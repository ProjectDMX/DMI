// ring/p2p_thread.h -- Pinned-to-pageable thread: copies assembled tensors from
// the pinned staging ring to pageable CPU memory, pops metadata from
// TensorMetaFifo, performs per-request slicing, and submits to host engine.
//
// Single-threaded: staging_.tail_ is monotonic; multi-threaded p2p would
// require gap tracking for out-of-order completion.

#pragma once
#include "drain_thread.h"
#include "drain_task.h"
#include "ring_config.h"
#include "record_consumer.h"
#include "tensor_meta.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace at { class Tensor; }

namespace ring {

class BufferPool;  // buffer_pool.h, included only by the .cpp

// Called by the p2p thread for each per-request tensor slice.
// Runs without the GIL -- all arguments are C++ types.  capture_ts_us is
// DrainTask::capture_ts_us; capture_lease is the step's lease.
using SubmitFn = std::function<void(
    const std::string&    model_id,
    int32_t               shard_rank,
    const std::string&    req_id,
    const std::string&    act_name,
    int32_t               layer_no,
    int32_t               start_token,
    int32_t               end_token,
    at::Tensor            slice,
    int64_t               capture_ts_us,
    ring_py::CaptureLease capture_lease)>;

// Called by the p2p thread ONCE per (hook x step) on the batch handoff
// (RingConfig::batch_submit, packed steps only): the whole reshaped tensor
// plus a segments table; the consumer does the per-request slicing.
//   segments: int32 CPU [N, 4] rows (dim0_offset, db_start, db_end,
//             orig_position)
//   req_ids:  N request ids, in segments row order
// last_in_step marks the last hook of a step.  Runs without the GIL.
//
// Per-slice fallback: when the batch sink is the only consumer, captures the
// whole-batch handoff does not cover (HF batched steps, token or attention)
// arrive here as ONE-REQUEST items, one call per request slice: N == 1,
// req_ids = {req_id}, segments = {(0, db_start, db_end, orig_position)} and
// `tensor` is that request's slice -- [rows, ...] for token hooks,
// [heads, rows, kv] for attention matrices (is_attn true; take it whole).
// last_in_step is set on the last item of the step's last hook.
using BatchSink = std::function<void(
    const std::string&       model_id,
    int32_t                  shard_rank,
    const std::string&       act_name,
    int32_t                  layer_no,
    uint64_t                 step_seq,
    bool                     last_in_step,
    at::Tensor               tensor,
    std::vector<std::string> req_ids,
    at::Tensor               segments,
    ring_py::CaptureLease    capture_lease,
    int64_t                  capture_ts_us,
    bool                     is_attn)>;

class P2PThread {
public:
    P2PThread(DrainThread& drain, ring_py::TensorMetaFifo& fifo,
              const RingConfig& cfg, SubmitFn submit_fn,
              BatchSink batch_sink = nullptr);
    ~P2PThread() noexcept;

    P2PThread(const P2PThread&)            = delete;
    P2PThread& operator=(const P2PThread&) = delete;

    void start();
    void stop();

    // -- consumer failures ---------------------------------------------------
    // Exceptions thrown by the consumer are logged once and counted here
    // instead of disappearing; the most recent message is kept.
    uint64_t submit_exceptions() const {
        return submit_exceptions_.load(std::memory_order_relaxed);
    }
    std::string last_error() const {
        std::lock_guard<std::mutex> lk(err_mu_);
        return last_error_;
    }
    void set_abort_on_sink_error(bool v) {
        abort_on_sink_error_.store(v, std::memory_order_relaxed);
    }
    // True once a consumer error occurred AND abort-on-sink-error is armed.
    bool sink_failed() const {
        return abort_on_sink_error_.load(std::memory_order_relaxed)
            && submit_exceptions_.load(std::memory_order_relaxed) > 0;
    }

    // -- delivery barrier -----------------------------------------------------
    // DrainTasks fully processed by this thread: every consumer call for the
    // task has returned.  Paired with DrainThread::tasks_enqueued().
    uint64_t tasks_processed() const {
        return tasks_processed_.load(std::memory_order_acquire);
    }
    // Block until tasks_processed() >= target, timeout_ms elapses (0 = no
    // timeout), or the thread has exited.  Returns true when the target was
    // reached.
    bool wait_until_processed(uint64_t target, uint32_t timeout_ms);
    // Same, bounded by an absolute deadline (a deadline already passed
    // checks once and returns).
    bool wait_until_processed_until(uint64_t target,
                                    std::chrono::steady_clock::time_point deadline);

private:
    DrainThread&             drain_;
    ring_py::TensorMetaFifo& fifo_;
    RingConfig               cfg_;
    SubmitFn                 submit_fn_;
    BatchSink                batch_sink_;
    // Incremented for every new StepContext; carried in each batch.
    uint64_t                 step_seq_{0};

    std::thread              thread_;
    ring_py::StepContext*    current_ctx_{nullptr};  // owned, freed on last_in_step

    // Warm pageable buffers for the staging -> pageable copy, recycled by
    // reference count: it must outlive every slice it hands out, so it is
    // destroyed last, in ~P2PThread, after stop() joins this thread.
    std::unique_ptr<BufferPool> pool_;

    std::atomic<uint64_t>    submit_exceptions_{0};
    mutable std::mutex       err_mu_;
    std::string              last_error_;
    std::atomic<bool>        abort_on_sink_error_{false};

    std::atomic<uint64_t>    tasks_processed_{0};
    std::mutex               barrier_mu_;
    std::condition_variable  barrier_cv_;
    bool                     stopped_{false};   // loop exited; under barrier_mu_

    void loop();
    void process(std::vector<DrainTask>& tasks);
    void do_post_processing(at::Tensor& tensor, const DrainTask& first_task);
    void note_submit_error(const char* what);
};

// Schema-driven record path.  It intentionally has a distinct worker class
// from P2PThread so the released inference path never selects a consumer per
// payload.  Construction fixes the consumer for the lifetime of the ring.
class RecordP2PThread {
public:
    RecordP2PThread(DrainThread& drain, std::shared_ptr<RecordSink> sink,
                    RecordFailurePolicy failure_policy =
                        RecordFailurePolicy::kRaiseAtProducer);
    ~RecordP2PThread() noexcept;

    RecordP2PThread(const RecordP2PThread&) = delete;
    RecordP2PThread& operator=(const RecordP2PThread&) = delete;

    void start();
    void stop();

    RecordConsumer& consumer() { return consumer_; }
    const RecordConsumer& consumer() const { return consumer_; }

private:
    DrainThread&  drain_;
    RecordConsumer consumer_;
    std::thread   thread_;

    void loop();
    void process(std::vector<DrainTask>& tasks);
};

}  // namespace ring
