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
#include "tensor_meta.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace at { class Tensor; }

namespace ring {

// Called by the p2p thread for each per-request tensor slice.
// Runs without the GIL -- all arguments are C++ types.
using SubmitFn = std::function<void(
    const std::string& model_id,
    int32_t            shard_rank,
    const std::string& req_id,
    const std::string& act_name,
    int32_t            layer_no,
    int32_t            start_token,
    int32_t            end_token,
    at::Tensor         slice)>;

class P2PThread {
public:
    P2PThread(DrainThread& drain, ring_py::TensorMetaFifo& fifo,
              const RingConfig& cfg, SubmitFn submit_fn);
    ~P2PThread() noexcept;

    P2PThread(const P2PThread&)            = delete;
    P2PThread& operator=(const P2PThread&) = delete;

    void start();
    void stop();

    // -- C0: fail-loud sink error accounting (observable from Python) -------
    // submit_fn exceptions (incl. re-raised Python SubmitFn errors) are caught
    // in do_post_processing; instead of being swallowed they are counted here.
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
    // True once a sink error occurred AND abort-on-sink-error is armed.
    bool sink_failed() const {
        return abort_on_sink_error_.load(std::memory_order_relaxed)
            && submit_exceptions_.load(std::memory_order_relaxed) > 0;
    }

    // -- C1: barrier through the SubmitFn ----------------------------------
    // Tasks fully processed by this thread (each = one DrainTask popped and run
    // through do_post_processing, i.e. all its per-request submit_fn calls have
    // returned).  Paired with DrainThread::tasks_enqueued() for a sink barrier.
    uint64_t tasks_processed() const {
        return tasks_processed_.load(std::memory_order_acquire);
    }
    // Block until tasks_processed() >= target, or timeout_ms elapses (0 = wait
    // forever).  Returns true if the target was reached.
    bool wait_until_processed(uint64_t target, uint32_t timeout_ms);

private:
    DrainThread&             drain_;
    ring_py::TensorMetaFifo& fifo_;
    RingConfig               cfg_;
    SubmitFn                 submit_fn_;

    std::thread              thread_;
    ring_py::StepContext*    current_ctx_{nullptr};  // owned, freed on last_in_step

    // C0 error accounting.
    std::atomic<uint64_t>    submit_exceptions_{0};
    mutable std::mutex       err_mu_;
    std::string              last_error_;
    std::atomic<bool>        abort_on_sink_error_{false};

    // C1 sink barrier.
    std::atomic<uint64_t>    tasks_processed_{0};
    std::mutex               barrier_mu_;
    std::condition_variable  barrier_cv_;

    void loop();
    void process(std::vector<DrainTask>& tasks);
    void do_post_processing(at::Tensor& tensor, const DrainTask& first_task);
    void note_submit_error(const char* what);
};

}  // namespace ring
