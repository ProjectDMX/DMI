// ring/ring_engine.h -- Top-level RAII engine combining all ring components.

#pragma once
#include "ring_alloc.h"
#include "pinned_staging.h"
#include "drain_thread.h"
#include "p2p_thread.h"
#include "record_sink.h"
#include "tensor_meta.h"

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

namespace ring {

class RingEngine {
public:
    RingEngine(const RingConfig& cfg, ring_py::TensorMetaFifo& fifo,
               SubmitFn submit_fn, BatchSink batch_sink = nullptr);
    RingEngine(const RingConfig& cfg, std::shared_ptr<RecordSinkLease> lease,
               RecordFailurePolicy failure_policy =
                   RecordFailurePolicy::kRaiseAtProducer);
    ~RingEngine() noexcept;

    RingEngine(const RingEngine&)            = delete;
    RingEngine& operator=(const RingEngine&) = delete;

    void init(cudaStream_t stream = 0);
    void start();
    void stop();

    RingState&   ring_state()    { return ring_.state(); }
    DrainThread& drain_thread()  { return *drain_; }
    RecordConsumer& record_consumer();
    const RecordConsumer& record_consumer() const;
    std::shared_ptr<RecordSink> record_sink() const { return record_sink_; }
    bool record_mode() const { return static_cast<bool>(record_p2p_); }
    // Legacy rings only (nullptr on a record ring).
    P2PThread*   p2p_thread()    { return p2p_.get(); }
    const P2PThread* p2p_thread() const { return p2p_.get(); }

    // Flush the ring, then wait until the P2P thread has processed every task
    // the drain has handed it: every consumer call for every captured slice
    // or batch has returned.  Returns true when delivered, false on timeout
    // (0 = wait forever).  The caller synchronizes the producer stream first.
    // timeout_ms == 0 waits forever; otherwise ONE deadline, taken at entry,
    // bounds the flush and the delivery wait together.
    bool drain_to_sink_and_wait(uint32_t timeout_ms);
    // Same with an absolute deadline (nullopt = no deadline).  Lets a caller
    // that already spent part of its budget (producer-stream completion)
    // pass the remainder on.
    bool drain_to_sink_and_wait_until(
        std::optional<std::chrono::steady_clock::time_point> deadline);

    uint64_t  payload_cap() const { return cfg_.payload_ring_bytes; }
    bool      drain_step_gated() const { return cfg_.drain_step_gated; }
    uint64_t  staging_cap() const { return staging_.capacity(); }
    uint64_t  task_cap()    const { return cfg_.task_ring_entries; }

private:
    RingConfig      cfg_;
    AllocatedRing   ring_;
    PinnedStaging   staging_;
    std::unique_ptr<DrainThread>  drain_;
    std::unique_ptr<P2PThread>    p2p_;
    std::shared_ptr<RecordSinkLease> record_sink_lease_;
    std::shared_ptr<RecordSink> record_sink_;
    std::unique_ptr<RecordP2PThread> record_p2p_;
    bool record_sink_released_{false};

    void release_record_sink() noexcept;
};

}  // namespace ring
