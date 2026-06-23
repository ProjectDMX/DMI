// ring/ring_engine.h -- Top-level RAII engine combining all ring components.

#pragma once
#include "ring_alloc.h"
#include "pinned_staging.h"
#include "drain_thread.h"
#include "p2p_thread.h"
#include "tensor_meta.h"

#include <memory>
#include <vector>

namespace ring {

class RingEngine {
public:
    RingEngine(const RingConfig& cfg, ring_py::TensorMetaFifo& fifo,
               SubmitFn submit_fn);
    ~RingEngine() noexcept;

    RingEngine(const RingEngine&)            = delete;
    RingEngine& operator=(const RingEngine&) = delete;

    void init(cudaStream_t stream = 0);
    void start();
    void stop();

    RingState&   ring_state()    { return ring_.state(); }
    DrainThread& drain_thread()  { return *drain_; }

    uint64_t  payload_cap() const { return cfg_.payload_ring_bytes; }
    uint64_t  staging_cap() const { return staging_.capacity(); }
    uint64_t  task_cap()    const { return cfg_.task_ring_entries; }

    // Flush the GPU ring AND barrier through the p2p -> SubmitFn stage, so
    // that on return every produced slice has been HANDED to the sink stage.
    // Caller must cudaStreamSynchronize first (see RingEnginePy).  Returns 0 if
    // all tasks were processed, 1 on timeout (timeout_ms==0 waits forever).
    // NOTE: "processed" means every per-request submit_fn was CALLED, not that
    // it succeeded -- a SubmitFn exception is caught + counted (submit_exceptions),
    // not propagated.  Pair this barrier with submit_exceptions()/sink_failed()
    // to tell delivered-and-succeeded from delivered-but-some-sink-errors.
    int drain_to_sink_and_wait(uint32_t timeout_ms) {
        drain_->force_flush_and_wait();                 // GPU ring -> p2p queue
        const uint64_t target = drain_->tasks_enqueued();
        return p2p_->wait_until_processed(target, timeout_ms) ? 0 : 1;
    }

    // Fail-loud sink error surface.
    uint64_t    submit_exceptions() const { return p2p_->submit_exceptions(); }
    std::string last_sink_error()   const { return p2p_->last_error(); }
    void set_abort_on_sink_error(bool v)  { p2p_->set_abort_on_sink_error(v); }
    bool sink_failed() const              { return p2p_->sink_failed(); }

private:
    RingConfig      cfg_;
    AllocatedRing   ring_;
    PinnedStaging   staging_;
    std::unique_ptr<DrainThread>  drain_;
    std::unique_ptr<P2PThread>    p2p_;
};

}  // namespace ring
