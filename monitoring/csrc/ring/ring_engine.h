// ring/ring_engine.h — Top-level RAII engine combining all ring components.

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

    // Allocate condition tensor after hook count is known.
    // Must be called before CUDA graph capture.
    void init_hooks(uint32_t num_hooks);

    // prepare_forward is called directly on DrainThread from
    // RingEnginePy::prepare_forward (ring_engine_py.cu).  It runs on the
    // Python thread, not the drain thread.  No RingEngine wrapper needed.

    RingState&   ring_state()    { return ring_.state(); }
    DrainThread& drain_thread()  { return *drain_; }

    uint32_t* d_condition()      { return ring_.d_condition(); }
    uint32_t  num_hooks() const  { return ring_.num_hooks(); }
    uint64_t  payload_cap() const { return cfg_.payload_ring_bytes; }

private:
    RingConfig      cfg_;
    AllocatedRing   ring_;
    PinnedStaging   staging_;
    std::unique_ptr<DrainThread>  drain_;
    std::unique_ptr<P2PThread>    p2p_;
};

}  // namespace ring
