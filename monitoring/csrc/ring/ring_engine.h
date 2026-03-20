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

private:
    RingConfig      cfg_;
    AllocatedRing   ring_;
    PinnedStaging   staging_;
    std::unique_ptr<DrainThread>  drain_;
    std::unique_ptr<P2PThread>    p2p_;
};

}  // namespace ring
