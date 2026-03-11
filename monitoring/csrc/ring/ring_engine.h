// ring/ring_engine.h — Top-level RAII engine combining all ring components.
//
// RingEngine owns:
//   AllocatedRing    — GPU ring buffers (task entries + payload buffer)
//   PinnedStaging    — pinned host staging ring for batch D2H
//   DrainThread      — CPU thread: scans GPU entries, batch D2H, pushes to task queue
//   P2PThread        — CPU thread: pinned→pageable copy, metadata, slicing, submit
//
// Typical usage:
//   RingEngine engine(cfg, fifo, submit_fn);
//   engine.init();
//   engine.start();
//   // In GPU hook / custom op:
//   launch_producer(engine.ring_state(), d_src, bytes, id,
//                   hook_type, hook_id, stream);
//   engine.stop();

#pragma once
#include "ring_alloc.h"
#include "pinned_staging.h"
#include "drain_thread.h"
#include "p2p_thread.h"
#include "tensor_meta.h"

#include <memory>

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

private:
    RingConfig      cfg_;
    AllocatedRing   ring_;
    PinnedStaging   staging_;
    std::unique_ptr<DrainThread>  drain_;
    std::unique_ptr<P2PThread>    p2p_;
};

}  // namespace ring
