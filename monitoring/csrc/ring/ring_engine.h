// ring/ring_engine.h — Top-level RAII engine combining all ring components.
//
// RingEngine owns:
//   AllocatedRing    — GPU ring buffers (task entries + payload buffer)
//   PinnedPool       — pinned host buffers for D2H copies
//   ChunkAssembler   — reassembles multi-chunk tensors
//   DrainThread      — CPU thread: drains ready entries, issues D2H copies
//
// Typical usage:
//   RingEngine engine(cfg, [](AssembledTensor&& t) { /* process t */ });
//   engine.init();
//   engine.start();
//
//   // In GPU hook / custom op:
//   launch_producer_with_notify(engine.ring_state(), d_src, bytes, id,
//                               hook_type, hook_id,
//                               DrainThread::hostfunc_cb,
//                               &engine.drain_thread(), stream);
//
//   engine.stop();  // waits for in-flight D2H to complete

#pragma once
#include "ring_alloc.h"
#include "pinned_pool.h"
#include "chunk_assembler.h"
#include "drain_thread.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace ring {

class RingEngine {
public:
    RingEngine(const RingConfig& cfg, TensorCallback cb);
    ~RingEngine() noexcept;

    RingEngine(const RingEngine&)            = delete;
    RingEngine& operator=(const RingEngine&) = delete;

    // Initialise GPU buffers (memset SENTINEL, zero counters).
    // Call once before start() and before any graph capture.
    void init(cudaStream_t stream = 0);

    void start();
    void stop();

    RingState&   ring_state()    { return ring_.state(); }
    DrainThread& drain_thread()  { return *drain_; }

private:
    RingConfig    cfg_;
    AllocatedRing ring_;
    PinnedPool    pool_;
    TensorCallback cb_;
    std::unique_ptr<ChunkAssembler> assembler_;
    std::unique_ptr<DrainThread>    drain_;

    // Callback thread: decouples Python on_tensor (GIL) from the drain loop.
    // The drain thread pushes assembled tensors here; this thread calls cb_.
    // This prevents GIL contention from stalling drain_ready() / task slot freeing.
    std::mutex              cb_mu_;
    std::condition_variable cb_cv_;
    std::queue<AssembledTensor> cb_queue_;
    std::atomic<bool>       cb_running_{false};
    std::thread             cb_thread_;

    void cb_loop();
};

}  // namespace ring
