// ring/drain_thread.h — CPU drain thread: reads published TaskEntries from the
// ring and issues async D2H copies into pinned pool buffers.
//
// Lifecycle:
//   DrainThread dt(ring_state, pool, callback);
//   dt.start();
//   // ... GPU producer kernels run, each followed by cudaLaunchHostFunc
//   //     calling DrainThread::hostfunc_cb(&dt) to wake the drain thread ...
//   dt.stop();
//
// Per-chunk protocol:
//   1. hostfunc_cb() signals the CV; drain thread wakes.
//   2. drain_ready(): for each consecutive ready TaskEntry:
//        a. Check ready_seq via task_cpu_ready() (acquire load).
//        b. Copy entry metadata; issue cudaMemcpyAsync D2H into pinned buffer.
//        c. Record cudaEvent.
//        d. task_release_cpu() + advance task_tail (frees the task slot).
//        e. Push PendingChunk onto pending_ deque.
//   3. poll_completed(): for each event at the head of pending_:
//        a. cudaEventQuery — if done, advance payload_tail, invoke callback.
//        b. Stop at the first non-completed entry (FIFO order preserved).
//
// payload_tail is advanced only after D2H completes so the GPU producer never
// sees payload bytes as free before the CPU has copied them out.

#pragma once
#include "ring_state.h"
#include "pinned_pool.h"

#include <cuda_runtime.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace ring {

// Metadata delivered to the callback for each completed D2H chunk.
struct DrainedChunk {
    uint8_t*  pinned;              // pinned buffer start; valid until pool.free(alloc_bytes)
    uint64_t  len;                 // bytes of tensor data in pinned (len1 + len2)
    uint64_t  alloc_bytes;         // bytes to return to the pinned ring via pool.free()
    uint64_t  logical_task_id;
    uint64_t  chunk_offset_bytes;
    uint64_t  tensor_total_bytes;
    uint32_t  chunk_idx;
    uint32_t  flags;
    uint32_t  hook_type;
    uint32_t  hook_id;
    uint32_t  reason;
};

using DrainCallback = std::function<void(DrainedChunk&&)>;

class DrainThread {
public:
    DrainThread(RingState& rs, PinnedPool& pool, DrainCallback cb);
    ~DrainThread() noexcept;

    DrainThread(const DrainThread&)            = delete;
    DrainThread& operator=(const DrainThread&) = delete;

    void start();
    void stop();

    // Called from the CUDA host-function callback to wake the drain thread.
    void notify();

    // Block until at least `target` chunks have completed D2H and their
    // assembler callback has been invoked.  Call after cudaStreamSynchronize
    // so that `target` (= *ring.task_head) is stable.
    void wait_until_completed(uint64_t target);

    // Static shim suitable for cudaLaunchHostFunc / cudaHostFn_t.
    static void CUDART_CB hostfunc_cb(void* arg);

private:
    struct PendingChunk {
        cudaEvent_t event;          // nullptr for drop entries (no D2H)
        uint8_t*    pinned;         // nullptr for drop entries
        uint64_t    payload_bytes;  // bytes to return to payload ring on completion
        uint64_t    alloc_bytes;    // bytes to return to pinned ring via pool.free()
        DrainedChunk meta;
    };

    RingState&    ring_;
    PinnedPool&   pool_;
    DrainCallback cb_;
    cudaStream_t  stream_{};

    std::thread             thread_;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::atomic<bool>       running_{false};
    bool                    notified_{false};

    std::deque<PendingChunk> pending_;
    uint64_t                 payload_tail_local_{0};
    // Local counters — drain thread is the sole writer of task_tail and
    // consumer_heartbeat; tracking locally avoids PCIe reads of GPU-HBM
    // managed memory and avoids LOCK-prefix atomics on PCIe-mapped pages.
    uint64_t                 task_tail_local_{0};
    uint64_t                 heartbeat_local_{0};

    // Flush barrier: counts chunks that have completed D2H + assembler call.
    // Written only by the drain thread; read by wait_until_completed().
    std::atomic<uint64_t>    completed_local_{0};
    std::mutex               flush_mu_;
    std::condition_variable  flush_cv_;

    void loop();
    void drain_ready();
    void poll_completed();
};

}  // namespace ring
