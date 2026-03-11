// ring/drain_thread.h — Batch drain thread: scans GPU task ring, issues batch
// D2H copies into a pinned staging ring, pushes DrainTasks to a locked queue
// for the p2p thread.
//
// Lifecycle:
//   DrainThread dt(ring_state, staging, cfg);
//   dt.start();
//   // ... GPU producer kernels run ...
//   dt.stop();   // final flush

#pragma once
#include "ring_state.h"
#include "ring_config.h"
#include "pinned_staging.h"
#include "drain_task.h"
#include "task_entry.h"

#include <cuda_runtime.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace ring {

class DrainThread {
public:
    DrainThread(RingState& rs, PinnedStaging& staging, const RingConfig& cfg);
    ~DrainThread() noexcept;

    DrainThread(const DrainThread&)            = delete;
    DrainThread& operator=(const DrainThread&) = delete;

    void start();
    void stop();

    // Wake the drain thread (non-blocking).
    void notify();

    // Static shim for cudaLaunchHostFunc / cudaHostFn_t.
    static void CUDART_CB hostfunc_cb(void* arg);

    // --- Task queue interface (drain → p2p) ---

    // Wait until can_pop_count > 0 or p2p_stop_requested. Returns number of
    // tasks to pop.  Sets can_pop_count_ to 0 under lock.
    uint64_t wait_for_tasks();

    // Pop exactly n tasks from the front of the queue into `out`.
    void pop_tasks(uint64_t n, std::vector<DrainTask>& out);

    // Signal the p2p thread to exit (called after drain thread has joined,
    // guaranteeing no more tasks will be pushed).
    void signal_p2p_stop();

    // Release staging bytes and notify drain thread that staging space has been freed.
    void notify_staging_freed_bytes(uint64_t nbytes);

    // Notify drain thread that bypass budget has been freed.
    void notify_bypass_freed();

    // --- Bypass backpressure interface ---
    std::mutex&              bypass_mu()  { return bypass_mu_; }
    std::condition_variable& bypass_cv()  { return bypass_cv_; }
    uint64_t& in_flight_bypass_bytes()    { return in_flight_bypass_bytes_; }

    bool is_running() const { return running_.load(std::memory_order_relaxed); }

private:
    RingState&      ring_;
    PinnedStaging&  staging_;
    RingConfig      cfg_;
    cudaStream_t    stream_{};

    std::thread             thread_;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::atomic<bool>       running_{false};
    bool                    notified_{false};

    // Scan progress (CPU-local to drain thread)
    uint64_t                visible_head_{0};
    uint64_t                pending_entries_{0};
    uint64_t                pending_bytes_{0};
    std::deque<TaskEntry>   scanned_;
    int64_t                 last_complete_idx_{-1};

    // Incremental flush-size tracking (avoids re-summing on every should_flush)
    uint64_t                scan_bytes_accum_{0};   // running sum of alloc_bytes for all scanned entries
    uint64_t                flushable_bytes_{0};    // sum of alloc_bytes for entries [0..last_complete_idx_]
    uint64_t                flushable_entries_{0};  // last_complete_idx_ + 1

    // Local mirrors of managed-memory counters
    uint64_t                payload_tail_local_{0};
    uint64_t                task_tail_local_{0};
    uint64_t                heartbeat_local_{0};

    // Task queue (drain → p2p)
    std::deque<DrainTask>   task_queue_;
    std::mutex              queue_mu_;
    uint64_t                can_pop_count_{0};
    std::mutex              pop_mu_;
    std::condition_variable pop_cv_;
    bool                    p2p_stop_requested_{false};
    uint64_t                pending_incomplete_cnt_{0};

    // Staging backpressure
    std::mutex              staging_mu_;
    std::condition_variable staging_cv_;

    // Bypass backpressure
    uint64_t                in_flight_bypass_bytes_{0};
    std::mutex              bypass_mu_;
    std::condition_variable bypass_cv_;

    void loop();
    void scan_ready();
    bool should_flush() const;
    void batch_flush();
    void handle_large_tensor();
};

}  // namespace ring
