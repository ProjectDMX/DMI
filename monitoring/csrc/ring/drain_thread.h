// ring/drain_thread.h — Batch drain thread with condition tensor management.
//
// Concurrency model: Optimistic Concurrency Control (OCC).
//   1. prepare_forward (Python thread): grants based on committed tail
//      (conservative), enqueues H2D on main stream — no sync, no blocking.
//   2. Forward runs (GPU): producers write to ring, kernels reset conditions.
//   3. Drain re-checks forward_start_seq_ after D2H sync — validation phase.
//   4. Re-grant if prepare_forward leaked in — compensation (monotonic, safe).
//
// mgmt_mu_ protects all shared management state.  prepare_forward and the
// drain loop both acquire it for state updates.  Stream ops (D2H, H2D)
// happen outside the lock.
//
// GIL note: prepare_forward is called with GIL released
// (py::call_guard<py::gil_scoped_release> in bindings).

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

    void set_condition(uint32_t* d_cond, uint32_t* h_cond, uint32_t num_hooks);
    void set_null_mode(bool enabled, cudaStream_t main_stream);

    // Runs on Python thread (GIL released).  Computes conditions under
    // mgmt_mu_, enqueues H2D on main_stream.  Non-blocking, no sync.
    void prepare_forward(const std::vector<uint64_t>& hook_tensor_bytes,
                         cudaStream_t main_stream);

    void notify();

    // Task queue interface (drain → p2p)
    uint64_t wait_for_tasks();
    void pop_tasks(uint64_t n, std::vector<DrainTask>& out);
    void signal_p2p_stop();

    void notify_staging_freed_bytes(uint64_t nbytes);
    void notify_bypass_freed();

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

    // ---- mgmt_mu_ protects all state below this line ----
    std::mutex              mgmt_mu_;

    uint32_t*               d_condition_{nullptr};
    uint32_t*               h_condition_{nullptr};
    uint32_t                num_hooks_{0};

    std::vector<uint64_t>   hook_tensor_bytes_;

    uint64_t                visible_head_{0};
    uint64_t                pending_entries_{0};
    uint64_t                pending_bytes_{0};
    std::deque<TaskEntry>   scanned_;

    uint64_t                cpu_task_head_{0};
    uint64_t                cpu_payload_head_{0};
    uint64_t                cpu_task_tail_{0};
    uint64_t                cpu_payload_tail_{0};           // pending
    uint64_t                cpu_payload_tail_committed_{0}; // safe after D2H

    uint64_t                forward_start_seq_{0};
    uint64_t                old_pending_{0};

    uint64_t                granted_count_{0};
    uint64_t                blocked_count_{0};
    uint64_t                scanned_complete_{0};
    uint32_t                next_ungrant_idx_{0};
    uint32_t                drain_hook_idx_{0};

    uint32_t                dirty_lo_{0};
    uint32_t                dirty_hi_{0};
    bool                    dirty_{false};
    bool                    null_mode_{false};

    std::chrono::steady_clock::time_point first_complete_time_{};
    bool                    has_complete_time_{false};
    // ---- end mgmt_mu_ protected state ----

    std::deque<DrainTask>   task_queue_;
    std::mutex              queue_mu_;
    uint64_t                can_pop_count_{0};
    std::mutex              pop_mu_;
    std::condition_variable pop_cv_;
    bool                    p2p_stop_requested_{false};

    std::mutex              staging_mu_;
    std::condition_variable staging_cv_;

    uint64_t                in_flight_bypass_bytes_{0};
    std::mutex              bypass_mu_;
    std::condition_variable bypass_cv_;

    void loop();

    // Called under mgmt_mu_:
    void scan_ready();
    bool should_flush() const;
    void flush_state_update(uint64_t flush_count, uint64_t flush_bytes);
    void grant_next_hooks();
    void handle_large_tensor();

    void mark_dirty(uint32_t idx);
    void enqueue_conditions_h2d();  // on drain stream_
    void sync_stream();
    void enqueue_d2h(uint64_t flush_bytes);

    // Split into two: submit_to_p2p pushes DrainTasks to the p2p queue
    // (uses queue_mu_/pop_mu_, NOT mgmt_mu_).  trim_scanned updates
    // scanned_/pending state (caller MUST hold mgmt_mu_).
    void submit_to_p2p(uint64_t flush_count, uint64_t flush_bytes);
    void trim_scanned(uint64_t flush_count, uint64_t flush_bytes);
};

}  // namespace ring
