// ring/drain_thread.h -- Batch drain thread (no condition tensor / backpressure).
//
// The drain thread scans GPU task entries, batches D2H copies via the staging
// ring, and submits completed tasks to the p2p thread for slicing and DB
// insertion.
//
// Space is guaranteed by the pre-forward capacity check in Python.  No
// cuStreamWaitValue32, no condition tensor, no large-tensor bypass.
//
// mgmt_mu_ protects all shared management state.
//
// GIL note: force_flush_and_wait is called with GIL released.

#pragma once
#include "ring_state.h"
#include "ring_config.h"
#include "pinned_staging.h"
#include "drain_task.h"

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ring {

struct RecordReservationItem {
    uint64_t reserved_payload_bytes{0};
    bool needs_reclaim{false};
};

class DrainThread {
public:
    DrainThread(RingState& rs, PinnedStaging& staging, const RingConfig& cfg);
    ~DrainThread() noexcept;

    DrainThread(const DrainThread&)            = delete;
    DrainThread& operator=(const DrainThread&) = delete;

    void start();
    void stop();

    void notify();

    // Request the drain thread to flush all pending entries, then block
    // until it finishes.  Caller must have done cudaStreamSynchronize on
    // the main stream first so all GPU writes are visible.
    void force_flush_and_wait();

    // Deadline-aware form used only by checked generic-record completion.
    // Returns false only when this request's flush generation has not
    // completed by deadline.  A timed-out request remains owned by the drain
    // thread; record-mode callers do not reuse the runtime after timeout.
    bool force_flush_and_wait_until(
        std::chrono::steady_clock::time_point deadline);

    // Submit a CPU-direct tensor to drain -> p2p pipeline.
    // The tensor is already in pageable CPU memory; skips D2H and staging.
    void submit_cpu_direct(at::Tensor cpu_tensor, uint64_t tensor_bytes);

    // Task queue interface (drain -> p2p)
    uint64_t wait_for_tasks();
    void pop_tasks(uint64_t n, std::vector<DrainTask>& out);
    void signal_p2p_stop();

    void notify_staging_freed_bytes(uint64_t nbytes);

    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    // Capacity query accessors (called from RingEnginePy::prepare_step).
    uint64_t cpu_payload_head() const;
    uint64_t cpu_payload_tail_committed() const;
    uint64_t cpu_task_head() const;
    uint64_t cpu_task_tail_committed() const;

    // Pre-allocate ring space for the next step's producer kernels.
    // Advances cpu_payload_head_ and cpu_task_head_ under mgmt_mu_.
    // Called from prepare_step after confirming space is available.
    void reserve(uint64_t payload_bytes, uint32_t num_tasks);

    // Record-mode reservation preserves each possibly-short task's upper bound.
    void reserve_record(const std::vector<RecordReservationItem>& items);
    void apply_pending_record_reclaims();
    uint64_t pending_record_reclaims() const;
    void rethrow_record_reclaim_failure() const;
    void rethrow_drain_failure();

private:
    RingState&      ring_;
    PinnedStaging&  staging_;
    RingConfig      cfg_;
    int             owner_device_{-1};
    cudaStream_t    stream_{};

    std::thread             thread_;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::atomic<bool>       running_{false};
    bool                    notified_{false};

    // ---- mgmt_mu_ protects all state below this line ----
    mutable std::mutex      mgmt_mu_;

    uint64_t                visible_head_{0};
    uint64_t                pending_entries_{0};
    uint64_t                pending_bytes_{0};
    std::deque<uint64_t>    scanned_;

    uint64_t                cpu_task_head_{0};
    uint64_t                cpu_payload_head_{0};
    uint64_t                cpu_task_tail_{0};
    uint64_t                cpu_payload_tail_{0};           // pending
    uint64_t                cpu_payload_tail_committed_{0}; // safe after D2H

    struct PendingTaskReclaim {
        uint64_t task_sequence{0};
        uint64_t reserved_payload_bytes{0};
    };
    std::deque<PendingTaskReclaim> pending_task_reclaims_;
    uint64_t                pending_reclaim_bytes_{0};
    std::exception_ptr      record_reclaim_failure_;

    std::chrono::steady_clock::time_point first_complete_time_{};
    bool                    has_complete_time_{false};
    // ---- end mgmt_mu_ protected state ----

    // Force-flush signalling (Python thread -> drain thread).  Generations
    // prevent one completed request from releasing a later overlapping
    // waiter.
    uint64_t                flush_requested_generation_{0};  // guarded by mu_
    uint64_t                flush_completed_generation_{0};  // guarded by mu_
    std::exception_ptr      drain_failure_;                  // guarded by mu_
    std::condition_variable flush_done_cv_;

    std::deque<DrainTask>   task_queue_;
    std::mutex              queue_mu_;
    uint64_t                can_pop_count_{0};
    std::mutex              pop_mu_;
    std::condition_variable pop_cv_;
    bool                    p2p_stop_requested_{false};

    std::mutex              staging_mu_;
    std::condition_variable staging_cv_;

    void loop();

    // Drain all pending entries -- called by the drain thread for an
    // outstanding flush generation.  Flushes repeatedly until empty.
    void do_full_flush();

    // Called under mgmt_mu_:
    void scan_ready();
    void account_record_task(uint64_t sequence, uint64_t actual_bytes);
    bool should_flush() const;
    void flush_state_update(uint64_t flush_count, uint64_t flush_bytes);

    void sync_stream();
    void enqueue_d2h(uint64_t flush_bytes);
    void record_drain_failure(std::exception_ptr failure);

    // Split into two: submit_to_p2p pushes DrainTasks to the p2p queue
    // (uses queue_mu_/pop_mu_, NOT mgmt_mu_).  trim_scanned updates
    // scanned_/pending state (caller MUST hold mgmt_mu_).
    void submit_to_p2p(uint64_t flush_count, uint64_t flush_bytes);
    void trim_scanned(uint64_t flush_count, uint64_t flush_bytes);
};

}  // namespace ring
