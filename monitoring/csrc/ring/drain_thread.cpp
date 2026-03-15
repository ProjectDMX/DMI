// ring/drain_thread.cpp — Batch drain thread implementation.
// Compiled with g++ (not nvcc); uses CUDA runtime C API via -lcudart.
//
// See drain_thread.h for OCC concurrency model overview.
//
// CONDITION TENSOR THREAD SAFETY (dirty flags, h_condition_, d_condition_):
//
// prepare_forward (Python thread) and the drain loop both write to
// h_condition_[], dirty_lo_/hi_/dirty_, and enqueue H2D to d_condition_.
// These accesses are safe WITHOUT special dirty-flag locking because the
// two writers are TEMPORALLY SEPARATED:
//
//   - prepare_forward runs BETWEEN forwards (after the previous forward
//     completes, before the next starts).
//
//   - drain's grant_next_hooks (which sets dirty) runs DURING forwards
//     (when entries arrive and forced flush fires).  Between forwards,
//     grant_next_hooks has nothing to grant (next_ungrant_idx_ == num_hooks,
//     all hooks were granted for the completed forward) — it does NOT
//     touch dirty state.
//
//   - OCC re-check (grant_next_hooks after sync): runs AFTER prepare_forward
//     has finished and cleared dirty.  prepare_forward won't run again
//     until the current forward completes.
//
// Therefore: when prepare_forward sets dirty, the drain is NOT setting
// dirty.  When the drain sets dirty (mid-forward grants or OCC re-check),
// prepare_forward is NOT running.  No concurrent dirty access.
//
// prepare_forward is still fully locked under mgmt_mu_ (including the
// H2D enqueue) for simplicity and to protect the management state
// (counters, grant state, forward bookkeeping).

#include "drain_thread.h"
#include "task_ring.cuh"
#include "task_entry.h"
#include "ring_config.h"
#include "ring_debug.h"

#include <ATen/ATen.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace ring {

// ---------------------------------------------------------------------------
DrainThread::DrainThread(RingState& rs, PinnedStaging& staging,
                         const RingConfig& cfg)
    : ring_(rs), staging_(staging), cfg_(cfg)
{
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("DrainThread: cudaStreamCreate failed");
}

DrainThread::~DrainThread() noexcept {
    stop();
    cudaStreamDestroy(stream_);
}

void DrainThread::set_condition(uint32_t* d_cond, uint32_t* h_cond, uint32_t n) {
    d_condition_ = d_cond;
    h_condition_ = h_cond;
    num_hooks_   = n;
}

void DrainThread::start() {
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread([this] { loop(); });
}

void DrainThread::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void DrainThread::notify() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        notified_ = true;
    }
    cv_.notify_one();
}

// ---------------------------------------------------------------------------
// prepare_forward — runs on Python thread (GIL released).
//
// Entirely under mgmt_mu_ (including H2D enqueue — cudaMemcpyAsync is
// non-blocking, ~microseconds).  This protects management state and
// simplifies reasoning about dirty flag ownership.
//
// Uses committed payload tail (only space confirmed safe after D2H) for
// conservative grants.  The drain's OCC re-check after sync will re-grant
// blocked hooks with newly freed resources.
//
// Only writes GRANTED hooks to h_condition_ (COND_GRANT_FULL / TASK_ONLY).
// Blocked hooks are NOT written — d_condition_ is already 0 from kernel
// reset of previous forward.  This ensures the H2D dirty range [0, N)
// does not overlap with drain's mid-forward grants [N, ...).
// ---------------------------------------------------------------------------
void DrainThread::prepare_forward(
    const std::vector<uint64_t>& hook_tensor_bytes,
    cudaStream_t main_stream)
{
    std::lock_guard<std::mutex> lk(mgmt_mu_);

    // Reclassify: all current-forward entries still in scanned_ become "old".
    // Use (pending - old) not scanned_complete_ — scanned_complete_ is
    // monotonic (total ever scanned) and may exceed entries still in queue.
    old_pending_ = pending_entries_;  // all entries in queue are now old
    scanned_complete_ = 0;

    forward_start_seq_ = cpu_task_head_;
    hook_tensor_bytes_ = hook_tensor_bytes;
    granted_count_     = 0;
    blocked_count_     = 0;
    next_ungrant_idx_  = 0;
    drain_hook_idx_    = 0;
    has_complete_time_  = false;
    dirty_              = false;

    // Use committed tail for safety (only space where D2H confirmed done).
    // If the drain is mid-D2H, committed < pending → conservative grants.
    // The drain's OCC re-check will re-grant after sync updates committed.
    uint64_t available_tasks   = ring_.task_cap - (cpu_task_head_ - cpu_task_tail_);
    uint64_t available_payload = ring_.payload_cap - (cpu_payload_head_ - cpu_payload_tail_committed_);

    uint32_t num = static_cast<uint32_t>(hook_tensor_bytes_.size());
    if (num > num_hooks_) num = num_hooks_;

    for (uint32_t i = 0; i < num; ++i) {
        uint64_t raw_bytes    = hook_tensor_bytes_[i];
        uint64_t padded_bytes = align_up(raw_bytes, PAYLOAD_ALIGN);
        // Bypass if tensor exceeds either ring or staging capacity.
        bool fits_normal      = (padded_bytes <= ring_.payload_cap) &&
                                (padded_bytes <= staging_.capacity());

        if (fits_normal) {
            if (available_tasks >= 1 && available_payload >= padded_bytes) {
                h_condition_[i] = COND_GRANT_FULL; mark_dirty(i);
                available_tasks   -= 1;
                available_payload -= padded_bytes;
                cpu_task_head_    += 1;
                cpu_payload_head_ += padded_bytes;
                granted_count_++;
            } else {
                // Must break: grants must be a contiguous prefix [0, N) so
                // the H2D dirty range doesn't include stale values for
                // skipped hooks, and doesn't overlap with drain's range.
                blocked_count_ += (num - i);
                break;
            }
        } else {
            if (available_tasks >= 1) {
                h_condition_[i] = COND_GRANT_TASK_ONLY; mark_dirty(i);
                available_tasks -= 1;
                cpu_task_head_  += 1;
                granted_count_++;
            } else {
                blocked_count_ += (num - i);
                break;
            }
        }
    }

    next_ungrant_idx_ = static_cast<uint32_t>(granted_count_);

    RING_DBG("[prepare_forward] num_hooks=%u granted=%lu blocked=%lu "
            "next_ungrant=%u avail_tasks=%lu avail_payload=%lu "
            "dirty=%d [%u,%u) d_cond=%p h_cond=%p\n",
            num, (unsigned long)granted_count_, (unsigned long)blocked_count_,
            next_ungrant_idx_,
            (unsigned long)(ring_.task_cap - (cpu_task_head_ - cpu_task_tail_)),
            (unsigned long)(ring_.payload_cap - (cpu_payload_head_ - cpu_payload_tail_committed_)),
            (int)dirty_, dirty_lo_, dirty_hi_,
            (void*)d_condition_, (void*)h_condition_);


    // H2D on main stream — ordered before graph's cuStreamWaitValue32.
    if (dirty_ && d_condition_ && h_condition_) {
        cudaError_t err = cudaMemcpyAsync(
                        d_condition_ + dirty_lo_,
                        h_condition_ + dirty_lo_,
                        (dirty_hi_ - dirty_lo_) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, main_stream);
        dirty_ = false;
        RING_DBG("[prepare_forward] H2D enqueued on main_stream=%p err=%d(%s)\n",
                (void*)main_stream, (int)err, cudaGetErrorString(err));

    } else if (!dirty_) {
        RING_DBG("[prepare_forward] WARNING: nothing dirty — no H2D enqueued!\n");

    } else {
        RING_DBG("[prepare_forward] WARNING: d_cond or h_cond null — no H2D!\n");

    }
}

// ---------------------------------------------------------------------------
// Task queue interface for p2p thread
// ---------------------------------------------------------------------------
uint64_t DrainThread::wait_for_tasks() {
    std::unique_lock<std::mutex> lk(pop_mu_);
    pop_cv_.wait(lk, [this] {
        return can_pop_count_ > 0 || p2p_stop_requested_;
    });
    uint64_t n = can_pop_count_;
    can_pop_count_ = 0;
    return n;
}

void DrainThread::signal_p2p_stop() {
    {
        std::lock_guard<std::mutex> lk(pop_mu_);
        p2p_stop_requested_ = true;
    }
    pop_cv_.notify_all();
}

void DrainThread::pop_tasks(uint64_t n, std::vector<DrainTask>& out) {
    out.clear();
    out.reserve(n);
    std::lock_guard<std::mutex> lk(queue_mu_);
    for (uint64_t i = 0; i < n; ++i) {
        out.push_back(std::move(task_queue_.front()));
        task_queue_.pop_front();
    }
}

void DrainThread::notify_staging_freed_bytes(uint64_t nbytes) {
    {
        std::lock_guard<std::mutex> lk(staging_mu_);
        staging_.advance_tail(nbytes);
    }
    staging_cv_.notify_one();
}

void DrainThread::notify_bypass_freed() {
    bypass_cv_.notify_one();
}

// ---------------------------------------------------------------------------
void DrainThread::mark_dirty(uint32_t idx) {
    if (!dirty_) {
        dirty_lo_ = idx;
        dirty_hi_ = idx + 1;
        dirty_ = true;
    } else {
        if (idx < dirty_lo_) dirty_lo_ = idx;
        if (idx + 1 > dirty_hi_) dirty_hi_ = idx + 1;
    }
}

void DrainThread::enqueue_conditions_h2d() {
    if (!dirty_ || !d_condition_ || !h_condition_) return;
    cudaMemcpyAsync(d_condition_ + dirty_lo_,
                    h_condition_ + dirty_lo_,
                    (dirty_hi_ - dirty_lo_) * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream_);
    dirty_ = false;
}

void DrainThread::sync_stream() {
    cudaStreamSynchronize(stream_);
}

// ---------------------------------------------------------------------------
// Main loop
//
// Phase 1 (under mgmt_mu_): scan entries, decide flush, update state.
// Phase 2 (outside lock): D2H + condition H2D + sync (stream ops).
// OCC re-check (under mgmt_mu_): detect leaked prepare_forward, re-grant.
//
// Dirty flag safety: the drain's enqueue_conditions_h2d() runs outside
// mgmt_mu_ but this is safe — see CONDITION TENSOR THREAD SAFETY comment
// at the top of this file.  When the drain sets dirty (mid-forward grants
// or OCC re-check), prepare_forward is not running.  When prepare_forward
// sets dirty (between forwards), the drain's grant_next_hooks has nothing
// to grant and does not touch dirty.
// ---------------------------------------------------------------------------
void DrainThread::loop() {
    RING_DBG("[drain_loop] started, poll_timeout=%lu us\n",
            (unsigned long)cfg_.drain_poll_timeout_us);

    uint64_t loop_iter = 0;
    auto last_log = std::chrono::steady_clock::now();

    while (running_.load(std::memory_order_relaxed)) {
        ++loop_iter;
        if (loop_iter <= 3 || loop_iter % 10000 == 0) {
            RING_DBG("[drain_loop] iter=%lu pending=%lu\n",
                    (unsigned long)loop_iter, (unsigned long)pending_entries_);

        }
        uint64_t flush_count = 0, flush_bytes = 0;
        bool needs_flush = false;
        uint64_t fss_snapshot = 0;

        {
            std::lock_guard<std::mutex> lk(mgmt_mu_);
            scan_ready();

            // Periodic status log (every 2s)
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log).count() > 2000) {
                RING_DBG("[drain_loop] iter=%lu pending=%lu scanned_complete=%lu "
                        "granted=%lu blocked=%lu old_pending=%lu "
                        "task_head=%lu task_tail=%lu payload_head=%lu "
                        "payload_tail=%lu committed=%lu should_flush=%d\n",
                        (unsigned long)loop_iter,
                        (unsigned long)pending_entries_,
                        (unsigned long)scanned_complete_,
                        (unsigned long)granted_count_,
                        (unsigned long)blocked_count_,
                        (unsigned long)old_pending_,
                        (unsigned long)cpu_task_head_,
                        (unsigned long)cpu_task_tail_,
                        (unsigned long)cpu_payload_head_,
                        (unsigned long)cpu_payload_tail_,
                        (unsigned long)cpu_payload_tail_committed_,
                        (int)should_flush());

                last_log = now;
            }

            if (should_flush()) {
                for (size_t i = 0; i < scanned_.size(); ++i) {
                    uint64_t ab = align_up(scanned_[i].tensor_total_bytes, PAYLOAD_ALIGN);
                    if (flush_bytes + ab > staging_.capacity()) break;
                    flush_bytes += ab;
                    flush_count++;
                }
                if (flush_count > 0) {
                    RING_DBG("[drain_flush] iter=%lu flush_count=%lu flush_bytes=%lu "
                            "scanned=%lu granted=%lu blocked=%lu staging_free=%lu\n",
                            (unsigned long)loop_iter, (unsigned long)flush_count,
                            (unsigned long)flush_bytes, (unsigned long)scanned_complete_,
                            (unsigned long)granted_count_, (unsigned long)blocked_count_,
                            (unsigned long)staging_.free_bytes());

                    flush_state_update(flush_count, flush_bytes);
                    fss_snapshot = forward_start_seq_;
                    needs_flush = true;
                }
            }
        }

        if (needs_flush) {
            {
                RING_DBG("[drain_flush] staging wait: need=%lu free=%lu cap=%lu\n",
                        (unsigned long)flush_bytes, (unsigned long)staging_.free_bytes(),
                        (unsigned long)staging_.capacity());

                std::unique_lock<std::mutex> lk(staging_mu_);
                staging_cv_.wait(lk, [&] {
                    return staging_.free_bytes() >= flush_bytes;
                });
                RING_DBG("[drain_flush] staging wait resolved\n");

            }

            RING_DBG("[drain_flush] enqueueing D2H\n"); 
            enqueue_d2h(flush_bytes);
            RING_DBG("[drain_flush] enqueueing H2D\n"); 
            enqueue_conditions_h2d();
            RING_DBG("[drain_flush] syncing drain stream\n"); 
            sync_stream();
            RING_DBG("[drain_flush] D2H+H2D synced\n");


            // OCC validation: re-check if prepare_forward leaked in during
            // D2H.  If forward_start_seq_ changed, prepare_forward ran with
            // stale committed tail.  Now D2H is done — update committed and
            // re-grant blocked hooks of the NEW forward.
            //
            // Dirty flag safety for the re-grant: prepare_forward already
            // finished and cleared dirty before this point.  It won't run
            // again until the current forward completes.  So the drain is
            // the sole dirty writer here.
            bool re_granted = false;
            {
                std::lock_guard<std::mutex> lk(mgmt_mu_);
                cpu_payload_tail_committed_ = cpu_payload_tail_;
                if (forward_start_seq_ != fss_snapshot) {
                    grant_next_hooks();
                    re_granted = dirty_;
                }
            }
            if (re_granted) {
                enqueue_conditions_h2d();
                sync_stream();
            }

            submit_to_p2p(flush_count, flush_bytes);
            {
                std::lock_guard<std::mutex> lk(mgmt_mu_);
                trim_scanned(flush_count, flush_bytes);
            }
        }

        {
            std::unique_lock<std::mutex> lk(mu_);
            auto pred = [this] {
                return notified_ || !running_.load(std::memory_order_relaxed);
            };
            cv_.wait_for(lk, std::chrono::microseconds(cfg_.drain_poll_timeout_us), pred);
            notified_ = false;
        }
    }

    // Final flush
    cudaDeviceSynchronize();
    for (;;) {
        uint64_t flush_count = 0, flush_bytes = 0;
        {
            std::lock_guard<std::mutex> lk(mgmt_mu_);
            scan_ready();
            if (pending_entries_ == 0) break;
            for (size_t i = 0; i < scanned_.size(); ++i) {
                uint64_t ab = align_up(scanned_[i].tensor_total_bytes, PAYLOAD_ALIGN);
                if (flush_bytes + ab > staging_.capacity()) break;
                flush_bytes += ab;
                flush_count++;
            }
            if (flush_count == 0) break;
            flush_state_update(flush_count, flush_bytes);
        }
        {
            std::unique_lock<std::mutex> lk(staging_mu_);
            staging_cv_.wait(lk, [&] { return staging_.free_bytes() >= flush_bytes; });
        }
        enqueue_d2h(flush_bytes);
        sync_stream();
        {
            std::lock_guard<std::mutex> lk(mgmt_mu_);
            cpu_payload_tail_committed_ = cpu_payload_tail_;
        }
        submit_to_p2p(flush_count, flush_bytes);
        {
            std::lock_guard<std::mutex> lk(mgmt_mu_);
            trim_scanned(flush_count, flush_bytes);
        }
    }
}

// ---------------------------------------------------------------------------
// scan_ready — under mgmt_mu_.
//
// INVARIANT: large tensor entries cannot coexist with prepare_forward
// contention on mgmt_mu_.
//
// Proof: large tensor bypass blocks the main stream at
// cuStreamWaitValue32(==0) until the drain acks (writes COND_RESET via
// H2D).  The forward cannot complete until all large tensor acks are done.
// prepare_forward only runs after the forward returns to Python.
// Therefore, by the time prepare_forward tries to acquire mgmt_mu_, no
// large tensor entries remain in the queue — the drain already processed
// and acked them all.
//
// Consequence: the inline batch_flush + handle_large_tensor calls (which
// do CUDA sync under mgmt_mu_) cannot block prepare_forward.  Normal
// entries in scan_ready are just fast DRAM reads + CPU state updates.
// ---------------------------------------------------------------------------
void DrainThread::scan_ready() {
    const uint64_t task_cap = ring_.task_cap;

    while (true) {
        if (pending_entries_ >= task_cap) break;
        if (!task_cpu_ready(ring_.task_entries, task_cap, visible_head_)) break;

        const uint64_t idx = visible_head_ % task_cap;
        TaskEntry ec = ring_.task_entries[idx];

        if (ec.device_src_ptr != nullptr) {
            // Large tensor — see INVARIANT above.
            // Must flush ALL pending normal entries to maintain FIFO ordering
            // with the TensorMeta FIFO.  Loop until scanned_ is empty.
            while (pending_entries_ > 0) {
                uint64_t fc = 0, fb = 0;
                for (size_t i = 0; i < scanned_.size(); ++i) {
                    uint64_t ab = align_up(scanned_[i].tensor_total_bytes, PAYLOAD_ALIGN);
                    if (fb + ab > staging_.capacity()) break;
                    fb += ab; fc++;
                }
                if (fc == 0) break;
                RING_DBG("[scan_ready inline] flushing %lu/%lu entries before large tensor\n",
                        (unsigned long)fc, (unsigned long)pending_entries_);

                flush_state_update(fc, fb);
                {
                    std::unique_lock<std::mutex> lk(staging_mu_);
                    staging_cv_.wait(lk, [&] { return staging_.free_bytes() >= fb; });
                }
                enqueue_d2h(fb);
                enqueue_conditions_h2d();
                sync_stream();
                cpu_payload_tail_committed_ = cpu_payload_tail_;
                submit_to_p2p(fc, fb);
                trim_scanned(fc, fb);  // already under mgmt_mu_
            }
            RING_DBG("[scan_ready] handling large tensor at visible_head=%lu\n",
                    (unsigned long)visible_head_);

            handle_large_tensor();
            continue;
        }

        scanned_.push_back(ec);
        pending_entries_++;
        pending_bytes_ += align_up(ec.tensor_total_bytes, PAYLOAD_ALIGN);

        if (visible_head_ < forward_start_seq_) {
            old_pending_++;
        } else {
            scanned_complete_++;
        }
        visible_head_++;

        if (!has_complete_time_) {
            first_complete_time_ = std::chrono::steady_clock::now();
            has_complete_time_ = true;
        }
    }
}

// ---------------------------------------------------------------------------
bool DrainThread::should_flush() const {
    if (pending_entries_ == 0) return false;

    const uint64_t fe = pending_entries_;
    const uint64_t fb = pending_bytes_;
    const uint64_t task_cap    = ring_.task_cap;
    const uint64_t payload_cap = ring_.payload_cap;
    const auto& fc = cfg_.drain_flush;

    // Forced: all current-forward granted hooks produced, blocked waiting.
    // GPU cannot produce more — must flush to free resources.
    if (granted_count_ > 0 &&
        scanned_complete_ == granted_count_ &&
        blocked_count_ > 0)
    {
        return true;
    }

    if (fe >= task_cap) return true;
    if (fb >= payload_cap) return true;
    if (fc.task_ratio > 0.0f &&
        fe >= static_cast<uint64_t>(fc.task_ratio * task_cap)) return true;
    if (fc.payload_ratio > 0.0f &&
        fb >= static_cast<uint64_t>(fc.payload_ratio * payload_cap)) return true;
    if (fc.entry_threshold > 0 && fe >= fc.entry_threshold) return true;
    if (fc.byte_threshold > 0 && fb >= fc.byte_threshold) return true;
    if (fc.timeout_us > 0 && has_complete_time_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - first_complete_time_).count();
        if (static_cast<uint64_t>(elapsed) >= fc.timeout_us) return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// flush_state_update — CPU-only state changes under mgmt_mu_.
// ---------------------------------------------------------------------------
void DrainThread::flush_state_update(uint64_t flush_count, uint64_t flush_bytes) {
    // Safe before D2H: task_entries (managed mem) independent of payload_buf.
    // Entry data already in scanned_.  Producer gated by cuStreamWaitValue32.
    for (uint64_t i = 0; i < flush_count; ++i) {
        task_release_cpu(ring_.task_entries, ring_.task_cap, cpu_task_tail_);
        ++cpu_task_tail_;

        if (old_pending_ > 0) {
            old_pending_--;
        } else {
            // No condition reset — kernel wrote COND_RESET on device.
            drain_hook_idx_++;
            // Do NOT decrement scanned_complete_: it counts total
            // current-forward entries ever scanned (monotonic).  The forced
            // flush (scanned_complete_ == granted_count_) compares total
            // scanned vs total granted — both are monotonic.
        }
    }
    cpu_payload_tail_ += flush_bytes;

    // Grant using pending tail — safe because drain's condition H2D is on
    // the same stream as D2H (producer can't proceed until D2H completes).
    grant_next_hooks();
}

// ---------------------------------------------------------------------------
void DrainThread::grant_next_hooks() {
    uint32_t num = static_cast<uint32_t>(hook_tensor_bytes_.size());
    if (num > num_hooks_) num = num_hooks_;

    uint64_t available_tasks   = ring_.task_cap - (cpu_task_head_ - cpu_task_tail_);
    uint64_t available_payload = ring_.payload_cap - (cpu_payload_head_ - cpu_payload_tail_);

    for (uint32_t i = next_ungrant_idx_; i < num; ++i) {
        uint64_t raw_bytes    = hook_tensor_bytes_[i];
        uint64_t padded_bytes = align_up(raw_bytes, PAYLOAD_ALIGN);
        // Bypass if tensor exceeds either ring or staging capacity.
        bool fits_normal      = (padded_bytes <= ring_.payload_cap) &&
                                (padded_bytes <= staging_.capacity());

        if (fits_normal) {
            if (available_tasks >= 1 && available_payload >= padded_bytes) {
                h_condition_[i] = COND_GRANT_FULL; mark_dirty(i);
                available_tasks   -= 1;
                available_payload -= padded_bytes;
                cpu_task_head_    += 1;
                cpu_payload_head_ += padded_bytes;
                granted_count_++;
                blocked_count_--;
                next_ungrant_idx_ = i + 1;
            } else {
                break;
            }
        } else {
            if (available_tasks >= 1) {
                h_condition_[i] = COND_GRANT_TASK_ONLY; mark_dirty(i);
                available_tasks -= 1;
                cpu_task_head_  += 1;
                granted_count_++;
                blocked_count_--;
                next_ungrant_idx_ = i + 1;
            } else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
void DrainThread::enqueue_d2h(uint64_t flush_bytes) {
    if (flush_bytes == 0) return;
    const uint64_t gpu_cap = ring_.payload_cap;
    const uint64_t stg_cap = staging_.capacity();
    uint64_t src_start = cpu_payload_tail_ - flush_bytes;
    uint64_t gpu_cursor = src_start % gpu_cap;
    uint64_t stg_cursor = staging_.head() % stg_cap;
    uint64_t remaining  = flush_bytes;
    int chunk_idx = 0;

    while (remaining > 0) {
        uint64_t gpu_avail = gpu_cap - gpu_cursor;
        uint64_t stg_avail = stg_cap - stg_cursor;
        uint64_t chunk = std::min({remaining, gpu_avail, stg_avail});
        RING_DBG("[enqueue_d2h] chunk=%d src_off=%lu dst_off=%lu "
                "size=%lu remaining=%lu gpu_cap=%lu stg_cap=%lu\n",
                chunk_idx, (unsigned long)gpu_cursor, (unsigned long)stg_cursor,
                (unsigned long)chunk, (unsigned long)remaining,
                (unsigned long)gpu_cap, (unsigned long)stg_cap);

        cudaError_t err = cudaMemcpyAsync(staging_.base() + stg_cursor,
                        ring_.payload_buf + gpu_cursor,
                        chunk, cudaMemcpyDeviceToHost, stream_);
        if (err != cudaSuccess) {
            RING_DBG("[enqueue_d2h] cudaMemcpyAsync FAILED: %s\n",
                    cudaGetErrorString(err));

        }
        RING_DBG("[enqueue_d2h] chunk=%d enqueued OK\n", chunk_idx);

        remaining  -= chunk;
        gpu_cursor  = (gpu_cursor + chunk) % gpu_cap;
        stg_cursor  = (stg_cursor + chunk) % stg_cap;
        chunk_idx++;
    }
}

// ---------------------------------------------------------------------------
// submit_to_p2p — push DrainTasks to p2p queue.  Uses queue_mu_/pop_mu_
// only (NOT mgmt_mu_).  Safe to call with or without mgmt_mu_ held.
// ---------------------------------------------------------------------------
void DrainThread::submit_to_p2p(uint64_t flush_count, uint64_t flush_bytes) {
    uint64_t cumulative = 0;
    const uint64_t staging_batch_start = staging_.head();

    for (uint64_t i = 0; i < flush_count; ++i) {
        const TaskEntry& ec = scanned_[i];
        uint64_t data_len = ec.payload_len1 + ec.payload_len2;
        uint64_t alloc    = align_up(ec.tensor_total_bytes, PAYLOAD_ALIGN);

        DrainTask task{};
        task.tensor_total_bytes = ec.tensor_total_bytes;
        task.alloc_bytes        = alloc;

        if (data_len > 0) {
            uint64_t staging_logical = staging_batch_start + cumulative;
            uint64_t staging_phys = staging_logical % staging_.capacity();

            task.data_ptr1 = staging_.base() + staging_phys;
            if (staging_phys + data_len <= staging_.capacity()) {
                task.data_len1 = data_len;
                task.data_ptr2 = nullptr;
                task.data_len2 = 0;
            } else {
                task.data_len1 = staging_.capacity() - staging_phys;
                task.data_ptr2 = staging_.base();
                task.data_len2 = data_len - task.data_len1;
            }
            cumulative += alloc;
        }

        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            task_queue_.push_back(std::move(task));
        }
        {
            std::lock_guard<std::mutex> lk(pop_mu_);
            can_pop_count_ += 1;
        }
        pop_cv_.notify_one();
    }

    staging_.advance_head(flush_bytes);
}

// ---------------------------------------------------------------------------
// trim_scanned — update scanned_/pending state after flush.
// Caller MUST hold mgmt_mu_.
// ---------------------------------------------------------------------------
void DrainThread::trim_scanned(uint64_t flush_count, uint64_t flush_bytes) {
    for (uint64_t i = 0; i < flush_count; ++i) {
        scanned_.pop_front();
    }
    pending_entries_ -= flush_count;
    pending_bytes_   -= flush_bytes;
    has_complete_time_ = false;
    if (pending_entries_ > 0) {
        first_complete_time_ = std::chrono::steady_clock::now();
        has_complete_time_ = true;
    }
}

// ---------------------------------------------------------------------------
// handle_large_tensor — under mgmt_mu_ (see scan_ready INVARIANT).
// ---------------------------------------------------------------------------
void DrainThread::handle_large_tensor() {
    bool is_old = (visible_head_ < forward_start_seq_);

    const uint64_t idx = visible_head_ % ring_.task_cap;
    TaskEntry ec = ring_.task_entries[idx];

    const uint64_t total_bytes = ec.tensor_total_bytes;
    const uint8_t* device_ptr  = ec.device_src_ptr;

    auto tensor = at::empty({static_cast<int64_t>(total_bytes)},
                            at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    uint8_t* dst = tensor.data_ptr<uint8_t>();

    cudaMemcpyAsync(dst, device_ptr, total_bytes,
                    cudaMemcpyDeviceToHost, stream_);

    // Safe while D2H in flight: task_entries independent of device_src_ptr,
    // device memory safe (main stream at waitValue32(==0)), no payload freed.
    task_release_cpu(ring_.task_entries, ring_.task_cap, visible_head_);
    ++cpu_task_tail_;
    visible_head_++;

    if (!is_old) {
        scanned_complete_++;
        if (drain_hook_idx_ < num_hooks_) {
            h_condition_[drain_hook_idx_] = COND_RESET; mark_dirty(drain_hook_idx_);
        }
        drain_hook_idx_++;
    }

    grant_next_hooks();
    enqueue_conditions_h2d();  // after D2H on same stream → ordered
    sync_stream();
    cpu_payload_tail_committed_ = cpu_payload_tail_;

    {
        std::unique_lock<std::mutex> lk(bypass_mu_);
        bypass_cv_.wait(lk, [&] {
            return in_flight_bypass_bytes_ == 0 ||
                   in_flight_bypass_bytes_ + total_bytes <= cfg_.bypass_budget_bytes;
        });
        in_flight_bypass_bytes_ += total_bytes;
    }

    DrainTask task{};
    task.tensor_total_bytes = total_bytes;
    task.large_tensor       = std::move(tensor);

    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        task_queue_.push_back(std::move(task));
    }
    {
        std::lock_guard<std::mutex> lk(pop_mu_);
        can_pop_count_ += 1;
    }
    pop_cv_.notify_one();
}

}  // namespace ring
