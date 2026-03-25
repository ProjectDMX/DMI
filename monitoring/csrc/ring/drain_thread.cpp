// ring/drain_thread.cpp -- Batch drain thread implementation.
// Compiled with g++ (not nvcc); uses CUDA runtime C API via -lcudart.
//
// No condition tensor, no cuStreamWaitValue32, no large-tensor bypass.
// Space is guaranteed by the pre-forward capacity check in Python.

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
// force_flush_and_wait -- signal drain thread to flush, block until done.
//
// Called from Python thread with GIL released.  Caller must have done
// cudaStreamSynchronize(main_stream) first so all GPU writes are visible.
// ---------------------------------------------------------------------------
void DrainThread::force_flush_and_wait() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        flush_requested_ = true;
        flush_done_      = false;
        notified_        = true;
    }
    cv_.notify_one();  // wake drain thread

    // Block until drain thread completes the flush
    std::unique_lock<std::mutex> lk(mu_);
    flush_done_cv_.wait(lk, [this] { return flush_done_; });
}

uint64_t DrainThread::force_flush_and_wait_timed() {
    auto start = std::chrono::steady_clock::now();
    force_flush_and_wait();
    const uint64_t wait_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start
        ).count()
    );
    {
        std::lock_guard<std::mutex> lk(mgmt_mu_);
        last_force_flush_wait_us_ = wait_us;
    }
    return wait_us;
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

// ---------------------------------------------------------------------------
// Capacity query accessors
// ---------------------------------------------------------------------------
uint64_t DrainThread::cpu_payload_head() const {
    return cpu_payload_head_;
}

uint64_t DrainThread::cpu_payload_tail_committed() const {
    return cpu_payload_tail_committed_;
}

FlushStats DrainThread::get_stats() const {
    std::lock_guard<std::mutex> lk(mgmt_mu_);
    FlushStats stats{};
    stats.pending_entries = pending_entries_;
    stats.pending_bytes = pending_bytes_;
    stats.cpu_payload_head = cpu_payload_head_;
    stats.cpu_payload_tail_committed = cpu_payload_tail_committed_;
    stats.total_flushes = total_flushes_;
    stats.last_flush_entries = last_flush_entries_;
    stats.last_flush_bytes = last_flush_bytes_;
    stats.last_flush_complete_monotonic_us = last_flush_complete_monotonic_us_;
    stats.last_force_flush_wait_us = last_force_flush_wait_us_;
    return stats;
}

// ---------------------------------------------------------------------------
// reserve -- pre-allocate ring space for the next step.
// Called from prepare_step after confirming space is available.
// ---------------------------------------------------------------------------
void DrainThread::reserve(uint64_t payload_bytes, uint32_t num_tasks) {
    std::lock_guard<std::mutex> lk(mgmt_mu_);
    cpu_payload_head_ += payload_bytes;
    cpu_task_head_    += num_tasks;
}

// ---------------------------------------------------------------------------
// submit_cpu_direct -- submit a CPU-direct tensor to drain -> p2p pipeline.
// ---------------------------------------------------------------------------
void DrainThread::submit_cpu_direct(at::Tensor cpu_tensor, uint64_t tensor_bytes) {
    DrainTask task{};
    task.tensor_total_bytes = tensor_bytes;
    task.cpu_paged_tensor   = std::move(cpu_tensor);

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

// ---------------------------------------------------------------------------
// do_full_flush -- drain all pending entries.  Called by drain thread only.
// ---------------------------------------------------------------------------
void DrainThread::do_full_flush() {
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
            total_flushes_ += 1;
            last_flush_entries_ = flush_count;
            last_flush_bytes_ = flush_bytes;
            last_flush_complete_monotonic_us_ = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count()
            );
        }
        submit_to_p2p(flush_count, flush_bytes);
        {
            std::lock_guard<std::mutex> lk(mgmt_mu_);
            trim_scanned(flush_count, flush_bytes);
        }
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
void DrainThread::loop() {
    RING_DBG("[drain_loop] started, poll_timeout=%lu us\n",
            (unsigned long)cfg_.drain_poll_timeout_us);

    uint64_t loop_iter = 0;

    while (running_.load(std::memory_order_relaxed)) {
        ++loop_iter;

        // Check for force-flush request
        bool flush_now = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (flush_requested_) {
                flush_now = true;
                flush_requested_ = false;
            }
        }

        if (flush_now) {
            do_full_flush();
            {
                std::lock_guard<std::mutex> lk(mu_);
                flush_done_ = true;
            }
            flush_done_cv_.notify_one();
            continue;  // skip normal sleep, re-check immediately
        }

        uint64_t flush_count = 0, flush_bytes = 0;
        bool needs_flush = false;

        {
            std::lock_guard<std::mutex> lk(mgmt_mu_);
            scan_ready();

            if (should_flush()) {
                for (size_t i = 0; i < scanned_.size(); ++i) {
                    uint64_t ab = align_up(scanned_[i].tensor_total_bytes, PAYLOAD_ALIGN);
                    if (flush_bytes + ab > staging_.capacity()) break;
                    flush_bytes += ab;
                    flush_count++;
                }
                if (flush_count > 0) {
                    RING_DBG("[drain_flush] iter=%lu flush_count=%lu flush_bytes=%lu "
                            "pending=%lu staging_free=%lu\n",
                            (unsigned long)loop_iter, (unsigned long)flush_count,
                            (unsigned long)flush_bytes, (unsigned long)pending_entries_,
                            (unsigned long)staging_.free_bytes());

                    flush_state_update(flush_count, flush_bytes);
                    needs_flush = true;
                }
            }
        }

        if (needs_flush) {
            {
                std::unique_lock<std::mutex> lk(staging_mu_);
                staging_cv_.wait(lk, [&] {
                    return staging_.free_bytes() >= flush_bytes;
                });
            }

            enqueue_d2h(flush_bytes);
            sync_stream();

            {
                std::lock_guard<std::mutex> lk(mgmt_mu_);
                cpu_payload_tail_committed_ = cpu_payload_tail_;
                total_flushes_ += 1;
                last_flush_entries_ = flush_count;
                last_flush_bytes_ = flush_bytes;
                last_flush_complete_monotonic_us_ = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count()
                );
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
                return notified_ || flush_requested_ ||
                       !running_.load(std::memory_order_relaxed);
            };
            cv_.wait_for(lk, std::chrono::microseconds(cfg_.drain_poll_timeout_us), pred);
            notified_ = false;
        }
    }

    // Final flush
    cudaDeviceSynchronize();
    do_full_flush();
}

// ---------------------------------------------------------------------------
// scan_ready -- under mgmt_mu_.
// ---------------------------------------------------------------------------
void DrainThread::scan_ready() {
    const uint64_t task_cap = ring_.task_cap;

    while (true) {
        if (pending_entries_ >= task_cap) break;
        if (!task_cpu_ready(ring_.task_entries, task_cap, visible_head_)) break;

        const uint64_t idx = visible_head_ % task_cap;
        TaskEntry ec = ring_.task_entries[idx];

        scanned_.push_back(ec);
        pending_entries_++;
        pending_bytes_ += align_up(ec.tensor_total_bytes, PAYLOAD_ALIGN);
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
// flush_state_update -- CPU-only state changes under mgmt_mu_.
// ---------------------------------------------------------------------------
void DrainThread::flush_state_update(uint64_t flush_count, uint64_t flush_bytes) {
    for (uint64_t i = 0; i < flush_count; ++i) {
        task_release_cpu(ring_.task_entries, ring_.task_cap, cpu_task_tail_);
        ++cpu_task_tail_;
    }
    cpu_payload_tail_ += flush_bytes;
}

void DrainThread::sync_stream() {
    cudaStreamSynchronize(stream_);
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
                "size=%lu remaining=%lu\n",
                chunk_idx, (unsigned long)gpu_cursor, (unsigned long)stg_cursor,
                (unsigned long)chunk, (unsigned long)remaining);

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
// submit_to_p2p -- push DrainTasks to p2p queue.  Uses queue_mu_/pop_mu_
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
// trim_scanned -- update scanned_/pending state after flush.
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

}  // namespace ring
