// ring/drain_thread.cpp — Batch drain thread implementation.
// Compiled with g++ (not nvcc); uses CUDA runtime C API via -lcudart.

#include "drain_thread.h"
#include "task_ring.cuh"   // task_cpu_ready, task_release_cpu
#include "task_entry.h"

#include <ATen/ATen.h>
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

// ---------------------------------------------------------------------------
void DrainThread::start() {
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread([this] { loop(); });
}

void DrainThread::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();       // wake drain thread from sleep to enter final flush
    // Do NOT notify pop_cv_ here — the final flush in loop() may push more
    // tasks.  The p2p thread is signalled separately via signal_p2p_stop()
    // after drain thread join completes.
    if (thread_.joinable()) thread_.join();
}

// ---------------------------------------------------------------------------
void DrainThread::notify() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        notified_ = true;
    }
    cv_.notify_one();
}

/*static*/ void CUDART_CB DrainThread::hostfunc_cb(void* arg) {
    static_cast<DrainThread*>(arg)->notify();
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
// Main loop
// ---------------------------------------------------------------------------
void DrainThread::loop() {
    uint64_t loop_iter = 0;
    uint64_t last_log_iter = 0;
    auto last_log_time = std::chrono::steady_clock::now();

    while (running_.load(std::memory_order_relaxed)) {
        ++loop_iter;
        uint64_t pre_pending = pending_entries_;
        scan_ready();
        uint64_t post_pending = pending_entries_;

        if (should_flush()) {
            fprintf(stderr, "[drain_diag] iter=%lu flush: entries=%lu bytes=%lu tail=%lu\n",
                    (unsigned long)loop_iter, (unsigned long)flushable_entries_,
                    (unsigned long)flushable_bytes_, (unsigned long)task_tail_local_);
            batch_flush();
        }

        // Periodic diagnostic: if >2s since last log, report state
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count();
        if (elapsed > 2000) {
            fprintf(stderr, "[drain_diag] STALL? iter=%lu pending=%lu scanned_new=%lu "
                    "last_complete_idx=%ld visible_head=%lu task_tail=%lu "
                    "staging_free=%lu staging_cap=%lu "
                    "flushable_entries=%lu flushable_bytes=%lu "
                    "task_cap=%lu payload_cap=%lu should_flush=%d\n",
                    (unsigned long)loop_iter,
                    (unsigned long)pending_entries_,
                    (unsigned long)(post_pending - pre_pending),
                    (long)last_complete_idx_,
                    (unsigned long)visible_head_,
                    (unsigned long)task_tail_local_,
                    (unsigned long)staging_.free_bytes(),
                    (unsigned long)staging_.capacity(),
                    (unsigned long)flushable_entries_,
                    (unsigned long)flushable_bytes_,
                    (unsigned long)ring_.task_cap,
                    (unsigned long)ring_.payload_cap,
                    (int)should_flush());
            last_log_time = now;
        }

        if (pending_entries_ == 0) {
            // No entries scanned: sleep until GPU produces something.
            std::unique_lock<std::mutex> lk(mu_);
            auto pred = [this] {
                return notified_ || !running_.load(std::memory_order_relaxed);
            };
            if (cfg_.drain_poll_timeout_us > 0) {
                cv_.wait_for(lk, std::chrono::microseconds(cfg_.drain_poll_timeout_us), pred);
            } else {
                cv_.wait(lk, pred);
            }
            notified_ = false;
        } else {
            // Entries scanned but flush threshold not met yet (e.g. last
            // tensor incomplete — waiting for its IS_LAST chunk to arrive
            // from the GPU).  Yield briefly to avoid busy-spinning.
            struct timespec ts{0, 1000};  // 1 us
            nanosleep(&ts, nullptr);
        }
    }

    // Final flush: drain all remaining entries.
    cudaDeviceSynchronize();
    for (;;) {
        scan_ready();
        if (pending_entries_ == 0) break;
        if (last_complete_idx_ < 0) {
            fprintf(stderr, "[drain_thread] WARNING: %lu incomplete entries at shutdown\n",
                    (unsigned long)pending_entries_);
            break;
        }
        batch_flush();
    }
    fprintf(stderr, "[drain_thread] final tail=%lu\n", (unsigned long)task_tail_local_);
}

// ---------------------------------------------------------------------------
// scan_ready — scan GPU task queue for ready entries.
//
// Maintains flushable_bytes_ / flushable_entries_ incrementally:
// - Every scanned entry's payload_alloc_bytes is added to a running sum
//   (scan_bytes_accum_).
// - When an IS_LAST entry is seen, we snapshot the running sum and entry
//   count as flushable_bytes_ / flushable_entries_.
// - After batch_flush(), the flushed portion is subtracted (see batch_flush).
// ---------------------------------------------------------------------------
void DrainThread::scan_ready() {
    const uint64_t task_cap = ring_.task_cap;

    while (true) {
        // Stop if entire ring scanned (cannot read further without wrapping
        // into unreleased slots).
        if (pending_entries_ >= task_cap) break;

        if (!task_cpu_ready(ring_.task_entries, task_cap, visible_head_)) break;

        const uint64_t idx = visible_head_ % task_cap;
        TaskEntry ec = ring_.task_entries[idx];  // full copy from managed memory

        // Large tensor detection: if IS_FIRST with tensor > staging capacity.
        // Must flush pending complete tensors first: the p2p thread expects
        // tasks in FIFO order, and the large tensor bypass processes entries
        // inline (not via staging), so earlier staged tensors must be pushed
        // to the task queue before the bypass tensor.
        if ((ec.flags & TASK_FLAG_IS_FIRST) &&
            !(ec.flags & TASK_FLAG_IS_DROP) &&
            ec.tensor_total_padded_bytes > staging_.capacity())
        {
            if (last_complete_idx_ >= 0) {
                batch_flush();
            }
            handle_large_tensor();
            continue;  // Re-enter scan loop
        }

        scanned_.push_back(ec);
        pending_entries_++;
        if (!(ec.flags & TASK_FLAG_IS_DROP)) {
            uint64_t ab = ec.payload_alloc_bytes;
            pending_bytes_ += ab;
            scan_bytes_accum_ += ab;
        }
        visible_head_++;

        if (ec.flags & TASK_FLAG_IS_LAST) {
            last_complete_idx_ = static_cast<int64_t>(scanned_.size()) - 1;
            // Snapshot running sums as flushable
            flushable_bytes_   = scan_bytes_accum_;
            flushable_entries_ = static_cast<uint64_t>(scanned_.size());
            // Update capped counters if this prefix still fits in staging
            if (scan_bytes_accum_ <= staging_.capacity()) {
                capped_flush_bytes_   = scan_bytes_accum_;
                capped_flush_entries_ = static_cast<uint64_t>(scanned_.size());
            }
            // Record when the first complete tensor became pending
            if (!has_complete_time_) {
                first_complete_time_ = std::chrono::steady_clock::now();
                has_complete_time_ = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// should_flush — check if any flush condition is met.
// Uses pre-computed flushable_bytes_ / flushable_entries_ (no loop).
// ---------------------------------------------------------------------------
bool DrainThread::should_flush() const {
    if (last_complete_idx_ < 0) return false;

    const uint64_t fe = flushable_entries_;
    const uint64_t fb = flushable_bytes_;
    const uint64_t task_cap    = ring_.task_cap;
    const uint64_t payload_cap = ring_.payload_cap;
    const auto& fc = cfg_.drain_flush;

    // 1. Force: entry full
    if (fe >= task_cap) return true;
    // 2. Force: payload full
    if (fb >= payload_cap) return true;
    // 3. Task ratio
    if (fc.task_ratio > 0.0f &&
        fe >= static_cast<uint64_t>(fc.task_ratio * task_cap)) return true;
    // 4. Payload ratio
    if (fc.payload_ratio > 0.0f &&
        fb >= static_cast<uint64_t>(fc.payload_ratio * payload_cap)) return true;
    // 5. Entry threshold
    if (fc.entry_threshold > 0 && fe >= fc.entry_threshold) return true;
    // 6. Byte threshold
    if (fc.byte_threshold > 0 && fb >= fc.byte_threshold) return true;
    // 7. Timeout: complete tensor pending longer than timeout_us
    if (fc.timeout_us > 0 && has_complete_time_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - first_complete_time_).count();
        if (static_cast<uint64_t>(elapsed) >= fc.timeout_us) return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// batch_flush — D2H copy + release GPU + push tasks + notify p2p.
// Uses pre-computed flushable_bytes_ / flushable_entries_ (no loop to sum).
// ---------------------------------------------------------------------------
void DrainThread::batch_flush() {
    const uint64_t flush_count = capped_flush_entries_;
    const uint64_t flush_bytes = capped_flush_bytes_;

    fprintf(stderr, "[drain_diag] batch_flush enter: capped_entries=%lu capped_bytes=%lu "
            "flushable_entries=%lu flushable_bytes=%lu staging_free=%lu staging_cap=%lu\n",
            (unsigned long)flush_count, (unsigned long)flush_bytes,
            (unsigned long)flushable_entries_, (unsigned long)flushable_bytes_,
            (unsigned long)staging_.free_bytes(), (unsigned long)staging_.capacity());

    // --- 1. Backpressure: wait for staging space ---
    if (flush_bytes > 0) {
        uint64_t free_before = staging_.free_bytes();
        if (free_before < flush_bytes) {
            fprintf(stderr, "[drain_diag] staging backpressure: need=%lu free=%lu cap=%lu\n",
                    (unsigned long)flush_bytes, (unsigned long)free_before,
                    (unsigned long)staging_.capacity());
        }
        std::unique_lock<std::mutex> lk(staging_mu_);
        staging_cv_.wait(lk, [&] {
            return staging_.free_bytes() >= flush_bytes;
        });
        fprintf(stderr, "[drain_diag] staging backpressure resolved\n");
        fflush(stderr);
    }

    // --- 2. Batch D2H (separate CUDA stream) ---
    // GPU payload region: [payload_tail_local_, +flush_bytes) in circular buffer.
    // Staging destination: [staging_.head(), +flush_bytes) in circular buffer.
    if (flush_bytes > 0) {
        // Log memory type of src (payload_buf) and dst (staging)
        {
            cudaPointerAttributes src_attr{}, dst_attr{};
            cudaPointerGetAttributes(&src_attr, ring_.payload_buf);
            cudaPointerGetAttributes(&dst_attr, staging_.base());
            // type: 0=unregistered, 1=host, 2=device, 3=managed
            fprintf(stderr, "[drain_diag] memcpy src type=%d device=%d ptr=%p | "
                    "dst type=%d device=%d ptr=%p\n",
                    (int)src_attr.type, src_attr.device, (void*)ring_.payload_buf,
                    (int)dst_attr.type, dst_attr.device, (void*)staging_.base());
            fflush(stderr);
        }
    }
    fprintf(stderr, "[drain_diag] step2: PRE cudaMemcpyAsync flush_bytes=%lu\n",
            (unsigned long)flush_bytes);
    fflush(stderr);
    if (flush_bytes > 0) {
        const uint64_t gpu_cap    = ring_.payload_cap;
        const uint64_t stg_cap    = staging_.capacity();
        uint64_t gpu_cursor = payload_tail_local_ % gpu_cap;
        uint64_t stg_cursor = staging_.head() % stg_cap;
        uint64_t remaining  = flush_bytes;

        while (remaining > 0) {
            
            uint64_t gpu_avail = gpu_cap - gpu_cursor;
            uint64_t stg_avail = stg_cap - stg_cursor;
            uint64_t chunk = std::min({remaining, gpu_avail, stg_avail});
            fprintf(stderr, "[drain_diag] step2: Just before one cudaMemcpyAsync\n");
            fflush(stderr);
            cudaMemcpyAsync(staging_.base() + stg_cursor,
                            ring_.payload_buf + gpu_cursor,
                            chunk, cudaMemcpyDeviceToHost, stream_);
            fprintf(stderr, "[drain_diag] step2: Just after one cudaMemcpyAsync\n");
            fflush(stderr);

            remaining  -= chunk;
            gpu_cursor  = (gpu_cursor + chunk) % gpu_cap;
            stg_cursor  = (stg_cursor + chunk) % stg_cap;
        }
    }

    fprintf(stderr, "[drain_diag] step2: POST cudaMemcpyAsync\n");
    fflush(stderr);

    // --- 3. cudaStreamSynchronize (D2H stream) ---
    if (flush_bytes > 0) {
        fprintf(stderr, "[drain_diag] step3: PRE cudaStreamSync\n");
        fflush(stderr);
        cudaStreamSynchronize(stream_);
        fprintf(stderr, "[drain_diag] step3: POST cudaStreamSync\n");
        fflush(stderr);
    }

    // --- 4. Release GPU resources (flushed entries only) ---
    uint64_t released_payload = 0;
    for (uint64_t i = 0; i < flush_count; ++i) {
        task_release_cpu(ring_.task_entries, ring_.task_cap, task_tail_local_);
        ++task_tail_local_;

        if (!(scanned_[i].flags & TASK_FLAG_IS_DROP)) {
            released_payload += scanned_[i].payload_alloc_bytes;
        }
    }
    __atomic_store_n(ring_.task_tail, task_tail_local_, __ATOMIC_RELEASE);
    if (released_payload > 0) {
        payload_tail_local_ += released_payload;
        __atomic_store_n(ring_.payload_tail, payload_tail_local_, __ATOMIC_RELEASE);
    }
    ++heartbeat_local_;
    __atomic_store_n(ring_.consumer_heartbeat, heartbeat_local_, __ATOMIC_RELEASE);

    // --- 5. Construct and push DrainTasks ---
    uint64_t cumulative = 0;
    const uint64_t staging_batch_start = staging_.head();

    for (uint64_t i = 0; i < flush_count; ++i) {
        const TaskEntry& ec = scanned_[i];
        DrainTask task{};
        task.logical_task_id    = ec.logical_task_id;
        task.chunk_offset_bytes = ec.chunk_offset_bytes;
        task.tensor_total_bytes = ec.tensor_total_bytes;
        task.hook_type          = ec.hook_type;
        task.hook_id            = ec.hook_id;
        task.flags              = ec.flags;
        task.alloc_bytes        = (ec.flags & TASK_FLAG_IS_DROP) ? 0 : ec.payload_alloc_bytes;

        if (!(ec.flags & TASK_FLAG_IS_DROP) && (ec.payload_len1 + ec.payload_len2) > 0) {
            uint64_t data_len = ec.payload_len1 + ec.payload_len2;
            uint64_t staging_logical = staging_batch_start + cumulative;
            uint64_t staging_phys = staging_logical % staging_.capacity();

            task.data_ptr1 = staging_.base() + staging_phys;
            if (staging_phys + data_len <= staging_.capacity()) {
                task.data_len1 = data_len;
                task.data_ptr2 = nullptr;
                task.data_len2 = 0;
            } else {
                task.data_len1 = staging_.capacity() - staging_phys;
                task.data_ptr2 = staging_.base();  // wrap to start
                task.data_len2 = data_len - task.data_len1;
            }
            cumulative += ec.payload_alloc_bytes;
        }

        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            task_queue_.push_back(std::move(task));
        }

        if (ec.flags & TASK_FLAG_IS_LAST) {
            {
                std::lock_guard<std::mutex> lk(pop_mu_);
                can_pop_count_ += pending_incomplete_cnt_ + 1;
            }
            pop_cv_.notify_one();
            pending_incomplete_cnt_ = 0;
        } else {
            pending_incomplete_cnt_++;
        }
    }

    // --- 6. Advance staging head + trim scanned state ---
    staging_.advance_head(flush_bytes);

    for (uint64_t i = 0; i < flush_count; ++i) {
        scanned_.pop_front();
    }
    pending_entries_ -= flush_count;
    pending_bytes_   -= flush_bytes;

    // Reset incremental counters: subtract flushed portion.
    // scan_bytes_accum_ tracks bytes scanned since last flush.
    // After flush, only trailing incomplete entries remain.
    scan_bytes_accum_ -= flush_bytes;
    flushable_bytes_   = 0;
    flushable_entries_ = 0;
    capped_flush_bytes_   = 0;
    capped_flush_entries_ = 0;
    last_complete_idx_ = -1;
    has_complete_time_ = false;

    // Re-scan remaining entries for any IS_LAST
    uint64_t reaccum = 0;
    for (size_t i = 0; i < scanned_.size(); ++i) {
        if (!(scanned_[i].flags & TASK_FLAG_IS_DROP)) {
            reaccum += scanned_[i].payload_alloc_bytes;
        }
        if (scanned_[i].flags & TASK_FLAG_IS_LAST) {
            last_complete_idx_ = static_cast<int64_t>(i);
            flushable_bytes_   = reaccum;
            flushable_entries_ = static_cast<uint64_t>(i + 1);
            if (reaccum <= staging_.capacity()) {
                capped_flush_bytes_   = reaccum;
                capped_flush_entries_ = static_cast<uint64_t>(i + 1);
            }
            if (!has_complete_time_) {
                first_complete_time_ = std::chrono::steady_clock::now();
                has_complete_time_ = true;
            }
        }
    }

    fprintf(stderr, "[drain_diag] batch_flush done: remaining_scanned=%lu remaining_pending=%lu "
            "new_flushable_entries=%lu new_capped_entries=%lu\n",
            (unsigned long)scanned_.size(), (unsigned long)pending_entries_,
            (unsigned long)flushable_entries_, (unsigned long)capped_flush_entries_);
}

// ---------------------------------------------------------------------------
// handle_large_tensor — bypass staging, D2H directly into pageable memory
// ---------------------------------------------------------------------------
void DrainThread::handle_large_tensor() {
    fprintf(stderr, "[drain_diag] handle_large_tensor: visible_head=%lu\n",
            (unsigned long)visible_head_);
    // The IS_FIRST entry is at visible_head_ (already confirmed ready)
    const uint64_t idx0 = visible_head_ % ring_.task_cap;
    TaskEntry first_entry = ring_.task_entries[idx0];

    const uint64_t total_bytes = first_entry.tensor_total_bytes;

    // ATen allocate pageable tensor
    auto tensor = at::empty({static_cast<int64_t>(total_bytes)},
                            at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    uint8_t* dst = tensor.data_ptr<uint8_t>();

    uint64_t task_id   = first_entry.logical_task_id;
    uint32_t hook_type = first_entry.hook_type;
    uint32_t hook_id   = first_entry.hook_id;

    // Process entries one at a time until IS_LAST
    bool done = false;
    while (!done) {
        while (!task_cpu_ready(ring_.task_entries, ring_.task_cap, visible_head_)) {
            struct timespec ts{0, 1000};  // 1 us
            nanosleep(&ts, nullptr);
        }

        const uint64_t idx = visible_head_ % ring_.task_cap;
        TaskEntry ec = ring_.task_entries[idx];

        if (ec.payload_len1 > 0) {
            cudaMemcpyAsync(dst + ec.chunk_offset_bytes,
                            ring_.payload_buf + ec.payload_off1,
                            ec.payload_len1,
                            cudaMemcpyDeviceToHost, stream_);
        }
        if (ec.payload_len2 > 0) {
            cudaMemcpyAsync(dst + ec.chunk_offset_bytes + ec.payload_len1,
                            ring_.payload_buf + ec.payload_off2,
                            ec.payload_len2,
                            cudaMemcpyDeviceToHost, stream_);
        }

        // Release GPU task slot + payload per-entry (allows producer to reuse)
        task_release_cpu(ring_.task_entries, ring_.task_cap, visible_head_);
        ++task_tail_local_;
        __atomic_store_n(ring_.task_tail, task_tail_local_, __ATOMIC_RELEASE);

        if (ec.payload_alloc_bytes > 0) {
            payload_tail_local_ += ec.payload_alloc_bytes;
            __atomic_store_n(ring_.payload_tail, payload_tail_local_, __ATOMIC_RELEASE);
        }
        ++heartbeat_local_;
        __atomic_store_n(ring_.consumer_heartbeat, heartbeat_local_, __ATOMIC_RELEASE);

        visible_head_++;
        done = (ec.flags & TASK_FLAG_IS_LAST) != 0;
    }

    cudaStreamSynchronize(stream_);

    // Bypass backpressure
    {
        std::unique_lock<std::mutex> lk(bypass_mu_);
        bypass_cv_.wait(lk, [&] {
            return in_flight_bypass_bytes_ == 0 ||
                   in_flight_bypass_bytes_ + total_bytes <= cfg_.bypass_budget_bytes;
        });
        in_flight_bypass_bytes_ += total_bytes;
    }

    // Construct ONE DrainTask
    DrainTask task{};
    task.logical_task_id    = task_id;
    task.chunk_offset_bytes = 0;
    task.tensor_total_bytes = total_bytes;
    task.hook_type          = hook_type;
    task.hook_id            = hook_id;
    task.flags              = TASK_FLAG_IS_FIRST | TASK_FLAG_IS_LAST | TASK_FLAG_LARGE_TENSOR;
    task.alloc_bytes        = 0;
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
