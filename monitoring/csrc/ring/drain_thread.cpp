// ring/drain_thread.cpp — CPU drain thread implementation.
// Compiled with g++ (not nvcc); uses CUDA runtime C API via -lcudart.

#include "drain_thread.h"
#include "task_ring.cuh"   // task_cpu_ready, task_release_cpu (CPU-side helpers)
#include "task_entry.h"    // TASK_FLAG_IS_DROP

#include <ctime>
#include <stdexcept>

namespace ring {

// ---------------------------------------------------------------------------
DrainThread::DrainThread(RingState& rs, PinnedPool& pool, DrainCallback cb)
    : ring_(rs), pool_(pool), cb_(std::move(cb))
{
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("DrainThread: cudaStreamCreate failed");
    // Local counters start at 0, matching AllocatedRing::init() which zeros
    // all ring counters before start() is called. Reading ring_.payload_tail
    // or ring_.task_tail here would require a PCIe round-trip (counters are
    // now GPU HBM); rely on the init() contract instead.
}

DrainThread::~DrainThread() noexcept {
    stop();
    // Clean up any pending events/buffers that weren't drained before stop.
    for (auto& pc : pending_) {
        if (pc.event)       cudaEventDestroy(pc.event);
        if (pc.alloc_bytes) pool_.free(pc.alloc_bytes);
    }
    cudaStreamDestroy(stream_);
}

// ---------------------------------------------------------------------------
void DrainThread::start() {
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread([this] { loop(); });
}

void DrainThread::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
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
void DrainThread::loop() {
    while (running_.load(std::memory_order_relaxed)) {
        poll_completed();
        drain_ready();

        if (pending_.empty()) {
            // No in-flight copies: sleep until notified.
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] {
                return notified_ || !running_.load(std::memory_order_relaxed);
            });
            notified_ = false;
        } else {
            // D2H copies in flight: yield briefly and keep polling.
            struct timespec ts{0, 1000};  // 1 µs
            nanosleep(&ts, nullptr);
        }
    }
    // Final drain: pick up any chunks whose D2H completed after stop().
    cudaStreamSynchronize(stream_);
    poll_completed();
}

// ---------------------------------------------------------------------------
void DrainThread::drain_ready() {
    while (true) {
        // task_tail_local_ is the authoritative tail: this thread is the sole
        // writer of ring_.task_tail. No need to re-read from GPU HBM.
        // task_head is no longer read: task_cpu_ready() returns false on an
        // empty ring (ready_seq == SENTINEL), which terminates the loop.
        const uint64_t tail = task_tail_local_;

        if (!task_cpu_ready(ring_.task_entries, ring_.task_cap, tail)) break;

        // Copy all needed fields out of the slot before releasing it.
        const uint64_t idx = tail % ring_.task_cap;
        const TaskEntry ec  = ring_.task_entries[idx];  // full copy

        const bool     is_drop   = (ec.flags & TASK_FLAG_IS_DROP) != 0;
        const uint64_t total_len = ec.payload_len1 + ec.payload_len2;

        PendingChunk pc{};
        pc.payload_bytes = is_drop ? 0 : total_len;
        pc.meta = {
            nullptr,           // pinned — filled below for data chunks
            total_len,
            0,                 // alloc_bytes — filled below for data chunks
            ec.logical_task_id,
            ec.chunk_offset_bytes,
            ec.tensor_total_bytes,
            ec.chunk_idx,
            ec.flags,
            ec.hook_type,
            ec.hook_id,
            ec.reason,
        };

        if (!is_drop && total_len > 0) {
            uint64_t alloc_bytes = 0;
            pc.pinned = pool_.alloc(total_len, alloc_bytes);
            if (pc.pinned == nullptr) {
                // Pinned ring full — stop draining; poll_completed() will free space.
                break;
            }
            pc.alloc_bytes      = alloc_bytes;
            pc.meta.pinned      = pc.pinned;
            pc.meta.alloc_bytes = alloc_bytes;

            cudaMemcpyAsync(pc.pinned,
                            ring_.payload_buf + ec.payload_off1,
                            ec.payload_len1,
                            cudaMemcpyDeviceToHost, stream_);
            if (ec.payload_len2 > 0) {
                cudaMemcpyAsync(pc.pinned + ec.payload_len1,
                                ring_.payload_buf + ec.payload_off2,
                                ec.payload_len2,
                                cudaMemcpyDeviceToHost, stream_);
            }

            cudaEvent_t ev;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            cudaEventRecord(ev, stream_);
            pc.event = ev;
        }
        // For drop entries: pc.event = nullptr, pc.pinned = nullptr.
        // They are treated as immediately complete in poll_completed().

        // Release the task slot so the producer can reuse it.
        // payload_tail is advanced later (after D2H completes).
        // Reset ready_seq to SENTINEL BEFORE advancing task_tail.
        // Ordering guarantee: producer sees the slot as free (task_tail++)
        // only after the slot is clean, preventing it from reusing a slot
        // whose ready_seq still signals "ready" to any late reader.
        task_release_cpu(ring_.task_entries, ring_.task_cap, tail);
        __atomic_store_n(ring_.task_tail, ++task_tail_local_, __ATOMIC_RELEASE);

        // Plain store (not fetch_add): drain thread is the sole writer of
        // consumer_heartbeat. Avoids LOCK-prefix atomics on PCIe-mapped
        // GPU HBM pages, which may not support hardware atomics from CPU.
        __atomic_store_n(ring_.consumer_heartbeat, ++heartbeat_local_,
                         __ATOMIC_RELEASE);

        pending_.push_back(std::move(pc));
    }
}

// ---------------------------------------------------------------------------
void DrainThread::poll_completed() {
    while (!pending_.empty()) {
        PendingChunk& pc = pending_.front();

        if (pc.event != nullptr) {
            cudaError_t status = cudaEventQuery(pc.event);
            if (status == cudaErrorNotReady) break;  // head not done, stop
            cudaEventDestroy(pc.event);
            pc.event = nullptr;
        }

        // Advance payload_tail (frees payload ring space for the producer).
        if (pc.payload_bytes > 0) {
            payload_tail_local_ += pc.payload_bytes;
            __atomic_store_n(ring_.payload_tail,
                             payload_tail_local_, __ATOMIC_RELEASE);
        }

        // Note: pool_.free() is called by the chunk assembler after it copies
        // the pinned data to pageable memory, not here.
        cb_(std::move(pc.meta));
        pending_.pop_front();
    }
}

}  // namespace ring
