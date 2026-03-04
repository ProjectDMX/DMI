// ring/producer.cuh — Multi-thread GPU producer kernel.
//
// producer_kernel uses a single block with PRODUCER_BLOCK_DIM threads.
// Each chunk is processed in three sequential phases on the same block:
//
//   Phase 1 — thread 0 only:
//     Spin-wait for space (backpressure), compute two-span reservation,
//     broadcast span descriptors via shared memory.
//
//   Phase 2 — all threads cooperate:
//     Grid-stride D2D copy of chunk data into payload_buf spans.
//     Every thread calls __threadfence() after its stores so its writes
//     are globally visible before phase 3.
//
//   Phase 3 — thread 0 only:
//     Publish TaskEntry: write metadata fields, __threadfence(), write
//     ready_seq = seq_no (the signal the consumer spins on).
//
// Correctness note on __threadfence():
//   In CUDA, __threadfence() only makes the CALLING thread's stores visible
//   system-wide.  For the consumer to see all copy threads' payload writes
//   before it reads them (after observing ready_seq), EVERY copy thread must
//   call __threadfence() before __syncthreads() hands control back to thread 0.
//   Thread 0's __threadfence() inside task_publish is then a second fence that
//   orders the entry-field writes relative to ready_seq — it does NOT substitute
//   for the per-thread fences in phase 2.
//
// Capture-safe: all device pointers in `ring` must be preallocated before
// graph capture.  No dynamic allocation, no host callbacks.

#pragma once
#ifdef __CUDACC__

#include "ring_state.h"
#include "payload_ring.cuh"
#include "task_ring.cuh"

namespace ring {

// Number of threads per block.  256 gives good occupancy on Ampere/Ada while
// keeping register pressure manageable.  Must be a multiple of warp size (32).
static constexpr uint32_t PRODUCER_BLOCK_DIM = 256;

// ---------------------------------------------------------------------------
// Shared-memory descriptor broadcast from thread 0 to all copy threads.
// ---------------------------------------------------------------------------
struct alignas(16) ProducerShmem {
    TwoSpan  spans;         // payload reservation for this chunk
    uint64_t task_head;     // seq_no / slot index for this entry
    uint64_t payload_head;  // payload head after this reservation
    int      dropped;       // 1 if timeout-drop occurred
};

// ---------------------------------------------------------------------------
// producer_kernel
// ---------------------------------------------------------------------------
__global__ void producer_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        src_bytes,
    uint64_t        logical_task_id,
    uint32_t        hook_type,
    uint32_t        hook_id)
{
    __shared__ ProducerShmem sh;

    const uint64_t chunk_sz = ring.cfg.chunk_bytes;
    const uint64_t n_chunks = (src_bytes == 0)
                              ? 1
                              : (src_bytes + chunk_sz - 1) / chunk_sz;

    // Thread 0 owns the ring head counters; other threads read them from shmem.
    uint64_t task_head    = 0;
    uint64_t payload_head = 0;
    if (threadIdx.x == 0) {
        task_head    = *ring.task_head;
        payload_head = *ring.payload_head;
    }

    for (uint64_t ci = 0; ci < n_chunks; ++ci) {
        const uint64_t chunk_off  = ci * chunk_sz;
        const uint64_t this_chunk = (src_bytes == 0) ? 0
                                    : (ci + 1 < n_chunks ? chunk_sz
                                                         : src_bytes - chunk_off);

        // ----------------------------------------------------------------
        // Phase 1: thread 0 — backpressure + span reservation
        // ----------------------------------------------------------------
        if (threadIdx.x == 0) {
            bool dropped = false;

            if (ring.cfg.wait_policy == WaitPolicy::INFINITE) {
                while (task_free_slots(task_head, *ring.task_tail, ring.task_cap) < 1 ||
                       payload_free_bytes(payload_head, *ring.payload_tail,
                                          ring.payload_cap) < this_chunk)
                {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(128);
#endif
                }
            } else {
                uint64_t hb_snap = *ring.consumer_heartbeat;
                uint64_t deadline = clock64() + ring.cfg.no_progress_timeout_cycles;

                while (task_free_slots(task_head, *ring.task_tail, ring.task_cap) < 1 ||
                       payload_free_bytes(payload_head, *ring.payload_tail,
                                          ring.payload_cap) < this_chunk)
                {
                    uint64_t hb_now = *ring.consumer_heartbeat;
                    if (hb_now != hb_snap) {
                        hb_snap  = hb_now;
                        deadline = clock64() + ring.cfg.no_progress_timeout_cycles;
                    } else if (static_cast<int64_t>(clock64() - deadline) > 0) {
                        dropped = true;
                        break;
                    }
#if __CUDA_ARCH__ >= 700
                    __nanosleep(128);
#endif
                }
            }

            sh.dropped      = dropped ? 1 : 0;
            sh.task_head    = task_head;
            sh.payload_head = payload_head;
            sh.spans        = dropped ? TwoSpan{0, 0, 0, 0}
                                      : payload_compute_spans(payload_head,
                                                              ring.payload_cap,
                                                              this_chunk);
        }

        __syncthreads();  // broadcast shmem to all threads

        if (sh.dropped) {
            // Thread 0 publishes DROP marker; others just exit this iteration.
            if (threadIdx.x == 0 &&
                ring.cfg.drop_reporting == DropReporting::DROP_TASK &&
                task_free_slots(task_head, *ring.task_tail, ring.task_cap) >= 1)
            {
                uint32_t drop_flags = TASK_FLAG_IS_DROP | TASK_FLAG_IS_LAST;
                if (ci == 0) drop_flags |= TASK_FLAG_IS_FIRST;

                TaskEntry drop{};
                drop.seq_no             = task_head;
                drop.logical_task_id    = logical_task_id;
                drop.chunk_offset_bytes = chunk_off;
                drop.tensor_total_bytes = src_bytes;
                drop.chunk_idx          = static_cast<uint32_t>(ci);
                drop.hook_type          = hook_type;
                drop.hook_id            = hook_id;
                drop.flags              = drop_flags;
                drop.reason             = DROP_REASON_TIMEOUT_NO_PROGRESS;

                task_publish(ring.task_entries, ring.task_cap, task_head, drop);
                ++task_head;
                *ring.task_head = task_head;
            }
            break;
        }

        // ----------------------------------------------------------------
        // Phase 2: all threads — cooperative D2D copy (grid-stride)
        //
        // Access pattern: thread T writes bytes T, T+blockDim, T+2*blockDim, …
        // → consecutive threads access consecutive bytes → coalesced 128-byte
        //   transactions → near-peak HBM bandwidth.
        //
        // After writing, every thread calls __threadfence() so its stores are
        // globally visible before thread 0 publishes ready_seq in phase 3.
        // ----------------------------------------------------------------
        const TwoSpan&  spans   = sh.spans;
        const uint8_t*  src_ptr = src + chunk_off;

        for (uint64_t i = threadIdx.x; i < spans.len1; i += blockDim.x)
            ring.payload_buf[spans.off1 + i] = src_ptr[i];

        for (uint64_t i = threadIdx.x; i < spans.len2; i += blockDim.x)
            ring.payload_buf[spans.off2 + i] = src_ptr[spans.len1 + i];

        // Each thread fences its own global stores system-wide.
        __threadfence();

        __syncthreads();  // thread 0 waits for all threads' fences

        // ----------------------------------------------------------------
        // Phase 3: thread 0 — publish TaskEntry
        //
        // task_publish writes entry fields, __threadfence(), ready_seq.
        // That second threadfence orders the entry-field writes relative to
        // ready_seq; the per-thread fences above order payload writes.
        // ----------------------------------------------------------------
        if (threadIdx.x == 0) {
            uint32_t flags = 0;
            if (ci == 0)            flags |= TASK_FLAG_IS_FIRST;
            if (ci == n_chunks - 1) flags |= TASK_FLAG_IS_LAST;

            payload_advance_head(payload_head, this_chunk);

            TaskEntry entry{};
            entry.seq_no             = task_head;
            entry.logical_task_id    = logical_task_id;
            entry.chunk_offset_bytes = chunk_off;
            entry.tensor_total_bytes = src_bytes;
            entry.payload_off1       = spans.off1;
            entry.payload_len1       = spans.len1;
            entry.payload_off2       = spans.off2;
            entry.payload_len2       = spans.len2;
            entry.chunk_idx          = static_cast<uint32_t>(ci);
            entry.hook_type          = hook_type;
            entry.hook_id            = hook_id;
            entry.flags              = flags;
            entry.reason             = DROP_REASON_NONE;

            task_publish(ring.task_entries, ring.task_cap, task_head, entry);
            ++task_head;
            *ring.task_head    = task_head;
            *ring.payload_head = payload_head;
        }

        __syncthreads();  // keep all threads in sync for next iteration
    }
}

// ---------------------------------------------------------------------------
// launch_producer — host-side launcher.
// ---------------------------------------------------------------------------
inline void launch_producer(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint64_t         logical_task_id,
    uint32_t         hook_type,
    uint32_t         hook_id,
    cudaStream_t     stream = 0)
{
    producer_kernel<<<1, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, src_bytes, logical_task_id, hook_type, hook_id);
}

}  // namespace ring

#endif  // __CUDACC__
