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
// producer_kernel — declaration only; definition in producer.cu.
// ---------------------------------------------------------------------------
__global__ void producer_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        src_bytes,
    uint64_t        logical_task_id,
    uint32_t        hook_type,
    uint32_t        hook_id);

// ---------------------------------------------------------------------------
// Host-side launchers — defined in producer.cu (not inline) so they are
// exported as symbols for cross-TU linking.
// ---------------------------------------------------------------------------
void launch_producer(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint64_t         logical_task_id,
    uint32_t         hook_type,
    uint32_t         hook_id,
    cudaStream_t     stream = 0);

// Launch producer kernel then enqueue a host callback on the same stream.
//
// Graph-capture semantics: the hostfunc becomes a graph node that fires after
// the producer kernel.  To avoid blocking the forward pass, callers should
// either:
//   (a) use a side stream for the producer kernel + hostfunc (with an event
//       recording the tensor-ready point on the main stream), so the forward
//       pass continues unblocked; or
//   (b) accept the ~1 µs callback overhead per hooked tensor on the main
//       stream (callback just signals a CV and returns immediately).
//
// `notify_fn` / `notify_arg` are typically DrainThread::hostfunc_cb / &dt.
void launch_producer_with_notify(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint64_t         logical_task_id,
    uint32_t         hook_type,
    uint32_t         hook_id,
    cudaHostFn_t     notify_fn,
    void*            notify_arg,
    cudaStream_t     stream = 0);

}  // namespace ring

#endif  // __CUDACC__
