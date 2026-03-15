// ring/producer.cuh — GPU producer kernel.
//
// producer_kernel uses a single block with PRODUCER_BLOCK_DIM threads.
//
// Normal path (large_bypass=false):
//   Phase 1 — thread 0: compute two-span reservation, broadcast via shmem.
//   Phase 2 — all threads: grid-stride D2D copy into payload_buf spans.
//   Phase 3 — thread 0: publish TaskEntry, advance heads,
//             write d_condition[hook_idx] = 0 (device-side reset).
//
// Large bypass path (large_bypass=true):
//   Thread 0 only: publish TaskEntry with device_src_ptr, advance task_head,
//   write d_condition[hook_idx] = 1 (kernel done, pending drain ack).
//   No D2D copy, no payload consumed.
//
// The kernel never spins.  Backpressure is handled by cuStreamWaitValue32
// on the condition tensor BEFORE the kernel is launched (host side).
// After the kernel, all conditions are either 0 (normal) or 1 (large).
//
// Capture-safe: all device pointers must be preallocated before graph capture.

#pragma once
#ifdef __CUDACC__

#include "ring_state.h"
#include "payload_ring.cuh"

namespace ring {

static constexpr uint32_t PRODUCER_BLOCK_DIM = 256;

// Condition values: see COND_* constants in ring_config.h.

struct alignas(16) ProducerShmem {
    TwoSpan  spans;
    uint64_t task_head;
    uint64_t payload_head;
};

__global__ void producer_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        src_bytes,
    uint32_t        hook_type,      // diagnostics only
    bool            large_bypass,
    uint32_t*       d_condition,    // condition tensor (device memory)
    uint32_t        hook_idx);      // index into d_condition

void launch_producer(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint32_t         hook_type,
    bool             large_bypass,
    uint32_t*        d_condition,
    uint32_t         hook_idx,
    cudaStream_t     stream = 0);

void set_ring_null_mode(bool enabled);

}  // namespace ring

#endif  // __CUDACC__
