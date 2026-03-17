// ring/producer.cuh — GPU producer kernel.
//
// Dynamic-dispatch multi-block producer.  launch_producer picks the grid size
// based on src_bytes at call time; under CUDA graphs this decision is baked
// into the recorded graph so replay has zero host-side dispatch cost.
//
// Normal path (large_bypass=false):
//   Phase 1 — all blocks: grid-stride D2D copy into payload_buf spans.
//   Phase 2 — last block to finish (atomicAdd counter): publish TaskEntry,
//             advance heads.
//
// Large bypass path (large_bypass=true):
//   Block 0, thread 0 only: publish TaskEntry with device_src_ptr.
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

// Size-tier thresholds for grid selection (bytes).
static constexpr uint64_t TIER1_THRESHOLD =   64 * 1024;       //  64 KB
static constexpr uint64_t TIER2_THRESHOLD =  4 * 1024 * 1024;  //   4 MB
static constexpr uint64_t TIER3_THRESHOLD = 32 * 1024 * 1024;  //  32 MB

// Block counts per tier.
static constexpr uint32_t TIER0_BLOCKS =  1;  // ≤ 64 KB
static constexpr uint32_t TIER1_BLOCKS =  4;  // ≤ 4 MB
static constexpr uint32_t TIER2_BLOCKS = 16;  // ≤ 32 MB
static constexpr uint32_t TIER3_BLOCKS = 64;  // > 32 MB

// Condition values: see COND_* constants in ring_config.h.

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
