// ring/producer.cuh -- GPU producer kernel.
//
// Dynamic-dispatch multi-block producer.  launch_producer picks the grid size
// based on src_bytes at call time; under CUDA graphs this decision is baked
// into the recorded graph so replay has zero host-side dispatch cost.
//
// All blocks: grid-stride D2D copy into payload_buf spans.
// Last block to finish (atomicAdd counter): publish TaskEntry, advance heads.
//
// The kernel never spins.  Space is guaranteed by the pre-forward capacity
// check before kernel launch.
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
static constexpr uint32_t TIER0_BLOCKS =  1;  // <= 64 KB
static constexpr uint32_t TIER1_BLOCKS =  4;  // <= 4 MB
static constexpr uint32_t TIER2_BLOCKS = 16;  // <= 32 MB
static constexpr uint32_t TIER3_BLOCKS = 64;  // > 32 MB

__global__ void producer_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        src_bytes,
    uint32_t        hook_type);      // diagnostics only

void launch_producer(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream = 0);

void set_ring_null_mode(bool enabled);

}  // namespace ring

#endif  // __CUDACC__
