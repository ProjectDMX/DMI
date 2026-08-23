// ring/producer.cuh -- GPU producer kernels and host-side launchers.
//
// Three distinct kernel variants, one per torch op:
//   producer_static_kernel    -- copy all `nbytes`; today's behavior.
//   producer_prefix_kernel    -- read row_count[0], copy row_count*row_bytes.
//   producer_chunked_kernel   -- K>1 chunked-suffix; per-chunk parallel.
//
// All blocks: grid-stride D2D copy (vectorized via uint4).
// Last block to finish (atomicAdd counter): publishes TaskEntry,
// advances heads, atomicAdd's actual_bytes_counter.
//
// The kernels never spin.  Space is guaranteed by the pre-forward
// capacity check before kernel launch.
//
// Capture-safe: all device pointers must be preallocated before
// graph capture.
//
// CUDA-graph capture: chosen launcher, n_blocks, K, row_bytes, and
// device pointer args are all baked at trace time.  The *values* at
// those device pointers are re-read each replay; the pointers and
// other args are not.  K and row_bytes being fixed is natural: they
// reflect structural properties tied to the captured shape signature
// (caller's contract; see ring_torch_op.cpp).

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

// Upper bound on K (chunk count) for shared-memory sizing in the
// chunked kernel.  Covers expected num_local_experts under EP.
static constexpr uint32_t PRODUCER_MAX_K = 64;

// Static: copy all `nbytes`; no actual_dev_ptr, no row_bytes.
__global__ void producer_static_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes,
    uint32_t        hook_type);

void launch_producer_static(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes,
    uint32_t         hook_type,
    cudaStream_t     stream = 0);

// Prefix: copy first (row_count[0] * row_bytes) bytes.  Shared-scalar
// pattern: multiple HookPoints may share the same row_count_dev_ptr.
__global__ void producer_prefix_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes_upper,
    const int64_t*  row_count_dev_ptr,
    uint64_t        row_bytes,
    uint32_t        hook_type);

void launch_producer_prefix(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   row_count_dev_ptr,
    uint64_t         row_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream = 0);

// Chunked: K>1 parallel-by-chunk; per-chunk bytes from
// chunk_bytes_dev_ptr[K]; packed contiguous output.
__global__ void producer_chunked_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes_upper,
    const int64_t*  chunk_bytes_dev_ptr,
    uint32_t        K,
    uint32_t        hook_type);

void launch_producer_chunked(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   chunk_bytes_dev_ptr,
    uint32_t         K,
    uint32_t         hook_type,
    cudaStream_t     stream = 0);

void set_ring_null_mode(bool enabled);

}  // namespace ring

#endif  // __CUDACC__
