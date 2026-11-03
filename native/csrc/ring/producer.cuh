// ring/producer.cuh -- GPU producer kernels and host-side launchers.
//
// Three distinct kernel variants, one per torch op:
//   producer_static_kernel    -- copy all `nbytes`; today's behavior.
//   producer_prefix_kernel    -- read row_count[0], copy row_count*row_bytes.
//   producer_chunked_kernel   -- K>1 chunked-suffix; per-chunk parallel.
//
// All blocks: grid-stride D2D copy (vectorized via uint4).
// Last block to finish (atomicAdd counter): publishes actual bytes,
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

#include <cstdint>

namespace ring {

// Grid geometry for the D2D producers: one block per
// PRODUCER_BYTES_PER_BLOCK of the bytes the launch may copy (the
// capture-time upper bound for the prefix and chunked producers), capped at
// PRODUCER_MAX_BLOCKS.  Host-only code (and CPU unit tests) can use these
// helpers without a CUDA compiler.
//
// 16 KiB per block is four uint4 stores per thread at 256 threads; 8, 16 and
// 32 KiB per block measured within 0.5 us of each other, and 16 KiB keeps a
// graph captured for many rows cheap when a replay copies only a few.
//
// PRODUCER_MAX_BLOCKS is a bounded heuristic cap, not a residency
// guarantee: 264 is sized for about two resident 256-thread blocks per SM on
// 128/132-SM parts.  On smaller GPUs, or when other kernels occupy SMs, a
// capped grid can still take more than one wave.  Correctness never depends
// on the grid fitting in one wave: the kernels stride over the bytes they
// actually copy, so a grid sized by the upper bound stays correct when a
// replay copies fewer bytes, and surplus blocks only join the arrival count.
static constexpr uint64_t PRODUCER_BYTES_PER_BLOCK = 16 * 1024;
static constexpr uint32_t PRODUCER_MAX_BLOCKS      = 264;

#if defined(__CUDACC__)
#define RING_PRODUCER_HD __host__ __device__
#else
#define RING_PRODUCER_HD
#endif

// Overflow-free ceiling division: nbytes + PRODUCER_BYTES_PER_BLOCK - 1
// would wrap for byte counts near UINT64_MAX.
RING_PRODUCER_HD inline uint32_t producer_grid_blocks(uint64_t nbytes) {
    if (nbytes <= PRODUCER_BYTES_PER_BLOCK) return 1;
    const uint64_t blocks = (nbytes - 1) / PRODUCER_BYTES_PER_BLOCK + 1;
    return blocks > PRODUCER_MAX_BLOCKS
        ? PRODUCER_MAX_BLOCKS
        : static_cast<uint32_t>(blocks);
}

// Grid for the chunked producers, which partition blocks evenly across
// `nchunks` chunks and therefore need a multiple of `nchunks` blocks.  The
// proportional grid is rounded up to that multiple while it stays within
// PRODUCER_MAX_BLOCKS; once rounding up would pass the cap, the largest
// multiple of `nchunks` not above the cap is used instead.  Only when
// `nchunks` itself exceeds the cap does the grid exceed it (one block per
// chunk is the minimum the partition needs).  `nchunks == 0` is rejected by
// the launchers; it maps to the unchunked grid here.
RING_PRODUCER_HD inline uint32_t producer_chunked_grid_blocks(
    uint64_t nbytes_upper, uint32_t nchunks) {
    const uint32_t raw = producer_grid_blocks(nbytes_upper);
    if (nchunks <= 1) return raw;
    if (nchunks >= PRODUCER_MAX_BLOCKS || raw <= nchunks) return nchunks;
    const uint32_t rounded_up = ((raw - 1) / nchunks + 1) * nchunks;
    if (rounded_up <= PRODUCER_MAX_BLOCKS) return rounded_up;
    return (PRODUCER_MAX_BLOCKS / nchunks) * nchunks;
}

#undef RING_PRODUCER_HD

// Upper bound on K (chunk count) for shared-memory sizing in the
// chunked kernel and host-side argument validation.
static constexpr uint32_t PRODUCER_MAX_K = 64;

}  // namespace ring

#ifdef __CUDACC__

#include "ring_state.h"
#include "payload_ring.cuh"

namespace ring {

static constexpr uint32_t PRODUCER_BLOCK_DIM = 256;

// Record-only row packers use one block per logical output row and cap the
// grid so a conservative upper bound cannot create an unbounded launch.
static constexpr uint32_t RECORD_PACK_MAX_BLOCKS = 1024;

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

// ---------------------------------------------------------------------------
// Additive generic-record producers.
//
// These are separately named so the released inference torch-op schemas and
// kernels above remain unchanged.  The optional int32 device scalar gates
// emission without a host read: nullptr emits unconditionally; otherwise the
// producer emits only when *emit_gate == emit_value.
// ---------------------------------------------------------------------------

__global__ void record_producer_static_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes,
    const int32_t*   emit_gate,
    int32_t          emit_value);

void launch_record_producer_static(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream = 0);

__global__ void record_producer_prefix_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   row_count_dev_ptr,
    uint64_t         row_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value);

void launch_record_producer_prefix(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   row_count_dev_ptr,
    uint64_t         row_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream = 0);

__global__ void record_producer_chunked_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   chunk_bytes_dev_ptr,
    uint32_t         chunk_count,
    const int32_t*   emit_gate,
    int32_t          emit_value);

void launch_record_producer_chunked(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   chunk_bytes_dev_ptr,
    uint32_t         chunk_count,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream = 0);

__global__ void record_producer_seq_prefix_pack_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   valid_count_dev_ptr,
    const int64_t*   valid_prefix_sum_dev_ptr,
    uint32_t         batch,
    uint64_t         feature_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value);

void launch_record_producer_seq_prefix_pack(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   valid_count_dev_ptr,
    const int64_t*   valid_prefix_sum_dev_ptr,
    uint32_t         batch,
    uint64_t         feature_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream = 0);

__global__ void record_producer_segmented_pack_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   segment_start_dev_ptr,
    const int64_t*   segment_end_dev_ptr,
    uint32_t         segment_count,
    uint64_t         feature_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value);

void launch_record_producer_segmented_pack(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   segment_start_dev_ptr,
    const int64_t*   segment_end_dev_ptr,
    uint32_t         segment_count,
    uint64_t         feature_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream = 0);

void set_ring_null_mode(bool enabled);

}  // namespace ring

#endif  // __CUDACC__
