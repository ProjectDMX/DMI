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

#include "row_table.h"

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

// GATHER_MAX_SEGS and the row-table segment-count clamp live in
// row_table.h (CUDA-free, shared with host code and CPU tests).

// In-graph linear projection heads (producer_head / producer_fused).
//
// A head node reads the same per-step row table as a gather producer and,
// instead of copying each selected row, writes K fp32 affine scores per
// selected row:  out[r, k] = sum_i w[k, i] * h[src_row(r), i] + b[k], with
// fp32 accumulation.  Weights and biases are data in a stable device arena:
// registering or retiring a slot writes the arena between steps (ordered on
// the model stream) and no graph is recaptured.
static constexpr uint32_t HEAD_MAX_SLOTS = 16;

// producer_head arena (fp32): K x hidden weights, K biases, then a scratch of
// HEAD_SPLIT_ROWS x HEAD_SPLITS x K partials.  Selections of at most
// HEAD_SPLIT_ROWS rows split each row's dot products over HEAD_SPLITS hidden
// slices so a one-row decode step is not bound by one SM reading K x hidden
// weights; the last block combines the partials in a fixed order.  A free
// slot holds w = 0, b = NaN, so its column is NaN.
static constexpr uint32_t HEAD_SPLIT_ROWS = 16;
static constexpr uint32_t HEAD_SPLITS     = 8;
inline uint64_t head_arena_floats(uint32_t K, uint32_t hidden) {
    return static_cast<uint64_t>(K) * hidden + K +
           static_cast<uint64_t>(HEAD_SPLIT_ROWS) * HEAD_SPLITS * K;
}

// producer_fused node arena (int32 words, stable address, written only
// between steps):
//   words [0, 16)    mode[k]   HEAD_FREE / SINGLE / FIRST / MID / LAST
//   words [16, 32)   acc[k]    accumulator column of a multi-layer slot
//   words [32, 48)   bias[k]   fp32
//   words [64, ...)  weights   K x hidden, bf16 or fp32 (a launch constant)
// Only non-free slots are computed.  A linear head over several layers holds
// one slot at every participating node: FIRST writes acc[row][a] = w.h, MID
// adds, LAST emits acc + w.h + b in fixed layer order; FIRST, MID and FREE
// columns are NaN.  acc is an fp32 [acc_rows, HEAD_MAX_ACC] buffer indexed by
// the row's position in the forward's token batch (the same at every layer).
enum HeadSlotMode : int32_t {
    HEAD_FREE = 0, HEAD_SINGLE = 1, HEAD_FIRST = 2, HEAD_MID = 3, HEAD_LAST = 4
};
struct HeadNodeHeader { int32_t mode[16]; int32_t acc[16]; };
static constexpr uint32_t HEAD_BIAS_WORD = 32;
static constexpr uint32_t HEAD_HDR_WORDS = 64;
static constexpr uint32_t HEAD_MAX_ACC   = 16;
static constexpr uint32_t HEAD_RT = 4;    // rows per head work item
static constexpr uint32_t HEAD_ST = 4;    // slots per head work item
inline uint64_t head_node_words(uint32_t K, uint32_t hidden, bool weight_bf16) {
    const uint64_t w = weight_bf16
        ? (static_cast<uint64_t>(K) * hidden + 1) / 2
        : static_cast<uint64_t>(K) * hidden;
    return (HEAD_HDR_WORDS + w + 3) & ~uint64_t(3);
}

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

// Gather: copy the row segments listed in a per-step row table, packed.
// Table layout (int64, stable device address, values re-read each replay --
// the same contract as row_count_dev_ptr):
//   table[0]       = n_segs (clamped to GATHER_MAX_SEGS and to the table's
//                    storage, row_table_segment_count); -1 disables the
//                    producer for the step: no reservation, no task, no
//                    payload.  0 publishes an empty task.
//   table[1 + 2k]  = source row of segment k
//   table[2 + 2k]  = rows in segment k
// table_numel is the table's int64 element count (>= 1), passed at launch
// so a count larger than the storage never reads past it.
// Destination offsets are the running prefix sum: the output is always
// packed, so the table holds (src, len) pairs, not (src, dst, len).  Each
// HookPoint passes its own table and row_bytes.
__global__ void producer_gather_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes_upper,
    const int64_t*  seg_table_dev_ptr,
    int64_t         table_numel,
    uint64_t        row_bytes,
    uint32_t        hook_type);

void launch_producer_gather(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   seg_table_dev_ptr,
    int64_t          table_numel,
    uint64_t         row_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream = 0);

// The same kernel and geometry, launched as a device-updatable kernel node
// (cudaLaunchAttributeDeviceUpdatableKernelNode) for device-side toggling
// (toggle_device.h).  Valid only under stream capture.  Returns the launch
// error; cudaErrorNotSupported when built without device graph updates.
cudaError_t launch_producer_gather_device_updatable(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   seg_table_dev_ptr,
    int64_t          table_numel,
    uint64_t         row_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream);

// In-graph linear heads at one HookPoint (see HEAD_* above): K fp32 scores
// per row selected by the gather-contract table.  Payload bytes are
// selected_rows x K x 4.  src_elem_bytes: 2 = bf16, 4 = fp32.
void launch_producer_head(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   seg_table_dev_ptr,
    int64_t          table_numel,
    uint32_t         hidden,
    uint32_t         src_elem_bytes,
    const float*     arena,
    uint32_t         K,
    uint32_t         hook_type,
    cudaStream_t     stream = 0);

// The HookPoint's row producer AND its heads in ONE node.  Each part has its
// own table (gather contract; -1 = that part publishes nothing); either table
// pointer may be null.  Publish order: the row task, then the head task
// (the FIFO metadata order of the two specs).  Head results are a fixed
// reduction order per (row, slot): bit-identical across batch compositions.
uint32_t producer_fused_grid(uint64_t nbytes_upper, uint64_t upper_rows);

void launch_producer_fused(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    uint32_t         hidden,
    uint32_t         src_elem_bytes,
    const int64_t*   gather_table,
    int64_t          gather_table_numel,
    const int64_t*   head_table,
    int64_t          head_table_numel,
    const int32_t*   node_arena,
    uint32_t         K,
    uint32_t         weight_bf16,
    float*           acc,
    uint64_t         acc_rows,
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
