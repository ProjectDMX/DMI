// ring/producer.cu -- Multi-block producer kernels and host-side launchers.
//
// Three distinct kernel templates, one per torch op:
//
//   producer_static_kernel
//     No __shared__, no extra __syncthreads.  Copies all `nbytes`
//     from `src` to the ring slot.  Bandwidth-identical to the pre
//     Phase 2 kernel.  Used by `ring::producer` (static path).
//
//   producer_prefix_kernel
//     No __shared__, no extra __syncthreads.  Reads a 1-element
//     int64 device tensor `row_count_dev_ptr` to a register
//     (coalesced cached load across the warp); copies
//     `row_count[0] * row_bytes` bytes from `src` to the ring slot.
//     Used by `ring::producer_prefix` (vLLM dense strip via the
//     shared-scalar pattern; Phase 3c per-expert MLP_POST).
//
//   producer_chunked_kernel
//     Uses __shared__ s_T[MAX_K] + s_prefix[MAX_K+1] + one
//     __syncthreads.  Source split into K equal chunks of
//     `nbytes_upper / K` bytes each; copies first `chunk_bytes[i]`
//     bytes of each chunk packed contiguously into the ring slot.
//     Block-to-chunk mapping; host rounds n_blocks up to a
//     multiple of K so the partition is exact.  Used by
//     `ring::producer_chunked` (Phase 3+ HF whole-row drop, DeepEP
//     LL combined-rank).
//
// All three publish via the same last-block-arrives metadata
// pattern; the only differences are the byte count and the kernel
// setup (none / scalar read / K-vector + prefix sum).

#include "producer.cuh"
#include "payload_ring.cuh"
#include "task_ring.cuh"
#include "ring_config.h"

namespace ring {

__device__ bool g_ring_null_mode = false;

// Counter for last-block-arrives pattern.  Reset by the last block after
// publishing.  Safe without host-side reset because producers are serialized
// on one stream -- the next launch cannot start until the current finishes.
__device__ uint32_t g_block_done_counter = 0;

void set_ring_null_mode(bool enabled) {
    cudaMemcpyToSymbol(g_ring_null_mode, &enabled, sizeof(bool));
}

void set_ring_null_mode(bool enabled);

static_assert(PAYLOAD_ALIGN == sizeof(uint4),
    "PAYLOAD_ALIGN must equal sizeof(uint4) for vectorized D2D copies");

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Grid-stride vectorized D2D copy.  Uses global thread ID (gtid) and total
// thread count (stride) so all blocks in the grid participate.
__device__ inline void d2d_copy_grid_stride(
    uint8_t*       dst,
    const uint8_t* src,
    uint64_t       nbytes,
    uint64_t       gtid,
    uint64_t       stride)
{
    constexpr uint64_t VEC = sizeof(uint4);
    const uint64_t n_vec = nbytes / VEC;
    const uint64_t tail  = nbytes - n_vec * VEC;

    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4*       d4 = reinterpret_cast<uint4*>(dst);
    for (uint64_t i = gtid; i < n_vec; i += stride)
        d4[i] = s4[i];

    // Tail bytes -- at most 15 bytes, only low-numbered threads participate.
    const uint64_t tb = n_vec * VEC;
    for (uint64_t i = gtid; i < tail; i += stride)
        dst[tb + i] = src[tb + i];
}

// Copy `nbytes` from `src` to the slot at logical offset `dst_off`, splitting
// across the ring's two-span layout if the range crosses the wrap boundary.
__device__ inline void copy_chunk_with_wrap(
    uint8_t*       payload_buf,
    const TwoSpan& spans,
    uint64_t       dst_off,           // logical offset within the slot
    uint64_t       nbytes,            // bytes to copy
    const uint8_t* src,
    uint64_t       gtid,
    uint64_t       stride)
{
    uint64_t in_span1 = 0;
    if (dst_off < spans.len1) {
        const uint64_t avail = spans.len1 - dst_off;
        in_span1 = (nbytes < avail) ? nbytes : avail;
        d2d_copy_grid_stride(
            payload_buf + spans.off1 + dst_off,
            src, in_span1, gtid, stride);
    }
    const uint64_t in_span2 = nbytes - in_span1;
    if (in_span2 > 0) {
        const uint64_t span2_off = (dst_off >= spans.len1)
            ? (dst_off - spans.len1) : 0;
        d2d_copy_grid_stride(
            payload_buf + spans.off2 + span2_off,
            src + in_span1, in_span2, gtid, stride);
    }
}

// Last-block-arrives publish.  Shared across all three kernels; the only
// differing input is `actual_total` (the bytes the producer wrote).
__device__ inline void publish_last_block_arrives(
    RingState&     ring,
    uint64_t       task_head,
    uint64_t       payload_head,
    uint64_t       alloc_bytes,
    const TwoSpan& spans,
    uint64_t       actual_total)
{
    if (threadIdx.x != 0) return;
    uint32_t finished = atomicAdd(&g_block_done_counter, 1);
    if (finished != gridDim.x - 1) return;

    // Last block: all copy stores are globally visible (caller __threadfence'd
    // before calling this helper).
    uint64_t ph = payload_head;
    payload_advance_head(ph, alloc_bytes);

    const uint64_t len1 = (actual_total < spans.len1) ? actual_total : spans.len1;

    TaskEntry entry{};
    entry.tensor_total_bytes = actual_total;
    entry.payload_off1       = spans.off1;
    entry.payload_len1       = len1;
    entry.payload_off2       = spans.off2;
    entry.payload_len2       = actual_total - len1;

    task_publish(ring.task_entries, ring.task_cap, task_head, entry);
    *ring.task_head    = task_head + 1;
    *ring.payload_head = ph;

    // Monotonic accumulator of actual bytes written.  The __threadfence()
    // before publish orders D2D stores before this work; a CPU observer that
    // sees the counter also sees the bytes whose write it accounts for.
    atomicAdd(reinterpret_cast<unsigned long long*>(ring.actual_bytes_counter),
              static_cast<unsigned long long>(actual_total));

    // Reset for the next launch on this stream.
    g_block_done_counter = 0;
}

// ---------------------------------------------------------------------------
// producer_static_kernel -- copy all nbytes; no actual_dev_ptr, no row_bytes.
// ---------------------------------------------------------------------------
__global__ void producer_static_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes,
    uint32_t        hook_type)
{
    if (g_ring_null_mode) return;

    const uint64_t gtid   = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x)  * blockDim.x;

    const uint64_t alloc_bytes = align_up(nbytes, PAYLOAD_ALIGN);
    const uint64_t task_head    = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan  spans        = payload_compute_spans(payload_head,
                                                         ring.payload_cap,
                                                         alloc_bytes);

    // Two-span grid-stride copy of all `nbytes`.
    {
        const uint64_t copy1 = (nbytes < spans.len1) ? nbytes : spans.len1;
        d2d_copy_grid_stride(ring.payload_buf + spans.off1, src, copy1,
                             gtid, stride);
    }
    {
        const uint64_t data_in_span1 = (nbytes < spans.len1) ? nbytes : spans.len1;
        const uint64_t copy2 = nbytes - data_in_span1;
        if (copy2 > 0) {
            d2d_copy_grid_stride(ring.payload_buf + spans.off2,
                                 src + data_in_span1, copy2,
                                 gtid, stride);
        }
    }

    __threadfence();
    __syncthreads();

    publish_last_block_arrives(ring, task_head, payload_head, alloc_bytes,
                               spans, nbytes);
}

// ---------------------------------------------------------------------------
// producer_prefix_kernel -- copy first (row_count[0] * row_bytes) bytes.
// Each thread reads row_count to its own register (coalesced cached load).
// ---------------------------------------------------------------------------
__global__ void producer_prefix_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes_upper,
    const int64_t*  row_count_dev_ptr,
    uint64_t        row_bytes,
    uint32_t        hook_type)
{
    if (g_ring_null_mode) return;

    // Each thread reads the count to its own register; the read coalesces
    // to one cached transaction across the warp.  Clamp defensively against
    // provider bugs.
    int64_t rc = *row_count_dev_ptr;
    if (rc < 0) rc = 0;
    uint64_t actual_bytes = static_cast<uint64_t>(rc) * row_bytes;
    if (actual_bytes > nbytes_upper) actual_bytes = nbytes_upper;

    const uint64_t gtid   = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x)  * blockDim.x;

    // Slot allocation must match Python's reservation, which is sized
    // to actual_bytes (not nbytes_upper) when padding_strip is active.
    // Otherwise the kernel advances payload_head past where Python
    // reserved, and subsequent steps land in wrong slots.
    const uint64_t alloc_bytes = align_up(actual_bytes, PAYLOAD_ALIGN);
    const uint64_t task_head    = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan  spans        = payload_compute_spans(payload_head,
                                                         ring.payload_cap,
                                                         alloc_bytes);

    // Two-span grid-stride copy of `actual_bytes`.
    {
        const uint64_t copy1 = (actual_bytes < spans.len1) ? actual_bytes : spans.len1;
        d2d_copy_grid_stride(ring.payload_buf + spans.off1, src, copy1,
                             gtid, stride);
    }
    {
        const uint64_t data_in_span1 = (actual_bytes < spans.len1) ? actual_bytes : spans.len1;
        const uint64_t copy2 = actual_bytes - data_in_span1;
        if (copy2 > 0) {
            d2d_copy_grid_stride(ring.payload_buf + spans.off2,
                                 src + data_in_span1, copy2,
                                 gtid, stride);
        }
    }

    __threadfence();
    __syncthreads();

    publish_last_block_arrives(ring, task_head, payload_head, alloc_bytes,
                               spans, actual_bytes);
}

// ---------------------------------------------------------------------------
// producer_chunked_kernel -- K>1 parallel-by-chunk; per-chunk bytes from
// chunk_bytes_dev_ptr[K]; packed output.
// ---------------------------------------------------------------------------
__global__ void producer_chunked_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes_upper,
    const int64_t*  chunk_bytes_dev_ptr,
    uint32_t        K,
    uint32_t        hook_type)
{
    if (g_ring_null_mode) return;

    __shared__ int64_t s_T[PRODUCER_MAX_K];
    __shared__ int64_t s_prefix[PRODUCER_MAX_K + 1];

    if (threadIdx.x == 0) {
        const int64_t chunk_in = static_cast<int64_t>(nbytes_upper / K);
        s_prefix[0] = 0;
        for (uint32_t k = 0; k < K; ++k) {
            int64_t v = chunk_bytes_dev_ptr[k];
            if (v < 0) v = 0;
            if (v > chunk_in) v = chunk_in;
            s_T[k]          = v;
            s_prefix[k + 1] = s_prefix[k] + v;
        }
    }
    __syncthreads();

    const uint64_t actual_total = static_cast<uint64_t>(s_prefix[K]);
    const uint64_t chunk_in_bytes = nbytes_upper / K;

    const uint64_t alloc_bytes  = align_up(nbytes_upper, PAYLOAD_ALIGN);
    const uint64_t task_head    = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan  spans        = payload_compute_spans(payload_head,
                                                         ring.payload_cap,
                                                         alloc_bytes);

    // Block-to-chunk mapping; host rounds gridDim.x to a multiple of K when
    // K > 1, so blocks_per_chunk * K == gridDim.x exactly.
    const uint32_t blocks_per_chunk = gridDim.x / K;
    const uint32_t chunk_id         = blockIdx.x / blocks_per_chunk;
    const uint32_t block_in_chunk   = blockIdx.x - chunk_id * blocks_per_chunk;

    const uint64_t chunk_gtid   = uint64_t(block_in_chunk) * blockDim.x + threadIdx.x;
    const uint64_t chunk_stride = uint64_t(blocks_per_chunk) * blockDim.x;

    const uint64_t T_k = static_cast<uint64_t>(s_T[chunk_id]);
    if (T_k > 0) {
        const uint64_t dst_off   = static_cast<uint64_t>(s_prefix[chunk_id]);
        const uint8_t* chunk_src = src + uint64_t(chunk_id) * chunk_in_bytes;
        copy_chunk_with_wrap(ring.payload_buf, spans, dst_off, T_k,
                             chunk_src, chunk_gtid, chunk_stride);
    }

    __threadfence();
    __syncthreads();

    publish_last_block_arrives(ring, task_head, payload_head, alloc_bytes,
                               spans, actual_total);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

static inline uint32_t pick_tier_blocks(uint64_t nbytes) {
    if      (nbytes <= TIER1_THRESHOLD) return TIER0_BLOCKS;
    else if (nbytes <= TIER2_THRESHOLD) return TIER1_BLOCKS;
    else if (nbytes <= TIER3_THRESHOLD) return TIER2_BLOCKS;
    else                                return TIER3_BLOCKS;
}

void launch_producer_static(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes,
    uint32_t         hook_type,
    cudaStream_t     stream)
{
    const uint32_t n_blocks = pick_tier_blocks(nbytes);
    producer_static_kernel<<<n_blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes, hook_type);
}

void launch_producer_prefix(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   row_count_dev_ptr,
    uint64_t         row_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream)
{
    const uint32_t n_blocks = pick_tier_blocks(nbytes_upper);
    producer_prefix_kernel<<<n_blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, row_count_dev_ptr, row_bytes, hook_type);
}

void launch_producer_chunked(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   chunk_bytes_dev_ptr,
    uint32_t         K,
    uint32_t         hook_type,
    cudaStream_t     stream)
{
    uint32_t n_blocks = pick_tier_blocks(nbytes_upper);
    // Round up so n_blocks % K == 0 (clean block-to-chunk partition).
    if (n_blocks < K) n_blocks = K;
    const uint32_t r = n_blocks % K;
    if (r != 0) n_blocks += (K - r);
    producer_chunked_kernel<<<n_blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, chunk_bytes_dev_ptr, K, hook_type);
}

}  // namespace ring
