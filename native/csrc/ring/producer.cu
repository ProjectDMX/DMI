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
#include "toggle_device.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ring {

__device__ bool g_ring_null_mode = false;

// IMPORTANT ASSUMPTION: Every producer kernel on this device runs on one
// stream. This last-block-arrives counter is shared
// by all launches, so overlapping launches would mix arrivals, publish early,
// and let one launch reset another's count. The last block resets it only after
// publishing, before the next producer launch on that stream can begin.
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

// Grid-stride D2D copy. Use uint4 only when both effective addresses satisfy
// its alignment requirement; tensor views, packed offsets, or wrap spans may
// make either address unaligned even when the ring allocation itself is
// aligned.
__device__ inline void d2d_copy_grid_stride(
    uint8_t*       dst,
    const uint8_t* src,
    uint64_t       nbytes,
    uint64_t       gtid,
    uint64_t       stride)
{
    constexpr uint64_t VEC = sizeof(uint4);
    const uintptr_t address_bits =
        reinterpret_cast<uintptr_t>(dst) |
        reinterpret_cast<uintptr_t>(src);
    if ((address_bits & (VEC - 1)) != 0) {
        for (uint64_t i = gtid; i < nbytes; i += stride)
            dst[i] = src[i];
        return;
    }

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
    uint64_t       actual_total)
{
    if (threadIdx.x != 0) return;

    // Every caller has just completed a block-wide __syncthreads(). This
    // thread-0 release fence therefore orders the joined payload writes from
    // the whole block before its relaxed arrival-counter RMW. The last block's
    // acquire fence below completes the cross-block fence/fence join.
    cuda::atomic_thread_fence(cuda::memory_order_release,
                              cuda::thread_scope_device);

    uint32_t finished = atomicAdd(&g_block_done_counter, 1);
    if (finished != gridDim.x - 1) return;

    // Pair with every block's thread-0 pre-arrival release fence. The relaxed
    // counter identifies the last block; this acquire fence imports every
    // block's joined payload writes before publication.
    cuda::atomic_thread_fence(cuda::memory_order_acquire,
                              cuda::thread_scope_device);

    uint64_t ph = payload_head;
    payload_advance_head(ph, alloc_bytes);

    task_publish(ring.publication_slots, ring.task_cap, task_head, actual_total);
    *ring.task_head    = task_head + 1;
    *ring.payload_head = ph;

    // Monotonic accumulator of actual bytes written. The per-block release
    // fences, last-block acquire, and system release publication order all
    // joined D2D stores before this accounting work.
    atomicAdd(reinterpret_cast<unsigned long long*>(ring.actual_bytes_counter),
              static_cast<unsigned long long>(actual_total));

    // Keep every access to the cross-block counter atomic. A plain store here
    // would race with the other blocks' relaxed atomic RMWs.
    atomicExch(&g_block_done_counter, 0);
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
                               nbytes);
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
                               actual_bytes);
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

    const uint64_t alloc_bytes  = align_up(actual_total, PAYLOAD_ALIGN);
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
                               actual_total);
}

// ---------------------------------------------------------------------------
// Generic-record producer helpers.  These deliberately do not alter or wrap
// the released inference kernels above.
// ---------------------------------------------------------------------------

__device__ inline bool record_emit_allowed(const int32_t* emit_gate,
                                           int32_t emit_value) {
    return emit_gate == nullptr || *emit_gate == emit_value;
}

// Generic records may pack rows or chunk prefixes whose byte widths are not
// multiples of PAYLOAD_ALIGN.  Keep the vectorized copy when both addresses
// are aligned, and use byte copies for an unaligned source or destination.
__device__ inline void record_d2d_copy_grid_stride(
    uint8_t*       dst,
    const uint8_t* src,
    uint64_t       nbytes,
    uint64_t       gtid,
    uint64_t       stride) {
    const uintptr_t address_bits =
        reinterpret_cast<uintptr_t>(dst) |
        reinterpret_cast<uintptr_t>(src);
    if ((address_bits & (PAYLOAD_ALIGN - 1)) == 0) {
        d2d_copy_grid_stride(dst, src, nbytes, gtid, stride);
        return;
    }

    for (uint64_t i = gtid; i < nbytes; i += stride) {
        dst[i] = src[i];
    }
}

__device__ inline void record_copy_chunk_with_wrap(
    uint8_t*       payload_buf,
    const TwoSpan& spans,
    uint64_t       dst_off,
    uint64_t       nbytes,
    const uint8_t* src,
    uint64_t       gtid,
    uint64_t       stride) {
    uint64_t in_span1 = 0;
    if (dst_off < spans.len1) {
        const uint64_t avail = spans.len1 - dst_off;
        in_span1 = nbytes < avail ? nbytes : avail;
        record_d2d_copy_grid_stride(
            payload_buf + spans.off1 + dst_off,
            src, in_span1, gtid, stride);
    }
    const uint64_t in_span2 = nbytes - in_span1;
    if (in_span2 > 0) {
        const uint64_t span2_off =
            dst_off >= spans.len1 ? dst_off - spans.len1 : 0;
        record_d2d_copy_grid_stride(
            payload_buf + spans.off2 + span2_off,
            src + in_span1, in_span2, gtid, stride);
    }
}

__device__ inline void record_copy_contiguous(
    uint8_t*       payload_buf,
    const TwoSpan& spans,
    const uint8_t* src,
    uint64_t       nbytes,
    uint64_t       gtid,
    uint64_t       stride) {
    const uint64_t first = nbytes < spans.len1 ? nbytes : spans.len1;
    d2d_copy_grid_stride(payload_buf + spans.off1, src, first, gtid, stride);
    const uint64_t second = nbytes - first;
    if (second > 0) {
        d2d_copy_grid_stride(payload_buf + spans.off2, src + first,
                             second, gtid, stride);
    }
}

__device__ inline void record_copy_row_block(
    uint8_t*       payload_buf,
    const TwoSpan& spans,
    uint64_t       dst_offset,
    const uint8_t* src,
    uint64_t       row_bytes) {
    record_copy_chunk_with_wrap(payload_buf, spans, dst_offset, row_bytes, src,
                                threadIdx.x, blockDim.x);
}

__global__ void record_producer_static_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes,
    const int32_t*   emit_gate,
    int32_t          emit_value) {
    if (g_ring_null_mode) return;
    if (!record_emit_allowed(emit_gate, emit_value)) return;

    const uint64_t gtid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    const uint64_t alloc_bytes = align_up(nbytes, PAYLOAD_ALIGN);
    const uint64_t task_head = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan spans = payload_compute_spans(
        payload_head, ring.payload_cap, alloc_bytes);

    record_copy_contiguous(ring.payload_buf, spans, src, nbytes, gtid, stride);
    __threadfence();
    __syncthreads();
    publish_last_block_arrives(
        ring, task_head, payload_head, alloc_bytes, nbytes);
}

__global__ void record_producer_prefix_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   row_count_dev_ptr,
    uint64_t         row_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value) {
    if (g_ring_null_mode) return;
    if (!record_emit_allowed(emit_gate, emit_value)) return;

    int64_t rows = *row_count_dev_ptr;
    if (rows < 0) rows = 0;
    uint64_t actual_bytes = nbytes_upper;
    if (row_bytes != 0 &&
        static_cast<uint64_t>(rows) <= nbytes_upper / row_bytes) {
        actual_bytes = static_cast<uint64_t>(rows) * row_bytes;
    }

    const uint64_t gtid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    const uint64_t alloc_bytes = align_up(actual_bytes, PAYLOAD_ALIGN);
    const uint64_t task_head = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan spans = payload_compute_spans(
        payload_head, ring.payload_cap, alloc_bytes);

    record_copy_contiguous(
        ring.payload_buf, spans, src, actual_bytes, gtid, stride);
    __threadfence();
    __syncthreads();
    publish_last_block_arrives(
        ring, task_head, payload_head, alloc_bytes, actual_bytes);
}

__global__ void record_producer_chunked_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   chunk_bytes_dev_ptr,
    uint32_t         chunk_count,
    const int32_t*   emit_gate,
    int32_t          emit_value) {
    if (g_ring_null_mode) return;
    if (!record_emit_allowed(emit_gate, emit_value)) return;

    __shared__ int64_t selected[PRODUCER_MAX_K];
    __shared__ int64_t prefix[PRODUCER_MAX_K + 1];
    if (threadIdx.x == 0) {
        const int64_t input_chunk =
            static_cast<int64_t>(nbytes_upper / chunk_count);
        prefix[0] = 0;
        for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
            int64_t bytes = chunk_bytes_dev_ptr[chunk];
            if (bytes < 0) bytes = 0;
            if (bytes > input_chunk) bytes = input_chunk;
            selected[chunk] = bytes;
            prefix[chunk + 1] = prefix[chunk] + bytes;
        }
    }
    __syncthreads();

    const uint64_t actual_bytes = static_cast<uint64_t>(prefix[chunk_count]);
    const uint64_t input_chunk = nbytes_upper / chunk_count;
    const uint64_t alloc_bytes = align_up(actual_bytes, PAYLOAD_ALIGN);
    const uint64_t task_head = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan spans = payload_compute_spans(
        payload_head, ring.payload_cap, alloc_bytes);

    const uint32_t blocks_per_chunk = gridDim.x / chunk_count;
    const uint32_t chunk = blockIdx.x / blocks_per_chunk;
    const uint32_t block_in_chunk = blockIdx.x % blocks_per_chunk;
    const uint64_t gtid =
        uint64_t(block_in_chunk) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(blocks_per_chunk) * blockDim.x;
    const uint64_t selected_bytes = static_cast<uint64_t>(selected[chunk]);
    if (selected_bytes > 0) {
        record_copy_chunk_with_wrap(
            ring.payload_buf, spans, static_cast<uint64_t>(prefix[chunk]),
            selected_bytes, src + uint64_t(chunk) * input_chunk, gtid, stride);
    }

    __threadfence();
    __syncthreads();
    publish_last_block_arrives(
        ring, task_head, payload_head, alloc_bytes, actual_bytes);
}

__global__ void record_producer_seq_prefix_pack_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   valid_count_dev_ptr,
    const int64_t*   valid_prefix_sum_dev_ptr,
    uint32_t         batch,
    uint64_t         feature_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value) {
    if (g_ring_null_mode) return;
    if (!record_emit_allowed(emit_gate, emit_value)) return;
    (void)valid_count_dev_ptr;

    int64_t encoded_rows = valid_prefix_sum_dev_ptr[batch];
    if (encoded_rows < 0) encoded_rows = 0;
    uint64_t total_rows = static_cast<uint64_t>(encoded_rows);
    const uint64_t max_rows = feature_bytes == 0
        ? 0 : nbytes_upper / feature_bytes;
    if (total_rows > max_rows) total_rows = max_rows;
    const uint64_t actual_bytes = total_rows * feature_bytes;
    const uint64_t alloc_bytes = align_up(actual_bytes, PAYLOAD_ALIGN);
    const uint64_t task_head = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan spans = payload_compute_spans(
        payload_head, ring.payload_cap, alloc_bytes);

    for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
        uint32_t sample = 0;
        while (sample + 1 < batch &&
               row >= static_cast<uint64_t>(
                   valid_prefix_sum_dev_ptr[sample + 1])) {
            ++sample;
        }
        const uint64_t sample_row = row - static_cast<uint64_t>(
            valid_prefix_sum_dev_ptr[sample]);
        const uint64_t source_row = sample_row * batch + sample;
        record_copy_row_block(
            ring.payload_buf, spans, row * feature_bytes,
            src + source_row * feature_bytes, feature_bytes);
    }

    __threadfence();
    __syncthreads();
    publish_last_block_arrives(
        ring, task_head, payload_head, alloc_bytes, actual_bytes);
}

__global__ void record_producer_segmented_pack_kernel(
    RingState        ring,
    const uint8_t*   src,
    uint64_t         nbytes_upper,
    const int64_t*   segment_start_dev_ptr,
    const int64_t*   segment_end_dev_ptr,
    uint32_t         segment_count,
    uint64_t         feature_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value) {
    if (g_ring_null_mode) return;
    if (!record_emit_allowed(emit_gate, emit_value)) return;

    const uint64_t input_rows = feature_bytes == 0
        ? 0 : nbytes_upper / feature_bytes;
    uint64_t total_rows = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        int64_t start = segment_start_dev_ptr[segment];
        int64_t end = segment_end_dev_ptr[segment];
        if (start < 0) start = 0;
        if (end < start) end = start;
        if (static_cast<uint64_t>(start) > input_rows) {
            start = static_cast<int64_t>(input_rows);
        }
        if (static_cast<uint64_t>(end) > input_rows) {
            end = static_cast<int64_t>(input_rows);
        }
        total_rows += static_cast<uint64_t>(end - start);
    }
    if (total_rows > input_rows) total_rows = input_rows;

    const uint64_t actual_bytes = total_rows * feature_bytes;
    const uint64_t alloc_bytes = align_up(actual_bytes, PAYLOAD_ALIGN);
    const uint64_t task_head = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan spans = payload_compute_spans(
        payload_head, ring.payload_cap, alloc_bytes);

    for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
        uint64_t remaining = row;
        uint64_t source_row = 0;
        for (uint32_t segment = 0; segment < segment_count; ++segment) {
            int64_t start = segment_start_dev_ptr[segment];
            int64_t end = segment_end_dev_ptr[segment];
            if (start < 0) start = 0;
            if (end < start) end = start;
            if (static_cast<uint64_t>(start) > input_rows) {
                start = static_cast<int64_t>(input_rows);
            }
            if (static_cast<uint64_t>(end) > input_rows) {
                end = static_cast<int64_t>(input_rows);
            }
            const uint64_t length = static_cast<uint64_t>(end - start);
            if (remaining < length) {
                source_row = static_cast<uint64_t>(start) + remaining;
                break;
            }
            remaining -= length;
        }
        record_copy_row_block(
            ring.payload_buf, spans, row * feature_bytes,
            src + source_row * feature_bytes, feature_bytes);
    }

    __threadfence();
    __syncthreads();
    publish_last_block_arrives(
        ring, task_head, payload_head, alloc_bytes, actual_bytes);
}


// ---------------------------------------------------------------------------
// Row tables: shared by the gather, head and fused producers.
// ---------------------------------------------------------------------------

// Load a gather-contract table into shared memory and turn its per-segment
// row counts into a destination prefix, clamped at upper_rows.  Every thread
// of the block must call this; it ends with a barrier.
//
// Defensive clamps: a host-side bug must not let a kernel read or write past
// the captured tensor (upper_rows).  The serial form of the clamp,
//     n'_k = min(n_k, upper_rows - prefix_k),
// telescopes to prefix_{k+1} = min(sum_{j<=k} n_j, upper_rows), so the table
// is loaded by all threads in parallel and the prefix is one warp's
// __shfl_up_sync scan.  A single serial thread-0 loop costs ~0.05 us per
// segment in every block (31 us for 150 one-row segments).
//
// The segment count is bounded by GATHER_MAX_SEGS and by the table's
// physical storage (table_numel words, row_table_segment_count), so a count
// word larger than the table never reads past it.
__device__ inline uint32_t load_row_table(const int64_t* table,
                                          int64_t table_numel,
                                          int64_t upper_rows,
                                          int64_t* s_src,
                                          int64_t* s_prefix)
{
    const uint32_t n_segs = row_table_segment_count(table[0], table_numel);
    for (uint32_t k = threadIdx.x; k < n_segs; k += blockDim.x) {
        int64_t rs = table[1 + 2 * k];
        int64_t nr = table[2 + 2 * k];
        if (rs < 0) rs = 0;
        if (rs > upper_rows) rs = upper_rows;
        if (nr < 0) nr = 0;
        if (nr > upper_rows - rs) nr = upper_rows - rs;
        s_src[k]        = rs;
        s_prefix[k + 1] = nr;          // per-segment count; scanned below
    }
    if (threadIdx.x == 0) s_prefix[0] = 0;
    __syncthreads();
    // A single segment's count is already <= upper_rows, so its scan is the
    // identity.  n_segs is block-uniform, so the barrier below is safe.
    if (n_segs > 1) {
        if (threadIdx.x < 32) {
            // Warp 0: each lane owns a contiguous run of PER_LANE counts.
            constexpr uint32_t PER_LANE = (GATHER_MAX_SEGS + 31) / 32;
            const uint32_t lane  = threadIdx.x;
            const uint32_t begin = lane * PER_LANE;
            int64_t local = 0;
            for (uint32_t j = 0; j < PER_LANE; ++j)
                if (begin + j < n_segs) local += s_prefix[begin + j + 1];
            int64_t incl = local;
            for (uint32_t off = 1; off < 32; off <<= 1) {
                const int64_t v = __shfl_up_sync(0xffffffffu, incl, off);
                if (lane >= off) incl += v;
            }
            int64_t run = incl - local;    // exclusive prefix of this lane's run
            for (uint32_t j = 0; j < PER_LANE; ++j) {
                if (begin + j < n_segs) {
                    run += s_prefix[begin + j + 1];
                    s_prefix[begin + j + 1] = (run < upper_rows) ? run : upper_rows;
                }
            }
        }
        __syncthreads();
    }
    return n_segs;
}

// Source row of packed output row r: the largest k with s_prefix[k] <= r.
__device__ inline uint64_t table_src_row(const int64_t* s_src,
                                         const int64_t* s_prefix,
                                         uint32_t n_segs, uint64_t r)
{
    uint32_t lo = 0, hi = n_segs;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) >> 1;
        if (static_cast<uint64_t>(s_prefix[mid]) <= r) lo = mid; else hi = mid;
    }
    return static_cast<uint64_t>(s_src[lo]) +
           (r - static_cast<uint64_t>(s_prefix[lo]));
}

__device__ inline void store_f32_with_wrap(const RingState& ring,
                                           const TwoSpan& spans,
                                           uint64_t offset, float value)
{
    uint8_t* dst = (offset < spans.len1)
        ? ring.payload_buf + spans.off1 + offset
        : ring.payload_buf + spans.off2 + (offset - spans.len1);
    *reinterpret_cast<float*>(dst) = value;
}

// Block-wide last-block-arrives join for kernels whose last block has work
// left before publishing (a combine step, or several tasks).  Every thread of
// a participating block calls it after its payload writes; it returns true in
// every thread of the last of `participants` blocks to arrive.  The same
// release/acquire fence pairing as publish_last_block_arrives.
__device__ inline bool join_last_block(uint32_t participants)
{
    __shared__ bool s_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        cuda::atomic_thread_fence(cuda::memory_order_release,
                                  cuda::thread_scope_device);
        const uint32_t finished = atomicAdd(&g_block_done_counter, 1);
        s_last = finished == participants - 1;
        if (s_last) {
            cuda::atomic_thread_fence(cuda::memory_order_acquire,
                                      cuda::thread_scope_device);
        }
    }
    __syncthreads();
    return s_last;
}

// Publish `count` consecutive tasks from thread 0 of the last block, after
// join_last_block() and a block barrier covering the last block's own writes.
__device__ inline void publish_tasks(RingState& ring,
                                     uint64_t task_head,
                                     uint64_t payload_head,
                                     const uint64_t* totals,
                                     const uint64_t* allocs,
                                     uint32_t count)
{
    if (threadIdx.x != 0) return;
    cuda::atomic_thread_fence(cuda::memory_order_release,
                              cuda::thread_scope_device);
    uint64_t th = task_head;
    uint64_t ph = payload_head;
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i) {
        task_publish(ring.publication_slots, ring.task_cap, th, totals[i]);
        ++th;
        payload_advance_head(ph, allocs[i]);
        total += totals[i];
    }
    *ring.task_head    = th;
    *ring.payload_head = ph;
    atomicAdd(reinterpret_cast<unsigned long long*>(ring.actual_bytes_counter),
              static_cast<unsigned long long>(total));
    atomicExch(&g_block_done_counter, 0);
}

// ---------------------------------------------------------------------------
// producer_gather_kernel -- copy the table's row runs, packed.
//
// Output ROWS are distributed to BLOCKS grid-stride (a row's source is a
// binary search over the shared prefix) and the block's threads copy that
// row vectorized.  Row-to-block (not segment-to-block) mapping keeps one
// large segment (a whole prefill chunk) spread across the grid, and keeps the
// grid independent of the runtime segment count -- required under CUDA-graph
// capture, where gridDim is baked but the table is re-read each replay.
// ---------------------------------------------------------------------------
__global__ void producer_gather_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        nbytes_upper,
    const int64_t*  seg_table_dev_ptr,
    int64_t         table_numel,
    uint64_t        row_bytes,
    uint32_t        hook_type)
{
    if (g_ring_null_mode) return;
    // -1 disables this producer for the step, including task publication.
    // Unlike 0 (an active, empty observation) it has no reservation or
    // metadata entry.  The table is stable for the launch.
    if (seg_table_dev_ptr[0] == -1) return;

    __shared__ int64_t s_src[GATHER_MAX_SEGS];
    __shared__ int64_t s_prefix[GATHER_MAX_SEGS + 1];

    const int64_t upper_rows =
        (row_bytes > 0) ? static_cast<int64_t>(nbytes_upper / row_bytes) : 0;
    const uint32_t n_segs = load_row_table(seg_table_dev_ptr, table_numel,
                                           upper_rows, s_src, s_prefix);

    const uint64_t out_rows     = static_cast<uint64_t>(s_prefix[n_segs]);
    const uint64_t actual_total = out_rows * row_bytes;
    const uint64_t alloc_bytes  = align_up(actual_total, PAYLOAD_ALIGN);
    const uint64_t task_head    = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan  spans        = payload_compute_spans(payload_head,
                                                         ring.payload_cap,
                                                         alloc_bytes);

    if (n_segs == 1) {
        // One contiguous run (a full-row table or one merged window): a flat
        // whole-grid copy, the same cost as producer_prefix.  The row-mapped
        // loop is paid only when the selection is actually fragmented.
        const uint64_t gtid   = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const uint64_t stride = uint64_t(gridDim.x)  * blockDim.x;
        copy_chunk_with_wrap(ring.payload_buf, spans, 0, actual_total,
                             src + static_cast<uint64_t>(s_src[0]) * row_bytes,
                             gtid, stride);
    } else {
        for (uint64_t r = blockIdx.x; r < out_rows; r += gridDim.x) {
            const uint64_t src_row = table_src_row(s_src, s_prefix, n_segs, r);
            copy_chunk_with_wrap(ring.payload_buf, spans, r * row_bytes,
                                 row_bytes, src + src_row * row_bytes,
                                 threadIdx.x, blockDim.x);
        }
    }

    __threadfence();
    __syncthreads();
    publish_last_block_arrives(ring, task_head, payload_head, alloc_bytes,
                               actual_total);
}

// ---------------------------------------------------------------------------
// producer_head_kernel -- in-graph linear heads (see producer.cuh).
//
// The table setup is the gather kernel's, so a head node and a gather node
// fed the same table select the same rows in the same packed order.  One
// output row (or, for decode-sized selections, one (row, hidden slice)) per
// block iteration; the block's threads split the hidden dimension, each
// accumulating K partial dot products in fp32 registers, then a warp-shuffle
// and shared-memory reduction yields the K scores.
// ---------------------------------------------------------------------------
__device__ inline void head_load8(const __nv_bfloat16* p, float* x) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h2[j]);
        x[2 * j] = f.x;
        x[2 * j + 1] = f.y;
    }
}
__device__ inline void head_load8(const float* p, float* x) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w;
    x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
}
__device__ inline float head_elem(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ inline float head_elem(const float* p) { return *p; }

template <typename T>
__global__ void producer_head_kernel(
    RingState       ring,
    const T*        src,
    uint64_t        upper_rows_u,
    const int64_t*  seg_table_dev_ptr,
    int64_t         table_numel,
    uint32_t        hidden,
    const float*    arena,
    uint32_t        K,
    uint32_t        hook_type)
{
    if (g_ring_null_mode) return;
    if (seg_table_dev_ptr[0] == -1) return;   // disabled: no reservation/task

    __shared__ int64_t s_src[GATHER_MAX_SEGS];
    __shared__ int64_t s_prefix[GATHER_MAX_SEGS + 1];
    __shared__ float   s_part[PRODUCER_BLOCK_DIM / 32][HEAD_MAX_SLOTS];

    const int64_t upper_rows = static_cast<int64_t>(upper_rows_u);
    if (K > HEAD_MAX_SLOTS) K = HEAD_MAX_SLOTS;   // the launcher refuses this
    const uint32_t n_segs = load_row_table(seg_table_dev_ptr, table_numel,
                                           upper_rows, s_src, s_prefix);

    const uint64_t out_rows     = static_cast<uint64_t>(s_prefix[n_segs]);
    const uint64_t row_out      = static_cast<uint64_t>(K) * sizeof(float);
    const uint64_t actual_total = out_rows * row_out;
    const uint64_t alloc_bytes  = align_up(actual_total, PAYLOAD_ALIGN);
    const uint64_t task_head    = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan  spans        = payload_compute_spans(payload_head,
                                                         ring.payload_cap,
                                                         alloc_bytes);
    const float* bias = arena + static_cast<uint64_t>(K) * hidden;
    float* scratch = const_cast<float*>(bias + K);
    // 8-wide loads need 16-byte aligned bases; with hidden % 8 == 0 every row
    // and slot offset is then a multiple of 16 bytes as well.  A tensor view
    // at an odd storage offset takes the scalar path.
    const bool vec8 = (hidden % 8u) == 0u
        && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u
        && (reinterpret_cast<uintptr_t>(arena) & 15u) == 0u;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n_warps = blockDim.x >> 5;
    // Decode-sized selections split each row over HEAD_SPLITS hidden slices;
    // the partials go to the arena's scratch and the last block sums them in
    // a fixed order (deterministic) and adds the bias.
    const uint32_t S = (out_rows <= HEAD_SPLIT_ROWS) ? HEAD_SPLITS : 1u;
    uint32_t chunk = (hidden + S - 1) / S;
    if (vec8) chunk = (chunk + 7u) & ~7u;
    const uint64_t items = out_rows * S;

    for (uint64_t it = blockIdx.x; it < items; it += gridDim.x) {
        const uint64_t r = it / S;
        const uint32_t sl = static_cast<uint32_t>(it - r * S);
        const uint32_t i_lo = sl * chunk;
        const uint32_t i_hi = (i_lo + chunk < hidden) ? i_lo + chunk : hidden;
        const T* h = src + table_src_row(s_src, s_prefix, n_segs, r) * hidden;

        float acc[HEAD_MAX_SLOTS];
#pragma unroll
        for (uint32_t k = 0; k < HEAD_MAX_SLOTS; ++k) acc[k] = 0.f;
        if (vec8) {
            for (uint32_t i = i_lo + threadIdx.x * 8u; i < i_hi; i += blockDim.x * 8u) {
                float x[8];
                head_load8(h + i, x);
#pragma unroll
                for (uint32_t k = 0; k < HEAD_MAX_SLOTS; ++k) {
                    if (k < K) {
                        const float* w = arena + static_cast<uint64_t>(k) * hidden + i;
                        const float4 w0 = *reinterpret_cast<const float4*>(w);
                        const float4 w1 = *reinterpret_cast<const float4*>(w + 4);
                        float a = acc[k];
                        a = fmaf(w0.x, x[0], a); a = fmaf(w0.y, x[1], a);
                        a = fmaf(w0.z, x[2], a); a = fmaf(w0.w, x[3], a);
                        a = fmaf(w1.x, x[4], a); a = fmaf(w1.y, x[5], a);
                        a = fmaf(w1.z, x[6], a); a = fmaf(w1.w, x[7], a);
                        acc[k] = a;
                    }
                }
            }
        } else {
            for (uint32_t i = i_lo + threadIdx.x; i < i_hi; i += blockDim.x) {
                const float xi = head_elem(h + i);
#pragma unroll
                for (uint32_t k = 0; k < HEAD_MAX_SLOTS; ++k)
                    if (k < K)
                        acc[k] = fmaf(arena[static_cast<uint64_t>(k) * hidden + i], xi, acc[k]);
            }
        }
#pragma unroll
        for (uint32_t k = 0; k < HEAD_MAX_SLOTS; ++k) {
            if (k < K) {
                float v = acc[k];
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, off);
                if (lane == 0) s_part[warp][k] = v;
            }
        }
        __syncthreads();
        if (threadIdx.x < K) {
            float v = 0.f;
            for (uint32_t w = 0; w < n_warps; ++w) v += s_part[w][threadIdx.x];
            if (S == 1) {
                store_f32_with_wrap(ring, spans,
                                    r * row_out + threadIdx.x * sizeof(float),
                                    v + bias[threadIdx.x]);
            } else {
                scratch[(r * S + sl) * K + threadIdx.x] = v;
            }
        }
        __syncthreads();   // s_part is reused by the next item
    }

    if (!join_last_block(gridDim.x)) return;
    if (S > 1) {
        // Fixed-order split-K combine.  __ldcg bypasses L1, which is not
        // coherent with the other blocks' scratch writes.
        for (uint64_t j = threadIdx.x; j < out_rows * K; j += blockDim.x) {
            const uint64_t r = j / K;
            const uint32_t k = static_cast<uint32_t>(j - r * K);
            float v = 0.f;
            for (uint32_t sl = 0; sl < S; ++sl)
                v += __ldcg(&scratch[(r * S + sl) * K + k]);
            store_f32_with_wrap(ring, spans, j * sizeof(float), v + bias[k]);
        }
        __syncthreads();
    }
    publish_tasks(ring, task_head, payload_head, &actual_total, &alloc_bytes, 1);
}

// ---------------------------------------------------------------------------
// producer_fused_kernel -- a HookPoint's row producer AND its heads in ONE
// captured node.
//
// Two independent parts, each driven by its own gather-contract table
// (-1 = part disabled: no reservation, no task, no payload):
//   row part:   copy the selected rows (exactly producer_gather's bytes);
//   head part:  K fp32 columns per selected row; only the node's active
//               slots are computed, free slots' columns are NaN.
// Publish order: the row task (if on), then the head task (if on), payloads
// back to back, each aligned.
//
// Head math: an item is a tile of HEAD_RT consecutive output rows x a group
// of HEAD_ST active slots.  The block's warps split the hidden dimension into
// fixed chunks; each lane accumulates fp32 FMAs in increasing order, each
// warp reduces its lanes with a shfl_down tree, and one thread sums the warp
// partials in warp order.  A (row, slot) result therefore depends only on
// that row's values and that slot's weights -- bit-identical across batch
// compositions and grids (fixed reduction order, no atomics, no split-K).
//
// Participating blocks: P = max(blocks the row part needs, blocks the head
// part needs) <= gridDim.x, computed identically by every block from the
// stable tables; the others return before reading the ring heads and never
// join the arrival count.
// ---------------------------------------------------------------------------
template <typename T, typename W>
__global__ void producer_fused_kernel(
    RingState       ring,
    const T*        src,
    uint64_t        upper_rows_u,
    uint32_t        hidden,
    const int64_t*  gtab,
    int64_t         gtab_numel,
    const int64_t*  htab,
    int64_t         htab_numel,
    const int32_t*  node,
    uint32_t        K,
    float*          acc,
    uint64_t        acc_rows,
    uint32_t        hook_type)
{
    if (g_ring_null_mode) return;
    const bool g_on = gtab != nullptr && gtab[0] != -1;
    const bool h_on = htab != nullptr && htab[0] != -1;
    if (!g_on && !h_on) return;

    __shared__ int64_t g_src[GATHER_MAX_SEGS];
    __shared__ int64_t g_pre[GATHER_MAX_SEGS + 1];
    __shared__ int64_t h_src[GATHER_MAX_SEGS];
    __shared__ int64_t h_pre[GATHER_MAX_SEGS + 1];
    __shared__ int32_t s_list[HEAD_MAX_SLOTS];      // compact active slot ids
    __shared__ int32_t s_mode[HEAD_MAX_SLOTS];      // by slot id
    __shared__ int32_t s_acc[HEAD_MAX_SLOTS];       // by slot id
    __shared__ float   s_bias[HEAD_MAX_SLOTS];      // by slot id
    __shared__ uint32_t s_nact;

    const int64_t  upper_rows = static_cast<int64_t>(upper_rows_u);
    const uint64_t row_bytes  = static_cast<uint64_t>(hidden) * sizeof(T);
    if (K > HEAD_MAX_SLOTS) K = HEAD_MAX_SLOTS;   // the launcher refuses this

    // Head header: the last warp loads modes, accumulator columns and biases
    // while the tables load.
    const uint32_t hdr_t0 = blockDim.x - 32;
    if (h_on && threadIdx.x >= hdr_t0 && threadIdx.x < hdr_t0 + HEAD_MAX_SLOTS) {
        const HeadNodeHeader* hdr = reinterpret_cast<const HeadNodeHeader*>(node);
        const float* bias = reinterpret_cast<const float*>(node + HEAD_BIAS_WORD);
        const uint32_t k = threadIdx.x - hdr_t0;
        int32_t m = (k < K) ? hdr->mode[k] : static_cast<int32_t>(HEAD_FREE);
        const int32_t ac = hdr->acc[k];
        if (m < HEAD_FREE || m > HEAD_LAST) m = HEAD_FREE;
        if ((m == HEAD_FIRST || m == HEAD_MID || m == HEAD_LAST)
                && (ac < 0 || ac >= static_cast<int32_t>(HEAD_MAX_ACC)))
            m = HEAD_FREE;
        s_mode[k] = m;
        s_acc[k]  = ac;
        s_bias[k] = bias[k];
    }
    const uint32_t gn = g_on ? load_row_table(gtab, gtab_numel, upper_rows, g_src, g_pre) : 0;
    const uint32_t hn = h_on ? load_row_table(htab, htab_numel, upper_rows, h_src, h_pre) : 0;
    if (h_on && threadIdx.x == 0) {
        // Compact list of computed slots in ascending slot id, derived from
        // the modes so no column is left unwritten.
        uint32_t c = 0;
        for (uint32_t k = 0; k < K; ++k)
            if (s_mode[k] != HEAD_FREE) s_list[c++] = static_cast<int32_t>(k);
        s_nact = c;
    }
    __syncthreads();

    const uint64_t g_rows  = g_on ? static_cast<uint64_t>(g_pre[gn]) : 0;
    const uint64_t h_rows  = h_on ? static_cast<uint64_t>(h_pre[hn]) : 0;
    const uint64_t g_total = g_rows * row_bytes;
    const uint64_t h_total = h_rows * K * sizeof(float);
    const uint64_t g_alloc = align_up(g_total, PAYLOAD_ALIGN);
    const uint64_t h_alloc = align_up(h_total, PAYLOAD_ALIGN);
    const uint32_t nact    = h_on ? s_nact : 0;

    // Participating blocks (block-uniform).
    const uint32_t NW = blockDim.x >> 5;
    uint64_t gb = 0;
    if (g_on)
        gb = (gn == 1)
            ? (g_total + PRODUCER_BYTES_PER_BLOCK - 1) / PRODUCER_BYTES_PER_BLOCK
            : g_rows;
    const uint64_t n_sg    = (nact + HEAD_ST - 1) / HEAD_ST;
    const uint64_t h_items = ((h_rows + HEAD_RT - 1) / HEAD_RT) * n_sg;
    uint64_t hb = h_items;                         // one block per head item
    const uint64_t fill = h_rows * K;              // free-column NaN fill
    const uint64_t fb = (fill + blockDim.x * 64 - 1) / (blockDim.x * 64);
    if (fb > hb) hb = fb;
    uint64_t P64 = gb > hb ? gb : hb;
    if (P64 < 1) P64 = 1;
    if (P64 > gridDim.x) P64 = gridDim.x;
    const uint32_t P = static_cast<uint32_t>(P64);
    if (blockIdx.x >= P) return;

    const uint64_t task_head    = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan  gsp = payload_compute_spans(payload_head, ring.payload_cap, g_alloc);
    const TwoSpan  hsp = payload_compute_spans(payload_head + (g_on ? g_alloc : 0),
                                               ring.payload_cap, h_alloc);

    // ---- row part (byte-identical to producer_gather) ----
    if (g_rows) {
        const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src);
        if (gn == 1) {
            const uint64_t gtid   = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            const uint64_t stride = uint64_t(P) * blockDim.x;
            copy_chunk_with_wrap(ring.payload_buf, gsp, 0, g_total,
                                 src8 + static_cast<uint64_t>(g_src[0]) * row_bytes,
                                 gtid, stride);
        } else {
            for (uint64_t r = blockIdx.x; r < g_rows; r += P) {
                const uint64_t sr = table_src_row(g_src, g_pre, gn, r);
                copy_chunk_with_wrap(ring.payload_buf, gsp, r * row_bytes, row_bytes,
                                     src8 + sr * row_bytes, threadIdx.x, blockDim.x);
            }
        }
    }

    // ---- head part ----
    if (h_rows) {
        const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
        const W* wbase = reinterpret_cast<const W*>(node + HEAD_HDR_WORDS);
        // As in producer_head_kernel: 8-wide loads only from 16-byte aligned
        // bases (row and slot offsets are multiples of 16 bytes then).
        const bool vec8 = (hidden % 8u) == 0u
            && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u
            && (reinterpret_cast<uintptr_t>(wbase) & 15u) == 0u;
        uint32_t chunk = (hidden + NW - 1) / NW;
        if (vec8) chunk = (chunk + 7u) & ~7u;
        const uint32_t w_lo = warp * chunk;
        const uint32_t w_hi = (w_lo + chunk < hidden) ? w_lo + chunk : hidden;
        __shared__ uint64_t s_hrow[HEAD_RT];
        __shared__ float    s_red[PRODUCER_BLOCK_DIM / 32][HEAD_RT * HEAD_ST];
        for (uint64_t it = blockIdx.x; it < h_items; it += P) {
            const uint64_t rt = it / n_sg;
            const uint32_t sg = static_cast<uint32_t>(it - rt * n_sg);
            const uint64_t r0 = rt * HEAD_RT;
            const uint32_t nr = static_cast<uint32_t>(
                (h_rows - r0) < HEAD_RT ? (h_rows - r0) : HEAD_RT);
            const uint32_t s0 = sg * HEAD_ST;
            const uint32_t ns = (nact - s0) < HEAD_ST ? (nact - s0) : HEAD_ST;
            if (threadIdx.x < HEAD_RT)
                s_hrow[threadIdx.x] = (threadIdx.x < nr)
                    ? table_src_row(h_src, h_pre, hn, r0 + threadIdx.x) : 0;
            __syncthreads();
            const T* hp[HEAD_RT];
            const W* wp[HEAD_ST];
#pragma unroll
            for (uint32_t rr = 0; rr < HEAD_RT; ++rr)
                hp[rr] = src + s_hrow[rr < nr ? rr : 0] * hidden;
#pragma unroll
            for (uint32_t ss = 0; ss < HEAD_ST; ++ss)
                wp[ss] = wbase + static_cast<uint64_t>(s_list[s0 + (ss < ns ? ss : 0)]) * hidden;
            float a[HEAD_RT][HEAD_ST];
#pragma unroll
            for (uint32_t rr = 0; rr < HEAD_RT; ++rr)
#pragma unroll
                for (uint32_t ss = 0; ss < HEAD_ST; ++ss) a[rr][ss] = 0.f;
            if (vec8) {
#pragma unroll 2
                for (uint32_t i = w_lo + lane * 8u; i < w_hi; i += 256u) {
                    float x[HEAD_RT][8], w[HEAD_ST][8];
#pragma unroll
                    for (uint32_t rr = 0; rr < HEAD_RT; ++rr)
                        if (rr < nr) head_load8(hp[rr] + i, x[rr]);
#pragma unroll
                    for (uint32_t ss = 0; ss < HEAD_ST; ++ss)
                        if (ss < ns) head_load8(wp[ss] + i, w[ss]);
#pragma unroll
                    for (uint32_t rr = 0; rr < HEAD_RT; ++rr)
#pragma unroll
                        for (uint32_t ss = 0; ss < HEAD_ST; ++ss)
                            if (rr < nr && ss < ns) {
                                float v = a[rr][ss];
#pragma unroll
                                for (int e = 0; e < 8; ++e) v = fmaf(w[ss][e], x[rr][e], v);
                                a[rr][ss] = v;
                            }
                }
            } else {
                for (uint32_t i = w_lo + lane; i < w_hi; i += 32u) {
                    float x[HEAD_RT], w[HEAD_ST];
#pragma unroll
                    for (uint32_t rr = 0; rr < HEAD_RT; ++rr)
                        x[rr] = (rr < nr) ? head_elem(hp[rr] + i) : 0.f;
#pragma unroll
                    for (uint32_t ss = 0; ss < HEAD_ST; ++ss)
                        w[ss] = (ss < ns) ? head_elem(wp[ss] + i) : 0.f;
#pragma unroll
                    for (uint32_t rr = 0; rr < HEAD_RT; ++rr)
#pragma unroll
                        for (uint32_t ss = 0; ss < HEAD_ST; ++ss)
                            if (rr < nr && ss < ns) a[rr][ss] = fmaf(w[ss], x[rr], a[rr][ss]);
                }
            }
#pragma unroll
            for (uint32_t rr = 0; rr < HEAD_RT; ++rr)
#pragma unroll
                for (uint32_t ss = 0; ss < HEAD_ST; ++ss) {
                    float v = a[rr][ss];
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        v += __shfl_down_sync(0xffffffffu, v, off);
                    if (lane == 0) s_red[warp][rr * HEAD_ST + ss] = v;
                }
            __syncthreads();
            const uint32_t pidx = threadIdx.x, rr = pidx / HEAD_ST, ss = pidx % HEAD_ST;
            if (pidx < HEAD_RT * HEAD_ST && rr < nr && ss < ns) {
                float v = 0.f;
                for (uint32_t w = 0; w < NW; ++w) v += s_red[w][pidx];
                const uint32_t k = static_cast<uint32_t>(s_list[s0 + ss]);
                const int32_t m = s_mode[k];
                const uint64_t sr = s_hrow[rr];
                float outv = __int_as_float(0x7fc00000);   // NaN: no score at this node
                if (m == HEAD_SINGLE) {
                    outv = v + s_bias[k];
                } else if (sr < acc_rows) {
                    float* cell = acc + sr * HEAD_MAX_ACC + static_cast<uint32_t>(s_acc[k]);
                    if (m == HEAD_FIRST)      *cell = v;
                    else if (m == HEAD_MID)   *cell = *cell + v;
                    else                      outv = (*cell + v) + s_bias[k];   // LAST
                }
                store_f32_with_wrap(ring, hsp, ((r0 + rr) * K + k) * sizeof(float), outv);
            }
            __syncthreads();   // s_hrow / s_red are reused by the next item
        }
        // Free slots' columns: NaN (never delivered; the host routes placed slots).
        for (uint64_t j = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; j < fill;
             j += uint64_t(P) * blockDim.x) {
            const uint32_t k = static_cast<uint32_t>(j % K);
            if (s_mode[k] == HEAD_FREE)
                store_f32_with_wrap(ring, hsp, j * sizeof(float), __int_as_float(0x7fc00000));
        }
    }

    if (!join_last_block(P)) return;
    uint64_t totals[2];
    uint64_t allocs[2];
    uint32_t count = 0;
    if (g_on) { totals[count] = g_total; allocs[count] = g_alloc; ++count; }
    if (h_on) { totals[count] = h_total; allocs[count] = h_alloc; ++count; }
    publish_tasks(ring, task_head, payload_head, totals, allocs, count);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void launch_producer_static(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes,
    uint32_t         hook_type,
    cudaStream_t     stream)
{
    const uint32_t n_blocks = producer_grid_blocks(nbytes);
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
    const uint32_t n_blocks = producer_grid_blocks(nbytes_upper);
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
    // A multiple of K (clean block-to-chunk partition) within the cap.
    const uint32_t n_blocks = producer_chunked_grid_blocks(nbytes_upper, K);
    producer_chunked_kernel<<<n_blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, chunk_bytes_dev_ptr, K, hook_type);
}

void launch_producer_gather(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   seg_table_dev_ptr,
    int64_t          table_numel,
    uint64_t         row_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream)
{
    if (seg_table_dev_ptr == nullptr || table_numel < 1) {
        throw std::invalid_argument(
            "gather producer requires a row table holding its segment count");
    }
    // Grid sized by the upper bound (a capture-time constant); the kernel
    // strides fewer rows when the table selects fewer.
    const uint32_t n_blocks = producer_grid_blocks(nbytes_upper);
    producer_gather_kernel<<<n_blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, seg_table_dev_ptr, table_numel, row_bytes,
        hook_type);
}

cudaError_t launch_producer_gather_device_updatable(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   seg_table_dev_ptr,
    int64_t          table_numel,
    uint64_t         row_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream)
{
#if !DMX_HAS_DEVICE_GRAPH_UPDATE
    (void)ring; (void)d_src; (void)nbytes_upper; (void)seg_table_dev_ptr;
    (void)table_numel; (void)row_bytes; (void)hook_type; (void)stream;
    return cudaErrorNotSupported;
#else
    if (seg_table_dev_ptr == nullptr || table_numel < 1)
        return cudaErrorInvalidValue;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3(producer_grid_blocks(nbytes_upper));
    cfg.blockDim = dim3(PRODUCER_BLOCK_DIM);
    cfg.dynamicSmemBytes = 0;
    cfg.stream   = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attr[0].val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    attr[0].val.deviceUpdatableKernelNode.devNode = nullptr;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, producer_gather_kernel,
                              ring, d_src, nbytes_upper, seg_table_dev_ptr,
                              table_numel, row_bytes, hook_type);
#endif
}

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
    cudaStream_t     stream)
{
    if (K == 0 || K > HEAD_MAX_SLOTS || hidden == 0 ||
        (src_elem_bytes != 2 && src_elem_bytes != 4) ||
        seg_table_dev_ptr == nullptr || table_numel < 1) {
        throw std::invalid_argument(
            "head producer requires 1 <= K <= HEAD_MAX_SLOTS, a positive "
            "hidden size, a bf16 or fp32 source and a row table holding its "
            "segment count");
    }
    const uint64_t row_bytes  = static_cast<uint64_t>(hidden) * src_elem_bytes;
    const uint64_t upper_rows = nbytes_upper / row_bytes;
    // One block per possible row (the capture-time bound) or, for
    // decode-sized bounds, per (row, hidden slice); capped at one wave.
    uint64_t blocks = upper_rows;
    const uint64_t split_rows = upper_rows < HEAD_SPLIT_ROWS ? upper_rows : HEAD_SPLIT_ROWS;
    if (blocks < split_rows * HEAD_SPLITS) blocks = split_rows * HEAD_SPLITS;
    if (blocks < 1) blocks = 1;
    if (blocks > PRODUCER_MAX_BLOCKS) blocks = PRODUCER_MAX_BLOCKS;
    const uint32_t grid = static_cast<uint32_t>(blocks);
    if (src_elem_bytes == 2) {
        producer_head_kernel<__nv_bfloat16><<<grid, PRODUCER_BLOCK_DIM, 0, stream>>>(
            ring, reinterpret_cast<const __nv_bfloat16*>(d_src), upper_rows,
            seg_table_dev_ptr, table_numel, hidden, arena, K, hook_type);
    } else {
        producer_head_kernel<float><<<grid, PRODUCER_BLOCK_DIM, 0, stream>>>(
            ring, reinterpret_cast<const float*>(d_src), upper_rows,
            seg_table_dev_ptr, table_numel, hidden, arena, K, hook_type);
    }
}

uint32_t producer_fused_grid(uint64_t nbytes_upper, uint64_t upper_rows)
{
    uint64_t blocks = producer_grid_blocks(nbytes_upper);
    // Head capacity at the capture-time bound: one block per (row tile, slot
    // group); at most HEAD_MAX_SLOTS / HEAD_ST groups.
    const uint64_t hb =
        ((upper_rows + HEAD_RT - 1) / HEAD_RT) * (HEAD_MAX_SLOTS / HEAD_ST);
    if (hb > blocks) blocks = hb;
    if (blocks < 1) blocks = 1;
    if (blocks > PRODUCER_MAX_BLOCKS) blocks = PRODUCER_MAX_BLOCKS;
    return static_cast<uint32_t>(blocks);
}

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
    cudaStream_t     stream)
{
    if (hidden == 0 || (src_elem_bytes != 2 && src_elem_bytes != 4) ||
        (head_table != nullptr && (K == 0 || K > HEAD_MAX_SLOTS)) ||
        (gather_table != nullptr && gather_table_numel < 1) ||
        (head_table != nullptr && head_table_numel < 1)) {
        throw std::invalid_argument(
            "fused producer requires a positive hidden size, a bf16 or fp32 "
            "source, row tables holding their segment counts and, with "
            "heads, 1 <= K <= HEAD_MAX_SLOTS");
    }
    const uint64_t row_bytes  = static_cast<uint64_t>(hidden) * src_elem_bytes;
    const uint64_t upper_rows = nbytes_upper / row_bytes;
    const uint32_t blocks = producer_fused_grid(nbytes_upper, upper_rows);
    if (src_elem_bytes == 2) {
        if (weight_bf16)
            producer_fused_kernel<__nv_bfloat16, __nv_bfloat16><<<blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
                ring, reinterpret_cast<const __nv_bfloat16*>(d_src), upper_rows, hidden,
                gather_table, gather_table_numel, head_table, head_table_numel,
                node_arena, K, acc, acc_rows, hook_type);
        else
            producer_fused_kernel<__nv_bfloat16, float><<<blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
                ring, reinterpret_cast<const __nv_bfloat16*>(d_src), upper_rows, hidden,
                gather_table, gather_table_numel, head_table, head_table_numel,
                node_arena, K, acc, acc_rows, hook_type);
    } else {
        if (weight_bf16)
            producer_fused_kernel<float, __nv_bfloat16><<<blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
                ring, reinterpret_cast<const float*>(d_src), upper_rows, hidden,
                gather_table, gather_table_numel, head_table, head_table_numel,
                node_arena, K, acc, acc_rows, hook_type);
        else
            producer_fused_kernel<float, float><<<blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
                ring, reinterpret_cast<const float*>(d_src), upper_rows, hidden,
                gather_table, gather_table_numel, head_table, head_table_numel,
                node_arena, K, acc, acc_rows, hook_type);
    }
}

void launch_record_producer_static(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream) {
    const uint32_t blocks = producer_grid_blocks(nbytes);
    record_producer_static_kernel<<<blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes, emit_gate, emit_value);
}

void launch_record_producer_prefix(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   row_count_dev_ptr,
    uint64_t         row_bytes,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream) {
    if (row_bytes == 0) {
        throw std::invalid_argument(
            "record prefix producer requires positive row bytes");
    }
    const uint32_t blocks = producer_grid_blocks(nbytes_upper);
    record_producer_prefix_kernel<<<blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, row_count_dev_ptr, row_bytes,
        emit_gate, emit_value);
}

void launch_record_producer_chunked(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes_upper,
    const int64_t*   chunk_bytes_dev_ptr,
    uint32_t         chunk_count,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream) {
    if (chunk_count == 0 || chunk_count > PRODUCER_MAX_K ||
        nbytes_upper % chunk_count != 0) {
        throw std::invalid_argument(
            "record chunked producer requires a valid evenly divided chunk count");
    }
    const uint32_t blocks =
        producer_chunked_grid_blocks(nbytes_upper, chunk_count);
    record_producer_chunked_kernel<<<blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, chunk_bytes_dev_ptr, chunk_count,
        emit_gate, emit_value);
}

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
    cudaStream_t     stream) {
    if (batch == 0 || feature_bytes == 0) {
        throw std::invalid_argument(
            "record sequence-prefix producer requires positive batch and feature bytes");
    }
    uint64_t max_rows = feature_bytes == 0 ? 0 : nbytes_upper / feature_bytes;
    if (max_rows == 0) max_rows = 1;
    const uint32_t blocks = static_cast<uint32_t>(
        max_rows < RECORD_PACK_MAX_BLOCKS ? max_rows : RECORD_PACK_MAX_BLOCKS);
    record_producer_seq_prefix_pack_kernel<<<
        blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, valid_count_dev_ptr,
        valid_prefix_sum_dev_ptr, batch, feature_bytes, emit_gate, emit_value);
}

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
    cudaStream_t     stream) {
    if (segment_count == 0 || feature_bytes == 0) {
        throw std::invalid_argument(
            "record segmented producer requires positive segment count and feature bytes");
    }
    uint64_t max_rows = feature_bytes == 0 ? 0 : nbytes_upper / feature_bytes;
    if (max_rows == 0) max_rows = 1;
    const uint32_t blocks = static_cast<uint32_t>(
        max_rows < RECORD_PACK_MAX_BLOCKS ? max_rows : RECORD_PACK_MAX_BLOCKS);
    record_producer_segmented_pack_kernel<<<
        blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, segment_start_dev_ptr,
        segment_end_dev_ptr, segment_count, feature_bytes,
        emit_gate, emit_value);
}

}  // namespace ring
