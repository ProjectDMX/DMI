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

#include <cstdlib>
#include <cstdio>
#include "producer.cuh"
#include "payload_ring.cuh"
#include "task_ring.cuh"
#include "ring_config.h"

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

    // Segment starts and prefix row offsets, clamped exactly as before, once
    // per block in shared memory.
    __shared__ uint64_t seg_start[SEGMENT_PACK_SHARED_SEGMENTS];
    __shared__ uint64_t seg_prefix[SEGMENT_PACK_SHARED_SEGMENTS + 1];
    const bool shared_segments = segment_count <= SEGMENT_PACK_SHARED_SEGMENTS;
    uint64_t total_rows = 0;
    if (shared_segments) {
        for (uint32_t segment = threadIdx.x; segment < segment_count; segment += blockDim.x) {
            int64_t start = segment_start_dev_ptr[segment];
            int64_t end = segment_end_dev_ptr[segment];
            if (start < 0) start = 0;
            if (end < start) end = start;
            if (static_cast<uint64_t>(start) > input_rows) start = static_cast<int64_t>(input_rows);
            if (static_cast<uint64_t>(end) > input_rows) end = static_cast<int64_t>(input_rows);
            seg_start[segment] = static_cast<uint64_t>(start);
            seg_prefix[segment + 1] = static_cast<uint64_t>(end - start);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            seg_prefix[0] = 0;
            for (uint32_t segment = 0; segment < segment_count; ++segment) {
                seg_prefix[segment + 1] += seg_prefix[segment];
            }
        }
        __syncthreads();
        total_rows = seg_prefix[segment_count];
    } else {
        for (uint32_t segment = 0; segment < segment_count; ++segment) {
            int64_t start = segment_start_dev_ptr[segment];
            int64_t end = segment_end_dev_ptr[segment];
            if (start < 0) start = 0;
            if (end < start) end = start;
            if (static_cast<uint64_t>(start) > input_rows) start = static_cast<int64_t>(input_rows);
            if (static_cast<uint64_t>(end) > input_rows) end = static_cast<int64_t>(input_rows);
            total_rows += static_cast<uint64_t>(end - start);
        }
    }
    if (total_rows > input_rows) total_rows = input_rows;

    const uint64_t actual_bytes = total_rows * feature_bytes;
    const uint64_t alloc_bytes = align_up(actual_bytes, PAYLOAD_ALIGN);
    const uint64_t task_head = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan spans = payload_compute_spans(
        payload_head, ring.payload_cap, alloc_bytes);

    if (shared_segments) {
        // Grid-stride over fixed-size units of the packed output: coalesced like
        // a contiguous copy. A unit never crosses a row (unit divides
        // feature_bytes) or the ring wrap (span lengths are multiples of 16).
        const uintptr_t src_bits = reinterpret_cast<uintptr_t>(src);
        uint64_t unit = 1;
        if (feature_bytes % 16 == 0 && src_bits % 16 == 0) unit = 16;
        else if (feature_bytes % 8 == 0 && src_bits % 8 == 0) unit = 8;
        else if (feature_bytes % 4 == 0 && src_bits % 4 == 0) unit = 4;
        const uint64_t units = actual_bytes / unit;
        const uint64_t gtid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
        for (uint64_t u = gtid; u < units; u += stride) {
            const uint64_t off = u * unit;
            const uint64_t row = off / feature_bytes;
            const uint64_t within = off - row * feature_bytes;
            uint32_t lo = 0, hi = segment_count;  // largest seg with prefix[seg] <= row
            while (hi - lo > 1) {
                const uint32_t mid = (lo + hi) >> 1;
                if (seg_prefix[mid] <= row) lo = mid; else hi = mid;
            }
            while (seg_prefix[lo + 1] <= row) ++lo;  // skip empty segments
            const uint8_t* sp = src + (seg_start[lo] + (row - seg_prefix[lo])) * feature_bytes + within;
            uint8_t* dp = off < spans.len1
                ? ring.payload_buf + spans.off1 + off
                : ring.payload_buf + spans.off2 + (off - spans.len1);
            if (unit == 16) {
                *reinterpret_cast<uint4*>(dp) = *reinterpret_cast<const uint4*>(sp);
            } else if (unit == 8) {
                *reinterpret_cast<uint2*>(dp) = *reinterpret_cast<const uint2*>(sp);
            } else if (unit == 4) {
                *reinterpret_cast<uint32_t*>(dp) = *reinterpret_cast<const uint32_t*>(sp);
            } else {
                *dp = *sp;
            }
        }
    } else {
        // More segments than fit in shared memory: row-by-row walk.
        for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
            uint64_t remaining = row;
            uint64_t source_row = 0;
            for (uint32_t segment = 0; segment < segment_count; ++segment) {
                int64_t start = segment_start_dev_ptr[segment];
                int64_t end = segment_end_dev_ptr[segment];
                if (start < 0) start = 0;
                if (end < start) end = start;
                if (static_cast<uint64_t>(start) > input_rows) start = static_cast<int64_t>(input_rows);
                if (static_cast<uint64_t>(end) > input_rows) end = static_cast<int64_t>(input_rows);
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
    }

    __threadfence();
    __syncthreads();
    publish_last_block_arrives(
        ring, task_head, payload_head, alloc_bytes, actual_bytes);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

// DMI_PRODUCER_LARGE_BLOCKS overrides the block count for copies above TIER3_THRESHOLD
// (default TIER3_BLOCKS). Smaller tiers are unchanged.
static uint32_t large_copy_blocks() {
    static const uint32_t blocks = [] {
        const char* v = std::getenv("DMI_PRODUCER_LARGE_BLOCKS");
        const unsigned long n = v ? std::strtoul(v, nullptr, 10) : 0;
        const uint32_t b = n > 0 ? static_cast<uint32_t>(n) : TIER3_BLOCKS;
        std::fprintf(stderr, "DMI_EFFECTIVE_PRODUCER_LARGE_BLOCKS=%u\n", b);
        return b;
    }();
    return blocks;
}

static inline uint32_t pick_tier_blocks(uint64_t nbytes) {
    if      (nbytes <= TIER1_THRESHOLD) return TIER0_BLOCKS;
    else if (nbytes <= TIER2_THRESHOLD) return TIER1_BLOCKS;
    else if (nbytes <= TIER3_THRESHOLD) return TIER2_BLOCKS;
    else                                return large_copy_blocks();
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

void launch_record_producer_static(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         nbytes,
    const int32_t*   emit_gate,
    int32_t          emit_value,
    cudaStream_t     stream) {
    const uint32_t blocks = pick_tier_blocks(nbytes);
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
    const uint32_t blocks = pick_tier_blocks(nbytes_upper);
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
    uint32_t blocks = pick_tier_blocks(nbytes_upper);
    if (blocks < chunk_count) blocks = chunk_count;
    const uint32_t remainder = blocks % chunk_count;
    if (remainder != 0) blocks += chunk_count - remainder;
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
    // Shared-segment path: one thread per 16-byte unit of the largest possible
    // selection, capped; the row-walk fallback uses the same grid.
    const uint64_t threads_needed = nbytes_upper / 16 + 1;
    uint64_t blocks64 = (threads_needed + PRODUCER_BLOCK_DIM - 1) / PRODUCER_BLOCK_DIM;
    if (blocks64 > RECORD_PACK_MAX_BLOCKS) blocks64 = RECORD_PACK_MAX_BLOCKS;
    const uint32_t blocks = static_cast<uint32_t>(blocks64);
    record_producer_segmented_pack_kernel<<<
        blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, nbytes_upper, segment_start_dev_ptr,
        segment_end_dev_ptr, segment_count, feature_bytes,
        emit_gate, emit_value);
}


// ---------------------------------------------------------------------------
// mask_shard_segment_ranges -- segment ranges of selected row shards in a
// column-major (stable) permutation of a [rows, cols] boolean mask.
//
// Rows of column c appear in increasing row order inside column c's block, and
// blocks follow column order. For selected shard k (rows [id_k*L, (id_k+1)*L)),
// column c's rows form one contiguous segment:
//   start[c*count+k] = block_start(c) + #{r < id_k*L : mask[r,c]}
//   end  [c*count+k] = start + #{id_k*L <= r < (id_k+1)*L : mask[r,c]}
// counts[c] = sum_k (end - start). One block; no host synchronization.
// ---------------------------------------------------------------------------
__global__ void mask_shard_segment_ranges_kernel(
    const bool* mask, int64_t rows, int64_t cols, int64_t shard_rows,
    ShardSelection selection, int64_t* starts, int64_t* ends, int64_t* counts) {
    extern __shared__ int32_t partial[];  // [groups][cols][1 + 2*count], per-group counts
    const int count = selection.count;
    const int width = 1 + 2 * count;
    const int groups = blockDim.x / static_cast<int>(cols);
    const int c = threadIdx.x % static_cast<int>(cols);
    const int g = threadIdx.x / static_cast<int>(cols);
    if (g < groups) {
        int64_t total = 0;
        int64_t before[MAX_SHARD_SELECTION] = {0};
        int64_t within[MAX_SHARD_SELECTION] = {0};
        for (int64_t r = g; r < rows; r += groups) {
            const int64_t v = mask[r * cols + c] ? 1 : 0;
            total += v;
            for (int k = 0; k < count; ++k) {
                const int64_t lo = selection.ids[k] * shard_rows;
                if (r < lo) before[k] += v;
                else if (r < lo + shard_rows) within[k] += v;
            }
        }
        int32_t* slot = partial + (static_cast<int64_t>(g) * cols + c) * width;
        slot[0] = static_cast<int32_t>(total);
        for (int k = 0; k < count; ++k) {
            slot[1 + k] = static_cast<int32_t>(before[k]);
            slot[1 + count + k] = static_cast<int32_t>(within[k]);
        }
    }
    __syncthreads();
    // One thread per column sums its groups; thread 0 then only scans the
    // column totals for the block starts.
    __shared__ int64_t col_total[1024];
    __shared__ int64_t col_start[1024];
    int64_t sums[1 + 2 * MAX_SHARD_SELECTION] = {0};
    if (g == 0) {
        for (int gg = 0; gg < groups; ++gg) {
            const int32_t* slot = partial + (static_cast<int64_t>(gg) * cols + c) * width;
            for (int j = 0; j < width; ++j) sums[j] += slot[j];
        }
        col_total[c] = sums[0];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int64_t block_start = 0;
        for (int64_t col = 0; col < cols; ++col) {
            col_start[col] = block_start;
            block_start += col_total[col];
        }
    }
    __syncthreads();
    if (g == 0) {
        int64_t selected = 0;
        for (int k = 0; k < count; ++k) {
            const int64_t start = col_start[c] + sums[1 + k];
            starts[static_cast<int64_t>(c) * count + k] = start;
            ends[static_cast<int64_t>(c) * count + k] = start + sums[1 + count + k];
            selected += sums[1 + count + k];
        }
        counts[c] = selected;
    }
}

void launch_mask_shard_segment_ranges(
    const bool* mask, int64_t rows, int64_t cols, int64_t shard_rows,
    const ShardSelection& selection, int64_t* starts, int64_t* ends,
    int64_t* counts, cudaStream_t stream) {
    if (rows < 0 || rows > INT32_MAX || cols <= 0 || cols > 1024 || selection.count <= 0 ||
        selection.count > static_cast<int32_t>(MAX_SHARD_SELECTION)) {
        throw std::invalid_argument(
            "mask_shard_segment_ranges requires 1 <= cols <= 1024 and 1 <= shard count <= 8");
    }
    const int64_t width = 1 + 2 * static_cast<int64_t>(selection.count);
    int64_t groups = 1024 / cols;
    const int64_t smem_cap = 32 * 1024;  // plus 16 KiB static per-column tables
    while (groups > 1 && groups * cols * width * 4 > smem_cap) --groups;
    const size_t smem = static_cast<size_t>(groups * cols * width) * sizeof(int32_t);
    if (smem > static_cast<size_t>(smem_cap)) {
        // Only very wide masks with several shards exceed the default limit.
        const cudaError_t attr = cudaFuncSetAttribute(
            mask_shard_segment_ranges_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem));
        if (attr != cudaSuccess) {
            throw std::runtime_error(std::string("mask_shard_segment_ranges shared memory: ") +
                                     cudaGetErrorString(attr));
        }
    }
    mask_shard_segment_ranges_kernel<<<1, static_cast<uint32_t>(groups * cols), smem, stream>>>(
        mask, rows, cols, shard_rows, selection, starts, ends, counts);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("mask_shard_segment_ranges launch failed: ") +
                                 cudaGetErrorString(err) + " threads=" +
                                 std::to_string(groups * cols) + " smem=" + std::to_string(smem));
    }
}

}  // namespace ring
