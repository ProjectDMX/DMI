// ring/producer.cu — Definitions of the producer kernel and host-side
// launchers.  The kernel body lives here (not in producer.cuh) so that
// multiple .cu files can include producer.cuh (for the declarations and
// helper types) without causing duplicate-symbol link errors.

#include "producer.cuh"
#include "payload_ring.cuh"
#include "task_ring.cuh"
#include <cstdio>

namespace ring {

// --- Per-hook-type diagnostic counters (device side) ---
#define HOOK_TYPE_MAX 32
__device__ unsigned long long g_diag_hook_writes[HOOK_TYPE_MAX];    // phase 3: published
__device__ unsigned long long g_diag_hook_entered[HOOK_TYPE_MAX];   // kernel entry (before null check)
__device__ unsigned long long g_diag_hook_active[HOOK_TYPE_MAX];    // after null check (real work)
__device__ unsigned long long g_diag_hook_dropped[HOOK_TYPE_MAX];   // drop path

void diag_reset_hook_counters() {
    unsigned long long zeros[HOOK_TYPE_MAX] = {};
    cudaMemcpyToSymbol(g_diag_hook_writes,  zeros, sizeof(zeros));
    cudaMemcpyToSymbol(g_diag_hook_entered, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(g_diag_hook_active,  zeros, sizeof(zeros));
    cudaMemcpyToSymbol(g_diag_hook_dropped, zeros, sizeof(zeros));
}

#define DIAG_PRINT_ARRAY(label, sym) do { \
    unsigned long long h[HOOK_TYPE_MAX]; \
    cudaMemcpyFromSymbol(h, sym, sizeof(h)); \
    unsigned long long total = 0; \
    fprintf(stderr, "[producer diag] %s:", label); \
    for (int i = 0; i < HOOK_TYPE_MAX; ++i) { \
        if (h[i]) { fprintf(stderr, " %d=%llu", i, h[i]); total += h[i]; } \
    } \
    fprintf(stderr, "  total=%llu\n", total); \
} while(0)

void diag_print_hook_counters() {
    DIAG_PRINT_ARRAY("entered",  g_diag_hook_entered);
    DIAG_PRINT_ARRAY("active",   g_diag_hook_active);
    DIAG_PRINT_ARRAY("dropped",  g_diag_hook_dropped);
    DIAG_PRINT_ARRAY("writes",   g_diag_hook_writes);
}

// PAYLOAD_ALIGN is defined in ring_config.h (included via producer.cuh → ring_state.h).

__device__ __host__ inline uint64_t align_up(uint64_t x, uint64_t a) {
    return (x + a - 1) & ~(a - 1);
}

// ---------------------------------------------------------------------------
// Null-mode device flag.
// When true, producer_kernel returns immediately without touching the ring.
// Kernel is still launched with the same <<<1, PRODUCER_BLOCK_DIM>>> so that
// CUDA graph topology is identical between null (warmup) and real (capture) runs.
// Set via set_ring_null_mode() — always call this outside any graph capture region.
// ---------------------------------------------------------------------------
__device__ bool g_ring_null_mode = false;

void set_ring_null_mode(bool enabled) {
    cudaMemcpyToSymbol(g_ring_null_mode, &enabled, sizeof(bool));
}

// ---------------------------------------------------------------------------
// producer_kernel definition
// ---------------------------------------------------------------------------
__global__ void producer_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        src_bytes,
    uint64_t        logical_task_id,
    uint32_t        hook_type,
    uint32_t        hook_id)
{
    if (threadIdx.x == 0 && hook_type < HOOK_TYPE_MAX)
        atomicAdd(&g_diag_hook_entered[hook_type], 1ULL);

    if (g_ring_null_mode) return;  // null/warmup mode: same launch, no ring writes

    if (threadIdx.x == 0 && hook_type < HOOK_TYPE_MAX)
        atomicAdd(&g_diag_hook_active[hook_type], 1ULL);

    __shared__ ProducerShmem sh;

    const uint64_t chunk_sz = ring.cfg.chunk_bytes;
    const uint64_t n_chunks = (src_bytes == 0)
                              ? 1
                              : (src_bytes + chunk_sz - 1) / chunk_sz;

    uint64_t task_head    = 0;
    uint64_t payload_head = 0;
    if (threadIdx.x == 0) {
        task_head    = *ring.task_head;
        payload_head = *ring.payload_head;
    }

    for (uint64_t ci = 0; ci < n_chunks; ++ci) {
        const uint64_t chunk_off  = ci * chunk_sz;
        const uint64_t this_chunk = (src_bytes == 0) ? 0
                                    : (ci + 1 < n_chunks ? chunk_sz
                                                         : src_bytes - chunk_off);
        // Round up to PAYLOAD_ALIGN so every span starts at an aligned offset,
        // enabling vectorized uint4 D2D copies.
        const uint64_t alloc_chunk = align_up(this_chunk, PAYLOAD_ALIGN);

        // Phase 1: thread 0 — backpressure + span reservation
        if (threadIdx.x == 0) {
            bool dropped = false;

            // Volatile reads bypass L1 so the GPU observes CPU-written updates
            // to managed memory (task_tail, payload_tail, consumer_heartbeat).
            const volatile uint64_t* task_tail_v    = ring.task_tail;
            const volatile uint64_t* payload_tail_v = ring.payload_tail;
            const volatile uint64_t* hb_v           = ring.consumer_heartbeat;

            if (ring.cfg.wait_policy == WaitPolicy::INFINITE) {
                while (task_free_slots(task_head, *task_tail_v, ring.task_cap) < 1 ||
                       payload_free_bytes(payload_head, *payload_tail_v,
                                          ring.payload_cap) < alloc_chunk)
                {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(128);
#endif
                }
            } else {
                uint64_t hb_snap = *hb_v;
                uint64_t deadline = clock64() + ring.cfg.no_progress_timeout_cycles;

                while (task_free_slots(task_head, *task_tail_v, ring.task_cap) < 1 ||
                       payload_free_bytes(payload_head, *payload_tail_v,
                                          ring.payload_cap) < alloc_chunk)
                {
                    uint64_t hb_now = *hb_v;
                    if (hb_now != hb_snap) {
                        hb_snap  = hb_now;
                        deadline = clock64() + ring.cfg.no_progress_timeout_cycles;
                    } else if (static_cast<int64_t>(clock64() - deadline) > 0) {
                        dropped = true;
                        break;
                    }
#if __CUDA_ARCH__ >= 700
                    __nanosleep(128);
#endif
                }
            }

            sh.dropped      = dropped ? 1 : 0;
            sh.task_head    = task_head;
            sh.payload_head = payload_head;
            // Reserve aligned size; spans.off1 is always PAYLOAD_ALIGN-aligned
            // because payload_head is kept aligned.
            sh.spans        = dropped ? TwoSpan{0, 0, 0, 0}
                                      : payload_compute_spans(payload_head,
                                                              ring.payload_cap,
                                                              alloc_chunk);
        }

        __syncthreads();

        if (sh.dropped) {
            if (threadIdx.x == 0 && hook_type < HOOK_TYPE_MAX)
                atomicAdd(&g_diag_hook_dropped[hook_type], 1ULL);
            if (threadIdx.x == 0 &&
                ring.cfg.drop_reporting == DropReporting::DROP_TASK &&
                task_free_slots(task_head, *ring.task_tail, ring.task_cap) >= 1)
            {
                uint32_t drop_flags = TASK_FLAG_IS_DROP | TASK_FLAG_IS_LAST;
                if (ci == 0) drop_flags |= TASK_FLAG_IS_FIRST;

                TaskEntry drop{};
                drop.seq_no             = task_head;
                drop.logical_task_id    = logical_task_id;
                drop.chunk_offset_bytes = chunk_off;
                drop.tensor_total_bytes = src_bytes;
                drop.chunk_idx          = static_cast<uint32_t>(ci);
                drop.hook_type          = hook_type;
                drop.hook_id            = hook_id;
                drop.flags              = drop_flags;
                drop.reason             = DROP_REASON_TIMEOUT_NO_PROGRESS;

                task_publish(ring.task_entries, ring.task_cap, task_head, drop);
                ++task_head;
                *ring.task_head = task_head;
            }
            break;
        }

        // Phase 2: all threads — cooperative D2D copy (grid-stride, vectorized)
        //
        // Payload offsets are always PAYLOAD_ALIGN-aligned (payload_head is
        // kept aligned, payload_cap must be a multiple of PAYLOAD_ALIGN).
        // Source tensors from PyTorch are 256-byte aligned (cudaMalloc).
        // So both src and dst are 16-byte aligned → safe to use uint4.
        const TwoSpan&  spans   = sh.spans;
        const uint8_t*  src_ptr = src + chunk_off;

        // --- Span 1: copy this_chunk bytes (spans.len1 may include padding) ---
        // Only the first span can contain actual data up to this_chunk bytes.
        // spans was computed for alloc_chunk; actual data in span1 is
        // min(this_chunk, spans.len1).
        {
            const uint64_t copy1 = (this_chunk < spans.len1) ? this_chunk : spans.len1;
            const uint64_t n16 = copy1 / 16;
            const uint64_t tail1 = copy1 - n16 * 16;
            const uint4* s4 = reinterpret_cast<const uint4*>(src_ptr);
            uint4*       d4 = reinterpret_cast<uint4*>(ring.payload_buf + spans.off1);
            for (uint64_t i = threadIdx.x; i < n16; i += blockDim.x)
                d4[i] = s4[i];
            const uint64_t tb = n16 * 16;
            for (uint64_t i = threadIdx.x; i < tail1; i += blockDim.x)
                ring.payload_buf[spans.off1 + tb + i] = src_ptr[tb + i];
        }

        // --- Span 2 (wrap region): remaining actual data ---
        {
            // Data in span2 = this_chunk - min(this_chunk, spans.len1)
            const uint64_t data_in_span1 = (this_chunk < spans.len1) ? this_chunk : spans.len1;
            const uint64_t copy2 = this_chunk - data_in_span1;
            if (copy2 > 0) {
                const uint64_t n16 = copy2 / 16;
                const uint64_t tail2 = copy2 - n16 * 16;
                const uint4* s4 = reinterpret_cast<const uint4*>(src_ptr + data_in_span1);
                uint4*       d4 = reinterpret_cast<uint4*>(ring.payload_buf + spans.off2);
                for (uint64_t i = threadIdx.x; i < n16; i += blockDim.x)
                    d4[i] = s4[i];
                const uint64_t tb = n16 * 16;
                for (uint64_t i = threadIdx.x; i < tail2; i += blockDim.x)
                    ring.payload_buf[spans.off2 + tb + i] = src_ptr[data_in_span1 + tb + i];
            }
        }

        __threadfence();
        __syncthreads();

        // Phase 3: thread 0 — publish TaskEntry
        if (threadIdx.x == 0) {
            uint32_t flags = 0;
            if (ci == 0)            flags |= TASK_FLAG_IS_FIRST;
            if (ci == n_chunks - 1) flags |= TASK_FLAG_IS_LAST;

            // Advance head by aligned amount (keeps payload_head aligned)
            payload_advance_head(payload_head, alloc_chunk);

            // Compute actual data lengths per span for the consumer D2H copy.
            const uint64_t data_in_span1 = (this_chunk < spans.len1) ? this_chunk : spans.len1;
            const uint64_t data_in_span2 = this_chunk - data_in_span1;

            TaskEntry entry{};
            entry.seq_no             = task_head;
            entry.logical_task_id    = logical_task_id;
            entry.chunk_offset_bytes = chunk_off;
            entry.tensor_total_bytes = src_bytes;
            entry.payload_off1       = spans.off1;
            entry.payload_len1       = data_in_span1;
            entry.payload_off2       = spans.off2;
            entry.payload_len2       = data_in_span2;
            entry.payload_alloc_bytes = alloc_chunk;
            entry.chunk_idx          = static_cast<uint32_t>(ci);
            entry.hook_type          = hook_type;
            entry.hook_id            = hook_id;
            entry.flags              = flags;
            entry.reason             = DROP_REASON_NONE;

            // Compute tensor_total_padded_bytes on IS_FIRST.
            // = (N-1)*chunk_bytes + align_up(last_chunk, 16)
            if (ci == 0) {
                if (src_bytes == 0) {
                    entry.tensor_total_padded_bytes = 0;
                } else {
                    uint64_t last_chunk_sz = src_bytes - (n_chunks - 1) * chunk_sz;
                    entry.tensor_total_padded_bytes =
                        (n_chunks - 1) * chunk_sz + align_up(last_chunk_sz, PAYLOAD_ALIGN);
                }
            } else {
                entry.tensor_total_padded_bytes = 0;
            }

            task_publish(ring.task_entries, ring.task_cap, task_head, entry);
            ++task_head;
            if (hook_type < HOOK_TYPE_MAX)
                atomicAdd(&g_diag_hook_writes[hook_type], 1ULL);
            *ring.task_head    = task_head;
            *ring.payload_head = payload_head;
        }

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
void launch_producer(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint64_t         logical_task_id,
    uint32_t         hook_type,
    uint32_t         hook_id,
    cudaStream_t     stream)
{
    producer_kernel<<<1, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, src_bytes, logical_task_id, hook_type, hook_id);
}

void launch_producer_with_notify(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint64_t         logical_task_id,
    uint32_t         hook_type,
    uint32_t         hook_id,
    cudaHostFn_t     notify_fn,
    void*            notify_arg,
    cudaStream_t     stream)
{
    producer_kernel<<<1, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, src_bytes, logical_task_id, hook_type, hook_id);
    cudaLaunchHostFunc(stream, notify_fn, notify_arg);
}

}  // namespace ring
