// ring/producer.cu — Producer kernel and host-side launcher.
//
// The kernel never spins.  Backpressure is handled by cuStreamWaitValue32
// on the condition tensor BEFORE the kernel is launched (host side).
// One tensor = one task entry.  No chunking.
//
// After publishing, the kernel writes to d_condition[hook_idx]:
//   Normal path:  d_condition[hook_idx] = 0  (COND_RESET — ready for next forward)
//   Large bypass: d_condition[hook_idx] = 1  (COND_PENDING — drain must D2H + ack)
//
// Large bypass: condition=1 is written BEFORE task_publish to prevent a race
// where the drain processes the entry and acks (writes 0) before the kernel
// writes 1 — which would overwrite the ack and cause deadlock.  The
// __threadfence inside task_publish ensures condition=1 is globally visible
// before ready_seq (which the drain polls).

#include "producer.cuh"
#include "payload_ring.cuh"
#include "task_ring.cuh"
#include "ring_config.h"
#include <cstdio>

namespace ring {

// --- Per-hook-type diagnostic counters (device side) ---
#define HOOK_TYPE_MAX 32
__device__ unsigned long long g_diag_hook_writes[HOOK_TYPE_MAX];
__device__ unsigned long long g_diag_hook_entered[HOOK_TYPE_MAX];
__device__ unsigned long long g_diag_hook_active[HOOK_TYPE_MAX];

void diag_reset_hook_counters() {
    unsigned long long zeros[HOOK_TYPE_MAX] = {};
    cudaMemcpyToSymbol(g_diag_hook_writes,  zeros, sizeof(zeros));
    cudaMemcpyToSymbol(g_diag_hook_entered, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(g_diag_hook_active,  zeros, sizeof(zeros));
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
    DIAG_PRINT_ARRAY("writes",   g_diag_hook_writes);
}

__device__ bool g_ring_null_mode = false;

void set_ring_null_mode(bool enabled) {
    cudaMemcpyToSymbol(g_ring_null_mode, &enabled, sizeof(bool));
}

static_assert(PAYLOAD_ALIGN == sizeof(uint4),
    "PAYLOAD_ALIGN must equal sizeof(uint4) for vectorized D2D copies");

__device__ inline void d2d_copy_vectorized(
    uint8_t*       dst,
    const uint8_t* src,
    uint64_t       nbytes)
{
    constexpr uint64_t VEC = sizeof(uint4);
    const uint64_t n_vec = nbytes / VEC;
    const uint64_t tail  = nbytes - n_vec * VEC;

    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4*       d4 = reinterpret_cast<uint4*>(dst);
    for (uint64_t i = threadIdx.x; i < n_vec; i += blockDim.x)
        d4[i] = s4[i];

    const uint64_t tb = n_vec * VEC;
    for (uint64_t i = threadIdx.x; i < tail; i += blockDim.x)
        dst[tb + i] = src[tb + i];
}

// ---------------------------------------------------------------------------
// producer_kernel
// ---------------------------------------------------------------------------
__global__ void producer_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        src_bytes,
    uint32_t        hook_type,
    bool            large_bypass,
    uint32_t*       d_condition,
    uint32_t        hook_idx)
{
    if (threadIdx.x == 0 && hook_type < HOOK_TYPE_MAX)
        atomicAdd(&g_diag_hook_entered[hook_type], 1ULL);

    if (g_ring_null_mode) return;

    if (threadIdx.x == 0 && hook_type < HOOK_TYPE_MAX)
        atomicAdd(&g_diag_hook_active[hook_type], 1ULL);

    // --- Large tensor bypass ---
    if (large_bypass) {
        if (threadIdx.x == 0) {
            uint64_t task_head = *ring.task_head;

            TaskEntry entry{};
            entry.tensor_total_bytes = src_bytes;
            entry.device_src_ptr     = src;

            // Write COND_PENDING BEFORE task_publish.  The __threadfence
            // inside task_publish ensures this write is globally visible
            // before ready_seq.  If we wrote AFTER, the drain could see
            // the entry, ack (write 0), and then this kernel would
            // overwrite the ack with 1 → deadlock.
            if (d_condition)
                d_condition[hook_idx] = COND_PENDING;

            task_publish(ring.task_entries, ring.task_cap, task_head, entry);
            *ring.task_head = task_head + 1;

            if (hook_type < HOOK_TYPE_MAX)
                atomicAdd(&g_diag_hook_writes[hook_type], 1ULL);
        }
        return;
    }

    // --- Normal path: D2D copy + publish + condition reset ---
    __shared__ ProducerShmem sh;

    const uint64_t alloc_bytes = align_up(src_bytes, PAYLOAD_ALIGN);

    if (threadIdx.x == 0) {
        uint64_t task_head    = *ring.task_head;
        uint64_t payload_head = *ring.payload_head;

        sh.task_head    = task_head;
        sh.payload_head = payload_head;
        sh.spans        = payload_compute_spans(payload_head, ring.payload_cap,
                                                 alloc_bytes);
    }

    __syncthreads();

    const TwoSpan& spans = sh.spans;

    {
        const uint64_t copy1 = (src_bytes < spans.len1) ? src_bytes : spans.len1;
        d2d_copy_vectorized(ring.payload_buf + spans.off1, src, copy1);
    }
    {
        const uint64_t data_in_span1 = (src_bytes < spans.len1) ? src_bytes : spans.len1;
        const uint64_t copy2 = src_bytes - data_in_span1;
        if (copy2 > 0) {
            d2d_copy_vectorized(ring.payload_buf + spans.off2,
                                src + data_in_span1, copy2);
        }
    }

    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t task_head    = sh.task_head;
        uint64_t payload_head = sh.payload_head;

        payload_advance_head(payload_head, alloc_bytes);

        const uint64_t data_in_span1 = (src_bytes < spans.len1) ? src_bytes : spans.len1;

        TaskEntry entry{};
        entry.tensor_total_bytes = src_bytes;
        entry.payload_off1       = spans.off1;
        entry.payload_len1       = data_in_span1;
        entry.payload_off2       = spans.off2;
        entry.payload_len2       = src_bytes - data_in_span1;
        entry.device_src_ptr     = nullptr;

        task_publish(ring.task_entries, ring.task_cap, task_head, entry);
        *ring.task_head    = task_head + 1;
        *ring.payload_head = payload_head;

        // Device-side condition reset (AFTER publish is fine for normal path:
        // nobody reads this condition until the next forward, which starts
        // after the graph completes — implicit global visibility).
        if (d_condition)
            d_condition[hook_idx] = COND_RESET;

        if (hook_type < HOOK_TYPE_MAX)
            atomicAdd(&g_diag_hook_writes[hook_type], 1ULL);
    }
}

// ---------------------------------------------------------------------------
void launch_producer(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint32_t         hook_type,
    bool             large_bypass,
    uint32_t*        d_condition,
    uint32_t         hook_idx,
    cudaStream_t     stream)
{
    producer_kernel<<<1, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, src_bytes, hook_type, large_bypass, d_condition, hook_idx);
}

}  // namespace ring
