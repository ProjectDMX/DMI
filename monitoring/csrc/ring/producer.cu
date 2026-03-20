// ring/producer.cu -- Multi-block producer kernel and host-side launcher.
//
// The kernel never spins.  Space is guaranteed by the pre-forward capacity
// check before kernel launch.  One tensor = one task entry.  No chunking.

#include "producer.cuh"
#include "payload_ring.cuh"
#include "task_ring.cuh"
#include "ring_config.h"

namespace ring {

__device__ bool g_ring_null_mode = false;

// Counter for last-block-arrives pattern.  Reset by the last block after
// publishing.  Safe without host-side reset because producers are serialized
// on one stream -- the next producer_kernel launch cannot start until the
// current one finishes.
__device__ uint32_t g_block_done_counter = 0;

void set_ring_null_mode(bool enabled) {
    cudaMemcpyToSymbol(g_ring_null_mode, &enabled, sizeof(bool));
}

static_assert(PAYLOAD_ALIGN == sizeof(uint4),
    "PAYLOAD_ALIGN must equal sizeof(uint4) for vectorized D2D copies");

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

// ---------------------------------------------------------------------------
// producer_kernel -- multi-block, size-tiered grid
// ---------------------------------------------------------------------------
__global__ void producer_kernel(
    RingState       ring,
    const uint8_t*  src,
    uint64_t        src_bytes,
    uint32_t        hook_type)
{
    if (g_ring_null_mode) return;

    const uint64_t gtid   = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x)  * blockDim.x;

    // All threads read ring heads from global memory (L1/L2 cached).
    // These values are stable for the duration of this kernel -- no other
    // kernel modifies them concurrently.
    const uint64_t alloc_bytes = align_up(src_bytes, PAYLOAD_ALIGN);

    const uint64_t task_head    = *ring.task_head;
    const uint64_t payload_head = *ring.payload_head;
    const TwoSpan  spans        = payload_compute_spans(payload_head,
                                                         ring.payload_cap,
                                                         alloc_bytes);

    // Phase 1: grid-stride D2D copy into payload ring spans.
    {
        const uint64_t copy1 = (src_bytes < spans.len1) ? src_bytes : spans.len1;
        d2d_copy_grid_stride(ring.payload_buf + spans.off1, src, copy1,
                             gtid, stride);
    }
    {
        const uint64_t data_in_span1 = (src_bytes < spans.len1) ? src_bytes : spans.len1;
        const uint64_t copy2 = src_bytes - data_in_span1;
        if (copy2 > 0) {
            d2d_copy_grid_stride(ring.payload_buf + spans.off2,
                                 src + data_in_span1, copy2,
                                 gtid, stride);
        }
    }

    // Ensure all D2D stores are globally visible before metadata publish.
    __threadfence();

    // Phase 2: last-block-arrives publishes metadata.
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t finished = atomicAdd(&g_block_done_counter, 1);
        if (finished == gridDim.x - 1) {
            // Last block: all copy stores are globally visible.
            uint64_t ph = payload_head;
            payload_advance_head(ph, alloc_bytes);

            const uint64_t data_in_span1 = (src_bytes < spans.len1) ? src_bytes : spans.len1;

            TaskEntry entry{};
            entry.tensor_total_bytes = src_bytes;
            entry.payload_off1       = spans.off1;
            entry.payload_len1       = data_in_span1;
            entry.payload_off2       = spans.off2;
            entry.payload_len2       = src_bytes - data_in_span1;

            task_publish(ring.task_entries, ring.task_cap, task_head, entry);
            *ring.task_head    = task_head + 1;
            *ring.payload_head = ph;

            // Reset counter for the next kernel launch on this stream.
            g_block_done_counter = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// launch_producer -- size-tiered grid selection
// ---------------------------------------------------------------------------
void launch_producer(
    const RingState& ring,
    const uint8_t*   d_src,
    uint64_t         src_bytes,
    uint32_t         hook_type,
    cudaStream_t     stream)
{
    // Size-tiered grid: baked into CUDA graph at capture time.
    uint32_t n_blocks;
    if      (src_bytes <= TIER1_THRESHOLD) n_blocks = TIER0_BLOCKS;
    else if (src_bytes <= TIER2_THRESHOLD) n_blocks = TIER1_BLOCKS;
    else if (src_bytes <= TIER3_THRESHOLD) n_blocks = TIER2_BLOCKS;
    else                                   n_blocks = TIER3_BLOCKS;

    producer_kernel<<<n_blocks, PRODUCER_BLOCK_DIM, 0, stream>>>(
        ring, d_src, src_bytes, hook_type);
}

}  // namespace ring
