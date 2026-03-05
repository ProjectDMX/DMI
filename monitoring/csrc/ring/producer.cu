// ring/producer.cu — Definitions of the producer kernel and host-side
// launchers.  The kernel body lives here (not in producer.cuh) so that
// multiple .cu files can include producer.cuh (for the declarations and
// helper types) without causing duplicate-symbol link errors.

#include "producer.cuh"
#include "payload_ring.cuh"
#include "task_ring.cuh"

namespace ring {

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
                                          ring.payload_cap) < this_chunk)
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
                                          ring.payload_cap) < this_chunk)
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
            sh.spans        = dropped ? TwoSpan{0, 0, 0, 0}
                                      : payload_compute_spans(payload_head,
                                                              ring.payload_cap,
                                                              this_chunk);
        }

        __syncthreads();

        if (sh.dropped) {
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

        // Phase 2: all threads — cooperative D2D copy (grid-stride)
        const TwoSpan&  spans   = sh.spans;
        const uint8_t*  src_ptr = src + chunk_off;

        for (uint64_t i = threadIdx.x; i < spans.len1; i += blockDim.x)
            ring.payload_buf[spans.off1 + i] = src_ptr[i];

        for (uint64_t i = threadIdx.x; i < spans.len2; i += blockDim.x)
            ring.payload_buf[spans.off2 + i] = src_ptr[spans.len1 + i];

        __threadfence();
        __syncthreads();

        // Phase 3: thread 0 — publish TaskEntry
        if (threadIdx.x == 0) {
            uint32_t flags = 0;
            if (ci == 0)            flags |= TASK_FLAG_IS_FIRST;
            if (ci == n_chunks - 1) flags |= TASK_FLAG_IS_LAST;

            payload_advance_head(payload_head, this_chunk);

            TaskEntry entry{};
            entry.seq_no             = task_head;
            entry.logical_task_id    = logical_task_id;
            entry.chunk_offset_bytes = chunk_off;
            entry.tensor_total_bytes = src_bytes;
            entry.payload_off1       = spans.off1;
            entry.payload_len1       = spans.len1;
            entry.payload_off2       = spans.off2;
            entry.payload_len2       = spans.len2;
            entry.chunk_idx          = static_cast<uint32_t>(ci);
            entry.hook_type          = hook_type;
            entry.hook_id            = hook_id;
            entry.flags              = flags;
            entry.reason             = DROP_REASON_NONE;

            task_publish(ring.task_entries, ring.task_cap, task_head, entry);
            ++task_head;
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
