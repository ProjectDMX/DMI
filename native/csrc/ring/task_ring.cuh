// ring/task_ring.cuh -- Task publication ring device operations.
//
// The task ring is a fixed-size FIFO of 64-bit publication slots. A counter
// tracks the producer's position:
//
//   task_head : next slot index the producer will claim (monotonically increasing)
//
// The consumer (CPU drain thread) tracks its own tail via a CPU-only shadow.
//
// Free slots available = capacity - (head - tail).
// Physical slot index  = logical_counter % capacity.
//
// Publish protocol (producer):
//   1. Finish writing the payload for the task.
//   2. System-scope release-store READY | actual_bytes.
//   3. Increment head.
//
// Consume protocol (CPU consumer):
//   1. Acquire-load slots[tail % cap] until READY is set.
//   2. Decode actual bytes from that acquired word.
//   3. Process (D2H, staging, ...).
//   4. Clear slots[tail % cap] to zero.
//   5. Increment tail (CPU-only shadow).
//
// CUDA graph constraints:
//   All device pointers must be preallocated before graph capture.

#pragma once

#include "publication_word.h"

#include <cstdint>

#ifdef __CUDACC__
#  include <cuda/atomic>
#  include <cuda_runtime.h>
#endif

namespace ring {

// ---------------------------------------------------------------------------
// task_free_slots -- available task slots the producer may claim.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
__host__ __device__
#endif
inline uint64_t task_free_slots(
    uint64_t head, uint64_t tail, uint64_t capacity)
{
    return capacity - (head - tail);
}

// ---------------------------------------------------------------------------
// Device-only ring operations (require CUDA compilation).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__

// ---------------------------------------------------------------------------
// task_ring_init -- initialise every publication slot to not-ready.
// ---------------------------------------------------------------------------
inline void task_ring_init(uint64_t* slots, uint64_t capacity,
                           cudaStream_t stream = 0)
{
    cudaMemsetAsync(slots, 0,
                    static_cast<size_t>(capacity) * sizeof(uint64_t),
                    stream);
}

// ---------------------------------------------------------------------------
// task_publish -- publish actual bytes to the CPU consumer.
// ---------------------------------------------------------------------------
__device__ inline void task_publish(
    uint64_t* slots,
    uint64_t  capacity,
    uint64_t  seq_no,
    uint64_t  actual_bytes)
{
    const uint64_t idx = seq_no % capacity;
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ready(
        slots[idx]);
    // The caller has already acquired the cross-block completion join. No
    // additional fence is needed here: this system-scope release store orders
    // the joined payload writes before the CPU can acquire the publication.
    ready.store(PUBLICATION_READY | actual_bytes,
                cuda::memory_order_release);
}

#endif  // __CUDACC__

}  // namespace ring
