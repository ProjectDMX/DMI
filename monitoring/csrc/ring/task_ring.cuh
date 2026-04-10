// ring/task_ring.cuh -- Task/control ring device operations.
//
// The task ring is a fixed-size FIFO of TaskEntry slots.  A 64-bit counter
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
//   1. Write all TaskEntry data fields at slot (head % capacity).
//   2. __threadfence() -- ensures data is visible before ready_seq.
//   3. Write ready_seq = head  (the slot's logical sequence number).
//   4. Increment head.
//
// Consume protocol (CPU consumer):
//   1. Poll until __atomic_load_n(entries[tail % cap].ready_seq) == tail.
//   2. Read entry fields (acquire ordering from the atomic load).
//   3. Process (D2H, staging, ...).
//   4. Reset entries[tail % cap].ready_seq = READY_SEQ_SENTINEL.
//   5. Increment tail (CPU-only shadow).
//
// CUDA graph constraints:
//   All device pointers must be preallocated before graph capture.

#pragma once

#include "task_entry.h"

#include <cstdint>

#ifdef __CUDACC__
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
// task_ring_init -- initialise a device-side task ring.
//
// Sets every byte in the entry array to 0xFF, which gives:
//   ready_seq = READY_SEQ_SENTINEL (0xFFFF...)  -- slot not published
// ---------------------------------------------------------------------------
inline void task_ring_init(TaskEntry* d_entries, uint64_t capacity,
                           cudaStream_t stream = 0)
{
    cudaMemsetAsync(d_entries, 0xFF,
                    static_cast<size_t>(capacity) * sizeof(TaskEntry),
                    stream);
}

// ---------------------------------------------------------------------------
// task_publish -- write a TaskEntry and publish it to the consumer.
//
// Copies all non-ready_seq fields from `src` into the slot at `seq_no %
// capacity`, issues a __threadfence() to enforce write ordering, then writes
// ready_seq = seq_no.
// ---------------------------------------------------------------------------
__device__ inline void task_publish(
    TaskEntry*       entries,
    uint64_t         capacity,
    uint64_t         seq_no,
    const TaskEntry& src)
{
    uint64_t   idx  = seq_no % capacity;
    TaskEntry& slot = entries[idx];

    // Write all data fields
    slot.tensor_total_bytes = src.tensor_total_bytes;
    slot.payload_off1       = src.payload_off1;
    slot.payload_len1       = src.payload_len1;
    slot.payload_off2       = src.payload_off2;
    slot.payload_len2       = src.payload_len2;

    // Release fence: all stores above must be visible before ready_seq.
    __threadfence();

    // Publish: consumer spins until it sees this value.
    *reinterpret_cast<volatile uint64_t*>(&slot.ready_seq) = seq_no;
}

// ---------------------------------------------------------------------------
// task_release -- consumer: mark slot `tail` as free for reuse (device-side).
// ---------------------------------------------------------------------------
__device__ inline void task_release(
    TaskEntry* entries,
    uint64_t   capacity,
    uint64_t   tail)
{
    uint64_t idx = tail % capacity;
    __threadfence();
    *reinterpret_cast<volatile uint64_t*>(&entries[idx].ready_seq) =
        READY_SEQ_SENTINEL;
}

#endif  // __CUDACC__

// ---------------------------------------------------------------------------
// CPU-side consumer helpers -- usable from host C++ compiled with g++ or nvcc.
// ---------------------------------------------------------------------------

// task_cpu_ready -- non-blocking: returns true if slot `tail` has been
// published by the GPU producer.
inline bool task_cpu_ready(
    const TaskEntry* entries,
    uint64_t         capacity,
    uint64_t         tail)
{
    const uint64_t  idx = tail % capacity;
    const uint64_t* rs  =
        reinterpret_cast<const uint64_t*>(&entries[idx].ready_seq);
    return __atomic_load_n(rs, __ATOMIC_ACQUIRE) == tail;
}

// task_release_cpu -- reset slot `tail` to SENTINEL so the producer can
// reuse the slot.
inline void task_release_cpu(
    TaskEntry* entries,
    uint64_t   capacity,
    uint64_t   tail)
{
    const uint64_t idx = tail % capacity;
    uint64_t*      rs  =
        reinterpret_cast<uint64_t*>(&entries[idx].ready_seq);
    __atomic_store_n(rs, READY_SEQ_SENTINEL, __ATOMIC_RELEASE);
}

}  // namespace ring
