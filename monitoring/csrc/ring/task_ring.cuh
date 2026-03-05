// ring/task_ring.cuh — Task/control ring device operations.
//
// The task ring is a fixed-size FIFO of TaskEntry slots.  Two 64-bit counters
// track the ring's state:
//
//   task_head : next slot index the producer will claim (monotonically increasing)
//   task_tail : next slot index the consumer expects to read  (monotonically increasing)
//
// Free slots available = capacity - (head - tail).
// Physical slot index  = logical_counter % capacity.
//
// Publish protocol (producer):
//   1. Verify free slots > 0 (or spin/drop in backpressure loop).
//   2. Write all TaskEntry data fields at slot (head % capacity).
//   3. __threadfence() — ensures data is visible before ready_seq.
//   4. Write ready_seq = head  (the slot's logical sequence number).
//   5. Increment head.
//
// Consume protocol (consumer):
//   1. Spin until volatile_read(entries[tail % cap].ready_seq) == tail.
//   2. __threadfence() — acquire: ensures data written before ready_seq is visible.
//   3. Read entry fields.
//   4. Process (D2H, staging, …).
//   5. Reset entries[tail % cap].ready_seq = READY_SEQ_SENTINEL.
//   6. __threadfence() — ensures reset is visible before tail advance.
//   7. Increment tail; write updated consumer_heartbeat.
//
// DROP marker entries (TASK_FLAG_IS_DROP):
//   Published the same way as regular entries.  payload_len1 == payload_len2 == 0.
//   Consumer increments its drop counter but does not stage any D2H transfer.
//
// CUDA graph constraints:
//   All device pointers passed to these functions must be preallocated before
//   graph capture.  No allocations, no host synchronisation in the device paths.

#pragma once

#include "task_entry.h"

#include <cstdint>

#ifdef __CUDACC__
#  include <cuda_runtime.h>
#endif

namespace ring {

// ---------------------------------------------------------------------------
// task_free_slots — available task slots the producer may claim.
// __host__ __device__ so it can be used in both host-side tests and device
// producer kernels.
// ---------------------------------------------------------------------------
__host__ __device__ inline uint64_t task_free_slots(
    uint64_t head, uint64_t tail, uint64_t capacity)
{
    return capacity - (head - tail);
}

// ---------------------------------------------------------------------------
// Device-only ring operations (require CUDA compilation).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__

// ---------------------------------------------------------------------------
// task_ring_init — initialise a device-side task ring.
//
// Sets every byte in the entry array to 0xFF, which gives:
//   ready_seq = READY_SEQ_SENTINEL (0xFFFF…)  — slot not published
//   all other fields = 0xFF…                  — garbage, safely ignored until
//                                               the producer writes them before
//                                               publishing ready_seq.
//
// Call once on host before the first graph capture.  `d_entries` must point to
// device memory of at least capacity * sizeof(TaskEntry) bytes.
// ---------------------------------------------------------------------------
inline void task_ring_init(TaskEntry* d_entries, uint64_t capacity,
                           cudaStream_t stream = 0)
{
    cudaMemsetAsync(d_entries, 0xFF,
                    static_cast<size_t>(capacity) * sizeof(TaskEntry),
                    stream);
}

// ---------------------------------------------------------------------------
// task_publish — write a TaskEntry and publish it to the consumer.
//
// Copies all non-ready_seq fields from `src` into the slot at `seq_no %
// capacity`, issues a __threadfence() to enforce write ordering, then writes
// ready_seq = seq_no.  The consumer's spin loop will observe ready_seq ==
// tail (== seq_no) and proceed.
//
// Call from device producer code only (inside the custom op kernel).
// `seq_no` is the value of task_head BEFORE the producer incremented it.
// ---------------------------------------------------------------------------
__device__ inline void task_publish(
    TaskEntry*       entries,
    uint64_t         capacity,
    uint64_t         seq_no,
    const TaskEntry& src)
{
    uint64_t   idx  = seq_no % capacity;
    TaskEntry& slot = entries[idx];

    // Write all data fields — ordering relative to each other doesn't matter;
    // only the final __threadfence() + ready_seq write establishes the happens-
    // before with the consumer.
    slot.seq_no             = src.seq_no;
    slot.logical_task_id    = src.logical_task_id;
    slot.chunk_offset_bytes = src.chunk_offset_bytes;
    slot.tensor_total_bytes = src.tensor_total_bytes;
    slot.payload_off1       = src.payload_off1;
    slot.payload_len1       = src.payload_len1;
    slot.payload_off2       = src.payload_off2;
    slot.payload_len2       = src.payload_len2;
    slot.chunk_idx          = src.chunk_idx;
    slot.hook_type          = src.hook_type;
    slot.hook_id            = src.hook_id;
    slot.flags              = src.flags;
    slot.reason             = src.reason;
    slot._pad0              = 0;

    // Release fence: all stores above must be visible before ready_seq.
    __threadfence();

    // Publish: consumer spins until it sees this value.
    *reinterpret_cast<volatile uint64_t*>(&slot.ready_seq) = seq_no;
}

// ---------------------------------------------------------------------------
// task_spin_wait — consumer: block until slot `tail` is published.
//
// Spins reading ready_seq with volatile semantics until ready_seq == tail.
// Issues an acquire __threadfence() before returning so that subsequent reads
// of the entry's data fields observe the values written by the producer.
//
// Returns a pointer to the entry.  Valid to read until task_release() resets
// the slot.
// ---------------------------------------------------------------------------
__device__ inline const TaskEntry* task_spin_wait(
    const TaskEntry* entries,
    uint64_t         capacity,
    uint64_t         tail)
{
    uint64_t         idx = tail % capacity;
    const volatile uint64_t* rs =
        reinterpret_cast<const volatile uint64_t*>(&entries[idx].ready_seq);

    while (*rs != tail) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);  // yield briefly to reduce interconnect pressure
#endif
    }

    // Acquire fence: ensures entry data fields are visible after ready_seq.
    __threadfence();
    return entries + idx;
}

// ---------------------------------------------------------------------------
// task_release — consumer: mark slot `tail` as free for reuse.
//
// Resets ready_seq to READY_SEQ_SENTINEL so the slot can be recycled by the
// producer once it sees (head - tail < capacity).
//
// Call AFTER D2H transfers for this entry's payload spans have been launched
// (they may still be in-flight; this only signals the ring slot is free).
// The caller must increment task_tail after this call.
// ---------------------------------------------------------------------------
__device__ inline void task_release(
    TaskEntry* entries,
    uint64_t   capacity,
    uint64_t   tail)
{
    uint64_t idx = tail % capacity;
    __threadfence();  // ensure all reads of the slot are done before reset
    *reinterpret_cast<volatile uint64_t*>(&entries[idx].ready_seq) =
        READY_SEQ_SENTINEL;
}

#endif  // __CUDACC__

// ---------------------------------------------------------------------------
// CPU-side consumer helpers — usable from host C++ compiled with g++ or with
// nvcc's host-compilation phase.  Use GCC/Clang __atomic builtins for
// acquire/release ordering on managed memory written by the GPU producer.
// ---------------------------------------------------------------------------

// task_cpu_ready — non-blocking: returns true if slot `tail` has been
// published by the GPU producer.  An acquire load ensures that, if true,
// all subsequent CPU reads of the entry's data fields are ordered after
// the ready_seq observation.
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

// task_release_cpu — reset slot `tail` to SENTINEL so the producer can
// reuse the slot once task_tail advances past it.  A release store ensures
// all prior CPU reads of the entry's data fields are ordered before the
// sentinel write (i.e., the sentinel is not visible before the reads).
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
