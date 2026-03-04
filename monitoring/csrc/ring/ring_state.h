// ring/ring_state.h — Capture-safe descriptor passed by value into the producer kernel.
//
// All fields are raw device pointers and integral constants (POD).  The struct
// is safe to copy into kernel arguments with no vtable, no destructor.
//
// Counter conventions:
//   task_head, payload_head     : written by producer (GPU)
//   task_tail, payload_tail     : written by consumer (CPU)
//   consumer_heartbeat          : incremented by consumer whenever tails advance;
//                                 read by producer to detect no-progress in
//                                 TIMEOUT_DROP mode.
//
// Tail pointers and consumer_heartbeat must reside in device-accessible memory
// (cudaMallocManaged or cudaHostAlloc+cudaHostAllocMapped) so the GPU producer
// can read them and the CPU drain loop can write them.

#pragma once
#include "task_entry.h"
#include "ring_config.h"
#include <cstdint>

namespace ring {

struct RingState {
    // Task/control ring
    TaskEntry*  task_entries;   // device buffer: task_cap × sizeof(TaskEntry)
    uint64_t    task_cap;       // number of task slots

    uint64_t*   task_head;      // monotonically increasing; producer writes
    uint64_t*   task_tail;      // monotonically increasing; consumer writes

    // Payload byte ring
    uint8_t*    payload_buf;    // device buffer: payload_cap bytes
    uint64_t    payload_cap;    // capacity in bytes

    uint64_t*   payload_head;   // monotonically increasing; producer writes
    uint64_t*   payload_tail;   // monotonically increasing; consumer writes

    // Consumer liveness signal (for TIMEOUT_DROP backpressure)
    uint64_t*   consumer_heartbeat;  // incremented by consumer; read by producer

    // Runtime config (copied by value — plain old data, capture-safe)
    RingConfig  cfg;
};

}  // namespace ring
