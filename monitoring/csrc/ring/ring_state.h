// ring/ring_state.h -- Capture-safe descriptor passed by value into the producer kernel.
//
// All fields are raw device pointers and integral constants (POD).  The struct
// is safe to copy into kernel arguments with no vtable, no destructor.
//
// Counter conventions:
//   task_head, payload_head : written by producer (GPU), managed memory preferred on GPU
//
// Tail pointers are CPU-only shadows in the drain thread -- not in this struct.
// The producer kernel never reads tail pointers.  Space is guaranteed by the
// pre-forward capacity check before kernel launch.

#pragma once
#include "task_entry.h"
#include <cstdint>

namespace ring {

struct RingState {
    // Task/control ring
    TaskEntry*  task_entries;   // managed memory: task_cap slots
    uint64_t    task_cap;       // number of task slots

    uint64_t*   task_head;      // monotonically increasing; producer writes

    // Payload byte ring
    uint8_t*    payload_buf;    // device buffer: payload_cap bytes
    uint64_t    payload_cap;    // capacity in bytes

    uint64_t*   payload_head;   // monotonically increasing; producer writes

    // Monotonic counter of actual bytes written by all producer kernels
    // (sum of src_bytes across every kernel invocation, ever).  Producer
    // atomicAdd's inside the last-block-arrives section; CPU reads the
    // value via delta vs a per-engine last_counter_read snapshot.  Used by
    // prepare_step to reclaim ring space when reservations over-estimate
    // actual writes (EP upper-bound reservations etc.).  Counter is never
    // reset -- delta tracking on the CPU side handles wrap-around.
    uint64_t*   actual_bytes_counter;
};

}  // namespace ring
