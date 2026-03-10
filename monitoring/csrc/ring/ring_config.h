// ring/ring_config.h — Runtime configuration for the GPU offload rings.
//
// All knobs live here.  Changing them at construction time does not require
// re-capturing the CUDA graph (Milestone 5 constraint).

#pragma once
#include <cstdint>

namespace ring {

// ---------------------------------------------------------------------------
// WaitPolicy — how the producer handles ring-full backpressure.
// ---------------------------------------------------------------------------
enum class WaitPolicy : uint32_t {
    // Spin indefinitely until both task slots and payload bytes are available.
    // Safe in steady state when consumer is alive; risks deadlock if consumer
    // dies (use a watchdog at a higher level in that case).
    INFINITE = 0,

    // If consumer_heartbeat / last_released_seq shows no progress for
    // no_progress_timeout_cycles, abandon the current logical task.
    // Producer does NOT advance head pointers (rollback-free).  If
    // drop_reporting == DROP_TASK, a small DROP marker entry is published.
    TIMEOUT_DROP = 1,
};

// ---------------------------------------------------------------------------
// DropReporting — what to do when a logical task is dropped.
// ---------------------------------------------------------------------------
enum class DropReporting : uint32_t {
    // Increment internal counter only; do not publish a task entry.
    COUNTER_ONLY = 0,

    // Publish a metadata-only TaskEntry with TASK_FLAG_IS_DROP set so the
    // consumer-side CPU pipeline can account for the missing tensor.
    DROP_TASK = 1,
};

// ---------------------------------------------------------------------------
// RingConfig — all tunable parameters.
// ---------------------------------------------------------------------------
struct RingConfig {
    // Task/control ring: number of fixed-size TaskEntry slots (power of 2
    // recommended for efficient modular index arithmetic).
    uint64_t task_ring_entries = 1024;

    // Payload byte ring: total size of the circular byte buffer.
    uint64_t payload_ring_bytes = 256ULL * 1024 * 1024;  // 256 MiB

    // Maximum bytes per chunk.  One logical tensor is split into chunks of at
    // most this many bytes so the producer never needs to reserve more than
    // payload_ring_bytes at once (forward-progress guarantee).
    // Default: payload_ring_bytes / 4.
    uint64_t chunk_bytes = 64ULL * 1024 * 1024;          // 64 MiB

    // Backpressure policy (see WaitPolicy).
    WaitPolicy wait_policy = WaitPolicy::INFINITE;

    // Duration without consumer progress before a TIMEOUT_DROP is triggered.
    // Units: GPU clock cycles (compare against clock64()).
    // Default: ~400 ms at 2.5 GHz.
    uint64_t no_progress_timeout_cycles = 1'000'000'000ULL;

    // Pinned host ring buffer size (shared across all in-flight chunks).
    uint64_t pinned_pool_bytes  = 256ULL * 1024 * 1024;  // 256 MiB

    // What to emit when a logical task is dropped (see DropReporting).
    DropReporting drop_reporting = DropReporting::DROP_TASK;

    // Drain thread poll timeout in microseconds.  0 = no timeout (infinite wait,
    // drain only when explicitly notified or at stop()).
    // WARNING: if both drain_poll_timeout_us == 0 and drain_notify_on_forward
    // == false, the drain thread will never wake during generation.  The ring
    // must be large enough to hold all data until stop(), or the producer will
    // deadlock on backpressure.
    uint64_t drain_poll_timeout_us = 0;

    // Whether to call notify_drain() before each forward pass from Python.
    bool drain_notify_on_forward = true;
};

}  // namespace ring
