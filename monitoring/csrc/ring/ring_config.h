// ring/ring_config.h — Runtime configuration for the GPU offload rings.
//
// All knobs live here.  Changing them at construction time does not require
// re-capturing the CUDA graph (Milestone 5 constraint).

#pragma once
#include <cstdint>

namespace ring {

// Payload allocation alignment (bytes).  Every reservation is rounded up to
// this so that D2D copies can use vectorized uint4 (16-byte) loads/stores.
// Used by: producer (D2D alignment), drain thread (staging alignment),
//          config validation (chunk_bytes % PAYLOAD_ALIGN == 0).
static constexpr uint64_t PAYLOAD_ALIGN = 16;

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
// DrainFlushConfig — controls when the batch drain thread flushes pending
// entries to host.  Force flush at 100% capacity is always active.
// ---------------------------------------------------------------------------
struct DrainFlushConfig {
    // Ratio thresholds (fraction of capacity; 0.0 = disabled)
    float task_ratio     = 0.0f;   // flush at N% task queue usage
    float payload_ratio  = 0.0f;   // flush at N% payload ring usage

    // Absolute thresholds (0 = disabled)
    uint64_t entry_threshold = 0;  // flush after N entries ready
    uint64_t byte_threshold  = 0;  // flush after N payload bytes ready

    // Time-based flush: if a complete tensor has been pending for longer
    // than this many microseconds, flush unconditionally.  0 = disabled.
    // WARNING: if this is set but drain_poll_timeout_us == 0 and
    // drain_notify_on_forward == false, the drain thread may never wake
    // to check the timeout.  If this is smaller than drain_poll_timeout_us,
    // the effective resolution is limited to drain_poll_timeout_us.
    uint64_t timeout_us = 0;

    // Force flush is always active:
    //   pending_entries >= task_cap  OR  pending_bytes >= payload_cap
    // This prevents deadlock.
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

    // Pinned staging ring size. 0 = default to payload_ring_bytes.
    // The bypass guard (tensor_total_padded_bytes > staging_capacity)
    // prevents deadlock regardless of the ratio to payload_ring_bytes.
    uint64_t pinned_staging_bytes = 0;

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

    // Batch drain flush rules.
    DrainFlushConfig drain_flush;

    // Bypass path: max bytes of bypass tensors queued between drain
    // and p2p threads. One tensor is always allowed even if it exceeds
    // this budget (prevents single-large-tensor deadlock).
    uint64_t bypass_budget_bytes = 256ULL * 1024 * 1024;  // 256 MiB

    // Clone per-request slices before submitting to host engine.
    // When true (and batch_size > 1), each slice is cloned so the full
    // tensor can be freed immediately. When false, slices are views
    // that keep the full tensor alive until consumed.
    bool clone_slices = false;

    // ClickHouse insert queue limits (host engine).
    // P2p thread blocks on submit_direct() when queue is full.
    uint64_t insert_queue_max_bytes = 512ULL * 1024 * 1024;  // 512 MiB
    uint64_t insert_queue_max_items = 4096;

    // Effective staging capacity (resolved at init time).
    // If pinned_staging_bytes == 0, defaults to payload_ring_bytes.
    uint64_t effective_staging_bytes() const {
        return pinned_staging_bytes > 0 ? pinned_staging_bytes : payload_ring_bytes;
    }
};

}  // namespace ring
