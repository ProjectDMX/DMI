// ring/ring_config.h -- Runtime configuration for the GPU offload rings.
//
// All knobs live here.  Changing them at construction time does not require
// re-capturing the CUDA graph (Milestone 5 constraint).

#pragma once
#include <cstdint>

namespace ring {

// Payload allocation alignment (bytes).  Every reservation is rounded up to
// this so that D2D copies can use vectorized uint4 (16-byte) loads/stores.
// Used by: producer (D2D alignment), drain thread (staging alignment).
static constexpr uint64_t PAYLOAD_ALIGN = 16;

// Align x up to a multiple of a (a must be a power of 2).
#ifdef __CUDACC__
__host__ __device__
#endif
inline uint64_t align_up(uint64_t x, uint64_t a) {
    return (x + a - 1) & ~(a - 1);
}

// ---------------------------------------------------------------------------
// DrainFlushConfig -- controls when the batch drain thread flushes pending
// entries to host.
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
    uint64_t timeout_us = 100ULL * 1000;
};

// ---------------------------------------------------------------------------
// RingConfig -- all tunable parameters.
// ---------------------------------------------------------------------------
struct RingConfig {
    // Task/control ring: number of fixed-size TaskEntry slots (power of 2
    // recommended for efficient modular index arithmetic).
    uint64_t task_ring_entries = 1024;

    // Payload byte ring: total size of the circular byte buffer.
    uint64_t payload_ring_bytes = 256ULL * 1024 * 1024;  // 256 MiB

    // Pinned staging ring size. 0 = default to payload_ring_bytes.
    uint64_t pinned_staging_bytes = 0;

    // Drain thread poll timeout in microseconds.  Must be > 0.
    uint64_t drain_poll_timeout_us = 100;

    // Batch drain flush rules.
    DrainFlushConfig drain_flush;

    // Clone per-request slices before submitting to host engine.
    // When true (and batch_size > 1), each slice is cloned so the full
    // tensor can be freed immediately. When false, slices are views
    // that keep the full tensor alive until consumed.
    bool clone_slices = false;

    // ClickHouse insert queue limits (host engine).
    // P2p thread blocks on submit_direct() when queue is full.
    uint64_t insert_queue_max_bytes = 4096ULL * 1024 * 1024;  // 4 GiB
    uint64_t insert_queue_max_items = 65536;

    // Effective staging capacity (resolved at init time).
    // If pinned_staging_bytes == 0, defaults to payload_ring_bytes.
    uint64_t effective_staging_bytes() const {
        return pinned_staging_bytes > 0 ? pinned_staging_bytes : payload_ring_bytes;
    }
};

}  // namespace ring
