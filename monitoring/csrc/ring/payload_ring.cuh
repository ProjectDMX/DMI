// ring/payload_ring.cuh — Payload byte-ring arithmetic.
//
// All functions are __host__ __device__ so they can be called from both host
// code (e.g., tests, init) and device kernels (e.g., the producer post-op).
//
// The payload ring is a circular byte buffer of `capacity` bytes.  Logical
// positions are tracked with two monotonically increasing 64-bit counters:
//
//   payload_head: next byte the producer will write (unwrapped)
//   payload_tail: oldest byte not yet freed by the consumer (unwrapped)
//
// Physical position = logical_position % capacity.
//
// The producer calls payload_compute_spans() to determine where to write
// before advancing payload_head.  The consumer calls payload_release() after
// the D2H transfer for that region is complete.
//
// IMPORTANT: this header contains only pure arithmetic.  It does not own any
// device memory; callers manage the buffers and counters.

#pragma once
#include <cstdint>

namespace ring {

// ---------------------------------------------------------------------------
// TwoSpan — result of a two-span payload reservation.
//
// When a reservation of `nbytes` crosses the physical end of the ring, it is
// split into two contiguous regions:
//   span 1: [off1, off1 + len1)   (runs to the end of the ring buffer)
//   span 2: [off2, off2 + len2)   (wraps to the start of the ring buffer)
//
// If len2 == 0, the reservation fits in a single span and off2 is undefined.
// ---------------------------------------------------------------------------
struct TwoSpan {
    uint64_t off1;  // byte offset of first  span in payload_buf
    uint64_t len1;  // byte length of first  span
    uint64_t off2;  // byte offset of second span in payload_buf (0 if unused)
    uint64_t len2;  // byte length of second span (0 if single span)
};

// ---------------------------------------------------------------------------
// payload_free_bytes — available space for a new reservation.
//
// Invariant: head - tail <= capacity (enforced by the producer before
// reserving; tail is only advanced by the consumer).
// ---------------------------------------------------------------------------
__host__ __device__ inline uint64_t payload_free_bytes(
    uint64_t head, uint64_t tail, uint64_t capacity)
{
    return capacity - (head - tail);
}

// ---------------------------------------------------------------------------
// payload_compute_spans — compute a two-span descriptor for `nbytes` bytes
// starting at the current `head` position.
//
// Precondition:
//   payload_free_bytes(head, tail, capacity) >= nbytes
//
// Does NOT modify head.  The caller is responsible for:
//   1. Copying data into the returned spans (D2D or cudaMemcpy).
//   2. Calling payload_advance_head() to commit the reservation.
// ---------------------------------------------------------------------------
__host__ __device__ inline TwoSpan payload_compute_spans(
    uint64_t head, uint64_t capacity, uint64_t nbytes)
{
    TwoSpan s{0, 0, 0, 0};
    uint64_t off    = head % capacity;
    uint64_t to_end = capacity - off;   // bytes from `off` to end-of-buffer

    if (nbytes <= to_end) {
        // Single contiguous span — no wrap needed.
        s.off1 = off;
        s.len1 = nbytes;
        // s.off2, s.len2 remain 0
    } else {
        // Two spans: first runs to the end, second wraps to the beginning.
        s.off1 = off;
        s.len1 = to_end;
        s.off2 = 0;
        s.len2 = nbytes - to_end;
    }
    return s;
}

// ---------------------------------------------------------------------------
// payload_advance_head — commit a reservation of `nbytes` bytes.
//
// Call AFTER writing data into the spans returned by payload_compute_spans(),
// but BEFORE publishing the corresponding TaskEntry (ready_seq must be the
// last thing written so the consumer sees consistent spans).
// ---------------------------------------------------------------------------
__host__ __device__ inline void payload_advance_head(
    uint64_t& head, uint64_t nbytes)
{
    head += nbytes;
}

// ---------------------------------------------------------------------------
// payload_release — consumer frees `nbytes` of payload space.
//
// Call AFTER the D2H transfer for the corresponding spans is complete and the
// data has been safely staged in pinned host memory.  After this call,
// payload_free_bytes() will increase by `nbytes`, which the producer can see.
// ---------------------------------------------------------------------------
__host__ __device__ inline void payload_release(
    uint64_t& tail, uint64_t nbytes)
{
    tail += nbytes;
}

// ---------------------------------------------------------------------------
// payload_chunk_bytes — payload bytes consumed by a TaskEntry.
//
// Convenience: just len1 + len2 (which equals the original nbytes argument
// passed to payload_compute_spans).
// ---------------------------------------------------------------------------
__host__ __device__ inline uint64_t payload_chunk_bytes(
    uint64_t len1, uint64_t len2)
{
    return len1 + len2;
}

}  // namespace ring
