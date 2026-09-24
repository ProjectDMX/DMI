// Bounded proof of the payload ring's two-span arithmetic.
//
// Source of truth: native/csrc/ring/payload_ring.cuh:44-86
//   payload_free_bytes(head, tail, capacity)      :49-53
//   payload_compute_spans(head, capacity, nbytes) :66-86
//
// The header states one precondition at :59-60:
//     payload_free_bytes(head, tail, capacity) >= nbytes
// and one producer-side invariant at :46-47:
//     head - tail <= capacity
//
// Build it twice -- with the precondition and with -DDROP_PRECONDITION -- to
// see whether the precondition is load-bearing or merely defensive.  See
// specs/README.md for the exact goto-cc / cbmc invocations.
//
// The file is C++ (.cpp) because payload_ring.cuh uses a namespace; the two
// CUDA qualifiers are defined away so the same header compiles for the host.

#define __host__
#define __device__
#include "../../native/csrc/ring/payload_ring.cuh"
#include <cassert>

int main() {
    uint64_t cap, head, tail, n;

    // Capacity is bounded only to keep the proof bounded; every capacity in
    // 1..64 is covered, wrap and non-wrap alike.
    __CPROVER_assume(cap >= 1 && cap <= 64);

    // The producer-side ring invariant (payload_ring.cuh:46-47).
    __CPROVER_assume(head >= tail);
    __CPROVER_assume(head - tail <= cap);

    // head is otherwise unconstrained up to 2^40, so `head % capacity` takes
    // every residue at an arbitrary number of wraps, not just the first lap.
    __CPROVER_assume(head <= (uint64_t)1 << 40);

#ifndef DROP_PRECONDITION
    // The documented precondition (payload_ring.cuh:59-60).
    __CPROVER_assume(n <= ring::payload_free_bytes(head, tail, cap));
#endif

    ring::TwoSpan s = ring::payload_compute_spans(head, cap, n);

    assert(s.len1 + s.len2 == n);                     // P1 spans cover the request
    assert(s.off1 + s.len1 <= cap);                   // P2 span 1 stays in the buffer
    assert(s.off2 + s.len2 <= cap);                   // P3 span 2 stays in the buffer
    assert(s.len2 == 0 || s.off2 + s.len2 <= s.off1); // P4 the two spans are disjoint

    // P5 no byte the spans cover lies in the unconsumed region [tail, head).
    // Pick any byte i of the reservation and any unconsumed position j; the
    // buffer offset the spans give byte i is not the one j occupies. j's
    // offset is found by stepping back head - j (1..cap, by the ring
    // invariant) from head's own offset, taken as s.off1: both branches of
    // payload_compute_spans set off1 = head % capacity on their first line.
    // Recomputing head % cap here instead would ask the solver to prove two
    // 64-bit dividers equal, which does not finish; P5 therefore checks the
    // lengths and the wrap point, and trusts that one assignment.
    uint64_t i, j;
    __CPROVER_assume(i < n);
    __CPROVER_assume(tail <= j && j < head);
    const uint64_t back = head - j;                   // 1..cap
    const uint64_t j_off = back <= s.off1 ? s.off1 - back
                                          : s.off1 + cap - back;
    const uint64_t written = i < s.len1 ? s.off1 + i : s.off2 + (i - s.len1);
    assert(written != j_off);                         // P5 spans avoid unconsumed bytes

    return 0;
}
