// Unbounded proof of the legacy ring's free-room arithmetic (#160), and of
// the record ring's admission check that #160 did not touch.
//
// Source of truth: native/csrc/ring/ring_engine_py.cu
//   legacy_ring_room(cap, head, tail)                    :141-144 (copied)
//   available_capacity() / available_task_slots()        :937-949 (call it)
//   reserve_record's admission check                      :715-720 (copied)
// and ring::payload_free_bytes, through the real header
// (payload_ring.cuh:49-53), and ring::task_free_slots (task_ring.cuh:45-50),
// copied: task_ring.cuh includes publication_word.h, whose
// `uint64_t{1} << 63` goto-cc 6.5 cannot parse.
//
// legacy_ring_room lives in an anonymous namespace of a .cu file that pulls
// in torch, so it cannot be #included; its body is copied verbatim between
// the BEGIN/END markers below, and specs/cbmc/sync_check.sh fails when the
// copy no longer occurs verbatim in the source.
//
// Every variable is a free uint64_t: no bound on cap, head or tail, so the
// proof covers every value the CPU accounting can take, including an
// accounting pushed past the ring (head - tail > cap) and a tail read later
// than the head (head - tail wrapping below zero).
//
// Build with -DUNSATURATED to check the pre-#160 formula instead,
// cap - (head - tail), and see which assertions the saturation carries.
//
// Assertions (their labels lead the assertion text):
//   R1 room <= cap: never more room than the ring has
//   R2 inside the producer invariant (head - tail <= cap) the room is
//      exactly the headers' payload_free_bytes and task_free_slots
//   R3 at or past capacity the room is 0
//   R4 a tail read later (tail <= tail2 <= head) never shows less room:
//      the non-atomic head-then-tail read in available_capacity() errs on
//      the side the comment at ring_engine_py.cu:925-934 claims
//   R5 from an accounting inside the ring, a reservation the room admits
//      keeps it inside the ring
//   R6 the record ring's check (:715-720), which #160 left unsaturated:
//      a nonzero reservation it admits leaves the accounting within the
//      ring, whatever the accounting was before.
//      Expected FAILURE: once payload_used > payload_cap the subtraction
//      wraps and every reservation is admitted. See the findings report
//      for whether the record path can reach payload_used > payload_cap.

#define __host__
#define __device__
#include "../../native/csrc/ring/payload_ring.cuh"

namespace ring {
// BEGIN COPY task_ring.cuh task_free_slots
inline uint64_t task_free_slots(
    uint64_t head, uint64_t tail, uint64_t capacity)
{
    return capacity - (head - tail);
}
// END COPY
}  // namespace ring

#ifndef UNSATURATED
// BEGIN COPY ring_engine_py.cu legacy_ring_room
uint64_t legacy_ring_room(uint64_t cap, uint64_t head, uint64_t tail) {
    const uint64_t outstanding = head - tail;
    return outstanding >= cap ? 0 : cap - outstanding;
}
// END COPY
#else
// The pre-#160 body of available_capacity() / available_task_slots().
uint64_t legacy_ring_room(uint64_t cap, uint64_t head, uint64_t tail) {
    return cap - (head - tail);
}
#endif

uint64_t nondet_u64();

int main() {
    const uint64_t cap = nondet_u64();
    const uint64_t head = nondet_u64();
    const uint64_t tail = nondet_u64();
    const uint64_t room = legacy_ring_room(cap, head, tail);

    __CPROVER_assert(room <= cap, "R1 room never exceeds the ring");

    if (head - tail <= cap) {
        __CPROVER_assert(room == ring::payload_free_bytes(head, tail, cap) &&
                             room == ring::task_free_slots(head, tail, cap),
                         "R2 room matches the headers inside the invariant");
    }

    if (head - tail >= cap)
        __CPROVER_assert(room == 0, "R3 room is 0 at or past capacity");

    // R4: the drain only moves the tail forward, never past what the head
    // has reserved, so a later read sees tail2 in [tail, head].
    const uint64_t tail2 = nondet_u64();
    if (tail <= tail2 && tail2 <= head && tail <= head)
        __CPROVER_assert(legacy_ring_room(cap, head, tail2) >= room,
                         "R4 a later tail never shows less room");

    // R5: n admitted by the room, from an accounting inside the ring; the
    // head does not wrap 2^64. (Past the ring the room is 0, so only n = 0
    // is admitted and nothing grows; R3 covers that side.)
    const uint64_t n = nondet_u64();
    if (n <= room && head - tail <= cap && head + n >= head && head >= tail)
        __CPROVER_assert((head + n) - tail <= cap,
                         "R5 an admitted reservation stays inside the ring");

    // R6: reserve_record's admission, ring_engine_py.cu:715-720, verbatim
    // but for the drain accessors, which are the free variables here.
    {
        const uint64_t payload_cap = cap;
        const uint64_t reservation_bytes = n;
        const uint64_t payload_used = head - tail;
        if (reservation_bytes <= payload_cap - payload_used &&
            n > 0 && head + n >= head && head >= tail)
            __CPROVER_assert(payload_used + reservation_bytes <= payload_cap,
                             "R6 record check: an admitted reservation stays inside the ring");
    }
    return 0;
}
