// Bounded proof of the wait arithmetic in Cancellation::SleepFor (#162).
//
// Source of truth: native/csrc/store/cancel.h
//   cancelled()  :65-69     SleepFor()  :78-93     NowNs()  :95-100
//
// cancel.h includes <atomic>, <chrono>, <mutex> and <condition_variable>,
// which goto-cc 6.5 cannot parse, so one iteration of SleepFor's loop is
// transcribed below with the chrono types replaced by the integers they
// hold: duration.count() is an int64_t, NowNs() a uint64_t, and
// cv_.wait_for(lock, nanoseconds(w), pred) is given w as the int64_t that
// std::chrono::nanoseconds(uint64_t) yields (modular conversion, C++20 and
// every compiler DMI builds with).
//
// What is not here: the lost-wakeup argument. Every write to cancelled_,
// deadline_ns_ and generation_ (Cancel, Reset, set_deadline) holds mutex_
// and notifies after the write; SleepFor reads cancelled() and generation_
// under mutex_ and waits on a predicate over generation_, so a change made
// after its check is seen by the predicate and one made before is seen by
// the check. That is the standard condition-variable pattern and needs no
// model; the arithmetic is what can go wrong silently.
//
// Clock reads are free but ordered: t_check (cancelled()'s read) <= t_now
// (SleepFor's `now`), both after t_start (the read that set `end`), each
// below 2^63 -- steady_clock counts from boot. -DANY_CLOCK drops that bound
// to show it is what keeps `end` from wrapping.
//
// Assertions (their labels lead the assertion text):
//   C1 a sleep that goes on to wait passes a positive wait no longer than
//      what is left of the requested duration
//   C2 with a deadline armed and not yet reached at the check, the wait
//      ends no later than the deadline
//   C3 a wait that is not positive happens only when the deadline passed
//      between the check and `now`, and then the next iteration's
//      cancelled() is true: the loop ends, it does not spin
//   C4 SleepFor never reports "slept it out" before the requested duration
//      has elapsed

#include <stdint.h>

uint64_t nondet_u64();
int64_t nondet_i64();
bool nondet_bool();

int main() {
    const int64_t count = nondet_i64();           // duration.count()
    const uint64_t t_start = nondet_u64();
    const uint64_t t_check = nondet_u64();
    const uint64_t t_now = nondet_u64();
    const uint64_t deadline = nondet_u64();       // deadline_ns_, 0 = none
    const bool cancelled_flag = nondet_bool();    // cancelled_

#ifndef ANY_CLOCK
    __CPROVER_assume(t_now < ((uint64_t)1 << 63));
#endif
    __CPROVER_assume(t_start <= t_check && t_check <= t_now);

    // const uint64_t end =
    //     NowNs() + static_cast<uint64_t>(std::max<int64_t>(0, duration.count()));
    // (goto-cc 6.5's C++ front end types `int64 ? int64 : 0` as int and
    // truncates, so std::max<int64_t>(0, count) is spelled out.)
    int64_t positive = 0;
    if (count > 0) positive = count;
    const uint64_t end = t_start + (uint64_t)positive;

    // if (cancelled()) return false;
    const bool cancelled_at_check =
        cancelled_flag || (deadline != 0 && t_check >= deadline);
    if (cancelled_at_check) return 0;

    // const uint64_t now = NowNs(); if (now >= end) return true;
    const uint64_t now = t_now;
    if (now >= end) {
        __CPROVER_assert(now - t_start >= (uint64_t)positive,
                         "C4 slept it out only once the duration elapsed");
        return 0;
    }

    // uint64_t wake = end; ... if (deadline != 0 && deadline < wake) wake = deadline;
    uint64_t wake = end;
    if (deadline != 0 && deadline < wake) wake = deadline;
    // cv_.wait_for(lock, std::chrono::nanoseconds(wake - now), ...)
    const int64_t wait = (int64_t)(wake - now);

    if (wait > 0) {
        __CPROVER_assert((uint64_t)wait <= end - now,
                         "C1 the wait is positive and within the duration");
        if (deadline != 0)
            __CPROVER_assert(now + (uint64_t)wait <= deadline || deadline >= end,
                             "C2 an armed deadline bounds the wait");
    } else {
        // wait_for with a non-positive duration returns at once; the loop
        // re-checks cancelled() with a clock read at or after `now`.
        __CPROVER_assert(deadline != 0 && deadline <= now,
                         "C3 a non-positive wait means the deadline passed");
    }
    return 0;
}
