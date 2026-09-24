#!/usr/bin/env python3
"""Two-host clock-skew bound for the fenced watermark INSERT.

Discharges the derivation in docs/catalog-descriptor-key.md:365-372, which the
document files under "Verification this repository cannot run" (:680-711)
because reproducing it was believed to need two hosts with a stepped clock.

The fence predicate (docs/catalog-descriptor-key.md:352-362, emitted by
native/csrc/catalog/catalog_writer.cpp:583) admits the holder's watermark
statement only while

    expires_at_ns > now64() + publish_timeout_ns + clock_skew_ns

evaluated on the holder's host, and native/csrc/catalog/catalog_writer.cpp:519
caps that statement at max_execution_time = publish_timeout_ns on the same
host.  A successor may claim once ITS host's clock passes expires_at_ns.

Model.  One true time line.  Host A (holder) reads true time + a, host B
(successor) reads true time + b; d = b - a is the step between them.  Clock
RATES are equal -- the fence is stamped and evaluated server-side, so only the
offset matters.  All quantities are reals: no discretisation, no bound on the
magnitudes.

    p  publish_timeout_ns (= max_execution_time)
    S  clock_skew_ns, the DECLARED bound carried in the fence margin
    e  expires_at_ns, as stamped on host A's clock
    t_adm   true time at which A's statement is admitted
    t_end   true time at which A's statement stops running
    t_claim true time at which B's claim becomes possible

Overlap -- the unsafety the bound is supposed to exclude -- is B claiming
while A's admitted statement is still in flight.

Expected: UNSAT for d <= S (no overlap exists, for any p, S, e and any
schedule), SAT for d > S (the documented race is reachable).  UNSAT is a proof
over all timings, which is strictly more than a two-host experiment can show.

Second obligation: the default START WAIT.  2b74d14 ("Count clock skew in the
default start wait") changed the default in
src/dmi/storage/native_capture.py:215-219 from

    lease_ttl_s + publish_timeout_s
to
    lease_ttl_s + publish_timeout_s + clock_skew_s

and the config comment at :139-145 records a caveat by hand:

    "None waits lease_ttl_s + publish_timeout_s + clock_skew_s, enough to
     outlast a crashed predecessor that ran with the same knobs; 0 fails at
     once.  A predecessor with a longer TTL (the native default is 30 s, which
     processes predating these knobs used) can outlast it"

start_wait_constraints() below encodes that claim and its caveat against the
code that decides it: the successor gives up at start + start_lease_wait_ns
(native/csrc/catalog/storage_service.cpp:642-670, a steady-clock deadline),
and a claim is refused while the predecessor's row still reads live under
LeaseCoordinator::reject_live -- head.live_until_ns > head.now_ns, with BOTH
sides of that comparison stamped by the ClickHouse replica serving the read
(native/csrc/catalog/lease_coordinator.cpp:138-153, :225-233).  The skew that
matters here is therefore between REPLICAS, not between DMI hosts: the
predecessor's expires_at_ns was stamped on the replica that took its INSERT,
and the successor's now64() comes from whichever replica answers head().
"""
from z3 import And, Reals, Solver, sat, unsat


def overlap_constraints(s, d_vs_S):
    p, S, e, a, b, d, t_adm, t_end, t_claim = Reals(
        "p S e a b d t_adm t_end t_claim")

    # Configuration is non-degenerate; catalog_writer.cpp:148-158 requires a
    # positive publish timeout and a non-negative declared skew bound.
    s.add(p > 0, S >= 0)

    # B's host is stepped ahead of A's by d.
    s.add(b == a + d, d >= 0)

    # The fence admits A's statement only while more than p + S of lease life
    # remains ON A'S OWN CLOCK.  A's clock reads (true time + a).
    s.add(e - (t_adm + a) > p + S)

    # max_execution_time caps the admitted statement at p, measured on the same
    # clock.  Equal rates, so p of A-clock time is p of true time.
    s.add(t_end <= t_adm + p)

    # B may claim only once B's clock has passed the recorded expiry.
    s.add(t_claim + b >= e)

    # THE UNSAFE STATE: B claims while A's statement is still in flight.
    s.add(t_claim < t_end)

    s.add(d_vs_S(d, S))
    return d, S


def start_wait_constraints(s, extra):
    """Can a crashed predecessor outlast the successor's default start wait?

    One true time line.  The replica that stamped the predecessor's lease row
    reads true time + ra; the replica answering the successor's head() reads
    true time + rb.  d = ra - rb is the step between them; |d| <= S, the
    DECLARED bound, is the hypothesis under test.  Rates are equal -- both
    stamps come from now64(9) server-side -- so only the offset matters.

        Tp  the PREDECESSOR's lease_ttl_ns
        Ts  the SUCCESSOR's lease_ttl_ns
        p   the successor's publish_timeout_ns
        S   the successor's clock_skew_ns
        t_ins    true time of the predecessor's LAST lease row, then it crashes
        t_start  true time the successor's start() begins waiting

    The row is stamped expires_at_ns = (t_ins + ra) + Tp.  A poll at true time
    t reads now_ns = t + rb and is refused while expires_at_ns > now_ns, i.e.
    while t < t_ins + Tp + d.  The successor polls until t_start + W, with

        W = Ts + p + S          (native_capture.py:217-219)

    THE FAILURE STATE: the whole window is refused, so start() raises kHeld on
    a predecessor that is already dead --

        t_start + W < t_ins + Tp + d
    """
    Tp, Ts, p, S, d, t_ins, t_start = Reals("Tp Ts p S d t_ins t_start")

    # Non-degenerate knobs (native_capture.py:147-166 rejects the rest).
    s.add(Tp > 0, Ts > 0, p > 0, S >= 0)

    # Real replica skew within the declared bound, either direction.
    s.add(d <= S, d >= -S)

    # A restart: the successor starts no earlier than the predecessor's last
    # lease row.  t_start == t_ins is the worst case (renew, crash, restart).
    s.add(t_start >= t_ins)

    # The default start wait.
    W = Ts + p + S

    # THE FAILURE STATE: every poll in the window is refused.
    s.add(t_start + W < t_ins + Tp + d)

    s.add(extra(Tp, Ts, p, S))
    return Tp, Ts, p, S


def check_start_wait(name, extra, expect):
    s = Solver()
    start_wait_constraints(s, extra)
    got = s.check()
    ok = got == expect
    print(f"{name:<44} {str(got):<7} expected {str(expect):<7} "
          f"{'OK' if ok else 'MISMATCH'}")
    if got == sat:
        print(f"    witness: {s.model()}")
    return ok


def check(name, d_vs_S, expect):
    s = Solver()
    overlap_constraints(s, d_vs_S)
    got = s.check()
    ok = got == expect
    print(f"{name:<44} {str(got):<7} expected {str(expect):<7} "
          f"{'OK' if ok else 'MISMATCH'}")
    if got == sat:
        print(f"    witness: {s.model()}")
    return ok


def main():
    results = [
        # 1. Real skew within the declared bound: no overlap, for any timing.
        check("real skew <= clock_skew_ns",
              lambda d, S: d <= S, unsat),
        # 2. Real skew above the declared bound: the documented race appears.
        check("real skew >  clock_skew_ns",
              lambda d, S: d > S, sat),
        # 3. What the + clock_skew_ns term in the margin actually buys: drop it
        #    (keep the cap) and ANY positive step between the hosts overlaps.
        check("margin without the skew term, any d > 0",
              lambda d, S: And(S == 0, d > 0), sat),

        # --- the default start wait (2b74d14) ---------------------------
        # (a) Same knobs on both sides: the wait outlasts the dead
        #     predecessor, for any TTL, any timeout, any declared bound and
        #     any real skew within it.
        check_start_wait("start wait, predecessor TTL == successor TTL",
                         lambda Tp, Ts, p, S: Tp == Ts, unsat),
        # (b) The hand-written caveat: a predecessor on the native 30 s
        #     default against a successor on 10 s / 10 s / 2 s.
        check_start_wait("start wait, predecessor 30s vs successor 10s",
                         lambda Tp, Ts, p, S: And(Tp == 30, Ts == 10,
                                                  p == 10, S == 2), sat),
        # (c) The exact threshold.  The S in the wait pays for the real skew
        #     exactly, so the whole slack for a TTL mismatch is p:
        #         the wait outlasts the predecessor  <=>  Tp <= Ts + p
        #     Checked as a pair -- no failure at or below it, a failure
        #     everywhere above it.
        check_start_wait("start wait, Tp <= Ts + p  (the threshold)",
                         lambda Tp, Ts, p, S: Tp <= Ts + p, unsat),
        check_start_wait("start wait, Tp >  Ts + p  (above it)",
                         lambda Tp, Ts, p, S: Tp > Ts + p, sat),
    ]
    print()
    if all(results):
        print("ALL CHECKS AS EXPECTED")
        return 0
    print("UNEXPECTED RESULT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
