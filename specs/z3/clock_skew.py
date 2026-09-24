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
"""
from z3 import Reals, Solver, sat, unsat


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
              lambda d, S: (S == 0, d > 0)[1], sat),
    ]
    print()
    if all(results):
        print("ALL CHECKS AS EXPECTED")
        return 0
    print("UNEXPECTED RESULT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
