#!/usr/bin/env python3
"""Clock-skew bounds of the publisher lease: the fence margin, the default
start wait, and (since #159, 8b7991d) the lease deadline.

Line references are to main @ 5b3b632.

Discharges the derivation in docs/catalog-descriptor-key.md:365-372, which the
document files under "Verification this repository cannot run" (:680-711)
because reproducing it was believed to need two hosts with a stepped clock.

The fence predicate (docs/catalog-descriptor-key.md:352-362, emitted by
native/csrc/catalog/catalog_writer.cpp:603) admits the holder's watermark
statement only while

    expires_at_ns > now64() + publish_timeout_ns + clock_skew_ns

evaluated on the holder's host, and native/csrc/catalog/catalog_writer.cpp:538
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

Second obligation: the default START WAIT.  #150's commit 204a8d2 ("Count
clock skew in the default start wait") changed the default in
src/dmi/storage/native_capture.py:482-485 from

    lease_ttl_s + publish_timeout_s
to
    lease_ttl_s + publish_timeout_s + clock_skew_s

and the config comment at :341-352 states when that wait suffices:

    "None waits lease_ttl_s + publish_timeout_s + clock_skew_s; 0 fails at
     once. That is guaranteed to outlast a crashed predecessor only when its
     TTL is at most lease_ttl_s + publish_timeout_s: clock_skew_s cancels,
     since the wait adds it and a lagging replica sees the row live that
     much longer"

start_wait_constraints() below encodes that claim against the code that
decides it: the successor gives up at start + start_lease_wait_ns
(native/csrc/catalog/storage_service.cpp:1547-1618, a steady-clock deadline),
and a claim is refused while the predecessor's row still reads live under
LeaseCoordinator::reject_live -- head.live_until_ns > head.now_ns, with BOTH
sides of that comparison stamped by the ClickHouse replica serving the read
(native/csrc/catalog/lease_coordinator.cpp:279-301, :367).  The skew that
matters here is therefore between REPLICAS, not between DMI hosts: the
predecessor's expires_at_ns was stamped on the replica that took its INSERT,
and the successor's now64() comes from whichever replica answers head().
"""
from z3 import And, Reals, Solver, sat, unsat


def overlap_constraints(s, d_vs_S):
    p, S, e, a, b, d, t_adm, t_end, t_claim = Reals(
        "p S e a b d t_adm t_end t_claim")

    # Configuration is non-degenerate: catalog_writer.cpp:132-147 requires a
    # positive publish timeout, and the declared skew bound is unsigned
    # (clock_skew_ns, catalog_writer.h:36), so never negative.
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

    All in seconds, as ClickHouseCatalogConfig takes them:

        Tp  the PREDECESSOR's lease TTL
        Ts  the SUCCESSOR's lease_ttl_s
        p   the successor's publish_timeout_s
        S   the successor's clock_skew_s
        t_ins    true time of the predecessor's LAST lease row, then it crashes
        t_start  true time the successor's start() begins waiting

    The row is stamped expires_at_ns = (t_ins + ra) + Tp.  A poll at true time
    t reads now_ns = t + rb and is refused while expires_at_ns > now_ns, i.e.
    while t < t_ins + Tp + d.  The successor polls until t_start + W, with

        W = Ts + p + S          (native_capture.py:482-485)

    THE FAILURE STATE: the whole window is refused, so start() raises kHeld on
    a predecessor that is already dead --

        t_start + W < t_ins + Tp + d
    """
    Tp, Ts, p, S, d, t_ins, t_start = Reals("Tp Ts p S d t_ins t_start")

    # Non-degenerate knobs (native_capture.py:439-448 rejects the rest).
    s.add(Tp > 0, Ts > 0, p > 0, S >= 0)

    # The successor's own fence margin: native_capture.py:454-462 (and the
    # native writer, catalog_writer.cpp:148-160) refuses a TTL that does not
    # exceed publish_timeout_s + clock_skew_s by at least 0.1 s, so a
    # successor outside it never starts at all.
    s.add(Ts - p - S >= 0.1)

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


# --- the lease deadline (#159, 8b7991d) ---------------------------------
#
# lease_coordinator.h:30-64, lease_coordinator.cpp:40-51: a lease whose
# claim INSERT was sent at steady time t0 has
#
#     deadline = t0 + lease_ttl - clock_skew - kLeaseDeadlineMarginNs (0.1 s)
#
# and every request made under it must be answered by then (run(), :86-110;
# the storage service's LeaseScope, storage_service.cpp:113-155).  The
# writer's steady clock and the replicas' now64() run at the same rate (as
# above), so an interval on one is an interval on the other.


def deadline_constraints(s, extra, drop_skew=False):
    """Can a rival see the holder's row dead while a request the holder made
    under its lease is still within the deadline?

        t0   true time the claim INSERT was sent (sent_ns, :221-224)
        t1   true time the replica that took it stamped it: t1 >= t0
        ra   that replica's clock offset; the row's expires_at = t1 + ra + T
        rb   the offset of the replica serving a rival's head()
        t    true time of the rival's head read
        m    the margin, kLeaseDeadlineMarginNs

    The holder's request is answered by the deadline, so whatever it did on
    the server happened at a true time <= t0 + T - S - m.  THE FAILURE STATE:
    a rival head read at such a time finds the row dead --
    reject_live's live_until > now (:365-399) fails: t + rb >= t1 + ra + T.
    """
    T, S, m, t0, t1, ra, rb, t = Reals("T S m t0 t1 ra rb t")
    s.add(T > 0, S >= 0, m >= 0)
    s.add(t1 >= t0)
    s.add(ra - rb <= S, rb - ra <= S)
    # drop_skew: the deadline WITHOUT its - clock_skew term, the real skew
    # still up to S.
    s.add(t <= t0 + T - (0 if drop_skew else S) - m)   # within the deadline
    s.add(t + rb >= t1 + ra + T)        # the rival sees the row dead
    s.add(extra(T, S, m))
    return s


def client_cut_constraints(s, extra):
    """Can the client give up, at its lease deadline, on a fenced publish
    statement the server is still entitled to run -- and by how much?

    The fence admits the statement at true time ta, on a replica with offset
    rc, only while expires_at > now + p + S (catalog_writer.cpp:603-626,
    fence() at lease_coordinator.cpp:303-316); the server stops it by
    ta + p (max_execution_time, catalog_writer.cpp:536-540).  The client's
    RequestDeadline (storage_service.cpp:113-155) is the lease deadline of
    the claim that stamped the fenced row, sent at t0, stamped at t1.
    """
    T, S, p, m, t0, t1, ra, rc, ta, t_end = Reals(
        "T S p m t0 t1 ra rc ta t_end")
    s.add(T > 0, S >= 0, p > 0, m > 0)
    s.add(T - p - S >= m)               # catalog_writer.cpp:148-160
    s.add(t1 >= t0, ta >= t1)
    s.add(ra - rc <= S, rc - ra <= S)
    s.add(t1 + ra + T > ta + rc + p + S)   # the fence admits
    s.add(t_end > ta, t_end <= ta + p)      # the statement still runs
    D = t0 + T - S - m
    s.add(t_end > D)                        # past the client's deadline
    s.add(extra(t_end - D, t1 - t0, m, S))
    return s


def confirm_constraints(s, extra):
    """The bug #159 fixed in claim_with_rival (lease_coordinator.cpp:
    216-231): a claim made without a lease gave its INSERT and its read-back
    a fresh claim bound each, B = min(request timeout, T / 3) (:74-83), so
    the read-back could confirm a lease whose deadline had already passed.

        t0   the INSERT is sent; its answer comes by t0 + B
        tc   the read-back's answer (the confirmation), by that + B
    """
    T, S, m, B, t0, ti, tc = Reals("T S m B t0 ti tc")
    s.add(T > 0, S >= 0, m > 0, B > 0, B <= T / 3)
    # The skew the storage service accepts: a renewal window of at least
    # 0.2 s (lease_coordinator.h:66-93, storage_service.cpp:196-212).
    s.add(T / 2 - S - m >= 0.2)
    s.add(ti > t0, ti <= t0 + B, tc > ti, tc <= ti + B)
    s.add(tc > t0 + T - S - m)          # confirmed past its own deadline
    s.add(extra(T, S, m, B))
    return s


def check_with(name, build, extra, expect):
    s = Solver()
    build(s, extra)
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

        # --- the default start wait (204a8d2) ---------------------------
        # (a) Same knobs on both sides: the wait outlasts the dead
        #     predecessor, for any TTL, any timeout, any declared bound and
        #     any real skew within it.
        check_start_wait("start wait, predecessor TTL == successor TTL",
                         lambda Tp, Ts, p, S: Tp == Ts, unsat),
        # (b) The config comment's 30 s case (native_capture.py:348-351): a
        #     predecessor on the native 30 s default against a successor on
        #     the shipped defaults, 15 s / 5 s / 0 s -- a 20 s wait.
        check_start_wait("start wait, predecessor 30s vs 15s/5s/0s",
                         lambda Tp, Ts, p, S: And(Tp == 30, Ts == 15,
                                                  p == 5, S == 0), sat),
        # (c) The exact threshold.  The S in the wait pays for the real skew
        #     exactly, so the whole slack for a TTL mismatch is p:
        #         the wait outlasts the predecessor  <=>  Tp <= Ts + p
        #     Checked as a pair -- no failure at or below it, a failure
        #     everywhere above it.
        check_start_wait("start wait, Tp <= Ts + p  (the threshold)",
                         lambda Tp, Ts, p, S: Tp <= Ts + p, unsat),
        check_start_wait("start wait, Tp >  Ts + p  (above it)",
                         lambda Tp, Ts, p, S: Tp > Ts + p, sat),

        # --- the lease deadline (#159) ----------------------------------
        # (d) Sound with the 0.1 s margin: no rival can find the row dead
        #     while a request under the lease is within its deadline.
        check_with("deadline, m = 0.1 s", deadline_constraints,
                   lambda T, S, m: m == 0.1, unsat),
        # (e) The margin is what makes it strict: at m = 0 the deadline
        #     instant itself can meet a rival whose replica runs S ahead.
        check_with("deadline, m = 0", deadline_constraints,
                   lambda T, S, m: m == 0, sat),
        # (f) The - clock_skew term is needed: drop it from the deadline,
        #     with real replica skew above the margin, and a rival finds
        #     the row dead inside it.
        check_with("deadline without the skew term",
                   lambda s, extra: deadline_constraints(s, extra, True),
                   lambda T, S, m: And(S > m, m == 0.1), sat),
        # (g) A client may give up, at its deadline, on a fenced statement
        #     the server still runs (an unknown outcome: quarantine) ...
        check_with("client cuts a fenced statement", client_cut_constraints,
                   lambda over, delay, m, S: over > 0, sat),
        # (h) ... but only within the stamp delay, the skew bound and the
        #     margin: the fence may run on a replica up to S behind the one
        #     that stamped the row.
        check_with("client cut >= stamp delay + S + margin",
                   client_cut_constraints,
                   lambda over, delay, m, S: over >= delay + S + m, unsat),
        # (i) Before #159: an accepted skew lets a claim confirm past its
        #     deadline exactly when S > T / 3 - m ...
        check_with("old confirm, S > T/3 - m", confirm_constraints,
                   lambda T, S, m, B: And(S > T / 3 - m, m == 0.1), sat),
        check_with("old confirm, S <= T/3 - m", confirm_constraints,
                   lambda T, S, m, B: And(S <= T / 3 - m, m == 0.1), unsat),
    ]
    print()
    if all(results):
        print("ALL CHECKS AS EXPECTED")
        return 0
    print("UNEXPECTED RESULT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
