--------------------------- MODULE PublisherLease ---------------------------
(***************************************************************************)
(* ClickHouse publisher lease + fenced snapshot publish.                   *)
(*                                                                         *)
(* Code: src/dmi/storage/capture/clickhouse_lease.py (coordinator) and     *)
(*       src/dmi/storage/capture/clickhouse_catalog.py (publish, quarantine)*)
(* See README.md for the action -> file:line map and the abstractions.     *)
(*                                                                         *)
(* Time is a discrete true clock `now`.  Every timestamp the protocol uses *)
(* is a server now64(); each server statement reads now + d for some d in  *)
(* Offsets, chosen per statement (any host, any moment).  Writers' own     *)
(* wall clocks never enter a decision in the code, so they are not state;  *)
(* the only client-side clock is the monotonic quarantine timer, modelled  *)
(* as a true-time duration.                                                *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS
    Writers,     \* publisher processes (model values)
    Ids,         \* lease_id pool (integers; integer order = UUID collation order)
    TTL,         \* lease_ttl_ns
    P,           \* publish_timeout_ns == max_execution_time
    S,           \* clock_skew_ns (the configured bound)
    MaxTime,     \* true-time horizon
    MaxTerm,     \* term bound (state-space bound only)
    MaxCrashes,  \* crash-restarts allowed per behaviour
    Unknowns,    \* TRUE: outcome-unknown failures (InsertUnknown, ClientError) enabled
    Mutation,    \* "none" = faithful; otherwise the ingredient removed
    ClockModel   \* "pairwise": host clocks differ pairwise by <= S (what
                 \*   clickhouse_catalog.py:72-90 says clock_skew_ns means);
                 \* "offset": each host within +-S of true time (pairwise 2S)

Mutations == {"none", "no_skew", "no_readback", "no_quarantine",
              "release_on_unknown", "old_release", "linger",
              "cpp_keep_lease"}  \* not a mutation: the C++ port's reject_if_gone

ASSUME Mutation \in Mutations
ASSUME ClockModel \in {"pairwise", "offset"}
\* ClickHouseCatalogConfig.__post_init__, clickhouse_catalog.py:134-146:
\* lease_ttl_ns - (publish_timeout_ns + clock_skew_ns) >= MINIMUM_FENCE_MARGIN_NS > 0
ASSUME TTL > P + S /\ P > 0 /\ S >= 0

Offsets  == IF ClockModel = "pairwise" THEN 0..S ELSE -S..S
SkewTerm == IF Mutation = "no_skew" THEN 0 ELSE S

Max(X) == CHOOSE x \in X : \A y \in X : y <= x
Min(X) == CHOOSE x \in X : \A y \in X : x <= y

VARIABLES
    now,        \* true time
    rows,       \* the append-only lease table: set of [t, id, e] (e = expires_at_ns)
    pending,    \* claim INSERTs whose outcome was unknown to the client and
                \*   that may still land later: set of [t, id]
    pc,         \* writer control state
    lid,        \* lease_id the writer holds / is claiming (0 = none)
    lterm,      \* term of the lease the writer holds (0 = none)
    tgt,        \* term being claimed (claim) or tombstoned (old_release)
    mode,       \* "acq" (acquire, new id) | "renew" (same id)
    inflight,   \* fence-admitted statements whose effect may still land:
                \*   set of [w, id, term, dl]  (dl = admission + P)
    cur,        \* the in-flight record the writer is waiting on
    quarUntil,  \* quarantine deadline (true time)
    used,       \* lease ids ever minted (uuid4 never repeats)
    granted,    \* history: every successful claim read-back [t, id, w]
    released,   \* history: ids a release tombstone was written for
    tombs,      \* history: [t, id] of every release tombstone
    crashes,
    admittedReleased \* history: fence admitted a statement for a released id

vars == <<now, rows, pending, pc, lid, lterm, tgt, mode, inflight, cur,
          quarUntil, used, granted, released, tombs, crashes, admittedReleased>>

NoRec == [w |-> "none", id |-> 0, term |-> 0, dl |-> -1]

-----------------------------------------------------------------------------
(* The head as head() / fence() compute it (clickhouse_lease.py:171-190,   *)
(* 213-225): rows at max(term), grouped by lease_id, expiry = MIN.         *)
HeadTerm   == IF rows = {} THEN 0 ELSE Max({r.t : r \in rows})
HeadIds    == {r.id : r \in {x \in rows : x.t = HeadTerm}}
ExpAt(i, T) == Min({r.e : r \in {x \in rows : x.t = T /\ x.id = i}})
TopId      == IF rows = {} THEN 0 ELSE Max(HeadIds)       \* ORDER BY lease_id DESC
LiveUntil  == IF rows = {} THEN 0 ELSE Max({ExpAt(i, HeadTerm) : i \in HeadIds})
Claimants  == Cardinality(HeadIds)

\* _reject_live (clickhouse_lease.py:258-262) returns without raising.
\* Empty table: head() returns all zeros (:179-180) and 0 <= now_ns.
ClaimMayProceed(i, nowns) ==
    \/ rows = {}
    \/ LiveUntil <= nowns
    \/ Claimants = 1 /\ TopId = i

\* fence() (clickhouse_lease.py:213-225), evaluated server-side at admission
\* with now64() = now + d on the host running the statement.
FenceOK(i, d) ==
    /\ rows # {}
    /\ TopId = i
    /\ ExpAt(i, HeadTerm) > now + d + P + SkewTerm

-----------------------------------------------------------------------------
Init ==
    /\ now = 0
    /\ rows = {}
    /\ pending = {}
    /\ pc = [w \in Writers |-> "idle"]
    /\ lid = [w \in Writers |-> 0]
    /\ lterm = [w \in Writers |-> 0]
    /\ tgt = [w \in Writers |-> 0]
    /\ mode = [w \in Writers |-> "acq"]
    /\ inflight = {}
    /\ cur = [w \in Writers |-> NoRec]
    /\ quarUntil = [w \in Writers |-> 0]
    /\ used = {}
    /\ granted = {}
    /\ released = {}
    /\ tombs = {}
    /\ crashes = 0
    /\ admittedReleased = FALSE

DropLease(w) == /\ lid' = [lid EXCEPT ![w] = 0]
                /\ lterm' = [lterm EXCEPT ![w] = 0]

\* Local bookkeeping of a finished claim is cleared (state-space hygiene only).
ResetClaim(w) == /\ tgt' = [tgt EXCEPT ![w] = 0]
                 /\ mode' = [mode EXCEPT ![w] = "acq"]

Quarantine(w) ==   \* _quarantine_locked, clickhouse_catalog.py:294-306
    /\ pc' = [pc EXCEPT ![w] = "quar"]
    /\ quarUntil' = [quarUntil EXCEPT ![w] = now + TTL]
    /\ DropLease(w)

(* acquire() with no held lease -> claim(): the head() read + _reject_live *)
(* (clickhouse_lease.py:58-67, 137-140).  One server statement.           *)
AcquireHead(w) ==
    /\ pc[w] = "idle"
    /\ \E i \in Ids \ used, d \in Offsets :
          /\ ClaimMayProceed(i, now + d)
          /\ HeadTerm + 1 <= MaxTerm
          /\ pc' = [pc EXCEPT ![w] = "insert"]
          /\ lid' = [lid EXCEPT ![w] = i]
          /\ tgt' = [tgt EXCEPT ![w] = HeadTerm + 1]
          /\ mode' = [mode EXCEPT ![w] = "acq"]
          /\ used' = used \cup {i}
    /\ UNCHANGED <<now, rows, pending, lterm, inflight, cur, quarUntil,
                   granted, released, tombs, crashes, admittedReleased>>

(* renew() -> claim() with the held lease_id (clickhouse_lease.py:69-78). *)
(* publish_snapshot renews before every fenced statement                  *)
(* (clickhouse_catalog.py:489, 541).                                      *)
RenewHead(w) ==
    /\ pc[w] \in {"held", "ready"}
    /\ \E d \in Offsets :
          IF ClaimMayProceed(lid[w], now + d)
          THEN /\ HeadTerm + 1 <= MaxTerm
               /\ pc' = [pc EXCEPT ![w] = "insert"]
               /\ tgt' = [tgt EXCEPT ![w] = HeadTerm + 1]
               /\ mode' = [mode EXCEPT ![w] = "renew"]
               /\ UNCHANGED <<lid, lterm>>
          ELSE \* PublisherLeaseHeldError; _lease = None (:263)
               /\ pc' = [pc EXCEPT ![w] = "idle"]
               /\ DropLease(w)
               /\ UNCHANGED <<tgt, mode>>
    /\ UNCHANGED <<now, rows, pending, inflight, cur, quarUntil, used,
                   granted, released, tombs, crashes, admittedReleased>>

(* _insert (clickhouse_lease.py:282-300): expires = server now64 + TTL.   *)
Insert(w) ==
    /\ pc[w] = "insert"
    /\ \E d \in Offsets :
          rows' = rows \cup {[t |-> tgt[w], id |-> lid[w], e |-> now + d + TTL]}
    /\ pc' = [pc EXCEPT ![w] = "readback"]
    /\ UNCHANGED <<now, pending, lid, lterm, tgt, mode, inflight, cur,
                   quarUntil, used, granted, released, tombs, crashes,
                   admittedReleased>>

(* The claim INSERT raises something that is not a CaptureStorageError   *)
(* (transport reset, quorum timeout): the row may land later.  The catalog *)
(* quarantines (clickhouse_catalog.py:684-701).                            *)
InsertUnknown(w) ==
    /\ Unknowns
    /\ pc[w] = "insert"
    /\ pending' = pending \cup {[t |-> tgt[w], id |-> lid[w]]}
    /\ IF Mutation = "no_quarantine"
       THEN IF mode[w] = "acq"
            THEN /\ pc' = [pc EXCEPT ![w] = "idle"] /\ DropLease(w)
                 /\ UNCHANGED quarUntil
            ELSE \* claim() raised before touching self._lease: lease kept
                 /\ pc' = [pc EXCEPT ![w] = "held"]
                 /\ UNCHANGED <<lid, lterm, quarUntil>>
       ELSE Quarantine(w)
    /\ ResetClaim(w)
    /\ UNCHANGED <<now, rows, inflight, cur, used, granted,
                   released, tombs, crashes, admittedReleased>>

(* A pending INSERT lands (server-side), stamped when it executes.        *)
LandPending ==
    /\ \E p \in pending, d \in Offsets :
          /\ rows' = rows \cup {[t |-> p.t, id |-> p.id, e |-> now + d + TTL]}
          /\ pending' = pending \ {p}
    /\ UNCHANGED <<now, pc, lid, lterm, tgt, mode, inflight, cur, quarUntil,
                   used, granted, released, tombs, crashes, admittedReleased>>

(* Sole-claimant read-back (clickhouse_lease.py:142-159).                 *)
Readback(w) ==
    /\ pc[w] = "readback"
    /\ LET ids == {r.id : r \in {x \in rows : x.t = tgt[w]}} IN
       IF ids = {lid[w]} \/ Mutation = "no_readback"
       THEN /\ lterm' = [lterm EXCEPT ![w] = tgt[w]]
            /\ granted' = granted \cup {[t |-> tgt[w], id |-> lid[w]]}
            \* after a renewal the writer may go straight to a fenced
            \* statement; after a bare acquire it renews first.
            /\ pc' = [pc EXCEPT ![w] = IF mode[w] = "renew" THEN "ready" ELSE "held"]
            /\ UNCHANGED lid
       ELSE /\ pc' = [pc EXCEPT ![w] = "idle"]
            /\ DropLease(w)
            /\ UNCHANGED granted
    /\ ResetClaim(w)
    /\ UNCHANGED <<now, rows, pending, inflight, cur, quarUntil,
                   used, released, tombs, crashes, admittedReleased>>

(* A fenced statement (manifest chunk or watermark INSERT ... WHERE fence, *)
(* clickhouse_catalog.py:521-535, 551-576).  Fence + write are ONE server  *)
(* statement: the fence is evaluated at admission; the write's effect may  *)
(* land any time up to admission + P (max_execution_time, :498-499,        *)
(* clickhouse_lease.py:235-239).                                           *)
Admit(w) ==
    /\ pc[w] = "ready"
    /\ \E d \in Offsets :
          IF FenceOK(lid[w], d)
          THEN LET rec == [w |-> w, id |-> lid[w], term |-> HeadTerm, dl |-> now + P]
               IN /\ inflight' = inflight \cup {rec}
                  /\ cur' = [cur EXCEPT ![w] = rec]
                  /\ pc' = [pc EXCEPT ![w] = "pub"]
                  /\ admittedReleased' = (admittedReleased \/ lid[w] \in released)
                  /\ UNCHANGED <<lid, lterm>>
          ELSE \* refused; reject_if_gone (clickhouse_lease.py:241-256) drops
               \* the local lease unless it still heads the table.  The C++
               \* port's reject_if_gone (lease_coordinator.cpp:187-201) is
               \* const and never drops it: variant "cpp_keep_lease".
               /\ IF TopId = lid[w] \/ Mutation = "cpp_keep_lease"
                  THEN pc' = [pc EXCEPT ![w] = "held"] /\ UNCHANGED <<lid, lterm>>
                  ELSE pc' = [pc EXCEPT ![w] = "idle"] /\ DropLease(w)
               /\ UNCHANGED <<inflight, cur, admittedReleased>>
    /\ UNCHANGED <<now, rows, pending, tgt, mode, quarUntil, used, granted,
                   released, tombs, crashes>>

(* The server finishes (or aborts) an admitted statement.                  *)
Land ==
    /\ \E f \in inflight : inflight' = inflight \ {f}
    /\ UNCHANGED <<now, rows, pending, pc, lid, lterm, tgt, mode, cur,
                   quarUntil, used, granted, released, tombs, crashes,
                   admittedReleased>>

(* The client sees the statement's result: a known outcome.               *)
Done(w) ==
    /\ pc[w] = "pub"
    /\ cur[w] \notin inflight
    /\ pc' = [pc EXCEPT ![w] = "held"]
    /\ cur' = [cur EXCEPT ![w] = NoRec]
    /\ UNCHANGED <<now, rows, pending, lid, lterm, tgt, mode, inflight,
                   quarUntil, used, granted, released, tombs, crashes,
                   admittedReleased>>

(* Outcome-unknown: the driver raises (Code 159, reset, interrupt) while  *)
(* the statement may still be running (clickhouse_catalog.py:430-470).    *)
ClientError(w) ==
    /\ Unknowns
    /\ pc[w] = "pub"
    /\ cur' = [cur EXCEPT ![w] = NoRec]
    /\ CASE Mutation = "no_quarantine" ->
              /\ pc' = [pc EXCEPT ![w] = "held"]
              /\ UNCHANGED <<lid, lterm, quarUntil, rows, released, tombs>>
         [] Mutation = "release_on_unknown" ->
              \E d \in Offsets :
                 /\ rows' = rows \cup {[t |-> lterm[w], id |-> lid[w], e |-> now + d]}
                 /\ tombs' = tombs \cup {[t |-> lterm[w], id |-> lid[w]]}
                 /\ released' = released \cup {lid[w]}
                 /\ pc' = [pc EXCEPT ![w] = "idle"]
                 /\ DropLease(w)
                 /\ UNCHANGED quarUntil
         [] OTHER ->
              /\ Quarantine(w)
              /\ UNCHANGED <<rows, released, tombs>>
    /\ UNCHANGED <<now, pending, tgt, mode, inflight, used, granted, crashes,
                   admittedReleased>>

(* _require_not_quarantined_locked clears an expired quarantine           *)
(* (clickhouse_catalog.py:268-277); the next acquire mints a fresh id.     *)
QuarantineEnds(w) ==
    /\ pc[w] = "quar"
    /\ now >= quarUntil[w]
    /\ pc' = [pc EXCEPT ![w] = "idle"]
    /\ quarUntil' = [quarUntil EXCEPT ![w] = 0]
    /\ UNCHANGED <<now, rows, pending, lid, lterm, tgt, mode, inflight, cur,
                   used, granted, released, tombs, crashes,
                   admittedReleased>>

(* release() (clickhouse_lease.py:80-111): tombstone at the holder's OWN  *)
(* term, expires_at = acquired_at = now64().  Reads nothing; one INSERT.   *)
Release(w) ==
    /\ Mutation # "old_release"
    /\ pc[w] \in {"held", "ready"}
    /\ \E d \in Offsets :
          rows' = rows \cup {[t |-> lterm[w], id |-> lid[w], e |-> now + d]}
    /\ tombs' = tombs \cup {[t |-> lterm[w], id |-> lid[w]]}
    /\ released' = released \cup {lid[w]}
    /\ pc' = [pc EXCEPT ![w] = "idle"]
    /\ DropLease(w)
    /\ UNCHANGED <<now, pending, tgt, mode, inflight, cur, quarUntil, used,
                   granted, crashes, admittedReleased>>

(* MUTATION old_release: the earlier form described at                    *)
(* clickhouse_lease.py:88-99, INSERT ... SELECT term + 1 ... WHERE         *)
(* lease_id = :me over the head.  Its SELECT and its row landing are not   *)
(* atomic with a concurrent claim, so they are two steps here.             *)
OldReleaseRead(w) ==
    /\ Mutation = "old_release"
    /\ pc[w] \in {"held", "ready"}
    /\ IF TopId = lid[w]
       THEN /\ pc' = [pc EXCEPT ![w] = "rel"]
            /\ tgt' = [tgt EXCEPT ![w] = HeadTerm + 1]
            /\ UNCHANGED <<lid, lterm>>
       ELSE /\ pc' = [pc EXCEPT ![w] = "idle"]
            /\ DropLease(w)
            /\ UNCHANGED tgt
    /\ UNCHANGED <<now, rows, pending, mode, inflight, cur, quarUntil, used,
                   granted, released, tombs, crashes, admittedReleased>>

OldReleaseLand(w) ==
    /\ pc[w] = "rel"
    /\ \E d \in Offsets :
          rows' = rows \cup {[t |-> tgt[w], id |-> lid[w], e |-> now + d]}
    /\ tombs' = tombs \cup {[t |-> tgt[w], id |-> lid[w]]}
    /\ released' = released \cup {lid[w]}
    /\ pc' = [pc EXCEPT ![w] = "idle"]
    /\ DropLease(w)
    /\ tgt' = [tgt EXCEPT ![w] = 0]
    /\ UNCHANGED <<now, pending, mode, inflight, cur, quarUntil, used,
                   granted, crashes, admittedReleased>>

(* Process crash + restart: all local state (lease, quarantine) is lost;  *)
(* whatever the server is running keeps running.                           *)
Crash(w) ==
    /\ crashes < MaxCrashes
    /\ pc[w] # "idle"
    /\ crashes' = crashes + 1
    /\ pc' = [pc EXCEPT ![w] = "idle"]
    /\ DropLease(w)
    /\ cur' = [cur EXCEPT ![w] = NoRec]
    /\ quarUntil' = [quarUntil EXCEPT ![w] = 0]
    /\ ResetClaim(w)
    /\ UNCHANGED <<now, rows, pending, inflight, used, granted,
                   released, tombs, admittedReleased>>

(* True time advances.  A statement cannot outlive its max_execution_time: *)
(* time cannot pass an in-flight deadline.  MUTATION linger drops that     *)
(* (a quorum INSERT that fails its quorum wait leaves its part behind to   *)
(* replicate later).                                                       *)
Tick ==
    /\ now < MaxTime
    /\ Mutation = "linger" \/ \A f \in inflight : f.dl > now
    /\ now' = now + 1
    /\ UNCHANGED <<rows, pending, pc, lid, lterm, tgt, mode, inflight, cur,
                   quarUntil, used, granted, released, tombs, crashes,
                   admittedReleased>>

Next ==
    \/ Tick
    \/ Land
    \/ LandPending
    \/ \E w \in Writers :
          \/ AcquireHead(w) \/ RenewHead(w) \/ Insert(w) \/ InsertUnknown(w)
          \/ Readback(w) \/ Admit(w) \/ Done(w) \/ ClientError(w)
          \/ QuarantineEnds(w) \/ Release(w) \/ OldReleaseRead(w)
          \/ OldReleaseLand(w) \/ Crash(w)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* SAFETY                                                                  *)

\* fence() docstring (clickhouse_lease.py:200-211): "the two cannot overlap
\* unless the real skew exceeds the configured bound."
MutualExclusion == \A f, g \in inflight : f.w = g.w

\* clickhouse_catalog.py:196-198 "One publish in flight per writer, and so per
\* lease", plus the quarantine's purpose (:229-243): at most one admitted
\* statement's effect window is open at any true time, even within one lease.
OnePublishInFlight == Cardinality(inflight) <= 1

\* clickhouse_catalog.py:64-69 / lease.py:116-122: no successor is granted a
\* lease while an older holder's admitted statement may still land.
NoGrantDuringFlight ==
    \A f \in inflight, g \in granted : g.t > f.term => g.id = f.id

\* The read-back: at most one claimant per term is ever returned a lease.
SoleGrantPerTerm == \A g, h \in granted : g.t = h.t => g.id = h.id

\* release() docstring: the tombstone "ends the lease wherever it lands" --
\* no clock reading lets the fence admit a released lease_id again ...
ReleasedNeverAdmittable ==
    \A i \in released : \A d \in Offsets : ~FenceOK(i, d)
\* ... and indeed no statement for a released id is ever admitted.
NoAdmitAfterRelease == ~admittedReleased

\* release() docstring (:88-99): the tombstone "cannot share or supersede a
\* successor's term".
ReleaseNeverContestsSuccessor ==
    \A tb \in tombs, g \in granted : g.t = tb.t => g.id = tb.id

\* Terms are monotonic (append-only table, head = max(term)); and a writer's
\* held term only grows while it keeps one lease_id.
TermsMonotonic == [][HeadTerm' >= HeadTerm]_vars
HeldTermMonotonic ==
    [][\A w \in Writers : (lid[w] # 0 /\ lid'[w] = lid[w]) => lterm'[w] >= lterm[w]]_vars

\* NOT claimed by the code; checked to characterise it.  A grant is at a term
\* above every earlier grant.  (Expected to FAIL benignly: see README.)
GrantsIncrease ==
    [][\A g \in granted' \ granted : \A h \in granted : g.t > h.t]_vars

-----------------------------------------------------------------------------
(* State-space reduction (TLC VIEW, see MC.tla).  Rows, grants and        *)
(* tombstones at terms below Floor can influence no future step and no     *)
(* future violation: every later head read / fence sees only HeadTerm, a   *)
(* read-back reads tgt[w], a tombstone lands at lterm[w] (or tgt[w]), a    *)
(* pending row lands at its own term, a new grant is at >= Floor, and an   *)
(* in-flight record's term bounds the grants NoGrantDuringFlight pairs     *)
(* it with.  Two states equal above Floor are therefore bisimilar for all  *)
(* the invariants above.                                                   *)
Floor ==
    Min({HeadTerm}
        \cup {tgt[w] : w \in {x \in Writers : pc[x] \in {"insert", "readback", "rel"}}}
        \cup {lterm[w] : w \in {x \in Writers : lid[x] # 0 /\ lterm[x] # 0}}
        \cup {p.t : p \in pending}
        \cup {f.term : f \in inflight})
View == <<now, {r \in rows : r.t >= Floor}, pending, pc, lid, lterm, tgt, mode,
          inflight, cur, quarUntil, used, {g \in granted : g.t >= Floor},
          released, {tb \in tombs : tb.t >= Floor}, crashes, admittedReleased>>

TypeOK ==
    /\ now \in 0..MaxTime
    /\ pc \in [Writers -> {"idle", "insert", "readback", "held", "ready",
                           "pub", "quar", "rel"}]
    /\ lid \in [Writers -> {0} \cup Ids]
    /\ used \subseteq Ids
=============================================================================
