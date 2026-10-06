---------------------------- MODULE PublisherLease ----------------------------
(***************************************************************************)
(* A TLA+ model of the DMI publisher lease + fenced publish protocol.      *)
(*                                                                         *)
(* SOURCE OF TRUTH, main @ 5b3b632 (everything below is modelled from the *)
(* CODE; where the code and the prose disagree the comment says so):     *)
(*                                                                         *)
(*   native/csrc/catalog/lease_coordinator.cpp                             *)
(*     :202-277  claim_with_rival  -- head read, reject_live, insert,      *)
(*               singleton read-back.  THREE round trips, so a rival row   *)
(*               can land in either gap.  Modelled as three actions:       *)
(*               ClaimHead / ClaimInsert / ClaimRead.                      *)
(*     :183       release_statement -- the TOMBSTONE: an already-expired    *)
(*               row at the holder's OWN term.  Modelled in Release.       *)
(*     :279-301  head()            -- one deciding read, GROUP BY          *)
(*               (term,lid) at max(term), min(expires_at_ns) per lease,    *)
(*               ORDER BY lease_id DESC, live_until = max over claimants.  *)
(*     :303-316  fence()           -- resolves ONE lease at the head term  *)
(*               (LIMIT 1 after lease_id DESC) and asks: is it mine, and   *)
(*               does it have more than publish_timeout + clock_skew left. *)
(*     :318-331  fence_eval        -- a DECIDING read (comment says why).  *)
(*     :365-399  reject_live       -- admits iff the head is wholly dead   *)
(*               OR (claimants == 1 AND the single head lease is mine).    *)
(*               The claimants>1 branch is the contested-head quarantine.  *)
(*                                                                         *)
(*   native/csrc/catalog/catalog_writer.cpp                                *)
(*     :498-689  publish_snapshot  -- renew, then per manifest chunk       *)
(*               (fenced INSERT, read-back, renew), then the fenced        *)
(*               watermark INSERT, then the owners read-back.              *)
(*     :603      "The barrier, the fence and the visibility write are ONE  *)
(*               server-side statement."  Modelled as WmAdmit (predicate   *)
(*               evaluated at admission) + WmLand (row becomes durable     *)
(*               LATER) -- these are deliberately NOT atomic, because the  *)
(*               doc's "takeover instant" residual lives in that gap.      *)
(*     :148-169  config precondition lease_ttl > publish_timeout +         *)
(*               clock_skew + margin, and clock_skew != 0 with quorum.     *)
(*     :538      max_execution_time = publish_timeout -- modelled as the   *)
(*               statement deadline `dl` and the ManAbort/WmAbort actions. *)
(*                                                                         *)
(*   native/csrc/catalog/clickhouse_client.cpp:457                         *)
(*     deciding_read() == {select_sequential_consistency = 1}.             *)
(*     Modelled by the constant Linearizable (see Views below).            *)
(*                                                                         *)
(*   docs/catalog-descriptor-key.md :305-375, :319-329, :476-575, :670+    *)
(*   src/dmi/storage/capture/clickhouse_lease.py :83-92, :198-210          *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
  Publishers,      \* publish OPERATIONS (one pc each)
  SelfRace,        \* TRUE maps every Publisher onto ONE Writer, i.e. two
                   \* concurrent publish_snapshot() calls on one
                   \* ClickHouseCatalogWriter sharing one lease_id --
                   \* the doc's "One writer racing itself" (:541).
  Lids,            \* the lease_id pool.  `<` on these models ClickHouse's
                   \* UUID collation (doc :319-329 is explicit that it is
                   \* NOT text order); the protocol must be correct for ANY
                   \* total order, so claims pick their id nondeterministically
                   \* from the pool rather than in increasing order.
  MaxTerm, MaxTime, MaxVersion, MaxAttempts,
  TTL,             \* lease_ttl_ns
  PT,              \* publish_timeout_ns  == max_execution_time
  SKEW,            \* clock_skew_ns
  NumChunks,       \* manifest chunks per publish (catalog_writer.cpp:551)
  Linearizable,    \* TRUE  = select_sequential_consistency=1 honoured
                   \* FALSE = a deciding read may MISS an accepted insert
  AllowOverrun,    \* TRUE models doc :518 "max_execution_time is checked
                   \* between processing blocks ... a statement blocked in a
                   \* lock can overrun it"
  WriterLock,      \* TRUE models the re-entrant writer lock (doc :554-558)
  FreshPublishId,  \* TRUE = publish_id minted per call (catalog_writer.cpp:549)
  Timeouts         \* TRUE models the per-request deadlines of #159
                   \* (8b7991d): any request of a claim or a publish may
                   \* time out client-side; the writer then gives the lease
                   \* up without a tombstone (CatalogWriter::quarantine,
                   \* catalog_writer.cpp:268-274, :311-316; abandon_lease
                   \* :326-330), while what it sent may still land.

VARIABLES
  rows,            \* the append-only {prefix}_publisher_lease table.
                   \* NO UPDATE ANYWHERE: every action only ever adds.
  settled,         \* rows guaranteed visible to every replica
  now,             \* the SERVER clock (doc :332-335: expiries and the fence are
                   \* both stamped server-side)
  lease,           \* [Writers -> PublisherLease or NoLease]
  pc, ret, ctm, clid, chunk, ver, att,
  manifest,        \* {prefix}_snapshot_manifest rows
  wm,              \* {prefix}_index_watermark rows
  inflight,        \* statements admitted (predicate evaluated) but not landed
  used,            \* lease_ids ever minted.  new_uuid_v4() never repeats, and
                   \* two DISTINCT writers can never mint the same id -- only
                   \* a fork (SelfRace) shares one.
  maxLanded, wmOutOfOrder,
  late             \* claim INSERTs the client gave up on that may still
                   \* land: [term, lid, landed].  Stamped when it lands.

vars == <<rows, settled, now, lease, pc, ret, ctm, clid, chunk, ver, att,
          manifest, wm, inflight, used, maxLanded, wmOutOfOrder, late>>

----------------------------------------------------------------------------
(* helpers *)
SetMax(S) == IF S = {} THEN 0 ELSE CHOOSE x \in S : \A y \in S : y <= x
SetMin(S) == CHOOSE x \in S : \A y \in S : x <= y

NoLease == [term |-> 0, lid |-> 0, exp |-> 0]

(* Client writer objects.  The PublisherLease object lives on the writer,
   the program counter on the operation. *)
Writers     == IF SelfRace THEN {"shared"} ELSE Publishers
WriterOf(p) == IF SelfRace THEN "shared" ELSE p

(***************************************************************************)
(* THE STORE MODEL.                                                        *)
(*                                                                         *)
(* Views is the set of table images a single deciding read may observe.    *)
(* With select_sequential_consistency=1 (clickhouse_client.cpp:457) a read *)
(* sees every accepted row.  Without it, the replica it lands on may be    *)
(* behind: it sees everything already replicated (`settled`) and any       *)
(* subset of what is accepted but still in flight.  This is the single     *)
(* load-bearing assumption of the whole safety argument -- doc :319        *)
(* "Where that safety comes from is the read-back" -- so it is a knob.     *)
(***************************************************************************)
Views == IF Linearizable THEN {rows}
                         ELSE {V \in SUBSET rows : settled \subseteq V}

(* head() -- lease_coordinator.cpp:279-301 *)
HTerm(V)     == SetMax({r.term : r \in V})
HeadSet(V)   == {r \in V : r.term = HTerm(V)}
HLids(V)     == {r.lid : r \in HeadSet(V)}
(* "A lease's expiry at a term is the MINIMUM expires_at_ns written under
   its (term, lease_id)" -- doc :342-343, clickhouse_lease.py:83-92.  This is
   exactly what makes the release tombstone end the lease. *)
MinExpOf(V, l) == SetMin({r.exp : r \in {q \in HeadSet(V) : q.lid = l}})
(* ORDER BY lease_id DESC -> the greatest id under the collation *)
TopLid(V)    == SetMax(HLids(V))
LiveUntil(V) == SetMax({MinExpOf(V, l) : l \in HLids(V)})
NClaim(V)    == Cardinality(HLids(V))

(* reject_live -- lease_coordinator.cpp:365-399.  Returns (admits) iff the
   whole head term is dead, or there is exactly ONE claimant at the head and
   it is me.  claimants > 1 quarantines the term until every claim at it
   expires (doc :523 "Lease acquisition is not the window it looks like"). *)
RejectLivePasses(V, l) ==
  \/ LiveUntil(V) <= now
  \/ (NClaim(V) = 1 /\ TopLid(V) = l)

(* fence() -- lease_coordinator.cpp:303-316, doc :367-377.
   ONE subquery reading ONE row (doc :393 explains why the two-subquery form
   is unsound).  The margin is publish cap PLUS host clock skew bound
   (clickhouse_lease.py:198-210, the S - (b - a) >= 0 derivation). *)
FenceOk(V, l) ==
  /\ V # {}
  /\ TopLid(V) = l
  /\ MinExpOf(V, TopLid(V)) > now + PT + SKEW

PubId(p) == IF FreshPublishId THEN <<p, att[p]>> ELSE <<p, ver[p]>>

Busy(w) == \E q \in Publishers :
             WriterOf(q) = w /\ pc[q] \notin {"idle", "done", "failed"}

MyStmt(p) == {s \in inflight : s.who = p}

----------------------------------------------------------------------------
Init ==
  /\ rows = {} /\ settled = {} /\ now = 0
  /\ lease = [w \in Writers |-> NoLease]
  /\ pc  = [p \in Publishers |-> "idle"]
  /\ ret = [p \in Publishers |-> "idle"]
  /\ ctm = [p \in Publishers |-> 0]
  /\ clid = [p \in Publishers |-> 0]
  /\ chunk = [p \in Publishers |-> 0]
  /\ ver = [p \in Publishers |-> 0]
  /\ att = [p \in Publishers |-> 0]
  /\ manifest = {} /\ wm = {} /\ inflight = {} /\ used = {}
  /\ maxLanded = 0 /\ wmOutOfOrder = FALSE
  /\ late = {}

(* The server clock.  Bounded so the model closes. *)
Tick ==
  /\ now < MaxTime
  /\ now' = now + 1
  /\ UNCHANGED <<rows, settled, lease, pc, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(* Replication catching up.  Only meaningful when ~Linearizable. *)
Settle ==
  /\ ~Linearizable
  /\ \E r \in rows \ settled : settled' = settled \cup {r}
  /\ UNCHANGED <<rows, now, lease, pc, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(* acquire_publisher_lease() then publish_snapshot().
   lease_coordinator.cpp:153-154: acquire() reuses the HELD lease_id and
   otherwise mints a fresh UUID -- minted BEFORE the head read. *)
StartPublish(p) ==
  /\ pc[p] = "idle"
  /\ att[p] < MaxAttempts
  /\ WriterLock => ~Busy(WriterOf(p))
  /\ \/ /\ lease[WriterOf(p)].term > 0
        /\ clid' = [clid EXCEPT ![p] = lease[WriterOf(p)].lid]
        /\ UNCHANGED used
     \/ /\ lease[WriterOf(p)].term = 0
        /\ \E l \in Lids \ used :
             /\ clid' = [clid EXCEPT ![p] = l]
             /\ used' = used \cup {l}
  /\ att' = [att EXCEPT ![p] = att[p] + 1]
  /\ pc'  = [pc  EXCEPT ![p] = "claim_head"]
  /\ ret' = [ret EXCEPT ![p] = "pub_alloc"]
  /\ chunk' = [chunk EXCEPT ![p] = 0]
  /\ UNCHANGED <<rows, settled, now, lease, ctm, ver,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, late>>

(* ---- claim: lease_coordinator.cpp:202-277, three round trips ---- *)

(* Round trip 1: head() + reject_live.  These are one query plus pure local
   computation on its result, so nothing can interleave INSIDE them.
   acquire() reuses the held lease_id, otherwise a fresh one (:153-154). *)
ClaimHead(p) ==
  LET w == WriterOf(p) IN
  LET l == IF lease[w].term > 0 THEN lease[w].lid ELSE clid[p] IN
  /\ pc[p] = "claim_head"
  /\ \E V \in Views :
       \/ /\ RejectLivePasses(V, l)
          /\ HTerm(V) + 1 <= MaxTerm          \* MODEL BOUND, not protocol
          /\ ctm'  = [ctm  EXCEPT ![p] = HTerm(V) + 1]
          /\ clid' = [clid EXCEPT ![p] = l]
          /\ pc'   = [pc   EXCEPT ![p] = "claim_insert"]
          /\ UNCHANGED lease
       \/ /\ ~RejectLivePasses(V, l)           \* throws kHeld; resets lease_
          /\ lease' = [lease EXCEPT ![w] = NoLease]
          /\ pc' = [pc EXCEPT ![p] = "failed"]
          /\ UNCHANGED <<ctm, clid, used, late>>
  /\ UNCHANGED <<rows, settled, now, ret, chunk, ver, att, used,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, late>>

(* Round trip 2: the claim INSERT (:350-363).  Append only. *)
ClaimInsert(p) ==
  /\ pc[p] = "claim_insert"
  /\ LET r == [term |-> ctm[p], lid |-> clid[p], exp |-> now + TTL] IN
       /\ rows' = rows \cup {r}
       /\ settled' = IF Linearizable THEN settled \cup {r} ELSE settled
  /\ pc' = [pc EXCEPT ![p] = "claim_read"]
  /\ UNCHANGED <<now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(* Round trip 3: the singleton read-back (:248-271).  THIS is where the
   safety comes from, per doc :319-329 -- not from the fence. *)
ClaimRead(p) ==
  LET w == WriterOf(p) IN
  /\ pc[p] = "claim_read"
  /\ \E V \in Views :
       LET mine == {q \in V : q.term = ctm[p]} IN
       LET owners == {q.lid : q \in mine} IN
       \/ /\ owners = {clid[p]}
          /\ lease' = [lease EXCEPT ![w] =
               [term |-> ctm[p], lid |-> clid[p],
                exp  |-> SetMin({q.exp : q \in mine})]]
          /\ pc' = [pc EXCEPT ![p] = ret[p]]
       \/ /\ owners # {clid[p]}                 \* :272-276, claim refused
          /\ lease' = [lease EXCEPT ![w] = NoLease]
          /\ pc' = [pc EXCEPT ![p] = "failed"]
  /\ UNCHANGED <<rows, settled, now, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(* release() -- lease_coordinator.cpp:170-190.  A TOMBSTONE, not an UPDATE and
   not a fenced head write: an already-expired row at the holder's OWN term,
   so min(expires_at_ns) for that (term,lease_id) collapses to now. *)
Release(w) ==
  /\ lease[w].term > 0
  /\ ~Busy(w)
  /\ LET r == [term |-> lease[w].term, lid |-> lease[w].lid, exp |-> now] IN
       /\ rows' = rows \cup {r}
       /\ settled' = IF Linearizable THEN settled \cup {r} ELSE settled
  /\ lease' = [lease EXCEPT ![w] = NoLease]
  /\ UNCHANGED <<now, pc, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(* ---- publish: catalog_writer.cpp:498-689 ---- *)

PubAlloc(p) ==
  /\ pc[p] = "pub_alloc"
  /\ SetMax({r.ver : r \in wm}) + 1 <= MaxVersion
  /\ ver' = [ver EXCEPT ![p] = SetMax({r.ver : r \in wm}) + 1]
  /\ pc' = [pc EXCEPT ![p] = IF NumChunks = 0 THEN "wm_admit" ELSE "man_next"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

ManNext(p) ==
  /\ pc[p] = "man_next"
  /\ chunk' = [chunk EXCEPT ![p] = chunk[p] + 1]
  /\ pc' = [pc EXCEPT ![p] = "man_admit"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(* The fenced manifest INSERT (catalog_writer.cpp:551-567).  A fence refusal
   writes ZERO rows WITHOUT raising (doc :432) -- hence the else branch goes
   straight to the read-back, which is what catches it. *)
ManAdmit(p) ==
  LET w == WriterOf(p) IN
  /\ pc[p] = "man_admit"
  /\ \E V \in Views :
       \/ /\ lease[w].term > 0 /\ FenceOk(V, lease[w].lid)
          /\ inflight' = inflight \cup
               {[who |-> p, kind |-> "manifest", ver |-> ver[p],
                 pid |-> PubId(p), pack |-> chunk[p], dl |-> now + PT]}
          /\ pc' = [pc EXCEPT ![p] = "man_land"]
       \/ /\ ~(lease[w].term > 0 /\ FenceOk(V, lease[w].lid))
          /\ pc' = [pc EXCEPT ![p] = "man_read"]
          /\ UNCHANGED inflight
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, maxLanded, wmOutOfOrder, used, late>>

ManLand(p) ==
  /\ pc[p] = "man_land"
  /\ \E s \in MyStmt(p) :
       /\ (now <= s.dl \/ AllowOverrun)
       /\ manifest' = manifest \cup
            {[ver |-> s.ver, pid |-> s.pid, pack |-> s.pack]}
       /\ inflight' = inflight \ {s}
  /\ pc' = [pc EXCEPT ![p] = "man_read"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 wm, maxLanded, wmOutOfOrder, used, late>>

(* max_execution_time / timeout_overflow_mode=throw (catalog_writer.cpp:538) *)
ManAbort(p) ==
  /\ pc[p] = "man_land"
  /\ ~AllowOverrun
  /\ \E s \in MyStmt(p) : now > s.dl /\ inflight' = inflight \ {s}
  /\ pc' = [pc EXCEPT ![p] = "failed"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, maxLanded, wmOutOfOrder, used, late>>

(* "Every conditional manifest INSERT is read back before the next renewal"
   -- doc :431, catalog_writer.cpp:568-587.  Then the renewal (:599). *)
ManRead(p) ==
  /\ pc[p] = "man_read"
  /\ \/ /\ \E m \in manifest :
             m.ver = ver[p] /\ m.pid = PubId(p) /\ m.pack = chunk[p]
        /\ pc'  = [pc  EXCEPT ![p] = "claim_head"]
        /\ ret' = [ret EXCEPT ![p] =
             IF chunk[p] < NumChunks THEN "man_next" ELSE "wm_admit"]
     \/ /\ ~\E m \in manifest :
             m.ver = ver[p] /\ m.pid = PubId(p) /\ m.pack = chunk[p]
        /\ pc' = [pc EXCEPT ![p] = "failed"]
        /\ UNCHANGED ret
  /\ UNCHANGED <<rows, settled, now, lease, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(* THE visibility write -- catalog_writer.cpp:603-626.  Barrier AND fence AND
   the INSERT are ONE server-side statement, so both predicates are evaluated
   HERE, at admission.  The row lands in WmLand, possibly later: doc :511
   "A's watermark statement must evaluate the fence before B's lease row
   commits, and still be in flight when B publishes." *)
WmAdmit(p) ==
  LET w == WriterOf(p) IN
  /\ pc[p] = "wm_admit"
  /\ \E V \in Views :
       \/ /\ SetMax({r.ver : r \in wm}) < ver[p]       \* the version barrier
          /\ lease[w].term > 0 /\ FenceOk(V, lease[w].lid)
          /\ inflight' = inflight \cup
               {[who |-> p, kind |-> "watermark", ver |-> ver[p],
                 pid |-> PubId(p), pack |-> 0, dl |-> now + PT]}
          /\ pc' = [pc EXCEPT ![p] = "wm_land"]
       \/ /\ ~( SetMax({r.ver : r \in wm}) < ver[p]
                /\ lease[w].term > 0 /\ FenceOk(V, lease[w].lid) )
          /\ pc' = [pc EXCEPT ![p] = "wm_read"]
          /\ UNCHANGED inflight
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, maxLanded, wmOutOfOrder, used, late>>

WmLand(p) ==
  /\ pc[p] = "wm_land"
  /\ \E s \in MyStmt(p) :
       /\ (now <= s.dl \/ AllowOverrun)
       /\ wm' = wm \cup {[ver |-> s.ver, pid |-> s.pid]}
       /\ inflight' = inflight \ {s}
       /\ wmOutOfOrder' = (wmOutOfOrder \/ (s.ver <= maxLanded))
       /\ maxLanded' = SetMax({maxLanded, s.ver})
  /\ pc' = [pc EXCEPT ![p] = "wm_read"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, used, late>>

WmAbort(p) ==
  /\ pc[p] = "wm_land"
  /\ ~AllowOverrun
  /\ \E s \in MyStmt(p) : now > s.dl /\ inflight' = inflight \ {s}
  /\ pc' = [pc EXCEPT ![p] = "failed"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, maxLanded, wmOutOfOrder, used, late>>

(* "Ownership, not occupancy" -- catalog_writer.cpp:628-648 *)
WmRead(p) ==
  /\ pc[p] = "wm_read"
  /\ \/ /\ \E r \in wm : r.ver = ver[p] /\ r.pid = PubId(p)
        /\ pc' = [pc EXCEPT ![p] = "done"]
     \/ /\ ~\E r \in wm : r.ver = ver[p] /\ r.pid = PubId(p)
        /\ pc' = [pc EXCEPT ![p] = "failed"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

Reset(p) ==
  /\ pc[p] \in {"done", "failed"}
  /\ pc' = [pc EXCEPT ![p] = "idle"]
  /\ UNCHANGED <<rows, settled, now, lease, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

(***************************************************************************)
(* TIMEOUTS -- #159 (8b7991d), main @ 5b3b632.                             *)
(*                                                                         *)
(* Every lease request now runs under a RequestDeadline                    *)
(* (lease_coordinator.cpp:86-110 run()): the held lease's deadline while   *)
(* that is ahead, else the claim bound; a claim is also bounded, from its  *)
(* INSERT on, by the deadline of the lease it takes (:216-231).  A request *)
(* that runs out of time throws a ClickHouseError, and                     *)
(*   - acquire_lease quarantines when the claim was a renewal or its       *)
(*     INSERT may have reached the server (catalog_writer.cpp:302-318);   *)
(*   - renew_for_publish quarantines on any non-CatalogError (:284-295);   *)
(*   - the storage service abandons a lease whose deadline passed          *)
(*     unrenewed (abandon_lease, :326-330), as a quarantine.               *)
(* A quarantine discards the local lease WITHOUT a tombstone (:268-274),   *)
(* and the timed-out statement may still land: an INSERT carries the time  *)
(* left as max_execution_time (add_write_caps, lease_coordinator.cpp      *)
(* :112-143) but "a request held up in transit can still land up to that   *)
(* delay after the client gave up".  A fenced publish statement keeps      *)
(* max_execution_time = publish_timeout (catalog_writer.cpp:536-540), so   *)
(* the client may give up on it while it still runs.                       *)
(*                                                                         *)
(* Over-approximations, all of which only ADD behaviours:                  *)
(*   - the quarantine's TTL wait before the next claim is not modelled, so *)
(*     a writer may claim again at once;                                   *)
(*   - a late claim row may land at ANY later time, not only within the    *)
(*     transit delay;                                                      *)
(*   - a time-out may strike any request, whatever the deadline.           *)
(***************************************************************************)

GiveUp(p) ==
  LET w == WriterOf(p) IN
  /\ Timeouts
  /\ pc[p] \notin {"idle", "done", "failed"}
  /\ lease' = [lease EXCEPT ![w] = NoLease]
  /\ pc' = [pc EXCEPT ![p] = "failed"]
  \* A claim INSERT that timed out may still land, or may never.
  /\ \/ /\ pc[p] = "claim_insert"
        /\ late' = late \cup {[term |-> ctm[p], lid |-> clid[p], landed |-> FALSE]}
     \/ UNCHANGED late
  \* A fenced statement the client gave up on stays in flight, ownerless.
  /\ inflight' = {IF s.who = p THEN [s EXCEPT !.who = "gone"] ELSE s :
                    s \in inflight}
  /\ UNCHANGED <<rows, settled, now, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, maxLanded, wmOutOfOrder, used>>

LateLand ==
  /\ \E x \in late :
       LET r == [term |-> x.term, lid |-> x.lid, exp |-> now + TTL] IN
       /\ ~x.landed
       /\ rows' = rows \cup {r}
       /\ settled' = IF Linearizable THEN settled \cup {r} ELSE settled
       /\ late' = (late \ {x}) \cup {[x EXCEPT !.landed = TRUE]}
  /\ UNCHANGED <<now, lease, pc, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used>>

(* abandon_lease (catalog_writer.cpp:326-330): between passes, a lease past
   its deadline is dropped without a tombstone. *)
Abandon(w) ==
  /\ Timeouts
  /\ lease[w].term > 0
  /\ ~Busy(w)
  /\ lease' = [lease EXCEPT ![w] = NoLease]
  /\ UNCHANGED <<rows, settled, now, pc, ret, ctm, clid, chunk, ver, att,
                 manifest, wm, inflight, maxLanded, wmOutOfOrder, used, late>>

Gone == {s \in inflight : s.who = "gone"}

GoneLand ==
  /\ \E s \in Gone :
       /\ (now <= s.dl \/ AllowOverrun)
       /\ inflight' = inflight \ {s}
       /\ IF s.kind = "manifest"
            THEN /\ manifest' = manifest \cup
                      {[ver |-> s.ver, pid |-> s.pid, pack |-> s.pack]}
                 /\ UNCHANGED <<wm, maxLanded, wmOutOfOrder>>
            ELSE /\ wm' = wm \cup {[ver |-> s.ver, pid |-> s.pid]}
                 /\ wmOutOfOrder' = (wmOutOfOrder \/ (s.ver <= maxLanded))
                 /\ maxLanded' = SetMax({maxLanded, s.ver})
                 /\ UNCHANGED manifest
  /\ UNCHANGED <<rows, settled, now, lease, pc, ret, ctm, clid, chunk, ver,
                 att, used, late>>

GoneAbort ==
  /\ ~AllowOverrun
  /\ \E s \in Gone : now > s.dl /\ inflight' = inflight \ {s}
  /\ UNCHANGED <<rows, settled, now, lease, pc, ret, ctm, clid, chunk, ver,
                 att, manifest, wm, maxLanded, wmOutOfOrder, used, late>>

Next ==
  \/ Tick \/ Settle
  \/ \E w \in Writers : Release(w) \/ Abandon(w)
  \/ LateLand \/ GoneLand \/ GoneAbort
  \/ \E p \in Publishers :
       \/ StartPublish(p) \/ ClaimHead(p) \/ ClaimInsert(p) \/ ClaimRead(p)
       \/ PubAlloc(p) \/ ManNext(p) \/ ManAdmit(p) \/ ManLand(p)
       \/ ManAbort(p) \/ ManRead(p)
       \/ WmAdmit(p) \/ WmLand(p) \/ WmAbort(p) \/ WmRead(p)
       \/ Reset(p) \/ GiveUp(p)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(***************************************************************************)
(* OBLIGATIONS                                                             *)
(***************************************************************************)

WatermarkStmts == {s \in inflight : s.kind = "watermark"}

(*  A statement whose max_execution_time has passed is guaranteed to be
    aborted by the server (timeout_overflow_mode = throw,
    catalog_writer.cpp:538-539), so it can no longer make anything visible.
    Counting it would be a MODELLING ARTIFACT: in this spec a statement sits
    in `inflight` until its Abort action is scheduled, which the real server
    does not allow.  AllowOverrun removes the guarantee (doc :518: the cap is
    "checked between processing blocks rather than pre-empted").             *)
CanStillLand(s)    == AllowOverrun \/ now <= s.dl
LandableWmStmts    == {s \in WatermarkStmts : CanStillLand(s)}

(*-------------------------------------------------------------------------*)
(* 1. NoOverlappingAdmit  --  THE safety property.                          *)
(*    catalog_writer.cpp:545-548 ("a publisher whose lease was taken over   *)
(*    makes NO snapshot visible") and doc :350-365.                         *)
(*    No two watermark-admitting statements are in flight at once.          *)
(*    EXPECTED: HOLDS with Linearizable /\ ~AllowOverrun.                   *)
(*-------------------------------------------------------------------------*)
NoOverlappingAdmit == Cardinality(LandableWmStmts) <= 1

(*  The same property stated over distinct WRITERS, so that the "one writer
    racing itself" configuration can be separated from the lease property. *)
NoOverlappingWriters ==
  \A s1, s2 \in WatermarkStmts :
     WriterOf(s1.who) = WriterOf(s2.who) \/ s1 = s2

(*-------------------------------------------------------------------------*)
(* 2. WatermarkMonotonic -- index_watermark.index_version strictly          *)
(*    increases.  doc :295-301 (the 1% residual) and :541 (one writer       *)
(*    racing itself: "an already-pinned watermark grew").                   *)
(*    EXPECTED: HOLDS with WriterLock; FAILS without it.                    *)
(*-------------------------------------------------------------------------*)
WatermarkMonotonic == ~wmOutOfOrder

(*-------------------------------------------------------------------------*)
(* 3. CompleteManifest -- every visible watermark row has a complete        *)
(*    manifest behind it.  catalog_writer.cpp:568-587, doc :431-434.        *)
(*    EXPECTED: HOLDS.                                                      *)
(*-------------------------------------------------------------------------*)
CompleteManifest ==
  \A r \in wm : \A c \in 1..NumChunks :
     \E m \in manifest : m.ver = r.ver /\ m.pid = r.pid /\ m.pack = c

(*  and the other direction: the snapshot a reader assembles for a watermark
    row is EXACTLY the packs that publish intended -- no orphan is ever
    attributed to it.  This is the non-vacuous half of obligation 5.        *)
SnapshotExact ==
  \A r \in wm : \A m \in manifest :
     (m.pid = r.pid) => (m.ver = r.ver /\ m.pack \in 1..NumChunks)

(*-------------------------------------------------------------------------*)
(* 4. TwoBelieversIsSafe -- doc :531.                                       *)
(*    Believers: writers that hold a PublisherLease object they consider    *)
(*    live.  The doc CLAIMS two can exist at once and that this is safe.    *)
(*-------------------------------------------------------------------------*)
Believers == {w \in Writers : lease[w].term > 0 /\ lease[w].exp > now}

(*  Refutation target, STRONG reading of doc :531: two writers hold leases
    that are BOTH unexpired on the server clock at the same instant.        *)
AtMostOneBeliever == Cardinality(Believers) <= 1

(*  Refutation target, WEAK reading of doc :531 -- and the one the doc's own
    last sentence uses: '"only one publisher holds the lease" is not
    literally true of the CLIENT OBJECTS, only of the row the fence
    resolves.'  A holder here is any writer still carrying a PublisherLease
    object it has not been told it lost.                                    *)
Holders == {w \in Writers : lease[w].term > 0}
AtMostOneHolder == Cardinality(Holders) <= 1

(*  and the safety question asked of the weak reading *)
TwoHoldersIsSafe ==
  (Cardinality(Holders) > 1) =>
     /\ Cardinality({w \in Holders : FenceOk(rows, lease[w].lid)}) <= 1
     /\ Cardinality(LandableWmStmts) <= 1

(*  The safety half: at most one believer can ever pass the fence, because
    the fence resolves ONE lease at the head term (lease_coordinator.cpp:310
    ORDER BY lease_id DESC LIMIT 1).  EXPECTED: HOLDS.                      *)
AtMostOneFenceable ==
  Cardinality({w \in Writers :
                 lease[w].term > 0 /\ FenceOk(rows, lease[w].lid)}) <= 1

TwoBelieversIsSafe == AtMostOneFenceable /\ NoOverlappingAdmit

(*-------------------------------------------------------------------------*)
(* 5. OrphanManifestInert -- doc :480-502.                                  *)
(*    Orphans: manifest rows whose publish never wrote a watermark row.     *)
(*    Refutation target NoOrphanManifestRows is EXPECTED TO FAIL (the       *)
(*    weakening is real); SnapshotExact above is the inertness claim.       *)
(*-------------------------------------------------------------------------*)
(*  A manifest row with no watermark row is not yet an orphan if its publish
    is still running -- that is just a publish in flight.  A REAL orphan is
    one whose publish has ENDED (refused at the watermark, or refused at the
    renewal between the two statements) with no watermark row behind it.     *)
LivePid(pid) == \E p \in Publishers :
                  PubId(p) = pid /\ pc[p] \notin {"idle", "done", "failed"}
Orphans == {m \in manifest :
              ~LivePid(m.pid) /\ ~\E r \in wm : r.pid = m.pid}
NoOrphanManifestRows == Orphans = {}

(*  "no worse than documented": an orphan is always a PREFIX of a publish's
    chunks -- the takeover cuts the loop, it does not scatter rows.        *)
OrphansArePrefixes ==
  \A m \in Orphans : \A c \in 1..m.pack :
     \E m2 \in manifest : m2.pid = m.pid /\ m2.pack = c

(*  Combined invariant used for the "everything at once" runs. *)
AllSafety ==
  /\ NoOverlappingAdmit
  /\ WatermarkMonotonic
  /\ CompleteManifest
  /\ SnapshotExact
  /\ AtMostOneFenceable

(*-------------------------------------------------------------------------*)
(* VACUITY GUARDS.  Each of these is an invariant we WANT TLC to refute:    *)
(* the violation trace is the proof that the model actually reaches the     *)
(* state the safety invariants are quantified over.  An invariant that      *)
(* holds only because its subject is unreachable proves nothing.            *)
(*-------------------------------------------------------------------------*)
NeverPublishes    == wm = {}                  \* refuted => snapshots do land
NeverFences       == ~\E w \in Writers :
                       lease[w].term > 0 /\ FenceOk(rows, lease[w].lid)
NeverAdmits       == LandableWmStmts = {}     \* refuted => the fence admits
NeverContested    == \A V \in {rows} : NClaim(V) <= 1   \* contested head
NeverTwoBelievers == Cardinality(Believers) <= 1
(* two believers WHILE one of them is mid-flight in a visibility write *)
NoBelieverDuringAdmit ==
  ~(Cardinality(Holders) > 1 /\ LandableWmStmts # {})

(* Timeouts: a late claim row lands; a gone statement is admitted and in
   flight while another writer holds a lease. *)
NeverLateLands    == ~\E x \in late : x.landed
NeverGoneInFlight == ~\E s \in Gone : CanStillLand(s) /\ s.kind = "watermark"

(* model bound, used as a TLC state constraint *)
Bound == /\ now <= MaxTime
         /\ HTerm(rows) <= MaxTerm
=============================================================================
