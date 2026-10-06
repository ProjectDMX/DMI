--------------------------- MODULE LeaseLifecycle ---------------------------
(***************************************************************************)
(* The LEASE LIFECYCLE LAYER of                                            *)
(*   native/csrc/catalog/storage_service.cpp                               *)
(* on projectdmx/dmi main @ 5b3b632 (1763 lines).                          *)
(*                                                                         *)
(* First written against 99ee4ae; re-synced to 5b3b632, after #159         *)
(* (8b7991d: a deadline for every request made under the lease, a renewal  *)
(* before every lease-locked request, start() past a slow claim, the       *)
(* service's own late claim rows kept out of the latch), #162 (0e1a112)    *)
(* and #163 (5b3b632: one owner process per spool directory).              *)
(*                                                                         *)
(* SCOPE.  This models the SERVICE's lease lifecycle -- the lease thread,  *)
(* the stretches of catalog work that hold the lease lock, the lease       *)
(* deadline, the quarantine window, the 2 x TTL latch, the start wait and  *)
(* the spool sweep -- NOT the claim/read-back protocol underneath it.  The *)
(* LeaseCoordinator is abstracted to its CONTRACT:                         *)
(*                                                                         *)
(*     a claim presenting lease id L is ADMITTED iff the head row is dead  *)
(*     or the head row IS L (lease_coordinator.cpp:365-401 reject_live),   *)
(*     REFUSED with kHeld otherwise, and may return an UNKNOWN outcome     *)
(*     when the request cannot be completed.                              *)
(*                                                                         *)
(* That contract -- three round trips, contested heads, the fence, the     *)
(* tombstone, replica staleness -- is discharged by PublisherLease.tla in  *)
(* this directory.  Anything this module says about the coordinator is     *)
(* only as strong as that discharge; see LIMITS at the foot of the file.   *)
(*                                                                         *)
(* REQUESTS TAKE TIME (new in this revision).  A ClickHouse request is     *)
(* issued by one step and completed by a later one, MaxReq ticks at most   *)
(* after it.  One request is in flight per service at a time, which is     *)
(* lease_mutex_: the lease thread and the cycle's stretches both hold it   *)
(* for every catalog request they make (LeaseScope).                       *)
(*                                                                         *)
(* SOURCE LINES (5b3b632).  Every action names the code it stands for.     *)
(*   storage_service.cpp                                                   *)
(*     :60-70     lease_tick_ns = max(ttl/6, 10ms)          -> Tick        *)
(*     :114-149   LeaseScope: abandon_lease_if_expired on entry and exit;  *)
(*                every request bounded by the lease deadline, and         *)
(*                keep_lease_in_pass() before each one                     *)
(*     :190-211   the constructor refuses a skew with no renewal window    *)
(*     :213-221   the constructor opens the spool: kOwned -> throws        *)
(*     :263-319   start(): schema, lease, lease THREAD, sweep, reconcile   *)
(*     :321-337   the spool sweep, AFTER the lease                         *)
(*     :372-413   stop()                                                   *)
(*     :576-581   run_cycle(): ensure_publisher_lease under a LeaseScope   *)
(*     :1194-1207 index_bounded(): plan and commit, each its own scope;    *)
(*                "Nothing here stamps a renewal"                          *)
(*     :1380-1436 keep_lease()        the lease thread                     *)
(*     :1401-1410   next_wake: due = sent + ttl/3, clamped to [1ms, tick]  *)
(*     :1438-1461 renew_lease_if_due()  due = lease_sent_ns + ttl/3        *)
(*     :1463-1474 keep_lease_in_pass()                                     *)
(*     :1476-1485 abandon_lease_if_expired()                               *)
(*     :1547-1618 acquire_lease_at_start()                                 *)
(*     :1620-1676 ensure_publisher_lease()                                 *)
(*     :1678-1711 lease_held_elsewhere()  the 2 x TTL latch, and the       *)
(*                refused_by_own_claims() reset at :1682-1696              *)
(*   catalog_writer.cpp                                                    *)
(*     :243-253   quarantine_in_force()  now < quarantine_until            *)
(*     :268-274   quarantine()   drop the lease, window = now + lease_ttl  *)
(*     :284-294   renew_for_publish()  std::exception -> quarantine()      *)
(*     :296-318   acquire_lease()  quarantine iff renewing or the claim    *)
(*                INSERT may have been sent (:315)                         *)
(*     :326-330   abandon_lease()  -> quarantine()                         *)
(*     :386-391, :449-454  a publish statement's std::exception ->         *)
(*                quarantine()                                             *)
(*   lease_coordinator.h:31-60  the lease deadline:                        *)
(*                sent + lease_ttl - clock_skew - 0.1 s                    *)
(*   lease_coordinator.cpp                                                 *)
(*     :86-110    run(): a held lease's deadline while it is ahead, else   *)
(*                the claim bound min(request_s, ttl/3)                    *)
(*     :112-143   add_write_caps(): a lease INSERT carries the attempt's   *)
(*                time to the server as max_execution_time                 *)
(*     :145-155   acquire()  reuses the HELD lease_id, else mints a fresh  *)
(*     :202-276   claim_with_rival(): sent_ns taken before the INSERT      *)
(*                (:219), confirmed by the new lease's deadline            *)
(*     :365-401   reject_live(); refused_by_own_claims_ at :372-378        *)
(*   clickhouse_client.cpp:489  before_request (the in-pass renewal) runs  *)
(*                first; :600-622 every attempt is cut to the tightest     *)
(*                RequestDeadline, and one already past is not sent        *)
(*   indexer.cpp:322  the publish is SKIPPED when every pack was already   *)
(*                committed -- which no longer moves the renewal schedule  *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
  Services,          \* the CaptureStorageService instances
  TTL,               \* writer.lease_ttl_ns, in TICKS.  6 keeps ttl/6 and
                     \* ttl/3 exact integers, which is what the code divides by
  PT,                \* publish_timeout_ns: the statement cap
  Skew,              \* clock_skew: ticks of extra life a row is seen to have
                     \* on a LAGGING replica
  PredTTL,           \* the TTL a crashed PREDECESSOR ran with
  StartWait,         \* start_lease_wait_ns
  MaxTime,
  MaxLate,           \* ticks of OS lateness allowed on a lease-thread wake
  Foreign,           \* TRUE: a rival publisher may claim the catalog
  ForeignStops,      \* TRUE: the rival may stop, releasing with a tombstone
  ForeignBudget,     \* how many times the rival may take the catalog
  ForeignStopBy,     \* the rival renews only while now < this
  MaxCuts,           \* how many ClickHouse cut/restore pairs are allowed
  AllowUnknown,      \* TRUE: a request to a LIVE ClickHouse may fail with
                     \* its outcome unknown (a transport error or timeout)
  AllowSkipPublish,  \* TRUE: a cycle may run a lease-locked stretch that
                     \* does NOT publish (an index pass's plan, a replay
                     \* guard, a pass whose packs were all committed): it
                     \* renews only if the renewal is due.  Before #159 such
                     \* a pass stamped last_renew_ns_ anyway (O1_skip).
  ReuseLid,          \* COUNTERFACTUAL for O2: TRUE makes a post-quarantine
                     \* claim present the DROPPED lease_id
  CycleOn,           \* TRUE: model run_cycle()'s lease-locked work
  PredHolds,         \* TRUE: a crashed predecessor's row is live at Init
  AllowStop,         \* TRUE: a service may call stop()
  MaxReacq,          \* cap on the re-acquisition counter (a state bound only)
  \* ---- new with the #159/#163 re-sync ----
  MaxReq,            \* ticks one ClickHouse request may take.  0 is the old
                     \* zero-time model
  CfgSkew,           \* writer.clock_skew_ns, in ticks: what the DEADLINE
                     \* subtracts (lease_coordinator.h:31-60)
  RivalEarly,        \* ticks before a row's (lagging-view) expiry at which a
                     \* rival may already see it dead: a rival reading a
                     \* replica whose clock runs ahead.  Skew + this is the
                     \* real two-way skew; the deadline is safe iff
                     \* CfgSkew >= RivalEarly - Skew
  SameSpool,         \* TRUE: every service in Services uses ONE spool dir
  OwnerLock,         \* TRUE: the #163 spool owner lock (flock on
                     \* <dir>/.owner.lock); FALSE: the pre-#163 COUNTERFACTUAL
  LateStart,         \* TRUE: a service may be constructed at any instant,
                     \* not only at t = 0
  AllowCrash         \* TRUE: a process may die (no tombstone; its flock goes)

ASSUME CfgSkew * 2 < TTL   \* storage_service.cpp:190-211 refuses the rest

VARIABLES
  now, chUp, cuts,
  hOwner, hLid, hExp, lidGen,          \* the lease table HEAD (abstracted)
  held, myLid, sent, qUntil,           \* per service: the writer's lease state
  heSince, nextClaim,                  \* held_elsewhere_since_ns_, next_claim_ns_
  phase, reacq, dropped,
  busy, who, rLid, rStart, rEnd, forced,   \* the request in flight
  wake, cwake, swept,
  runBroken, startWaited, selfRef, coSweep, lost,  \* history variables
  startDL, stretched, startUnk,        \* acquire_lease_at_start's locals
  spoolLk,                             \* the spool owner lock (SameSpool)
  fWake, fLid, fBudget, fHeld          \* the rival publisher

envVars  == <<now, chUp, cuts>>
headVars == <<hOwner, hLid, hExp, lidGen>>
locVars  == <<held, myLid, sent, qUntil, heSince, nextClaim, phase, reacq,
              dropped>>
reqVars  == <<busy, who, rLid, rStart, rEnd, forced>>
resVars  == <<wake, cwake, swept, runBroken, startWaited, selfRef, coSweep,
              lost>>
stVars   == <<startDL, stretched, startUnk, spoolLk>>
rivVars  == <<fWake, fLid, fBudget, fHeld>>
vars     == <<envVars, headVars, locVars, reqVars, resVars, stVars, rivVars>>

-----------------------------------------------------------------------------
(* Derived constants, exactly as the code computes them.                    *)

Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b

\* storage_service.cpp:68-70  lease_tick_ns(ttl) = max(ttl/6, 10ms).
Tick      == IF TTL \div 6 > 0 THEN TTL \div 6 ELSE 1
DueAfter  == TTL \div 3        \* storage_service.cpp:1458, :1407
LatchWin  == 2 * TTL           \* storage_service.cpp:1700
\* lease_coordinator.cpp:66-77  claim bound = min(request_s, ttl/3).  The
\* default request_s (60 s) exceeds ttl/3 at the default 15 s TTL.
CB        == TTL \div 3
\* storage_service.cpp:1552-1553  clamp(ttl/10, 50ms, 500ms): one tick is
\* the finest grain this model has, COARSER than the real poll.
StartPoll == 1

Owners == Services \cup {"none", "F", "P"}
Never  == MaxTime + 99      \* a wake time that never arrives

\* lease_coordinator.cpp:41-45  the lease deadline, sent + ttl - skew -
\* 0.1 s, on the writer's steady clock.  The 0.1 s margin is far below a
\* tick: the model keeps it as STRICTNESS -- a request must complete at a
\* time < DL, one that would not is cut at DL, and a lease is usable only
\* while now < DL.
DL(s)    == sent[s] + TTL - CfgSkew
Holds(s) == held[s] /\ now < DL(s)
RenewDue(s) == now - sent[s] >= DueAfter

-----------------------------------------------------------------------------
(* The abstracted LeaseCoordinator.                                         *)
(* hExp already carries Skew: a row stamped at t is SEEN as live until      *)
(* t + TTL + Skew by a replica lagging by Skew.                             *)
HeadLive == now < hExp

\* lease_coordinator.cpp:365-370.  claimants is always 1 here: a contested
\* head is PublisherLease.tla's obligation, and can only ADD refusals.
Admits(lid)  == (~HeadLive) \/ (hLid = lid)
ForeignLive  == (hOwner = "F") /\ HeadLive
Quarantined(s) == now < qUntil[s]          \* catalog_writer.cpp:245

CanOk(lid)      == chUp /\ Admits(lid)
CanRefused(lid) == chUp /\ ~Admits(lid)
CanUnknown      == (~chUp) \/ AllowUnknown

\* storage_service.cpp:1697-1700, in the code's own evaluation order.
NewSince(s) == IF heSince[s] = 0 THEN now ELSE heSince[s]
LatchNow(s) == now - NewSince(s) >= LatchWin

\* The process is past its lease and its sink is writing into the spool.
Alive(t) == phase[t] \in {"sweep", "run", "failed"}

-----------------------------------------------------------------------------
(* Requests.                                                                *)

SetReq(s, kind, by, lid, endT, f) ==
  /\ busy'   = [busy   EXCEPT ![s] = kind]
  /\ who'    = [who    EXCEPT ![s] = by]
  /\ rLid'   = [rLid   EXCEPT ![s] = lid]
  /\ rStart' = [rStart EXCEPT ![s] = now]
  /\ rEnd'   = [rEnd   EXCEPT ![s] = endT]
  /\ forced' = [forced EXCEPT ![s] = f]

\* A request that would still be running at `cap` is cut at `cutAt` with
\* its outcome unknown (clickhouse_client.cpp:600-622).
Issue(s, kind, by, lid, cap, cutAt) ==
  \E d \in 0..MaxReq :
     IF now + d >= cap THEN SetReq(s, kind, by, lid, cutAt, TRUE)
                       ELSE SetReq(s, kind, by, lid, now + d, FALSE)

\* Under the lease: bounded by the lease deadline (LeaseScope, :114-149).
IssueL(s, kind, by, sn) ==
  Issue(s, kind, by, myLid[s], sn + TTL - CfgSkew, sn + TTL - CfgSkew)
\* A claim made without a lease: each request bounded by CB
\* (lease_coordinator.cpp:86-110).  sent_ns is taken at the issue (the head
\* read is given no time: the earliest send, the conservative choice for
\* the deadline), and the INSERT and the read-back follow, each within CB:
\* the claim is confirmed up to 2 x CB after the send, or cut at CB by one
\* request over its bound.
IssueC(s, by, lid) ==
  \E d \in 0..MaxReq :
     \/ d <= 2 * CB /\ SetReq(s, "claim", by, lid, now + d, FALSE)
     \/ d > CB      /\ SetReq(s, "claim", by, lid, now + CB, TRUE)

\* The request fields are cleared when it ends, so that states differing
\* only in a finished request's leftovers are one state.
EndReq(s) == /\ busy'   = [busy   EXCEPT ![s] = "idle"]
             /\ who'    = [who    EXCEPT ![s] = "lt"]
             /\ rLid'   = [rLid   EXCEPT ![s] = 0]
             /\ rStart' = [rStart EXCEPT ![s] = 0]
             /\ rEnd'   = [rEnd   EXCEPT ![s] = 0]
             /\ forced' = [forced EXCEPT ![s] = FALSE]

\* A fresh lease_id that no row and no variable will ever show -- a claim
\* refused, or one whose INSERT was never sent -- is handed back, so that
\* repeated refusals do not make every state distinct.  Only when nothing
\* was minted since (and never under ReuseLid, where the claim may not have
\* minted at all).
Unminted(lid) == IF ~ReuseLid /\ lidGen = lid + 1 THEN lid ELSE lidGen

LateWake(s, t) == \E late \in 0..MaxLate : wake' = [wake EXCEPT ![s] = t + late]

\* keep_lease()'s next_wake (:1401-1410): the renewal due time, clamped to
\* [1 ms, tick].  1 ms is "at once" at this grain.
NextWakeAt(sn) == IF sn + DueAfter > now THEN now + Min(Tick, sn + DueAfter - now)
                                          ELSE now

\* A stretch of the cycle that holds the lease lock: before each request
\* keep_lease_in_pass() renews if due (:1463-1474), otherwise the request
\* runs.  `sn` is the lease's send time as of now.
NextInStretch(s, sn) ==
  IF now - sn >= DueAfter THEN IssueL(s, "renew", "cy", sn)
                          ELSE IssueL(s, "stmt",  "cy", sn)

\* After a request of the cycle's stretch went through: the stretch ends,
\* or sends its next request at once.  With zero-time requests a longer
\* stretch adds nothing, so it is cut to one request there.
StretchGoesOn(s, sn) ==
  \/ EndReq(s)
  \/ MaxReq > 0 /\ NextInStretch(s, sn)

\* storage_service.cpp:1678-1711  lease_held_elsewhere().
HeldElsewhere(s) ==
  /\ nextClaim' = [nextClaim EXCEPT ![s] = now + Tick]
  /\ IF hOwner = s
       \* :1682-1696  refused_by_own_claims(): every claimant at the head is
       \* one of this coordinator's own claim rows (a claim whose request
       \* gave up but which the server still ran).  It restarts the refusal
       \* clock and is no rival.
       THEN /\ heSince' = [heSince EXCEPT ![s] = 0]
            /\ UNCHANGED phase
       ELSE /\ heSince' = [heSince EXCEPT ![s] = NewSince(s)]
            /\ phase'   = [phase   EXCEPT ![s] =
                             IF LatchNow(s) THEN "failed" ELSE @]

\* catalog_writer.cpp:268-274 quarantine(), and abandon_lease() (:326-330).
QuarantineAt(s, lid) ==
  /\ held'    = [held    EXCEPT ![s] = FALSE]
  /\ qUntil'  = [qUntil  EXCEPT ![s] = now + TTL]
  /\ dropped' = [dropped EXCEPT ![s] = lid]

\* An outcome-unknown lease INSERT may have landed.  add_write_caps
\* (lease_coordinator.cpp:112-143) gives the server the attempt's time as
\* max_execution_time, so it lands, if at all, by the time the client gave
\* up: its stamp is the send or the completion.
MayLand(s, lid) ==
  \/ UNCHANGED <<hOwner, hLid, hExp>>
  \/ /\ chUp /\ Admits(lid)
     /\ \E st \in {rStart[s], now} :
           hOwner' = s /\ hLid' = lid /\ hExp' = st + TTL + Skew

-----------------------------------------------------------------------------
(* storage_service.cpp:1620-1676  ensure_publisher_lease(), issuing side.   *)
(* Called from the lease thread (:1411-1415) and from every cycle (:580).   *)

EnsureLease(s, by) ==
  \/ \* :1627 latched; :1631-1637 quarantined: NO claim, not even a fresh
     \* lease_id; :1638-1639 the post-refusal / unsent-claim backoff.
     /\ \/ phase[s] = "failed"
        \/ Quarantined(s)
        \/ now < nextClaim[s]
     /\ IF by = "lt" THEN LateWake(s, now + Tick) ELSE UNCHANGED wake
     /\ UNCHANGED <<headVars, locVars, reqVars>>
  \/ \* :1643  writer_.acquire_lease(holder)
     /\ phase[s] # "failed" /\ ~Quarantined(s) /\ now >= nextClaim[s]
     /\ LET lid == IF ReuseLid /\ myLid[s] # 0
                     THEN myLid[s]        \* the COUNTERFACTUAL
                     ELSE lidGen          \* a fresh lease_id
        IN /\ lidGen' = IF lid = lidGen THEN lidGen + 1 ELSE lidGen
           /\ IssueC(s, by, lid)
     /\ UNCHANGED <<hOwner, hLid, hExp, locVars, wake>>

-----------------------------------------------------------------------------
(* storage_service.cpp:1380-1436  the lease thread.                         *)

LeaseThreadTick(s) ==
  /\ phase[s] \in {"run", "sweep"}
  /\ busy[s] = "idle" /\ now >= wake[s]
  /\ IF held[s] /\ now >= DL(s)
       THEN \* LeaseScope entry (:122): abandon_lease_if_expired, which
            \* quarantines (catalog_writer.cpp:326-330); then
            \* ensure_publisher_lease() finds the writer quarantined.
            /\ QuarantineAt(s, myLid[s])
            /\ lost' = [lost EXCEPT ![s] = TRUE]
            /\ LateWake(s, now + Tick)
            /\ UNCHANGED <<headVars, myLid, sent, heSince, nextClaim, phase,
                           reacq, reqVars>>
       ELSE IF ~held[s]
         THEN /\ EnsureLease(s, "lt")                          \* :1411-1415
              /\ UNCHANGED lost
         ELSE IF RenewDue(s)
           THEN /\ IssueL(s, "renew", "lt", sent[s])            \* :1458-1459
                /\ UNCHANGED <<headVars, locVars, wake, lost>>
           ELSE /\ LateWake(s, NextWakeAt(sent[s]))             \* not due
                /\ UNCHANGED <<headVars, locVars, reqVars, lost>>
  /\ UNCHANGED <<envVars, cwake, swept, runBroken, startWaited, selfRef,
                 coSweep, stVars, rivVars>>

\* :1395-1396  the thread returns for good once the service has latched.
LeaseThreadExit(s) ==
  /\ phase[s] = "failed" /\ wake[s] # Never /\ busy[s] = "idle"
  /\ wake'  = [wake  EXCEPT ![s] = Never]
  /\ cwake' = [cwake EXCEPT ![s] = Never]
  /\ UNCHANGED <<envVars, headVars, locVars, reqVars, swept, runBroken,
                 startWaited, selfRef, coSweep, lost, stVars, rivVars>>

-----------------------------------------------------------------------------
(* The cycle's lease-locked work: ensure_publisher_lease() (:576-581) and   *)
(* the stretches of index_bounded() and the chunk loop.  A PUBLISH renews   *)
(* first, unconditionally (renew_for_publish, catalog_writer.cpp:510); any  *)
(* other stretch renews only when the renewal is due (keep_lease_in_pass).  *)

CycleTick(s) ==
  /\ CycleOn /\ phase[s] = "run"
  /\ busy[s] = "idle" /\ now >= cwake[s]
  /\ cwake' = [cwake EXCEPT ![s] = now + 1]
  /\ IF held[s] /\ now >= DL(s)
       THEN \* LeaseScope entry: abandon_lease_if_expired (:122).
            /\ QuarantineAt(s, myLid[s])
            /\ lost' = [lost EXCEPT ![s] = TRUE]
            /\ UNCHANGED <<headVars, myLid, sent, heSince, nextClaim, phase,
                           reacq, reqVars>>
       ELSE IF ~held[s]
         THEN EnsureLease(s, "cy") /\ UNCHANGED lost
         ELSE /\ \/ \* a cycle with nothing to index
                    UNCHANGED reqVars
                 \/ \* a publish: renew_for_publish first
                    IssueL(s, "renew", "cy", sent[s])
                 \/ \* a stretch that does not publish
                    AllowSkipPublish /\ NextInStretch(s, sent[s])
              /\ UNCHANGED <<headVars, locVars, lost>>
  /\ UNCHANGED <<envVars, wake, swept, runBroken, startWaited, selfRef,
                 coSweep, stVars, rivVars>>

-----------------------------------------------------------------------------
(* Completions.                                                             *)

\* A renewal: the lease thread's (:1458-1459), keep_lease_in_pass's, or a
\* publish's renew_for_publish.  All three present the HELD lease_id
\* (lease_coordinator.cpp:157-168) and quarantine on any std::exception
\* (catalog_writer.cpp:284-294).
CompleteRenew(s) ==
  /\ busy[s] = "renew" /\ now = rEnd[s]
  /\ LET lid == myLid[s] IN
     \/ \* confirmed: the new lease's send time is the INSERT's
        /\ ~forced[s] /\ CanOk(lid)
        /\ \E st \in {rStart[s], now} :
              hOwner' = s /\ hLid' = lid /\ hExp' = st + TTL + Skew
        /\ sent' = [sent EXCEPT ![s] = rStart[s]]
        /\ UNCHANGED <<lidGen, held, myLid, qUntil, heSince, nextClaim,
                       phase, reacq, dropped, lost>>
        /\ IF who[s] = "lt"
             THEN EndReq(s) /\ LateWake(s, NextWakeAt(rStart[s]))
             ELSE StretchGoesOn(s, rStart[s]) /\ UNCHANGED wake
     \/ \* kHeld: reject_live reset the lease (lease_coordinator.cpp:370).
        \* Only the lease thread reports it to lease_held_elsewhere
        \* (:1422-1425); a stretch's refusal fails the pass (:1467).
        /\ ~forced[s] /\ CanRefused(lid)
        /\ held' = [held EXCEPT ![s] = FALSE]
        /\ IF who[s] = "lt"
             THEN HeldElsewhere(s) /\ LateWake(s, now + Tick)
             ELSE UNCHANGED <<heSince, nextClaim, phase, wake>>
        /\ EndReq(s)
        /\ UNCHANGED <<headVars, myLid, sent, qUntil, reacq, dropped, lost>>
     \/ \* cut at the deadline, or failed: quarantine at once
        /\ forced[s] \/ CanUnknown
        /\ MayLand(s, lid) /\ UNCHANGED lidGen
        /\ QuarantineAt(s, lid)
        /\ lost' = [lost EXCEPT ![s] = @ \/ forced[s]]
        /\ IF who[s] = "lt" THEN LateWake(s, now + Tick) ELSE UNCHANGED wake
        /\ EndReq(s)
        /\ UNCHANGED <<myLid, sent, heSince, nextClaim, phase, reacq>>
  /\ UNCHANGED <<envVars, cwake, swept, runBroken, startWaited, selfRef,
                 coSweep, stVars, rivVars>>

\* Any other request a stretch makes under the lease: a read, a replay
\* guard, a fenced statement.  It does not touch the lease row.  Cut at the
\* deadline, it fails the stretch and the lease goes: a publish statement
\* quarantines (catalog_writer.cpp:386-391, :449-454), and a read leaves the
\* LeaseScope past the deadline, which abandons (:130).
CompleteStmt(s) ==
  /\ busy[s] = "stmt" /\ now = rEnd[s]
  /\ IF ~forced[s]
       THEN /\ StretchGoesOn(s, sent[s])
            /\ UNCHANGED <<locVars, lost>>
       ELSE /\ QuarantineAt(s, myLid[s])
            /\ lost' = [lost EXCEPT ![s] = TRUE]
            /\ EndReq(s)
            /\ UNCHANGED <<myLid, sent, heSince, nextClaim, phase, reacq>>
  /\ UNCHANGED <<envVars, headVars, wake, cwake, swept, runBroken,
                 startWaited, selfRef, coSweep, stVars, rivVars>>

\* storage_service.cpp:1559-1612, a refusal or a failed claim at start.
StartRefused(s) ==
  IF StartWait = 0 \/ now >= startDL[s]                       \* :1563-1572
    THEN /\ phase' = [phase EXCEPT ![s] = "refused"]
         /\ UNCHANGED <<wake, startWaited>>
    ELSE /\ UNCHANGED phase
         /\ wake' = [wake EXCEPT ![s] = now + Min(StartPoll, startDL[s] - now)]
         /\ startWaited' = [startWaited EXCEPT ![s] = TRUE]

\* :1574-1611  a claim at start that failed with a ClickHouseError.  Only a
\* TIMEOUT is retried; one whose INSERT may have landed quarantined the
\* writer, and the wait is stretched -- once -- to cover the quarantine and
\* a claim after it.
StartFailed(s, ins, timedOut) ==
  IF ~timedOut \/ StartWait = 0
    THEN /\ phase' = [phase EXCEPT ![s] = "refused"]
         /\ UNCHANGED <<wake, startDL, stretched>>
    ELSE LET q      == now + TTL
             grow   == ins /\ ~stretched[s] /\ q + 3 * CB + StartPoll > startDL[s]
             newDL  == IF grow THEN q + 3 * CB + StartPoll ELSE startDL[s]
             resume == IF ins THEN Max(now, q) ELSE now
         IN /\ startDL'   = [startDL   EXCEPT ![s] = newDL]
            /\ stretched' = [stretched EXCEPT ![s] = @ \/ grow]
            /\ IF resume >= newDL
                 THEN /\ phase' = [phase EXCEPT ![s] = "refused"]
                      /\ UNCHANGED wake
                 ELSE /\ UNCHANGED phase
                      /\ wake' = [wake EXCEPT ![s] = resume]

\* A claim: the lease thread's or the cycle's ensure_publisher_lease, or
\* start()'s acquire_lease_at_start.
CompleteClaim(s) ==
  /\ busy[s] = "claim" /\ now = rEnd[s]
  /\ LET lid == rLid[s] IN
     \/ \* admitted.  sent_ns is taken just before the INSERT (:219)
        /\ ~forced[s] /\ CanOk(lid)
        /\ \E st \in {rStart[s], now} :
              hOwner' = s /\ hLid' = lid /\ hExp' = st + TTL + Skew
        /\ held'  = [held  EXCEPT ![s] = TRUE]
        /\ myLid' = [myLid EXCEPT ![s] = lid]
        /\ sent'  = [sent  EXCEPT ![s] = rStart[s]]
        /\ IF who[s] = "st"
             THEN \* start(): the lease thread starts now, its first wait a
                  \* tick (:1386), and the sweep follows (:288-289).
                  /\ phase' = [phase EXCEPT ![s] = "sweep"]
                  /\ LateWake(s, now + Tick)
                  /\ cwake' = [cwake EXCEPT ![s] = now + 1]
                  /\ UNCHANGED <<heSince, nextClaim, reacq>>
             ELSE /\ heSince'   = [heSince   EXCEPT ![s] = 0]     \* :1663
                  /\ nextClaim' = [nextClaim EXCEPT ![s] = 0]     \* :1664
                  /\ reacq'     = [reacq     EXCEPT ![s] =
                                     IF @ < MaxReacq THEN @ + 1 ELSE @]
                  /\ UNCHANGED phase
                  /\ IF who[s] = "lt" THEN LateWake(s, NextWakeAt(rStart[s]))
                                      ELSE UNCHANGED wake
                  /\ UNCHANGED cwake
        /\ UNCHANGED <<lidGen, qUntil, dropped, selfRef, startWaited, stVars>>
     \/ \* kHeld.  The step is the claim's refusal: was it by a head row
        \* this service itself wrote?
        /\ ~forced[s] /\ CanRefused(lid)
        /\ selfRef' = (selfRef \/ hOwner = s)
        /\ IF who[s] = "st"
             THEN StartRefused(s) /\ UNCHANGED <<heSince, nextClaim>>
             ELSE /\ HeldElsewhere(s)
                  /\ UNCHANGED startWaited
                  /\ IF who[s] = "lt" THEN LateWake(s, now + Tick)
                                      ELSE UNCHANGED wake
        /\ lidGen' = Unminted(lid)
        /\ UNCHANGED <<hOwner, hLid, hExp, held, myLid, sent, qUntil, reacq,
                       dropped, cwake, stVars>>
     \/ \* failed.  `ins`: the INSERT may have reached the server
        \* (claim_insert_sent); only then does the writer quarantine
        \* (catalog_writer.cpp:315).
        /\ forced[s] \/ CanUnknown
        /\ \E ins \in BOOLEAN :
             /\ IF ins THEN /\ MayLand(s, lid) /\ QuarantineAt(s, lid)
                            /\ UNCHANGED lidGen
                       ELSE /\ lidGen' = Unminted(lid)
                            /\ UNCHANGED <<hOwner, hLid, hExp, held, qUntil,
                                           dropped>>
             /\ IF who[s] = "st"
                  THEN \* a timeout, unless the connection was refused
                       /\ StartFailed(s, ins, forced[s] \/ chUp)
                       /\ startUnk' = [startUnk EXCEPT ![s] = Min(@ + 1, 2)]
                       /\ UNCHANGED <<nextClaim, spoolLk>>
                  ELSE \* :1648-1662 an unsent claim backs off a tick
                       /\ nextClaim' = IF ins THEN nextClaim
                                       ELSE [nextClaim EXCEPT ![s] = now + Tick]
                       /\ IF who[s] = "lt" THEN LateWake(s, now + Tick)
                                           ELSE UNCHANGED wake
                       /\ UNCHANGED <<phase, stVars>>
        /\ UNCHANGED <<myLid, sent, heSince, reacq, selfRef, startWaited,
                       cwake>>
  /\ EndReq(s)
  /\ UNCHANGED <<envVars, swept, runBroken, coSweep, lost, rivVars>>

-----------------------------------------------------------------------------
(* Construction and start().                                                *)

\* storage_service.cpp:213-221: the constructor opens the spool, which
\* takes the directory's owner lock; another LIVE process holding it makes
\* Open() return kOwned and the constructor throw.
TakeLock(s) ==
  /\ phase[s] = "init" /\ (LateStart \/ now = 0)
  /\ IF SameSpool /\ OwnerLock /\ spoolLk # "none"
       THEN /\ phase' = [phase EXCEPT ![s] = "nolock"]
            /\ UNCHANGED <<spoolLk, startDL>>
       ELSE /\ phase'   = [phase EXCEPT ![s] = "start"]
            /\ spoolLk' = IF SameSpool /\ OwnerLock THEN s ELSE spoolLk
            /\ startDL' = [startDL EXCEPT ![s] = now + StartWait]
  /\ UNCHANGED <<envVars, headVars, held, myLid, sent, qUntil, heSince,
                 nextClaim, reacq, dropped, reqVars, resVars, stretched,
                 startUnk, rivVars>>

\* acquire_lease_at_start's loop issues a claim (:1558-1560).
StartBegin(s) ==
  /\ phase[s] = "start" /\ busy[s] = "idle"
  /\ (wake[s] = Never \/ now >= wake[s])
  /\ ~Quarantined(s)   \* resume = max(now, quarantine_until): never inside
  /\ lidGen' = lidGen + 1
  /\ IssueC(s, "st", lidGen)
  /\ UNCHANGED <<envVars, hOwner, hLid, hExp, locVars, resVars, stVars,
                 rivVars>>

\* storage_service.cpp:321-337  the sweep, AFTER the lease and never before.
Sweep(s) ==
  /\ phase[s] = "sweep"
  /\ swept' = [swept EXCEPT ![s] = TRUE]
  /\ phase' = [phase EXCEPT ![s] = "run"]
  \* O5: was ANOTHER live process writing into THIS spool when it ran?
  /\ coSweep' = (coSweep \/ (SameSpool /\ \E t \in Services \ {s} : Alive(t)))
  /\ UNCHANGED <<envVars, headVars, held, myLid, sent, qUntil, heSince,
                 nextClaim, reacq, dropped, reqVars, wake, cwake, runBroken,
                 startWaited, selfRef, lost, stVars, rivVars>>

\* A process dies: no tombstone, so its row stays live to its TTL; the
\* kernel drops its flock.
Crash(s) ==
  /\ AllowCrash /\ phase[s] \in {"start", "sweep", "run", "failed"}
  /\ phase' = [phase EXCEPT ![s] = "crashed"]
  /\ held'  = [held  EXCEPT ![s] = FALSE]
  /\ busy'  = [busy  EXCEPT ![s] = "idle"]
  /\ wake'  = [wake  EXCEPT ![s] = Never]
  /\ cwake' = [cwake EXCEPT ![s] = Never]
  /\ spoolLk' = IF spoolLk = s THEN "none" ELSE spoolLk
  /\ UNCHANGED <<envVars, headVars, myLid, sent, qUntil, heSince, nextClaim,
                 reacq, dropped, who, rLid, rStart, rEnd, forced, swept,
                 runBroken, startWaited, selfRef, coSweep, lost, startDL,
                 stretched, startUnk, rivVars>>

\* storage_service.cpp:372-413  stop(): the release goes through a
\* LeaseScope, which abandons a lease past its deadline first; a
\* quarantined or abandoned writer writes no tombstone.
Stop(s) ==
  /\ AllowStop /\ phase[s] \in {"run", "failed"} /\ busy[s] = "idle"
  /\ phase' = [phase EXCEPT ![s] = "stopped"]
  /\ wake'  = [wake  EXCEPT ![s] = Never]
  /\ cwake' = [cwake EXCEPT ![s] = Never]
  /\ IF Holds(s) /\ chUp
       THEN hExp' = now /\ UNCHANGED <<hOwner, hLid, lidGen>>
       ELSE UNCHANGED headVars
  /\ held' = [held EXCEPT ![s] = FALSE]
  /\ spoolLk' = IF spoolLk = s THEN "none" ELSE spoolLk
  /\ UNCHANGED <<envVars, myLid, sent, qUntil, heSince, nextClaim, reacq,
                 dropped, reqVars, swept, runBroken, startWaited, selfRef,
                 coSweep, lost, startDL, stretched, startUnk, rivVars>>

-----------------------------------------------------------------------------
(* The rival publisher: another publisher on the same catalog, obeying the  *)
(* same contract and renewing on the same ttl/3 schedule.  It may read a    *)
(* replica that sees rows dead RivalEarly ticks early.                      *)

RivalClaim ==
  /\ Foreign /\ ~fHeld /\ fBudget > 0 /\ chUp /\ now + RivalEarly >= hExp
  /\ hOwner' = "F" /\ hLid' = lidGen /\ hExp' = now + TTL + Skew
  /\ lidGen' = lidGen + 1
  /\ fLid' = lidGen /\ fHeld' = TRUE /\ fBudget' = fBudget - 1
  /\ fWake' = now + DueAfter
  /\ UNCHANGED <<envVars, locVars, reqVars, resVars, stVars>>

RivalRenew ==
  /\ fHeld /\ chUp /\ now = fWake /\ Admits(fLid) /\ now < ForeignStopBy
  /\ hOwner' = "F" /\ hLid' = fLid /\ hExp' = now + TTL + Skew
  /\ UNCHANGED lidGen
  /\ fWake' = now + DueAfter
  /\ UNCHANGED <<envVars, locVars, reqVars, resVars, stVars, fLid, fBudget,
                 fHeld>>

RivalLost ==
  /\ fHeld /\ chUp /\ now = fWake /\ (~Admits(fLid) \/ now >= ForeignStopBy)
  /\ fHeld' = FALSE /\ fWake' = Never
  /\ UNCHANGED <<envVars, headVars, locVars, reqVars, resVars, stVars, fLid,
                 fBudget>>

RivalStop ==
  /\ ForeignStops /\ fHeld /\ chUp
  /\ hExp' = now /\ UNCHANGED <<hOwner, hLid, lidGen>>
  /\ fHeld' = FALSE /\ fWake' = Never
  /\ UNCHANGED <<envVars, locVars, reqVars, resVars, stVars, fLid, fBudget>>

-----------------------------------------------------------------------------
Cut     == /\ chUp /\ cuts < MaxCuts
           /\ chUp' = FALSE /\ cuts' = cuts + 1
           /\ UNCHANGED <<now, headVars, locVars, reqVars, resVars, stVars,
                          rivVars>>
Restore == /\ ~chUp /\ chUp' = TRUE
           /\ UNCHANGED <<now, cuts, headVars, locVars, reqVars, resVars,
                          stVars, rivVars>>

-----------------------------------------------------------------------------
(* Time.  A discrete-event clock: it only advances when nothing is due at   *)
(* the current instant, so every scheduled wake and completion is served.   *)

Due(s) ==
  \/ phase[s] = "init" /\ ~LateStart
  \/ busy[s] # "idle" /\ now >= rEnd[s]
  \/ busy[s] = "idle" /\ phase[s] \in {"run", "sweep"} /\ now >= wake[s]
  \/ busy[s] = "idle" /\ phase[s] = "run" /\ CycleOn /\ now >= cwake[s]
  \/ busy[s] = "idle" /\ phase[s] = "start"
       /\ (wake[s] = Never \/ now >= wake[s])
  \/ phase[s] = "sweep"
  \/ phase[s] = "failed" /\ wake[s] # Never /\ busy[s] = "idle"

TimeTick ==
  /\ now < MaxTime
  /\ \A s \in Services : ~Due(s)
  /\ ~(fHeld /\ now = fWake)
  /\ now' = now + 1
  \* History for O3: since the latch clock started, was there an instant
  \* at which NO foreign publisher held a live row?
  /\ runBroken' = [s \in Services |->
        runBroken[s] \/ (heSince[s] # 0 /\ ~ForeignLive)]
  /\ UNCHANGED <<chUp, cuts, headVars, locVars, reqVars, wake, cwake, swept,
                 startWaited, selfRef, coSweep, lost, stVars, rivVars>>

-----------------------------------------------------------------------------
Init ==
  /\ now = 0  /\ chUp = TRUE  /\ cuts = 0  /\ lidGen = 1
  /\ hOwner = IF PredHolds THEN "P" ELSE "none"
  /\ hLid   = 0
  /\ hExp   = IF PredHolds THEN PredTTL + Skew ELSE 0
  /\ held      = [s \in Services |-> FALSE]
  /\ myLid     = [s \in Services |-> 0]
  /\ sent      = [s \in Services |-> 0]
  /\ qUntil    = [s \in Services |-> 0]
  /\ heSince   = [s \in Services |-> 0]
  /\ nextClaim = [s \in Services |-> 0]
  /\ phase     = [s \in Services |-> "init"]
  /\ reacq     = [s \in Services |-> 0]
  /\ dropped   = [s \in Services |-> 0]
  /\ busy      = [s \in Services |-> "idle"]
  /\ who       = [s \in Services |-> "lt"]
  /\ rLid      = [s \in Services |-> 0]
  /\ rStart    = [s \in Services |-> 0]
  /\ rEnd      = [s \in Services |-> 0]
  /\ forced    = [s \in Services |-> FALSE]
  /\ wake      = [s \in Services |-> Never]
  /\ cwake     = [s \in Services |-> Never]
  /\ swept     = [s \in Services |-> FALSE]
  /\ runBroken = [s \in Services |-> FALSE]
  /\ startWaited = [s \in Services |-> FALSE]
  /\ selfRef   = FALSE
  /\ coSweep   = FALSE
  /\ lost      = [s \in Services |-> FALSE]
  /\ startDL   = [s \in Services |-> StartWait]
  /\ stretched = [s \in Services |-> FALSE]
  /\ startUnk  = [s \in Services |-> 0]
  /\ spoolLk   = "none"
  /\ fWake = Never  /\ fLid = 0
  /\ fBudget = IF Foreign THEN ForeignBudget ELSE 0
  /\ fHeld = FALSE

Next ==
  \/ \E s \in Services : LeaseThreadTick(s)
  \/ \E s \in Services : LeaseThreadExit(s)
  \/ \E s \in Services : CycleTick(s)
  \/ \E s \in Services : CompleteRenew(s)
  \/ \E s \in Services : CompleteStmt(s)
  \/ \E s \in Services : CompleteClaim(s)
  \/ \E s \in Services : TakeLock(s)
  \/ \E s \in Services : StartBegin(s)
  \/ \E s \in Services : Sweep(s)
  \/ \E s \in Services : Crash(s)
  \/ \E s \in Services : Stop(s)
  \/ RivalClaim \/ RivalRenew \/ RivalLost \/ RivalStop
  \/ Cut \/ Restore
  \/ TimeTick

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(*                            THE OBLIGATIONS                               *)
-----------------------------------------------------------------------------

TypeOK ==
  /\ now \in 0..MaxTime
  /\ hOwner \in Owners
  /\ spoolLk \in Services \cup {"none"}
  /\ \A s \in Services :
       /\ phase[s] \in {"init", "nolock", "start", "sweep", "run", "failed",
                        "refused", "stopped", "crashed"}
       /\ busy[s] \in {"idle", "renew", "stmt", "claim"}

\* --- O1  renewal keeps the lease alive -----------------------------------
\* O1a.  A lease the service may still USE -- held, and its deadline ahead
\* -- is the live head.  Since #159 a lease past its deadline is not used:
\* every request under it is cut at the deadline or not sent
\* (clickhouse_client.cpp:600-622), and LeaseScope abandons it on the way
\* in and out (storage_service.cpp:122, :130).
NoPhantomLease ==
  \A s \in Services :
     (Holds(s) /\ phase[s] \in {"run", "sweep"}) => (hLid = myLid[s] /\ HeadLive)

\* O1a'.  The pre-#159 reading: the service never even BELIEVES it holds a
\* row that is not the live head.  Expected refuted under OS lateness: the
\* belief outlives the deadline until the next LeaseScope boundary, though
\* nothing is sent under it.
NoPhantomBelief ==
  \A s \in Services :
     (held[s] /\ phase[s] \in {"run", "sweep"}) => (hLid = myLid[s] /\ HeadLive)

\* O1b.  No request in flight under a lease outlives its deadline.
NoRequestPastDeadline ==
  \A s \in Services :
     (busy[s] \in {"renew", "stmt"} /\ held[s]) => rEnd[s] <= DL(s)

\* O1b'.  storage_service.cpp:1442-1446 and lease_coordinator.h:66-82: "with
\* the lease lock free a renewal starts within half the TTL of that send".
\* Checked for the lease thread's renewals.
RenewalStartsByHalf ==
  \A s \in Services :
     (busy[s] = "renew" /\ who[s] = "lt") => rStart[s] - sent[s] <= TTL \div 2

\* O1c.  A slow but HEALTHY catalog never costs the lease: no request was
\* cut at the deadline and no lease abandoned at it.  How slow is too slow
\* is what the O1_slow* configs measure.
NoDeadlineLoss == \A s \in Services : ~lost[s]

\* O1d.  Is a renewal failure ABSORBED?  (Expected REFUTED.)
OneFailureAbsorbed ==
  \A s \in Services : (phase[s] = "run" /\ Quarantined(s)) => held[s]

\* --- O2  quarantine ------------------------------------------------------
QuarantineTakesNothing ==
  \A s \in Services : Quarantined(s) => ~held[s]

\* Nobody but the head owner can USE a lease, and the rival never holds a
\* row a service can still use.
NoConcurrentHolder ==
  /\ \A s \in Services : Holds(s) => (hOwner = s /\ hLid = myLid[s] /\ HeadLive)
  /\ fHeld => (hOwner = "F" \/ ~HeadLive)

\* Checked for a service that can still claim: a start that failed has
\* thrown, and its window no longer matters.
QuarantineOutlastsItsRow ==
  \A s \in Services :
     (phase[s] \in {"start", "sweep", "run"} /\ qUntil[s] > 0
      /\ now >= qUntil[s] /\ dropped[s] # 0)
        => ~(hLid = dropped[s] /\ HeadLive)

NoSelfRefusal == ~selfRef

\* --- O3  the 2 x TTL latch ----------------------------------------------
NoFalsePositiveLatch ==
  \A s \in Services : (phase[s] = "failed") => ~runBroken[s]

NeverLatches == \A s \in Services : phase[s] # "failed"

\* --- O4  the start wait --------------------------------------------------
StartAlwaysSucceeds == \A s \in Services : phase[s] # "refused"

\* #159 (:1574-1611): ONE timed-out claim at start does not fail start(),
\* whether or not its INSERT was sent.
StartPastOneSlowClaim ==
  \A s \in Services : (phase[s] = "refused") => startUnk[s] >= 2

\* --- O5  spool ordering --------------------------------------------------
RefusedStartNeverSweeps ==
  \A s \in Services : (phase[s] = "refused") => ~swept[s]

\* No sweep while another live process writes into the same spool.  Since
\* #163 the owner lock is what guarantees it, not the lease.
SweepOnlyWhenAlone == ~coSweep

\* --- vacuity guards.  EVERY ONE OF THESE MUST BE REFUTED. ---------------
VacQuarantine  == \A s \in Services : ~Quarantined(s)
VacLatch       == \A s \in Services : phase[s] # "failed"
VacRecovered   == \A s \in Services : ~(reacq[s] > 0 /\ held[s])
VacRefusal     == \A s \in Services : heSince[s] = 0
VacStartWaited == \A s \in Services : ~startWaited[s]
VacRivalHeld   == ~fHeld
VacCut         == chUp
VacHeld        == \A s \in Services : ~held[s]
VacCoSweep     == ~coSweep
VacLost        == \A s \in Services : ~lost[s]
VacNoLock      == \A s \in Services : phase[s] # "nolock"
VacStartUnk    == \A s \in Services :
                    ~(startUnk[s] > 0 /\ phase[s] \in {"sweep", "run"})
VacOwnRefusal  == ~selfRef
VacSecondRunsAfterCrash ==
  ~(\E s, t \in Services : phase[s] = "crashed" /\ phase[t] = "run")

\* A state CONSTRAINT for the start-only configs: a state past start() is
\* checked, not expanded.
StartPhaseOnly == \A s \in Services : phase[s] \in {"init", "start"}

-----------------------------------------------------------------------------
(*        VERDICTS at 5b3b632  (TLC rev 94d0c50, 2026-10-06)            *)
(*                                                                          *)
(*  config               invariant                verdict   was (99ee4ae)   *)
(*  -------------------  -----------------------  --------  -------------   *)
(*  O1_clean/cut/late    NoPhantomLease           HOLDS     holds           *)
(*  O1_slowreq3/slowreq  NoPhantomLease (late 3/4) HOLDS    holds/REFUTED   *)
(*  O1_slowreq_req       NoPhantomLease +3 (req 4) HOLDS    (new)           *)
(*  O1_slowreq_cycle     the same, with stretches HOLDS     (new)           *)
(*  O1_belief            NoPhantomBelief          REFUTED   (new) reporting *)
(*  O1_skip              NoPhantomLease           HOLDS     REFUTED         *)
(*  O1_absorb            OneFailureAbsorbed       REFUTED   refuted         *)
(*  O1_window            RenewalStartsByHalf      HOLDS     (new)           *)
(*  O1_window_slowclaim  RenewalStartsByHalf      REFUTED   (new) FINDING   *)
(*  O1_fast_lt (req 2)   NoDeadlineLoss           HOLDS     (new)           *)
(*  O1_slow_lt (req 3)   NoDeadlineLoss           REFUTED   (new)           *)
(*  O1_fast_cy (req 2)   NoDeadlineLoss           HOLDS     (new)           *)
(*  O1_slow_cy (req 3)   NoDeadlineLoss           REFUTED   (new)           *)
(*  O1_skew_cy           NoDeadlineLoss           REFUTED   (new)           *)
(*  O2_quar              QuarantineOutlastsItsRow HOLDS     holds           *)
(*  O2_skew/O2_reuse     QuarantineOutlastsItsRow REFUTED   refuted         *)
(*  O2_selfref           NoSelfRefusal            REFUTED   refuted         *)
(*  O3_rival             NeverLatches             REFUTED   refuted         *)
(*  O3_rivaljust         NoFalsePositiveLatch     HOLDS     holds           *)
(*  O3_stops2, O3_false  NeverLatches / NoFPLatch HOLDS     holds           *)
(*  O3_falsenocut        NoFalsePositiveLatch     HOLDS     holds           *)
(*  O3_selflatch(_latch) NoFPLatch / NeverLatches HOLDS     REFUTED         *)
(*  O3_unkrival          NoFalsePositiveLatch     REFUTED   (new) FINDING   *)
(*  O3_stops             NeverLatches             HOLDS     holds, VACUOUS  *)
(*  O4_same/O4_skew      StartAlwaysSucceeds      HOLDS     holds           *)
(*  O4_bigger            StartAlwaysSucceeds      REFUTED   refuted         *)
(*  O4_compose           NoConcurrentHolder       HOLDS     holds           *)
(*  O4_deadline_skew     NoConcurrentHolder       HOLDS     (new)           *)
(*  O4_deadline_noskew   NoConcurrentHolder       REFUTED   (new) precond.  *)
(*  O4_slowstart         StartPastOneSlowClaim    HOLDS     (new)           *)
(*  O4_slowstart_two     StartAlwaysSucceeds      REFUTED   (new)           *)
(*  O4_slowstart_pred    StartPastOneSlowClaim    REFUTED   (new) by design *)
(*  O5_refused           RefusedStartNeverSweeps  HOLDS     holds           *)
(*  O5_cosweep           SweepOnlyWhenAlone       HOLDS     REFUTED         *)
(*  O5_cosweep_nolock    SweepOnlyWhenAlone       REFUTED   (counterfactual)*)
(*  O5_crash             SweepOnlyWhenAlone +1    HOLDS     (new)           *)
(*                                                                          *)
(*  VACUITY GUARDS -- every one REFUTED, as it must be: vac_held, vac_quar, *)
(*  vac_recov, vac_refus, vac_rival, vac_cut, vac_latch, vac_start,         *)
(*  vac_o4waited, vac_refusedstart, vac_stops2refusal, vac_stops2rival,     *)
(*  vac_lost, vac_deadline_rival, vac_slowstart, vac_nolock, vac_crash,     *)
(*  vac_ownrefusal.  ONE GUARD HOLDS, AND CONDEMNS ITS RUN, as before:      *)
(*  vac_stopsrefusal, so O3_stops is vacuous; O3_stops2 replaces it.        *)
(*  O1_tries/tries5/tries12/tries60 are retired: the fixed-tick comment     *)
(*  they pinned is gone (the thread now wakes when the renewal falls due).  *)
-----------------------------------------------------------------------------
(*                                 LIMITS                                   *)
(*                                                                          *)
(* 1. The LeaseCoordinator is a contract, not a protocol (see SCOPE).       *)
(* 2. Time is TTL/6 per tick.  The deadline's 0.1 s margin is kept only as  *)
(*    strictness (a request must finish at a time < DL).  The start poll    *)
(*    is coarser than the real one.                                        *)
(* 3. A request's duration is chosen when it is issued, 0..MaxReq ticks.    *)
(*    A claim's head read takes no time; its INSERT and read-back together  *)
(*    take up to 2 x CB, or one of them is cut at CB.  A renewal's three    *)
(*    requests are one, bounded by the deadline.  The admission decision    *)
(*    is taken at completion.                                               *)
(* 4. An outcome-unknown lease INSERT lands, if at all, stamped at its      *)
(*    send or its completion.  add_write_caps now gives the server the      *)
(*    attempt's time as max_execution_time, which is what justifies this;   *)
(*    what the caps do not cover (the part commit can overrun the limit;    *)
(*    the server's clock starts on arrival) is not modelled.  The code's    *)
(*    answer to a late landing is refused_by_own_claims(), which IS.        *)
(* 5. Skew: rows are seen live Skew ticks longer by a lagging replica, and  *)
(*    a rival may see them dead RivalEarly ticks before that.               *)
(* 6. The cycle is reduced to its lease-locked work.  Uploads, backoff,     *)
(*    pending_index_ and the object-store reads (outside the lock since     *)
(*    #159) never touch the lease and are omitted.                          *)
(* 7. CatalogSchema::ensure()'s install lease at start() is not modelled.   *)
(* 8. Stop is never enabled in a shipped config; its tombstone overwrites   *)
(*    the head's expiry instantly.                                          *)
(* 9. Two services are two processes.  The owner lock is modelled as        *)
(*    exclusive per spool directory while the holder lives; its             *)
(*    implementation (flock, shared filesystems, fork) is spool.cpp's.      *)
=============================================================================
