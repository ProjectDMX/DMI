--------------------------- MODULE LeaseLifecycle ---------------------------
(***************************************************************************)
(* The LEASE LIFECYCLE LAYER of                                            *)
(*   native/csrc/catalog/storage_service.cpp                               *)
(* on projectdmx/dmi branch fix/lease-recovery @ 2b74d14 (783 lines).      *)
(*                                                                         *)
(* SCOPE.  This models the SERVICE's lease lifecycle -- the lease thread,  *)
(* the quarantine window, the 2 x TTL latch, the start wait and the spool  *)
(* sweep -- NOT the claim/read-back protocol underneath it.  The           *)
(* LeaseCoordinator is abstracted to its CONTRACT:                         *)
(*                                                                         *)
(*     a claim presenting lease id L is ADMITTED iff the head row is dead  *)
(*     or the head row IS L (lease_coordinator.cpp:220-247 reject_live),   *)
(*     REFUSED with kHeld otherwise, and may return an UNKNOWN outcome     *)
(*     when the request cannot be completed.                              *)
(*                                                                         *)
(* That contract -- three round trips, contested heads, the fence, the     *)
(* tombstone, replica staleness -- is discharged by PublisherLease.tla in  *)
(* this directory.  Anything this module says about the coordinator is     *)
(* only as strong as that discharge; see LIMITS at the foot of the file.   *)
(*                                                                         *)
(* SOURCE LINES (2b74d14).  Every action names the code it stands for.     *)
(*   storage_service.cpp                                                   *)
(*     :65-67   lease_tick_ns   = max(ttl/6, 10ms)      -> Tick            *)
(*     :104-175 start()         schema, lease, sweep, reconcile            *)
(*     :121-135 the spool sweep, AFTER the lease                           *)
(*     :177-207 stop()          release only if a lease is held            *)
(*     :278-400 run_cycle()     ensure_publisher_lease at :286             *)
(*     :433-437 "a publish renews the lease"                               *)
(*     :588-628 keep_lease()    the lease thread                           *)
(*     :630-640 renew_lease_if_due()                                       *)
(*     :642-670 acquire_lease_at_start()                                   *)
(*     :672-721 ensure_publisher_lease()                                   *)
(*     :723-739 lease_held_elsewhere()  the 2 x TTL latch                  *)
(*     :762-781 latch_failure()  permanent                                 *)
(*   catalog_writer.cpp                                                    *)
(*     :243-253 quarantine_in_force()  now < quarantine_until              *)
(*     :268-274 quarantine()     drop the lease, window = now + lease_ttl  *)
(*     :285-293 renew_for_publish()  std::exception  -> quarantine()       *)
(*     :296-310 acquire_lease()       std::exception  -> quarantine()      *)
(*   lease_coordinator.cpp                                                 *)
(*     :45-55   acquire()  reuses the HELD lease_id, else mints a fresh one *)
(*     :57-68   renew()    always presents the held lease_id               *)
(*     :70-81   release()  the tombstone: expiry := now                    *)
(*     :220-247 reject_live()                                              *)
(*   indexer.cpp:258  "if (!all_rows.empty() || !indexed.empty())" -- the   *)
(*                    publish is SKIPPED when every pack was already       *)
(*                    committed, yet storage_service.cpp:435 still treats  *)
(*                    skipped_packs > 0 as "a publish renews the lease".   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
  Services,          \* the CaptureStorageService instances
  TTL,               \* writer.lease_ttl_ns, in TICKS.  6 keeps ttl/6 and
                     \* ttl/3 exact integers, which is what the code divides by
  PT,                \* publish_timeout_ns: the statement cap
  Skew,              \* clock_skew: ticks of extra life a row is seen to have
                     \* on a LAGGING replica (lease_coordinator.cpp:222)
  PredTTL,           \* the TTL a crashed PREDECESSOR ran with.  The config
                     \* comment (native_capture.py:140-145) admits a larger
                     \* one can outlast the default start wait
  StartWait,         \* start_lease_wait_ns (storage_service.cpp:647)
  MaxTime,
  MaxLate,           \* ticks of OS lateness allowed on a lease-thread wake
  Foreign,           \* TRUE: a rival publisher may claim the catalog
  ForeignStops,      \* TRUE: the rival may stop, releasing with a tombstone
  ForeignBudget,     \* how many times the rival may take the catalog
  ForeignStopBy,     \* the rival renews only while now < this (so a
                     \* "rival that stops within two TTLs" can be pinned)
  MaxCuts,           \* how many ClickHouse cut/restore pairs are allowed
  AllowUnknown,      \* TRUE: a write to a LIVE ClickHouse may time out with
                     \* its outcome unknown (the branch's bounded timeouts)
  AllowSkipPublish,  \* TRUE: model indexer.cpp:258 -- an index pass whose
                     \* packs were all already committed returns
                     \* skipped_packs > 0 WITHOUT publishing, while
                     \* storage_service.cpp:435-437 bumps last_renew_ns_ anyway
  ReuseLid,          \* COUNTERFACTUAL for obligation 2: TRUE makes a
                     \* post-quarantine acquire present the DROPPED lease_id
                     \* instead of a fresh one
  CycleOn,           \* TRUE: model run_cycle()'s ensure_publisher_lease (:286)
  PredHolds,         \* TRUE: a crashed predecessor's row is live at Init
  AllowStop,         \* TRUE: a service may call stop()
  MaxReacq           \* cap on the re-acquisition counter (a state bound only)

VARIABLES
  now, chUp, cuts,
  hOwner, hLid, hExp, lidGen,          \* the lease table HEAD (abstracted)
  held, myLid, qUntil, lastRenew,      \* per service: the writer's lease state
  heSince, nextClaim,                  \* held_elsewhere_since_ns_, next_claim_ns_
  wake, cwake, phase, swept, reacq, dropped,
  runBroken, startWaited, selfRef, coSweep,      \* history variables
  fWake, fLid, fBudget, fHeld                    \* the rival publisher

envVars  == <<now, chUp, cuts>>
headVars == <<hOwner, hLid, hExp, lidGen>>
locVars  == <<held, myLid, qUntil, lastRenew, heSince, nextClaim, phase,
              reacq, dropped>>
resVars  == <<wake, cwake, swept, runBroken, startWaited, selfRef, coSweep>>
rivVars  == <<fWake, fLid, fBudget, fHeld>>
vars     == <<envVars, headVars, locVars, resVars, rivVars>>

-----------------------------------------------------------------------------
(* Derived constants, exactly as the code computes them.                    *)

\* storage_service.cpp:65-67  lease_tick_ns(ttl) = max(ttl/6, 10ms).  The
\* 10 ms floor only bites for a TTL under 60 ms, which the writer's own
\* config precondition (catalog_writer.cpp:148-158) already forbids.
Tick      == IF TTL \div 6 > 0 THEN TTL \div 6 ELSE 1
DueAfter  == TTL \div 3        \* storage_service.cpp:634
LatchWin  == 2 * TTL           \* storage_service.cpp:731
\* storage_service.cpp:648-649  clamp(ttl/10, 50ms, 500ms).  One tick is the
\* finest grain this model has and is COARSER than the real poll, so the
\* model can only under-report how promptly a start wait notices an expiry.
StartPoll == 1

Owners == Services \cup {"none", "F", "P"}
Never  == MaxTime + 99      \* a wake time that never arrives

-----------------------------------------------------------------------------
(* The abstracted LeaseCoordinator.                                         *)
(* hExp already carries Skew: a row written at t is SEEN as live until      *)
(* t + TTL + Skew by a replica lagging by Skew.  A tombstone                *)
(* (lease_coordinator.cpp:83-89) reads its own expiry and now_ns from the   *)
(* same replica, so it is dead at once and carries no Skew.                 *)
HeadLive == now < hExp

\* lease_coordinator.cpp:220-247.  claimants is always 1 here: a contested
\* head (two rows at one term) is PublisherLease.tla's obligation, and it
\* can only ADD refusals, never remove them.
Admits(lid)  == (~HeadLive) \/ (hLid = lid)
ForeignLive  == (hOwner = "F") /\ HeadLive
Quarantined(s) == now < qUntil[s]          \* catalog_writer.cpp:245

CanOk(lid)      == chUp /\ Admits(lid)
CanRefused(lid) == chUp /\ ~Admits(lid)
CanUnknown      == (~chUp) \/ AllowUnknown
CanLand(lid)    == chUp /\ AllowUnknown /\ Admits(lid)

\* storage_service.cpp:723-739, in the code's own evaluation order: :726
\* sets held_elsewhere_since_ns_ to now when it was 0, and only THEN does
\* :731 compare, so the FIRST refusal in a run never latches.
NewSince(s) == IF heSince[s] = 0 THEN now ELSE heSince[s]
LatchNow(s) == now - NewSince(s) >= LatchWin

-----------------------------------------------------------------------------
Init ==
  /\ now = 0  /\ chUp = TRUE  /\ cuts = 0  /\ lidGen = 1
  \* A crashed predecessor that ran with PredTTL and was renewed at t = 0:
  \* its row stays live with no tombstone (storage_service.cpp:188-190 -- a
  \* killed or quarantined holder writes none).
  /\ hOwner = IF PredHolds THEN "P" ELSE "none"
  /\ hLid   = 0
  /\ hExp   = IF PredHolds THEN PredTTL + Skew ELSE 0
  /\ held      = [s \in Services |-> FALSE]
  /\ myLid     = [s \in Services |-> 0]
  /\ qUntil    = [s \in Services |-> 0]
  /\ lastRenew = [s \in Services |-> 0]
  /\ heSince   = [s \in Services |-> 0]
  /\ nextClaim = [s \in Services |-> 0]
  /\ wake      = [s \in Services |-> Never]
  /\ cwake     = [s \in Services |-> Never]
  /\ phase     = [s \in Services |-> "start"]
  /\ swept     = [s \in Services |-> FALSE]
  /\ reacq     = [s \in Services |-> 0]
  /\ dropped   = [s \in Services |-> 0]   \* the lease id a quarantine dropped
  /\ runBroken = [s \in Services |-> FALSE]
  /\ startWaited = [s \in Services |-> FALSE]
  /\ selfRef   = FALSE
  /\ coSweep   = FALSE
  /\ fWake = Never  /\ fLid = 0
  /\ fBudget = IF Foreign THEN ForeignBudget ELSE 0
  /\ fHeld = FALSE

-----------------------------------------------------------------------------
(* storage_service.cpp:672-721  ensure_publisher_lease().                   *)
(* Called from the lease thread (:608) and from every cycle (:286).         *)
(* Constrains headVars and locVars only.                                    *)

EnsureLease(s) ==
  \/ \* :673 already holds one; :676 already failed; :678-684 quarantined,
     \* which takes NO claim at all, not even a fresh lease_id; :686 the
     \* post-refusal backoff.  All four are no-ops on the lease state.
     /\ \/ held[s]
        \/ phase[s] = "failed"
        \/ Quarantined(s)
        \/ now < nextClaim[s]
     /\ UNCHANGED <<headVars, locVars>>
  \/ \* :690  writer_.acquire_lease(holder)
     /\ ~held[s] /\ phase[s] # "failed"
     /\ ~Quarantined(s) /\ now >= nextClaim[s]
     /\ LET lid == IF ReuseLid /\ myLid[s] # 0
                     THEN myLid[s]        \* the COUNTERFACTUAL
                     ELSE lidGen          \* :688-689 a fresh lease_id
        IN \/ \* admitted  :707-713
              /\ CanOk(lid)
              /\ hOwner' = s /\ hLid' = lid /\ hExp' = now + TTL + Skew
              /\ lidGen' = IF lid = lidGen THEN lidGen + 1 ELSE lidGen
              /\ held'      = [held      EXCEPT ![s] = TRUE]
              /\ myLid'     = [myLid     EXCEPT ![s] = lid]
              /\ lastRenew' = [lastRenew EXCEPT ![s] = now]      \* :707
              /\ heSince'   = [heSince   EXCEPT ![s] = 0]        \* :708
              /\ nextClaim' = [nextClaim EXCEPT ![s] = 0]        \* :709
              /\ reacq'     = [reacq     EXCEPT ![s] =
                                 IF @ < MaxReacq THEN @ + 1 ELSE @]
              /\ UNCHANGED <<qUntil, phase, dropped>>
           \/ \* :692-693  refused as held -> lease_held_elsewhere()
              /\ CanRefused(lid)
              /\ heSince'   = [heSince   EXCEPT ![s] = NewSince(s)]
              /\ nextClaim' = [nextClaim EXCEPT ![s] = now + Tick]
              /\ phase'     = [phase     EXCEPT ![s] =
                                 IF LatchNow(s) THEN "failed" ELSE "run"]
              /\ UNCHANGED <<headVars, held, myLid, qUntil, lastRenew,
                             reacq, dropped>>
           \/ \* :700-706 unknown outcome -> catalog_writer.cpp:307 quarantine()
              /\ CanUnknown
              /\ \E landed \in {TRUE, FALSE} :
                   /\ landed => CanLand(lid)
                   /\ IF landed
                        THEN /\ hOwner' = s /\ hLid' = lid
                             /\ hExp' = now + TTL + Skew
                             /\ lidGen' = IF lid = lidGen THEN lidGen+1
                                                          ELSE lidGen
                        ELSE /\ lidGen' = IF lid = lidGen THEN lidGen+1
                                                          ELSE lidGen
                             /\ UNCHANGED <<hOwner, hLid, hExp>>
              /\ qUntil'  = [qUntil  EXCEPT ![s] = now + TTL]   \* writer:273
              /\ dropped' = [dropped EXCEPT ![s] = lid]
              /\ UNCHANGED <<held, myLid, lastRenew, heSince, nextClaim,
                             phase, reacq>>

-----------------------------------------------------------------------------
(* storage_service.cpp:630-640  renew_lease_if_due().                       *)

RenewIfDue(s) ==
  \/ \* :634  not due yet
     /\ now - lastRenew[s] < DueAfter
     /\ UNCHANGED <<headVars, locVars>>
  \/ /\ now - lastRenew[s] >= DueAfter
     /\ \/ \* :635-639  renewed.  lease_coordinator.cpp:67 presents the HELD id
           /\ CanOk(myLid[s])
           /\ hOwner' = s /\ hLid' = myLid[s] /\ hExp' = now + TTL + Skew
           /\ lastRenew' = [lastRenew EXCEPT ![s] = now]   \* :636
           /\ heSince'   = [heSince   EXCEPT ![s] = 0]     \* :637
           /\ UNCHANGED <<lidGen, held, myLid, qUntil, nextClaim, phase,
                          reacq, dropped>>
        \/ \* :613-617  kHeld from reject_live.  lease_coordinator.cpp:226
           \* resets lease_ BEFORE throwing, so the local lease is gone too.
           /\ CanRefused(myLid[s])
           /\ held'      = [held      EXCEPT ![s] = FALSE]
           /\ heSince'   = [heSince   EXCEPT ![s] = NewSince(s)]
           /\ nextClaim' = [nextClaim EXCEPT ![s] = now + Tick]
           /\ phase'     = [phase     EXCEPT ![s] =
                              IF LatchNow(s) THEN "failed" ELSE "run"]
           /\ UNCHANGED <<headVars, myLid, qUntil, lastRenew, reacq, dropped>>
        \/ \* :621-624 unknown outcome.  catalog_writer.cpp:285-292
           \* renew_for_publish(): ONE std::exception quarantines the writer
           \* and discards the lease.  There is no second try.
           /\ CanUnknown
           /\ \E landed \in {TRUE, FALSE} :
                /\ landed => CanLand(myLid[s])
                /\ IF landed
                     THEN /\ hOwner' = s /\ hLid' = myLid[s]
                          /\ hExp' = now + TTL + Skew /\ UNCHANGED lidGen
                     ELSE UNCHANGED headVars
           /\ held'    = [held    EXCEPT ![s] = FALSE]      \* discard_local_lease
           /\ qUntil'  = [qUntil  EXCEPT ![s] = now + TTL]  \* writer:273
           /\ dropped' = [dropped EXCEPT ![s] = myLid[s]]
           /\ UNCHANGED <<myLid, lastRenew, heSince, nextClaim, phase, reacq>>

-----------------------------------------------------------------------------
(* storage_service.cpp:596-627  the lease thread's body.                    *)

LeaseThreadTick(s) ==
  /\ phase[s] = "run"
  /\ now = wake[s]
  /\ \E late \in 0..MaxLate :
       wake' = [wake EXCEPT ![s] = now + Tick + late]
  /\ IF ~held[s] THEN EnsureLease(s) ELSE RenewIfDue(s)   \* :607-612
  /\ selfRef' = (selfRef \/ (~held[s] /\ hOwner = s /\ HeadLive
                             /\ chUp /\ ~Admits(myLid[s])))
  /\ UNCHANGED <<envVars, cwake, swept, runBroken, startWaited, coSweep,
                 rivVars>>

\* :603-604 the thread returns for good once the service has latched.
LeaseThreadExit(s) ==
  /\ phase[s] = "failed" /\ wake[s] # Never
  /\ wake'  = [wake  EXCEPT ![s] = Never]
  /\ cwake' = [cwake EXCEPT ![s] = Never]
  /\ UNCHANGED <<envVars, headVars, locVars, swept, runBroken, startWaited,
                 selfRef, coSweep, rivVars>>

-----------------------------------------------------------------------------
(* storage_service.cpp:278-400  the cycle loop, reduced to its two          *)
(* lease-relevant acts: ensure_publisher_lease() at :286, and the           *)
(* last_renew_ns_ bump at :433-437.                                         *)

CycleTick(s) ==
  /\ CycleOn
  /\ phase[s] = "run"
  /\ now = cwake[s]
  /\ cwake' = [cwake EXCEPT ![s] = now + 1]
  /\ \/ EnsureLease(s)                                          \* :286
     \/ \* :433-437  a publish that DID reach the catalog: the fenced
        \* statement renewed the row (catalog_writer.cpp:489) and :436
        \* records that.
        /\ held[s] /\ CanOk(myLid[s])
        /\ hOwner' = s /\ hLid' = myLid[s] /\ hExp' = now + TTL + Skew
        /\ lastRenew' = [lastRenew EXCEPT ![s] = now]
        /\ UNCHANGED <<lidGen, held, myLid, qUntil, heSince, nextClaim,
                       phase, reacq, dropped>>
     \/ \* :435 + indexer.cpp:258 -- every pack in the batch was already
        \* committed, so `all_rows` and `indexed` are both empty, the
        \* publish block is SKIPPED and renew_for_publish() is never
        \* called.  index() still returns skipped_packs > 0, and
        \* :435-437 bumps last_renew_ns_ as if the row had been renewed.
        /\ AllowSkipPublish /\ held[s] /\ chUp
        /\ lastRenew' = [lastRenew EXCEPT ![s] = now]
        /\ UNCHANGED <<headVars, held, myLid, qUntil, heSince, nextClaim,
                       phase, reacq, dropped>>
  /\ UNCHANGED <<envVars, wake, swept, runBroken, startWaited, selfRef,
                 coSweep, rivVars>>

-----------------------------------------------------------------------------
(* storage_service.cpp:104-175  start().                                    *)

StartClaim(s) ==
  /\ phase[s] = "start"
  /\ (wake[s] = Never \/ now = wake[s])
  /\ LET lid == lidGen IN
     \/ \* :652-654 acquired
        /\ CanOk(lid)
        /\ hOwner' = s /\ hLid' = lid /\ hExp' = now + TTL + Skew
        /\ lidGen' = lidGen + 1
        /\ held'      = [held      EXCEPT ![s] = TRUE]
        /\ myLid'     = [myLid     EXCEPT ![s] = lid]
        /\ lastRenew' = [lastRenew EXCEPT ![s] = now]      \* :653
        /\ phase'     = [phase     EXCEPT ![s] = "sweep"]
        /\ wake'      = [wake      EXCEPT ![s] = now + Tick]
        /\ cwake'     = [cwake     EXCEPT ![s] = now + 1]
        /\ UNCHANGED <<qUntil, heSince, nextClaim, reacq, dropped, startWaited>>
     \/ \* :655-667 refused; retry every poll until the deadline, then throw
        /\ CanRefused(lid)
        /\ IF now >= StartWait                                  \* :658
             THEN /\ phase' = [phase EXCEPT ![s] = "refused"]   \* :659-664
                  /\ UNCHANGED <<wake, startWaited>>
             ELSE /\ UNCHANGED phase
                  /\ wake' = [wake EXCEPT ![s] = now + StartPoll]   \* :666
                  /\ startWaited' = [startWaited EXCEPT ![s] = TRUE]
        /\ UNCHANGED <<headVars, held, myLid, qUntil, lastRenew, heSince,
                       nextClaim, reacq, dropped, cwake>>
     \/ \* a ClickHouse error at start is NOT a lease refusal (:656), so it
        \* propagates and start() fails outright.
        /\ CanUnknown
        /\ phase' = [phase EXCEPT ![s] = "refused"]
        /\ UNCHANGED <<headVars, held, myLid, qUntil, lastRenew, heSince,
                       nextClaim, reacq, dropped, wake, cwake, startWaited>>
  /\ UNCHANGED <<envVars, swept, runBroken, selfRef, coSweep, rivVars>>

\* storage_service.cpp:121-135  the sweep, AFTER the lease and never before.
Sweep(s) ==
  /\ phase[s] = "sweep"
  /\ swept' = [swept EXCEPT ![s] = TRUE]
  /\ phase' = [phase EXCEPT ![s] = "run"]
  \* Obligation 5: was ANOTHER service live (its sink writing into a spool)
  \* when this sweep ran?  storage_service.cpp:115-120 admits this is only
  \* "usually" prevented.
  /\ coSweep' = (coSweep \/ \E t \in Services \ {s} : phase[t] = "run")
  /\ UNCHANGED <<envVars, headVars, held, myLid, qUntil, lastRenew, heSince,
                 nextClaim, reacq, dropped, wake, cwake, runBroken,
                 startWaited, selfRef, rivVars>>

\* storage_service.cpp:177-207  stop().  A quarantined writer holds no lease,
\* so it writes NO tombstone (:188-190) and its row stays live to its TTL.
Stop(s) ==
  /\ AllowStop /\ phase[s] \in {"run", "failed"}
  /\ phase' = [phase EXCEPT ![s] = "stopped"]
  /\ wake'  = [wake  EXCEPT ![s] = Never]
  /\ cwake' = [cwake EXCEPT ![s] = Never]
  /\ IF held[s] /\ chUp
       THEN /\ hExp' = now /\ UNCHANGED <<hOwner, hLid, lidGen>>  \* :195
       ELSE UNCHANGED headVars
  /\ held' = [held EXCEPT ![s] = FALSE]
  /\ UNCHANGED <<envVars, myLid, qUntil, lastRenew, heSince, nextClaim,
                 reacq, dropped, swept, runBroken, startWaited, selfRef,
                 coSweep, rivVars>>

-----------------------------------------------------------------------------
(* The rival publisher: another CaptureStorageService on the same catalog.  *)
(* It obeys the same contract, and renews on the same ttl/3 schedule.       *)

RivalClaim ==
  /\ Foreign /\ ~fHeld /\ fBudget > 0 /\ chUp /\ ~HeadLive
  /\ hOwner' = "F" /\ hLid' = lidGen /\ hExp' = now + TTL + Skew
  /\ lidGen' = lidGen + 1
  /\ fLid' = lidGen /\ fHeld' = TRUE /\ fBudget' = fBudget - 1
  /\ fWake' = now + DueAfter
  /\ UNCHANGED <<envVars, locVars, resVars>>

RivalRenew ==
  /\ fHeld /\ chUp /\ now = fWake /\ Admits(fLid) /\ now < ForeignStopBy
  /\ hOwner' = "F" /\ hLid' = fLid /\ hExp' = now + TTL + Skew
  /\ UNCHANGED lidGen
  /\ fWake' = now + DueAfter
  /\ UNCHANGED <<envVars, locVars, resVars, fLid, fBudget, fHeld>>

\* Renewal refused: the rival lost the head and gives up (it has its own
\* lifecycle, which this model does not need).
RivalLost ==
  /\ fHeld /\ chUp /\ now = fWake /\ (~Admits(fLid) \/ now >= ForeignStopBy)
  /\ fHeld' = FALSE /\ fWake' = Never
  /\ UNCHANGED <<envVars, headVars, locVars, resVars, fLid, fBudget>>

\* stop() with a tombstone (lease_coordinator.cpp:83-89): expiry := now.
RivalStop ==
  /\ ForeignStops /\ fHeld /\ chUp
  /\ hExp' = now /\ UNCHANGED <<hOwner, hLid, lidGen>>
  /\ fHeld' = FALSE /\ fWake' = Never
  /\ UNCHANGED <<envVars, locVars, resVars, fLid, fBudget>>

-----------------------------------------------------------------------------
Cut     == /\ chUp /\ cuts < MaxCuts
           /\ chUp' = FALSE /\ cuts' = cuts + 1
           /\ UNCHANGED <<now, headVars, locVars, resVars, rivVars>>
Restore == /\ ~chUp /\ chUp' = TRUE
           /\ UNCHANGED <<now, cuts, headVars, locVars, resVars, rivVars>>

-----------------------------------------------------------------------------
(* Time.  A discrete-event clock: it only advances when nothing is due at   *)
(* the current instant, so every scheduled wake is served.                  *)

Due(s) ==
  \/ (phase[s] = "run"   /\ now = wake[s])
  \/ (phase[s] = "run"   /\ CycleOn /\ now = cwake[s])
  \/ (phase[s] = "start" /\ (wake[s] = Never \/ now = wake[s]))
  \/ (phase[s] = "sweep")
  \/ (phase[s] = "failed" /\ wake[s] # Never)

TimeTick ==
  /\ now < MaxTime
  /\ \A s \in Services : ~Due(s)
  /\ ~(fHeld /\ now = fWake)
  /\ now' = now + 1
  \* History for obligation 3: since the latch clock started, was there an
  \* instant at which NO foreign publisher held a live row?  If so, the
  \* "refusal that has lasted 2 x TTL" (storage_service.cpp:728-730) did
  \* not in fact last.
  /\ runBroken' = [s \in Services |->
        runBroken[s] \/ (heSince[s] # 0 /\ ~ForeignLive)]
  /\ UNCHANGED <<chUp, cuts, headVars, locVars, wake, cwake, swept,
                 startWaited, selfRef, coSweep, rivVars>>

-----------------------------------------------------------------------------
Next ==
  \/ \E s \in Services : LeaseThreadTick(s)
  \/ \E s \in Services : LeaseThreadExit(s)
  \/ \E s \in Services : CycleTick(s)
  \/ \E s \in Services : StartClaim(s)
  \/ \E s \in Services : Sweep(s)
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
  /\ \A s \in Services : phase[s] \in
       {"start","sweep","run","failed","refused","stopped"}

\* --- O1  renewal keeps the lease alive -----------------------------------
\* O1a.  The service never BELIEVES it holds a lease whose row is not the
\* live head.  storage_service.cpp:588-593: "so neither the cycle backoff
\* nor a slow upload can let the lease lapse while the service still runs."
NoPhantomLease ==
  \A s \in Services :
     (held[s] /\ phase[s] \in {"run","sweep"}) => (hLid = myLid[s] /\ HeadLive)

\* O1b.  storage_service.cpp:631-632: "Renew once a third of the TTL has
\* passed without a publish, which leaves two more tries before a rival
\* could claim it."  Read literally: between the instant the renewal falls
\* due (last_renew + ttl/3) and the instant the row dies (last_renew + ttl),
\* at least THREE lease-thread wakes fall -- the due one and two more.
\* Pure arithmetic over the tick grid, quantified over every phase offset.
TriesInWindow(off) ==
  Cardinality({k \in 0..(3 * TTL) : /\ k * Tick + off >= DueAfter
                                    /\ k * Tick + off <  TTL})
ThreeTriesFit == \A off \in 1..Tick : TriesInWindow(off) >= 3
FourTriesFit  == \A off \in 1..Tick : TriesInWindow(off) >= 4
FiveTriesFit  == \A off \in 1..Tick : TriesInWindow(off) >= 5

\* O1c.  Is a renewal failure ABSORBED -- does the service still hold its
\* lease after one?  (catalog_writer.cpp:285-292 quarantines on the first
\* std::exception, so this is expected to be REFUTED.)
OneFailureAbsorbed ==
  \A s \in Services : (phase[s] = "run" /\ Quarantined(s)) => held[s]

\* --- O2  quarantine ------------------------------------------------------
\* storage_service.cpp:678-684 -- a quarantined writer takes no claim at all.
QuarantineTakesNothing ==
  \A s \in Services : Quarantined(s) => ~held[s]

\* Nobody but the head owner believes it holds a lease.
NoConcurrentHolder ==
  /\ \A s \in Services : held[s] => (hOwner = s /\ hLid = myLid[s] /\ HeadLive)
  /\ fHeld => (hOwner = "F" \/ ~HeadLive)

\* storage_service.cpp:679-681: "no claim ... until the window (one TTL) has
\* passed AND THAT LEASE'S ROW HAS EXPIRED WITH IT".  Checked directly: at
\* every instant at or after the window's end, the row the quarantine
\* dropped is dead.
QuarantineOutlastsItsRow ==
  \A s \in Services :
     (qUntil[s] > 0 /\ now >= qUntil[s] /\ dropped[s] # 0)
        => ~(hLid = dropped[s] /\ HeadLive)

\* Does a writer ever get refused by its OWN dropped row?  (The price of
\* the fresh-lease_id rule: reject_live's claimants==1 exemption at
\* lease_coordinator.cpp:223 cannot recognise a fresh id.)
NoSelfRefusal == ~selfRef

\* --- O3  the 2 x TTL latch ----------------------------------------------
\* storage_service.cpp:728-730: "our own dropped row is dead within one TTL
\* of the loss, and a handover ends sooner still.  A refusal that has lasted
\* 2 x TTL is a publisher that means to stay."  The latch is justified only
\* if the refusals really did LAST -- a foreign publisher held a live row at
\* every instant of the window.
NoFalsePositiveLatch ==
  \A s \in Services : (phase[s] = "failed") => ~runBroken[s]

NeverLatches == \A s \in Services : phase[s] # "failed"

\* --- O4  the start wait --------------------------------------------------
StartAlwaysSucceeds == \A s \in Services : phase[s] # "refused"

\* --- O5  spool ordering --------------------------------------------------
\* test_a_start_refused_the_lease_leaves_the_spool_unswept
RefusedStartNeverSweeps ==
  \A s \in Services : (phase[s] = "refused") => ~swept[s]

\* storage_service.cpp:115-120, weakened by 2b74d14 from a guarantee to
\* "only usually".  The STRONG form, which the header (storage_service.h
\* :100-104) still states:
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

-----------------------------------------------------------------------------
(*                    VERDICTS  (TLC 2.19, Java 21, 4 workers)              *)
(* Verbatim output for every run is in LLruns/<config>.txt.                 *)
(*                                                                          *)
(*  config              invariant                 verdict     distinct      *)
(*  ------------------  ------------------------  ----------  ---------     *)
(*  O1_tries            ThreeTriesFit             HOLDS               9     *)
(*  O1_tries12/60       Three+FourTriesFit        HOLDS               6     *)
(*  O1_tries5           FiveTriesFit              REFUTED             -     *)
(*                      -> exactly FOUR wakes fall in the window            *)
(*  O1_clean            NoPhantomLease            HOLDS             126     *)
(*  O1_cut              NoPhantomLease            HOLDS           3,879     *)
(*  O1_late (MaxLate 1) NoPhantomLease            HOLDS           9,880     *)
(*  O1_absorb           OneFailureAbsorbed        REFUTED            90     *)
(*  O1_skip             NoPhantomLease            REFUTED           196     *)
(*  O2_quar             QuarantineOutlastsItsRow  HOLDS           3,879     *)
(*  O2_skew  (Skew 1)   QuarantineOutlastsItsRow  REFUTED           854     *)
(*  O2_selfref          NoSelfRefusal             REFUTED         1,001     *)
(*  O2_reuse            QuarantineOutlastsItsRow  REFUTED           765     *)
(*  O3_rival            NeverLatches              REFUTED        54,130     *)
(*  O3_rivaljust        NoFalsePositiveLatch      HOLDS          81,516     *)
(*                      -> a persistent rival latches, and legitimately     *)
(*  O3_stops2           NeverLatches              HOLDS         170,480     *)
(*  O3_false (Skew 0)   NoFalsePositiveLatch      HOLDS         170,480     *)
(*  O3_selflatch        NoFalsePositiveLatch      REFUTED         9,559     *)
(*  O3_selflatch_latch  NeverLatches              REFUTED        11,614     *)
(*                      -> a latch with NO rival in existence               *)
(*  O4_same             StartAlwaysSucceeds       HOLDS              66     *)
(*  O4_skew             StartAlwaysSucceeds       HOLDS              59     *)
(*  O4_bigger           StartAlwaysSucceeds       REFUTED            16     *)
(*  O4_compose          NoConcurrentHolder        HOLDS           5,872     *)
(*  O5_refused          RefusedStartNeverSweeps   HOLDS              14     *)
(*  O5_cosweep          SweepOnlyWhenAlone        REFUTED         2,653     *)
(*                                                                          *)
(*  VACUITY GUARDS -- every one REFUTED, as it must be:                     *)
(*    vac_held, vac_quar, vac_recov, vac_refus, vac_rival, vac_cut,         *)
(*    vac_latch, vac_start, vac_o4waited, vac_refusedstart, vac_cosweep,    *)
(*    vac_stops2refusal, vac_stops2rival.                                   *)
(*  ONE GUARD HELD, AND CONDEMNS ITS RUN: vac_stopsrefusal (VacRefusal)     *)
(*  HOLDS on LeaseLifecycle_O3_stops.cfg, so THAT run is vacuous -- with    *)
(*  no ClickHouse cut the rival can never claim.  O3_stops2.cfg replaces    *)
(*  it and is proved non-vacuous by vac_stops2refusal/vac_stops2rival.      *)
-----------------------------------------------------------------------------
(*                                 LIMITS                                   *)
(*                                                                          *)
(* 1. The LeaseCoordinator is a contract, not a protocol.  A contested head *)
(*    (two rows at one term, lease_coordinator.cpp:227-236) is omitted; it  *)
(*    can only ADD kHeld refusals, so every "this latches / this lapses"    *)
(*    refutation below is preserved under it, and every "this HOLDS"        *)
(*    verdict is conditional on PublisherLease.tla's discharge.             *)
(* 2. Time is TTL/6 per tick, so ttl/6, ttl/3 and 2*ttl are exact but       *)
(*    anything finer than a sixth of the TTL is invisible.  The real start  *)
(*    poll is ttl/10 clamped to 50-500 ms, FINER than one tick, so a start  *)
(*    the model reports as refused purely at an expiry boundary would be    *)
(*    retried sooner in reality.                                            *)
(* 3. MaxLate = 0 means the lease thread wakes exactly on its tick.  Real   *)
(*    OS scheduling can only make it LATER, which strictly reduces the      *)
(*    number of renewal attempts in the window; MaxLate = 1 is run          *)
(*    separately.                                                           *)
(* 4. The cycle loop is reduced to ensure_publisher_lease() plus the        *)
(*    last_renew_ns_ bump.  Uploads, backoff and pending_index_ never touch *)
(*    the lease and are omitted.                                            *)
(* 5. An outcome-unknown statement's own late landing is not modelled as a  *)
(*    separate actor: catalog_writer.cpp:519 caps it at publish_timeout,    *)
(*    which the writer's config precondition (:149) forces below the TTL,   *)
(*    so it is always dead before the quarantine window ends.  What IS      *)
(*    modelled is the row it may have left behind -- see                    *)
(*    QuarantineOutlastsItsRow.                                             *)
=============================================================================
