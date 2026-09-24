---------------------------- MODULE PayloadRing ----------------------------
(***************************************************************************)
(* GPU-to-host payload ring of ProjectDMX/DMI (native/csrc/ring, a987dfe). *)
(*                                                                         *)
(* Processes: the Python step thread (prepare_step / reserve_record /      *)
(* reserve_one safety net / final flush), the GPU producer stream, the     *)
(* drain thread, the p2p thread (staging release only) and one extra       *)
(* concurrent force_flush_and_wait caller ("Ext").                         *)
(*                                                                         *)
(* Units: one payload unit == PAYLOAD_ALIGN (16 B), so align_up is the     *)
(* identity.  All counters are the code's monotonic logical counters;      *)
(* physical index = counter % cap, so task-slot wraparound is modeled      *)
(* exactly and payload wraparound is modeled through the capacity check    *)
(* (the two-span split is byte arithmetic, payload_ring.cuh:291-311).      *)
(*                                                                         *)
(* Every mutex-protected region is ONE atomic action; the lock-free        *)
(* publication store (task_ring.cuh:422-436) and acquire load              *)
(* (drain_thread.cpp:26-30) are separate actions of different processes.   *)
(* No code path nests two of mu_/mgmt_mu_/queue_mu_/pop_mu_/staging_mu_    *)
(* (every lock scope in drain_thread.cpp / p2p_thread.cpp is block-local), *)
(* so there is no lock-order deadlock to model; condition-variable waits   *)
(* are guarded actions.                                                    *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    PCAP,          \* payload ring capacity (units)
    TCAP,          \* task ring entries (publication slots)
    SCAP,          \* pinned staging capacity (units)
    MaxTasks,      \* total hook occurrences planned in one run
    MaxPerStep,    \* max hooks/items per step
    MaxU,          \* max (upper-bound) bytes per item, units
    MaxExtFlush,   \* number of flush_and_wait calls from the Ext thread
    RECORD,        \* TRUE: record ring (reserve_record); FALSE: legacy ring
    SAFETY_NET,    \* legacy: model HookPoint.forward eager safety net
    MUT_NO_GEN,        \* mutation: completion releases every waiter
    MUT_EARLY_COMMIT,  \* mutation: committed tail advanced before D2H
    MUT_RECLAIM_OOO,   \* mutation: reclaim credited without FIFO sequence check
    MUT_UNDER_RESERVE, \* mutation: record item reserves < its upper bound
    MUT_LOST_WAKEUP,   \* mutation: waiter blocks without re-checking predicate
    MUT_GATE_SKIP      \* contract violation: gated-off producer that WAS reserved

ASSUME PCAP >= 1 /\ TCAP >= 1 /\ SCAP >= 1 /\ MaxU >= 1
ASSUME SAFETY_NET => (~RECORD /\ MaxU <= IF PCAP < SCAP THEN PCAP ELSE SCAP)

Min(a, b) == IF a < b THEN a ELSE b
EffCap    == Min(PCAP, SCAP)          \* ring_engine_py.cu:433-435, 494-496
Tasks     == 0..(MaxTasks - 1)
NoTask    == -1
EmptySlot == [t |-> NoTask, sz |-> 0]

VARIABLES
    \* ---- step thread (Python caller, GIL released) ----
    stPc, stItems, stIdx, stRet, stGen, stSnapP, stSnapT, stNotified,
    planned, nTasks,
    \* ---- per-task ghost/bookkeeping (index = CPU reservation order) ----
    tU, tRec, tRes, tAct, tState,
    \* ---- GPU stream and device-visible memory ----
    launched, executed, gpuPHead, gpuTHead, slots,
    \* ---- mgmt_mu_ state written by the step thread ----
    cpuPHead, cpuTHead,
    \* ---- drain thread (mgmt_mu_ state + private) ----
    dPc, dFull, dGen, visHead, scanned, cpuTTail, cpuPTail, committed,
    d2hDone, fCnt, fBytes,
    \* ---- record reclaim state (mgmt_mu_) + ghosts ----
    pendRec, pendRecBytes, recFail, rclCnt, rclAmt, applied,
    \* ---- flush generations (mu_) and Ext requester ----
    reqGen, compGen, extPc, extGen, extSnapP, extSnapT, extNotified, extN,
    \* ---- pinned staging (staging_mu_) and p2p queue (queue_mu_/pop_mu_) ----
    stgHead, stgTail, p2pQ,
    \* ---- reached error / property-violation markers ----
    errs

stVars   == <<stPc, stItems, stIdx, stRet, stGen, stSnapP, stSnapT, stNotified, planned, nTasks>>
taskVars == <<tU, tRec, tRes, tAct, tState>>
gpuVars  == <<launched, executed, gpuPHead, gpuTHead, slots>>
cpuVars  == <<cpuPHead, cpuTHead>>
drVars   == <<dPc, dFull, dGen, visHead, scanned, cpuTTail, cpuPTail, committed, d2hDone, fCnt, fBytes>>
recVars  == <<pendRec, pendRecBytes, recFail, rclCnt, rclAmt, applied>>
extVars  == <<extPc, extGen, extSnapP, extSnapT, extNotified, extN>>
genVars  == <<reqGen, compGen>>
stgVars  == <<stgHead, stgTail, p2pQ>>
vars == <<stVars, taskVars, gpuVars, cpuVars, drVars, recVars, extVars, genVars, stgVars, errs>>

-----------------------------------------------------------------------------
(* Helpers *)

ItemSet == [u : 1..MaxU, rec : IF RECORD THEN BOOLEAN ELSE {FALSE}]

\* Bytes the CPU reserves for one item.  Faithful: the aligned upper bound
\* (records.py:367-373 -> reserve_record ring_engine_py.cu:479-489).
ResAmt(it) == IF RECORD /\ it.rec /\ MUT_UNDER_RESERVE THEN it.u - 1 ELSE it.u

StepBytes(items) ==
    LET RECURSIVE S(_)
        S(i) == IF i > Len(items) THEN 0 ELSE ResAmt(items[i]) + S(i + 1)
    IN S(1)

SeqSum(sq) ==
    LET RECURSIVE S(_)
        S(i) == IF i > Len(sq) THEN 0 ELSE sq[i].sz + S(i + 1)
    IN S(1)

RECURSIVE FSum(_, _)
FSum(f, S) == IF S = {} THEN 0
              ELSE LET x == CHOOSE x \in S : TRUE IN f[x] + FSum(f, S \ {x})

\* Flush batch: longest prefix of scanned_ whose aligned bytes fit the staging
\* CAPACITY (drain_thread.cpp:348-353 and 432-437).
Batch(sc) ==
    LET RECURSIVE B(_, _, _)
        B(i, c, b) == IF i > Len(sc) \/ b + sc[i].sz > SCAP
                      THEN [cnt |-> c, bytes |-> b]
                      ELSE B(i + 1, c + 1, b + sc[i].sz)
    IN B(1, 0, 0)

\* account_record_task, drain_thread.cpp:533-562 (called under mgmt_mu_).
Account(s, sl) ==
    IF Len(s.pr) = 0 \/ s.rf THEN s                                   \* 535
    ELSE LET p == Head(s.pr) IN
      IF ~MUT_RECLAIM_OOO /\ s.vh < p.seq THEN s                       \* 538
      ELSE IF ~MUT_RECLAIM_OOO /\ s.vh # p.seq                         \* 539-543
        THEN [s EXCEPT !.rf = TRUE, !.err = @ \cup {"SeqMismatch"}]
      ELSE IF sl.sz > p.res                                            \* 545-551
        THEN [s EXCEPT !.rf = TRUE, !.err = @ \cup {"OverReservation"},
                       !.pr = Tail(@)]
      ELSE [s EXCEPT !.prb = @ + (p.res - sl.sz),                      \* 552-561
                     !.pr  = Tail(@),
                     !.rc  = [@ EXCEPT ![p.t] = @ + 1],
                     !.ra  = [@ EXCEPT ![p.t] = p.res - sl.sz]]

\* scan_ready, drain_thread.cpp:508-531: acquire-load slots in sequence
\* order while READY and fewer than task_cap entries are pending.  Doing the
\* whole loop atomically is equivalent to the real loop: a slot observed
\* READY stays READY until this drain clears it.
RECURSIVE ScanR(_)
ScanR(s) ==
    LET sl == slots[s.vh % TCAP] IN
    IF Len(s.sc) >= TCAP \/ sl.t = NoTask THEN s                     \* 512-516
    ELSE ScanR([Account(s, sl) EXCEPT !.vh = @ + 1, !.sc = Append(@, sl)])

ScanInit == [vh |-> visHead, sc |-> scanned, pr |-> pendRec, prb |-> pendRecBytes,
             rf |-> recFail, rc |-> rclCnt, ra |-> rclAmt, err |-> {}]

ApplyScan(s) ==
    /\ visHead' = s.vh /\ scanned' = s.sc /\ pendRec' = s.pr
    /\ pendRecBytes' = s.prb /\ recFail' = s.rf /\ rclCnt' = s.rc /\ rclAmt' = s.ra

ClearSlots(from, cnt) ==
    [i \in 0..(TCAP - 1) |->
        IF \E k \in from..(from + cnt - 1) : k % TCAP = i THEN EmptySlot ELSE slots[i]]

-----------------------------------------------------------------------------
Init ==
    /\ stPc = "plan" /\ stItems = <<>> /\ stIdx = 1 /\ stRet = "none"
    /\ stGen = 0 /\ stSnapP = 0 /\ stSnapT = 0 /\ stNotified = FALSE
    /\ planned = 0 /\ nTasks = 0
    /\ tU = [t \in Tasks |-> 0] /\ tRec = [t \in Tasks |-> FALSE]
    /\ tRes = [t \in Tasks |-> 0] /\ tAct = [t \in Tasks |-> -1]
    /\ tState = [t \in Tasks |-> "none"]
    /\ launched = 0 /\ executed = 0 /\ gpuPHead = 0 /\ gpuTHead = 0
    /\ slots = [i \in 0..(TCAP - 1) |-> EmptySlot]
    /\ cpuPHead = 0 /\ cpuTHead = 0
    /\ dPc = "loop" /\ dFull = FALSE /\ dGen = 0 /\ visHead = 0 /\ scanned = <<>>
    /\ cpuTTail = 0 /\ cpuPTail = 0 /\ committed = 0 /\ d2hDone = 0
    /\ fCnt = 0 /\ fBytes = 0
    /\ pendRec = <<>> /\ pendRecBytes = 0 /\ recFail = FALSE
    /\ rclCnt = [t \in Tasks |-> 0] /\ rclAmt = [t \in Tasks |-> 0] /\ applied = 0
    /\ reqGen = 0 /\ compGen = 0
    /\ extPc = "idle" /\ extGen = 0 /\ extSnapP = 0 /\ extSnapT = 0
    /\ extNotified = FALSE /\ extN = 0
    /\ stgHead = 0 /\ stgTail = 0 /\ p2pQ = <<>>
    /\ errs = {}

-----------------------------------------------------------------------------
(***************************** Step thread *********************************)

\* Python plans one step (plan_step / ProducerPlan).  Bounded by MaxTasks.
StPlan ==
    /\ stPc = "plan" /\ planned < MaxTasks
    /\ \E n \in 1..Min(MaxPerStep, MaxTasks - planned) :
         \E items \in [1..n -> ItemSet] :
            /\ stItems' = items
            /\ planned' = planned + n
            /\ stPc' = IF RECORD THEN "apply" ELSE "check"
    /\ UNCHANGED <<stIdx, stRet, stGen, stSnapP, stSnapT, stNotified, nTasks,
                   taskVars, gpuVars, cpuVars, drVars, recVars, extVars, genVars, stgVars, errs>>

\* End of run: flush_records_and_wait (ring_engine_py.cu:558-578) or
\* flush_and_wait (677-681).  Both sync the stream and force-flush.
StFinal ==
    /\ stPc = "plan" /\ planned = MaxTasks
    /\ stRet' = "final" /\ stPc' = "sync"
    /\ UNCHANGED <<stItems, stIdx, stGen, stSnapP, stSnapT, stNotified, planned, nTasks,
                   taskVars, gpuVars, cpuVars, drVars, recVars, extVars, genVars, stgVars, errs>>

\* reserve_record prologue, ring_engine_py.cu:499-501: rethrow drain/reclaim
\* failure, then apply_pending_record_reclaims (drain_thread.cpp:287-294).
StApply ==
    /\ stPc = "apply"
    /\ LET ex == pendRecBytes > cpuPHead IN
       /\ errs' = errs \cup (IF recFail THEN {"ReclaimFailureRaised"} ELSE {})
                       \cup (IF ex THEN {"ReclaimExceedsHead"} ELSE {})
       /\ cpuPHead' = IF ex THEN cpuPHead ELSE cpuPHead - pendRecBytes
       /\ applied' = IF ex THEN applied ELSE applied + pendRecBytes
       /\ pendRecBytes' = IF ex THEN pendRecBytes ELSE 0
    /\ stPc' = "check"
    /\ UNCHANGED <<stItems, stIdx, stRet, stGen, stSnapP, stSnapT, stNotified, planned, nTasks,
                   taskVars, gpuVars, cpuTHead, drVars, pendRec, recFail, rclCnt, rclAmt,
                   extVars, genVars, stgVars>>

\* Capacity decision: prepare_step ring_engine_py.cu:444-460 /
\* reserve_record 503-525.  The step thread is the only writer of the heads
\* and the drain only advances tails, so reading the four counters under
\* separate mgmt_mu_ acquisitions is equivalent to one atomic read.
StCheck ==
    /\ stPc = "check"
    /\ LET b == StepBytes(stItems)
           n == Len(stItems)
       IN IF b > EffCap \/ n > TCAP
          THEN stRet' = "oversized" /\ stPc' = "sync"
          ELSE IF b <= PCAP - (cpuPHead - committed) /\ n <= TCAP - (cpuTHead - cpuTTail)
               THEN stPc' = "reserve" /\ UNCHANGED stRet
               ELSE stRet' = "flushed" /\ stPc' = "sync"
    /\ UNCHANGED <<stItems, stIdx, stGen, stSnapP, stSnapT, stNotified, planned, nTasks,
                   taskVars, gpuVars, cpuVars, drVars, recVars, extVars, genVars, stgVars, errs>>

\* cudaStreamSynchronize(current stream): every launched producer has run.
StSync ==
    /\ stPc = "sync" /\ executed = launched
    /\ stPc' = "req"
    /\ UNCHANGED <<stItems, stIdx, stRet, stGen, stSnapP, stSnapT, stNotified, planned, nTasks,
                   taskVars, gpuVars, cpuVars, drVars, recVars, extVars, genVars, stgVars, errs>>

\* force_flush_and_wait request half, drain_thread.cpp:123-132 (mu_).
\* Snapshots of the device heads are ghosts for the PrematureRelease check.
StReq ==
    /\ stPc = "req"
    /\ reqGen' = reqGen + 1 /\ stGen' = reqGen + 1
    /\ stSnapP' = gpuPHead /\ stSnapT' = gpuTHead /\ stNotified' = FALSE
    /\ stPc' = "prewait"
    /\ UNCHANGED <<stItems, stIdx, stRet, planned, nTasks, compGen,
                   taskVars, gpuVars, cpuVars, drVars, recVars, extVars, stgVars, errs>>

StWakeTo ==
    CASE stRet = "flushed"   -> IF RECORD THEN "post" ELSE "reserve"
      [] stRet = "oversized" -> IF RECORD THEN "post" ELSE IF SAFETY_NET THEN "sn" ELSE "plan"
      [] stRet = "final"     -> IF RECORD THEN "post" ELSE "done"
      [] stRet = "snflush"   -> "snres"

\* Waiter returns (predicate compGen >= stGen held under mu_).
StWake ==
    /\ errs' = errs \cup (IF committed < stSnapP \/ cpuTTail < stSnapT
                          THEN {"PrematureRelease"} ELSE {})
    /\ stPc' = StWakeTo
    /\ stIdx' = IF stRet = "oversized" THEN 1 ELSE stIdx
    /\ UNCHANGED stNotified

\* Wait half, drain_thread.cpp:135-138: re-lock mu_, wait(lk, pred).
\* A predicate wait checks before blocking; MUT_LOST_WAKEUP blocks blindly.
StPreWait ==
    /\ stPc = "prewait"
    /\ IF ~MUT_LOST_WAKEUP /\ compGen >= stGen
       THEN StWake
       ELSE stPc' = "blk" /\ UNCHANGED <<stIdx, stNotified, errs>>
    /\ UNCHANGED <<stItems, stRet, stGen, stSnapP, stSnapT, planned, nTasks,
                   taskVars, gpuVars, cpuVars, drVars, recVars, extVars, genVars, stgVars>>

StBlk ==
    /\ stPc = "blk" /\ stNotified
    /\ IF compGen >= stGen
       THEN StWake
       ELSE stNotified' = FALSE /\ UNCHANGED <<stPc, stIdx, errs>>
    /\ UNCHANGED <<stItems, stRet, stGen, stSnapP, stSnapT, planned, nTasks,
                   taskVars, gpuVars, cpuVars, drVars, recVars, extVars, genVars, stgVars>>

\* Record post-flush checks: reserve_record 507-513 / 530-536,
\* flush_records_and_wait 572-578.
StPost ==
    /\ stPc = "post"
    /\ LET ex == pendRecBytes > cpuPHead IN
       /\ errs' = errs \cup (IF recFail THEN {"ReclaimFailureRaised"} ELSE {})
                       \cup (IF ex THEN {"ReclaimExceedsHead"} ELSE {})
                       \cup (IF Len(pendRec) # 0 THEN {"IncompleteReclaims"} ELSE {})
       /\ cpuPHead' = IF ex THEN cpuPHead ELSE cpuPHead - pendRecBytes
       /\ applied' = IF ex THEN applied ELSE applied + pendRecBytes
       /\ pendRecBytes' = IF ex THEN pendRecBytes ELSE 0
    /\ stPc' = CASE stRet = "flushed"   -> "reserve"
                 [] stRet = "oversized" -> "plan"
                 [] stRet = "final"     -> "done"
    /\ UNCHANGED <<stItems, stIdx, stRet, stGen, stSnapP, stSnapT, stNotified, planned, nTasks,
                   taskVars, gpuVars, cpuTHead, drVars, pendRec, recFail, rclCnt, rclAmt,
                   extVars, genVars, stgVars>>

\* DrainThread::reserve (drain_thread.cpp:256-260) / reserve_record (262-285)
\* under mgmt_mu_, followed by the forward pass launching one producer per
\* item on the single producer stream (launch folded in: the GPU action is
\* independent and may still be delayed arbitrarily).  Note the FLUSHED
\* paths reach here with no re-check (ring_engine_py.cu:467, 537).
StReserve ==
    /\ stPc = "reserve"
    /\ LET n    == Len(stItems)
           base == nTasks
           newRec == [i \in 1..n |-> [seq |-> cpuTHead + i - 1, res |-> ResAmt(stItems[i]),
                                      t |-> base + i - 1, rec |-> stItems[i].rec]]
           In(t) == t >= base /\ t < base + n
       IN /\ cpuPHead' = cpuPHead + StepBytes(stItems)
          /\ cpuTHead' = cpuTHead + n
          /\ pendRec' = pendRec \o SelectSeq(newRec, LAMBDA r : r.rec)
          /\ tU'   = [t \in Tasks |-> IF In(t) THEN stItems[t - base + 1].u ELSE tU[t]]
          /\ tRec' = [t \in Tasks |-> IF In(t) THEN stItems[t - base + 1].rec ELSE tRec[t]]
          /\ tRes' = [t \in Tasks |-> IF In(t) THEN ResAmt(stItems[t - base + 1]) ELSE tRes[t]]
          /\ tState' = [t \in Tasks |-> IF In(t) THEN "launched" ELSE tState[t]]
          /\ nTasks' = nTasks + n
          /\ launched' = launched + n
    /\ stPc' = "plan"
    /\ UNCHANGED <<stItems, stIdx, stRet, stGen, stSnapP, stSnapT, stNotified, planned,
                   tAct, executed, gpuPHead, gpuTHead, slots, drVars,
                   pendRecBytes, recFail, rclCnt, rclAmt, applied,
                   extVars, genVars, stgVars, errs>>

\* reserve_one, ring_engine_py.cu:667-671 -> DrainThread::reserve(nbytes, 1):
\* one payload reservation + ONE TASK, no task-capacity check.
ReserveOne(u) ==
    /\ cpuPHead' = cpuPHead + u
    /\ cpuTHead' = cpuTHead + 1
    /\ tU' = [tU EXCEPT ![nTasks] = u]
    /\ tRes' = [tRes EXCEPT ![nTasks] = u]
    /\ tState' = [tState EXCEPT ![nTasks] = "launched"]
    /\ nTasks' = nTasks + 1
    /\ launched' = launched + 1
    /\ stIdx' = stIdx + 1
    /\ UNCHANGED <<tRec, tAct>>

\* Legacy eager safety net for a force_eager step, src/dmi/hooks/point.py:
\* 321-324 (fits slack -> reserve_one + producer), 325-329 (fits after a
\* flush_and_wait -> flush, then reserve_one + producer).  The CPU-direct
\* branch (330-339) is unreachable here because MaxU <= EffCap.
StSN ==
    /\ stPc = "sn"
    /\ IF stIdx > Len(stItems)
       THEN stPc' = "plan"
            /\ UNCHANGED <<stIdx, stRet, cpuVars, taskVars, nTasks, launched>>
       ELSE LET u == stItems[stIdx].u
                avail == PCAP - (cpuPHead - committed)      \* available_capacity 658-662
            IN IF u <= Min(avail, EffCap)
               THEN ReserveOne(u) /\ UNCHANGED <<stPc, stRet>>
               ELSE stRet' = "snflush" /\ stPc' = "sync"
                    /\ UNCHANGED <<stIdx, cpuVars, taskVars, nTasks, launched>>
    /\ UNCHANGED <<stItems, stGen, stSnapP, stSnapT, stNotified, planned,
                   executed, gpuPHead, gpuTHead, slots, drVars, recVars,
                   extVars, genVars, stgVars, errs>>

StSNRes ==
    /\ stPc = "snres"
    /\ ReserveOne(stItems[stIdx].u)
    /\ stPc' = "sn"
    /\ UNCHANGED <<stItems, stRet, stGen, stSnapP, stSnapT, stNotified, planned,
                   executed, gpuPHead, gpuTHead, slots, drVars, recVars,
                   extVars, genVars, stgVars, errs>>

StepNext == StPlan \/ StFinal \/ StApply \/ StCheck \/ StSync \/ StReq \/ StPreWait
            \/ StBlk \/ StPost \/ StReserve \/ StSN \/ StSNRes

-----------------------------------------------------------------------------
(***************************** GPU producer ********************************)
(* One producer kernel runs at a time, in launch order (single-stream      *)
(* assumption, producer.cu:44-48).  The last block writes the payload at   *)
(* device payload_head, release-stores READY|actual into                   *)
(* slots[task_head % task_cap], then advances the device heads             *)
(* (producer.cu:128-172, task_ring.cuh:422-436).  The kernel never reads a *)
(* tail (ring_state.h:9-11), so we only RECORD whether it overwrote an     *)
(* unread slot or unread payload bytes.                                    *)
GpuRun ==
    /\ executed < launched
    /\ LET t   == executed
           idx == gpuTHead % TCAP
       IN \E a \in (IF tRec[t] THEN 0..tU[t] ELSE {tU[t]}) :
          \E skip \in (IF MUT_GATE_SKIP /\ tRec[t] THEN BOOLEAN ELSE {FALSE}) :
            IF skip
            THEN \* record_emit_allowed false (producer.cu:431): no publication
                 /\ tState' = [tState EXCEPT ![t] = "skipped"]
                 /\ executed' = executed + 1
                 /\ UNCHANGED <<tAct, gpuPHead, gpuTHead, slots, errs>>
            ELSE
                 /\ errs' = errs
                      \cup (IF slots[idx].t # NoTask THEN {"SlotOverwrite"} ELSE {})
                      \cup (IF gpuPHead + a - d2hDone > PCAP THEN {"PayloadOverwrite"} ELSE {})
                 /\ slots' = [slots EXCEPT ![idx] = [t |-> t, sz |-> a]]
                 /\ gpuPHead' = gpuPHead + a
                 /\ gpuTHead' = gpuTHead + 1
                 /\ tAct' = [tAct EXCEPT ![t] = a]
                 /\ tState' = [tState EXCEPT ![t] = "produced"]
                 /\ executed' = executed + 1
    /\ UNCHANGED <<stVars, tU, tRec, tRes, launched, cpuVars, drVars, recVars,
                   extVars, genVars, stgVars>>

-----------------------------------------------------------------------------
(***************************** Drain thread ********************************)

\* flush_state_update, drain_thread.cpp:594-601 (under mgmt_mu_): clear the
\* flushed publication slots, advance cpu_task_tail_ (this is what
\* cpu_task_tail_committed() returns, 247-250) and the PENDING payload tail.
FlushStateUpdate(b) ==
    /\ slots' = ClearSlots(cpuTTail, b.cnt)
    /\ cpuTTail' = cpuTTail + b.cnt
    /\ cpuPTail' = cpuPTail + b.bytes
    /\ fCnt' = b.cnt /\ fBytes' = b.bytes
    /\ committed' = IF MUT_EARLY_COMMIT THEN cpuPTail + b.bytes ELSE committed

\* Loop head with an outstanding flush generation, drain_thread.cpp:401-411.
DrainLoopForce ==
    /\ dPc = "loop" /\ reqGen > compGen
    /\ dGen' = reqGen /\ dFull' = TRUE /\ dPc' = "fullscan"
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, visHead, scanned, cpuTTail,
                   cpuPTail, committed, d2hDone, fCnt, fBytes, recVars, extVars,
                   genVars, stgVars, errs>>

\* Normal iteration, drain_thread.cpp:424-449: scan_ready + should_flush +
\* batch + flush_state_update in one mgmt_mu_ region.  should_flush is
\* over-approximated: it must flush at >= task_cap entries or >= payload_cap
\* bytes (574-575); every other threshold/timeout is a free choice.  Only
\* iterations that change state are modeled (the rest is the poll sleep).
DrainLoopNormal ==
    /\ dPc = "loop" /\ reqGen <= compGen
    /\ LET s      == ScanR(ScanInit)
           pend   == Len(s.sc)
           forced == pend >= TCAP \/ SeqSum(s.sc) >= PCAP
           b      == Batch(s.sc)
       IN \E doFlush \in (IF pend = 0 THEN {FALSE} ELSE IF forced THEN {TRUE} ELSE BOOLEAN) :
            LET flushing == doFlush /\ b.cnt > 0 IN
            /\ s.vh # visHead \/ flushing
            /\ ApplyScan(s)
            /\ errs' = errs \cup s.err
            /\ IF flushing
               THEN FlushStateUpdate(b) /\ dPc' = "stgwait"
               ELSE UNCHANGED <<slots, cpuTTail, cpuPTail, fCnt, fBytes, committed, dPc>>
    /\ UNCHANGED <<stVars, taskVars, launched, executed, gpuPHead, gpuTHead, cpuVars,
                   dFull, dGen, d2hDone, applied, extVars, genVars, stgVars>>

\* do_full_flush iteration head, drain_thread.cpp:344-356.
DrainFullScan ==
    /\ dPc = "fullscan"
    /\ LET s == ScanR(ScanInit)
           b == Batch(s.sc)
       IN /\ ApplyScan(s)
          /\ IF Len(s.sc) = 0                                        \* 347
             THEN errs' = errs \cup s.err /\ dPc' = "complete"
                  /\ UNCHANGED <<slots, cpuTTail, cpuPTail, fCnt, fBytes, committed>>
             ELSE IF b.cnt = 0                                        \* 354
             THEN errs' = errs \cup s.err \cup {"StagingStuck"} /\ dPc' = "complete"
                  /\ UNCHANGED <<slots, cpuTTail, cpuPTail, fCnt, fBytes, committed>>
             ELSE errs' = errs \cup s.err /\ FlushStateUpdate(b) /\ dPc' = "stgwait"
    /\ UNCHANGED <<stVars, taskVars, launched, executed, gpuPHead, gpuTHead, cpuVars,
                   dFull, dGen, d2hDone, applied, extVars, genVars, stgVars>>

\* staging_cv_.wait(free_bytes() >= flush_bytes), drain_thread.cpp:357-360 / 453-458.
DrainStg ==
    /\ dPc = "stgwait" /\ SCAP - (stgHead - stgTail) >= fBytes
    /\ dPc' = "d2h"
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, dFull, dGen, visHead, scanned,
                   cpuTTail, cpuPTail, committed, d2hDone, fCnt, fBytes, recVars,
                   extVars, genVars, stgVars, errs>>

\* enqueue_d2h + sync_stream (drain_thread.cpp:361-362 / 460-461, 603-639):
\* bytes [cpuPTail - fBytes, cpuPTail) are now safely in pinned staging.
DrainD2H ==
    /\ dPc = "d2h"
    /\ d2hDone' = cpuPTail
    /\ errs' = errs \cup (IF cpuPTail > gpuPHead THEN {"CopiedUnwritten"} ELSE {})
    /\ dPc' = "commit"
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, dFull, dGen, visHead, scanned,
                   cpuTTail, cpuPTail, committed, fCnt, fBytes, recVars,
                   extVars, genVars, stgVars>>

\* cpu_payload_tail_committed_ = cpu_payload_tail_ (mgmt_mu_, 363-366 / 463-466),
\* then submit_to_p2p (367 / 468; 645-687: queue_mu_, pop_mu_, staging head)
\* and trim_scanned (368-371 / 469-472; 693-704, mgmt_mu_).  Folding the
\* last two in is sound: they touch only drain-private and p2p state.
DrainCommit ==
    /\ dPc = "commit"
    /\ committed' = cpuPTail
    /\ p2pQ' = p2pQ \o [i \in 1..fCnt |-> scanned[i].sz]
    /\ stgHead' = stgHead + fBytes
    /\ scanned' = SubSeq(scanned, fCnt + 1, Len(scanned))
    /\ dPc' = IF dFull THEN "fullscan" ELSE "loop"
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, dFull, dGen, visHead,
                   cpuTTail, cpuPTail, d2hDone, fCnt, fBytes, recVars,
                   extVars, genVars, stgTail, errs>>

\* flush_completed_generation_ = flush_generation; notify_all (416-420).
\* MUT_NO_GEN: completion satisfies whatever has been requested by now.
DrainComplete ==
    /\ dPc = "complete"
    /\ compGen' = IF MUT_NO_GEN THEN reqGen ELSE dGen
    /\ stNotified' = (stNotified \/ stPc = "blk")
    /\ extNotified' = (extNotified \/ extPc = "blk")
    /\ dFull' = FALSE /\ dPc' = "loop"
    /\ UNCHANGED <<stPc, stItems, stIdx, stRet, stGen, stSnapP, stSnapT, planned, nTasks,
                   taskVars, gpuVars, cpuVars, dGen, visHead, scanned, cpuTTail, cpuPTail,
                   committed, d2hDone, fCnt, fBytes, recVars, extPc, extGen, extSnapP,
                   extSnapT, extN, reqGen, stgVars, errs>>

DrainNext == DrainLoopForce \/ DrainLoopNormal \/ DrainFullScan \/ DrainStg
             \/ DrainD2H \/ DrainCommit \/ DrainComplete

-----------------------------------------------------------------------------
(***************************** p2p thread **********************************)
\* wait_for_tasks / pop_tasks / notify_staging_freed_bytes
\* (drain_thread.cpp:193-227; p2p_thread.cpp:144-181, 353-404).  Single
\* consumer, so the staging tail is monotonic (p2p_thread.h:5-6).
P2PFree ==
    /\ Len(p2pQ) > 0
    /\ stgTail' = stgTail + Head(p2pQ)
    /\ p2pQ' = Tail(p2pQ)
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, drVars, recVars, extVars,
                   genVars, stgHead, errs>>

-----------------------------------------------------------------------------
(***************** Ext: concurrent force_flush_and_wait caller ***************)
ExtReq ==
    /\ extPc = "idle" /\ extN < MaxExtFlush
    /\ reqGen' = reqGen + 1 /\ extGen' = reqGen + 1
    /\ extSnapP' = gpuPHead /\ extSnapT' = gpuTHead /\ extNotified' = FALSE
    /\ extPc' = "prewait"
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, drVars, recVars, extN,
                   compGen, stgVars, errs>>

ExtWake ==
    /\ errs' = errs \cup (IF committed < extSnapP \/ cpuTTail < extSnapT
                          THEN {"PrematureRelease"} ELSE {})
    /\ extPc' = "idle" /\ extN' = extN + 1
    /\ UNCHANGED extNotified

ExtPreWait ==
    /\ extPc = "prewait"
    /\ IF ~MUT_LOST_WAKEUP /\ compGen >= extGen
       THEN ExtWake
       ELSE extPc' = "blk" /\ UNCHANGED <<extN, extNotified, errs>>
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, drVars, recVars, extGen,
                   extSnapP, extSnapT, genVars, stgVars>>

ExtBlk ==
    /\ extPc = "blk" /\ extNotified
    /\ IF compGen >= extGen
       THEN ExtWake
       ELSE extNotified' = FALSE /\ UNCHANGED <<extPc, extN, errs>>
    /\ UNCHANGED <<stVars, taskVars, gpuVars, cpuVars, drVars, recVars, extGen,
                   extSnapP, extSnapT, genVars, stgVars>>

ExtNext == ExtReq \/ ExtPreWait \/ ExtBlk

\* Run finished: allow stuttering so TLC's deadlock check only flags real
\* stuck states before the step thread is done.
Done == stPc = "done" /\ extPc = "idle" /\ UNCHANGED vars

Next == StepNext \/ GpuRun \/ DrainNext \/ P2PFree \/ ExtNext \/ Done

Fairness ==
    /\ WF_vars(StepNext)
    /\ WF_vars(GpuRun)
    /\ WF_vars(DrainLoopForce) /\ WF_vars(DrainFullScan) /\ WF_vars(DrainStg)
    /\ WF_vars(DrainD2H) /\ WF_vars(DrainCommit) /\ WF_vars(DrainComplete)
    /\ WF_vars(P2PFree)
    /\ WF_vars(ExtPreWait) /\ WF_vars(ExtBlk)
    \* No fairness on DrainLoopNormal (below-threshold entries may sit until a
    \* forced flush) nor on ExtReq (Ext may never call).

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(******************************* Properties *********************************)

\* reserved >= actual for every produced task.
ReservedGeActual == \A t \in Tasks : tState[t] = "produced" => tAct[t] <= tRes[t]

\* No slot is drained before its READY bit: every scanned entry came from a
\* published word of a produced task, and the drain never copies bytes the
\* producer has not written.
DrainAfterReady ==
    /\ \A i \in 1..Len(scanned) : tState[scanned[i].t] = "produced"
    /\ "CopiedUnwritten" \notin errs

\* Committed tail never passes an undrained (not yet D2H-completed) byte.
CommittedSafe == committed <= d2hDone /\ d2hDone <= cpuPTail /\ cpuPTail <= gpuPHead

\* Outstanding bytes/tasks never exceed capacity; GPU never overwrites
\* payload bytes or a publication word the drain has not consumed.
NoPayloadOverwrite == "PayloadOverwrite" \notin errs
NoSlotOverwrite    == "SlotOverwrite" \notin errs
CapacityRespected  ==
    /\ cpuPHead - committed <= PCAP
    /\ cpuTHead - cpuTTail <= TCAP
    /\ gpuPHead - committed <= PCAP

\* Reclaim accounting: each needs-reclaim task reclaimed at most once, only
\* after it was produced, by exactly reserved - actual; head bookkeeping
\* balances (cpu head = reservations - applied reclaims).
ReclaimOnce  == \A t \in Tasks : rclCnt[t] <= 1
ReclaimExact == \A t \in Tasks : rclCnt[t] = 1 =>
                    /\ tState[t] = "produced" /\ tRec[t]
                    /\ rclAmt[t] = tRes[t] - tAct[t]
HeadAccounting ==
    /\ cpuPHead + applied = FSum(tRes, 0..(nTasks - 1))
    /\ applied + pendRecBytes =
         FSum([t \in Tasks |-> IF rclCnt[t] >= 1 THEN rclAmt[t] ELSE 0], Tasks)

\* The code's hard-error paths are never reached.
HardErrorsUnreachable ==
    errs \cap {"SeqMismatch", "OverReservation", "IncompleteReclaims",
               "ReclaimExceedsHead", "ReclaimFailureRaised", "StagingStuck"} = {}

\* A completed flush never releases a waiter whose request it did not cover.
NoPrematureRelease == "PrematureRelease" \notin errs

\* End state: nothing leaked, every reclaim applied exactly once.
NoLeakAtEnd ==
    stPc = "done" =>
        /\ Len(pendRec) = 0 /\ pendRecBytes = 0
        /\ cpuPHead = gpuPHead /\ committed = gpuPHead
        /\ cpuTHead = cpuTTail /\ cpuTHead = gpuTHead
        /\ \A t \in 0..(nTasks - 1) : tRec[t] => rclCnt[t] = 1

\* Used only by MutGateSkipSilent: if the run ends leaked, did the code at
\* least raise one of its hard errors?  (A violation = silent leak.)
HardErrs == {"SeqMismatch", "OverReservation", "IncompleteReclaims",
             "ReclaimExceedsHead", "ReclaimFailureRaised", "StagingStuck"}
LeakImpliesHardError == ~NoLeakAtEnd => errs \cap HardErrs # {}

\* Liveness.
StepWaiterWakes == [](stPc \in {"prewait", "blk"} => <>(stPc \notin {"prewait", "blk"}))
ExtWaiterWakes  == [](extPc \in {"prewait", "blk"} => <>(extPc \notin {"prewait", "blk"}))
Termination     == <>(stPc = "done")

=============================================================================
