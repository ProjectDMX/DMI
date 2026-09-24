---------------------------- MODULE RingCapacity ----------------------------
(***************************************************************************)
(* A TLA+ model of the ring's capacity protocol: who guarantees that a     *)
(* producer kernel never writes into payload bytes (or a publication slot) *)
(* the drain has not yet released.                                         *)
(*                                                                         *)
(* WHY THIS SPEC EXISTS                                                    *)
(*   cbmc/payload_ring_span.cpp proves payload_compute_spans is correct    *)
(*   GIVEN its precondition                                                *)
(*       payload_free_bytes(head, tail, capacity) >= nbytes                *)
(*   (payload_ring.cuh:59-60).  The kernel cannot check it -- it never     *)
(*   reads the tail (ring_state.h:8-11) -- so the whole obligation sits on *)
(*   the host: "Space is guaranteed by the pre-forward capacity check"     *)
(*   (producer.cuh:12, drain_thread.h:7, ring_engine_py.cu:224).  This     *)
(*   spec is that check, and every path that reserves ring space.          *)
(*                                                                         *)
(* SOURCE OF TRUTH                                                         *)
(*   native/csrc/ring/ring_engine_py.cu:367-447   prepare_step             *)
(*   native/csrc/ring/ring_engine_py.cu:449-520   reserve_record           *)
(*   native/csrc/ring/ring_engine_py.cu:634-656   available_capacity,      *)
(*                                                reserve_one,             *)
(*                                                flush_and_wait           *)
(*   native/csrc/ring/drain_thread.cpp:253-294    reserve, reserve_record, *)
(*                                                apply_pending_record_    *)
(*                                                reclaims                 *)
(*   native/csrc/ring/drain_thread.cpp:341-367    do_full_flush            *)
(*   native/csrc/ring/drain_thread.cpp:507-561    scan_ready,              *)
(*                                                account_record_task      *)
(*   native/csrc/ring/drain_thread.cpp:594-600    flush_state_update       *)
(*   native/csrc/ring/producer.cu:176-320,420-700 every producer kernel:   *)
(*                                                alloc = align_up(actual) *)
(*   native/csrc/ring/task_ring.cuh:72-86         task_publish (no check)  *)
(*   src/dmi/adapters/base.py:281-335             commit_step              *)
(*   src/dmi/hooks/point.py:259-346               HookPoint.forward, the   *)
(*                                                eager safety net         *)
(*   src/dmi/records.py:317-386                   emit_output,             *)
(*                                                prepare_replay,          *)
(*                                                _needs_reclaim           *)
(*                                                                         *)
(* ABSTRACTION                                                             *)
(*   Bytes are counted in PAYLOAD_ALIGN units.  That is exact, not an      *)
(*   approximation: every reservation is a sum of align_up(...) values,    *)
(*   every kernel advances payload_head by align_up(actual), and the drain *)
(*   advances the tail by align_up(scanned) -- so every quantity the       *)
(*   protocol compares is a multiple of PAYLOAD_ALIGN.                     *)
(*                                                                         *)
(*   The uint64 subtraction  cap - (head - tail)  is modelled as written,  *)
(*   wrap included: if head - tail > cap the C++ result is a huge unsigned *)
(*   number, represented here by the constant Wrap, larger than any        *)
(*   request.  That is the only place the model needs wrap-around, and     *)
(*   leaving it out would hide a real failure mode (see NoWrap).           *)
(*                                                                         *)
(*   One CUDA stream.  Producers are enqueued in program order and run     *)
(*   asynchronously; cudaStreamSynchronize is modelled as "the queue is    *)
(*   empty" enabling the action that syncs.  This is the documented        *)
(*   serialized-stream contract (capture-storage-design.md:298-300).       *)
(*                                                                         *)
(*   The drain thread is fully nondeterministic: it may release any        *)
(*   staging-sized prefix of published tasks at any moment, or never.  A   *)
(*   drain that is slow because staging is full or the sink is slow is the *)
(*   realistic case this covers.                                           *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    Cap,            \* payload_cap, in PAYLOAD_ALIGN units
    Staging,        \* staging_cap, same units
    TaskCap,        \* task_cap (publication slots)
    MaxBytes,       \* largest transport size of one hook, in units
    MaxHooks,       \* most hooks that fire in one step
    MaxSteps,       \* number of steps (prepare_step / reserve_record calls)
    Wrap,           \* the uint64 underflow result: > any request
    Record,         \* FALSE: adapter path (prepare_step + eager safety net)
                    \* TRUE : generic-record path (reserve_record + reclaim)
    NeedsEager,     \* adapter path: some adapter overrides
                    \* _spec_needs_eager (base.py:150-155) to return TRUE.
                    \* No shipped adapter does; it is a documented
                    \* extension point "for dynamic-shape backends".
    PrefixMismatch, \* adapter path: the device row count a prefix producer
                    \* reads may exceed the CPU actual_q_len the step was
                    \* sized with (base.py:359-366 vs producer.cu:234-246).
    GateMismatch    \* record path: a device gate may skip an occurrence the
                    \* host reserved, violating integration-api-v1.md:370-374

ASSUME Cap \in Nat \ {0} /\ Staging \in Nat \ {0} /\ TaskCap \in Nat \ {0}
ASSUME MaxBytes \in Nat \ {0} /\ MaxHooks \in Nat \ {0} /\ MaxSteps \in Nat
ASSUME Wrap > MaxBytes * MaxHooks + Cap + TaskCap
ASSUME \A b \in {Record, NeedsEager, PrefixMismatch, GateMismatch} : b \in BOOLEAN

Min(a, b) == IF a <= b THEN a ELSE b

\* effective_cap = min(payload_cap, staging_cap)  (ring_engine_py.cu:405)
Eff == Min(Cap, Staging)

\* pcap - (head - tail), as uint64 arithmetic computes it.
\* head < tail (only reachable once the GPU has run past its reservation)
\* gives cap + (tail - head), which is also what the unsigned C++ yields.
Avail(head, tl, cap) == IF head - tl <= cap THEN cap - (head - tl) ELSE Wrap

(***************************************************************************)
(* One hook firing within a step.                                          *)
(*   res  : the CPU's transport size for it -- what the step reservation   *)
(*          (or reserve_one, or the record reservation item) counts.       *)
(*   dev  : what the kernel actually allocates: align_up(actual) where     *)
(*          actual is computed ON THE DEVICE from a device-side count.     *)
(*   skip : the kernel returns before publishing (record emit gate).       *)
(*   recl : record item reserved with needs_reclaim (records.py:376-386).  *)
(***************************************************************************)
PrefixDev(r) == IF PrefixMismatch THEN r .. MaxBytes ELSE {r}

AdapterHooks ==
    {[kind |-> "static", res |-> r, dev |-> r, skip |-> FALSE, recl |-> FALSE]
        : r \in 1 .. MaxBytes}
    \cup
    UNION {{[kind |-> "prefix", res |-> r, dev |-> d, skip |-> FALSE,
             recl |-> FALSE] : d \in PrefixDev(r)} : r \in 1 .. MaxBytes}

GateSkips == IF GateMismatch THEN BOOLEAN ELSE {FALSE}

RecordHooks ==
    \* identity, not gated: reserved exactly, cannot be short, cannot skip
    {[kind |-> "record", res |-> r, dev |-> r, skip |-> FALSE, recl |-> FALSE]
        : r \in 1 .. MaxBytes}
    \cup
    \* packed/prefix/chunked or device-gated: reserved at the upper bound,
    \* the kernel may write less, and the unused tail is reclaimed
    UNION {{[kind |-> "record", res |-> r, dev |-> d, skip |-> s,
             recl |-> TRUE] : d \in 0 .. r, s \in GateSkips}
           : r \in 1 .. MaxBytes}

Hooks == IF Record THEN RecordHooks ELSE AdapterHooks

StepPlans == UNION {[1 .. n -> Hooks] : n \in 1 .. MaxHooks}

EagerChoices == IF NeedsEager THEN BOOLEAN ELSE {FALSE}

RECURSIVE SumRes(_), SumSeq(_), SumA(_), Drainable(_)
SumRes(s) == IF s = <<>> THEN 0 ELSE Head(s).res + SumRes(Tail(s))
SumA(s)   == IF s = <<>> THEN 0 ELSE Head(s).a   + SumA(Tail(s))
SumSeq(s) == IF s = <<>> THEN 0 ELSE Head(s)     + SumSeq(Tail(s))

\* do_full_flush drains whole entries in order and stops at the first one
\* larger than staging (drain_thread.cpp:348-351).
Drainable(s) == IF s = <<>> \/ Head(s) > Staging THEN 0
                ELSE 1 + Drainable(Tail(s))

VARIABLES
    cpuHead,   \* DrainThread::cpu_payload_head_  -- reserved
    cpuTask,   \* DrainThread::cpu_task_head_
    tail,      \* cpu_payload_tail_committed_     -- released after D2H
    taskTail,  \* cpu_task_tail_                  -- slots cleared
    gpuHead,   \* *ring.payload_head              -- what kernels wrote
    gpuTask,   \* *ring.task_head
    queue,     \* producer kernels enqueued on the stream, not yet run:
               \* Seq of [a: alloc units, skip: BOOLEAN, rc: reclaim units]
    pub,       \* published, not yet drained: Seq of alloc units
    reclaim,   \* pending_reclaim_bytes_
    hooks,     \* hooks of the current step still to fire
    eager,     \* transport.force_eager (adapter) / OVERSIZED (record)
    pc,        \* "plan" | "prepare" | "hooks"
    step,
    event      \* ghost: the path the last CPU action took (vacuity probes)

vars == <<cpuHead, cpuTask, tail, taskTail, gpuHead, gpuTask, queue, pub,
          reclaim, hooks, eager, pc, step, event>>

TypeOK ==
    /\ cpuHead \in Int /\ cpuTask \in Int
    /\ tail \in Nat /\ taskTail \in Nat /\ gpuHead \in Nat /\ gpuTask \in Nat
    /\ queue \in Seq([a : Nat, skip : BOOLEAN, rc : Nat])
    /\ pub \in Seq(Nat)
    /\ reclaim \in Nat
    /\ hooks \in Seq(Hooks)
    /\ eager \in BOOLEAN
    /\ pc \in {"plan", "prepare", "hooks"}
    /\ step \in 0 .. MaxSteps

Init ==
    /\ cpuHead = 0 /\ cpuTask = 0 /\ tail = 0 /\ taskTail = 0
    /\ gpuHead = 0 /\ gpuTask = 0
    /\ queue = <<>> /\ pub = <<>> /\ reclaim = 0
    /\ hooks = <<>> /\ eager = FALSE
    /\ pc = "plan" /\ step = 0 /\ event = "init"

(***************************************************************************)
(* Synchronous flush: cudaStreamSynchronize (queue empty -- the action is  *)
(* only enabled then) followed by force_flush_and_wait, which drains every *)
(* published entry that fits staging.  Primed values below are those after *)
(* the flush; callers add their reservation on top.                        *)
(***************************************************************************)
FlushK        == Drainable(pub)
FlushedTail   == tail + SumSeq(SubSeq(pub, 1, FlushK))
FlushedTTail  == taskTail + FlushK
FlushedPub    == SubSeq(pub, FlushK + 1, Len(pub))

SyncFlush ==
    /\ queue = <<>>
    /\ tail' = FlushedTail
    /\ taskTail' = FlushedTTail
    /\ pub' = FlushedPub

NoFlush == UNCHANGED <<tail, taskTail, pub>>

(***************************************************************************)
(* CPU: pick the step's hooks (plan_step / the record producer plan).      *)
(***************************************************************************)
Plan ==
    /\ pc = "plan"
    /\ step < MaxSteps
    /\ hooks' \in StepPlans
    /\ pc' = "prepare"
    /\ step' = step + 1
    /\ UNCHANGED <<cpuHead, cpuTask, tail, taskTail, gpuHead, gpuTask, queue,
                   pub, reclaim, eager, event>>

(***************************************************************************)
(* Adapter path: prepare_step (ring_engine_py.cu:367-447), called from     *)
(* commit_step (base.py:299-314).                                          *)
(***************************************************************************)
PrepareStep ==
    /\ pc = "prepare"
    /\ ~Record
    /\ LET n     == Len(hooks)
           total == SumRes(hooks)
       IN
       IF total > Eff \/ n > TaskCap THEN
           \* Case B: STEP_OVERSIZED.  Flush, no reservation, safety net on.
           /\ SyncFlush
           /\ eager' = TRUE
           /\ UNCHANGED <<cpuHead, cpuTask>>
           /\ event' = "oversized"
       ELSE IF total <= Avail(cpuHead, tail, Cap)
               /\ n <= Avail(cpuTask, taskTail, TaskCap) THEN
           \* Case A fast path: STEP_RING_OK.
           /\ NoFlush
           /\ cpuHead' = cpuHead + total
           /\ cpuTask' = cpuTask + n
           /\ eager' \in EagerChoices
           /\ event' = "ok"
       ELSE
           \* STEP_RING_FLUSHED: sync, flush, then reserve WITHOUT
           \* re-checking the available space (ring_engine_py.cu:442-446).
           /\ SyncFlush
           /\ cpuHead' = cpuHead + total
           /\ cpuTask' = cpuTask + n
           /\ eager' \in EagerChoices
           /\ event' = "flushed"
    /\ pc' = "hooks"
    /\ UNCHANGED <<gpuHead, gpuTask, queue, reclaim, hooks, step>>

(***************************************************************************)
(* Record path: reserve_record (ring_engine_py.cu:449-520).  Reclaims      *)
(* already accounted by the drain are applied first, and again after any   *)
(* flush; all of them are in `reclaim` by then because the flush syncs.    *)
(***************************************************************************)
ReserveRecord ==
    /\ pc = "prepare"
    /\ Record
    /\ LET n     == Len(hooks)
           total == SumRes(hooks)
           head0 == cpuHead - reclaim      \* apply_pending_record_reclaims
       IN
       IF total > Eff \/ n > TaskCap THEN
           \* OVERSIZED: flush; the entries go CPU-direct, no producer runs
           /\ SyncFlush
           /\ cpuHead' = head0
           /\ reclaim' = 0
           /\ UNCHANGED cpuTask
           /\ eager' = TRUE
           /\ event' = "oversized"
       ELSE IF total <= Avail(head0, tail, Cap)
               /\ n <= Avail(cpuTask, taskTail, TaskCap) THEN
           /\ NoFlush
           /\ cpuHead' = head0 + total
           /\ cpuTask' = cpuTask + n
           /\ reclaim' = 0
           /\ eager' = FALSE
           /\ event' = "ok"
       ELSE
           \* flush, apply reclaims, reserve without re-checking
           /\ SyncFlush
           /\ cpuHead' = head0 + total
           /\ cpuTask' = cpuTask + n
           /\ reclaim' = 0
           /\ eager' = FALSE
           /\ event' = "flushed"
    /\ pc' = "hooks"
    /\ UNCHANGED <<gpuHead, gpuTask, queue, hooks, step>>

(***************************************************************************)
(* CPU: the next hook fires (HookPoint.forward, point.py:259-346).         *)
(***************************************************************************)
Kernel(h) == [a |-> h.dev, skip |-> h.skip,
              rc |-> IF h.recl THEN h.res - h.dev ELSE 0]

Advance ==
    /\ hooks' = Tail(hooks)
    /\ pc' = IF Len(hooks) = 1 THEN "plan" ELSE "hooks"

\* Fast path: dispatch_producer into the step's reservation.
FireFast ==
    /\ pc = "hooks"
    /\ ~eager
    /\ queue' = Append(queue, Kernel(Head(hooks)))
    /\ Advance
    /\ event' = "fast"
    /\ UNCHANGED <<cpuHead, cpuTask, tail, taskTail, gpuHead, gpuTask, pub,
                   reclaim, eager, step>>

\* Record OVERSIZED: the entry was submitted CPU-direct; the hook skips its
\* producer launch (records.py:321-335).
FireRecordOversized ==
    /\ pc = "hooks"
    /\ Record
    /\ eager
    /\ Advance
    /\ event' = "cpu_direct"
    /\ UNCHANGED <<cpuHead, cpuTask, tail, taskTail, gpuHead, gpuTask, queue,
                   pub, reclaim, eager, step>>

\* The eager safety net (point.py:292-339).  Note what is checked:
\* available_capacity() is PAYLOAD space only; reserve_one then claims one
\* task entry as well (ring_engine_py.cu:643-646) without checking the task
\* ring.  The flushed branch reserves without re-checking either.
FireEager ==
    /\ pc = "hooks"
    /\ ~Record
    /\ eager
    /\ LET h == Head(hooks)
           t == h.dev      \* align_up(x_cont.nbytes, 16): the real tensor
       IN
       IF h.kind = "prefix" THEN
           \* stripped producers never reserve in eager mode: flush, CPU-direct
           /\ SyncFlush
           /\ UNCHANGED <<cpuHead, cpuTask, queue>>
           /\ event' = "cpu_direct"
       ELSE IF t <= Min(Avail(cpuHead, tail, Cap), Eff) THEN
           /\ NoFlush
           /\ cpuHead' = cpuHead + t
           /\ cpuTask' = cpuTask + 1
           /\ queue' = Append(queue, Kernel(h))
           /\ event' = "eager_reserve"
       ELSE IF t <= Eff THEN
           /\ SyncFlush
           /\ cpuHead' = cpuHead + t
           /\ cpuTask' = cpuTask + 1
           /\ queue' = Append(queue, Kernel(h))
           /\ event' = "eager_flush_reserve"
       ELSE
           /\ SyncFlush
           /\ UNCHANGED <<cpuHead, cpuTask, queue>>
           /\ event' = "cpu_direct"
    /\ Advance
    /\ UNCHANGED <<gpuHead, gpuTask, reclaim, eager, step>>

(***************************************************************************)
(* GPU: the next producer kernel on the stream runs.  It reads            *)
(* payload_head/task_head, writes its spans, publishes, and advances both  *)
(* heads (producer.cu:155-160).  It never looks at a tail.                 *)
(***************************************************************************)
GpuRun ==
    /\ queue # <<>>
    /\ LET k == Head(queue) IN
       IF k.skip THEN
           \* record_emit_allowed() false: return before any write/publish
           UNCHANGED <<gpuHead, gpuTask, pub, reclaim>>
       ELSE
           /\ gpuHead' = gpuHead + k.a
           /\ gpuTask' = gpuTask + 1
           /\ pub' = Append(pub, k.a)
           \* account_record_task: the drain credits the unused part of a
           \* conservative reservation once it sees the publication.  Doing it
           \* at publish time is the most generous timing the code allows.
           /\ reclaim' = reclaim + k.rc
    /\ queue' = Tail(queue)
    /\ UNCHANGED <<cpuHead, cpuTask, tail, taskTail, hooks, eager, pc, step,
                   event>>

(***************************************************************************)
(* Drain thread, in the background: release a staging-sized batch of the   *)
(* oldest published entries (flush_state_update + D2H + commit).           *)
(***************************************************************************)
Drain ==
    /\ \E j \in 1 .. Drainable(pub) :
          /\ SumSeq(SubSeq(pub, 1, j)) <= Staging
          /\ tail' = tail + SumSeq(SubSeq(pub, 1, j))
          /\ taskTail' = taskTail + j
          /\ pub' = SubSeq(pub, j + 1, Len(pub))
    /\ UNCHANGED <<cpuHead, cpuTask, gpuHead, gpuTask, queue, reclaim, hooks,
                   eager, pc, step, event>>

Done ==
    /\ pc = "plan" /\ step = MaxSteps
    /\ UNCHANGED vars

Next ==
    \/ Plan \/ PrepareStep \/ ReserveRecord
    \/ FireFast \/ FireEager \/ FireRecordOversized
    \/ GpuRun \/ Drain \/ Done

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(***************************************************************************)
(* THE OBLIGATIONS                                                         *)
(***************************************************************************)

\* 1. The CBMC precondition, at the moment the kernel can run: the next
\*    producer on the stream fits in the payload bytes the drain has
\*    released.  A state predicate, not an action one: the GPU may run the
\*    head of the queue in ANY state, so if it does not fit in some
\*    reachable state, an overrun is reachable.
PayloadFits ==
    (queue # <<>> /\ ~Head(queue).skip) =>
        (gpuHead - tail) + Head(queue).a <= Cap

\* 2. The same for publication slots: task_publish overwrites slot
\*    seq % task_cap unconditionally (task_ring.cuh:78-85).  Publishing into
\*    a slot the drain has not cleared destroys an unread publication.
TaskFits ==
    (queue # <<>> /\ ~Head(queue).skip) =>
        (gpuTask - taskTail) + 1 <= TaskCap

Safety == PayloadFits /\ TaskFits

\* Supporting invariants -- the argument the code comments make.
\*  Covered: the GPU never runs ahead of what the CPU reserved.
Covered ==
    /\ gpuHead + reclaim + SumA(queue) <= cpuHead
    /\ gpuTask + Len(SelectSeq(queue, LAMBDA k : ~k.skip)) <= cpuTask

\*  NoWrap: the CPU's own accounting never claims more than the ring holds.
\*  If this fails, cap - (head - tail) underflows and every later capacity
\*  check -- prepare_step's fast path, available_capacity(), reserve_record
\*  -- passes unconditionally: the ring loses backpressure for good.
NoWrap ==
    /\ cpuHead - reclaim - tail <= Cap
    /\ cpuTask - taskTail <= TaskCap

\*  Exact: every reservation is eventually consumed -- nothing leaks.
Exact ==
    (queue = <<>> /\ pc = "plan") =>
        /\ cpuHead - reclaim = gpuHead
        /\ cpuTask = gpuTask

-----------------------------------------------------------------------------
(***************************************************************************)
(* VACUITY PROBES -- each should be REFUTED.  A refutation proves the      *)
(* model reaches the state the obligations quantify over.                  *)
(***************************************************************************)
NeverFull         == gpuHead - tail < Cap                \* ring really fills
NeverTasksFull    == gpuTask - taskTail < TaskCap
NeverFlushedPath  == event # "flushed"
NeverOversized    == event # "oversized"
NeverEagerReserve == event # "eager_reserve"
NeverReclaims     == reclaim = 0
=============================================================================
