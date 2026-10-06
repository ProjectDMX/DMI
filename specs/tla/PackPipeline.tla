---------------------------- MODULE PackPipeline ----------------------------
(***************************************************************************)
(* The IN-PROCESS PACK PIPELINE of one capture process, on main @ 5b3b632: *)
(*                                                                         *)
(*   ring -> NativePackSink/PackSink -> spool (.ready)                     *)
(*        -> CaptureStorageService cycle: list, upload a chunk, index it   *)
(*        -> object store + catalog                                        *)
(*                                                                         *)
(* with flush(timeout), engine.close()'s drain, stop(), the two            *)
(* Cancellations, a crash at any point, and the next process's start()     *)
(* with or without reconcile_on_start.                                     *)
(*                                                                         *)
(* ABSTRACTED (other specs, other workstreams):                            *)
(*   - the publisher lease is an environment signal, "held" or "lost"      *)
(*     (quarantined, refused, abandoned at its deadline all look alike to  *)
(*     the cycle: writer_.held_lease() == nullptr).  LeaseLifecycle.tla    *)
(*     and PublisherLease.tla model how it is kept;                        *)
(*   - the spool has one owner, this process (store/spool.h owner lock);   *)
(*     adoption of dead siblings is not modelled;                         *)
(*   - an outcome-unknown publish is resolved at once (landed or not):     *)
(*     the quarantine that follows it outlasts the statement               *)
(*     (LeaseLifecycle O2), so the next replay guard sees its outcome;     *)
(*   - records are packs: one pack per admitted record; sealing reasons    *)
(*     (size, linger, flush, release) are one Stage action;                *)
(*   - time: "the flush deadline has passed" and "the read grace has       *)
(*     passed" are booleans that only become TRUE while a flush runs.      *)
(*                                                                         *)
(* SOURCE (5b3b632).                                                       *)
(*   storage_service.cpp                                                   *)
(*     :424-478   flush()       try_lock_until, run_cycle(dl,false,false), *)
(*                              "if (drained) return true", sleep, again   *)
(*     :552-747   run_cycle()                                              *)
(*       :558-563   cancelled_for_good -> cut short, no work               *)
(*       :570-573   ArmedDeadline: uploads at dl, reads at dl + grace      *)
(*       :577-581   catalog = ensure_publisher_lease()     -> CycLease     *)
(*       :592-596   step 1: index pending_index_           -> CycRetry     *)
(*       :627-637   step 2: listing, cut by upload_cancel_ -> CycList      *)
(*       :641-655   per-chunk gate: owed, cancelled, lease -- for chunks   *)
(*                  after the first ONLY ("if (next != 0)")-> CycChunk     *)
(*       :657-674   upload_chunk + index_or_owe            -> Upload*/Idx  *)
(*       :697-705   step 4: reconcile, loop cycles only    -> CycRecon     *)
(*       :715-719   failed / drained                       -> CycEnd       *)
(*       :722-731   a lease refusal ends the cycle (catch)                 *)
(*     :749-770   index_or_owe: unindexed -> pending_index_                *)
(*     :1127-1271 index_bounded: no batch once reads are cut, none past    *)
(*                the deadline but the first; a split halves; a failure    *)
(*                ends the pass; kLease rethrown               -> IdxBatch *)
(*     :1284-1378 reconcile(): unindexed packs are NOT owed                *)
(*     :321-370   sweep_and_reconcile_at_start: only a lease refusal sets  *)
(*                reconcile_owed_; any other failure is recorded only      *)
(*     :372-413   stop(): cancels both for good, joins the loop            *)
(*   indexer.cpp:282-293  commit() throws kLease without a held lease      *)
(*   uploader.cpp:103-265 UploadOne: Remove() from the spool only after    *)
(*                the object is verified; cancel checked before attempts   *)
(*   native_pack_sink.cpp:256-282 on_engine_release(): the backstop flush  *)
(*   pack_sink.cpp:254-353, 461-520 Flush barrier, stager                  *)
(*   engine.py:944-970, 992-1062  close(): seal, stop ring, flush, stop    *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
  Packs,                 \* pack ids (one record each)
  Chunk,                 \* indexer.max_packs: the upload chunk and batch size
  ReconcileOnStart,      \* StorageServiceConfig::reconcile_on_start
  ReconcilePeriodic,     \* reconcile_interval_ns > 0
  MaxRestarts,           \* processes after the first (crash or close+start)
  MaxCrashes,            \* how many of the process ends may be crashes
  MaxLeaseLosses,        \* environment: times the lease may be lost
  MaxFaults,             \* transient upload/index/reconcile failures
  AllowClose,            \* engine.close() may run (drain, then stop())
  AllowUserFlush,        \* flush_and_wait() may run while capturing
  \* --- code-shape switches.  The value marked MAIN is what 5b3b632 does.
  CheckLeaseBeforeFirstChunk, \* MAIN: FALSE (gate only "if (next != 0)")
  GateUploadsOnLease,    \* MAIN: TRUE.  FALSE: upload without the lease
  GateUploadsOnOwed,     \* MAIN: TRUE.  FALSE: upload while packs are owed
  ReleaseStagesTail,     \* MAIN: TRUE.  FALSE: release reports sealed
                         \* without staging the open pack
  FirstBatchOnly,        \* MAIN: TRUE.  FALSE: index_bounded ignores the
                         \* deadline after the first batch
  ReconcileFailuresOwed, \* MAIN: FALSE.  TRUE: a reconcile pass (at start
                         \* or the loop's) that failed, or left a pack
                         \* unindexed, is owed to the loop again -- not
                         \* only one a lease refusal cut short
  FlushRunsReconcile     \* MAIN: FALSE.  TRUE: flush()'s cycles reconcile

ASSUME Chunk \in 1..Cardinality(Packs)

VARIABLES
  \* durable
  spool, s3, catalog,
  \* the sink and ring
  born, bornHere, sinkMem, lostInSink, ring, sealedOnRelease, sealedBefore,
  \* the process
  alive, started, restarts, crashes, stopReq, cl,
  \* environment
  lease, losses, faults,
  \* the service's memory
  pending, reconcileOwed,
  \* the cycle in flight (cycle_mutex_ holder)
  cyc, runner, catOk, checked, first, listed, upl, uplCut, done, idxQ,
  idxFirst, idxSrc, idxSplit, cutC, failC, upFail, uploadedAll,
  \* flush()
  fl, flWant, pastDl, pastGrace, pdBatches, flTarget,
  \* history for the properties
  badChunkStart, flushReconciled, remembered

durable == <<spool, s3, catalog>>
sinkv   == <<born, bornHere, sinkMem, lostInSink, ring, sealedOnRelease, sealedBefore>>
procv   == <<alive, started, restarts, crashes, stopReq, cl>>
envv    == <<lease, losses, faults>>
memv    == <<pending, reconcileOwed>>
cycv    == <<cyc, runner, catOk, checked, first, listed, upl, uplCut, done,
             idxQ, idxFirst, idxSrc, idxSplit, cutC, failC, upFail, uploadedAll>>
flv     == <<fl, flWant, pastDl, pastGrace, pdBatches, flTarget>>
histv   == <<badChunkStart, flushReconciled, remembered>>
vars    == <<durable, sinkv, procv, envv, memv, cycv, flv, histv>>

Min(a, b) == IF a < b THEN a ELSE b

\* Cancellation::cancelled() for each of the service's two Cancellations.
\* stop() cancels both for good; a flush's cycle arms the uploads' at its
\* deadline and the reads' one request timeout later (run_cycle :570-573).
UploadCancelled == stopReq \/ (cyc # "idle" /\ runner = "flush" /\ pastDl)
ReadCancelled   == stopReq \/ (cyc # "idle" /\ runner = "flush" /\ pastGrace)

\* What only this process remembers: uploaded (gone from the spool) and not
\* yet in the catalog.
Remembered ==
  ((pending \cup done \cup (IF idxSrc = "recon" THEN {} ELSE idxQ)) \ catalog)
    \ spool

CycleReset ==
  /\ cyc' = "idle" /\ catOk' = FALSE /\ checked' = FALSE /\ first' = TRUE
  /\ listed' = {} /\ upl' = {} /\ uplCut' = FALSE /\ done' = {} /\ idxQ' = {}
  /\ idxFirst' = TRUE /\ idxSrc' = "none" /\ idxSplit' = FALSE
  /\ cutC' = FALSE /\ failC' = FALSE /\ upFail' = FALSE /\ uploadedAll' = FALSE

Init ==
  /\ spool = {} /\ s3 = {} /\ catalog = {}
  /\ born = {} /\ bornHere = {} /\ sinkMem = {} /\ lostInSink = {}
  /\ ring = "open" /\ sealedOnRelease = FALSE /\ sealedBefore = FALSE
  /\ alive = TRUE /\ started = FALSE /\ restarts = 0 /\ crashes = 0
  /\ stopReq = FALSE /\ cl = "run"
  /\ lease = "held" /\ losses = 0 /\ faults = 0
  /\ pending = {} /\ reconcileOwed = FALSE
  /\ cyc = "idle" /\ runner = "loop" /\ catOk = FALSE /\ checked = FALSE
  /\ first = TRUE /\ listed = {} /\ upl = {} /\ uplCut = FALSE /\ done = {}
  /\ idxQ = {} /\ idxFirst = TRUE /\ idxSrc = "none" /\ idxSplit = FALSE
  /\ cutC = FALSE /\ failC = FALSE /\ upFail = FALSE /\ uploadedAll = FALSE
  /\ fl = "none" /\ flWant = FALSE /\ pastDl = FALSE /\ pastGrace = FALSE
  /\ pdBatches = 0 /\ flTarget = {}
  /\ badChunkStart = FALSE /\ flushReconciled = FALSE /\ remembered = 0

-----------------------------------------------------------------------------
(* The environment: the lease comes and goes, time passes a flush deadline *)

LoseLease ==
  /\ alive /\ lease = "held" /\ losses < MaxLeaseLosses
  /\ lease' = "lost" /\ losses' = losses + 1
  /\ UNCHANGED <<durable, sinkv, procv, faults, memv, cycv, flv, histv>>

\* ensure_publisher_lease() from the lease thread or a cycle, once the
\* quarantine (or the rival) is gone.
Reacquire ==
  /\ alive /\ lease = "lost"
  /\ lease' = "held"
  /\ UNCHANGED <<durable, sinkv, procv, losses, faults, memv, cycv, flv, histv>>

PassDeadline ==
  /\ fl = "running" /\ ~pastDl
  /\ pastDl' = TRUE
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, cycv, fl, flWant,
                 pastGrace, pdBatches, flTarget, histv>>

PassGrace ==
  /\ fl = "running" /\ pastDl /\ ~pastGrace
  /\ pastGrace' = TRUE
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, cycv, fl, flWant,
                 pastDl, pdBatches, flTarget, histv>>

-----------------------------------------------------------------------------
(* The ring and the sink                                                   *)

\* The ring releases a record to the sink's open pack.
Admit(p) ==
  /\ alive /\ ring = "open" /\ p \notin born
  /\ born' = born \cup {p} /\ bornHere' = bornHere \cup {p}
  /\ sinkMem' = sinkMem \cup {p}
  /\ UNCHANGED <<durable, lostInSink, ring, sealedOnRelease, sealedBefore,
                 procv, envv, memv, cycv, flv, histv>>

\* A seal (size, linger, flush, release) and the stager's spool.Stage(): the
\* pack is a .ready file.  A sink that did not seal may still stage after
\* its release (native_pack_sink.h:86-95), so this is not gated on the ring.
Stage(p) ==
  /\ alive /\ p \in sinkMem
  /\ sinkMem' = sinkMem \ {p} /\ spool' = spool \cup {p}
  /\ UNCHANGED <<s3, catalog, born, bornHere, lostInSink, ring,
                 sealedOnRelease, sealedBefore, procv, envv, memv, cycv, flv,
                 histv>>

-----------------------------------------------------------------------------
(* The process: start, crash, restart                                       *)

\* start(): the lease, the sweep (staging is atomic here, so it has no .open
\* file to sweep), and the reconcile at start.  Only a lease refusal makes
\* the pass owed to the loop (:357-358); any other failure is recorded only.
StartService ==
  /\ alive /\ ~started /\ lease = "held"
  /\ started' = TRUE
  /\ \/ /\ ~ReconcileOnStart
        /\ UNCHANGED <<catalog, reconcileOwed, lease, losses, faults>>
     \/ /\ ReconcileOnStart                              \* the pass completes
        /\ catalog' = catalog \cup s3
        /\ UNCHANGED <<reconcileOwed, lease, losses, faults>>
     \/ /\ ReconcileOnStart /\ losses < MaxLeaseLosses   \* loses the lease
        /\ lease' = "lost" /\ losses' = losses + 1
        /\ reconcileOwed' = TRUE
        /\ UNCHANGED <<catalog, faults>>
     \/ /\ ReconcileOnStart /\ faults < MaxFaults        \* a listing, HEAD or
        /\ faults' = faults + 1                          \* catalog failure
        /\ reconcileOwed' = ReconcileFailuresOwed
        /\ UNCHANGED <<catalog, lease, losses>>
  /\ UNCHANGED <<spool, s3, sinkv, alive, restarts, crashes, stopReq, cl,
                 pending, cycv, flv, histv>>

ResetMemory ==
  /\ pending' = {} /\ reconcileOwed' = FALSE
  /\ CycleReset /\ runner' = "loop"
  /\ fl' = "none" /\ flWant' = FALSE /\ pastDl' = FALSE /\ pastGrace' = FALSE
  /\ pdBatches' = 0 /\ flTarget' = {}

\* The process dies anywhere.  An upload in flight may have put its object
\* without the Remove() that follows (uploader.cpp:221-240): the pack is
\* then in both places.  What only this process remembered is now only in
\* the bucket.
Crash ==
  /\ alive /\ crashes < MaxCrashes /\ cl # "done"
  /\ \E landed \in SUBSET upl:
       s3' = s3 \cup landed
  /\ alive' = FALSE /\ started' = FALSE /\ crashes' = crashes + 1
  /\ lostInSink' = lostInSink \cup sinkMem /\ sinkMem' = {}
  /\ remembered' = remembered + Cardinality(Remembered)
  /\ ResetMemory
  /\ UNCHANGED <<spool, catalog, born, bornHere, ring, sealedOnRelease,
                 sealedBefore, restarts, stopReq, cl, envv, badChunkStart,
                 flushReconciled>>

\* The next process on this spool (after a crash, or after a close).
Restart ==
  /\ ~alive /\ restarts < MaxRestarts
  /\ alive' = TRUE /\ restarts' = restarts + 1 /\ stopReq' = FALSE
  /\ cl' = "run" /\ ring' = "open" /\ sealedOnRelease' = FALSE
  /\ sealedBefore' = FALSE /\ bornHere' = {}
  /\ UNCHANGED <<durable, born, sinkMem, lostInSink, started, crashes, envv,
                 memv, cycv, flv, histv>>

-----------------------------------------------------------------------------
(* The cycle.  One at a time (cycle_mutex_): the loop's, or a flush's.     *)

\* The loop's cycle.  Time is abstract: the loop may run one whenever no
\* cycle is in flight, until stop().  (The loop's wait-again rule is
\* PackPipelineLoop.tla's.)
LoopCycle ==
  /\ alive /\ started /\ ~stopReq /\ cyc = "idle"
  /\ cyc' = "lease" /\ runner' = "loop"
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, catOk, checked, first,
                 listed, upl, uplCut, done, idxQ, idxFirst, idxSrc, idxSplit,
                 cutC, failC, upFail, uploadedAll, flv, histv>>

\* flush()'s cycle (:445-460).  try_lock_until(deadline) on a free mutex
\* succeeds even past the deadline.  Past stop()'s cancel it returns false.
FlushCycle ==
  /\ alive /\ started /\ fl = "running" /\ flWant /\ cyc = "idle"
  /\ IF stopReq
       THEN /\ fl' = "false" /\ flWant' = FALSE
            /\ UNCHANGED <<cyc, runner>>
       ELSE /\ cyc' = "lease" /\ runner' = "flush" /\ flWant' = FALSE
            /\ UNCHANGED fl
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, catOk, checked, first,
                 listed, upl, uplCut, done, idxQ, idxFirst, idxSrc, idxSplit,
                 cutC, failC, upFail, uploadedAll, pastDl, pastGrace,
                 pdBatches, flTarget, histv>>

\* A loop cycle holds the mutex past the flush's deadline (:449).
FlushGivesUp ==
  /\ fl = "running" /\ flWant /\ pastDl /\ cyc # "idle"
  /\ fl' = "false" /\ flWant' = FALSE
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, cycv, pastDl, pastGrace,
                 pdBatches, flTarget, histv>>

\* :558-563, then :577-581.  A cycle started after stop()'s cancel does
\* nothing.  ensure_publisher_lease() is "is a lease held now".
CycLease ==
  /\ alive /\ cyc = "lease"
  /\ IF stopReq
       THEN /\ cyc' = "end" /\ cutC' = TRUE
            /\ UNCHANGED <<catOk, checked>>
       ELSE /\ catOk' = (lease = "held") /\ checked' = (lease = "held")
            /\ cyc' = "retry"
            /\ UNCHANGED cutC
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, runner, first, listed,
                 upl, uplCut, done, idxQ, idxFirst, idxSrc, idxSplit, failC,
                 upFail, uploadedAll, flv, histv>>

\* Step 1 (:592-596): what earlier cycles owe, before anything new.
CycRetry ==
  /\ alive /\ cyc = "retry"
  /\ IF catOk /\ pending # {}
       THEN /\ idxQ' = pending /\ pending' = {} /\ idxSrc' = "retry"
            /\ idxFirst' = TRUE /\ idxSplit' = FALSE /\ cyc' = "index"
       ELSE /\ cyc' = "list"
            /\ UNCHANGED <<idxQ, pending, idxSrc, idxFirst, idxSplit>>
  /\ UNCHANGED <<durable, sinkv, procv, envv, reconcileOwed, runner, catOk,
                 checked, first, listed, upl, uplCut, done, cutC, failC,
                 upFail, uploadedAll, flv, histv>>

\* Step 2's listing (:627-637): only with the lease (GateUploadsOnLease) and
\* nothing owed (GateUploadsOnOwed).  It hashes every staged pack -- long
\* work, after which the cycle's lease check is stale -- and a cancel stops
\* it between packs: an empty spool still lists as empty, not cut.
CycList ==
  /\ alive /\ cyc = "list"
  /\ IF (catOk \/ ~GateUploadsOnLease) /\ (pending = {} \/ ~GateUploadsOnOwed)
       THEN IF UploadCancelled /\ spool # {}
              THEN /\ cutC' = TRUE /\ cyc' = "recon"
                   /\ UNCHANGED <<listed, first, checked, uploadedAll>>
              ELSE /\ listed' = spool /\ first' = TRUE /\ cyc' = "chunk"
                   /\ checked' = (checked /\ spool = {})
                   /\ UNCHANGED <<cutC, uploadedAll>>
       ELSE /\ cyc' = "recon"
            /\ UNCHANGED <<listed, first, checked, cutC, uploadedAll>>
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, runner, catOk, upl,
                 uplCut, done, idxQ, idxFirst, idxSrc, idxSplit, failC,
                 upFail, flv, histv>>

\* The per-chunk gate (:641-655), and the chunk's start (:657).
CycChunk ==
  /\ alive /\ cyc = "chunk"
  /\ IF listed = {}
       THEN /\ uploadedAll' = ~upFail /\ cyc' = "recon"
            /\ UNCHANGED <<listed, upl, uplCut, first, checked, cutC, failC,
                           badChunkStart>>
       ELSE LET gate == ~first \/ CheckLeaseBeforeFirstChunk IN
            IF gate /\ ~first /\ pending # {} /\ GateUploadsOnOwed
              THEN /\ cyc' = "recon"                      \* :643
                   /\ UNCHANGED <<listed, upl, uplCut, first, checked, cutC,
                                  failC, uploadedAll, badChunkStart>>
            ELSE IF gate /\ ~first /\ UploadCancelled
              THEN /\ cutC' = TRUE /\ cyc' = "recon"      \* :644-647
                   /\ UNCHANGED <<listed, upl, uplCut, first, checked, failC,
                                  uploadedAll, badChunkStart>>
            ELSE IF gate /\ GateUploadsOnLease /\ lease # "held"
              THEN /\ failC' = TRUE /\ cyc' = "recon"     \* :650-654
                   /\ UNCHANGED <<listed, upl, uplCut, first, checked, cutC,
                                  uploadedAll, badChunkStart>>
            ELSE
              /\ \E c \in SUBSET listed:
                   /\ Cardinality(c) = Min(Chunk, Cardinality(listed))
                   /\ upl' = c /\ listed' = listed \ c
              \* Decision 7: no chunk starts while the writer holds no
              \* lease.  The lease thread may have lost it at any step
              \* since the cycle's last look (here: the listing).
              /\ badChunkStart' = (badChunkStart \/ lease # "held")
              /\ uplCut' = UploadCancelled                \* uploader.cpp:126
              /\ first' = FALSE /\ checked' = FALSE /\ cyc' = "upload"
              /\ UNCHANGED <<cutC, failC, uploadedAll>>
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, runner, catOk, done, idxQ,
                 idxFirst, idxSrc, idxSplit, upFail, flv, flushReconciled,
                 remembered>>

\* One pack of the chunk ends.  ok: object verified, then Remove()d from
\* the spool (uploader.cpp:221-240).  A cancel cuts it (stays staged); a
\* failure leaves it staged too.  An upload the cancel reached before its
\* first attempt cannot succeed (uploader.cpp:126-129).
UploadOne(p) ==
  /\ alive /\ cyc = "upload" /\ p \in upl
  /\ upl' = upl \ {p}
  /\ \/ /\ ~uplCut
        /\ s3' = s3 \cup {p} /\ spool' = spool \ {p} /\ done' = done \cup {p}
        /\ UNCHANGED <<cutC, upFail, faults>>
     \/ /\ UploadCancelled
        /\ cutC' = TRUE
        /\ UNCHANGED <<s3, spool, done, upFail, faults>>
     \/ /\ faults < MaxFaults
        /\ faults' = faults + 1 /\ upFail' = TRUE
        /\ UNCHANGED <<s3, spool, done, cutC>>
  /\ UNCHANGED <<catalog, sinkv, procv, lease, losses, memv, cyc, runner,
                 catOk, checked, first, listed, uplCut, idxQ, idxFirst,
                 idxSrc, idxSplit, failC, uploadedAll, flv, histv>>

\* The chunk is over: index what went up (:674).
UploadChunkDone ==
  /\ alive /\ cyc = "upload" /\ upl = {}
  /\ idxQ' = done /\ done' = {} /\ idxSrc' = "chunk" /\ idxFirst' = TRUE
  /\ idxSplit' = FALSE /\ cyc' = "index"
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, runner, catOk, checked,
                 first, listed, upl, uplCut, cutC, failC, upFail, uploadedAll,
                 flv, histv>>

\* A reconcile pass that ends at a failure inside index_bounded still
\* returns true (give_up records it; :1369), so main clears the owed pass.
PassFailedOwed == IF idxSrc = "recon" THEN ReconcileFailuresOwed
                  ELSE reconcileOwed

AfterIndex == IF idxSrc = "chunk" THEN "chunk"
              ELSE IF idxSrc = "retry" THEN "list" ELSE "end"

\* Where a pass's unindexed refs go: owed in pending_index_ (index_or_owe),
\* except a reconcile's, which stay in the bucket for the next pass.
Owe(S) == IF idxSrc = "recon" THEN pending ELSE pending \cup S

\* index_bounded (:1127-1271), one batch at a time.
IdxBatch ==
  /\ alive /\ cyc = "index"
  /\ IF idxQ = {}
       THEN /\ cyc' = AfterIndex
            \* reconcile() returned true: :701-704 clear the owed pass
            /\ reconcileOwed' = IF idxSrc = "recon" THEN FALSE
                                ELSE reconcileOwed
            /\ UNCHANGED <<idxQ, pending, catalog, cutC, failC, lease, losses,
                           faults, idxFirst, idxSplit, checked, pdBatches>>
       ELSE LET armed == runner = "flush" /\ pastDl IN
            IF ReadCancelled \/ (~idxFirst /\ armed /\ FirstBatchOnly)
              THEN \* :1182-1187 no batch starts: the rest is owed, deferred
                   /\ pending' = Owe(idxQ) /\ idxQ' = {} /\ cutC' = TRUE
                   /\ cyc' = AfterIndex
                   /\ UNCHANGED <<catalog, failC, lease, losses, faults,
                                  idxFirst, idxSplit, checked, pdBatches,
                                  reconcileOwed>>
              ELSE
              /\ idxFirst' = FALSE /\ checked' = FALSE
              /\ pdBatches' = IF armed THEN pdBatches + 1 ELSE pdBatches
              /\ \E b \in SUBSET idxQ:
                   /\ Cardinality(b) =
                        Min(IF idxSplit THEN 1 ELSE Chunk, Cardinality(idxQ))
                   /\ \/ \* committed (the replay guard skips what already is)
                         /\ lease = "held"
                         /\ catalog' = catalog \cup b /\ idxQ' = idxQ \ b
                         /\ cyc' = "index"
                         /\ UNCHANGED <<pending, cutC, failC, lease, losses,
                                        faults, idxSplit, reconcileOwed>>
                      \/ \* kBatchTooLarge: halves, back on the stack
                         /\ Cardinality(b) > 1 /\ ~idxSplit
                         /\ idxSplit' = TRUE /\ cyc' = "index"
                         /\ UNCHANGED <<idxQ, pending, catalog, cutC, failC,
                                        lease, losses, faults, reconcileOwed>>
                      \/ \* the store or the catalog failed: the pass ends
                         \* (give_up), and reconcile() still returns true
                         /\ faults < MaxFaults /\ faults' = faults + 1
                         /\ pending' = Owe(idxQ) /\ idxQ' = {} /\ failC' = TRUE
                         /\ cyc' = AfterIndex
                         /\ reconcileOwed' = PassFailedOwed
                         /\ UNCHANGED <<catalog, cutC, lease, losses, idxSplit>>
                      \/ \* an outcome-unknown publish: maybe landed; the
                         \* writer quarantines (drops the lease)
                         /\ lease = "held" /\ faults < MaxFaults
                         /\ losses < MaxLeaseLosses
                         /\ faults' = faults + 1
                         /\ lease' = "lost" /\ losses' = losses + 1
                         /\ \E landed \in BOOLEAN:
                              catalog' = IF landed THEN catalog \cup b
                                         ELSE catalog
                         /\ pending' = Owe(idxQ) /\ idxQ' = {} /\ failC' = TRUE
                         /\ cyc' = AfterIndex
                         /\ reconcileOwed' = PassFailedOwed
                         /\ UNCHANGED <<cutC, idxSplit>>
                      \/ \* commit() without a lease: kLease, rethrown; the
                         \* cycle ends in its catch (:722-728)
                         /\ lease = "lost"
                         /\ pending' = Owe(idxQ) /\ idxQ' = {} /\ failC' = TRUE
                         /\ cyc' = "end"
                         /\ UNCHANGED <<catalog, cutC, lease, losses, faults,
                                        idxSplit, reconcileOwed>>
  /\ UNCHANGED <<spool, s3, sinkv, procv, runner, catOk, first, listed, upl,
                 uplCut, done, idxSrc, upFail, uploadedAll, fl, flWant,
                 pastDl, pastGrace, flTarget, badChunkStart, flushReconciled,
                 remembered>>

\* Step 4 (:697-705): the loop's cycles only, with the lease, no cancel.
CycRecon ==
  /\ alive /\ cyc = "recon"
  /\ LET may == (runner = "loop" \/ FlushRunsReconcile) /\ catOk
                /\ ~UploadCancelled /\ (reconcileOwed \/ ReconcilePeriodic)
     IN IF may /\ (s3 \ catalog) # {}
          THEN /\ idxQ' = s3 \ catalog /\ idxSrc' = "recon" /\ idxFirst' = TRUE
               /\ idxSplit' = FALSE /\ cyc' = "index"
               /\ flushReconciled' = (flushReconciled \/ runner = "flush")
               /\ UNCHANGED reconcileOwed
          ELSE /\ cyc' = "end"
               /\ reconcileOwed' = (reconcileOwed /\ ~may)
               /\ UNCHANGED <<idxQ, idxSrc, idxFirst, idxSplit, flushReconciled>>
  /\ UNCHANGED <<durable, sinkv, procv, envv, pending, runner, catOk, checked,
                 first, listed, upl, uplCut, done, cutC, failC, upFail,
                 uploadedAll, flv, badChunkStart, remembered>>

\* :715-719 and flush()'s decision (:460-476).
CycEnd ==
  /\ alive /\ cyc = "end"
  /\ LET failed  == ~catOk \/ failC \/ upFail
         drained == uploadedAll /\ ~failed /\ ~cutC /\ pending = {}
     IN /\ IF runner = "flush"
             THEN IF drained
                    THEN fl' = "true" /\ flWant' = FALSE
                    ELSE IF pastDl
                           THEN fl' = "false" /\ flWant' = FALSE
                           ELSE flWant' = TRUE /\ UNCHANGED fl
             ELSE UNCHANGED <<fl, flWant>>
  /\ CycleReset
  /\ UNCHANGED <<durable, sinkv, procv, envv, memv, runner, pastDl,
                 pastGrace, pdBatches, flTarget, histv>>

-----------------------------------------------------------------------------
(* flush_and_wait() and engine.close()                                      *)

FlushBegin ==
  /\ alive /\ started /\ ~stopReq /\ fl # "running"
  /\ \/ cl = "storeflush" /\ cl' = "flushing"
     \/ cl = "run" /\ AllowUserFlush /\ UNCHANGED cl
  /\ fl' = "running" /\ flWant' = TRUE /\ pastDl' = FALSE /\ pastGrace' = FALSE
  /\ pdBatches' = 0
  \* Everything this process has staged by now (the sink's own flush came
  \* first): flush() returning true must have put it in the catalog.
  /\ flTarget' = bornHere \ (sinkMem \cup lostInSink)
  /\ UNCHANGED <<durable, sinkv, alive, started, restarts, crashes, stopReq,
                 envv, memv, cycv, histv>>

\* close(): seal the sink (its flush may time out), stop the ring, whose
\* release runs the backstop flush (may time out), then the storage flush,
\* then stop().
CloseBegin ==
  /\ AllowClose /\ alive /\ started /\ cl = "run" /\ fl # "running"
  /\ \/ /\ spool' = spool \cup sinkMem /\ sinkMem' = {}
        /\ sealedBefore' = TRUE
     \/ /\ sealedBefore' = FALSE /\ UNCHANGED <<spool, sinkMem>>
  /\ cl' = "ringstop"
  /\ UNCHANGED <<s3, catalog, born, bornHere, lostInSink, ring,
                 sealedOnRelease, alive, started, restarts, crashes, stopReq,
                 envv, memv, cycv, flv, histv>>

\* RingEngine::stop drains its worker into the sink (Admit stays enabled
\* until here), then releases it: on_engine_release.
RingRelease ==
  /\ alive /\ cl = "ringstop"
  /\ ring' = "released" /\ cl' = "storeflush"
  /\ IF ReleaseStagesTail
       THEN \/ /\ spool' = spool \cup sinkMem /\ sinkMem' = {}
               /\ sealedOnRelease' = TRUE
            \/ /\ sealedOnRelease' = FALSE /\ UNCHANGED <<spool, sinkMem>>
       ELSE /\ sealedOnRelease' = TRUE /\ UNCHANGED <<spool, sinkMem>>
  /\ UNCHANGED <<s3, catalog, born, bornHere, lostInSink, sealedBefore, alive,
                 started, restarts, crashes, stopReq, envv, memv, cycv, flv,
                 histv>>

CloseStop ==
  /\ alive /\ cl = "flushing" /\ fl \in {"true", "false"}
  /\ stopReq' = TRUE /\ cl' = "stop"
  /\ UNCHANGED <<durable, sinkv, alive, started, restarts, crashes, envv, memv,
                 cycv, flv, histv>>

\* stop() has joined the loop: the process is done with this service, and
\* what it still owed is now only in the bucket.
CloseDone ==
  /\ alive /\ cl = "stop" /\ cyc = "idle"
  /\ cl' = "done" /\ alive' = FALSE /\ started' = FALSE
  /\ remembered' = remembered + Cardinality(Remembered)
  /\ ResetMemory
  /\ UNCHANGED <<durable, sinkv, restarts, crashes, stopReq, envv,
                 badChunkStart, flushReconciled>>

-----------------------------------------------------------------------------

Next ==
  \/ LoseLease \/ Reacquire \/ PassDeadline \/ PassGrace
  \/ \E p \in Packs: Admit(p) \/ Stage(p)
  \/ StartService \/ Crash \/ Restart
  \/ LoopCycle \/ FlushCycle \/ FlushGivesUp
  \/ CycLease \/ CycRetry \/ CycList \/ CycChunk
  \/ \E p \in Packs: UploadOne(p)
  \/ UploadChunkDone \/ IdxBatch \/ CycRecon \/ CycEnd
  \/ FlushBegin \/ CloseBegin \/ RingRelease \/ CloseStop \/ CloseDone

\* Fairness: the service makes progress while it lives; a lost lease comes
\* back; a crashed process restarts; staged packs get staged.
Service ==
  \/ LoopCycle \/ CycLease \/ CycRetry \/ CycList \/ CycChunk
  \/ \E p \in Packs: UploadOne(p)
  \/ UploadChunkDone \/ IdxBatch \/ CycRecon \/ CycEnd \/ StartService
Spec == Init /\ [][Next]_vars

\* Fault-free progress: an attempt that succeeds is always among the
\* choices, so strong fairness on the service's step under a bounded
\* fault budget forces success eventually.
LiveSpec == Spec /\ WF_vars(Reacquire) /\ WF_vars(Restart)
            /\ SF_vars(Service) /\ \A p \in Packs: WF_vars(Stage(p))

-----------------------------------------------------------------------------
(* Properties                                                               *)

TypeOK ==
  /\ spool \subseteq Packs /\ s3 \subseteq Packs /\ catalog \subseteq Packs
  /\ pending \subseteq Packs /\ lease \in {"held", "lost"}
  /\ cyc \in {"idle", "lease", "retry", "list", "chunk", "upload", "index",
              "recon", "end"}

\* A pack the sink staged is never lost: in the spool, or in the bucket.
\* (A pack still in sink memory dies with the process: lostInSink.)
Durable == (born \ (sinkMem \cup lostInSink)) \subseteq (spool \cup s3)

\* Only uploaded packs are indexed.
CatalogSound == catalog \subseteq s3

\* Decision 7, as the code states it (header :237-246, :598-602): a cycle
\* checks the lease before each chunk it uploads, so no chunk STARTS while
\* the writer holds no lease.  (One in flight when the lease goes finishes:
\* the accepted window, bounded by OwedAtMostOneChunk.)
ChunkStartsWithLease == ~badChunkStart

\* At most one chunk is ever out of the spool and not in the catalog,
\* remembered by this process alone (:12-16, :471-472).
OwedAtMostOneChunk == Cardinality(Remembered) <= Chunk

\* The strict reading of decision 7: nothing only one process remembers is
\* ever lost with it.  NOT a claim of the code (it accepts one chunk in the
\* gap); a witness config shows it fails, and how.
NoOrphanEver == remembered = 0

\* Past a flush's deadline no index batch starts but the first (:338-344).
OneBatchPastDeadline == pdBatches <= 1

\* flush()'s cycles never reconcile (:456-460).
FlushNeverReconciles == ~flushReconciled

\* The release backstop: a sink reported sealed holds nothing in memory.
TailDelivered == sealedOnRelease => sinkMem = {}

\* flush() returning true: everything staged before it is in the catalog.
FlushBoundary == fl = "true" => flTarget \subseteq catalog

\* close() with both seals through and a drained flush: every record this
\* process admitted is in the catalog.
CloseDelivers ==
  (cl = "stop" /\ fl = "true" /\ sealedOnRelease)
    => (bornHere \ lostInSink) \subseteq catalog

\* Liveness: a staged pack ends up in the catalog.
EventuallyIndexed ==
  \A p \in Packs: (p \in spool \/ p \in s3) ~> (p \in catalog)

\* Vacuity guards: each must be VIOLATED (the run reaches the state).
VacUploaded    == s3 = {}
VacIndexed     == catalog = {}
VacFlushTrue   == fl # "true"
VacPastBatch   == pdBatches = 0
VacOwed        == pending = {}
VacTwoChunks   == ~(cyc = "upload" /\ ~first /\ s3 # {} /\ catalog # {})
VacClosed      == ~(cl = "stop" /\ fl = "true" /\ sealedOnRelease /\ catalog # {})
=============================================================================
