--------------------------- MODULE SpoolOwnership ---------------------------
(***************************************************************************)
(* ONE OWNER PER SPOOL DIRECTORY, AND ADOPTION OF A DEAD PROCESS'S SPOOL   *)
(* on projectdmx/dmi main @ 5b3b632 (#163).                                *)
(*                                                                         *)
(* SCOPE.  The rank directories of the section 2.3 layout under ONE        *)
(* catalog key on ONE node,                                                *)
(*     <base>/<catalog_key>/r<rank>-<incarnation>/                         *)
(* each created by one process (its "own" directory, fresh per process     *)
(* start), and the processes that write, drain, adopt and remove them:     *)
(*   - the engine's claim: SpoolOwnerLock::Acquire -> CreateLocked, the    *)
(*     directory built under a staging name with its locks held and        *)
(*     renamed into place (atomic to every observer);                      *)
(*   - the storage service: Recover() of its own directory at start(),     *)
(*     its uploads (only with the lease, nothing owed), its index pass,    *)
(*     and adoption: scan_siblings, begin_adoption (TryAdopt ->            *)
(*     LockInPlace, Spool::Open held_by_caller, BeginRecovery's sweep),    *)
(*     upload_adopted_round, finish_adoption (block_sibling or             *)
(*     ReleaseAndRemoveIfEmpty), let_go_of_adoption at stop();             *)
(*   - the sink: Stage = write .open, link to .ready, unlink .open;        *)
(*     create_directories(dir) first, which RECREATES a removed root with  *)
(*     no lock;                                                            *)
(*   - the engine's close(): stop the service, then release the claim      *)
(*     (ReleaseAndRemoveIfEmpty) only if the sink sealed, else keep it     *)
(*     held until the process exits (_keep_spool_claim_held);              *)
(*   - crash: the kernel drops every flock, files stay; a child forked     *)
(*     without exec closed its copies of the lock descriptors at fork      *)
(*     (pthread_atfork), so it keeps no lock.                              *)
(*                                                                         *)
(* THE LOCK.  flock(LOCK_EX) on <dir>/.owner.lock AND on <dir> itself.     *)
(* The lock file is modelled by inode, so a remover's unlink and a         *)
(* taker's O_CREAT of a NEW file at the same path are distinct; the        *)
(* directory lock is per directory.  LockInPlace's steps are separate      *)
(* actions: open (O_CREAT), flock the file, IsFileAt, flock the directory, *)
(* with the code's retry bounds (2 retries on EWOULDBLOCK).                *)
(*                                                                         *)
(* SOURCE LINES (5b3b632).                                                 *)
(*   native/csrc/store/spool.cpp                                           *)
(*     :607-614   FlockHeld (the probe)                                    *)
(*     :616-638   ReadSpoolOwner: file lock, else the directory's lock     *)
(*     :823-848   LockDirectory                                            *)
(*     :854-908   LockInPlace: open, flock, IsFileAt, LockDirectory        *)
(*     :914-994   CreateLocked: staging copy, renameat2(NOREPLACE)         *)
(*     :1040-1051 Release                                                  *)
(*     :1105-1121 TryAdopt                                                 *)
(*     :1123-1128 MarkBlocked                                              *)
(*     :1130-1171 ReleaseAndRemoveIfEmpty: walk, unlink lock file, rmdir   *)
(*     :1357-1407 ChargedSiblingBytes                                      *)
(*     :1502-1754 Stage (create_directories :1680, link :1690)             *)
(*     :1756-1781 Recover / BeginRecovery: unlink every .open not in      *)
(*                THIS object's inflight_temps_ (ListReadyLocked :1846)    *)
(*     :1883-1936 Remove (after a verified upload)                         *)
(*   native/csrc/catalog/storage_service.cpp                               *)
(*     :311-336   sweep_and_reconcile_at_start: Recover, adoption owed     *)
(*     :605-678   run_cycle step 2/3: upload only with lease, nothing owed *)
(*     :686-692   run_cycle step 3a: adopt only after own uploads          *)
(*     :800-814   let_go_of_adoption, adoption_owed                        *)
(*     :821-908   adopt_step: scan when owed or recheck_due (:828-832)     *)
(*     :910-959   scan_siblings: probe each rank dir, queue dead ones,     *)
(*                live_siblings_ = (live != 0)                             *)
(*     :961-994   begin_adoption                                           *)
(*     :996-1058  upload_adopted_round                                     *)
(*     :1060-1081 finish_adoption                                          *)
(*     :1083-1101 block_sibling                                            *)
(*   src/dmi/engine.py                                                     *)
(*     :495-553   _start_capture_storage: claim, service, start()          *)
(*     :579-599   the sink opens held_by_caller AFTER the service started  *)
(*     :955-990   close(): flush, stop, release only if sealed            *)
(*                                                                         *)
(* ABSTRACTIONS.  Pack contents, hashing and validation (quarantine of a   *)
(* corrupt ready file) are not modelled; a "Bad" pack is one no upload by  *)
(* any service can ever take (UploadFailure::retryable false).  The probe  *)
(* is atomic (the real probe takes the lock for microseconds, which can    *)
(* only make a concurrent take fail, never succeed).  One chunk is one     *)
(* pack.  The lease is a boolean the environment can take away a bounded   *)
(* number of times (a quarantine) and the service takes back.  Time        *)
(* (adoption_recheck_interval_ns, adoption_slice_ns) is abstracted away:   *)
(* a recheck that is due can happen at any moment.  Only node-local        *)
(* filesystems are modelled (a shared one is refused by statfs).  The flat *)
(* spool_root modes (sink-only, explicit record_sink) and the nesting      *)
(* check are not modelled.                                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  Procs,            \* process incarnations; each owns the directory named
                    \* after it (the rank directory r<rank>-<incarnation>)
  Stagers,          \* the processes whose sink stages a pack (a reduction:
                    \* the others capture nothing)
  Adopters,         \* the processes whose service adopts (a reduction: in
                    \* the engine every service does)
  NoP,              \* "nobody"
  NoPack,
  CrashProcs,       \* the processes that may crash
  Survivor,         \* a process that never crashes or closes (liveness), or NoP
  BadProcs,         \* processes whose pack no service can ever upload
  MaxQuar,          \* how many times the environment may take a lease away
  ReconcileOnStart, \* reconcile_on_start: start() indexes what the bucket holds
  \* TRUE: today's single publisher per (database, table_prefix) -- and so
  \* per catalog key: one service holds the lease at a time; a start() that
  \* cannot get it fails, and its engine releases the claim.  FALSE: every
  \* service may hold a lease of its own at once (several services serving
  \* one catalog key, as upload-only services per rank would).
  ExclusiveLease,
  \* The fix under test for the recheck gap: look at the siblings on every
  \* recheck, not only while the last look found a live one.
  RescanAlways,
  \* MUTATIONS (each FALSE in the real protocol)
  SkipFileCheck,    \* LockInPlace without IsFileAt
  SkipDirLock,      \* LockInPlace without LockDirectory
  SinkFirst,        \* the sink opens before the service's start() sweep
  ReleaseUnsealed,  \* close() releases the claim even if the sink did not seal
  NoAtfork,         \* a child forked without exec keeps the lock
  NoLeaseGate,      \* uploads start without the lease
  Track             \* record the vacuity witnesses' events in hist (they
                    \* multiply the state space, so only the _vac configs do)

Packs == Procs                    \* one pack per process, staged into its own dir
Home(pk) == pk
BadPacks == BadProcs
Dirs == Procs
Inos == 1..(Cardinality(Procs) + 3)

VARIABLES
  alive, ph, sinkOpen, sealed, child, lease, nQuar, leaseHolder,
  dirEx, lf, fl, dl, nIno,
  ownHeld, ownIno, oSt,
  pst, inStore, inCat, pend, upl,
  aSt, aDir, aFd, aTry, aList, aBad, queue, scanOwed, liveSib, blocked,
  blockMark, clobbered, hist

procVars == <<alive, ph, sinkOpen, sealed, child, lease, nQuar, leaseHolder>>
lockVars == <<dirEx, lf, fl, dl, nIno>>
ownVars  == <<ownHeld, ownIno, oSt>>
packVars == <<pst, inStore, inCat, pend, upl>>
adVars   == <<aSt, aDir, aFd, aTry, aList, aBad, queue, scanOwed, liveSib, blocked>>
vars == <<procVars, lockVars, ownVars, packVars, adVars, blockMark, clobbered, hist>>

HeldStates == {"sweep", "held", "rmcheck", "unlink", "rmdir"}

\* ReadSpoolOwner (spool.cpp:616-638): the lock file's flock, else the
\* directory's own.
Held(d) == (lf[d] # 0 /\ fl[lf[d]] # NoP) \/ dl[d] # NoP

FilesIn(d) == {pk \in Packs : Home(pk) = d /\ pst[pk] \in {"open", "ready"}}

Init ==
  /\ alive = [p \in Procs |-> TRUE]
  /\ ph = [p \in Procs |-> "new"]
  /\ sinkOpen = [p \in Procs |-> FALSE]
  /\ sealed = [p \in Procs |-> FALSE]
  /\ child = [p \in Procs |-> FALSE]
  /\ lease = [p \in Procs |-> FALSE]
  /\ nQuar = 0
  /\ leaseHolder = NoP
  /\ dirEx = [d \in Dirs |-> FALSE]
  /\ lf = [d \in Dirs |-> 0]
  /\ fl = [i \in Inos |-> NoP]
  /\ dl = [d \in Dirs |-> NoP]
  /\ nIno = 0
  /\ ownHeld = [p \in Procs |-> FALSE]
  /\ ownIno = [p \in Procs |-> 0]
  /\ oSt = [p \in Procs |-> "none"]
  /\ pst = [pk \in Packs |-> "unborn"]
  /\ inStore = [pk \in Packs |-> FALSE]
  /\ inCat = [pk \in Packs |-> FALSE]
  /\ pend = [p \in Procs |-> {}]
  /\ upl = [p \in Procs |-> NoPack]
  /\ aSt = [p \in Procs |-> "idle"]
  /\ aDir = [p \in Procs |-> NoP]
  /\ aFd = [p \in Procs |-> 0]
  /\ aTry = [p \in Procs |-> 0]
  /\ aList = [p \in Procs |-> {}]
  /\ aBad = [p \in Procs |-> FALSE]
  /\ queue = [p \in Procs |-> {}]
  /\ scanOwed = [p \in Procs |-> FALSE]
  /\ liveSib = [p \in Procs |-> FALSE]
  /\ blocked = [p \in Procs |-> {}]
  /\ blockMark = [d \in Dirs |-> FALSE]
  /\ clobbered = FALSE
  /\ hist = {}

Note(tag) == IF Track THEN hist \cup {tag} ELSE hist

\* The publisher lease: free for p to take.  Exclusive: nobody holds it
\* (a crashed or quarantined holder's row has lapsed: a TTL, abstracted).
LeaseFree(p) == ~ExclusiveLease \/ leaseHolder \in {NoP, p}
TakeLease(p) == leaseHolder' = IF ExclusiveLease THEN p ELSE leaseHolder
DropLease(p) == leaseHolder' = IF leaseHolder = p THEN NoP ELSE leaseHolder

\* Release every flock p holds except those on `keepIno` / `keepDir`.
ReleaseExcept(p, keepIno, keepDir) ==
  /\ fl' = [i \in Inos |-> IF fl[i] = p /\ i # keepIno THEN NoP ELSE fl[i]]
  /\ dl' = [d \in Dirs |-> IF dl[d] = p /\ d # keepDir THEN NoP ELSE dl[d]]

ReleaseAll(p) == ReleaseExcept(p, 0, NoP)

\* Recover()/BeginRecovery() on d: every .open file not in this Spool
\* object's inflight_temps_ is unlinked -- including a sink's in the same
\* process, which is a different Spool object.  The writer of d's packs is
\* the process d; an .open file of a live writer is a stage in flight.
SweepOpen(d) ==
  /\ pst' = [pk \in Packs |-> IF Home(pk) = d /\ pst[pk] = "open"
                                THEN "swept" ELSE pst[pk]]
  /\ clobbered' = (clobbered \/
                   \E pk \in Packs : Home(pk) = d /\ pst[pk] = "open" /\ alive[d])

ResetLocal(p) ==
  /\ upl' = [upl EXCEPT ![p] = NoPack]
  /\ pend' = [pend EXCEPT ![p] = {}]
  /\ aSt' = [aSt EXCEPT ![p] = "idle"]
  /\ aDir' = [aDir EXCEPT ![p] = NoP]
  /\ aFd' = [aFd EXCEPT ![p] = 0]
  /\ aTry' = [aTry EXCEPT ![p] = 0]
  /\ aList' = [aList EXCEPT ![p] = {}]
  /\ aBad' = [aBad EXCEPT ![p] = FALSE]
  /\ queue' = [queue EXCEPT ![p] = {}]

-----------------------------------------------------------------------------
(* Process lifecycle *)

\* claim_spool_directory -> SpoolOwnerLock::Acquire -> CreateLocked: the
\* directory appears already locked (staging copy + renameat2 NOREPLACE).
Claim(p) ==
  /\ alive[p] /\ ph[p] = "new" /\ nIno < Cardinality(Inos)
  /\ ph' = [ph EXCEPT ![p] = "claimed"]
  /\ dirEx' = [dirEx EXCEPT ![p] = TRUE]
  /\ nIno' = nIno + 1
  /\ lf' = [lf EXCEPT ![p] = nIno + 1]
  /\ fl' = [fl EXCEPT ![nIno + 1] = p]
  /\ dl' = [dl EXCEPT ![p] = p]
  /\ ownHeld' = [ownHeld EXCEPT ![p] = TRUE]
  /\ ownIno' = [ownIno EXCEPT ![p] = nIno + 1]
  /\ UNCHANGED <<alive, sinkOpen, sealed, child, lease, nQuar, leaseHolder, oSt,
                 packVars, adVars, blockMark, clobbered, hist>>

\* start(): lease, then Recover() of its own directory, the reconcile, and
\* adoption owed from the loop's first cycle (storage_service.cpp:311-336).
Start(p) ==
  /\ alive[p] /\ ph[p] = "claimed" /\ LeaseFree(p)
  /\ ph' = [ph EXCEPT ![p] = "serving"]
  /\ lease' = [lease EXCEPT ![p] = TRUE]
  /\ TakeLease(p)
  /\ SweepOpen(p)
  /\ inCat' = IF ReconcileOnStart
                THEN [pk \in Packs |-> inCat[pk] \/ inStore[pk]] ELSE inCat
  /\ scanOwed' = [scanOwed EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<alive, sinkOpen, sealed, child, nQuar, lockVars, ownVars,
                 inStore, pend, upl, aSt, aDir, aFd, aTry, aList, aBad, queue,
                 liveSib, blocked, blockMark, hist>>

\* start() cannot take the lease another service holds (acquire_lease_at_start
\* throws kHeld after start_lease_wait_ns): the engine releases the claim
\* (engine.py:546-548) -- a directory holding nothing is removed.
StartFail(p) ==
  /\ alive[p] /\ ph[p] = "claimed" /\ ~LeaseFree(p) /\ p # Survivor
  /\ ph' = [ph EXCEPT ![p] = "closed"]
  /\ oSt' = [oSt EXCEPT ![p] = "check"]
  /\ UNCHANGED <<alive, sinkOpen, sealed, child, lease, nQuar, leaseHolder,
                 lockVars, ownHeld, ownIno, packVars, adVars, blockMark,
                 clobbered, hist>>

\* engine.py:579-599: the sink opens held_by_caller after the service's
\* start() (mutation SinkFirst: before it).
OpenSink(p) ==
  /\ alive[p] /\ ~sinkOpen[p]
  /\ \/ ph[p] = "serving"
     \/ SinkFirst /\ ph[p] = "claimed"
  /\ sinkOpen' = [sinkOpen EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<alive, ph, sealed, child, lease, nQuar, leaseHolder, lockVars, ownVars,
                 packVars, adVars, blockMark, clobbered, hist>>

\* Stage, first half: create_directories(dir) (recreating a removed root,
\* with no lock), then the .open temp.
StageOpen(p) ==
  /\ p \in Stagers /\ alive[p] /\ sinkOpen[p] /\ ~sealed[p] /\ pst[p] = "unborn"
  /\ pst' = [pst EXCEPT ![p] = "open"]
  /\ dirEx' = [dirEx EXCEPT ![p] = TRUE]
  /\ hist' = IF dirEx[p] THEN hist ELSE Note("recreated")
  /\ UNCHANGED <<procVars, lf, fl, dl, nIno, ownVars, inStore, inCat, pend, upl,
                 adVars, blockMark, clobbered>>

\* Stage, second half: link(temp, ready), unlink(temp).  A swept temp fails
\* the link (ENOENT) and the stage reports kIo: the pack is lost to capture.
StageLink(p) ==
  /\ alive[p] /\ pst[p] = "open"
  /\ pst' = [pst EXCEPT ![p] = "ready"]
  /\ UNCHANGED <<procVars, lockVars, ownVars, inStore, inCat, pend, upl,
                 adVars, blockMark, clobbered, hist>>

Seal(p) ==
  /\ alive[p] /\ sinkOpen[p] /\ ~sealed[p] /\ pst[p] # "open"
  /\ sealed' = [sealed EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<alive, ph, sinkOpen, child, lease, nQuar, leaseHolder, lockVars, ownVars,
                 packVars, adVars, blockMark, clobbered, hist>>

\* Only the NoAtfork mutation gives a child anything to keep; in the real
\* protocol a fork is a no-op for the lock, so it is not taken at all.
Fork(p) ==
  /\ NoAtfork /\ alive[p] /\ ph[p] # "new" /\ ~child[p]
  /\ child' = [child EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<alive, ph, sinkOpen, sealed, lease, nQuar, leaseHolder, lockVars, ownVars,
                 packVars, adVars, blockMark, clobbered, hist>>

\* Death, by crash or by exit after close().  The kernel drops every flock
\* the process held -- unless (mutation NoAtfork) a child forked without
\* exec kept a copy of the descriptions.  In-memory state (pend) is lost.
Die(p) ==
  /\ alive' = [alive EXCEPT ![p] = FALSE]
  /\ DropLease(p)
  /\ IF NoAtfork /\ child[p] THEN UNCHANGED <<fl, dl>> ELSE ReleaseAll(p)
  /\ ResetLocal(p)
  /\ UNCHANGED <<ph, sinkOpen, sealed, child, lease, nQuar, dirEx, lf, nIno,
                 ownVars, pst, inStore, inCat, scanOwed, liveSib, blocked,
                 blockMark, clobbered, hist>>

Crash(p) ==
  /\ p \in CrashProcs /\ alive[p] /\ ph[p] # "new" /\ p # Survivor
  /\ Die(p)

Exit(p) ==
  /\ alive[p] /\ ph[p] = "closed" /\ oSt[p] \in {"done", "kept"} /\ p # Survivor
  /\ Die(p)

\* close(): flush, stop() (let_go_of_adoption: the adoption's lock goes;
\* pend_index_ is dropped), then release the claim only if the sink sealed.
Close(p) ==
  /\ alive[p] /\ ph[p] = "serving" /\ p # Survivor /\ upl[p] = NoPack
  /\ ph' = [ph EXCEPT ![p] = "closed"]
  /\ DropLease(p)
  /\ lease' = [lease EXCEPT ![p] = FALSE]
  /\ ReleaseExcept(p, ownIno[p], p)
  /\ oSt' = [oSt EXCEPT ![p] = IF sealed[p] \/ ReleaseUnsealed THEN "check" ELSE "kept"]
  /\ ResetLocal(p)
  /\ UNCHANGED <<alive, sinkOpen, sealed, child, nQuar, dirEx, lf, nIno,
                 ownHeld, ownIno, pst, inStore, inCat, scanOwed, liveSib,
                 blocked, blockMark, clobbered, hist>>

\* ReleaseAndRemoveIfEmpty on the process's own directory (SpoolClaim.release).
OwnRmCheck(p) ==
  /\ alive[p] /\ oSt[p] = "check"
  /\ IF FilesIn(p) # {}
       THEN /\ fl' = [fl EXCEPT ![ownIno[p]] = NoP]
            /\ dl' = [dl EXCEPT ![p] = NoP]
            /\ ownHeld' = [ownHeld EXCEPT ![p] = FALSE]
            /\ oSt' = [oSt EXCEPT ![p] = "done"]
            /\ UNCHANGED lf
       ELSE /\ oSt' = [oSt EXCEPT ![p] = "unlink"]
            /\ UNCHANGED <<fl, dl, ownHeld, lf>>
  /\ UNCHANGED <<procVars, dirEx, nIno, ownIno, packVars, adVars, blockMark,
                 clobbered, hist>>

OwnRmUnlink(p) ==
  /\ alive[p] /\ oSt[p] = "unlink"
  /\ lf' = IF lf[p] = ownIno[p] THEN [lf EXCEPT ![p] = 0] ELSE lf
  /\ oSt' = [oSt EXCEPT ![p] = "rmdir"]
  /\ UNCHANGED <<procVars, dirEx, fl, dl, nIno, ownHeld, ownIno, packVars,
                 adVars, blockMark, clobbered, hist>>

OwnRmRmdir(p) ==
  /\ alive[p] /\ oSt[p] = "rmdir"
  /\ dirEx' = IF lf[p] = 0 /\ FilesIn(p) = {}
                THEN [dirEx EXCEPT ![p] = FALSE] ELSE dirEx
  /\ fl' = [fl EXCEPT ![ownIno[p]] = IF fl[ownIno[p]] = p THEN NoP ELSE fl[ownIno[p]]]
  /\ dl' = [dl EXCEPT ![p] = IF dl[p] = p THEN NoP ELSE dl[p]]
  /\ ownHeld' = [ownHeld EXCEPT ![p] = FALSE]
  /\ oSt' = [oSt EXCEPT ![p] = "done"]
  /\ UNCHANGED <<procVars, lf, nIno, ownIno, packVars, adVars, blockMark,
                 clobbered, hist>>

-----------------------------------------------------------------------------
(* The catalog: quarantine, uploads, index *)

Quarantine(p) ==
  /\ alive[p] /\ lease[p] /\ nQuar < MaxQuar
  /\ lease' = [lease EXCEPT ![p] = FALSE]
  /\ nQuar' = nQuar + 1
  /\ DropLease(p)
  /\ UNCHANGED <<alive, ph, sinkOpen, sealed, child, lockVars, ownVars,
                 packVars, adVars, blockMark, clobbered, hist>>

Regain(p) ==
  /\ alive[p] /\ ~lease[p] /\ ph[p] = "serving" /\ LeaseFree(p)
  /\ lease' = [lease EXCEPT ![p] = TRUE]
  /\ TakeLease(p)
  /\ UNCHANGED <<alive, ph, sinkOpen, sealed, child, nQuar, lockVars, ownVars,
                 packVars, adVars, blockMark, clobbered, hist>>

UploadGate(p) ==
  /\ alive[p] /\ ph[p] = "serving" /\ upl[p] = NoPack
  /\ lease[p] \/ NoLeaseGate
  /\ pend[p] = {}

\* run_cycle step 2: ListPending of its own directory, one chunk.
OwnUploadBegin(p) ==
  /\ UploadGate(p) /\ ownHeld[p] /\ pst[p] = "ready"
  /\ upl' = [upl EXCEPT ![p] = p]
  /\ hist' = IF lease[p] THEN hist ELSE hist \cup {"nolease"}
  /\ UNCHANGED <<procVars, lockVars, ownVars, pst, inStore, inCat, pend,
                 adVars, blockMark, clobbered>>

\* upload_adopted_round: the lease and nothing owed, checked before every round.
AdoptUploadBegin(p) ==
  /\ UploadGate(p) /\ aSt[p] = "held"
  /\ \E pk \in aList[p] :
       /\ pst[pk] = "ready"
       /\ upl' = [upl EXCEPT ![p] = pk]
  /\ hist' = IF lease[p] THEN hist ELSE hist \cup {"nolease"}
  /\ UNCHANGED <<procVars, lockVars, ownVars, pst, inStore, inCat, pend,
                 adVars, blockMark, clobbered>>

\* The transfer, its verification and Spool::Remove: gone from the spool,
\* owed to the catalog in pending_index_.  The lease may have been lost since
\* the chunk began: a chunk in flight finishes.  A Bad pack fails
\* non-retryably and stays staged; an adoption then blocks the directory.
UploadEnd(p) ==
  /\ alive[p] /\ upl[p] # NoPack
  /\ LET pk == upl[p] IN
     /\ IF pk \in BadPacks
          THEN /\ UNCHANGED <<pst, inStore, pend, hist>>
               /\ aBad' = IF pk # p THEN [aBad EXCEPT ![p] = TRUE] ELSE aBad
          ELSE /\ pst' = [pst EXCEPT ![pk] = "gone"]
               /\ inStore' = [inStore EXCEPT ![pk] = TRUE]
               /\ pend' = [pend EXCEPT ![p] = pend[p] \cup {pk}]
               /\ hist' = IF pk # p THEN Note("adopted") ELSE hist
               /\ UNCHANGED aBad
     /\ aList' = [aList EXCEPT ![p] = aList[p] \ {pk}]
  /\ upl' = [upl EXCEPT ![p] = NoPack]
  /\ UNCHANGED <<procVars, lockVars, ownVars, inCat, aSt, aDir, aFd, aTry,
                 queue, scanOwed, liveSib, blocked, blockMark, clobbered>>

Index(p) ==
  /\ alive[p] /\ ph[p] = "serving" /\ lease[p]
  /\ \E pk \in pend[p] :
       /\ inCat' = [inCat EXCEPT ![pk] = TRUE]
       /\ pend' = [pend EXCEPT ![p] = pend[p] \ {pk}]
  /\ UNCHANGED <<procVars, lockVars, ownVars, pst, inStore, upl, adVars,
                 blockMark, clobbered, hist>>

-----------------------------------------------------------------------------
(* Adoption (storage_service.cpp:821-1101) *)

\* adopt_step runs in the loop's cycles only after every own staged pack is
\* up (run_cycle :686-692), in a cycle that held the lease at its start: a
\* look (Scan) and a new take (Take) need the lease; the steps of a take
\* under way go on if it is lost meanwhile (an over-approximation of a
\* slice that outlives the lease).
AdoptGate(p) ==
  /\ alive[p] /\ ph[p] = "serving" /\ upl[p] = NoPack
  /\ pst[p] # "ready" \/ p \in BadPacks

\* adopt_step :825-834 and scan_siblings :910-959.  A look is due when owed
\* (from start() on) or, while the last look found a live sibling, on the
\* recheck interval (abstracted: any time).
Scan(p) ==
  /\ p \in Adopters /\ AdoptGate(p) /\ lease[p] /\ aSt[p] = "idle" /\ queue[p] = {}
  /\ scanOwed[p] \/ liveSib[p] \/ RescanAlways
  /\ queue' = [queue EXCEPT ![p] =
                 {d \in Dirs \ {p} : dirEx[d] /\ d \notin blocked[p] /\ ~Held(d)}]
  /\ liveSib' = [liveSib EXCEPT ![p] =
                   \E d \in Dirs \ {p} : dirEx[d] /\ d \notin blocked[p] /\ Held(d)]
  /\ scanOwed' = [scanOwed EXCEPT ![p] = FALSE]
  /\ UNCHANGED <<procVars, lockVars, ownVars, packVars, aSt, aDir, aFd, aTry,
                 aList, aBad, blocked, blockMark, clobbered, hist>>

Take(p) ==
  /\ AdoptGate(p) /\ lease[p] /\ aSt[p] = "idle"
  /\ \E d \in queue[p] :
       /\ queue' = [queue EXCEPT ![p] = queue[p] \ {d}]
       /\ aDir' = [aDir EXCEPT ![p] = d]
       /\ aSt' = [aSt EXCEPT ![p] = IF dirEx[d] THEN "open" ELSE "idle"]
       /\ aTry' = [aTry EXCEPT ![p] = 0]
  /\ UNCHANGED <<procVars, lockVars, ownVars, packVars, aFd, aList, aBad,
                 scanOwed, liveSib, blocked, blockMark, clobbered, hist>>

GiveUp(p) ==   \* begin_adoption returns without a lock
  /\ aSt' = [aSt EXCEPT ![p] = "idle"]
  /\ aDir' = [aDir EXCEPT ![p] = NoP]
  /\ aFd' = [aFd EXCEPT ![p] = 0]

\* LockInPlace: open(O_CREAT) of <dir>/.owner.lock.  A directory removed
\* meanwhile: TryAdopt's realpath fails, or the open does (kIo / kBadArgument)
\* -- "another adopter drained and removed it", nothing owed.
LkOpen(p) ==
  /\ alive[p] /\ aSt[p] = "open"
  /\ LET d == aDir[p] IN
     IF ~dirEx[d]
       THEN /\ GiveUp(p) /\ UNCHANGED <<lf, nIno, aTry>>
       ELSE IF lf[d] = 0
         THEN /\ nIno < Cardinality(Inos)
              /\ nIno' = nIno + 1
              /\ lf' = [lf EXCEPT ![d] = nIno + 1]
              /\ aFd' = [aFd EXCEPT ![p] = nIno + 1]
              /\ aSt' = [aSt EXCEPT ![p] = "flock"]
              /\ UNCHANGED <<aDir, aTry>>
         ELSE /\ aFd' = [aFd EXCEPT ![p] = lf[d]]
              /\ aSt' = [aSt EXCEPT ![p] = "flock"]
              /\ UNCHANGED <<lf, nIno, aDir, aTry>>
  /\ UNCHANGED <<procVars, dirEx, fl, dl, ownVars, packVars, aList, aBad, queue,
                 scanOwed, liveSib, blocked, blockMark, clobbered, hist>>

\* flock(LOCK_EX|LOCK_NB) on the file.  EWOULDBLOCK: retried twice after
\* 2 ms (attempt < 2), then kOwned -- "alive after all" (live_siblings_).
LkFlock(p) ==
  /\ alive[p] /\ aSt[p] = "flock"
  /\ IF fl[aFd[p]] = NoP
       THEN /\ fl' = [fl EXCEPT ![aFd[p]] = p]
            /\ aSt' = [aSt EXCEPT ![p] = "check"]
            /\ UNCHANGED <<aDir, aFd, aTry, liveSib>>
       ELSE /\ UNCHANGED fl
            /\ IF aTry[p] < 2
                 THEN /\ aSt' = [aSt EXCEPT ![p] = "open"]
                      /\ aTry' = [aTry EXCEPT ![p] = aTry[p] + 1]
                      /\ UNCHANGED <<aDir, aFd, liveSib>>
                 ELSE /\ GiveUp(p) /\ UNCHANGED aTry
                      /\ liveSib' = [liveSib EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<procVars, dirEx, lf, dl, nIno, ownVars, packVars, aList, aBad,
                 queue, scanOwed, blocked, blockMark, clobbered, hist>>

\* IsFileAt: the locked file is still the one at the path.  If not, the
\* lock is let go and the take starts over (`continue`), or kIo once the
\* directory is gone.  (8 loops in the code; 3 here.)
LkCheck(p) ==
  /\ alive[p] /\ aSt[p] = "check"
  /\ LET d == aDir[p] IN
     IF SkipFileCheck \/ lf[d] = aFd[p]
       THEN /\ aSt' = [aSt EXCEPT ![p] = "dir"]
            /\ UNCHANGED <<fl, aDir, aFd, aTry, hist>>
       ELSE /\ fl' = [fl EXCEPT ![aFd[p]] = NoP]
            /\ hist' = Note("mismatch")
            /\ IF dirEx[d] /\ aTry[p] < 2
                 THEN /\ aSt' = [aSt EXCEPT ![p] = "open"]
                      /\ aTry' = [aTry EXCEPT ![p] = aTry[p] + 1]
                      /\ UNCHANGED <<aDir, aFd>>
                 ELSE /\ GiveUp(p) /\ UNCHANGED aTry
  /\ UNCHANGED <<procVars, dirEx, lf, dl, nIno, ownVars, packVars, aList, aBad,
                 queue, scanOwed, liveSib, blocked, blockMark, clobbered>>

\* LockDirectory: flock on the directory itself.
LkDir(p) ==
  /\ alive[p] /\ aSt[p] = "dir"
  /\ LET d == aDir[p] IN
     IF ~dirEx[d]
       THEN /\ fl' = [fl EXCEPT ![aFd[p]] = NoP]
            /\ GiveUp(p) /\ UNCHANGED <<dl, aTry, liveSib>>
       ELSE IF SkipDirLock \/ dl[d] = NoP
         THEN /\ dl' = IF SkipDirLock THEN dl ELSE [dl EXCEPT ![d] = p]
              /\ aSt' = [aSt EXCEPT ![p] = "sweep"]
              /\ UNCHANGED <<fl, aDir, aFd, aTry, liveSib>>
         ELSE /\ fl' = [fl EXCEPT ![aFd[p]] = NoP]
              /\ UNCHANGED dl
              /\ IF aTry[p] < 2
                   THEN /\ aSt' = [aSt EXCEPT ![p] = "open"]
                        /\ aTry' = [aTry EXCEPT ![p] = aTry[p] + 1]
                        /\ UNCHANGED <<aDir, aFd, liveSib>>
                   ELSE /\ GiveUp(p) /\ UNCHANGED aTry
                        /\ liveSib' = [liveSib EXCEPT ![p] = TRUE]
  /\ UNCHANGED <<procVars, dirEx, lf, nIno, ownVars, packVars, aList, aBad,
                 queue, scanOwed, blocked, blockMark, clobbered, hist>>

\* Spool::Open(held_by_caller) + BeginRecovery: sweep the .open files, list
\* the ready packs (validated a pack a step later; not modelled).
AdSweep(p) ==
  /\ alive[p] /\ aSt[p] = "sweep"
  /\ LET d == aDir[p] IN
     /\ SweepOpen(d)
     /\ aList' = [aList EXCEPT ![p] =
                    {pk \in Packs : Home(pk) = d /\ pst[pk] = "ready"}]
  /\ aBad' = [aBad EXCEPT ![p] = FALSE]
  /\ aSt' = [aSt EXCEPT ![p] = "held"]
  /\ hist' = Note("took")
  /\ UNCHANGED <<procVars, lockVars, ownVars, inStore, inCat, pend, upl, aDir,
                 aFd, aTry, queue, scanOwed, liveSib, blocked, blockMark>>

\* finish_adoption :1060-1081.
Finish(p) ==
  /\ AdoptGate(p) /\ aSt[p] = "held" /\ aList[p] = {}
  /\ LET d == aDir[p] IN
     IF aBad[p]
       THEN \* block_sibling with the lock: MarkBlocked, Release.
            /\ blockMark' = [blockMark EXCEPT ![d] = TRUE]
            /\ fl' = [fl EXCEPT ![aFd[p]] = NoP]
            /\ dl' = [dl EXCEPT ![d] = IF dl[d] = p THEN NoP ELSE dl[d]]
            /\ blocked' = [blocked EXCEPT ![p] = blocked[p] \cup {d}]
            /\ GiveUp(p)
            /\ hist' = Note("blocked")
       ELSE /\ aSt' = [aSt EXCEPT ![p] = "rmcheck"]
            /\ UNCHANGED <<fl, dl, blocked, blockMark, aDir, aFd, hist>>
  /\ UNCHANGED <<procVars, dirEx, lf, nIno, ownVars, packVars, aTry, aList, aBad,
                 queue, scanOwed, liveSib, clobbered>>

\* ReleaseAndRemoveIfEmpty :1130-1171: walk; anything but the lock file
\* keeps the directory (Release, block_sibling without the lock).
RmCheck(p) ==
  /\ alive[p] /\ aSt[p] = "rmcheck"
  /\ LET d == aDir[p] IN
     IF FilesIn(d) # {}
       THEN /\ fl' = [fl EXCEPT ![aFd[p]] = NoP]
            /\ dl' = [dl EXCEPT ![d] = IF dl[d] = p THEN NoP ELSE dl[d]]
            /\ blocked' = [blocked EXCEPT ![p] = blocked[p] \cup {d}]
            /\ GiveUp(p)
       ELSE /\ aSt' = [aSt EXCEPT ![p] = "unlink"]
            /\ UNCHANGED <<fl, dl, blocked, aDir, aFd>>
  /\ UNCHANGED <<procVars, dirEx, lf, nIno, ownVars, packVars, aTry, aList, aBad,
                 queue, scanOwed, liveSib, blockMark, clobbered, hist>>

RmUnlink(p) ==
  /\ alive[p] /\ aSt[p] = "unlink"
  /\ lf' = IF lf[aDir[p]] = aFd[p] THEN [lf EXCEPT ![aDir[p]] = 0] ELSE lf
  /\ aSt' = [aSt EXCEPT ![p] = "rmdir"]
  /\ UNCHANGED <<procVars, dirEx, fl, dl, nIno, ownVars, packVars, aDir, aFd,
                 aTry, aList, aBad, queue, scanOwed, liveSib, blocked, blockMark,
                 clobbered, hist>>

\* rmdir: fails ENOTEMPTY if a taker created a new lock file meanwhile (or
\* a sink staged); finish_adoption then blocks it (without a mark).
RmRmdir(p) ==
  /\ alive[p] /\ aSt[p] = "rmdir"
  /\ LET d == aDir[p] IN
     /\ IF lf[d] = 0 /\ FilesIn(d) = {}
          THEN /\ dirEx' = [dirEx EXCEPT ![d] = FALSE]
               /\ hist' = Note("removed")
               /\ UNCHANGED blocked
          ELSE /\ blocked' = [blocked EXCEPT ![p] = blocked[p] \cup {d}]
               /\ UNCHANGED <<dirEx, hist>>
     /\ fl' = [fl EXCEPT ![aFd[p]] = IF fl[aFd[p]] = p THEN NoP ELSE fl[aFd[p]]]
     /\ dl' = [dl EXCEPT ![d] = IF dl[d] = p THEN NoP ELSE dl[d]]
  /\ GiveUp(p)
  /\ UNCHANGED <<procVars, lf, nIno, ownVars, packVars, aTry, aList, aBad,
                 queue, scanOwed, liveSib, blockMark, clobbered>>

-----------------------------------------------------------------------------

Next ==
  \E p \in Procs :
    \/ Claim(p) \/ Start(p) \/ StartFail(p) \/ OpenSink(p) \/ StageOpen(p) \/ StageLink(p)
    \/ Seal(p) \/ Fork(p) \/ Crash(p) \/ Exit(p) \/ Close(p)
    \/ OwnRmCheck(p) \/ OwnRmUnlink(p) \/ OwnRmRmdir(p)
    \/ Quarantine(p) \/ Regain(p)
    \/ OwnUploadBegin(p) \/ AdoptUploadBegin(p) \/ UploadEnd(p) \/ Index(p)
    \/ Scan(p) \/ Take(p) \/ LkOpen(p) \/ LkFlock(p) \/ LkCheck(p) \/ LkDir(p)
    \/ AdSweep(p) \/ Finish(p) \/ RmCheck(p) \/ RmUnlink(p) \/ RmRmdir(p)

\* Every live process makes progress with its own work; the environment
\* (crash, exit, close, fork, quarantine) and the sink's choices are free.
\* The Survivor also claims, starts and opens its sink.
Progress(p) ==
  /\ WF_vars(Regain(p)) /\ WF_vars(OwnUploadBegin(p))
  /\ WF_vars(AdoptUploadBegin(p)) /\ WF_vars(UploadEnd(p)) /\ WF_vars(Index(p))
  /\ WF_vars(Scan(p)) /\ WF_vars(Take(p)) /\ WF_vars(LkOpen(p))
  /\ WF_vars(LkFlock(p)) /\ WF_vars(LkCheck(p)) /\ WF_vars(LkDir(p))
  /\ WF_vars(AdSweep(p)) /\ WF_vars(Finish(p)) /\ WF_vars(RmCheck(p))
  /\ WF_vars(RmUnlink(p)) /\ WF_vars(RmRmdir(p)) /\ WF_vars(StageLink(p))
  /\ WF_vars(OwnRmCheck(p)) /\ WF_vars(OwnRmUnlink(p)) /\ WF_vars(OwnRmRmdir(p))

Fairness ==
  /\ \A p \in Procs : Progress(p)
  /\ \A p \in Procs \cap {Survivor} :
       WF_vars(Claim(p)) /\ WF_vars(Start(p)) /\ WF_vars(OpenSink(p))

Spec == Init /\ [][Next]_vars
FairSpec == Spec /\ Fairness

-----------------------------------------------------------------------------
(* PROPERTIES *)

\* The processes that believe they own d: its creator while its claim is
\* held, and an adopter from LockDirectory on until it lets go.
Owners(d) ==
  {p \in Procs : alive[p] /\ ((d = p /\ ownHeld[p]) \/
                              (aDir[p] = d /\ aSt[p] \in HeldStates))}

MutualExclusion == \A d \in Dirs : Cardinality(Owners(d)) <= 1

\* No sweep ever deletes the .open file of a stage still in flight.
NoClobber == ~clobbered

\* A staged pack is never lost: a ready file stays in an existing directory,
\* and one gone from the spool is in the object store.
NoLostPack ==
  \A pk \in Packs :
    /\ pst[pk] = "ready" => dirEx[Home(pk)]
    /\ pst[pk] = "gone" => inStore[pk]

\* No pack is in flight to the store from two uploaders at once.
NoConcurrentUpload ==
  \A pk \in Packs : Cardinality({p \in Procs : alive[p] /\ upl[p] = pk}) <= 1

\* At most one chunk is out of a spool and not yet in the catalog per process
\* (decision 7: no uploads while the writer cannot index).
OwedBounded == \A p \in Procs : Cardinality(pend[p]) <= 1

\* A pack in the store but not in the catalog is remembered by a live
\* service, or the next start()'s reconcile finds it.
NoOrphan ==
  \A pk \in Packs :
    (inStore[pk] /\ ~inCat[pk]) =>
      \/ ReconcileOnStart
      \/ \E p \in Procs : alive[p] /\ ph[p] = "serving" /\ pk \in pend[p]

\* Decision 7: no chunk starts while the service holds no lease.
UploadOnlyWithLease == "nolease" \notin hist

\* A service never keeps holding a sibling it has blocked.
BlockedLetGo ==
  \A p \in Procs : alive[p] /\ aSt[p] \in HeldStates => aDir[p] \notin blocked[p]

TypeOK ==
  /\ nIno \in 0..Cardinality(Inos)
  /\ \A d \in Dirs : lf[d] \in 0..Cardinality(Inos)

Safety ==
  /\ TypeOK /\ MutualExclusion /\ NoClobber /\ NoLostPack
  /\ NoConcurrentUpload /\ OwedBounded /\ NoOrphan /\ BlockedLetGo
  /\ UploadOnlyWithLease

\* LIVENESS.  A pack staged by a process that has died is eventually taken
\* out of its spool by some live service (uploaded), unless no service can
\* ever upload it (Bad).
AdoptLive ==
  \A pk \in Packs \ BadPacks :
    (pst[pk] = "ready" /\ ~alive[Home(pk)]) ~> (pst[pk] # "ready")

\* charge_dead_siblings: what a sink is charged for a dead sibling
\* (ChargedSiblingBytes, spool.cpp:1357-1407) is eventually given back.
Charged(p, d) ==
  /\ d # p /\ dirEx[d] /\ sinkOpen[p] /\ ~sealed[p] /\ alive[p]
  /\ \E pk \in Packs : Home(pk) = d /\ pst[pk] \in {"ready", "open"}
  /\ \/ ~Held(d) /\ ~blockMark[d]
     \/ aDir[p] = d /\ aSt[p] \in HeldStates

ChargeDrains ==
  \A d \in Dirs : Survivor # NoP => (Charged(Survivor, d) ~> ~Charged(Survivor, d))

\* VACUITY WITNESSES: each is expected to be VIOLATED, showing the model
\* reaches the behaviour the properties above quantify over.
VacAdopted   == "adopted" \notin hist   \* a dead process's pack uploaded by another
VacRemoved   == "removed" \notin hist   \* an adopter removed a drained directory
VacMismatch  == "mismatch" \notin hist  \* IsFileAt caught a replaced lock file
VacBlocked   == "blocked" \notin hist   \* a directory with a Bad pack blocked
VacTook      == "took" \notin hist      \* an adoption took a lock at all
VacLiveSib   == \A p \in Procs : ~liveSib[p]  \* a look found a live sibling
Perms == Permutations(Procs)
AdopterPerms == Permutations(Adopters)
=============================================================================
