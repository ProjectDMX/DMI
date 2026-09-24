--------------------------- MODULE VersionPublish ---------------------------
(***************************************************************************)
(* Sole-claimant version allocation, the single-statement watermark        *)
(* publish barrier, and the publish retry loop with descriptor rewrite.    *)
(*                                                                         *)
(* Code (HEAD a987dfe):                                                    *)
(*   allocator   src/dmi/storage/capture/clickhouse_catalog.py:957-1034    *)
(*               native/csrc/catalog/version_allocator.cpp:49-89           *)
(*   publish     src/dmi/storage/capture/clickhouse_catalog.py:472-672     *)
(*               native/csrc/catalog/catalog_writer.cpp:479-670            *)
(*   retry loop  src/dmi/storage/capture/catalog.py:355-420, 463-600       *)
(*               native/csrc/catalog/indexer.cpp:107-125, 264-340          *)
(*   reader      src/dmi/storage/capture/clickhouse_reader.py:145,417-521  *)
(*               src/dmi/storage/capture/clickhouse_sql.py:131-139         *)
(*                                                                         *)
(* Every indexer runs ONE CatalogIndexer.index pass over ONE pack.  All    *)
(* packs describe the same capture, so the reader's supersession choice    *)
(* between packs is observable.  See README.md for the action/code map.    *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Layout,          \* which indexer/writer/pack topology; see WriterOf, PackOf
    AllocAttempts,   \* ClickHouseCatalogConfig.allocation_attempts (code: 16)
    PublishAttempts, \* CatalogIndexerConfig.max_publish_attempts (code: 8)
    MaxSkip,         \* cap on the randomized skip randbelow(8*attempt+1)
    MaxTakeovers,    \* lease takeovers the environment may perform
    MaxCrashes,      \* crashes / outcome-unknown failures the env may inject
    \* ---- consistency parameter ------------------------------------------
    STALE_READS,     \* FALSE: every read sees every committed row (single node,
                     \*   or quorum insert + select_sequential_consistency).
                     \* TRUE: a row is at first visible only on its writer's
                     \*   own replica; other writers' reads may miss it until an
                     \*   unordered Replicate step (per-table replication logs).
    READER_LAG,      \* TRUE: the reader resolves on a lagging replica, i.e.
                     \*   consistent_snapshot_reads=False (clickhouse_reader.py:177)
    \* ---- mutations (all FALSE = the code as written) --------------------
    SPLIT_BARRIER,        \* barrier+fence as a SELECT, then an unconditional INSERT
    SKIP_WM_READBACK,     \* no identity read-back after the watermark INSERT
    OCCUPANCY_READBACK,   \* read-back asks "any row at V?" (count() > 0)
    SKIP_REWRITE,         \* retry does not rewrite descriptors at the new version
    SKIP_CLAIM_READBACK,  \* allocator returns its candidate without the read-back
    UNSAFE_TAKEOVER       \* lease may move while the holder's statement is in flight

(* WriterOf[i]: the ClickHouseCatalogWriter indexer i uses (threads sharing  *)
(* one writer share its _serial lock and its lease).  PackOf[i]: the pack    *)
(* indexer i indexes; all packs describe the same capture.                   *)
(*  "shared": i1,i2 are two threads on writer 1 (the lower-version-publishes-*)
(*            second race of clickhouse_catalog.py:973-979); i3 is writer 2, *)
(*            which can only publish after a lease takeover.                 *)
(*  "replay": i1 and i3 both index pack 1 (a rebuild beside the live indexer *)
(*            or a pass re-indexing after a crash before commit_packs);      *)
(*            i2 indexes pack 2, a second pack describing the same capture.  *)
(*  "replaycrash": as "replay", but i3 is a LATER pass: it starts only after *)
(*            i1 has finished, so pack 1 is re-indexed only if i1 stopped    *)
(*            between its publish and commit_packs (the documented           *)
(*            "redundant work next pass", clickhouse_catalog.py:1049-1056).  *)
(*  "pair":   two writers, one thread each, distinct packs.                  *)
WriterOf == CASE Layout = "shared" -> <<1, 1, 2>>
              [] Layout = "replay" -> <<1, 1, 1>>
              [] Layout = "replaycrash" -> <<1, 1, 1>>
              [] Layout = "pair"   -> <<1, 2>>
PackOf   == CASE Layout = "shared" -> <<1, 2, 3>>
              [] Layout = "replay" -> <<1, 2, 1>>
              [] Layout = "replaycrash" -> <<1, 2, 1>>
              [] Layout = "pair"   -> <<1, 2>>

Indexers == 1 .. Len(WriterOf)
Writers  == {WriterOf[i] : i \in Indexers}
Packs    == {PackOf[i] : i \in Indexers}
NoOne    == 0
None     == -1       \* CatalogIndexer._published_version = None

Terminal == {"done", "skipped", "noLease", "allocFail", "allocErr", "leaseLost",
             "quarantined", "exhausted", "conflict", "crashed"}

VARIABLES
    claims,      \* version-claims table: [v, by, n]
    wm,          \* index_watermark table: [v, by, n]  (publish_id = <<by, n>>)
    manifest,    \* snapshot_manifest: [v, by, n, pack]
    descr,       \* capture_raw descriptor rows for the one capture: [v, pack]
    committed,   \* pack_inventory (replay guard): set of packs
    inflight,    \* watermark statements admitted (barrier+fence passed) but not landed
    lag,         \* rows not yet replicated off their writer's replica (STALE_READS)
    holder,      \* server-side lease head: the writer the fence admits
    everHeld,    \* writers that have held the lease (takeover is one-way)
    hasLease,    \* [w -> BOOLEAN] the writer's LOCAL lease object is not None
    quarantined, \* [w -> BOOLEAN] outcome-unknown quarantine (clickhouse_catalog.py:294)
    lock,        \* [w -> indexer or NoOne] the writer's _serial lock
    pc, ver, att, aatt, fl, cand, cn, cache,
    takeovers, crashes,
    allocated,   \* history: <<v, i, n>> of every version allocate_version returned
    refused,     \* history: publish ids whose attempt ended "race" or "lease lost"
    pins         \* history: W -> snapshot resolved when W first became the head

vars == <<claims, wm, manifest, descr, committed, inflight, lag, holder,
          everHeld, hasLease, quarantined, lock, pc, ver, att, aatt, fl, cand,
          cn, cache, takeovers, crashes, allocated, refused, pins>>

-----------------------------------------------------------------------------
(* Helpers *)
MaxV(S) == IF S = {} THEN 0 ELSE CHOOSE x \in S : \A y \in S : y <= x
Min(a, b) == IF a < b THEN a ELSE b
LexLe(a, b) == a[1] < b[1] \/ (a[1] = b[1] /\ a[2] <= b[2])
W(i) == WriterOf[i]

\* The replica a writer's reads land on sees every replicated row plus its
\* own writes (read-your-writes on one's own replica).  Faithful: everything.
Vis(tag, rows, w, originOf(_)) ==
    {r \in rows : ~STALE_READS \/ <<tag, r>> \notin lag \/ originOf(r) = w}
ByW(r) == WriterOf[r.by]
VisClaims(w) == Vis("c", claims, w, ByW)
VisWm(w)     == Vis("w", wm, w, ByW)
Lagged(tag, rows) == IF STALE_READS THEN {<<tag, r>> : r \in rows} ELSE {}

Skips(a) == IF a = 0 THEN {0} ELSE 0 .. Min(8 * a, MaxSkip)   \* :998

\* The lease, abstracted (modelled in detail in formal/tla/PublisherLease):
\* the fence admits exactly the head holder; renew succeeds only for a writer
\* whose local lease object still stands at the head.
Fence(w) == holder = w
Renew(w) == hasLease[w] /\ holder = w

LockFree(i) == lock[W(i)] \in {NoOne, i}
Take(i)     == lock' = [lock EXCEPT ![W(i)] = i]
Drop(i)     == lock' = [lock EXCEPT ![W(i)] = NoOne]
Me(i)       == <<i, att[i]>>
Pid(r)      == <<r.by, r.n>>

(* Reader: membership_predicate(bounded=True), clickhouse_sql.py:131-139,  *)
(* and argMax over (index_version, store_id, pack_id), reader.py:145,521.  *)
(* Descriptor rows are NOT bounded by the watermark: only their pack is.   *)
Paired(wmS, manS, Wm) ==
    {m \in manS : m.v <= Wm /\ \E r \in wmS : r.v <= Wm /\ r.v = m.v /\ Pid(r) = Pid(m)}
Members(wmS, manS, Wm) == {m.pack : m \in Paired(wmS, manS, Wm)}
Resolve(wmS, manS, dS, Wm) ==
    LET M == Members(wmS, manS, Wm)
        D == {d \in dS : d.pack \in M}
    IN  IF D = {} THEN 0
        ELSE (CHOOSE d \in D : \A e \in D : LexLe(<<e.v, e.pack>>, <<d.v, d.pack>>)).pack
\* The pack a snapshot SHOULD resolve to: the member whose publish is newest.
PubV(wmS, manS, Wm, p) == MaxV({m.v : m \in {x \in Paired(wmS, manS, Wm) : x.pack = p}})
Newest(wmS, manS, Wm) ==
    LET M == Members(wmS, manS, Wm)
    IN  IF M = {} THEN 0
        ELSE CHOOSE p \in M : \A q \in M :
                 LexLe(<<PubV(wmS, manS, Wm, q), q>>, <<PubV(wmS, manS, Wm, p), p>>)
Snap(wmS, manS, dS, Wm) ==
    [mem |-> Members(wmS, manS, Wm), res |-> Resolve(wmS, manS, dS, Wm)]
PubHead == MaxV({r.v : r \in wm})

\* Record the snapshot a reader pinning the new head would get.
PinNew(newWm) ==
    LET H == MaxV({r.v : r \in newWm})
    IN  pins' = IF H = 0 \/ H \in DOMAIN pins THEN pins
                ELSE pins @@ (H :> Snap(newWm, manifest, descr, H))

-----------------------------------------------------------------------------
Init ==
    /\ claims = {} /\ wm = {} /\ manifest = {} /\ descr = {} /\ committed = {}
    /\ inflight = {} /\ lag = {}
    /\ holder = WriterOf[1] /\ everHeld = {WriterOf[1]}
    \* Every writer starts with a LOCAL lease object (it acquired one once);
    \* only writer 1's stands at the server-side head.  So another writer
    \* passes the precheck (catalog.py:443-463) with a stale lease and is
    \* first refused by renew or the fence.
    /\ hasLease = [w \in Writers |-> TRUE]
    /\ quarantined = [w \in Writers |-> FALSE]
    /\ lock = [w \in Writers |-> NoOne]
    /\ pc = [i \in Indexers |-> "start"]
    /\ ver = [i \in Indexers |-> 0] /\ att = [i \in Indexers |-> 0]
    /\ aatt = [i \in Indexers |-> 0] /\ fl = [i \in Indexers |-> 0]
    /\ cand = [i \in Indexers |-> 0] /\ cn = [i \in Indexers |-> 0]
    /\ cache = [i \in Indexers |-> None]
    /\ takeovers = 0 /\ crashes = 0
    /\ allocated = {} /\ refused = {} /\ pins = <<>>

-----------------------------------------------------------------------------
(* CatalogIndexer.index, catalog.py:357-406 / indexer.cpp:196-264:           *)
(* committed_pack_ids (replay guard), then the publisher-lease precheck.     *)
Start(i) ==
    /\ pc[i] = "start"
    /\ (Layout = "replaycrash" /\ i = 3) => pc[1] \in Terminal
    /\ pc' = [pc EXCEPT ![i] = IF PackOf[i] \in committed THEN "skipped"
                               ELSE IF ~hasLease[W(i)] THEN "noLease"
                               ELSE "a_claims"]
    /\ aatt' = [aatt EXCEPT ![i] = 0]
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, hasLease, quarantined, lock, ver, att, fl, cand, cn,
                   cache, takeovers, crashes, allocated, refused, pins>>

(* allocate_version, clickhouse_catalog.py:984-1020: whole call under _serial. *)
(* Floor read in two deciding statements: max(claims) (:990) ...             *)
AllocClaims(i) ==
    /\ pc[i] = "a_claims" /\ LockFree(i) /\ Take(i)
    /\ fl' = [fl EXCEPT ![i] = MaxV({r.v : r \in VisClaims(W(i))})]
    /\ pc' = [pc EXCEPT ![i] = "a_wm"]
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, hasLease, quarantined, ver, att, aatt, cand, cn,
                   cache, takeovers, crashes, allocated, refused, pins>>

(* ... then last_published_version() (:995); floor = max of the two.         *)
AllocWm(i) ==
    /\ pc[i] = "a_wm"
    /\ fl' = [fl EXCEPT ![i] = MaxV({fl[i]} \cup {r.v : r \in VisWm(W(i))})]
    /\ pc' = [pc EXCEPT ![i] = "a_claim"]
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, hasLease, quarantined, lock, ver, att, aatt, cand, cn,
                   cache, takeovers, crashes, allocated, refused, pins>>

(* candidate = floor + 1 (+ randomized skip after a collision), :998; INSERT  *)
(* the claim, :1001-1006.  One INSERT: atomic.                               *)
AllocInsert(i) ==
    /\ pc[i] = "a_claim"
    /\ \E s \in Skips(aatt[i]) :
          LET row == [v |-> fl[i] + 1 + s, by |-> i, n |-> cn[i]]
          IN  /\ claims' = claims \cup {row}
              /\ lag' = lag \cup Lagged("c", {row})
              /\ cand' = [cand EXCEPT ![i] = row.v]
    /\ cn' = [cn EXCEPT ![i] = @ + 1]
    /\ pc' = [pc EXCEPT ![i] = "a_readback"]
    /\ UNCHANGED <<wm, manifest, descr, committed, inflight, holder, everHeld,
                   hasLease, quarantined, lock, ver, att, aatt, fl, cache,
                   takeovers, crashes, allocated, refused, pins>>

(* Read the claims at the candidate back (:1007-1015): proceed only as the    *)
(* sole claimant; otherwise abandon it and retry above it.                   *)
AllocReadback(i) ==
    /\ pc[i] = "a_readback"
    /\ LET owners == {Pid(r) : r \in {x \in VisClaims(W(i)) : x.v = cand[i]}}
           sole   == SKIP_CLAIM_READBACK \/ owners = {<<i, cn[i] - 1>>}
       IN  IF sole
           THEN /\ ver' = [ver EXCEPT ![i] = cand[i]]
                /\ allocated' = allocated \cup {<<cand[i], i, cn[i] - 1>>}
                /\ pc' = [pc EXCEPT ![i] = "a_check"]
                /\ Drop(i)
                /\ UNCHANGED aatt
           ELSE IF aatt[i] + 1 >= AllocAttempts
           THEN /\ pc' = [pc EXCEPT ![i] = "allocFail"]   \* :1018-1020
                /\ Drop(i)
                /\ UNCHANGED <<ver, allocated, aatt>>
           ELSE /\ aatt' = [aatt EXCEPT ![i] = @ + 1]
                /\ pc' = [pc EXCEPT ![i] = "a_claims"]    \* keeps _serial
                /\ UNCHANGED <<ver, allocated, lock>>
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, hasLease, quarantined, att, fl, cand, cn, cache,
                   takeovers, crashes, refused, pins>>

(* CatalogIndexer._allocate_version cross-check, catalog.py:476-491:          *)
(* version must be above the (possibly cached) published head.              *)
AllocCheck(i) ==
    /\ pc[i] = "a_check" /\ LockFree(i)
    /\ LET h == IF cache[i] = None THEN MaxV({r.v : r \in VisWm(W(i))}) ELSE cache[i]
       IN  /\ cache' = [cache EXCEPT ![i] = h]
           /\ pc' = [pc EXCEPT ![i] =
                       IF ver[i] <= h THEN "allocErr"
                       ELSE IF att[i] > 0 /\ SKIP_REWRITE THEN "p_begin"
                       ELSE "write_desc"]
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, hasLease, quarantined, lock, ver, att, aatt, fl,
                   cand, cn, takeovers, crashes, allocated, refused, pins>>

(* write_descriptors (one batch), clickhouse_catalog.py:361-375.  Also the   *)
(* retry's rewrite at the new version, catalog.py:590-592.                   *)
WriteDesc(i) ==
    /\ pc[i] = "write_desc" /\ LockFree(i)
    /\ LET row == [v |-> ver[i], pack |-> PackOf[i]]
       IN  /\ descr' = descr \cup {row}
           /\ lag' = lag \cup Lagged("d", {row})
    /\ pc' = [pc EXCEPT ![i] = "p_begin"]
    /\ UNCHANGED <<claims, wm, manifest, committed, inflight, holder,
                   everHeld, hasLease, quarantined, lock, ver, att, aatt, fl,
                   cand, cn, cache, takeovers, crashes, allocated, refused, pins>>

(* publish_snapshot, clickhouse_catalog.py:454-470 + 495: under _serial;      *)
(* quarantine check, then renew_publisher_lease.                            *)
PubBegin(i) ==
    /\ pc[i] = "p_begin" /\ LockFree(i)
    /\ IF quarantined[W(i)]
       THEN /\ pc' = [pc EXCEPT ![i] = "quarantined"] /\ UNCHANGED <<lock, hasLease>>
       ELSE IF ~Renew(W(i))
       THEN /\ pc' = [pc EXCEPT ![i] = "leaseLost"]
            /\ hasLease' = [hasLease EXCEPT ![W(i)] = FALSE]
            /\ UNCHANGED lock
       ELSE /\ pc' = [pc EXCEPT ![i] = "p_manifest"] /\ Take(i)
            /\ UNCHANGED hasLease
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, quarantined, ver, att, aatt, fl, cand, cn, cache,
                   takeovers, crashes, allocated, refused, pins>>

(* Fenced manifest INSERT + chunk read-back + reject_if_gone, :510-547, then  *)
(* the renew at :548.  One chunk.  Folded into one step: the read-back reads *)
(* only this publish's own rows, which nothing else writes or (GC is not     *)
(* modelled) deletes, and the renew succeeds exactly when the fence just did *)
(* (the lease abstraction has no expiry of its own).  A takeover can still   *)
(* land between this renew and the watermark statement's admission.         *)
PubManifest(i) ==
    /\ pc[i] = "p_manifest"
    /\ IF Fence(W(i))
       THEN LET row == [v |-> ver[i], by |-> i, n |-> att[i], pack |-> PackOf[i]]
            IN  /\ manifest' = manifest \cup {row}
                /\ lag' = lag \cup Lagged("m", {row})
                /\ pc' = [pc EXCEPT ![i] = IF SPLIT_BARRIER THEN "p_check" ELSE "p_admit"]
                /\ UNCHANGED <<lock, hasLease, refused>>
       ELSE /\ pc' = [pc EXCEPT ![i] = "leaseLost"]
            /\ hasLease' = [hasLease EXCEPT ![W(i)] = FALSE]
            /\ refused' = refused \cup {Me(i)}
            /\ Drop(i)
            /\ UNCHANGED <<manifest, lag>>
    /\ UNCHANGED <<claims, wm, descr, committed, inflight, holder, everHeld,
                   quarantined, ver, att, aatt, fl, cand, cn, cache, takeovers,
                   crashes, allocated, pins>>

(* The ONE statement: INSERT ... SELECT ... WHERE barrier AND fence,          *)
(* :559-583.  Barrier and fence are evaluated when the statement is ADMITTED *)
(* and the row LANDS later (:643-650): admission is this step, landing is    *)
(* Land.  A refused statement completes with zero rows.                      *)
PubAdmit(i) ==
    /\ pc[i] = "p_admit" /\ ~SPLIT_BARRIER
    /\ IF MaxV({r.v : r \in VisWm(W(i))}) < ver[i] /\ Fence(W(i))
       THEN inflight' = inflight \cup {[v |-> ver[i], by |-> i, n |-> att[i]]}
       ELSE UNCHANGED inflight
    /\ pc' = [pc EXCEPT ![i] = "p_readback"]
    /\ UNCHANGED <<hasLease, refused, lock>>
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, lag, holder, everHeld,
                   quarantined, ver, att, aatt, fl, cand, cn, cache, takeovers,
                   crashes, allocated, pins>>

(* MUTATION SPLIT_BARRIER: "a separate SELECT then INSERT" (:552-555).       *)
PubCheck(i) ==
    /\ pc[i] = "p_check" /\ SPLIT_BARRIER
    /\ pc' = [pc EXCEPT ![i] =
                 IF MaxV({r.v : r \in VisWm(W(i))}) < ver[i] /\ Fence(W(i))
                 THEN "p_insert" ELSE "p_readback"]
    /\ UNCHANGED <<hasLease, refused, lock>>
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, quarantined, ver, att, aatt, fl, cand, cn, cache,
                   takeovers, crashes, allocated, pins>>
PubInsert(i) ==
    /\ pc[i] = "p_insert"
    /\ LET row == [v |-> ver[i], by |-> i, n |-> att[i]]
       IN  /\ wm' = wm \cup {row}
           /\ lag' = lag \cup Lagged("w", {row})
           /\ PinNew(wm \cup {row})
    /\ pc' = [pc EXCEPT ![i] = "p_readback"]
    /\ UNCHANGED <<claims, manifest, descr, committed, inflight, holder, everHeld,
                   hasLease, quarantined, lock, ver, att, aatt, fl, cand, cn,
                   cache, takeovers, crashes, allocated, refused>>

(* Identity read-back, :594-642.  execute() is synchronous, so it runs after  *)
(* this publish's own statement has landed.  "Is MY row there?" then "is it  *)
(* the ONLY row there?".  Race -> catalog.py:571-593: drop the cached head,  *)
(* re-allocate, rewrite descriptors, publish again, bounded by attempts.     *)
PubReadback(i) ==
    /\ pc[i] = "p_readback"
    /\ \A s \in inflight : Pid(s) # Me(i)
    /\ LET owners  == {Pid(r) : r \in {x \in VisWm(W(i)) : x.v = ver[i]}}
           success == IF SKIP_WM_READBACK THEN TRUE
                      ELSE IF OCCUPANCY_READBACK THEN owners # {}
                      ELSE owners = {Me(i)}
       IN  IF success
           THEN /\ pc' = [pc EXCEPT ![i] = "commit"]
                /\ UNCHANGED <<att, aatt, cache, refused, committed, hasLease>>
           ELSE IF Me(i) \in owners
           THEN \* SnapshotPublishConflictError: visible, so commit, then raise.
                /\ pc' = [pc EXCEPT ![i] = "conflict"]
                /\ committed' = committed \cup {PackOf[i]}
                /\ UNCHANGED <<att, aatt, cache, refused, hasLease>>
           ELSE IF ~Fence(W(i))
           THEN \* reject_if_gone -> PublisherLeaseError (:608)
                /\ pc' = [pc EXCEPT ![i] = "leaseLost"]
                /\ hasLease' = [hasLease EXCEPT ![W(i)] = FALSE]
                /\ refused' = refused \cup {Me(i)}
                /\ UNCHANGED <<att, aatt, cache, committed>>
           ELSE /\ refused' = refused \cup {Me(i)}   \* SnapshotPublishRaceError
                /\ IF att[i] + 1 >= PublishAttempts
                   THEN /\ pc' = [pc EXCEPT ![i] = "exhausted"]
                        /\ UNCHANGED <<att, aatt, cache>>
                   ELSE /\ att' = [att EXCEPT ![i] = @ + 1]
                        /\ aatt' = [aatt EXCEPT ![i] = 0]
                        /\ cache' = [cache EXCEPT ![i] = None]
                        /\ pc' = [pc EXCEPT ![i] = "a_claims"]
                /\ UNCHANGED <<committed, hasLease>>
    /\ Drop(i)
    /\ UNCHANGED <<claims, wm, manifest, descr, inflight, lag, holder, everHeld,
                   quarantined, ver, fl, cand, cn, takeovers, crashes, allocated,
                   pins>>

(* commit_packs AFTER a successful publish, catalog.py:404-406.             *)
Commit(i) ==
    /\ pc[i] = "commit" /\ LockFree(i)
    /\ committed' = committed \cup {PackOf[i]}
    /\ pc' = [pc EXCEPT ![i] = "done"]
    /\ UNCHANGED <<claims, wm, manifest, descr, inflight, lag, holder, everHeld,
                   hasLease, quarantined, lock, ver, att, aatt, fl, cand, cn,
                   cache, takeovers, crashes, allocated, refused, pins>>

-----------------------------------------------------------------------------
(* Environment *)

(* An admitted watermark statement lands.                                    *)
Land ==
    \E s \in inflight :
        /\ wm' = wm \cup {s}
        /\ inflight' = inflight \ {s}
        /\ lag' = lag \cup Lagged("w", {s})
        /\ PinNew(wm \cup {s})
        /\ UNCHANGED <<claims, manifest, descr, committed, holder, everHeld,
                       hasLease, quarantined, lock, pc, ver, att, aatt, fl, cand,
                       cn, cache, takeovers, crashes, allocated, refused>>

(* A statement whose client is gone is capped by max_execution_time.       *)
Abort ==
    \E s \in inflight :
        /\ pc[s.by] # "p_readback"
        /\ inflight' = inflight \ {s}
        /\ UNCHANGED <<claims, wm, manifest, descr, committed, lag, holder,
                       everHeld, hasLease, quarantined, lock, pc, ver, att, aatt,
                       fl, cand, cn, cache, takeovers, crashes, allocated,
                       refused, pins>>

(* Lease takeover by another writer (abstract).  The timing argument of the  *)
(* fence (clickhouse_lease.py:192-211, clickhouse_catalog.py:66-94) is that  *)
(* a successor cannot claim while the holder's fenced statement can still    *)
(* land; quarantine (:294-307) keeps the server row live for a full TTL.     *)
Takeover ==
    /\ takeovers < MaxTakeovers
    /\ UNSAFE_TAKEOVER \/ \A s \in inflight : WriterOf[s.by] # holder
    /\ \E w \in Writers \ everHeld :
          /\ holder' = w
          /\ everHeld' = everHeld \cup {w}
          /\ hasLease' = [hasLease EXCEPT ![w] = TRUE]
    /\ takeovers' = takeovers + 1
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag,
                   quarantined, lock, pc, ver, att, aatt, fl, cand, cn, cache,
                   crashes, allocated, refused, pins>>

(* Per-table replication catches up on one row.                            *)
Replicate ==
    /\ STALE_READS
    /\ \E x \in lag : lag' = lag \ {x}
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, holder,
                   everHeld, hasLease, quarantined, lock, pc, ver, att, aatt, fl,
                   cand, cn, cache, takeovers, crashes, allocated, refused, pins>>

(* A crash or unexpected exception at any point of a pass.  Inside           *)
(* publish_snapshot (lock held) it is outcome-unknown: quarantine and drop   *)
(* the local lease, no tombstone (:462-470, :294-307).  A statement already  *)
(* admitted may still land.  The exception releases _serial.                 *)
InPublish(i) == pc[i] \in {"p_manifest", "p_admit", "p_check", "p_insert", "p_readback"}
Crash(i) ==
    /\ crashes < MaxCrashes
    /\ pc[i] \notin Terminal \cup {"start"}
    /\ crashes' = crashes + 1
    /\ pc' = [pc EXCEPT ![i] = "crashed"]
    /\ lock' = IF lock[W(i)] = i THEN [lock EXCEPT ![W(i)] = NoOne] ELSE lock
    /\ IF InPublish(i)
       THEN /\ quarantined' = [quarantined EXCEPT ![W(i)] = TRUE]
            /\ hasLease' = [hasLease EXCEPT ![W(i)] = FALSE]
       ELSE UNCHANGED <<quarantined, hasLease>>
    /\ UNCHANGED <<claims, wm, manifest, descr, committed, inflight, lag, holder,
                   everHeld, ver, att, aatt, fl, cand, cn, cache, takeovers,
                   allocated, refused, pins>>

Proc(i) ==
    \/ Start(i) \/ AllocClaims(i) \/ AllocWm(i) \/ AllocInsert(i)
    \/ AllocReadback(i) \/ AllocCheck(i) \/ WriteDesc(i) \/ PubBegin(i)
    \/ PubManifest(i) \/ PubAdmit(i) \/ PubCheck(i) \/ PubInsert(i)
    \/ PubReadback(i) \/ Commit(i)

AllDone == \A i \in Indexers : pc[i] \in Terminal
Finished == AllDone /\ inflight = {} /\ UNCHANGED vars

Next ==
    \/ \E i \in Indexers : Proc(i) \/ Crash(i)
    \/ Land \/ Abort \/ Takeover \/ Replicate
    \/ Finished

Fairness == /\ \A i \in Indexers : WF_vars(Proc(i))
            /\ WF_vars(Land)
Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(* Properties *)

TypeOK ==
    /\ pc \in [Indexers -> Terminal \cup {"start", "a_claims", "a_wm", "a_claim",
               "a_readback", "a_check", "write_desc", "p_begin", "p_manifest",
               "p_admit", "p_check", "p_insert", "p_readback", "commit"}]
    /\ lock \in [Writers -> Indexers \cup {NoOne}]

\* The allocator's contract (clickhouse_catalog.py:960-972): every returned
\* version is unique.
AllocUnique == \A a, b \in allocated : a[1] = b[1] => a = b

\* No two publishes are visible at one version.
NoDupVersion == \A r, s \in wm : r.v = s.v => r = s

\* A publish that reported "lost race" or "lease lost" left nothing visible
\* (the error messages at :536-542, :609-615, and catalog.py:524-526).
LoserInvisible == \A r \in wm : Pid(r) \notin refused

\* A pack in the replay inventory is visible at the head (catalog.py:398-403:
\* "a pack recorded there but never made visible is skipped forever").
CommittedVisible == \A p \in committed : p \in Members(wm, manifest, PubHead)

\* An indexer that finished "done" published its own pack at its version.
DoneIsPublished ==
    \A i \in Indexers : pc[i] = "done" =>
        \E m \in Paired(wm, manifest, PubHead) : m.pack = PackOf[i] /\ m.v = ver[i]

\* A pinned watermark resolves to the same packs and the same pack for the
\* capture for as long as it lives (clickhouse_reader.py:3, :36-39).
PinnedStable ==
    \A Wm \in DOMAIN pins : Snap(wm, manifest, descr, Wm) = pins[Wm]

\* Supersession: a snapshot resolves the capture to the member pack whose
\* publish is newest -- never to a superseded pack's locator (catalog.py:520-531).
ResolvesNewest ==
    \A Wm \in DOMAIN pins :
        Resolve(wm, manifest, descr, Wm) = Newest(wm, manifest, Wm)

\* The same, for a reader on a lagging replica (consistent_snapshot_reads off;
\* only meaningful with STALE_READS).  Lag may OMIT the capture (a short page,
\* documented at clickhouse_reader.py:163-170); it must not resolve it to a
\* superseded pack.
RView(tag, rows) == {r \in rows : <<tag, r>> \notin lag}
ReaderLagNoSuperseded ==
    READER_LAG =>
        LET vw == RView("w", wm)  vm == RView("m", manifest)  vd == RView("d", descr)
            H  == MaxV({r.v : r \in vw})
            res == Resolve(vw, vm, vd, H)
        IN  res = 0 \/ res = Newest(vw, vm, H)

\* Published versions strictly increase: a watermark row never lands at or
\* below the published head (:552-558, reader pins, :643-650).
MonotonicLanding == [][\A r \in wm' \ wm : r.v > PubHead]_vars

\* Reachability witnesses: each is EXPECTED TO FAIL in the faithful model,
\* proving the scenario it negates is actually explored (Witness*.cfg).
NeverLostRace     == ~\E i \in Indexers : att[i] > 0
NeverRetryWins    == ~\E i \in Indexers : pc[i] = "done" /\ att[i] > 0
NeverSuccessorPub == ~\E r \in wm : WriterOf[r.by] # WriterOf[1]
NeverFencedOut    == ~\E i \in Indexers : pc[i] = "leaseLost" /\ Me(i) \in refused
NeverOrphanLands  == ~\E r \in wm : pc[r.by] = "crashed"
NeverContested    == ~\E r, s \in claims : r.v = s.v /\ r # s
\* Expected to HOLD in the faithful model: no publish ever conflicts.
NeverConflict     == ~\E i \in Indexers : pc[i] = "conflict"

\* The bounded retry loops terminate: every indexer reaches a terminal state.
Termination == <>AllDone
=============================================================================
