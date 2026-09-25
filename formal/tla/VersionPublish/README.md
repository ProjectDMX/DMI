# VersionPublish: version allocation, watermark publish barrier, publish retry

A TLA+ model of three pieces, written against HEAD `a987dfe`:

- the sole-claimant version allocator,
- the single-statement watermark publish (barrier, lease fence and INSERT in one
  statement) and its identity read-back,
- the indexer's bounded publish retry, which rewrites descriptors at the new version.

It also models the reader's snapshot resolution, so that supersession can be
checked. The lease itself is abstracted to "the fence admits the head holder".
`formal/tla/PublisherLease` models it in detail.

One module, `VersionPublish.tla`, backs every config. The configs differ only in
constants: `Layout`, the bounds, the consistency flag, the mutation flags and
the reader-ranking flags. `gen_cfgs.sh` regenerates every `.cfg`.

**Status of the replay finding: fixed by PR #156** (first-publish ranking).
With `RANK_BY_MEMBERSHIP = FALSE` the module is main at `a987dfe` and the
`Replay*` configs still fail, as the bug witness. With
`RANK_BY_MEMBERSHIP = TRUE, RANK_MIN = TRUE` it is PR #156 as merged, and the
`FixMin*` configs pass. See [PR #156](#pr-156-first-publish-ranking-expected-to-pass).

## What is modelled

Each indexer runs one `CatalogIndexer.index` pass over one pack. Every pack
describes the same capture, so the reader's choice between packs can be observed.
The tables are sets of rows:

| Table | Fields |
|---|---|
| `claims` | `[v, by, n]` |
| `wm` (watermark) | `[v, by, n]`, where publish_id = `<<by, n>>` |
| `manifest` | `[v, by, n, pack]` |
| `descr` | `[v, pack]` |
| `committed` | pack inventory |

`inflight` holds watermark statements that have been admitted but have not
landed. Each writer has its own `_serial` lock (`lock`), its local lease object
(`hasLease`) and a quarantine flag.

Layouts (`WriterOf`, `PackOf` in the module):

- `shared`: i1 and i2 are two threads on writer 1, so the lower version can
  publish second (clickhouse_catalog.py:973-979). i3 is on writer 2, which can
  publish only after a takeover.
- `replay`: i1 and i3 both index pack 1. i2 indexes pack 2, a second pack that
  describes the same capture.
- `replaycrash`: the same as `replay`, but i3 starts only after i1 has finished.
- `pair`: two writers, one thread each.

### Actions mapped to code

| Action | Python | C++ |
|---|---|---|
| `Start`: replay guard and lease precheck | catalog.py:365, 400, 443-463 | indexer.cpp:212, 261-268 |
| `AllocClaims`: `max(claims)`, under `_serial` | clickhouse_catalog.py:984-994 | version_allocator.cpp:52 |
| `AllocWm`: `floor = max(claims, last_published)` | clickhouse_catalog.py:995 | version_allocator.cpp:53-54 |
| `AllocInsert`: candidate is floor+1+skip, INSERT claim | clickhouse_catalog.py:998-1006 | version_allocator.cpp:57-73 |
| `AllocReadback`: sole claimant, or retry or give up | clickhouse_catalog.py:1007-1020 | version_allocator.cpp:74-88 |
| `AllocCheck`: strictly above the cached head | catalog.py:465-491 | indexer.cpp:107-125 |
| `WriteDesc`: first write and retry rewrite | catalog.py:402, 493-502, 600 | indexer.cpp:270-284, 325 |
| `PubBegin`: `_serial`, quarantine, renew | clickhouse_catalog.py:454-470, 489 | catalog_writer.cpp:483-489 |
| `PubManifest`: fenced chunk, read-back, renew | clickhouse_catalog.py:496-549 | catalog_writer.cpp:530-580 |
| `PubAdmit` then `Land`: barrier + fence + INSERT | clickhouse_catalog.py:551-583 (admit and land: 643-650) | catalog_writer.cpp:585-605 |
| `PubReadback`: mine? only mine? | clickhouse_catalog.py:584-642; catalog.py:566-606 | catalog_writer.cpp:607-660; indexer.cpp:290-340 |
| `Commit`: inventory after publish | catalog.py:404-418 | indexer.cpp:337-339 |
| `Crash`: exception or crash anywhere; quarantine inside publish | clickhouse_catalog.py:462-470, 294-307 | catalog_writer.cpp:661-668 |
| `Abort`: `max_execution_time` caps an orphaned statement | clickhouse_catalog.py:66-72 | none |
| `Takeover`: abstract lease move | clickhouse_lease.py:192-211 | none |
| `Replicate`: per-table replication catches up | clickhouse_catalog.py:95-118 | none |
| `Merge` (`MERGES` only): ReplacingMergeTree collapses one pack's descriptor rows to its highest version | ReplacingMergeTree on the descriptor table | none |

The reader is modelled at clickhouse_reader.py:145, 417-448 and 521, and at
clickhouse_sql.py:131-139. Membership is manifest rows paired with a watermark row
on `(index_version, publish_id)`, both at or below W. Resolution is `argMax` over
`(rank, pack)` across ALL descriptor rows of the member packs. The descriptor
rows are not bounded by W. The rank (`Rank` in the module) is:

| Flags | Rank of a pack's row | Models |
|---|---|---|
| `RANK_BY_MEMBERSHIP = FALSE` | the row's own `index_version` | main at `a987dfe` (the bug) |
| `RANK_BY_MEMBERSHIP = TRUE, RANK_MIN = TRUE` | the pack's FIRST paired publish <= W, `min(index_version)` | PR #156 as merged |
| `RANK_BY_MEMBERSHIP = TRUE, RANK_MIN = FALSE` | the pack's NEWEST paired publish <= W, `max(index_version)` | the alternative (kept selectable; it fails, see below) |

### What is atomic and why

**One INSERT or SELECT is one step.** Each is a single server statement. The one
exception is the watermark statement.

**The watermark statement is two steps, `PubAdmit` then `Land`.** The code says
that its barrier and fence are evaluated at admission and that the row lands
later (clickhouse_catalog.py:643-650).

**Some steps are folded.** `PubManifest` folds the chunk INSERT, its read-back
and the following renew into one step. The read-back sees only this publish's
own rows, and GC is not modelled. The takeover window between that renew and the
watermark statement is kept.

**The floor is read in two steps**, as the code does it.

**`_serial` is modelled explicitly.**

- `allocate_version` and `publish_snapshot` each hold it for the whole call.
- `write_descriptors`, `commit_packs` and `last_published_version` each take it
  for one step.

**The randomized skip** `randbelow(8*attempt+1)` becomes a nondeterministic choice
from `0..min(8*attempt, MaxSkip)`.

### Abstractions and assumptions

- **Lease.** `Fence(w) == holder = w`. A `Takeover` moves the head to a writer
  that has never held it. The move is one-way and bounded by `MaxTakeovers`.
  - In the faithful model a takeover is not allowed while the holder has an
    admitted, unlanded statement. This is the TTL > timeout + skew argument, plus
    quarantine keeping the server row live (clickhouse_lease.py:192-211). The
    `UNSAFE_TAKEOVER` flag removes that condition.
  - Every writer starts with a local lease object, but only writer 1 is the head.
    So a writer can pass the precheck while holding a stale lease.
- **Consistency.**
  - `STALE_READS=FALSE`: every read sees every committed row. This is a single
    node, or `insert_quorum` plus `select_sequential_consistency`.
  - `STALE_READS=TRUE`: a row is visible only on its writer's own replica until
    `Replicate` runs. Replication is independent per row, which models the
    per-table replication logs. This applies to the allocator's reads, the
    barrier's subquery and the read-backs.
  - `READER_LAG=TRUE`: the reader resolves on a lagging replica, as when
    `consistent_snapshot_reads=False`.
- **Not modelled.**
  - `collect_garbage`. This makes the `_manifest_is_whole` check
    (clickhouse_catalog.py:641-672) vacuous.
  - Multi-chunk manifests.
  - Multiple batches per pass.
  - Store ids.
  - Lease re-acquisition within a pass.
  - Quarantine expiry. It is irrelevant, because each writer's passes end at the
    quarantine.
- **Pinning.** `pins[W]` records the snapshot at the moment W first becomes the
  head. Readers pin `current_watermark()`, which is the head.

## Properties

| Name | Meaning |
|---|---|
| `AllocUnique` | Every version returned by the allocator is unique (:960-972). |
| `NoDupVersion` | At most one watermark row per version. |
| `MonotonicLanding` (action) | A watermark row never lands at or below the published head. Published versions strictly increase and are never reused. |
| `LoserInvisible` | An attempt that ended "race" or "lease lost" has no watermark row, so none of its manifest rows are members. |
| `CommittedVisible` | Every pack in the replay inventory is a member at the head (catalog.py:410-416). |
| `DoneIsPublished` | An indexer that returned success has its pack paired at its version. |
| `PinnedStable` | A pinned W keeps its members and its resolved pack for as long as it lives (clickhouse_reader.py:3, 36-39). |
| `ResolvesNewest` | For every W that was a head, the capture resolves to the member pack whose publish is newest, never to a superseded pack (catalog.py:520-531). This encodes the **pre-fix** intent (newest publish wins). Under it, a replay that republishes a superseded pack at a newer version is *supposed* to win, so it is expected to fail under PR #156's ranking on the replay layouts and is not checked by the `FixMin*` configs. |
| `ResolvesFirstPublished` | PR #156's supersession: for every W that was a head, the capture resolves to the member whose FIRST publish is newest, so republishing a superseded pack never wins. |
| `NoSupersededComeback` | The same claim over history, independent of min vs max: a pack that was a member at some head W1 but did not win there never wins at a later head W2. |
| `ReaderLagNoSuperseded` | The same as `ResolvesNewest` on the lagging replica. Omitting the capture is allowed, because it is documented as a short page. |
| `NeverConflict` | `SnapshotPublishConflictError` is unreachable when the lease holds. |
| `Termination` (liveness) | Under weak fairness of every indexer and of `Land`, every indexer reaches a terminal state. |
| `Witness*` | Reachability checks. Each one must FAIL. Together they show that the model does reach a contested claim, a lost race, a retry that wins, a publish by the successor writer, a fence-out, and an orphaned statement that lands after its client crashed. |

## How to run

```
SP=<scratch dir>
java -XX:+UseParallelGC -Xmx4g -cp $SP/tla2tools.jar tlc2.TLC -noGenerateSpecTE -workers auto \
  -metadir $SP/meta-<Config> -cleanup -config <Config>.cfg VersionPublish.tla
```

`tla2tools.jar` is a source build of TLC, `Version 2026.09.24`. Its state dirs are
kept out of the repo through `-metadir`. The machine had 4 cores shared with
other TLC jobs.

## Results

Wall times are on a shared 4-core machine and are only a rough guide. The
state counts of violating runs are what TLC had explored when it stopped.

Default bounds, unless a config says otherwise:

- AllocAttempts 2, PublishAttempts 2, MaxSkip 1.
- The code's defaults are 16 and 8, with a skip of up to 8*attempt.

In the takeover (T) and crash (C) columns, a `-` means 0.

### Faithful configs (expected to pass)

| Config | Layout, bounds | Checks | Result | States generated / distinct, depth | Wall |
|---|---|---|---|---|---|
| Faithful | shared, T=1 C=1 | all safety + NeverConflict + MonotonicLanding + Termination | **PASS** | 5,239,843 / 2,110,944, depth 76 | 734 s |
| FaithfulLarge | shared, alloc 3, publish 3, T=1 C=1 | all safety + NeverConflict + MonotonicLanding | **PASS** | 28,454,387 / 11,530,782, depth 92 | 959 s |
| ReplayRest | replay, T=- C=- | everything except PinnedStable and ResolvesNewest, + Termination | **PASS** | 24,433 / 17,228, depth 60 | 7 s |
| SkipClaimReadbackRest | pair, T=1 C=1, allocator read-back removed | everything except AllocUnique, + Termination | **PASS** | 5,138 / 2,275, depth 38 | 3 s |

### Finding (main at `a987dfe`, with a replayed pack; expected to fail)

| Config | Layout | Violated | Trace | Wall |
|---|---|---|---|---|
| Replay | replay (concurrent passes) | ResolvesNewest | 31 states | 3 s |
| ReplayCrash | replaycrash, C=1 (replay only after a crash) | ResolvesNewest | 32 states | 3 s |
| ReplayCrashPinned | replaycrash, C=1 | PinnedStable | 33 states | 4 s |

### PR #156, first-publish ranking (expected to pass)

`RANK_BY_MEMBERSHIP = TRUE, RANK_MIN = TRUE`. Each config has a `FixMinMerge_*`
twin that adds `MERGES = TRUE` (background merges collapse a pack's descriptor
rows to its highest version at any point). The `*_Faithful` numbers come from
runs of this model before it was merged into this module, with identical
constants. The short configs were re-run on this module and match exactly.

| Config | Layout | Checks | Result | States generated / distinct, depth | Wall |
|---|---|---|---|---|---|
| FixMin_Faithful | shared, T=1 C=1 | ResolvesFirstPublished, NoSupersededComeback, all safety except ResolvesNewest, NeverConflict, MonotonicLanding, Termination | **PASS** | 5,239,843 / 2,110,944, depth 76 | 395 s |
| FixMin_Replay | replay | ResolvesFirstPublished, NoSupersededComeback, all safety except ResolvesNewest, MonotonicLanding | **PASS** | 23,767 / 16,724, depth 60 | 4 s |
| FixMin_ReplayRest | replay | ResolvesFirstPublished, NoSupersededComeback, everything except PinnedStable and ResolvesNewest, MonotonicLanding, Termination | **PASS** | 23,767 / 16,724, depth 60 | 5 s |
| FixMin_ReplayCrash | replaycrash, C=1 | ResolvesFirstPublished, NoSupersededComeback, PinnedStable | **PASS** | 5,821 / 4,119, depth 60 | 2 s |
| FixMin_ReplayCrashPinned | replaycrash, C=1 | PinnedStable | **PASS** | 5,821 / 4,119, depth 60 | 1 s |
| FixMinMerge_Faithful | shared, T=1 C=1, merges | as FixMin_Faithful | **PASS** | 8,523,437 / 3,057,864, depth 78 | 635 s |
| FixMinMerge_Replay | replay, merges | as FixMin_Replay | **PASS** | 78,183 / 38,746, depth 62 | 7 s |
| FixMinMerge_ReplayRest | replay, merges | as FixMin_ReplayRest | **PASS** | 78,183 / 38,746, depth 62 | 9 s |
| FixMinMerge_ReplayCrash | replaycrash, C=1, merges | as FixMin_ReplayCrash | **PASS** | 9,093 / 5,377, depth 62 | 2 s |
| FixMinMerge_ReplayCrashPinned | replaycrash, C=1, merges | PinnedStable | **PASS** | 9,093 / 5,377, depth 62 | 1 s |

`ResolvesNewest` is left out on purpose (see Properties). Checked anyway, it
fails under first-publish ranking on `replay` and `replaycrash`, because the
replayed pack's republish is newest but no longer wins. `PinnedStable` holds.

### The max alternative (expected to fail)

| Config | Flags | Violated | Trace |
|---|---|---|---|
| FixMax_ReplayCrash | `RANK_BY_MEMBERSHIP = TRUE, RANK_MIN = FALSE`, replaycrash, C=1 | NoSupersededComeback | 37 states |

Ranking by the newest paired publish at or below W keeps every pin stable,
and it satisfies `ResolvesNewest`. But once the replayed pass republishes P1
at v3, every head from 3 onwards resolves to P1, which lost to P2 at W=2. With
`MERGES` the result is the same. First-publish ranking keeps P1 at rank 1,
below P2, at every head.

### Consistency parameter (expected to fail)

| Config | Violated | Trace | What breaks |
|---|---|---|---|
| StaleAlloc | AllocUnique | 11 | Two writers each read only their own claim at v=1, and both think they are the sole claimant. This is the case the `insert_quorum` note at clickhouse_catalog.py:95-104 warns about. |
| StalePublish | NoDupVersion | 24 | Writer 1 publishes v=1, but its watermark row has not replicated. The lease moves to writer 2. Writer 2's allocator and the barrier subquery both see max=0, so v=1 is published a second time. |
| StaleReader | ReaderLagNoSuperseded | 29 | The reader's replica holds wm rows 1 and 2 and both manifest rows, but not P2's descriptor row. The capture resolves to P1, the superseded pack (clickhouse_reader.py:163-177). |

### Mutations (expected to fail)

| Config | Mutation | Violated | Trace | Counterexample |
|---|---|---|---|---|
| SplitBarrier | barrier+fence as a SELECT, then an unconditional INSERT | PinnedStable (and MonotonicLanding) | 24 | Writer 1 passes the check at v=1. The lease moves, since no statement is in flight. Writer 2 publishes v=2, and a reader pins 2. Writer 1's INSERT then lands v=1 below the head, and the members of snapshot 2 grow. |
| UnsafeTakeover | lease moves while a statement is in flight | PinnedStable | 24 | Writer 1's v=1 statement is admitted, and the lease moves. Writer 2's v=2 lands first. Then v=1 lands below the head. This documents the dependency on the lease timing argument. |
| SkipWmReadback | no identity read-back | CommittedVisible | 25 | i2 publishes v=2. i1's v=1 is refused by the barrier but treated as a success, so pack 1 is committed to the inventory while invisible: skipped forever. |
| OccupancyReadback | `count() > 0` read-back (with the allocator read-back also removed) | CommittedVisible | 25 | Both writers get v=1. Writer 1 is fenced out after a takeover, then sees writer 2's row at v=1, calls it its own, and commits invisible pack 1. |
| SkipRewrite | the retry does not rewrite descriptors | ResolvesNewest | 33 | i1's P1 loses at v=1 to i2's P2 at v=2, then republishes at v=3. P1's descriptors stay at v=1, so snapshot 3 resolves to the superseded P2. |
| SkipClaimReadback | allocator trusts its candidate | AllocUnique | 11 | Two writers claim v=1 concurrently. As SkipClaimReadbackRest shows, nothing downstream breaks in this model: the fence and the barrier alone keep the watermark sound. |

### Reachability witnesses (each must be VIOLATED; all were)

Each witness uses the shared layout with T=1 and C=1. The number is the length of
the witness trace.

- NeverContested: 9
- NeverFencedOut: 11
- NeverLostRace: 24
- NeverOrphanLands: 13
- NeverRetryWins: 36
- NeverSuccessorPub: 13


## Finding: a replayed pack re-promotes itself over a newer pack

**Status: a real gap in main at `a987dfe`, fixed by PR #156.** The `Replay*`
configs keep modelling main as it was, with `RANK_BY_MEMBERSHIP = FALSE`, so they
still fail: they are the bug witness. The `FixMin*` configs model the fix.

The Replay configs use the unmodified protocol. The only thing they add is a pass
that re-indexes a pack which is already published, and the code explicitly
permits that pass:

- It happens after a crash between publish and `commit_packs`. This is "redundant
  work next pass" at clickhouse_catalog.py:1049-1056 and catalog.py:410-416.
- It also happens after an outcome-unknown publish that did land
  (clickhouse_catalog.py:447-452), and after a lagging `committed_pack_ids`
  (:349-353).

`ReplayCrash` / `ReplayCrashPinned` counterexample, in plain steps:

1. Pass A indexes P1, allocates v1, writes P1's descriptor rows at v1 and
   publishes P1@1. The watermark lands and the read-back succeeds.
2. Pass A dies before `commit_packs` (catalog.py:417-418). P1 is visible but not
   in the inventory.
3. Pass B indexes P2, a second pack describing the same capture: "a pack
   mirrored to a second store, or a producer retrying a capture_id"
   (capture-storage-design.md:444-447). It publishes P2@2. A reader pins W=2 and
   resolves the capture to P2, the newest publish.
4. Pass C, a later pass, finds P1 missing from the inventory (catalog.py:365) and
   re-indexes it. It allocates v3 and writes P1's descriptor rows at v3
   (catalog.py:402).
5. The reader pinned at W=2 now resolves the capture to **P1**.
   - The membership clause admits P1, because P1 is published at 1 ≤ 2.
   - `argMax(..., (index_version, store_id, pack_id))` ranks P1's new row
     (3, P1) above (2, P2). Descriptor rows are not bounded by the watermark
     (clickhouse_reader.py:417-448, 521-525).
   - This happens before pass C has published anything. If C then loses its
     lease, exhausts its attempts or crashes, the rows stay and the flip is
     permanent at every head from 2 onwards.

In the other ordering (`ReplayCrash`), C's rows at v3 are written before B
publishes. P2's publish is then shadowed from the moment it lands.

**Contracts this violates.**

- clickhouse_reader.py:36-39 says "the pin still resolves to the pack it was
  taken over".
- The newest-wins supersession argument at catalog.py:520-531 assumes that a
  pack's descriptor rows sit only at the version that published it.
- The differential review (catalog-differential-review-2026-09-01.md:145) judges
  a re-publish "Reader-visible corruption: none (rows are byte-identical and
  collapse)". That holds for one pack. It fails once a second pack describes the
  capture.

**Impact.** The reader gets a different locator for the same capture bytes, which
is allowed by the identity rule. It breaks pinned-selection determinism, and it
can point readers at a pack that was deliberately superseded.

**Fix (PR #156).** The reader ranks a member pack by its paired manifest version
at or below W, which the membership subquery already computes, instead of by the
descriptor row's `index_version`. Of the pack's paired versions it takes the
first, `min(index_version)`. Taking the newest (`max`) would fix pinned
stability but still let a superseded pack come back once its republish becomes
the head (`FixMax_ReplayCrash`).


## Python vs C++ divergences noted

1. **`commit_packs` failing on the conflict path.**
   - Python (catalog.py:566-583) re-raises `SnapshotPublishConflictError` from
     the commit failure.
   - C++ (indexer.cpp:300-306) calls `commit_packs` unguarded. A transport error
     there replaces `kPublishConflict`, and the visible packs stay out of the
     inventory, so the next pass re-publishes them.
   - This is the issue at catalog-differential-review-2026-09-01.md:145, fixed
     in Python only. It is also one more route into the replay finding above.
     This model does not include it: `NeverConflict` holds in the faithful model,
     so the path is unreachable here.
2. **Exhaustion.** C++ uses `continue` where Python uses `break`
   (indexer.cpp:311-318). The two are equivalent: both raise after
   `max_publish_attempts` without committing.
3. **Randomized skip.** C++ uses `rng() % (8*attempt+1)` from an `mt19937_64`
   seeded per call. Python uses `secrets.randbelow`. The range is the same; C++
   has a negligible modulo bias.
4. **Allocator floor and lock.** Both read the floor with the same two deciding
   statements. Both serialize the allocator on the writer lock (Python
   `_serial`; C++ `serial_`, catalog_writer.cpp:462-468).
5. **Watermark publish.** The statement, the two-question read-back and the
   manifest-whole check match line for line.
