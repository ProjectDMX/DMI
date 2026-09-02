# Catalog descriptor key

A proposal to add pack identity to the descriptor table's sort key, and a note
on where the catalog goes after that.

## The problem

**This section describes the schema as it was when this document was written.
The proposal below shipped**, so the sort key here is version 1's and not the
one `ensure_schema` creates; the current key is in *The proposal*.

Every capture gets one row in the descriptor table saying, in effect, "capture
X lives in pack P at offset N". The table was a `ReplacingMergeTree` ordered by

```
(tenant_id, experiment_id, run_id, captured_at_ns, capture_id)     -- since replaced
```

which is a capture's identity and nothing more. That engine collapses rows
sharing a sort key during background merges, keeping the one with the highest
`index_version` and deleting the rest.

That collapse is only safe while two rows with the same key are byte identical.
The design says they always are, because a capture belongs to exactly one pack.
Nothing enforces it.

When two rows for one capture disagree, a merge silently deletes one of them --
at an unpredictable later time, with nothing connecting the deletion to whatever
caused it. A reader whose pinned snapshot covers only the deleted row's pack
then fails with `selection no longer resolves at its catalog watermark`. The
bytes are still in object storage; the pointer to them is gone.

## When two rows disagree

Three ways, and only the third needs code that does not exist yet.

**A pack copied to a second store.** An operator copies a pack object from the
filesystem store into an S3 bucket for redundancy, then reconciles that bucket
-- a supported operation. Replay dedup is keyed on `(store_id, pack_id)`, so a
new store is not a replay: the indexer writes a fresh descriptor row for every
capture in that pack. Same captures, same sort key, different `store_id` and
`object_key`. Possible today with shipped components.

**A producer retry across a pack boundary.** `capture_id` comes from the
producer under an at-most-once contract. Within one open pack a repeat is
caught (`DuplicateCaptureError`), but once a pack is sealed and uploaded the
pipeline has no memory of it. A producer that retries after an ambiguous
failure -- a crash, a timeout, a restart -- writes the same capture into a
second pack. Possible today, and retry-after-ambiguity is the ordinary
distributed failure, not an exotic one.

**Re-packing or compaction tooling.** A maintenance job that rewrites old packs
into new ones produces exactly this shape deliberately. Does not exist yet; it
is the natural next tool for any object-store system.

## The proposal

Append pack identity to the sort key:

```
ORDER BY (tenant_id, experiment_id, run_id, captured_at_ns, capture_id,
          store_id, pack_id)
```

Rows describing one capture in different packs now have different keys, so no
merge can delete either. Rows describing the same capture in the same pack --
the ordinary replay case the engine is there for -- still share a full key and
still collapse, and those rows really are identical.

Supersession then falls out of machinery the reader already has. Keep the
`GROUP BY` on the five capture-identity columns, decoupled from the now longer
sort key, and an `argMax` grouped on capture identity still picks one row per
capture. A reader pinned before the new pack was committed does not see its
rows at all, because the commit-log membership clause excludes that pack; a
reader on a fresh watermark sees both and takes the newer. Re-packing goes from
"would corrupt pinned snapshots" to "supported, under the same commit-log rules
as everything else".

> **What shipped is not the per-column `argMax(..., index_version)` this
> section originally specified.** `e93a2c8` replaced it, because
> `index_version` does not order the rows it was being asked to order: one
> `index()` call allocates ONE version for a whole batch, so two packs
> describing a capture in a single indexing pass produce rows whose ordering
> key is EQUAL, and ClickHouse leaves the winner of a tie undefined. Measured
> on 25.12, one corpus pinned at a single watermark resolved to a different
> pack at `max_threads = 1` than above it, and to a different one again once a
> merge had put both rows in one part.
>
> The shipped projection is **one** `argMax` over a tuple of every resolved
> column, ordered on the tuple `(index_version, store_id, pack_id)`. The tuple
> key restores a total order (supersession is unchanged -- `index_version`
> still leads); the single aggregate makes a mixed descriptor -- `store_id`
> from one row and `object_key` from another -- structurally impossible rather
> than merely unobserved. Twenty-seven aggregates ordered on the tuple are also
> correct and cost +291% at a 100-row page, because ClickHouse compares a tuple
> ordering argument through a generic `Field` once per row per aggregate. See
> `clickhouse_reader._projection`.

### What it costs

- **Query performance: nothing measurable.** The two columns append after the
  existing prefix, so tenant-range pruning and the `capture_id` bloom filter
  behave exactly as they do now.
- **Code: small.** The DDL in `ensure_schema`, decoupling `GROUP BY` from the
  sort-key constant in the two reader queries, tests, and one live test proving
  a re-packed capture resolves to the old pack under an old pin and the new pack
  under a fresh one.
- **Disk: slightly more rows** if re-packing ever happens, since superseded
  pointers are kept rather than collapsed. Negligible.
- **The public `FINAL` views** show one row per pack for a superseded capture,
  since `FINAL` collapses on the new key. That is deliberate: `{prefix}_capture`
  means "published descriptor rows, engine-deduplicated", one row per
  `(capture, store, pack)`, and choosing between two packs is supersession,
  which belongs to the reader. *(This bullet used to add that those views
  "bypass the snapshot bound". `d50c778` bounded `{prefix}_capture` to the
  packs the published snapshot contains, using the same `(index_version,
  publish_id)` pair the reader's membership clause uses.
  `{prefix}_pack_inventory` still carries no bound and needs none: `index()`
  writes the inventory only after a successful publish. What the view still
  does not do is PIN -- it re-evaluates membership on every query, so two reads
  a moment apart can see different sets of packs; a caller that needs a stable
  selection uses the reader.)*

### Why the timing matters

ClickHouse cannot change a table's `ORDER BY` in place. Before this schema
ships anywhere, the change is one line in `ensure_schema` and every deployment
starts correct. After it ships, the same change is a create-new-table-and-copy
migration over the whole descriptor table.

That is the entire argument for doing it now rather than when the first
re-packing tool is written.

## Alternatives considered

**Enforce uniqueness at write time.** Check per capture whether it is already
described before indexing a pack. Expensive on the ingest path, racy between
concurrent indexers, and it forbids re-packing rather than supporting it.

**Leave it.** Defensible while nothing re-packs, but scenarios A and B above are
reachable today, and the cost of the fix rises sharply after the first
deployment.

## The mid-batch gap, and why the obvious fix fails

Snapshot membership was a predicate over a live table (`{prefix}_pack_commit_log`,
since replaced by the manifest described below):

```
(store_id, pack_id) IN (SELECT store_id, pack_id FROM {prefix}_pack_commit_log
                        WHERE index_version <= W)
```

Commit rows are written before the watermark is published, so indexer A can
write its rows at version 5, indexer B can publish version 6, a reader can pin
W = 6, and A's rows then appear inside that pinned snapshot. The snapshot grew
after it was pinned.

The natural repair -- write membership rows at publish time instead, and have
the publisher verify afterwards that it is the highest published version --
does not work, and this was demonstrated against a live server rather than
argued:

```
pinned watermark          : 6
packs in pinned snapshot  : one
publisher of 5 detects it lost the race to 6, as designed
packs in pinned snapshot  : two
```

The loser writes its membership rows before it discovers it lost. Those rows
sit at a version below W, the clause admits everything below W, and the check
that would have vetoed them runs after they are already durable.

The general form is worth stating, because it rules out a whole family of
attempted fixes: **no `<= W` predicate over an append-only table can be made
sound by a post-write check.** The write that confers visibility necessarily
precedes the check that would withdraw it. Soundness has to come from the
shape of the data, not from a verification step.

### Three repairs, in increasing cost

**Gate visibility on a marker written after the check.** Order the publish as
membership rows, then verify, then the watermark row, and require a version to
appear in the watermark table before its rows count. A loser never writes that
row, so its rows are permanently inert. This closes the window that is
reachable today -- the whole descriptor-writing phase, seconds wide -- and
leaves a residual of roughly one round trip, when two publishers both read the
barrier before either writes its watermark row. Cheap, strictly better than
today, still not sound.

**Write the full membership set per version.** Each publish enumerates every
pack in the snapshot, and membership becomes `index_version = W` exactly. The
set for W is complete before W's watermark row exists, so nothing another
publisher does later can alter it. Airtight, and it pays a per-publish cost
proportional to the whole catalog rather than to the batch, which grows without
bound.

**Require contiguity.** Publish version V only once every lower version is
either published or durably abandoned. Then publish order equals version order
by construction and `<= W` is sound again, at O(new packs) per publish and with
no chain to walk at read time. The cost moves to liveness: a publisher that
crashes holding a version blocks the ones behind it until an abandonment
protocol reclaims it, and abandonment has to be raced correctly against a
publish that is merely slow. The same append-and-read-back pattern the version
allocator already uses works here, because an abandonment marker is inert --
unlike membership rows, writing one confers nothing.

Parent chaining -- each publish records the snapshot it builds on, and
membership follows the chain back from W -- is sound at the same write cost,
and moves the expense to resolving the chain at pin time, which needs periodic
checkpointing to stay cheap as publishes accumulate. It is the table-format
answer, and worth reaching for if the catalog ever grows a second writer that
cannot be serialised.

### What shipped

None of this is reachable with a single indexer: one writer cannot race itself.
The whole concern is about the multi-indexer future the version allocator was
built for -- and "single indexer" was an assumption nothing enforced, which is
what the publisher lease below changes.

The marker-gated variant shipped first, with the barrier tightened as far as it
goes without a contract change:

- Membership moved to `{prefix}_snapshot_manifest`, written by
  `publish_snapshot` rather than by `commit_packs`, and a row counts only once
  its version also appears in `{prefix}_index_watermark`. A publish that loses
  never writes that watermark row, so the rows it already wrote are inert.
- The barrier and the visibility write are **one server-side statement**:
  `INSERT INTO watermark SELECT ... WHERE (SELECT max(index_version) FROM
  watermark) < V`. There is no client round trip between "am I the highest?"
  and "I am now visible", so no network hop, GC pause or scheduler stall sits
  inside the window. This also subsumes the indexer's non-monotonic-version
  guard -- the server now refuses a version that is not strictly above the
  head.
- A losing publish raises `SnapshotPublishRaceError`; the indexer re-allocates
  and republishes, bounded at `max_publish_attempts` and terminating in
  `SnapshotPublishExhaustedError` (chained to the last race) when the budget
  runs out. The manifest is rewritten at the new version, and so are the
  DESCRIPTORS. Visibility would survive without the rewrite -- **it is decided
  by membership, never by `index_version`**: a descriptor row is in a snapshot
  when its pack is in the manifest of a publish that reached the watermark
  table -- but supersession would not. `index_version` leads
  `_RESOLUTION_ORDER`, the primary ordering between two rows describing one
  capture in two DIFFERENT packs, so rows left at the lost version would rank
  below another pack's rows written between the lost and the winning version:
  the reader would resolve the capture to the OLDER publish's pack, and to its
  locator, which is exactly the field that may differ. The rewrite is
  byte-identical rows at the winning version; the superseded rows share their
  full sort key with them (pack identity included), so the engine collapses
  each pair and `argMax` resolves the new version in the meantime.
- **The publish verifies that it owns the version, not that the version is
  occupied.** Each attempt mints a `publish_id`, writes it on its manifest rows
  and on its watermark row, and reads that column back. The check it replaced --
  `count() > 0` for the version -- treated a row written by anything else as
  success, so a publisher could be told it had published a snapshot it did not
  publish and would then record those packs in the replay inventory, where no
  later pass would pick them up. Membership pairs `(index_version, publish_id)`
  for the same reason: on the version alone, the *contents* of snapshot V are
  whatever anyone wrote at V. The sole-claimant allocator makes a foreign row at
  V unlikely, not impossible, and reading the identity back costs exactly what
  counting cost.
- `index()` orders itself descriptors -> publish -> pack inventory. The
  inventory is the replay guard, so writing it before a successful publish
  would let a crash leave a pack skipped forever *and* invisible. Last means a
  crash costs redundant work, never invisibility.

**That narrowed the gap and did not close it.** The window went from the whole
descriptor-writing phase -- seconds, many INSERTs -- down to the server-side
overlap of two conditional INSERTs. Two publishers could still both evaluate
`max(index_version) < V` before either row was durable and both land; the lower
one then became visible underneath a watermark that was already pinned, and it
happened silently.

Measured on 25.12, on the statement shape itself: two clients issuing the
conditional publish from a barrier, with a third polling `max(index_version)`
throughout, 200 trials. In **2 of them (1%)** the poller saw the higher version
alone -- a pin a reader would have taken -- and the lower version landed
underneath it afterwards. That is the residual, and one percent of concurrent
publishes is not a rounding error.

### The fenced publisher lease

The residual has a precondition: **two publishers writing concurrently.** All
three sound repairs above attack the other half of it -- they change the shape
of the data so that concurrent publication is safe. The lease attacks the
precondition instead, and enforces the single-indexer invariant the design has
assumed from the start rather than merely documenting it.

`{prefix}_publisher_lease` holds `(term, lease_id, holder, acquired_at_ns,
expires_at_ns)`. It is append-only and claimed by the **same sole-claimant
protocol as a version**, but with an extra late-claim rule. Writing a row does
not confer ownership until its read-back succeeds. A competing row can arrive
after that success, however, and the fence resolves one row from a contested
term. The protocol therefore quarantines a contested head until the latest
`expires_at_ns` recorded at that term. Only after every possible returned lease
has expired may a publisher claim a higher term.

**Where that safety comes from is the read-back, not the fence.** Both this
document and `capture-storage-design.md` used to say that a contested head term
"satisfies neither" of the fence's two conditions. It does not: the fence
resolves ONE row with `ORDER BY term DESC, lease_id DESC`, so the
higher-ordering of two claimants at the top term satisfies it exactly as a real
holder would. Verified on 25.12, and note that the ordering is ClickHouse's UUID
collation -- the low 64 bits first -- so it is not the text order either. What
makes the term safe is `ClickHouseLeaseCoordinator.claim`: a claimant that sees two rows is not
returned a `PublisherLease`, and nobody may advance past that term while a
claim at it remains live. This also protects a claimant that completed
read-back before the competing row arrived.

`term` is the monotonic slot, not the clock -- a wall-clock ordering could tie,
and then "the head of the table" would be ambiguous. `acquired_at_ns` and
`expires_at_ns` are stamped by the **server**, and the fence compares them
against the **server's** clock, so no publisher's own clock, and no skew between
two of them, decides whether a lease is live.

The point of the whole thing is one sentence:

> **The fencing check rides inside the same server-side statement as the
> visibility write. A publisher whose lease has been taken over makes NO
> SNAPSHOT VISIBLE -- it does not make something visible and then discover it
> lost.**

That is what separates it from the two designs this document has already
rejected. Both of those wrote first and checked afterwards, and no check that
runs after a write can withdraw what the write already made durable.

The sentence used to read "writes NOTHING", and that is a stronger claim than
the code makes. A publish is TWO fenced statements, the manifest rows and then
the watermark row, so the guarantee is per statement: a takeover landing
between them leaves the manifest rows of the first behind while the second is
refused. Those rows are inert -- membership pairs them with a watermark row
that will never exist -- so the safety claim holds exactly as stated. See
*What this does not close*.

```sql
INSERT INTO {prefix}_index_watermark (...)
SELECT ... FROM system.one
WHERE (SELECT max(index_version) FROM {prefix}_index_watermark) < V
  AND (SELECT count() FROM (SELECT lease_id, expires_at_ns
                            FROM {prefix}_publisher_lease
                            ORDER BY term DESC, lease_id DESC LIMIT 1)
       WHERE lease_id = :my_lease
         AND expires_at_ns > toUnixTimestamp64Nano(now64(9))) = 1
```

**One subquery reading one row, not two returning a column each.** The two-read
form was implemented first and is unsound, which is worth stating because the
shape is the obvious one to reach for: two scalar subqueries are two reads of
the lease table, so a takeover landing between them answers with the OLD
holder's `lease_id` and the NEW holder's `expires_at_ns`, and the fence passes
for a publisher that has already been replaced -- the exact failure the fence
exists to stop, reintroduced by the way it is written. Here the ordered
`LIMIT 1` resolves the head once and both conditions are asked of that one row.
It is cheaper too, but that is the smaller half: 3.06 ms unfenced, 3.85 ms with
a fence, 4.84 ms with two subqueries, measured on 25.12.

`count() ... = 1` rather than a tuple comparison, because an empty lease table
has to make the predicate FALSE and a tuple cannot. A scalar subquery selecting
a tuple from no rows is not NULL on 25.12; it raises `Code: 125, Scalar
subquery returned empty result of type Tuple(UUID, UInt8) which cannot be
Nullable` -- a raw `ServerException`, not a `CaptureStorageError`. `count()`
over the same one-row read always returns a row, so an empty table answers 0.
No publish reaches that state today, because a publish renews first and a
renewal leaves a row; the GC obligation below makes it reachable.

The manifest INSERT carries the same predicate, over an `arrayJoin` of the
packs it is admitting. Those rows are inert either way -- membership pairs them
with a watermark row that would never exist -- and the fence is what keeps a
publisher fenced out *before* its first statement from writing them at all.
What it cannot do is make the pair of statements atomic; see *What this does
not close*.

A publisher renews before every manifest chunk and before the watermark. Each
renewal costs **three** round trips -- the head read, the claim INSERT and the
read-back -- measured at 5.69 ms median on 25.12. Every fenced statement
requires more than `publish_timeout_ns` of lease life remaining and carries
`max_execution_time` set to that same timeout converted to the **seconds** the
setting takes (whole seconds, validated at construction and sent as an integer
-- a fractional value can truncate to 0 on older serialization paths, which
ClickHouse reads as *no* limit -- and required to be below the TTL). Thus a
later statement cannot start on the tail of an earlier renewal, and the server
aborts any admitted statement that remains in flight for the full margin.

Every conditional manifest INSERT is read back before the next renewal. This
matters because a fence refusal writes zero rows without raising: renewing and
continuing without the read-back could make the watermark visible after one
manifest chunk was silently omitted.
A statement the server aborts this way raises the driver's own exception
(`timeout_overflow_mode='throw'`, Code 159) out of `publish_snapshot` before
the ownership read-back -- the same species as the Code 125 path above, and
like it outside the module's error taxonomy. The publish outcome is unknown at
that point; the safe reading is "re-acquire and re-index", and because
`commit_packs` has not run, a row that did land costs only redundant work on
the next pass.

**Measured, with the lease in place**: 200 trials, two writers each racing to
acquire the lease, allocate a version and publish from a barrier. Both published
in the same trial **0 times**; 173 publishes succeeded, 227 attempts were
refused at the lease, and **0** reached the version barrier. The lease
serialises publication before the barrier's window can open. Removing the fence
from the two writing statements and rerunning
`test_a_taken_over_publisher_writes_nothing_at_all` makes it fail with `DID NOT
RAISE`: the publisher that had demonstrably lost the lease published anyway.

#### What it costs

The earlier one-renewal implementation measured on 25.12 as follows:

| | before | after | |
|---|---:|---:|---|
| one `publish_snapshot` (16 packs) | 6.7 ms | 15.2 ms | +8.5 ms |
| one `index()` pass (16 packs, 4096 rows) | 160 ms | 173 ms | +8% |
| `bench_capture_search` pages and selectivity | -- | -- | inside the base's own 4-25% run-to-run spread |

The read path is untouched, as it should be: the lease is on the write path, and
the membership clause gained one column in a join between two tables that hold
one row per publish.

Those figures predate per-statement renewal and manifest read-back. The current
protocol adds one renewal and one verification per manifest chunk, so it needs
remeasurement; its write-path cost scales with the number of byte-budgeted
chunks. The read path remains untouched.

Renewing on **every fenced statement** is deliberate. It removes the client
clock from the decision and restores the full margin before later chunks and
the watermark. Renew-when-stale would need a trusted staleness decision between
statements, which is the condition this protocol avoids.

#### What this does not close

Stated as precisely as the rest of this document tries to be.

**The fence is per STATEMENT, not per publish.** A publish issues two of them:
the manifest rows for the packs it is admitting, then the watermark row that
admits them. Both carry the fence, so neither can land after a takeover -- but
the gap between them is a full client round trip, and `max_execution_time`
bounds each statement rather than the pair, so nothing bounds the gap. A
takeover landing there leaves the first statement's manifest rows behind while
the second is refused.

Those rows are **inert**: membership requires the manifest row and a watermark
row from the SAME publish, and that watermark row does not exist, so no
snapshot admits them, no reader sees them and the public view does not show
them. The safety claim -- no snapshot becomes visible -- is unaffected. What is
wrong is the stronger claim, which this document, `capture-storage-design.md`,
`catalog.py` and the writer's own refusal message all used to make: that such a
publisher "wrote nothing".

Reproduced on 25.12 by wedging a takeover into the gap
(`test_a_takeover_between_the_two_publish_statements_leaves_orphan_rows`), and
quantified with a hostile TTL -- 7 ms, just long enough for the renewal and the
manifest INSERT and not for the watermark INSERT -- over 50 publishes of three
packs each: **150 orphan manifest rows, 0 published versions**, every attempt
refused. They fall under the GC obligation below, which already covers manifest
rows whose version never reached the watermark table.

Closing it properly means making the two statements one, which means
`arrayJoin`-ing the membership rows and the watermark row out of a single
INSERT into two tables -- ClickHouse has no such statement -- or moving
membership into the watermark row itself, which changes the read path. Neither
is worth doing for rows that are inert.

**The takeover instant.** For both publishers to make a snapshot visible,
publisher A's watermark statement must evaluate the fence before B's lease row
commits, and still be in flight when B publishes. Every timestamp in that
sequence is on the server's own clock, and it forces A's single INSERT to stay
in flight from before its lease expired until after B took over -- essentially a
whole `lease_ttl_ns`, when A renewed immediately before issuing it, and past a
`max_execution_time` set below that. So the window is bounded by two knobs
rather than by an unbounded scheduler artifact. It is not *zero*:
`max_execution_time` is checked between processing blocks rather than
pre-empted, so a statement blocked in a lock can overrun it. This has not been
reproduced; it also has not been proven impossible, and the difference matters.

**Lease acquisition is not the window it looks like.** A claimant can complete
its singleton read-back before a delayed competitor inserts at the same term.
The contested head therefore locks *everyone* out of acquiring a higher term
until the maximum expiry at that term. The fence may continue to admit its
ordering winner during that interval, but no successor can overlap it. The cost
is bounded liveness; the benefit is preserving the returned lease's full safety
margin.

**Two writers can both believe they hold a lease.** The sole-claimant read-back
proves a claimant is alone at ITS term, not that its term is the head. A
claimant whose head read preceded a rival's row claims below that rival, reads
back alone, and is handed a live `PublisherLease` while the rival sits above it.
Verified on 25.12: two writers holding terms 1 and 9, both live by their own
reckoning. Safety holds -- only the head publishes, the fence says so, and the
loser is refused at its next renewal with a message naming the actual holder --
but "only one publisher holds the lease" is not literally true of the client
objects, only of the row the fence resolves.

**A crash between a claim and its read-back** leaves a claim row nobody uses.
It looks like a live lease until it expires, so the next publisher waits out one
`lease_ttl_ns`. `release_publisher_lease()` exists so an orderly restart does
not pay that; a crash does.

**Replication.** All of this rests on a later write observing an earlier one.
The read-backs carry `select_sequential_consistency`, which is the read half.
`ClickHouseCatalogConfig.insert_quorum` supplies the opt-in write half for lease
and descriptor rows, including the fenced release tombstone.

Two consequences to carry forward:

**Dead manifest rows accumulate.** Every lost publish leaves membership rows at
a version that will never reach the watermark table. They are inert -- no
snapshot can ever admit them, and versions are unique so nobody will publish
that version later -- but they are never collected. That is a new GC
obligation: any retention job may delete manifest rows whose `index_version` is
absent from the watermark table. Nothing does today.

**Detection remains available, and it still costs a contract change.** The
takeover-instant residual can be made loud instead of silent by the same means
the wider race could: pin `(W, generation)` where the generation counts
watermark rows `<= W`, carried as a scalar subquery in the same statement as
the page so the number is consistent with the rows returned -- checking it in a
separate round trip has its own window and is a false guarantee. Verified live:
a late lower publish moves the count, and the scalar folds as a constant
alongside the `GROUP BY`/`argMax`/`LIMIT` the reader already emits. It is still
left out, because it changes what a pinned read promises -- a walk becomes
"stable, or `SnapshotMoved` and re-pin" -- and it bumps the cursor format,
invalidating in-flight cursors. That is a reviewer decision, not an
implementation detail. It is a much smaller prize now than it was: the thing it
would detect went from one percent of concurrent publishes to a window nobody
has managed to reach.

## Where the catalog goes later

The deeper version of this problem is that immutable facts ("capture X is in
pack P") are stored in an engine whose job is keeping the latest version of a
mutable thing, and that a snapshot is an open predicate (`index_version <= W`)
over a table still being written. The sort-key change removes the sharp edge;
it does not change that shape. The same shape is also behind the mid-batch
membership gap and the fact that no retention or GC policy exists yet.

The clean-slate alternative is the table-format design: publish an immutable
manifest per snapshot ("snapshot 42 = 41 plus these packs"), advance a single
pointer by compare-and-swap, and demote ClickHouse to a rebuildable search
index over those manifests. Pinned snapshots become closed sets written before
they are named, so the mid-batch gap cannot occur; merges stop being
load-bearing because the index is rebuildable; and GC becomes "delete any pack
no live manifest references".

If that is ever built, evaluate Apache Iceberg and Delta Lake (`delta-rs`)
before hand-rolling manifests -- snapshot isolation, atomic multi-writer
commits, and snapshot expiry are their core competency, and they are hardened
in exactly the places our reviews keep finding holes. The deciding experiment
is whether Garage supports the conditional-put semantics a catalog-free commit
needs.

**Performance is not a reason to move.** The pack data path -- ranged reads
against immutable objects -- is identical under every option. What differs is
the metadata path, and the current design is the fastest of them: membership,
watermark and descriptors live in one engine, so a pinned search is a single
query (measured: 1.9 ms watermark pin, 120-150 ms pinned pages, point lookups
pruned to one tenant's granule range). Moving authority out of ClickHouse adds
a manifest fetch and, more importantly, makes the serving index trail the
authority. A table format adds commit round-trips and small-file compaction on
top, and still needs ClickHouse for serving latency -- so it would be adopted
for correctness generality and GC, never for speed.

Keep the catalog ClickHouse-native and harden it. Revisit when re-packing or
retention actually forces the question; the descriptor rows migrate as plain
data whichever way it is answered.
