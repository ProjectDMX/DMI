# Catalog descriptor key

A proposal to add pack identity to the descriptor table's sort key, and a note
on where the catalog goes after that.

## The problem

Every capture gets one row in the descriptor table saying, in effect, "capture
X lives in pack P at offset N". The table is a `ReplacingMergeTree` ordered by

```
(tenant_id, experiment_id, run_id, captured_at_ns, capture_id)
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
sort key, and `argMax(..., index_version)` still picks one row per capture. A
reader pinned before the new pack was committed does not see its rows at all,
because the commit-log membership clause excludes that pack; a reader on a
fresh watermark sees both and takes the newer. Re-packing goes from "would
corrupt pinned snapshots" to "supported, under the same commit-log rules as
everything else".

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
- **The public `FINAL` views** would show one row per pack for a superseded
  capture, since `FINAL` collapses on the new key. That extends the existing
  caveat that those views bypass the snapshot bound; it is not a new problem.

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
  and republishes, bounded at `max_publish_attempts`. Only the manifest is
  rewritten. Descriptors keep the version they were written with, which is
  harmless now that pack identity is in the descriptor sort key: `index_version`
  there is only a tiebreaker among byte-identical rows and never decides
  visibility.
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
protocol as a version**, which is safe here for the same reason: a lease claim
row is inert. Writing one confers nothing on its own, because the only thing
that ever reads the table is the fencing predicate inside a publish, and that
predicate names a single row. A term claimed by two publishers is abandoned by
both, holds no lease, and is takeable at once by whoever claims a higher term.

`term` is the monotonic slot, not the clock -- a wall-clock ordering could tie,
and then "the head of the table" would be ambiguous. `acquired_at_ns` and
`expires_at_ns` are stamped by the **server**, and the fence compares them
against the **server's** clock, so no publisher's own clock, and no skew between
two of them, decides whether a lease is live.

The point of the whole thing is one sentence:

> **The fencing check rides inside the same server-side statement as the
> visibility write. A publisher whose lease has been taken over writes NOTHING
> -- it does not write and then discover it lost.**

That is what separates it from the two designs this document has already
rejected. Both of those wrote first and checked afterwards, and no check that
runs after a write can withdraw what the write already made durable.

```sql
INSERT INTO {prefix}_index_watermark (...)
SELECT ... FROM system.one
WHERE (SELECT max(index_version) FROM {prefix}_index_watermark) < V
  AND (SELECT (lease_id, expires_at_ns > toUnixTimestamp64Nano(now64(9)))
       FROM {prefix}_publisher_lease
       ORDER BY term DESC, lease_id DESC LIMIT 1) = (:my_lease, true)
```

**One subquery returning a tuple, not two returning a column each.** The
two-read form was implemented first and is unsound, which is worth stating
because the shape is the obvious one to reach for: two scalar subqueries are two
reads of the lease table, so a takeover landing between them answers with the
OLD holder's `lease_id` and the NEW holder's `expires_at_ns`, and the fence
passes for a publisher that has already been replaced -- the exact failure the
fence exists to stop, reintroduced by the way it is written. One subquery reads
one row, and the pair it returns describes that row. It is cheaper too, but that
is the smaller half: 3.06 ms unfenced, 3.85 ms with this form, 4.84 ms with two
subqueries, measured on 25.12.

The manifest INSERT carries the same predicate, over an `arrayJoin` of the
packs it is admitting. Those rows would be inert either way -- membership pairs
them with a watermark row that would never exist -- but "wrote nothing" is a
property a test can assert directly, and "wrote something harmless" is one that
has to be re-argued every time the membership clause changes.

A publisher renews before every publish, which costs a round trip and buys the
entire safety margin: at the moment the fence is evaluated its lease has
essentially a full `lease_ttl_ns` left, and a takeover cannot happen until that
expires. The publish statement carries `max_execution_time = publish_timeout_ns`
(required to be below the TTL), so the server itself aborts a publish that has
been in flight that long.

**Measured, with the lease in place**: 200 trials, two writers each racing to
acquire the lease, allocate a version and publish from a barrier. Both published
in the same trial **0 times**; 173 publishes succeeded, 227 attempts were
refused at the lease, and **0** reached the version barrier. The lease
serialises publication before the barrier's window can open. Removing the fence
from the two writing statements and rerunning
`test_a_taken_over_publisher_writes_nothing_at_all` makes it fail with `DID NOT
RAISE`: the publisher that had demonstrably lost the lease published anyway.

#### What it costs

Measured on 25.12 against a local server, base = the same code without the
lease:

| | before | after | |
|---|---:|---:|---|
| one `publish_snapshot` (16 packs) | 6.7 ms | 15.2 ms | +8.5 ms |
| one `index()` pass (16 packs, 4096 rows) | 160 ms | 173 ms | +8% |
| `bench_capture_search` pages and selectivity | -- | -- | inside the base's own 4-25% run-to-run spread |

The read path is untouched, as it should be: the lease is on the write path, and
the membership clause gained one column in a join between two tables that hold
one row per publish.

The 8.5 ms is three round trips of renewal (head read, claim, read-back) plus
0.8 ms of fence on each of the two writing statements. It is paid once per
`index()` call, not once per pack or per row, which is why it disappears into a
pass that already spends 160 ms reading footers and inserting descriptors.

Renewing on **every** publish rather than only when the lease is close to
expiring is deliberate. It removes the client clock from the design entirely --
this branch has spent two rounds getting wall clocks out of the ordering -- and
it maximises the margin the residual argument below depends on. Renew-when-stale
would recover most of the 8.5 ms and needs a monotonic clock and a staleness
branch to do it; that is a trade worth making only if publish frequency ever
rises enough for 8.5 ms to matter.

#### What this does not close

Two things, stated as precisely as the rest of this document tries to be.

**The takeover instant.** For both publishers to write, publisher A's publish
statement must evaluate the fence before B's lease row commits, and still be in
flight when B publishes. Every timestamp in that sequence is on the server's own
clock, and it forces A's single INSERT to stay in flight from before its lease
expired until after B took over -- essentially a whole `lease_ttl_ns`, when A
renewed immediately before issuing it, and past a `max_execution_time` set below
that. So the window is bounded by two knobs rather than by an unbounded
scheduler artifact. It is not *zero*: `max_execution_time` is checked between
processing blocks rather than pre-empted, so a statement blocked in a lock can
overrun it. This has not been reproduced; it also has not been proven
impossible, and the difference matters.

**Lease acquisition is not the window it looks like.** Two publishers claiming
one term both abandon it, so a contested term holds no lease and the fence
matches nobody -- the same argument that makes the version allocator sound. The
cost is liveness, not safety: a contested term locks *everyone* out, including a
previous holder whose lease was still live, until somebody claims a higher term.
That is deliberate. A contested head means the catalog does not know who is in
charge, and refusing to publish is the conservative answer.

**A crash between a claim and its read-back** leaves a claim row nobody uses.
It looks like a live lease until it expires, so the next publisher waits out one
`lease_ttl_ns`. `release_publisher_lease()` exists so an orderly restart does
not pay that; a crash does.

**Replication.** All of this rests on a later write observing an earlier one.
The read-backs carry `select_sequential_consistency`, which is the read half. The
write half -- `insert_quorum`, and quorum-durable descriptor inserts before their
watermark row -- is a deployment decision and is deliberately not claimed here.

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
