# Capture storage design

Status: Accepted; host storage reference through Phase 5 implemented

This document defines a clean-slate host persistence architecture after Ring².
It does not change the CUDA producer, ring layout, or device-to-host transport.
The new path remains opt-in until CPU-only, live-store, and compatibility gates
pass.

The visual companion is
[`capture-storage-pipeline.html`](capture-storage-pipeline.html).

## Implementation status

The first host-only slice is available under `dmi.storage.capture`:

| Capability | Status |
|---|---|
| Versioned pack writer and full validator | Implemented |
| Two-range footer index | Implemented |
| Immutable filesystem store | Implemented |
| Stable selection, byte estimation, and range hydration | Implemented |
| CPU pack benchmark and package checks | Implemented |
| Bounded asynchronous pack pipeline and spool recovery | Implemented |
| Garage/S3 store and bounded parallel uploader | Implemented |
| ClickHouse metadata projection | Implemented and live-tested |
| Opt-in Ring² to Python pack adapter | Reference implementation |
| Core tensor summaries and the extension registry | Implemented |
| Summary artifact stores and long-form scalar metric tables | Planned |

This slice remains opt-in. A reference adapter can now connect the generic
record sink boundary to the Python pack pipeline without changing the CUDA
producer or the existing ClickHouse payload sink.

The Phase 2 implementation adds `HostCapturePipeline`, bounded blocking or
drop-newest admission, size/record/linger/session/shutdown sealing, direct local
persistence, and a durable filesystem spool. The durable sink commits locally;
`SpoolUploader` independently retries staged packs and removes them only after
remote size and SHA-256 verification. One process owns a spool directory.

Phase 3 adds `S3PackStore` and `ParallelSpoolUploader`. The store streams packs
through Boto3 managed multipart transfers, persists DMI SHA-256 and pack
identity as object metadata, supports exact byte ranges and bounded paginated
listing, and resolves ambiguous retries by checking the existing key. The
uploader bounds outer workers and aggregate bytes in flight, applies bounded
backoff only to transient failures, and leaves permanent failures in the spool.

Phase 4 adds bounded notification and prefix-scan discovery, footer-only pack
indexing, client-side ClickHouse batching, and pack commit markers. Descriptor
rows are inserted before their pack marker. An interrupted or ambiguous batch
may replay physical rows, while `ReplacingMergeTree` tables and public `FINAL`
views preserve immediate logical results. Object storage remains sufficient to
rebuild the projection.

## Decision

Treat immutable, self-describing tensor packs in object storage as the only
durable source of truth. A successful object upload is the capture commit.
ClickHouse is a rebuildable analytical projection populated by an independent
catalog indexer.

```text
Ring² host drain
      |
      v
bounded slabs -> pack assembler -> direct upload or NVMe spool
                                      |
                                      v
                         canonical object-store packs
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
            catalog indexer                      summary workers
                    |                                   |
                    v                                   v
            ClickHouse catalog             scalar rows + object artifacts
                    |
                    v
       metadata-first query -> estimate -> selective range hydration
```

The capture host does not run a ClickHouse client, compute summaries, or
coordinate two durable writes. Object-created notifications reduce indexing
latency, while periodic listing and reconciliation provide completeness.

### The other sink, and how one is chosen

This document describes ONE of two storage paths, and they are mutually
exclusive per record runtime:

| | native path | capture path (this document) |
|---|---|---|
| Sink | `ClickHouseRecordSink` (C++) | `ReferencePythonCaptureSink` (C++ bridge) → `CapturePackReferenceSink` (Python) |
| Durable form | one ClickHouse row per record, tensor bytes inline | immutable packs in object storage |
| ClickHouse role | the store itself | a rebuildable index over the packs |
| ClickHouse footprint | one configured table (`offload` by default) | `{prefix}_*` (`dmi_*` by default) |
| Selected by | `create_record_runtime(fmt)` with no `record_sink`, plus a `host_engine`/`db_config` on the engine | `create_record_runtime(fmt, record_sink=reference.native_sink)` |
| Declared by | `MonitoringConfig(storage_backend="native")` | `MonitoringConfig(storage_backend="capture")` |
| Status | production | explicitly reference-only; production sinks remain native-only |

Both are `ring::RecordSink` implementations and the record engine takes exactly
one of them: `RingEngine.create_record` is handed either the host engine or a
sink lease, never both, so no record can reach both backends. Passing an
explicit `record_sink` bypasses the ClickHouse host entirely -- it is neither
validated nor used, which
`test_explicit_record_sink_lease_is_owned_by_native_ring` asserts by failing if
the host is touched at all; `test_create_record_runtime_is_additive_and_uses_active_transport` pins
the other direction.

`MonitoringConfig.storage_backend` declares which of the two an engine is for,
and the engine holds the runtime to it. The choice is otherwise made in two
different places -- a host engine at construction, a `record_sink` at
`create_record_runtime` -- with nothing but the caller's memory connecting
them, so a mismatch was a silent outcome rather than an error: passing a
`record_sink` while a host engine is configured writes packs and leaves a
ClickHouse insert pipeline started, connected and never fed.

```python
engine = MonitoringEngine(
    config=MonitoringConfig(storage_backend="capture"),
    model_id="...",
    ring_config=ring_config,
)                                    # a host_engine here is now refused
runtime = engine.create_record_runtime(
    reference.record_format,
    record_sink=reference.native_sink,  # omitting it is now refused
)
```

`"native"` is the mirror image: it requires a host engine and refuses an
explicit sink. `"none"` is capture and transport with no persistence at all.
The default is `"auto"`, which infers the backend from what was passed -- what
every caller did before the field existed, so nothing that predates it
changes.

Their ClickHouse footprints are disjoint, so the two can share one server: the
catalog's schema guard only ever names `{prefix}_*` objects, and `drop_schema`
only drops what it owns.
`test_the_capture_catalog_ignores_the_native_paths_offload_table` runs a full
publish and teardown beside a populated `offload` table and asserts it is
untouched. A deployment that wants both isolated further should give them
separate databases.

## Goals

- Keep persistence and analytics off the inference path.
- Scale payload upload, catalog indexing, and summarization independently.
- Avoid one object-store request or ClickHouse insert per tensor.
- Bound host memory, disk, concurrency, retries, and read amplification.
- Make the catalog reconstructable from canonical packs.
- Let users and agents inspect summaries before transferring tensor bytes.
- Add storage providers and summarizers without changing Ring².
- Prove the design on CPU-only hosts before changing CUDA code.

## Non-goals

- Changing hook placement, the CUDA producer, or Ring².
- Querying arbitrary tensor contents directly in ClickHouse.
- Providing a cross-system exactly-once transaction.
- Using an embedded database as a second source of truth.
- Selecting production pack sizes or codecs without measurement.

## Performance model

The hot path is deliberately short:

```text
drain -> acquire reusable slab -> append record -> seal pack -> enqueue upload
```

The architecture removes four scaling costs from capture hosts:

- per-tensor network requests;
- ClickHouse insert latency and merge behavior;
- summary computation;
- coordination between payload and catalog durability.

Packs amortize object-store request and protocol overhead. An independent
indexer reads only pack footers and batches rows across all producers before
inserting into ClickHouse. Readers coalesce adjacent selected ranges instead of
downloading complete packs.

Initial tuning ranges are hypotheses:

| Setting | Initial sweep | Purpose |
|---|---:|---|
| Target pack size | 64–256 MiB | Amortize upload and object overhead |
| Maximum linger | 50–100 ms | Bound visibility latency at low volume |
| Upload workers | 1–16 | Find the store/network saturation point |
| Index batch rows | 10k–100k | Avoid small ClickHouse inserts |
| Index batch bytes | 16–64 MiB | Bound memory while preserving batch efficiency |

The pack-and-upload plane must sustain at least 1.2 times the expected
device-to-host rate on representative hardware. A value becomes a default only
after repeated measurements exceed run-to-run variance and correctness tests
remain green.

## Host capture agent

### Admission

The drain thread hands each `CaptureRecord` to a queue bounded by both bytes and
record count. A record owns or references a contiguous CPU payload and immutable
metadata. Queue saturation follows an explicit policy: bounded blocking,
sampling, or dropping. The system never silently grows memory or disk.

### Pack assembly

Pack assemblers use reusable slabs and partition work by stable producer scope
so independent capture streams do not share a global lock. They seal on target
size, linger deadline, session boundary, or shutdown.

Each record is independently encoded. Whole-pack streaming compression is not
used because it would require preceding bytes to decode one selected tensor.
The first implementation supports `none` and one measured block codec.

### Direct and durable modes

Both modes implement the same `PackSink` contract. Direct mode writes a
`PackSource` to a `PackStore`; durable mode first writes that source to the
local spool and exposes the staged file as another streaming `PackSource`.

Direct mode uploads sealed packs from bounded memory. Durable mode writes sealed
packs to local NVMe and uses atomic rename:

```text
.open -> .ready -> upload -> remote verification -> delete
```

At restart, valid `.ready` packs are retried with the same deterministic key.
Ambiguous uploads use remote metadata and checksum verification. The filesystem
state machine is sufficient initially; RocksDB or SQLite is added only if
measurements demonstrate a recovery or scheduling bottleneck.

The spool absorbs bursts and process failure. It cannot compensate for
sustained object-store throughput below the capture rate.

The CPU reference exposes the modes through a common `PackSink` boundary:

```python
config = PipelineConfig(
    max_queue_records=256,
    max_queue_bytes=16 * 1024**2,
    max_pack_bytes=128 * 1024**2,
    max_pack_records=10_000,
    max_linger_ns=100_000_000,
)

sink = DirectPackSink(store)
# Or: sink = DurablePackSink(DurablePackSpool(path, max_bytes=...))

pipeline = HostCapturePipeline(config, sink)
pipeline.start()
result = pipeline.submit(record)
snapshot = pipeline.close(timeout=30)
```

### Opt-in Ring² reference adapter

`CapturePackReferenceSink` is a correctness bridge for exercising this storage
path from a real record ring. It is selected explicitly; omitting
`record_sink` preserves the existing native ClickHouse path, and the two sinks
are never active at the same time.

```python
pipeline = HostCapturePipeline(config, sink)
pipeline.start()
reference = CapturePackReferenceSink(pipeline)

runtime = engine.create_record_runtime(
    reference.record_format,
    record_sink=reference.native_sink,
)

# Bind HookPointV1 and run normal single-worker, single-stream forwards.
engine.flush_and_wait(30.0)  # non-closing; earlier records are durable
engine.close()               # stop Ring/drain/P2P before the Python queue
reference.close(timeout=30.0)
```

The versioned reference wire carries canonical `CaptureMetadata` JSON beside
one or more fixed-shape tensor payload slices. The native sink validates each
slice, then acquires the GIL; the Python target copies it to immutable `bytes`
before admission to the dedicated `HostCapturePipeline`. This adds one copy
and a Python callback per row, so it is intentionally separate from the future
production native pack writer.

The generic record engine owns an exclusive sink lease. Acquisition happens
before replacing an active Ring, while release happens only after the record
worker stops; the reference adapter does not add a separate lifecycle path.

`HostCapturePipeline.flush()` is a FIFO, repeatable, non-closing barrier. It
seals the current pack and waits for `PackSink.persist()` for every earlier
admission. With `DirectPackSink` that means object-store commit; with
`DurablePackSink` it means local spool fsync/rename, not remote upload.

The adapter does not broaden Ring² concurrency: each process/rank owns its own
Ring and pipeline, and producers retain DMI's existing serialized CUDA-stream
contract. The reference format rejects device-gated hooks because the current
record envelope cannot distinguish a gated-off empty tensor from a real empty
capture; adding an explicit skip marker belongs to the future native protocol.

Admission returns an explicit `AdmissionResult`. It never silently expands the
queue. Durable mode intentionally separates local commit from remote upload so
a remote outage cannot invalidate a completed local commit.

## Pack format

`dmi-pack-v1` is one immutable object:

```text
+------------------------+
| fixed header           | magic, format version, pack ID
+------------------------+
| independently encoded  | tensor record 0
+------------------------+
| independently encoded  | tensor record 1
+------------------------+
| ...                    |
+------------------------+
| footer manifest        | IDs, provenance, shape, offsets, codecs, checksums
+------------------------+
| fixed trailer          | footer offset/length, pack checksum
+------------------------+
```

The fixed trailer allows an indexer to locate the footer with one small suffix
range read. A second range read retrieves the footer. Tensor bytes are not read
during catalog indexing.

The footer is authoritative for reconstruction. It includes stable capture
identity, format identity, object-relative ranges, and sufficient metadata to
recreate catalog rows. Readers reject unknown major versions and allow only
documented additive minor changes.

Object keys are immutable and deterministic for one persistence intent:

```text
v1/tenant=<tenant-id>/date=<yyyy-mm-dd>/session=<session-id>/rank=<rank>/<pack-id>.dmi-pack
```

Key components are percent-encoded. Values that would exceed portable
filesystem component limits use a stable SHA-256 component; the full value
remains in the pack footer and catalog.

The same retry reuses the key and checksum. A different pack never reuses that
key.

## Commit and discovery semantics

Successful completion of the pack upload is the commit. There is no separate
manifest object or catalog acknowledgement.

Discovery combines:

1. Object-created notifications for low latency.
2. Periodic prefix scans for correctness.
3. Stable pack IDs for idempotent replay.

Notifications may be delayed, duplicated, or reordered. The indexer therefore
treats them as hints. Reconciliation scans recent time partitions and scheduled
older partitions, compares pack identities with indexed state, and replays any
missing work.

### A pack's footer is not evidence of whose it is

The footer names the tenant; the key prefix names where a writer with
credentials for that tenant is allowed to put objects. Until descriptors are
built, nothing compares them -- and integrity proves nothing here, because a
forged pack is perfectly well formed. Anyone able to PUT into the bucket could
therefore write a pack whose footer carried another tenant's `tenant_id` and
`capture_id`, have it indexed under the victim's tenant, and -- since the
reader resolves a capture with `argMax` over `(index_version, store_id,
pack_id)` -- become the pack that capture resolves to at every fresh watermark.

`_descriptors` now refuses a pack whose records name a tenant other than the
one its key belongs to, comparing against the same `key_component` encoding
that wrote the key (including the digest form an over-long identifier takes,
which is one-way and so has to be re-encoded rather than decoded). A key with
no `tenant=` segment is left alone: a filesystem store, a test fixture and a
hand-placed object are all legitimately laid out some other way, and a check
that guessed would refuse them all. What is enforced is that a key which DOES
name a tenant names the pack's own.

This is a bound on damage, not a substitute for object-store authorization.
Per-tenant buckets or prefix-scoped credentials remain the primary control; the
check is what makes a breach of that control visible instead of silent.

## Catalog indexer

The indexer is isolated from capture hosts and scales independently. For each
discovered pack it:

1. Reads the fixed trailer.
2. Reads and validates the footer.
3. Converts descriptors into catalog rows.
4. Batches rows across many packs.
5. Inserts into ClickHouse using stable IDs and versions.
6. Reports discovery lag, validation failures, and batch health.

At-least-once discovery can produce physical duplicates. Private raw tables may
use `ReplacingMergeTree`, but public views must provide deterministic logical
deduplication. Correctness does not depend on asynchronous merges having run.

`{prefix}_capture` means **published descriptor rows, engine-deduplicated**.
`FINAL` supplies the deduplication; it applies no membership, so the view is
additionally bounded to the packs the latest published snapshot contains:

```sql
WHERE (store_id, pack_id) IN (
  SELECT store_id, pack_id FROM {prefix}_snapshot_manifest
  WHERE (index_version, publish_id) IN (
    SELECT index_version, publish_id FROM {prefix}_index_watermark))
```

The bound pairs `(index_version, publish_id)` rather than matching the version
alone, exactly as the reader's membership clause does. Every row in the
watermark table is at or below the published head by definition, so the pair
test subsumes an `index_version <= max(...)` bound and additionally requires the
manifest row and the watermark row to come from the **same publish**. See
*Publish identity* below.

Without that bound the view showed descriptor rows from batches that were
written and never published, and rows orphaned by a crashed indexing pass --
data the reader correctly reports as nonexistent. Filtering under `FINAL` is
sound here only because `(store_id, pack_id)` belongs to the table's sort key:
rows a merge may collapse into one all share those columns, so the predicate
keeps or drops a whole group and can never delete the representative `FINAL`
would have kept. A predicate on `index_version` has no such guarantee, which is
why `FINAL ... WHERE index_version <= W` is not a snapshot (see *Phase 5*).

The view emits **one row per `(capture, store, pack)`**, not one per capture. A
capture described by two packs -- a pack mirrored to a second store, or a
producer retrying a `capture_id` after the first pack was sealed -- is two
published rows and appears twice. Choosing between them is supersession, which
belongs to the reader (one `argMax` grouped on capture identity, ordered on
`(index_version, store_id, pack_id)` -- see *Phase 5*); a second copy of those
semantics in the view's SQL could drift away from the reader's without either
side failing.

`{prefix}_pack_inventory` carries no such bound and needs none: `index()` writes
the inventory only after a successful publish, so every pack in it is already
published.

### The publisher lease

**Only the holder of the publisher lease can make a snapshot visible, and the
check rides inside the statement that makes it visible.**

`{prefix}_publisher_lease` holds `(term, lease_id, holder, acquired_at_ns,
expires_at_ns)`, append-only. A publisher calls `acquire_publisher_lease(holder)`
before it indexes anything; `publish_snapshot` refuses without one. Both of the
statements a publish writes -- the manifest rows and the watermark row -- carry
the same predicate:

```sql
  AND (SELECT count() FROM (SELECT lease_id, expires_at_ns
                            FROM {prefix}_publisher_lease
                            ORDER BY term DESC, lease_id DESC LIMIT 1)
       WHERE lease_id = :my_lease
         AND expires_at_ns > toUnixTimestamp64Nano(now64(9))) = 1
```

so a publisher whose lease has been taken over makes **no snapshot visible**.
It does not make one visible and then discover it lost, which is the failure
mode every post-write check in this design's history has had.

That is per STATEMENT, not per publish, and the two are not the same claim.
This paragraph used to say such a publisher "writes nothing"; a takeover
landing in the gap between the two statements leaves the manifest rows of the
first behind. They are inert -- membership needs the manifest row and the
watermark row of the SAME publish -- so the safety claim is unaffected, but the
rows are durable. See docs/catalog-descriptor-key.md, *What this does not
close*.

**One subquery reading one row, not two subqueries returning a column each.**
That is a correctness point before it is a cost one, and the two-read form was
implemented before it was understood: two scalar subqueries are two reads of
the lease table, and a takeover landing between them can be answered with the
OLD holder's `lease_id` and the NEW holder's `expires_at_ns` -- so the fence
passes for a publisher that has already been replaced. The ordered `LIMIT 1`
resolves the head once and both conditions are asked of that one row. It is
also cheaper: measured on 25.12, the conditional publish costs 3.06 ms
unfenced, 3.85 ms with a fence, and 4.84 ms with the two-subquery form.

`count() ... = 1` rather than a tuple comparison, because an empty lease table
has to make this false and a tuple cannot: a scalar subquery selecting a tuple
from no rows raises `Code: 125 ... cannot be Nullable` on 25.12 rather than
answering NULL.

The lease is claimed by the same sole-claimant append-and-read-back protocol as
a catalog version, but with an extra late-claim rule. A claimant can finish a
singleton read-back before a competitor inserts at the same term. Once a term
is contested, no publisher may claim above it until the maximum expiry at that
term; this preserves any lease returned before the second row arrived.

**The credit there belongs to the read-back, not to the fence.** This section
used to argue that because the fence names one row, a contested term satisfies
neither of its conditions. It satisfies both, for the claimant whose `lease_id`
sorts higher under ClickHouse's UUID collation -- which compares the low 64 bits
first and is therefore not the text order either. What makes the term safe is
that a claimant which sees the contest receives no `PublisherLease`, while a
claimant that returned earlier remains protected by the contested term's expiry
quarantine.

| Concept | Where it lives |
|---|---|
| the monotonic slot a takeover has to beat | `term` |
| the fencing token the publish statement checks | `lease_id` |
| when the lease was taken and when it lapses | `acquired_at_ns`, `expires_at_ns`, both stamped by the **server** |

Clocks: `term` orders the table, so a wall-clock tie can never make the head
ambiguous. Expiry is decided by the server's clock on both sides -- the row is
stamped server-side and the fence compares against `now64(9)` -- so no
publisher's own clock, and no skew between two of them, participates.

Lifecycle:

- **Acquire.** A live lease held by somebody else raises
  `PublisherLeaseHeldError`, naming the holder and the remaining time. An
  expired singleton lease is taken over. A contested head is unavailable until
  every claim at that term has expired; only then is a higher term claimed.
  A writer that already holds a lease is re-acquiring its OWN and refreshes it,
  keeping the same `lease_id`, because "acquire before publishing" is the
  documented path and anything restarting above the writer calls it again.
- **Renew.** Every publish renews first, keeping the same `lease_id` at a fresh
  term. That costs **three** round trips -- head read, claim INSERT, read-back,
  5.69 ms median on 25.12 -- and buys the safety margin: at the moment the
  fence runs, the lease has essentially a full `lease_ttl_ns` left.
- **Fail.** A publish fenced out raises `PublisherLeaseError`, deliberately
  *not* `SnapshotPublishRaceError` -- a lost version is repaired by allocating a
  higher one, which `CatalogIndexer` does automatically, while a lost lease
  would fail the same fence at every version. The recovery is to acquire again
  and re-index; no snapshot became visible, though a takeover between the two
  publish statements can leave inert manifest rows behind.
- **Release.** `release_publisher_lease()` writes an already-expired tombstone
  so an orderly restart does not cost the next publisher a whole TTL. A crash
  does; that is what expiry is for. It is **fenced on the head being this
  writer's own lease, inside the statement that writes**: the tombstone lands
  at `head.term + 1` and so becomes the head whatever was there before, which
  unfenced let a writer whose lease had long since lapsed revoke the current
  holder's live one just by shutting down cleanly -- and fenced by a client
  read followed by a separate INSERT, the same revocation fit in the window
  between the two, contesting a successor's freshly granted term. The head is
  resolved and the tombstone written in one server-side statement, exactly as
  the publish fence is.

**One publisher holds the lease, but two client objects can believe they do.**
The sole-claimant read-back proves a claimant is alone at ITS term, not that its
term is the head: a claimant whose head read preceded a rival's row claims below
that rival, reads back alone, and is handed a live lease while the rival sits
above it. Verified on 25.12 with two writers holding terms 1 and 9. Only the
head publishes -- the fence says so, and the loser is refused at its next
renewal with a message naming the actual holder -- so this is a statement about
the client objects rather than about safety.

`publish_timeout_ns` caps each publish statement's `max_execution_time` and must
be below `lease_ttl_ns`. The writer renews before every fenced statement, and
each fence requires more than that timeout of lease life remaining. A later
manifest chunk or watermark therefore cannot start near the end of an earlier
renewal. Each manifest chunk is read back before proceeding, so a conditional
INSERT that wrote zero rows cannot be followed by the watermark.
`max_execution_time` is in
**seconds**, so the writer requires `publish_timeout_ns` to be a whole number
of seconds and sends the integer -- a fractional value can truncate to 0 on
older serialization paths, which the server reads as *no* limit. The cap is per
statement, not the pair, so it does not bound the client round trip between
them.

Contested lease terms wait for expiry rather than retrying at randomized higher
terms; the legacy `lease_attempts` retry knob was removed.

The earlier one-renewal implementation measured 15.2 ms for a 16-pack publish
and 173 ms for the full 4096-row indexing pass on 25.12. The current protocol
adds one renewal and one manifest verification per chunk, so those figures are
no longer representative and need remeasurement. The read path is unchanged.

What this does **not** close is written up in
docs/catalog-descriptor-key.md, "The fenced publisher lease", with the
measurements: the takeover instant, the liveness cost of a contested term, a
crash between a claim and its read-back, and the write half of replication.

### Publish identity

Every call to `publish_snapshot` mints one `publish_id` and writes it on both
rows it produces: the manifest rows for the packs it is admitting, and the
watermark row that admits them. It then reads that column back and compares it
to its own value.

That check answers *"is version V mine?"*. The one it replaced -- `SELECT
count() FROM {prefix}_index_watermark WHERE index_version = V` -- answered
*"does a row for V exist?"*, and a row written by anything else read as success:
an operator's `INSERT`, a second build sharing the prefix, a publisher whose
conditional statement overlapped this one. The caller was then told it had
published a snapshot it did not publish, and `CatalogIndexer` went on to record
those packs in the replay inventory, so no later pass would index them again.
The sole-claimant allocator makes a foreign row at V unlikely, not impossible,
and reading the identity back costs what counting cost: the same one-row scan
of the same key range.

The identity is per **attempt**, not per allocated version. A second attempt at
one version is a different write, and reusing the allocator's `claim_id` would
let it read the first attempt's row back as its own and report success for a
statement that inserted nothing.

Carrying the same identity on the manifest rows is what makes membership a
claim about a publish rather than about a number. On `index_version` alone the
*contents* of snapshot V would be whatever anyone wrote at V, while the winner
of V unwittingly published them; owning a version and owning its membership are
separate claims and the catalog needs both.

#### Reads that decide something

The claim read-back in `allocate_version`, the published-head reads, the
conditional watermark `INSERT` and the publish verification all carry
`select_sequential_consistency`. Each is a read-back of the reader's own write,
and the sole-claimant protocols are sound only while a later write always
observes an earlier one. A single ClickHouse node gives that; a
`ReplicatedMergeTree` replica serves reads from whatever log entries it has
fetched, so a read-back can miss a row another claimant has already committed
and both claimants can see themselves alone.

It is set on the reads rather than validated at construction because there is
nothing to validate at construction: the tables need not exist yet, an operator
can convert them to `Replicated` afterwards, and a warning nobody reads is not
enforcement. Setting it is unconditional and costs nothing on a non-replicated
table, where the server accepts and ignores it.

The write-side half is opt-in through `ClickHouseCatalogConfig.insert_quorum`.
When configured, descriptor inserts and protocol writes are quorum-durable;
pack inventory stays asynchronous because a stale replay check causes redundant
work rather than a skipped capture.

### Retention

Three tables grow with every pass and nothing collected them: a publisher
publishing once a second appends ~86k `{prefix}_publisher_lease` rows a day,
every allocation leaves a `{prefix}_capture_version_claims` row, and every
publish that loses its version race leaves a full set of manifest rows behind.
The only removal path was `drop_schema()`, which destroys the catalog.

`ClickHouseCatalogWriter.collect_garbage()` deletes exactly the rows that can
never be resolved again, and returns how many it removed per table. It is
explicit and belongs in a maintenance job, never on the write path: a mutation
is a background rewrite of the parts it touches, and an indexer that ran one
inline would pay for it in the middle of a publish.

| Table | Deleted | Why it can never be resolved again |
|---|---|---|
| `{prefix}_snapshot_manifest` | rows below the published head whose `(index_version, publish_id)` has no watermark row | below the head, that publish's watermark INSERT can no longer land -- the barrier requires strictly above -- so the pair membership needs will never exist. An in-flight publish always sits ABOVE the head, which is what keeps this from deleting membership out from under one |
| `{prefix}_publisher_lease` | rows below the head `term` | the fence resolves exactly one row, the highest `(term, lease_id)`, and terms only increase. The head is kept even when expired: it is what a takeover has to sort above |
| `{prefix}_capture_version_claims` | rows at or below the published head | the allocator picks above `max(claims.version)` AND above `last_published_version()`, so the watermark keeps the floor once these are gone. Claims ABOVE the head stay -- one may be a version a pass has allocated and not yet published |

`{prefix}_index_watermark` and the descriptor and inventory tables are never
collected here. The watermark log IS the floor the other two bounds are
measured against, and the descriptors are the catalog.

### Reading under replication lag

A bounded read resolves against a pinned watermark, and the statement that
reads the watermark has always carried `select_sequential_consistency`. The
statement that reads DESCRIPTORS now carries it too, because the first does not
imply the second: a `ReplicatedMergeTree` keeps a replication log **per table**,
so a replica caught up on `{prefix}_index_watermark` can still be behind on
`{prefix}_snapshot_manifest` and `{prefix}_capture_raw`. Left plain, a read
pinned to a genuinely published watermark returns a SHORT page -- and `search`
issues its cursor at that watermark, so the next page resumes past the rows
that were missing, and a walk skips captures while reporting success. Lag can
only omit, never invent, so nothing unpublished or cross-tenant is exposed
either way.

On a single node the server accepts and ignores the setting (verified on
25.12), so this is free unless the tables are replicated; there it makes every
page wait for the replica. `ClickHouseReaderConfig.consistent_snapshot_reads`
turns it off for an operator who would rather have the latency and accept the
gap.

The WRITE half is available and still off by default:
`ClickHouseCatalogConfig.insert_quorum` sets `insert_quorum` (and clears
`insert_quorum_parallel`, which `select_sequential_consistency` does not work
with) on version claims, lease claims and releases, membership, the watermark,
and descriptor inserts. Without it the sole-claimant protocols and consistent
descriptor reads are sound on a single node and on a quorum-writing cluster,
and not in between. Pack inventory remains asynchronous: it is only the replay
guard, so lag there causes redundant indexing rather than an incomplete page.

**It is a for-the-life-of-the-catalog setting, not a toggle.** Measured on
25.12 against two replicas of each protocol table: once a replicated table has
taken a quorum insert, a later NON-quorum insert into it is invisible to a
`select_sequential_consistency` read -- the plain read returns two rows and the
sequential read returns one. Every read-back in this module is a sequential
read of the writer's own insert, so turning `insert_quorum` off on a catalog
that had it on leaves the sole-claimant protocols unable to confirm themselves:
`allocate_version` exhausts its attempts and raises
`CatalogVersionAllocationError`. That is the safe direction -- loud, not blind
-- and it is still an outage. Turning it ON mid-life is safe; turning it off
requires a rebuild.

The same measurement is why `insert_quorum_timeout` is bounded to
`publish_timeout_ns` rather than left at ClickHouse's 600-second default: the
fenced publish statements are capped by `max_execution_time`, but a version or
lease claim is not, so an unreachable replica would otherwise park a claim for
ten minutes -- twenty times the publish cap and twenty times the lease TTL --
while the writer believed it was mid-publish. With the bound, an unsatisfiable
quorum is refused immediately (Code 285, `TOO_FEW_LIVE_REPLICAS`, measured at
0.0s with one replica detached).

`tests/tools/verify_replicated_quorum.py` drives all of this against two
replicas and is the only way to exercise it: no CI here runs a replicated
cluster.

### Catalog schema versions and rebuild

`{prefix}_schema_version` holds one row naming the schema the catalog was
created with. The current version is **4**. `ensure_schema` writes that row
last, once every other object exists, so a stamped catalog is a complete one.

**Version 4 is the first version that stamps itself**, so a catalog with no such
table is one of versions 1, 2 or 3 and the stamp cannot say which. It is not
"version 1 by definition"; reading it that way is how a refusal comes to state
differences the catalog in front of it does not have.

| Version | What it changed |
|---|---|
| 1 | the original schema; membership in `{prefix}_pack_commit_log`, descriptors sorted on capture identity alone |
| 2 | `(store_id, pack_id)` appended to the descriptor sort key; membership moved to `{prefix}_snapshot_manifest` |
| 3 | `publish_id` added to the watermark and the manifest |
| 4 | `{prefix}_publisher_lease` and `{prefix}_schema_version` added |

`ensure_schema` checks compatibility before issuing any DDL and refuses
anything it did not create, raising `CatalogSchemaVersionError`:

| State found | Outcome |
|---|---|
| no catalog objects at all, with object visibility confirmed | fresh install: create everything, stamp the current version |
| version table stamped at the current version, all objects present | proceed; the DDL is idempotent |
| version table present, no row | an install of this build that died before stamping: rerun the DDL, then stamp -- *after* the last three rows below, which an empty stamp does not skip |
| only superseded objects (`{prefix}_pack_commit_log`) | refuse -- naming that object and saying to drop it; there is nothing to rebuild |
| an object present under the wrong kind (a table where a view belongs, or the reverse) | refuse -- naming the object and its engine |
| catalog objects present, no version table | refuse -- unstamped, with the differences read off `system.tables` / `system.columns` |
| stamped at any other version | refuse -- a newer writer owns this catalog, or an older one this build cannot upgrade |
| a superseded object (`{prefix}_pack_commit_log`) standing BESIDE this build's own | refuse -- an earlier build has been writing this prefix |
| a TABLE missing, and some data table holds rows | refuse -- partially dropped |
| a TABLE missing, and every data table is empty | complete it: rerun the DDL |
| a VIEW missing | recreate it; a view is a projection of the tables and holds no rows |
| any pack inventory identity lacks effective published manifest membership | refuse -- membership is incomplete |

Before any verdict above -- and so before any catalog DDL -- `ensure_schema`
checks object-scoped `SHOW TABLES` access for every current and superseded
object. After DDL it re-reads `system.tables` and requires the complete current
object set, with no superseded object, before writing the stamp. An empty
result caused by hidden objects therefore cannot be mistaken for a fresh
install, and a partially visible catalog cannot be stamped as complete.

**"Before any verdict" is the load-bearing half.** `system.tables` is
grant-filtered per role, so an object this role holds no privilege on is
absent from the state every row of the table above is read off -- there,
indistinguishable from an object that was dropped. Checked only before the
DDL, this check was unreachable for exactly the catalog it exists for:
measured on 25.12, a healthy stamped catalog whose `{prefix}_snapshot_manifest`
was merely ungranted was refused as "missing `{prefix}_snapshot_manifest`" with
the rebuild instruction, telling the operator to drop all ten objects of a
catalog whose only fault was a missing `GRANT`. The grant probes name objects
rather than resolving them, so they need neither the database nor the objects
to exist and can run first on a fresh install too.

**An empty stamp narrows the version; it does not excuse anything else.** The
last three rows apply whether or not `{prefix}_schema_version` holds a row.
Returning as soon as it was empty skipped both refusals, and clearing the stamp
is the obvious workaround for an operator who has just been refused -- which
made it the likeliest route into the state this check exists to prevent.
Measured on 25.12: with the stamp truncated, a version 3 catalog was accepted
and re-stamped as 4 with no `{prefix}_publisher_lease` table, performing the
in-place upgrade this design says is never performed; on a version 2 catalog
the DDL then died mid-way with `Code: 47` over `publish_id`, leaving the
catalog half written; and an inventory beside an empty manifest was accepted.

**A missing table is dangerous because of what SURVIVED it.** Recreating one
empty is the state that hides every capture only while its neighbours kept their
rows: a surviving pack inventory makes the next pass skip every pack it lists,
so nothing refills what was dropped. With every data table absent or empty there
is nothing to hide and nothing to lose, and re-running the DDL is the
completion -- which is the whole reason `{prefix}_schema_version` is created
first, "so an install interrupted partway leaves a catalog that says *this
build, unfinished* rather than one refused forever". Refusing regardless made
that false: an install that died between two `CREATE`s was refused on every
later start, reciting a surviving pack inventory that was itself one of the
tables that had never been created, and the only recovery it offered was a
manual drop of every object.

**An earlier build sharing the prefix is refused, not tolerated.** A superseded
object standing beside this build's own is not an unfinished cleanup -- that is
the "only superseded objects" row above -- it is two builds writing one catalog,
and the older one is the dangerous half. Its `ensure_schema` is all
`CREATE ... IF NOT EXISTS`, so it no-ops over these tables and recreates its
own; its publish writes the pack inventory and an unconditional watermark row
carrying no publish identity, and never a manifest row. Every pack it touches is
therefore recorded as indexed and admitted by no snapshot: invisible to every
reader, and skipped by every later pass including a rebuild. Nothing else here
notices -- the stamp reads this version, no table is missing, and the
inventory-without-membership check reports those packs only once they are
already durable and invisible.

**An unstamped catalog is diagnosed, not assumed.** The refusal reads the
descriptor table's `sorting_key`, which of the two membership tables exist,
whether the watermark and the manifest carry `publish_id`, and which of this
build's objects are absent -- and reports the differences it actually found,
including saying so where a difference is *not* there. An operator who checks a
named fact, finds it false and concludes the guard is broken works around it,
which lands them on the one path the guard exists to prevent; an inaccurate
diagnosis is worse than a vague one. It is still refused whichever version it
turns out to be: those probes compare object names, one sort key and one column,
not column types, view definitions, codecs or skip indices, so "the differences
listed are all of them" is not something this build can know.

No version is reachable from the one below it by any statement `ensure_schema`
could issue. Between them the versions change the descriptor sort key, the
membership table, the publish identity columns and the set of tables, and
`CREATE TABLE IF NOT EXISTS` alters neither an existing table's columns nor its
sort key. Version 4 adds only new tables and could in principle be created in
place, but the compatibility check runs before any DDL and cannot tell a table
that was never written from one that was dropped -- and one of those is the
state that hides every capture. The version 1 to version 2 step is the worked
example, and both ways it fails are silent. `CREATE TABLE IF NOT EXISTS` is
a no-op against a live table, so the descriptor table keeps the version 1 sort
key and stays open to the merge deletion the new key exists to prevent.
Membership is worse: it moved from `{prefix}_pack_commit_log` to
`{prefix}_snapshot_manifest`, and on an upgraded catalog the manifest is empty
while `{prefix}_pack_inventory_raw` is full, so the next indexing pass skips
every pre-existing pack as already committed, no pack ever reaches the
manifest, and every capture indexed before the upgrade becomes invisible to
every reader. Measured on a version 1 catalog holding four captures in two
packs: `ensure_schema()` returned cleanly, the sort key was unchanged, the
following rebuild reported `skipped=2 indexed=0`, and the reader returned 0 of
4 captures.

A **missing view** is the one recoverable state, and deliberately so.
`{prefix}_capture` and `{prefix}_pack_inventory` hold no rows: `ensure_schema`
recreates them outright (`CREATE OR REPLACE VIEW` / `CREATE VIEW IF NOT
EXISTS`), and a recreated projection cannot disagree with the tables that
survived. Demanding the full rebuild -- which re-reads every pack footer in the
store and leaves readers on an empty and then partial catalog while it runs --
because somebody dropped a view is a cost with no risk behind it. A missing
TABLE is the opposite and stays refused.

#### Rebuilding the catalog

Every `{prefix}_` object is derived from immutable packs, so the supported
recovery -- from a version mismatch, from a partial drop, from any catalog
state that is not trusted -- is to delete all of it and reconcile the object
store again:

1. Stop every indexer writing that prefix.
2. Drop all of its objects, views before the tables they read:
   `{prefix}_capture`, `{prefix}_pack_inventory`, `{prefix}_capture_raw`,
   `{prefix}_pack_inventory_raw`, `{prefix}_capture_version_claims`,
   `{prefix}_publisher_lease`, `{prefix}_index_watermark`,
   `{prefix}_snapshot_manifest`, `{prefix}_schema_version`, and version 1's
   `{prefix}_pack_commit_log`. This list and the one
   `CatalogSchemaVersionError` prints are both checked against the writer's own
   object table by `test_the_documented_rebuild_drops_exactly_what_the_writer_owns`,
   because a table missed here is a table that survives the rebuild -- and one
   superseded table left standing wedges every start after it. `drop_schema()`
   is the supported way to do this and issues `DROP TABLE IF EXISTS` for every
   object regardless of the kind this build creates it as: the catalog that
   most needs tearing down is the one whose kinds are WRONG, and `DROP VIEW`
   against a table is refused even with IF EXISTS (Code 80 on 25.12).
3. Run `ensure_schema()`. It finds nothing, creates the current schema, and
   stamps it.
4. Run `acquire_publisher_lease(holder)` on the writer the rebuild will use.
   Only the lease holder can make a snapshot visible, and `rebuild()`
   publishes every page it indexes, so a rebuild without the lease fails its
   first page with `PublisherLeaseError`.
5. Run `CatalogReconciler.rebuild(prefix=...)` once per pack store. It lists
   the objects, range-reads each pack footer, and repopulates the descriptors,
   the snapshot manifest, the pack inventory and the watermark from them.

**Dropping `{prefix}_pack_inventory_raw` is mandatory, not optional.**
`committed_pack_ids` reads that inventory to decide which packs are already
indexed. An inventory left in place reports every pre-existing pack as done, so
step 5 skips all of them, writes no descriptors, publishes nothing at all,
and returns an `IndexResult` with no failures -- a rebuild that reports success
and produces an empty catalog, while the captures stay durable in object
storage and unreachable through every reader. That state is why the last row of
the table above exists: any inventory identity without effective manifest
membership is refused at `ensure_schema`, which is the earliest point at which
anything can still see the mistake.

The check lives there rather than in `CatalogReconciler` deliberately.
`rebuild()` is also the periodic full sweep of a healthy catalog -- it is
expected to skip everything it has already indexed -- so it cannot refuse a
populated catalog, and it reaches the writer only through `CatalogWriter`,
which exposes no way to ask whether membership is intact. `ensure_schema` runs
before any indexer starts and sees the tables directly.

ClickHouse stores:

- pack inventory and integrity state;
- capture descriptors and payload ranges;
- stable core summaries;
- extensible scalar metrics;
- references to large summary artifacts;
- indexing and enrichment health.

ClickHouse does not store raw tensor payloads or the only copy of essential
capture metadata.

## Summary model

Summary computation is asynchronous and versioned by:

```text
(capture_id, summarizer_name, summarizer_version, config_hash)
```

Stable, commonly filtered values such as minimum, maximum, mean, standard
deviation, norms, sparsity, NaN count, and infinity count use typed ClickHouse
columns. Extensible scalar metrics use a bounded long-form table. Large arrays,
histograms, embeddings, sketches, and sampled tensors remain immutable objects
with catalog locators.

New summarizers do not change the capture host or pack format.

## Reader and agent workflow

The public API is metadata-first and bounded:

```text
search(filters, page)
query_metrics(filters, metric_names, page)
estimate_hydration(capture_ids)
hydrate(capture_ids, byte_limit, request_limit)
get_artifacts(capture_ids)
export(capture_ids, format)
```

A typical flow is:

1. Query ClickHouse with bounded filters and pagination.
2. Inspect core or custom summaries.
3. Select capture IDs.
4. Estimate total read bytes and request count.
5. Coalesce adjacent ranges within each pack.
6. Fetch with byte, request, and concurrency limits.
7. Verify checksums and decode selected records.

Hydration also binds every catalog descriptor to the pack footer before any
payload range is fetched: the footer index is loaded with the usual two small
range reads (trailer, then footer), cached per pack in an LRU bounded by both
pack count and serialized footer bytes, and charged to the same request and
byte limits as payload reads. The zero-I/O estimate uses a safe cold-cache
upper bound because the exact footer length is available only after reading
the trailer. A descriptor whose metadata or record locator contradicts the
footer fails hydration as a format error. This closes the catalog-trust gap
where a re-described row (same bytes and CRC, different dtype or shape) would
decode garbage. The catalog remains the query index and the footer the
authority; estimation stays metadata-only and reads nothing.

List and summary operations never hydrate tensor bytes implicitly. Storage
credentials and provider endpoints are resolved from deployment configuration,
not returned in catalog rows.

## Extension contracts

```python
class PackWriter:                      # pack.PackWriter, a concrete class
    def append(self, record: CaptureRecord) -> None: ...
    def seal(self) -> SealedPack: ...


class PackSource(Protocol):            # model.PackSource
    pack_id: str
    checksum: str
    @property
    def object_bytes(self) -> int: ...
    def open(self) -> BinaryIO: ...


class PackStore(Protocol):             # model.PackStore
    store_id: str
    def put(self, pack: PackSource, object_key: str) -> PackRef: ...
    def stat(self, ref: PackRef) -> ObjectInfo: ...
    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes: ...


class PackInventory(PackStore, Protocol):          # catalog.PackInventory
    def inspect(self, object_key: str) -> PackRef: ...
    def list_objects(
        self, *, prefix: str = "", cursor: str | None = None, limit: int = 1000
    ) -> ObjectPage: ...


class PackSink(Protocol):              # pipeline.PackSink
    def persist(self, ready: ReadyPack) -> PackRef | PackSource: ...


class CatalogWriter(Protocol):         # catalog.CatalogWriter
    def committed_pack_ids(self, identities) -> set[PackIdentity]: ...
    def write_descriptors(self, descriptors, *, index_version: int) -> None: ...
    def commit_packs(self, refs, *, index_version: int) -> None: ...
    def publish_snapshot(
        self, *, index_version: int, refs, published_at_ns: int,
        indexed_rows: int, indexed_packs: int,
    ) -> None: ...
    def last_published_version(self) -> int: ...
    def allocate_version(self) -> int: ...


class CaptureReader:                   # reader.CaptureReader, a concrete class
    def search(self, query: CaptureQuery) -> CapturePage: ...
    def select(self, query: CaptureQuery) -> CaptureSelection: ...
    def estimate(self, selection: CaptureSelection) -> HydrationEstimate: ...
    def hydrate(
        self, selection: CaptureSelection, *, byte_limit: int,
        request_limit: int = 1024,
    ) -> tuple[HydratedCapture, ...]: ...
    def summarize(
        self, selection: CaptureSelection, *, byte_limit: int, ...
    ) -> tuple[CaptureSummary, ...]: ...


# Summaries are extension VALUES registered with ExtensionRegistry rather than
# a summarizer protocol: a named, versioned callable over a decoded tensor.
@dataclass(frozen=True, slots=True)
class ScalarMetric:                    # extensions.ScalarMetric
    name: str
    version: int
    compute: Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class ArtifactProducer:                # extensions.ArtifactProducer
    kind: str
    version: int
    produce: Callable[[np.ndarray], tuple[bytes, str]]


class ArtifactSink(Protocol):          # extensions.ArtifactSink
    def put(
        self, *, capture_id: str, kind: str, version: int, data: bytes,
        content_type: str,
    ) -> ArtifactRef: ...
```

*(This block used to name about a dozen types that were never written --
`CommitFeed`, `CaptureSummarizer`, `ScanCursor`, `Page`, `EventCursor`,
`TensorView`, `SummaryBatch`, `CaptureId`, `HydrationRequest`, `TensorRecord`,
`ClickHouseCatalogIndexer`, `CoreTensorStatsSummarizer`. What shipped instead
is above: discovery is `CatalogReconciler` over a `PackInventory` rather than a
`CommitFeed`; paging is `CapturePage` / `ObjectPage` and an opaque cursor
string rather than a generic `Page[T]`; hydration takes a `CaptureSelection`
rather than a request object; and summarization is the extension registry
rather than a summarizer protocol.)*

Initial implementations:

```text
PackStore / PackInventory
  - FilesystemPackStore
  - S3PackStore

CatalogWriter
  - ClickHouseCatalogWriter          (driven by CatalogIndexer/CatalogReconciler)

CaptureCatalog (the reader's query side)
  - ClickHouseCaptureCatalog

Summaries
  - summarize_tensor -> CoreTensorSummary
  - ExtensionRegistry, for ScalarMetric and ArtifactProducer
```

External configuration and provider responses are validated at their
boundaries. Internal stages exchange typed records without repeated validation.

`S3PackStore` is the Garage implementation; Garage is selected through its
endpoint, region, and credentials rather than a provider-specific subclass.
This avoids duplicating the S3 contract while keeping provider compatibility in
the live test matrix.

## Failure semantics

| Failure | Visible result | Recovery |
|---|---|---|
| Upload fails | Pack is not committed | Retry the deterministic key with bounded backoff |
| Upload succeeds, event is lost | Pack exists but catalog is late | Prefix reconciliation discovers it |
| Event is duplicated | Same pack may be indexed again | Stable IDs and public deduplication preserve logical results |
| ClickHouse is unavailable | Capture continues; catalog lag grows | Indexer retries and replays canonical packs |
| Footer is corrupt | Pack is not indexed | Quarantine identity and emit an integrity failure |
| Reader gets a short range | Hydration fails closed | Retry, then report pack and range |
| Checksum mismatches | Tensor is not decoded | Quarantine pack and emit an integrity failure |
| Host queue or spool fills | Configured overload policy applies | Report pressure and drops; never grow without bound |
| Process exits with `.ready` packs | Packs remain locally recoverable | Restart uploader and verify ambiguous remote writes |

Deleting a pack requires a separate retention workflow. Garbage collection and
pack compaction never run in capture or indexing critical paths.

## Observability and acceptance gates

| Plane | Required measurements |
|---|---|
| Admission | enqueue latency, blocked time, drops, queue bytes |
| Packing | GiB/s, CPU, copies, compression ratio, flush reason |
| Upload | GiB/s, p50/p95 latency, active requests, retries, spool growth |
| Indexing | catalog lag, rows/s, rows/insert, bytes/insert, duplicate rate |
| Hydration | range latency, useful GiB/s, read amplification, checksum failures |
| Process | CPU, peak RSS, network, local-spool I/O |
| End-to-end | capture-to-committed and capture-to-queryable p50/p95/p99 |

Correctness gates every performance result. A faster path that loses required
records, skips validation, or changes reader semantics is a regression.

## Package layout

```text
src/dmi/storage/capture/
  model.py                 # contracts and bounded request types
  pack.py                  # dmi-pack-v1 writer, validator, footer index
  filesystem.py            # immutable local reference store
  reader.py                # selection, estimation, coalesced hydration
  pipeline.py              # bounded admission, assembly, sinks, metrics
  spool.py                 # atomic local commit, recovery, retry upload
  s3.py                    # Garage/S3 streaming, listing, exact range reads
  catalog.py               # discovery, reconciliation, footer indexing
  clickhouse_catalog.py    # raw tables, logical views, batched inserts
  clickhouse_reader.py     # pinned snapshot reads, keyset pagination
  cursor.py                # opaque validated keyset cursors
  summary.py               # core tensor statistics, tensor decode
  extensions.py            # scalar metric / artifact extension registry
  record_adapter.py        # opt-in Ring² reference bridge

# Planned additions
native/csrc/storage/
  capture_record.h
  pack_builder.h
  persistence_pipeline.h
  spool.h
```

*(The five modules after `clickhouse_catalog.py` shipped without this list
being updated. The `summaries/` package this section used to plan was never
created; `summary.py` and `extensions.py` are what filled that role.)*

## Implementation plan

### Phase 1 — contract and format

- Specify `dmi-pack-v1`, stable identifiers, checksums, and compatibility.
- Implement writer/reader round trips and filesystem storage.
- Test truncation, corruption, unknown versions, and deterministic retries.

Exit gate: CPU-only format tests pass and pack throughput is reproducible.

Status: CPU reference exit gate passed. The local five-trial, 2,000-record
baseline used 64 KiB payloads and one roughly 126 MiB pack per trial. It measured
0.521 GiB/s median construction throughput and 1.009× space amplification. This
is a regression baseline for the Python reference, not evidence that the future
native pipeline meets the 1.2× production-capacity gate.

Run it from a source checkout with:

```bash
PYTHONPATH=src python -m benchmarks.bench_capture_pack \
  --records 2000 --payload-bytes 64KiB \
  --target-pack-bytes 128MiB --trials 5
```

### Phase 2 — bounded CPU pipeline

- Implement reusable slabs and size/linger sealing.
- Add direct upload and durable-spool policies behind an opt-in mode.
- Add saturation, restart, and ambiguous-upload tests.

Exit gate: memory and disk remain bounded under sustained overload.

Status: CPU reference exit gate passed. Queue tests sustain 100 submissions
against a three-record/24-byte queue and preserve the exact bound under
drop-newest overload. The measured blocking-admission workload used 2,000 ×
64 KiB records, a 256-record/16 MiB queue, and one roughly 126 MiB pack. Across
five local trials it reported:

| Mode | Median logical throughput | Drops | Peak queue |
|---|---:|---:|---:|
| Direct filesystem | 0.328 GiB/s | 0 | 256 records / 16 MiB |
| Durable spool | 0.345 GiB/s | 0 | 256 records / 16 MiB |

Those two figures are from the original Apple Silicon run and predate the
removal of the write path's redundant hashing and buffer copies. Re-measured
on the Linux reference host afterwards, with interleaved A/B sampling, the
same shape of workload sustains 0.291 GiB/s direct and 0.282 GiB/s spooled;
the pack writer alone reaches 0.472 GiB/s. The capacity gate later in this
document uses the post-change numbers.

Both modes produced 1.0095× space amplification. The difference between local
modes is within filesystem and scheduling variance; it is not treated as an
optimization result. Neither number includes Garage or network upload.

Run the same workload with:

```bash
PYTHONPATH=src python -m benchmarks.bench_capture_pipeline \
  --mode direct --records 2000 --payload-bytes 64KiB \
  --target-pack-bytes 128MiB --queue-records 256 \
  --queue-bytes 16MiB --trials 5
```

### Phase 3 — Garage integration

- Implement the S3-compatible store contract.
- Sweep pack size, multipart threshold, and upload concurrency.
- Test retries, restart recovery, listing, and range reads against a pinned
  Garage single-node release.

Exit gate: pack and upload capacity exceeds target input by at least 1.2 times.

Status: implementation and local compatibility gates passed. A reference
device-to-host rate for a single RTX 4090 host is derived below and the gate is
evaluated against it; the gate remains open for production hardware, where the
network and storage path differ from this loopback single-node setup. The pinned Garage
v2.3.0 live test covers multipart upload, retry idempotency, object metadata,
listing, and the two range reads used to load a pack footer.

The released Garage binary is now verified on Linux as well as from source.
`garage v2.3.0 [features: bundled-libs, consul-discovery, fjall, journald, k2v,
kubernetes-discovery, lmdb, metrics, sqlite, syslog, telemetry-otlp]`, the
published x86-64 Linux download, runs the whole live suite unmodified; the
harness accepts both that version string and the `cargo:2.3.0` a source build
reports.

#### Server-side checksums

`S3PackStore.put` sends `ChecksumAlgorithm=SHA256`, and Garage 2.3.0 both
supports and enforces it, measured against the ephemeral server:

- A `PutObject` whose `ChecksumSHA256` contradicts the body is rejected with
  `InvalidDigest` (HTTP 400); a matching one is stored and returned verbatim by
  `HeadObject` with `ChecksumMode=ENABLED`.
- An `UploadPart` whose `ChecksumSHA256` contradicts the part is rejected the
  same way, so every part of a multipart pack is checked on arrival rather than
  only at completion.
- For a multipart object the stored `ChecksumSHA256` is S3's composite
  checksum-of-checksums — `SHA256` of the concatenated per-part digests, which
  reproduces Garage's value exactly — and therefore is **not** the digest of the
  whole object. Garage returns that composite without the `-<parts>` suffix AWS
  appends, so it cannot even be recognised as composite by shape. It is
  consequently never compared against `PackRef.checksum`; DMI identity keeps
  using the client-side `dmi-sha256` object metadata, which the upload tee
  computes over the exact byte stream boto3 sends.

The server-side checksum is therefore defence in depth against corruption
between the bytes boto3 read and the bytes Garage stored, not a replacement for
the DMI digest.

#### Live coverage

`tests/test_garage_live.py` (markers `garage` + `manual`) covers the isolated
store contract and the production write path end to end:

- `test_garage_multipart_retry_listing_and_two_range_footer_read` — multipart
  upload, idempotent retry, object metadata, listing, footer range reads.
- `test_garage_pipeline_spool_upload_commits_every_pack` —
  `HostCapturePipeline` -> `DurablePackSpool` -> `ParallelSpoolUploader` ->
  Garage. Every staged pack becomes exactly one object, the stored bytes are
  byte-identical to the sealed pack, `HeadObject` agrees with the `PackRef`, the
  spool is empty afterwards, and the uploader reports zero retries and zero
  failures.
- `test_garage_restart_recovers_exactly_the_un_uploaded_packs` — a batch is
  interrupted part way, then a fresh spool and uploader over the same directory
  recover exactly the packs that were never uploaded and complete them. No
  duplicated objects, no lost packs.
- `test_garage_failed_upload_keeps_the_local_copy` — a scripted permanent
  upload failure leaves the pack staged and its object absent, and the next
  pass uploads it successfully. Local durability survives a failed commit.

`tests/test_capture_garage_e2e.py` (markers `garage` + `clickhouse` + `manual`)
closes the loop through the catalog. It uploads real packs to Garage, discovers
them by bucket listing only — `S3PackStore` acting as the `PackInventory` a
`CatalogReconciler` reads, so every `PackRef` is rebuilt from Garage object
metadata — indexes them into ClickHouse under a unique table prefix, pins a
watermark, searches and resolves by ID at that pin, hydrates payloads by range
read out of Garage, and asserts both the raw bytes and the decoded tensors match
what was captured. It also asserts that re-reconciling the same bucket indexes
nothing new, and that a foreign object in the prefix is reported as one failure
without holding back the healthy packs. Both the ClickHouse tables and the
bucket objects it created are removed on teardown.

An Apple Silicon local sweep used four packs per trial, a 16 MiB multipart
threshold and chunk, two multipart requests per pack, and three trials per
point. All objects and byte caps verified with zero retries:

| Pack payload | 1 outer worker | 2 outer workers | 4 outer workers |
|---:|---:|---:|---:|
| 32 MiB | 0.248 GiB/s | 0.388 GiB/s | 0.473 GiB/s |
| 64 MiB | 0.265 GiB/s | 0.397 GiB/s | 0.502 GiB/s |

The same sweep against the released Linux binary (AMD Ryzen Threadripper PRO
5955WX, Linux 5.15, loopback, single node, `sqlite` metadata engine), median of
three trials per point, again with zero retries:

| Pack payload | 1 outer worker | 2 outer workers | 4 outer workers |
|---:|---:|---:|---:|
| 32 MiB | 0.207 GiB/s | 0.327 GiB/s | 0.363 GiB/s |
| 64 MiB | 0.259 GiB/s | 0.394 GiB/s | 0.438 GiB/s |

Both platforms show useful parallel scaling and saturation beginning near four
outer workers on this loopback, single-node setup. They do not establish a
production default. Outer pack concurrency and per-pack multipart concurrency
multiply; their product and the configured HTTP connection pool bound the
potential active S3 requests.

Install the optional client and run the isolated live contract with:

```bash
python -m pip install -e '.[s3]'
DMI_GARAGE_BINARY=/path/to/garage \
  python tests/tools/run_garage_live.py
```

The harness pins Garage 2.3.0 by default, creates temporary credentials and
storage, and deletes the entire instance on exit. The official Garage download
page publishes Linux binaries; macOS can build the same pinned tag from source
using Garage's documented Cargo workflow.

`--tests` and `--marker` select what runs under that ephemeral server; the
defaults reproduce the isolated store and pipeline suite. The Garage plus
ClickHouse end-to-end suite needs both, so it is selected explicitly:

```bash
DMI_CLICKHOUSE_HOST=127.0.0.1 DMI_GARAGE_BINARY=/path/to/garage \
  python tests/tools/run_garage_live.py \
  --tests tests/test_capture_garage_e2e.py \
  --marker "garage and clickhouse and manual"
```

Run a sweep against the same ephemeral server with:

```bash
DMI_GARAGE_BINARY=/path/to/garage \
  python tests/tools/run_garage_live.py --benchmark -- \
  --pack-payload-bytes 32MiB,64MiB \
  --multipart-threshold-bytes 16MiB \
  --upload-workers 1,2,4 --packs-per-trial 4 \
  --multipart-chunk-bytes 16MiB --multipart-concurrency 2 --trials 3
```


#### Reference capture rate: single RTX 4090 server

The capacity gate needs a target input rate. This section derives one for a
single-GPU RTX 4090 host so the gate can be evaluated; it is a derived figure
from measured hardware limits and model shapes, not a measurement of a
production capture workload.

The PCIe link is not the constraint. Measured device-to-host bandwidth on an
RTX 4090 (PCIe 4.0 x16, pinned host memory) is 24.5 GiB/s at 16 MiB transfers
and above, 20.8 GiB/s at 1 MiB, and 9.3 GiB/s for pageable memory. That is two
orders of magnitude above anything the capture plane needs, so the link only
matters as a reason to keep hook copies pinned and batched, not as a capacity
bound.

What bounds capture volume is decode throughput times bytes captured per token.
Decode is weight-read bound, so tokens per second is approximately
(memory bandwidth / weight bytes) x batch, derated to 75% for attention, KV
traffic, and scheduling. With 1008 GB/s of device bandwidth:

| Model (bf16) | Batch | Decode | resid, all layers | resid, every 4th layer |
|---|---:|---:|---:|---:|
| Qwen3-4B (36 layers, d=2560) | 32 | ~3,000 tok/s | 0.519 GiB/s | 0.130 GiB/s |
| Llama-3.1-8B (32 layers, d=4096) | 32 | ~1,500 tok/s | 0.369 GiB/s | 0.092 GiB/s |
| Qwen3-14B (40 layers, d=5120) | 16 | ~430 tok/s | 0.165 GiB/s | 0.041 GiB/s |

A smaller model produces more capture bytes per second, not fewer: it decodes
proportionally faster, and per-token capture volume falls more slowly than
decode throughput rises. Attention-pattern hooks are excluded because they
scale with sequence length squared rather than per token; any policy capturing
them must sample aggressively and needs its own budget.

Against the measured plane capacity on this host, after the redundant hashing
and buffer copies were removed from the write path (the figures the earlier
sections of this document quote, 0.216 and 0.370 GiB/s, predate that change):

| Stage | Capacity | Supported input at the 1.2x gate |
|---|---:|---:|
| Pipeline, spool mode, one instance | 0.282 GiB/s | 0.235 GiB/s |
| Pipeline, direct mode, one instance | 0.291 GiB/s | 0.243 GiB/s |
| Pack writer alone | 0.472 GiB/s | 0.393 GiB/s |
| Garage upload, 4 workers, 64 MiB packs | 0.438 GiB/s | 0.365 GiB/s |

**Adopted target: 0.235 GiB/s of sustained capture per pipeline instance**,
taking the durable spool path as the production shape and leaving the pipeline
as the binding constraint rather than packing or upload.

The gate outcome follows directly:

- A sampled policy -- every 4th layer, all tokens -- passes on one 4090 for
  every model above, with 1.8x to 5.7x margin beyond the required 1.2x.
- Full-fidelity capture of every layer passes for Qwen3-14B (0.165 GiB/s of
  input against a 0.235 GiB/s budget) but not for the smaller, faster-decoding
  models: Llama-3.1-8B needs 0.369 GiB/s and Qwen3-4B 0.519 GiB/s. Those need
  either two to three pipeline instances sharded by layer range or producer
  rank, or a native writer.

The pipeline ceiling is a Python-side limit -- the pack writer alone runs at
0.472 GiB/s, so roughly 40% of achievable throughput is still lost between the
queue and persistence. That gap, not the object store, is where full-fidelity
single instance capture would have to be won, and it is the same boundary the
production-writer discussion in this document points at.

On a three-GPU host these numbers multiply: full capture across three 4090s is
about 1.1 GiB/s, requiring roughly six pipeline instances at the current
per-instance rate, or the native path.

### Phase 4 — derived ClickHouse catalog

- Implement notification and reconciliation discovery.
- Range-read and validate pack footers.
- Batch catalog rows across packs and expose logically deduplicated views.
- Prove full catalog rebuild from object storage.

Exit gate: forced missed and duplicate events converge to the expected catalog.

Status: exit gate passed in the CPU contract suite and against ClickHouse
26.9.1. Duplicate notifications are collapsed before footer work; missed
notifications are recovered by bounded prefix scans; corrupt footers never get
a pack commit marker; and an ambiguous marker insert safely replays descriptor
rows. Two physical descriptor and pack rows returned one logical row through
each public `FINAL` view in the live test.

The local 100,000-row, three-trial batch sweep measured:

| Rows per insert | Inserts per trial | Median rows/s |
|---:|---:|---:|
| 1,000 | 100 | 13,954 |
| 10,000 | 10 | 88,458 |
| 50,000 | 2 | 157,567 |

The result supports large client batches and the existing 10k–100k tuning
range, but does not establish a production default. It excludes object-store
discovery time and should be repeated on representative ClickHouse hardware.
ClickHouse recommends client batching and documents `FINAL` as the query-time
correctness mechanism for `ReplacingMergeTree` data; see its
[insert guidance](https://clickhouse.com/docs/concepts/best-practices/selecting-an-insert-strategy)
and [`FINAL` guidance](https://clickhouse.com/resources/engineering/clickhouse-optimize-table-final).

Run the benchmark with:

```bash
PYTHONPATH=src python -m benchmarks.bench_capture_catalog \
  --rows 100000 --batch-rows 10000 --trials 3
```

### Phase 5 — reader and summaries

- Implement bounded search, estimation, and coalesced range hydration.
- Add the core tensor summarizer, plus scalar metric and artifact extension
  points.
- Add agent-safe query and hydration limits.

Exit gate: selected hydration returns identical decoded tensors while avoiding
unrelated payload bytes.

Status: exit gate passed in the CPU contract suite, with snapshot and
pagination behaviour proven against ClickHouse 26.9.1.

Both halves of the gate are tested. *Identical decoded tensors* is a full round
trip per dtype -- tensor, pack, store, catalog, search, select, hydrate,
decode -- asserting array and byte equality; `bfloat16` is checked against
hand-chosen bit patterns including both NaN encodings and both infinities.
*No unrelated payload bytes* is not literal, because coalescing deliberately
spans small gaps: with `max_coalesce_gap_bytes = 0` every read falls exactly
inside a selected extent, and with the 4 KiB default the unrelated bytes stay
within `gap x joins`. `HydrationEstimate.request_bytes` additionally includes
a conservative cold-cache bound for footer verification.

Reads are pinned to a watermark. `CaptureQuery.filter_hash` identifies a query
independently of its page, keyset cursors carry that hash and the pinned
watermark, and `ClickHouseCaptureCatalog` resolves a capture out of
`*_capture_raw` with **one** `argMax` over a tuple of every non-grouped column,
ordered on the tuple `(index_version, store_id, pack_id)`.

Both halves of that shape are load-bearing, and the per-column
`argMax(<column>, index_version)` this document used to describe has been
removed:

- **The tuple ordering key.** One `index()` call allocates one `index_version`
  for the whole batch, so two packs describing a capture in a single pass
  produce rows whose ordering key is EQUAL, and ClickHouse leaves the winner of
  a tie undefined. Measured on 26.9/25.12, one corpus pinned at a single
  watermark resolved to a different pack at `max_threads = 1` than above it, and
  to a different one again once a merge had put both rows in one part -- a
  pinned selection silently reading different bytes before and after a
  background merge. `(index_version, store_id, pack_id)` is a total order over
  the rows in a group, and `index_version` still leads, so supersession is
  unchanged.
- **One aggregate, not one per column.** Twenty-seven separate `argMax` calls
  leave nothing forbidding `store_id` from one row and `object_key` from
  another -- a descriptor describing no pack that exists. It could not be
  reproduced on 25.12 across forty-two aggregation configurations, and the
  reason looks structural, but that is an observation about one build rather
  than a promise the engine makes. Keeping the per-column form and merely
  giving it the tuple key is also correct and costs +291% at a 100-row page,
  because ClickHouse compares a tuple ordering argument through a generic
  `Field` once per row per aggregate.

Every descriptor field except the locator is immutable for a
`(tenant_id, capture_id)`, which is what makes the pre-aggregation `WHERE`
filters safe; the rule is written out in `clickhouse_reader`'s module
docstring and pinned by a test.

`FINAL` is not a snapshot mechanism. It collapses duplicates to the highest
version present and only then applies predicates, so
`FINAL ... WHERE index_version <= W` drops a capture re-indexed above `W`
instead of returning its value at `W`. Measured on 26.9.1, snapshotting at
watermark 1 where `capture-a` was re-indexed at v2:

| Query shape | Result |
|---|---|
| `FINAL` + `index_version <= 1` | `capture-b` only; `capture-a` missing |
| `argMax` at watermark 1 | `capture-a@64`, `capture-b@128` |

Catalog facets -- `element_count`, `tensor_rank`, `token_span`,
`compression_ratio` -- are `MATERIALIZED` columns derived from data the writer
already stores, so they cost no indexer change and no extra object reads.
Anything derived from tensor *contents* runs at hydration time instead, because
`CatalogIndexer.index` range-reads pack footers only.

The 50,000-row, two-version measurement, re-run on the Linux reference host
against ClickHouse 25.12 after `e93a2c8` changed the projection shape, median
of three rounds of `--trials 3`:

| Measurement | Median |
|---|---:|
| `argMax` snapshot read | 35.6 ms |
| `FINAL` read (not a snapshot) | 16.8 ms |
| `max(index_version)` watermark | 2.3 ms |
| Page, `limit=100` | 171.7 ms |
| Page, `limit=1000` | 188.5 ms |
| Page, `limit=5000` | 256.9 ms |
| Page 1 vs page 25 at `limit=100` | 176.8 ms vs 173.9 ms |

**These are not comparable with the figures this table used to carry** (22.3 /
12.1 / 1.7 / 21.4 / 78.8 ms, and 21.5 vs 24.4 ms for depth). Those were taken
on an Apple Silicon laptop against ClickHouse 26.9.1 and, more importantly,
before `e93a2c8` -- so they describe a reader that resolved each column with its
own `argMax(<column>, index_version)`, which is the shape that commit removed
for the tie it left undefined. Two changes at once, and the hardware is the
larger of them.

The cost of the shape change alone was measured A/B on this host by that
commit: a 100-row page went from 140.1 ms to 171.7 ms, **+22.6%**, and across
page sizes, pagination depth and the selectivity cases the current shape runs
+17% to +43%, worst at `unfiltered` (136.4 -> 195.6 ms). That is what
determinism and a structurally coherent descriptor cost.

Correctness still costs about 2.1x a plain `FINAL` read at this size, with one
replay -- not a production figure, and it should be repeated on representative
hardware and duplicate ratios. Page cost remains flat with depth, which is the
property keyset pagination exists to provide. The watermark aggregate is a
second round trip per search but a small fraction of a page's cost, so caching
it is not yet worth the staleness.

Run the benchmark with:

```bash
PYTHONPATH=src python -m benchmarks.bench_capture_search \
  --rows 50000 --replays 2 --trials 3
```

### Phase 6 — migration

- Keep the current ClickHouse payload sink as the default initially.
- Compare golden workloads by identity, logical bytes, checksums, decoded
  tensors, and query results.
- Run fault injection and record performance variance.
- Switch the default only after compatibility, recovery, and throughput gates
  pass; preserve configuration rollback.

Status: not started. Two of the instruments it depends on now exist; the
comparison itself has not been run, and the production writer it compares
against has not been built.

#### Decision: the production writer is native

The Python implementation is a **reference implementation and conformance
suite**, permanently -- not a candidate production writer. The reason is
structural rather than performance-related: the ring transport reconstructs
tensors on a native callback thread specifically to avoid touching Python or the
GIL, and `DMXHostEngine` receives pre-assembled rows from that thread. There is
no hot-path caller for a Python pack sink, and creating one would reintroduce
per-tensor GIL contention that the ring exists to avoid.

The pack-and-upload plane will therefore be written in C++ alongside the
existing `ClickHouseInsertStage`, reusing `batching_queue.hpp` and
`pipelined_engine.hpp`. This supersedes the Phase 2 limitation describing the
Python pipeline as an interim stand-in for "the final reusable native slab
allocator": that is now the plan of record rather than a gap.

#### Fault injection

`tests/_faults.py` wraps the three boundaries that can misbehave -- object
store, ClickHouse client, and pack sink. Faults are scripted rather than random:
a schedule names which calls fail and how, so a failure reproduces exactly.

The characterised behaviour, which a native writer must reproduce:

| Fault | Required behaviour |
|---|---|
| Short read from the store | Refused, not silently truncated |
| Read failure mid-index | Aborts that pack; no partial pack is produced |
| Immutable key written twice | Converges; not a conflict |
| Sink failure | Pipeline fails loudly and refuses further admission |
| Insert failure | Pack left uncommitted; the batch is replayable |
| Duplicated insert | Absorbed by replay semantics |
| One corrupt pack in a batch | Fails only itself; the batch still indexes |

```bash
python -m pytest tests/test_capture_faults.py -q
```

#### Conformance manifest

`tests/tools/golden_workload.py` produces the golden-workload comparison this
phase requires, as a single JSON document over a deterministic corpus covering
every dtype the format accepts: pack identity and checksum, per-capture payload
sha256 and crc32, decoded-tensor sha256, placement, and the full summary
contract. Every value is language-neutral -- byte counts, hex digests and
integers -- so a native writer is conformant exactly when the same corpus
produces the same manifest.

```bash
python tests/tools/golden_workload.py generate --out golden.json
python tests/tools/golden_workload.py verify --manifest golden.json
```

The recorded manifest lives at `tests/data/capture_golden_manifest.json` and is
checked on every CPU run. `verify` diffs field by field, so a mismatch names the
capture and field that moved.

#### Remaining before the default can switch

- The native pack-and-upload plane, and configuration to select a sink with
  rollback preserved. Until a sink is selectable, "keep the current sink as the
  default initially" has nothing to compare against.
- Phase 3's capacity gate, which is still pending hardware. A throughput gate
  cannot pass while the 1.2x requirement is unmeasured.
- Performance variance under fault injection, which is not yet recorded.
- Three decisions the migration forces: whether the public views keep `FINAL`
  now that the correct read's cost against it is measured at 2.1x; where
  `index_version` comes from once more than one indexer runs, since the current
  per-process clock assumes a single writer; and whether catalog facets belong
  in the public views.

No phase requires a CUDA-side change.

## Operating signals

The Phase 2 metrics answer four initial on-call questions:

| Question | Signal |
|---|---|
| Is admission saturated? | accepted, dropped, timed-out, oversized, and closed counters; queue peaks; admission histogram |
| Is persistence keeping up? | persisted records/bytes, pack counts, flush reasons, persistence histogram |
| Did the worker fail? | failure counter and typed `pipeline_failed` event |
| Is durable work accumulating? | current/peak spool bytes and pending entry count |

Event callbacks receive bounded structured fields. They include pack identity for
correlation but never tensor payloads or capture metadata. Callback failures are
counted and cannot fail persistence. Deployment adapters can export these
snapshots and events to OpenTelemetry without coupling the storage core to one
telemetry vendor.

Phase 3 adds upload attempts, successful packs and bytes, failures, retries,
peak active uploads, peak bytes in flight, duration totals/maxima, and callback
failures. Events are `pack_upload_retry`, `pack_upload_committed`, and
`pack_upload_failed`; they contain pack identity and bounded diagnostic fields,
never tensor contents or credentials.

Phase 4 returns requested, skipped, indexed, and failed pack counts; indexed
rows; descriptor insert count; estimated metadata bytes; elapsed time; and a
bounded set of failure details. The `catalog_index_completed` event exposes the
same aggregate fields without object keys, capture metadata, or payloads.
Notification size, page size, packs per indexing call, estimated metadata
bytes, and retained failure details all have explicit caps.

## Phase 2 limitations

- The implementation is a Python CPU reference, not the final reusable native
  slab allocator. As of the Phase 6 decision this is permanent: the Python
  implementation is the reference and conformance suite, and the production
  writer will be native. See *Phase 6 -- Decision: the production writer is
  native*.
- One process owns a spool directory; cross-process locking is not implemented.
- Durable mode stages synchronously and uploads through a separate explicit
  uploader, so remote backpressure is isolated from local commit.
- The pipeline remains opt-in and is not connected to Ring².

## Phase 3 limitations

- Garage has no object versioning or object locks. UUID-based object keys have
  one designated writer; preflight and post-upload metadata checks detect
  retries and conflicts but are not a cross-writer compare-and-swap primitive.
- Boto3 is an optional dependency. Importing `dmi.storage.capture` does not
  require it; constructing an S3 client does.
- One uploader owns a spool. Cross-process scheduling and locking remain out of
  scope.
- The benchmark measures upload from recovered spool files. It deliberately
  excludes pack construction so pack and store saturation can be diagnosed
  independently.
- The Phase 3 code remains opt-in and does not change CUDA or the current
  ClickHouse payload sink.

## Phase 4 limitations

- Reconciliation is an explicit bounded call; deployment scheduling and event
  transport remain outside the storage library.
- The metadata byte bound is a conservative serialized-size estimate, not
  ClickHouse wire size.
- The current public views use `FINAL` for immediate replay correctness. Hot
  query workloads must measure its cost before choosing a different projection.
- `{prefix}_capture` re-evaluates its membership bound on every query. The bound
  carries no `max()` and no version predicate: it pairs `(index_version,
  publish_id)` between `{prefix}_snapshot_manifest` and
  `{prefix}_index_watermark`, so the view shows every pack whose publish has
  reached the watermark table *at the moment the query runs*. It therefore
  tracks published state rather than pinning a snapshot, and two reads of the
  view a moment apart can see different sets of packs. A caller that needs a
  stable selection uses the reader, which pins a watermark and carries it in
  the cursor.
- The catalog schema is versioned and checked, never migrated. There is no
  in-place upgrade path and no online one: the recovery from a version mismatch
  is the full rebuild above, which re-reads every pack footer in the store, so
  its cost scales with the corpus and readers see a catalog that is empty and
  then partial while it runs. Nothing is lost -- the packs are the durable copy
  -- but a large deployment needs to plan the window.

## Phase 5 limitations

- `index_version` no longer comes from any process clock. It is allocated by
  the catalog itself through the sole-claimant protocol in `allocate_version`,
  so it is unique and monotonic across writers; the wall clock stamps only the
  diagnostic `published_at_ns`. Making a snapshot visible is additionally
  fenced on a durable publisher lease, so only the lease holder can publish.
  What remains is not clock skew but the residual publication windows recorded
  in `catalog-descriptor-key.md` under "What this does not close".
- Cursors are validated, not authenticated. A tampered cursor is rejected as
  malformed -- strict base64 and envelope checks -- but nothing binds a cursor to
  the caller who received it. A cursor can only address the keyspace its own
  filters already reach.
- Query filters apply before aggregation. That is safe only because a descriptor
  derives from an immutable pack footer, so re-indexing a capture rewrites
  identical values. Mutable descriptors would require filtering after `argMax`.
- Each search issues a second round trip for `max(index_version)`, needed to
  reject cursors ahead of the catalog.
- `get_by_ids` matches on `capture_id`, the last element of the sort key, so it
  does not benefit from the primary index.
- Catalog facets are on `*_capture_raw` only; the public `FINAL` views are
  unchanged.
- The per-extension time budget is checked after each call. A runaway extension
  is reported, not interrupted; preemption needs a worker boundary.
- Core summary statistics cover finite elements only, with `nan_count`,
  `inf_count` and `finite_count` reported alongside. `l2_norm` factors out the
  largest magnitude before squaring, because the direct `sqrt(sum(x**2))` form
  overflows float64 for large-magnitude tensors and returns infinity where the
  true norm is finite.
- A selection is one bounded page. Callers paginate explicitly.

## Alternatives considered

### Host writes payload and catalog directly

This makes ClickHouse latency, availability, and schema part of the capture
commit path. It also fragments inserts across inference hosts. Rejected in favor
of a centralized, independently scalable indexer.

### Separate payload and manifest objects

Standard manifest formats can be convenient, but object storage does not offer
an atomic transaction across two keys. A footer inside one pack supplies the
same reconstruction data with one commit boundary. Parquet manifest export can
remain an optional downstream interoperability feature.

### Raw payloads in ClickHouse

Operationally simple, but large binary columns compete with searchable metadata
for inserts, merges, caches, and storage. ClickHouse remains the derived catalog
rather than the raw byte store.

### One object per tensor

Simple addressing, but request rate and object count dominate for small
tensors. Packs amortize those costs while retaining record-level range reads.

### RocksDB or SQLite in the host path

An embedded index introduces another recovery and compaction surface. Bounded
pack files plus atomic rename are sufficient for the initial spool. Reconsider
only after measurement identifies a concrete need.

### Object storage without ClickHouse

Durable and inexpensive, but poorly suited to interactive high-cardinality
discovery and aggregation. ClickHouse supplies the rebuildable hot query layer.

## References

- [Amazon S3 data consistency model](https://docs.aws.amazon.com/console/s3/UsingObjects.html)
- [Amazon S3 event notifications](https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventNotifications.html)
- [Amazon S3 event ordering and duplication](https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-how-to-event-types-and-destinations.html)
- [Amazon S3 performance guidance](https://docs.aws.amazon.com/pdfs/whitepapers/latest/s3-optimizing-performance-best-practices/s3-optimizing-performance-best-practices.pdf)
- [ClickHouse insert strategy](https://clickhouse.com/docs/concepts/best-practices/selecting-an-insert-strategy)
- [ClickHouse ReplacingMergeTree and FINAL](https://clickhouse.com/resources/engineering/clickhouse-optimize-table-final)
- [ClickHouse and Amazon S3](https://clickhouse.com/integrations/amazon_s3)
- [Garage documentation](https://garagehq.deuxfleurs.fr/)
- [Garage 2.3 quick start](https://garagehq.deuxfleurs.fr/documentation/quick-start/)
- [Garage S3 compatibility](https://garagehq.deuxfleurs.fr/documentation/reference-manual/s3-compatibility/)
- [Garage release downloads](https://garagehq.deuxfleurs.fr/download/)
- [Boto3 managed S3 transfers](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html#file-transfer-configuration)
- [Amazon S3 multipart upload](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html)
