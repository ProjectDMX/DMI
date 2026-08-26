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
| Summaries | Planned |

This slice is opt-in and has no connection to the CUDA producer or current
ClickHouse payload sink.

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
4. Estimate payload bytes and request count.
5. Coalesce adjacent ranges within each pack.
6. Fetch with byte, request, and concurrency limits.
7. Verify checksums and decode selected records.

List and summary operations never hydrate tensor bytes implicitly. Storage
credentials and provider endpoints are resolved from deployment configuration,
not returned in catalog rows.

## Extension contracts

```python
class PackWriter(Protocol):
    def append(self, record: CaptureRecord) -> None: ...
    def seal(self) -> SealedPack: ...


class PackSource(Protocol):
    pack_id: str
    object_bytes: int
    checksum: str
    def open(self) -> BinaryIO: ...


class PackStore(Protocol):
    def put(self, pack: PackSource, object_key: str) -> PackRef: ...
    def stat(self, ref: PackRef) -> ObjectInfo: ...
    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes: ...
    def list_committed(self, cursor: ScanCursor) -> Page[PackRef]: ...


class PackSink(Protocol):
    def persist(self, ready: ReadyPack) -> object: ...


class CommitFeed(Protocol):
    def watch(self, cursor: EventCursor) -> Iterable[PackRef]: ...
    def scan(self, cursor: ScanCursor) -> Page[PackRef]: ...


class CatalogIndexer(Protocol):
    def index(self, packs: Sequence[PackRef]) -> IndexResult: ...


class CaptureSummarizer(Protocol):
    name: str
    version: str
    def summarize(self, capture: CaptureDescriptor, tensor: TensorView) -> SummaryBatch: ...


class CaptureReader(Protocol):
    def search(self, query: CaptureQuery) -> Page[CaptureDescriptor]: ...
    def estimate(self, ids: Sequence[CaptureId]) -> HydrationEstimate: ...
    def hydrate(self, request: HydrationRequest) -> Iterable[TensorRecord]: ...
```

Initial implementations:

```text
PackStore
  - FilesystemPackStore
  - S3PackStore

CatalogIndexer
  - ClickHouseCatalogIndexer

CaptureSummarizer
  - CoreTensorStatsSummarizer
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

# Planned additions
src/dmi/storage/
  summaries/
    contracts.py
    core.py

native/csrc/storage/
  capture_record.h
  pack_builder.h
  persistence_pipeline.h
  spool.h
```

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

Status: implementation and local compatibility gates passed; the production
capacity gate remains pending because no target device-to-host rate or
representative network/storage hardware has been supplied. The pinned Garage
v2.3.0 live test covers multipart upload, retry idempotency, object metadata,
listing, and the two range reads used to load a pack footer.

An Apple Silicon local sweep used four packs per trial, a 16 MiB multipart
threshold and chunk, two multipart requests per pack, and three trials per
point. All objects and byte caps verified with zero retries:

| Pack payload | 1 outer worker | 2 outer workers | 4 outer workers |
|---:|---:|---:|---:|
| 32 MiB | 0.248 GiB/s | 0.388 GiB/s | 0.473 GiB/s |
| 64 MiB | 0.265 GiB/s | 0.397 GiB/s | 0.502 GiB/s |

These results show useful parallel scaling and saturation beginning near four
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
page currently publishes Linux binaries; macOS can build the same pinned tag
from source using Garage's documented Cargo workflow.

Run a sweep against the same ephemeral server with:

```bash
DMI_GARAGE_BINARY=/path/to/garage \
  python tests/tools/run_garage_live.py --benchmark -- \
  --pack-payload-bytes 32MiB,64MiB \
  --multipart-threshold-bytes 16MiB \
  --upload-workers 1,2,4 --packs-per-trial 4 \
  --multipart-chunk-bytes 16MiB --multipart-concurrency 2 --trials 3
```

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
within `gap x joins` and match `HydrationEstimate.request_bytes` exactly.

Reads are pinned to a watermark. `CaptureQuery.filter_hash` identifies a query
independently of its page, keyset cursors carry that hash and the pinned
watermark, and `ClickHouseCaptureCatalog` resolves each column with `argMax`
over `*_capture_raw`.

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

The 50,000-row, two-version measurement:

| Measurement | Median |
|---|---:|
| `argMax` snapshot read | 22.3 ms |
| `FINAL` read (not a snapshot) | 12.1 ms |
| `max(index_version)` watermark | 1.7 ms |
| Page, `limit=100` | 21.4 ms |
| Page, `limit=1000` | 78.8 ms |
| Page 1 vs page 25 at `limit=100` | 21.5 ms vs 24.4 ms |

Correctness costs about 1.85x a plain `FINAL` read at this size, on a laptop
build with one replay -- it is not a production figure and should be repeated on
representative hardware and duplicate ratios. Page cost is flat with depth,
which is the property keyset pagination exists to provide. The watermark
aggregate is a second round trip per search but under a tenth of a page's cost,
so caching it is not yet worth the staleness.

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
  slab allocator.
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

## Phase 5 limitations

- `index_version` is `time_ns()` from the indexing process's own clock. With
  more than one indexer, clock skew makes versions non-monotone across writers,
  and a watermark taken from one indexer can permanently exclude rows written by
  another. Phase 5 assumes a single indexer owns a catalog; coordinating the
  version source is deferred to Phase 6, which already revisits indexer
  topology.
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
  `inf_count` and `finite_count` reported alongside.
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
