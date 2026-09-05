# TODO: Native capture storage pipeline

Loop state file. Re-read at the start of every work cycle. Check off only with
verification evidence (test output / JSON artifact path) noted inline.

Legend: `[ ]` pending · `[~]` in progress · `[x]` done (with evidence) · `[!]` blocked/escalated

## Phase 0 — Measure & attribute (gate)

- [x] T0.1 Baseline: spool 0.2349 GiB/s, direct 0.2318 GiB/s (median of 5, 5955WX).
      Evidence: `benchmarks/data/native-pipeline/baseline-{spool,direct}.json`,
      ledger section in `docs/benchmarks.md` ("Native capture pipeline ledger").
      NOTE: below the doc's 0.282/0.291 — re-derive attribution target in T0.2 from
      measured, not doc, numbers.
- [x] T0.2 Attribution: writer-only 0.359 vs pipeline 0.235 GiB/s → ~35% GIL+queue tax.
      Inside writer: 29% asdict+deepcopy metadata dict, 24% crc32 (1.17 GiB/s),
      13% sha256 seal. Spool stage 1.15 GiB/s (NVMe, fsync incl.) — not the bottleneck;
      direct-mode FilesystemPackStore.put 0.199 GiB/s IS fsync-bound.
      Evidence: ledger T0.2 section in `docs/benchmarks.md`; harness
      /tmp/opencode/profile_pipeline_stages.py (to be checked in as
      benchmarks/profile_pipeline_stages.py in T0.4 commit).
- [x] T0.3 Native kernel micro-bench. CRC32C 8-chain 13.7 GiB/s (6.5× Python
      zlib), sha256 2.09 GiB/s (parity w/ Python), append memcpy 2.6 GiB/s,
      footer 247 GiB/s, spool 0.88 GiB/s fsync-bound. Modeled native ceiling
      ~1.5–2.0 GiB/s/instance (~8× Python). Two invalid harness readings
      (cache-resident corpus) caught, reverted, logged.
      Evidence: ledger T0.3 section; harnesses checked in at
      native/csrc/pack/bench_{crc32c,sha256,kernels}.cpp +
      benchmarks/profile_pipeline_stages.py
- [ ] T0.4 Scope decision: derived per-instance target; full port vs narrow. Pre-approved
      by human: proceed toward best performance with parallelism; do not stop while
      goals unmet — narrow scope only if the profile indicts non-Python bottlenecks.

## Phase A — Hot path

- [ ] A1 Pack writer core (`native/csrc/pack/`) + golden byte-equality test
      (`tests/test_native_pack_conformance.py`)
- [ ] A2a Object-store client core: PUT / GET-range / HEAD + SigV4 (`native/csrc/store/`)
- [ ] A2b List + fault matrix (short read, mid-write failure, 5xx retry, timeout)
- [ ] A3a Spool state machine + restart recovery; Python SpoolUploader drains native spool
- [ ] A3b NativePackSink : RecordSink (single worker → spool), latched failure, counters
- [ ] A4 Uploader: bounded workers, bytes-in-flight, checksum verify; e2e ring→Garage
- [ ] A5a Scope-hash worker pool via pipelined_engine; N-scaling curve
- [ ] A5b Sink-selection config (opt-in) + rollback demo
- **Checkpoint A** (human review): byte-equality ✓, CPU suite green ✓, N=1 ≥ derived
  target ✓, N=4 store-bound ✓, rollback ✓

## Phase B — Cold write path

- [ ] B1a Lease coordinator native (textual SQL port) + ported lease live tests
- [ ] B1b Sole-claimant allocate_version + contention tests
- [ ] B2a Descriptor batches + commit_packs/committed_pack_ids
- [ ] B2b Fenced publish_snapshot; verify_replicated_quorum.py PASS vs C++ writer
- [ ] B3 indexer-native (footer read → batch → publish) + e2e with Python CaptureReader oracle
- [ ] B4 schema-native + maintenance-native (GC)
- **Checkpoint B** (human review)

## Phase C — Serving path (deferred follow-up)

- [ ] C1 reader-native + read-parity suite
- [ ] C2 hydration/summary/extensions native
- [ ] C3 default-switch + final docs

## Checkpoints log

(none yet)

## Escalations

(none yet)
