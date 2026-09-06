# TODO: Native capture storage pipeline

Loop state file. Re-read at the start of every work cycle. Check off only with
verification evidence (test output / JSON artifact path) noted inline.

Legend: `[ ]` pending · `[~]` in progress · `[x]` done (with evidence) · `[!]` blocked/escalated

## Loop guardrails (controller rules — this file is the controller)

- **Terminal condition:** this loop is done when Checkpoint B passes (its
  criteria are defined inline at the Checkpoint B entry below — cold-path
  gates, not a copy of Checkpoint A's hot-path gates). Phase C is a separate
  follow-up loop, not work this loop picks up by default.
- **Budget exit:** one task gets at most 4 wall-clock hours and 3 verification
  attempts per work cycle before it must stop for the cycle.
- **No-progress exit:** 2 consecutive work cycles with no state transition
  (no task checked off, no ledger entry, no revert logged) → stop and
  escalate, never silently continue.
- **Escalation:** `[!]` = a task failed its verification gate twice, or is
  blocked on an external dependency (PR merge, human review, GPU
  availability). Always carry a one-line reason and a named unblocker.
- **Single authority:** this tracked copy is the controller. The untracked
  `tasks/` snapshot in the main checkout (feat/dmi-configurator working tree)
  is historical; its corrected T0.2 evidence and harness fix were merged here
  on 2026-09-06.

## Phase 0 — Measure & attribute (gate)

- [x] T0.1 Baseline: spool 0.2349 GiB/s, direct 0.2318 GiB/s (median of 5, 5955WX).
      Evidence: `benchmarks/data/native-pipeline/baseline-{spool,direct}.json`,
      ledger section in `docs/benchmarks.md` ("Native capture pipeline ledger").
      NOTE: below the doc's 0.282/0.291 — re-derive attribution target in T0.2 from
      measured, not doc, numbers.
- [x] T0.2 Attribution: writer-only 0.359 vs pipeline 0.235 GiB/s → ~35% GIL+queue tax.
      Inside writer: 29% asdict+deepcopy metadata dict, 24% crc32 (1.17 GiB/s),
      13% sha256 seal. Spool stage ~0.86 GiB/s and direct-mode
      FilesystemPackStore.put ~0.84 GiB/s (re-measured 2026-09-06 after a
      harness fix — the first run swallowed PackCapacityError and dropped ~19%
      of records, inflating both rows ~1.25x; the old 1.15/0.199 figures were
      wrong, and absolute I/O rates also drift with machine state while
      CPU-bound rows reproduce) — neither store row is the bottleneck.
      T0.1 baselines and all Phase-A bench numbers are unaffected: they come
      from bench_capture_pipeline.py / bench_builder / bench_sink, not the
      buggy stage harness, and none of them swallowed append errors.
      Evidence: ledger T0.2 section in `docs/benchmarks.md` (incl. the
      re-measurement note); fixed harness at benchmarks/profile_pipeline_stages.py
      with regression test tests/test_profile_pipeline_stages.py.
- [x] T0.3 Native kernel micro-bench. CRC32C 8-chain 13.7 GiB/s (6.5× Python
      zlib), sha256 2.09 GiB/s (parity w/ Python), append memcpy 2.6 GiB/s,
      footer 247 GiB/s, spool 0.88 GiB/s fsync-bound. Modeled native ceiling
      ~1.5–2.0 GiB/s/instance (~8× Python). Two invalid harness readings
      (cache-resident corpus) caught, reverted, logged.
      Evidence: ledger T0.3 section; harnesses checked in at
      native/csrc/pack/bench_{crc32c,sha256,kernels}.cpp +
      benchmarks/profile_pipeline_stages.py
- [x] T0.4 Scope decision: derived per-instance target 1.1 GiB/s/host (full-fidelity
      3×4090); Python 0.235, modeled native ceiling ~1.5–2.0 GiB/s/instance →
      **full native port, Phases A+B**. Pre-approved by human: proceed toward best
      performance with parallelism; narrow scope only if the profile indicts
      non-Python bottlenecks — it did not.
      Evidence: ledger "T0.4 scope decision (2026-09-05)" in `docs/benchmarks.md`;
      Checkpoint A subsequently passed against the derived target.
      (Closed 2026-09-06 during plan reconciliation — the decision was recorded in
      the ledger on 2026-09-05 but the box was never checked.)

## Phase A — Hot path

- [x] A1 Pack writer core (`native/csrc/pack/`) + golden byte-equality test
      (`tests/test_native_pack_conformance.py`).
      Evidence: 7/7 conformance tests green (golden corpus, escaped text,
      alignment, empty/multi-dim shapes, rejection parity, zlib CRC parity);
      bench 0.63 GiB/s best-of-5 vs 0.359 Python (+75%).
      Gates: `make -C native build/conformance_main`,
      `make -C native build/bench_builder` (no CUDA needed).
- [x] A2a Object-store client core: PUT / GET-range / HEAD + SigV4 (`native/csrc/store/`).
      Evidence: `tests/test_native_s3_sign.py` 11/11 differential vs botocore.
- [x] A2b List + fault matrix (short read, mid-write failure, 5xx retry, timeout).
      Evidence: `tests/test_native_s3_client.py` 8/8 vs fake S3 with
      server-side botocore re-signing (PUT/GET/HEAD/DELETE round trip,
      3 MiB multipart, pagination, retry taxonomy, short-body refusal).
      Gates: `make -C native build/conformance_sign build/conformance_store`
      (no CUDA; libcurl headers via CURL_INCDIR sysroot).
- [x] A3a Spool state machine + restart recovery; Python drains native spool.
      Evidence: `tests/test_native_spool.py` 7/7 (both cross directions,
      idempotence, capacity, quarantine, stale-.open cleanup, removal).
      Gate: `make -C native build/conformance_spool`.
- [x] A3b NativePackSink single worker → spool, latched failure, counters.
      Evidence: `tests/test_native_pack_sink.py` 9/9 (golden e2e with
      Python PackReader read-back, session/linger sealing, drop/block
      policies, duplicates, oversized, size splits, object-key parity).
      CPU suite green (1155 passed). Gate:
      `make -C native build/conformance_sink`.
- [x] A4 Uploader: bounded workers, bytes-in-flight, checksum verify; e2e ring→Garage.
      Evidence: `tests/test_native_uploader.py` 6/6 (sink→spool→fake-S3
      e2e with Python PackReader read-back, preflight idempotence with
      PUT-count proof, parallel batch by position, layered retry,
      corrupt refusal, byte gate). Native suite 48/48.
- [x] A5a Scope-hash worker pool + N-scaling curve + throughput gate.
      Evidence: `tests/test_native_pack_sink.py` 11/11 (incl. N=4 scope
      isolation + cross-worker flush); `bench_sink` NVMe 0.42→0.52
      monotonic to N=8 (+121% over Python 0.235); tmpfs to 0.65.
      Variance ±20% (shared host) — gates clear with margin; re-measure
      quiet before published claims. Gate:
      `make -C native build/bench_sink` (single|interleaved, spooldir=).
- [x] A5b Sink-selection config (opt-in) + rollback demo.
      Evidence: `tests/test_native_adapter_torch.py` 19/19 (torch
      envelopes, all dtypes, validation, lease guards);
      `tests/test_native_rollback.py` 3/3 (shared layout, config
      validation, native→Python catalog indexing). CPU suite 1187 green.
- **Checkpoint A: PASS** (human review): byte-equality ✓, CPU suite green
  ✓, N=1 ≥ derived target ✓ (worst reading +45%), N-scaling monotonic ✓,
  rollback ✓. Follow-ups deferred: CRC fusion, Phase B, Phase C.

## Phase B — Cold write path

- [x] B1a Lease coordinator native (textual SQL port) + ported lease live tests.
      Evidence: tests/test_native_catalog_lease_live.py 9/9 live (refusal names
      holder, expired takeover, release tombstone at own term, contested claim,
      fence on empty table, statement byte-identity vs the Python module);
      `native/csrc/catalog/` speaks ClickHouse HTTP (libcurl, POST body).
      Gate: `make -C native build/conformance_catalog`.
- [x] B1b Sole-claimant allocate_version + contention tests.
      Evidence: same suite — monotonic versions, allocator floor above an
      external watermark head, 3 racing driver processes → distinct versions,
      each returned version claimed exactly once.
- [x] B2a Descriptor batches + commit_packs/committed_pack_ids.
      Evidence: head-to-head capture_raw rows byte-identical vs the Python
      writer, MATERIALIZED facets included; replay-guard round trip.
      Statement byte-identity where clickhouse-driver substitutes client-side
      (the committed_pack_ids SELECT incl. chunking at _inline_chunks
      boundaries); INSERT parity is behavioral (native-written rows read back
      identical through the Python client) since the Python driver ships row
      data in the native protocol, not statement text.
- [x] B2b Fenced publish_snapshot; verify_replicated_quorum.py PASS vs the C++ writer.
      Evidence: the verifier now runs BOTH legs — the Python writer and the
      native writer through conformance_catalog — against two replicas of
      each protocol table under one Keeper (tests/tools/quorum_harness,
      ports 9181/9010/8110). All 12 checks PASS, verdict 0, two runs: the
      native publish cycle with insert_quorum=2, the settings recorded in
      the query log on every native deciding INSERT, native retention
      admitted on ReplicatedMergeTree, the quorum-unset control, quorum-off
      mid-life failing loudly, and the unsatisfiable quorum refused with
      Code 285 promptly.
      Evidence: publish below the head loses the race; taken-over publisher
      writes NOTHING (error names the successor); takeover between the two
      statements leaves the documented inert orphan rows; injected transport
      death quarantines without the tombstone until the TTL lapses. 15/15
      live, three consecutive runs.
      NOTE (#125, post-#119): the publish protocol is no longer SQL-only — one
      writer serialises publishes (`_serial`), binds to its process
      (`_owned_by_this_process`), and quarantines on outcome-unknown failures
      (driver/transport error during a fenced statement → discard lease WITHOUT
      tombstone, refuse acquire/renew/publish for a full lease_ttl_ns). The
      native writer must reproduce these client-side semantics, and the #125
      concurrency tests (two concurrent publishes on one writer serialised;
      cross-process use refused; failed publish releases the writer) port
      alongside the SQL suites.
      STATUS (2026-09-06, Checkpoint B review): the QUARANTINE half is
      ported and tested (injected transport death, and a lapsed window that
      reports itself over). The serialisation and process-binding halves are
      NOT: the native writer relies on the driver being one process running
      one op at a time instead of holding a lock or checking its pid, and
      the three #125 concurrency tests are not ported. Sound for the shipped
      path — pybind and the driver are single-threaded per writer — and a
      real gap for any other embedding, so it is recorded as a deferred
      follow-up at Checkpoint B rather than claimed. `catalog_writer.h`
      names the assumption at the class it applies to.
- [x] B3 indexer-native (footer read → batch → publish) + e2e with Python CaptureReader oracle.
      Evidence: native sink → native uploader → native pack-index read → native
      publish, and the PYTHON CaptureReader resolves both captures with
      metadata intact; second pass skips via the replay guard and moves no
      watermark; no-lease refusal burns no version. Pack-index validation
      ported from PackIndex.from_store (trailer/footer/ranges/dtype-size/
      dup-ids), external data refused at the boundary. 18/18 live ×2 runs.
- [x] B4 schema-native + maintenance-native (GC) — complete.
      Evidence: orphan manifest rows below the head collected after the
      two-read settle (publish timeout apart); superseded lease rows and
      spent claims collected; membership of published versions, the head
      lease row, and the pending claim above the head kept; second pass
      idempotent. Counted-then-deleted per table, literal-pair chunks.
      23/23 live ×4 consecutive runs. Schema port: ensure/drop with the
      install serialised on the publisher lease (borrow-or-take, the
      budget covering one contested-term wait), server-side conditional
      stamp, and the compatibility refusals ported — kinds, sort keys,
      the ReplacingMergeTree version-argument property (first balanced
      engine-argument group; bare calls render without parentheses),
      stamp version, and inventory-without-membership. Concurrent
      installers: the second arrives DURING the first's install and
      completes by waiting; a forced-zero-stagger start can cascade
      contested terms past the ttl+margin budget, and the Python
      implementation fails identically there.
- **Checkpoint B: PASS** (reviewed 2026-09-06 against PR #127 + #128) —
  criteria (cold-path, defined 2026-09-06), each re-verified at `7c7f725`:
  ported live suites green next to their Python oracles ✓ (lease, allocator,
  descriptors, publish, indexer, schema, GC — 44/44 ×2; the #125 concurrency
  trio is NOT among them, deferred below) · statement byte-identity gate
  green ✓ (scope: `release` + `fence` only — the parameterized statements are
  semantic-equivalence, not textual) · verify_replicated_quorum.py PASS
  against the C++ writer ✓ (12/12 both legs, ×2, Keeper-backed two-replica
  harness) · e2e indexer run read back identically by the Python
  CaptureReader ✓ · CPU suite green ✓ (1202) · rollback (native→Python
  catalog path) still proven ✓ · clean rebuild carries zero warnings ✓ ·
  before any published throughput claim: re-measure on a quiet host and
  re-baseline on the reference host (owns the A5a/plan obligation — STILL
  OPEN, and the only thing between these numbers and a published one).
  Deferred follow-ups: the #125 serialisation + process-binding halves and
  their three tests (see B2b STATUS); a byte-identity gate over the
  parameterized statements, or a risk-table correction saying they are
  semantically equivalent rather than textually identical.

## Phase C — Serving path

- [x] C1 reader-native + read-parity suite.
      Evidence: tests/test_native_reader_parity_live.py 5/5 live (×3
      consecutive): search parity with and without filters (identical
      32-field descriptors both readers), CURSORS CROSS THE
      IMPLEMENTATIONS (the native codec is byte-compatible with
      cursor.py — same envelope, unpadded url-safe base64, canonical
      alphabet, filter_hash binding — so each side accepts and walks the
      other's cursor and the union covers the corpus in order),
      get_by_ids parity with the watermark bound refused by both, and
      supersession (a later pack's locator wins on both sides). Native
      bugs the parity suite caught: the projection's aggregate tuple
      carried the sort-key columns (a mixed-row shape), the cursor keyed
      on the row BEYOND the page, and the base64url encoder emitted a
      phantom trailing byte in both remainder branches. 44/44 native
      catalog tests; CPU gate 1202 passed.
- [~] C2 hydration/summary/extensions native.
      Scope (from reader.py, 445 lines): select (search → one bounded page
      → CaptureSelection), estimate, hydrate (plan per pack with coalesced
      ranges, footer-authoritative verification phase before any payload
      fetch, byte/request budgets, per-descriptor verify_payload), summarize
      (decode_tensor + core tensor stats; the ExtensionRegistry stays
      Python-side — extensions are Python pluggables, the native leg covers
      hydrate + core stats). Parity tests must pin: identical payload bytes
      native vs Python hydrate, identical core summary numbers, budget
      refusals both sides. Native pieces already in place: pack_index
      (footer read + validation), S3Client GetRange, reader search/select
      surface. NOT started.
- [ ] C3 default-switch + final docs

## Checkpoints log

- Checkpoint B (2026-09-06): PASS (human review of #127 + #128). Phase C
  authorized: C1 reader-native next.

- Checkpoint B (2026-09-06): PASS with two recorded gaps, reviewed against
  PR #127 (Phase A → main, head 6473871) and PR #128 (Phase B stacked, head
  7c7f725). Every criterion re-verified in one sitting rather than read off
  earlier commits: 44/44 ported live ×2, CPU 1202, Python oracles 63,
  clean rebuild 0 warnings, rollback and reader-oracle both green, and the
  quorum verifier 12/12 on BOTH legs ×2 against a real Keeper-backed
  two-replica cluster — the criterion PR #128's body still describes as
  "not runnable in this environment" (it was: the harness keeper was live,
  so only `quorum_harness/server.xml` needed starting).
  Gaps recorded, neither blocking: (1) the #125 serialisation and
  process-binding halves are argued from single-threaded embedding rather
  than ported, and their three tests are absent — `catalog_writer.h` now
  states that limit at the class; (2) the byte-identity gate covers only
  `release` and `fence`, so "textual SQL identity" in the risk table
  overstates what is checked for the parameterized statements.
  Terminal condition of this loop reached; Phase C remains a separate loop.
- Plan reconciliation (2026-09-06): merged the corrected T0.2 evidence and
  harness fix (PackCapacityError swallow) from the main-checkout snapshot into
  this controller; closed T0.4 against the ledger; re-grounded B2b on #125
  (client-side serialization/process binding/quarantine are part of the port
  surface); defined Checkpoint B's own criteria; declared this tracked copy
  the single authority.
- Checkpoint A (2026-09-05): PASS. 74 native tests green, CPU suite 1187
  green, N-scaling monotonic, rollback proven. Commits cdc774a..A5b.
  Open follow-ups: CRC+memcpy fusion, Phase B (catalog-native),
  Phase C (reader-native + default-switch).
- Differential round (2026-09-05): native vs Python reference head-to-head
  — recorded manifest digest, pipeline descriptor unions, uploader object
  bytes+metadata all identical. 77 native tests, CPU suite 1190 green.

## Escalations

(none yet)
