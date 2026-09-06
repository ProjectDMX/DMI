# Implementation Plan: End-to-end native capture storage pipeline

Branch: `feat/native-capture-pipeline` (rebased onto `origin/main` @ `9bf0db8`,
post-#125; originally cut from `0543ea5`, post-#119). Phase B continues on
`feat/native-catalog-b1`.

## Overview

Replace the Python capture-pack storage path with a C++ pipeline — ring callback →
pack → durable spool → object store → catalog index — with Python kept permanently
as conformance oracle and rollback path. Performance-oriented: Phase 0 measures and
attributes the Python bottleneck before any port; every phase lands with a
re-measured number against the recorded baseline.

## Architecture decisions

- **Vertical slices**: every task ends runnable and measurable.
- **Fail fast**: Phase 0 (profile + native kernel ceiling) gates the rewrite.
- **Python is the oracle**: byte-equality for packs, behavioral parity for catalog
  protocols and query results; the reference suite must stay green in every task.
- **SQL ports textually**: #119's lease/allocator/publish protocols ported as
  identical SQL sequences, verified by ported live suites + the replicated-quorum
  verifier (`tests/tools/verify_replicated_quorum.py`). Post-#125 the publish
  protocol also carries client-side semantics — per-writer publish
  serialisation, process binding, outcome-unknown quarantine — which the native
  writer must reproduce and the #125 concurrency tests verify (see B2b).
- **D1**: S3 client = libcurl + hand-rolled SigV4 (no aws-sdk-cpp).
- **D2**: durable spool mode only for native v1; direct mode stays Python reference.
- **D3**: scope-hash worker partition; per-scope serialism is an inherent ceiling,
  documented, not engineered away.
- **D4**: default sink stays `clickhouse` until all gates pass.
- **D5**: C++17, pybind11, existing `native/Makefile`, no CUDA changes.

## Task list

Ordered index — the live checklist with state lives in `tasks/todo.md`.

- Phase 0 (gate): T0.1 baseline → T0.2 attribution → T0.3 kernel ceiling → T0.4 scope decision
- Phase A (hot path): A1 pack writer → A2a/b object store → A3a/b spool+sink →
  A4 uploader → A5a/b workers+selection → Checkpoint A
- Phase B (cold write path): B1a/b lease+allocator → B2a/b descriptors+publish →
  B3 indexer → B4 schema+GC → Checkpoint B
- Phase C (serving path, deferred follow-up): C1 reader, C2 hydration, C3 flip

Full task detail with acceptance criteria and verification: `tasks/todo.md`.

## Performance discipline

- Same harness, same host, ≥3 trials, JSON artifacts per measurement.
- Delta must beat run-to-run variance; neutral or worse → revert, logged.
- Correctness gates the metric: no number counts if an oracle suite went red.
- CI: kernel micro-bench budget check (CPU-marked).

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Catalog protocol port re-opens #119/#125 surface | High | Textual SQL identity where gated (`release`, `fence`); semantic equivalence elsewhere; 1:1 test port (#125 concurrency trio deferred — see Checkpoint B); quorum verifier PASS ✓ 12/12 both legs |
| Profile indicts non-GIL bottleneck | High | Phase 0 gate before any port beyond T0.3 |
| Two implementations drift | Med | Behavioral tests drive both; Python oracle-only after flip |
| libcurl/SigV4 subtleties | Med | Garage conformance + fault matrix (A2b) |
| Scale (~6.2k Python → est. 9–12k C++) | Med | Phases land independently; A alone delivers hot-path win |

## Open questions / resolved

- Phase C deferred to follow-up (approved).
- Baseline host: current machine; re-baseline on the reference host before any
  published claim.
