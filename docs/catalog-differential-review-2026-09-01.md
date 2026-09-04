# Differential Security Review — PR #119 "Capture catalog: pack identity in the descriptor key, fenced snapshot publication"

**Repository:** ProjectDMX/DMI · **Branch:** `design/catalog-descriptor-key` · **Range:** `cb4e490` (`origin/main`) … `a808812`
**Date:** 2026-09-01 · **Method:** trailofbits `differential-review` (FOCUSED strategy) · **Status of this file:** preserved review artifact; section 0 records the later remediation

---

## 0. Resolution status (updated after remediation)

Every finding below has been addressed on the branch:

| Finding | Status | Where |
|---|---|---|
| M1 stale pre-PR indexer undetected | **Fixed** | `clickhouse_schema.py` refuses a superseded object beside this build's own |
| M2 contested lease term overtaken | **Fixed** (independently, different design) | `clickhouse_lease.py` quarantines a contested head through its latest expiry, and the fence now requires `expires_at_ns > now + publish_timeout_ns` |
| M3 conflict handler masks the conflict | **Fixed** | `catalog.py` chains the commit failure |
| M4 deciding read half applied | **Fixed** | bounded reads carry it on the DATA statement; `consistent_snapshot_reads` opts out |
| P1 cross-tenant pack forgery | **Fixed** | `_descriptors` binds the footer tenant to the key prefix |
| L1 chunk overrun at long `store_id` | **Fixed** (independently) | `clickhouse_sql.py` budgets inline parameters in rendered bytes |
| L2 `select_sequential_consistency` overstated | **Fixed** | opt-in `insert_quorum` on the deciding writes |
| L3 lease wedge states | **Fixed** | early lease refusal, plus `collect_garbage()` for the rows the lease protocol appends |
| L4 live fixture `UnboundLocalError` | **Fixed** | writer constructed before the `try` |
| I1–I4 | **Fixed** | count validation, object-scoped visibility preflight and full post-DDL shape check, byte-bounded holder, summary rename |
| R1 publish verified visibility but not membership | **Fixed** (independently) | per-chunk membership verification in `publish_snapshot` |
| R3 interrupted install refused forever | **Fixed** | the missing-table refusal is gated on surviving rows |
| R5 bare `RuntimeError`s outside the taxonomy | **Fixed** | `CatalogVersionAllocationError`, `CatalogRebuildExhaustedError` |
| R6 `drop_schema` aborts on a wrong-kind catalog | **Fixed** | `DROP TABLE IF EXISTS` for every object |

**Nothing from this review is open.** The four items that needed a decision
rather than a patch were resolved as follows: bounded reads are consistent by
default with an operator opt-out; the write half of replication consistency is
opt-in config, because quorum size is a deployment choice; retention is an
explicit maintenance call rather than a TTL, so a mutation never lands inside a
publish; and the tenant binding is enforced where the footer and the key first
meet, as a bound on damage rather than a replacement for object-store
authorization.

Two things came out of *verifying* those fixes rather than writing them, and
both are recorded in `docs/capture-storage-design.md`:
`insert_quorum_timeout` is bounded to `publish_timeout_ns`, because ClickHouse's
600-second default would park a version or lease claim far past the lease TTL
when a replica is unreachable; and `insert_quorum` turns out to be a
for-the-life-of-the-catalog setting, since a non-quorum insert into a table that
has taken a quorum insert is invisible to a `select_sequential_consistency`
read. `tests/tools/verify_replicated_quorum.py` reproduces both against two
replicas.

---

## 1. Executive Summary

| Severity | Count | Of which new in this PR |
|----------|-------|-------------------------|
| 🔴 CRITICAL | 0 | 0 |
| 🟠 HIGH | 0 | 0 |
| 🟡 MEDIUM | 5 | 4 (one is pre-existing and *improved* by the PR) |
| 🟢 LOW | 5 | 4 |
| ⚪ INFO | 8 | 4 |

**Overall risk:** MEDIUM
**Recommendation:** **CONDITIONAL APPROVE** — merge after the four new MEDIUM items (§3 M1–M4) and the three "HIGH by rule" test gaps (§4) are addressed. Nothing found is exploitable by a party without catalog write credentials, and no validation was removed relative to `main`.

**Key metrics**
- Production files analyzed: 6/6 (100%); test/doc/benchmark files: triaged LOW, sampled.
- Changed executable units: 48 → 36 covered by CPU tests (75%), 11 partially, 1 not at all (`drop_schema`).
- High-blast-radius changes: `search` (78 refs), `ensure_schema` (97), `write_descriptors` (65), `publish_snapshot` (46), `allocate_version` (44), `acquire_publisher_lease` (41), `index()` (40).
- Security regressions: **0**. Every removed guard traces to the baseline squash `08ea071` (#113) and was relocated or strengthened, never dropped.
- SQL injection: **refuted at every site** (hostile values rendered through clickhouse-driver 0.2.11 `escape_param`; `INSERT … VALUES` paths use native blocks, not text).
- The most significant *security* issue in the module — an object-store writer forging another tenant's descriptors — is **pre-existing** and the PR makes it strictly less bad (§6).

**What the PR gets right (verified):** the sort key now carries pack identity so a merge cannot delete one of two packs' rows; membership is a manifest row *paired* with a watermark row on `(index_version, publish_id)`, so orphaned/losing writes are inert; the watermark write carries a server-side barrier and lease fence in one statement; the public view is bounded by the same predicate; the inventory (replay guard) is written only after visibility; timestamps that decide lease liveness are all server-side; large inlined lists are chunked.

---

## 2. What Changed

**Commits:** 27 (2026-08-30 → 2026-09-01) · **Total:** +8,026 / −536 across 30 files
**PR:** https://github.com/ProjectDMX/DMI/pull/119

| File | + | − | Risk | Blast radius | Why |
|------|---|---|------|--------------|-----|
| `src/dmi/storage/capture/clickhouse_catalog.py` | 1,420 | ~100 | **HIGH** | CRITICAL | Publisher lease (write authorization), fenced publish, barrier, schema-version refusal, SQL text with inlined values |
| `src/dmi/storage/capture/clickhouse_reader.py` | 357 | ~90 | **HIGH** | CRITICAL | Tenant-scoped reads, cursor bound, deciding reads, chunked `get_by_ids` |
| `src/dmi/storage/capture/catalog.py` | 286 | ~80 | **HIGH** | HIGH | Publish retry loop, error taxonomy, conflict handler, allocator guard |
| `src/dmi/storage/capture/summary.py` | 80 | ~20 | MEDIUM | MEDIUM | `CORE_SUMMARY_VERSION` 1→2, exact integer order statistics |
| `src/dmi/storage/capture/model.py` | 21 | 0 | LOW | — | Docstrings only |
| `src/dmi/storage/capture/__init__.py` | 19 | 2 | LOW | — | Re-exports |
| `tests/**` (16 files), `tests/_catalog_fakes.py` | ~5,300 | ~230 | LOW | — | New CPU fakes and live suites |
| `docs/**`, `benchmarks/**`, `.gitignore` | ~1,240 | ~30 | LOW | — | Design docs, teardown via `drop_schema` |

Baseline codebase: 44 Python files under `src/` → MEDIUM → FOCUSED strategy (full depth on HIGH files, 1-hop dependencies `cursor.py`, `reader.py`, `s3.py`, `pack.py`, `pipeline.py` read as needed).

---

## 3. Findings introduced or left open by this PR

### 🟡 M1 — A stale pre-PR indexer sharing the prefix makes packs invisible-and-skipped forever, and the new schema guard does not notice
**File:** `clickhouse_catalog.py:618-697` (`_verify_schema_compatibility`), `:998-1018` (`committed_pack_ids`), `:276-297` (`_membership_predicate`) · **Commits:** `9c2bcfc`, `c96ebac`, `bea3ed8` · **Blast radius:** every pack the stale writer touches · **Test coverage:** NO · **Baseline:** NEW (migration hazard created by the schema change)

**Description.** The pre-PR build's `ensure_schema` is all `CREATE … IF NOT EXISTS`; against a version-4 catalog it no-ops on the shared tables and quietly recreates `{prefix}_pack_commit_log`. Its `index()` then: allocates from the shared claims table, writes descriptors (same columns), writes the **inventory**, and inserts an **unconditional** watermark row with `publish_id` at the column default. No manifest row is ever written. In the v4 world: `_membership_predicate` never admits that pack (no paired manifest row) → invisible to every reader and the public view; `committed_pack_ids` sees it in the inventory → skipped by every later pass including `rebuild()`; the stale watermark row raises `max(index_version)`, so the v4 indexer's barrier refuses and it burns up to `max_publish_attempts` descriptor rewrites per call. On the next v4 `ensure_schema()`, the stamp is 4, no table is missing, and `_inventory_without_membership()` is false as soon as *any* pack has membership; the surviving legacy `pack_commit_log` is examined only on the leftovers/unstamped paths (`:764`, `:828-857`) — never when stamped.

**Scenario.** Rolling deploy with two indexer replicas; the old one keeps a schedule for a few hours. Every pack it lists first is durably stored, reported as indexed, and unreachable. Recovery is the full drop-and-rebuild in `docs/capture-storage-design.md`.

**Recommendation (cheap, makes the state loud):**
```python
# in _verify_schema_compatibility, after the recorded-version check
leftovers = [name for _, name in self._legacy_objects if name in found]
if leftovers:
    raise CatalogSchemaVersionError(
        f"catalog ... {stamp} but `{'`, `'.join(leftovers)}` exist(s) beside it: "
        "a writer from an earlier build has run against this catalog. Its packs "
        "are in the inventory without membership. " + self._rebuild_instruction())
```
and, as a data-level detector, refuse when `{prefix}_index_watermark` holds a row whose `publish_id` is the zero UUID. Neither stops the stale writer; both stop the silent state. Document that the prefix must not be shared across builds.

### 🟡 M2 — A contested head term is treated as free even when one claimant genuinely holds it, voiding the TTL safety margin
**File:** `clickhouse_catalog.py:1370-1374`, comment `:1418-1419`, docstrings `:44-49`, `:1073-1084`, `:1252-1254`, `:1506-1510`; `docs/catalog-descriptor-key.md:377-381,466-475` · **Commits:** `c007844`, `34366e7` · **Blast radius:** every lease operation and every publish (transitively CRITICAL) · **Test coverage:** PARTIAL — `test_a_contested_lease_term_is_abandoned_by_everyone_who_sees_it` and the two live contested-head tests assert the premise "both claimants see two rows", which the protocol does not guarantee · **Baseline:** NEW code; strictly narrower than `main` (which had no lease), but the documented bound is overstated. Found independently by two analysts.

**Description.** `_claim_lease` raises `PublisherLeaseHeldError` only when `head.claimants == 1 and live and foreign`. The sole-claimant protocol guarantees *at most one* claimant reads back a singleton — not that both see the contest. Schedule (preconditions: the previous lease has just expired, or the table is empty — the normal state at the start of every run when the indexing period exceeds `lease_ttl_ns`):
1. A reads head, computes term T, inserts `(T, id_A)`.
2. B reads head before A's row is visible, computes T, inserts `(T, id_B)`.
3. A reads back T with `select_sequential_consistency`, sees only `id_A` → **granted**; A proceeds into `allocate_version`/`write_descriptors`/`publish_snapshot`.
4. B reads back T, sees `{id_A, id_B}` → "contested" → re-reads head: `claimants == 2` → the guard is skipped → B claims T+1 → **granted**. Elapsed since A's grant: three round trips (~5 ms), not `lease_ttl_ns`.
5. Any third writer C arriving while T is contested-but-held also overtakes A at once.

Safety of the snapshot now rests on the fence alone: for both A's and B's watermarks to land, A's `INSERT … SELECT` must evaluate its fence subquery before B's row lands and commit after B's watermark does — a stall of ~10–20 ms inside one INSERT rather than the 25 s margin (`lease_ttl_ns − publish_timeout_ns`) the config promises. The lower version then becomes visible underneath a watermark a reader already pinned — exactly the mid-batch gap the PR set out to close. Exploitability: HARD (needs a second publisher, a ~1 ms coincidence, and a server stall); impact: pinned snapshot silently grows; no data loss; not `SnapshotPublishConflictError` (different versions).

**Recommendation.** The fence honors row[0] alone, so `_claim_lease` should reason about row[0] alone:
```python
if head.expires_at_ns > head.now_ns and head.lease_id != lease_id:
    self._lease = None
    raise PublisherLeaseHeldError(...)
```
Cost: an abandoned contested head blocks the lower-sorted contender for one TTL (the higher-sorted one is row[0] and may still claim). Update the four docstrings and two live-test docstrings that state the stronger claim; add a fake-driven test in which the rival row lands between INSERT and read-back for only one side. Optionally, re-read `_lease_head()` after the read-back and require `term == term and claimants == 1 and lease_id == mine` before returning a grant.

### 🟡 M3 — A `commit_packs` failure inside the conflict handler masks the conflict and re-arms the retry it forbids
**File:** `catalog.py:494-499` · **Commit:** `2cccb1f` · **Blast radius:** `index()` (40 refs) · **Test coverage:** NO · **Baseline:** NEW

```python
except SnapshotPublishConflictError:
    if refs:
        self._writer.commit_packs(refs, index_version=version)
    raise
```
If the inventory INSERT raises (transport error, Code 159 timeout), the bare `raise` is never reached; the driver exception propagates with the conflict demoted to `__context__`. No `except CaptureStorageError` supervisor sees the must-not-retry anomaly, and the packs are *visible but not in the inventory*, so the next pass re-indexes and re-publishes the batch at a higher version — the retry `SnapshotPublishConflictError`'s contract (`catalog.py:35-50`) exists to forbid, while the foreign-writer anomaly is buried. Reader-visible corruption: none (rows are byte-identical and collapse).

**Recommendation:**
```python
except SnapshotPublishConflictError as conflict:
    if refs:
        try:
            self._writer.commit_packs(refs, index_version=version)
        except Exception as commit_failure:
            raise conflict from commit_failure
    raise
```
plus a test that fails `commit_packs` inside the handler and asserts the conflict type surfaces.

### 🟡 M4 — The deciding-read remedy stops at the validation; the reads that follow can still hit a lagging replica
**Files:** `clickhouse_reader.py:301` (search data query), `:374` (`get_by_ids` chunks); `clickhouse_catalog.py:1010-1014` (`committed_pack_ids`), `:944-949` (`_inventory_without_membership`) · **Commits:** `a808812`, `2cccb1f` · **Blast radius:** `search` 78, `get_by_ids` 39 (CRITICAL/HIGH) · **Test coverage:** PARTIAL (head read only: `test_a_cursor_at_a_published_watermark_survives_replica_lag`) · **Baseline:** partly pre-existing (the no-cursor `search` path had the same exposure on `main`)

**Description.** The head is read with `select_sequential_consistency` so a genuinely published watermark is not falsely rejected — but the statement that then reads descriptors/membership runs on plain settings. On a replica holding the watermark row for V but not yet V's manifest/descriptor rows, `get_by_ids(watermark=V)` returns a **short** result and `search()` returns a page pinned at V missing rows in its key range; the issued cursor carries V, so the next page resumes *past* the missing keys — captures silently skipped in a walk. Lag can only omit, never add, so no unpublished or cross-tenant data is exposed. On the writer side, `committed_pack_ids` is the sole input to the no-op skip and to "which packs get re-indexed" (lag → redundant re-publish), and `_inventory_without_membership` can falsely refuse a healthy young catalog (fail-closed).

**Recommendation.** Either carry `_DECIDING_READ` on the data statement of every *bounded* read (quorum wait per query), or two-phase — plain read, escalate to deciding only when the head read says the requested watermark is at the edge (e.g. `requested == head` and the page is short). Add `settings=_DECIDING_READ` to the two writer reads regardless; add a lagging-client test for the data query, not just the head.

### 🟢 L1 — `_MAX_INLINE_TUPLES = 1_000` overruns `max_query_size` at the `store_id` length the code permits
**File:** `clickhouse_catalog.py:259-273`; call sites `:1009`, `:1127` · **Commits:** `2cccb1f`, `76c5c5c` · **Test coverage:** NO (reader has boundary tests; writer has none; none use a long `store_id`) · **Baseline:** NEW

`store_id` is operator config bounded to **255 bytes** (`s3.py:31-37,170`). A rendered member `('<store_id>', '<uuid>')` is `len(store_id)+36+~8` bytes, doubled for escaped characters. At 255 plain bytes ≈ 300 KB per chunk > 262,144. The comment's "an order of magnitude under the limit even with generous store ids" holds for short ids only. Reachability: `committed_pack_ids` accepts up to `query_pack_limit = 10,000` identities; the manifest path needs a publish batch > 1,000 packs (default `max_packs = 64`). Breach = Code 62 mid-publish, deterministic for that catalog (availability, not attacker-triggerable). **Fix:** size chunks by rendered bytes, or cap at 200 like the reader, and add a live test with a 255-byte `store_id`.

### 🟢 L2 — `select_sequential_consistency` is necessary but not sufficient on `ReplicatedMergeTree` without quorum inserts
**File:** `clickhouse_catalog.py:232-257` (comment), all `_DECIDING_READ` sites · **Baseline:** NEW comment · **Needs live confirmation on the target ClickHouse version**

ClickHouse documents that the setting constrains SELECTs only relative to INSERTs executed with `insert_quorum`, and that it does not work with `insert_quorum_parallel` (the default). None of the protocol writes set `insert_quorum`, so on a replicated deployment two claimants on different replicas can still both read back alone. The module already disclaims the write half (`docs/catalog-descriptor-key.md:501-504`); this is an assurance/documentation gap, not a new hole. **Fix:** a config flag that sets `insert_quorum`/`insert_quorum_parallel=0` on the four protocol tables' inserts, or reword the comment to "necessary but not sufficient" and name the deployment requirement.

### 🟢 L3 — New lease-induced wedge states (liveness)
**File:** `clickhouse_catalog.py:1380`, `:1294-1300`, `:1276-1281`; `catalog.py:380-385` · **Baseline:** NEW operational surface

`PublisherLeaseHeldError` sets `self._lease = None`; every later `index()` raises `PublisherLeaseError("no publisher lease is held")` until something above the writer re-acquires — nothing in `src/` does, so a supervisor that treats the error as fatal keeps the live indexer down after a single collision with a sweep. A restart without `release_publisher_lease()` waits a full TTL (identity is `lease_id`, not `holder`). A periodic `rebuild()` beside the live indexer will, on the first page with any uncommitted pack, either fail itself or take the lease and knock the live indexer out. **Fix:** document the supervisor contract (re-acquire on `PublisherLeaseHeldError`), or let `acquire_publisher_lease` be idempotent-by-holder.

### 🟢 L4 — Test fixture: `writer` assigned inside `try`, referenced in `finally`
**File:** `tests/test_clickhouse_facets_live.py:48` · **Commit:** `76c5c5c` · A `ClickHouseCatalogWriter(...)` construction failure dies with `UnboundLocalError`, masking the real error. Move the construction above the `try`.

### ⚪ INFO (new in PR)
- **I1** `indexed_rows`/`indexed_packs` interpolated into statement text without the `type is int` validation `index_version` gets (`clickhouse_catalog.py:1157-1159,1174-1175`). `escape_param` renders unknown types via `str()` unescaped. Only `CatalogIndexer` calls it with `len(...)` — unreachable with untrusted data. Cheap hardening: `_validate_version` both.
- **I2** Schema guard treats an empty `system.tables` answer as "fresh install" (`:618-620`, `:712-716`): a role lacking `SHOW TABLES` would let `CREATE … IF NOT EXISTS` run against an old catalog. Needs a misconfigured grant; consider asserting the stamp table is visible after `ensure_schema` writes it.
- **I3** `holder` bound is 256 *characters*; message says *bytes* (`:1276-1277`). Use `len(holder.encode()) <= 256` like `_validate_bounded_text`.
- **I4** `bool` dtype now reports `int` order statistics (0/1); intended by version 2, but the golden manifest has no `bool` entry to pin it.

---

## 4. Test Coverage Analysis

**Coverage:** 36/48 changed units fully covered by CPU tests (75%); 11 PARTIAL; 1 NO. By added-line volume ~90% executes under `pytest -m cpu`. Live suites (`*_live.py`, `manual and clickhouse`) are **not collected in CI** (`pyproject.toml:32`).

**Untested or under-pinned changes (methodology elevation applied)**

| Function | Rule → Risk | Why it matters |
|----------|-------------|----------------|
| `ClickHouseCatalogConfig.__post_init__` whole-second check (`clickhouse_catalog.py:73-84`) | NEW validation + NO test → **HIGH** | Sole reason `max_execution_time` is a whole int; a fraction truncating to 0 = *unlimited*, removing the bound that keeps fence evaluation and row landing inside the TTL. The settings test asserts `5.0`, which `== 5`, so the int type is unpinned too. |
| `CatalogIndexer._allocate_version` zero rejection (`catalog.py:410-414`) | MODIFIED validation + UNCHANGED tests → **HIGH** (by rule) | Guard went `< 0` → `< 1`; parametrization `[-1, "7", True, 4.0]` never includes `0`. |
| `CatalogIndexerConfig.max_publish_attempts` (`catalog.py:155-165`) | MODIFIED validation + UNCHANGED test list → **HIGH** (by rule) | `test_indexer_config_rejects_non_positive_fields` does not list the new field; `max_publish_attempts=0` makes `_publish`'s loop body never run. |
| `_inline_chunks` at writer sites (`:1009`, `:1127`) | NEW + NO boundary test → MEDIUM-HIGH | An off-by-one drops members from the manifest (published but invisible) or oversizes a statement. The reader has −1/0/+1 tests; the writer none. |
| `drop_schema` (`:980-996`) | NEW + NO CPU test (live only) → MEDIUM | Destructive DDL; emitted list/order asserted nowhere in CI. |
| `_membership_predicate(bounded=True)` via reader `_membership` | PARTIAL → MEDIUM | CPU tests pin only the `(store_id, pack_id) IN (SELECT … FROM` fragment; dropping `index_version <= %(watermark)s` keeps CPU green. Live-only guard. |
| `get_by_ids` deciding head read (`clickhouse_reader.py:341`) | PARTIAL → MEDIUM | Settings asserted only for the cursor-bearing `search`; removing `deciding=True` here passes CI. |
| `acquire_publisher_lease` holder `len == 256` | PARTIAL → LOW | Only 257 rejected. |

**Behaviours whose semantics are verified only by live tests (not in CI):** fence on an empty lease table returns 0 rather than throwing; takeover between the two publish statements leaves inert rows; `select_sequential_consistency` accepted server-side; merge stability of `argMax(tuple, (index_version, store_id, pack_id))`; bounded membership excludes later publishes; `drop_schema`; manifest `arrayJoin(%(members)s)` rendering.

**Fake-fidelity gaps that let a real bug pass CI**
1. **The manifest INSERT is unfenced in two of three writer fakes** (`tests/test_clickhouse_capture_catalog.py:152-166`, `tests/test_capture_faults.py:219-235` — unrecognised INSERTs `return []`). Removing `WHERE {self._lease_fence()}` from the manifest statement (`:1134`) fails nothing in those suites. Only `test_capture_version_allocation.py::_CatalogServer` fences it.
2. `FakeLeaseTable.execute` has no `settings` kwarg; wrappers drop it, so sequential consistency never affects fake behaviour and there is no writer-side lag model (the reader has `_LaggingClient`).
3. Nothing renders `%(members)s`/`%(identities)s`, so chunk size vs 256 KiB is arithmetic-only for the reader and absent for the writer.
4. Catalog `_Client` answers `_inventory_without_membership` by fragment match with a canned pair; the effective-membership subquery text is unverified.
5. Fakes cannot interleave statements (documented), so takeover-in-the-gap is live-only.
6. `fence_admits` is a regex shape heuristic, not a parser (documented; acceptable).

---

## 5. Blast Radius Analysis

| Symbol | References (src+tests+bench) | Class | Note |
|--------|------------------------------|-------|------|
| `ensure_schema` | 97 | CRITICAL | now runs the schema guard (M1, I2) |
| `search` | 78 | CRITICAL | deciding head + plain data (M4) |
| `write_descriptors` | 65 | CRITICAL | body unchanged |
| `publish_snapshot` (new) | 46 | HIGH | replaces `publish_watermark` (0 remaining) |
| `allocate_version` | 44 | HIGH | deciding reads |
| `acquire_publisher_lease` (new) | 41 | HIGH | M2, L3 |
| `CatalogIndexer.index` | 40 | HIGH | retry loop, no-op skip, M3 |
| `get_by_ids` | 39 | HIGH | chunked, M4 |
| `current_watermark` | 33 | HIGH | |
| `commit_packs` | 30 | HIGH | membership write removed |
| `committed_pack_ids` | 27 | HIGH | chunked, M4, L1 |
| `release_publisher_lease` (new) | 21 | HIGH | fenced tombstone |
| `summarize_tensor` | 19 | MEDIUM | version 2 |
| `_lease_fence` | 13 | MEDIUM | pinned by fakes |
| `_claim_lease` / `_lease_head` / `renew_publisher_lease` | 5 / 4 / 4 | LOW direct, CRITICAL transitive | every lease op and publish passes through (M2) |
| `drop_schema` (new) | 9 | MEDIUM | live-only |
| `_membership_predicate`, `_inline_chunks`, `_verify_schema_compatibility`, `_inventory_without_membership`, `_published_head`, `_projection`, `_descriptor` | 2–5 | LOW | internal, no direct tests of the helpers |

Priority: M1/M2 are HIGH-risk changes on CRITICAL-radius paths → **P0/P1**. M3/M4 HIGH-risk on HIGH radius → **P1**.

---

## 6. Historical Context and Pre-existing Issues

**Security-related removals:** none. Every removed guard originates in the single baseline squash `08ea071` (2026-08-30, "Capture storage: immutable packs … (#113)") and was relocated:

| Removed line (baseline) | Where it lives now |
|---|---|
| `query_pack_limit must be positive`, `allocation_attempts must be positive` | one `__post_init__` loop over five fields, `clickhouse_catalog.py:57-66` (stricter: `type is int`) |
| `allocate_version must return a non-negative integer` (`< 0`) | `< 1`, `catalog.py:410-414` (stricter) |
| `if requested > int(self.current_watermark())` | `if requested > self._published_head(deciding=True)`, `clickhouse_reader.py:341` (stricter) |
| `if len(row) != len(_PROJECTION)` | `_ROW_WIDTH` check plus tuple-shape check, `clickhouse_reader.py:550-563` (stricter) |
| `ClickHouse returned an invalid pack identity` | `_text` kept, message generalized (`:1672`) |
| unconditional `publish_watermark` `INSERT … VALUES` | fenced, barriered `INSERT … SELECT` (`:1153-1179`) |
| unbounded `CREATE VIEW IF NOT EXISTS {capture_view}` | bounded `CREATE OR REPLACE VIEW` (`:568-572`) |

**Regression check:** `git log -S` on each removed and added guard finds no earlier remove/re-add cycle.

**Pre-existing issues surfaced by the review (not introduced by the PR; recorded because the PR touches the paths):**

- 🟡 **P1 — An object-store writer can forge descriptors into another tenant and supersede a victim's capture pointer.** Anyone with bucket PUT writes a well-formed pack whose footer says `tenant_id="victim", capture_id=C`; nothing compares the footer `tenant_id` to the `v1/tenant=<id>/…` key prefix (`pipeline.py:301` writes it; no reader checks it), and `PackIndex.from_store` (`pack.py:309-339`) checks only pack id/CRC/format. The reader's `argMax` over `(index_version, store_id, pack_id)` then resolves C to the attacker's pack for any fresh watermark. **On `main` this was worse:** without pack identity in the sort key a background merge deleted the victim's row outright. The PR keeps both rows and makes the winner deterministic; pinned older selections still resolve to the original. **Fix belongs outside this PR:** bind footer `tenant_id` to the key prefix at `inspect`/`from_store`, or use per-tenant buckets. This is the most significant security finding in the module.
- ⚪ **P2** `tenant_id` is optional on `CaptureQuery`; a tenant-less `search` spans tenants (`model.py:277`, `clickhouse_reader.py:510-514`). `get_by_ids` correctly requires one. Policy question for the service layer.
- ⚪ **P3** Cursor is unsigned (`cursor.py:90-150`); a forged cursor can only reach an older *published* snapshot of the caller's own filters (`filter_hash` binds filters incl. tenant; watermark bounded by head). Harmless.
- ⚪ **P4** `get_by_ids` does not type-check `capture_ids` elements; `escape_param` renders a non-`str` via `str()` unescaped. Inside the Python trust boundary; defense-in-depth only (`clickhouse_reader.py:314-325`).
- ⚪ **P5** Two hostile packs adjacent in a listing page whose estimated descriptor bytes exceed 128 MiB abort that page on every run (`catalog.py:348-353`); attacker controls adjacency via key names. Identical on `main`.

**Fixed by this PR relative to `main` (verified):** inventory written before visibility (crash → skipped-forever pack); unconditional watermark (broken allocator could publish under a pinned watermark); mid-batch membership gap with concurrent indexers (narrowed to the in-flight residual in M2); public view serving unpublished/orphaned rows; unchunked `get_by_ids`/`committed_pack_ids` Code 62 DoS; ReplacingMergeTree deleting one of two packs' rows for one capture; blind `release` revoking a successor (introduced and fixed within the PR).

---

## 7. Recommendations

### Immediate (before merge)
- [ ] **M1** Refuse in `_verify_schema_compatibility` when a legacy object exists beside a version-4 stamp; optionally detect zero `publish_id` watermark rows. Document "one build per prefix".
- [ ] **M2** Make `_claim_lease` refuse whenever row[0] is live and foreign regardless of claimant count; fix the six docstrings that state the TTL bound; add the singleton-winner schedule as a fake-driven test.
- [ ] **M3** Wrap `commit_packs` in the conflict handler and re-raise the conflict chaining the commit failure; add a test.
- [ ] **M4** Add `_DECIDING_READ` to `committed_pack_ids` and `_inventory_without_membership`; decide (and document) the reader data-query policy.
- [ ] Tests: whole-second `publish_timeout_ns` rejection and `isinstance(max_execution_time, int)`; allocator returning `0`; `max_publish_attempts` in the non-positive parametrization; fence the manifest INSERT in the catalog and faults fakes.
- [ ] **L4** Move `writer = ClickHouseCatalogWriter(...)` above the `try` in `test_clickhouse_facets_live.py`.

### Before production
- [ ] **L1** Size inline chunks by rendered bytes (or cap at 200) and live-test a 255-byte `store_id`.
- [ ] **L2** Confirm `select_sequential_consistency` semantics on the target ClickHouse version; add an `insert_quorum` option or reword the guarantee.
- [ ] **L3** Define the supervisor contract for `PublisherLeaseHeldError`/`PublisherLeaseError`; consider holder-idempotent re-acquire.
- [ ] Writer-side chunk-boundary tests (−1/0/+1 at 1,000) for both call sites; CPU pin of `index_version <= %(watermark)s` in both halves of the bounded membership predicate; settings assertion for `get_by_ids`'s head read; a CPU test of `drop_schema`'s emitted statements.
- [ ] **P1** (outside this PR, highest security value): bind footer `tenant_id` to the object key prefix at ingest.

### Technical debt
- [ ] I1–I3 hardening; I4 add a `bool` entry to the golden manifest.
- [ ] `get_by_ids` should reuse `_inline_chunks` with a size parameter; `_allocate_version`'s non-monotonic check should raise a `CaptureStorageError` subclass, not bare `RuntimeError`; remove dead `_ALL_OBJECTS` in `test_clickhouse_schema_migration_live.py`.
- [ ] `.gitignore`: `uv.lock` policy and `.code-improver/`.

---

## 8. Analysis Methodology

**Strategy:** FOCUSED (44-file `src/`, 30 changed files, 6 production modules).

**Scope:** 6/6 production files read in full at head and baseline; 1-hop dependencies (`cursor.py`, `reader.py`, `s3.py`, `pack.py`, `pipeline.py`) read where a finding depended on them; tests sampled for coverage mapping (all 17 changed test files grepped, fakes read in full); docs read for the stated trust model.

**Techniques:** git blame / `git log -S` on every removed guard; per-region BEFORE/AFTER/CHANGE/SECURITY analysis; invariant tables for the lease and membership protocols; blast radius by reference counting across `src/`, `tests/`, `benchmarks/`; test-coverage mapping of 48 changed units with CPU-vs-live classification and fake-fidelity audit; adversarial modeling against four attacker models (object-store writer, second publisher, reader/API caller, infrastructure faults); clickhouse-driver 0.2.11 `escape_param`/`process_insert_query` read in the venv and hostile values rendered offline (`inj.py`, `size.py` in the job temp dir).

**Limitations:** no ClickHouse server was driven during this review (a prior live run of the suites passed: 57 passed / 3 S3 skips); replica-lag and `select_sequential_consistency` claims are reasoned from documentation and settings, not reproduced; `max_execution_time` kill granularity for a stalled MergeTree write not verified empirically; multi-replica `insert_quorum` behaviour explicitly out of the module's scope.

**Confidence:** HIGH for the injection refutations, the invariant table, and M1–M3 (code-traced schedules); MEDIUM for M4/L2 (deployment-dependent); the pre-existing tenant-forgery issue (P1) is HIGH-confidence but out of the PR's scope.

---

## 9. Appendix — Commit reference

| SHA | Subject |
|-----|---------|
| `0728c37` / `a0749bf` | Propose / add pack identity to the descriptor sort key |
| `bea3ed8` | Publish snapshot membership behind a conditional watermark write |
| `296c681` | Report integer order statistics exactly instead of rounded |
| `d50c778` | Bound the public capture view to the published snapshot |
| `e93a2c8` | Resolve every descriptor column from the same row |
| `9c2bcfc` / `c96ebac` | Refuse to run against a catalog this build cannot read / diagnose a schema refusal |
| `25fac88` | Verify a publish owns its version rather than that one exists |
| `c007844` / `34366e7` | Fence publication on a durable publisher lease / make it recoverable |
| `cf58876` | Stop an empty schema stamp from waiving the rest of the guard |
| `0c0da97` | Make the catalog fakes enforce the fence they claim to enforce |
| `2cccb1f` | Fix 15 review findings in the capture catalog publish protocol |
| `4c06ec3` | Adapt lease-takeover live tests to whole-second publish timeouts |
| `76c5c5c` | Simplify the chunked-inline and deciding-read plumbing |
| `aa7ce77` | Close three test-suite blind spots around the lease and fence |
| `c8875a8` | Address the three Copilot review comments |
| `a808812` | Apply the deciding-read remedy everywhere it decides, chunk get_by_ids |

Definitions: **deciding read** — a SELECT whose answer changes a write decision, run with `select_sequential_consistency=1`; **fence** — the `WHERE` conjunct on every visibility-making INSERT requiring the lease head row to be this writer's and unexpired by the server clock; **barrier** — `ifNull((SELECT max(index_version) FROM watermark),0) < %(index_version)s`; **membership** — a manifest row whose `(index_version, publish_id)` also appears in the watermark table.
