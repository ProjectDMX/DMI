# Polish run — C++ native path

Scope: `native/csrc/` (the C++ native path), as chosen by the human.
Landing rule, also from the human: each fix lands on the PR that OWNS the
file, then the stack is forward-merged so all three PRs carry it.

## Baseline refs (Phase 0)

Tree clean at start. One baseline per branch, because the scope spans three
stacked PRs rather than one:

| PR | Branch | Baseline ref |
|---|---|---|
| 127 | `feat/native-capture-pipeline` | `92c2225` |
| 128 | `feat/native-catalog-b1` | `7eac029` |
| 129 | `feat/native-c-reader` | `a4d3dcb` |

`a4d3dcb` contains both others (verified with `git merge-base --is-ancestor`),
so review runs once against C and sees the whole native path.

## File ownership (routes each fix to its PR)

| PR | Owns under `native/csrc/` |
|---|---|
| 127 | `pack/`, `sink/`, `store/`, `common/`, `ring/`, root — 69 files |
| 128 | `catalog/{catalog_writer,clickhouse_client,conformance_catalog,indexer,lease_coordinator,pack_index,schema,version_allocator}.{cpp,h}` — 15 files |
| 129 | `catalog/{hydration,reader}.{cpp,h}` — 4 files |

Nuance: `conformance_catalog.cpp` is owned by 128 but carries ops added on
129 (reader/hydration). A defect in one of those later ops routes to 129,
not to the file's owner — ownership is per defect, not per filename.

## Baseline numbers (Phase 1), measured at `a4d3dcb`

Measured in a dedicated detached worktree (`/tmp/opencode/polish-c`), NOT in
the shared one: another session edits `/tmp/opencode/native-wt` roughly every
20 minutes, and its in-flight edits have already contaminated one measurement
this session (3 phantom test failures traced to uncommitted `pack_builder.cpp`
work).

| Signal | Baseline |
|---|---|
| Build errors (`make -C native cpu-goals`, from empty `build/`) | **0** |
| Build warnings, same build | **0** |
| cpu suite (`pytest -m cpu`) | **1203 passed**, 0 failed, 0 skipped |
| live suites (`clickhouse and manual and not garage`) | **128 passed, 1 failed** |
| Lint | not measured — no `.clang-tidy`/`.clang-format`, no ruff/flake8 in `pyproject.toml`. The compiler's `-Wall -Wextra` IS this project's C++ lint signal, counted above. |
| Types | not measured — no mypy/ty configured. Adding tooling is out of scope. |
| Coverage | not measured — no coverage tooling configured. Out of scope. |

The one live failure is pre-existing and environmental, not a finding:
`test_a_role_that_cannot_see_one_object_is_told_to_grant_it_not_to_rebuild`
needs `CREATE USER` on the server, and the local standalone ClickHouse has no
access management. The same test passes in CI. It is the baseline, so a final
count of "128 passed / 1 failed" is unchanged, not a regression.

Warning baseline is genuinely zero rather than filtered: the whole build log
is 55 lines with no `warning:` matches. Earlier in the session this build
carried four torch-header warnings; those disappeared when the torch
extension moved to C++20.

## Rounds

### Round 1 — verdicts

Four lenses reported 40 findings (see `polish-seen.md`). Verifier verdicts as
they land. A verifier may CONFIRM the mechanism and still downgrade impact to
`optional`; those do NOT enter the fix queue — they go to the human list. That
distinction is the whole point of the verify step, so it is recorded per row.

| Finding | Verdict | Impact | Queued? | Owner |
|---|---|---|---|---|
| hydration.cpp:586 squared element budget | CONFIRMED | correctness | YES | 129 |
| hydration.cpp:692/:646 int64 abs_max sign + >2^53 cast | CONFIRMED | correctness | YES | 129 |
| schema.cpp:127 rebuild_instruction omits legacy_objects_ | CONFIRMED | requirement | YES | 128 |
| reader.cpp:396 missing CaptureQuery bounds | CONFIRMED (3 of 4 sub-claims) | requirement | YES | 129 |
| schema.cpp:385 dead legacy-refusal loop | CONFIRMED | optional (downgraded from requirement) | no — human list | 128 |
| s3_sign.cpp:75 header ordering | CONFIRMED (downgraded) | optional (downgraded from requirement) | no — human list | 127 |
| pack_builder.cpp:353 ParseUuid discarded | CONFIRMED mechanism, 2 of 3 sub-claims REFUTED | optional (downgraded from correctness/requirement), sev -> low | no — human list | 127 |
| spool.cpp:190 `a..b` rejected → sink latches | CONFIRMED | correctness | YES | 127 |
| spool.cpp:190 symlink escape → silent permanent loss | CONFIRMED | correctness | YES | 127 |
| catalog_writer.cpp:21 + clickhouse_client.cpp:26/111 escaping + NUL truncation | CONFIRMED (all 3 parts) | requirement | YES | 128 |
| pack_index.cpp:179 + indexer.cpp:36 unescaped pack_id | running | | | 128 |

Two verifiers usefully corrected their reviewer rather than rubber-stamping:

- The SigV4 reviewer's own example (`Content-MD5`) does NOT reproduce — `'C'`
  sorts before every lowercase letter either way. The verifier had to find
  `X-Amz-Meta-Zed` to break it, then showed every production path already
  lowercases before signing, so the hazard is reachable only through the
  conformance driver. Mechanism real, blast radius nil → `optional`.
- The query-bounds reviewer claimed an unbounded `IN` list was dangerous. The
  verifier disproved that specific consequence (the server refuses with
  `Code: 62 Max query size exceeded` — fails closed, and `sql_quote` escapes
  each name), while confirming the three real divergences. Queued for the
  bounds, not for the imagined injection.

### Round 1 — fix queue and dispatch

Ordered high severity first, then correctness before requirement. Dispatch is
grouped BY WORKTREE, not one builder per finding: three builders running
concurrently in one worktree would race on the shared `native/build/`, and two
builders editing one function would conflict. Commit-per-fix is preserved
inside each builder, which is the invariant that actually matters for
traceability.

| # | Finding | Sev/Impact | PR | Worktree | Builder |
|---|---|---|---|---|---|
| 1 | hydration.cpp:586 squared budget | high/corr | 129 | native-wt | dispatched |
| 2 | hydration.cpp:692 int64 order stats | high/corr | 129 | native-wt | dispatched (same builder, 2nd commit) |
| 3 | spool.cpp:190 `a..b` latch | med/corr | 127 | polish-a | dispatched |
| 4 | spool.cpp:190 symlink escape | med/corr | 127 | polish-a | dispatched (same builder, 2nd commit) |
| 5 | pack_id SQL injection | med/corr | 128 | phase-a-ci | dispatched |
| 6 | escaper 4-of-10 + NUL truncation | med/req | 128 | phase-a-ci | dispatched (2nd commit) |
| 7 | schema.cpp:127 rebuild instruction | med/req | 128 | phase-a-ci | dispatched (3rd commit) |
| 8 | reader.cpp:396 query bounds | med/req | 129 | native-wt | HELD — waits for the hydration builder to release the worktree |

Findings added to the queue by later verdicts (all held behind the builders
already in flight, since they land in worktrees currently occupied):

| # | Finding | Sev/Impact | PR | State |
|---|---|---|---|---|
| 9 | indexer.cpp:105 conflict emission per ref + :107 message text | low/corr + req | 128 | HELD |
| 10 | schema.cpp:162 visibility probe omits the legacy object | med/corr | 128 | HELD |
| 11 | reader.cpp:475 cursor captured_at_ns unvalidated | med/corr | 129 | HELD |

### PR 127 — COMPLETE and independently verified

Both 127 findings landed: `cdc5286` (component-wise key check) and `0db751d`
(weakly_canonical containment). cpu on that branch went 1198 -> 1200 passed,
0 failed, 0 skipped; +2 is exactly the two new tests and no pre-existing test
was modified.

I did not take the builder's red-then-green claim on trust. I reverted ONLY
`native/csrc/store/spool.cpp` to `92c2225`, rebuilt both drivers, and re-ran
the new tests:

```
FAILED tests/test_native_spool.py::test_stage_rejects_a_key_escaping_through_a_symlinked_directory
FAILED tests/test_native_pack_sink.py::test_dotted_tenant_stages_and_does_not_latch_the_sink
2 failed, 1 passed, 20 deselected
```

Then restored the fix and confirmed 3 passed. So both tests are genuinely red
at the baseline and green after — the fixes are load-bearing, not decorative.
The third test in that selection (`test_object_key_parity_with_python`) passes
in BOTH states, which is the point: it only ever compared the key string and
never reached the spool guard, which is why the defect was invisible to it.

One judgement call the builder made and flagged: the component-walk refusal
now reports "object key is invalid" (Python's `validate_object_key` wording)
while "object key escapes the spool root" is reserved for the containment
check, matching the filesystem.py / spool.py split. No test asserted on the
old native message.

### PR 129 — first two findings landed, independently verified

`3d59cea` (element budget) and `2626b46` (integer order statistics). Verified
the same way as 127, but from a fresh detached worktree at the commit rather
than the builder's own scratch copy: built all five drivers, ran the three
affected parity tests green, then reverted ONLY `hydration.cpp` to `a4d3dcb`,
rebuilt, and got

```
FAILED test_summary_parity_on_int64_beyond_the_double_mantissa
FAILED test_the_summary_element_budget_counts_each_element_once
2 failed, 1 passed
```

then restored and confirmed 3 passed. Both new tests are load-bearing.

Honest note on the third: `test_summary_core_stats_parity` passes in BOTH
states. The builder widened it to assert `abs_max_int`, which it had silently
omitted, but its corpus is uint8 where `abs_max_int` is already at parity — so
that extension is a widened gate, not a proof. The two new tests carry the
proof.

Two decisions the builder made that I checked rather than accepted:

- It REFUSES `uint64` in the new integer decode rather than representing it.
  Verified independently: `uint64` is absent from `_DTYPE_BYTES` in
  `src/dmi/storage/capture/model.py` and `kDtypes` in `pack_builder.cpp` has
  14 entries without it, so there is no oracle to match and failing closed is
  right. (The recent `a5e15a2 feat(capture): fp8 and the wide integers are
  first-class dtypes` does NOT add uint64.)
- `abs_max_int` stays `int64_t` and SATURATES at `INT64_MAX` for the single
  unrepresentable magnitude, documented in code and pinned by the test as
  `min(core.abs_max, 2**63-1)`. Widening the wire field is a contract change
  and stays on the human list. Same residual applies to `minimum`/`maximum`,
  which are `double` on the wire and so remain lossy above 2^53 even though
  `minimum_int`/`maximum_int` are now exact.

### Concurrent-session hazard, and how the 129 work was isolated

`/tmp/opencode/native-wt` is NOT clean: another session holds uncommitted
edits to `hydration.cpp`, `clickhouse_client.cpp`, `pack_index.cpp`,
`bindings_sink.cpp`, `native_pack_sink.cpp`, `spool.cpp`, `uploader.cpp`,
`native_sink.py` and `summary.py`, and that tree passed through a
non-compiling intermediate state during this run.

The builder therefore verified in an isolated copy and committed only its own
hunks via `git hash-object -w` + `git update-index --cacheinfo`, leaving the
other session's working-tree files untouched. I confirmed both halves:
`git show --stat` on each commit lists only `hydration.cpp` and the parity
test, and `git status --porcelain` still shows all nine of their files as
modified-uncommitted. Their work is intact.

Consequence for measurement: suite counts taken in the shared worktree would
measure their unfinished edits, not ours. Every number recorded for 129 comes
from an isolated worktree at the commit.

Findings 3 and 4 share one code site and the verifier established that neither
change fixes the other: the component-wise check does not stop a symlink
(`link` is a legal component) and canonicalisation does not admit `a..b` (it is
rejected before any path is built). Both are required, hence one builder, two
commits.

Both int64 sub-claims came back worse than reported: `static_cast<int64_t>` of
`2^63` yields `INT64_MIN` on x86-64, so a strictly positive tensor
(`[2**63-2, 2**63-1]`) summarises as maximally negative in BOTH `minimum_int`
and `maximum_int`, not merely a rounded magnitude.
