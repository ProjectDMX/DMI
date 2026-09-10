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

### PR 128 — three findings landed, independently verified

`8f23d0e` (sql_uuid validation), `359ba00` (one ten-character escaper in a new
`sql_escape.h` + `CURLOPT_POSTFIELDSIZE_LARGE`), `631c1c6` (legacy object in
the rebuild instruction). Branch counts: cpu 1202 -> 1204, live 106+1 -> 109+1,
the one failure being the same pre-existing `CREATE USER` test.

Red check: reverted `native/csrc` and `native/Makefile` to `7eac029`, rebuilt,
and all four new live tests went red, one per finding:

```
FAILED test_the_parameterized_statements_are_byte_identical_too
FAILED test_an_embedded_nul_does_not_truncate_the_statement_on_the_wire
FAILED test_a_pack_id_that_is_not_a_uuid_is_refused_before_it_reaches_the_sql
FAILED test_an_earlier_builds_object_beside_this_builds_is_refused
4 failed, 1 passed
```

Restored: 5 passed live, 2 passed cpu.

HONEST QUALIFICATION on the two cpu escape tests. They also fail at the
baseline, but NOT because of the defect — they fail with
`{'op': 'escape', 'ok': False, 'what': 'call open first'}`, because the
`escape` driver op is itself new. So `test_native_catalog_sql_escape.py` is a
forward regression guard, not a proof the bug existed. The builder was right
to say `test_the_ordinary_characters_are_left_alone` is not a red test (it
guards against OVER-escaping), and by the same token neither cpu test can
carry the red-green proof. The four live tests carry it.

### Two mismatches in my dispatch brief, corrected by the builder

Both were my error, inherited from a verifier that reasoned against
`polish-c` — which contains all three PRs at once, so it sees files that
branch 128 alone does not:

- I listed `reader.cpp:497`, `reader.cpp:669` and `catalog_writer.cpp:806` as
  sites the shared escaper would fix. `reader.cpp` does not exist on 128 at
  all (it is a 129 file). The builder consolidated the four copies that do
  exist on this branch instead, which is the correct reading.
- I told it to EXTEND `test_the_parameterized_statements_are_byte_identical_too`.
  That test does not exist on 128 — I wrote it earlier this session on 129.
  The builder added one under that name rather than inventing a different one.

### Duplicate-test hazard from the second mismatch, checked and contained

Because both branches now define
`test_the_parameterized_statements_are_byte_identical_too` in the same file
(129 at line 303, 128 at line 308), the forward-merge could in principle have
landed two same-named functions in one module, where Python keeps only the
last and the other silently never runs.

Trial-merged `631c1c6` into `2626b46` in a throwaway worktree to find out.
Git raises it as a conflict, and both definitions fall inside ONE conflict
block (lines 292-433), so a human resolution is forced and no silent
shadowing is possible. Conflicts also appear in `native/Makefile` and
`native/csrc/catalog/catalog_writer.cpp`. Merge aborted, worktree removed.

Resolution when the stack is forward-merged: keep 128's version and drop
129's. 128's is strictly stronger — all ten of the driver's characters plus
the NUL case — and 129's is the weaker original whose fixture (`it's\odd`)
only covered the two characters both implementations already agreed on.

### Round 2 — a /code-review of the LANDED work, all five findings closed

The human ran `/code-review` over the 129 diff. It found five things, one of
them a defect in the fix I had just landed. All five are now fixed and
red-checked; none was refuted.

| Finding | PR | Commit |
|---|---|---|
| cursor `v`/`w` still wrapped (high) | 129 | `a6b5260` |
| `number()` accepted leading zeros | 129 | `40e5ada` |
| `search()` never validated its text filters | 129 | `343bef6` |
| harness decoded a negative bound to UINT64_MAX | 129 | `dee5d95` |
| `substitute()` rescanned from 0 per parameter | 128 | `560b229`, `33ea8d7` |

A verification mistake of mine worth recording: my first red check of
`substitute` reported "5 passed" — a FALSE GREEN. Reverting the `.h` broke the
build (`'substitute' is not a member of 'dmi_catalog'`) and the tests ran
against a STALE binary. Same trap as earlier in the session. The CPU gate
cannot be red-checked by reverting at all, because the driver op it drives is
part of the fix; the LIVE test can, because it uses the pre-existing `acquire`
op. Redone properly with `build errors: 0` asserted in BOTH states: red at
`8ee2856`, green at HEAD.

The `substitute` red also exposed something the verifier had missed — TWO rows
at one term with DIFFERENT holders, because `release_statement()` binds no
`ttl_ns`, so the tombstone stored the literal `%(ttl_ns)s` while the claim
stored `30000000000`. The test reads back a sorted distinct holder list rather
than a single value, so that shows up in the failure message instead of
tripping a `len(rows) == 1` guard.

Sibling check, because this run has been bitten four times by fixing one site
and leaving its twin: `native/csrc/clickhouse_client.cpp` (a different 53 KB
file sharing the basename) has ZERO `%(` matches and no `substitute`, so it
does not carry this defect.

### THE RUN'S MAIN TECHNICAL LESSON: fix the primitive, not the caller

Four separate findings this run were the same shape — a defect fixed at one
call site while its siblings kept it:

1. the SELECT-row splitter got `[`/`]` depth tracking; the footer splitter did not
2. `hydration.cpp` tightened `uint64`; `pack_index.cpp` stayed lax
3. the cursor's `captured_at_ns` was bounded; its `v` and `w` were not
4. and the root of #3: `jc::FindInt` in `common/json.cpp` is STILL an
   unbounded `int64_t` accumulator for ~100 other call sites

#4 is the open one and the most valuable next step. The cursor path no longer
depends on it, but the sink, store and pack drivers and `pack_index.cpp` all
do, and none was audited. It needs its own finding and its own review because
changing shared JSON semantics touches files PR 128 also modifies.

### CORRECTION — I reported the rank>=2 hydration bug as live. It was already fixed.

The verifier reasoned against the FROZEN baseline `a4d3dcb`, where the bug is
real. The branch had moved: the other session's `422dc6e` ("seven review
findings across the three PRs", Sep 7 00:19) already fixed both hydration
defects. I checked rather than taking the builder's word:

```
3af38c3 hydration.cpp:  595 int depth = 0;  616 ++depth;  625 if (c == ',' && depth == 0)
                        231 kCatalogToFooter()   650 for (... : kCatalogToFooter())
a4d3dcb hydration.cpp:  depth matches: 0
```

Present at the tip, absent at the baseline. So my previous statement to the
human — that every rank>=2 capture is refused at hydration — was true of the
frozen review baseline and NOT true of the shipping branch. Stated plainly
because it materially changes the severity of what I reported.

What was genuinely missing was the TESTS, and that residue is not small: the
rank-2 bug survived a fix of the identical bug class in the SELECT-row
splitter precisely because the hydration suite was 100% rank 1, and the
9-field comparison survived because the only mismatch cases anyone had
written were the three fields already inside the list.

`ea3bcd1` and `7a22e66` add them, and the builder proved each genuinely red by
temporarily reverting the exact code the finding names, then restoring:

- removing the depth tracking -> `catalog descriptor does not match the pack
  footer: field 3` on the rank-2 case while the rank-1 test passed in the same
  run. (Field 3, not the verifier's field 20, because the widened comparison
  now trips on the first shifted field rather than on kShape — same shift,
  earlier detection.)
- narrowing the comparison back to 9 fields -> 5 failures, `layer_number`,
  `hook_name`, `step_number`, `request_id`, `captured_at_ns`, with the 3
  locator cases still passing, exactly the split the docstrings claim.

Parity suite 23 -> **32 passed**; cpu unchanged at 1203. Branch fast-forwarded
to `7a22e66` and re-verified there.

Field accounting, since I had asked for it: Python compares 21 CaptureMetadata
fields plus the 5-tuple locator = 26. Native's `kCatalogToFooter()` compares
all 32 catalog columns, so all 26 are covered and nothing Python checks is
missing natively. The 6 extras (`pack_id`, `store_id`, `object_key`,
`object_bytes`, `pack_checksum`, `pack_record_count`) are tautological rather
than wrong — Python excludes them deliberately because they came from the row
under test — so they were left alone.

### PR 128 — footer metadata validation landed, with one cross-branch defect I caught

`e10653b`, `cb39890`, `2413153`. 17 bug proofs and 12 regression guards, each
labelled as such, plus a control test (`test_a_well_formed_pack_still_indexes`)
without which a reader that refused everything would pass every refusal case.
cpu 1204 unchanged, live 111 -> **140 passed**, 1 pre-existing failure.

The poison-shape test is built in two halves so the harm is recorded
independently of the fix: it writes `shape=[0,4294967295]` through the
untouched `write_descriptors` path, shows ClickHouse storing it verbatim in
`Array(UInt32)`, shows `reader.search()` then raising, and only then shows the
footer carrying it refused at the boundary.

CROSS-BRANCH DEFECT, caught before it shipped. Among its "additions beyond the
confirmed list" the builder narrowed `pack_index.cpp`'s dtype table to the ten
names in `_DTYPE_BYTES`. Correct against 128's own oracle — and it breaks 129.
The stack is 127 -> 128 -> 129, and the oracle GREW on 129:

```
128 tip  _DTYPE_BYTES  10 names
129 tip  _DTYPE_BYTES  14 names   (+ float8_e4m3fn, float8_e5m2, uint16, uint32)
a5e15a2 "fp8 and the wide integers are first-class dtypes"  -> 129 ONLY
```

and 129's `test_fp8_and_wide_int_summaries_parity` stages exactly those four
and drives them through `op="hydrate"` into this very function. After the
forward-merge the narrowed table would refuse dtypes 129 declares first-class,
failing that test. The builder could not see this from inside 128 — my
dispatch never told it the branch is stacked.

Note this also RESOLVES the apparent contradiction with the earlier dtype
verifier, which found 14 oracle names and zero width disagreements: it was
reading `polish-c`, which contains all three PRs. Both agents were right about
different branches. Same root cause as the two `reader.cpp` mismatches earlier
— a verifier reasoning against the combined tree and a builder reasoning
against one branch.

Sent back for a per-branch-correct derivation from `kDtypes` in
`pack_builder.cpp`, which is `std::array<const char*, 10>` on 128 and
`std::array<const char*, 14>` on 129 — so it tracks the oracle automatically
and the merge resolves itself instead of encoding a count that a merge can
invalidate.

### Follow-up `8ee2856` — the derivation, and the merge verified by trial rather than argument

`pack_index.cpp` no longer answers the dtype membership question: the gate is
`if (!dmi_pack::DtypeSupported(dtype))`, reading `kDtypes`, which tracks the
oracle per branch. It kept only the element WIDTHS, which already cover 129's
four additions (fp8 -> 1, uint16 -> 2, uint32 -> 4). No new dependency —
`pack_index.cpp` already includes `pack_builder.h` for `Crc32`. It considered
and rejected routing through the sink's `DtypeWidth`, which would have made
the reader depend on the writer's sink and needed a Makefile edit.

Verified on this branch, name for name: `float32`/`bool`/`int64` admitted;
`uint16`, `uint32`, `float8_e4m3fn`, `float8_e5m2`, `uint64`, `float128`,
`not-a-dtype` refused — the live test reaching the same split independently.

I trial-merged the two branch tips rather than accepting the merge-safety
argument, and it corrected one of the builder's claims:

| file | result |
|---|---|
| `pack_builder.cpp`, `pack_builder.h` | auto-merged CLEAN, as claimed |
| `pack_index.cpp` | **CONFLICTS** — the builder said 129 never touches it, but 129's `422dc6e` did |
| `catalog_writer.cpp`, `tests/test_native_catalog_lease_live.py` | conflict (already known) |

The `pack_index.cpp` conflict is benign and informative: BOTH branches
independently fixed the same rank-0 bug. 129 added
`if (jc::Unwrap(shape_json).empty()) return 1;` inside the old
`shape_product`; 128 replaced that function with a validated `parse_shape`
that also treats `[]` as rank-0 (`pack_index.cpp:217`) and additionally checks
bounds, overflow, and refusal ORDER against the oracle.

RESOLUTION when forward-merging: take 128's side (`8ee2856`) for that region.
It is strictly stronger and preserves the rank-0 behaviour 129 was fixing.
Coverage survives either way — 129 pins the neighbouring case with
`test_zero_element_tensor_hydrates_to_empty_bytes`
(test_native_reader_parity_live.py:1831). Note the two are DIFFERENT cases:
`shape=[]` is rank-0 with product 1, `shape=[0]` is a zero-element tensor with
`logical_bytes == 0`. The merged tree handles both.

Merge-safe test design worth copying: the builder noted a "native refuses
uint32" test PASSES on 128 (128's oracle refuses it too), so such a test could
never have caught the narrowing. Its ten probes instead assert AGREEMENT in
whichever direction the contract moves — for each name, both readers admit the
footer or both refuse with the same sentence. On 128 that is the 3/7 split;
after the merge four flip to admitted on both sides and the test still holds.
`uint64` is deliberately included as the case that catches a width table
quietly widening membership.

Two stale comments it flagged and left alone, both outside the finding and
both invalidated by the merge: `pack_builder.h:83` ("one of the ten supported
names") and `record_row.h:58` ("the ten supported dtype names"). `a5e15a2`
fixes neither, so they will be stale on the merged tree.

### COORDINATION PROBLEM with the other session

`422dc6e` also committed MY `.loop/polish-seen.md` (+106) and
`.loop/polish-state.md` (+221). The other session is reading this ledger and
fixing findings out of it, which explains the overlap and the two reverts of
`reader.cpp`. Useful, but it means confirmed findings can be fixed twice and
a verifier's baseline can go stale mid-run. Checked the blast radius: its
`pack_index.cpp` change in that commit is only +5 lines and contains none of
the metadata-bound validation, so the 128 work now in flight is NOT
duplicated.

### PR 128 — last two findings landed, and my prescription was wrong

`28aff34` (one failure per distinct claimant, plus the oracle's message text)
and `6cb6f6b` (grant-probe the superseded object). Red check: reverted
`indexer.cpp` and `schema.cpp` to `631c1c6`, rebuilt, both new tests failed;
restored, both pass. cpu 1204 (unchanged — the new tests are live-marked),
live 109+1 -> 111+1.

I told the builder to emit conflict failures "in map order". That was WRONG
and it overrode me correctly. Python's `conflicted` is a dict, and dicts are
insertion-ordered, so `conflicted.items()` yields FIRST-CONFLICT order. I
confirmed it directly:

```
dict order for z,w,z,w -> ['z', 'w']
```

A `std::map` sorted by identity would have emitted `[w..., z...]` where the
oracle emits `[z..., w...]` — preserving the very defect class the fix
targets, since `merge` truncates `failures[:failure_limit]`. The builder used
a `conflict_order` key vector to reproduce insertion order and demonstrated
the divergence with a `[z1,w1,z2,w2]` case. It also verified its Python-repr
helper byte-for-byte against the oracle, including
`store_id = "it's\todd\\"`.

NEW finding it surfaced and correctly declined to fix: the NON-conflicted
`unique` list has the same ordering divergence in a different loop — native
walks `by_identity` as a `std::map` (sorted by `(store_id, pack_id)`) where
Python's dict yields first-appearance order. That affects descriptor/inventory
row order and the published member list, so it is a wider blast radius than
the finding it sat next to. Recorded for the human, not fixed.

### PR 129 — reader findings landed, after being reverted TWICE by the other session

`7118b6b` (cursor timestamp validated, carried as a typed `uint64_t`) and
`3af38c3` (the four missing CaptureQuery bounds). Verified as before: both new
tests red against the pre-fix `reader.cpp`, green after, and the full parity
suite is `23 passed`.

INCIDENT worth remembering. The surgical-index commit technique
(`git hash-object` + `git update-index --cacheinfo`) creates a correct commit
but leaves the shared worktree's FILE stale — the index and the working tree
disagree. The other session commits with `git commit -a`, which stages that
stale file and silently reverts the fix. It happened twice:

| Their commit | Effect |
|---|---|
| `422dc6e` "seven review findings" | `reader.cpp | 46 +------`, test file `| 60 ------` — pure removal of the fix, no content of their own in those two files |
| `79addbb` "zero-element tensor" | `reader.cpp | 70 +-------` — reverted the reland |

The builder re-landed a third time, additively on top of `79addbb`, preserving
their new `test_zero_element_tensor_hydrates_to_empty_bytes` verbatim.

I then closed the loop by refreshing the two stale working-tree copies in
`/tmp/opencode/native-wt` to HEAD, which the builder was forbidden to do.
Before overwriting I checked exactly what would be lost: the diff was 186
deletions and just 2 insertions, and both insertions were the PRE-FIX
originals — including the exact buggy line the finding names,
`after = {literal(0), literal(1), literal(2), parts[3], literal(4)};`. So
nothing of the other session's was in those files. After the refresh,
`git status` shows only my own `.loop/polish-state.md` as modified.

LESSON for any future run sharing a worktree: the index technique is only half
a solution. Either refresh the working-tree file immediately after committing,
or do not share the worktree at all.

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

## Round 3 — the primitive, then all 89 of its call sites

The run's recurring lesson was fixing a defect at one call site and leaving
its siblings. `jc::FindInt` was the root: an unbounded `int64_t` accumulator
shared by the whole native tree.

`329f7a4` bounded it before the multiply behind a new `FindIntChecked` with
`IntFind { kOk, kAbsent, kOutOfRange }`, and `1c45872` migrated the one
production caller. The design point worth keeping: the accepted range is the
UNION `[-2^63, 2^64-1]`, stored as the two's-complement pattern. For a
literal in `[2^63, 2^64-1]` the OLD wrap produced the same bit pattern, so
`static_cast<uint64_t>` was ACCIDENTALLY CORRECT, and a conformance test
depends on it (`step_number = 2**64-1`, legal per model.py). A tightening to
`INT64_MAX` would have broken parity in the other direction.

Then the remaining 89 sites, one commit per driver (`aa7dc59`, `3c833b0`,
`6bb4aa9`, `7325269`, `ab8687e`, `cd6d039`, `18b7f9c`). Red check: reverting
the six source files put **77 tests red** with `build errors: 0` asserted, so
genuinely rebuilt. cpu 1228 -> **1331**; live unchanged at 188 plus the known
`CREATE USER` failure.

Corrections made during that round:

- My site count of ~92 was wrong: `grep -c` counts LINES and swept in
  comments. The real figure is 89 — `reader.cpp` has ZERO (both matches are
  comments), `pack_index.cpp` 2 not 3, `conformance_catalog.cpp` 46 not 47.
- The prior builder's stated reason for skipping the drivers — "a refusal
  needs a new `ok:false` shape no test exercises" — did not survive contact.
  NO driver needed one; each was routed through the refusal shape it already
  had, and the catalog driver through the `CatalogError(kValue, ...)` its own
  neighbouring lambda already throws.
- The builder corrected its own commit messages: it had described the
  spool/store bounds as falling back to a 1 TiB default, but the fallback is
  keyed on ZERO, so `-1` cast to `UINT64_MAX` and the capacity limit became
  NO limit. Worse than first written.

Worst thing found in that sweep: in the catalog driver an out-of-range
`uint`/`int` parameter rendered `18446744073709551615` or `-1` straight into
SQL, and out-of-range reader read-bounds LIFTED the bound rather than
tightening it — the inverse of what a bound is for.

## Closing state

`8a228ef` drops the now-callerless `FindInt` wrapper. It only ever existed to
return -1 for both `kAbsent` and `kOutOfRange` — the exact conflation the
migration removed — so keeping it exported left a way back to the bug.

`b289356` removes the false parity claim at the uploader byte gate, in both
`uploader.cpp` and `uploader.h`. Behaviour is deliberately UNCHANGED: whether
to adopt the oracle's whole-batch refusal is a maintainer call, pinned by two
purpose-named tests and a `docs/benchmarks.md` entry. Only the claim that it
already matches the oracle is gone, with the divergence named in its place.

Everything is on `feat/native-capture-pipeline` (PR #127) — the only open PR
after #128 and #129 merged mid-run. Both CI jobs green.

Still open, for the human:

- the uploader byte-gate POLICY (documented now, not decided)
- six confirmed-but-`optional` findings, listed in the PR description's
  "deliberately not fixed" table
- the quiet-host benchmark re-measure, unchanged from Checkpoint B
