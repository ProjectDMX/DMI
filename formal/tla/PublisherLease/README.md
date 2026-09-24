# PublisherLease: TLA+ model of the ClickHouse publisher lease and fenced publish

This spec models the leader lease that decides which writer may make a catalog snapshot visible, together with the server-side fence that every visibility write carries. It was written against `main` at `a987dfe`.

- Python: `src/dmi/storage/capture/clickhouse_lease.py` (the coordinator) and `src/dmi/storage/capture/clickhouse_catalog.py` (publish, quarantine, config).
- C++ port: `native/csrc/catalog/lease_coordinator.{h,cpp}` and `native/csrc/catalog/catalog_writer.cpp`.

| File | What it is |
|---|---|
| `PublisherLease.tla` | The spec. The constant `Mutation` removes one safety ingredient at a time, and `ClockModel` selects the clock assumption. |
| `MC.tla` | TLC wrapper. It adds the `Permutations(Writers)` symmetry. |
| `PublisherLease.cfg` | Faithful model, no faults. |
| `PublisherLease_unknowns.cfg` | Faithful model with outcome-unknown failures. |
| `PublisherLease_crash.cfg` | Faithful model with one crash-restart. |
| `Var_cpp_keep_lease.cfg` | Faithful model with the C++ `reject_if_gone` semantics. |
| `Mut_*.cfg` | Mutations. Each one must fail. |
| `Chk_grants_increase.cfg` | A characterisation check, not a property the code claims. |

## What is modelled

- **Writers.** There are N writer processes, each running `acquire`, `renew`, the publish and `release`. Each writer runs one lease operation at a time, because the catalog's `_serial` RLock enforces that (`clickhouse_catalog.py:196-222`).
- **The lease table.** An append-only set of rows `[t, id, e]` (term, lease_id, expires_at_ns).
  - `HeadTerm`, `HeadIds`, `ExpAt` (the minimum over a key), `TopId` (the highest lease_id, standing in for `ORDER BY lease_id DESC`), `LiveUntil` and `Claimants` follow `head()` and `fence()` exactly (`clickhouse_lease.py:171-190`, `:213-225`).
  - Lease ids are integers drawn nondeterministically from a pool, never reused (uuid4), so every UUID collation order is explored.
- **Time.** A discrete true clock `now`.
  - Every timestamp the protocol uses is a server `now64()`. Each server statement reads `now + d`, with `d` chosen per statement from `Offsets`, so any host can serve any statement at any moment.
  - `ClockModel = "pairwise"` gives `d ∈ 0..S`: any two readings differ by at most S. This is what `clock_skew_ns` means according to its docstring, "how far apart two ClickHouse HOSTS' clocks may be" (`clickhouse_catalog.py:72-90`).
  - `ClockModel = "offset"` gives `d ∈ -S..S`: each host is within S of true time, so two hosts can be 2S apart.
  - The code never lets a writer's wall clock decide anything, so writer clocks are not state. The only client-side clock is the monotonic quarantine timer, modelled as a true-time duration.
- **Publish window.** A fenced statement is admitted at `now` and its effect can land at any point up to `now + P`. `Tick` may not advance past an in-flight deadline. P is `max_execution_time` (`clickhouse_catalog.py:497-501`, `clickhouse_lease.py:235-239`). Whole seconds are enforced at `clickhouse_catalog.py:163-174`.

### Actions, and what is atomic

Each action below is one server statement, or one purely local step. Consecutive statements are separate actions, so time and other writers can interleave between them.

| Spec action | Code | Atomicity argument |
|---|---|---|
| `AcquireHead` / `RenewHead` | `acquire`/`renew` → `claim`: `head()` + `_reject_live` (`clickhouse_lease.py:58-78, 137-140, 161-190, 258-280`) | `head()` is one `SELECT`. Rows at max(term) and `now64()` come from one snapshot, as the docstring at :167-169 says. `_reject_live` is local. A rejection drops the local lease (:263). |
| `Insert` | `_insert` (`:282-300`) | One `INSERT ... SELECT now64()`. `expires = now64 + TTL` is stamped server-side. |
| `InsertUnknown`, `LandPending` | a claim INSERT that raises a non-`CaptureStorageError` (transport, quorum timeout). The catalog quarantines (`clickhouse_catalog.py:680-701`). | The row may land later, stamped when it runs, or never. The claim leaves `self._lease` untouched. |
| `Readback` | `claim` read-back (`clickhouse_lease.py:142-159`) | A separate `SELECT ... WHERE term = T` with `DECIDING_READ`. It grants only if the ids at T are exactly `{mine}`. Otherwise `_lease = None`. |
| `Admit` | a fenced manifest/watermark INSERT (`clickhouse_catalog.py:521-535, 551-576`) with `fence()` (`clickhouse_lease.py:192-225`) | The fence and the write are ONE statement. The fence is evaluated at admission, and the effect lands at or before admission + P. If the fence refuses, `reject_if_gone` (`:241-256`) runs. |
| `Land`, `Done` | the server finishes the statement, and the client sees the result | |
| `ClientError` | an outcome-unknown publish (`clickhouse_catalog.py:430-470`), handled by `_quarantine_locked` (`:294-306`) | The statement may still be running. The lease is discarded with no tombstone. |
| `QuarantineEnds` | `_require_not_quarantined_locked` (`:268-277`) | After `lease_ttl_ns` of monotonic time. The next acquire mints a fresh id. |
| `Release` | `release()` (`clickhouse_lease.py:80-111, 127-134`) | One INSERT: an already-expired row at the holder's own term. It reads nothing. |
| `Crash` | process restart | Local lease and quarantine are lost. Server statements keep running. |
| `OldReleaseRead`/`Land` | mutation only: the pre-fix release that the docstring describes (`:88-99`) | Its SELECT and its row landing race a concurrent claim. |

A publish is `RenewHead → Insert → Readback → Admit`. The code renews before every fenced statement (`clickhouse_catalog.py:489, 549`), and time may pass between the renewal and the fence. Manifest chunks are further renew-then-fence cycles, so a single fenced statement per cycle loses nothing.

## Properties

| Name | Meaning | Source claim |
|---|---|---|
| `MutualExclusion` | No two writers have admitted statements in flight at the same true time. | fence docstring, `clickhouse_lease.py:200-211` |
| `OnePublishInFlight` | At most one admitted statement is in flight at all, even under one lease. | `clickhouse_catalog.py:196-198`, and the reason for the quarantine at `:229-243` |
| `NoGrantDuringFlight` | No other lease is granted while an older holder's statement may still land. | `clickhouse_catalog.py:64-69`, `clickhouse_lease.py:116-122` |
| `SoleGrantPerTerm` | At most one lease_id is ever granted per term. | the read-back; `docs/catalog-descriptor-key.md` "Where that safety comes from is the read-back" |
| `ReleasedNeverAdmittable` | For every released id, the fence predicate is false under every clock reading. | `release()` docstring: the tombstone "ends the lease wherever it lands" |
| `NoAdmitAfterRelease` | The fence never admits a statement for a released id. | same |
| `ReleaseNeverContestsSuccessor` | A tombstone never shares a term with another id's grant. | `clickhouse_lease.py:96-99` |
| `TermsMonotonic`, `HeldTermMonotonic` | The head term never decreases. A writer's term under one lease_id never decreases. | |
| `GrantsIncrease` (characterisation) | Each grant is at a term above every earlier grant. | not claimed |

Liveness was not checked. The protocol has no deterministic progress guarantee:

- Two claimants whose inserts tie at a term both abandon (`_reject_live` with claimants > 1).
- An adversarial scheduler can repeat that tie indefinitely.
- Progress relies on timing and randomness.

A bounded-time model also makes "eventually" meaningless at the horizon.

## Bounds

All configs use:

- Writers: `{w1, w2}`, symmetric.
- Ids: `{1, 2, 3}`.
- `TTL = 3`, `P = 1`, `S = 1`. This is the tightest integer setting that satisfies `TTL > P + S`, the `__post_init__` margin at `clickhouse_catalog.py:134-146`.
- `MaxTerm = 4`. Every renewal costs a term, so a full takeover (A acquire, A renew, B acquire, B renew) needs 4.
- `MaxTime = 4`. This covers a claim, its expiry, and a takeover with offsets.

The `VIEW` fingerprints rows, grants and tombstones only at or above `Floor`, the lowest term any future step can read or write. The argument for why this is sound is next to `Floor` in the spec.

Deadlock checking is off (`CHECK_DEADLOCK FALSE` in every config, the same as `-deadlock`) because behaviours stop at the time horizon. Symmetry is used only with safety properties.

## Commands and results

From this directory, with `J=/path/to/tla2tools.jar` and `M=<scratch dir>`:

```
java -XX:+UseParallelGC -cp $J tlc2.TLC -workers 2 -noGenerateSpecTE \
     -metadir $M/<cfg> -config <cfg>.cfg MC.tla
```

The machine had 4 cores shared with other TLC jobs, so wall times are pessimistic.

For failing configs, "depth" is the BFS depth at which TLC stopped. The counterexample has that many states.

| Config | Expected | Result | States generated | Distinct | Depth | Wall |
|---|---|---|---|---|---|---|
| `PublisherLease` (faithful, no faults) | pass | **PASS**, all 8 invariants and 2 action properties | 8,076,278 | 2,886,614 | 25 | 5m01s (10m28s on a busier run) |
| `PublisherLease_unknowns` (faithful, outcome-unknown faults) | pass | **PASS** | 33,850,460 | 10,240,656 | 26 | 16m01s |
| `PublisherLease_crash` (faithful, 1 crash-restart) | pass | **PASS** | 16,785,902 | 4,676,420 | 25 | 8m13s |
| `Var_cpp_keep_lease` (C++ `reject_if_gone`) | pass | **PASS** (the same reachable state set as faithful) | 8,076,278 | 2,886,614 | 25 | 6m07s |
| `Mut_no_skew` | fail | **FAIL** `MutualExclusion` | 1,156,491 | 524,087 | 17 | 1m10s |
| `Mut_offset_clocks` | fail | **FAIL** `MutualExclusion` | 4,142,033 | 1,659,438 | 17 | 2m08s |
| `Mut_linger` | fail | **FAIL** `MutualExclusion` | 1,523,496 | 582,762 | 18 | 47s |
| `Mut_release_on_unknown` | fail | **FAIL** `MutualExclusion` | 2,340,621 | 1,010,264 | 17 | 39s |
| `Mut_no_quarantine` | fail | **FAIL** `OnePublishInFlight` | 516,415 | 223,837 | 14 | 15s |
| `Mut_no_readback` | fail | **FAIL** `SoleGrantPerTerm` | 1,336 | 677 | 7 | 3s |
| `Mut_no_readback_mutex` | fail | **FAIL** `MutualExclusion` | 2,058,466 | 861,539 | 17 | 1m03s |
| `Mut_old_release` | fail | **FAIL** `ReleaseNeverContestsSuccessor` | 23,986 | 12,425 | 11 | 5s |
| `Chk_grants_increase` | (characterise) | **FAIL**, benign, see below | 7,463 | 4,000 | 9 | 3s |


### Counterexamples (mutations)

- **no_skew** (depth 17):
  1. A claims term 1. At t=0 it renews to term 2 on a host at +0, stamped `e=3`, then waits.
  2. At t=1, A's publish is admitted on a host at +0, because `3 > 1+0+P` (the margin without S), with deadline t=2.
  3. At t=2 B reads the head on a host at +1: `now_ns=3 >= e`, so the lease looks dead.
  4. B claims term 3, renews to term 4, and is admitted at t=2 while A's statement can still land.
- **offset_clocks** (the faithful fence, but clocks at ±S):
  1. A's lease is stamped on a host at -1 (`e=2`).
  2. A is admitted at t=0 on a host at -1, because `2 > 0-1+P+S`, with deadline t=1.
  3. At t=1 B reads the head on a host at +1, `now_ns=2 >= e`, claims, renews, and is admitted at t=1.

  The two hosts are 2S apart. `clock_skew_ns` has to bound the **pairwise** difference between hosts, not each host's NTP offset.
- **linger** (the effect can outlive `max_execution_time`): A is admitted at t=1 (deadline 2). Its effect has not landed at t=3, when B, seeing A expired, claims, renews and is admitted.
- **release_on_unknown**:
  1. A is admitted.
  2. The client sees an outcome-unknown error and writes the release tombstone instead of discarding the lease.
  3. B immediately sees the head dead, claims, renews and is admitted while A's statement runs.

  With `OnePublishInFlight` enabled, the same thing is found with A itself re-acquiring under a fresh id.
- **no_quarantine**: A is admitted, then gets an outcome-unknown error and keeps its lease. A renews the same lease_id and is admitted again while the first statement is in flight. This violates `OnePublishInFlight`: the hazard at `clickhouse_catalog.py:229-243`.
- **no_readback** (depth 7): two claimants both insert at term 1, and both are "granted" (`SoleGrantPerTerm`).
- **no_readback_mutex** (depth 17): without the read-back, `MutualExclusion` also fails, even though every publish renews first.
  1. A and B both claim term 1, and neither checks.
  2. A's renewal reads the head at t=0, before B's row lands, sees only itself, and targets term 2.
  3. At t=2 B's renewal sees term 1 expired and also targets term 2.
  4. A inserts at term 2 and is admitted.
  5. B then inserts at term 2. B's id sorts higher, so the fence resolves B and admits it while A's statement is in flight.

  A renewal is a claim, and without the read-back nothing stops the late co-claimant. This confirms the design doc's statement that the safety comes from the read-back.
- **old_release** (depth 11):
  1. A holds term 1, and its lease has expired by t=2.
  2. A's old-style release reads the head (still A at term 1) and targets term 2.
  3. B claims term 2 and passes its read-back.
  4. A's tombstone lands at term 2, so term 2 is contested with B.

  This is exactly the bug described at `clickhouse_lease.py:88-96`. When A's UUID sorts higher, B is fenced out at once.

### Characterisation: `GrantsIncrease` fails benignly

1. A inserts term 1 at t=0, and its read-back is delayed until t=2.
2. B sees term 1 expired, claims term 2, and passes its read-back.
3. A's read-back then sees `{A}` at term 1 and `claim()` returns a `PublisherLease` for term 1, which has already been superseded and has expired.

This is safe: A's first fenced statement resolves B at the head and is refused, and A's renewal is rejected because B is live. But `claim()` (`clickhouse_lease.py:148-156`) does not check that the lease it returns is still live or still at the head. Code that treats `publisher_lease is not None` as "I am the leader" (for example the guard at `catalog.py:457`) can hold a stale answer for up to a round trip. The only consequence is wasted work before the fence refuses.

## Python / C++ divergences

- **`reject_if_gone` keeps the local lease in C++.** In C++ (`lease_coordinator.cpp:187-201`) the method is `const` and never resets `lease_`. In Python (`clickhouse_lease.py:245`) it sets `self._lease = None`. So after a fenced-out publish, the C++ writer's next renewal reuses the old lease_id at a new term, while Python requires a fresh `acquire`.

  `Var_cpp_keep_lease` models this and reaches exactly the same set of states as the faithful model. Both "fenced out and kept the id" and "fenced out and dropped it" are states the faithful model already reaches: through a writer that simply did not publish, and through a writer rejected by its renewal. The divergence is therefore safe for the lease protocol. It only changes which error the caller sees next.
- **No term-range check in C++.** The C++ claim (`lease_coordinator.cpp:107`) computes `current.term + 1` as `uint64_t`, with no `_validate_term` (Python `clickhouse_lease.py:140, 310-312`). At 2^64-1 it would wrap to 0 and insert below the head. This is unreachable in practice and not modelled.
- **Exception classification.** Python quarantines on any `BaseException` that is not a `CaptureStorageError` or `ValueError` (`clickhouse_catalog.py:466-470, 686-701`). C++ quarantines on `std::exception` that is not a `CatalogError` (`catalog_writer.cpp:284-293, 661-667`). A non-`std::exception` throw would skip quarantine in C++. This is not modelled.
- **`fence_eval` exists only in C++.** It runs the fence as a separate SELECT (`lease_coordinator.cpp:171-185`). It is used only by the conformance/test harness (`conformance_catalog.cpp:1027`). No production path fences with a separate read.

## Abstractions and assumptions

1. **The lease table is linearizable.** Every statement is atomic against the whole table. The code relies on `select_sequential_consistency` plus `insert_quorum` with `insert_quorum_parallel=0` (`clickhouse_sql.py:22, 32-60`) for this on a replicated deployment. The code comments say it is not guaranteed without `insert_quorum` on a replicated deployment (`clickhouse_catalog.py:92-101`), and that case is not modelled.
2. **An admitted statement's effect lands within P of admission.** `max_execution_time` is measured from query start, which is at or before the fence evaluation. `Mut_linger` shows this assumption is load-bearing. A quorum INSERT whose quorum wait times out may leave a part that replicates later; whether ClickHouse can make that row visible after the error is outside this model.
3. **Monotonic time.** Server clocks may be offset, but elapsed-time measurements (`max_execution_time`, the client's monotonic quarantine timer) run at true rate. There is no drift and no clock step.
4. **Scalar subqueries and `now64()` are evaluated once at admission.** This matches ClickHouse's handling of constant scalar subqueries.
5. **Watermark and manifest are not modelled.** The version barrier, the manifest/watermark contents, and `publish_id` ownership read-backs are omitted. They do not feed the lease decision.
6. **Small integers.** Integer time, with `TTL=3`, `P=1`, `S=1`. The strict `>` in the fence and the `<=` in `_reject_live` are preserved exactly.
