# Formal specifications

Machine-checked models of five pieces of DMI whose correctness arguments are
currently carried by prose: the catalog version allocator, the publisher lease
and fenced publish protocol, the two-host clock-skew bound in the fence margin,
the payload ring's span arithmetic, and the host-side capacity check that span
arithmetic depends on.

Nothing here is built, imported or executed by DMI. The specs are checked by
hand with the commands below; none of them runs in CI.

```text
specs/
├── tla/    TLA+ models, checked with TLC
├── z3/     SMT encoding, checked with Z3
└── cbmc/   C++ harness, checked with CBMC
```

## What each spec models

| Spec | Models | Source of truth |
|---|---|---|
| `tla/VersionAllocator.tla` | the sole-claimant version allocation loop: floor read, jittered candidate, claim INSERT, singleton read-back, retry | `native/csrc/catalog/version_allocator.cpp:49-89`, header claim at `version_allocator.h:5-7`, watermark publish at `native/csrc/catalog/catalog_writer.cpp:587-620`, `clickhouse_client.cpp:107` |
| `tla/PublisherLease.tla` | the lease claim/renew/release protocol and the fenced publish: `claim_with_rival`, `head()`, `fence()`, `fence_eval()`, `reject_live()`, and `publish_snapshot`'s manifest chunks and watermark INSERT | `native/csrc/catalog/lease_coordinator.cpp:83-247`, `native/csrc/catalog/catalog_writer.cpp:148-158,478-640`, `clickhouse_client.cpp:107`, `docs/catalog-descriptor-key.md:290-360,461-560`, `src/dmi/storage/capture/clickhouse_lease.py:83-92,198-210` |
| `z3/clock_skew.py` | the two-host derivation behind the fence margin `publish_timeout_ns + clock_skew_ns` | the SQL at `docs/catalog-descriptor-key.md:352-362` (emitted by `catalog_writer.cpp:583`), the derivation at `:365-372`, the cap at `catalog_writer.cpp:519` |
| `cbmc/payload_ring_span.cpp` | `payload_compute_spans` and its stated precondition | `native/csrc/ring/payload_ring.cuh:44-86` |
| `tla/RingCapacity.tla` | who establishes that precondition: `prepare_step`, the eager safety net (`available_capacity`, `reserve_one`), `reserve_record` with reclaims, one async CUDA stream, the drain | `native/csrc/ring/ring_engine_py.cu:367-656`, `native/csrc/ring/drain_thread.cpp:253-600`, `native/csrc/ring/producer.cu`, `native/csrc/ring/task_ring.cuh:72-86`, `src/dmi/adapters/base.py:281-366`, `src/dmi/hooks/point.py:259-346`, `src/dmi/records.py:317-386` |

The TLA+ models are written against the **code**, not the prose. Where the two
disagree, the spec follows the code and a comment in the `.tla` says so.

## Getting the tools

```sh
# TLC -- a single jar, no install
curl -fsSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar

# Z3 (Python bindings include the solver)
pip install z3-solver

# CBMC (Debian/Ubuntu; or grab a release from github.com/diffblue/cbmc)
apt-get install cbmc
```

Checked with TLC 2.19, Z3 4.13 (`z3-solver` 4.13.x) and CBMC 5.95. Any recent
version of each should do; nothing here relies on a version-specific feature.

## Running the checks

### TLA+

Every `.cfg` in `specs/tla/` is one model: one set of constants, one invariant.
Run any of them from `specs/tla/`:

```sh
cd specs/tla
java -XX:+UseParallelGC -Xmx3g -cp /path/to/tla2tools.jar tlc2.TLC \
     -workers 4 -config lin_distinct.cfg VersionAllocator.tla
```

```sh
java -XX:+UseParallelGC -Xmx3g -cp /path/to/tla2tools.jar tlc2.TLC \
     -workers 4 -config PublisherLease_believers.cfg PublisherLease.tla
```

`PublisherLease.cfg` is the base model — the protocol exactly as shipped, with
the combined `AllSafety` invariant. Each `PublisherLease_*.cfg` turns a single
knob away from it or swaps in a single invariant; its header comment says which.

`RingCapacity.tla` follows the same convention with a `Ring_` prefix:
`Ring_base.cfg` (adapter path) and `Ring_rec_base.cfg` (record path) are the
shipped protocol, and each other `Ring_*.cfg` turns one knob or checks one
invariant. `Ring_rec_taskover` and `Ring_rec_wide` are the largest, at ~14M
distinct states and a few minutes each.

The largest runs (`believers`, `holderssafe`, `overrun`, `PublisherLease.cfg`)
generate ~20M states and take roughly 15-25 minutes on four workers with a 3 GB
heap. The rest finish in seconds to a couple of minutes.

### Z3

```sh
python3 specs/z3/clock_skew.py
```

Prints one line per check and exits non-zero if any result differs from the
expected one.

### CBMC

`cbmc` does not accept `-std=`, so the harness is compiled to a goto-binary
with `goto-cc` first and verified in a second step. The file is `.cpp` because
`payload_ring.cuh` uses a namespace; the two CUDA qualifiers are defined away
so the header compiles for the host.

```sh
cd specs/cbmc
B=$(mktemp -d)            # goto-binaries are build output; keep them out of the tree

# the shipped contract: the documented precondition is assumed
goto-cc -std=c++11 payload_ring_span.cpp -o "$B/span.gb"
cbmc --unwind 80 --unwinding-assertions "$B/span.gb"

# the same harness with the precondition dropped
goto-cc -std=c++11 -DDROP_PRECONDITION payload_ring_span.cpp -o "$B/span_nopre.gb"
cbmc --unwind 80 --unwinding-assertions --trace "$B/span_nopre.gb"
```

## Results

State counts are TLC's own, as `generated / distinct`. Counterexample lengths
are the number of states in the trace TLC printed, including the initial state.
TLC's parallel BFS does not guarantee a minimal counterexample, so a re-run may
report a different length for the same violation; `NoOrphanManifestRows` was
seen at both 9 and 14 states on the same config.

State counts for runs that **completed** are exact and reproducible. State
counts for runs that were **refuted** are not: TLC stops as soon as any worker
hits the violation, so the count depends on the worker count and on scheduling.
The figures below were recorded with 3 or 4 workers. Only the verdict is stable
for a refuted run, not the number beside it.

### Version allocator (`VersionAllocator.tla`)

All configs use `Attempts = 3`, `MaxSpread = 1`.

| Config | Store | Allocators | Invariant | Verdict | States |
|---|---|---|---|---|---|
| `lin_distinct` | linearizable | 3 | `Distinct` | **holds** | 669,421 / 407,083 |
| `nocap3_distinct` | linearizable | 3 | `Distinct` | **holds** | 1,094,245 / 689,368 |
| `nocap3_ceiling` | linearizable | 3 | `CeilingNeverBinds` | **holds** | 1,094,245 / 689,368 |
| `nocap_distinct` | linearizable | 2 | `Distinct` | **holds** | 1,117 / 836 |
| `nocap_ceiling` | linearizable | 2 | `CeilingNeverBinds` | **holds** | 1,117 / 836 |
| `lin_ceiling` | linearizable | 3 | `CeilingNeverBinds` | violated | 37,503 / 21,944 |
| `lin_solerow` | linearizable | 3 | `SoleRow` | violated (6) | 134 / 102 |
| `lin_floor` | linearizable | 3 | `FloorMonotonic` | violated (12) | 905 / 584 |
| `lin_publish` | linearizable | 3 | `NoPublishRefused` | violated (9) | 628 / 412 |
| `lin_budget` | linearizable | 3 | `NoBudgetExhaustion` | violated (17) | 38,286 / 22,370 |
| `nocapf_distinct` | shared stale frontier | 2 | `Distinct` | **holds** | 2,886,043 / 639,601 |
| `nocapf_ceiling` | shared stale frontier | 2 | `CeilingNeverBinds` | **holds** | 2,886,043 / 639,601 |
| `frontier_distinct` | shared stale frontier | 2 | `Distinct` | **holds** | 2,232,679 / 456,499 |
| `ec_distinct` | eventually consistent | 2 | `Distinct` | violated (9) | 1,545 / 735 |
| `nocap_ec` | eventually consistent | 2 | `Distinct` | violated (9) | 1,013 / 533 |

`MaxVersion` is a state-space bound, not a quantity in the code, so
`CeilingNeverBinds` probes whether the bound hid behaviour. It binds at
`MaxVersion = 6` (`lin_ceiling`), which is why `nocap3_*` exists: at
`MaxVersion = 18` the ceiling is never reached and `Distinct` holds over a
state space the bound did not truncate.

`SoleRow`, `FloorMonotonic`, `NoPublishRefused` and `NoBudgetExhaustion` are
refutation targets — claims the allocator is sometimes read as making but does
not make. Their counterexamples are the point, not a defect.

### Publisher lease (`PublisherLease.tla`)

Base model: two publishers, `Lids = {1,2,3}`, `MaxTerm = 3`, `MaxTime = 5`,
`MaxVersion = 2`, `MaxAttempts = 2`, `TTL = 2`, `PT = 1`, `SKEW = 0`,
`NumChunks = 1`, linearizable store, cap enforced, writer lock held, fresh
`publish_id`. Each config below differs from it only as its name says.

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `PublisherLease.cfg` (base) | `AllSafety` | **holds** | 19,807,266 / 9,665,700 |
| `base5` (`MaxTerm 5`, `MaxTime 8`) | `AllSafety` | **holds** | 5,195,821 / 2,408,145 |
| `believers` | `AtMostOneBeliever` | **holds** | 19,807,266 / 9,665,700 |
| `holderssafe` | `TwoHoldersIsSafe` | **holds** | 19,807,266 / 9,665,700 |
| `holders` | `AtMostOneHolder` | violated (11) | 10,106 / 5,855 |
| `noovr0` (empty refs, cap enforced) | `NoOverlappingAdmit` | **holds** | 1,761,711 / 799,611 |
| `ovr0` (empty refs, `AllowOverrun`) | `NoOverlappingAdmit` | violated (17) | 108,545 / 56,237 |
| `overrun` (`AllowOverrun`, 1 chunk) | `NoOverlappingAdmit` | **holds** (vacuously) | 20,855,772 / 9,873,900 |
| `selfrace` (shared writer, no lock) | `WatermarkMonotonic` | violated (29) | 11,411,403 / 5,835,034 |
| `selfracelocked` (shared writer, lock) | `AllSafety` | **holds** | 947,826 / 524,274 |
| `orphans` | `NoOrphanManifestRows` | violated (9) | 3,921 / 2,315 |
| `chunks2_orphans` (2 chunks) | `NoOrphanManifestRows` | violated (10) | 4,818 / 2,832 |
| `chunks2_prefix` (2 chunks) | `OrphansArePrefixes` | **holds** | 1,892,754 / 909,324 |
| `chunks2` (2 chunks) | `AllSafety` | **holds** | 1,892,754 / 909,324 |
| `stalepid` (reused `publish_id`) | `AllSafety` | **holds** | 15,073,338 / 7,368,006 |
| `nonlin_admit` (non-linearizable) | `NoOverlappingAdmit` | violated (30) | 4,990,872 / 1,250,982 |
| `nonlin_all` (non-linearizable) | `AllSafety` | violated (28) | 4,985,963 / 1,249,885 |

#### Vacuity guards

Each of these is an invariant we *want* refuted: the counterexample is the proof
that the model reaches the state the safety invariants are quantified over. An
invariant that holds because its subject is unreachable proves nothing.

| Config | Guard | Refuted at | States |
|---|---|---|---|
| `vac_publish` | `NeverPublishes` | yes | 69,435 / 35,732 |
| `vac_fence` | `NeverFences` | yes | 212 / 121 |
| `vac_admit` | `NeverAdmits` | yes | 54,661 / 28,790 |
| `vac_contested` | `NeverContested` | yes | 955 / 542 |
| `vac_bothadmit` | `NoBelieverDuringAdmit` | yes | 387,558 / 195,122 |
| `vac0` | `NeverAdmits` at the `ovr0` / `noovr0` constants | yes | 1,323 / 754 |

`vac0` is the one that matters for reading the table above. At the base
constants, `overrun` reports `NoOverlappingAdmit` as holding — but it holds
**vacuously**: the term budget at `MaxTerm = 3` with a manifest chunk is too
small to reach a takeover at all, so no two watermark statements are ever in
flight to compare. `base5`, `noovr0`, `ovr0` and `vac0` exist for that reason.
`vac0` refutes `NeverAdmits` at the `ovr0`/`noovr0` constants, which proves
admissions are reached there; at those constants the cap-enforced run holds
(799,611 states) and the cap-overrun run is violated. That pair, not the
`overrun` row, is the evidence about the takeover instant.

### Z3 — clock-skew bound

| Check | Result |
|---|---|
| real skew `d <= clock_skew_ns` overlaps a holder's admitted statement | `unsat` |
| real skew `d > clock_skew_ns` overlaps | `sat` |
| margin with the `+ clock_skew_ns` term dropped, any `d > 0` | `sat` |

`unsat` in the first row is a proof over all timings, for any publish timeout,
any declared bound, any expiry and any schedule — not a sample of one.

### CBMC — payload ring spans

Four assertions over `payload_compute_spans`, for every capacity in `1..64`,
every `head` up to `2^40` and every `nbytes`:

| Assertion | With the precondition | Without it |
|---|---|---|
| P1 `len1 + len2 == n` | SUCCESS | SUCCESS |
| P2 `off1 + len1 <= cap` | SUCCESS | SUCCESS |
| P3 `off2 + len2 <= cap` | SUCCESS | **FAILURE** |
| P4 spans disjoint | SUCCESS | **FAILURE** |

Witness for the failures: `cap = 22`, `head = 15`, `tail = 0` — so seven bytes
are free — and `n = 1152921504606846983`. The second span runs past the end of
the buffer. Who establishes the precondition is the subject of the next model.

### Ring capacity (`RingCapacity.tla`)

The CBMC harness proves the span arithmetic *given*
`payload_free_bytes(head, tail, capacity) >= nbytes`. The kernel cannot check
that — it never reads a tail — so the obligation sits entirely on the host's
reservation paths. This model is those paths: `prepare_step`, the eager safety
net's `available_capacity` / `reserve_one`, `reserve_record` with its
upper-bound reservations and reclaims, a single asynchronous CUDA stream, and a
drain thread that may release any staging-sized prefix of published tasks at
any moment, or never.

Two obligations are checked in every reachable state, for the producer at the
head of the stream (it may run in any state, so a state where it does not fit
is a reachable overrun):

- `PayloadFits` — the CBMC precondition: `(gpuHead - tail) + alloc <= Cap`.
- `TaskFits` — the same for publication slots. `task_publish` stores to slot
  `seq % task_cap` unconditionally (`task_ring.cuh:78-85`); publishing into a
  slot the drain has not cleared destroys an unread publication.

Three supporting invariants state the argument the code comments make:
`Covered` (the GPU never runs ahead of what the CPU reserved), `NoWrap` (the
CPU's own accounting never claims more than the ring holds — if it does,
`cap - (head - tail)` underflows in `uint64` and every later capacity check
passes unconditionally) and `Exact` (every reservation is eventually consumed).

Base constants: `Cap = 4`, `Staging = 4`, `TaskCap = 4`, `MaxBytes = 3`,
`MaxHooks = 2`, `MaxSteps = 4`, in `PAYLOAD_ALIGN` units. Every quantity the
protocol compares is a multiple of `PAYLOAD_ALIGN`, so the unit abstraction is
exact.

| Config | Differs from base | Invariants | Verdict | States |
|---|---|---|---|---|
| `base` | adapter path as shipped | `Safety` `Covered` `NoWrap` `Exact` | **holds** | 265,955 / 101,358 |
| `wide` | `Cap 6`, `Staging 5`, `MaxSteps 5` | `Safety` `Covered` `NoWrap` `Exact` | **holds** | 1,707,055 / 543,119 |
| `taskover` | `MaxHooks 3 > TaskCap 2` | `Safety` | `Safety` violated (8) | 79,575 / 50,754 |
| `taskover_payload` | `MaxHooks 3 > TaskCap 2` | `PayloadFits` | **holds** | 5,036,539 / 1,825,901 |
| `taskover_nowrap` | `MaxHooks 3 > TaskCap 2` | `NoWrap` | `NoWrap` violated (6) | 13,639 / 10,183 |
| `eager` | `NeedsEager` | `Exact` | `Exact` violated (4) | 178 / 176 |
| `eager_nowrap` | `NeedsEager` | `NoWrap` | `NoWrap` violated (4) | 211 / 208 |
| `eager_payload` | `NeedsEager`, `MaxSteps 5` | `PayloadFits` | `PayloadFits` violated (8) | 4,730 / 3,614 |
| `prefix` | `PrefixMismatch` | `PayloadFits` | `PayloadFits` violated (8) | 4,285 / 3,400 |
| `rec_base` | record path as shipped | `Safety` `Covered` `NoWrap` `Exact` | **holds** | 9,795,483 / 2,957,562 |
| `rec_wide` | record, `Cap 6`, `Staging 5` | `Safety` `Covered` `NoWrap` `Exact` | **holds** | 44,914,022 / 13,756,982 |
| `rec_taskover` | record, `MaxHooks 3 > TaskCap 2` | `Safety` `Covered` `NoWrap` `Exact` | **holds** | 35,452,350 / 13,328,511 |
| `rec_gate` | record, `GateMismatch` | `Exact` | `Exact` violated (5) | 3,343 / 3,340 |
| `rec_gate_safety` | record, `GateMismatch`, `MaxSteps 5` | `Safety` | `Safety` violated (14) | 10,845,115 / 6,597,762 |

**The shipped paths discharge the obligation.** On the adapter path with no
knob turned, and on the generic-record path with reclaims, `PayloadFits`,
`TaskFits`, `Covered`, `NoWrap` and `Exact` all hold. The CBMC precondition is
established on every path the code takes — under the assumptions below, and
with one exception.

**Exception: more hooks per step than `task_cap` (`taskover`).** This needs no
override and no broken contract. `prepare_step` returns `STEP_OVERSIZED` when
`num_hooks > task_cap` (`ring_engine_py.cu:422-427`), which turns on the eager
safety net. The safety net checks only *payload* space
(`point.py:321-323`, `available_capacity()` at `ring_engine_py.cu:634-638`),
then `reserve_one` claims one task entry each without checking the task ring
(`ring_engine_py.cu:643-646`). Payload room is plentiful — the step was
oversized by count, not bytes — so no hook flushes, and the stream carries more
producers than there are publication slots. If the drain has not cleared the
oldest slot when the `task_cap + 1`-th producer runs, that producer overwrites
an unread publication. The payload obligation still holds on this path
(`taskover_payload`); the task obligation does not, and the CPU's task
accounting wraps (`taskover_nowrap`), after which the task-ring check in
`prepare_step` passes unconditionally. `estimate.py:816-826` describes exactly
this configuration as "capture keeps working". It does only while the drain
keeps up. The default `task_cap` is 65,536 (`engine.py:102`), so reaching this
takes a ring configured with fewer task entries or a very wide hook selection
— both of which the estimator supports and reports on.

**The obligation rests on three assumptions the code does not check.** Each
was given a knob; each knob, turned alone, breaks `PayloadFits`.

- *An adapter overriding `_spec_needs_eager` (`eager`).* `commit_step`
  reserves the whole step through `prepare_step` and *then* sets
  `force_eager`, so every hook also `reserve_one`s its own bytes. The step
  reservation is never consumed. In the counterexample a single such step with
  one 3-unit hook on a 4-unit ring reserves 3, finds 1 free, flushes, and
  `reserve_one`s 3 more without re-checking: `cpuHead - tail = 6 > Cap`. The
  next ordinary step's `cap - (head - tail)` underflows, the fast path passes,
  and the GPU writes 5 units into a 4-unit ring — eight states in all. No
  shipped adapter overrides `_spec_needs_eager` (`base.py:150-155` documents it
  as the extension point for dynamic-shape backends), so this is latent.
- *The device row count a prefix producer reads never exceeds the CPU
  `actual_q_len` the step was sized with (`prefix`).* `plan_step` sizes a
  strip-eligible hook from `ctx.actual_q_len` (`base.py:359-366`); the kernel
  sizes its allocation from `*row_count_dev_ptr`, clamped only to
  `nbytes_upper` (`producer.cu:234-252`). Any disagreement in the device's
  favour writes past the reservation. The contract is stated in
  `point.py:135-148`; nothing enforces it.
- *A device gate never skips an occurrence the host reserved (`rec_gate`).*
  `integration-api-v1.md:370-374` requires the integration to apply the gate
  before reserving. If it does not, the reservation is never consumed or
  reclaimed — a skipped producer publishes nothing for `account_record_task`
  to see — and the leak compounds through `reserve_record`'s flushed branch
  into the same underflow.

**Every flushed branch reserves without re-checking.** `prepare_step`
(`ring_engine_py.cu:440-446`), `reserve_record` (`:505-516`) and the safety
net's middle branch (`point.py:326-329`) all flush and then reserve
unconditionally. That is sound exactly when `Exact` holds, because a full flush
then leaves `cpuHead - tail` at zero and the request is already known to fit
the effective capacity. It is also why each assumption above fails as badly as
it does: once any reservation leaks, a flushed branch can push the CPU's
accounting past `Cap`, and the unsigned subtraction removes backpressure for
the rest of the run instead of reporting the leak. A re-check after the flush
that throws when the request still does not fit would turn the two leak-driven
overruns (`eager`, `rec_gate`) into loud errors. It would not catch `prefix`:
there the CPU's accounting is right and the device writes past it, so a fix has
to bound the device's allocation by what was reserved. Today its only clamp is
`nbytes_upper`, the padded tensor size — and under CUDA-graph capture that
bound is baked into the graph, which is why this is not a one-line change.

Vacuity probes, each refuted as intended at the base constants: the payload
ring fills (`vac_full`), the task ring fills (`vac_tasks`), the flushed branch
is taken (`vac_flushed`), `STEP_OVERSIZED` is returned (`vac_oversized`), the
safety net reserves (`vac_eager`), and on the record path the ring fills, the
flushed branch is taken and a reclaim is credited (`vac_rec_*`).

| Config | Guard | Refuted at | States |
|---|---|---|---|
| `vac_full` | `NeverFull` | 7 states | 2,272 / 1,654 |
| `vac_tasks` | `NeverTasksFull` | 15 states | 53,801 / 27,161 |
| `vac_flushed` | `NeverFlushedPath` | 7 states | 699 / 540 |
| `vac_oversized` | `NeverOversized` | 3 states | 71 / 68 |
| `vac_eager` | `NeverEagerReserve` | 4 states | 234 / 231 |
| `vac_rec_full` | `NeverFull` | 9 states | 32,184 / 23,015 |
| `vac_rec_flushed` | `NeverFlushedPath` | 7 states | 4,321 / 3,340 |
| `vac_rec_reclaim` | `NeverReclaims` | 5 states | 1,126 / 1,123 |

## Limitations

Read this section before quoting any result above.

**The two-host clock skew is assumed inside the TLA+ model, not verified by
it.** `PublisherLease.tla` runs with `SKEW = 0` and a single server clock
(`now`). It therefore takes the skew bound as given and checks the rest of the
protocol on top of it. The bound itself is closed separately, by
`z3/clock_skew.py`, against the derivation at
`docs/catalog-descriptor-key.md:365-372`. The two results are independent; the
TLA+ run does not corroborate the Z3 one, or the reverse.

**The catalog tables are always read linearizably in the lease model, except
where a config says otherwise.** Every deciding read in `PublisherLease.tla`
sees every accepted row whenever `Linearizable = TRUE`, which is every config
except the `nonlin_*` pair. That is the single load-bearing assumption of the
safety argument. It models `select_sequential_consistency=1`
(`clickhouse_client.cpp:107`) being honoured; it does not check that it is.

**The non-linearizable store model is a generous over-approximation.** With
`Linearizable = FALSE` a deciding read may observe any subset of the in-flight
inserts on top of what has replicated. Real ClickHouse replicas are not that
adversarial. The `nonlin_*` counterexamples are therefore "this is what you are
exposed to if the setting is not in force", not "this exact interleaving will
occur". Their value is the shape of the failure and which obligations fall, not
a probability.

**`overrun` holding at the base constants proves nothing.** See the vacuity
note above: at `MaxTerm = 3` the takeover race is out of budget, so obligation 1
holds because its subject is unreachable. `base5`, `noovr0` and `ovr0` are the
configs that carry that claim.

**Every result is bounded.** TLC explores the state space cut off by the
constants in each `.cfg` — at most three lease ids, five or eight time steps,
two manifest chunks, two or three concurrent actors. An invariant reported as
holding holds *within that bound*. `CeilingNeverBinds` and the vacuity guards
probe whether a specific bound hid behaviour; they do not turn a bounded check
into a proof. CBMC's result is likewise bounded at capacity 64 and `head < 2^40`.
Only the Z3 result is unbounded.

**The ring model trusts the kernel to be the arithmetic it is modelled as.**
`RingCapacity.tla` treats a producer as one atomic step that allocates
`align_up(actual)`, publishes, and advances both heads. The CUDA memory
ordering that makes that true across blocks and to the host (the last-block
join, the system-scope release at `task_ring.cuh:84`) is assumed, not checked;
none of these tools reaches it. It also assumes one CUDA stream: a producer
launched on a stream other than the one `flush_and_wait` synchronises would be
outside what the flush waits for.

**The drain in the ring model can stall forever.** That is deliberate: a drain
held up by a full staging buffer or a slow sink is realistic, and the capacity
check exists precisely so that correctness does not depend on the drain keeping
up. It means a counterexample shows an overrun is *reachable*, not how often it
occurs under a healthy drain.

**Two actors, not N.** The catalog TLA+ models run with two or three concurrent
processes. A protocol bug that needs four simultaneous claimants would not be
found.

**Liveness is not modelled.** Every invariant here is a safety property. The
specs say nothing about a publisher making progress, and the contested-head
quarantine — a deliberate liveness cost — is not measured.

**The specs model the code as of the revision they were written against.** They
are not regenerated from the source and nothing checks that they still match it.
Re-read the `.tla` header comments against the cited lines before trusting a
result after the catalog code changes.
