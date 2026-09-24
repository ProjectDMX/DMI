# Formal specifications

Machine-checked models of four pieces of DMI whose correctness arguments are
currently carried by prose: the catalog version allocator, the publisher lease
and fenced publish protocol, the two-host clock-skew bound in the fence margin,
and the payload ring's span arithmetic.

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

Both TLA+ models are written against the **code**, not the prose. Where the two
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
the buffer.

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

**Two actors, not N.** Both TLA+ models run with two or three concurrent
processes. A protocol bug that needs four simultaneous claimants would not be
found.

**Liveness is not modelled.** Every invariant here is a safety property. The
specs say nothing about a publisher making progress, and the contested-head
quarantine — a deliberate liveness cost — is not measured.

**The specs model the code as of the revision they were written against.** They
are not regenerated from the source and nothing checks that they still match it.
Re-read the `.tla` header comments against the cited lines before trusting a
result after the catalog code changes.
