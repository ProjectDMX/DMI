# Formal specifications

Machine-checked models of five pieces of DMI whose correctness arguments are
currently carried by prose: the catalog version allocator, the publisher lease
and fenced publish protocol, the lease *lifecycle* inside
`CaptureStorageService`, the clock-skew bounds in the fence margin and the
default start wait, and the payload ring's span arithmetic.

Nothing here is built, imported or executed by DMI. The specs are checked by
hand with the commands below; none of them runs in CI.

```text
specs/
├── tla/    TLA+ models, checked with TLC
├── z3/     SMT encoding, checked with Z3
└── cbmc/   C++ harness, checked with CBMC
```

Everything below was checked against `fix/lease-recovery` @ `2b74d14`.

## What each spec models

| Spec | Models | Source of truth |
|---|---|---|
| `tla/VersionAllocator.tla` | the sole-claimant version allocation loop: floor read, jittered candidate, claim INSERT, singleton read-back, retry | `native/csrc/catalog/version_allocator.cpp:49-89`, header claim at `version_allocator.h:5-7`, watermark publish at `native/csrc/catalog/catalog_writer.cpp:587-620`, `clickhouse_client.cpp:107` |
| `tla/PublisherLease.tla` | the lease claim/renew/release protocol and the fenced publish: `claim_with_rival`, `head()`, `fence()`, `fence_eval()`, `reject_live()`, and `publish_snapshot`'s manifest chunks and watermark INSERT | `native/csrc/catalog/lease_coordinator.cpp:83-247`, `native/csrc/catalog/catalog_writer.cpp:148-158,478-640`, `clickhouse_client.cpp:107`, `docs/catalog-descriptor-key.md:290-360,461-560`, `src/dmi/storage/capture/clickhouse_lease.py:83-92,198-210` |
| `tla/LeaseLifecycle.tla` | the lease lifecycle *above* that protocol: the lease thread, the quarantine window, the `2 x TTL` latch, the start wait and the spool sweep | `native/csrc/catalog/storage_service.cpp:65-67,104-175,177-207,278-400,433-437,588-628,630-640,642-670,672-721,723-739,762-781`, `catalog_writer.cpp:243-253,268-274,285-293,296-310`, `lease_coordinator.cpp:45-55,57-68,70-81,220-247`, `indexer.cpp:258`, `src/dmi/storage/native_capture.py:139-145,215-219` |
| `z3/clock_skew.py` | two obligations: the two-host derivation behind the fence margin `publish_timeout_ns + clock_skew_ns`, and the default start wait `lease_ttl_s + publish_timeout_s + clock_skew_s` | the SQL at `docs/catalog-descriptor-key.md:352-362` (emitted by `catalog_writer.cpp:583`), the derivation at `:365-372`, the cap at `catalog_writer.cpp:519`; for the start wait, `native_capture.py:215-219`, `storage_service.cpp:642-670`, `lease_coordinator.cpp:138-153,225-233` |
| `cbmc/payload_ring_span.cpp` | `payload_compute_spans` and its stated precondition | `native/csrc/ring/payload_ring.cuh:44-86` |

All three TLA+ models are written against the **code**, not the prose. Where
the two disagree, the spec follows the code and a comment in the `.tla` says so.

`LeaseLifecycle.tla` sits on top of `PublisherLease.tla` rather than beside it:
it abstracts `LeaseCoordinator` to its contract (*a claim presenting lease id L
is admitted iff the head row is dead or is L, refused with `kHeld` otherwise,
and may return an unknown outcome*) and models the service around it. That
contract is what `PublisherLease.tla` discharges. See **Limitations**.

## Getting the tools

```sh
# TLC -- a single jar, no install
curl -fsSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar

# Z3 (Python bindings include the solver)
pip install z3-solver

# CBMC (Debian/Ubuntu; or grab a release from github.com/diffblue/cbmc)
apt-get install cbmc
```

Checked with TLC 2.19 on Java 21, Z3 4.13 (`z3-solver` 4.13.x) and CBMC 5.95.
Any recent version of each should do; nothing here relies on a version-specific
feature.

## Running the checks

### TLA+

Every `.cfg` in `specs/tla/` is one model: one set of constants, one invariant.
Run any of them from `specs/tla/`.

**The allocator configs need `-deadlock` on the command line.**
`VersionAllocator.tla` has no stutter step: once every allocator reaches
`published` there is no next state, and TLC reports `Error: Deadlock reached.`
and stops — on `nocap_distinct` that happens after 97 of the 836 distinct
states, so without the flag the run *looks* clean for two seconds and has
checked almost nothing. `-deadlock` turns deadlock checking off (that is what
the flag does, despite the name), and the run completes. The
`PublisherLease_*` and `LeaseLifecycle_*` configs do not need it: they carry
`CHECK_DEADLOCK FALSE` in the `.cfg` itself.

```sh
cd specs/tla

# version allocator -- note -deadlock
java -XX:+UseParallelGC -Xmx3g -cp /path/to/tla2tools.jar tlc2.TLC \
     -workers 4 -deadlock -config lin_distinct.cfg VersionAllocator.tla

# publisher lease
java -XX:+UseParallelGC -Xmx3g -cp /path/to/tla2tools.jar tlc2.TLC \
     -workers 4 -config PublisherLease_believers.cfg PublisherLease.tla

# lease lifecycle
java -XX:+UseParallelGC -Xmx6g -cp /path/to/tla2tools.jar tlc2.TLC \
     -workers 4 -config LeaseLifecycle_O3_selflatch.cfg LeaseLifecycle.tla
```

`PublisherLease.cfg` is the base model — the protocol exactly as shipped, with
the combined `AllSafety` invariant. Each `PublisherLease_*.cfg` turns a single
knob away from it or swaps in a single invariant; its header comment says which.
`LeaseLifecycle_*.cfg` is grouped by obligation: `O1_*` renewal, `O2_*`
quarantine, `O3_*` the latch, `O4_*` the start wait, `O5_*` the spool sweep, and
`vac_*` the vacuity guards.

The largest runs (`believers`, `holderssafe`, `overrun`, `nonlin_fence`,
`PublisherLease.cfg`) generate 20M-61M states and take roughly 15-60 minutes on
four workers with a 3 GB heap. Every `LeaseLifecycle` run finishes in under two
minutes. The rest finish in seconds.

### Z3

```sh
python3 specs/z3/clock_skew.py
```

Prints one line per check and exits non-zero if any result differs from the
expected one. Six checks, no arguments, a couple of seconds.

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

State counts are TLC's own, as `generated / distinct`. Counterexample lengths,
in parentheses, are the number of states in the trace TLC printed, including
the initial state.

**State counts for runs that completed are exact and reproducible.** Every
completed run in the tables below reproduced its previously recorded distinct
count to the state when re-run against `2b74d14`.

**State counts and trace depths for refuted runs are not reproducible.** TLC
stops as soon as any worker hits the violation, so both the count and the
trace length depend on the worker count and on scheduling. The figures here
were recorded with 2 or 4 workers; `NoOrphanManifestRows` has been seen at both
9 and 14 states on the same config. Only the verdict is stable for a refuted
run, not the number beside it.

### Version allocator (`VersionAllocator.tla`)

All configs use `Attempts = 3`, `MaxSpread = 1`. Run with `-deadlock`.

| Config | Store | Allocators | Invariant | Verdict | States |
|---|---|---|---|---|---|
| `lin_distinct` | linearizable | 3 | `Distinct` | **holds** | 669,421 / 407,083 |
| `nocap3_distinct` | linearizable | 3 | `Distinct` | **holds** | 1,094,245 / 689,368 |
| `nocap3_ceiling` | linearizable | 3 | `CeilingNeverBinds` | **holds** | 1,094,245 / 689,368 |
| `nocap_distinct` | linearizable | 2 | `Distinct` | **holds** | 1,117 / 836 |
| `nocap_ceiling` | linearizable | 2 | `CeilingNeverBinds` | **holds** | 1,117 / 836 |
| `lin_ceiling` | linearizable | 3 | `CeilingNeverBinds` | violated (17) | 39,417 / 23,193 |
| `lin_solerow` | linearizable | 3 | `SoleRow` | violated (6) | 77 / 61 |
| `lin_floor` | linearizable | 3 | `FloorMonotonic` | violated (8) | 597 / 390 |
| `lin_publish` | linearizable | 3 | `NoPublishRefused` | violated (9) | 834 / 531 |
| `lin_budget` | linearizable | 3 | `NoBudgetExhaustion` | violated (17) | 39,376 / 23,154 |
| `nocapf_distinct` | shared stale frontier | 2 | `Distinct` | **holds** | 2,886,043 / 639,601 |
| `nocapf_ceiling` | shared stale frontier | 2 | `CeilingNeverBinds` | **holds** | 2,886,043 / 639,601 |
| `frontier_distinct` | shared stale frontier | 2 | `Distinct` | **holds** | 2,232,679 / 456,499 |
| `ec_distinct` | eventually consistent | 2 | `Distinct` | violated (9) | 1,464 / 724 |
| `nocap_ec` | eventually consistent | 2 | `Distinct` | violated (9) | 782 / 409 |

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
| `holders` | `AtMostOneHolder` | violated (12) | 10,865 / 6,280 |
| `noovr0` (empty refs, cap enforced) | `NoOverlappingAdmit` | **holds** | 1,761,711 / 799,611 |
| `ovr0` (empty refs, `AllowOverrun`) | `NoOverlappingAdmit` | violated (15) | 64,379 / 34,191 |
| `overrun` (`AllowOverrun`, 1 chunk) | `NoOverlappingAdmit` | **holds** (vacuously) | 20,855,772 / 9,873,900 |
| `selfrace` (shared writer, no lock) | `WatermarkMonotonic` | violated (29) | 11,890,931 / 6,074,002 |
| `selfracelocked` (shared writer, lock) | `AllSafety` | **holds** | 947,826 / 524,274 |
| `orphans` | `NoOrphanManifestRows` | violated (14) | 43,456 / 23,086 |
| `chunks2_orphans` (2 chunks) | `NoOrphanManifestRows` | violated (14) | 33,994 / 18,091 |
| `chunks2_prefix` (2 chunks) | `OrphansArePrefixes` | **holds** | 1,892,754 / 909,324 |
| `chunks2` (2 chunks) | `AllSafety` | **holds** | 1,892,754 / 909,324 |
| `stalepid` (reused `publish_id`) | `AllSafety` | **holds** | 15,073,338 / 7,368,006 |
| `nonlin_fence` (non-linearizable) | `AtMostOneFenceable` | **holds** | 61,324,208 / 12,215,084 |
| `nonlin_admit` (non-linearizable) | `NoOverlappingAdmit` | violated (27) | 5,093,065 / 1,273,001 |
| `nonlin_all` (non-linearizable) | `AllSafety` | violated (27) | 5,176,956 / 1,293,712 |

`nonlin_fence` is the one thing that survives a non-linearizable store: at most
one publisher can *pass the fence* at a time even then. What it does not
survive is the admission window — `nonlin_admit` — so the fence being sole does
not make the publish sole. Read the three `nonlin_*` rows together.

#### Vacuity guards

Each of these is an invariant we *want* refuted: the counterexample is the proof
that the model reaches the state the safety invariants are quantified over. An
invariant that holds because its subject is unreachable proves nothing.

| Config | Guard | Refuted at | States |
|---|---|---|---|
| `vac_publish` | `NeverPublishes` | yes (16) | 74,958 / 38,597 |
| `vac_fence` | `NeverFences` | yes (5) | 169 / 98 |
| `vac_admit` | `NeverAdmits` | yes (15) | 49,953 / 26,385 |
| `vac_contested` | `NeverContested` | yes (7) | 745 / 417 |
| `vac_bothadmit` | `NoBelieverDuringAdmit` | yes (20) | 335,283 / 167,905 |
| `vac0` | `NeverAdmits` at the `ovr0` / `noovr0` constants | yes (7) | 763 / 434 |

`vac0` is the one that matters for reading the table above. At the base
constants, `overrun` reports `NoOverlappingAdmit` as holding — but it holds
**vacuously**: the term budget at `MaxTerm = 3` with a manifest chunk is too
small to reach a takeover at all, so no two watermark statements are ever in
flight to compare. `base5`, `noovr0`, `ovr0` and `vac0` exist for that reason.
`vac0` refutes `NeverAdmits` at the `ovr0`/`noovr0` constants, which proves
admissions are reached there; at those constants the cap-enforced run holds
(799,611 states) and the cap-overrun run is violated. That pair, not the
`overrun` row, is the evidence about the takeover instant.

### Lease lifecycle (`LeaseLifecycle.tla`)

Time is in **ticks** of `TTL/6`, the lease thread's own period
(`storage_service.cpp:65-67`), so `ttl/6`, `ttl/3` and `2*ttl` are all exact
integers. Most configs run one service with `TTL = 6`.

#### O1 — renewal keeps the lease alive

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `O1_tries` | `ThreeTriesFit` | **holds** | 11 / 9 |
| `O1_tries12` (`TTL 12`) | `ThreeTriesFit` + `FourTriesFit` | **holds** | 6 / 6 |
| `O1_tries60` (`TTL 60`) | `ThreeTriesFit` + `FourTriesFit` | **holds** | 6 / 6 |
| `O1_tries5` | `FiveTriesFit` | violated (constant-level) | — |
| `O1_clean` | `NoPhantomLease` | **holds** | 206 / 126 |
| `O1_cut` (ClickHouse cut) | `NoPhantomLease` | **holds** | 7,795 / 3,879 |
| `O1_late` (`MaxLate 1`) | `NoPhantomLease` | **holds** | 22,268 / 9,880 |
| `O1_absorb` | `OneFailureAbsorbed` | violated (9) | 148 / 90 |
| `O1_skip` (`AllowSkipPublish`) | `NoPhantomLease` | violated (19) | 344 / 196 |

`O1_tries` / `O1_tries5` together pin the comment at
`storage_service.cpp:631-632` — *"which leaves two more tries"*. Exactly **four**
lease-thread wakes fall between the instant the renewal falls due
(`last_renew + ttl/3`) and the instant the row dies (`last_renew + ttl`), at
every phase offset: three fit, four fit, five do not. The arithmetic is
correct and conservative. `FiveTriesFit` is refuted at the constant level — the
arithmetic is decided before any state is explored, so TLC reports no state
count.

Two caveats on that comment, neither a model result:

* The four wakes are room for a renewal that runs **late**, not for one that
  **fails**: a failed renewal costs the lease at once, whatever the cause. A
  refusal drops it in the coordinator (`reject_live`,
  `lease_coordinator.cpp:226`, or the failed read-back at `:128`). A transport
  error, timeout or parse error is a `ClickHouseError` and takes the
  `std::exception` path at `catalog_writer.cpp:285-292`, which quarantines on
  the *first* one. No renewal failure is retried under the same lease.
  `O1_absorb` is that, refuted as expected.
* `O1_skip` is a real defect, not a modelling artefact. See below.

**`O1_skip` — a lease can lapse under a service that still believes it holds
it.** `storage_service.cpp:435-437` bumps `last_renew_ns_` when
`indexed_packs > 0 || skipped_packs > 0`, with the comment *"a publish renews
the lease"*. But `indexer.cpp:258` gates the whole publish on
`!all_rows.empty() || !indexed.empty()`: an index pass whose packs were all
already committed returns `skipped_packs > 0` and never calls
`renew_for_publish()`. The bump pushes the next renewal out by `ttl/3` without
anything having touched the row. The trace, at `TTL = 6` so `ttl/3 = 2` ticks:

1. Tick 0. `s1` starts, claims lease 1. Its row expires at tick 6.
2. Tick 1. A cycle indexes a batch whose packs were all already committed.
   `skipped_packs > 0`, nothing is published, `last_renew_ns_ := 1`. The
   renewal is now not due until tick 3.
3. Tick 3. The lease thread wakes, finds `now - last_renew_ns_ < ttl/3`, and
   returns without renewing. Another skip-bump lands in the same tick:
   `last_renew_ns_ := 3`. Due date moves to 5.
4. Tick 5. Same again: the lease thread stands down, a third skip-bump sets
   `last_renew_ns_ := 5` and the due date to 7.
5. Tick 6. The row expires. `held_lease() != nullptr` is still true and
   `phase = run`, so `NoPhantomLease` falls at depth 23.

Because the bumps arrive at exactly the renewal interval, the lease thread is
starved indefinitely — it never once reaches its own due test as true.
**Reachability caveat:** this needs skip-only passes landing close enough
together to keep resetting the clock, and no single concrete deployment
scenario chaining them was demonstrated. The state machine reaches it; a
production trace has not been shown. The guard should test `indexed_packs > 0`
alone, or the indexer should report whether it actually published.

#### O2 — the quarantine window

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `O2_quar` | `QuarantineOutlastsItsRow` | **holds** | 7,795 / 3,879 |
| `O2_skew` (`Skew 1`) | `QuarantineOutlastsItsRow` | violated (27) | 1,632 / 854 |
| `O2_reuse` (`ReuseLid`) | `QuarantineOutlastsItsRow` | violated (27) | 1,422 / 765 |
| `O2_selfref` | `NoSelfRefusal` | violated (29) | 1,856 / 1,001 |

`QuarantineTakesNothing` holds everywhere it is checked: a quarantined writer
takes no claim at all, not even with a fresh `lease_id`
(`storage_service.cpp:678-684`).

With `Skew = 0` the window at `catalog_writer.cpp:273` does outlast the row it
dropped. With one tick of skew it does not — the window is
`now_monotonic_ns() + lease_ttl_ns` on the **local** steady clock, while the
row's expiry is read from whichever replica answers, which may report it live a
skew later. `O2_reuse` is a counterfactual: presenting the *dropped* lease id
after the window would be worse still, which is why the fresh-id rule at
`lease_coordinator.cpp:45-55` is right. `O2_selfref` is the price of that rule:
`reject_live`'s `claimants == 1` exemption cannot recognise a fresh id, so a
writer can be refused by its own dropped row.

#### O3 — the `2 x TTL` latch

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `O3_rival` (persistent rival) | `NeverLatches` | violated (68) | 110,815 / 54,130 |
| `O3_rivaljust` | `NoFalsePositiveLatch` | **holds** | 172,656 / 81,516 |
| `O3_stops2` (rival stops < 2 TTL) | `NeverLatches` | **holds** | 355,934 / 170,480 |
| `O3_false` (`Skew 0`) | `NoFalsePositiveLatch` | **holds** | 355,934 / 170,480 |
| `O3_falsenocut` (`Skew 0`, no cut) | `NoFalsePositiveLatch` | **holds** | 1,743 / 1,166 |
| `O3_selflatch` (`Skew 1`, no rival) | `NoFalsePositiveLatch` | violated (65) | 13,938 / 9,559 |
| `O3_selflatch_latch` (`Skew 1`, no rival) | `NeverLatches` | violated (68) | 16,865 / 11,614 |
| `O3_stops` | `NeverLatches` | **holds** — but see below | 821 / 563 |

The latch is justified when it is meant to be: a rival that keeps renewing does
latch (`O3_rival`), and it latches *legitimately* — `O3_rivaljust` shows the
rival really did hold a live row at every instant of the window. A rival that
stops inside two TTLs does **not** latch (`O3_stops2`, 170,480 states). With no
skew, no false positive is reachable at all (`O3_false`).

**`O3_selflatch` — the latch fires with no rival in existence.** `Foreign =
FALSE`, `MaxCuts = 0`: no other publisher, no ClickHouse outage, only bounded
request timeouts and `Skew = 1`. The service latches
`"publisher lease held by another publisher for over 2 x TTL"` against itself,
at depth 70. Two independent causes, both needed:

1. The quarantine window at `catalog_writer.cpp:273` is measured on the local
   steady clock with **no skew allowance**, while the row it dropped is read
   from a replica that may still report it live. `2b74d14` gave the start wait
   a `+ clock_skew_s` term; the quarantine window did not get one. The writer
   therefore comes out of quarantine and is refused by its own corpse.
2. `storage_service.cpp:731` tests `now - held_elsewhere_since_ns_ >= 2 * ttl`,
   and `held_elsewhere_since_ns_` is set on the **first** refusal (`:726`) and
   cleared only by a **successful** claim (`:707`). The justification written
   directly above it at `:728-730` — *"a refusal that has lasted 2 x TTL is a
   publisher that means to stay"* — requires an unbroken **run** of refusals.
   The code measures elapsed time since the first one instead, and nothing in
   between resets it.

The trace, at `TTL = 6` and `Skew = 1`:

1. Tick 0. `s1` starts, claims lease 1, sweeps, runs. Its row expires at 7.
2. Tick 2. A renewal returns an unknown outcome. `renew_for_publish` takes the
   `std::exception` path and quarantines: lease 1 is dropped locally with no
   tombstone, the window is set to `now + ttl` = tick 8. The statement had in
   fact landed, so the server row now lives to tick 9.
3. Tick 8. The window ends and `ensure_publisher_lease` claims with a fresh
   lease id. The row from step 2 is dead on the true clock at 8 but reads live
   on a replica one tick behind, so the claim comes back `kHeld`.
   `held_elsewhere_since_ns_ := 8`. **This is the only refusal in the trace.**
4. Tick 9. The row really does expire. The refusal run is broken here
   (`runBroken := TRUE`), which is exactly what `NoFalsePositiveLatch` watches.
5. Ticks 9 and 15. Two further acquire attempts return unknown outcomes and
   take the `catalog_writer.cpp:296-310` path, quarantining again each time.
   Neither is a refusal, and neither is a success — so `:707` never runs and
   `held_elsewhere_since_ns_` stays at 8. (The second of these does leave a
   live row on the server, expiring at 22.)
6. Tick 21. The last window ends, the claim is refused by that row, and
   `21 - 8 = 13 >= 2 * ttl = 12`. `latch_failure` fires with
   *"publisher lease held by another publisher for over 2 x TTL"*.

`Foreign = FALSE` throughout: no other publisher exists in the model. The only
thing that ever held the row was the service itself, and the single refusal it
is latching on happened thirteen ticks earlier and lasted one tick.

Suggested fixes, one for each: add `clock_skew_ns` to the quarantine window at
`catalog_writer.cpp:273`, matching what the start wait now does; and reset
`held_elsewhere_since_ns_ = 0` whenever a claim is *not* refused with `kHeld`
— on a quarantine, on a transport error, on anything that breaks the run —
rather than only on success. Either one alone kills this trace; both are
independently wrong.

#### O4 — the start wait

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `O4_same` (`PredTTL == TTL`) | `StartAlwaysSucceeds` | **holds** | 98 / 66 |
| `O4_skew` (`Skew 1`) | `StartAlwaysSucceeds` | **holds** | 85 / 59 |
| `O4_bigger` (`PredTTL > TTL + PT`) | `StartAlwaysSucceeds` | violated (16) | 16 / 16 |
| `O4_compose` | `NoConcurrentHolder` | **holds** | 11,809 / 5,872 |

The start wait composes correctly: `O4_compose` finds no instant at which two
parties both believe they hold the lease. The `O4_same` / `O4_bigger` pair
reproduces, inside the state machine, the threshold the Z3 script proves in the
reals — a predecessor whose TTL exceeds the successor's `ttl + publish_timeout`
outlasts the wait.

#### O5 — the spool sweep

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `O5_refused` | `RefusedStartNeverSweeps` | **holds** | 15 / 14 |
| `O5_cosweep` | `SweepOnlyWhenAlone` | violated (22) | 4,498 / 2,653 |

A start that is refused the catalog never touches the spool — `O5_refused`
holds, which is `test_a_start_refused_the_lease_leaves_the_spool_unswept`
generalised over the model.

**`O5_cosweep` — two processes sweep one spool at once.** No ClickHouse cut is
needed, and no skew: `MaxCuts = 0`, `Skew = 0`. The trace, at `TTL = 6`:

1. Tick 0. `s1` starts, claims lease 1 (row expires at tick 6), sweeps the
   spool, and enters `run`. It is now the live sink, writing `.open` packs.
2. Tick 0. `s2` starts on the same spool and is refused the lease, so it begins
   polling inside `acquire_lease_at_start`.
3. Tick 2. `s1`'s lease thread takes an unknown-outcome error and quarantines.
   `quarantine()` discards the lease **without** the release tombstone, so the
   server row stays live to its TTL. `s1` does not stop — it is still in
   `run`, still holding the spool open.
4. Tick 6. The row expires by itself. `s2`'s next poll succeeds and it takes
   lease 2.
5. Tick 6. `s2` proceeds past the lease to `sweep_spool_on_start` and calls
   `Recover()`, which deletes every `.open` file it does not own — including
   `s1`'s in-progress packs — and records the count in
   `state_.swept_on_start` as a success. Both services are now in `run` and
   `coSweep` is set: `SweepOnlyWhenAlone` falls at depth 23.

Nothing reports an error. `s1` keeps writing into files that have been
unlinked, and `s2`'s `swept_on_start` counts a live sink's work as recovered
debris.

This is not a new discovery so much as the **witness for the comment `2b74d14`
already weakened**. `storage_service.cpp:115-120` now says *"usually"* and
names exactly this gap: *"a holder that is quarantined has let its row lapse,
and a second process can take the lease in that gap"*. That comment is correct.
`storage_service.h:100-104` was not updated with it and still reads as the
strong form — *"It runs after the lease is taken, so a start refused the
catalog never touches the spool"* — which is true of a *refused* start and
says nothing about this one. The real fix is the spool owner lock the comment
defers.

#### Vacuity guards

Every one of these must be refuted, or the run above it proves nothing.

| Config | Guard | Refuted at | States |
|---|---|---|---|
| `vac_held` | `VacHeld` | yes (2) | 2 / 2 |
| `vac_quar` | `VacQuarantine` | yes (8) | 142 / 86 |
| `vac_recov` | `VacRecovered` | yes (26) | 1,417 / 757 |
| `vac_refus` | `VacRefusal` | yes (27) | 2,191 / 1,158 |
| `vac_rival` | `VacRivalHeld` | yes (3) | 70 / 43 |
| `vac_cut` | `VacCut` | yes (2) | 6 / 6 |
| `vac_latch` | `VacLatch` | yes (68) | 112,205 / 54,658 |
| `vac_start` | `VacStartWaited` | yes (2) | 2 / 2 |
| `vac_o4waited` | `VacStartWaited` at the `O4` constants | yes (2) | 2 / 2 |
| `vac_refusedstart` | `StartAlwaysSucceeds` at the `O5_refused` constants | yes (2) | 2 / 2 |
| `vac_stops2refusal` | `VacRefusal` at the `O3_stops2` constants | yes (29) | 4,989 / 2,775 |
| `vac_stops2rival` | `VacRivalHeld` at the `O3_stops2` constants | yes (2) | 44 / 30 |
| `vac_stopsrefusal` | `VacRefusal` at the `O3_stops` constants | **NO — holds** | 821 / 563 |

`SweepOnlyWhenAlone` is its own vacuity guard: it is refuted, so the co-sweep
state is reachable by construction. No separate guard config is shipped for it.

**One guard held, and it condemns its own run.** `LeaseLifecycle_O3_stops.cfg`
was the first attempt at *"a rival that stops inside two TTLs does not latch"*,
and it reported `NeverLatches` as holding over 563 states. `vac_stopsrefusal`
shows why: at those constants, with `MaxCuts = 0`, the rival can **never claim
the catalog at all**, so the service is never refused, so of course it never
latches. That run is vacuous and carries no weight. `O3_stops2.cfg` replaces
it — the same claim at constants where the rival does claim, does hold, and
does stop — and is proved non-vacuous by `vac_stops2refusal` and
`vac_stops2rival`. Both configs are kept here so the disclosure is checkable.

### Z3 — clock skew, two obligations

| Check | Result |
|---|---|
| real skew `d <= clock_skew_ns` overlaps a holder's admitted statement | `unsat` |
| real skew `d > clock_skew_ns` overlaps | `sat` |
| margin with the `+ clock_skew_ns` term dropped, any `d > 0` | `sat` |
| start wait, predecessor TTL `==` successor TTL | `unsat` |
| start wait, predecessor 30 s vs successor 10 s / 10 s / 2 s | `sat` |
| start wait, `Tp <= Ts + p` (the threshold) | `unsat` |
| start wait, `Tp >  Ts + p` (above it) | `sat` |

`unsat` here is a proof over all timings — for any publish timeout, any
declared bound, any expiry and any schedule — not a sample of one. All
quantities are reals: no discretisation and no bound on the magnitudes.

The second obligation discharges the change `2b74d14` made at
`native_capture.py:215-219`, which added `+ clock_skew_s` to the default start
wait. The result: **the wait outlasts a crashed predecessor iff
`predecessor_ttl <= successor_ttl + publish_timeout`**, and `clock_skew_s`
cancels out of that condition entirely. It pays for real replica skew exactly
and buys **zero** headroom against a TTL mismatch.

The hand-written caveat at `native_capture.py:139-145` is therefore true but
conservative. On the shipped Python defaults — `lease_ttl_s = 15`,
`publish_timeout_s = 5` — any predecessor TTL up to **20 s** is outlasted. The
native default TTL is 30 s (`catalog_writer.h:34`), which processes predating
these knobs used, so a restart after one of those gives up 10 s early. The
script's second start-wait check pins the same failure at a different knob set
(successor 10 s / 10 s / 2 s, so a 22 s wait against a 30 s row, 8 s short).

One clarification the caveat does not make: the skew that matters *at start* is
between ClickHouse **replicas**, not between DMI hosts. `reject_live` compares
`head.live_until_ns > head.now_ns` with both sides stamped server-side inside
one query (`lease_coordinator.cpp:138-153,225-233`), so the successor's own
clock never enters it. The relevant step is between the replica that took the
predecessor's INSERT and the replica that answers the successor's `head()`.

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

**The two-host clock skew is assumed inside the TLA+ models, not verified by
them.** `PublisherLease.tla` runs with `SKEW = 0` and a single server clock
(`now`), and `LeaseLifecycle.tla`'s `Skew` is a whole tick of `TTL/6`, not a
derivation. Both therefore take the skew bound as *given* and check the rest of
the protocol on top of it. The bound itself, and the start-wait obligation, are
closed separately by `z3/clock_skew.py`. The results are independent: the TLA+
runs do not corroborate the Z3 ones, or the reverse.

**The catalog tables are always read linearizably in `PublisherLease`, except
where a config says otherwise.** Every deciding read in `PublisherLease.tla`
sees every accepted row whenever `Linearizable = TRUE`, which is every config
except the three `nonlin_*`. That is the single load-bearing assumption of the
safety argument. It models `select_sequential_consistency=1`
(`clickhouse_client.cpp:107`) being honoured; it does not check that it is.

**The non-linearizable store model is a generous over-approximation.** With
`Linearizable = FALSE` a deciding read may observe any subset of the in-flight
inserts on top of what has replicated. Real ClickHouse replicas are not that
adversarial. The `nonlin_*` counterexamples are therefore "this is what you are
exposed to if the setting is not in force", not "this exact interleaving will
occur". Their value is the shape of the failure and which obligations fall, not
a probability.

**`LeaseLifecycle` abstracts `LeaseCoordinator` to its contract, so every
HOLDS verdict in its tables is conditional on `PublisherLease.tla` discharging
that contract.** A contested head — two rows at one term,
`lease_coordinator.cpp:227-236` — is not modelled. It can only *add* `kHeld`
refusals, so the refutations (`O1_skip`, `O2_*`, `O3_selflatch*`, `O5_cosweep`)
survive under it; the HOLDS verdicts do not stand on their own. Read them as
"holds, given the coordinator behaves as `PublisherLease.tla` says it does".

**`LeaseLifecycle`'s `ttl/6` time grain cannot see a sub-tick race.** One tick
is the lease thread's own period, which makes `ttl/6`, `ttl/3` and `2*ttl`
exact, but anything finer than a sixth of the TTL is invisible to it. The real
start poll is `ttl/10` clamped to 50-500 ms — *finer* than one tick — so a
start the model reports as refused purely at an expiry boundary would be
retried sooner in reality. In the other direction, `MaxLate = 0` means the
lease thread wakes exactly on its tick; real OS scheduling can only make it
later, which strictly reduces the number of renewal attempts in the window.
`O1_late` runs `MaxLate = 1` separately.

**Counts and trace depths for REFUTED runs are scheduling-dependent; counts
for completed runs are exact.** Stated again here because it is the most
commonly misread number in the tables. A refuted run's count tells you nothing
reproducible. A completed run's count does, and every completed run above
reproduced its recorded figure exactly when re-run against `2b74d14`.

**`overrun` holding at the base constants proves nothing.** See the vacuity
note in the `PublisherLease` section: at `MaxTerm = 3` the takeover race is out
of budget, so the obligation holds because its subject is unreachable. `base5`,
`noovr0` and `ovr0` are the configs that carry that claim. The same trap caught
`LeaseLifecycle_O3_stops`, disclosed above.

**Every result is bounded.** TLC explores the state space cut off by the
constants in each `.cfg` — at most three lease ids, five or eight time steps,
two manifest chunks, two or three concurrent actors, twenty-four ticks of lease
lifetime. An invariant reported as holding holds *within that bound*.
`CeilingNeverBinds` and the vacuity guards probe whether a specific bound hid
behaviour; they do not turn a bounded check into a proof. CBMC's result is
likewise bounded at capacity 64 and `head < 2^40`. Only the Z3 results are
unbounded.

**Two or three actors, not N.** The TLA+ models run with one to three
concurrent processes. A protocol bug that needs four simultaneous claimants
would not be found.

**Liveness is not modelled.** Every invariant here is a safety property. The
specs say nothing about a publisher making progress, and the contested-head
quarantine — a deliberate liveness cost — is not measured. `NeverLatches` is a
safety invariant about a latch being *reachable*, not a claim about recovery.

**The specs model the code as of the revision they were written against.**
They are not regenerated from the source and nothing checks that they still
match it. Re-read the `.tla` header comments against the cited lines before
trusting a result after the catalog code changes.
