# Formal specifications

Machine-checked models of the parts of DMI's native capture storage path whose
correctness arguments are otherwise carried by prose:

- the catalog version allocator;
- the publisher lease and fenced publish protocol, including #159's
  per-request deadlines;
- the lease *lifecycle* inside `CaptureStorageService`;
- the reader's pack ranking at a pinned watermark (#161);
- spool-directory ownership and the adoption of a dead process's spool (#163);
- one process's pack pipeline: sink staging, the upload gate, chunked
  upload and index, `flush`/`close`/`stop` deadlines and cancellation (#162);
- the clock-skew bounds in the fence margin, the start wait and the lease
  deadline;
- the ring arithmetic: the payload ring's span split, the saturated free-room
  computation, and the eager safety net's reserve/flush/drain (#160);
- `Cancellation::SleepFor`'s wait arithmetic.

The ring's publish/consume protocol (the ready words, the head and tail
updates and their memory ordering, the CUDA side) is not modelled.

Nothing here is built, imported or executed by DMI. The specs are checked by
hand, with `specs/check.sh` or the commands below; none of them runs in CI.

```text
specs/
├── tla/    TLA+ models, checked with TLC
├── z3/     SMT encoding, checked with Z3
└── cbmc/   C++ harnesses, checked with CBMC, and sync_check.sh
```

Line references point at `main` @ `5b3b632`. The first three TLA+ models, the
Z3 script and the span harness were written against `204a8d2`/`99ee4ae` (#157)
and re-synced to `5b3b632`, after #159-#163, on 2026-10-06. Each `.tla`
header says what changed. `PackRank`, `SpoolOwnership`, the three
`PackPipeline` models and the new CBMC harnesses were written against
`5b3b632`.

## Findings at `5b3b632`

Open defects the models found in the current code, each traced line by line
in the source. Every one has a config that shows it and, where a fix was
modelled, a config that shows the fix closes it. None is a safety violation of
the catalog's single-publisher protocol.

| # | Severity | Finding | Shown by | Fix modelled by |
|---|---|---|---|---|
| E1 | medium | `run_cycle` checks the lease before every upload chunk except the first (`storage_service.cpp:642`, `if (next != 0)`). The first relies on the check at cycle start (`:577-581`), which comes before `ListPending` hashes the whole spool. A lease quarantined during the listing lets up to `max_packs` (64) packs upload that cannot be indexed and live only in `pending_index_`; a crash then, with `reconcile_on_start=False`, orphans them -- what decision 7 forbids. The header (`storage_service.h:237-246`) says every chunk is checked. | `PackPipeline_firstchunk` | `PackPipeline_firstchunk_fixed` |
| E2 | low-medium, liveness | Only a lease refusal makes a failed start reconcile owed (`storage_service.cpp:357-358`). A listing, HEAD or catalog error is only recorded, and a reconcile whose index batch fails still returns true and clears the flag. With the default `reconcile_interval_s = 0` a crash's orphaned packs wait for the next process start. | `PackPipeline_live_startonly` | `PackPipeline_live_failuresowed` |
| A1 | low | The `std::exception` branch of `ensure_publisher_lease` (`storage_service.cpp:1648`) never resets `held_elsewhere_since_ns_`, even when the claim INSERT was sent -- which means the head read found no live holder. Rival refusals on either side of a stretch with no holder then add up to the `2 x TTL` latch, and the service stops claiming for good. | `LeaseLifecycle_O3_unkrival` | -- (reset when `claim_insert_sent()`) |
| A2 | low, liveness | A renewal can start later than half the TTL after the claim was sent: a claim can be confirmed up to 2/3 TTL after its send, and after `start()` the lease thread waits a full tick before its first renewal (`:1386`). The comments at `storage_service.cpp:1442-1446` and `lease_coordinator.h:66-82` say otherwise. The deadline keeps it safe; the cost is a needless lease loss after start on a slow catalog. | `LeaseLifecycle_O1_window_slowclaim`, `O1_slow_lt` | -- (start the thread on `next_wake()`) |
| C1 | low today, routine under multi-rank | `adopt_step` looks at sibling spool directories again only while its last look found a live one (`storage_service.cpp:826-829`; flags set at `:953-954`). A rank directory created after that look and then left with packs is never adopted while the service runs, and `ChargedSiblingBytes` (`spool.cpp:1376-1390`) charges it against the live sink's budget all the while, contrary to `spool.h:104-107`. Today it needs a lease takeover; under per-rank upload-only services it needs nothing. | `SpoolOwnership_live_takeover`, `_live_multi`, `_charge` | `SpoolOwnership_live_fix`, `_live_multi_fix`, `_charge_fix` |
| D1 | latent | The record ring's admission check (`ring_engine_py.cu:718-719`) still computes `payload_cap - payload_used` and `task_cap - task_used` unsaturated, the form #160 replaced with `legacy_ring_room` on the legacy ring. If the CPU accounting ever passes capacity, every later reservation is admitted. No path to that state was shown. | `cbmc_room` (R6) | -- (use `legacy_ring_room`) |
| B1 | doc | `native_capture.py:427-436` says a publish is never abandoned while the server may still commit it. Under `LeaseScope` the lease deadline can cut a fenced statement before its `max_execution_time`, by at most stamp delay + `clock_skew` + 0.1 s. That costs a quarantine, not safety. | `z3/clock_skew.py` (lease-deadline give-up checks) | -- |
| E3 | doc | #162 says no loop cycle runs between `close()`'s flush and its `stop()`; the wait-again-once rule (`storage_service.cpp:523-538`) still lets one run. The trade-off is deliberate: always deferring lets back-to-back flushes starve the loop. | `PackPipelineLoop_between` | `_between_always` holds, but `_starve_always` fails |

Fixed since `99ee4ae`, as the re-synced models confirm: the lease no longer
outlives a slow request or a late wake (`LeaseLifecycle_O1_slowreq` and
`O1_skip` now hold), a refusal by the service's own late row no longer counts
towards the latch (`O3_selflatch` holds), two services no longer sweep one
spool (`O5_cosweep` holds under the owner lock), and a replayed pack no longer
flips a pinned read (`PackRank_first` holds; `PackRank_rowversion` shows the
old ranking failing).

## What each spec models

Every `.tla` header and harness comment carries the exact line references at
`5b3b632`; the table names the files.

| Spec | Models | Source of truth |
|---|---|---|
| `tla/VersionAllocator.tla` | the sole-claimant version allocation loop: floor read, jittered candidate, claim INSERT, singleton read-back, retry | `native/csrc/catalog/version_allocator.cpp:49-89`, `version_allocator.h:5-7`, watermark publish at `catalog_writer.cpp:603-648`, `clickhouse_client.cpp:457` |
| `tla/PublisherLease.tla` | the lease claim/renew/release protocol and the fenced publish: `claim_with_rival`, `head()`, `fence()`, `fence_eval()`, `reject_live()`, `publish_snapshot`'s manifest chunks and watermark INSERT; with `Timeouts = TRUE`, #159's per-request deadlines, late-landing claim rows and statements left running past their client | `native/csrc/catalog/lease_coordinator.{h,cpp}`, `catalog_writer.cpp`, `clickhouse_client.cpp`, `docs/catalog-descriptor-key.md`, `src/dmi/storage/capture/clickhouse_lease.py` |
| `tla/LeaseLifecycle.tla` | the lease lifecycle *above* that protocol: the lease thread and its deadline-driven wake, requests that take time and are cut at the lease deadline, the quarantine window, the `2 x TTL` latch, the start wait, and the spool sweep under the owner lock | `native/csrc/catalog/storage_service.{h,cpp}`, `catalog_writer.cpp`, `lease_coordinator.{h,cpp}`, `indexer.cpp`, `src/dmi/storage/native_capture.py` |
| `tla/PackRank.tla` | the reader's pick of a capture's pack at a pinned watermark: the `snapshot()` join and the `argMax` over `(member_version, store_id, pack_id, index_version)`, against indexer passes that fail part-way or replay a published pack, and ReplacingMergeTree merges | `native/csrc/catalog/reader.cpp:20-41,481-520`, `schema.cpp:27-29,498-512`, `object_key.cpp:66-85` |
| `tla/SpoolOwnership.tla` | one owner per spool directory and the adoption of a dead process's spool (#163): the claim, `LockInPlace`'s flock / `IsFileAt` / directory lock, `ReleaseAndRemoveIfEmpty`, the sink's stage, `Recover()`'s sweep, the upload gate, sibling scan / adoption / block, the engine's start and close order, crash and fork | `native/csrc/store/spool.cpp:607-638,823-994,1040-1171,1357-1407,1502-1781,1883-1936`, `native/csrc/catalog/storage_service.cpp:311-345,605-692,800-1101`, `src/dmi/engine.py:425-431,495-553,579-599,955-990` |
| `tla/PackPipeline.tla` | one process's pack pipeline: the sink staging packs as the ring releases them, a service cycle (lease check, owed retry, spool listing, chunked upload then index, reconcile), `flush(timeout)`, `close()`'s drain, `stop()`'s cancel, crash, and the next `start()`'s reconcile | `native/csrc/catalog/storage_service.{h,cpp}`, `indexer.cpp`, `store/uploader.cpp`, `sink/native_pack_sink.cpp`, `sink/pack_sink.cpp`, `src/dmi/engine.py:944-970,992-1062` |
| `tla/PackPipelineLoop.tla` | who runs a cycle when: `loop()`'s wait and its wait-again-once rule, `flush()`'s `try_lock_until` loop, `close()`'s flush followed by `stop()` | `storage_service.cpp:424-478,490-550`, `engine.py:944-970` |
| `tla/PackPipelineUpload.tla` | one pack's `UploadOne` over `S3Client::ExchangeWith` under a `Cancellation`: attempts, both backoffs, preflight, PUT, HEAD, `Remove`, and how the end is booked | `store/uploader.cpp:66-82,103-265`, `store/s3_client.cpp:76-99,214-386`, `store/cancel.h` |
| `z3/clock_skew.py` | three obligations: the two-host derivation behind the fence margin `publish_timeout_ns + clock_skew_ns`; the default start wait `lease_ttl_s + publish_timeout_s + clock_skew_s`; and #159's lease deadline `sent + TTL - clock_skew - 0.1 s` | `docs/catalog-descriptor-key.md`, `catalog_writer.cpp:148-160`, `native_capture.py:454-462,482-485`, `lease_coordinator.h:30-64`, `lease_coordinator.cpp:40-51` |
| `cbmc/payload_ring_span.cpp` | `payload_compute_spans` and its stated precondition (includes the real header) | `native/csrc/ring/payload_ring.cuh:44-86` |
| `cbmc/legacy_ring_room.cpp` | #160's saturated free-room arithmetic over all 64-bit values, and the record ring's admission check | `native/csrc/ring/ring_engine_py.cu:141-144,715-720`, `task_ring.cuh` |
| `cbmc/eager_safety_net.cpp` | a bounded model of reserve / flush / publish / drain: `prepare_step`, `available_*`, `reserve_one`, `flush_and_wait`, and HookPoint's eager branch | `ring_engine_py.cu:618-651,937-949,957-974,980-984`, `src/dmi/adapters/base.py`, `src/dmi/hooks/point.py` |
| `cbmc/cancel_sleep.cpp` | `Cancellation::SleepFor`'s wait arithmetic (#162) | `native/csrc/store/cancel.h` |

`LeaseLifecycle.tla` sits on top of `PublisherLease.tla` rather than beside it:
it abstracts `LeaseCoordinator` to its contract (*a claim presenting lease id L
is admitted iff the head row is dead or is L, refused with `kHeld` otherwise,
and may return an unknown outcome*) and models the service around it. That
contract is what `PublisherLease.tla` discharges. `PackPipeline.tla` in turn
treats the lease as an environment signal (held or lost) and the spool as
owned by this process; how the lease is kept is `LeaseLifecycle.tla`'s job,
and who owns the spool is `SpoolOwnership.tla`'s. See **Limitations**.

All TLA+ models are written against the **code**, not the prose. Where the two
disagree, the spec follows the code and a comment in the `.tla` says so. The
CBMC harnesses include the real header where goto-cc can parse it and copy the
code verbatim, between `BEGIN COPY`/`END COPY` markers, where it cannot;
`cbmc/sync_check.sh` fails when a copy no longer matches `native/csrc`.

## Getting the tools

```sh
# TLC -- a single jar, no install
curl -fsSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar

# Z3 (Python bindings include the solver), in a venv of its own
uv venv .tools/venv && uv pip install -p .tools/venv/bin/python z3-solver

# CBMC: a release package from github.com/diffblue/cbmc/releases. The .deb
# also unpacks without installing: dpkg-deb -x ubuntu-*-cbmc-*.deb DIR puts
# the binaries in DIR/usr/bin.
```

Do not use apt's `cbmc` on Ubuntu 20.04: it is too old for these harnesses.
Take a release package instead, and mind the glibc it was built for: the
`ubuntu-22.04` and `ubuntu-24.04` packages need glibc 2.32 or later, so on
Ubuntu 20.04 (glibc 2.31) the newest that runs is CBMC 6.5.0's
`ubuntu-20.04-cbmc-6.5.0-Linux.deb`.

Recorded with TLC 2.19 on Java 21, Z3 4.13 and CBMC 5.95 (#157), and re-run
at `5b3b632` with TLC v1.8.0 (`tla2tools.jar`, 2026-10 nightly) on Java 21,
Z3 5.1.0 (`z3-solver`) and CBMC 6.5.0. Any recent version of each should do;
nothing here relies on a version-specific feature, except that goto-cc 6.5's
C++ front end cannot parse some libstdc++ headers (see CBMC below).

## Running the checks

### All at once

```sh
TLA2TOOLS_JAR=/path/to/tla2tools.jar CBMC=/path/to/cbmc PYTHON=/path/to/python \
    specs/check.sh
```

`check.sh` runs every row not marked manual -- the TLC configs,
`z3/clock_skew.py`, the CBMC builds and `cbmc/sync_check.sh` -- compares each
verdict with the one it expects, and exits non-zero on any mismatch. The
expected verdicts are a table at the top of the script (`specs/check.sh
--list`); a `.cfg` without a row, or a row without a `.cfg`, is an error. A
violated temporal property counts as `violated`, like a violated invariant.
`--all` adds the manual rows, which take minutes each; `--list | awk '$3 ==
"manual"'` names them. Arguments that are not options select checks by shell
glob, e.g. `specs/check.sh 'LeaseLifecycle_O3_*' 'cbmc_*'`. TLC runs in a
scratch copy of `specs/tla`, so no `states/` directory or trace file lands in
the tree. `TLC_WORKERS` (default 4) and `TLC_HEAP` (default `4g`) tune TLC,
and `GOTO_CC` overrides the `goto-cc` found next to `$CBMC`.

The sections below give the commands for running one check by hand.

### TLA+

Every `.cfg` in `specs/tla/` is one model: one set of constants, one or more
invariants or temporal properties. `check.sh` picks the module from the
config's name: `LeaseLifecycle_*`, `PublisherLease*`, `PackRank_*`,
`SpoolOwnership*`, `PackPipelineLoop_*`, `PackPipelineUpload_*` and
`PackPipeline_*` go to the module of that name, and everything else to
`VersionAllocator.tla`. Run any of them from `specs/tla/`.

**The allocator and `PackRank` configs need `-deadlock` on the command line.**
`VersionAllocator.tla` has no stutter step: once every allocator reaches
`published` there is no next state, and TLC reports `Error: Deadlock reached.`
and stops — on `nocap_distinct` that happens after 97 of the 836 distinct
states, so without the flag the run *looks* clean for two seconds and has
checked almost nothing. `-deadlock` turns deadlock checking off (that is what
the flag does, despite the name), and the run completes. `PackRank`'s bounded
runs end the same way. The other modules' configs carry `CHECK_DEADLOCK FALSE`
in the `.cfg` itself.

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
     -workers 4 -config LeaseLifecycle_O3_unkrival.cfg LeaseLifecycle.tla

# pack pipeline
java -XX:+UseParallelGC -Xmx6g -cp /path/to/tla2tools.jar tlc2.TLC \
     -workers 4 -config PackPipeline_firstchunk.cfg PackPipeline.tla
```

`PublisherLease.cfg` is the base model — the protocol exactly as shipped, with
the combined `AllSafety` invariant. Each `PublisherLease_*.cfg` turns a single
knob away from it or swaps in a single invariant; its header comment says
which. `LeaseLifecycle_*.cfg` is grouped by obligation: `O1_*` renewal, `O2_*`
quarantine, `O3_*` the latch, `O4_*` the start wait and the deadline, `O5_*`
the spool sweep, and `vac_*` the vacuity guards. In the newer modules,
`m_*`/`mut_*` configs are mutations of the code that must be caught, `*_fix`/
`*_fixed` configs model a proposed fix, `live_*` configs check liveness under
fairness, and `vac_*` configs are vacuity guards.

The manual rows are the largest runs: `PublisherLease.cfg`, `believers`,
`holderssafe`, `overrun`, `nonlin_fence`, `stalepid`, `noovr1`, `to_retry`
and `to_base5` generate 15M-61M states each; `SpoolOwnership_race` and
`SpoolOwnership_race_takeover` take one to five and about 12 minutes; and the
full three-process `SpoolOwnership` and `SpoolOwnership_multi3` have not
finished within an hour (256M distinct states explored, no violation) and are
recorded as expected values, not observed ones. `cbmc_eager_3x2` takes about
six minutes. On a loaded 32-core host the fast set takes about 45 minutes with
`TLC_WORKERS=16`.

### Z3

```sh
python3 specs/z3/clock_skew.py
```

Prints one line per check and exits non-zero if any result differs from the
expected one. Fourteen checks, no arguments, a few seconds.

### CBMC

`cbmc` does not accept `-std=`, so each harness is compiled to a goto-binary
with `goto-cc` first and verified in a second step. The harnesses are `.cpp`
because the ring headers use namespaces; the CUDA qualifiers are defined away
so they compile for the host. goto-cc 6.5's C++ front end cannot parse
libstdc++'s `<stdexcept>`, `<chrono>`, `<mutex>` or `<atomic>`, nor
`task_ring.cuh`'s `publication_word.h`, so code that lives behind those, or in
a `.cu` file that pulls in torch, is copied between `// BEGIN COPY` and
`// END COPY` markers. `specs/cbmc/sync_check.sh` fails when a copied block
no longer occurs verbatim in `native/csrc`; run it before trusting a verdict.

```sh
cd specs/cbmc
B=$(mktemp -d)
./sync_check.sh
goto-cc -std=c++11 payload_ring_span.cpp -o "$B/span.gb"
cbmc --unwind 80 --unwinding-assertions "$B/span.gb"
goto-cc -std=c++11 legacy_ring_room.cpp -o "$B/room.gb" && cbmc "$B/room.gb"
goto-cc -std=c++11 -DSTEPS=2 -DHMAX=2 eager_safety_net.cpp -o "$B/eager.gb"
cbmc --unwind 4 --unwinding-assertions "$B/eager.gb"
goto-cc -std=c++11 cancel_sleep.cpp -o "$B/cancel.gb" && cbmc "$B/cancel.gb"
```

Each harness except `payload_ring_span.cpp` leads every assertion's text
with a label (`R1`, `E4`, `C2`); its verdict string lists the labels in
order, S for SUCCESS and F for FAILURE.

## Results

State counts are TLC's own, as `generated / distinct`. Counterexample lengths,
in parentheses, are the number of states in the trace TLC printed, including
the initial state.

**State counts for runs that completed are exact and reproducible.** Every
completed `VersionAllocator` and `PublisherLease` run in the tables below
reproduced its previously recorded distinct count to the state when re-run
with `check.sh`, on #157's models and again on the re-synced ones at
`5b3b632` (whose `Timeouts = FALSE` reproduces the old `PublisherLease` state
space exactly). The `LeaseLifecycle` counts are new at `5b3b632`: the re-sync
changed its state space.

**State counts and trace depths for refuted runs are not reproducible.** TLC
stops as soon as any worker hits the violation, so both the count and the
trace length depend on the worker count and on scheduling. The figures here
were recorded with 2, 4 or 8 workers; `NoOrphanManifestRows` has been seen at both
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

Re-checked at `5b3b632`. `version_allocator.{h,cpp}` have not changed since
`99ee4ae`, and every verdict reproduced. Since #151/#159 a claim INSERT that
times out may still land after its allocator has given up. The model does not
add this, because it cannot hurt `Distinct`. A late row only adds an owner at
a candidate version, and that can only make another allocator's read-back
fail. It can never let two allocators *return* one version. It can raise the
floor, which `FloorMonotonic` already treats as allowed.

### Publisher lease (`PublisherLease.tla`)

Base model: two publishers, `Lids = {1,2,3}`, `MaxTerm = 3`, `MaxTime = 5`,
`MaxVersion = 2`, `MaxAttempts = 2`, `TTL = 2`, `PT = 1`, `SKEW = 0`,
`NumChunks = 1`, linearizable store, cap enforced, writer lock held, fresh
`publish_id`. Each config below differs from it only as its name says. Every config also sets `Timeouts = FALSE` unless
its name starts with `to_` or it is one of the timeout vacuity guards
(`vac_late`, `vac_gone`).

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `PublisherLease.cfg` (base) | `AllSafety` | **holds** (its `NoOverlappingAdmit` conjunct vacuously; see below) | 19,807,266 / 9,665,700 |
| `base5` (`MaxTerm 5`, `MaxTime 8`) | `AllSafety` | **holds** | 5,195,821 / 2,408,145 |
| `base5_ovr` (`base5`, `AllowOverrun`) | `NoOverlappingAdmit` | violated (29) | 1,432,673 / 678,409 |
| `believers` | `AtMostOneBeliever` | **holds** | 19,807,266 / 9,665,700 |
| `holderssafe` | `TwoHoldersIsSafe` | **holds** | 19,807,266 / 9,665,700 |
| `holders` | `AtMostOneHolder` | violated (12) | 10,865 / 6,280 |
| `noovr0` (empty refs, cap enforced) | `NoOverlappingAdmit` | **holds** | 1,761,711 / 799,611 |
| `ovr0` (empty refs, `AllowOverrun`) | `NoOverlappingAdmit` | violated (15) | 64,379 / 34,191 |
| `noovr1` (`noovr0` with 1 chunk) | `NoOverlappingAdmit` | **holds** | 15,555,231 / 7,204,077 |
| `ovr1` (`noovr1`, `AllowOverrun`) | `NoOverlappingAdmit` | violated (32) | 4,235,316 / 2,001,749 |
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
flight to compare. The same goes for the base run itself: `AllSafety` includes
`NoOverlappingAdmit`, so `PublisherLease.cfg` holding says nothing about
overlapping admissions. That does not make its other four conjuncts vacuous,
but it does not show them reached either; the guards above do that for the
publish, fence and admission states. `base5`, `noovr0`, `ovr0`, `noovr1`, `ovr1` and `vac0` exist for that
reason, and each holding run has a partner that differs only in allowing the
statement cap to overrun and is refuted, which proves the takeover is reached
at its constants:

| Holds (cap enforced) | Refuted (`AllowOverrun`) | Publish |
|---|---|---|
| `base5` (`AllSafety`) | `base5_ovr` | one manifest chunk, `MaxTerm 5` |
| `noovr0` | `ovr0` | empty refs, `MaxTerm 4` |
| `noovr1` | `ovr1` | one manifest chunk, `MaxTerm 4` |

`vac0` also refutes `NeverAdmits` at the `ovr0`/`noovr0` constants, which proves
admissions are reached there. Those pairs, not the `overrun` row or the base
run, are the evidence about the takeover instant.

#### Timeouts (#159)

Since `8b7991d`, every lease request has a deadline
(`lease_coordinator.cpp:86-110`, `storage_service.cpp:113-155`). When a request
runs out of time, the writer drops its lease without a tombstone
(`CatalogWriter::quarantine`, `catalog_writer.cpp:268-274`). That happens for a
renewal, a claim whose INSERT may have been sent (`:302-318`) and a publish
(`:284-295`, `:681-688`). An unrenewed lease is dropped the same way
(`abandon_lease`, `:326-330`). What the writer sent may still land: a claim
INSERT up to its transit delay late (`add_write_caps`,
`lease_coordinator.cpp:112-143`), and a fenced publish statement until its own
`max_execution_time` (`catalog_writer.cpp:536-540`). `Timeouts = TRUE` adds all
of this. A time-out may strike any request (`GiveUp`). A claim INSERT given up
on may land at *any* later time, stamped when it lands (`LateLand`). A
statement given up on stays in flight with no owner, and lands or aborts by its
cap (`GoneLand`/`GoneAbort`). A lease may also be abandoned between passes
(`Abandon`). The quarantine's TTL wait before the next claim is left out, so a
writer may claim again at once. Each of these choices only adds behaviours.

| Config | Invariant | Verdict | States |
|---|---|---|---|
| `to_noovr0` (`noovr0` + timeouts) | `AllSafety` | **holds** | 4,466,223 / 1,573,881 |
| `to_ovr0` (`to_noovr0`, `AllowOverrun`, `MaxAttempts 1`) | `NoOverlappingAdmit` | violated (20) | 403,477 / 163,727 |
| `to_chunks2` (`chunks2` + timeouts) | `AllSafety` | **holds** | 8,264,802 / 3,083,616 |
| `to_chunks2_prefix` | `OrphansArePrefixes` | **holds** | 8,264,802 / 3,083,616 |
| `to_retry` (`to_chunks2`, empty refs, `MaxAttempts 2`; manual) | `AllSafety` | **holds** | 59,029,066 / 21,106,310 |
| `to_base5` (`base5` + timeouts; manual) | `AllSafety` | **holds** | 15,604,703 / 5,593,245 |
| `vac_late` | `NeverLateLands` | refuted (5) | 321 / 206 |
| `vac_gone` | `NeverGoneInFlight` | refuted (8) | 7,784 / 3,596 |
| `ovr0_mono` (`ovr0`, timeouts off) | `WatermarkMonotonic` | violated (17) | 186,453 / 93,966 |

The safety argument does not lean on the quarantine. A late claim row or an
abandoned statement is still bounded by the server-side fence: the fence
admitted the statement with more than `publish_timeout + clock_skew` of life
left on its row. A rival can claim only after that row expires, which comes
after the statement's cap. A late claim row at the head term adds a claimant,
and `reject_live` and the fence's single-row resolution already handle that.
The two guards show that both behaviours are reached at these constants.
`to_ovr0` shows the takeover is reached there too, so `to_noovr0` holding is
not vacuous.

`ovr0_mono` checks `WatermarkMonotonic` under `AllowOverrun`, the takeover-instant
residual already documented. Two watermark rows in flight can land out of
order. `PackRank_unordered` shows that this is exactly what makes a pinned read
change, so the residual reaches readers as well as publishers.

### Lease lifecycle (`LeaseLifecycle.tla`)

Re-synced to `5b3b632`, which follows #159 (a deadline for every request under
the lease), #162 and #163 (one owner process per spool directory). Time is in
**ticks** of `TTL/6`, the lease thread's tick (`storage_service.cpp:60-70`).
Most configs run one service with `TTL = 6`.

**Requests take time.** A ClickHouse request is issued by one step and
completed by a later one, `0..MaxReq` ticks on. One is in flight per service,
which is `lease_mutex_`: the lease thread and the cycle's stretches hold it for
every catalog request they make (`LeaseScope`, `:114-149`).

- **Under the lease.** A request is cut at the lease deadline,
  `sent + TTL − clock_skew` (`lease_coordinator.h:31-60`). The 0.1 s margin is
  kept as strictness. A lease past its deadline is abandoned at the next
  `LeaseScope` boundary.
- **Claims.** A claim's `sent_ns` is taken before its INSERT. The claim is
  confirmed up to `2 × CB` later, or cut at `CB = min(request_s, ttl/3)`.
- **The cycle.** A cycle stretch renews before each request once the renewal
  is due (`keep_lease_in_pass`). A publish renews first, unconditionally.
- **Constants that bound the trouble.** `MaxLate` is OS lateness on a
  lease-thread wake. `CfgSkew` is the configured `clock_skew`. `RivalEarly` is
  how much earlier than the lagging view a rival may see a row dead.

#### O1 — renewal keeps the lease alive

| Config | Invariant | Verdict | was at 99ee4ae |
|---|---|---|---|
| `O1_clean`, `O1_cut`, `O1_late` | `NoPhantomLease` | **holds** | holds |
| `O1_slowreq3`, `O1_slowreq` (`MaxLate` 3, 4) | `NoPhantomLease` | **holds** | holds / violated |
| `O1_slowreq_req` (`MaxReq 4`) | `NoPhantomLease`, `NoRequestPastDeadline`, `NoConcurrentHolder` | **holds** | — |
| `O1_slowreq_cycle` (+ stretches, cut, unknowns) | the same | **holds** | — |
| `O1_belief` | `NoPhantomBelief` | violated (reporting gap) | — |
| `O1_skip` (non-publishing stretches) | `NoPhantomLease` | **holds** | violated |
| `O1_absorb` | `OneFailureAbsorbed` | violated | violated |
| `O1_window` / `O1_window_slowclaim` | `RenewalStartsByHalf` | holds / **violated** | — |
| `O1_fast_lt` / `O1_slow_lt` (`MaxReq` 2 / 3) | `NoDeadlineLoss` | holds / violated | — |
| `O1_fast_cy` / `O1_slow_cy` (stretches, 2 / 3) | `NoDeadlineLoss` | holds / violated | — |
| `O1_skew_cy` (stretches, 2, `CfgSkew 1`) | `NoDeadlineLoss` | violated | — |

`NoPhantomLease` now reads: a lease the service can still *use*, held and
before its deadline, is the live head. It holds under slow requests and late
wakes alike. #159 closed the pre-#159 `O1_slowreq` and `O1_skip` defects: every
request is cut to the deadline, a lease past it is abandoned, and nothing but a
confirmed claim moves the renewal schedule (`storage_service.cpp:1194-1200`).
`vac_lost` proves the slow path is exercised. `O1_belief` keeps the old, stronger
reading: `held_lease()` outlives the deadline until the next `LeaseScope`
boundary, though nothing is sent under it.

**`O1_window_slowclaim`: a renewal can start later than `TTL/2` after the
send.** The comments at `storage_service.cpp:1442-1446` and
`lease_coordinator.h:66-82` say a renewal starts within half the TTL of the
claim's send. Two things break it:

- a claim's INSERT and read-back each get the claim bound, so the claim can be
  confirmed up to `2 × TTL/3` after the send;
- after `start()`, the lease thread's first wait is a full tick
  (`storage_service.cpp:1386`), not `next_wake()`.

The trace: a claim sent at 0 is confirmed at 3, the thread's first look comes
at 4, and the renewal starts at 4 > 3. `O1_slow_lt` shows the cost: 3-tick
requests lose the lease after start, though a renewal on schedule would
finish. It is safe, since the deadline still guards the row, but the lease is
lost needlessly. Starting the thread with `next_wake()` removes the extra tick.

`NoDeadlineLoss` measures how slow a healthy catalog can be: requests of up to
`TTL/3` never cost the lease. With stretches, a request that starts just
before the renewal falls due delays it by its own length, which the comment
states ("a third of the TTL plus one request"). `clock_skew` comes straight
off that budget.

#### O2 — the quarantine window

| Config | Invariant | Verdict |
|---|---|---|
| `O2_quar` | `QuarantineOutlastsItsRow` | **holds** |
| `O2_skew` (`Skew 1`) | `QuarantineOutlastsItsRow` | violated |
| `O2_reuse` (`ReuseLid`) | `QuarantineOutlastsItsRow` | violated |
| `O2_selfref` (`Skew 1`) | `NoSelfRefusal` | violated |

The window (`catalog_writer.cpp:273`) is still `now + lease_ttl` on the local
clock, with no skew allowance, so a lagging replica can show the dropped row
live past it. Since #159 the price is one retry tick, not a latch: the
refusal comes from the writer's own claim row, which `lease_held_elsewhere()`
recognises (`:1682-1696`). `O2_quar` lands an unknown-outcome INSERT no later
than the client gave up on it. `add_write_caps` now gives the server that
time as `max_execution_time` (`lease_coordinator.cpp:112-143`). What the caps do
not cover — a part commit that overruns, time in transit — is not modelled.

#### O3 — the `2 x TTL` latch

| Config | Invariant | Verdict | was |
|---|---|---|---|
| `O3_rival` | `NeverLatches` | violated | violated |
| `O3_rivaljust` | `NoFalsePositiveLatch` | **holds** | holds |
| `O3_stops2` / `O3_false` | `NeverLatches` / `NoFalsePositiveLatch` | **holds** | holds |
| `O3_falsenocut` | `NoFalsePositiveLatch` | **holds** | holds |
| `O3_selflatch`, `O3_selflatch_latch` | `NoFalsePositiveLatch`, `NeverLatches` | **holds** | violated |
| `O3_unkrival` | `NoFalsePositiveLatch` | **violated** | — |
| `O3_stops` | `NeverLatches` | holds, **vacuous** (`vac_stopsrefusal`) | the same |

#159 fixed the self-latch: a refusal by the service's own late or dropped row
restarts the clock. `vac_ownrefusal` proves the self-refusal is still reached.

**`O3_unkrival`: the latch can still count across a broken refusal run.** The
latch clock is reset in two places only:

- by a successful claim;
- by a refusal from the service's own rows.

It is not reset by a claim whose INSERT was sent and then timed out. Such a
claim passed `reject_live`, so its head read found no live rival, which is
the very argument the comment at `:1687-1689` makes. The trace, at TTL 6:

1. Rival F refuses the service at 8.
2. F stops at 8.
3. The service's claims at 9 and 15 time out with their INSERTs sent and
   quarantine it.
4. F returns at 19.
5. At 21 the service latches, 21 − 8 ≥ 12, though nobody held the catalog
   from 8 to 19.

The fix is to reset `held_elsewhere_since_ns_` in
`ensure_publisher_lease`'s `std::exception` branch when
`claim_insert_sent()`.

#### O4 — the start wait and the deadline

| Config | Invariant | Verdict |
|---|---|---|
| `O4_same`, `O4_skew` | `StartAlwaysSucceeds` | **holds** |
| `O4_bigger` | `StartAlwaysSucceeds` | violated |
| `O4_compose` (cycle off) | `NoConcurrentHolder` | **holds** |
| `O4_deadline_skew` (`CfgSkew = RivalEarly = 1`) | `NoConcurrentHolder`, `NoPhantomLease` | **holds** |
| `O4_deadline_noskew` (`CfgSkew 0`, `RivalEarly 1`) | `NoConcurrentHolder` | violated |
| `O4_slowstart` | `StartPastOneSlowClaim` | **holds** |
| `O4_slowstart_two` | `StartAlwaysSucceeds` | violated |
| `O4_slowstart_pred` | `StartPastOneSlowClaim` | violated (by design) |

The deadline keeps a lease from being used once a rival can take its row,
exactly when `clock_skew_s` covers the real skew (`O4_deadline_*`).

The start wait handles slow claims as follows:

- One timed-out start claim never fails `start()`, whether or not its INSERT
  was sent (`O4_slowstart`, with `vac_slowstart`).
- Two timed-out claims can fail it.
- A predecessor's row together with a slow claim at the end of the wait can
  also fail it. Only a quarantine stretches the wait (`:1588-1602`).

#### O5 — the spool sweep

| Config | Invariant | Verdict | was |
|---|---|---|---|
| `O5_refused` | `RefusedStartNeverSweeps` | **holds** | holds |
| `O5_cosweep` | `SweepOnlyWhenAlone` | **holds** | violated |
| `O5_cosweep_nolock` (pre-#163 counterfactual) | `SweepOnlyWhenAlone` | violated | — |
| `O5_crash` | `SweepOnlyWhenAlone`, `RefusedStartNeverSweeps` | **holds** | — |

#163 made the spool owner lock the guarantee. The constructor's `Spool::Open`
takes `<dir>/.owner.lock` and throws on `kOwned` (`storage_service.cpp:213-221`),
so a second live process never constructs on the directory (`vac_nolock`).
After a crash the kernel drops the flock. A successor then takes it, waits out
the dead row and sweeps (`O5_crash`, with `vac_crash`). Without the lock, the
pre-#163 co-sweep trace still reproduces (`O5_cosweep_nolock`).

#### Vacuity guards

Every one must be refuted, and every one is, except `vac_stopsrefusal`, which
holds and so marks `O3_stops` as vacuous, as before.

`vac_held`, `vac_quar`, `vac_recov`, `vac_refus`, `vac_rival`, `vac_cut`,
`vac_latch`, `vac_start`, `vac_o4waited`, `vac_refusedstart`,
`vac_stops2refusal`, `vac_stops2rival`, `vac_lost`, `vac_deadline_rival`,
`vac_slowstart`, `vac_nolock`, `vac_crash`, `vac_ownrefusal`.

`O1_tries*` are retired. The fixed-tick comment they pinned is gone: the
thread now wakes when the renewal falls due.

#### Limitations specific to this model

See LIMITS at the foot of the `.tla`. In particular:

- the coordinator is a contract;
- a claim's head read takes no time;
- a renewal's three requests are one;
- skew is one tick at this grain;
- the schema install lease at `start()` and the object-store side of the
  cycle are not modelled;
- `Stop` is never enabled;
- the owner lock is modelled as exclusive per directory while its holder
  lives. Its implementation — flock, shared filesystems, fork — is
  `spool.cpp`'s.

### Pack ranking (`PackRank.tla`)

Models how the native reader picks the pack a capture resolves to at a pinned
watermark (#161, `d3fe7c2`). This is the `snapshot()` join and the single
`argMax` over `(member_version, store_id, pack_id, index_version)`
(`reader.cpp:20-41,481-520`). The model includes indexer passes that write
descriptor rows, then manifest rows, then the watermark row. Any pass may fail
part-way, and any pass may replay a pack that is already published. It also
includes ReplacingMergeTree merges that collapse a `(capture, pack)` pair to its
highest-version row (`schema.cpp:27-29,498-512`). Each read pinned at a published
watermark is recorded and compared against every later state.

Base model: two packs competing for one capture, two concurrent passes,
`MaxVer = 3`, replays byte-identical, watermark rows landing in version order.
Run with `-deadlock`.

| Config | Ranking / knob | Invariant | Verdict | States |
|---|---|---|---|---|
| `first` | `(min member version, pack, iv)` — shipped | `TypeOK`, `PinnedReadStable`, `UniqueWinner`, `LatestPackWins` | **holds** | 476,073 / 137,332 |
| `first_v4` | shipped, one pass, `MaxVer = 4` | same | **holds** | 1,538,933 / 344,241 |
| `rowversion` | `(index_version, pack)` — `99ee4ae` | `PinnedReadStable` | violated (8) | 7,154 / 4,010 |
| `newest_pin` | `(max member version, ...)` — #156 | `PinnedReadStable`, `UniqueWinner` | **holds** | 476,073 / 137,332 |
| `newest` | the same | `LatestPackWins` | violated (9) | 13,580 / 7,050 |
| `diffreplay` | shipped, a replay's rows differ | `PinnedReadStable` | violated (8) | 3,397 / 2,009 |
| `unordered` | shipped, watermark rows may land out of order | `PinnedReadStable` | violated (10) | 28,924 / 13,400 |

Vacuity guards, each refuted: `vac_replay` (`NeverReplayed`, 5), `vac_merge`
(`NeverMerged`, 7), `vac_contest` (a pin is recorded while two member packs
compete, 6), `vac_abovepin` (a pack that was a member at a pin is published
again above it, 10).

What the rows show:

- **`rowversion` is the bug #161 fixed.** A pass replays pack 1, which is
  already a member at pin 1. It rewrites pack 1's rows at version 2 before
  publishing anything. The join is on pack identity and is not bounded by the
  pin, so those rows enter the read at pin 1 and outrank pack 2. The pinned
  read flips without anything new being published.
- **`newest` shows why it is the *first* publish.** Ranking on the newest
  publish keeps pins stable, but a replay that publishes promotes the
  superseded pack at every later head.
- **The shipped ranking rests on two premises, which `diffreplay` and
  `unordered` remove one at a time.**
  - A replay rewrites byte-identical rows. Because `index_version` is the last
    component of the order, the read resolves to the replay's row. Content
    descriptors come from the pack itself, and the object key is a function of
    the pack's own fields (`object_key.cpp:66-85`), so in code the rows are
    identical.
  - Watermark rows land in version order (`PublisherLease.tla`'s
    `WatermarkMonotonic`). That holds under the cap-enforced, linearizable
    lease, and fails under the documented overrun residual (`ovr0_mono`).

Not modelled: the two-query page shape and its cursor, manifest garbage
collection (`catalog_writer.cpp:776-857`, which deletes only unpaired manifest
rows below the published head), and store mirroring beyond its role as a
tiebreak.

### Spool ownership and adoption (`SpoolOwnership.tla`)

Written against `main` @ `5b3b632` (#163). The model covers the rank
directories under one catalog key on one node, `<base>/<key>/r<rank>-<inc>/`,
each created by one process. It covers what those processes do to the
directories: claim, write, sweep, upload, adopt, block and remove. Crash and
fork are included too.

How the model represents the protocol:

- **The lock file.** It is modelled by inode, so a remover's unlink and a
  taker's `O_CREAT` of a new file at the same path are different files.
- **LockInPlace.** Its open, flock, `IsFileAt` and directory-lock steps are
  separate actions, retried as in the code (twice on `EWOULDBLOCK`).
- **ReleaseAndRemoveIfEmpty.** It is split into its check, its unlink and
  its rmdir.
- **The probe** (`ReadSpoolOwner`) is atomic.
- **The publisher lease** is a boolean that the environment can take away a
  bounded number of times (a quarantine). With `ExclusiveLease`, one service
  holds it at a time, as with today's single publisher: a `start()` that
  cannot get it fails, and its engine releases the claim. Without it, every
  service has a lease of its own, which is the shape of several services
  serving one catalog key.
- **Time** is abstracted away: a recheck that is due may run at any moment.
- **Reductions.** One pack per process and one pack per chunk. `Stagers`
  and `Adopters` restrict which processes stage and which adopt in the
  three-process race configs.

Properties (invariants unless marked liveness):

| Property | Meaning |
|---|---|
| `MutualExclusion` | at most one live process believes it owns a directory: its creator while its claim is held, and an adopter from its directory lock until it lets go |
| `NoClobber` | no sweep (`Recover`, `BeginRecovery`) deletes the `.open` file of a stage still in flight |
| `NoLostPack` | a ready pack stays in an existing directory, and one gone from the spool is in the object store |
| `NoConcurrentUpload` | no pack is in flight to the store from two uploaders |
| `OwedBounded` | at most one chunk per service is uploaded and not yet indexed |
| `UploadOnlyWithLease` | no chunk starts without the lease (decision 7) |
| `NoOrphan` | a pack in the store and not in the catalog is owed by a live service, or reconcile_on_start finds it |
| `BlockedLetGo` | a service never keeps a sibling it blocked |
| `AdoptLive` (liveness) | a ready pack whose process has died is eventually uploaded by some live service, unless no service can ever upload it |
| `ChargeDrains` (liveness) | what a sink is charged for a dead sibling is eventually given back |

**Holds.** All the invariants hold in every real-protocol config:

- `SpoolOwnership_quar`: two processes and two quarantines;
- `_multi`: a lease per service;
- `_race` and `_race_takeover`: three processes, two adopters racing for one
  dead directory while one removes it;
- `_bad`: a pack no service can upload, so the directory is blocked, marked,
  let go of, and taken again by the other adopter.

**Fails.** `AdoptLive` fails under a lease takeover (`_live_takeover`) and
under several services (`_live_multi`):

- `adopt_step` (storage_service.cpp:828-832) looks at the siblings again
  only while `live_siblings_`, which `scan_siblings` (:953-954) sets from
  its own last look.
- So a directory that appears after that look, and is left with packs, is
  not adopted while the service runs.
- It stays charged against the engine's sink budget for as long
  (`ChargedSiblingBytes`, spool.cpp:1376-1390; `_charge`).

The single lease keeps the gap unreachable without a takeover (`_live`
holds). Looking on every recheck, whether or not the last look found a live
sibling, closes it (`_live_fix`, `_live_multi_fix`, `_charge_fix`).

**Mutations, and what they show:**

- **The service sweeps before the sink opens** (engine.py:428 before :430).
  Opening the sink first breaks `NoClobber` (`_m_sinkfirst`).
- **An unsealed sink's claim is kept until exit.** Releasing it breaks
  `NoClobber` (`_m_releaseunsealed`).
- **A forked child closes its lock descriptors.** If it does not, a dead
  directory reads as live forever, and `AdoptLive` fails (`_m_noatfork`).
- **Uploads wait for the lease.** Dropping the gate breaks
  `UploadOnlyWithLease` (`_m_noleasegate`).
- **reconcile_on_start off** orphans the one owed chunk on a crash
  (`_noreconcile`, `NoOrphan`). This is the documented one-chunk residue.
- **IsFileAt and the directory lock are not symmetric.**
  - Without `IsFileAt`, `MutualExclusion` still holds (`_m_nofilecheck`).
  - Without the directory lock it fails (`_m_nodirlock`): in
    `ReleaseAndRemoveIfEmpty`'s window between unlink and rmdir, the
    directory reads as unowned, and a second adopter takes a new lock file.
  - The overlap is benign for the data: with either check or both removed,
    every other invariant still holds (`_m_nodirlock_data`,
    `_m_nolocks_data`), since the directory is drained and the remover's
    rmdir fails on the new file.

The `_vac_*` configs witness that each adoption path is reachable: an
adopted upload, a removal, an `IsFileAt` mismatch, a live sibling, and a
block.

**Limitations:**

- Pack validation and quarantine are not modelled.
- The probe's brief hold on a dead directory's lock is not modelled. It can
  only make a concurrent take fail.
- The nesting check and the flat `spool_root` modes are not modelled.
- An outside cleaner removing a lock file, which is the directory lock's
  documented reason, is not modelled.
- The pid-and-host fallback of `SpoolOwnedByThisProcess` (gVisor, WSL1) is
  not modelled.
- Shared filesystems are not modelled, since statfs refuses them.
- The full three-process configs, where every process stages and adopts
  (`SpoolOwnership`, `SpoolOwnership_multi3`), are `manual` and did not
  finish within an hour. See the results.

### Pack pipeline (`PackPipeline.tla`, `PackPipelineLoop.tla`, `PackPipelineUpload.tla`)

| Spec | Models | Source of truth |
|---|---|---|
| `tla/PackPipeline.tla` | one capture process's pack pipeline. The ring admits records to the sink, and the sink stages packs. A `CaptureStorageService` cycle checks the lease, retries what is owed, lists the spool, uploads a chunk and indexes it before the next, and runs the reconcile. Also covered: `flush(timeout)` with its two armed deadlines, `engine.close()`'s drain, `stop()`'s cancel, a crash anywhere, and the next process's `start()` reconcile | `native/csrc/catalog/storage_service.cpp:321-370,372-413,424-478,552-747,749-770,1127-1271,1284-1378`, `storage_service.h:12-25,237-246`, `indexer.cpp:208-224,282-293`, `store/uploader.cpp:103-265`, `sink/native_pack_sink.cpp:256-282`, `sink/pack_sink.cpp:254-353,461-520`, `src/dmi/engine.py:944-970,992-1062` |
| `tla/PackPipelineLoop.tla` | who runs a cycle when: `loop()`'s wait and its wait-again-once rule, `flush()`'s `try_lock_until` loop, and close()'s flush followed at once by `stop()` | `storage_service.cpp:424-478,490-550`, `engine.py:944-970` |
| `tla/PackPipelineUpload.tla` | one pack's `UploadOne` (attempts, uploader backoff, preflight HEAD/GET, PUT, HEAD, `Remove`, the booking of its end) over `S3Client::ExchangeWith` (a request's own retries and backoff, the progress-callback abort) under a `Cancellation` | `store/uploader.cpp:66-82,103-265`, `store/s3_client.cpp:76-99,214-386`, `store/cancel.h` |

The lease is an environment signal here, `held` or `lost`. Quarantined,
refused and abandoned all look alike to a cycle, which only asks
`writer_.held_lease()`. How the lease is kept is LeaseLifecycle.tla's job.
The spool has one owner, this process. An outcome-unknown publish is
resolved at once, landed or not: the quarantine after it outlasts the
statement (LeaseLifecycle O2), so the next replay guard sees how it ended.
Records are packs, and a seal plus the stager's stage is one step.

Properties (`PackPipeline.tla`):

- **Durable**: a pack the sink staged is always in the spool or the bucket.
- **CatalogSound**: only uploaded packs are indexed.
- **ChunkStartsWithLease**: decision 7 as the header states it. No chunk
  *starts* while the writer holds no lease.
- **OwedAtMostOneChunk**: at most one chunk is uploaded and not in the
  catalog, remembered by this process alone.
- **NoOrphanEver**: the strict reading of decision 7. Not the code's claim;
  a witness.
- **OneBatchPastDeadline**: per flush() call, at most one index batch
  starts past the deadline.
- **FlushNeverReconciles**: no flush cycle reconciles.
- **TailDelivered**: `sealed_on_release` implies nothing is left in sink
  memory.
- **FlushBoundary**: flush() returning true means everything staged before
  the call is in the catalog.
- **CloseDelivers**: a drained close() with both seals through has every
  admitted record in the catalog.
- **EventuallyIndexed** (liveness): a staged pack ends up in the catalog.

Constants marked MAIN in the module header are 5b3b632's code shape. The
others are the fix or a mutation.

| Config | Expect | What it shows |
|---|---|---|
| `PackPipeline_main` | holds | every safety property but the two below, through a crash and restart, a lease loss, a transient fault, a user flush and close() |
| `PackPipeline_main3` | holds | the same with three packs in chunks of two (a split, a second chunk), without the crash |
| `PackPipeline_firstchunk` | **violated** | the first chunk of a cycle starts after the lease was lost during the listing (E1) |
| `PackPipeline_firstchunk_fixed` | holds | with the lease checked before the first chunk too |
| `PackPipeline_strictorphan` | violated | a crash between upload and index orphans one chunk with `reconcile_on_start` off: the accepted gap |
| `PackPipeline_live_nocrash` | holds | without a crash every staged pack is indexed, reconcile off |
| `PackPipeline_live_periodic`, `_periodic2` | hold | across a crash, with the reconcile at start and periodic |
| `PackPipeline_live_startonly` | **violated** | the default (`reconcile_interval_s = 0`): a start reconcile that fails other than by a lease refusal is never retried (E2) |
| `PackPipeline_live_failuresowed` | holds | with every failed or partial reconcile pass owed to the loop |
| `PackPipeline_live_noreconcile` | violated | with `reconcile_on_start` off a crash's chunk is never indexed (documented) |
| `PackPipeline_mut_*` | violated | uploads without the lease or while packs are owed, a release that reports a seal it did not stage, batches past the deadline, a flush that reconciles: each breaks its property |
| `PackPipelineLoop_between` | **violated** | a loop cycle between close()'s flush and its stop(), through `waited_again` (E3) |
| `PackPipelineLoop_between_always` / `_starve_always` | holds / violated | deferring every time closes that window but lets endless flushes starve the loop |
| `PackPipelineLoop_starve` | holds | main's "once" never starves the loop |
| `PackPipelineUpload_main` | holds | no request after a cancel; `Remove` only once the object is ours; a foreign object kept; "cancelled" booked only when a cancel ended the upload, and always for a cut last attempt; a cancel ends the upload without any timer firing |
| `PackPipelineUpload_mut_*` | violated | a backoff that sleeps out its timer (uploader's or S3 client's), a cut last attempt booked as a failure |
| `*_vac_*` | violated | each property's interesting state is reached |

Limitations:

- Time is abstract: "past the deadline" and "past the read grace" are flags,
  so the overrun is counted in batches, not seconds. The abort of a cut
  multipart upload (≤ 5 s, `s3_client.cpp:609-619`) is not modelled.
- A chunk's uploads may each succeed, fail or be cut, but the uploader's
  worker pool is not modelled.
- Adoption of dead siblings is not modelled.
- The spool's own validation and quarantine of a bad `.ready` file are not
  modelled.

### Z3 — clock skew, three obligations

| Check | Result |
|---|---|
| real skew `d <= clock_skew_ns` overlaps a holder's admitted statement | `unsat` |
| real skew `d > clock_skew_ns` overlaps | `sat` |
| margin with the `+ clock_skew_ns` term dropped, any `d > 0` | `sat` |
| start wait, predecessor TTL `==` successor TTL | `unsat` |
| start wait, predecessor 30 s vs successor 15 s / 5 s / 0 s | `sat` |
| start wait, `Tp <= Ts + p` (the threshold) | `unsat` |
| start wait, `Tp >  Ts + p` (above it) | `sat` |
| lease deadline (`TTL - skew - 0.1 s` after the claim was sent): a rival finds the row dead within it | `unsat` |
| same, margin `0` | `sat` (only the deadline instant itself) |
| same, deadline without `- clock_skew`, real skew > margin | `sat` |
| client gives up at the lease deadline on a fenced statement the server still runs | `sat` |
| ... by `>=` stamp delay + `clock_skew` + `0.1 s` | `unsat` |
| pre-#159 claim confirmation (fresh claim bound for the read-back) past its own deadline, `S > TTL/3 - 0.1 s` | `sat` |
| same, `S <= TTL/3 - 0.1 s` | `unsat` |

`unsat` here is a proof over all timings — for any publish timeout, any
declared bound, any expiry and any schedule — not a sample of one. All
quantities are reals: no discretisation and no bound on the magnitudes.

The second obligation discharges the change `204a8d2` made at
`native_capture.py:374-378`, which added `+ clock_skew_s` to the default start
wait. The result: **the wait outlasts a crashed predecessor iff
`predecessor_ttl <= successor_ttl + publish_timeout`**, and `clock_skew_s`
cancels out of that condition entirely. It pays for real replica skew exactly
and buys **zero** headroom against a TTL mismatch.

The config comment at `native_capture.py:270-280` states that threshold: the
default wait *"is guaranteed to outlast a crashed predecessor only when its
TTL is at most lease_ttl_s + publish_timeout_s"*, *"20 s on these defaults"*.
On the shipped Python defaults — `lease_ttl_s = 15`, `publish_timeout_s = 5` —
any predecessor TTL up to **20 s** is outlasted. The native default TTL is
30 s (`catalog_writer.h:34`), which processes predating these knobs used, so a
restart after one of those gives up 10 s early. The script's second start-wait
check pins exactly that case (successor 15 s / 5 s / 0 s, so a 20 s wait
against a 30 s row, 10 s short). Every start-wait check also carries the
successor's own fence margin,
`lease_ttl_s - publish_timeout_s - clock_skew_s >= 0.1 s`
(`native_capture.py:366-372`, `catalog_writer.cpp:148-160`): a successor
outside it is refused at construction and never waits at all.

The comment also names the skew that matters *at start*: it *"assumes
clock_skew_s bounds the offset between the replica that stamped the
predecessor's row and the one serving the read"*, so between ClickHouse
**replicas**, not between DMI hosts. `reject_live` compares
`head.live_until_ns > head.now_ns` with both sides stamped server-side inside
one query (`lease_coordinator.cpp:148-163,232`), so the successor's own clock
never enters it.

The third obligation is #159's lease deadline
(`lease_coordinator.h:30-64`, `lease_coordinator.cpp:40-51`). It holds over all
timings: within `deadline = sent + TTL - clock_skew - 0.1 s`, no head read on
any replica within the declared skew finds the row dead. Without the 0.1 s
margin it fails only at the deadline instant, because `live_until > now` is
strict. Without the skew term it fails as soon as the real skew passes the
margin.

Two further results follow from it:

- **The client can give up on a fenced publish statement that is still
  running.** This holds even though `clickhouse_request_timeout_s >=
  2 x publish_timeout_s`. The lease deadline binds first, and it can end before
  the statement's `max_execution_time`. The overlap is bounded by the claim's
  stamp delay plus `clock_skew` plus 0.1 s. Such a give-up is an unknown
  outcome: the writer quarantines, and safety is unaffected.
- **Before #159, a claim made without a lease could be confirmed past its own
  deadline exactly when `clock_skew > TTL/3 - 0.1 s`.** That is the threshold
  the code comment at `lease_coordinator.cpp:222-229` gives. It sits inside
  what the service accepts: its window floor is `clock_skew <= TTL/2 - 0.3 s`,
  and that is wider than `TTL/3 - 0.1 s` whenever `TTL > 1.2 s`. So the fix was
  needed in reachable configurations.

The start-wait obligation and the fence margin are unchanged at `5b3b632`
(`native_capture.py:482-485`, `:454-462`; `catalog_writer.cpp:148-160`).

### CBMC — ring arithmetic, the eager safety net, cancellation

#### Payload ring spans (`payload_ring_span.cpp`)

Five assertions over `payload_compute_spans`, for every capacity in `1..64`,
every `head` up to `2^40` and every `nbytes`:

| Assertion | With the precondition | Without it |
|---|---|---|
| P1 `len1 + len2 == n` | SUCCESS | SUCCESS |
| P2 `off1 + len1 <= cap` | SUCCESS | SUCCESS |
| P3 `off2 + len2 <= cap` | SUCCESS | **FAILURE** |
| P4 spans disjoint | SUCCESS | **FAILURE** |
| P5 no span byte lies in the unconsumed region `[tail, head)` | SUCCESS | **FAILURE** |

Witness for the P3/P4 failures: `cap = 22`, `head = 15`, `tail = 0` — so seven
bytes are free — and `n = 1152921504606846983`. The second span runs past the
end of the buffer.

P5 is the property the precondition exists for: a reservation never overwrites
bytes the consumer has not released. It picks any byte of the reservation and
any unconsumed position and asserts that the spans put them at different buffer
offsets. It takes head's offset from `off1`, which both branches of
`payload_compute_spans` set to `head % capacity` on their first line, rather
than recomputing `head % cap`: asking the solver to prove two 64-bit dividers
equal does not finish. P5 therefore checks the span lengths and the wrap point
and trusts that one assignment. With it the proof takes a few seconds.

Re-run at `5b3b632`: `payload_ring.cuh` is unchanged since `71e6663`, and
the verdicts are the same.

#### Legacy ring room (`legacy_ring_room.cpp`, #160)

`legacy_ring_room(cap, head, tail)` (ring_engine_py.cu:141-144), over every
64-bit `cap`, `head` and `tail` — no bounds:

| | Shipped | `-DUNSATURATED` (pre-#160) |
|---|---|---|
| R1 room ≤ cap | SUCCESS | **FAILURE** |
| R2 equals `payload_free_bytes` / `task_free_slots` when head − tail ≤ cap | SUCCESS | SUCCESS |
| R3 room = 0 at or past capacity | SUCCESS | **FAILURE** |
| R4 a later tail read never shows less room | SUCCESS | **FAILURE** |
| R5 an admitted reservation keeps the accounting in the ring | SUCCESS | SUCCESS |
| R6 the record ring's check (:715-720) keeps it in the ring | **FAILURE** | **FAILURE** |

R6 is the record ring's admission, which #160 did not change: it still
computes `payload_cap - payload_used` unsaturated and so admits everything
once the accounting is past the ring. Whether the record path can get there
is open; see the findings.

#### Eager safety net (`eager_safety_net.cpp`, #160)

A bounded model of the legacy ring's reserve / flush / publish / drain
protocol. `prepare_step`, `available_capacity`, `available_task_slots`,
`reserve_one` and `flush_and_wait` are copied from ring_engine_py.cu;
commit_step's reserve / force_eager decision (base.py:395-404) and
HookPoint's eager branch (point.py:290-337) are transcribed; the producer
publishes at the device heads, never reading the tails; the drain may consume
any prefix of what is published at every lock boundary, fail at any point
(mid-flush included), and every step may stop after prepare_step, leaving a
reservation nothing publishes. Task ring 1-3 entries, payload and staging
16-64 bytes, hooks of 1-64 bytes; the fast set runs 2 steps of up to 2 hooks,
`cbmc_eager_3x2` 3 steps.

| | Shipped | `PRE160_TASK_CHECK` | `PRE160_STEP_RESERVE` | `UNSATURATED` | `PRE160` |
|---|---|---|---|---|---|
| E1 no publish into an unconsumed task slot | S | **F** | S | S | **F** |
| E2 no payload write over unconsumed bytes | S | S | S | **F** | **F** |
| E3 room never exceeds the ring | S | S | S | **F** | **F** |
| E4 every reservation published once (no failure) | S | S | **F** | S | **F** |
| E5 refusal only after a phantom or a failed drain | S | S | **F** | S | S |
| E6 a refusal on a failed drain raises the drain's failure | S | S | S | S | S |
| V1 vacuity: eager hook publishes after a flush | F | F | F | F | F |
| V2 vacuity: logic_error refusal reachable | F | S | F | F | S |
| V3 vacuity: task accounting passes the ring | F | F | F | F | F |

Each variant turns one #160 change back, and each fails the property that
change's commit message names: bytes-only admission overwrites slot 0 at
task_cap + 1 hooks (E1); a needs_eager step reserved twice leaks entries
(E4) until reserve_one refuses with nothing to blame (E5); the unsaturated
room wraps after a phantom and admits an overwrite (E2, E3). V2 is SUCCESS
where reserve_one has no task check, because then it never refuses.

#### Cancellation sleep (`cancel_sleep.cpp`, #162)

One iteration of `Cancellation::SleepFor`'s loop (cancel.h:78-93),
transcribed with the chrono types as the integers they hold; clock reads
ordered and below 2^63:

| | Shipped | `-DANY_CLOCK` |
|---|---|---|
| C1 a wait is positive and within the remaining duration | SUCCESS | SUCCESS |
| C2 an armed deadline bounds the wait | SUCCESS | SUCCESS |
| C3 a non-positive wait only after the deadline passed (no spin) | SUCCESS | SUCCESS |
| C4 "slept it out" only once the duration elapsed | SUCCESS | **FAILURE** |

The lost-wakeup property is argued, not modelled: every write the sleeper
waits on holds `mutex_` and notifies, and the sleeper checks and waits under
that mutex with a predicate over `generation_`.

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
safety argument. On a replicated deployment it takes two settings, not one:
`select_sequential_consistency=1` on every deciding read
(`clickhouse_client.cpp:374`) is only the read half, and means something only
if every deciding write waited for the same quorum, which is `insert_quorum`
(`LeaseCoordinator::quorum_write`, `lease_coordinator.cpp:37-43`, and its twin
in `version_allocator.cpp`). `insert_quorum` is optional and unset by default,
so a replicated catalog must set it. The model assumes both are in force; it
does not check either.

**The non-linearizable store model is a generous over-approximation.** With
`Linearizable = FALSE` a deciding read may observe any subset of the in-flight
inserts on top of what has replicated. Real ClickHouse replicas are not that
adversarial. The `nonlin_*` counterexamples are therefore "this is what you are
exposed to if the setting is not in force", not "this exact interleaving will
occur". Their value is the shape of the failure and which obligations fall, not
a probability.

**`LeaseLifecycle`'s requests take time, but a renewal's three are one.**
Since the re-sync a request is issued by one step and completed `0..MaxReq`
ticks later, and cut at the lease deadline, so `O1` no longer assumes requests
finish in half the TTL; `O1_fast_*`, `O1_slowreq*` and `O1_slow_*` find where
latency costs the lease. A renewal's three requests are still modelled as one,
and a claim's head read takes no time.

**Late-landing statements are modelled in `PublisherLease` only.** With
`Timeouts = TRUE` a claim INSERT given up on may land at any later time, and a
fenced statement left running lands or aborts by its cap (`LateLand`,
`GoneLand`, `GoneAbort`); `AllSafety` holds under them. `LeaseLifecycle` and
`PackPipeline` still settle an outcome-unknown statement at once, landed or
not, and rely on the quarantine outlasting it (`O2`). The quarantine window
still has no skew allowance (`O2_skew`, `O2_selfref`); since #159 that costs
a retry tick, not the lease.

**`LeaseLifecycle`'s skew runs one way.** Every live row is seen `Skew` ticks
longer than its true expiry (`hExp` adds `Skew`), as if every read went to a
replica lagging by the full bound. A replica that reports a row dead early, or
successive reads that disagree in opposite directions, are not modelled.

**`Stop` is never enabled in `LeaseLifecycle`.** Every shipped config sets
`AllowStop = FALSE`, so no verdict depends on it. Its tombstone also differs
from the code's: the model overwrites the head's expiry with `now`, instantly
and as seen by every replica, and only while ClickHouse is up; the code inserts
a separate tombstone row (`lease_coordinator.cpp:70-90`) that can fail, be read
through a lagging replica, or land with an unknown outcome.

**Each allocator in `VersionAllocator` allocates once.** An allocator runs one
`allocate_version()` call to `done`, `published`, `refused` or `failed` and
stops, so the model says nothing about successive calls from one process, and
it has no cross-call monotonicity invariant (that a process's second version
exceeds its first). `FloorMonotonic` compares a returned version with the
watermark at that moment, not with the same process's earlier versions.

**The ring checks are arithmetic and a bounded protocol model.**
`payload_ring_span.cpp` checks `payload_compute_spans` against its
precondition, and P5 takes head's buffer offset from `off1` rather than
recomputing `head % cap`. `eager_safety_net.cpp` checks reserve / flush /
publish / drain for two steps of two hooks (three steps in `cbmc_eager_3x2`);
three hooks did not finish in 30 minutes. The ready words, the head and tail
atomics, their memory ordering and the CUDA side are not modelled.
`cancel_sleep.cpp` checks `SleepFor`'s arithmetic; its lost-wakeup freedom is
argued from the mutex discipline, not checked.

**`LeaseLifecycle` abstracts `LeaseCoordinator` to its contract, so every
HOLDS verdict in its tables is conditional on `PublisherLease.tla` discharging
that contract.** A contested head — two rows at one term,
`lease_coordinator.cpp:237-246` — is not modelled. It can only *add* `kHeld`
refusals, so the refutations (`O2_*`, `O3_rival`, `O3_unkrival`,
`O5_cosweep_nolock`) survive under it; the HOLDS verdicts do not stand on their own. Read them as
"holds, given the coordinator behaves as `PublisherLease.tla` says it does".

**`LeaseLifecycle`'s `ttl/6` time grain cannot see a sub-tick race.** One tick
is the lease thread's own period, which makes `ttl/6`, `ttl/3` and `2*ttl`
exact, but anything finer than a sixth of the TTL is invisible to it. The real
start poll is `ttl/10` clamped to 50-500 ms — *finer* than one tick — so a
start the model reports as refused purely at an expiry boundary would be
retried sooner in reality. In the other direction, `MaxLate = 0` means the
lease thread wakes exactly on its tick; real OS scheduling and slow requests
can only make it later, which strictly reduces the number of renewal attempts
in the window. `O1_late` runs `MaxLate = 1`, and `O1_slowreq3` / `O1_slowreq`
find where lateness breaks `O1`.

**Counts and trace depths for REFUTED runs are scheduling-dependent; counts
for completed runs are exact.** Stated again here because it is the most
commonly misread number in the tables. A refuted run's count tells you nothing
reproducible. A completed run's count does, and every completed run above
reproduced its recorded figure exactly when re-run with `check.sh` (the
`LeaseLifecycle` ones since `NoSelfRefusal` was tightened).

**`overrun` holding at the base constants proves nothing.** See the vacuity
note in the `PublisherLease` section: at `MaxTerm = 3` the takeover race is out
of budget, so the obligation holds because its subject is unreachable, and
the same goes for the base run's `AllSafety`. The `base5` / `base5_ovr`,
`noovr0` / `ovr0` and `noovr1` / `ovr1` pairs carry that claim. The same trap
caught `LeaseLifecycle_O3_stops`, disclosed above.

**Every result is bounded.** TLC explores the state space cut off by the
constants in each `.cfg` — at most three lease ids, five or eight time steps,
two manifest chunks, two or three concurrent actors, twenty-four ticks of lease
lifetime. An invariant reported as holding holds *within that bound*.
`CeilingNeverBinds` and the vacuity guards probe whether a specific bound hid
behaviour; they do not turn a bounded check into a proof. `payload_ring_span.cpp` is
likewise bounded at capacity 64 and `head < 2^40`, and `eager_safety_net.cpp`
at two steps of two hooks. The Z3 results, `legacy_ring_room.cpp` (every
64-bit value) and `cancel_sleep.cpp` are unbounded.

**Two or three actors, not N.** The TLA+ models run with one to three
concurrent processes. A protocol bug that needs four simultaneous claimants
would not be found. The full three-process `SpoolOwnership` configs, in which
every process stages and adopts, have not finished; three-process races are
covered piecewise, with one stager and two adopters.

**Liveness is modelled only in `SpoolOwnership` and `PackPipeline`.** Their
`live_*` configs check `AdoptLive`, `ChargeDrains`, `EventuallyIndexed`,
`LoopNotStarved` and `CancelEnds` under weak fairness. Every other property
here is a safety property: the lease specs say nothing about a publisher making
progress, and the contested-head quarantine — a deliberate liveness cost — is
not measured. `NeverLatches` is a safety invariant about a latch being
*reachable*, not a claim about recovery.

**The specs model the code as of the revision they were written against**
(`5b3b632`). They are not regenerated from the source, and only the CBMC
copies are checked against it (`cbmc/sync_check.sh`). Re-read the `.tla`
header comments against the cited lines before trusting a result after the
code changes.
