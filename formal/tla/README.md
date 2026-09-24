# TLA+ specs

TLC models of three concurrency protocols in DMI, each checked against the
code at `a987dfe`. Each directory has one module, a set of `.cfg` files that
are variants of it (faithful, mutations, findings), and a README that maps
spec actions to code lines and lists every config with its expected and
actual result.

## PublisherLease

The ClickHouse publisher lease (`clickhouse_lease.py`, `clickhouse_catalog.py`):
term-based claim with read-back, renew-before-publish, the admission fence,
release tombstones, and quarantine after outcome-unknown errors. The faithful
model (the Python lease, plus the C++ `reject_if_gone` variant) passes
`MutualExclusion` and 8 other safety properties, including with outcome-unknown
faults and crash-restart. All 8 mutations fail. The result rests on three
assumptions: `clock_skew` bounds the difference between *any two* ClickHouse
hosts (not each host's NTP offset); an admitted write lands within
`max_execution_time`; and table reads are linearizable.

## VersionPublish

Snapshot version allocation and publish (allocator claim + read-back,
barrier-and-fence insert, watermark read-back, pinned readers). The faithful
model passes all safety properties and termination. The stale-read configs
and 7 mutation configs fail, as expected. The `Replay` / `ReplayCrash` configs
expose a real gap in the faithful protocol: re-indexing an already-published
pack re-points pinned readers to it over a newer pack.

## PayloadRing

The GPU-to-host payload ring (`native/csrc/ring`): capacity reservation,
producer publication, drain, record-mode byte reclaim, and force-flush
waiters. The record ring passes all safety and liveness properties; 8
mutations fail. `FaithfulLegacy` exposes the legacy task-slot overflow (the
eager safety net reserves tasks with no task-capacity check), fixed in PR #155.

## Getting TLC

You need Java 11+ and `tla2tools.jar`. These results used TLC built from
tlaplus master at commit `4260e471245fce04ad5610bae7125bdc9ef7eec6`
(reports `Version 2026.09.24`). The GitHub release download was blocked in the
environment where the specs were written, so the jar was built from source:

```
git clone https://github.com/tlaplus/tlaplus && cd tlaplus
git checkout 4260e471245fce04ad5610bae7125bdc9ef7eec6
cd tlatools/org.lamport.tlatools
ant -f customBuild.xml info compile compile-test dist   # -> dist/tla2tools.jar
```

(Plain `javac` over `tlatools/org.lamport.tlatools/src` with the jars in its
`lib/` also works.) A release jar from
https://github.com/tlaplus/tlaplus/releases should behave the same for these
specs, but is not what the numbers in the READMEs came from.

## Running

```
TLA2TOOLS=/path/to/tla2tools.jar TLC_META=/tmp/tlc-meta ./run.sh
```

runs every faithful config (the ones expected to pass) and prints PASS/FAIL
per config. It takes about an hour on 4 cores. Pass `Spec/Config` names to
run a subset, e.g. `./run.sh VersionPublish/ReplayRest PayloadRing/LegacyHooksWithinTaskCap`.
Expected-fail configs are run the same way; a FAIL there is the expected result.
Keep `TLC_META` outside the repo; `.gitignore` covers TLC output if it lands here.
