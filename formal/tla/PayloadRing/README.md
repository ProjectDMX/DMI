# PayloadRing: TLA+ model of the GPU-to-host payload ring

This spec models `native/csrc/ring` at `a987dfe`: pre-forward capacity
reservation, producer publication, drain, record-mode byte reclaim, and
force-flush waiters. `PayloadRing.tla` is the one module. Each `*.cfg` sets
the module's constant flags. The configs are variants of each other, not
copies of the spec.

## What is modeled

| Spec action | Code |
|---|---|
| `StPlan` | the Python step builds a plan (`records.py:367-373` `_reservation_items`; legacy `adapters/base.py:425-473`) |
| `StApply` | `reserve_record` prologue: rethrow the failures, then `apply_pending_record_reclaims` (`ring_engine_py.cu:499-501`, `drain_thread.cpp:287-294`) |
| `StCheck` | capacity decision, OK / FLUSHED / OVERSIZED (`ring_engine_py.cu:444-460` legacy, `503-525` record) |
| `StSync` | `cudaStreamSynchronize` of the producer stream (`ring_engine_py.cu:445,464,504,527`) |
| `StReq`, `StPreWait`, `StBlk` | `force_flush_and_wait`: request half (`drain_thread.cpp:123-132`, under `mu_`) and wait half `wait(lk, pred)` (`135-138`) |
| `StPost` | record post-flush checks: rethrow, apply, "incomplete producer reclaims" (`ring_engine_py.cu:507-513, 530-536, 572-578`) |
| `StReserve` | `DrainThread::reserve` (`drain_thread.cpp:256-260`) or `reserve_record` (`262-285`), then the producers launch. The FLUSHED paths get here with no re-check (`ring_engine_py.cu:467, 537`) |
| `StSN`, `StSNRes` | the legacy eager safety net `hooks/point.py:321-329`: `available_capacity` (`ring_engine_py.cu:658-662`) then `reserve_one` (`667-671` -> `reserve(nbytes, 1)`) |
| `GpuRun` | one producer kernel. Last-block publish is a release-store of `READY|actual` to `slots[task_head % cap]`, then the device heads advance (`producer.cu:128-172`, `task_ring.cuh:422-436`, `publication_word.h`). It is serialized on one stream (`producer.cu:44-48`). The gated skip is `producer.cu:431` |
| `DrainLoopForce` | loop head that snapshots `flush_requested_generation_` (`drain_thread.cpp:401-411`) |
| `DrainLoopNormal` | `scan_ready` + `should_flush` + batch + `flush_state_update` in one `mgmt_mu_` region (`424-449`, `508-531`, `565-601`) |
| `DrainFullScan` | `do_full_flush` iteration head (`344-356`), including the `flush_count==0` break |
| `Account` (operator) | `account_record_task` (`533-562`) |
| `DrainStg` | `staging_cv_` wait (`357-360`, `453-458`) |
| `DrainD2H` | `enqueue_d2h` + `sync_stream` (`361-362`, `603-639`) |
| `DrainCommit` | `cpu_payload_tail_committed_ = cpu_payload_tail_` (`363-366`) plus `submit_to_p2p` (`645-687`) plus `trim_scanned` (`693-704`) |
| `DrainComplete` | `flush_completed_generation_ = flush_generation; notify_all` (`416-420`) |
| `P2PFree` | p2p pop and `notify_staging_freed_bytes` (`drain_thread.cpp:193-227`, `p2p_thread.cpp:353-404`) |
| `ExtReq`, `ExtPreWait`, `ExtBlk` | a second thread calling `flush_and_wait` concurrently (`ring_engine_py.cu:677-681`) |

**Abstractions and assumptions**

- **Units.** 1 unit = `PAYLOAD_ALIGN` (16 B), so `align_up` is the identity. uint64 overflow is not modeled, so the overflow hard errors are out of scope.
- **Counters.** Counters are monotonic, as in the code. Task-slot wraparound is exact (`seq % TCAP`). Payload wraparound is covered by the capacity arithmetic; the two-span split is pure byte arithmetic.
- **Atomic regions.** Each `mgmt_mu_` / `mu_` / `staging_mu_` / `queue_mu_` / `pop_mu_` region is one atomic step. No code path nests two of these mutexes: every lock scope in `drain_thread.cpp` and `p2p_thread.cpp` is block-local and sequential. So lock-order deadlock cannot happen, and only blocking on conditions is checked (TLC deadlock plus liveness).
- **Scan.** `scan_ready`'s loop is atomic. This is equivalent to the real loop because a READY word stays READY until this same drain clears it.
- **Step-thread reads.** The step thread reads the four counters under separate locks, but it modeled as one read. This is sound because only the step thread writes the heads and the tails only grow (`ring_engine_py.cu:632-655`).
- **`should_flush` and the poll sleep.** `should_flush` is over-approximated: it must flush at `>= task_cap` entries or `>= payload_cap` bytes, and any other threshold is a free choice. The poll sleep is stuttering. The normal path gets no fairness, so progress comes from forced flushes only.
- **Not modeled.** The TensorMeta/descriptor pairing in the consumers, CUDA errors (`record_drain_failure`), stop/shutdown, the `strip_t` CPU-direct branch, and byte contents are all left out.
- **Caller contracts assumed.** Producers run on the stream that `prepare_step` syncs. A record item with `needs_reclaim=false` writes exactly its reservation (`records.py:376-387`). A gated-off occurrence is never reserved (`docs/integration-api-v1.md:409-412`). `MUT_GATE_SKIP` breaks that last contract on purpose.

## Properties (invariants unless noted)

- `ReservedGeActual`: reserved >= actual for every produced task.
- `DrainAfterReady`: nothing is scanned unless it was published, and the drain never copies unwritten bytes.
- `CommittedSafe`: `committed <= d2hDone <= cpuPTail <= gpuPHead`. The committed tail never passes a byte whose D2H has not finished.
- `NoPayloadOverwrite`, `NoSlotOverwrite`, `CapacityRespected`: the GPU never writes over unread payload or over an unconsumed READY word, and outstanding bytes/tasks are <= capacity.
- `ReclaimOnce`, `ReclaimExact`, `HeadAccounting`: each reclaim happens at most once, only after its task is produced, and equals exactly reserved - actual. `cpu head + applied reclaims = sum of reservations`.
- `HardErrorsUnreachable`: `SeqMismatch`, `OverReservation`, `IncompleteReclaims`, `ReclaimExceedsHead`, `ReclaimFailureRaised`, `StagingStuck` are never reached.
- `NoPrematureRelease`: a woken flush waiter finds every task that was published before its request already committed.
- `NoLeakAtEnd`: after the final flush, heads equal tails, no reclaim is pending, and every needs-reclaim task was reclaimed once.
- Liveness (`*Liveness.cfg`): `StepWaiterWakes`, `ExtWaiterWakes`, `Termination`, under weak fairness of the drain, GPU, p2p, step thread and waiters.

## How to run

```
TLA2TOOLS=/path/to/tla2tools.jar TLC_META=/tmp/tlc-meta TLC_WORKERS=2 ./run.sh <Config>
# = java -XX:+UseParallelGC -Xmx4g -jar $TLA2TOOLS -workers $TLC_WORKERS \
#        -metadir $TLC_META/<Config>-<pid> -noGenerateSpecTE -config <Config>.cfg PayloadRing.tla
```

Common bounds: `PCAP=3 TCAP=2 SCAP=2 MaxU=2 MaxExtFlush=1 MaxTasks=3`, except
where noted. Runs used TLC built from tlaplus master source (2026-09-24), because the
GitHub release download was blocked by the sandbox proxy. The box was a shared
4-core machine, so wall times are inflated. Mutation configs also list the liveness
properties, but each stops at its first invariant violation or deadlock.

## Results

| Config | Flags | Expect | Result | Generated / distinct / depth | Time |
|---|---|---|---|---|---|
| FaithfulRecord | record, all safety invariants | pass | **PASS** | 30,079,812 / 10,422,853 / 58 | 11m03s |
| FaithfulRecordLiveness | record, MaxTasks=2, liveness | pass | **PASS** (all 3) | 991,010 / 372,161 / 41 | 4m05s |
| LegacyHooksWithinTaskCap | legacy + safety net, MaxPerStep=2 (<= TCAP) | pass | **PASS** | 965,264 / 346,389 / 58 | 33s |
| LegacyHooksWithinTaskCapLiveness | same, MaxTasks=2, liveness | pass | **PASS** (all 3) | 118,302 / 44,515 / 43 | 39s |
| FaithfulLegacy | legacy + safety net, MaxPerStep=3 (> TCAP) | pass | **FAIL** `CapacityRespected` | 12,108 / 6,018 / 12 | 10s |
| FaithfulLegacySlotOverwrite | same, without `CapacityRespected` | (consequence) | **FAIL** `NoSlotOverwrite` | 33,429 / 15,361 / 15 | 16s |
| MutNoGen | completion satisfies all requests | fail | FAIL `NoPrematureRelease` | 287,407 / 127,972 / 16 | 1m22s |
| MutEarlyCommit | committed tail at `flush_state_update` | fail | FAIL `CommittedSafe` | 995 / 593 / 7 | 4s |
| MutEarlyCommitOverwrite | same, without `CommittedSafe` | fail | FAIL `NoPayloadOverwrite` | 54,568 / 25,338 / 12 | 23s |
| MutReclaimOOO | no FIFO sequence check in reclaim | fail | FAIL `ReclaimExact` | 1,593 / 931 / 7 | 6s |
| MutUnderReserve | record item reserves upper-1 | fail | FAIL `ReservedGeActual` | 527 / 318 / 6 | 4s |
| MutUnderReserveConsequence | same, without `ReservedGeActual`/`CapacityRespected` | fail | FAIL `HardErrorsUnreachable` (OverReservation) | 1,195 / 723 / 7 | 4s |
| MutLostWakeup | waiter blocks without checking its predicate | fail | FAIL deadlock | 108,621 / 50,137 / 13 | 33s |
| MutGateSkip | gated-off producer that was reserved | fail | FAIL `ReclaimExact` | 5,237 / 2,660 / 8 | 10s |
| MutGateSkipSilent | same, checking only `LeakImpliesHardError` | (probe) | FAIL: leak with no hard error | 9,043,383 / 3,432,499 / 28 | 2m46s |

### Finding: the legacy eager safety net oversubscribes the task ring

`FaithfulLegacy` reproduces this in 12 steps:

1. A step has 3 firing hooks and `task_ring_entries = 2`. `prepare_step` returns `STEP_OVERSIZED` because `num_hooks > tcap` (`ring_engine_py.cu:444-448`), after syncing and flushing, so the ring is empty. `adapters/base.py:395-397` sets `force_eager`.
2. Each hook takes the safety net (`hooks/point.py:321-324`). It checks only payload bytes (`available_capacity`, `ring_engine_py.cu:658-662`) and calls `reserve_one`. That calls `DrainThread::reserve(nbytes, 1)` (`drain_thread.cpp:256-260`), which adds one task with **no task-capacity check**. After the third hook, `cpu_task_head_ - cpu_task_tail_ = 3 > 2` (`CapacityRespected` fails).
3. The GPU runs the producers. Producer 3 release-stores its word into `slots[2 % 2] = slot 0`, while slot 0 still holds producer 1's READY word that the drain has not consumed. The producer never reads tails (`ring_state.h:9-11`), so nothing stops this (`NoSlotOverwrite` fails, `FaithfulLegacySlotOverwrite`).
4. What happens next depends on timing. If the drain has not scanned seq 0, it reads producer 3's size as seq 0's: wrong byte count, and the TensorMeta pairing shifts by one. If it scanned seq 0 but has not flushed, `flush_state_update` (`drain_thread.cpp:596-597`) then clears producer 3's word. The drain then waits forever at that sequence, and `do_full_flush` still reports completion because it breaks on `pending_entries_ == 0` (`347`). In both cases capture data is lost or misattributed.

`src/dmi/configuration/estimate.py:816-823` says a step over the task cap "falls back to eager
CPU-direct dispatch", but the safety net goes CPU-direct only for bytes. A
likely fix: have `reserve_one` / the safety net also check
`task_cap - (cpu_task_head - cpu_task_tail)` and flush when it is zero.
Record rings are not affected: every record item goes through `reserve_record`,
which checks both caps.

### Other observations

- **Gate contract.** If a gated-off occurrence *is* reserved, which violates the contract, the ring layer may raise no hard error at all. `MutGateSkipSilent` ends with a leaked task slot and payload unit, and nothing in `HardErrs` was raised. The ring layer's checks alone do not detect this contract break. The record consumer's leftover-descriptor check (`record_consumer.cpp:119-127`, surfaced as a flush timeout) is outside this model.
- **FLUSHED paths do not re-check capacity.** Both FLUSHED paths reserve without re-checking capacity (`ring_engine_py.cu:466-467`, `536-537`). The model shows this is safe given the synced stream, exact non-reclaim sizes and the incomplete-reclaim check. It is not safe after a drain failure: legacy `force_flush_and_wait` returns silently (`drain_thread.cpp:128`, `175`). That path is not modeled.
