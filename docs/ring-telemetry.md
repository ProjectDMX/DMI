# Legacy ring telemetry

`RingEngine.get_stats()` returns a `RingFlushStats` snapshot of host-only,
cumulative counters for waits and copies the legacy ring already performs.
Reading it does no CUDA work and adds no synchronization; each field is
monotonic on its own.

| fields | what is counted |
|---|---|
| `cpu_payload_head`, `cpu_payload_tail_committed`, `cpu_task_head`, `cpu_task_tail_committed` | reservation heads and drain tails |
| `prepare_ok_calls`, `prepare_flushed_calls`, `prepare_oversized_calls` | `prepare_step` outcomes |
| `prepare_sync_{calls,ns,errors}` | stream synchronizations inside `prepare_step` (flushed and oversized steps) |
| `prepare_drain_{calls,ns,errors}` | forced flushes inside `prepare_step` |
| `toggle_wait_{calls,ns,errors}` | node-toggle pre-apply waits (`set_toggle_pre_apply_sync(True)` only) |
| `staging_wait_{calls,ns,errors}` | drain waits for pinned staging space (an uncontended check reads no clock) |
| `ready_commit_{tasks,us,invalid}` | per task, summed: first seen published to committed after its D2H copy |
| `d2h_copy_errors`, `d2h_sync_errors` | failed D2H copies and synchronizations |
| `pageable_copy_{calls,ns,bytes}` | staging-to-pageable copies in the P2P thread |

`dmi.sync_stats` reads them as versioned dicts:
`synchronization_stats(engine)` (prepare outcomes and waits, toggle waits),
`pipeline_stats(engine)` (drain and P2P stages) and
`gather_sync_stats(source)` (an integration's per-HookPoint row-table
uploads, reuses and pre-write waits, from `source.row_table_stats()`). A reader returns `{"available": False, ...}` rather than
zeros when a counter is missing, and raises on a negative one.

Building a legacy ring logs its device payload ring, pinned staging and task
entries at INFO (`dmi.engine`). The payload ring is allocated when the ring
is built, so a serving framework that sizes its KV cache from the free memory
it profiles afterwards loses KV capacity byte for byte.
