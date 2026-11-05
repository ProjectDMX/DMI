# In-process consumers of the legacy ring

The legacy ring delivers each captured tensor, sliced per request, to one
consumer. Besides the C++ ClickHouse host engine (`DMXHostEngine`), a ring can
deliver to an in-process consumer in the same Python process:

```python
from dmi.engine import MonitoringEngine
from dmi.transport import native

queue = native.BatchQueue(max_bytes=512 << 20)
engine = MonitoringEngine(model_id="m", ring_consumer=queue)
...
for (model_id, shard_rank, act_name, layer_no, node_id, step_seq,
     last_in_step, tensor, req_ids, segments, *lease) in queue.drain_batches(64, 50):
    ...
```

| consumer | handoff | GIL on the P2P thread |
|---|---|---|
| `DMXHostEngine` | one row per (request, hook, step) to ClickHouse | no |
| `SliceQueue` | one item per (request, hook, step); `drain_slices()` | no |
| `BatchQueue` | one item per (hook, step) with a segments table; `drain_batches()` | no |
| a Python callable | one call per (request, hook, step) | yes, per call |

`MonitoringEngine(ring_consumer=...)` accepts a `SliceQueue`, a `BatchQueue`
(which switches `RingConfig.batch_submit` on) or a callable. A ring built with
no consumer warns (`RingWithoutConsumerWarning`): its captures are drained and
dropped.

## The batch handoff

With `RingConfig.batch_submit` and a `BatchQueue`, a packed step hands each
hook's whole tensor over once, with an int32 `[N, 4]` segments table of
`(dim0_offset, db_start, db_end, orig_position)` rows and the matching
request ids; requests with no rows are omitted. `final_logits` rows are
indexed by the request's original position.

Batched (HF) steps are not packed, so they cannot be handed over whole. With a
`BatchQueue` they arrive as **one-request items**, one per request slice --
the per-slice fallback, delivered through the same queue:

| capture | `tensor` | `req_ids` | `segments` |
|---|---|---|---|
| HF token hook, `[batch, q_len, hidden...]` | the request's rows, `[rows, hidden...]`: its last `db_end - db_start` query positions (all `q_len` if fewer) | `[req_id]` | `[[0, db_start, db_end, orig_position]]` |
| HF attention matrix, `[batch, heads, q_len, kv_len]` | the request's slice, `[heads, rows, keys]`: its query rows and its real keys | `[req_id]` | `[[0, db_start, db_end, orig_position]]` |
| HF `final_logits` | `[logits_to_keep, vocab]` | `[req_id]` | `[[0, end_token - logits_to_keep, end_token, orig_position]]` |

For token hooks the usual reconstruction `tensor[dim0_offset : dim0_offset +
(db_end - db_start)]` yields the request's rows (`dim0_offset` is 0). An
attention item is taken whole: its first dimension is heads, not rows. The
attention hooks are `blocks.attn.hook_pattern` and
`blocks.attn.hook_attn_scores` (C++ batch sinks also receive an `is_attn`
flag). The items of one hook share its `step_seq`; `last_in_step` is set on
the last item of the step's last hook. A consumer exception on a fallback item
is counted in `submit_exceptions` like any other. Packed attention matrices
are not delivered on either path.

The tensor is a view of a pooled pageable buffer: consumers must not keep a
view past their handling of the item (clone what they keep). Pool blocks
return to the pool when the last view dies, on any thread, also after the
engine stopped. `RingConfig.p2p_pool_cap_bytes` bounds the idle bytes the pool
retains; `0` disables the pool, and every step's buffer is then allocated at
its exact size (pool blocks are size classes: powers of two, 64 KiB at
least).

## Backpressure and drop attribution

`BatchQueue` is bounded by retained bytes, `SliceQueue` by items and,
optionally, retained bytes (`SliceQueue(max_bytes=...)`, 0 = no byte cap).
Retained bytes are the allocations the queued items keep alive, not their
logical lengths: a 32-byte payload in a 64 KiB pool block is charged 64 KiB,
and items that are views of one buffer (the slices of one step) share a single
charge until the last of them is drained. Item metadata (segments table,
request ids) is not charged; `dropped_bytes` counts logical payload bytes. An
empty queue always admits one item. On overflow the item is shed, counted,
and attributed: a drop notice names the requests that lost data
(`drain_drop_notices()`). No notice means nothing was lost; a
saturated notice buffer raises a sticky flag for one poll window, meaning
every request in flight over that window may be incomplete.

`BatchQueue.set_layer_class_map([(act_name, layer, rank), ...], default)`
installs drop ranks (0 = needed every step, 1 = needed when the request ends,
2 = no deadline). On overflow the queue evicts queued items strictly less
urgent than the arriving one (worst rank first, oldest first) until it fits;
when evicting all of them still would not make room, it evicts nothing and
refuses the newcomer. Evicting a view frees its shared buffer only once no
remaining item, of any rank, and not the arriving item itself still refers to
it. Without a map it is a FIFO that drops the newcomer.

## Capture leases

`RingTransport.set_step_context(capture_lease=native.CaptureLease(id))`
attaches an integration-defined identity to a step. The lease arrives with
every slice, batch and drop notice of the step, and `lease.idle` turns true
once every payload carrying it has been consumed or dropped. A lease built as
`CaptureLease(id, [(act_name, layer, rank), ...])` carries a drop-rank map
that orders eviction for the steps sharing it (`CaptureLease(id, policy)`
shares the map without copying it).

## Delivery barrier and consumer failures

`RingEngine.drain_to_sink_and_wait(timeout_ms)` flushes the ring and waits
until every flushed task has been processed by the P2P thread -- every
consumer call for every slice or batch captured so far has returned. It
returns False on timeout (0 waits forever). A finite `timeout_ms` bounds the
whole barrier: one deadline, taken at entry, covers the producer stream's
completion, the flush (which can wait behind staging saturated by a slow
consumer) and the delivery wait. `flush_and_wait()` only waits for the
ring-to-P2P handoff.

Pending step metadata belongs to the ring: when an engine is torn down with
steps whose payloads never arrived (a forward that failed after publishing its
metadata), their contexts are released with it, so their capture leases turn
idle.

An exception thrown by a consumer is logged once and counted:
`submit_exceptions`, `last_sink_error`. With `set_abort_on_sink_error(True)`,
`sink_failed` turns true at the first one and the next `prepare_step` raises.

## Freshness and flushing

Every delivered payload carries `capture_ts_us`, the steady-clock time the
drain first saw the task published, so `now - capture_ts_us` measures the
export latency including the drain's flush policy
(`drain_flush_entry_threshold`, `drain_flush_timeout_us`, ...).
`RingEngine.request_flush()` asks the drain to flush every task reserved so
far as soon as it is published, without waiting on the producer stream.
