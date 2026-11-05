# Per-HookPoint request-row gather and in-graph heads

In packed (vLLM-style) execution one activation tensor per HookPoint holds the
rows of every scheduled request. Request-row gather lets each HookPoint copy
only the rows it needs: the rows of some requests, or only some token
positions of them, chosen independently per HookPoint and per step. The
selection is data in a per-HookPoint row table, so a captured CUDA graph
replays it without recapture.

In-graph projection heads go one step further: instead of copying the
selected rows, a fused node computes `K` fp32 affine scores per selected row
inside the graph, and only the scores leave the GPU.

## The row-table contract

A row table is a stable int64 device tensor read by the producer at every
launch or replay:

| word | meaning |
|---|---|
| `t[0]` | number of segments `n` (clamped to `GATHER_MAX_SEGS` = 256 and to the segments the table's storage holds); `-1` disables the producer for the step |
| `t[1 + 2k]` | source row of segment `k` |
| `t[2 + 2k]` | rows in segment `k` |

The output is always packed: destination offsets are the running prefix of
the segment lengths. `n = -1` publishes nothing (no reservation, no task, no
payload) and needs no metadata; `n = 0` publishes an empty task. A producer
clamps every segment to the captured tensor, so a host-side bug cannot read
or write out of bounds. The torch ops pass the table's element count to the
kernels (never a host read of `t[0]`, which would sync and be baked into a
captured graph), and the device loader reads at most
`min(n, GATHER_MAX_SEGS, (numel - 1) / 2)` segments
(`native/csrc/ring/row_table.h`), so a count word larger than the table
never reads past it.

The table is loaded by all threads in parallel and the destination prefix is
one warp's `__shfl_up_sync` scan; a single segment (a full-row table) takes a
flat whole-grid copy, the cost of the prefix producer.

`configure_hook_gather(hook_point, table, row_bytes)` (in `dmi.hook_rows`)
arms gather mode on an installed HookPoint: it takes precedence over the
strip modes, and `row_bytes` is the packed row stride. Under the eager safety net
a gather HookPoint reserves its full tensor, so on an eager step the
integration writes a full-row table.

Gather tables and heads belong to one attachment: `install_ring_hooks` and
`uninstall_ring_hooks` clear them (`clear_hook_row_features`), releasing the
table, arena and accumulator references and restoring the strip stride that
`configure_hook_gather` replaced. Integrations configure gather and heads
after installing the HookPoints.

`RingEngine.set_gather_host_mirror(dev_base, host_base, row_bytes, rows)`
registers the pinned host copies of the tables. Outside stream capture a
launch whose mirrored row is `-1` is skipped (`gather_host_skips()`); a
capture always launches, because the graph re-reads the device row at every
replay.

## Planning rows: `dmi.hook_rows`

A `HookRows` plan holds the producer's `segments` and the consumer's
attribution: `req_ids`, `token_ranges`, destination `offsets`,
`source_offsets` and `orig_positions`.

* `plan_hook_rows` plans one HookPoint from a request selection and optional
  absolute token windows (`canonical_token_windows` validates and unions
  them; `None` as an end is unbounded).
* `plan_hook_groups` / `HookRowPlanner` plan many HookPoints at once and
  share one plan object between HookPoints whose selections agree.
  `HookRowPlanner` caches request membership for a repeated request order,
  never positions or rows.
* `PackedDemandPlanner` plans declared per-request demand with NumPy array
  operations: a request's rule (HookPoint index and window pieces) is
  compiled once per demand change and reused every step; the step derives
  every HookPoint's rows for the whole batch with one clip, compaction and
  merge. Its plans compare equal to the reference planner's and expose
  `layout_arrays()` so the native side ingests int64 arrays instead of one
  Python tuple per row.
* `full_row_plan` is the static full-row baseline: the whole physical
  tensor `(0, total_rows)` in one run with the step's physical offsets, so a
  gather producer fed it copies exactly what an ungathered producer copies
  (padding rows included) and attributes the same rows to each request. A
  step without scheduled requests gives the empty plan.
* `encode_row_table` writes a plan (or `None` for disabled) into the table
  layout above; `segment_array` gives a plan's segments as an `[n, 2]` array
  and `row_signature` its value identity, so an integration can skip
  rewriting a table whose plan did not change.

A plan with more segments than the table holds falls back to the
un-compacted full payload while keeping the consumer-side selection
(`capacity_fallback`).

## Publishing per-hook layouts

`StepContext.hook_rows` (and `RingTransport.set_step_context(hook_rows=...)`)
maps `(hook_type, layer_no)` to a plan. A planned hook reserves and publishes
its plan's `row_count` rows and carries its own request layout in its
metadata; other hooks use the step layout. Equal plans share one native
layout per step. `orig_positions` keeps each request's index in the
unfiltered step, which position-indexed hooks (`final_logits`) use.

## The toggle and the table gate

The gather, head and fused producers record their kernel nodes inside the
node-toggle capture window (see [node-toggle.md](node-toggle.md)); a fused
node carries both of its keys. A gather producer captured without its node
would keep firing after its key was disabled, while metadata and reservation
excluded it.

`RingTransport.set_prefill_table_gate(True)` is an integration's guarantee that
every producer reads a per-HookPoint table whose `-1` marker also gates eager
and partial-model execution. The per-step spec set then follows the toggle on
every step, and excluded graphs may replay on any step.

In device-toggle mode, a capture outside the toggle window (graphs the host
does not toggle) captures gather producers as device-updatable nodes behind
one planner node per captured forward, each gated by its own table row.

## In-graph projection heads

`ring::producer_fused` runs a HookPoint's row producer and its heads in one
captured node. Each part has its own table and toggle key: the rows under
the HookPoint's key, the heads under `HOOK_TYPE_*_HEAD` (`resid_pre_head`,
`final_ln_head`) with shape `[rows, head_slots]` fp32. The node arena (int32
words, written only between steps) holds per-slot modes, accumulator
columns, fp32 biases and `K x hidden` weights (bf16 or fp32):

| mode | output column |
|---|---|
| `FREE` | NaN |
| `SINGLE` | `w . h + b` |
| `FIRST` | NaN; writes `acc[row][a] = w . h` |
| `MID` | NaN; adds `w . h` to `acc[row][a]` |
| `LAST` | `acc[row][a] + w . h + b` |

A linear head over several layers holds one slot per participating node;
layers run in order inside a forward, so the sum has a fixed order.
Registering or retiring a head rewrites the arena; nothing is recaptured.
Each `(row, slot)` score uses a fixed reduction order (no atomics, no
split-K), so it is bit-identical across batch compositions and grids.

`ring::producer_head` is the standalone head node (fp32 arena:
`K x hidden` weights, `K` biases, split scratch). Selections of at most 16
rows split each row's dot products over 8 hidden slices so a one-row decode
step is not bound by one SM.

`configure_hook_heads(hook_point, head_type=..., layer_no=..., table=...,
arena=..., acc=..., slots=..., weight_bf16=...)` arms the heads. The
heads' spec names the same HookPoint and follows its row spec
(`BackendAdapter.replace_active_specs`), because the heads publish after the
rows. Dispatch keeps the planned tasks and their order on the fast path and
under the eager safety net alike:

| HookPoint | launches |
|---|---|
| row spec + gather table + heads | one fused node (rows, then heads) |
| row spec (static, prefix or chunked) + heads | the row producer, then a head-only fused node |
| heads only (no row spec) | a head-only fused node |

A head-only node carries no row table and is recorded under the head key
alone, so the row and head toggle states stay independent. A head spec
passed to `install_ring_hooks` binds the payload but never replaces the
HookPoint's row key.

## Tests

* `tests/native/ring/test_producer_equivalence.cu`: static, prefix, chunked
  and gather producers (and planner-gated device nodes) against a host model
  of the whole ring state over thousands of eager launches and graph replays.
* `tests/native/ring/test_inline_head.cu`, `test_fused_head.cu`: heads
  against double-precision references, multi-layer chains, batch invariance,
  graph replays with tables and arenas rewritten between replays. Each takes
  `NVCC_EXTRA=-DDMI_TEST_MUTATE_*` mutants that must fail.
* `tests/test_hook_rows.py`: planners against independent enumerations.
* `tests/test_hook_row_features.py`: gather/head attachment lifecycle and
  plan / fast path / eager task agreement for every row mode with heads.
* `tests/test_row_table_clamp.py`: the row-table segment-count clamp
  against table storage, on the CPU.
* `tests/test_per_hook_gather_gpu.py`: end-to-end delivery through
  HookPoints, the transport and the native consumer.
