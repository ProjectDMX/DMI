# DMI integration API v1

`dmi.api.v1` is the supported boundary between DMI core and
an external framework integration. It specifies how an integration constructs
the runtime, describes model hooks, publishes metadata before a forward pass,
and reads captured tensors back.

This is not an end-user inference API. Framework-specific scheduler, worker,
model, compilation, and serving assumptions belong in that framework's own
integration repository.

## Import and version check

```python
import dmi.api.v1 as dmi

if dmi.DMI_INTEGRATION_API_VERSION != 1:
    raise RuntimeError("This integration requires DMI integration API v1")
```

`DMI_INTEGRATION_API_VERSION` identifies this facade's contract. It is not the
DMI package version or native-extension ABI version. This module always
exports `1`, even after a future `v2` module is added.

Importing the facade loads DMI's pure-Python mirror of the native hook
definitions. It does not load the compiled extension, construct an engine,
allocate ring buffers, start host threads, import an inference framework, or
register framework-specific presets. Re-exported objects use the same
transport and selection state as DMI core.

## Complete lifecycle

An integration follows this order:

1. Create the optional ClickHouse host pipeline.
2. Create one `MonitoringEngine`, which starts the host pipeline and GPU ring.
3. Construct a framework-specific `BackendAdaptor` subclass.
4. Call `attach_model()` after model loading and before compilation, warmup, or
   a monitored forward.
5. If the framework performs warmup or graph capture, call
   `set_capture_enabled(False)` first. Call `set_capture_enabled(True)`
   afterward only when capture should begin; permanent null mode stays false.
6. Call `before_forward()` once immediately before each monitored forward.
   Installed `HookPoint` producers fire during that forward.
7. After the final producer activity, call `close()` and do not use that engine,
   adaptor, or attached model for further monitored work.

Only one ring transport may produce in a process. Hook dispatch uses one
process-global active native ring binding. Construct the adaptor only after its
engine's ring exists. The adaptor remains bound to that ring; if it is
replaced, recreate the adaptor and reinstall hooks.

### Minimal construction

```python
import dmi.api.v1 as dmi

clickhouse = dmi.ClickHouseClientConfig()
clickhouse.host = "localhost"
clickhouse.port = 9000
clickhouse.database = "default"
clickhouse.table = "offload"

stage = dmi.StageConfig.clickhouse_insert(clickhouse, parallelism=1)
queue = stage.input_queue
queue.max_batch_items = 1024
queue.max_batch_size = 2 * 1024**3
queue.high_watermark_items = 1024
queue.high_watermark_size = 2 * 1024**3
stage.input_queue = queue
host = dmi.DMXHostEngine(stage)

ring = dmi.RingConfig()
ring.payload_ring_bytes = 256 * 1024 * 1024
ring.pinned_staging_bytes = 256 * 1024 * 1024

engine = dmi.MonitoringEngine(
    model_id="my-run",
    host_engine=host,
    ring_config=ring,
)
try:
    adaptor = MyFrameworkAdaptor(engine, model_id="my-run")
    adaptor.attach_model(model, hook_selection="full")
    for framework_step in steps:
        adaptor.before_forward(framework_step)
        output = model(framework_step.inputs)
finally:
    engine.close()

# An authoritative successful run also checks asynchronous host failures.
host.raise_if_failed()
```

Passing `host_engine` to `MonitoringEngine` causes the wrapper to call
`host.start()`. Do not start it first. To exercise capture without database
writes, omit `host_engine`; the ring still runs but has no host submit target.
Construction is not transactional: if ring creation fails after host startup,
the caller must explicitly stop the retained `host` object.

## Runtime and adapter API

### `CaptureSchedule` and `MonitoringConfig`

```python
CaptureSchedule(
    step_stride: int = 1,
    step_offset: int = 0,
    warmup_steps: int = 0,
    capture_prefill: bool = True,
    capture_decode: bool = True,
    request_stride: int = 1,
    request_offset: int = 0,
    warmup_requests: int = 0,
)

MonitoringConfig()
MonitoringConfig(schedule=CaptureSchedule(...))
```

`CaptureSchedule` is DMI's framework-neutral capture policy. Strides must be at
least one; offsets and warmup counts must be nonnegative. Invalid values raise
`ValueError` during construction.

```python
schedule.should_capture_request(request_id: int) -> bool
schedule.should_capture_step(
    step_id: int,
    phase: Literal["prefill", "decode"],
) -> bool
```

The predicates apply warmup, then offset, then stride. Step selection also
honors `capture_prefill`/`capture_decode`; an unknown phase raises `ValueError`.
`MonitoringConfig` currently contains only this schedule. Its default factory
creates a distinct `CaptureSchedule` for each config instance.
`MonitoringEngine` stores the config, while concrete adaptors decide whether
and how to apply it; the engine does not enforce the schedule by itself.

### `HostEngineConfig`

```python
HostEngineConfig(
    stages: Sequence[StageConfig],
    start_on_init: bool = True,
)
```

This Python wrapper is the `MonitoringEngine(db_config=...)` construction path.
Current `MonitoringEngine` accepts exactly one ClickHouse insert stage. With
`start_on_init=True` it constructs and starts a `DMXHostEngine`.
`start_on_init=False` leaves that engine stopped, but `MonitoringEngine` does
not expose the constructed engine through a public accessor. External v1
integrations should therefore keep this flag true, or construct and retain a
`DMXHostEngine` themselves and pass it through `host_engine=` when they need
explicit lifecycle control or diagnostics.

### `MonitoringEngine`

```python
MonitoringEngine(
    *,
    config=None,
    model_id: str | None = None,
    host_engine=None,
    db_config=None,
    enable_ring_transport: bool = True,
    ring_config=None,
    ring_payload_mb: int = 4096,
    ring_pinned_mb: int = 4096,
    ring_task_entries: int = 65536,
)
```

`MonitoringEngine` owns the native host pipeline and capture transport. The
transport and its native ring engine are implementation details; framework
integrations use the lifecycle, capacity, and adaptor APIs below instead of
accessing those objects.

- `model_id` is required when `host_engine` or `db_config` is supplied. Captured
  rows actually use `StepContext.model_id`; integrations must keep them equal.
- `host_engine` accepts a constructed `DMXHostEngine` and is the self-contained
  v1 path for database offload.
- `db_config` accepts public `HostEngineConfig` and constructs the native host
  engine from its one stage. Prefer explicit `host_engine` when the caller
  needs direct access to `raise_if_failed()` and other diagnostics.
- `host_engine` and `db_config` are mutually exclusive.
- Supplying `ring_config` starts a ring even when
  `enable_ring_transport=False`.
- With no explicit `ring_config`, the wrapper defaults to a 4096 MiB GPU
  payload ring, 4096 MiB pinned staging ring, and 65,536 task entries. These
  are much larger than raw `RingConfig()` defaults.
- `config` is retained for DMI's higher-level configuration path; this driver
  does not interpret it.

```python
engine.enable_ring_transport(ring_config, model_shape=None)
engine.ring_capacities() -> RingCapacities
engine.capture_enabled -> bool
engine.set_capture_enabled(enabled: bool) -> None
engine.next_auto_group_id() -> int
engine.close() -> None
```

`enable_ring_transport()` stops the engine's previous ring, clears its active
binding, constructs and starts the replacement, optionally sets a
`ModelShapeConfig`, and makes the replacement process-global. Ignore its legacy
return value; v1 does not expose the underlying transport type. Replacement is
not transactional: if new-ring construction fails, the old ring is already
gone.

`ring_capacities()` returns an immutable `RingCapacities` snapshot:

| Field | Meaning |
| --- | --- |
| `payload_bytes` | GPU payload-ring capacity. |
| `staging_bytes` | Pinned host staging capacity. |
| `task_entries` | Metadata/task-ring entry capacity. |
| `effective_bytes` | `min(payload_bytes, staging_bytes)`, the per-step byte ceiling. |

It raises `RuntimeError` if no ring is active. Reading it calls three native
capacity getters; integrations should cache the snapshot for a loaded engine
instead of querying it per forward.

`capture_enabled` is false when no ring is active or capture is suppressed.
It reports the engine-owned suppression flag; it does not detect whether a
caller separately cleared the process-global binding with
`deactivate_ring_transport()`.
`set_capture_enabled()` changes native producer behavior and metadata emission
together. It is a lifecycle operation: the caller must ensure no forward can
overlap it. An actual state change performs the native CUDA synchronization
needed by the device-global null-mode flag; asking for the current state is a
no-op. Disabling also clears any previous per-step eager fallback decision and
leaves installed hooks attached. It does not drain or flush database work and
does not apply `MonitoringConfig.schedule`; the framework adapter owns that
scheduling decision. Model attachment remains valid while capture is disabled.
It raises `RuntimeError` when no ring is active. If the native transition
fails, Python-visible capture flags remain unchanged.

`next_auto_group_id()` returns engine-scoped integers starting at zero. It is
not synchronized for concurrent callers.

`close()` stops and flushes the ring, clears the active binding, closes host
input, and stops the host engine. It is terminal and effectively idempotent.
Shutdown exceptions are suppressed, so an integration requiring an
authoritative final read must ensure every worker reaches this close path and
should separately check native host failures.

Closing does not disable or uninstall HookPoints: they retain hook IDs and the
old payload tensor. Treat the attached model as terminal too. A later CUDA
forward—especially after another engine becomes active—can combine stale hook
bindings with the new global engine.

### `deactivate_ring_transport`

```python
deactivate_ring_transport() -> None
```

Clears the process-global Python transport and native active-engine pointer.
It does not synchronize, drain, stop, close, or mutate a `MonitoringEngine`,
and it does not uninstall model hooks. Call it only after producer activity is
quiescent. v1 has no corresponding activation API; resuming requires
re-enabling or replacing the ring, recreating the adaptor, and reinstalling
hooks. `set_capture_enabled(True)` does not reverse this teardown operation.
This is teardown plumbing, not a substitute for `MonitoringEngine.close()`.

### `BackendAdaptor`

```python
BackendAdaptor(engine: MonitoringEngine, model_id: str)
```

This abstract base implements hook attachment and the shared pre-forward
driver. A framework integration implements:

```python
detect_model_shape(self, model) -> ModelShapeConfig
detect_parallel_ranks(self) -> tuple[int, int, int, int]  # TP, DP, EP, PP
is_pp_first(self) -> bool
is_pp_last(self) -> bool
build_step_context(self, *framework_state) -> StepContext | None
on_capacity_exceeded(self, ctx: StepContext) -> None
```

Optional overrides are:

```python
adapt_for_cpu_direct(self, ctx: StepContext) -> StepContext
_spec_needs_eager(self, spec: HookSpec) -> bool
_warn_once_capacity(self, ctx, total_bytes: int, n_hooks: int) -> None
```

Their defaults are identity, `False`, and no-op. Stable instance state used by
integrations is:

| Attribute | Meaning |
| --- | --- |
| `engine` | Owning `MonitoringEngine`. |
| `model_id` | Identifier associated with this adaptor. |
| `model_shape` | Detected `ModelShapeConfig`, or `None` before attachment. |
| `active_hook_specs` | Tuple snapshot of bound, selected, PP/TP-filtered hooks in firing order. |

The `HookSpec` objects inside `active_hook_specs` remain mutable. Internal
transport handles and mutable inventory fields are not part of the v1
integration contract.

`detect_model_shape()` must put the real TP world size into
`ModelShapeConfig.tp_size`. The base copies only `tp_rank` from
`detect_parallel_ranks()` and otherwise merely clamps the existing TP size to
at least one.

#### `attach_model(model, hook_selection="full")`

Attachment requires an existing ring, but capture may be temporarily disabled.
The model must implement
`get_hook_specs()` and return bound `HookSpec` objects in exactly the order
their `HookPoint` modules fire during a forward.

The method detects model shape/ranks, resolves selection, disables unavailable
or non-owning PP/TP hooks, installs ring fields on remaining HookPoints, and
publishes the selected inventory. It mutates HookPoints and is not
transactional; call it before graph capture. An unknown
selection raises `ValueError`, and an executable inventory containing
`module=None` raises `RuntimeError`.

One engine supports one active model inventory. Attaching a second adaptor
invalidates the first inventory while the first model's HookPoints remain
installed, so forwarding the first model afterward can mismatch FIFO metadata
and producers.

#### `before_forward(*framework_state)`

Call this once immediately before the corresponding model forward. It returns
early when capture is disabled or `build_step_context()` returns `None`.
Otherwise it:

1. builds a `StepContext`;
2. calls `plan_step()` to compute each firing hook's shape and 16-byte-aligned
   payload size;
3. calls `commit_step()` to reserve the complete step;
4. invokes capacity fallback callbacks if one step cannot fit;
5. publishes request/range/rank context; and
6. pushes hook metadata in `active_hook_specs` order.

The forward must then fire the same hooks in the same order. Native consumers
pair FIFO metadata with producer arrivals, so an inventory/order mismatch can
mislabel tensors.

`commit_step()` records in its returned `StepReservation` whether capacity was
already available, existing work had to be flushed, the complete step is
oversized, or no reservation was needed. `before_forward()` itself still
returns `None`. With no computable hook shape, reservation is skipped and no
hook metadata is emitted. Exceptions propagate; the driver does not roll back
partial state.

For each firing spec, `_spec_needs_eager()` is ORed into the step's eager
decision even when reservation succeeds. For an oversized step the callback
order is `adapt_for_cpu_direct()`, `on_capacity_exceeded()`, then
`_warn_once_capacity()`, and metadata is built from the adapted context.

`close()` delegates to `MonitoringEngine.close()`.

### `StepPlan`, `StepReservation`, `plan_step()`, and `commit_step()`

```python
StepPlan(total_bytes: int, hook_count: int, needs_eager: bool)

adaptor.plan_step(ctx: StepContext) -> StepPlan
adaptor.commit_step(
    ctx: StepContext,
    plan: StepPlan | None = None,
) -> StepReservation
```

`StepPlan` is an immutable tuple value describing one forward's reservation.
Its fields are the aligned payload bytes, the number of nonempty hook tensors,
and whether any selected hook requires eager execution. `plan_step()` walks
`active_hook_specs` once. It performs no ring operation, CUDA synchronization,
tensor allocation, or database work.

`commit_step()` owns reservation, eager-fallback selection, request/range
context publication, and FIFO hook-metadata publication. Passing a plan avoids
recomputing hook shapes; omitting it makes `commit_step()` call `plan_step()`
once. Its result is:

| `StepReservation` | Value | Meaning |
| --- | ---: | --- |
| `SKIPPED` | -1 | Capture is disabled, no transport exists, or no hook has a computable shape. |
| `RESERVED` | 0 | The step was reserved in current ring capacity. |
| `FLUSHED` | 1 | Existing ring work was flushed before the step was reserved. |
| `OVERSIZED` | 2 | The complete step cannot fit; per-hook eager fallback is active. |

For `SKIPPED` caused only by zero computable hooks, `commit_step()` still
publishes the step context; the metadata loop emits no hook records. A supplied
plan is not revalidated against its context or the current hook inventory, so
the integration must keep all three consistent.

A framework that must decide dispatch before the real forward may call
`plan_step()` during its read-only preflight and later pass the same object to
`commit_step()`. It must commit exactly once, after the real request layout is
known and before any installed `HookPoint` fires. Ordinary integrations call
`before_forward()`, which performs the same plan-then-commit sequence and
returns `None`.

### `StepContext`

```python
StepContext(
    model_id: str,
    flattened: bool,
    req_ids: list[str],
    token_ranges: list[tuple[int, int]],
    dim0_offsets: list[int],
    kv_offsets: list[int],
    tp_rank: int = 0,
    dp_rank: int = 0,
    ep_rank: int = 0,
    pp_rank: int = 0,
    batch: int = 0,
    q_len: int = 0,
    kv_dim: int = 0,
    logits_to_keep: int = 0,
    token_ids_dtype: torch.dtype | None = None,
    actual_q_len: int | None = None,
)
```

This mutable dataclass performs no validation or copying. The adaptor must make
every field match the tensor layout that will execute.

| Field | Contract |
| --- | --- |
| `model_id` | Stored row namespace; normally matches engine/adaptor ID. |
| `flattened` | `False` for `[batch, q_len, ...]`; `True` for packed `[total_rows, ...]`. |
| `req_ids` | Requests in tensor slicing order, not scheduler-map order. |
| `token_ranges` | Per-request half-open logical intervals `[start, end)` represented by this forward. |
| `dim0_offsets` | Batched: request batch index. Packed: starting tensor row. |
| `kv_offsets` | Per-request first real key position in attention matrices. |
| rank fields | Physical producer coordinates stored in metadata. |
| `batch` | Positive for batched shapes; exactly zero selects packed formulas. |
| `q_len` | Execution query rows, including graph padding when present. |
| `kv_dim` | Key dimension used by attention-score/pattern shapes. |
| `logits_to_keep` | Materialized logit rows; zero means all `q_len` rows. |
| `token_ids_dtype` | Per-step fallback dtype for token-ID metadata. |
| `actual_q_len` | Optional unpadded rows for specs marked `dim0_is_actual_tokens`. |

Request IDs, ranges, and offsets must have matching lengths and order. Set
`actual_q_len` only when the corresponding producers really strip padding;
otherwise metadata can describe fewer rows than the producer writes.

`plan_step()` does not consult `token_ids_dtype`; it uses
`HookSpec.dtype` or model dtype. Therefore an executable token-ID spec must set
its real dtype explicitly. Treat `token_ids_dtype` as metadata fallback, not
as the sole dtype declaration.

`transport_kwargs()` returns request/range/rank fields for the internal
transport's step-context call and intentionally excludes shape fields.

## Hook specification API

### `HookPoint`

`HookPoint()` marks an activation boundary in a model. Before ring
installation, while disabled, or for CPU tensors, it returns its input
unchanged. Once installed and enabled for a CUDA tensor, it makes the tensor
contiguous, dispatches one native producer, and returns that contiguous tensor.
It preserves values but may change object identity and memory layout.

Integration-relevant state is:

- `enabled`, selected before compilation; changing it can trigger Dynamo
  recompilation;
- hook type, layer ID, and shared payload, installed by
  `install_ring_hooks()`; and
- padding-strip state, installed by `configure_hook_padding_strip()`.

With the eager safety net active, a HookPoint reserves current capacity,
flushes if needed, or copies the complete tensor through CPU-direct fallback
when one tensor is larger than the ring. CPU-direct fallback does not apply GPU
padding stripping.

Legacy Python forward/backward callbacks are not a CUDA-graph replay mechanism.
DMI capture uses the native producer called directly from
`HookPoint.forward()`. The legacy callback objects and callback-management
methods are not part of the v1 framework-integration contract.

### `HookSpec`

```python
HookSpec(
    hook_type: int,
    module: torch.nn.Module | None,
    layer_no: int = -1,
    dtype: torch.dtype | None = None,
    allow_token_cnt_mismatch: bool = False,
    dim0_is_actual_tokens: bool = False,
)
```

| Field | Contract |
| --- | --- |
| `hook_type` | Stable `HOOK_TYPE_*` ID determining stored name, shape class, and ownership. |
| `module` | Bound `HookPoint`, or `None` only in a non-executable planning inventory. |
| `layer_no` | Global layer index; `-1` denotes a model-global hook. |
| `dtype` | Metadata dtype override; `None` uses model dtype except for a per-step token-ID override. |
| `allow_token_cnt_mismatch` | Lets the consumer derive dim 0 from actual bytes for dynamically routed payloads. |
| `dim0_is_actual_tokens` | Makes planning/metadata use `StepContext.actual_q_len`; it does not itself strip padding. |

`HookSpec` is mutable and unvalidated. List order is part of the transport
contract because metadata is consumed FIFO.

### `HookRowBasis` and `hook_row_basis`

```python
hook_row_basis(hook_type: int) -> HookRowBasis
```

`HookRowBasis.TOKEN_ROWS` means per-step payload cardinality follows
token/execution rows. `HookRowBasis.REQUEST_ROWS` means it follows
request/logit rows. In v1 only `HOOK_TYPE_FINAL_LOGITS` is request-row based;
all other registered hooks are token-row based. Unknown IDs raise `ValueError`.

Row basis does not specify dimension order, TP/PP ownership, or padding-strip
eligibility.

### `ModelShapeConfig`

```python
ModelShapeConfig(
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    vocab_size: int = 0,
    intermediate_dim: int = 0,
    num_experts: int = 0,
    top_k: int = 0,
    tp_size: int = 1,
    tp_rank: int = 0,
)
```

The first five fields are required. A zero optional dimension disables shapes
that need it: `vocab_size` for logits, `intermediate_dim` for `MLP_POST`,
`num_experts` for router logits, and `top_k` for top-k IDs/weights. Shape
calculation uses `tp_size` for local attention/MLP dimensions; it does not
validate divisibility, positivity, or rank bounds.

### `compute_hook_shape`

```python
compute_hook_shape(
    hook_type: int,
    cfg: ModelShapeConfig,
    batch: int,
    q_len: int,
    kv_dim: int,
    logits_to_keep: int = 0,
) -> list[int]
```

`batch > 0` selects batched shapes with a leading batch dimension. `batch == 0`
selects packed shapes without it. Let `T = cfg.tp_size`:

| Shape class | Batched | Packed |
| --- | --- | --- |
| Hidden | `[batch, q_len, hidden_dim]` | `[q_len, hidden_dim]` |
| Q | `[batch, q_len, num_heads//T, head_dim]` | `[q_len, num_heads//T, head_dim]` |
| K or V | `[batch, q_len, max(1, num_kv_heads//T), head_dim]` | `[q_len, max(1, num_kv_heads//T), head_dim]` |
| Z | `[batch, q_len, num_heads//T, head_dim]` | `[q_len, (num_heads//T)*head_dim]` |
| Attention scores/pattern | `[batch, num_heads//T, q_len, kv_dim]` | `[num_heads//T, q_len, kv_dim]` |
| MLP post | `[batch, q_len, intermediate_dim//T]` | `[q_len, intermediate_dim//T]` |
| Token IDs | `[batch, q_len]` | `[q_len]` |
| Final logits | `[batch, logit_rows, vocab_size]` | `[logit_rows, vocab_size]` |
| Router logits | `[batch, q_len, num_experts]` | `[q_len, num_experts]` |
| Top-k IDs/weights | `[batch, q_len, top_k]` | `[q_len, top_k]` |

For batched logits, `logit_rows` is `q_len` when `logits_to_keep <= 0`,
otherwise `min(q_len, logits_to_keep)`. For packed logits it is `q_len` when
nonpositive, otherwise exactly `logits_to_keep`.

The function returns `[]` for an unknown type or unavailable optional
dimension. It does not inspect dtype or validate dimensions, divisibility, or
`tp_size`; invalid inputs can produce invalid shapes or ordinary Python errors.

### `align_up`

```python
align_up(x: int, a: int) -> int
```

Rounds `x` up to `a`. The caller must supply a positive power-of-two
alignment; this function does not check that precondition. DMI reservations
use 16-byte alignment per hook.

### `make_model_shape_from_hf_config`

```python
make_model_shape_from_hf_config(
    hf_config,
    dtype: torch.dtype | None = None,
) -> ModelShapeConfig | None
```

The object need not inherit from a Transformers class. Attribute mapping is:

| Output | Input attribute |
| --- | --- |
| `hidden_dim` | `hidden_size`, otherwise `n_embd` when the first attribute is absent |
| `num_heads` | `num_attention_heads`, otherwise `n_head` when absent |
| `num_kv_heads` | `num_key_value_heads`, otherwise `num_heads` |
| `head_dim` | `head_dim`, otherwise `hidden_dim // num_heads` |
| `dtype` | Explicit argument, otherwise `torch_dtype`, otherwise `torch.float16` |
| `vocab_size` | `vocab_size` or zero |
| `intermediate_dim` | Truthy `intermediate_size`, then `n_inner`, otherwise zero |
| `num_experts` | `num_experts` or zero |
| `top_k` | Truthy `num_experts_per_tok`, then `top_k`, otherwise zero |
| TP fields | Always size 1, rank 0 |

For GPT-2, absent intermediate size becomes `4 * hidden_dim`. The function
returns `None` only when hidden size or attention-head count is absent. It does
little validation: a present-but-`None` primary attribute does not fall back to
its alias, and conversion/division errors propagate.

### `install_ring_hooks`

```python
install_ring_hooks(
    specs: list[HookSpec],
    ring_payload: torch.Tensor | None = None,
) -> None
```

Assigns each spec's hook type, layer ID, and shared engine payload to its bound
HookPoint. A spec with `module is None` raises `RuntimeError`. The list is not
prevalidated, so a later failure leaves earlier modules changed.

Repeated installation overwrites the three ring fields, but omitted HookPoints
remain installed. The function does not select hooks, change `enabled`, set
strip mode, compute shapes, activate a transport, or validate capacity. A live
CUDA producer requires the payload tensor belonging to the active ring.

### `configure_hook_padding_strip`

```python
configure_hook_padding_strip(
    hook_point: HookPoint,
    row_count_tensor: torch.Tensor | None,
    row_bytes: int = 0,
) -> None
```

| State | Producer mode |
| --- | --- |
| `row_count_tensor is None` | Static producer copies the full tensor. |
| Tensor present and `row_bytes > 0` | Prefix producer copies `row_count_tensor[0] * row_bytes`. |
| Tensor present and `row_bytes <= 0` | Chunked producer treats each of the tensor's `K` values as one chunk's byte count. |

The function only assigns HookPoint state. It validates neither dtype, device,
shape, values, nor `row_bytes`. Prefix mode expects a one-element device
`int64` tensor; chunked mode expects a device `int64[K]` tensor. During
compilation/CUDA-graph replay, tensor address and element count and the Python
`row_bytes` value must remain fixed. Update only tensor values on the correct
stream between steps.

## Selection and rank ownership

### `register_preset` and `is_preset_registered`

```python
register_preset(name: str, hook_types: frozenset[int]) -> None
is_preset_registered(name: str) -> bool
```

`register_preset()` permanently adds a name to process-global selection state.
Duplicate names raise `ValueError`; hook IDs and emptiness are not validated.
`is_preset_registered()` checks the same table and returns `True` for presets,
individual selectors, and aliases.

Core v1 supplies:

- `full`: every native hook;
- `hf-only`: residual-pre, final normalization, attention pattern, and logits;
- one selector for every hook's short name; and
- aliases `hidden-states`, `hidden_states`, `logits`, and `token-ids`.

Framework integrations may register additional names. Importing v1 alone does
not register them.

### `select_hook_specs`

```python
select_hook_specs(
    specs: list[HookSpec],
    mode: str,
    cfg: ModelShapeConfig | None = None,
) -> list[HookSpec]
```

`mode` is a comma-separated union of preset, individual, or alias names. An
unknown token or empty expression raises `ValueError`. The returned list holds
the original spec objects in original order. This function does not mutate
specs, HookPoints, or selection state and does not apply PP/TP ownership.

With `cfg`, it removes MLP-post when `intermediate_dim == 0`, router logits
when `num_experts == 0`, and top-k IDs/weights when `top_k == 0`.

### `hook_belongs_to_pp_rank` and `hook_belongs_to_tp_rank`

```python
hook_belongs_to_pp_rank(
    spec: HookSpec,
    is_first_rank: bool,
    is_last_rank: bool,
) -> bool

hook_belongs_to_tp_rank(spec: HookSpec, tp_rank: int) -> bool
```

The PP function enforces only global first/last-stage restrictions. Per-layer
specs must already correspond to layers owned by the local PP stage.

The TP function keeps all hooks on rank 0 and only TP-sharded hooks on nonzero
ranks. Neither function validates rank bounds or mutates the spec.

`ALL_HOOK_TYPES` is the frozen set of all native hook IDs.
`ATTENTION_WEIGHT_HOOK_TYPES` is the frozen subset containing only attention
scores and attention patterns; it is not every hook in the attention group.

## Hook catalog

These constants are stable numeric IDs shared with the native backend and
stored metadata. ID 10 is intentionally absent because the old attention
`result` hook duplicated `ATTN_OUT`.

The catalog is a universe of valid public activations, not a promise that every
model emits every hook. For valid registered IDs, selection filters the model's
supplied inventory. The low-level preset API does not reject unknown numeric
IDs, so integrations must build presets and inventories from this catalog.
Exact placement is ultimately defined by the model implementation.

In the shape column, `B?` is present only for batched execution, `T` is
`q_len`, `H` is `hidden_dim`, `P` is `tp_size`, `QH` is `num_heads//P`, `KVH`
is `max(1, num_kv_heads//P)`, `D` is `head_dim`, `S` is `kv_dim`, `I` is
`intermediate_dim`, `E` is `num_experts`, `K` is `top_k`, and `Vocab` is
`vocab_size`.

| Constant (ID) | Selector / stored name | Activation | Shape | Ownership |
| --- | --- | --- | --- | --- |
| `HOOK_TYPE_RESID_PRE` (0) | `resid_pre` / `blocks.hook_resid_pre` | Block residual before first norm | `[B?, T, H]` | Per-layer, TP0 |
| `HOOK_TYPE_LN1` (1) | `ln1` / `blocks.hook_ln1` | First/input norm output fed to attention | `[B?, T, H]` | Per-layer, TP0 |
| `HOOK_TYPE_ATTN_OUT` (2) | `attn_out` / `blocks.hook_attn_out` | Complete attention branch result before residual add | `[B?, T, H]` | Per-layer, TP0 |
| `HOOK_TYPE_RESID_MID` (3) | `resid_mid` / `blocks.hook_resid_mid` | Residual after attention, before second norm | `[B?, T, H]` | Per-layer, TP0 |
| `HOOK_TYPE_ATTN_SCORES` (4) | `attn_scores` / `blocks.attn.hook_attn_scores` | Masked/scaled QK scores before softmax | `[B?, QH, T, S]` | Per-layer, all TP shards |
| `HOOK_TYPE_PATTERN` (5) | `pattern` / `blocks.attn.hook_pattern` | Softmax-normalized attention weights | `[B?, QH, T, S]` | Per-layer, all TP shards |
| `HOOK_TYPE_Q` (6) | `q` / `blocks.attn.hook_q` | Current-step projected queries at the model-defined Q hook | `[B?, T, QH, D]` | Per-layer, all TP shards |
| `HOOK_TYPE_K` (7) | `k` / `blocks.attn.hook_k` | Current-step projected keys at the model-defined K hook | `[B?, T, KVH, D]` | Per-layer, all TP shards |
| `HOOK_TYPE_V` (8) | `v` / `blocks.attn.hook_v` | Current-step projected values | `[B?, T, KVH, D]` | Per-layer, all TP shards |
| `HOOK_TYPE_Z` (9) | `z` / `blocks.attn.hook_z` | Attention-weighted values before output projection | Batched `[B,T,QH,D]`; packed `[T,QH*D]` | Per-layer, all TP shards |
| `HOOK_TYPE_LN2` (11) | `ln2` / `blocks.hook_ln2` | Post-attention/second norm output | `[B?, T, H]` | Per-layer, TP0 |
| `HOOK_TYPE_MLP_IN` (12) | `mlp_in` / `blocks.hook_mlp_in` | Exact tensor passed into the MLP/MoE module | `[B?, T, H]` | Per-layer, TP0 |
| `HOOK_TYPE_MLP_OUT` (13) | `mlp_out` / `blocks.hook_mlp_out` | Complete MLP/MoE branch before residual add | `[B?, T, H]` | Per-layer, TP0 |
| `HOOK_TYPE_RESID_FINAL` (14) | `resid_final` / `hook_resid_final` | Last block residual before final norm | `[B?, T, H]` | Global, TP0, PP last |
| `HOOK_TYPE_EMBED` (15) | `embed` / `hook_embed` | Token embeddings or supplied input embeddings | `[B?, T, H]` | Global, TP0, PP first |
| `HOOK_TYPE_POS_EMBED` (16) | `pos_embed` / `hook_pos_embed` | Explicit additive position embedding | `[B?, T, H]` | Global, TP0, PP first |
| `HOOK_TYPE_FINAL_LN` (17) | `final_ln` / `hook_final_ln` | Final model norm output | `[B?, T, H]` | Global, TP0, PP last |
| `HOOK_TYPE_TOKEN_IDS` (18) | `token_ids` / `token_ids` | Input or scheduled token IDs | `[B?, T]` | Global, TP0, PP first |
| `HOOK_TYPE_FINAL_LOGITS` (19) | `final_logits` / `final_logits` | Materialized full-vocabulary logits | `[B?, R, Vocab]` | Global, TP0, PP last |
| `HOOK_TYPE_MLP_POST` (20) | `mlp_post` / `blocks.hook_mlp_post` | Activated dense MLP intermediate before down projection | `[B?, T, I//P]` | Per-layer, all TP shards |
| `HOOK_TYPE_ROUTER_LOGITS` (21) | `router_logits` / `blocks.mlp.hook_router_logits` | Raw expert-gate output before selection | `[B?, T, E]` | Per-layer, TP0 |
| `HOOK_TYPE_TOPK_IDS` (22) | `topk_ids` / `blocks.mlp.hook_topk_ids` | Expert IDs returned by routing and consumed by experts | `[B?, T, K]` | Per-layer, TP0 |
| `HOOK_TYPE_TOPK_WEIGHTS` (23) | `topk_weights` / `blocks.mlp.hook_topk_weights` | Weights paired with selected experts | `[B?, T, K]` | Per-layer, TP0 |

“Per-layer” means the native table permits any PP stage; the integration still
must include only layers owned by that stage. `FINAL_LOGITS` is the only
request-row hook. All others, including attention matrices, are token-row
hooks.

A hook ID does not imply one universal dtype. Metadata dtype precedence is:

1. `HookSpec.dtype`;
2. `StepContext.token_ids_dtype`, only for token IDs when the spec has no
   override; and
3. `ModelShapeConfig.dtype`.

For example, built-in integrations may use int64 or int32 token IDs, int32
expert IDs, and float32 routing weights. The supplying model inventory is
authoritative.

## Native host and ring API

The following classes are loaded lazily from DMI's compiled native backend.

### `RingConfig`

Construct `RingConfig()` and set fields before engine construction. The native
engine copies the object and allocates resources immediately; later mutations
do not reconfigure it.

| Field | Native default | Meaning |
| --- | ---: | --- |
| `task_ring_entries` | `1024` | Task/control slots; power of two is recommended. |
| `payload_ring_bytes` | `256 MiB` | GPU circular payload capacity; must be 16-byte aligned. |
| `pinned_staging_bytes` | `0` | Pinned host staging; zero inherits payload capacity. |
| `drain_poll_timeout_us` | `100` | Drain poll cadence; must be positive, not a persistence deadline. |
| `drain_flush_task_ratio` | `0.0` | Task-capacity flush fraction; zero disables it. |
| `drain_flush_payload_ratio` | `0.5` | Payload-capacity flush fraction. |
| `drain_flush_entry_threshold` | `0` | Absolute ready-entry trigger; zero disables it. |
| `drain_flush_byte_threshold` | `0` | Absolute ready-byte trigger; zero disables it. |
| `drain_flush_timeout_us` | `0` | Pending-data age trigger; zero disables it. |
| `clone_slices` | `False` | Clone multi-request slices so full assembled tensors can be released sooner. |
| `insert_queue_max_bytes` | `4 GiB` | Reserved field; current v1 does not apply it to host queue limits. |
| `insert_queue_max_items` | `65536` | Reserved field; current v1 does not apply it to host queue limits. |

Flush triggers are ORed, and full ring capacity always forces a flush. For
live readback of small workloads, configure `drain_flush_timeout_us`; the poll
timeout alone does not flush pending rows. Allocation or invalid native
configuration can fail during ring construction.

Actual ClickHouse queue batching/backpressure is controlled by
`StageConfig.input_queue` and its public `QueueConfig`; do not rely on the two
reserved ring fields.

### `ClickHouseClientConfig`

Construct `ClickHouseClientConfig()` and set all fields before passing it to
`StageConfig.clickhouse_insert()`, which copies it by value.

| Field | Default | Meaning |
| --- | --- | --- |
| `host`, `port` | `localhost`, `9000` | Native ClickHouse endpoint. |
| `username`, `password` | `default`, empty | Authentication. |
| `database`, `table` | `default`, `offload` | Destination objects. |
| `secure` | `False` | Enable secure native transport. |
| `client_settings` | empty | Session settings; values must be `bool`, `int`, or `str`. |
| `create_database_if_missing` | `True` | Create the configured database when absent. |
| `drop_existing_database` | `False` | Drop the entire configured database before setup. Destructive; isolated tests only. |
| `client_side_compress` | `none` | `none`, `lz4`, `zstd`, `true`, or `false`. |
| `index_granularity` | `8192` | MergeTree index granularity for a created table. |

Database connection and schema initialization occur asynchronously in stage
worker threads after host start, not in the config/factory constructor.
Existing incompatible tables are not migrated. Current v1 also protects schema
DDL with one process-global one-time guard: use one writer destination per
process or pre-create any later destination yourself.

### `StageConfig`

The supported construction path is:

```python
StageConfig.clickhouse_insert(
    clickhouse_config: ClickHouseClientConfig,
    parallelism: int = 1,
    name: str = "clickhouse_insert",
) -> StageConfig
```

The factory creates the single ClickHouse sink stage accepted by
`DMXHostEngine`. A bare `StageConfig()` has no Python-settable processing
callback and is unusable as a sink. `name` must be nonempty and `parallelism`
positive when the host engine is constructed.

The object also exposes mutable `name`, `parallelism`, and
`thread_name_prefix` fields, plus `input_queue: QueueConfig` and
`ingress_policy: EnqueuePolicy`. `thread_name_prefix` changes only diagnostic
worker labels. Configure all fields before constructing `DMXHostEngine`; the
engine copies the stage.

### `QueueConfig`

```python
QueueConfig()
```

| Field | Default | Meaning |
| --- | --- | --- |
| `min_batch_items` | `1` | Minimum items that make a batch ready. |
| `min_batch_size` | `None` | Optional minimum ready bytes. |
| `max_linger_s` | `None` | Optional maximum wait before releasing a partial batch. |
| `max_batch_items` | `None` | Maximum items dequeued into one insert batch. |
| `max_batch_size` | `None` | Maximum bytes in one batch; also rejects one larger item. |
| `high_watermark_items` | `None` | Maximum queued items before backpressure. |
| `high_watermark_size` | `None` | Maximum queued bytes before backpressure. |

Minimum readiness conditions are ORed. With no linger and thresholds above
actual traffic, rows can wait until more work arrives, a high watermark is
reached, or input closes. Both high watermarks default to unbounded.

### `EnqueuePolicy`, `OnFullPolicy`, and `OnClosedPolicy`

```python
EnqueuePolicy()
```

| Field | Default | Meaning |
| --- | --- | --- |
| `block` | `True` | Wait for queue capacity instead of returning immediately. |
| `timeout_s` | `None` | Optional wait timeout. |
| `on_full` | `OnFullPolicy.RAISE` | Action after a full/timeout outcome. |
| `max_retries` | `0` | Retries used with retry policy. |
| `retry_backoff_s` | `0` | Delay between retries. |
| `on_closed` | `OnClosedPolicy.RAISE` | Action when input is closed. |
| `drop_if_stopping` | `True` | Drop new work while the engine is stopping. |

`OnFullPolicy` values are `RAISE`, `DROP`, `RETRY`, and `ABORT`.
`OnClosedPolicy` values are `RAISE` and `DROP`. Assign the policy to
`stage.ingress_policy` before constructing `DMXHostEngine`.

### `DMXHostEngine`

```python
DMXHostEngine(insert_stage: StageConfig)
```

Construction validates/copies stage configuration but does not connect to the
database. Public lifecycle and diagnostics are:

```python
start() -> None
close_input() -> None
stop(graceful: bool = True, timeout_s: float | None = None) -> bool
request_abort() -> None
join(timeout_s: float | None = None) -> bool
failures() -> list[ThreadFailure]
raise_if_failed() -> None
```

`start()` is asynchronous: it can return before a worker fails to connect or
initialize. `stop()` returning true means threads joined, not that inserts
succeeded. After shutdown, call `raise_if_failed()`; `failures()` returns
records with `stage`, `thread_name`, `where`, `exc_type`, and `exc_what`.

### `ThreadFailure`

`ThreadFailure` is the immutable diagnostic record returned by
`DMXHostEngine.failures()`. Its read-only string fields are:

| Field | Meaning |
| --- | --- |
| `stage` | Stage name in which the failure occurred. |
| `thread_name` | Generated diagnostic worker label, normally the configured prefix plus `.tN`; it is not guaranteed to be the OS thread name. |
| `where` | Failure location such as `worker`, `thread_entry`, `thread_cleanup`, or `enqueue_outputs`. |
| `exc_type` | Exception type name. |
| `exc_what` | Exception message. |

The list is a snapshot. Use `raise_if_failed()` when a failed authoritative
run should raise instead of being inspected manually.

The basic direct lifecycle is:

```python
host.start()
try:
    produce()
finally:
    joined = host.stop(graceful=True, timeout_s=30)
    host.raise_if_failed()
    if not joined:
        raise TimeoutError("DMI host engine did not stop")
```

`close_input()` half-closes submission and lets queued work drain. `join()`
alone does not close input and can wait indefinitely. Abort paths can discard
queued rows, and the engine cannot restart after stop/abort. The `stop()`
timeout applies once to graceful joining and again to abort joining after a
timeout, so the example can wait about 60 seconds. A false result means a
worker may still be active; abort cannot cancel an in-flight database call.

```python
host.submit_direct(
    model_id,
    shard_rank,
    req_id,
    act_name,
    layer_no,
    start_token,
    end_token,
    tensor,
) -> None
```

This queues one already-sliced row and is normally used by the ring engine,
not a framework adaptor. Submission is asynchronous; do not mutate a retained
tensor until draining completes. Worker initialization, processing, and insert
errors appear in host failure records. Synchronous submission state/queue
errors instead raise to the caller; the current ring P2P path can log and drop
such a submission without adding a host failure record.

## Readback API

### `CHClickhouseDriverReadOnly`

```python
CHClickhouseDriverReadOnly(
    host: str = "localhost",
    port: int = 9000,
    username: str = "default",
    password: str = "",
    database: str = "default",
    table: str = "offload",
    secure: bool = False,
    client_settings: dict | None = None,
    primary_key_column_names=(
        "model_id", "request_id", "act_name", "layer_no",
        "shard_rank", "start_token_idx", "end_token_idx",
    ),
    order_by_column_names=None,
    value_column_names=("dtype", "shape", "bytes"),
    decode_strings: bool = True,
    **ignored,
)
```

Construction imports `clickhouse-driver`, validates table/column identifiers,
and stores settings. The network client is lazy until the first query. Known
HTTP ports 8123 and 8443 are rejected because this class uses ClickHouse's
native protocol. Extra keyword arguments are silently ignored.

```python
reader.prefix_get(
    prefix_key: tuple,
    *,
    return_full_key_tuple: bool = True,
)
```

The prefix must be nonempty and no longer than the configured primary key. The
default returns `(complete_key, CPU_tensor)` pairs ordered by configured
`ORDER BY`; `False` returns tensors only. `decode_strings` affects returned key
cells, while tensor payloads are always decoded.

```python
reader.custom_select(query: str, params: dict | None = None) -> list[tuple]
```

Only text whose first token is `SELECT` is accepted. ClickHouse String columns
remain raw bytes and tensors are not decoded automatically. The text check is
not a security boundary; use server-side read-only credentials.

```python
reader.close() -> None
CHClickhouseDriverReadOnly.bytes_to_torch_dtype(value) -> torch.dtype
CHClickhouseDriverReadOnly.torch_decode(dtype, shape, payload) -> torch.Tensor
```

`close()` disconnects idempotently; a later query reconnects. `torch_decode()`
uses `torch.frombuffer()` plus reshape, so the CPU tensor may share the payload
buffer. Clone it before mutation if ownership or writability matters.

Properties `database`, `table`, `primary_keys_columns`, `value_columns`, and
`columns` expose the resolved schema configuration.

### `make_lazy_internal`

```python
make_lazy_internal(
    model_id: str,
    reader: CHClickhouseDriverReadOnly | None = None,
    requirements: InternalRequirements | None = None,
    request_ids: tuple[str, ...] | list[str] | None = None,
    token_ranges: dict[
        str,
        tuple[tuple[int, int], ...] | list[tuple[int, int]],
    ] | None = None,
) -> LazyInternal
```

Construction performs no I/O. The returned handle queries and reassembles a
field on first access, then caches successful results. Missing, failed, or
requirement-detected incomplete reads are not cached. Without a sufficient
requirement, an asynchronously incomplete value can look successful and remain
cached until `clear_cache()` is called.

- `model_id` should uniquely identify one run; otherwise rows from separate
  runs can be merged.
- Nonempty `request_ids` restrict reads. Reassembled tensor fields sort request
  IDs naturally by colon-separated components rather than preserving this
  list, while `token_mask` preserves supplied order. Do not pair a mask with a
  reassembled field unless those orders are known to match. An empty list is
  treated like omission and falls back to an unscoped whole-model query.
- `token_ranges` are exact physical intervals actually captured for those
  requests. Nonempty ranges together with nonempty `request_ids` enable
  `token_mask`; do not invent uncaptured prefix rows. The mask width uses
  absolute token coordinates up to the largest interval end, while tensor
  reassembly concatenates only captured segments. Their sequence axes align
  only when captured intervals form contiguous prefix coverage from zero with
  no gaps; a suffix-only cache-hit capture intentionally produces a wider mask
  than captured tensor. Use the mask as coverage metadata, not as a direct
  tensor index in that case.
- With `reader=None`, defaults use only `DMX_DB_HOST` and `DMX_DB_PORT` and do
  not inherit custom database, table, credentials, or TLS. Production
  integrations should pass an explicit reader and close it themselves.

### `InternalRequirement` and `InternalRequirements`

`InternalRequirement` is the immutable per-field policy record:

```python
InternalRequirement(
    count: int,
    retry: bool = False,
    timeout_s: float | None = 30.0,
    poll_s: float = 0.25,
    match_token_ranges: bool = False,
)
```

Direct record construction does not validate its values. Prefer the validated
builder below, which rejects negative counts/timeouts and non-positive polling
intervals:

```python
requirements = InternalRequirements()
requirements.require(
    field: str,
    *,
    count: int,
    retry: bool = False,
    timeout_s: float | None = 30.0,
    poll_s: float = 0.25,
    match_token_ranges: bool = False,
) -> InternalRequirements
```

`InternalRequirements(counts=None)` may also be initialized from a mapping of
field names to integer counts or `InternalRequirement` records. It copies the
mapping, but it preserves the immutable records and does not revalidate them.
Its remaining methods are:

```python
requirements.copy() -> InternalRequirements
requirements.expected_count(field: str) -> int | None
requirements.requirement(field: str) -> InternalRequirement | None
```

`make_lazy_internal()` copies the supplied requirements, so later changes to
the caller's policy do not alter an existing handle.

### `LazyInternal` and `IncompleteInternalError`

`LazyInternal` is the concrete lazy handle returned by
`make_lazy_internal()`. Construction performs no I/O; dynamic field access
queries and reassembles captured data. `IncompleteInternalError` is the
`RuntimeError` subclass raised when a configured count/range requirement is
not satisfied. Count/retry failures expose diagnostic attributes `field`,
`expected`, and `found`; callers must not assume those attributes exist on
every instance, including all token-range mismatch paths.

The handle supports:

```python
handle.available -> list[str]
handle.clear_cache(field: str | None = None) -> None
handle.require(
    field: str,
    *,
    count: int,
    retry: bool = False,
    timeout_s: float | None = 30.0,
    poll_s: float = 0.25,
    match_token_ranges: bool = False,
) -> handle
```

`count` validates `len(field_value)`: it is a layer count for per-layer tuples
and a batch count for global tensors, not a token or database-row count. With
`retry=True`, synchronous field access polls missing/incomplete data until
success or timeout. `timeout_s=None` can block forever. Database/runtime errors
are not retried. `match_token_ranges=True` performs validation only when both
nonempty request IDs and ranges were supplied; otherwise it is a no-op. For a
per-layer field the current v1 implementation checks only its highest present
layer, not every layer.

Mapped dynamic attributes are:

`attention_output`, `attention_scores`, `attention_values`, `attentions`,
`embeddings`, `expert_ids`, `expert_weights`, `final_hidden`, `final_residual`,
`hidden_states`, `k`, `ln1`, `ln2`, `logits`, `middle_residual`,
`mlp_activation`, `mlp_input`, `mlp_output`, `position_embeddings`, `q`,
`router_logits`, `token_ids`, `v`, and—when both nonempty request IDs and ranges
are supplied—`token_mask`.

Per-layer values are tuples ordered by layer, with tensors left-padded into
batch form. Global fields are left-padded batched tensors. Attention matrices
are left-padded on query and key axes. Outputs are CPU tensors.

Current v1 reassembly does not reconstruct TP-sharded fields across
`shard_rank`, and ClickHouse MergeTree keys do not enforce uniqueness. An
integration needing distributed reassembly or duplicate detection must handle
that explicitly rather than treating the lazy view as an authoritative
cross-rank oracle.

Example:

```python
reader = dmi.CHClickhouseDriverReadOnly(table="offload")
try:
    internal = dmi.make_lazy_internal(
        "my-run",
        reader=reader,
        request_ids=["request-7"],
        token_ranges={"request-7": ((0, 80),)},
    )
    internal.require(
        "hidden_states",
        count=28,
        retry=True,
        timeout_s=30,
        match_token_ranges=True,
    )
    hidden_by_layer = internal.hidden_states
finally:
    reader.close()
```

## Compatibility policy

External integration code should import DMI functionality only from
`dmi.api.v1`.

Compatible additions to v1 include a new export that does not change existing
behavior, an optional argument with a backward-compatible default, or new hook
IDs that do not renumber/reinterpret existing IDs.

A new integration API version is required for:

- removing or renaming an export;
- changing a required signature, field meaning, lifecycle order, or failure
  behavior relied on by integrations;
- changing an existing hook ID, shape class, stored name, row basis, or TP/PP
  ownership; or
- changing producer/metadata ordering.

DMI retains v1 while a supported integration depends on it. Framework version
support belongs to the framework integration package and is independent of the
DMI package release number.

## Non-goals

This contract does not define framework scheduler ordering, worker lifecycle,
model implementations, compilation or graph-dispatch behavior, serving
endpoints, or supported framework versions. It does not make unexported
`dmi.*` implementation details stable.
