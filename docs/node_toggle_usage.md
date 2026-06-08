# Node-Toggle — Usage & Configuration

Runtime enable/disable of individual monitoring (producer) hooks inside an
already-captured CUDA graph, between replays, via `cudaGraphNodeSetEnabled` — no
re-capture. Idle (all hooks off) is near-baseline; cost scales with the number of
*enabled* hooks. vLLM decode path only.

## Quick start (vLLM)

Pass node-toggle options in `--additional-config` (or the matching `DMX_*` env
vars). Node-toggle **requires full-decode CUDA graphs**:

```bash
vllm serve <model> \
  --compilation-config '{"cudagraph_mode": "FULL"}' \
  --worker-cls integration.vllm_adapter.DMXGPUWorker \
  --additional-config '{
    "dmx_hook_selection": "hidden-states",
    "dmx_node_toggle": true,
    "dmx_enabled_hooks": "0:24,0:25,0:26,0:27",
    "dmx_db_host": "localhost", "dmx_db_port": 9000, "dmx_db_table": "offload"
  }'
```

## Configuration keys

| Key (`additional_config`) | Env | Default | Meaning |
|---|---|---|---|
| `dmx_node_toggle` | `DMX_NODE_TOGGLE` | `false` | Master switch. Off → stock always-on behaviour (unchanged). |
| `dmx_enabled_hooks` | `DMX_ENABLED_HOOKS` | `""` | Which hooks fire. See **enabled-set semantics** below. |
| `dmx_lazy_toggle` | `DMX_LAZY_TOGGLE` | `false` | Defer the device apply to each graph's first replay (per-graph lazy). See **eager vs lazy**. |
| `dmx_drain_flush_timeout_us` | `DMX_DRAIN_FLUSH_TIMEOUT_US` | **50000 when toggle on**, else 0 | Bound export latency for sparse-hook monitoring (see **sparse-hook export**). |

### Enabled-set semantics (`dmx_enabled_hooks`)

Comma-separated `hook_type:layer` pairs. `hook_type` 0 = hidden-states
(`resid_pre`); layer omitted (`"14"`) → global hook (layer −1).

- **unset / `""`** → toggle gate **inactive**: every active hook fires (all-on).
- **`"none"`** → explicit **empty set**: every hook disabled (toggle-0, near-baseline).
- **`"0:24,0:25"`** → exactly those hooks fire; all others disabled.

(Do **not** use a nonexistent layer like `"0:99"` to mean all-off — use `"none"`.)

## Requirements & constraints

- **`cudagraph_mode=FULL`** for decode. The worker raises at startup if a subset
  is requested but no full-decode graph was bound. *Piecewise* graphs hold only a
  model subset, so different graphs capture different hook subsets and fail the
  uniform-hook-set requirement. Note: vLLM may **auto-downgrade FULL →
  FULL_AND_PIECEWISE** if the attention backend lacks full-graph support
  (e.g. FlashAttention `UNIFORM_BATCH`); node-toggle still binds the full *decode*
  graphs, and if an unregistered (runtime-captured) graph ever replays the worker
  **fails loud** rather than silently desyncing.
- **`gpu_padding_strip` is auto-forced off** when node-toggle is on (only the
  basic producer's kernel node is toggle-recorded; the prefix/chunked strip
  producers are not). A warning is emitted if you set it on explicitly.
- Model code needs **no changes** — toggle is entirely backend + adaptor; hooked
  models only place `HookPoint`s and declare `get_hook_specs()`.

## eager vs lazy

- **Eager** (default): `set_active_hooks` flips every bound exec immediately.
  Safe **only at a quiescent point** (no replay of any bound exec in flight) —
  i.e. a static config applied once after warmup. It does **not** wait on replay
  events, so mutating an exec mid-replay is UB.
- **Lazy** (`dmx_lazy_toggle=true`): the device flip is deferred to each graph's
  next replay (`ensure_graph_current`), guarded by a per-graph replay event. Use
  this for **dynamic / per-step reconfigure** (adaptive toggling at runtime).

## Sparse-hook export

With a small enabled set, per-step data volume is low; without a timed flush the
drain waits on byte/entry thresholds that sparse steps never hit, so data buffers
until the ring fills or shutdown. The default `dmx_drain_flush_timeout_us=50000`
(50 ms, when toggle is on) forces a periodic flush. Lower it for tighter export
latency; raise/zero it if you batch large volumes and prefer threshold-only.

## Scope (current)

The enabled set is **engine/step-level**, applied at a step boundary — **not
per-request-within-a-batch**. When multiple requests are batched into one decode
step, a hook is either on or off for the whole batch tensor. True per-request
adaptive monitoring (different hooks per request in the same step) would need a
request → step-union → graph layering plus row-sparse producer copy; that's a
future feature, not implemented here. For now, take the per-step union of what
any request needs and route/filter downstream.

## Runtime (programmatic) reconfigure

For dynamic toggling from your own code (e.g. an adaptive monitor), the transport
exposes:

```python
from monitoring import ring_transport
t = ring_transport.get_active()
t.set_active_hooks([(0, 24), (0, 25)])        # eager: flip now (quiescent only)
t.set_active_hooks_lazy([(0, 26)])            # lazy: deferred per-graph apply
t.clear_toggle()                              # teardown: gate off, registry cleared
```

`(hook_type, layer)` pairs; the enabled set is the single source of truth driving
capacity-reserve, meta-push, and device node-enable in lockstep.

## Fail-loud guards (what raises)

Node-toggle treats any desync risk as fatal (the worker must terminate):
- requesting a subset with no bound full-decode graph,
- captured graphs with non-uniform hook sets, or an incomplete graph↔exec registry,
- a runtime-captured (unregistered) graph reaching replay,
- a lazy device apply / replay-event error.

## Mechanism / internals

See `docs/node_toggle_implementation.md` (design + optimizations) and
`docs/node_toggle_migration_scope.md`. The self-checking end-to-end test is
`tests/ring/smoke_toggle_vllm.py`.
