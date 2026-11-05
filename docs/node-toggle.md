# Runtime CUDA-graph node toggling

Node toggling enables or disables individual HookPoint producers inside
already-captured CUDA graphs, between replays, with `cudaGraphNodeSetEnabled`
on the instantiated exec. Nothing is recaptured or re-instantiated and the
model is unchanged. A disabled producer costs an empty-node traversal, so a
step pays only for the hooks it enables, and the enabled set can change from
one step to the next.

The feature is opt-in and applies to the legacy ring transport. With it off
the only code that runs is the per-step spec seam described under
[Behaviour when off](#behaviour-when-off).

## The invariant

Every step, three lanes must agree on which hooks fire:

1. capacity reservation (`BackendAdapter.plan_step` → `prepare_step`),
2. metadata publication (`RingTransport.pre_push_all_metas`),
3. the device node state (the engine's enabled set).

If they disagree, the ring desynchronizes: reserved bytes differ from written
bytes, or metadata is paired with the wrong payload. All three read
`RingTransport.specs_for_step()`:

| Step | `specs_for_step()` |
|---|---|
| toggle inactive | every active spec |
| gated step (a toggled graph replays) | active ∩ enabled ∩ captured |
| ungated step (eager, prefill, partial-model graph) | every active spec |

The integration marks each step with `transport.set_gated_step(...)`. A hook that is
enabled but has no captured node is not effective (it would push metadata
with no payload), and the eager safety net in `HookPoint.forward` skips
disabled hooks on a gated step that fell back to eager dispatch.

## Using it

```python
from dmi import node_toggle

engine = transport._ring_engine          # legacy RingEngine

# 1. Open the capture window before the framework captures its graphs.
#    exclude_capture() is asked at the start of every capture; return True
#    for captures that must not be toggled (e.g. partial-model segment
#    graphs, which would make the registry non-uniform).
node_toggle.begin_capture_window(engine, exclude_capture=is_partial_capture)

framework.capture_graphs()               # creates torch.cuda.CUDAGraph()s

# 2. Bind every graph you want toggled, then close the window.
for graph in framework.full_graphs():
    node_toggle.bind_graph(engine, graph)
node_toggle.end_capture_window(engine)

# 3. Apply a static set (or None to leave the gate inactive) and arm the
#    replay guard.
node_toggle.activate(transport, [(HOOK_TYPE_RESID_PRE, 24)], lazy=True)

# Later, per step:
transport.set_active_hooks_lazy(new_set)
transport.set_gated_step(step_replays_a_toggled_graph)
```

`parse_enabled_hooks()` reads the setting format `"hook_type:layer,..."`:
`""`/unset leaves the gate inactive, `"none"` is an explicit empty set, and a
token without a layer (`"14"`) names a global hook (layer `-1`).

### Eager and lazy application

* `set_active_hooks` (eager) flips every bound exec immediately. It does not
  wait for replays in flight, so call it only at a quiescent point, such as
  once after warmup.
* `set_active_hooks_lazy` updates the enabled set and the host gate now; the
  guarded replay calls `ensure_graph_current()` for the graph about to replay,
  so a reconfiguration costs O(changed nodes × graphs actually replayed).

`cudaGraphNodeSetEnabled` affects only future launches of an exec; launches
already enqueued keep the node state they were enqueued with. The registry
therefore does not wait for a graph's previous replay before mutating it,
which keeps one-step-ahead schedulers asynchronous.
`engine.set_toggle_pre_apply_sync(True)` restores the wait (an event recorded
after every replay, synchronized before the next mutation of that exec).

### The CUDA-graph patch and replay guard

`begin_capture_window` installs a `torch.cuda.CUDAGraph` subclass. It forces
`keep_graph=True` (otherwise the captured template is freed at instantiation
and the recorded node handles dangle; keeping it costs host memory only),
suspends node recording for excluded captures and tags them, and guards
replays once `activate` armed the guard:

* an excluded graph may replay only on an ungated step;
* a graph that is not registered and bound (for example captured after the
  window closed) is refused;
* a bound graph re-instantiated since `bind_graph` is bound again to its new
  exec (the registry then resets that graph's per-exec node state);
* in lazy mode the graph is brought up to date before it replays.

Each refusal raises: the step's metadata is already published, so replaying
would desynchronize the ring. `bind_graph` refuses graphs created from the
unpatched class.

The guard covers only graphs that belong to the active transport's engine:
every capture is stamped with the monitoring session it was captured in and
with whether it captured DMI producers (the process-wide
`producer_launch_count()` across the capture, or the nodes recorded for that
graph). A graph that captured no producer and is not bound -- another CUDA
graph user in the same process -- replays untouched, as does a graph captured
for a different engine. A graph that captured producers, or whose producers
cannot be ruled out, keeps the fail-closed checks above.

The patch is process-global, so monitoring sessions own it.
`begin_capture_window` starts the engine's session; it ends when the window
closes with no graph bound, on `NodeToggleController.clear()`, or on
`dmi.transport.ring.deactivate()` (binding a graph after an empty window
renews it). When the last session ends, the original `torch.cuda.CUDAGraph`
is restored and the replay guard disarmed: graphs created afterwards get the
constructor arguments they ask for, and graphs created while patched replay
as plain graphs.

### Activation guards

Activation fails closed when no node is registered or no graph is bound, when
graphs registered different hook sets, when the registered graphs are not
exactly the bound graphs, or when a capture recorded a producer the toggle
cannot manage (a non-kernel tail dependency, unreadable capture state, or the
chunked producer). The verdict is cached on the registry version, which every
registry mutation bumps.

## Device-side toggling

Device-side toggling applies to producers paired with a per-hook device
control row; the static and prefix producers are always host-toggled with
`cudaGraphNodeSetEnabled`. With `begin_capture_window(..., device_toggle=True)`
(CUDA 12.4+ toolkit and driver), such producers are captured as
device-updatable kernel nodes, each paired with its control row; a planner
kernel captured
upstream of them calls `cudaGraphKernelNodeSetEnabled(node, row[0] != -1)` at
every replay. A `-1` row already makes a producer return without a
reservation, a task or a payload, so a disabled node is the same observation
as an early exit and the host never toggles these nodes. An exec holding
device-updatable nodes must be uploaded before its first launch. Because the
patched class keeps the template (`keep_graph=True` defers instantiation),
a capture in device mode that may hold producers -- excluded captures
included -- is instantiated and uploaded at capture end, and a later
re-instantiation is uploaded before it replays. Each exec is uploaded once;
excluded captures are never enrolled in the registry.
The mode is chosen per capture window (`clear_toggle_registry()` turns it
off) and stays off when the device, driver or build lacks device graph
updates.

`capture_window(engine, ...)` is the context-manager form of
`begin_capture_window` / `end_capture_window`; the window's reference to the
engine is also dropped by `NodeToggleController.clear()` and by
`dmi.transport.ring.deactivate()`.

## Strict eager fallback

`BackendAdapter(..., allow_eager_fallback=False)` refuses a step that would
need the eager safety net: an eager-only plan raises before anything is
reserved, and a step larger than the ring raises after `prepare_step` (which
reserves nothing for it) but before metadata is published or a producer is
armed. `capture_dispatch_state()` reports cumulative preparation counts.

## Behaviour when off

| Runs when off | Behaviour |
|---|---|
| `specs_for_step()` in `pre_push_all_metas` | returns `_active_specs` (the same object) |
| `plan_step` | iterates `active_specs` unless the gate is active |
| node recording in the producer ops | returns on a process-global flag |
| `HookPoint` eager gate | inside the `force_eager` branch, short-circuits on `toggle_gate_active` |
| `legacy_ring_room` | additionally saturates when the drain tail is ahead of the reservation head |
