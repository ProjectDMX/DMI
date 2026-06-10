# Runtime CUDA-Graph Node-Toggle — Implementation & Optimizations

> Branch: `feature/dmi_kernel_node_toggle`
>
> This document explains **how** DMI enables/disables individual monitoring
> (producer) kernels inside an *already-captured* CUDA graph at runtime — with no
> re-capture and no re-instantiation — and **which optimizations** make it cheap
> enough to be "free to have, pay only for what you enable."
>
> For the user-facing API and the interactive walkthrough see
> `docs/node_toggle_mechanism_api.html` / `docs/node_toggle_interactive.html`.
> For raw numbers see `docs/node_toggle_local_perf.md`.

---

## 1. The problem

DMI captures model internals by inserting `HookPoint` modules into the forward
graph. Each one calls `torch.ops.ring.producer(x, hook_type, hook_id)`, a custom
CUDA op that does a D2D copy of the tensor into the Ring² payload buffer and
publishes a task descriptor. Under vLLM the decode path runs as a **full CUDA
graph**, so every producer becomes a kernel *node* baked into the captured graph.

Originally hook selection was **static**: which producers exist is decided once,
*before* graph capture. To change what you monitor you had to tear down and
re-capture the graph — far too expensive to do online, per request, or
adaptively (e.g. "only turn on hidden-state capture for the last 4 layers when a
hallucination detector flags a request").

CUDA 12.2+ exposes `cudaGraphNodeSetEnabled(exec, node, 0|1)`: it flips a single
node in an *instantiated* graph exec on or off between replays, in microseconds,
with no re-instantiate. A **disabled kernel node is still traversed** by the
replay engine but launches as a grid-0 no-op (~0.18–0.34 µs/node residual on
H100/4090). That primitive is the foundation of node-toggle.

The hard part is not calling that function. It is keeping the **three lanes that
must agree** in lockstep, and doing the flip safely with respect to in-flight
GPU work.

---

## 2. The three lanes that must stay in lockstep

A producer firing is not a self-contained event. Each producer that fires must
have:

1. **A device kernel that actually runs** — the producer node enabled in the exec.
2. **A metadata entry** pushed to the C++ `TensorMetaFifo` *before* the forward,
   so the p2p thread can pop it and slice/route the payload the kernel wrote.
   FIFO pop order must match producer firing order.
3. **Reserved ring capacity** — `prepare_step(bytes, num_hooks)` advances the
   payload/task ring head by exactly what the producers will write. The drain
   thread advances the tail as it consumes *actual* producer task entries.

If these three diverge, you get silent corruption:

- Enabled device node but **no meta** → p2p has a payload with no descriptor → desync.
- Meta pushed but **node disabled** → descriptor with no payload → desync.
- Reserve for hooks that **don't fire** (or vice versa) → the ring
  `head − tail` gap drifts monotonically → eventually a uint64 underflow in the
  capacity check → a spurious "ring full" flush storm (measured ~+16% TPOT before
  the fix).

The design therefore funnels all three lanes through **one source of truth**.

### 2.1 The single source of truth: `effective_specs`

```
effective_specs = active_specs  ∩  enabled_hooks  ∩  registered_hooks
```

- `active_specs` — hooks installed for this model/preset.
- `enabled_hooks` — the runtime on-set (last value passed to `set_active_hooks`).
- `registered_hooks` — hooks that actually captured a producer node during
  capture (the `#14` guard: enabled-but-uncaptured hooks must NOT be gated on,
  else a meta is pushed with no kernel behind it).

The C++ engine owns `enabled_hooks` and `registered_hooks`. It is asked **once**
per reconfigure (not per step) for the resulting mask, and the Python transport
caches the surviving specs in `_effective_enabled_specs`
(`ring_transport.py:548`, property at `:624`). Every lane reads it:

- **capacity-reserve** → `vllm_integration.execute_model` sizes the step from
  `transport.effective_specs` (version-keyed cache, `:534`).
- **meta-push** → `pre_push_all_metas` iterates `effective_specs` (`:661`).
- **device-enable** → `apply_toggle` / `ensure_graph_current` flip exactly the
  registered nodes whose `(ht, layer)` is in `enabled_hooks`.

Because all three derive from the same set, recomputed atomically when the
enabled-set changes, they can never diverge by construction.

---

## 3. Lifecycle

### 3.1 Capture: record each producer's graph node

During warmup, vLLM captures the decode graph. We open a recording window and,
inside the producer op, snapshot the node CUDA just created:

```cpp
// ring_torch_op.cpp — after launching the producer kernel, if capturing:
if (g_toggle_capture) {
    cudaStreamCaptureStatus st; cudaGraph_t cap_graph; const cudaGraphNode_t* deps; size_t nd;
    if (cudaStreamGetCaptureInfo(stream, &st, &id, &cap_graph, &deps, &edges, &nd) == cudaSuccess
        && st == cudaStreamCaptureStatusActive && nd >= 1) {
        // the just-added producer kernel is the current tail dependency
        g_active_engine->register_capture_node(cap_graph, hook_type, hook_id, deps[nd - 1]);
    }
}
```

`cudaStreamGetCaptureInfo` returns the current capture graph plus its tail
dependency nodes; the producer kernel we just enqueued is `deps[nd-1]`. We store
`(graph → [{hook_type, layer, node}])` in `reg_nodes`, plus the global
`registered_hooks` set (`ring_engine_py.cu:169`).

**`keep_graph=True` (`vllm_integration.py:108`).** vLLM creates
`torch.cuda.CUDAGraph()` with `keep_graph=False`, which frees the captured
*template* graph right after `instantiate()` — the node handles we recorded would
dangle. We monkeypatch `torch.cuda.CUDAGraph` to force `keep_graph=True`
(host-memory only; no device or latency cost), as a subclass so
`isinstance(x, torch.cuda.CUDAGraph)` still holds. Applied only when node-toggle
is on.

### 3.2 Bind: associate each captured graph with its exec

After warmup, before serving (no replay in flight), we walk vLLM's
`CUDAGraphWrapper.concrete_cudagraph_entries` (one `CUDAGraph` per batch size),
unwrap nested wrappers, and bind each `(graph → exec)` into the registry
(`_dmx_bind_captured_graphs`, `:406`). We instantiate exactly once if needed
(`raw_cuda_graph_exec()` raises until instantiated — guard `#3`: never
re-instantiate an existing exec, that would destroy it). Then we **close the
capture window** (guard `#1`).

A typical Qwen3-8B run binds ~35 graphs (one per captured batch size).

### 3.3 Reconfigure: `set_active_hooks(enabled)`

The single entry point (`ring_transport.py:695`). At a step boundary with the
prior replay complete it:

1. **Guards** — refuse to arm the host gate if nothing is bound / nothing
   captured (`#1`, would filter metas while every default-enabled producer still
   fires → desync); refuse if captured graphs have **non-uniform** hook sets
   (`#4`, the meta gate keys on `(ht,layer)` globally, so a hook present in one
   graph but not the one replayed this step would push an orphan meta). vLLM
   captures all hooks in every graph, so uniform holds.
2. `eng.set_enabled_hooks(pairs)` — updates `enabled_hooks`, bumps
   `target_version`.
3. `eng.apply_toggle()` — flips the device nodes (eager path).
4. Activates the host meta gate and recomputes `_effective_enabled_specs` via
   **one** batched `effective_enabled_mask` call, bumps `_enabled_version`
   (invalidates the worker's capacity cache).

### 3.4 Replay

Each step: `prepare_step` reserves capacity for `effective_specs`,
`pre_push_all_metas` pushes metas for `effective_specs`, the graph replays —
enabled producers run, disabled ones are grid-0 no-ops — the drain/p2p threads
route the payloads exactly as in always-on DMI.

---

## 4. Optimizations

### Opt 1 — Per-node diff (`last_enabled`)

`apply_toggle` only calls `cudaGraphNodeSetEnabled` for nodes whose desired state
**changed** since the last apply (`ring_engine_py.cu:198`). Each `RegEntry`
tracks `last_enabled` (DMI is the sole toggler of its own nodes, so this mirrors
the exec's real state). An adaptive flip of 4 hooks costs O(4) SetEnabled calls,
not O(all nodes × all graphs).

### Opt 2 — Batched pybind crossing (`effective_enabled_mask`)

**The reconfigure bottleneck was the Python↔C++ boundary, not the C++ walk.**
The original code called `is_hook_enabled(ht, layer)` once per spec to recompute
the effective set — N pybind round-trips, each taking a lock. We replaced that
with a single `effective_enabled_mask(query)` call that takes one lock and
returns the whole mask (`ring_engine_py.cu:299`, used at `ring_transport.py:744`).
C++ stays the source of truth; the host just stops crossing the boundary N times.

> An earlier attempt ("eager-delta": making `apply_toggle` itself diff harder)
> gave **zero** benefit because it optimized the wrong layer — the C++ walk was
> already cheap. It was reverted. Batching the pybind crossings was the real win.

### Opt 3 — Lazy per-graph apply (Phase 4)

Eager `apply_toggle` touches **every bound graph** (~35) on every reconfigure,
even graphs that may not be replayed again soon. Lazy mode defers the device flip
to the moment a graph is *actually about to replay*:

- `set_enabled_hooks` bumps a global `target_version`; each graph tracks its own
  `applied_version`.
- `ensure_graph_current(graph)` (`ring_engine_py.cu:253`) runs in the patched
  `CUDAGraph.replay()` just before replay. If `applied_version == target_version`
  it's a fast no-op (the common path); otherwise it applies the diff to *that one
  graph* and stamps it current.
- A reconfigure then costs **O(changed × graphs-actually-used)** instead of
  **O(changed × all-graphs)**.

Measured: pure reconfigure host cost dropped **6.1×** (≈80 µs → ≈13 µs).
Opt-in via `dmx_lazy_toggle` (eager is default; lazy only matters for high-
frequency per-step reconfigure, not static configs).

### Opt 4 — Reconfigure caches (guard verdict + preset memo)

Per-step reconfigure is the access pattern of the planned consumers (graduated
hallucination monitoring, per-layer profiler sweeps, debugging), and at
production scale the dominant reconfigure cost turned out to be **re-validating
an immutable registry**: the `set_active_hooks[_lazy]` guards (uniformity /
completeness, O(graphs × hooks)) scale with bound graphs — measured host-only
lazy cost 9.9 / 15.5 / 36.6 µs at 1 / 8 / 35 graphs
(`docs/node_toggle_probe/probe_reconfig_scaling.py`).

The registry only mutates at capture / bind / clear time, so the engine keeps a
`registry_version` bumped by **every** mutation, and the transport memoizes on
it:

- **Guard verdict**: a guard PASS is cached per registry version; the next
  reconfigure on an unchanged registry skips the checks entirely. Failures are
  never cached, and any mutation (including `note_capture_anomaly`) forces full
  re-validation — the fail-loud semantics are unchanged, gated by
  `test_toggle_reconfig_cache_e2e.py`.
- **Preset memo**: `(registry version, enabled set, active_specs identity) →
  effective_specs`, so cycling presets skips the batched-mask recompute.

Measured after: lazy reconfigure **4–6 µs flat, independent of graph count**
(35 graphs: 36.6 → 4.4 µs); per-step full lazy price (call + ensure + event)
17.8 → 10.5 µs on the ~4 ms probe step.

> A fifth idea — **exec ping-pong** (two execs per template; apply to the idle
> one to hide the ensure event-wait) — was probed and rejected: the event-wait
> bubble is only ~6 µs, while a second exec costs ~7 KiB/node of device memory
> (≈0.25–0.5 GiB across 35 production graphs).

### Opt 3a — Event guard (safety for lazy)

Mutating an exec while a prior replay of it is still running is **undefined
behavior** — and the host can run ahead of the GPU. So after each lazy replay we
record a timing-disabled CUDA event on the stream (`record_replay_event`,
`:285`); the next `ensure_graph_current` for that graph calls
`cudaEventSynchronize` on it before mutating (`:270`). This serializes
"mutate exec" strictly after "prior replay of that exec finished," only when a
mutation is actually pending.

(The eager `null_mode` device-flag path has a *different* discipline — full
`cudaDeviceSynchronize` around a `cudaMemcpyToSymbol` on the legacy default
stream — documented at `set_null_mode`, `ring_engine_py.cu:134`.)

### Opt 4 — `effective_specs` single-source (correctness *and* speed)

Folding active∩enabled∩registered into one cached list (§2.1) removed a per-step
`is_hook_enabled` loop from the meta-push hot path **and** closed the reserve-
invariant bug. It is the optimization that makes "toggle off ≈ baseline"
possible: when nothing is enabled, `effective_specs` is empty, so reserve is 0
bytes / 0 hooks and meta-push pushes nothing — the host floor collapses to almost
nothing and only the disabled graph nodes remain (traversed, not launched).

### Opt 5 — Reserve-invariant fix + two-sided clamp

`prepare_step` (`ring_engine_py.cu:385`) must never under/over-reserve relative to
what producers write. Two robustness fixes:

- **`reserve()` keyed on `effective_specs`** — not all `active_specs`. Reserving
  for disabled hooks was the original integration bug (monotonic gap drift →
  underflow).
- **Two-sided saturating clamp** on `used = head − tail`:
  - `head < tail`: a small benign skew exists because producers that fire during
    *capture* get drained (tail advances) with no matching `reserve()` (which
    only runs per real step). Naive `head − tail` underflows uint64 → avail=0 →
    spurious flush every step. Clamp `used = 0` (avail = full cap).
  - `head − tail > cap`: over-reserve drift; fail safe with avail = 0 rather than
    underflowing to a huge bogus "available."

This removed the flush storm and made the capacity check robust to warmup skew.

### Opt 6 — Low-volume drain latency

Unrelated to toggle topology but on the same path: the drain thread's flush
thresholds were tuned so low-volume steps (e.g. a few enabled hooks) don't sit
waiting on a byte/entry threshold that never trips. See the drain-flush config in
`init_device` and `docs/node_toggle_local_perf.md` "Low-volume drain latency."

---

## 5. Correctness guards (numbered, as in code)

| # | Guard | Where | Prevents |
|---|-------|-------|----------|
| #1 | Refuse to arm host gate with no exec bound / nothing captured | `set_active_hooks` `:715` + close window after bind `:375` | host filters metas while default-on producers still fire → desync |
| #2 | `clear_toggle` resets enabled-set + deactivates gate together | `clear_toggle_registry` `:322`, `clear_toggle` `:788` | stale gate skipping every meta after a clear |
| #3 | Never re-instantiate an existing exec | `_dmx_bind_captured_graphs` `:431` | destroying a live exec |
| #4 | Require uniform hook sets across captured graphs | `toggle_registry_uniform` `:225`, checked `:724` | orphan meta when the replayed graph lacks a globally-gated hook |
| #14 | Gate on enabled **AND** registered | `is_hook_enabled` `:242`, `effective_enabled_mask` `:299` | meta pushed for an enabled-but-uncaptured hook (no kernel behind it) |

A debug accessor `get_stats()` (`:118`) exposes live `payload/task head & tail`
so a long-running probe can confirm `reserve == actually-written` across configs.

---

## 6. Measured results (Qwen3-8B, H100)

Per-step decode overhead, batch=1, `cudagraph_mode=FULL`, 100 steps × 30 reps,
median. Baseline per-step = 6.5220 ms.

| Config | per-step (ms) | Δ µs/step | overhead |
|--------|--------------:|----------:|---------:|
| baseline (no DMI)         | 6.5220 | —      | —      |
| **toggle off** (0/36)     | 6.5282 | +6.3   | +0.10% |
| DMI null mode             | 6.5455 | +23.5  | +0.36% |
| **toggle partial** (4/36) | 6.5490 | +27.0  | +0.41% |
| DMI on (full transport)   | 6.6411 | +119.2 | +1.83% |
| **toggle full** (36/36)   | 6.6543 | +132.4 | +2.03% |

1. **toggle off ≈ baseline (+0.10%)** — armed-but-idle is essentially free.
2. **toggle off (+0.10%) < null_mode (+0.36%)** — disabling removes the kernel
   *launch*, not just the work; ~3.7× cheaper idle than the old `null_mode`.
3. **Scales with #enabled** — 0/4/36 → +0.10/+0.41/+2.03% (~3.5 µs/enabled hook;
   ~0.18 µs residual per disabled node).
4. **toggle full ≈ DMI on** (+2.03% vs +1.83%) — toggle is ~free to have
   (~13 µs machinery over plain always-on) and recovers **~95%** of the transport
   overhead when idle.

The host costs overlap GPU replay, so these are not pure single-stream TPOT
deltas — the host-CPU savings matter most for throughput under concurrency. See
`node_toggle_local_perf.md` for the per-block host-path decomposition (RTX 4090 /
Qwen3-0.6B).

---

## 7. API summary

Three engine calls span capture → bind → toggle; the transport wraps them.

```python
# capture (warmup): record producer nodes
engine.enable_toggle_capture(True)        # producer op records nodes via GetCaptureInfo
...super().compile_or_warm_up_model()...   # vLLM captures graphs (keep_graph forced True)
# bind (post-warmup, no replay in flight)
engine.bind_graph_exec(raw_graph, raw_exec)   # per captured graph
engine.enable_toggle_capture(False)           # close the window (#1)
# reconfigure (any step boundary)
transport.set_active_hooks([(ht, layer), ...])      # eager  (set_enabled_hooks + apply_toggle)
transport.set_active_hooks_lazy([(ht, layer), ...]) # lazy   (defer device flip to replay)
transport.clear_toggle()                            # paired teardown
```

vLLM config (`--additional-config`):

```json
{
  "dmx_node_toggle": true,
  "dmx_enabled_hooks": "0:32,0:33,0:34,0:35",   // hook_type:layer; "0:99" => all off
  "dmx_lazy_toggle": false                       // true => Phase 4 lazy apply
}
```

Requires `cudagraph_mode=FULL` (decode) so producer nodes are captured + bindable;
`DMXGPUWorker` raises if a subset is requested but nothing bound.

---

## 8. File map

| Concern | File / symbol |
|---------|---------------|
| Capture-time node recording | `csrc/ring/ring_torch_op.cpp` — `g_toggle_capture`, `cudaStreamGetCaptureInfo`, `register_capture_node` |
| Toggle registry + apply + lazy | `csrc/ring/ring_engine_py.cu` — `Impl` registry, `apply_toggle`, `set_enabled_hooks`, `ensure_graph_current`, `record_replay_event`, `effective_enabled_mask`, `prepare_step` clamp, `get_stats` |
| Bindings / decls | `csrc/bindings.cpp`, `csrc/ring/ring_engine_py.h` |
| Host single-source + gate | `ring_transport.py` — `effective_specs`, `set_active_hooks`, `set_active_hooks_lazy`, `pre_push_all_metas`, `clear_toggle` |
| vLLM wiring | `vllm_integration.py` — `_patch_cudagraph_keep_graph`, `_KeepGraphCUDAGraph.replay`, `compile_or_warm_up_model`, `_dmx_bind_captured_graphs`, `execute_model` capacity loop |
| Tests | `tests/ring/test_rings.cu`, `tests/test_reconfig_sequence_e2e.py`, `test_reconfig_multigraph_e2e.py`, `test_reconfig_lazy_e2e.py` |
| Experiments | `experiments/online_serving/script/sbatch/adapt_toggle_{qwen4b,llama8b,qwen14b}.sbatch` |
| Perf / showcase | `docs/node_toggle_local_perf.md`, `docs/node_toggle_mechanism_api.html`, `docs/node_toggle_interactive.html` |
</content>
</invoke>
