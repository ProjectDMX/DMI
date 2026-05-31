# Node-Toggle — Phase 2 Results

> Phase 2 = the runtime "toggle list" API. Status: **C++ controller done + verified
> end-to-end.** Python binding (Phase 2b) deferred — needs the backend build.
> Invariants: `node_toggle_design_notes.md` §1. Phase 1: `node_toggle_phase1_results.md`.

## What was built

`monitoring/csrc/ring/node_toggle.h` — `ring::NodeToggleController`, header-only, no
ATen/engine deps (usable from the native backend and from standalone tests).

It makes the lockstep invariant **structural** rather than hand-coordinated: a single
enabled-set drives BOTH
- which producer nodes fire — `apply(exec)` → `cudaGraphNodeSetEnabled`, and
- which hook metas the host pushes — `enabled_in_capture_order()`.

Because both read the same source, you cannot enable a node without its meta being in
the push list, or vice versa — the exact divergence that desyncs `p2p_thread`.

API:
```cpp
ctrl.register_node({hook_type, layer_no}, node);   // capture time, in capture order
ctrl.set_enabled_if(pred);   // or set_all(bool)    // between steps
ctrl.apply(exec);                                    // -> cudaGraphNodeSetEnabled, checked
auto hooks = ctrl.enabled_in_capture_order();        // -> the meta-push list (lockstep)
```

Lifecycle contract (enforced by the caller, per design-notes §1): register at capture
time; between steps only, with the prior replay complete, `set_*` → `apply` → push metas
for `enabled_in_capture_order()` → launch.

## Verification

`tests/ring/test_node_toggle_e2e.cu` was refactored to drive everything through the
controller (replacing the ad-hoc node map + meta-id logic). Same two scenarios, same
real consumer pipeline on a non-blocking stream:

| scenario | source of meta-push list | desync | mismatch |
|---|---|---|---|
| (1) lockstep | `ctrl.enabled_in_capture_order()` (single source) | **0** | **0** |
| (2) violation | bypass controller, push for ALL hooks | **22** | **22** |

4/4 assertions pass. Scenario 1 shows the controller yields correct lockstep by
construction; scenario 2 shows what the API prevents (the bug is only possible by
*bypassing* it).

## What Phase 2 does NOT yet cover (→ Phase 2b / 3)

- **Python binding** mirroring `set_null_mode`'s chain (`ring_engine_py.cu/.h` →
  `bindings.cpp`): a `set_enabled_hooks([...])`-style runtime knob exposed to Python,
  plus wiring the controller into `RingEnginePy`. Mechanical but needs the backend
  build (ClickHouse client + `make -C monitoring`) to compile-test — deferred to 2b.
- **Populating the registry from a real captured graph.** Here nodes come from a
  DMI-owned sandbox capture; in vLLM/HF the controller must be fed node handles from
  the framework's captured exec graph — the Phase 3 blocker.
- **`pre_push_all_metas` integration**: `ring_transport.py` must call
  `enabled_in_capture_order()` (or equivalent) to gate the real shape-computing meta
  path — Phase 1b.

## Reproduce

```bash
make -C tests/ring test_node_toggle_e2e
```
