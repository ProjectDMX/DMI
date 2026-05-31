# Node-Toggle — Phase 2 Results

> Phase 2 = the runtime "toggle list" API. Status: **C++ controller done + verified
> end-to-end.** Python binding (Phase 2b) deferred — needs the backend build.
> Invariants: `node_toggle_design_notes.md` §1. Phase 1: `node_toggle_phase1_results.md`.

## What was built

`monitoring/csrc/ring/node_toggle.h` — `ring::NodeToggleController`, header-only, no
ATen/engine deps (usable from the native backend and from standalone tests).

It keeps the lockstep invariant on a **single source of truth**: one enabled-set drives
BOTH
- which producer nodes fire — `apply(exec)` → `cudaGraphNodeSetEnabled`, and
- which hook metas the host pushes — `enabled_in_capture_order()`.

This is NOT unconditional "lockstep by construction" — it holds only under the caller
contract below. Two conditions matter (both reviewed and hardened):

1. **Single snapshot, single thread.** `apply()` and `enabled_in_capture_order()` are
   two reads of mutable state; a mutation between them (or a second thread) re-diverges
   the lanes. Use **`apply_and_get_enabled(exec, out)`** — applies and returns the
   meta-push list in one call, so no mutation can interleave. The controller has no
   internal locking: single-threaded, step-boundary ownership only.
2. **Register in producer PUBLISH order, not hook identity.** p2p matches payloads to
   metas POSITIONALLY, so the meta order (= `register_node` order) must equal the order
   producers fire. Same enabled *set* in a different *order* still desyncs. The
   controller cannot verify capture order (caller contract); `validate()` checks what it
   can — null/duplicate nodes, duplicate hook ids.

API:
```cpp
ctrl.register_node({hook_type, layer_no}, node);   // capture time, IN PUBLISH ORDER
ctrl.validate(&reason);                             // null/dup node, dup hook id
ctrl.set_enabled_if(pred);   // or set_all(bool)    // between steps
ctrl.apply_and_get_enabled(exec, push_hooks);       // PREFERRED: apply + meta list, one snapshot
//   or, separately: ctrl.apply(exec); ctrl.enabled_in_capture_order();
```

Lifecycle contract (caller-enforced, per design-notes §1): register at capture time in
publish order; between steps only, with the prior replay complete,
`set_*` → `apply_and_get_enabled` → push metas for the returned list → launch.

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
