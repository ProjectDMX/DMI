"""Node-toggle binding through the REAL backend + producer op.

Drives torch.ops.ring.producer inside a torch CUDA-graph capture (keep_graph=True)
with the engine active and toggle-capture on, so the producer op records its kernel
node via cudaStreamGetCaptureInfo. Then exercises the Python binding:
  - toggle_node_count(): capture-time registration worked
  - is_hook_enabled():   enabled-set single-source gate (for pre_push_all_metas)
  - effective_enabled_mask(): batched gate matches is_hook_enabled
  - apply_toggle():      cudaGraphNodeSetEnabled on torch's raw_cuda_graph_exec using
                         the capture-recorded handles SUCCEEDS (err 0) -> the
                         toggle mechanism works through the real DMI backend.

Note: the producer op signature is
``producer(Tensor(a!) ring_payload, Tensor x, int hook_type, int hook_id)``.
This test drives the *basic* producer op only (producer_prefix has its own gate,
test_toggle_prefix_e2e).

Run:  CUDA_MODULE_LOADING=EAGER python tests/ring/test_toggle_binding.py
Requires the built backend (monitoring_native_backend*.so at repo root) + torch CUDA.
"""
import os
import sys

# Make the repo-root monitoring_native_backend*.so importable regardless of cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import monitoring_native_backend as ne

N = 4                       # producer "hooks"
HT = 0                      # hook_type (RESID_PRE)
ELEMS = 4096
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 64 * 1024 * 1024
    cfg.task_ring_entries = 4096
    engine = ne.RingEngine(cfg, None)          # no host engine; we don't observe submits here
    engine.init(0)
    ne.ring_set_active_engine(engine)
    engine.enable_toggle_capture(True)

    # The Tensor(a!) mutation-alias arg every producer op now takes.
    payload = engine.payload_tensor()
    src = [torch.full((ELEMS,), float(j), device="cuda") for j in range(N)]

    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            # hook_id == layer_no (see ring_transport._hook_id_from_name)
            torch.ops.ring.producer(payload, src[j], HT, j)
    g.instantiate()

    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())

    print("[registration]")
    check(engine.toggle_node_count() == N,
          f"toggle_node_count == {N} (got {engine.toggle_node_count()})")
    check(engine.bound_graph_count() == 1,
          f"bound_graph_count == 1 (got {engine.bound_graph_count()})")
    check(engine.toggle_registry_uniform(), "single graph -> registry uniform")

    print("[enabled-set gate is_hook_enabled]")
    # Before any set_enabled_hooks, toggle inactive -> all enabled.
    check(all(engine.is_hook_enabled(HT, j) for j in range(N)),
          "all hooks enabled before set_enabled_hooks (toggle inactive)")
    keep = [0, 2]
    engine.set_enabled_hooks([(HT, j) for j in keep])
    gate_ok = all(engine.is_hook_enabled(HT, j) == (j in keep) for j in range(N))
    check(gate_ok, f"is_hook_enabled matches enabled set {keep}")

    print("[effective_enabled_mask (batched gate)]")
    mask = engine.effective_enabled_mask([(HT, j) for j in range(N)])
    check(mask == [1 if j in keep else 0 for j in range(N)],
          f"effective_enabled_mask matches is_hook_enabled (got {mask})")

    # #14 desync guard: a hook that was never captured must gate OFF even if
    # someone puts it in the enabled set (else meta pushed with no payload).
    engine.set_enabled_hooks([(HT, 0), (HT, 99)])   # layer 99 was not captured
    check(engine.is_hook_enabled(HT, 0) and not engine.is_hook_enabled(HT, 99),
          "enabled-but-uncaptured hook (HT,99) gated off (#14 guard)")

    print("[apply_toggle on torch exec with capture-recorded handles]")
    err = engine.apply_toggle()
    check(err == 0, f"apply_toggle() returns cudaSuccess (got {err})")

    # re-enable all and re-apply
    engine.set_enabled_hooks([(HT, j) for j in range(N)])
    check(engine.apply_toggle() == 0, "apply_toggle() ok after re-enable")

    print("[clear_toggle_registry resets state]")
    engine.clear_toggle_registry()
    check(engine.toggle_node_count() == 0 and engine.bound_graph_count() == 0,
          "registry empty after clear")
    check(engine.is_hook_enabled(HT, 0),
          "toggle inactive after clear -> all-on gate")

    ne.ring_clear_active_engine()
    engine.stop()

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
