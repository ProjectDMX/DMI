"""Lazy apply failure-path gates (fault-injected).

Uses the engine's test-only fault injection (_test_force_apply_error) to drive
the device-apply error path deterministically -- cudaGraphNodeSetEnabled itself
can't be forced to fail from Python. Verifies:

  - apply failure -> applied_version NOT marked current (graph stays stale, so
    a later ensure retries); success -> marked current.
  - with the replay-time guard armed, a failed ensure_graph_current at replay
    RAISES FATAL (and does NOT replay); a clean ensure replays normally.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_lazy_failpath_e2e.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig
import integration.vllm_adapter as va

HT = rt.HOOK_TYPE_RESID_PRE
N = 3
QLEN, HID = 4, 8
INJ = 999  # arbitrary nonzero "CUDA error" code
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    va._patch_cudagraph_keep_graph()
    va._DMX_TOGGLE_REPLAY_GUARD = True

    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 32 * 1024 * 1024
    eng = ne.RingEngine(cfg, None)
    eng.init(0)
    eng.set_null_mode(True)
    eng.start()
    transport = RingTransport(eng)
    rt.activate(transport)
    transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                                             head_dim=HID // 2, dtype=torch.float32))
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    payload = eng.payload_tensor()

    eng.enable_toggle_capture(True)
    src = [torch.full((QLEN, HID), float(j), device="cuda", dtype=torch.float32) for j in range(N)]
    gA = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(gA):
        for j in range(N):
            torch.ops.ring.producer(payload, src[j], HT, j)
    gA.instantiate()
    eng.enable_toggle_capture(False)
    eng.bind_graph_exec(gA.raw_cuda_graph(), gA.raw_cuda_graph_exec())
    raw = gA.raw_cuda_graph()

    # ---- version-on-success-only ----
    print("[applied_version only on full success]")
    transport.set_active_hooks_lazy([(HT, j) for j in range(N)])   # target bumped; A stale
    check(not eng._test_applied_current(raw), "A stale right after lazy reconfigure (deferred)")

    eng._test_force_apply_error(INJ)
    err = eng.ensure_graph_current(raw)
    check(err == INJ, f"ensure_graph_current returns injected error ({err})")
    check(not eng._test_applied_current(raw),
          "apply FAILED -> applied_version NOT marked current (stays stale, retryable)")

    eng._test_force_apply_error(0)
    err = eng.ensure_graph_current(raw)
    check(err == 0, "ensure_graph_current succeeds after clearing injection")
    check(eng._test_applied_current(raw), "apply SUCCEEDED -> applied_version marked current")

    # ---- replay raises FATAL on apply error ----
    print("[replay FATAL on lazy apply error]")
    transport.set_active_hooks_lazy([(HT, 0)])      # bump target -> A stale again
    eng._test_force_apply_error(INJ)
    raised = False
    try:
        gA.replay(); torch.cuda.synchronize()
    except RuntimeError as e:
        raised = "FATAL" in str(e) and "ensure_graph_current" in str(e)
    check(raised, "failed lazy apply at replay RAISES fatal (no silent desync)")

    eng._test_force_apply_error(0)
    ok = True
    try:
        gA.replay(); torch.cuda.synchronize()
    except RuntimeError:
        ok = False
    check(ok, "clean lazy apply replays without raising (recovery)")

    rt.deactivate()
    va._DMX_TOGGLE_REPLAY_GUARD = False
    eng.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
