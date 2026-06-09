"""Graph-guard failure paths.

Registry completeness: set_active_hooks must RAISE when the set of graphs with
   recorded producer nodes != the set of bound execs (partial/mismatched bind).

Replay-time guard: with a toggle gate active, replaying a graph that is NOT
   registered+bound (e.g. one vLLM captured at RUNTIME after DMI closed its
   capture window) must RAISE -- such a graph's producers run default-ON while
   the meta gate filters to the enabled subset, which would desync the ring.
   A registered+bound graph must replay fine.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_graph_guard_e2e.py
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
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def _new_engine():
    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 32 * 1024 * 1024
    eng = ne.RingEngine(cfg, None)
    eng.init(0)
    return eng


def _capture(engine, payload, record):
    """Capture a graph of N producer ops. record=True -> toggle nodes recorded."""
    engine.enable_toggle_capture(record)
    src = [torch.full((QLEN, HID), float(j), device="cuda", dtype=torch.float32) for j in range(N)]
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer(payload, src[j], HT, j)
    g.instantiate()
    engine.enable_toggle_capture(False)
    return g, src   # keep src alive


def test_completeness_guard():
    print("[registry completeness]")
    eng = _new_engine()
    ne.ring_set_active_engine(eng)
    eng.set_null_mode(True)
    payload = eng.payload_tensor()
    transport = RingTransport(eng)
    transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                                             head_dim=HID // 2, dtype=torch.float32))
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]

    gA, _sa = _capture(eng, payload, record=True)    # reg_nodes = {A}
    gB, _sb = _capture(eng, payload, record=False)   # B has NO recorded nodes
    eng.bind_graph_exec(gA.raw_cuda_graph(), gA.raw_cuda_graph_exec())
    eng.bind_graph_exec(gB.raw_cuda_graph(), gB.raw_cuda_graph_exec())  # reg_exec = {A, B}

    check(not eng.toggle_registry_complete(),
          "toggle_registry_complete() False when reg_nodes={A} != reg_exec={A,B}")
    raised = False
    try:
        transport.set_active_hooks([(HT, 0)])
    except RuntimeError as e:
        raised = "incomplete" in str(e)
    check(raised, "set_active_hooks RAISES on incomplete registry (partial/extra bind)")

    ne.ring_clear_active_engine()
    eng.stop()


def test_replay_guard():
    print("[replay-time unknown-graph guard]")
    va._patch_cudagraph_keep_graph()
    va._DMX_TOGGLE_REPLAY_GUARD = True

    eng = _new_engine()
    eng.set_null_mode(True)
    transport = RingTransport(eng)
    rt.activate(transport)                            # get_active() -> transport; sets engine ptr
    transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                                             head_dim=HID // 2, dtype=torch.float32))
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    payload = eng.payload_tensor()

    gA, _sa = _capture(eng, payload, record=True)     # registered
    eng.bind_graph_exec(gA.raw_cuda_graph(), gA.raw_cuda_graph_exec())  # bound -> ready
    transport.set_active_hooks([(HT, j) for j in range(N)])             # eager gate active

    check(transport.is_graph_ready(gA.raw_cuda_graph()), "graph A is registered+bound (ready)")

    # A is ready -> replay must NOT raise (the guard lets it through).
    ok = True
    try:
        gA.replay(); torch.cuda.synchronize()
    except RuntimeError:
        ok = False
    check(ok, "ready graph A replays through the guard without raising")

    # B is captured WITHOUT recording (simulates a runtime-new graph) -> unknown.
    gB, _sb = _capture(eng, payload, record=False)
    check(not transport.is_graph_ready(gB.raw_cuda_graph()), "graph B is NOT ready (unregistered)")
    raised = False
    try:
        gB.replay(); torch.cuda.synchronize()
    except RuntimeError as e:
        raised = "FATAL" in str(e) and "NOT registered" in str(e)
    check(raised, "unknown graph B RAISES fatal at replay (would-be desync blocked)")

    rt.deactivate()
    va._DMX_TOGGLE_REPLAY_GUARD = False
    eng.stop()


def test_capture_anomaly_guard():
    """A capture-time non-kernel tail node (recorded as an anomaly) must make
    set_active_hooks fail loud (fail-closed) -- registering a wrong node would
    toggle the wrong graph node. Simulated via the test-only note_capture_anomaly."""
    print("[capture-anomaly fail-closed]")
    eng = _new_engine()
    ne.ring_set_active_engine(eng)
    eng.set_null_mode(True)
    payload = eng.payload_tensor()
    transport = RingTransport(eng)
    transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                                             head_dim=HID // 2, dtype=torch.float32))
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    gA, _sa = _capture(eng, payload, record=True)
    eng.bind_graph_exec(gA.raw_cuda_graph(), gA.raw_cuda_graph_exec())
    check(eng.capture_anomaly_count() == 0, "real producer capture -> 0 anomalies (validation ran clean)")

    eng.note_capture_anomaly()   # simulate a non-kernel tail node recorded during capture
    raised = False
    try:
        transport.set_active_hooks([(HT, 0)])
    except RuntimeError as e:
        raised = "non-kernel" in str(e)
    check(raised, "set_active_hooks RAISES when a capture anomaly was recorded (fail-closed)")

    ne.ring_clear_active_engine()
    eng.stop()


def main():
    test_completeness_guard()
    test_replay_guard()
    test_capture_anomaly_guard()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
