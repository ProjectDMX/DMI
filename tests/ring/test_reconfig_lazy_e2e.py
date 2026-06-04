"""Lazy per-graph toggle — engine-level correctness (Phase 4, sub-step 4.1).

Same two-graph reconfigure scenarios as test_reconfig_multigraph_e2e, but driven
through the LAZY path:
  - reconfigure via set_active_hooks_lazy() (bumps target_version, updates the
    meta gate, does NOT apply to the device),
  - ensure_graph_current(raw_graph) just BEFORE each replay (applies the deferred
    toggle to that one graph if stale),
  - record_replay_event(raw_graph) right AFTER each replay (event guard).

Asserts the same observable invariant: replaying graph G delivers exactly the
current enabled set, aligned, from G — regardless of how many reconfigures
happened since G last ran. This proves the lazy core: if ensure applies the wrong
graph / misses a version / leaves a graph stale, the stale graph fires its OLD
enabled set while the gate pushed the NEW set -> misalignment -> failure here.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> python tests/ring/test_reconfig_lazy_e2e.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig

HT = rt.HOOK_TYPE_RESID_PRE
N = 4
QLEN, HID = 4, 8
OFF_A, OFF_B = 0.0, 100.0
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    collected = []
    def submit(model_id, shard_rank, req_id, act_name, layer_no, s, e, slice_):
        collected.append((int(layer_no), float(slice_.reshape(-1)[0].item())))

    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 64 * 1024 * 1024
    cfg.drain_poll_timeout_us = 200
    engine = ne.RingEngine(cfg, submit)
    engine.init(0)
    ne.ring_set_active_engine(engine)
    engine.enable_toggle_capture(True)
    engine.set_null_mode(True)
    engine.start()

    transport = RingTransport(engine)
    mcfg = ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                            head_dim=HID // 2, dtype=torch.float32)
    transport.set_model_cfg(mcfg)
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    transport.set_step_context(
        model_id="m", req_ids=["r"], token_ranges=[(0, QLEN)],
        dim0_offsets=[0], kv_offsets=[0], flattened=True)

    srcA = [torch.full((QLEN, HID), OFF_A + j, device="cuda", dtype=torch.float32) for j in range(N)]
    srcB = [torch.full((QLEN, HID), OFF_B + j, device="cuda", dtype=torch.float32) for j in range(N)]

    gA = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(gA):
        for j in range(N):
            torch.ops.ring.producer(srcA[j], HT, j)
    gA.instantiate()
    gB = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(gB):
        for j in range(N):
            torch.ops.ring.producer(srcB[j], HT, j)
    gB.instantiate()
    engine.bind_graph_exec(gA.raw_cuda_graph(), gA.raw_cuda_graph_exec())
    engine.bind_graph_exec(gB.raw_cuda_graph(), gB.raw_cuda_graph_exec())
    engine.set_null_mode(False)

    graphs = {"A": (gA, OFF_A), "B": (gB, OFF_B)}
    cur = {"enabled": None}

    def reconfig(enabled):                         # LAZY: no device apply here
        cur["enabled"] = sorted(enabled)
        transport.set_active_hooks_lazy([(HT, j) for j in enabled])

    def replay(name):
        g, off = graphs[name]
        keep = cur["enabled"]
        collected.clear()
        transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        transport.ensure_graph_current(g.raw_cuda_graph())   # lazy: apply deferred toggle to G
        g.replay()
        transport.record_replay_event(g.raw_cuda_graph())    # lazy: event guard
        torch.cuda.synchronize()
        engine.flush_and_wait()
        for _ in range(200):
            if len(collected) >= len(keep):
                break
            time.sleep(0.01)
        time.sleep(0.03)
        delivered = sorted(l for l, _ in collected)
        aligned = all(m == l + off for l, m in collected)
        tag = f"replay {name} (enabled={keep})"
        check(delivered == keep, f"{tag}: delivered {delivered} == {keep}")
        check(aligned, f"{tag}: aligned + from graph {name} (marker==layer+{int(off)})")

    # Same scenarios as the eager multi-graph guardrail, via the lazy path.
    reconfig([0, 2]); replay("B"); replay("B"); replay("A")     # deferred apply: first-time A
    reconfig([1, 3]); replay("A"); replay("B")                  # one-version stale (B)
    reconfig([0, 1, 2, 3]); reconfig([2]); replay("A"); replay("B")  # two-version stale
    reconfig([]); replay("A"); replay("B")                      # empty

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
