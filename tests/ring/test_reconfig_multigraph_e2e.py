"""Multi-graph reconfigure guardrail (prerequisite for lazy per-graph toggle).

Binds TWO captured graphs (A, B) that register the same hook set, then drives a
sequence of reconfigures and *selective* replays. After every replay it asserts
the OBSERVABLE invariant:

    replaying graph G delivers exactly the CURRENT enabled set, aligned, and the
    data came from G (per-graph marker), regardless of how many reconfigures
    happened since G was last replayed.

This is implementation-agnostic: it passes under today's eager apply (every graph
updated on every reconfigure) AND is the safety net for lazy per-graph apply
(reconfigure only bumps a version; each graph is applied just before it replays).
The lazy bugs it catches: a stale graph firing its OLD enabled set while the meta
gate pushed the NEW set (-> desync), applying to the wrong graph, or missing a
multi-version-stale catch-up. Covered transitions: deferred apply (replay B then
first-time A), one-version stale (alternate A/B across a reconfigure), two-version
stale (two reconfigures with no replay between), and empty.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> python tests/ring/test_reconfig_multigraph_e2e.py
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
OFF_A, OFF_B = 0.0, 100.0     # per-graph marker offset: A src[j]=j, B src[j]=100+j
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    collected = []   # (layer_no, marker)
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

    # Two graphs, same hook set (so toggle_registry_uniform passes), distinct
    # markers so we can tell which graph delivered.
    payload = engine.payload_tensor()  # Tensor(a!) alias for the 4-arg producer op
    srcA = [torch.full((QLEN, HID), OFF_A + j, device="cuda", dtype=torch.float32) for j in range(N)]
    srcB = [torch.full((QLEN, HID), OFF_B + j, device="cuda", dtype=torch.float32) for j in range(N)]

    gA = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(gA):
        for j in range(N):
            torch.ops.ring.producer(payload, srcA[j], HT, j)
    gA.instantiate()
    gB = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(gB):
        for j in range(N):
            torch.ops.ring.producer(payload, srcB[j], HT, j)
    gB.instantiate()

    engine.bind_graph_exec(gA.raw_cuda_graph(), gA.raw_cuda_graph_exec())
    engine.bind_graph_exec(gB.raw_cuda_graph(), gB.raw_cuda_graph_exec())
    check(engine.bound_graph_count() == 2, f"bound 2 graphs (got {engine.bound_graph_count()})")
    check(engine.toggle_registry_uniform(), "registries uniform across A,B")

    engine.set_null_mode(False)

    graphs = {"A": (gA, OFF_A), "B": (gB, OFF_B)}
    cur = {"enabled": None}

    def reconfig(enabled):
        cur["enabled"] = sorted(enabled)
        transport.set_active_hooks([(HT, j) for j in enabled])

    def replay(name):
        g, off = graphs[name]
        keep = cur["enabled"]
        collected.clear()
        transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        g.replay()
        torch.cuda.synchronize()
        engine.flush_and_wait()
        for _ in range(200):
            if len(collected) >= len(keep):
                break
            time.sleep(0.01)
        time.sleep(0.03)
        delivered = sorted(l for l, _ in collected)
        aligned = all(m == l + off for l, m in collected)          # right enabled set + right GRAPH
        tag = f"replay {name} (enabled={keep})"
        check(delivered == keep, f"{tag}: delivered {delivered} == {keep}")
        check(aligned, f"{tag}: aligned + from graph {name} (marker==layer+{int(off)})")

    # --- deferred apply: reconfigure, replay B twice, THEN first-time A ---
    reconfig([0, 2]); replay("B"); replay("B"); replay("A")
    # --- one-version stale: reconfigure, replay A, then B (B was at prev set) ---
    reconfig([1, 3]); replay("A"); replay("B")
    # --- two-version stale: two reconfigures, no replay between, then A then B ---
    reconfig([0, 1, 2, 3]); reconfig([2]); replay("A"); replay("B")
    # --- empty ---
    reconfig([]); replay("A"); replay("B")

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
