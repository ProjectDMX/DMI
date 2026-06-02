"""Phase C (1b) end-to-end: the meta gate + device toggle through the REAL path.

Drives the actual RingTransport.pre_push_all_metas gate + set_active_hooks, the real
torch.ops.ring.producer (node registration), and the real RingEngine -> drain -> p2p
consumer, observed via a Python SubmitFn collector. Verifies:
  - disabled hooks produce NOTHING (no submission), and
  - enabled hooks arrive ALIGNED (delivered layer_no == the payload's marker value),
    i.e. host meta set == device enabled set end-to-end.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> python tests/ring/test_meta_gate_e2e.py
Requires the built backend + torch CUDA.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
# Use the SAME backend module ring_transport uses (monitoring._native_engine), or
# two copies of the .so load and TORCH_LIBRARY(ring) double-registers -> crash.
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig

HT = rt.HOOK_TYPE_RESID_PRE          # hidden-dim hook: shape [q_len, hidden_dim]
N = 4
QLEN, HID = 4, 8                     # tiny; payload = QLEN*HID*4 bytes (float32)
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    collected = []   # (layer_no, marker) per submitted slice
    def submit(model_id, shard_rank, req_id, act_name, layer_no, s, e, slice_):
        collected.append((int(layer_no), float(slice_.reshape(-1)[0].item())))

    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 64 * 1024 * 1024
    cfg.drain_poll_timeout_us = 200
    engine = ne.RingEngine(cfg, submit)        # Python SubmitFn collector
    engine.init(0)
    ne.ring_set_active_engine(engine)
    engine.enable_toggle_capture(True)
    engine.set_null_mode(True)                 # no real writes during warmup/capture
    engine.start()

    # Real RingTransport with a minimal-but-real model shape + specs.
    transport = RingTransport(engine)
    mcfg = ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                            head_dim=HID // 2, dtype=torch.float32)
    transport.set_model_cfg(mcfg)
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    transport.set_step_context(
        model_id="m", req_ids=["r"], token_ranges=[(0, QLEN)],
        dim0_offsets=[0], kv_offsets=[0], flattened=True)

    # src[j] filled with value j -> slice marker identifies the producer.
    src = [torch.full((QLEN, HID), float(j), device="cuda", dtype=torch.float32) for j in range(N)]

    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer(src[j], HT, j)   # real producer op (registers node)
    g.instantiate()
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    check(engine.toggle_node_count() == N, f"registered {N} nodes (got {engine.toggle_node_count()})")

    engine.set_null_mode(False)                # real writes now

    # ---- controlled step: enable a subset, gate metas, replay ----
    keep = [0, 2]
    transport.set_active_hooks([(HT, j) for j in keep])   # device toggle + activate gate
    transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)  # gated meta push
    g.replay()
    torch.cuda.synchronize()

    # wait for drain/p2p to deliver
    for _ in range(200):
        if len(collected) >= len(keep):
            break
        time.sleep(0.01)
    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()

    print(f"\ncollected: {sorted(collected)}  (expected layers {keep})")
    delivered_layers = sorted(l for l, _ in collected)
    check(delivered_layers == keep, f"only enabled hooks delivered: {delivered_layers} == {keep}")
    check(all(layer == marker for layer, marker in collected),
          "each slice aligned: delivered layer_no == payload marker (no desync)")
    check(all(l not in (1, 3) for l, _ in collected), "disabled hooks (1,3) produced nothing")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
