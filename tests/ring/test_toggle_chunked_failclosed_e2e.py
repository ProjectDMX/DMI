"""Chunked producer fail-closed under node-toggle.

The chunked producer is not toggle-managed (the prefix-path Option A covers basic
+ prefix only). If one fires during capture, its node is NOT recorded and a
capture anomaly is flagged, so set_active_hooks / set_active_hooks_lazy must
REFUSE to activate (fail-closed) -- otherwise it would be a fired-but-unregistered
producer (silent desync). This gate captures a MIXED graph (basic producers
registered + one chunked producer) and asserts:
  - capture_anomaly_count() > 0, chunked node NOT registered,
  - set_active_hooks raises, message names 'chunked' + the remedy,
  - set_active_hooks_lazy raises too.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_chunked_failclosed_e2e.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig

HT = rt.HOOK_TYPE_RESID_PRE
HID = 8
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 64 * 1024 * 1024
    engine = ne.RingEngine(cfg, None)
    engine.init(0)
    ne.ring_set_active_engine(engine)
    engine.enable_toggle_capture(True)
    engine.set_null_mode(True)
    engine.start()

    transport = RingTransport(engine)
    payload = engine.payload_tensor()
    transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                                             head_dim=HID // 2, dtype=torch.float32))
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(3)]

    # Mixed graph: two BASIC producers (layers 0,1 -> registered) + one CHUNKED
    # producer (layer 2 -> anomaly, not registered).
    src = [torch.full((4, HID), float(j), device="cuda", dtype=torch.float32) for j in range(2)]
    cx = torch.full((32,), 1.0, device="cuda", dtype=torch.float32)
    chunk_bytes = torch.tensor([16, 16], dtype=torch.int64, device="cuda")  # K=2
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        torch.ops.ring.producer(payload, src[0], HT, 0)
        torch.ops.ring.producer(payload, src[1], HT, 1)
        torch.ops.ring.producer_chunked(payload, cx, chunk_bytes, HT, 2)
    g.instantiate()
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())

    check(engine.toggle_node_count() == 2,
          f"only the 2 basic producers registered (got {engine.toggle_node_count()})")
    check(engine.capture_anomaly_count() >= 1,
          f"chunked producer flagged a capture anomaly (got {engine.capture_anomaly_count()})")

    print("[set_active_hooks must fail closed]")
    raised = False
    try:
        transport.set_active_hooks([(HT, 0)])
    except RuntimeError as e:
        msg = str(e)
        raised = "chunked" in msg and "gpu_padding_strip" in msg
        print("   ->", msg.split(" -- ")[0] if " -- " in msg else msg[:80])
    check(raised, "set_active_hooks RAISES naming 'chunked' + the remedy (fail-closed)")

    print("[set_active_hooks_lazy must fail closed too]")
    raised_lazy = False
    try:
        transport.set_active_hooks_lazy([(HT, 0)])
    except RuntimeError as e:
        raised_lazy = "chunked" in str(e)
    check(raised_lazy, "set_active_hooks_lazy RAISES (fail-closed)")

    ne.ring_clear_active_engine()
    engine.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
