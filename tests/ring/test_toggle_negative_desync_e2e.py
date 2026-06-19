"""Negative test: breaking meta/producer lockstep MUST desync (and the correct
order MUST stay aligned).

This is the complement of test_reconfig_sequence_e2e. It proves the FIFO-order
invariant is load-bearing: the drain/p2p match metas to payloads purely by
arrival order, so if the host meta ORDER diverges from the device producer
firing order, payloads get mis-associated with the wrong layer (desync). The
production path never does this because all lanes iterate effective_specs in the
single firing order; here we deliberately push a REVERSED meta order to show the
corruption is real.

Each producer writes marker == its layer, so a correctly-aligned delivery has
layer_no == marker for every row; a desync shows layer_no != marker.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_negative_desync_e2e.py
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
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def collect_round(engine, collected):
    """Replay + drain + wait for p2p submit."""
    torch.cuda.synchronize()
    engine.flush_and_wait()
    for _ in range(200):
        if len(collected) >= N:
            break
        time.sleep(0.01)
    time.sleep(0.03)


def main():
    collected = []
    def submit(model_id, shard_rank, req_id, act_name, layer_no, s, e, slice_):
        collected.append((int(layer_no), int(round(float(slice_.reshape(-1)[0].item())))))

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
    payload = engine.payload_tensor()
    mcfg = ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                            head_dim=HID // 2, dtype=torch.float32)
    transport.set_model_cfg(mcfg)
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    transport.set_step_context(model_id="m", req_ids=["r"], token_ranges=[(0, QLEN)],
                               dim0_offsets=[0], kv_offsets=[0], flattened=True)

    src = [torch.full((QLEN, HID), float(j), device="cuda", dtype=torch.float32) for j in range(N)]
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer(payload, src[j], HT, j)
    g.instantiate()
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    engine.set_null_mode(False)

    # All producers enabled (device fires layers 0..3 in order, markers 0..3).
    transport.set_active_hooks([(HT, j) for j in range(N)])

    # --- POSITIVE CONTROL: correct meta order via the production path -> aligned.
    print("[positive: correct order via pre_push_all_metas]")
    collected.clear()
    transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
    g.replay()
    collect_round(engine, collected)
    aligned = all(layer == marker for layer, marker in collected)
    check(sorted(l for l, _ in collected) == [0, 1, 2, 3] and aligned,
          f"correct order -> all aligned (layer==marker): {sorted(collected)}")

    # --- NEGATIVE: deliberately REVERSED meta order -> must desync.
    print("[negative: reversed meta order pushed directly -> must desync]")
    collected.clear()
    shape = [QLEN, HID]
    engine.push_all_metas(
        hook_types=[HT] * N,
        layer_nos=[3, 2, 1, 0],                 # REVERSED vs producer firing order
        shapes=[shape] * N,
        dtypes=[torch.float32] * N,
        flags=[0] * N,
        model_id="m", tp_rank=0, dp_rank=0, ep_rank=0, pp_rank=0,
        flattened=True, req_ids=["r"], token_ranges=[(0, QLEN)],
        dim0_offsets=[0], kv_offsets=[0],
    )
    g.replay()
    collect_round(engine, collected)
    # FIFO pairs meta[i] (layer 3,2,1,0) with payload[i] (marker 0,1,2,3) -> mismatched.
    mismatched = sum(1 for layer, marker in collected if layer != marker)
    check(len(collected) == N and mismatched == N,
          f"reversed order -> ALL rows mis-associated (desync detected): {sorted(collected)}")
    check(mismatched > 0,
          "lockstep is load-bearing: breaking meta order corrupts layer<->payload mapping")

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
