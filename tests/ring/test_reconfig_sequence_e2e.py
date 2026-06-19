"""Reconfigure-sequence alignment guardrail (transport-level lockstep).

Runs a SEQUENCE of set_active_hooks() reconfigures (including empty/full and
arbitrary transitions) on one captured graph, replaying after each, and verifies
the meta gate stays aligned every round:
  - delivered layers == the round's enabled set (host meta set == device enabled),
  - each delivered slice is aligned (layer_no == payload marker, no desync),
  - disabled hooks deliver nothing.

This guards the toggle *reconfigure* path through the REAL pipeline
(producer -> drain -> p2p -> submit): a wrong reconfigure (stale node, wrong
graph, bad diff seed, or a meta-gate that diverges from the device enabled set)
would desync and fail here.

Every hook uses the basic producer
``producer(ring_payload, x, hook_type, hook_id)`` (no gpu_padding_strip), so all
nodes are toggle-recorded.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_reconfig_sequence_e2e.py
Requires the built backend + torch CUDA.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig

HT = rt.HOOK_TYPE_RESID_PRE          # hidden-dim hook: shape [q_len, hidden_dim]
N = 4
QLEN, HID = 4, 8
fails = 0

# Reconfigure sequence -- exercises every transition kind: partial->partial,
# disjoint swap, grow to full, full->empty, empty->single, and empty->empty.
ROUNDS = [[0, 2], [1, 3], [0, 1, 2, 3], [], [2], [0, 3], [], [1]]


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
    engine = ne.RingEngine(cfg, submit)
    engine.init(0)
    ne.ring_set_active_engine(engine)
    engine.enable_toggle_capture(True)
    engine.set_null_mode(True)
    engine.start()

    transport = RingTransport(engine)
    payload = engine.payload_tensor()    # the Tensor(a!) alias every producer op takes
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
            torch.ops.ring.producer(payload, src[j], HT, j)
    g.instantiate()
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    check(engine.toggle_node_count() == N, f"registered {N} nodes (got {engine.toggle_node_count()})")

    engine.set_null_mode(False)

    for r_i, keep in enumerate(ROUNDS):
        collected.clear()
        transport.set_active_hooks([(HT, j) for j in keep])          # device toggle + gate
        # effective_specs (the single source) must equal the enabled set.
        eff = sorted(s.layer_no for s in transport.effective_specs)
        check(eff == sorted(keep), f"round {r_i} keep={keep}: effective_specs {eff} == enabled")
        transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)  # gated meta push
        g.replay()
        torch.cuda.synchronize()
        engine.flush_and_wait()                                      # force drain
        for _ in range(200):                                         # wait for p2p submit
            if len(collected) >= len(keep):
                break
            time.sleep(0.01)
        time.sleep(0.03)

        delivered = sorted(l for l, _ in collected)
        aligned = all(layer == marker for layer, marker in collected)
        only_enabled = all(l in keep for l, _ in collected)
        tag = f"round {r_i} keep={keep}"
        check(delivered == sorted(keep), f"{tag}: delivered {delivered} == enabled {sorted(keep)}")
        check(aligned, f"{tag}: each slice aligned (layer_no == marker, no desync)")
        check(only_enabled, f"{tag}: no disabled hook delivered")

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
