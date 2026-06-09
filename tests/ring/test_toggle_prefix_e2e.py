"""Node-toggle composes with the gpu_padding_strip (prefix) producer.

Drives torch.ops.ring.producer_prefix (the strip producer: reads a device
row_count scalar at execution, copies only the first row_count rows) inside a
captured graph with toggle-capture on, and verifies:
  - the prefix producer's kernel node IS recorded (toggle_node_count == N) and
    no capture anomaly (capture_anomaly_count == 0),
  - reconfigure lockstep holds: delivered layers == enabled set every round,
    each delivered slice is the correct stripped prefix (marker == layer),
    disabled hooks deliver nothing.

The chunked producer is deliberately NOT covered here -- it is not
toggle-managed and fails loud at set_active_hooks
(see test_toggle_chunked_failclosed_e2e).

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_prefix_e2e.py
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
PADDED, ACTUAL, HID = 8, 4, 8          # captured tensor is [PADDED,HID]; strip to ACTUAL rows
ROW_BYTES = HID * 4                     # float32 bytes per token row
ROUNDS = [[0, 2], [1, 3], [0, 1, 2, 3], [], [2], [0, 3], [], [1]]
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    collected = []   # (layer_no, marker, n_rows)
    def submit(model_id, shard_rank, req_id, act_name, layer_no, s, e, slice_):
        flat = slice_.reshape(-1)
        collected.append((int(layer_no), int(round(float(flat[0].item()))), slice_.shape[0]))

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
    transport.set_step_context(model_id="m", req_ids=["r"], token_ranges=[(0, ACTUAL)],
                               dim0_offsets=[0], kv_offsets=[0], flattened=True)

    # Padded source tensors (value == layer); device row_count scalar = ACTUAL.
    src = [torch.full((PADDED, HID), float(j), device="cuda", dtype=torch.float32) for j in range(N)]
    row_count = torch.tensor([ACTUAL], dtype=torch.int64, device="cuda")

    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer_prefix(payload, src[j], row_count, ROW_BYTES, HT, j)
    g.instantiate()
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())

    check(engine.toggle_node_count() == N,
          f"prefix producer nodes recorded: toggle_node_count == {N} (got {engine.toggle_node_count()})")
    check(engine.capture_anomaly_count() == 0,
          f"no capture anomaly for prefix producers (got {engine.capture_anomaly_count()})")

    engine.set_null_mode(False)

    for r_i, keep in enumerate(ROUNDS):
        collected.clear()
        transport.set_active_hooks([(HT, j) for j in keep])
        eff = sorted(s.layer_no for s in transport.effective_specs)
        check(eff == sorted(keep), f"round {r_i} keep={keep}: effective_specs {eff} == enabled")
        # meta shape uses the STRIPPED row count (ACTUAL), matching the prefix copy.
        transport.pre_push_all_metas(batch=0, q_len=ACTUAL, kv_dim=0)
        g.replay()
        torch.cuda.synchronize()
        engine.flush_and_wait()
        for _ in range(200):
            if len(collected) >= len(keep):
                break
            time.sleep(0.01)
        time.sleep(0.03)

        delivered = sorted(l for l, _, _ in collected)
        aligned = all(layer == marker for layer, marker, _ in collected)
        stripped = all(nrows == ACTUAL for _, _, nrows in collected)
        tag = f"round {r_i} keep={keep}"
        check(delivered == sorted(keep), f"{tag}: delivered {delivered} == enabled {sorted(keep)}")
        check(aligned, f"{tag}: each slice aligned (layer==marker, no desync)")
        check(stripped, f"{tag}: each slice stripped to {ACTUAL} rows (prefix copy correct)")

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
