"""force_eager (capacity-overflow) on a gated decode step must still gate.

The eager safety net (HookPoint.forward's force_eager branch) fires producers
WITHOUT a graph, so cudaGraphNodeSetEnabled cannot gate them. On a gated decode
step that overflowed -> force_eager, a disabled hook must skip there too, or it
fires while meta/reserve counted only the enabled subset -> ring desync (same
shape as the prefill over-fire bug, via a different eager path).

This gate builds real HookPoints, captures them (so they're registered), sets a
decode subset active, then drives every HookPoint.forward EAGERLY with
force_eager=True and _gated_step=True (the gated-decode-overflow case).

NOTE the failure mode is content corruption, NOT a ring gap: the eager path
reserves per-hook (reserve_one, 1:1 with dispatch), so head/tail stay balanced
even when over-firing. The damage is meta<->payload MISPAIRING -- N producers
fire but only K metas were pushed, so the flat FIFO pairs the K metas with the
WRONG K payloads. Each producer writes its layer index as a marker, so the test
checks delivered (layer_no == payload marker); a disabled hook firing shifts the
pairing and a row's marker stops matching its label. Teeth: neutering the eager
gate makes all N fire -> mispaired markers -> FAIL.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_force_eager_gate_e2e.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig
from monitoring.hook_points import HookPoint

HT = rt.HOOK_TYPE_RESID_PRE
N = 4
QLEN, HID = 4, 8
SUBSET = [0, 2]
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    collected = []      # (layer_no from meta, marker value from payload)
    def submit(model_id, shard_rank, req_id, act_name, layer_no, s, e, slice_):
        marker = int(round(float(slice_.flatten()[0].item())))
        collected.append((int(layer_no), marker))

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
    rt.activate(transport)                      # _active_transport -> transport
    mcfg = ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                            head_dim=HID // 2, dtype=torch.float32)
    transport.set_model_cfg(mcfg)
    specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    transport._active_specs = specs

    payload = engine.payload_tensor()
    # Build real HookPoints; capture them into a graph so they register, bind it.
    hps = []
    for j in range(N):
        hp = HookPoint()
        hp._ring_hook_type = HT
        hp._ring_hook_id = j
        hp._ring_payload = payload
        hp._strip_tensor = None
        hp._strip_row_bytes = 0
        hp.enabled = True
        hps.append(hp)

    xs = [torch.full((QLEN, HID), float(j), device="cuda", dtype=torch.float32)
          for j in range(N)]
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            hps[j](xs[j])               # captures each producer node
    g.instantiate()
    engine.enable_toggle_capture(False)
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    engine.set_null_mode(False)

    transport.set_active_hooks([(HT, j) for j in SUBSET])   # gate active, subset

    # Simulate a gated decode step forced to the eager safety net.
    transport._gated_step = True
    transport.force_eager = True
    transport.set_step_context(model_id="m", req_ids=["r"], token_ranges=[(0, QLEN)],
                               dim0_offsets=[0], kv_offsets=[0], flattened=True)
    transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)   # subset metas

    for j in range(N):                  # every HookPoint.forward runs (eager)
        hps[j](xs[j])
    torch.cuda.synchronize()

    engine.flush_and_wait()
    for _ in range(200):
        if len(collected) >= len(SUBSET):
            break
        time.sleep(0.01)
    time.sleep(0.03)
    engine.flush_and_wait()

    got_layers = sorted(l for l, _ in collected)
    check(got_layers == sorted(SUBSET),
          f"eager force_eager delivered layers {got_layers} == subset {sorted(SUBSET)}")
    # The teeth: every delivered row's payload marker must equal its label.
    # A disabled hook firing shifts the FIFO pairing -> marker != layer_no.
    mispaired = [(l, m) for l, m in collected if l != m]
    check(not mispaired,
          f"every row's payload marker matches its layer label "
          f"(mispaired={mispaired}) -- disabled hooks did NOT fire into the FIFO")

    transport.force_eager = False
    engine.flush_and_wait()
    time.sleep(0.05)
    rt.deactivate()
    ne.ring_clear_active_engine()
    engine.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
