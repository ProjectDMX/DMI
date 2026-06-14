"""Prefill (eager) steps must NOT be node-toggle-gated -- the separate-path fix.

The toggle's SetEnabled only gates producers running inside a replayed decode
graph. A prefill / eager step runs producers UNGATED (no graph), so its
meta-push + capacity-reserve must use the FULL active set that step, not the
toggle subset. If they used the subset, the eager step would fire N producers
while reserving/pushing only K metas -> a permanent surplus of (N-K) orphan
task entries that cascade-corrupt every later row (flat FIFO 1:1 pairing in
p2p_thread). transport._gated_step selects the per-step set; the vLLM adapter
sets it False for prefill/eager, True for decode-graph steps.

This gate drives the REAL capacity path (_compute_step_plan -> prepare_step ->
pre_push_all_metas -> drain) for BOTH a decode-graph step (gated subset, via
graph replay) and a prefill step (full set, via eager producer launches), and
asserts each delivers exactly its expected layers AND the ring fully drains
(gap==0) -- i.e. no surplus, no cascade.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_prefill_passthrough_e2e.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig
from monitoring.adaptor_base import BackendAdaptor
from monitoring.step_context import StepContext

HT = rt.HOOK_TYPE_RESID_PRE
N = 4
QLEN, HID = 4, 8
SUBSET = [0, 2]          # decode-step enabled subset
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


class _ProbeAdaptor(BackendAdaptor):
    def detect_model_shape(self, model):     return None
    def detect_parallel_ranks(self):         return (0, 0, 0, 0)
    def is_pp_first(self):                    return True
    def is_pp_last(self):                     return True
    def build_step_context(self, *raw):       return None
    def on_capacity_exceeded(self, ctx):     pass


def drain_collect(engine, collected, want):
    engine.flush_and_wait()
    for _ in range(200):
        if len(collected) >= want:
            break
        time.sleep(0.01)
    time.sleep(0.03)
    engine.flush_and_wait()


def gaps(engine):
    st = engine.get_stats()
    return (st.cpu_payload_head - st.cpu_payload_tail_committed,
            st.cpu_task_head - st.cpu_task_tail_committed)


def main():
    collected = []
    def submit(model_id, shard_rank, req_id, act_name, layer_no, s, e, slice_):
        collected.append(int(layer_no))

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
    specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    transport._active_specs = specs

    adaptor = _ProbeAdaptor.__new__(_ProbeAdaptor)
    adaptor.model_cfg = mcfg
    adaptor.active_specs = specs
    adaptor.transport = transport
    adaptor.ring_engine = engine
    adaptor._warned_shapes = set()

    # Capture the decode graph (all N producers) and bind it.
    src = [torch.full((QLEN, HID), float(j), device="cuda", dtype=torch.float32)
           for j in range(N)]
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer(engine.payload_tensor(), src[j], HT, j)
    g.instantiate()
    engine.enable_toggle_capture(False)
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    engine.set_null_mode(False)

    transport.set_active_hooks([(HT, j) for j in SUBSET])   # gate active, subset enabled

    def run_step(gated, fire):
        """fire: 'graph' (replay, SetEnabled-gated) or 'eager' (direct launches)."""
        collected.clear()
        transport._gated_step = gated
        expect = [s.layer_no for s in transport.specs_for_step()]
        ctx = StepContext(model_id="m", flattened=True, req_ids=["r"],
                          token_ranges=[(0, QLEN)], dim0_offsets=[0], kv_offsets=[0],
                          batch=0, q_len=QLEN, kv_dim=0)
        total_bytes, n_hooks, _ = adaptor._compute_step_plan(ctx)
        if n_hooks > 0:
            assert engine.prepare_step(total_bytes, n_hooks) == 0
        transport.set_step_context(model_id="m", req_ids=["r"], token_ranges=[(0, QLEN)],
                                   dim0_offsets=[0], kv_offsets=[0], flattened=True)
        transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        if fire == "graph":
            g.replay()
        else:
            for j in range(N):     # eager: ALL producers fire, ungated
                torch.ops.ring.producer(engine.payload_tensor(), src[j], HT, j)
        torch.cuda.synchronize()
        drain_collect(engine, collected, len(expect))
        return sorted(expect), sorted(collected)

    print("[decode step: graph replay, gated subset]")
    expect, got = run_step(gated=True, fire="graph")
    check(got == expect == sorted(SUBSET), f"decode delivered {got} == subset {sorted(SUBSET)}")
    pg, tg = gaps(engine)
    check(pg == 0 and tg == 0, f"ring drained after decode (payload_gap={pg}, task_gap={tg})")

    print("[prefill step: eager launches, FULL set (not gated)]")
    expect, got = run_step(gated=False, fire="eager")
    check(got == expect == list(range(N)),
          f"prefill delivered {got} == full {list(range(N))} (all hooks, ungated)")
    pg, tg = gaps(engine)
    check(pg == 0 and tg == 0,
          f"ring FULLY drained after prefill -- no surplus/cascade (payload_gap={pg}, task_gap={tg})")

    print("[interleave decode again: still aligned after a prefill]")
    expect, got = run_step(gated=True, fire="graph")
    check(got == sorted(SUBSET), f"post-prefill decode still {got} == {sorted(SUBSET)}")
    pg, tg = gaps(engine)
    check(pg == 0 and tg == 0, f"ring drained after 2nd decode (payload_gap={pg}, task_gap={tg})")

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
