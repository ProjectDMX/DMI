"""Reserve invariant for the prefix (strip) producer under node-toggle.

Complements test_toggle_prefix_e2e (which checks lockstep + stripped output) by
driving the REAL capacity path -- adaptor._compute_step_plan -> prepare_step
(reserve) -> prefix replay -> drain -- and asserting, every round, that the ring
fully drains (payload_gap == task_gap == 0). That proves CPU-reserved bytes ==
prefix-kernel written bytes, i.e. the strip byte-accounting stays in lockstep
even as the per-step actual token count varies.

Each round varies BOTH the enabled subset AND the actual row count (row_count
device scalar + ctx.actual_q_len), so the reserve must track the stripped size,
not the padded capture size.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_prefix_reserve_e2e.py
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
PADDED, HID = 8, 8
ROW_BYTES = HID * 4
# (enabled subset, actual row count) -- vary both, incl. full/empty/partial.
ROUNDS = [([0, 1, 2, 3], 4), ([0, 2], 2), ([], 4), ([1], 1), ([0, 1, 2, 3], 7), ([3], 3)]
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


class _ProbeAdaptor(BackendAdaptor):
    def detect_model_shape(self, model):        return None
    def detect_parallel_ranks(self):            return (0, 0, 0, 0)
    def is_pp_first(self):                        return True
    def is_pp_last(self):                         return True
    def build_step_context(self, *raw):          return None
    def on_capacity_exceeded(self, ctx):         pass


def main():
    collected = []
    def submit(model_id, shard_rank, req_id, act_name, layer_no, s, e, slice_):
        collected.append((int(layer_no), slice_.shape[0]))

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
    # dim0_is_actual_tokens=True -> _compute_step_plan sizes by ctx.actual_q_len.
    specs = [HookSpec(hook_type=HT, module=None, layer_no=j, dim0_is_actual_tokens=True)
             for j in range(N)]
    transport._active_specs = specs

    adaptor = _ProbeAdaptor.__new__(_ProbeAdaptor)
    adaptor.model_cfg = mcfg
    adaptor.active_specs = specs
    adaptor.transport = transport
    adaptor.ring_engine = engine
    adaptor._warned_shapes = set()

    src = [torch.full((PADDED, HID), float(j), device="cuda", dtype=torch.float32) for j in range(N)]
    row_count = torch.tensor([PADDED], dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer_prefix(payload, src[j], row_count, ROW_BYTES, HT, j)
    g.instantiate()
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    engine.set_null_mode(False)

    for r_i, (keep, actual) in enumerate(ROUNDS):
        collected.clear()
        transport.set_active_hooks([(HT, j) for j in keep])
        row_count.fill_(actual)                       # device count the prefix kernel reads
        torch.cuda.synchronize()

        ctx = StepContext(model_id="m", flattened=True, req_ids=["r"],
                          token_ranges=[(0, actual)], dim0_offsets=[0], kv_offsets=[0],
                          batch=0, q_len=PADDED, kv_dim=0, actual_q_len=actual)
        total_bytes, n_hooks, _ = adaptor._compute_step_plan(ctx)
        check(n_hooks == len(keep),
              f"round {r_i} keep={keep} actual={actual}: plan n_hooks={n_hooks} == {len(keep)}")
        if n_hooks > 0:
            res = engine.prepare_step(total_bytes, n_hooks)
            check(res == 0, f"round {r_i}: prepare_step RING_OK (got {res})")
        transport.set_step_context(model_id="m", req_ids=["r"], token_ranges=[(0, actual)],
                                   dim0_offsets=[0], kv_offsets=[0], flattened=True)
        transport.pre_push_all_metas(batch=0, q_len=PADDED, kv_dim=0, actual_q_len=actual)
        g.replay()
        torch.cuda.synchronize()
        engine.flush_and_wait()
        for _ in range(200):
            if len(collected) >= len(keep):
                break
            time.sleep(0.01)
        time.sleep(0.03)
        engine.flush_and_wait()

        st = engine.get_stats()
        pgap = st.cpu_payload_head - st.cpu_payload_tail_committed
        tgap = st.cpu_task_head - st.cpu_task_tail_committed
        check(pgap == 0 and tgap == 0,
              f"round {r_i} keep={keep} actual={actual}: ring fully drained "
              f"(payload_gap={pgap}, task_gap={tgap}) -> reserve == prefix writes")
        check(sorted(l for l, _ in collected) == sorted(keep),
              f"round {r_i}: delivered {sorted(l for l,_ in collected)} == enabled")
        check(all(nrows == actual for _, nrows in collected),
              f"round {r_i}: each slice stripped to actual={actual} rows")

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
