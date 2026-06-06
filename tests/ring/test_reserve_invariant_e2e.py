"""Reserve-invariant guardrail (Layer 3: capacity path reads effective_specs).

Drives the REAL adaptor capacity path -- BackendAdaptor._compute_step_plan ->
prepare_step -> pre_push_all_metas -> replay -> drain -- across a sequence of
node-toggle reconfigures, and asserts the reserve invariant every round:

  - _compute_step_plan's n_hooks == #enabled hooks (capacity-reserve set ==
    meta-push set == device-enabled set, all read transport.effective_specs),
  - after flush the ring FULLY DRAINS (payload/task head == tail_committed):
    reserve() matched what producers actually wrote, so the head-tail gap does
    NOT drift. Reserving for the full active_specs while only the enabled subset
    fires (the pre-Layer-3 bug) would leave head > tail by the over-reserved
    amount, growing every step -> caught here.

Targets the post-#40/#51 backend (4-arg producer op). Option-B: basic producer
only (no gpu_padding_strip), so every node is toggle-recorded.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_reserve_invariant_e2e.py
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
fails = 0

# Includes over-reserve-prone transitions: full->empty, partial, full again.
ROUNDS = [[0, 1, 2, 3], [0, 2], [], [1], [0, 1, 2, 3], [3], []]


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


class _ProbeAdaptor(BackendAdaptor):
    """Minimal concrete adaptor: we only exercise the inherited
    _compute_step_plan, so the abstract methods are stubs."""
    def detect_model_shape(self, model):        return None
    def detect_parallel_ranks(self):            return (0, 0, 0, 0)
    def is_pp_first(self):                        return True
    def is_pp_last(self):                         return True
    def build_step_context(self, *raw):          return None
    def on_capacity_exceeded(self, ctx):         pass


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
    payload = engine.payload_tensor()
    mcfg = ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                            head_dim=HID // 2, dtype=torch.float32)
    transport.set_model_cfg(mcfg)
    specs = [HookSpec(hook_type=HT, module=None, layer_no=j) for j in range(N)]
    transport._active_specs = specs

    # Minimal adaptor wired to our hand-built transport (bypass __init__, which
    # expects a MonitoringEngine).
    adaptor = _ProbeAdaptor.__new__(_ProbeAdaptor)
    adaptor.model_cfg = mcfg
    adaptor.active_specs = specs
    adaptor.transport = transport
    adaptor.ring_engine = engine
    adaptor._warned_shapes = set()

    src = [torch.full((QLEN, HID), float(j), device="cuda", dtype=torch.float32) for j in range(N)]
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer(payload, src[j], HT, j)
    g.instantiate()
    engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    engine.set_null_mode(False)

    def ctx_for():
        return StepContext(model_id="m", flattened=True, req_ids=["r"],
                           token_ranges=[(0, QLEN)], dim0_offsets=[0],
                           kv_offsets=[0], batch=0, q_len=QLEN, kv_dim=0)

    for r_i, keep in enumerate(ROUNDS):
        collected.clear()
        transport.set_active_hooks([(HT, j) for j in keep])

        ctx = ctx_for()
        total_bytes, n_hooks, needs_eager = adaptor._compute_step_plan(ctx)
        check(n_hooks == len(keep),
              f"round {r_i} keep={keep}: _compute_step_plan n_hooks={n_hooks} == {len(keep)}")

        if n_hooks > 0:
            res = engine.prepare_step(total_bytes, n_hooks)
            check(res == 0, f"round {r_i}: prepare_step RING_OK (got {res})")
        transport.set_step_context(**ctx.transport_kwargs())
        transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)

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
              f"round {r_i} keep={keep}: ring fully drained "
              f"(payload_gap={pgap}, task_gap={tgap}) -> reserve == writes")
        check(sorted(collected) == sorted(keep),
              f"round {r_i} keep={keep}: delivered {sorted(collected)} == enabled")

    engine.flush_and_wait()
    time.sleep(0.05)
    ne.ring_clear_active_engine()
    engine.stop()

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
