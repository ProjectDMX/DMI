"""Per-HookPoint request-row gather and in-graph heads on a real legacy ring.

HookPoints are armed in gather mode (a per-HookPoint row table) and, for the
heads, with a fused-node arena; plans from dmi.hook_rows feed both the
producer tables and the per-hook request layouts in the metadata.  An
in-process consumer checks that every delivered slice is attributed to the
right request and token range and carries the right rows (or scores).

Needs CUDA and the full native backend.  Memory use is a few MiB.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]

HID = 16
MAX_SEGS = 8


class Ring:
    def __init__(self, payload=8 << 20):
        from dmi.hooks.specs import ModelShapeConfig
        from dmi.transport import native as ne
        from dmi.transport import ring as rt

        self.slices = []
        cfg = ne.RingConfig()
        cfg.payload_ring_bytes = payload
        cfg.pinned_staging_bytes = payload
        cfg.drain_poll_timeout_us = 200

        def consume(model_id, shard_rank, req_id, act_name, layer_no,
                    start, end, tensor):
            self.slices.append((act_name, int(layer_no), req_id, int(start),
                                int(end), tensor.clone()))

        self.engine = ne.RingEngine(cfg, consume)
        self.engine.init(0)
        self.engine.start()
        self.transport = rt.RingTransport(self.engine)
        rt.activate(self.transport)
        self.transport.set_model_cfg(ModelShapeConfig(
            hidden_dim=HID, num_heads=2, num_kv_heads=2, head_dim=HID // 2,
            dtype=torch.float32, head_slots=0))
        self.payload = self.engine.payload_tensor()

    def drain(self, expected):
        torch.cuda.synchronize()
        self.engine.flush_and_wait()
        deadline = time.monotonic() + 5.0
        while len(self.slices) < expected and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.02)
        got = list(self.slices)
        self.slices.clear()
        return got

    def close(self):
        from dmi.transport import ring as rt
        torch.cuda.synchronize()
        self.engine.flush_and_wait()
        rt.deactivate()
        self.engine.stop()


@pytest.fixture
def ring():
    r = Ring()
    try:
        yield r
    finally:
        r.close()


def _hook(ring, hook_type, layer, table, row_bytes=HID * 4):
    from dmi.hook_rows import configure_hook_gather
    from dmi.hooks.point import HookPoint

    hook = HookPoint()
    hook._ring_hook_type = hook_type
    hook._ring_hook_id = layer
    hook._ring_payload = ring.payload
    configure_hook_gather(hook, table, row_bytes)
    return hook


def _spec(hook_type, layer):
    from dmi.hooks.specs import HookSpec
    return HookSpec(hook_type=hook_type, module=None, layer_no=layer,
                    dim0_is_actual_tokens=True)


def _write_table(table, plan):
    from dmi.hook_rows import encode_row_table
    table.copy_(torch.from_numpy(encode_row_table(plan, MAX_SEGS)))


def _expected(x, plan, act, layer):
    """Consumer slices a correct gather must produce for `plan`."""
    out = []
    for rid, (lo, hi), source in zip(plan.req_ids, plan.token_ranges,
                                     plan.source_offsets):
        out.append((act, layer, rid, lo, hi, x[source:source + (hi - lo)].cpu()))
    return out


def _compare(got, want):
    key = lambda s: (s[0], s[1], s[2], s[3])
    got, want = sorted(got, key=key), sorted(want, key=key)
    assert [key(s) for s in got] == [key(s) for s in want]
    for g, w in zip(got, want):
        assert torch.equal(g[5], w[5]), key(g)


def _step_layout():
    # Three requests packed into 9 rows with a padding row at 4.
    req_ids = ["a", "b", "c"]
    ranges = [(10, 14), (0, 3), (7, 8)]
    offsets = [0, 5, 8]
    return req_ids, ranges, offsets, [0, 1, 2], 9


def test_two_hooks_with_independent_selections_eager_and_graph(ring):
    from dmi.hook_rows import plan_hook_rows
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    ht = HOOK_TYPE_RESID_PRE
    req_ids, ranges, offsets, pos, total = _step_layout()
    x = torch.arange(total * HID, dtype=torch.float32, device="cuda").reshape(total, HID)
    tables = [torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
              for _ in range(2)]
    hooks = [_hook(ring, ht, layer, tables[layer]) for layer in range(2)]
    ring.transport._active_specs = [_spec(ht, 0), _spec(ht, 1)]

    selections = [({"a", "c"}, {"b"}), ({"b"}, set()), (None, {"a", "b", "c"})]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for hook in hooks:
            hook(x)
    ring.drain(0)
    for mode in ("eager", "graph"):
        for sel0, sel1 in selections:
            plans = [plan_hook_rows(req_ids, ranges, offsets, pos, sel, total, MAX_SEGS)
                     for sel in (sel0, sel1)]
            for table, plan in zip(tables, plans):
                _write_table(table, plan)
            ring.transport.set_step_context(
                model_id="m", req_ids=req_ids, token_ranges=ranges,
                dim0_offsets=offsets, flattened=True,
                hook_rows={(ht, 0): plans[0], (ht, 1): plans[1]})
            ring.transport.pre_push_all_metas(batch=0, q_len=total, kv_dim=0)
            if mode == "eager":
                for hook in hooks:
                    hook(x)
            else:
                graph.replay()
            want = (_expected(x, plans[0], "blocks.hook_resid_pre", 0)
                    + _expected(x, plans[1], "blocks.hook_resid_pre", 1))
            _compare(ring.drain(len(want)), want)


def test_disabled_rows_publish_nothing_and_skip_eager_launches(ring):
    from dmi.hook_rows import plan_hook_rows
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    ht = HOOK_TYPE_RESID_PRE
    req_ids, ranges, offsets, pos, total = _step_layout()
    x = torch.randn(total, HID, device="cuda")
    host = torch.zeros(2, 1 + 2 * MAX_SEGS, dtype=torch.int64).pin_memory()
    dev = torch.zeros_like(host, device="cuda")
    hooks = [_hook(ring, ht, layer, dev[layer]) for layer in range(2)]
    ring.engine.set_gather_host_mirror(dev.data_ptr(), host.data_ptr(),
                                       host.stride(0) * 8, 2)
    plan = plan_hook_rows(req_ids, ranges, offsets, pos, {"b"}, total, MAX_SEGS)
    from dmi.hook_rows import encode_row_table
    host[0] = torch.from_numpy(encode_row_table(plan, MAX_SEGS))
    host[1] = torch.from_numpy(encode_row_table(None, MAX_SEGS))   # disabled
    dev.copy_(host, non_blocking=True)
    # Only the enabled hook has metadata: a disabled row has none.
    ring.transport._active_specs = [_spec(ht, 0)]
    ring.transport.set_step_context(model_id="m", req_ids=req_ids,
                                    token_ranges=ranges, dim0_offsets=offsets,
                                    flattened=True, hook_rows={(ht, 0): plan})
    ring.transport.pre_push_all_metas(batch=0, q_len=total, kv_dim=0)
    skips = ring.engine.gather_host_skips()
    for hook in hooks:
        hook(x)
    _compare(ring.drain(1), _expected(x, plan, "blocks.hook_resid_pre", 0))
    assert ring.engine.gather_host_skips() == skips + 1
    assert ring.engine.available_task_slots() == ring.engine.task_cap()
    ring.engine.set_gather_host_mirror(0, 0, 0, 0)


def test_gather_nodes_are_toggle_managed(ring):
    """A gather producer captured inside the toggle window records its node,
    so disabling its key disables it (otherwise it would fire while the
    metadata gate excluded it)."""
    from dmi.hook_rows import plan_hook_rows
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    ht = HOOK_TYPE_RESID_PRE
    req_ids, ranges, offsets, pos, total = _step_layout()
    x = torch.randn(total, HID, device="cuda")
    tables = [torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
              for _ in range(3)]
    hooks = [_hook(ring, ht, layer, tables[layer]) for layer in range(3)]
    ring.transport._active_specs = [_spec(ht, layer) for layer in range(3)]
    ring.engine.enable_toggle_capture(True)
    ring.engine.set_null_mode(True)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for hook in hooks:
            hook(x)
    graph.instantiate()
    ring.engine.set_null_mode(False)
    ring.engine.enable_toggle_capture(False)
    ring.engine.bind_graph_exec(graph.raw_cuda_graph(), graph.raw_cuda_graph_exec())
    assert ring.engine.toggle_node_count() == 3
    assert ring.engine.capture_anomaly_count() == 0

    plan = plan_hook_rows(req_ids, ranges, offsets, pos, {"a", "b"}, total, MAX_SEGS)
    for table in tables:
        _write_table(table, plan)
    ring.transport.set_active_hooks_lazy([(ht, 2)])
    ring.transport.set_step_context(
        model_id="m", req_ids=req_ids, token_ranges=ranges, dim0_offsets=offsets,
        flattened=True, hook_rows={(ht, layer): plan for layer in range(3)})
    ring.transport.pre_push_all_metas(batch=0, q_len=total, kv_dim=0)
    assert ring.transport.ensure_graph_current(graph.raw_cuda_graph()) == 0
    graph.replay()
    _compare(ring.drain(2), _expected(x, plan, "blocks.hook_resid_pre", 2))
    assert ring.engine.available_task_slots() == ring.engine.task_cap()


def test_table_gate_follows_the_toggle_on_ungated_steps(ring):
    """With the table gate armed, an ungated (eager) step still publishes
    only the enabled hooks: disabled hooks' tables hold -1 and the eager
    safety net skips them."""
    from dmi.hook_rows import encode_row_table, full_row_plan
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    ht = HOOK_TYPE_RESID_PRE
    req_ids, ranges, offsets, pos, total = _step_layout()
    x = torch.randn(total, HID, device="cuda")
    tables = [torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
              for _ in range(2)]
    hooks = [_hook(ring, ht, layer, tables[layer]) for layer in range(2)]
    ring.transport._active_specs = [_spec(ht, 0), _spec(ht, 1)]
    ring.engine.enable_toggle_capture(True)
    ring.engine.set_null_mode(True)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for hook in hooks:
            hook(x)
    graph.instantiate()
    ring.engine.set_null_mode(False)
    ring.engine.enable_toggle_capture(False)
    ring.engine.bind_graph_exec(graph.raw_cuda_graph(), graph.raw_cuda_graph_exec())
    ring.transport.set_active_hooks_lazy([(ht, 1)])
    ring.transport.set_prefill_table_gate(True)
    ring.transport.set_gated_step(False)
    try:
        plan = full_row_plan(req_ids, ranges, offsets, pos, total)
        tables[0].copy_(torch.from_numpy(encode_row_table(None, MAX_SEGS)))
        _write_table(tables[1], plan)
        assert [(s.hook_type, s.layer_no) for s in ring.transport.specs_for_step()] == [(ht, 1)]
        ring.transport.set_step_context(
            model_id="m", req_ids=req_ids, token_ranges=ranges,
            dim0_offsets=offsets, flattened=True, hook_rows={(ht, 1): plan})
        ring.transport.pre_push_all_metas(batch=0, q_len=total, kv_dim=0)
        ring.transport.force_eager = True
        for hook in hooks:
            hook(x)
        ring.transport.force_eager = False
        _compare(ring.drain(3), _expected(x, plan, "blocks.hook_resid_pre", 1))
        assert ring.transport.gated_step_requires_registry() is False
    finally:
        ring.transport.force_eager = False
        ring.transport.set_prefill_table_gate(False)
        ring.transport.set_gated_step(True)


def _fused_arena(weights, bias, modes, acc_cols):
    """int32 node arena: modes, accumulator columns, fp32 biases, fp32 weights."""
    K, hidden = weights.shape
    words = (64 + K * hidden + 3) & ~3
    arena = np.zeros(words, dtype=np.int32)
    arena[0:K] = modes
    arena[16:16 + K] = acc_cols
    arena[32:32 + K] = np.asarray(bias, dtype=np.float32).view(np.int32)
    arena[64:64 + K * hidden] = weights.astype(np.float32).reshape(-1).view(np.int32)
    return torch.from_numpy(arena).cuda()


def test_fused_heads_score_selected_rows_and_chain_across_layers(ring):
    from dmi.hook_rows import configure_hook_heads, plan_hook_rows
    from dmi.hooks.specs import (HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_PRE_HEAD,
                                 ModelShapeConfig)

    ht, hh = HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_PRE_HEAD
    K = 3
    ring.transport.set_model_cfg(ModelShapeConfig(
        hidden_dim=HID, num_heads=2, num_kv_heads=2, head_dim=HID // 2,
        dtype=torch.float32, head_slots=K))
    req_ids, ranges, offsets, pos, total = _step_layout()
    rng = np.random.default_rng(7)
    xs = [torch.from_numpy(rng.standard_normal((total, HID)).astype(np.float32)).cuda()
          for _ in range(2)]
    w = [rng.standard_normal((K, HID)).astype(np.float32) for _ in range(2)]
    bias = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    # Slot 0: single-layer head at layer 0.  Slot 1: a linear head over
    # layers 0 and 1 (FIRST at 0, LAST at 1, accumulator column 3).
    # Slot 2: free.
    arenas = [_fused_arena(w[0], bias, [1, 2, 0], [0, 3, 0]),
              _fused_arena(w[1], bias, [0, 4, 0], [0, 3, 0])]
    acc = torch.zeros(64, 16, dtype=torch.float32, device="cuda")
    head_tables = [torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
                   for _ in range(2)]
    row_tables = [torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
                  for _ in range(2)]
    hooks = []
    for layer in range(2):
        hook = _hook(ring, ht, layer, row_tables[layer])
        configure_hook_heads(hook, head_type=hh, layer_no=layer,
                             table=head_tables[layer], arena=arenas[layer],
                             acc=acc, slots=K, weight_bf16=False)
        hooks.append(hook)
    rows_plan = plan_hook_rows(req_ids, ranges, offsets, pos, {"b"}, total, MAX_SEGS)
    head_plan = plan_hook_rows(req_ids, ranges, offsets, pos, {"a", "c"}, total, MAX_SEGS)
    for layer in range(2):
        _write_table(row_tables[layer], rows_plan)
        _write_table(head_tables[layer], head_plan)
    ring.transport._active_specs = [_spec(ht, 0), _spec(hh, 0), _spec(ht, 1), _spec(hh, 1)]
    hook_rows = {}
    for layer in range(2):
        hook_rows[(ht, layer)] = rows_plan
        hook_rows[(hh, layer)] = head_plan
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for hook, x in zip(hooks, xs):
            hook(x)
    ring.drain(0)
    for mode in ("eager", "graph"):
        ring.transport.set_step_context(
            model_id="m", req_ids=req_ids, token_ranges=ranges,
            dim0_offsets=offsets, flattened=True, hook_rows=hook_rows)
        ring.transport.pre_push_all_metas(batch=0, q_len=total, kv_dim=0)
        if mode == "eager":
            for hook, x in zip(hooks, xs):
                hook(x)
        else:
            graph.replay()
        got = ring.drain(2 + 2 * len(head_plan.req_ids))
        rows = [s for s in got if s[0] == "blocks.hook_resid_pre"]
        _compare(rows, _expected(xs[0], rows_plan, "blocks.hook_resid_pre", 0)
                 + _expected(xs[1], rows_plan, "blocks.hook_resid_pre", 1))
        heads = {(s[1], s[2]): s[5] for s in got if s[0] == "blocks.hook_resid_pre_head"}
        assert sorted(heads) == [(0, "a"), (0, "c"), (1, "a"), (1, "c")]
        for rid, (lo, hi), source in zip(head_plan.req_ids, head_plan.token_ranges,
                                         head_plan.source_offsets):
            h0 = xs[0][source:source + hi - lo].double().cpu()
            h1 = xs[1][source:source + hi - lo].double().cpu()
            w0 = torch.from_numpy(w[0]).double()
            w1 = torch.from_numpy(w[1]).double()
            layer0, layer1 = heads[(0, rid)], heads[(1, rid)]
            assert layer0.shape == (hi - lo, K)
            torch.testing.assert_close(layer0[:, 0].double(), h0 @ w0[0] + 0.5,
                                       rtol=1e-5, atol=1e-5)
            assert torch.isnan(layer0[:, 1]).all() and torch.isnan(layer0[:, 2]).all()
            chained = h0 @ w0[1] + h1 @ w1[1] - 1.0
            torch.testing.assert_close(layer1[:, 1].double(), chained,
                                       rtol=1e-5, atol=1e-5)
            assert torch.isnan(layer1[:, 0]).all() and torch.isnan(layer1[:, 2]).all()
    assert ring.engine.head_launches() >= 2


def test_original_positions_select_final_logits_rows(ring):
    """A filtered request list keeps each request's position in the
    unfiltered step, so final_logits rows stay attributed correctly."""
    from dmi.hooks.point import HookPoint
    from dmi.hooks.specs import HOOK_TYPE_FINAL_LOGITS, HookSpec, ModelShapeConfig

    vocab = 8
    ring.transport.set_model_cfg(ModelShapeConfig(
        hidden_dim=HID, num_heads=2, num_kv_heads=2, head_dim=HID // 2,
        dtype=torch.float32, vocab_size=vocab))
    logits = torch.arange(4 * vocab, dtype=torch.float32, device="cuda").reshape(4, vocab)
    hook = HookPoint()
    hook._ring_hook_type = HOOK_TYPE_FINAL_LOGITS
    hook._ring_hook_id = -1
    hook._ring_payload = ring.payload
    ring.transport._active_specs = [HookSpec(hook_type=HOOK_TYPE_FINAL_LOGITS,
                                             module=None, layer_no=-1)]
    # Requests 1 and 3 of four survive the filter.
    ring.transport.set_step_context(model_id="m", req_ids=["q1", "q3"],
                                    token_ranges=[(0, 5), (0, 2)],
                                    dim0_offsets=[3, 9], flattened=True,
                                    orig_positions=[1, 3])
    ring.transport.pre_push_all_metas(batch=0, q_len=11, kv_dim=0, logits_to_keep=4)
    hook(logits)
    got = {s[2]: s[5] for s in ring.drain(2)}
    assert torch.equal(got["q1"], logits[1:2].cpu())
    assert torch.equal(got["q3"], logits[3:4].cpu())


def test_native_layout_parser_accepts_rows_and_arrays():
    from dmi.transport import native as ne

    rows = [("b", 0, 3, 0, 0, 1), ("a", 10, 14, 3, 0, 0)]
    assert ne.parse_hook_requests(["a", "b"], rows) == rows
    arrays = (np.array([1, 0]), np.array([0, 10]), np.array([3, 14]),
              np.array([0, 3]), np.array([1, 0]))
    assert ne.parse_hook_requests(["a", "b"], arrays) == rows
    with pytest.raises(ValueError, match="outside the step"):
        ne.parse_hook_requests(["a"], tuple(np.array([v]) for v in (2, 0, 1, 0, 0)))
    with pytest.raises(ValueError, match="five fields"):
        ne.parse_hook_requests(["a"], (np.array([0]),))
    with pytest.raises(ValueError, match="range or offset"):
        ne.parse_hook_requests(["a"], [("a", 4, 2, 0, 0, 0)])


def test_device_toggle_gates_gather_producers_of_excluded_captures(
        ring, monkeypatch):
    """In device-toggle mode a capture excluded from the host registry
    (e.g. a partial-model segment graph) captures its gather producers as
    device-updatable nodes behind one planner node: each node follows its
    own table row every replay, with no host SetEnabled call."""
    from dmi import node_toggle
    from dmi.hook_rows import encode_row_table, plan_hook_rows
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    monkeypatch.setattr(torch.cuda, "CUDAGraph", torch.cuda.CUDAGraph)
    monkeypatch.setattr(node_toggle, "_PATCHED_CLASS", None)
    node_toggle.begin_capture_window(ring.engine, exclude_capture=lambda: True,
                                     device_toggle=True)
    if not ring.engine.toggle_device_mode():
        node_toggle.end_capture_window(ring.engine)
        pytest.skip("device graph updates unavailable")
    ht = HOOK_TYPE_RESID_PRE
    req_ids, ranges, offsets, pos, total = _step_layout()
    x = torch.randn(total, HID, device="cuda")
    tables = [torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
              for _ in range(3)]
    hooks = [_hook(ring, ht, layer, tables[layer]) for layer in range(3)]
    ring.engine.set_null_mode(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for hook in hooks:
            hook(x)
    graph.instantiate()
    ring.engine.set_null_mode(False)
    node_toggle.end_capture_window(ring.engine)
    assert graph._dmi_toggle_excluded
    assert ring.engine.toggle_device_node_count() == 3
    assert ring.engine.toggle_node_count() == 0      # nothing host-toggled
    assert ring.engine.capture_anomaly_count() == 0

    plan = plan_hook_rows(req_ids, ranges, offsets, pos, {"a", "c"}, total, MAX_SEGS)
    for enabled in ([0, 2], [1], [], [0, 1, 2]):
        for layer, table in enumerate(tables):
            table.copy_(torch.from_numpy(encode_row_table(
                plan if layer in enabled else None, MAX_SEGS)))
        ring.transport._active_specs = [_spec(ht, layer) for layer in enabled]
        ring.transport.set_step_context(
            model_id="m", req_ids=req_ids, token_ranges=ranges,
            dim0_offsets=offsets, flattened=True,
            hook_rows={(ht, layer): plan for layer in enabled})
        ring.transport.pre_push_all_metas(batch=0, q_len=total, kv_dim=0)
        graph.replay()
        want = []
        for layer in enabled:
            want += _expected(x, plan, "blocks.hook_resid_pre", layer)
        _compare(ring.drain(len(want)), want)
    assert ring.engine.toggle_device_errors() == 0


def _misaligned(rows, hidden, seed):
    """A contiguous [rows, hidden] fp32 view at storage offset 1 (4-byte, not
    16-byte, aligned base)."""
    gen = torch.Generator().manual_seed(seed)
    buf = torch.randn(1 + rows * hidden, generator=gen).cuda()
    view = buf[1:1 + rows * hidden].view(rows, hidden)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


def test_heads_score_tensors_with_misaligned_bases(ring):
    """hidden % 8 == 0 alone must not select 16-byte vector loads: a view at
    an odd storage offset takes the scalar path, with the same scores."""
    from dmi.hook_rows import full_row_plan
    from dmi.hooks.specs import (HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_PRE_HEAD,
                                 ModelShapeConfig)

    hid, K, rows = 64, 2, 5
    ht, hh = HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_PRE_HEAD
    ring.transport.set_model_cfg(ModelShapeConfig(
        hidden_dim=hid, num_heads=2, num_kv_heads=2, head_dim=hid // 2,
        dtype=torch.float32, head_slots=K))
    req_ids, ranges, offsets, pos = ["a"], [(0, rows)], [0], [0]
    plan = full_row_plan(req_ids, ranges, offsets, pos, rows)
    table = torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
    _write_table(table, plan)
    x = _misaligned(rows, hid, 1)
    rng = np.random.default_rng(3)
    w = rng.standard_normal((K, hid)).astype(np.float32)
    bias = np.array([0.25, -0.5], dtype=np.float32)
    want = x.double().cpu() @ torch.from_numpy(w).double().T + torch.from_numpy(bias).double()

    # Standalone head node: fp32 arena (weights, biases, split scratch), also
    # at a misaligned base.
    from dmi.transport import native as ne
    floats = K * hid + K + 16 * 8 * K
    arena_buf = torch.zeros(1 + floats, dtype=torch.float32, device="cuda")
    arena = arena_buf[1:]
    arena[:K * hid] = torch.from_numpy(w.reshape(-1)).cuda()
    arena[K * hid:K * hid + K] = torch.from_numpy(bias).cuda()
    ring.transport._active_specs = [_spec(hh, 0)]
    ring.transport.set_step_context(model_id="m", req_ids=req_ids, token_ranges=ranges,
                                    dim0_offsets=offsets, flattened=True,
                                    hook_rows={(hh, 0): plan})
    ring.transport.pre_push_all_metas(batch=0, q_len=rows, kv_dim=0)
    ring.engine.reserve_one(rows * K * 4)
    torch.ops.ring.producer_head(ring.payload, x, table, arena, K, hh, 0)
    got = ring.drain(1)
    assert len(got) == 1
    torch.testing.assert_close(got[0][5].double(), want, rtol=1e-5, atol=1e-5)

    # Fused node: rows and heads from the misaligned view.
    node = _fused_arena(w, bias, [1, 1], [0, 0])
    acc = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    head_table = table.clone()
    hook = _hook(ring, ht, 0, table, row_bytes=hid * 4)
    from dmi.hook_rows import configure_hook_heads
    configure_hook_heads(hook, head_type=hh, layer_no=0, table=head_table,
                         arena=node, acc=acc, slots=K, weight_bf16=False)
    ring.transport._active_specs = [_spec(ht, 0), _spec(hh, 0)]
    ring.transport.set_step_context(model_id="m", req_ids=req_ids, token_ranges=ranges,
                                    dim0_offsets=offsets, flattened=True,
                                    hook_rows={(ht, 0): plan, (hh, 0): plan})
    ring.transport.pre_push_all_metas(batch=0, q_len=rows, kv_dim=0)
    align = lambda n: (n + 15) & ~15
    ring.engine.prepare_step(align(rows * hid * 4) + align(rows * K * 4), 2)
    hook(x)
    got = {s[0]: s[5] for s in ring.drain(2)}
    torch.testing.assert_close(got["blocks.hook_resid_pre"], x.cpu())
    torch.testing.assert_close(got["blocks.hook_resid_pre_head"].double(), want,
                               rtol=1e-5, atol=1e-5)
    torch.cuda.synchronize()       # no misaligned-address fault poisoned the context


@pytest.mark.parametrize("op", ["gather", "head", "fused"])
@pytest.mark.parametrize("bad", ["int32", "strided", "host", "empty"])
def test_producers_refuse_malformed_row_tables(ring, op, bad):
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_PRE_HEAD

    x = torch.zeros(4, HID, device="cuda")
    good = torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
    table = {"int32": good.to(torch.int32), "strided": torch.zeros(
                 2 * (1 + 2 * MAX_SEGS), dtype=torch.int64, device="cuda")[::2],
             "host": good.cpu(), "empty": good[:0]}[bad]
    K = 2
    with pytest.raises(RuntimeError, match="row table"):
        if op == "gather":
            torch.ops.ring.producer_gather(ring.payload, x, table, HID * 4,
                                           HOOK_TYPE_RESID_PRE, 0)
        elif op == "head":
            arena = torch.zeros(K * HID + K + 16 * 8 * K, device="cuda")
            torch.ops.ring.producer_head(ring.payload, x, table, arena, K,
                                         HOOK_TYPE_RESID_PRE_HEAD, 0)
        else:
            node = torch.zeros((64 + K * HID + 3) & ~3, dtype=torch.int32, device="cuda")
            acc = torch.zeros(4, 16, device="cuda")
            torch.ops.ring.producer_fused(ring.payload, x, None, 0, table, node, acc,
                                          K, 0, HOOK_TYPE_RESID_PRE, 0,
                                          HOOK_TYPE_RESID_PRE_HEAD)


def test_head_arena_and_accumulator_must_share_the_tensor_device(ring):
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_PRE_HEAD

    K = 2
    x = torch.zeros(4, HID, device="cuda")
    table = torch.zeros(1 + 2 * MAX_SEGS, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="arena"):
        torch.ops.ring.producer_head(ring.payload, x, table,
                                     torch.zeros(K * HID + K + 16 * 8 * K), K,
                                     HOOK_TYPE_RESID_PRE_HEAD, 0)
    node = torch.zeros((64 + K * HID + 3) & ~3, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="acc"):
        torch.ops.ring.producer_fused(ring.payload, x, None, 0, table, node,
                                      torch.zeros(4, 16), K, 0,
                                      HOOK_TYPE_RESID_PRE, 0, HOOK_TYPE_RESID_PRE_HEAD)
