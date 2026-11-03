"""Runtime CUDA-graph node toggling on a real legacy ring.

Each test captures ``torch.ops.ring.producer`` launches into
``torch.cuda.CUDAGraph`` graphs, binds them to the engine's toggle registry,
and checks what an in-process consumer receives after replays.  Every source
tensor is filled with a marker (graph offset + layer), so a delivery shows
which graph's node produced it and whether payload and metadata stayed
paired: a stale or misapplied toggle shows up as a missing, extra or
misattributed slice.

Needs CUDA and the full native backend.  Memory use is a few MiB.
"""
from __future__ import annotations

import time

import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]

QLEN, HID = 4, 8


class Ring:
    """A legacy ring engine + transport with an in-process consumer."""

    def __init__(self, n_hooks=4, payload=4 << 20, task_entries=1024):
        from dmi.hooks.specs import (HOOK_TYPE_RESID_PRE, HookSpec,
                                     ModelShapeConfig)
        from dmi.transport import native as ne
        from dmi.transport import ring as rt

        self.ht = HOOK_TYPE_RESID_PRE
        self.collected = []
        cfg = ne.RingConfig()
        cfg.payload_ring_bytes = payload
        cfg.pinned_staging_bytes = payload
        cfg.task_ring_entries = task_entries
        cfg.drain_poll_timeout_us = 200

        def consume(model_id, shard_rank, req_id, act_name, layer_no,
                    start, end, tensor):
            self.collected.append(
                (int(layer_no), float(tensor.reshape(-1)[0].item())))

        self.engine = ne.RingEngine(cfg, consume)
        self.engine.init(0)
        self.engine.start()
        self.transport = rt.RingTransport(self.engine)
        rt.activate(self.transport)
        self.transport.set_model_cfg(ModelShapeConfig(
            hidden_dim=HID, num_heads=2, num_kv_heads=2, head_dim=HID // 2,
            dtype=torch.float32))
        self.transport._active_specs = [
            HookSpec(hook_type=self.ht, module=None, layer_no=j,
                     dim0_is_actual_tokens=True)
            for j in range(n_hooks)]
        self.transport.set_step_context(
            model_id="m", req_ids=["r"], token_ranges=[(0, QLEN)],
            dim0_offsets=[0], kv_offsets=[0], flattened=True)
        self.payload = self.engine.payload_tensor()
        self.n = n_hooks
        self.keep_alive = []

    def sources(self, offset):
        return [torch.full((QLEN, HID), offset + j, device="cuda",
                           dtype=torch.float32) for j in range(self.n)]

    def produce(self, sources, layers=None):
        for j, src in enumerate(sources):
            if layers is None or j in layers:
                torch.ops.ring.producer(self.payload, src, self.ht, j)

    def capture(self, sources, graph=None):
        """Capture one producer per hook; null mode keeps any eager work out
        of the ring."""
        self.engine.set_null_mode(True)
        graph = graph if graph is not None else torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            self.produce(sources)
        graph.instantiate()
        self.keep_alive.append(sources)   # the graph reads these addresses
        torch.cuda.synchronize()
        self.engine.set_null_mode(False)
        return graph

    def bind(self, graph):
        self.engine.bind_graph_exec(graph.raw_cuda_graph(),
                                    graph.raw_cuda_graph_exec())

    def drain(self, expected):
        torch.cuda.synchronize()
        self.engine.flush_and_wait()
        deadline = time.monotonic() + 5.0
        while len(self.collected) < expected and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.02)
        got = sorted(self.collected)
        self.collected.clear()
        return got

    def close(self):
        from dmi.transport import ring as rt
        torch.cuda.synchronize()
        self.engine.flush_and_wait()
        rt.deactivate()
        self.engine.stop()


def _keys(ring, layers):
    return [(ring.ht, j) for j in layers]


def _expect(layers, offset):
    return sorted((j, float(offset + j)) for j in layers)


@pytest.fixture
def ring():
    r = Ring()
    try:
        yield r
    finally:
        r.close()


def test_eager_toggle_delivers_exactly_the_enabled_hooks(ring):
    ring.engine.enable_toggle_capture(True)
    graph = ring.capture(ring.sources(0.0))
    ring.engine.enable_toggle_capture(False)
    ring.bind(graph)
    assert ring.engine.toggle_node_count() == ring.n
    assert ring.engine.toggle_registry_complete()

    # Gate inactive: every hook fires.
    ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
    graph.replay()
    assert ring.drain(ring.n) == _expect(range(ring.n), 0.0)

    for enabled in ([1, 3], [0], [], [0, 1, 2, 3], [2]):
        ring.transport.set_active_hooks(_keys(ring, enabled))
        ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        graph.replay()
        assert ring.drain(len(enabled)) == _expect(enabled, 0.0), enabled
    assert ring.engine.capture_anomaly_count() == 0


def test_lazy_reconfiguration_applies_per_graph_before_replay(ring):
    ring.engine.enable_toggle_capture(True)
    graphs = {"A": (ring.capture(ring.sources(0.0)), 0.0),
              "B": (ring.capture(ring.sources(100.0)), 100.0)}
    ring.engine.enable_toggle_capture(False)
    for graph, _ in graphs.values():
        ring.bind(graph)
    assert ring.engine.toggle_registry_uniform()

    current = {}

    def reconfigure(enabled):
        current["enabled"] = sorted(enabled)
        ring.transport.set_active_hooks_lazy(_keys(ring, enabled))

    def replay(name):
        graph, offset = graphs[name]
        raw = graph.raw_cuda_graph()
        ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        assert ring.transport.ensure_graph_current(raw) == 0
        assert ring.engine._test_applied_current(raw)
        graph.replay()
        assert ring.transport.record_replay_event(raw) == 0
        assert ring.drain(len(current["enabled"])) == _expect(
            current["enabled"], offset), (name, current["enabled"])

    # First-time apply, one version stale, two versions stale, empty.
    reconfigure([0, 2]); replay("B"); replay("B"); replay("A")
    reconfigure([1, 3]); replay("A"); replay("B")
    reconfigure([0, 1, 2, 3]); reconfigure([2]); replay("A"); replay("B")
    reconfigure([]); replay("A"); replay("B")


def test_pre_apply_sync_mode_waits_for_the_previous_replay(ring):
    ring.engine.set_toggle_pre_apply_sync(True)
    ring.engine.enable_toggle_capture(True)
    graph = ring.capture(ring.sources(0.0))
    ring.engine.enable_toggle_capture(False)
    ring.bind(graph)
    raw = graph.raw_cuda_graph()
    for enabled in ([0, 1], [3], [1, 2, 3]):
        ring.transport.set_active_hooks_lazy(_keys(ring, enabled))
        ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        assert ring.transport.ensure_graph_current(raw) == 0
        graph.replay()
        assert ring.transport.record_replay_event(raw) == 0
        assert ring.drain(len(enabled)) == _expect(enabled, 0.0)


def test_reservation_matches_what_fires_on_gated_and_ungated_steps():
    """Plan each step from specs_for_step(), as BackendAdapter.plan_step
    does, over a long sequence of gated replays and ungated eager steps with
    a small ring: if reservation and production disagree, the CPU accounting
    drifts and the ring never drains back to empty."""
    from dmi.hooks.specs import align_up_py

    r = Ring(payload=64 * 1024, task_entries=16)
    try:
        r.engine.enable_toggle_capture(True)
        sources = r.sources(0.0)
        graph = r.capture(sources)
        r.engine.enable_toggle_capture(False)
        r.bind(graph)
        delivered = 0
        expected = 0
        presets = [[0, 1], [3], [0, 1, 2, 3], [], [2, 3]]
        for step in range(60):
            if step % 3 == 0:
                r.transport.set_active_hooks_lazy(
                    _keys(r, presets[(step // 3) % len(presets)]))
            gated = step % 4 != 3
            r.transport._gated_step = gated
            specs = r.transport.specs_for_step()
            total = sum(align_up_py(QLEN * HID * 4, 16) for _ in specs)
            if specs:
                result = r.engine.prepare_step(total, len(specs))
                assert result in (0, 1), result
            r.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
            if gated:
                assert r.transport.ensure_graph_current(graph.raw_cuda_graph()) == 0
                graph.replay()
            else:
                r.produce(sources)   # eager: every hook fires
            expected += len(specs)
            if step % 7 == 6:
                delivered += len(r.drain(0))
        delivered += len(r.drain(expected - delivered))
        assert delivered == expected
        assert r.engine.available_capacity() == r.engine.payload_cap()
        assert r.engine.available_task_slots() == r.engine.task_cap()
    finally:
        r.close()


def test_capacity_protection_engages_after_unreserved_warmup_producers():
    """Warmup runs of captured producers publish without a reservation, so
    the drain's tails end up ahead of the reservation heads.  Reservations
    must start from the tails: otherwise, for as many tasks as the skew, the
    room check reads a full ring as empty and producers overwrite unread
    slots."""
    from dmi.hooks.specs import align_up_py

    r = Ring(n_hooks=1, payload=64 * 1024, task_entries=8)
    try:
        sources = r.sources(0.0)
        for _ in range(3):                   # 21 unreserved tasks, drained
            for _ in range(7):
                r.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
                r.produce(sources)
            assert len(r.drain(7)) == 7
        assert r.engine.available_task_slots() == r.engine.task_cap()
        step = align_up_py(QLEN * HID * 4, 16)
        results = [r.engine.prepare_step(step, 1) for _ in range(9)]
        # Eight reservations fill the 8-entry task ring; the ninth must flush.
        assert results[:8] == [0] * 8
        assert results[8] == 1
        assert r.engine.available_task_slots() == 0
    finally:
        r.close()


def test_activation_refuses_unsafe_registries(ring):
    # Nothing captured or bound.
    with pytest.raises(RuntimeError, match="no producer nodes registered"):
        ring.transport.set_active_hooks(_keys(ring, [0]))

    ring.engine.enable_toggle_capture(True)
    graph = ring.capture(ring.sources(0.0))
    partial = torch.cuda.CUDAGraph(keep_graph=True)
    partial_sources = ring.sources(50.0)
    ring.engine.set_null_mode(True)
    with torch.cuda.graph(partial):
        ring.produce(partial_sources, layers={0, 1})
    partial.instantiate()
    ring.engine.set_null_mode(False)
    ring.engine.enable_toggle_capture(False)
    ring.bind(graph)
    # `partial` holds nodes but is unbound, and registered a different set.
    assert not ring.engine.is_graph_ready(partial.raw_cuda_graph())
    assert ring.engine.is_graph_ready(graph.raw_cuda_graph())
    assert not ring.engine.toggle_registry_complete()
    with pytest.raises(RuntimeError, match="different hook sets"):
        ring.transport.set_active_hooks_lazy(_keys(ring, [0]))
    ring.bind(partial)
    with pytest.raises(RuntimeError, match="different hook sets"):
        ring.transport.set_active_hooks_lazy(_keys(ring, [0]))
    assert ring.transport.toggle_gate_active is False


def test_chunked_producer_captured_in_the_window_fails_closed(ring):
    counts = torch.full((2,), QLEN * HID * 2, dtype=torch.int64, device="cuda")
    src = torch.zeros(2 * QLEN, HID, device="cuda")
    ring.engine.enable_toggle_capture(True)
    graph = ring.capture(ring.sources(0.0))
    ring.engine.set_null_mode(True)
    chunked = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(chunked):
        torch.ops.ring.producer_chunked(ring.payload, src, counts, ring.ht, 9)
    ring.engine.set_null_mode(False)
    ring.engine.enable_toggle_capture(False)
    ring.bind(graph)
    assert ring.engine.capture_anomaly_count() == 1
    with pytest.raises(RuntimeError, match="cannot manage"):
        ring.transport.set_active_hooks(_keys(ring, [0]))
    del chunked


def test_failed_lazy_apply_is_reported_and_retried(ring):
    ring.engine.enable_toggle_capture(True)
    graph = ring.capture(ring.sources(0.0))
    ring.engine.enable_toggle_capture(False)
    ring.bind(graph)
    raw = graph.raw_cuda_graph()
    ring.transport.set_active_hooks_lazy(_keys(ring, [2]))
    ring.engine._test_force_apply_error(999)
    assert ring.transport.ensure_graph_current(raw) == 999
    assert not ring.engine._test_applied_current(raw)
    ring.engine._test_force_apply_error(0)
    assert ring.transport.ensure_graph_current(raw) == 0
    ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
    graph.replay()
    assert ring.drain(1) == _expect([2], 0.0)


def test_clear_re_enables_nodes_and_deactivates_the_gate(ring):
    ring.engine.enable_toggle_capture(True)
    graph = ring.capture(ring.sources(0.0))
    ring.engine.enable_toggle_capture(False)
    ring.bind(graph)
    ring.transport.set_active_hooks(_keys(ring, [1]))
    ring.transport.clear_toggle()
    assert not ring.transport.toggle_gate_active
    assert ring.engine.toggle_node_count() == 0
    # The graph keeps replaying after the clear with every producer on.
    ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
    graph.replay()
    assert ring.drain(ring.n) == _expect(range(ring.n), 0.0)


def test_eager_safety_net_skips_disabled_hooks_on_gated_steps(ring):
    from dmi.hooks.point import HookPoint

    ring.engine.enable_toggle_capture(True)
    graph = ring.capture(ring.sources(0.0))
    ring.engine.enable_toggle_capture(False)
    ring.bind(graph)
    ring.transport.set_active_hooks(_keys(ring, [0, 3]))
    hooks = []
    for j in range(ring.n):
        hook = HookPoint()
        hook._ring_hook_type = ring.ht
        hook._ring_hook_id = j
        hook._ring_payload = ring.payload
        hooks.append(hook)
    sources = ring.sources(200.0)
    ring.transport.force_eager = True
    try:
        # Gated step that fell back to eager: only the enabled hooks fire.
        ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        for hook, src in zip(hooks, sources):
            hook(src)
        assert ring.drain(2) == _expect([0, 3], 200.0)
        # Ungated step: the full active set fires.
        ring.transport._gated_step = False
        ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        for hook, src in zip(hooks, sources):
            hook(src)
        assert ring.drain(ring.n) == _expect(range(ring.n), 200.0)
    finally:
        ring.transport.force_eager = False
        ring.transport._gated_step = True


@pytest.fixture
def patched_cudagraph(monkeypatch):
    from dmi import node_toggle

    monkeypatch.setattr(torch.cuda, "CUDAGraph", torch.cuda.CUDAGraph)
    monkeypatch.setattr(node_toggle, "_PATCHED_CLASS", None)
    monkeypatch.setattr(node_toggle, "_ORIG_CUDAGRAPH", None)
    monkeypatch.setattr(node_toggle, "_TOGGLE_CLASS", None)
    monkeypatch.setattr(node_toggle, "_SESSIONS", {})
    monkeypatch.setattr(node_toggle, "_REPLAY_GUARD", False)
    yield node_toggle
    if node_toggle._WINDOW.engine is not None:
        node_toggle.end_capture_window(node_toggle._WINDOW.engine)


def test_capture_window_excludes_partial_captures_and_guards_replays(
        ring, patched_cudagraph):
    """A capture the integration excludes (e.g. a partial-model segment
    graph) records no nodes; the guarded replay allows it only on an
    ungated step, refuses it on a gated one, and refuses any graph captured
    after the window closed."""
    node_toggle = patched_cudagraph
    exclude = {"now": False}
    node_toggle.begin_capture_window(ring.engine,
                                     exclude_capture=lambda: exclude["now"])
    cls = torch.cuda.CUDAGraph
    assert getattr(cls, "_dmi_keep_graph", False)

    full = ring.capture(ring.sources(0.0), graph=cls())
    exclude["now"] = True
    segment = cls()
    segment_sources = ring.sources(300.0)
    ring.engine.set_null_mode(True)
    with torch.cuda.graph(segment):
        ring.produce(segment_sources, layers={1, 2})
    segment.instantiate()
    ring.engine.set_null_mode(False)
    exclude["now"] = False
    assert segment._dmi_toggle_excluded and not full._dmi_toggle_excluded
    assert node_toggle.excluded_capture_count() == 1
    assert ring.engine.toggle_capture_enabled()   # restored after the segment

    node_toggle.bind_graph(ring.engine, full)
    with pytest.raises(RuntimeError, match="excluded"):
        node_toggle.bind_graph(ring.engine, segment)
    node_toggle.end_capture_window(ring.engine)
    assert ring.engine.toggle_node_count() == ring.n
    assert ring.engine.toggle_registry_complete()

    late = ring.capture(ring.sources(400.0), graph=cls())   # window closed

    node_toggle.activate(ring.transport, _keys(ring, [0, 2]), lazy=True)
    # Gated step: the bound full graph replays with the lazy apply.
    ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
    full.replay()
    assert ring.drain(2) == _expect([0, 2], 0.0)
    # The excluded segment on a gated step would desynchronize the ring.
    with pytest.raises(RuntimeError, match="excluded from the toggle registry"):
        segment.replay()
    # On an ungated step the full set's metadata is published and the
    # excluded segment's all-on producers match it.
    ring.transport._gated_step = False
    ring.transport._active_specs, saved = (
        [s for s in ring.transport._active_specs if s.layer_no in (1, 2)],
        ring.transport._active_specs)
    try:
        ring.transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
        segment.replay()
        assert ring.drain(2) == _expect([1, 2], 300.0)
    finally:
        ring.transport._active_specs = saved
        ring.transport._gated_step = True
    # A graph captured after the window closed is refused before it runs.
    with pytest.raises(RuntimeError, match="not registered and bound"):
        late.replay()
    node_toggle.set_replay_guard(False)


def test_bind_refuses_graphs_of_the_unpatched_class(ring):
    from dmi import node_toggle

    class UnpatchedGraph:   # what torch.cuda.CUDAGraph() is without the patch
        pass

    with pytest.raises(RuntimeError, match="unpatched"):
        node_toggle.bind_graph(ring.engine, UnpatchedGraph())


def test_activate_refuses_a_requested_set_without_bound_graphs(ring):
    from dmi import node_toggle

    node_toggle.activate(ring.transport, None, lazy=False)   # inactive: fine
    with pytest.raises(RuntimeError, match="silently inert"):
        node_toggle.activate(ring.transport, _keys(ring, [0]), lazy=False)


def test_python_callable_consumers_are_validated():
    from dmi.transport import native as ne

    with pytest.raises(TypeError, match="callable consumer"):
        ne.RingEngine(ne.RingConfig(), 42)


def test_toggle_capture_is_refused_on_a_record_ring():
    from dmi.transport import native as ne

    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 1 << 20
    cfg.pinned_staging_bytes = 1 << 20
    try:
        engine = ne.RingEngine.create_record(cfg, None, "raise", 0)
    except Exception as exc:   # pragma: no cover - record runtime unavailable
        pytest.skip(f"record ring unavailable: {exc}")
    with pytest.raises(RuntimeError, match="record ring"):
        engine.enable_toggle_capture(True)
