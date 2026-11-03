"""CPU tests for the host side of runtime CUDA-graph node toggling.

A fake engine stands in for the native ToggleRegistry so the controller's
guards, its effective-set recomputation and caches, the transport's per-step
spec seam, the adapter's reservation set and the HookPoint eager gate can be
checked without a GPU.  tests/test_node_toggle_gpu.py drives the same paths
against real captured graphs, and tests/native/ring/test_toggle_registry.cu
covers the native registry.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from dmi.hooks.specs import (
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_RESID_PRE,
    HookSpec,
    ModelShapeConfig,
)
from dmi.node_toggle import NodeToggleController, parse_enabled_hooks
from dmi.transport.ring import RingTransport

pytestmark = pytest.mark.cpu


class FakeToggleEngine:
    """The engine surface NodeToggleController uses, with a registry state."""

    def __init__(self, registered=(), *, bound=1, uniform=True, complete=True,
                 anomalies=0):
        self.registered = set(registered)
        self.bound = bound
        self.uniform = uniform
        self.complete = complete
        self.anomalies = anomalies
        self.version = 7
        self.enabled = set()
        self.active = False
        self.apply_calls = 0
        self.apply_result = 0
        self.mask_calls = []
        self.cleared = 0

    def toggle_registry_version(self):
        return self.version

    def bound_graph_count(self):
        return self.bound

    def toggle_node_count(self):
        return len(self.registered)

    def toggle_registry_uniform(self):
        return self.uniform

    def toggle_registry_complete(self):
        return self.complete

    def capture_anomaly_count(self):
        return self.anomalies

    def set_enabled_hooks(self, pairs):
        self.enabled = set(map(tuple, pairs))
        self.active = True

    def apply_toggle(self):
        self.apply_calls += 1
        return self.apply_result

    def effective_enabled_mask(self, query):
        self.mask_calls.append(list(query))
        return [int(not self.active
                    or (tuple(k) in self.enabled and tuple(k) in self.registered))
                for k in query]

    def is_hook_enabled(self, hook_type, layer_no):
        return not self.active or (
            (hook_type, layer_no) in self.enabled
            and (hook_type, layer_no) in self.registered)

    def clear_toggle_registry(self):
        self.cleared += 1
        self.active = False
        self.enabled = set()

    def is_graph_ready(self, raw):
        return raw == 1

    def ensure_graph_current(self, raw):
        return 0

    def record_replay_event(self, raw):
        return 0


def _specs(n=6):
    keys = [(HOOK_TYPE_RESID_PRE, layer) for layer in range(n)]
    keys.append((HOOK_TYPE_FINAL_LN, -1))
    return [HookSpec(hook_type=ht, module=None, layer_no=ln) for ht, ln in keys]


def _transport(engine, specs):
    transport = RingTransport.__new__(RingTransport)
    transport._ring_engine = engine
    transport._active_specs = specs
    transport._toggle = None
    transport._gated_step = True
    return transport


def _keys(specs):
    return [(s.hook_type, s.layer_no) for s in specs]


# ---------------------------------------------------------------------------
# Parsing


@pytest.mark.parametrize("value, expected", [
    (None, None),
    ("", None),
    ("  ", None),
    ("none", []),
    ("NONE", []),
    ("0:1, 0:2", [(0, 1), (0, 2)]),
    ("14", [(14, -1)]),
    ("3:4,,5", [(3, 4), (5, -1)]),
])
def test_parse_enabled_hooks(value, expected):
    assert parse_enabled_hooks(value) == expected


# ---------------------------------------------------------------------------
# Controller


def test_gate_is_inactive_until_activated_and_every_spec_fires():
    specs = _specs()
    transport = _transport(FakeToggleEngine(_keys(specs)), specs)
    assert transport.toggle_gate_active is False
    assert transport.effective_specs is specs
    assert transport.specs_for_step() is specs


def test_eager_activation_applies_and_selects_enabled_and_captured_in_order():
    specs = _specs()
    registered = set(_keys(specs)) - {(HOOK_TYPE_RESID_PRE, 3)}
    engine = FakeToggleEngine(registered)
    transport = _transport(engine, specs)
    transport.set_active_hooks([(HOOK_TYPE_FINAL_LN, -1),
                                (HOOK_TYPE_RESID_PRE, 3),   # not captured
                                (HOOK_TYPE_RESID_PRE, 1),
                                (HOOK_TYPE_RESID_PRE, 99)])  # not installed
    assert engine.apply_calls == 1
    assert transport.toggle_gate_active and not transport.toggle_lazy_active
    assert _keys(transport.effective_specs) == [
        (HOOK_TYPE_RESID_PRE, 1), (HOOK_TYPE_FINAL_LN, -1)]
    # Only enabled-and-installed pairs are queried.
    assert engine.mask_calls[-1] == [(HOOK_TYPE_RESID_PRE, 1),
                                     (HOOK_TYPE_RESID_PRE, 3),
                                     (HOOK_TYPE_FINAL_LN, -1)]


def test_lazy_activation_defers_the_device_apply():
    specs = _specs()
    engine = FakeToggleEngine(_keys(specs))
    transport = _transport(engine, specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 0)])
    assert engine.apply_calls == 0
    assert transport.toggle_gate_active and transport.toggle_lazy_active
    assert _keys(transport.effective_specs) == [(HOOK_TYPE_RESID_PRE, 0)]


def test_failed_eager_apply_raises_and_leaves_the_gate_inactive():
    specs = _specs()
    engine = FakeToggleEngine(_keys(specs))
    engine.apply_result = 400
    transport = _transport(engine, specs)
    with pytest.raises(RuntimeError, match="CUDA error 400"):
        transport.set_active_hooks([(HOOK_TYPE_RESID_PRE, 0)])
    assert transport.toggle_gate_active is False


@pytest.mark.parametrize("state, message", [
    (dict(bound=0), "no producer nodes registered or no graph exec bound"),
    (dict(uniform=False), "different hook sets"),
    (dict(complete=False), "registry incomplete"),
    (dict(anomalies=2), "cannot manage"),
])
def test_activation_fails_closed_on_an_unsafe_registry(state, message):
    specs = _specs()
    engine = FakeToggleEngine(_keys(specs), **state)
    transport = _transport(engine, specs)
    for activate in (transport.set_active_hooks, transport.set_active_hooks_lazy):
        with pytest.raises(RuntimeError, match=message):
            activate([(HOOK_TYPE_RESID_PRE, 0)])
    assert transport.toggle_gate_active is False


def test_activation_fails_closed_without_any_registered_node():
    specs = _specs()
    transport = _transport(FakeToggleEngine(()), specs)
    with pytest.raises(RuntimeError, match="toggle_node_count=0"):
        transport.set_active_hooks([(HOOK_TYPE_RESID_PRE, 0)])


def test_guard_verdict_is_cached_per_registry_version():
    specs = _specs()
    engine = FakeToggleEngine(_keys(specs))
    transport = _transport(engine, specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 0)])
    # Same registry version: a registry going bad is not re-checked...
    engine.uniform = False
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 1)])
    # ...but any mutation (a version bump) re-validates.
    engine.version += 1
    with pytest.raises(RuntimeError, match="different hook sets"):
        transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 2)])


def test_repeated_presets_reuse_the_effective_set():
    specs = _specs()
    engine = FakeToggleEngine(_keys(specs))
    transport = _transport(engine, specs)
    a = [(HOOK_TYPE_RESID_PRE, 0), (HOOK_TYPE_RESID_PRE, 4)]
    b = [(HOOK_TYPE_RESID_PRE, 2)]
    transport.set_active_hooks_lazy(a)
    first = transport.effective_specs
    transport.set_active_hooks_lazy(b)
    transport.set_active_hooks_lazy(list(reversed(a)))
    assert transport.effective_specs is first
    assert len(engine.mask_calls) == 2
    # A different active list (re-attach) is not served from the memo.
    transport._active_specs = list(specs)
    transport.set_active_hooks_lazy(a)
    assert len(engine.mask_calls) == 3
    assert _keys(transport.effective_specs) == _keys(first)


def test_randomized_reconfiguration_matches_the_reference_set():
    specs = _specs(32)
    keys = _keys(specs)
    registered = set(keys) - {keys[7]}
    engine = FakeToggleEngine(registered)
    transport = _transport(engine, specs)
    rng = random.Random(83)
    for step in range(200):
        enabled = set(rng.sample(keys, rng.randrange(0, 12)))
        transport.set_active_hooks_lazy(sorted(enabled))
        expected = [s for s in specs
                    if (s.hook_type, s.layer_no) in enabled & registered]
        assert transport.effective_specs == expected, step


def test_ungated_steps_use_the_full_active_set():
    specs = _specs()
    transport = _transport(FakeToggleEngine(_keys(specs)), specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 0)])
    transport.set_gated_step(False)
    assert transport.specs_for_step() is specs
    assert transport.gated_step_requires_registry() is False
    transport.set_gated_step(True)
    assert _keys(transport.specs_for_step()) == [(HOOK_TYPE_RESID_PRE, 0)]
    assert transport.gated_step_requires_registry() is True


def test_table_gate_makes_every_step_follow_the_toggle():
    specs = _specs()
    transport = _transport(FakeToggleEngine(_keys(specs)), specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 0)])
    transport.set_gated_step(False)
    transport.set_prefill_table_gate(True)
    assert _keys(transport.specs_for_step()) == [(HOOK_TYPE_RESID_PRE, 0)]
    assert transport.gated_step_requires_registry() is False
    transport.set_gated_step(True)
    assert transport.gated_step_requires_registry() is False
    transport.set_prefill_table_gate(False)
    assert transport.gated_step_requires_registry() is True


def test_clear_resets_registry_and_gate_together():
    specs = _specs()
    engine = FakeToggleEngine(_keys(specs))
    transport = _transport(engine, specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 0)])
    transport.clear_toggle()
    assert engine.cleared == 1
    assert transport.toggle_gate_active is False
    assert transport.toggle_lazy_active is False
    assert transport.effective_specs is specs


def test_minimal_transport_without_a_controller_is_inactive():
    transport = RingTransport.__new__(RingTransport)
    transport._active_specs = _specs()
    assert transport.toggle_gate_active is False
    assert transport.specs_for_step() is transport._active_specs


# ---------------------------------------------------------------------------
# Metadata, reservation and the eager gate use the per-step set


class _RecordingRingEngine(FakeToggleEngine):
    def __init__(self, registered):
        super().__init__(registered)
        self.pushed = []
        self.prepared = []

    def push_all_metas(self, hook_types, layer_nos, *args, **kwargs):
        self.pushed.append(list(zip(hook_types, layer_nos)))

    def prepare_step(self, total_bytes, n_hooks, reserve=True):
        self.prepared.append((total_bytes, n_hooks))
        return 0


def _cfg():
    import torch
    return ModelShapeConfig(hidden_dim=8, num_heads=2, num_kv_heads=2,
                            head_dim=4, dtype=torch.float32)


def _real_transport(engine, specs):
    transport = RingTransport.__new__(RingTransport)
    transport._ring_engine = engine
    transport._ring_payload = None
    transport._effective_cap = None
    transport.null_offload = False
    transport.force_eager = False
    transport.capture_step = True
    transport._model_cfg = _cfg()
    transport._active_specs = specs
    transport._using_forward_hooks = True
    transport._toggle = None
    transport._gated_step = True
    transport._hook_selection = None
    transport._warned_shapes = set()
    transport.set_step_context(model_id="m", req_ids=["r"],
                               token_ranges=[(0, 4)], flattened=True)
    return transport


def test_metadata_follows_the_per_step_set():
    specs = _specs()
    engine = _RecordingRingEngine(_keys(specs))
    transport = _real_transport(engine, specs)
    transport.pre_push_all_metas(batch=0, q_len=4, kv_dim=0)
    assert engine.pushed[-1] == _keys(specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 2),
                                     (HOOK_TYPE_FINAL_LN, -1)])
    transport.pre_push_all_metas(batch=0, q_len=4, kv_dim=0)
    assert engine.pushed[-1] == [(HOOK_TYPE_RESID_PRE, 2), (HOOK_TYPE_FINAL_LN, -1)]
    transport._gated_step = False
    transport.pre_push_all_metas(batch=0, q_len=4, kv_dim=0)
    assert engine.pushed[-1] == _keys(specs)


def test_adapter_reservation_counts_the_per_step_set():
    from dmi.adapters.base import BackendAdapter
    from dmi.adapters.types import StepContext

    specs = _specs()
    engine = _RecordingRingEngine(_keys(specs))
    transport = _real_transport(engine, specs)

    class Adapter(BackendAdapter):
        def detect_model_shape(self, model):
            raise NotImplementedError

        def detect_parallel_ranks(self):
            return (0, 0, 0, 0)

        def is_pp_first(self):
            return True

        def is_pp_last(self):
            return True

        def build_step_context(self, *raw):
            return StepContext(model_id="m", flattened=True, req_ids=["r"],
                               token_ranges=[(0, 4)], dim0_offsets=[0],
                               kv_offsets=[0], batch=0, q_len=4, kv_dim=0,
                               logits_to_keep=0)

        def on_capacity_exceeded(self, ctx):
            pass

    host = SimpleNamespace(_ring_transport=transport, _ring_engine=engine)
    adapter = Adapter(host, "m")
    adapter.model_cfg = _cfg()
    adapter.active_specs = specs
    ctx = adapter.build_step_context()
    full = adapter.plan_step(ctx)
    assert full.hook_count == len(specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 1)])
    gated = adapter.plan_step(ctx)
    assert gated.hook_count == 1
    assert gated.total_bytes == 4 * 8 * 4
    transport._gated_step = False
    assert adapter.plan_step(ctx) == full
    # Reservation and metadata agree on every step kind.
    for gated_step in (True, False):
        transport._gated_step = gated_step
        adapter.before_forward()
        assert engine.prepared[-1][1] == len(engine.pushed[-1])


def test_hook_point_eager_net_skips_a_disabled_hook_on_a_gated_step(monkeypatch):
    import torch

    from dmi.hooks import point as point_module
    from dmi.hooks.point import HookPoint
    from dmi.transport import ring as ring_module

    specs = _specs()
    engine = _RecordingRingEngine(_keys(specs))
    transport = _real_transport(engine, specs)
    transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 1)])
    transport.force_eager = True
    monkeypatch.setattr(ring_module, "_active_transport", transport)
    dispatched = []
    monkeypatch.setattr(point_module, "dispatch_producer",
                        lambda *args: dispatched.append(args[-1]))

    class CudaLike(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    hook = HookPoint()
    hook._ring_hook_type = HOOK_TYPE_RESID_PRE
    hook._ring_hook_id = 0          # disabled
    x = torch.zeros(2, 2).as_subclass(CudaLike)
    assert hook(x) is not None
    assert dispatched == []
    assert engine.prepared == []    # nothing reserved for the disabled hook


# ---------------------------------------------------------------------------
# Capture window lifecycle (fake engine and graph base class; no GPU)


class FakeWindowEngine:
    def __init__(self):
        self.capture = False
        self.device_mode = False
        self.device_calls = []
        self.cleared = 0
        self.uploads = []

    def clear_toggle_registry(self):
        self.cleared += 1
        self.capture = False
        self.device_mode = False

    def enable_device_toggle(self, enabled):
        self.device_calls.append(enabled)
        self.device_mode = enabled
        return enabled

    def enable_toggle_capture(self, enabled):
        self.capture = enabled

    def toggle_capture_enabled(self):
        return self.capture

    def toggle_device_mode(self):
        return self.device_mode

    def upload_graph_exec(self, graph, exec_ptr):
        self.uploads.append((graph, exec_ptr))
        return 0


def _reset_patch_state(monkeypatch, node_toggle):
    monkeypatch.setattr(node_toggle, "_PATCHED_CLASS", None)
    monkeypatch.setattr(node_toggle, "_ORIG_CUDAGRAPH", None)
    monkeypatch.setattr(node_toggle, "_TOGGLE_CLASS", None)
    monkeypatch.setattr(node_toggle, "_SESSIONS", {})
    monkeypatch.setattr(node_toggle, "_REPLAY_GUARD", False)


@pytest.fixture
def window(monkeypatch):
    """dmi.node_toggle with a fake torch.cuda.CUDAGraph base class."""
    import torch

    from dmi import node_toggle

    class FakeGraph:
        fail_begin = False

        def __init__(self, keep_graph=False):
            self.keep_graph = keep_graph

        def __new__(cls, keep_graph=False):
            return object.__new__(cls)

        def capture_begin(self, *args, **kwargs):
            if FakeGraph.fail_begin:
                raise RuntimeError("capture_begin failed")

        def capture_end(self):
            pass

        def raw_cuda_graph(self):
            return 0x10

        def raw_cuda_graph_exec(self):
            return 0x20

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    _reset_patch_state(monkeypatch, node_toggle)
    yield node_toggle, FakeGraph
    node_toggle._WINDOW.engine = None
    node_toggle._WINDOW.exclude_capture = None


def test_device_toggle_is_chosen_per_window(window):
    node_toggle, _ = window
    engine = FakeWindowEngine()
    node_toggle.begin_capture_window(engine, device_toggle=True)
    assert engine.toggle_device_mode() is True
    node_toggle.end_capture_window(engine)
    node_toggle.begin_capture_window(engine)
    assert engine.device_calls == [True, False]
    assert engine.toggle_device_mode() is False
    node_toggle.end_capture_window(engine)


def test_failed_capture_begin_restores_node_recording(window):
    node_toggle, FakeGraph = window
    engine = FakeWindowEngine()
    node_toggle.begin_capture_window(engine, exclude_capture=lambda: True)
    cls = node_toggle.install_cudagraph_patch()
    graph = cls()
    FakeGraph.fail_begin = True
    try:
        with pytest.raises(RuntimeError, match="capture_begin failed"):
            graph.capture_begin()
    finally:
        FakeGraph.fail_begin = False
    assert engine.toggle_capture_enabled() is True      # recording restored
    assert graph._dmi_toggle_suspended is False
    # A successful excluded capture suspends and restores around the capture.
    graph.capture_begin()
    assert engine.toggle_capture_enabled() is False
    graph.capture_end()
    assert engine.toggle_capture_enabled() is True
    node_toggle.end_capture_window(engine)


def test_capture_end_uploads_through_the_registry_guard(window):
    node_toggle, _ = window
    engine = FakeWindowEngine()
    node_toggle.begin_capture_window(engine, device_toggle=True)
    graph = node_toggle.install_cudagraph_patch()()
    graph.capture_begin()
    graph.capture_end()
    assert engine.uploads == [(0x10, 0x20)]   # the registry decides per graph
    node_toggle.end_capture_window(engine)
    node_toggle.begin_capture_window(engine)  # host-toggle window: no upload
    graph = node_toggle.install_cudagraph_patch()()
    graph.capture_begin()
    graph.capture_end()
    assert engine.uploads == [(0x10, 0x20)]
    node_toggle.end_capture_window(engine)


def test_capture_window_releases_the_engine_on_exception(window):
    node_toggle, _ = window
    engine = FakeWindowEngine()
    with pytest.raises(KeyError):
        with node_toggle.capture_window(engine, exclude_capture=lambda: False):
            assert node_toggle._WINDOW.engine is engine
            assert engine.toggle_capture_enabled()
            raise KeyError("boom")
    assert node_toggle._WINDOW.engine is None
    assert node_toggle._WINDOW.exclude_capture is None
    assert engine.toggle_capture_enabled() is False


def test_clear_and_deactivate_release_the_window_engine(window, monkeypatch):
    node_toggle, _ = window
    from dmi.transport import ring as ring_module

    engine = FakeWindowEngine()
    node_toggle.begin_capture_window(engine)
    controller = NodeToggleController(engine, SimpleNamespace(_active_specs=[]))
    controller.clear()
    assert node_toggle._WINDOW.engine is None

    node_toggle.begin_capture_window(engine)
    other = FakeWindowEngine()
    NodeToggleController(other, SimpleNamespace(_active_specs=[])).clear()
    assert node_toggle._WINDOW.engine is engine     # a different engine
    monkeypatch.setattr(ring_module, "_active_transport",
                        SimpleNamespace(_ring_engine=engine))
    ring_module.deactivate()
    assert node_toggle._WINDOW.engine is None


# ---------------------------------------------------------------------------
# Patch ownership, replay-guard scope and exec lifecycle (fake engine and a
# keep_graph-faithful graph base class; no GPU)


class LifecycleGraph:
    """torch.cuda.CUDAGraph's keep_graph contract: with keep_graph=True the
    exec does not exist until instantiate() or the first replay, and
    raw_cuda_graph_exec() raises until then."""

    next_raw = 0x100

    def __new__(cls, keep_graph=False):
        return object.__new__(cls)

    def __init__(self, keep_graph=False):
        self.keep_graph = keep_graph
        self.exec_ptr = None
        self.raw = LifecycleGraph.next_raw
        LifecycleGraph.next_raw += 1
        self.instantiations = 0
        self.replays = 0
        self.trace = []

    def capture_begin(self, *args, **kwargs):
        self.trace.append("capture_begin")

    def capture_end(self):
        if not self.keep_graph:
            self.instantiate()
        self.trace.append("capture_end")

    def instantiate(self):
        self.instantiations += 1
        self.exec_ptr = self.raw * 0x100 + self.instantiations
        self.trace.append("instantiate")

    def raw_cuda_graph(self):
        return self.raw

    def raw_cuda_graph_exec(self):
        if self.exec_ptr is None:
            raise RuntimeError("instantiate must be called when keep_graph=True")
        return self.exec_ptr

    def replay(self):
        if self.exec_ptr is None:
            self.instantiate()
        self.replays += 1
        self.trace.append("replay")


class LifecycleEngine(FakeWindowEngine):
    """Window + registry surface.  `produce()` stands in for a producer op:
    it bumps the process-wide launch counter and, while node recording is on,
    records a node for the graph being captured."""

    launches = 0

    def __init__(self, *, launch_counter=True):
        super().__init__()
        self.nodes = {}
        self.bound = {}
        self.capturing = None
        self.ensured = []
        if not launch_counter:
            self.producer_launch_count = None

    def produce(self, layer=0, *, counted=True):
        if counted:
            LifecycleEngine.launches += 1
        if self.capture and self.capturing is not None:
            self.nodes.setdefault(self.capturing.raw, []).append((0, layer))

    def producer_launch_count(self):
        return LifecycleEngine.launches

    def toggle_node_count(self):
        return sum(len(v) for v in self.nodes.values())

    def toggle_node_count_for_graph(self, raw):
        return len(self.nodes.get(raw, ()))

    def clear_toggle_registry(self):
        super().clear_toggle_registry()
        self.nodes.clear()
        self.bound.clear()

    def bind_graph_exec(self, raw, exec_ptr):
        self.bound[raw] = exec_ptr

    def is_graph_ready(self, raw):
        return raw in self.nodes and raw in self.bound


class GatedTransport:
    def __init__(self, engine, *, lazy=True, requires_registry=True):
        self._ring_engine = engine
        self.toggle_gate_active = True
        self.toggle_lazy_active = lazy
        self.requires_registry = requires_registry

    def gated_step_requires_registry(self):
        return self.requires_registry

    def is_graph_ready(self, raw):
        return self._ring_engine.is_graph_ready(raw)

    def ensure_graph_current(self, raw):
        self._ring_engine.ensured.append((raw, self._ring_engine.bound.get(raw)))
        return 0

    def record_replay_event(self, raw):
        return 0


@pytest.fixture
def lifecycle(monkeypatch):
    import torch

    from dmi import node_toggle
    from dmi.transport import ring as ring_module

    monkeypatch.setattr(torch.cuda, "CUDAGraph", LifecycleGraph)
    _reset_patch_state(monkeypatch, node_toggle)
    monkeypatch.setattr(ring_module, "_active_transport", None)

    def capture(engine, *, producers=0):
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        engine.capturing = graph
        for layer in range(producers):
            engine.produce(layer)
        engine.capturing = None
        graph.capture_end()
        return graph

    def serve(transport):
        monkeypatch.setattr(ring_module, "_active_transport", transport)

    yield node_toggle, torch, capture, serve
    node_toggle._WINDOW.engine = None
    node_toggle._WINDOW.exclude_capture = None


def _monitor(node_toggle, capture, engine, **window):
    node_toggle.begin_capture_window(engine, **window)
    graph = capture(engine, producers=2)
    node_toggle.bind_graph(engine, graph)
    node_toggle.end_capture_window(engine)
    node_toggle.set_replay_guard(True)
    return graph


def test_clearing_the_controller_restores_the_cudagraph_class(lifecycle):
    node_toggle, torch, capture, serve = lifecycle
    engine = LifecycleEngine()
    monitored = _monitor(node_toggle, capture, engine)
    assert torch.cuda.CUDAGraph is not LifecycleGraph     # session open
    assert torch.cuda.CUDAGraph(keep_graph=False).keep_graph is True
    NodeToggleController(engine, SimpleNamespace(_active_specs=[])).clear()
    assert torch.cuda.CUDAGraph is LifecycleGraph
    assert node_toggle._REPLAY_GUARD is False
    assert node_toggle.active_session_count() == 0
    assert torch.cuda.CUDAGraph(keep_graph=False).keep_graph is False
    # The graph created while patched still replays (its guard is inert).
    serve(GatedTransport(engine))
    monitored.replay()
    assert monitored.replays == 1


def test_an_unactivated_window_and_a_deactivated_transport_end_the_session(lifecycle):
    node_toggle, torch, capture, serve = lifecycle
    from dmi.transport import ring as ring_module

    engine = LifecycleEngine()
    node_toggle.begin_capture_window(engine)
    capture(engine)
    node_toggle.end_capture_window(engine)                # nothing bound
    assert torch.cuda.CUDAGraph is LifecycleGraph
    with node_toggle.capture_window(engine):
        capture(engine)
    assert torch.cuda.CUDAGraph is LifecycleGraph

    _monitor(node_toggle, capture, engine)
    assert torch.cuda.CUDAGraph is not LifecycleGraph
    serve(SimpleNamespace(_ring_engine=engine))
    ring_module.deactivate()
    assert torch.cuda.CUDAGraph is LifecycleGraph
    assert node_toggle._REPLAY_GUARD is False


def test_binding_after_the_window_closed_renews_the_session(lifecycle):
    node_toggle, torch, capture, _ = lifecycle
    engine = LifecycleEngine()
    node_toggle.begin_capture_window(engine)
    graph = capture(engine, producers=1)
    node_toggle.end_capture_window(engine)
    assert torch.cuda.CUDAGraph is LifecycleGraph
    node_toggle.bind_graph(engine, graph)
    assert torch.cuda.CUDAGraph is not LifecycleGraph
    assert node_toggle._session_for(engine) is graph._dmi_owner


def test_the_patch_stays_until_the_last_session_ends(lifecycle):
    node_toggle, torch, capture, _ = lifecycle
    first, second = LifecycleEngine(), LifecycleEngine()
    _monitor(node_toggle, capture, first)
    _monitor(node_toggle, capture, second)
    NodeToggleController(first, SimpleNamespace(_active_specs=[])).clear()
    assert torch.cuda.CUDAGraph is not LifecycleGraph
    assert node_toggle.active_session_count() == 1
    NodeToggleController(second, SimpleNamespace(_active_specs=[])).clear()
    assert torch.cuda.CUDAGraph is LifecycleGraph


def test_a_graph_without_producers_replays_unguarded(lifecycle):
    node_toggle, _, capture, serve = lifecycle
    engine = LifecycleEngine()
    node_toggle.begin_capture_window(engine)
    monitored = capture(engine, producers=2)
    unrelated = capture(engine)                 # a helper graph, no producer
    node_toggle.bind_graph(engine, monitored)
    node_toggle.end_capture_window(engine)
    after_window = capture(engine)              # captured later, no producer
    node_toggle.set_replay_guard(True)
    serve(GatedTransport(engine))
    monitored.replay()
    unrelated.replay()
    after_window.replay()
    assert (monitored.replays, unrelated.replays, after_window.replays) == (1, 1, 1)
    assert [raw for raw, _ in engine.ensured] == [monitored.raw]


@pytest.mark.parametrize("launch_counter", [True, False])
def test_an_unbound_producer_graph_still_fails_closed(lifecycle, launch_counter):
    node_toggle, _, capture, serve = lifecycle
    engine = LifecycleEngine(launch_counter=launch_counter)
    _monitor(node_toggle, capture, engine)
    # Captured after the window closed: producers, but no node and no bind.
    late = capture(engine, producers=1)
    serve(GatedTransport(engine))
    with pytest.raises(RuntimeError, match="not registered and bound"):
        late.replay()
    assert late.replays == 0


def test_an_unbound_graph_with_recorded_producers_fails_closed(lifecycle):
    node_toggle, _, capture, serve = lifecycle
    engine = LifecycleEngine(launch_counter=False)
    node_toggle.begin_capture_window(engine)
    monitored = capture(engine, producers=1)
    forgotten = capture(engine, producers=1)   # recorded, never bound
    node_toggle.bind_graph(engine, monitored)
    node_toggle.end_capture_window(engine)
    node_toggle.set_replay_guard(True)
    serve(GatedTransport(engine))
    with pytest.raises(RuntimeError, match="not registered and bound"):
        forgotten.replay()


def test_a_later_session_does_not_adopt_another_engines_graph(lifecycle):
    node_toggle, _, capture, serve = lifecycle
    old_engine, new_engine = LifecycleEngine(), LifecycleEngine()
    old_graph = _monitor(node_toggle, capture, old_engine)
    NodeToggleController(old_engine, SimpleNamespace(_active_specs=[])).clear()
    _monitor(node_toggle, capture, new_engine)
    serve(GatedTransport(new_engine))
    old_graph.replay()          # its producers write old_engine's ring
    assert old_graph.replays == 1
    assert new_engine.ensured == []


def test_a_stale_graph_of_the_same_engine_fails_closed(lifecycle):
    node_toggle, _, capture, serve = lifecycle
    engine = LifecycleEngine()
    stale = _monitor(node_toggle, capture, engine)
    _monitor(node_toggle, capture, engine)      # recapture: registry cleared
    serve(GatedTransport(engine))
    with pytest.raises(RuntimeError, match="not registered and bound"):
        stale.replay()


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_a_reinstantiated_bound_graph_is_rebound_before_it_replays(lifecycle, lazy):
    node_toggle, _, capture, serve = lifecycle
    engine = LifecycleEngine()
    graph = _monitor(node_toggle, capture, engine)
    first_exec = engine.bound[graph.raw]
    serve(GatedTransport(engine, lazy=lazy))
    graph.replay()
    engine.ensured.clear()
    graph.instantiate()                          # a new exec, same template
    replays = []
    graph.trace.clear()
    real_replay = LifecycleGraph.replay

    def replay_after_ensure(self):
        # The new exec must already carry the enabled set when it launches.
        replays.append(list(engine.ensured))
        real_replay(self)

    LifecycleGraph.replay = replay_after_ensure
    try:
        graph.replay()
    finally:
        LifecycleGraph.replay = real_replay
    assert engine.bound[graph.raw] == graph.exec_ptr != first_exec
    # Eager mode too: the rebound exec is brought current before its first
    # replay, exactly once.
    assert replays == [[(graph.raw, graph.exec_ptr)]]
    engine.ensured.clear()
    graph.replay()                               # same exec again
    assert engine.ensured == ([(graph.raw, graph.exec_ptr)] if lazy else [])


def test_eager_replays_of_an_unchanged_exec_do_not_reapply(lifecycle):
    node_toggle, _, capture, serve = lifecycle
    engine = LifecycleEngine()
    graph = _monitor(node_toggle, capture, engine)
    serve(GatedTransport(engine, lazy=False))
    graph.replay()
    graph.replay()
    assert graph.replays == 2 and engine.ensured == []


def test_recorded_producer_nodes_win_over_an_unchanged_launch_counter(lifecycle):
    # A producer op that recorded a node but did not bump the launch counter
    # must still classify the capture as carrying producers: an unbound
    # capture fails closed instead of replaying around the gate.
    node_toggle, torch, capture, serve = lifecycle
    engine = LifecycleEngine()
    node_toggle.begin_capture_window(engine)
    monitored = capture(engine, producers=1)
    forgotten = torch.cuda.CUDAGraph()
    forgotten.capture_begin()
    engine.capturing = forgotten
    engine.produce(0, counted=False)
    engine.capturing = None
    forgotten.capture_end()
    assert forgotten._dmi_producers is True
    node_toggle.bind_graph(engine, monitored)
    node_toggle.end_capture_window(engine)
    node_toggle.set_replay_guard(True)
    serve(GatedTransport(engine))
    with pytest.raises(RuntimeError, match="not registered and bound"):
        forgotten.replay()
    assert forgotten.replays == 0


@pytest.mark.parametrize("launch_counter", [True, False])
def test_excluded_device_toggle_capture_uploads_before_first_replay(
        lifecycle, launch_counter):
    node_toggle, _, capture, serve = lifecycle
    engine = LifecycleEngine(launch_counter=launch_counter)
    node_toggle.begin_capture_window(engine, exclude_capture=lambda: True,
                                     device_toggle=True)
    graph = capture(engine, producers=1)
    assert graph.keep_graph is True
    # Instantiated and uploaded at capture end, before any launch.
    assert graph.trace == ["capture_begin", "capture_end", "instantiate"]
    assert engine.uploads == [(graph.raw, graph.exec_ptr)]
    assert engine.nodes == {}                     # never enrolled
    with pytest.raises(RuntimeError, match="excluded"):
        node_toggle.bind_graph(engine, graph)
    node_toggle.end_capture_window(engine)
    node_toggle.set_replay_guard(True)
    serve(GatedTransport(engine, requires_registry=False))
    graph.replay()
    graph.replay()
    assert engine.uploads == [(graph.raw, graph.exec_ptr)]    # once per exec
    graph.instantiate()                                       # new exec
    graph.replay()
    assert engine.uploads[-1] == (graph.raw, graph.exec_ptr)
    assert graph.trace[-2:] == ["instantiate", "replay"]
    assert len(engine.uploads) == 2 and graph.replays == 3


def test_device_toggle_capture_without_producers_is_not_instantiated(lifecycle):
    node_toggle, _, capture, _ = lifecycle
    engine = LifecycleEngine()
    node_toggle.begin_capture_window(engine, exclude_capture=lambda: True,
                                     device_toggle=True)
    graph = capture(engine)
    node_toggle.end_capture_window(engine)
    assert graph.exec_ptr is None and engine.uploads == []
    graph.replay()
    assert engine.uploads == []
