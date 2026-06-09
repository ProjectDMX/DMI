"""Unit tests for BackendAdaptor.before_forward call ordering.

Uses fakes for MonitoringEngine, RingTransport, and RingEngine to verify
the driver flow:

    build_step_context -> _compute_step_plan -> [prepare_step
        -> adapt_for_cpu_direct (if result==2)
        -> on_capacity_exceeded (if result==2)
        -> _warn_once_capacity (if result==2)]
    set transport.force_eager from (result == 2) OR needs_eager.
    -> set_step_context -> pre_push_all_metas

No GPU / native engine required.
"""
from __future__ import annotations

import dataclasses

import pytest

from monitoring.adaptor_base import BackendAdaptor
from monitoring.step_context import StepContext

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTransport:
    def __init__(self) -> None:
        self.null_offload = False
        self.force_eager = False
        self.set_step_context_calls: list = []
        self.pre_push_all_metas_calls: list = []
        self._active_specs: list = []
        self._using_forward_hooks = False
        self._model_cfg = None

    def set_step_context(self, **kwargs):
        self.set_step_context_calls.append(kwargs)

    def pre_push_all_metas(self, **kwargs):
        self.pre_push_all_metas_calls.append(kwargs)

    def set_model_cfg(self, cfg):
        self._model_cfg = cfg


class FakeRingEngine:
    def __init__(self, prepare_step_result: int = 0) -> None:
        self._result = prepare_step_result
        self.prepare_step_calls: list = []

    def prepare_step(self, total_bytes: int, n_hooks: int) -> int:
        self.prepare_step_calls.append((total_bytes, n_hooks))
        return self._result


class FakeEngine:
    def __init__(self, prepare_step_result: int = 0) -> None:
        self._ring_transport = FakeTransport()
        self._ring_engine = FakeRingEngine(prepare_step_result)


class StubAdaptor(BackendAdaptor):
    """Concrete BackendAdaptor with fixed StepContext + recordable callbacks."""

    def __init__(self, engine, model_id, ctx, step_plan=(1024, 3, False)):
        super().__init__(engine, model_id)
        self._ctx = ctx
        self._step_plan_value = step_plan
        self.adapt_for_cpu_direct_calls: list = []
        self.on_capacity_exceeded_calls: list = []
        self.warn_calls: list = []
        self.call_order: list = []

    def detect_model_shape(self, model):
        raise NotImplementedError

    def detect_parallel_ranks(self):
        return (0, 0, 0, 0)

    def is_pp_first(self):
        return True

    def is_pp_last(self):
        return True

    def build_step_context(self, *raw):
        self.call_order.append("build_step_context")
        return self._ctx

    def on_capacity_exceeded(self, ctx):
        self.call_order.append("on_capacity_exceeded")
        self.on_capacity_exceeded_calls.append(ctx)

    def adapt_for_cpu_direct(self, ctx):
        self.call_order.append("adapt_for_cpu_direct")
        self.adapt_for_cpu_direct_calls.append(ctx)
        return dataclasses.replace(ctx, q_len=ctx.q_len + 100)

    def _warn_once_capacity(self, ctx, total_bytes, n_hooks):
        self.call_order.append("_warn_once_capacity")
        self.warn_calls.append((ctx, total_bytes, n_hooks))

    def _compute_step_plan(self, ctx):
        self.call_order.append("_compute_step_plan")
        return self._step_plan_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> StepContext:
    return StepContext(
        model_id="test_model",
        flattened=False,
        req_ids=["0:0", "0:1"],
        token_ranges=[(0, 4), (0, 4)],
        dim0_offsets=[0, 1],
        kv_offsets=[0, 0],
        batch=2, q_len=4, kv_dim=4,
        logits_to_keep=0,
    )


def _make_adaptor(prepare_result, ctx_override=..., step_plan=(1024, 3, False)):
    engine = FakeEngine(prepare_step_result=prepare_result)
    ctx = _make_ctx() if ctx_override is ... else ctx_override
    return StubAdaptor(engine, "test_model", ctx, step_plan=step_plan)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_result_zero():
    """prepare_step -> 0: no capacity hooks fire; force_eager stays False."""
    a = _make_adaptor(prepare_result=0)
    a.before_forward(None)

    assert a.call_order == ["build_step_context", "_compute_step_plan"]
    assert a.transport.force_eager is False
    assert len(a.engine._ring_engine.prepare_step_calls) == 1
    assert a.engine._ring_engine.prepare_step_calls[0] == (1024, 3)
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


def test_flushed_result_one():
    """prepare_step -> 1 (RING_FLUSHED): same as 0 from the adapter's view."""
    a = _make_adaptor(prepare_result=1)
    a.before_forward(None)

    assert a.transport.force_eager is False
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


def test_capacity_exceeded_result_two():
    """prepare_step -> 2: adapt_for_cpu_direct + on_capacity_exceeded +
    _warn_once_capacity fire in order; force_eager True; rest of path runs."""
    a = _make_adaptor(prepare_result=2)
    a.before_forward(None)

    assert a.call_order == [
        "build_step_context", "_compute_step_plan",
        "adapt_for_cpu_direct", "on_capacity_exceeded", "_warn_once_capacity",
    ]
    assert a.transport.force_eager is True
    assert len(a.adapt_for_cpu_direct_calls) == 1
    assert len(a.on_capacity_exceeded_calls) == 1
    # on_capacity_exceeded receives the post-adapt ctx (StubAdaptor.adapt_for_cpu_direct
    # bumps q_len by 100).
    assert a.on_capacity_exceeded_calls[0].q_len == 4 + 100
    assert a.warn_calls[0][1:] == (1024, 3)
    # set_step_context still runs after the capacity branch -- uses the
    # adapted ctx, so q_len in the kwargs reflects the bump.
    assert len(a.transport.set_step_context_calls) == 1
    pushed_meta = a.transport.pre_push_all_metas_calls[0]
    assert pushed_meta["q_len"] == 4 + 100


def test_needs_eager_from_plan_sets_force_eager():
    """When _compute_step_plan returns needs_eager=True (dynamic-shape
    spec in active selection), force_eager is True even when
    prepare_step returns 0 (no overflow)."""
    a = _make_adaptor(prepare_result=0, step_plan=(1024, 3, True))
    a.before_forward(None)

    assert a.transport.force_eager is True
    # No overflow, so the code-2 branch doesn't fire.
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []


def test_force_eager_cleared_on_normal_step_after_overflow():
    """force_eager is per-batch.  After an overflow step sets it True,
    a follow-up normal step must reassign to False -- no leak."""
    a = _make_adaptor(prepare_result=2)
    a.before_forward(None)
    assert a.transport.force_eager is True

    # Swap the fake engine to return 0 (normal) and re-run.
    a.engine._ring_engine._result = 0
    a.before_forward(None)
    assert a.transport.force_eager is False


def test_n_hooks_zero_skips_prepare_step():
    """When _compute_step_plan returns (0, 0, False), prepare_step is
    skipped but set_step_context + pre_push_all_metas still run."""
    a = _make_adaptor(prepare_result=2, step_plan=(0, 0, False))
    a.before_forward(None)

    assert a.engine._ring_engine.prepare_step_calls == []
    assert a.transport.force_eager is False
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


def test_null_context_skips_everything_after_build():
    """build_step_context returning None: no further calls."""
    a = _make_adaptor(prepare_result=0, ctx_override=None)
    a.before_forward(None)

    assert a.call_order == ["build_step_context"]
    assert a.engine._ring_engine.prepare_step_calls == []
    assert a.transport.set_step_context_calls == []
    assert a.transport.pre_push_all_metas_calls == []


def test_null_offload_short_circuits():
    """transport.null_offload=True: build_step_context not even called."""
    a = _make_adaptor(prepare_result=0)
    a.transport.null_offload = True
    a.before_forward(None)

    assert a.call_order == []
    assert a.engine._ring_engine.prepare_step_calls == []
    assert a.transport.set_step_context_calls == []
    assert a.transport.pre_push_all_metas_calls == []


def test_register_preset_raises_on_duplicate():
    """selection.register_preset is strict-by-default."""
    import pytest
    from monitoring import selection

    # "full" is registered at module load -- re-registering must raise.
    with pytest.raises(ValueError, match="already registered"):
        selection.register_preset("full", frozenset())


def test_register_preset_adds_new_name():
    """A novel name registers successfully and is resolvable."""
    from monitoring import selection

    name = "_test_phase1_preset"
    assert name not in selection._HOOK_SELECTIONS
    try:
        selection.register_preset(name, frozenset({0, 1, 2}))
        assert selection._HOOK_SELECTIONS[name] == frozenset({0, 1, 2})
        assert selection.resolve_hook_selection(name) == frozenset({0, 1, 2})
    finally:
        # Clean up so the test doesn't pollute the global preset table.
        selection._HOOK_SELECTIONS.pop(name, None)
