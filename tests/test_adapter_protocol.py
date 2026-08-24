"""Unit tests for BackendAdapter step planning and commit ordering.

Uses fakes for MonitoringEngine, RingTransport, and RingEngine to verify
the driver flow:

    build_step_context -> plan_step -> commit_step -> [prepare_step
        -> adapt_for_cpu_direct (if result==2)
        -> on_capacity_exceeded (if result==2)
        -> _warn_once_capacity (if result==2)]
    set transport.force_eager from (result == 2) OR needs_eager.
    -> set_step_context -> pre_push_all_metas

No GPU or compiled backend is required.  The adapter base depends only on the
Python hook-definition and dispatch layers, so this suite belongs in the CPU
PR gate.
"""
from __future__ import annotations

import dataclasses

import pytest

from dmi.adapters.base import BackendAdapter, StepPlan, StepReservation
from dmi.adapters.types import StepContext

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


class StubAdapter(BackendAdapter):
    """Concrete BackendAdapter with fixed StepContext and recorded callbacks."""

    def __init__(self, engine, model_id, ctx, step_plan=(1024, 3, False)):
        super().__init__(engine, model_id)
        self._ctx = ctx
        self._step_plan_value = StepPlan(*step_plan)
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

    def plan_step(self, ctx):
        self.call_order.append("plan_step")
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
    return StubAdapter(engine, "test_model", ctx, step_plan=step_plan)


# ---------------------------------------------------------------------------
# Public planning / commit API
# ---------------------------------------------------------------------------


def test_step_plan_is_frozen_slotted_value():
    plan = StepPlan(total_bytes=1024, hook_count=3, needs_eager=True)

    assert tuple(plan) == (1024, 3, True)
    assert plan[:2] == (1024, 3)
    assert not hasattr(plan, "__dict__")
    with pytest.raises(AttributeError):
        plan.total_bytes = 2048


def test_step_reservation_matches_native_result_codes():
    assert int(StepReservation.SKIPPED) == -1
    assert int(StepReservation.RESERVED) == 0
    assert int(StepReservation.FLUSHED) == 1
    assert int(StepReservation.OVERSIZED) == 2


def test_commit_step_uses_supplied_plan_without_replanning():
    a = _make_adaptor(prepare_result=1, step_plan=(999, 9, False))
    supplied = StepPlan(total_bytes=2048, hook_count=4, needs_eager=True)

    reservation = a.commit_step(_make_ctx(), supplied)

    assert reservation is StepReservation.FLUSHED
    assert a.call_order == []
    assert a.engine._ring_engine.prepare_step_calls == [(2048, 4)]
    assert a.transport.force_eager is True
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


def test_commit_step_plans_once_when_plan_is_omitted():
    a = _make_adaptor(prepare_result=0)

    reservation = a.commit_step(_make_ctx())

    assert reservation is StepReservation.RESERVED
    assert a.call_order == ["plan_step"]
    assert a.engine._ring_engine.prepare_step_calls == [(1024, 3)]


def test_commit_step_returns_skipped_without_hooks_but_publishes_context():
    a = _make_adaptor(prepare_result=2)
    plan = StepPlan(total_bytes=0, hook_count=0, needs_eager=False)

    reservation = a.commit_step(_make_ctx(), plan)

    assert reservation is StepReservation.SKIPPED
    assert a.call_order == []
    assert a.engine._ring_engine.prepare_step_calls == []
    assert a.transport.force_eager is False
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


def test_commit_step_returns_oversized_after_fallback_callbacks():
    a = _make_adaptor(prepare_result=2)
    plan = StepPlan(total_bytes=1024, hook_count=3, needs_eager=False)

    reservation = a.commit_step(_make_ctx(), plan)

    assert reservation is StepReservation.OVERSIZED
    assert a.call_order == [
        "adapt_for_cpu_direct",
        "on_capacity_exceeded",
        "_warn_once_capacity",
    ]
    assert a.transport.force_eager is True
    assert a.transport.pre_push_all_metas_calls[0]["q_len"] == 104


def test_compute_step_plan_compatibility_wrapper_returns_tuple():
    a = _make_adaptor(prepare_result=0, step_plan=(2048, 4, True))

    assert a._compute_step_plan(_make_ctx()) == (2048, 4, True)
    assert a.call_order == ["plan_step"]


def test_read_only_adaptor_state_properties_return_snapshots():
    a = _make_adaptor(prepare_result=0)
    first_spec = object()
    second_spec = object()
    model_shape = object()
    a.active_specs = [first_spec]
    a.model_cfg = model_shape

    snapshot = a.active_hook_specs
    a.active_specs.append(second_spec)

    assert snapshot == (first_spec,)
    assert a.active_hook_specs == (first_spec, second_spec)
    assert a.model_shape is model_shape
    with pytest.raises(AttributeError):
        a.active_hook_specs = ()
    with pytest.raises(AttributeError):
        a.model_shape = None


# ---------------------------------------------------------------------------
# before_forward compatibility
# ---------------------------------------------------------------------------


def test_happy_path_result_zero():
    """prepare_step -> 0: no capacity hooks fire; force_eager stays False."""
    a = _make_adaptor(prepare_result=0)
    a.before_forward(None)

    assert a.call_order == ["build_step_context", "plan_step"]
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
        "build_step_context", "plan_step",
        "adapt_for_cpu_direct", "on_capacity_exceeded", "_warn_once_capacity",
    ]
    assert a.transport.force_eager is True
    assert len(a.adapt_for_cpu_direct_calls) == 1
    assert len(a.on_capacity_exceeded_calls) == 1
    # on_capacity_exceeded receives the post-adapt ctx (StubAdapter.adapt_for_cpu_direct
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
    from dmi.hooks import selection

    # "full" is registered at module load -- re-registering must raise.
    with pytest.raises(ValueError, match="already registered"):
        selection.register_preset("full", frozenset())


def test_register_preset_adds_new_name():
    """A novel name registers successfully and is resolvable."""
    from dmi.hooks import selection

    name = "_test_phase1_preset"
    assert name not in selection._HOOK_SELECTIONS
    try:
        selection.register_preset(name, frozenset({0, 1, 2}))
        assert selection._HOOK_SELECTIONS[name] == frozenset({0, 1, 2})
        assert selection.resolve_hook_selection(name) == frozenset({0, 1, 2})
    finally:
        # Clean up so the test doesn't pollute the global preset table.
        selection._HOOK_SELECTIONS.pop(name, None)
