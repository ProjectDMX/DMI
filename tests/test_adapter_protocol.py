"""Unit tests for BackendAdaptor.before_forward call ordering.

Phase 1 verification gate (§11.1 of unified_adaptor_plan.md).  Uses fakes for
MonitoringEngine, RingTransport, and RingEngine to verify the driver flow:

    build_step_context -> _step_bytes -> [prepare_step
        -> adapt_for_cpu_direct (if result==2)
        -> on_capacity_exceeded (if result==2)
        -> _warn_once_capacity (if result==2)]
    -> set_step_context -> pre_push_all_metas

No GPU / native engine required.
"""
from __future__ import annotations

import dataclasses
from unittest.mock import patch

from monitoring.adaptor_base import BackendAdaptor
from monitoring.step_context import StepContext


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTransport:
    def __init__(self) -> None:
        self.null_offload = False
        self.cpu_direct = False
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

    def __init__(self, engine, model_id, ctx, step_bytes=(1024, 3)):
        super().__init__(engine, model_id)
        self._ctx = ctx
        self._step_bytes_value = step_bytes
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

    def _step_bytes(self, ctx):
        self.call_order.append("_step_bytes")
        return self._step_bytes_value


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


def _make_adaptor(prepare_result, ctx_override=..., step_bytes=(1024, 3)):
    engine = FakeEngine(prepare_step_result=prepare_result)
    ctx = _make_ctx() if ctx_override is ... else ctx_override
    return StubAdaptor(engine, "test_model", ctx, step_bytes=step_bytes)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("monitoring.adaptor_base.set_cpu_direct")
def test_happy_path_result_zero(mock_set_cpu_direct):
    """prepare_step -> 0: no capacity hooks fire; cpu_direct stays False."""
    a = _make_adaptor(prepare_result=0)
    a.before_forward(None)

    assert a.call_order == ["build_step_context", "_step_bytes"]
    assert a.transport.cpu_direct is False
    mock_set_cpu_direct.assert_called_once_with(False)
    assert len(a.engine._ring_engine.prepare_step_calls) == 1
    assert a.engine._ring_engine.prepare_step_calls[0] == (1024, 3)
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


@patch("monitoring.adaptor_base.set_cpu_direct")
def test_flushed_result_one(mock_set_cpu_direct):
    """prepare_step -> 1 (RING_FLUSHED): same as 0 from the adapter's view."""
    a = _make_adaptor(prepare_result=1)
    a.before_forward(None)

    assert a.transport.cpu_direct is False
    mock_set_cpu_direct.assert_called_once_with(False)
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


@patch("monitoring.adaptor_base.set_cpu_direct")
def test_capacity_exceeded_result_two(mock_set_cpu_direct):
    """prepare_step -> 2: adapt_for_cpu_direct + on_capacity_exceeded +
    _warn_once_capacity fire in order; cpu_direct True; rest of path runs."""
    a = _make_adaptor(prepare_result=2)
    a.before_forward(None)

    assert a.call_order == [
        "build_step_context", "_step_bytes",
        "adapt_for_cpu_direct", "on_capacity_exceeded", "_warn_once_capacity",
    ]
    assert a.transport.cpu_direct is True
    mock_set_cpu_direct.assert_called_once_with(True)
    assert len(a.adapt_for_cpu_direct_calls) == 1
    assert len(a.on_capacity_exceeded_calls) == 1
    # on_capacity_exceeded receives the post-adapt ctx (StubAdaptor.adapt_for_cpu_direct
    # bumps q_len by 100).
    assert a.on_capacity_exceeded_calls[0].q_len == 4 + 100
    assert a.warn_calls[0][1:] == (1024, 3)
    # set_step_context still runs after the capacity branch — uses the
    # adapted ctx, so q_len in the kwargs reflects the bump.
    assert len(a.transport.set_step_context_calls) == 1
    pushed_meta = a.transport.pre_push_all_metas_calls[0]
    assert pushed_meta["q_len"] == 4 + 100


@patch("monitoring.adaptor_base.set_cpu_direct")
def test_force_cpu_direct_skips_prepare_step(mock_set_cpu_direct):
    """When _force_cpu_direct=True, prepare_step is NOT called and the
    capacity branch is fully skipped; set_step_context + pre_push_all_metas
    still run with the original ctx."""
    a = _make_adaptor(prepare_result=2)  # would trigger if invoked
    a._force_cpu_direct = True
    a.before_forward(None)

    assert a.engine._ring_engine.prepare_step_calls == []
    mock_set_cpu_direct.assert_not_called()
    assert "_step_bytes" not in a.call_order
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


@patch("monitoring.adaptor_base.set_cpu_direct")
def test_n_hooks_zero_skips_prepare_step(mock_set_cpu_direct):
    """When _step_bytes returns (0, 0), prepare_step is skipped but
    set_step_context + pre_push_all_metas still run."""
    a = _make_adaptor(prepare_result=2, step_bytes=(0, 0))
    a.before_forward(None)

    assert a.engine._ring_engine.prepare_step_calls == []
    mock_set_cpu_direct.assert_not_called()
    assert a.adapt_for_cpu_direct_calls == []
    assert a.on_capacity_exceeded_calls == []
    assert a.warn_calls == []
    assert len(a.transport.set_step_context_calls) == 1
    assert len(a.transport.pre_push_all_metas_calls) == 1


@patch("monitoring.adaptor_base.set_cpu_direct")
def test_null_context_skips_everything_after_build(mock_set_cpu_direct):
    """build_step_context returning None: no further calls."""
    a = _make_adaptor(prepare_result=0, ctx_override=None)
    a.before_forward(None)

    assert a.call_order == ["build_step_context"]
    assert a.engine._ring_engine.prepare_step_calls == []
    mock_set_cpu_direct.assert_not_called()
    assert a.transport.set_step_context_calls == []
    assert a.transport.pre_push_all_metas_calls == []


@patch("monitoring.adaptor_base.set_cpu_direct")
def test_null_offload_short_circuits(mock_set_cpu_direct):
    """transport.null_offload=True: build_step_context not even called."""
    a = _make_adaptor(prepare_result=0)
    a.transport.null_offload = True
    a.before_forward(None)

    assert a.call_order == []
    assert a.engine._ring_engine.prepare_step_calls == []
    mock_set_cpu_direct.assert_not_called()
    assert a.transport.set_step_context_calls == []
    assert a.transport.pre_push_all_metas_calls == []


def test_register_preset_raises_on_duplicate():
    """selection.register_preset is strict-by-default."""
    import pytest
    from monitoring import selection

    # "full" is registered at module load — re-registering must raise.
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
