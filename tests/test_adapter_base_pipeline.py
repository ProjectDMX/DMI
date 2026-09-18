"""Behavioral tests for the *base* BackendAdapter attach/plan pipeline.

``tests/test_adapter_protocol.py`` pins the ``before_forward`` driver ordering
with a stub whose ``plan_step`` is overridden, and
``tests/test_hook_selection_filters.py`` pins each selection/PP/TP filter in
isolation.  Neither one ever executes the base-class bodies of
``BackendAdapter.attach_model`` and ``BackendAdapter.plan_step``, so a
regression inside either method -- a dropped filter in the attach pipeline, a
lost per-hook alignment or an off-by-one in the byte walk -- would ship
silently.

This suite therefore drives those two methods directly through a concrete
adapter (:class:`PlanningAdapter`) that deliberately does **not** override
them, and then chains them: ``attach_model`` decides which hooks exist and
``plan_step`` prices exactly that surviving list.

Only the Python hook-definition, selection and dispatch layers are involved,
so no GPU or compiled backend is required and this suite belongs in the CPU
PR gate.
"""
from __future__ import annotations

import pytest
import torch

from dmi.adapters.base import BackendAdapter, StepPlan
from dmi.adapters.types import StepContext
from dmi.hooks.specs import (
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_PATTERN,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HookSpec,
    ModelShapeConfig,
    align_up_py,
    compute_hook_shape,
)

pytestmark = pytest.mark.cpu

# Sentinels, not ``True``/``None``: they keep "the pipeline wrote this field"
# distinguishable from "the pipeline left this hook alone".  Seeding them with
# real values would make the "not installed" assertions below vacuous.
_ENABLED_UNTOUCHED = "enabled-never-written"
_PAYLOAD_UNSET = "ring-payload-never-written"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubHookPoint:
    """Minimal HookPoint stand-in: the filters assign ``.enabled`` and
    ``install_ring_hooks`` assigns the three ``_ring_*`` attributes."""

    def __init__(self) -> None:
        self.enabled = _ENABLED_UNTOUCHED
        self._ring_hook_type = None
        self._ring_hook_id = None
        self._ring_payload = _PAYLOAD_UNSET


class FakeTransport:
    def __init__(self) -> None:
        self.null_offload = False
        self.force_eager = False
        self._ring_payload = object()
        self._active_specs = None
        self._using_forward_hooks = False
        self.model_cfgs: list = []

    def set_model_cfg(self, cfg):
        self.model_cfgs.append(cfg)

    def set_step_context(self, **kwargs):
        pass

    def pre_push_all_metas(self, **kwargs):
        pass


class FakeEngine:
    def __init__(self, transport) -> None:
        self._ring_transport = transport
        self._ring_engine = None


class RecordingRingEngine:
    def __init__(self, prepare_step_result: int = 2) -> None:
        self._result = prepare_step_result
        self.prepare_step_calls: list = []

    def prepare_step(self, total_bytes: int, n_hooks: int) -> int:
        self.prepare_step_calls.append((total_bytes, n_hooks))
        return self._result


class PlanningAdapter(BackendAdapter):
    """Concrete adapter that does NOT override ``plan_step`` /
    ``attach_model`` -- every test here runs the real base implementations."""

    def __init__(self, engine, cfg=None, specs=(), ranks=(0, 0, 0, 0),
                 pp_first=True, pp_last=True, eager_types=frozenset()):
        super().__init__(engine, "m")
        self._detect_cfg = cfg
        self._model_specs = list(specs)
        self._ranks = ranks
        self._pp_first, self._pp_last = pp_first, pp_last
        self._eager_types = eager_types

    def detect_model_shape(self, model):
        return self._detect_cfg

    def detect_parallel_ranks(self):
        return self._ranks

    def is_pp_first(self):
        return self._pp_first

    def is_pp_last(self):
        return self._pp_last

    def build_step_context(self, *raw):
        return None

    def on_capacity_exceeded(self, ctx):
        pass

    def _spec_needs_eager(self, spec):
        return spec.hook_type in self._eager_types


class ToyModel:
    """Stands in for the framework model: only ``get_hook_specs`` is used."""

    def __init__(self, specs) -> None:
        self._specs = specs

    def get_hook_specs(self):
        return list(self._specs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> ModelShapeConfig:
    """A minimal shape config; ``intermediate_dim`` defaults to unavailable."""
    kwargs = dict(hidden_dim=16, num_heads=4, num_kv_heads=4, head_dim=4,
                  dtype=torch.float16, vocab_size=32, intermediate_dim=0)
    kwargs.update(overrides)
    return ModelShapeConfig(**kwargs)


def _ctx(batch=2, q_len=4, kv_dim=4, logits_to_keep=0,
         actual_q_len=None) -> StepContext:
    return StepContext(
        model_id="m", flattened=False, req_ids=["0:0", "0:1"],
        token_ranges=[(0, q_len), (0, q_len)], dim0_offsets=[0, 1],
        kv_offsets=[0, 0], batch=batch, q_len=q_len, kv_dim=kv_dim,
        logits_to_keep=logits_to_keep, actual_q_len=actual_q_len,
    )


def _adapter(**kwargs) -> PlanningAdapter:
    return PlanningAdapter(FakeEngine(FakeTransport()), **kwargs)


# ---------------------------------------------------------------------------
# plan_step
# ---------------------------------------------------------------------------


def test_plan_step_returns_empty_plan_before_attach():
    a = _adapter()
    assert a.plan_step(_ctx()) == StepPlan(0, 0, False)
    # A shape config alone is not enough: an empty spec list still plans zero.
    a.model_cfg = _cfg()
    assert a.plan_step(_ctx()) == StepPlan(0, 0, False)


def test_plan_step_sums_16_byte_aligned_bytes_and_counts_firing_hooks():
    a = _adapter()
    a.model_cfg = _cfg()
    # resid_pre: [2,4,16] fp16 = 256 B (already aligned)
    # q:         [2,4,4,4] fp16 = 256 B
    # token_ids: [2,4] int64  =  64 B
    a.active_specs = [
        HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0),
        HookSpec(HOOK_TYPE_Q, StubHookPoint(), 0),
        HookSpec(HOOK_TYPE_TOKEN_IDS, StubHookPoint(), -1,
                 dtype=torch.int64),
    ]
    plan = a.plan_step(_ctx())
    assert plan.hook_count == 3
    assert plan.total_bytes == 256 + 256 + 64
    assert plan.needs_eager is False


def test_plan_step_rounds_each_hook_up_to_16_bytes_independently():
    # Alignment is per hook, not per sum: 3 hooks of 6 B each must reserve
    # 3 * 16 = 48 B, not align_up(18) = 32 B and not a raw 18 B.
    cfg = _cfg(hidden_dim=3, num_heads=1, num_kv_heads=1, head_dim=3)
    a = _adapter()
    a.model_cfg = cfg
    a.active_specs = [HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), i)
                      for i in range(3)]
    plan = a.plan_step(_ctx(batch=0, q_len=1))
    assert compute_hook_shape(HOOK_TYPE_RESID_PRE, cfg, 0, 1, 4) == [1, 3]
    assert align_up_py(6, 16) == 16
    assert plan.total_bytes == 48
    assert plan.hook_count == 3


def test_plan_step_skips_empty_shape_specs_entirely():
    # intermediate_dim=0 -> mlp_post shape is [] -> neither counted nor sized.
    cfg = _cfg(intermediate_dim=0)
    a = _adapter()
    a.model_cfg = cfg
    a.active_specs = [
        HookSpec(HOOK_TYPE_MLP_POST, StubHookPoint(), 0),
        HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0),
    ]
    assert compute_hook_shape(HOOK_TYPE_MLP_POST, cfg, 2, 4, 4) == []
    plan = a.plan_step(_ctx())
    assert plan.hook_count == 1
    assert plan.total_bytes == 256


def test_plan_step_empty_shape_spec_does_not_set_needs_eager():
    # A spec that never fires must not drag the whole step into eager
    # dispatch, even when its hook type would otherwise demand it.
    cfg = _cfg(intermediate_dim=0)
    a = _adapter(eager_types={HOOK_TYPE_MLP_POST})
    a.model_cfg = cfg
    a.active_specs = [HookSpec(HOOK_TYPE_MLP_POST, StubHookPoint(), 0)]
    assert a.plan_step(_ctx()) == StepPlan(0, 0, False)


def test_plan_step_ors_needs_eager_over_firing_specs_only():
    a = _adapter(eager_types={HOOK_TYPE_Q})
    a.model_cfg = _cfg()
    a.active_specs = [HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0)]
    assert a.plan_step(_ctx()).needs_eager is False
    a.active_specs.append(HookSpec(HOOK_TYPE_Q, StubHookPoint(), 0))
    plan = a.plan_step(_ctx())
    assert plan.needs_eager is True
    assert plan.hook_count == 2


def test_plan_step_uses_actual_q_len_only_for_strip_eligible_specs():
    a = _adapter()
    a.model_cfg = _cfg()
    strip = HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0,
                     dim0_is_actual_tokens=True)
    nostrip = HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 1)
    a.active_specs = [strip, nostrip]
    plan = a.plan_step(_ctx(q_len=8, actual_q_len=2))
    # strip spec sized at q_len=2 -> [2,2,16] fp16 = 128 B
    # nostrip spec sized at q_len=8 -> [2,8,16] fp16 = 512 B
    assert plan.total_bytes == 128 + 512
    # Without actual_q_len both use ctx.q_len=8.
    assert a.plan_step(_ctx(q_len=8)).total_bytes == 512 + 512


def test_plan_step_honors_per_spec_dtype_override():
    a = _adapter()
    a.model_cfg = _cfg()
    a.active_specs = [HookSpec(HOOK_TYPE_TOKEN_IDS, StubHookPoint(), -1,
                               dtype=torch.int64)]
    assert a.plan_step(_ctx()).total_bytes == 2 * 4 * 8
    # No override -> the model dtype (fp16) sizes the payload.
    a.active_specs = [HookSpec(HOOK_TYPE_TOKEN_IDS, StubHookPoint(), -1)]
    assert a.plan_step(_ctx()).total_bytes == align_up_py(2 * 4 * 2, 16)


def test_plan_step_feeds_commit_step_reservation_arguments():
    a = _adapter()
    a.ring_engine = RecordingRingEngine(prepare_step_result=2)
    a.model_cfg = _cfg()
    a.active_specs = [HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0)]
    a.commit_step(_ctx())
    # commit_step reserves with exactly the plan the base computed.
    assert a.ring_engine.prepare_step_calls == [(256, 1)]
    assert a.transport.force_eager is True


# ---------------------------------------------------------------------------
# attach_model
# ---------------------------------------------------------------------------


def test_attach_model_requires_ring_transport():
    a = PlanningAdapter(FakeEngine(None), cfg=_cfg())
    with pytest.raises(RuntimeError, match="enable_ring_transport"):
        a.attach_model(ToyModel([]))


def test_attach_model_runs_selection_then_pp_then_tp_filters_and_installs():
    cfg = _cfg(intermediate_dim=32, tp_size=2)
    specs = {
        "resid_pre": HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0),
        "q": HookSpec(HOOK_TYPE_Q, StubHookPoint(), 0),
        "mlp_post": HookSpec(HOOK_TYPE_MLP_POST, StubHookPoint(), 0),
        "token_ids": HookSpec(HOOK_TYPE_TOKEN_IDS, StubHookPoint(), -1),
        "final_logits": HookSpec(HOOK_TYPE_FINAL_LOGITS, StubHookPoint(), -1),
        "pattern": HookSpec(HOOK_TYPE_PATTERN, StubHookPoint(), 0),
    }
    a = _adapter(cfg=cfg, specs=list(specs.values()), ranks=(1, 0, 0, 0),
                 pp_first=False, pp_last=True)
    a.attach_model(ToyModel(list(specs.values())), hook_selection="full")

    kept = {s.hook_type for s in a.active_specs}
    # token_ids is PP_FIRST_ONLY and this rank is not first -> dropped.
    assert HOOK_TYPE_TOKEN_IDS not in kept
    # final_logits survives PP (last rank) but is unsharded on tp_rank 1 ->
    # dropped by filter_by_tp_rank.
    assert HOOK_TYPE_FINAL_LOGITS not in kept
    # resid_pre is unsharded and not PP-restricted -> dropped by the TP filter
    # only.
    assert HOOK_TYPE_RESID_PRE not in kept
    # Sharded hooks survive on a non-zero TP rank.
    assert kept == {HOOK_TYPE_Q, HOOK_TYPE_MLP_POST, HOOK_TYPE_PATTERN}

    # Dropped specs are explicitly disabled by their filter.
    assert specs["token_ids"].module.enabled is False
    assert specs["final_logits"].module.enabled is False
    assert specs["resid_pre"].module.enabled is False

    # install_ring_hooks ran on exactly the surviving specs.
    for name in ("q", "mlp_post", "pattern"):
        hook_point = specs[name].module
        assert hook_point._ring_hook_type == specs[name].hook_type
        assert hook_point._ring_payload is a.transport._ring_payload
    for name in ("token_ids", "final_logits", "resid_pre"):
        assert specs[name].module._ring_payload == _PAYLOAD_UNSET

    # Transport + adapter state are published from the same filtered list.
    assert a.transport._active_specs is a.active_specs
    assert a.transport._using_forward_hooks is True
    assert a.transport.model_cfgs == [cfg]
    assert a.model_cfg.tp_rank == 1
    assert a.model_cfg.tp_size == 2


def test_attach_model_selection_narrows_before_rank_filters():
    cfg = _cfg(intermediate_dim=32)
    specs = [
        HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0),
        HookSpec(HOOK_TYPE_Q, StubHookPoint(), 0),
    ]
    a = _adapter(cfg=cfg, specs=specs)
    a.attach_model(ToyModel(specs), hook_selection="resid_pre")

    assert [s.hook_type for s in a.active_specs] == [HOOK_TYPE_RESID_PRE]
    # q would have survived both rank filters; only the selection dropped it.
    assert specs[1].module.enabled is False
    assert specs[1].module._ring_payload == _PAYLOAD_UNSET


def test_attach_model_result_drives_plan_step():
    """attach_model -> plan_step is one pipeline: only installed hooks
    contribute to the byte budget."""
    cfg = _cfg(intermediate_dim=32)
    specs = [
        HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0),
        HookSpec(HOOK_TYPE_TOKEN_IDS, StubHookPoint(), -1,
                 dtype=torch.int64),
    ]
    a = _adapter(cfg=cfg, specs=specs, pp_first=False, pp_last=True)
    a.attach_model(ToyModel(specs), hook_selection="full")

    plan = a.plan_step(_ctx())
    # token_ids was dropped by the PP filter -> its 64 B is not reserved.
    assert plan.hook_count == 1
    assert plan.total_bytes == 256


def test_attach_model_tp_rank_zero_keeps_unsharded_hooks():
    cfg = _cfg()
    specs = [HookSpec(HOOK_TYPE_RESID_PRE, StubHookPoint(), 0)]
    a = _adapter(cfg=cfg, specs=specs, ranks=(0, 0, 0, 0))
    a.attach_model(ToyModel(specs), hook_selection="full")
    assert [s.hook_type for s in a.active_specs] == [HOOK_TYPE_RESID_PRE]
    assert specs[0].module.enabled is True
