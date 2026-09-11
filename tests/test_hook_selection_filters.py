"""Behavioral tests for the hook-selection / PP-rank / TP-rank filters.

``dmi.hooks.selection`` decides which hooks an adapter installs: adapters call
``filter_by_pp_rank`` / ``filter_by_tp_rank`` (and ``apply_hook_selection``)
on every ``attach_model``, so a filter that silently keeps or drops the wrong
hook types either duplicates unsharded payloads across TP ranks or loses
capture on PP stages.  These tests pin the decision surface only; the
transport-side effect of ``HookPoint.enabled`` is covered elsewhere.

The filters touch nothing but ``spec.module.enabled``, so a stub hook point
stands in for the real one.  Its ``enabled`` starts as a sentinel so "left
untouched" can be distinguished from "explicitly set to True/False".
"""
from __future__ import annotations

import pytest
import torch

from dmi.hooks.selection import (
    apply_hook_selection,
    filter_by_pp_rank,
    filter_by_tp_rank,
    hook_belongs_to_pp_rank,
    hook_belongs_to_tp_rank,
    select_hook_specs,
)
from dmi.hooks.specs import (
    HOOK_TYPE_TO_SHORT_NAME,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HookSpec,
    ModelShapeConfig,
    PP_FIRST_ONLY,
    PP_LAST_ONLY,
)

pytestmark = pytest.mark.cpu

_UNTOUCHED = "enabled-never-written"


class _StubHookPoint:
    """Minimal HookPoint stand-in: the filters only assign ``.enabled``."""

    def __init__(self) -> None:
        self.enabled = _UNTOUCHED


def _spec(hook_type: int, layer_no: int = -1) -> HookSpec:
    """A bound spec whose module records whether ``enabled`` was written."""
    return HookSpec(hook_type=hook_type, module=_StubHookPoint(),
                    layer_no=layer_no)


def _cfg(**overrides) -> ModelShapeConfig:
    """A minimal shape config; MoE/MLP fields default to unavailable (0)."""
    kwargs = dict(hidden_dim=8, num_heads=2, num_kv_heads=2, head_dim=4,
                  dtype=torch.float32, vocab_size=10)
    kwargs.update(overrides)
    return ModelShapeConfig(**kwargs)


_ALL_HOOK_TYPES = sorted(HOOK_TYPE_TO_SHORT_NAME)
_CONFIG_GATED_TYPES = {
    HOOK_TYPE_MLP_POST, HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOPK_IDS, HOOK_TYPE_TOPK_WEIGHTS,
}


# ---------------------------------------------------------------------------
# TP-rank filter
# ---------------------------------------------------------------------------

def test_hook_belongs_to_tp_rank_keeps_only_sharded_on_nonzero_rank():
    """Non-zero TP ranks own sharded hooks only; rank 0 owns everything."""
    sharded = _spec(HOOK_TYPE_Q, layer_no=0)
    unsharded = _spec(HOOK_TYPE_RESID_PRE, layer_no=0)

    assert hook_belongs_to_tp_rank(sharded, 1) is True
    assert hook_belongs_to_tp_rank(unsharded, 1) is False

    assert hook_belongs_to_tp_rank(sharded, 0) is True
    assert hook_belongs_to_tp_rank(unsharded, 0) is True


def test_filter_by_tp_rank_disables_dropped_modules():
    """Dropped specs are disabled in place; kept specs are not rewritten."""
    specs = [_spec(HOOK_TYPE_RESID_PRE, layer_no=0), _spec(HOOK_TYPE_Q, layer_no=0)]

    kept = filter_by_tp_rank(specs, 1)

    assert kept == [specs[1]]
    assert specs[0].module.enabled is False
    assert specs[1].module.enabled == _UNTOUCHED

    # Rank 0 short-circuits: the very same list object comes back untouched.
    rank0_specs = [_spec(HOOK_TYPE_RESID_PRE, layer_no=0), _spec(HOOK_TYPE_Q, layer_no=0)]
    assert filter_by_tp_rank(rank0_specs, 0) is rank0_specs
    assert [s.module.enabled for s in rank0_specs] == [_UNTOUCHED, _UNTOUCHED]


# ---------------------------------------------------------------------------
# PP-rank filter
# ---------------------------------------------------------------------------

def test_hook_belongs_to_pp_rank_matrix():
    """First-only hooks need the first stage, last-only the last, rest any."""
    for hook_type in sorted(PP_FIRST_ONLY):
        spec = _spec(hook_type)
        assert hook_belongs_to_pp_rank(spec, True, False) is True
        assert hook_belongs_to_pp_rank(spec, True, True) is True
        assert hook_belongs_to_pp_rank(spec, False, True) is False
        assert hook_belongs_to_pp_rank(spec, False, False) is False

    for hook_type in sorted(PP_LAST_ONLY):
        spec = _spec(hook_type)
        assert hook_belongs_to_pp_rank(spec, False, True) is True
        assert hook_belongs_to_pp_rank(spec, True, True) is True
        assert hook_belongs_to_pp_rank(spec, True, False) is False
        assert hook_belongs_to_pp_rank(spec, False, False) is False

    unrestricted = _spec(HOOK_TYPE_RESID_PRE, layer_no=0)
    for is_first in (True, False):
        for is_last in (True, False):
            assert hook_belongs_to_pp_rank(unrestricted, is_first, is_last) is True


def test_filter_by_pp_rank_drops_stage_restricted_hooks_on_middle_stage():
    """A middle stage keeps only the stage-agnostic hooks."""
    restricted = [_spec(ht) for ht in sorted(PP_FIRST_ONLY) + sorted(PP_LAST_ONLY)]
    survivor = _spec(HOOK_TYPE_RESID_PRE, layer_no=0)
    specs = restricted + [survivor]

    kept = filter_by_pp_rank(specs, False, False)

    assert kept == [survivor]
    assert not ({s.hook_type for s in kept} & (PP_FIRST_ONLY | PP_LAST_ONLY))
    for spec in restricted:
        assert spec.module.enabled is False, HOOK_TYPE_TO_SHORT_NAME[spec.hook_type]
    assert survivor.module.enabled == _UNTOUCHED


def test_pp_and_tp_filters_reject_unbound_specs():
    """Disabling a dropped hook needs a bound module, not a metadata spec."""
    with pytest.raises(RuntimeError, match="requires bound executable HookSpecs"):
        filter_by_pp_rank([HookSpec(HOOK_TYPE_RESID_FINAL, None)], True, False)

    with pytest.raises(RuntimeError, match="requires bound executable HookSpecs"):
        filter_by_tp_rank([HookSpec(HOOK_TYPE_RESID_PRE, None)], 1)


# ---------------------------------------------------------------------------
# Selection strings and config availability
# ---------------------------------------------------------------------------

def test_select_hook_specs_drops_hooks_with_unavailable_config():
    """Hooks whose shape needs a zeroed config field are not selectable.

    ``ModelShapeConfig.num_experts`` is read off the model config by name, so
    it stays 0 for architectures that spell it differently (e.g. Mixtral's
    ``num_local_experts``) -- this filter is what keeps those hooks out.
    """
    specs = [_spec(ht) for ht in _ALL_HOOK_TYPES]

    unavailable_cfg = _cfg(intermediate_dim=0, num_experts=0, top_k=0)
    selected = select_hook_specs(specs, "full", unavailable_cfg)
    assert {s.hook_type for s in selected} == set(_ALL_HOOK_TYPES) - _CONFIG_GATED_TYPES
    assert HOOK_TYPE_RESID_PRE in {s.hook_type for s in selected}

    available_cfg = _cfg(intermediate_dim=256, num_experts=8, top_k=2)
    selected = select_hook_specs(specs, "full", available_cfg)
    assert {s.hook_type for s in selected} == set(_ALL_HOOK_TYPES)


def test_select_hook_specs_honours_selection_string():
    """A single-hook selection returns exactly that hook's specs, in order."""
    resid_pre = [_spec(HOOK_TYPE_RESID_PRE, layer_no=0),
                 _spec(HOOK_TYPE_RESID_PRE, layer_no=1)]
    specs = [resid_pre[0], _spec(HOOK_TYPE_Q, layer_no=0), resid_pre[1],
             _spec(HOOK_TYPE_MLP_POST, layer_no=0)]

    assert select_hook_specs(specs, "hidden-states") == resid_pre


def test_apply_hook_selection_sets_enabled_flags_and_warns():
    """apply_hook_selection returns the selected specs and flags every spec."""
    resid_pre = [_spec(HOOK_TYPE_RESID_PRE, layer_no=0),
                 _spec(HOOK_TYPE_RESID_PRE, layer_no=1)]
    others = [_spec(HOOK_TYPE_Q, layer_no=0), _spec(HOOK_TYPE_RESID_FINAL)]
    specs = [resid_pre[0], others[0], resid_pre[1], others[1]]

    enabled = apply_hook_selection(specs, "resid_pre",
                                   _cfg(intermediate_dim=16, num_experts=8, top_k=2))

    assert enabled == resid_pre
    assert all(s.module.enabled is True for s in resid_pre)
    assert all(s.module.enabled is False for s in others)

    # A spec whose shape config is unavailable is reported, not silently kept.
    gated = [_spec(HOOK_TYPE_RESID_PRE, layer_no=0), _spec(HOOK_TYPE_MLP_POST, layer_no=0)]
    with pytest.warns(UserWarning, match="required model config unavailable"):
        enabled = apply_hook_selection(gated, "full", _cfg(intermediate_dim=0))
    assert gated[1] not in enabled
    assert gated[1].module.enabled is False

    with pytest.raises(RuntimeError, match="requires bound executable HookSpecs"):
        apply_hook_selection([HookSpec(HOOK_TYPE_RESID_PRE, None)], "resid_pre", None)
