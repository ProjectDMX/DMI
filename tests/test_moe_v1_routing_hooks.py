from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.framework_fork


@lru_cache(maxsize=1)
def _mods() -> SimpleNamespace:
    try:
        from transformers import Qwen2MoeConfig
        from transformers.models.qwen2_moe_compare.modeling_qwen2_moe import CompareQwen2MoeForCausalLM
        from transformers.models.qwen2_moe_p.modeling_qwen2_moe import HookedQwen2MoeForCausalLM

        from integration.model_shape import _make_model_shape_from_hf_config
        from monitoring.ring_transport import (
            HOOK_TYPE_ROUTER_LOGITS,
            HOOK_TYPE_TOPK_IDS,
            HOOK_TYPE_TOPK_WEIGHTS,
            _compute_hook_shape,
            _id_by_short,
        )
    except ImportError as exc:
        pytest.skip(f"modified Transformers fork required: {exc}")
    return SimpleNamespace(**locals())


def test_moe_v1_routing_hook_types_registered() -> None:
    m = _mods()
    assert m._id_by_short["router_logits"] == m.HOOK_TYPE_ROUTER_LOGITS
    assert m._id_by_short["topk_ids"] == m.HOOK_TYPE_TOPK_IDS
    assert m._id_by_short["topk_weights"] == m.HOOK_TYPE_TOPK_WEIGHTS


def test_moe_v1_routing_shapes_from_qwen2_moe_config() -> None:
    m = _mods()
    cfg = m.Qwen2MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=60,
        num_experts_per_tok=4,
        vocab_size=128,
    )
    model_shape = m._make_model_shape_from_hf_config(cfg)
    assert model_shape is not None

    q_len = 17
    kv_dim = 17

    assert m._compute_hook_shape(
        m.HOOK_TYPE_ROUTER_LOGITS, model_shape, batch=0, q_len=q_len, kv_dim=kv_dim
    ) == [q_len, 60]
    assert m._compute_hook_shape(
        m.HOOK_TYPE_TOPK_IDS, model_shape, batch=0, q_len=q_len, kv_dim=kv_dim
    ) == [q_len, 4]
    assert m._compute_hook_shape(
        m.HOOK_TYPE_TOPK_WEIGHTS, model_shape, batch=0, q_len=q_len, kv_dim=kv_dim
    ) == [q_len, 4]


def test_hf_hooked_qwen2_moe_exposes_routing_hook_specs() -> None:
    m = _mods()
    model = m.HookedQwen2MoeForCausalLM(
        m.Qwen2MoeConfig(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            vocab_size=128,
        )
    )
    emitted = {spec.hook_type for spec in model.get_hook_specs()}
    assert m.HOOK_TYPE_ROUTER_LOGITS in emitted
    assert m.HOOK_TYPE_TOPK_IDS in emitted
    assert m.HOOK_TYPE_TOPK_WEIGHTS in emitted


def test_hf_compare_qwen2_moe_exposes_compare_api() -> None:
    m = _mods()
    model = m.CompareQwen2MoeForCausalLM(
        m.Qwen2MoeConfig(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            vocab_size=128,
        )
    )
    assert hasattr(model, "allocate_compare_buffers")
    assert hasattr(model, "get_ref_buffers")
