"""Modified-Transformers contracts for Qwen2-MoE routing hooks."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import pytest

from tests.test_moe_v1_routing_hooks import _vllm_mods


pytestmark = pytest.mark.framework_fork


@lru_cache(maxsize=1)
def _hf_mods() -> SimpleNamespace:
    base = vars(_vllm_mods())
    try:
        from transformers.models.qwen2_moe_compare.modeling_qwen2_moe import (
            CompareQwen2MoeForCausalLM,
        )
        from transformers.models.qwen2_moe_p.modeling_qwen2_moe import (
            HookedQwen2MoeForCausalLM,
        )
    except ImportError as exc:
        pytest.skip(f"modified Transformers fork required: {exc}")
    return SimpleNamespace(
        **base,
        CompareQwen2MoeForCausalLM=CompareQwen2MoeForCausalLM,
        HookedQwen2MoeForCausalLM=HookedQwen2MoeForCausalLM,
    )


def _config(m: SimpleNamespace):
    return m.Qwen2MoeConfig(
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


def test_hf_hooked_qwen2_moe_exposes_routing_hook_specs() -> None:
    m = _hf_mods()
    model = m.HookedQwen2MoeForCausalLM(_config(m))
    emitted = {spec.hook_type for spec in model.get_hook_specs()}
    assert m.HOOK_TYPE_ROUTER_LOGITS in emitted
    assert m.HOOK_TYPE_TOPK_IDS in emitted
    assert m.HOOK_TYPE_TOPK_WEIGHTS in emitted


def test_hf_compare_qwen2_moe_exposes_compare_api() -> None:
    m = _hf_mods()
    model = m.CompareQwen2MoeForCausalLM(_config(m))
    assert hasattr(model, "allocate_compare_buffers")
    assert hasattr(model, "get_ref_buffers")
