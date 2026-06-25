from __future__ import annotations

import json

import pytest

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

pytestmark = pytest.mark.cpu


def test_moe_v1_routing_hook_types_registered() -> None:
    assert _id_by_short["router_logits"] == HOOK_TYPE_ROUTER_LOGITS
    assert _id_by_short["topk_ids"] == HOOK_TYPE_TOPK_IDS
    assert _id_by_short["topk_weights"] == HOOK_TYPE_TOPK_WEIGHTS


def test_moe_v1_routing_shapes_from_qwen2_moe_config() -> None:
    cfg = Qwen2MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=60,
        num_experts_per_tok=4,
        vocab_size=128,
    )
    model_shape = _make_model_shape_from_hf_config(cfg)
    assert model_shape is not None

    q_len = 17
    kv_dim = 17

    assert _compute_hook_shape(
        HOOK_TYPE_ROUTER_LOGITS, model_shape, batch=0, q_len=q_len, kv_dim=kv_dim
    ) == [q_len, 60]
    assert _compute_hook_shape(
        HOOK_TYPE_TOPK_IDS, model_shape, batch=0, q_len=q_len, kv_dim=kv_dim
    ) == [q_len, 4]
    assert _compute_hook_shape(
        HOOK_TYPE_TOPK_WEIGHTS, model_shape, batch=0, q_len=q_len, kv_dim=kv_dim
    ) == [q_len, 4]


def test_vllm_adapter_remaps_qwen2_moe_to_hooked_variant() -> None:
    from integration.vllm_adapter import _ARCH_REMAP

    assert _ARCH_REMAP["Qwen2MoeForCausalLM"] == "Qwen2MoePForCausalLM"


def test_ref_disk_worker_remaps_qwen2_moe_to_ref_variant() -> None:
    from tests.ref_disk_worker import _ARCH_REMAP as _REF_ARCH_REMAP

    assert _REF_ARCH_REMAP["Qwen2MoeForCausalLM"] == "Qwen2MoeRefForCausalLM"


def test_hf_hooked_qwen2_moe_exposes_routing_hook_specs() -> None:
    model = HookedQwen2MoeForCausalLM(
        Qwen2MoeConfig(
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
    assert HOOK_TYPE_ROUTER_LOGITS in emitted
    assert HOOK_TYPE_TOPK_IDS in emitted
    assert HOOK_TYPE_TOPK_WEIGHTS in emitted


def test_hf_compare_qwen2_moe_exposes_compare_api() -> None:
    model = CompareQwen2MoeForCausalLM(
        Qwen2MoeConfig(
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


def test_qwen2_moe_ref_preset_adds_routing_hooks(tmp_path) -> None:
    from integration.vllm.vllm.model_executor.models.enable_ref_hooks import enable_ref_hooks

    model_file = tmp_path / "qwen2_moe_ref.py"
    model_file.write_text("class Dummy:\n    pass\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    cfg_out = tmp_path / "ref_config.json"

    enable_ref_hooks(
        model_file=str(model_file),
        hooks="vllm-full",
        max_len=128,
        output_dir=str(out_dir),
        config_out=str(cfg_out),
    )

    cfg = json.loads(cfg_out.read_text(encoding="utf-8"))
    assert "router_logits" in cfg["enabled_hooks"]
    assert "topk_ids" in cfg["enabled_hooks"]
    assert "topk_weights" in cfg["enabled_hooks"]
