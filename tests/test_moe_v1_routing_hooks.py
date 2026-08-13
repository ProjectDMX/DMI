from __future__ import annotations

from functools import lru_cache
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.framework_fork


@lru_cache(maxsize=1)
def _vllm_mods() -> SimpleNamespace:
    try:
        from transformers import Qwen2MoeConfig

        from integration.model_shape import _make_model_shape_from_hf_config
        from integration.vllm_adapter import _ARCH_REMAP
        from integration.vllm.vllm.model_executor.models.enable_ref_hooks import enable_ref_hooks
        from integration.vllm.vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
        from monitoring.ring_transport import (
            HOOK_TYPE_ROUTER_LOGITS,
            HOOK_TYPE_TOPK_IDS,
            HOOK_TYPE_TOPK_WEIGHTS,
            _compute_hook_shape,
            _id_by_short,
        )
        from tests.ref_disk_worker import _ARCH_REMAP as _REF_ARCH_REMAP
    except ImportError as exc:
        pytest.skip(f"modified framework forks required: {exc}")
    return SimpleNamespace(**locals())


def test_moe_v1_routing_hook_types_registered() -> None:
    m = _vllm_mods()
    assert m._id_by_short["router_logits"] == m.HOOK_TYPE_ROUTER_LOGITS
    assert m._id_by_short["topk_ids"] == m.HOOK_TYPE_TOPK_IDS
    assert m._id_by_short["topk_weights"] == m.HOOK_TYPE_TOPK_WEIGHTS


def test_moe_v1_routing_shapes_from_qwen2_moe_config() -> None:
    m = _vllm_mods()
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


def test_vllm_adapter_remaps_qwen2_moe_to_hooked_variant() -> None:
    m = _vllm_mods()
    assert m._ARCH_REMAP["Qwen2MoeForCausalLM"] == "Qwen2MoePForCausalLM"


def test_vllm_compare_model_is_registered() -> None:
    m = _vllm_mods()
    assert m._TEXT_GENERATION_MODELS["Qwen2MoeCompareForCausalLM"] == (
        "qwen2_moe_compare",
        "Qwen2MoeCompareForCausalLM",
    )


def test_vllm_ref_model_is_registered() -> None:
    m = _vllm_mods()
    assert m._TEXT_GENERATION_MODELS["Qwen2MoeRefForCausalLM"] == (
        "qwen2_moe_ref",
        "Qwen2MoeRefForCausalLM",
    )
    assert m._REF_ARCH_REMAP["Qwen2MoeForCausalLM"] == "Qwen2MoeRefForCausalLM"


def test_qwen2_moe_ref_preset_adds_routing_hooks(tmp_path) -> None:
    m = _vllm_mods()
    model_file = tmp_path / "qwen2_moe_ref.py"
    model_file.write_text("class Dummy:\n    pass\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    cfg_out = tmp_path / "ref_config.json"

    m.enable_ref_hooks(
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
