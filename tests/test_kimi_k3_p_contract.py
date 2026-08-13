"""CPU contracts for bounded Kimi K3 decoder monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP  # isort: skip
from tests.compare_worker import (  # isort: skip
    _ARCH_REMAP as _COMPARE_ARCH_REMAP,
    _COMPARE_MODEL_VARIANTS,
)

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.kimi_k3_compare import (
    KimiK3CompareForConditionalGeneration,
)
from vllm.model_executor.models.kimi_k3_p import (
    _FULL_ATTN_LAYERS,
    _KDA_LAYERS,
    KimiK3PDecoderLayer,
    KimiK3PForConditionalGeneration,
    KimiK3PLinearForCausalLM,
    KimiK3PModel,
    _require_supported_kimi_k3_config,
)
from vllm.models.kimi_k3.nvidia.model import (
    KimiDecoderLayer,
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config

from integration.model_shape import _make_model_shape_from_hf_config
from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)

pytestmark = pytest.mark.framework_fork


def _quantization() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "format": "mxfp4-pack-quantized",
                "weights": {
                    "group_size": 32,
                    "num_bits": 4,
                    "scale_dtype": "torch.uint8",
                },
            }
        },
        "format": "mxfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def _text_values(**overrides) -> dict:
    values = {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 12,
        "first_k_dense_replace": 1,
        "hidden_act": "situ",
        "hidden_size": 7168,
        "intermediate_size": 33_792,
        "kv_lora_rank": 512,
        "latent_moe_use_norm": True,
        "max_position_embeddings": 1_048_576,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "moe_intermediate_size": 3072,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": 96,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_key_value_heads": 96,
        "num_nextn_predict_layers": 0,
        "num_shared_experts": 2,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "quantization_config": _quantization(),
        "rms_norm_eps": 1e-5,
        "routed_expert_hidden_size": 3584,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
        "use_grouped_topk": True,
        "v_head_dim": 128,
        "vocab_size": 163_840,
        "linear_attn_config": {
            "full_attn_layers": _FULL_ATTN_LAYERS,
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "kda_layers": _KDA_LAYERS,
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
    }
    values.update(overrides)
    return values


def _vision_values(**overrides) -> dict:
    values = {
        "patch_size": 14,
        "merge_kernel_size": (2, 2),
        "merge_type": "sd2_tpool",
        "mm_projector_type": "patchmergerv2",
        "mm_hidden_size": 1024,
        "qkv_hidden_size": 1536,
        "text_hidden_size": 7168,
        "vt_hidden_size": 1024,
        "vt_intermediate_size": 4096,
        "vt_num_attention_heads": 12,
        "vt_num_hidden_layers": 27,
    }
    values.update(overrides)
    return values


def _config(
    *,
    text_overrides: dict | None = None,
    vision_overrides: dict | None = None,
    **overrides,
) -> SimpleNamespace:
    values = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "bos_token_id": 163_584,
        "eos_token_id": 163_586,
        "pad_token_id": 163_839,
        "image_placeholder": "<|kimi_image_placeholder|>",
        "media_placeholder_token_id": 163_605,
        "tie_word_embeddings": False,
        "text_config": SimpleNamespace(**_text_values(**(text_overrides or {}))),
        "vision_config": SimpleNamespace(**_vision_values(**(vision_overrides or {}))),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parallel(**overrides) -> SimpleNamespace:
    values = {
        "tensor_parallel_size": 32,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "use_sequence_parallel_moe": False,
        "enable_eplb": False,
        "use_ubatching": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quant_config(name: str = "compressed-tensors") -> SimpleNamespace:
    return SimpleNamespace(get_name=lambda: name)


def _add_hooks(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


def _fake_language_model() -> KimiK3PLinearForCausalLM:
    subject = KimiK3PLinearForCausalLM.__new__(KimiK3PLinearForCausalLM)
    nn.Module.__init__(subject)
    subject.config = SimpleNamespace(**_text_values())
    _add_hooks(subject, ("token_ids", "final_ln", "final_logits"))

    model = nn.Module()
    model.start_layer = 0
    model.end_layer = 93
    model.layers = nn.ModuleList()
    _add_hooks(model, ("embed", "resid_final"))
    for _ in range(93):
        layer = nn.Module()
        _add_hooks(layer, ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"))
        model.layers.append(layer)
    subject.model = model
    return subject


def test_kimi_k3_preserves_public_wrapper_and_loader_contracts() -> None:
    assert (
        _ARCH_REMAP["KimiK3ForConditionalGeneration"]
        == "KimiK3PForConditionalGeneration"
    )
    assert issubclass(
        KimiK3PForConditionalGeneration,
        KimiK3ForConditionalGeneration,
    )
    assert issubclass(
        KimiK3CompareForConditionalGeneration,
        KimiK3PForConditionalGeneration,
    )
    assert issubclass(KimiK3PLinearForCausalLM, KimiLinearForCausalLM)
    assert issubclass(KimiK3PModel, KimiLinearModel)
    assert issubclass(KimiK3PDecoderLayer, KimiDecoderLayer)
    assert (
        KimiK3PForConditionalGeneration.load_weights
        is KimiK3ForConditionalGeneration.load_weights
    )
    assert (
        KimiK3PForConditionalGeneration.hf_to_vllm_mapper
        is KimiK3ForConditionalGeneration.hf_to_vllm_mapper
    )
    assert KimiK3PForConditionalGeneration.get_placeholder_str("image", 0) == (
        "<|kimi_image_placeholder|>"
    )
    assert (
        ModelRegistry.models["KimiK3PForConditionalGeneration"].load_model_cls()
        is KimiK3PForConditionalGeneration
    )
    assert (
        _COMPARE_ARCH_REMAP["KimiK3ForConditionalGeneration"]
        == "KimiK3CompareForConditionalGeneration"
    )
    assert _COMPARE_MODEL_VARIANTS["KimiK3CompareForConditionalGeneration"] == (
        "kimi_k3_compare:KimiK3CompareForConditionalGeneration"
    )
    assert (
        ModelRegistry.models["KimiK3CompareForConditionalGeneration"].load_model_cls()
        is KimiK3CompareForConditionalGeneration
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"architectures": ["KimiK25ForConditionalGeneration"]}, "architectures"),
        ({"model_type": "kimi_k25"}, "model_type"),
        ({"image_placeholder": "<image>"}, "image_placeholder"),
        ({"media_placeholder_token_id": 1}, "media_placeholder_token_id"),
        ({"tie_word_embeddings": True}, "tie_word_embeddings"),
    ],
)
def test_kimi_k3_outer_contract_fails_closed(overrides, match) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_kimi_k3_config(
            _config(**overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"hidden_act": "silu"}, "hidden_act"),
        ({"hidden_size": 4096}, "hidden_size"),
        ({"intermediate_size": 11_008}, "intermediate_size"),
        ({"num_hidden_layers": 92}, "num_hidden_layers"),
        ({"num_attention_heads": 64}, "num_attention_heads"),
        ({"num_experts": 256}, "num_experts"),
        ({"num_experts_per_token": 8}, "num_experts_per_token"),
        ({"attn_res_block_size": None}, "attn_res_block_size"),
        ({"first_k_dense_replace": 0}, "first_k_dense_replace"),
        ({"mla_use_output_gate": False}, "mla_use_output_gate"),
        ({"num_nextn_predict_layers": 1}, "num_nextn_predict_layers"),
        ({"routed_expert_hidden_size": 3072}, "routed_expert_hidden_size"),
        ({"linear_attn_config": {}}, "69-KDA/24-MLA"),
    ],
)
def test_kimi_k3_text_contract_fails_closed(overrides, match) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_kimi_k3_config(
            _config(text_overrides=overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"patch_size": 16}, "patch_size"),
        ({"merge_kernel_size": (1, 1)}, "merge_kernel_size"),
        ({"mm_projector_type": "linear"}, "mm_projector_type"),
        ({"vt_num_hidden_layers": 26}, "vt_num_hidden_layers"),
    ],
)
def test_kimi_k3_vision_contract_fails_closed(overrides, match) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_kimi_k3_config(
            _config(vision_overrides=overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"tensor_parallel_size": 16}, "TP32/PP1/DP1"),
        ({"pipeline_parallel_size": 2}, "TP32/PP1/DP1"),
        ({"data_parallel_size": 2}, "TP32/PP1/DP1"),
        ({"enable_expert_parallel": True}, "enable_expert_parallel"),
        ({"use_sequence_parallel_moe": True}, "use_sequence_parallel_moe"),
        ({"enable_eplb": True}, "enable_eplb"),
        ({"use_ubatching": True}, "use_ubatching"),
        ({"decode_context_parallel_size": 2}, "context parallelism"),
    ],
)
def test_kimi_k3_fails_closed_for_unverified_topologies(overrides, match) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_kimi_k3_config(
            _config(),
            _parallel(**overrides),
        )


def test_kimi_k3_rejects_wrong_quantizer_dtype_speculation_and_megamoe() -> None:
    with pytest.raises(NotImplementedError, match="compressed-tensors"):
        _require_supported_kimi_k3_config(
            _config(),
            _parallel(),
            quant_config=_quant_config("mxfp4"),
        )
    with pytest.raises(NotImplementedError, match="BF16"):
        _require_supported_kimi_k3_config(
            _config(),
            _parallel(),
            dtype=torch.float16,
        )
    with pytest.raises(NotImplementedError, match="speculative/MTP"):
        _require_supported_kimi_k3_config(
            _config(),
            _parallel(),
            speculative_config=object(),
        )
    with pytest.raises(NotImplementedError, match="SM100 MegaMoE"):
        _require_supported_kimi_k3_config(
            _config(),
            _parallel(),
            moe_backend="deep_gemm_mega_moe",
        )


def test_kimi_k3_accepts_vllm_normalized_official_config() -> None:
    config = KimiK3Config(
        text_config=_text_values(),
        vision_config=_vision_values(),
        architectures=["KimiK3ForConditionalGeneration"],
        bos_token_id=163_584,
        eos_token_id=163_586,
        pad_token_id=163_839,
        image_placeholder="<|kimi_image_placeholder|>",
        media_placeholder_token_id=163_605,
        tie_word_embeddings=False,
    )

    _require_supported_kimi_k3_config(
        config,
        _parallel(),
        quant_config=_quant_config(),
        dtype=torch.bfloat16,
        moe_backend="auto",
    )
    assert config.vision_config.merge_kernel_size == (2, 2)


def test_kimi_k3_model_shape_uses_nested_moe_geometry() -> None:
    shape = _make_model_shape_from_hf_config(_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 7168
    assert shape.num_heads == 96
    assert shape.num_kv_heads == 96
    assert shape.head_dim == 74
    assert shape.intermediate_dim == 33_792
    assert shape.num_experts == 896
    assert shape.top_k == 16


class _Attention(nn.Module):
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states + 1


class _MLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2


def test_kimi_k3_decoder_hooks_observe_uniform_authoritative_boundaries(
    monkeypatch,
) -> None:
    subject = KimiK3PDecoderLayer.__new__(KimiK3PDecoderLayer)
    nn.Module.__init__(subject)
    subject.use_sequence_parallel = False
    subject._self_attn_writes_output = False
    subject.self_attn = _Attention()
    subject.mlp = _MLP()
    _add_hooks(subject, ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"))
    residual = torch.ones(1, 8, 1)
    prefix_sum = torch.tensor([[2.0]])
    monkeypatch.setattr(
        subject,
        "_pre_attn_norm",
        lambda hidden, out_residual, prefix: (
            torch.tensor([[3.0]]),
            prefix,
            out_residual,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_post_attn_norm",
        lambda hidden, out_residual, prefix: (
            torch.tensor([[5.0]]),
            prefix,
            out_residual,
        ),
    )
    observed: dict[str, torch.Tensor] = {}
    for name in ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"):
        getattr(subject, f"hook_{name}").register_forward_hook(
            lambda _module, _inputs, output, name=name: observed.setdefault(
                name, output.clone()
            )
        )

    output, out_prefix, out_residual = subject(
        torch.tensor([0]),
        None,
        residual,
        prefix_sum,
    )

    assert {name: value.item() for name, value in observed.items()} == {
        "ln1": 3.0,
        "attn_out": 4.0,
        "ln2": 5.0,
        "mlp_in": 5.0,
        "mlp_out": 10.0,
    }
    assert output.item() == 10.0
    assert out_prefix is prefix_sum
    assert out_residual is residual


def test_kimi_k3_disabled_layer_hooks_delegate_to_upstream(monkeypatch) -> None:
    subject = KimiK3PDecoderLayer.__new__(KimiK3PDecoderLayer)
    nn.Module.__init__(subject)
    for name in ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"):
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = (torch.tensor([[17.0]]), None, torch.tensor([[19.0]]))
    observed = []

    def upstream_forward(_self, *args, **kwargs):
        observed.append((args, kwargs))
        return marker

    monkeypatch.setattr(KimiDecoderLayer, "forward", upstream_forward)
    args = (
        torch.tensor([0]),
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[3.0]]),
    )

    result = subject(*args)

    assert result is marker
    assert observed == [(args, {})]


def test_kimi_k3_manifest_is_reduced_and_exact() -> None:
    subject = _fake_language_model()
    specs = subject.get_hook_specs()
    model_wide = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 470
    assert len(model_wide) == 470
    assert all(spec.module is not None for spec in specs)
    assert all(spec.module is None for spec in model_wide)
    assert [spec.hook_type for spec in specs[:2]] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
    ]
    expected_layer = [
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_OUT,
    ]
    for layer_no in range(93):
        start = 2 + layer_no * len(expected_layer)
        layer_specs = specs[start : start + len(expected_layer)]
        assert [spec.hook_type for spec in layer_specs] == expected_layer
        assert {spec.layer_no for spec in layer_specs} == {layer_no}
    assert [spec.hook_type for spec in specs[-3:]] == [
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    omitted = {
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_ROUTER_LOGITS,
        HOOK_TYPE_TOPK_IDS,
        HOOK_TYPE_TOPK_WEIGHTS,
    }
    assert omitted.isdisjoint(spec.hook_type for spec in specs)


def test_kimi_k3_compare_buffers_match_reduced_manifest(monkeypatch) -> None:
    subject = KimiK3CompareForConditionalGeneration.__new__(
        KimiK3CompareForConditionalGeneration
    )
    nn.Module.__init__(subject)
    subject.language_model = _fake_language_model()
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    subject.allocate_compare_buffers(8, vllm_config)

    buffers = subject.get_ref_buffers()
    assert len(buffers) == 470
    assert buffers["embed"].shape == (8, 7168)
    assert buffers["ln1_L92"].shape == (8, 7168)
    assert buffers["final_logits"].shape == (4, 163_840)
    assert buffers["token_ids"].shape == (8,)
    assert not any(name.startswith(("resid_pre_", "resid_mid_")) for name in buffers)
    assert not any(name.startswith(("q_", "k_", "v_", "z_")) for name in buffers)
    assert not any(name.startswith(("router_", "topk_")) for name in buffers)
