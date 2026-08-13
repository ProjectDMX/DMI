"""CPU contracts for bounded Gemma 4 E2B decoder monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP  # isort: skip

from vllm.model_executor.models.gemma4 import (
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4Model,
)
from vllm.model_executor.models.gemma4_compare import (
    Gemma4CompareForConditionalGeneration,
)
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.model_executor.models.gemma4_p import (
    _EXPECTED_LAYER_TYPES,
    Gemma4PDecoderLayer,
    Gemma4PForCausalLM,
    Gemma4PForConditionalGeneration,
    Gemma4PModel,
    _require_supported_gemma4_e2b_config,
)

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
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)

pytestmark = pytest.mark.framework_fork


def _text_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "gemma4_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 1536,
        "hidden_size_per_layer_input": 256,
        "intermediate_size": 6144,
        "layer_types": list(_EXPECTED_LAYER_TYPES),
        "max_position_embeddings": 131072,
        "num_attention_heads": 8,
        "num_hidden_layers": 35,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 20,
        "quantization_config": None,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
        "sliding_window": 512,
        "tie_word_embeddings": True,
        "use_bidirectional_attention": None,
        "use_double_wide_mlp": True,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _outer_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "gemma4",
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
        "text_config": _text_config(),
        "vision_config": SimpleNamespace(
            model_type="gemma4_vision",
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=16,
            num_attention_heads=12,
            patch_size=16,
            pooling_kernel_size=3,
            position_embedding_size=10240,
            default_output_length=280,
        ),
        "audio_config": SimpleNamespace(
            model_type="gemma4_audio",
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=8,
            output_proj_dims=1536,
            conv_kernel_size=5,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parallel(**overrides) -> SimpleNamespace:
    values = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "use_sequence_parallel_moe": False,
        "enable_eplb": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _add_hooks(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


def _fake_language_model() -> Gemma4PForCausalLM:
    subject = Gemma4PForCausalLM.__new__(Gemma4PForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _text_config()
    _add_hooks(subject, ("token_ids", "final_logits"))

    model = nn.Module()
    model.start_layer = 0
    model.end_layer = 35
    model.layers = nn.ModuleList()
    _add_hooks(model, ("embed", "resid_final", "final_ln"))
    for _ in range(35):
        layer = nn.Module()
        _add_hooks(
            layer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
        )
        model.layers.append(layer)
    subject.model = model
    return subject


def test_gemma4_preserves_public_wrapper_and_loader_contracts() -> None:
    assert (
        _ARCH_REMAP["Gemma4ForConditionalGeneration"]
        == "Gemma4PForConditionalGeneration"
    )
    assert issubclass(
        Gemma4PForConditionalGeneration,
        Gemma4ForConditionalGeneration,
    )
    assert issubclass(
        Gemma4CompareForConditionalGeneration,
        Gemma4PForConditionalGeneration,
    )
    assert issubclass(Gemma4PForCausalLM, Gemma4ForCausalLM)
    assert issubclass(Gemma4PModel, Gemma4Model)
    assert issubclass(Gemma4PDecoderLayer, Gemma4DecoderLayer)
    assert (
        Gemma4PForConditionalGeneration.load_weights
        is Gemma4ForConditionalGeneration.load_weights
    )
    assert (
        Gemma4PForConditionalGeneration.get_mm_mapping
        is Gemma4ForConditionalGeneration.get_mm_mapping
    )
    assert (
        Gemma4PForConditionalGeneration.packed_modules_mapping
        == Gemma4ForConditionalGeneration.packed_modules_mapping
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"model_type": "gemma3"}, "model_type"),
        ({"hidden_activation": "silu"}, "hidden_activation"),
        ({"hidden_size": 2048}, "hidden_size"),
        ({"intermediate_size": 8192}, "intermediate_size"),
        ({"num_hidden_layers": 34}, "num_hidden_layers"),
        ({"num_attention_heads": 12}, "num_attention_heads"),
        ({"num_key_value_heads": 2}, "num_key_value_heads"),
        ({"head_dim": 128}, "head_dim"),
        ({"global_head_dim": 256}, "global_head_dim"),
        ({"num_kv_shared_layers": 0}, "num_kv_shared_layers"),
        ({"use_double_wide_mlp": False}, "use_double_wide_mlp"),
        ({"enable_moe_block": True}, "enable_moe_block"),
        ({"layer_types": ["full_attention"] * 35}, "layer_types"),
        ({"rope_parameters": {}}, "rope_parameters"),
        ({"quantization_config": {"quant_method": "fp8"}}, "BF16"),
    ],
)
def test_gemma4_text_fails_closed_outside_official_e2b_contract(
    overrides,
    match,
) -> None:
    config = _outer_config(text_config=_text_config(**overrides))
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_gemma4_e2b_config(config, _parallel())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"image_token_id": 1}, "image_token_id"),
        ({"vision_soft_tokens_per_image": 256}, "vision_soft_tokens_per_image"),
        ({"vision_config": None}, "image/audio towers"),
        ({"audio_config": None}, "image/audio towers"),
    ],
)
def test_gemma4_outer_tower_contract_fails_closed(overrides, match) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_gemma4_e2b_config(
            _outer_config(**overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"tensor_parallel_size": 2}, "TP1/PP1/DP1"),
        ({"pipeline_parallel_size": 2}, "TP1/PP1/DP1"),
        ({"data_parallel_size": 2}, "TP1/PP1/DP1"),
        ({"enable_expert_parallel": True}, "enable_expert_parallel"),
        ({"use_sequence_parallel_moe": True}, "use_sequence_parallel_moe"),
        ({"enable_eplb": True}, "enable_eplb"),
        ({"decode_context_parallel_size": 2}, "context parallelism"),
        ({"prefill_context_parallel_size": 2}, "context parallelism"),
    ],
)
def test_gemma4_lite_cell_fails_closed_for_unverified_topologies(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_gemma4_e2b_config(
            _outer_config(),
            _parallel(**overrides),
        )


def test_gemma4_rejects_quantization_fp16_speculation_and_fast_prefill() -> None:
    with pytest.raises(NotImplementedError, match="BF16 checkpoint"):
        _require_supported_gemma4_e2b_config(
            _outer_config(),
            _parallel(),
            quant_config=object(),
        )
    with pytest.raises(NotImplementedError, match="runtime dtype"):
        _require_supported_gemma4_e2b_config(
            _outer_config(),
            _parallel(),
            dtype=torch.float16,
        )
    with pytest.raises(NotImplementedError, match="speculative/Eagle"):
        _require_supported_gemma4_e2b_config(
            _outer_config(),
            _parallel(),
            speculative_config=object(),
        )
    with pytest.raises(NotImplementedError, match="fast prefill"):
        _require_supported_gemma4_e2b_config(
            _outer_config(),
            _parallel(),
            kv_sharing_fast_prefill=True,
        )


def test_gemma4_accepts_official_bf16_tp1_contract() -> None:
    _require_supported_gemma4_e2b_config(
        _outer_config(),
        _parallel(),
        dtype=torch.bfloat16,
    )


def test_gemma4_model_shape_uses_nested_decoder_geometry() -> None:
    shape = _make_model_shape_from_hf_config(_outer_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 1536
    assert shape.num_heads == 8
    assert shape.num_kv_heads == 1
    assert shape.head_dim == 256
    assert shape.intermediate_dim == 6144
    assert shape.vocab_size == 262144


class _Scale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return value * self.scale


class _Attention(nn.Module):
    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states + 1


def test_gemma4_decoder_hooks_observe_exact_upstream_boundaries() -> None:
    subject = Gemma4PDecoderLayer.__new__(Gemma4PDecoderLayer)
    nn.Module.__init__(subject)
    subject.input_layernorm = _Scale(2)
    subject.self_attn = _Attention()
    subject.post_attention_layernorm = _Scale(3)
    subject.pre_feedforward_layernorm = _Scale(4)
    subject.mlp = _Scale(5)
    subject.post_feedforward_layernorm = _Scale(6)
    subject.per_layer_input_gate = None
    subject.layer_scalar = torch.tensor([0.5])
    _add_hooks(
        subject,
        (
            "resid_pre",
            "ln1",
            "attn_out",
            "resid_mid",
            "ln2",
            "mlp_in",
            "mlp_out",
        ),
    )
    observed: dict[str, torch.Tensor] = {}
    for name in (
        "resid_pre",
        "ln1",
        "attn_out",
        "resid_mid",
        "ln2",
        "mlp_in",
        "mlp_out",
    ):
        getattr(subject, f"hook_{name}").register_forward_hook(
            lambda _module, _inputs, output, name=name: observed.setdefault(
                name, output.clone()
            )
        )

    output, residual = subject(
        torch.tensor([0]),
        torch.tensor([[2.0]]),
        None,
    )

    assert residual is None
    assert {name: value.item() for name, value in observed.items()} == {
        "resid_pre": 2.0,
        "ln1": 4.0,
        "attn_out": 15.0,
        "resid_mid": 17.0,
        "ln2": 68.0,
        "mlp_in": 68.0,
        "mlp_out": 2040.0,
    }
    assert output.item() == 1028.5


def test_gemma4_manifest_is_exact_and_omits_heterogeneous_shapes() -> None:
    subject = _fake_language_model()
    specs = subject.get_hook_specs()
    model_wide = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 250
    assert len(model_wide) == 250
    assert all(spec.module is not None for spec in specs)
    assert all(spec.module is None for spec in model_wide)
    assert [spec.hook_type for spec in specs[:2]] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
    ]
    expected_layer = [
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_OUT,
    ]
    for layer_no in range(35):
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
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_MLP_POST,
    }
    assert omitted.isdisjoint(spec.hook_type for spec in specs)


def test_gemma4_compare_buffer_inventory_matches_manifest(monkeypatch) -> None:
    subject = Gemma4CompareForConditionalGeneration.__new__(
        Gemma4CompareForConditionalGeneration
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
    assert len(buffers) == 250
    assert buffers["embed"].shape == (8, 1536)
    assert buffers["resid_pre_L34"].shape == (8, 1536)
    assert buffers["final_logits"].shape == (4, 262144)
    assert buffers["token_ids"].shape == (8,)
    assert not any(name.startswith(("q_", "k_", "v_", "z_")) for name in buffers)
    assert not any(name.startswith("mlp_post_") for name in buffers)
