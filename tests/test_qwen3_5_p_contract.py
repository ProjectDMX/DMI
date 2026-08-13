"""CPU contracts for decoder-only Qwen3.6-27B monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP  # isort: skip

from vllm.model_executor.models import qwen3_5_compare, qwen3_5_p
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_5_compare import (
    Qwen3_5CompareForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_5_p import (
    Qwen3_5PAttention,
    Qwen3_5PDecoderLayer,
    Qwen3_5PForCausalLM,
    Qwen3_5PForConditionalGeneration,
    Qwen3_5PMLP,
    Qwen3_5PModel,
    _require_supported_qwen36_config,
    _require_supported_qwen36_text_config,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
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


def _layer_types(count: int = 64) -> list[str]:
    return [
        "full_attention" if (layer_no + 1) % 4 == 0 else "linear_attention"
        for layer_no in range(count)
    ]


def _text_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "qwen3_5_text",
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 17_408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "max_position_embeddings": 262_144,
        "vocab_size": 248_320,
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "qkv_bias": False,
        "tie_word_embeddings": False,
        "layer_scale": None,
        "layer_types": _layer_types(),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        "quantization_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _outer_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "qwen3_5",
        "language_model_only": False,
        "text_config": _text_config(),
        "vision_config": SimpleNamespace(
            model_type="qwen3_5_vision",
            hidden_size=1152,
            intermediate_size=4304,
            hidden_act="gelu_pytorch_tanh",
            depth=27,
            num_heads=16,
            in_channels=3,
            num_position_embeddings=2304,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=5120,
            deepstack_visual_indexes=[],
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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen36_preserves_public_wrapper_and_loader_contracts() -> None:
    assert (
        _ARCH_REMAP["Qwen3_5ForConditionalGeneration"]
        == "Qwen3_5PForConditionalGeneration"
    )
    assert issubclass(
        Qwen3_5PForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
    )
    assert issubclass(
        Qwen3_5CompareForConditionalGeneration,
        Qwen3_5PForConditionalGeneration,
    )
    assert issubclass(Qwen3_5PForCausalLM, Qwen3_5ForCausalLM)
    assert issubclass(Qwen3_5PModel, Qwen3_5Model)
    assert issubclass(Qwen3_5PDecoderLayer, Qwen3_5DecoderLayer)
    assert issubclass(Qwen3_5PAttention, Qwen3NextAttention)
    assert issubclass(Qwen3_5PMLP, Qwen3NextMLP)
    assert (
        Qwen3_5PForConditionalGeneration.load_weights
        is Qwen3_5ForConditionalGeneration.load_weights
    )
    assert (
        Qwen3_5PForConditionalGeneration.get_mm_mapping
        is Qwen3_5ForConditionalGeneration.get_mm_mapping
    )
    assert (
        Qwen3_5PForConditionalGeneration.packed_modules_mapping
        == Qwen3_5ForConditionalGeneration.packed_modules_mapping
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"model_type": "qwen3_next"}, "model_type"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"hidden_size": 4096}, "hidden_size"),
        ({"intermediate_size": 16_384}, "intermediate_size"),
        ({"num_hidden_layers": 60}, "num_hidden_layers"),
        ({"num_attention_heads": 32}, "num_attention_heads"),
        ({"num_key_value_heads": 8}, "num_key_value_heads"),
        ({"head_dim": 128}, "head_dim"),
        ({"linear_num_key_heads": 8}, "linear_num_key_heads"),
        ({"linear_num_value_heads": 32}, "linear_num_value_heads"),
        ({"linear_conv_kernel_dim": 3}, "linear_conv_kernel_dim"),
        ({"attn_output_gate": False}, "attn_output_gate"),
        ({"output_gate_type": "silu"}, "output_gate_type"),
        ({"attention_dropout": 0.1}, "attention_dropout"),
        ({"rms_norm_eps": 1e-5}, "rms_norm_eps"),
        ({"attention_bias": True}, "bias-free"),
        ({"qkv_bias": True}, "bias-free QKV"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"layer_scale": 1.0}, "layer scaling"),
        ({"layer_types": ["full_attention"] * 64}, "3-linear/1-full"),
        ({"rope_parameters": None}, "interleaved MRoPE"),
        (
            {
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1_000_000,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10],
                    "mrope_interleaved": True,
                }
            },
            "interleaved MRoPE",
        ),
        ({"quantization_config": {"quant_method": "fp8"}}, "BF16"),
    ],
)
def test_qwen36_text_fails_closed_outside_the_27b_contract(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_qwen36_text_config(
            _text_config(**overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"tensor_parallel_size": 2}, "TP1/PP1/DP1"),
        ({"pipeline_parallel_size": 2}, "TP1/PP1/DP1"),
        ({"data_parallel_size": 2}, "TP1/PP1/DP1"),
        ({"enable_expert_parallel": True}, "TP1/PP1/DP1"),
        ({"use_sequence_parallel_moe": True}, "sequence-parallel"),
    ],
)
def test_qwen36_lite_cell_fails_closed_for_unverified_topologies(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_qwen36_text_config(
            _text_config(),
            _parallel(**overrides),
        )


def test_qwen36_lite_cell_rejects_runtime_fp16() -> None:
    with pytest.raises(NotImplementedError, match="runtime dtype"):
        _require_supported_qwen36_text_config(
            _text_config(),
            _parallel(),
            dtype=torch.float16,
        )


@pytest.mark.parametrize(
    "outer_overrides,match",
    [
        ({"model_type": "qwen3_vl"}, "model_type"),
        ({"language_model_only": True}, "multimodal public wrapper"),
        ({"vision_config": None}, "text and vision"),
        (
            {
                "vision_config": SimpleNamespace(
                    model_type="qwen3_5_vision",
                    hidden_size=1024,
                    intermediate_size=4304,
                    hidden_act="gelu_pytorch_tanh",
                    depth=27,
                    num_heads=16,
                    in_channels=3,
                    num_position_embeddings=2304,
                    patch_size=16,
                    spatial_merge_size=2,
                    temporal_patch_size=2,
                    out_hidden_size=5120,
                    deepstack_visual_indexes=[],
                )
            },
            "vision hidden_size",
        ),
        (
            {
                "vision_config": SimpleNamespace(
                    model_type="qwen3_5_vision",
                    hidden_size=1152,
                    intermediate_size=4304,
                    hidden_act="gelu_pytorch_tanh",
                    depth=27,
                    num_heads=16,
                    in_channels=3,
                    num_position_embeddings=2304,
                    patch_size=16,
                    spatial_merge_size=2,
                    temporal_patch_size=2,
                    out_hidden_size=5120,
                    deepstack_visual_indexes=[7],
                )
            },
            "deepstack",
        ),
    ],
)
def test_qwen36_outer_contract_fails_closed(
    outer_overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_qwen36_config(
            _outer_config(**outer_overrides),
            _parallel(),
        )


def test_qwen36_accepts_the_official_decoder_and_vision_identity() -> None:
    _require_supported_qwen36_config(_outer_config(), _parallel())


def test_qwen36_model_shape_uses_the_nested_text_config() -> None:
    shape = _make_model_shape_from_hf_config(_outer_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 5120
    assert shape.intermediate_dim == 17_408
    assert shape.num_heads == 24
    assert shape.num_kv_heads == 4
    assert shape.head_dim == 256
    assert shape.num_experts == 0
    assert shape.top_k == 0


def test_qwen36_mlp_post_hook_observes_post_activation_values() -> None:
    subject = Qwen3_5PMLP.__new__(Qwen3_5PMLP)
    nn.Module.__init__(subject)

    class GateUp(nn.Module):
        def forward(self, value):
            return value + 2, None

    class Activation(nn.Module):
        def forward(self, value):
            return value * 3

    class Down(nn.Module):
        def forward(self, value):
            return value - 5, None

    subject.gate_up_proj = GateUp()
    subject.act_fn = Activation()
    subject.down_proj = Down()
    subject.expert_gate = None
    subject.hook_post = HookPoint()
    captured = []
    subject.hook_post.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )
    inputs = torch.tensor([[1.0, 2.0]])

    output = subject(inputs)

    expected_post = (inputs + 2) * 3
    assert torch.equal(captured[0], expected_post)
    assert torch.equal(output, expected_post - 5)


def test_qwen36_full_attention_hooks_observe_gated_decoder_values() -> None:
    subject = Qwen3_5PAttention.__new__(Qwen3_5PAttention)
    nn.Module.__init__(subject)
    subject.num_heads = 2
    subject.num_kv_heads = 1
    subject.head_dim = 2

    class Projection(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    class Attention(nn.Module):
        def forward(self, q, k, v):
            assert k.shape == v.shape == (2, 2)
            return q + 1

    class OutputProjection(nn.Module):
        def forward(self, value):
            return value + 10, None

    subject.qkv_proj = Projection()
    subject._project_qkv_gate = lambda _qkv, _positions: (
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        torch.arange(4, dtype=torch.float32).reshape(2, 2) + 20,
        torch.zeros(2, 4),
    )
    subject.attn = Attention()
    subject.o_proj = OutputProjection()
    captured = {}
    for name in ("q", "k", "v", "z"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output.clone(),
            )
        )
        setattr(subject, f"hook_{name}", hook)

    output = subject(torch.tensor([0]), torch.zeros(2, 4))

    expected_z = (torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1) * 0.5
    assert captured["q"].shape == (2, 2, 2)
    assert captured["k"].shape == captured["v"].shape == (2, 1, 2)
    assert torch.equal(captured["z"], expected_z)
    assert torch.equal(output, expected_z + 10)


def test_qwen36_multimodal_wrapper_delegates_only_decoder_manifest() -> None:
    subject = Qwen3_5PForConditionalGeneration.__new__(Qwen3_5PForConditionalGeneration)
    nn.Module.__init__(subject)
    sentinel = [object()]
    calls = []
    subject.language_model = SimpleNamespace(
        get_hook_specs=lambda **kwargs: calls.append(kwargs) or sentinel
    )
    subject.visual = SimpleNamespace()

    result = subject.get_hook_specs(model_wide=True)

    assert result is sentinel
    assert calls == [{"model_wide": True}]
    assert not hasattr(subject.visual, "get_hook_specs")


def test_qwen36_outer_forward_captures_tokens_before_decoder_bypass(
    monkeypatch,
) -> None:
    subject = Qwen3_5PForConditionalGeneration.__new__(Qwen3_5PForConditionalGeneration)
    nn.Module.__init__(subject)
    hook = HookPoint()
    captured = []
    hook.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )
    subject.language_model = SimpleNamespace(hook_token_ids=hook)
    monkeypatch.setattr(
        qwen3_5_p,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    marker = torch.tensor([[19.0]])
    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration,
        "forward",
        lambda _self, *args, **kwargs: marker,
    )
    input_ids = torch.tensor([1, 2], dtype=torch.int32)

    output = subject(input_ids, torch.tensor([0, 1]))

    assert output is marker
    assert torch.equal(captured[0], input_ids)


def test_qwen36_compare_buffers_cover_heterogeneous_decoder(monkeypatch) -> None:
    subject = Qwen3_5CompareForConditionalGeneration.__new__(
        Qwen3_5CompareForConditionalGeneration
    )
    nn.Module.__init__(subject)
    config = _text_config(num_hidden_layers=4, layer_types=_layer_types(4))
    layers = []
    for layer_type in config.layer_types:
        layer = SimpleNamespace(layer_type=layer_type, mlp=SimpleNamespace())
        if layer_type == "full_attention":
            layer.self_attn = SimpleNamespace()
        layers.append(layer)
    subject.language_model = SimpleNamespace(
        config=config,
        model=SimpleNamespace(start_layer=0, end_layer=4, layers=layers),
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(qwen3_5_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        qwen3_5_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 41
    assert len(buffers) == 41
    assert "q_L0" not in buffers
    assert buffers["q_L3"].shape == (16, 24, 256)
    assert buffers["k_L3"].shape == (16, 4, 256)
    assert buffers["v_L3"].shape == (16, 4, 256)
    assert buffers["z_L3"].shape == (16, 6144)
    assert buffers["mlp_post_L0"].shape == (16, 17_408)
    assert buffers["final_logits"].shape == (4, 248_320)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names", "args", "kwargs"),
    [
        (
            Qwen3_5PAttention,
            Qwen3NextAttention,
            ("q", "k", "v", "z"),
            (torch.tensor([0]), torch.tensor([[1.0]])),
            {},
        ),
        (
            Qwen3_5PMLP,
            Qwen3NextMLP,
            ("post",),
            (torch.tensor([[1.0]]),),
            {},
        ),
        (
            Qwen3_5PDecoderLayer,
            Qwen3_5DecoderLayer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
            (torch.tensor([[1.0]]), None),
            {"positions": torch.tensor([0])},
        ),
    ],
)
def test_qwen36_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
    args,
    kwargs,
) -> None:
    subject = subject_cls.__new__(subject_cls)
    nn.Module.__init__(subject)
    for name in hook_names:
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, *forward_args, **forward_kwargs):
        observed.append((forward_args, forward_kwargs))
        return marker

    monkeypatch.setattr(upstream_cls, "forward", upstream_forward)

    result = subject(*args, **kwargs)

    assert result is marker
    assert observed == [(args, kwargs)]


def test_qwen36_model_wide_manifest_has_581_truthful_families() -> None:
    subject = Qwen3_5PForCausalLM.__new__(Qwen3_5PForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _text_config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=64,
        layers=[None] * 64,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    common = [
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
    ]
    full_only = [HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V, HOOK_TYPE_Z]
    expected = [HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_EMBED]
    for layer_no, layer_type in enumerate(subject.config.layer_types):
        layer = common[:2]
        if layer_type == "full_attention":
            layer += full_only
        layer += common[2:]
        expected.extend(layer)
    expected.extend([HOOK_TYPE_RESID_FINAL, HOOK_TYPE_FINAL_LN, HOOK_TYPE_FINAL_LOGITS])

    specs = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 581
    assert [spec.hook_type for spec in specs] == expected
    assert all(spec.module is None for spec in specs)
    assert sum(spec.hook_type == HOOK_TYPE_Q for spec in specs) == 16
    assert sum(spec.hook_type == HOOK_TYPE_MLP_POST for spec in specs) == 64
