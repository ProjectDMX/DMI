"""CPU contracts for bounded GPT-OSS monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP  # isort: skip

from vllm.model_executor.models import gpt_oss_compare, gpt_oss_p
from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    MLPBlock,
    OAIAttention,
    TransformerBlock,
)
from vllm.model_executor.models.gpt_oss_compare import (
    GptOssCompareForCausalLM,
)
from vllm.model_executor.models.gpt_oss_p import (
    GptOssPAttention,
    GptOssPForCausalLM,
    GptOssPMLP,
    GptOssPModel,
    GptOssPTransformerBlock,
    _require_supported_gpt_oss_config,
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


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "gpt_oss",
        "hidden_act": "silu",
        "attention_bias": True,
        "tie_word_embeddings": False,
        "output_router_logits": False,
        "swiglu_limit": 7,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "experts_per_token": 4,
        "sliding_window": 128,
        "vocab_size": 201_088,
        "rope_theta": 150_000,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 150_000,
            "factor": 32,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "truncate": False,
        },
        "quantization_config": {"quant_method": "mxfp4"},
    }
    values.update(overrides)
    if "layer_types" not in overrides:
        values["layer_types"] = [
            "sliding_attention" if layer_no % 2 == 0 else "full_attention"
            for layer_no in range(values["num_hidden_layers"])
        ]
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


def test_gpt_oss_preserves_upstream_class_and_loader_contracts() -> None:
    assert _ARCH_REMAP["GptOssForCausalLM"] == "GptOssPForCausalLM"
    assert issubclass(GptOssPForCausalLM, GptOssForCausalLM)
    assert issubclass(GptOssCompareForCausalLM, GptOssPForCausalLM)
    assert issubclass(GptOssPModel, GptOssModel)
    assert issubclass(GptOssPTransformerBlock, TransformerBlock)
    assert issubclass(GptOssPAttention, OAIAttention)
    assert issubclass(GptOssPMLP, MLPBlock)
    assert (
        GptOssPForCausalLM.packed_modules_mapping
        == GptOssForCausalLM.packed_modules_mapping
    )
    assert vars(GptOssPForCausalLM.hf_to_vllm_mapper) == vars(
        GptOssForCausalLM.hf_to_vllm_mapper
    )
    assert GptOssPForCausalLM.is_3d_moe_weight is True


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"model_type": "llama"}, "model_type"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"attention_bias": False}, "biased attention"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"swiglu_limit": 0}, "swiglu_limit"),
        ({"num_local_experts": 0}, "local-expert"),
        ({"num_experts_per_tok": 33, "experts_per_token": 33}, "top-k"),
        ({"experts_per_token": 2}, "aliases"),
        ({"layer_types": ["full_attention"] * 24}, "alternating"),
        ({"sliding_window": 0}, "sliding window"),
        ({"num_key_value_heads": 7}, "GQA"),
        ({"rope_parameters": None}, "YaRN"),
        (
            {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 10_000,
                    "factor": 32,
                    "original_max_position_embeddings": 4096,
                    "beta_fast": 32,
                    "beta_slow": 1,
                }
            },
            "theta",
        ),
        ({"quantization_config": {"quant_method": "quark"}}, "BF16 or native"),
    ],
)
def test_gpt_oss_fails_closed_outside_the_audited_contract(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_gpt_oss_config(_config(**overrides), _parallel())


def test_gpt_oss_rejects_sequence_parallel_routing_in_lite_cell() -> None:
    with pytest.raises(NotImplementedError, match="sequence-parallel"):
        _require_supported_gpt_oss_config(
            _config(),
            _parallel(use_sequence_parallel_moe=True),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"tensor_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"data_parallel_size": 2},
        {"enable_expert_parallel": True},
    ],
)
def test_gpt_oss_lite_cell_fails_closed_for_unverified_topologies(
    overrides,
) -> None:
    with pytest.raises(NotImplementedError, match="TP1/PP1/DP1"):
        _require_supported_gpt_oss_config(_config(), _parallel(**overrides))


def test_gpt_oss_accepts_official_mxfp4_and_unquantized_contracts() -> None:
    _require_supported_gpt_oss_config(_config(), _parallel())
    _require_supported_gpt_oss_config(
        _config(quantization_config=None),
        _parallel(),
    )


def test_gpt_oss_model_shape_reads_local_expert_geometry() -> None:
    shape = _make_model_shape_from_hf_config(_config(), torch.bfloat16)

    assert shape is not None
    assert shape.num_experts == 32
    assert shape.top_k == 4
    assert shape.head_dim == 64


def test_gpt_oss_routing_hooks_observe_authoritative_router_inputs(
    monkeypatch,
) -> None:
    subject = GptOssPMLP.__new__(GptOssPMLP)
    nn.Module.__init__(subject)
    subject.is_sequence_parallel = False
    subject.hidden_size = 4
    events = []

    class Router(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            events.append("router")
            return torch.arange(
                hidden_states.shape[0] * 4,
                dtype=torch.float32,
            ).reshape(hidden_states.shape[0], 4)

    class ExpertRouter:
        def select_experts(self, *, hidden_states, router_logits):
            events.append("select")
            assert hidden_states.shape == (2, 4)
            assert router_logits.shape == (2, 4)
            return (
                torch.full((2, 2), 0.5, dtype=torch.float32),
                torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            )

    class Experts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = ExpertRouter()

        def forward(self, *, hidden_states, router_logits):
            events.append("experts")
            assert router_logits.shape == (2, 4)
            return hidden_states + 1

    subject.router = Router()
    subject.experts = Experts()
    captured = {}
    for name in ("router_logits", "topk_ids", "topk_weights"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output,
            )
        )
        setattr(subject, f"hook_{name}", hook)
    monkeypatch.setattr(
        gpt_oss_p,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: False),
    )
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = subject(hidden_states)

    assert events == ["router", "select", "experts"]
    assert torch.equal(output, hidden_states + 1)
    assert captured["router_logits"].shape == (2, 4)
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_weights"].dtype == torch.float32


def test_gpt_oss_compare_buffers_cover_every_manifest_family(monkeypatch) -> None:
    subject = GptOssCompareForCausalLM.__new__(GptOssCompareForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config(num_hidden_layers=2)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[
            SimpleNamespace(attn=SimpleNamespace(), mlp=SimpleNamespace())
            for _ in range(2)
        ],
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(gpt_oss_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        gpt_oss_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 33
    assert len(buffers) == 33
    assert buffers["q_L0"].shape == (16, 32, 64)
    assert buffers["k_L0"].shape == (16, 4, 64)
    assert buffers["router_logits_L0"].shape == (16, 32)
    assert buffers["topk_ids_L0"].dtype == torch.int32
    assert buffers["topk_weights_L0"].dtype == torch.float32
    assert buffers["final_logits"].shape == (4, 201_088)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names"),
    [
        (GptOssPAttention, OAIAttention, ("q", "k", "v", "z")),
        (
            GptOssPMLP,
            MLPBlock,
            ("router_logits", "topk_ids", "topk_weights"),
        ),
        (
            GptOssPTransformerBlock,
            TransformerBlock,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
        ),
    ],
)
def test_gpt_oss_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
) -> None:
    subject = subject_cls.__new__(subject_cls)
    nn.Module.__init__(subject)
    for name in hook_names:
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, *args):
        observed.append(args)
        return marker

    monkeypatch.setattr(upstream_cls, "forward", upstream_forward)
    args = (
        (torch.tensor([[1.0]]), torch.tensor([0]))
        if subject_cls is GptOssPAttention
        else (
            (torch.tensor([[1.0]]),)
            if subject_cls is GptOssPMLP
            else (torch.tensor([[1.0]]), torch.tensor([0]), None)
        )
    )

    result = subject(*args)

    assert result is marker
    assert observed == [args]


def test_gpt_oss_model_wide_manifest_has_341_truthful_families() -> None:
    subject = GptOssPForCausalLM.__new__(GptOssPForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=24,
        layers=[None] * 24,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    layer_types = [
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_ROUTER_LOGITS,
        HOOK_TYPE_TOPK_IDS,
        HOOK_TYPE_TOPK_WEIGHTS,
        HOOK_TYPE_MLP_OUT,
    ]

    specs = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 341
    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(layer_types * 24),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is None for spec in specs)
    assert specs[13].dtype == torch.int32
    assert specs[14].dtype == torch.float32
