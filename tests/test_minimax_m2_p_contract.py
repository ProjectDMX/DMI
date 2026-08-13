"""CPU contracts for bounded MiniMax-M2.7 monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP  # isort: skip

from vllm.model_executor.models import minimax_m2_compare, minimax_m2_p
from vllm.model_executor.models.minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2DecoderLayer,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2MoE,
)
from vllm.model_executor.models.minimax_m2_compare import (
    MiniMaxM2CompareForCausalLM,
)
from vllm.model_executor.models.minimax_m2_p import (
    MiniMaxM2PAttention,
    MiniMaxM2PDecoderLayer,
    MiniMaxM2PForCausalLM,
    MiniMaxM2PModel,
    MiniMaxM2PMoE,
    _require_supported_minimax_m27_config,
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


def _quantization_config() -> dict:
    return {
        "activation_scheme": "dynamic",
        "fmt": "float8_e4m3fn",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "gate",
            "e_score_correction_bias",
            "lm_head",
        ],
    }


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "minimax_m2",
        "hidden_act": "silu",
        "hidden_size": 3072,
        "intermediate_size": 1536,
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rotary_dim": 64,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "shared_intermediate_size": 0,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "output_router_logits": False,
        "use_qk_norm": True,
        "qk_norm_type": "per_layer",
        "max_position_embeddings": 204_800,
        "vocab_size": 200_064,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "router_jitter_noise": 0.0,
        "num_mtp_modules": 3,
        "mtp_transformer_layers": 1,
        "use_mtp": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "attn_type_list": [1] * 62,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 5_000_000,
        },
        "quantization_config": _quantization_config(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parallel(**overrides) -> SimpleNamespace:
    values = {
        "tensor_parallel_size": 4,
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


def test_minimax_m27_preserves_upstream_class_and_loader_contracts() -> None:
    assert _ARCH_REMAP["MiniMaxM2ForCausalLM"] == "MiniMaxM2PForCausalLM"
    assert issubclass(MiniMaxM2PForCausalLM, MiniMaxM2ForCausalLM)
    assert issubclass(MiniMaxM2CompareForCausalLM, MiniMaxM2PForCausalLM)
    assert issubclass(MiniMaxM2PModel, MiniMaxM2Model)
    assert issubclass(MiniMaxM2PDecoderLayer, MiniMaxM2DecoderLayer)
    assert issubclass(MiniMaxM2PAttention, MiniMaxM2Attention)
    assert issubclass(MiniMaxM2PMoE, MiniMaxM2MoE)
    assert MiniMaxM2PForCausalLM.load_weights is MiniMaxM2ForCausalLM.load_weights
    assert (
        MiniMaxM2PForCausalLM.packed_modules_mapping
        == MiniMaxM2ForCausalLM.packed_modules_mapping
    )
    assert vars(MiniMaxM2PForCausalLM.hf_to_vllm_mapper) == vars(
        MiniMaxM2ForCausalLM.hf_to_vllm_mapper
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"model_type": "minimax_m1"}, "model_type"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"hidden_size": 4096}, "hidden_size"),
        ({"num_hidden_layers": 60}, "num_hidden_layers"),
        ({"num_attention_heads": 32}, "num_attention_heads"),
        ({"num_key_value_heads": 4}, "num_key_value_heads"),
        ({"head_dim": 64}, "head_dim"),
        ({"rotary_dim": 128}, "rotary_dim"),
        ({"num_local_experts": 128}, "num_local_experts"),
        ({"num_experts_per_tok": 4}, "num_experts_per_tok"),
        ({"shared_intermediate_size": 512}, "shared_intermediate_size"),
        ({"scoring_func": "softmax"}, "scoring_func"),
        ({"use_routing_bias": False}, "use_routing_bias"),
        ({"output_router_logits": True}, "output_router_logits"),
        ({"use_qk_norm": False}, "use_qk_norm"),
        ({"qk_norm_type": "per_head"}, "qk_norm_type"),
        ({"attention_bias": True}, "bias-free attention"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"attn_type_list": [0] * 62}, "full attention"),
        ({"rope_parameters": {"rope_type": "yarn"}}, "half-head RoPE"),
        (
            {
                "quantization_config": {
                    **_quantization_config(),
                    "weight_block_size": [64, 64],
                }
            },
            "block-FP8",
        ),
    ],
)
def test_minimax_m27_fails_closed_outside_the_official_contract(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_minimax_m27_config(
            _config(**overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"tensor_parallel_size": 2}, "TP4/PP1/DP1"),
        ({"tensor_parallel_size": 8}, "TP4/PP1/DP1"),
        ({"pipeline_parallel_size": 2}, "TP4/PP1/DP1"),
        ({"data_parallel_size": 2}, "TP4/PP1/DP1"),
        ({"enable_expert_parallel": True}, "TP4/PP1/DP1"),
        ({"use_sequence_parallel_moe": True}, "sequence-parallel"),
        ({"enable_eplb": True}, "EPLB"),
        ({"decode_context_parallel_size": 2}, "context parallelism"),
        ({"prefill_context_parallel_size": 2}, "context parallelism"),
    ],
)
def test_minimax_m27_fails_closed_for_unverified_topologies(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_minimax_m27_config(
            _config(),
            _parallel(**overrides),
        )


def test_minimax_m27_rejects_wrong_quantizer_dtype_and_speculation() -> None:
    with pytest.raises(NotImplementedError, match="vLLM FP8"):
        _require_supported_minimax_m27_config(
            _config(),
            _parallel(),
            quant_config=SimpleNamespace(get_name=lambda: "gptq"),
        )
    with pytest.raises(NotImplementedError, match="runtime dtype"):
        _require_supported_minimax_m27_config(
            _config(),
            _parallel(),
            dtype=torch.float16,
        )
    with pytest.raises(NotImplementedError, match="speculative/MTP"):
        _require_supported_minimax_m27_config(
            _config(),
            _parallel(),
            speculative_config=object(),
        )


def test_minimax_m27_accepts_official_fp8_bf16_tp4_contract() -> None:
    _require_supported_minimax_m27_config(
        _config(),
        _parallel(),
        quant_config=SimpleNamespace(get_name=lambda: "fp8"),
        dtype=torch.bfloat16,
    )
    normalized = _config(
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5_000_000,
            "partial_rotary_factor": 0.5,
        }
    )
    _require_supported_minimax_m27_config(normalized, _parallel())


def test_minimax_m27_model_shape_reads_moe_geometry() -> None:
    shape = _make_model_shape_from_hf_config(_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 3072
    assert shape.num_heads == 48
    assert shape.num_kv_heads == 8
    assert shape.head_dim == 128
    assert shape.num_experts == 256
    assert shape.top_k == 8


def test_minimax_m27_attention_hooks_observe_normalized_pre_rope_qkv(
    monkeypatch,
) -> None:
    subject = MiniMaxM2PAttention.__new__(MiniMaxM2PAttention)
    nn.Module.__init__(subject)
    subject.num_heads = 2
    subject.num_kv_heads = 1
    subject.head_dim = 2
    subject.q_size = 4
    subject.kv_size = 2
    subject.q_norm = object()
    subject.k_norm = object()

    class Projection(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    class Rotary(nn.Module):
        def forward(self, _positions, q, k):
            return q + 100, k + 200

    class Attention(nn.Module):
        def forward(self, q, k, v):
            assert q.min() >= 100
            assert k.min() >= 200
            assert v.shape == (2, 2)
            return q - 90

    class OutputProjection(nn.Module):
        def forward(self, value):
            return value + 5, None

    q = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    k = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10
    v = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 20
    monkeypatch.setattr(
        minimax_m2_p.MiniMaxText01RMSNormTP,
        "forward_qkv",
        lambda *_args: (q, k, v),
    )
    subject.qkv_proj = Projection()
    subject.rotary_emb = Rotary()
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

    output = subject(torch.tensor([0]), torch.zeros(2, 8))

    assert torch.equal(captured["q"].flatten(1), q)
    assert torch.equal(captured["k"].flatten(1), k)
    assert torch.equal(captured["v"].flatten(1), v)
    assert torch.equal(captured["z"], q + 10)
    assert torch.equal(output, q + 15)


def test_minimax_m27_routing_hooks_observe_fp32_gate_values() -> None:
    subject = MiniMaxM2PMoE.__new__(MiniMaxM2PMoE)
    nn.Module.__init__(subject)
    events = []

    class Gate(nn.Module):
        def forward(self, hidden_states):
            events.append("gate")
            return (
                torch.arange(
                    hidden_states.shape[0] * 4,
                    dtype=torch.float32,
                ).reshape(hidden_states.shape[0], 4),
                None,
            )

    class Router:
        def select_experts(self, *, hidden_states, router_logits):
            events.append("select")
            return (
                torch.full((2, 2), 0.5, dtype=torch.float32),
                torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = Router()

        def forward(self, *, hidden_states, router_logits):
            events.append("experts")
            assert router_logits.dtype == torch.float32
            return hidden_states + 1

    subject.gate = Gate()
    subject.experts = Experts()
    captured = {}
    for name in ("router_logits", "topk_ids", "topk_weights"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output.clone(),
            )
        )
        setattr(subject, f"hook_{name}", hook)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = subject(hidden_states)

    assert events == ["gate", "select", "experts"]
    assert torch.equal(output, hidden_states + 1)
    assert captured["router_logits"].dtype == torch.float32
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_weights"].dtype == torch.float32


def test_minimax_m27_compare_buffers_cover_every_family(monkeypatch) -> None:
    subject = MiniMaxM2CompareForCausalLM.__new__(MiniMaxM2CompareForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config(num_hidden_layers=2)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(),
                block_sparse_moe=SimpleNamespace(),
            )
            for _ in range(2)
        ],
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(minimax_m2_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        minimax_m2_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 33
    assert len(buffers) == 33
    assert buffers["q_L0"].shape == (16, 12, 128)
    assert buffers["k_L0"].shape == (16, 2, 128)
    assert buffers["v_L0"].shape == (16, 2, 128)
    assert buffers["z_L0"].shape == (16, 1536)
    assert buffers["router_logits_L0"].shape == (16, 256)
    assert buffers["router_logits_L0"].dtype == torch.float32
    assert buffers["topk_ids_L0"].dtype == torch.int32
    assert buffers["topk_weights_L0"].dtype == torch.float32
    assert buffers["final_logits"].shape == (4, 200_064)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names", "args"),
    [
        (
            MiniMaxM2PAttention,
            MiniMaxM2Attention,
            ("q", "k", "v", "z"),
            (torch.tensor([0]), torch.tensor([[1.0]])),
        ),
        (
            MiniMaxM2PMoE,
            MiniMaxM2MoE,
            ("router_logits", "topk_ids", "topk_weights"),
            (torch.tensor([[1.0]]),),
        ),
        (
            MiniMaxM2PDecoderLayer,
            MiniMaxM2DecoderLayer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
            (torch.tensor([0]), torch.tensor([[1.0]]), None),
        ),
    ],
)
def test_minimax_m27_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
    args,
) -> None:
    subject = subject_cls.__new__(subject_cls)
    nn.Module.__init__(subject)
    for name in hook_names:
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, *forward_args):
        observed.append(forward_args)
        return marker

    monkeypatch.setattr(upstream_cls, "forward", upstream_forward)

    result = subject(*args)

    assert result is marker
    assert observed == [args]


def test_minimax_m27_model_wide_manifest_has_873_families() -> None:
    subject = MiniMaxM2PForCausalLM.__new__(MiniMaxM2PForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=62,
        layers=[None] * 62,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    per_layer = [
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

    assert len(specs) == 873
    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(per_layer * 62),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is None for spec in specs)
