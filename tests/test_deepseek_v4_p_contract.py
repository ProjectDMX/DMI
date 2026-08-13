"""CPU contracts for bounded DeepSeek V4 Flash monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP  # isort: skip

from vllm.model_executor.models import deepseek_v4_p
from vllm.model_executor.models.deepseek_v4_compare import (
    DeepseekV4CompareForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_p import (
    _EXPECTED_COMPRESS_RATIOS,
    DeepseekV4PDecoderLayer,
    DeepseekV4PForCausalLM,
    DeepseekV4PModel,
    _require_supported_deepseek_v4_flash_config,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
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
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "expert_dtype": "fp4",
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "head_dim": 512,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "max_position_embeddings": 1_048_576,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 43,
        "num_hash_layers": 3,
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 1,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65_536,
            "type": "yarn",
        },
        "rope_theta": 10_000,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "sliding_window": 128,
        "swiglu_limit": 10.0,
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
        "vocab_size": 129_280,
        "compress_rope_theta": 160_000,
        "compress_ratios": list(_EXPECTED_COMPRESS_RATIOS),
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
        "use_ubatching": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quant_config(
    name: str = "deepseek_v4_fp8",
    block_size: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        get_name=lambda: name,
        weight_block_size=block_size or [128, 128],
    )


def _add_hooks(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


def _fake_model() -> DeepseekV4PForCausalLM:
    subject = DeepseekV4PForCausalLM.__new__(DeepseekV4PForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config()
    _add_hooks(subject, ("token_ids", "final_logits"))

    model = nn.Module()
    model.start_layer = 0
    model.end_layer = 43
    model.layers = nn.ModuleList()
    _add_hooks(model, ("embed", "resid_final", "final_ln"))
    for _ in range(43):
        layer = nn.Module()
        _add_hooks(layer, ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"))
        model.layers.append(layer)
    subject.model = model
    return subject


def test_deepseek_v4_preserves_plugin_and_loader_contracts() -> None:
    assert _ARCH_REMAP["DeepseekV4ForCausalLM"] == "DeepseekV4PForCausalLM"
    assert issubclass(DeepseekV4PForCausalLM, DeepseekV4ForCausalLM)
    assert issubclass(
        DeepseekV4CompareForCausalLM,
        DeepseekV4PForCausalLM,
    )
    assert issubclass(DeepseekV4PModel, DeepseekV4Model)
    assert issubclass(DeepseekV4PDecoderLayer, DeepseekV4DecoderLayer)
    assert DeepseekV4PForCausalLM.load_weights is DeepseekV4ForCausalLM.load_weights
    assert (
        DeepseekV4PForCausalLM.get_expert_mapping
        is DeepseekV4ForCausalLM.get_expert_mapping
    )
    assert (
        DeepseekV4PForCausalLM.hf_to_vllm_mapper
        is DeepseekV4ForCausalLM.hf_to_vllm_mapper
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"architectures": ["DeepseekV3ForCausalLM"]}, "architectures"),
        ({"model_type": "deepseek_v3"}, "model_type"),
        ({"expert_dtype": "fp8"}, "expert_dtype"),
        ({"hc_mult": 2}, "hc_mult"),
        ({"head_dim": 256}, "head_dim"),
        ({"hidden_size": 7168}, "hidden_size"),
        ({"index_n_heads": 32}, "index_n_heads"),
        ({"index_topk": 2048}, "index_topk"),
        ({"moe_intermediate_size": 1536}, "moe_intermediate_size"),
        ({"n_routed_experts": 160}, "n_routed_experts"),
        ({"n_shared_experts": 2}, "n_shared_experts"),
        ({"num_attention_heads": 128}, "num_attention_heads"),
        ({"num_experts_per_tok": 8}, "num_experts_per_tok"),
        ({"num_hidden_layers": 61}, "num_hidden_layers"),
        ({"num_hash_layers": 0}, "num_hash_layers"),
        ({"o_lora_rank": 512}, "o_lora_rank"),
        ({"q_lora_rank": 1536}, "q_lora_rank"),
        ({"scoring_func": "sigmoid"}, "scoring_func"),
        ({"topk_method": "greedy"}, "topk_method"),
        ({"compress_ratios": [0] * 43}, "compress_ratios"),
    ],
)
def test_deepseek_v4_fails_closed_outside_flash_contract(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_deepseek_v4_flash_config(
            _config(**overrides),
            _parallel(),
        )


def test_deepseek_v4_accepts_normalized_rope_type_key() -> None:
    config = _config(
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65_536,
            "rope_type": "yarn",
        }
    )
    _require_supported_deepseek_v4_flash_config(config, _parallel())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"tensor_parallel_size": 2}, "TP4/PP1/DP1"),
        ({"tensor_parallel_size": 8}, "TP4/PP1/DP1"),
        ({"pipeline_parallel_size": 2}, "TP4/PP1/DP1"),
        ({"data_parallel_size": 2}, "TP4/PP1/DP1"),
        ({"enable_expert_parallel": True}, "enable_expert_parallel"),
        ({"use_sequence_parallel_moe": True}, "use_sequence_parallel_moe"),
        ({"enable_eplb": True}, "enable_eplb"),
        ({"use_ubatching": True}, "use_ubatching"),
        ({"decode_context_parallel_size": 2}, "context parallelism"),
        ({"prefill_context_parallel_size": 2}, "context parallelism"),
    ],
)
def test_deepseek_v4_fails_closed_for_unverified_topologies(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_deepseek_v4_flash_config(
            _config(),
            _parallel(**overrides),
        )


def test_deepseek_v4_rejects_wrong_quantizer_dtype_mtp_and_megamoe() -> None:
    with pytest.raises(NotImplementedError, match="deepseek_v4_fp8"):
        _require_supported_deepseek_v4_flash_config(
            _config(),
            _parallel(),
            quant_config=_quant_config("fp8"),
        )
    with pytest.raises(NotImplementedError, match="128x128"):
        _require_supported_deepseek_v4_flash_config(
            _config(),
            _parallel(),
            quant_config=_quant_config(block_size=[64, 64]),
        )
    with pytest.raises(NotImplementedError, match="runtime dtype"):
        _require_supported_deepseek_v4_flash_config(
            _config(),
            _parallel(),
            dtype=torch.float16,
        )
    with pytest.raises(NotImplementedError, match="speculative/MTP"):
        _require_supported_deepseek_v4_flash_config(
            _config(),
            _parallel(),
            speculative_config=object(),
        )
    with pytest.raises(NotImplementedError, match="SM100 MegaMoE"):
        _require_supported_deepseek_v4_flash_config(
            _config(),
            _parallel(),
            moe_backend="deep_gemm_mega_moe",
        )


def test_deepseek_v4_accepts_official_bf16_tp4_contract() -> None:
    _require_supported_deepseek_v4_flash_config(
        _config(),
        _parallel(),
        quant_config=_quant_config(),
        dtype=torch.bfloat16,
        moe_backend="auto",
    )


def test_deepseek_v4_model_shape_reads_moe_geometry() -> None:
    shape = _make_model_shape_from_hf_config(_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 4096
    assert shape.num_heads == 64
    assert shape.num_kv_heads == 1
    assert shape.head_dim == 512
    assert shape.intermediate_dim == 0
    assert shape.num_experts == 256
    assert shape.top_k == 6


class _Attention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache,
    ) -> torch.Tensor:
        return hidden_states + 1


class _FFN(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return hidden_states * 2


def test_deepseek_v4_decoder_hooks_observe_mhc_boundaries(monkeypatch) -> None:
    subject = DeepseekV4PDecoderLayer.__new__(DeepseekV4PDecoderLayer)
    nn.Module.__init__(subject)
    subject.attn = _Attention()
    subject.ffn = _FFN()
    subject.attn_norm = SimpleNamespace(
        weight=nn.Parameter(torch.ones(1)),
        variance_epsilon=1e-6,
    )
    subject.ffn_norm = SimpleNamespace(
        weight=nn.Parameter(torch.ones(1)),
        variance_epsilon=1e-6,
    )
    subject.hc_attn_fn = nn.Parameter(torch.ones(1))
    subject.hc_attn_scale = nn.Parameter(torch.ones(1))
    subject.hc_attn_base = nn.Parameter(torch.ones(1))
    subject.hc_attn_fn_broadcast = torch.ones(1)
    subject.hc_ffn_fn = nn.Parameter(torch.ones(1))
    subject.hc_ffn_scale = nn.Parameter(torch.ones(1))
    subject.hc_ffn_base = nn.Parameter(torch.ones(1))
    subject.rms_norm_eps = 1e-6
    subject.hc_eps = 1e-6
    subject.hc_post_alpha = 2.0
    subject.hc_sinkhorn_iters = 20
    subject.use_sequence_parallel = False
    _add_hooks(subject, ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"))

    residual = torch.ones(1, 4, 1)
    post_mix = torch.ones(1)
    res_mix = torch.ones(1)

    def fake_pre(*args, **kwargs):
        return residual, post_mix, res_mix, torch.tensor([[3.0]])

    def fake_post_pre(*args, **kwargs):
        return residual, post_mix, res_mix, torch.tensor([[5.0]])

    monkeypatch.setattr(deepseek_v4_p, "mhc_pre_broadcast_tilelang", fake_pre)
    monkeypatch.setattr(deepseek_v4_p, "mhc_fused_post_pre_tilelang", fake_post_pre)
    observed: dict[str, torch.Tensor] = {}
    for name in ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"):
        getattr(subject, f"hook_{name}").register_forward_hook(
            lambda _module, _inputs, output, name=name: observed.setdefault(
                name, output.clone()
            )
        )

    output, out_residual, out_post_mix, out_res_mix = subject(
        torch.tensor([[2.0]]),
        torch.tensor([0]),
        torch.tensor([7]),
    )

    assert {name: value.item() for name, value in observed.items()} == {
        "ln1": 3.0,
        "attn_out": 4.0,
        "ln2": 5.0,
        "mlp_in": 5.0,
        "mlp_out": 10.0,
    }
    assert output.item() == 10.0
    assert out_residual is residual
    assert out_post_mix is post_mix
    assert out_res_mix is res_mix


def test_deepseek_v4_manifest_is_reduced_and_exact() -> None:
    subject = _fake_model()
    specs = subject.get_hook_specs()
    model_wide = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 220
    assert len(model_wide) == 220
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
    for layer_no in range(43):
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


def test_deepseek_v4_compare_buffers_match_reduced_manifest(monkeypatch) -> None:
    subject = DeepseekV4CompareForCausalLM.__new__(DeepseekV4CompareForCausalLM)
    nn.Module.__init__(subject)
    fake = _fake_model()
    subject.config = fake.config
    subject.model = fake.model
    _add_hooks(subject, ("token_ids", "final_logits"))
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
    assert len(buffers) == 220
    assert buffers["embed"].shape == (8, 4096)
    assert buffers["ln1_L42"].shape == (8, 4096)
    assert buffers["final_logits"].shape == (4, 129_280)
    assert buffers["token_ids"].shape == (8,)
    assert not any(name.startswith(("resid_pre_", "resid_mid_")) for name in buffers)
    assert not any(name.startswith(("q_", "k_", "v_", "z_")) for name in buffers)
    assert not any(name.startswith(("router_", "topk_")) for name in buffers)
