"""CPU contracts for bounded dense MiniCPM monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP
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
from vllm.model_executor.models.minicpm import (
    MiniCPMAttention,
    MiniCPMDecoderLayer,
    MiniCPMForCausalLM,
    MiniCPMMLP,
    MiniCPMModel,
)
import vllm.model_executor.models.minicpm_compare as minicpm_compare
from vllm.model_executor.models.minicpm_p import (
    MiniCPMPAttention,
    MiniCPMPDecoderLayer,
    MiniCPMPForCausalLM,
    MiniCPMPMLP,
    MiniCPMPModel,
    _instrument_minicpm_model,
    _require_supported_minicpm_config,
)


pytestmark = pytest.mark.framework_fork


def _config(**overrides) -> SimpleNamespace:
    head_dim = overrides.get("hidden_size", 64) // overrides.get(
        "num_attention_heads", 2
    )
    values = {
        "model_type": "minicpm",
        "hidden_act": "silu",
        "num_experts": 0,
        "sparse_config": None,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 32,
        "tie_word_embeddings": True,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 65_536,
        "vocab_size": 73_448,
        "rope_parameters": {
            "rope_type": "longrope",
            "rope_theta": 10_000.0,
            "original_max_position_embeddings": 65_536,
            "short_factor": [1.0] * (head_dim // 2),
            "long_factor": [1.0] * (head_dim // 2),
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_minicpm_preserves_upstream_loader_and_class_contracts() -> None:
    assert _ARCH_REMAP["MiniCPMForCausalLM"] == "MiniCPMPForCausalLM"
    assert issubclass(MiniCPMPForCausalLM, MiniCPMForCausalLM)
    assert issubclass(MiniCPMPModel, MiniCPMModel)
    assert issubclass(MiniCPMPDecoderLayer, MiniCPMDecoderLayer)
    assert issubclass(MiniCPMPAttention, MiniCPMAttention)
    assert issubclass(MiniCPMPMLP, MiniCPMMLP)
    assert (
        MiniCPMPForCausalLM.packed_modules_mapping
        == MiniCPMForCausalLM.packed_modules_mapping
    )
    assert (
        MiniCPMPForCausalLM.embedding_modules
        == MiniCPMForCausalLM.embedding_modules
    )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"model_type": "llama"}, "model_type"),
        ({"hidden_act": "fatrelu"}, "hidden_act"),
        ({"num_experts": 8}, "MoE"),
        ({"sparse_config": {"topk": 64}}, "sparse-attention"),
        ({"scale_emb": None}, "scale_emb"),
        ({"scale_depth": 0}, "scale_depth"),
        ({"dim_model_base": -1}, "dim_model_base"),
        ({"tie_word_embeddings": None}, "tied/untied"),
        ({"max_position_embeddings": 8192}, "65536"),
        ({"rope_parameters": None}, "rope_parameters"),
        (
            {
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                }
            },
            "LongRoPE",
        ),
    ],
)
def test_minicpm_fails_closed_outside_the_audited_dense_contract(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_minicpm_config(_config(**overrides))


def test_minicpm_accepts_tiny_and_official_embedding_ownership() -> None:
    _require_supported_minicpm_config(_config(tie_word_embeddings=True))
    _require_supported_minicpm_config(_config(tie_word_embeddings=False))


def test_minicpm_disabled_layer_hooks_delegate_to_upstream(monkeypatch) -> None:
    layer = MiniCPMPDecoderLayer.__new__(MiniCPMPDecoderLayer)
    nn.Module.__init__(layer)
    for name in (
        "resid_pre",
        "ln1",
        "attn_out",
        "resid_mid",
        "ln2",
        "mlp_in",
        "mlp_out",
    ):
        hook = HookPoint()
        hook.enabled = False
        setattr(layer, f"hook_{name}", hook)
    marker = (torch.tensor([[17.0]]), None)
    observed = []

    def upstream_forward(_self, positions, hidden_states, residual):
        observed.append((positions, hidden_states, residual))
        return marker

    monkeypatch.setattr(MiniCPMDecoderLayer, "forward", upstream_forward)
    positions = torch.tensor([0])
    hidden_states = torch.tensor([[1.0]])

    result = layer(positions, hidden_states, None)

    assert result is marker
    assert observed == [(positions, hidden_states, None)]


class _Scale(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.amount


class _Attention(nn.Module):
    def forward(self, *, positions, hidden_states):
        del positions
        return hidden_states + 3


class _Add(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.amount


def test_minicpm_layer_hooks_preserve_depth_scaled_residual_order() -> None:
    layer = MiniCPMPDecoderLayer.__new__(MiniCPMPDecoderLayer)
    nn.Module.__init__(layer)
    layer.config = _config(num_hidden_layers=4, scale_depth=1.0)
    layer.input_layernorm = _Scale(2)
    layer.self_attn = _Attention()
    layer.post_attention_layernorm = _Scale(2)
    layer.mlp = _Add(5)
    captured: dict[str, torch.Tensor] = {}
    for name in (
        "resid_pre",
        "ln1",
        "attn_out",
        "resid_mid",
        "ln2",
        "mlp_in",
        "mlp_out",
    ):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key,
                output.clone(),
            )
        )
        setattr(layer, f"hook_{name}", hook)

    output, residual = layer(torch.tensor([0]), torch.tensor([[1.0]]), None)

    assert residual is None
    assert captured["resid_pre"].item() == 1.0
    assert captured["ln1"].item() == 2.0
    assert captured["attn_out"].item() == 5.0
    assert captured["resid_mid"].item() == 3.5
    assert captured["ln2"].item() == 7.0
    assert captured["mlp_in"].item() == 7.0
    assert captured["mlp_out"].item() == 12.0
    assert output.item() == 9.5


class _Projection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _value: torch.Tensor):
        return self.output, None


class _Rotary(nn.Module):
    def forward(self, _positions, q, k):
        return q + 100, k + 100


class _AttentionKernel(nn.Module):
    def forward(self, q, _k, _v):
        return q


def test_minicpm_attention_hooks_bind_pre_rope_qkv_and_pre_o_proj_z() -> None:
    attention = MiniCPMPAttention.__new__(MiniCPMPAttention)
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 2
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    attention.qkv_proj = _Projection(qkv)
    attention.rotary_emb = _Rotary()
    attention.attn = _AttentionKernel()
    attention.o_proj = _Projection(torch.full((1, 4), 99.0))
    captured: dict[str, torch.Tensor] = {}
    for name in ("q", "k", "v", "z"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key,
                output.clone(),
            )
        )
        setattr(attention, f"hook_{name}", hook)

    output = attention(torch.tensor([0]), torch.zeros(1, 4))

    assert output.tolist() == [[99.0, 99.0, 99.0, 99.0]]
    assert torch.equal(captured["q"], qkv[:, :4].view(1, 2, 2))
    assert torch.equal(captured["k"], qkv[:, 4:6].view(1, 1, 2))
    assert torch.equal(captured["v"], qkv[:, 6:].view(1, 1, 2))
    assert torch.equal(captured["z"], qkv[:, :4] + 100)


def test_minicpm_instruments_the_upstream_dense_tree_in_place() -> None:
    model = MiniCPMModel.__new__(MiniCPMModel)
    nn.Module.__init__(model)
    model.config = _config(num_hidden_layers=1)
    model.start_layer = 0
    model.end_layer = 1
    layer = MiniCPMDecoderLayer.__new__(MiniCPMDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = MiniCPMAttention.__new__(MiniCPMAttention)
    nn.Module.__init__(layer.self_attn)
    layer.mlp = MiniCPMMLP.__new__(MiniCPMMLP)
    nn.Module.__init__(layer.mlp)
    model.layers = nn.ModuleList([layer])
    identities = (id(model), id(layer), id(layer.self_attn), id(layer.mlp))

    result = _instrument_minicpm_model(model)

    assert (id(result), id(layer), id(layer.self_attn), id(layer.mlp)) == identities
    assert isinstance(result, MiniCPMPModel)
    assert isinstance(layer, MiniCPMPDecoderLayer)
    assert isinstance(layer.self_attn, MiniCPMPAttention)
    assert isinstance(layer.mlp, MiniCPMPMLP)


def _fake_hooked_model(num_layers: int = 2) -> MiniCPMPForCausalLM:
    subject = MiniCPMPForCausalLM.__new__(MiniCPMPForCausalLM)
    nn.Module.__init__(subject)
    layers = []
    for _ in range(num_layers):
        attention = SimpleNamespace(
            **{f"hook_{name}": HookPoint() for name in ("q", "k", "v", "z")}
        )
        mlp = SimpleNamespace(hook_post=HookPoint())
        layers.append(
            SimpleNamespace(
                self_attn=attention,
                mlp=mlp,
                **{
                    f"hook_{name}": HookPoint()
                    for name in (
                        "resid_pre",
                        "ln1",
                        "attn_out",
                        "resid_mid",
                        "ln2",
                        "mlp_in",
                        "mlp_out",
                    )
                },
            )
        )
    subject.config = _config(num_hidden_layers=num_layers)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=num_layers,
        layers=layers,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    return subject


def test_minicpm_manifest_is_truthful_and_in_firing_order() -> None:
    specs = _fake_hooked_model(num_layers=1).get_hook_specs()

    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
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
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is not None for spec in specs)


def test_minicpm_official_model_wide_manifest_has_389_families() -> None:
    subject = _fake_hooked_model(num_layers=2)
    subject.config.num_hidden_layers = 32
    specs = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 389
    assert all(spec.module is None for spec in specs)


def test_minicpm_compare_uses_the_constructed_mlp_width(monkeypatch) -> None:
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            down_proj=SimpleNamespace(input_size_per_partition=256)
        ),
        self_attn=SimpleNamespace(),
    )
    model = SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])
    subject = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=73_448,
        ),
        model=model,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(
        minicpm_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        minicpm_compare.torch,
        "empty",
        lambda *shape, **_kwargs: shape,
    )

    minicpm_compare.MiniCPMCompareForCausalLM.allocate_compare_buffers(
        subject,
        8,
        vllm_config,
    )

    assert layer.mlp._buf_mlp_post == (8, 256)
