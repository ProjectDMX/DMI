"""CPU contracts for bounded Apertus monitoring support."""

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
from vllm.model_executor.models.apertus import (
    ApertusAttention,
    ApertusDecoderLayer,
    ApertusForCausalLM,
    ApertusMLP,
    ApertusModel,
)
import vllm.model_executor.models.apertus_compare as apertus_compare
from vllm.model_executor.models.apertus_p import (
    ApertusPAttention,
    ApertusPDecoderLayer,
    ApertusPForCausalLM,
    ApertusPMLP,
    ApertusPModel,
    _require_supported_apertus_config,
)


pytestmark = pytest.mark.framework_fork


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "apertus",
        "hidden_act": "xielu",
        "post_norm": False,
        "qk_norm": True,
        "attention_bias": False,
        "mlp_bias": False,
        "is_causal": True,
        "layer_types": None,
        "logit_scale": None,
        "tie_word_embeddings": False,
        "num_hidden_layers": 2,
        "rope_parameters": {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
            "rope_theta": 12_000_000,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_apertus_preserves_upstream_loader_and_class_contracts() -> None:
    assert _ARCH_REMAP["ApertusForCausalLM"] == "ApertusPForCausalLM"
    assert issubclass(ApertusPForCausalLM, ApertusForCausalLM)
    assert issubclass(ApertusPModel, ApertusModel)
    assert issubclass(ApertusPDecoderLayer, ApertusDecoderLayer)
    assert issubclass(ApertusPAttention, ApertusAttention)
    assert issubclass(ApertusPMLP, ApertusMLP)
    assert (
        ApertusPForCausalLM.packed_modules_mapping
        == ApertusForCausalLM.packed_modules_mapping
    )
    assert (
        ApertusPForCausalLM.hf_to_vllm_mapper
        is ApertusForCausalLM.hf_to_vllm_mapper
    )
    assert ApertusPForCausalLM.embedding_modules == (
        ApertusForCausalLM.embedding_modules
    )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"model_type": "llama"}, "model_type"),
        ({"hidden_act": "silu"}, "hidden_act"),
        ({"post_norm": True}, "post_norm"),
        ({"qk_norm": False}, "qk_norm"),
        ({"attention_bias": True}, "bias-free"),
        ({"mlp_bias": True}, "bias-free"),
        ({"is_causal": False}, "causal"),
        ({"layer_types": ["sliding_attention"] * 2}, "schedules"),
        ({"logit_scale": 0.5}, "logit scale"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"rope_parameters": None}, "rope_parameters"),
        (
            {
                "rope_parameters": {
                    "factor": 8.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "default",
                    "rope_theta": 12_000_000,
                }
            },
            "rope_type",
        ),
    ],
)
def test_apertus_fails_closed_outside_audited_8b_contract(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_apertus_config(_config(**overrides))


def test_apertus_accepts_the_official_8b_config_contract() -> None:
    _require_supported_apertus_config(_config())


def test_apertus_disabled_layer_hooks_delegate_to_upstream(monkeypatch) -> None:
    layer = ApertusPDecoderLayer.__new__(ApertusPDecoderLayer)
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
    marker = (torch.tensor([[17.0]]), torch.tensor([[19.0]]))
    observed = []

    def upstream_forward(_self, positions, hidden_states, residual):
        observed.append((positions, hidden_states, residual))
        return marker

    monkeypatch.setattr(ApertusDecoderLayer, "forward", upstream_forward)
    positions = torch.tensor([0])
    hidden_states = torch.tensor([[1.0]])
    residual = torch.tensor([[2.0]])

    result = layer(positions, hidden_states, residual)

    assert result is marker
    assert observed == [(positions, hidden_states, residual)]


class _FusedNorm(nn.Module):
    def forward(self, value: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return value * 2
        completed = value + residual
        return completed * 2, completed


class _Attention(nn.Module):
    def forward(self, *, positions, hidden_states):
        return hidden_states + 3


class _Add(nn.Module):
    def forward(self, value: torch.Tensor):
        return value + 5


def test_apertus_hooks_follow_fused_prenorm_execution_order() -> None:
    layer = ApertusPDecoderLayer.__new__(ApertusPDecoderLayer)
    nn.Module.__init__(layer)
    layer.attention_layernorm = _FusedNorm()
    layer.self_attn = _Attention()
    layer.feedforward_layernorm = _FusedNorm()
    layer.mlp = _Add()
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
                key, output.clone()
            )
        )
        setattr(layer, f"hook_{name}", hook)

    hidden_states, residual = layer(
        torch.tensor([0]), torch.tensor([[1.0]]), None
    )

    assert {name: value.item() for name, value in captured.items()} == {
        "resid_pre": 1.0,
        "ln1": 2.0,
        "attn_out": 5.0,
        "resid_mid": 6.0,
        "ln2": 12.0,
        "mlp_in": 12.0,
        "mlp_out": 17.0,
    }
    assert hidden_states.item() == 17.0
    assert residual.item() == 6.0


def test_apertus_resid_pre_uses_completed_fused_norm_residual() -> None:
    layer = ApertusPDecoderLayer.__new__(ApertusPDecoderLayer)
    nn.Module.__init__(layer)
    layer.attention_layernorm = _FusedNorm()
    layer.self_attn = _Attention()
    layer.feedforward_layernorm = _FusedNorm()
    layer.mlp = _Add()
    captured: list[torch.Tensor] = []
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
        hook.enabled = name == "resid_pre"
        if name == "resid_pre":
            hook.register_forward_hook(
                lambda _module, _args, output: captured.append(output.clone())
            )
        setattr(layer, f"hook_{name}", hook)

    layer(
        torch.tensor([0]),
        torch.tensor([[2.0]]),
        torch.tensor([[3.0]]),
    )

    assert len(captured) == 1
    assert captured[0].item() == 5.0


class _FinalLayer(nn.Module):
    def forward(self, _positions, _hidden_states, _residual):
        return torch.tensor([[2.0]]), torch.tensor([[3.0]])


class _FinalNorm(nn.Module):
    def forward(self, value, residual):
        return value * 10, value + residual


def test_apertus_resid_final_uses_completed_fused_norm_residual(
    monkeypatch,
) -> None:
    import vllm.model_executor.models.apertus_p as apertus_p

    model = ApertusPModel.__new__(ApertusPModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_FinalLayer()])
    model.start_layer = 0
    model.end_layer = 1
    model.norm = _FinalNorm()
    model.hook_embed = HookPoint()
    model.hook_embed.enabled = False
    model.hook_final_ln = HookPoint()
    model.hook_final_ln.enabled = False
    model.hook_resid_final = HookPoint()
    captured: list[torch.Tensor] = []
    model.hook_resid_final.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )
    model._maybe_add_hidden_state = lambda values, *_args: values
    monkeypatch.setattr(
        apertus_p,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output = model.forward(
        None,
        torch.tensor([0]),
        None,
        inputs_embeds=torch.tensor([[1.0]]),
    )

    assert output.item() == 20.0
    assert len(captured) == 1
    assert captured[0].item() == 5.0


class _Projection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _value: torch.Tensor):
        return self.output, None


class _Norm(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.amount


class _Rotary(nn.Module):
    def forward(self, _positions, q, k):
        return q + 100, k + 100


class _AttentionKernel(nn.Module):
    def forward(self, q, _k, _v):
        return q


def test_apertus_attention_hooks_bind_post_qk_norm_pre_rope_values() -> None:
    attention = ApertusPAttention.__new__(ApertusPAttention)
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 2
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    attention.qkv_proj = _Projection(qkv)
    attention.q_norm = _Norm(10)
    attention.k_norm = _Norm(20)
    attention.rotary_emb = _Rotary()
    attention.attn = _AttentionKernel()
    attention.o_proj = _Projection(torch.full((1, 4), 99.0))
    captured: dict[str, torch.Tensor] = {}
    for name in ("q", "k", "v", "z"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key, output.clone()
            )
        )
        setattr(attention, f"hook_{name}", hook)

    output = attention(torch.tensor([0]), torch.zeros(1, 4))

    assert output.tolist() == [[99.0, 99.0, 99.0, 99.0]]
    assert torch.equal(captured["q"], (qkv[:, :4].view(1, 2, 2) + 10))
    assert torch.equal(captured["k"], (qkv[:, 4:6].view(1, 1, 2) + 20))
    assert torch.equal(captured["v"], qkv[:, 6:].view(1, 1, 2))
    assert torch.equal(captured["z"], qkv[:, :4] + 110)


def test_apertus_mlp_post_hook_binds_xielu_before_down_projection() -> None:
    mlp = ApertusPMLP.__new__(ApertusPMLP)
    nn.Module.__init__(mlp)
    mlp.up_proj = _Projection(torch.tensor([[1.0, 2.0]]))
    mlp.act_fn = _Norm(10)
    mlp.down_proj = _Projection(torch.tensor([[19.0, 23.0]]))
    captured: list[torch.Tensor] = []
    mlp.hook_post = HookPoint()
    mlp.hook_post.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )

    output = mlp(torch.zeros(1, 2))

    assert output.tolist() == [[19.0, 23.0]]
    assert len(captured) == 1
    assert captured[0].tolist() == [[11.0, 12.0]]


def _fake_hooked_model() -> ApertusPForCausalLM:
    subject = ApertusPForCausalLM.__new__(ApertusPForCausalLM)
    nn.Module.__init__(subject)
    attention = SimpleNamespace(
        **{f"hook_{name}": HookPoint() for name in ("q", "k", "v", "z")}
    )
    mlp = SimpleNamespace(hook_post=HookPoint())
    layer = SimpleNamespace(
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
    subject.config = _config(num_hidden_layers=1)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[layer],
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    return subject


def test_apertus_manifest_is_truthful_and_in_firing_order() -> None:
    specs = _fake_hooked_model().get_hook_specs()

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


def test_apertus_model_wide_manifest_has_no_runtime_modules() -> None:
    specs = _fake_hooked_model().get_hook_specs(model_wide=True)

    assert len(specs) == 17
    assert all(spec.module is None for spec in specs)


def test_apertus_compare_uses_the_constructed_mlp_width(monkeypatch) -> None:
    layer = SimpleNamespace(
        mlp=SimpleNamespace(down_proj=SimpleNamespace(input_size_per_partition=256)),
        self_attn=SimpleNamespace(),
    )
    subject = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=128,
            vocab_size=65_536,
            head_dim=None,
        ),
        model=SimpleNamespace(start_layer=0, end_layer=1, layers=[layer]),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(
        apertus_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        apertus_compare.torch,
        "empty",
        lambda *shape, **_kwargs: shape,
    )

    apertus_compare.ApertusCompareForCausalLM.allocate_compare_buffers(
        subject,
        8,
        vllm_config,
    )

    assert layer.mlp._buf_mlp_post == (8, 256)


def test_apertus_compare_constructs_concrete_model_and_layer(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_init(
        self,
        *,
        vllm_config,
        prefix,
        model_type=ApertusPModel,
        layer_type=ApertusPDecoderLayer,
    ) -> None:
        nn.Module.__init__(self)
        observed.update(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=model_type,
            layer_type=layer_type,
        )

    monkeypatch.setattr(ApertusPForCausalLM, "__init__", fake_init)
    config = object()

    apertus_compare.ApertusCompareForCausalLM(
        vllm_config=config,
        prefix="model",
    )

    assert observed == {
        "vllm_config": config,
        "prefix": "model",
        "model_type": apertus_compare.ApertusCompareModel,
        "layer_type": apertus_compare.ApertusCompareDecoderLayer,
    }
