"""CPU contracts for bounded OLMo 3 monitoring support."""

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
from vllm.model_executor.models.olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3MLP,
    Olmo3Model,
)
import vllm.model_executor.models.olmo3_compare as olmo3_compare
from vllm.model_executor.models.olmo3_p import (
    Olmo3PAttention,
    Olmo3PDecoderLayer,
    Olmo3PForCausalLM,
    Olmo3PMLP,
    Olmo3PModel,
    _instrument_olmo3_model,
    _require_supported_olmo3_config,
)


pytestmark = pytest.mark.framework_fork


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "olmo3",
        "hidden_act": "silu",
        "attention_bias": False,
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 4096,
        "rope_parameters": {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 500000.0,
            },
            "full_attention": {
                "rope_type": "yarn",
                "factor": 8.0,
            },
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_olmo3_preserves_upstream_loader_and_class_contracts() -> None:
    assert _ARCH_REMAP["Olmo3ForCausalLM"] == "Olmo3PForCausalLM"
    assert issubclass(Olmo3PForCausalLM, Olmo3ForCausalLM)
    assert issubclass(Olmo3PModel, Olmo3Model)
    assert issubclass(Olmo3PDecoderLayer, Olmo3DecoderLayer)
    assert issubclass(Olmo3PAttention, Olmo3Attention)
    assert issubclass(Olmo3PMLP, Olmo3MLP)
    assert (
        Olmo3PForCausalLM.packed_modules_mapping
        == Olmo3ForCausalLM.packed_modules_mapping
    )
    assert (
        Olmo3PForCausalLM.hf_to_vllm_mapper
        is Olmo3ForCausalLM.hf_to_vllm_mapper
    )


@pytest.mark.parametrize(
    "overrides, error, match",
    [
        ({"model_type": "olmo2"}, NotImplementedError, "model_type"),
        ({"hidden_act": "gelu"}, NotImplementedError, "hidden_act"),
        ({"attention_bias": True}, NotImplementedError, "attention_bias"),
        ({"layer_types": ["full_attention"]}, ValueError, "layer_types"),
        (
            {"layer_types": ["linear_attention", "full_attention"]},
            NotImplementedError,
            "layer types",
        ),
        ({"sliding_window": 0}, NotImplementedError, "sliding_window"),
        ({"rope_parameters": None}, NotImplementedError, "rope_parameters"),
        (
            {"rope_parameters": {"sliding_attention": {}}},
            NotImplementedError,
            "full_attention",
        ),
    ],
)
def test_olmo3_fails_closed_outside_audited_attention_contract(
    overrides,
    error,
    match,
) -> None:
    with pytest.raises(error, match=match):
        _require_supported_olmo3_config(_config(**overrides))


def test_olmo3_accepts_mixed_sliding_and_full_attention() -> None:
    _require_supported_olmo3_config(_config())


def test_olmo3_disabled_layer_hooks_delegate_to_upstream(monkeypatch) -> None:
    layer = Olmo3PDecoderLayer.__new__(Olmo3PDecoderLayer)
    nn.Module.__init__(layer)
    for name in (
        "resid_pre",
        "attn_out",
        "ln1",
        "resid_mid",
        "mlp_in",
        "mlp_out",
        "ln2",
    ):
        hook = HookPoint()
        hook.enabled = False
        setattr(layer, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, positions, hidden_states):
        observed.append((positions, hidden_states))
        return marker

    monkeypatch.setattr(Olmo3DecoderLayer, "forward", upstream_forward)
    positions = torch.tensor([0])
    hidden_states = torch.tensor([[1.0]])

    result = layer(positions, hidden_states)

    assert result is marker
    assert observed == [(positions, hidden_states)]


class _Scale(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.amount


class _Attention(nn.Module):
    def forward(self, _positions, hidden_states):
        return hidden_states + 3


class _Add(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.amount


def test_olmo3_hooks_follow_post_norm_execution_order() -> None:
    layer = Olmo3PDecoderLayer.__new__(Olmo3PDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = _Attention()
    layer.post_attention_layernorm = _Scale(2)
    layer.mlp = _Add(5)
    layer.post_feedforward_layernorm = _Scale(2)
    captured: dict[str, torch.Tensor] = {}
    for name in (
        "resid_pre",
        "attn_out",
        "ln1",
        "resid_mid",
        "mlp_in",
        "mlp_out",
        "ln2",
    ):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key,
                output.clone(),
            )
        )
        setattr(layer, f"hook_{name}", hook)

    output = layer(torch.tensor([0]), torch.tensor([[1.0]]))

    assert captured["resid_pre"].item() == 1.0
    assert captured["attn_out"].item() == 4.0
    assert captured["ln1"].item() == 8.0
    assert captured["resid_mid"].item() == 9.0
    assert captured["mlp_in"].item() == 9.0
    assert captured["mlp_out"].item() == 14.0
    assert captured["ln2"].item() == 28.0
    assert output.item() == 37.0


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


class _Activation(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :2] * 3


def test_olmo3_attention_hooks_bind_post_qk_norm_pre_rope_values() -> None:
    attention = Olmo3PAttention.__new__(Olmo3PAttention)
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 2
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    attention.qkv_proj = _Projection(qkv)
    attention._apply_qk_norm = lambda q, k: (q + 10, k + 20)
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
    assert torch.equal(captured["q"], (qkv[:, :4] + 10).view(1, 2, 2))
    assert torch.equal(captured["k"], (qkv[:, 4:6] + 20).view(1, 1, 2))
    assert torch.equal(captured["v"], qkv[:, 6:].view(1, 1, 2))
    assert torch.equal(captured["z"], qkv[:, :4] + 110)


def test_olmo3_mlp_post_hook_binds_pre_down_projection_activation() -> None:
    mlp = Olmo3PMLP.__new__(Olmo3PMLP)
    nn.Module.__init__(mlp)
    gate_up = torch.tensor([[1.0, 2.0, 7.0, 11.0]])
    mlp.gate_up_proj = _Projection(gate_up)
    mlp.act_fn = _Activation()
    mlp.down_proj = _Projection(torch.tensor([[19.0, 23.0]]))
    captured: list[torch.Tensor] = []
    mlp.hook_post = HookPoint()
    mlp.hook_post.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )

    output = mlp(torch.zeros(1, 2))

    assert output.tolist() == [[19.0, 23.0]]
    assert len(captured) == 1
    assert captured[0].tolist() == [[3.0, 6.0]]


def test_olmo3_instruments_the_upstream_tree_in_place() -> None:
    model = Olmo3Model.__new__(Olmo3Model)
    nn.Module.__init__(model)
    model.config = _config()
    model.start_layer = 0
    model.end_layer = 1
    layer = Olmo3DecoderLayer.__new__(Olmo3DecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = Olmo3Attention.__new__(Olmo3Attention)
    nn.Module.__init__(layer.self_attn)
    layer.mlp = Olmo3MLP.__new__(Olmo3MLP)
    nn.Module.__init__(layer.mlp)
    model.layers = nn.ModuleList([layer])
    model.config.num_hidden_layers = 1
    model.config.layer_types = ["full_attention"]
    identities = (id(model), id(layer), id(layer.self_attn), id(layer.mlp))

    result = _instrument_olmo3_model(model)

    assert (id(result), id(layer), id(layer.self_attn), id(layer.mlp)) == identities
    assert isinstance(result, Olmo3PModel)
    assert isinstance(layer, Olmo3PDecoderLayer)
    assert isinstance(layer.self_attn, Olmo3PAttention)
    assert isinstance(layer.mlp, Olmo3PMLP)


def _fake_hooked_model() -> Olmo3PForCausalLM:
    subject = Olmo3PForCausalLM.__new__(Olmo3PForCausalLM)
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
                "attn_out",
                "ln1",
                "resid_mid",
                "mlp_in",
                "mlp_out",
                "ln2",
            )
        },
    )
    subject.config = _config(num_hidden_layers=1, layer_types=["full_attention"])
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


def test_olmo3_manifest_is_truthful_and_in_firing_order() -> None:
    specs = _fake_hooked_model().get_hook_specs()

    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_LN1,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
        HOOK_TYPE_LN2,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is not None for spec in specs)


def test_olmo3_model_wide_manifest_has_no_runtime_modules() -> None:
    specs = _fake_hooked_model().get_hook_specs(model_wide=True)

    assert len(specs) == 17
    assert all(spec.module is None for spec in specs)


def test_olmo3_compare_uses_the_constructed_mlp_width(monkeypatch) -> None:
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
            intermediate_size=128,
            vocab_size=65_536,
        ),
        model=model,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(
        olmo3_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        olmo3_compare.torch,
        "empty",
        lambda *shape, **_kwargs: shape,
    )

    olmo3_compare.Olmo3CompareForCausalLM.allocate_compare_buffers(
        subject,
        8,
        vllm_config,
    )

    assert layer.mlp._buf_mlp_post == (8, 256)


def test_olmo3_compare_constructs_concrete_compiled_backbone(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_init(
        self,
        *,
        vllm_config,
        prefix,
        model_type=Olmo3PModel,
    ) -> None:
        nn.Module.__init__(self)
        observed.update(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=model_type,
        )

    monkeypatch.setattr(Olmo3PForCausalLM, "__init__", fake_init)
    config = object()

    olmo3_compare.Olmo3CompareForCausalLM(
        vllm_config=config,
        prefix="model",
    )

    assert observed == {
        "vllm_config": config,
        "prefix": "model",
        "model_type": olmo3_compare.Olmo3CompareModel,
    }
