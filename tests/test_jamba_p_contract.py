"""CPU contracts for bounded dense Jamba DMI support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import integration.vllm_adapter  # noqa: F401
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
    HOOK_TYPE_SSM_IN,
    HOOK_TYPE_SSM_OUT,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)
from vllm.model_executor.models.jamba import (
    JambaAttentionDecoderLayer,
    JambaForCausalLM,
    JambaMLP,
    JambaMambaDecoderLayer,
    JambaModel,
)
import vllm.model_executor.models.jamba_compare as jamba_compare
from vllm.model_executor.models.jamba_p import (
    JambaPAttentionDecoderLayer,
    JambaPForCausalLM,
    JambaPMLP,
    JambaPMambaDecoderLayer,
    JambaPModel,
    _instrument_upstream_jamba_model,
    _require_dense_jamba_config,
)


class _FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hook_post = HookPoint()


class _FakeLayer(nn.Module):
    def __init__(self, *, attention: bool) -> None:
        super().__init__()
        self.feed_forward = _FakeMLP()
        for name in (
            "resid_pre",
            "ln1",
            "resid_mid",
            "ln2",
            "mlp_in",
            "mlp_out",
        ):
            setattr(self, f"hook_{name}", HookPoint())
        if attention:
            for name in ("q", "k", "v", "z", "attn_out"):
                setattr(self, f"hook_{name}", HookPoint())
        else:
            self.hook_ssm_in = HookPoint()
            self.hook_ssm_out = HookPoint()


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeLayer(attention=False), _FakeLayer(attention=True)]
        )
        self.start_layer = 0
        self.end_layer = 2
        self.hook_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()


def _fake_model() -> JambaPForCausalLM:
    subject = JambaPForCausalLM.__new__(JambaPForCausalLM)
    nn.Module.__init__(subject)
    subject.config = SimpleNamespace(
        num_hidden_layers=2,
        layers_block_type=["mamba", "attention"],
    )
    subject.model = _FakeModel()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    return subject


def _config(*, experts: tuple[int, ...] = (1, 1)) -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=2,
        layers_block_type=("mamba", "attention"),
        layers_num_experts=experts,
    )


def test_jamba_preserves_hybrid_loader_and_state_contracts() -> None:
    assert _ARCH_REMAP["JambaForCausalLM"] == "JambaPForCausalLM"
    assert issubclass(JambaPForCausalLM, JambaForCausalLM)
    assert issubclass(JambaPAttentionDecoderLayer, JambaAttentionDecoderLayer)
    assert issubclass(JambaPMambaDecoderLayer, JambaMambaDecoderLayer)
    assert (
        JambaPForCausalLM.packed_modules_mapping
        == JambaForCausalLM.packed_modules_mapping
    )
    assert JambaPForCausalLM.hf_to_vllm_mapper is JambaForCausalLM.hf_to_vllm_mapper
    assert (
        JambaPForCausalLM.get_mamba_state_shape_from_config.__func__
        is JambaForCausalLM.get_mamba_state_shape_from_config.__func__
    )
    assert (
        JambaPForCausalLM.get_mamba_state_copy_func.__func__
        is JambaForCausalLM.get_mamba_state_copy_func.__func__
    )


def test_jamba_support_is_bounded_to_dense_attention_mamba_configs() -> None:
    _require_dense_jamba_config(_config())

    with pytest.raises(NotImplementedError, match="MoE layers"):
        _require_dense_jamba_config(_config(experts=(1, 16)))
    with pytest.raises(NotImplementedError, match="layer kinds"):
        config = _config()
        config.layers_block_type = ("mamba", "linear_attention")
        _require_dense_jamba_config(config)


def test_jamba_disabled_layer_hooks_use_exact_upstream_forward(
    monkeypatch,
) -> None:
    layer = JambaPMambaDecoderLayer.__new__(JambaPMambaDecoderLayer)
    nn.Module.__init__(layer)
    for name in (
        "resid_pre",
        "ln1",
        "ssm_in",
        "ssm_out",
        "resid_mid",
        "ln2",
        "mlp_in",
        "mlp_out",
    ):
        hook = HookPoint()
        hook.enabled = False
        setattr(layer, f"hook_{name}", hook)
    expected = (torch.tensor([[3.0]]), torch.tensor([[4.0]]))
    observed: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def upstream_forward(_self, hidden_states, residual, **_kwargs):
        observed.append((hidden_states, residual))
        return expected

    monkeypatch.setattr(
        JambaMambaDecoderLayer,
        "forward",
        upstream_forward,
    )
    hidden_states = torch.tensor([[1.0]])
    residual = torch.tensor([[2.0]])

    result = layer(hidden_states, residual)

    assert result is expected
    assert observed == [(hidden_states, residual)]


def test_jamba_instruments_upstream_module_tree_in_place() -> None:
    model = JambaModel.__new__(JambaModel)
    nn.Module.__init__(model)
    model.config = _config()
    model.start_layer = 0
    model.end_layer = 2
    mamba = JambaMambaDecoderLayer.__new__(JambaMambaDecoderLayer)
    attention = JambaAttentionDecoderLayer.__new__(JambaAttentionDecoderLayer)
    for layer in (mamba, attention):
        nn.Module.__init__(layer)
        layer.feed_forward = JambaMLP.__new__(JambaMLP)
        nn.Module.__init__(layer.feed_forward)
    model.layers = nn.ModuleList([mamba, attention])
    identities = (
        id(model),
        id(mamba),
        id(mamba.feed_forward),
        id(attention),
        id(attention.feed_forward),
    )

    result = _instrument_upstream_jamba_model(model)

    assert (
        id(result),
        id(result.layers[0]),
        id(result.layers[0].feed_forward),
        id(result.layers[1]),
        id(result.layers[1].feed_forward),
    ) == identities
    assert isinstance(result, JambaPModel)
    assert isinstance(result.layers[0], JambaPMambaDecoderLayer)
    assert isinstance(result.layers[1], JambaPAttentionDecoderLayer)
    assert isinstance(result.layers[0].feed_forward, JambaPMLP)
    assert isinstance(result.layers[1].hook_q, HookPoint)


def test_jamba_inventory_is_truthful_and_in_firing_order() -> None:
    specs = _fake_model().get_hook_specs()
    by_layer = {
        layer_no: [spec.hook_type for spec in specs if spec.layer_no == layer_no]
        for layer_no in (0, 1)
    }
    prefix = [HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1]
    suffix = [
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
    ]
    assert (
        by_layer[0]
        == prefix
        + [
            HOOK_TYPE_SSM_IN,
            HOOK_TYPE_SSM_OUT,
        ]
        + suffix
    )
    assert (
        by_layer[1]
        == prefix
        + [
            HOOK_TYPE_Q,
            HOOK_TYPE_K,
            HOOK_TYPE_V,
            HOOK_TYPE_Z,
            HOOK_TYPE_ATTN_OUT,
        ]
        + suffix
    )
    assert {spec.hook_type for spec in specs if spec.layer_no == -1} == {
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    }
    assert len(specs) == 26
    assert all(spec.module is not None for spec in specs)


def test_jamba_model_wide_inventory_is_module_free() -> None:
    specs = _fake_model().get_hook_specs(model_wide=True)

    assert len(specs) == 26
    assert all(spec.module is None for spec in specs)


def test_jamba_compare_buffer_uses_constructed_mlp_width(
    monkeypatch,
) -> None:
    layer = SimpleNamespace(
        feed_forward=SimpleNamespace(
            down_proj=SimpleNamespace(input_size_per_partition=256)
        )
    )
    model = SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])
    subject = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=128,
            layers_block_type=["mamba"],
            vocab_size=65_536,
        ),
        model=model,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(
        jamba_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        jamba_compare.torch,
        "empty",
        lambda *shape, **_kwargs: shape,
    )

    jamba_compare.JambaCompareForCausalLM.allocate_compare_buffers(
        subject, 8, vllm_config
    )

    assert layer.feed_forward._buf_mlp_post == (8, 256)


def test_jamba_compare_constructs_the_concrete_compiled_backbone(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_init(
        self,
        *,
        vllm_config,
        prefix,
        model_type=JambaPModel,
    ) -> None:
        nn.Module.__init__(self)
        observed.update(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=model_type,
        )

    monkeypatch.setattr(JambaPForCausalLM, "__init__", fake_init)
    config = object()

    jamba_compare.JambaCompareForCausalLM(
        vllm_config=config,
        prefix="model",
    )

    assert observed == {
        "vllm_config": config,
        "prefix": "model",
        "model_type": jamba_compare.JambaCompareModel,
    }


class _Projection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _value: torch.Tensor):
        return self.output, None


class _AttentionKernel(nn.Module):
    def forward(self, q, _k, _v):
        return q


def test_jamba_attention_hooks_bound_qkv_and_attention_output() -> None:
    layer = JambaPAttentionDecoderLayer.__new__(JambaPAttentionDecoderLayer)
    nn.Module.__init__(layer)
    layer.q_size = 4
    layer.kv_size = 2
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.head_dim = 2
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    layer.qkv_proj = _Projection(qkv)
    layer.attn = _AttentionKernel()
    layer.o_proj = _Projection(torch.full((1, 4), 99.0))
    captured: dict[str, torch.Tensor] = {}
    for name in ("q", "k", "v", "z"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key, output.clone()
            )
        )
        setattr(layer, f"hook_{name}", hook)

    output = layer.self_attention(torch.tensor([0]), torch.zeros(1, 4))

    assert output.tolist() == [[99.0, 99.0, 99.0, 99.0]]
    assert torch.equal(captured["q"], qkv[:, :4].view(1, 2, 2))
    assert torch.equal(captured["k"], qkv[:, 4:6].view(1, 1, 2))
    assert torch.equal(captured["v"], qkv[:, 6:].view(1, 1, 2))
    assert torch.equal(captured["z"], qkv[:, :4])


class _Scale(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor, residual=None):
        if residual is None:
            return value * self.amount
        return value * self.amount, value + residual


class _Mamba(nn.Module):
    def forward(self, hidden_states, output):
        output.copy_(hidden_states + 3)


class _Add(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1


def test_jamba_mamba_hooks_bound_operator_input_and_output() -> None:
    layer = JambaPMambaDecoderLayer.__new__(JambaPMambaDecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = _Scale(2)
    layer.mamba = _Mamba()
    layer.pre_ff_layernorm = _Scale(7)
    layer.feed_forward = _Add()
    captured: dict[str, torch.Tensor] = {}
    for name in (
        "resid_pre",
        "ln1",
        "ssm_in",
        "ssm_out",
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

    output, residual = layer(torch.tensor([[1.0]]), None)

    assert captured["ssm_in"].item() == 2
    assert captured["ssm_out"].item() == 5
    assert captured["resid_mid"].item() == 6
    assert output.item() == 36
    assert residual.item() == 6
