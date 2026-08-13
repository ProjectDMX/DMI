"""CPU contracts for Falcon-H1 hybrid DMI support."""

from __future__ import annotations

from types import SimpleNamespace

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
from vllm.config import CompilationMode
from vllm.model_executor.models.falcon_h1 import FalconH1ForCausalLM
from vllm.model_executor.models.falcon_h1_p import (
    FalconH1PAttentionDecoderLayer,
    FalconH1PForCausalLM,
    FalconH1PModel,
    FalconH1PParallelHybrid,
)
import vllm.model_executor.models.falcon_h1_p as falcon_h1_p


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("q", "k", "v", "z"):
            setattr(self, f"hook_{name}", HookPoint())


class _FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hook_post = HookPoint()


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()
        self.feed_forward = _FakeMLP()
        for name in (
            "resid_pre",
            "ln1",
            "attn_out",
            "ssm_in",
            "ssm_out",
            "resid_mid",
            "ln2",
            "mlp_in",
            "mlp_out",
        ):
            setattr(self, f"hook_{name}", HookPoint())


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])
        self.start_layer = 0
        self.end_layer = len(self.layers)
        self.hook_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()


def _fake_model() -> FalconH1PForCausalLM:
    model = FalconH1PForCausalLM.__new__(FalconH1PForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=2)
    model.model = _FakeModel()
    model.hook_token_ids = HookPoint()
    model.hook_final_logits = HookPoint()
    return model


def test_falcon_h1_preserves_hybrid_and_weight_loader_contracts() -> None:
    assert _ARCH_REMAP["FalconH1ForCausalLM"] == "FalconH1PForCausalLM"
    assert issubclass(FalconH1PForCausalLM, FalconH1ForCausalLM)
    assert (
        FalconH1PForCausalLM.packed_modules_mapping
        == FalconH1ForCausalLM.packed_modules_mapping
    )
    assert (
        FalconH1PForCausalLM.hf_to_vllm_mapper
        is FalconH1ForCausalLM.hf_to_vllm_mapper
    )
    assert (
        FalconH1PForCausalLM.get_mamba_state_shape_from_config.__func__
        is FalconH1ForCausalLM.get_mamba_state_shape_from_config.__func__
    )
    assert (
        FalconH1PForCausalLM.get_mamba_state_copy_func.__func__
        is FalconH1ForCausalLM.get_mamba_state_copy_func.__func__
    )


def test_falcon_h1_model_factory_accepts_make_layers_prefix_keyword(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    class _FactoryLayer(nn.Module):
        def __init__(
            self,
            config,
            layer_idx,
            model_config,
            cache_config,
            *,
            quant_config,
            prefix,
        ) -> None:
            super().__init__()
            observed.update(
                layer_idx=layer_idx,
                prefix=prefix,
                quant_config=quant_config,
            )

    def _make_layers(num_layers, layer_fn, *, prefix):
        layer = layer_fn(prefix=f"{prefix}.0")
        return 0, num_layers, nn.ModuleList([layer])

    rank = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(falcon_h1_p, "get_pp_group", lambda: rank)
    monkeypatch.setattr(falcon_h1_p, "make_layers", _make_layers)
    monkeypatch.setattr(
        falcon_h1_p,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        falcon_h1_p,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        falcon_h1_p,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        embedding_multiplier=1.0,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=object(),
        quant_config="quant",
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    model = FalconH1PModel(
        vllm_config=vllm_config,
        prefix="model",
        layer_type=_FactoryLayer,
    )

    assert len(model.layers) == 1
    assert observed == {
        "layer_idx": 0,
        "prefix": "model.layers.0",
        "quant_config": "quant",
    }


def test_falcon_h1_inventory_declares_each_hybrid_layer_capability() -> None:
    specs = _fake_model().get_hook_specs()
    per_layer = {
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_SSM_IN,
        HOOK_TYPE_SSM_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
    }

    assert {spec.hook_type for spec in specs if spec.layer_no == 0} == per_layer
    assert {spec.hook_type for spec in specs if spec.layer_no == 1} == per_layer
    assert {spec.hook_type for spec in specs if spec.layer_no == -1} == {
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    }
    assert len(specs) == 33
    assert all(spec.module is not None for spec in specs)


def test_falcon_h1_model_wide_inventory_is_module_free() -> None:
    specs = _fake_model().get_hook_specs(model_wide=True)

    assert len(specs) == 33
    assert all(spec.module is None for spec in specs)
    assert all(
        spec.dim0_is_actual_tokens
        for spec in specs
        if spec.hook_type != HOOK_TYPE_FINAL_LOGITS
    )


class _Projection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _value: torch.Tensor):
        return self.output, None


class _Rotary(nn.Module):
    def __init__(self, events: list[tuple[str, torch.Tensor]]) -> None:
        super().__init__()
        self.events = events

    def forward(self, _positions, q, k):
        self.events.extend(
            [("rotary_q", q.clone()), ("rotary_k", k.clone())]
        )
        return q + 100, k + 200


class _AttentionKernel(nn.Module):
    def __init__(self, events: list[tuple[str, torch.Tensor]]) -> None:
        super().__init__()
        self.events = events

    def forward(self, q, k, v):
        self.events.extend(
            [
                ("kernel_q", q.clone()),
                ("kernel_k", k.clone()),
                ("kernel_v", v.clone()),
            ]
        )
        return q


def test_falcon_h1_attention_hooks_scaled_k_before_rope() -> None:
    attention = FalconH1PAttentionDecoderLayer.__new__(
        FalconH1PAttentionDecoderLayer
    )
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 2
    attention.key_multiplier = 10
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    attention.qkv_proj = _Projection(qkv)
    events: list[tuple[str, torch.Tensor]] = []
    attention.rotary_emb = _Rotary(events)
    attention.attn = _AttentionKernel(events)
    attention.o_proj = _Projection(torch.full((1, 4), 999.0))
    captured: dict[str, torch.Tensor] = {}
    for name in ("q", "k", "v", "z"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key, output.clone()
            )
        )
        setattr(attention, f"hook_{name}", hook)

    output = attention.self_attention(
        torch.tensor([0]), torch.zeros(1, 4)
    )

    assert output.tolist() == [[999.0, 999.0, 999.0, 999.0]]
    assert torch.equal(captured["q"].flatten(), qkv[:, :4].flatten())
    assert torch.equal(captured["k"].flatten(), qkv[:, 4:6].flatten() * 10)
    assert torch.equal(captured["v"].flatten(), qkv[:, 6:].flatten())
    assert torch.equal(events[0][1], captured["q"].flatten(-2, -1))
    assert torch.equal(events[1][1], captured["k"].flatten(-2, -1))
    assert torch.equal(events[4][1], captured["v"].flatten(-2, -1))
    assert torch.equal(captured["z"], events[2][1])


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.value


class _Branch(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, *, hidden_states: torch.Tensor, **kwargs):
        return hidden_states + self.amount, kwargs.get("residual")


class _Add(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.amount


def test_falcon_h1_ssm_hooks_bound_scaled_branch_contribution() -> None:
    layer = FalconH1PParallelHybrid.__new__(FalconH1PParallelHybrid)
    nn.Module.__init__(layer)
    layer.input_layernorm = _Scale(2)
    layer.pre_ff_layernorm = _Scale(13)
    layer.self_attn = _Branch(1)
    layer.mamba = _Branch(2)
    layer.feed_forward = _Add(3)
    layer.attention_in_multiplier = 3
    layer.attn_out_multiplier = 5
    layer.ssm_in_multiplier = 7
    layer.ssm_out_multiplier = 11
    captured: dict[str, torch.Tensor] = {}
    for name in (
        "resid_pre",
        "ln1",
        "attn_out",
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

    output = layer(torch.tensor([0]), torch.tensor([[1.0]]))

    assert captured["ssm_in"].item() == 14
    assert captured["ssm_out"].item() == 176
    assert captured["attn_out"].item() == 35
    assert captured["resid_mid"].item() == 212
    assert output.item() == 2971
