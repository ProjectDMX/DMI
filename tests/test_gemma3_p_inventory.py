"""CPU contracts for Gemma 3's DMI model and hook inventory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

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
from vllm.config import CompilationMode

import integration.vllm_adapter  # noqa: F401
import vllm.model_executor.models.gemma3_p as gemma3_p_module
from vllm.model_executor.models.gemma3 import (
    Gemma3Attention as UpstreamGemma3Attention,
    Gemma3ForCausalLM as UpstreamGemma3ForCausalLM,
    Gemma3MLP as UpstreamGemma3MLP,
    Gemma3Model as UpstreamGemma3Model,
)
from vllm.model_executor.models.gemma3_p import (
    Gemma3Attention,
    Gemma3MLP,
    Gemma3Model,
    Gemma3PForCausalLM,
)


pytestmark = pytest.mark.framework_fork


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(
            hook_q=HookPoint(),
            hook_k=HookPoint(),
            hook_v=HookPoint(),
            hook_z=HookPoint(),
        )
        self.mlp = SimpleNamespace(hook_post=HookPoint())
        for name in (
            "hook_resid_pre",
            "hook_ln1",
            "hook_attn_out",
            "hook_resid_mid",
            "hook_ln2",
            "hook_mlp_in",
            "hook_mlp_out",
        ):
            setattr(self, name, HookPoint())


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])
        self.start_layer = 0
        self.end_layer = len(self.layers)
        self.hook_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()


def _fake_gemma3_p() -> Gemma3PForCausalLM:
    model = Gemma3PForCausalLM.__new__(Gemma3PForCausalLM)
    nn.Module.__init__(model)
    model.model = _FakeModel()
    model.hook_token_ids = HookPoint()
    model.hook_final_logits = HookPoint()
    return model


def test_gemma3_variant_preserves_upstream_class_contracts() -> None:
    assert issubclass(Gemma3MLP, UpstreamGemma3MLP)
    assert issubclass(Gemma3Attention, UpstreamGemma3Attention)
    assert issubclass(Gemma3Model, UpstreamGemma3Model)
    assert issubclass(Gemma3PForCausalLM, UpstreamGemma3ForCausalLM)
    assert Gemma3PForCausalLM.hf_to_vllm_mapper is UpstreamGemma3ForCausalLM.hf_to_vllm_mapper
    assert (
        Gemma3PForCausalLM.packed_modules_mapping
        == UpstreamGemma3ForCausalLM.packed_modules_mapping
    )


def test_gemma3_layer_factory_accepts_vllm_prefix_keyword(monkeypatch) -> None:
    config = SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        vocab_size=32,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        quant_config=None,
    )
    observed: dict[str, object] = {}

    def fake_make_layers(count, layer_fn, *, prefix):
        observed["count"] = count
        observed["prefix"] = prefix
        observed["layer"] = layer_fn(prefix=f"{prefix}.0")
        return 0, count, nn.ModuleList([observed["layer"]])

    monkeypatch.setattr(
        gemma3_p_module,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        gemma3_p_module,
        "GemmaRMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        gemma3_p_module,
        "Gemma3DecoderLayer",
        lambda *_args, **kwargs: nn.Identity()
        if kwargs["prefix"] == "root.layers.0"
        else pytest.fail(f"unexpected layer prefix: {kwargs['prefix']}"),
    )
    monkeypatch.setattr(gemma3_p_module, "make_layers", fake_make_layers)
    monkeypatch.setattr(
        gemma3_p_module,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        gemma3_p_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    model = Gemma3Model(
        vllm_config=vllm_config,
        prefix="root",
        decoder_layer_type=gemma3_p_module.Gemma3DecoderLayer,
    )

    assert observed == {
        "count": 1,
        "prefix": "root.layers",
        "layer": model.layers[0],
    }


def test_gemma3_inventory_has_all_dense_hook_types_per_layer() -> None:
    specs = _fake_gemma3_p().get_hook_specs()
    per_layer = {
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
    assert all(spec.module is not None for spec in specs)


def test_gemma3_model_wide_inventory_is_module_free() -> None:
    specs = _fake_gemma3_p().get_hook_specs(model_wide=True)

    assert len(specs) == 29
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


class _Add(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.amount


class _Rotary(nn.Module):
    def __init__(self, events: list[tuple[str, torch.Tensor]]) -> None:
        super().__init__()
        self.events = events

    def forward(
        self,
        _positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.events.extend(
            [("rotary_q", q.clone()), ("rotary_k", k.clone())]
        )
        return q + 100, k + 200


class _AttentionKernel(nn.Module):
    def __init__(self, events: list[tuple[str, torch.Tensor]]) -> None:
        super().__init__()
        self.events = events

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        self.events.extend(
            [
                ("kernel_q", q.clone()),
                ("kernel_k", k.clone()),
                ("kernel_v", v.clone()),
            ]
        )
        return q


def test_gemma3_attention_hooks_normalized_heads_before_rope() -> None:
    attention = Gemma3Attention.__new__(Gemma3Attention)
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 2
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    attention.qkv_proj = _Projection(qkv)
    attention.q_norm = _Add(10)
    attention.k_norm = _Add(20)
    events: list[tuple[str, torch.Tensor]] = []
    attention.rotary_emb = _Rotary(events)
    attention.attn = _AttentionKernel(events)
    attention.o_proj = _Projection(torch.full((1, 4), 999.0))
    attention.hook_q = HookPoint()
    attention.hook_k = HookPoint()
    attention.hook_v = HookPoint()
    attention.hook_z = HookPoint()
    captured: dict[str, torch.Tensor] = {}
    for name in ("q", "k", "v", "z"):
        getattr(attention, f"hook_{name}").register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key, output.clone()
            )
        )

    output = attention(torch.tensor([0]), torch.zeros(1, 4))

    assert output.tolist() == [[999.0, 999.0, 999.0, 999.0]]
    assert captured["q"].shape == (1, 2, 2)
    assert captured["k"].shape == (1, 1, 2)
    assert captured["v"].shape == (1, 1, 2)
    assert torch.equal(captured["q"].flatten(), qkv[:, :4].flatten() + 10)
    assert torch.equal(captured["k"].flatten(), qkv[:, 4:6].flatten() + 20)
    assert torch.equal(captured["v"].flatten(), qkv[:, 6:].flatten())
    assert torch.equal(events[0][1], captured["q"].flatten(-2, -1))
    assert torch.equal(events[1][1], captured["k"].flatten(-2, -1))
    assert torch.equal(events[4][1], captured["v"].flatten(-2, -1))
    assert torch.equal(captured["z"], events[2][1])
