"""CPU contracts for LFM2 heterogeneous DMI support."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import CompilationMode

import integration.vllm_adapter  # noqa: F401
from integration.vllm_adapter import _ARCH_REMAP
from integration.model_shape import _make_model_shape_from_hf_config
from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_CONV_IN,
    HOOK_TYPE_CONV_OUT,
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
from vllm.model_executor.models.lfm2 import (
    Lfm2Attention,
    Lfm2AttentionDecoderLayer,
    Lfm2ForCausalLM,
    Lfm2MLP,
    Lfm2Model,
    Lfm2ShortConvDecoderLayer,
)
import vllm.model_executor.models.lfm2_compare as lfm2_compare
import vllm.model_executor.models.lfm2_p as lfm2_p
from vllm.model_executor.models.lfm2_p import (
    Lfm2PAttention,
    Lfm2PAttentionDecoderLayer,
    Lfm2PForCausalLM,
    Lfm2PModel,
    Lfm2PShortConvDecoderLayer,
    _instrument_upstream_lfm2_model,
)


class _FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hook_post = HookPoint()


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("q", "k", "v", "z"):
            setattr(self, f"hook_{name}", HookPoint())


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
            self.self_attn = _FakeAttention()
            self.hook_attn_out = HookPoint()
        else:
            self.hook_conv_in = HookPoint()
            self.hook_conv_out = HookPoint()


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


def _fake_model() -> Lfm2PForCausalLM:
    model = Lfm2PForCausalLM.__new__(Lfm2PForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
    )
    model.model = _FakeModel()
    model.hook_token_ids = HookPoint()
    model.hook_final_logits = HookPoint()
    return model


def test_lfm2_preserves_hybrid_loader_and_state_contracts() -> None:
    assert _ARCH_REMAP["Lfm2ForCausalLM"] == "Lfm2PForCausalLM"
    assert issubclass(Lfm2PForCausalLM, Lfm2ForCausalLM)
    assert issubclass(
        Lfm2PAttentionDecoderLayer, Lfm2AttentionDecoderLayer
    )
    assert issubclass(
        Lfm2PShortConvDecoderLayer, Lfm2ShortConvDecoderLayer
    )
    assert (
        Lfm2PForCausalLM.packed_modules_mapping
        == Lfm2ForCausalLM.packed_modules_mapping
    )
    assert (
        Lfm2PForCausalLM.hf_to_vllm_mapper
        is Lfm2ForCausalLM.hf_to_vllm_mapper
    )
    assert (
        Lfm2PForCausalLM.get_mamba_state_shape_from_config.__func__
        is Lfm2ForCausalLM.get_mamba_state_shape_from_config.__func__
    )
    assert (
        Lfm2PForCausalLM.get_mamba_state_copy_func.__func__
        is Lfm2ForCausalLM.get_mamba_state_copy_func.__func__
    )


def test_lfm2_disabled_layer_hooks_use_the_exact_upstream_forward(
    monkeypatch,
) -> None:
    layer = Lfm2PShortConvDecoderLayer.__new__(
        Lfm2PShortConvDecoderLayer
    )
    nn.Module.__init__(layer)
    for name in (
        "resid_pre",
        "ln1",
        "conv_in",
        "conv_out",
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
        Lfm2ShortConvDecoderLayer,
        "forward",
        upstream_forward,
    )
    hidden_states = torch.tensor([[1.0]])
    residual = torch.tensor([[2.0]])

    result = layer(hidden_states, residual)

    assert result is expected
    assert observed == [(hidden_states, residual)]


def test_lfm2_instruments_the_upstream_module_tree_in_place() -> None:
    model = Lfm2Model.__new__(Lfm2Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        layer_types=["conv", "full_attention"]
    )
    model.start_layer = 0
    model.end_layer = 2
    conv = Lfm2ShortConvDecoderLayer.__new__(
        Lfm2ShortConvDecoderLayer
    )
    nn.Module.__init__(conv)
    attention = Lfm2AttentionDecoderLayer.__new__(
        Lfm2AttentionDecoderLayer
    )
    nn.Module.__init__(attention)
    for layer in (conv, attention):
        layer.feed_forward = Lfm2MLP.__new__(Lfm2MLP)
        nn.Module.__init__(layer.feed_forward)
    attention.self_attn = Lfm2Attention.__new__(Lfm2Attention)
    nn.Module.__init__(attention.self_attn)
    model.layers = nn.ModuleList([conv, attention])
    identities = {
        "model": id(model),
        "conv": id(conv),
        "conv_mlp": id(conv.feed_forward),
        "attention": id(attention),
        "attention_module": id(attention.self_attn),
        "attention_mlp": id(attention.feed_forward),
    }

    result = _instrument_upstream_lfm2_model(model)

    assert id(result) == identities["model"]
    assert id(result.layers[0]) == identities["conv"]
    assert id(result.layers[0].feed_forward) == identities["conv_mlp"]
    assert id(result.layers[1]) == identities["attention"]
    assert id(result.layers[1].self_attn) == identities["attention_module"]
    assert id(result.layers[1].feed_forward) == identities["attention_mlp"]
    assert isinstance(result, Lfm2PModel)
    assert isinstance(result.layers[0], Lfm2PShortConvDecoderLayer)
    assert isinstance(result.layers[1], Lfm2PAttentionDecoderLayer)
    assert isinstance(result.layers[1].self_attn, Lfm2PAttention)
    assert isinstance(result.layers[0].hook_conv_in, HookPoint)
    assert isinstance(result.layers[1].self_attn.hook_q, HookPoint)


def test_lfm2_model_shape_uses_the_effective_aligned_mlp_width() -> None:
    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=None,
        block_multiple_of=256,
        vocab_size=65_536,
    )

    shape = _make_model_shape_from_hf_config(config, torch.bfloat16)

    assert shape is not None
    assert shape.intermediate_dim == 256


def test_lfm2_compare_buffer_uses_the_constructed_mlp_width(
    monkeypatch,
) -> None:
    layer = SimpleNamespace(
        feed_forward=SimpleNamespace(
            w2=SimpleNamespace(input_size_per_partition=256)
        )
    )
    model = SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])
    subject = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=128,
            layer_types=["conv"],
            vocab_size=65_536,
        ),
        model=model,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(
        lfm2_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        lfm2_compare.torch,
        "empty",
        lambda *shape, **_kwargs: shape,
    )

    lfm2_compare.Lfm2CompareForCausalLM.allocate_compare_buffers(
        subject, 8, vllm_config
    )

    assert layer.feed_forward._buf_mlp_post == (8, 256)


def test_lfm2_inventory_is_truthful_per_layer_kind() -> None:
    specs = _fake_model().get_hook_specs()
    by_layer = {
        layer_no: {spec.hook_type for spec in specs if spec.layer_no == layer_no}
        for layer_no in (0, 1)
    }
    common = {
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
    }

    assert by_layer[0] == common | {HOOK_TYPE_CONV_IN, HOOK_TYPE_CONV_OUT}
    assert by_layer[1] == common | {
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
    }
    assert {spec.hook_type for spec in specs if spec.layer_no == -1} == {
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    }
    assert len(specs) == 26
    assert all(spec.module is not None for spec in specs)


def test_lfm2_inventory_matches_forward_firing_order() -> None:
    specs = _fake_model().get_hook_specs()

    by_layer = {
        layer_no: [
            spec.hook_type for spec in specs if spec.layer_no == layer_no
        ]
        for layer_no in (0, 1)
    }
    common_prefix = [HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1]
    common_suffix = [
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
    ]
    assert by_layer[0] == common_prefix + [
        HOOK_TYPE_CONV_IN,
        HOOK_TYPE_CONV_OUT,
    ] + common_suffix
    assert by_layer[1] == common_prefix + [
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
    ] + common_suffix


def test_lfm2_model_wide_inventory_is_module_free() -> None:
    specs = _fake_model().get_hook_specs(model_wide=True)

    assert len(specs) == 26
    assert all(spec.module is None for spec in specs)


def test_lfm2_model_factory_selects_layer_kind_with_prefix_keyword(
    monkeypatch,
) -> None:
    observed: list[tuple[str, int, str]] = []

    class _AttentionLayer(nn.Module):
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
            observed.append(("attention", layer_idx, prefix))

    class _ConvLayer(_AttentionLayer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            kind, layer_idx, prefix = observed.pop()
            observed.append(("conv", layer_idx, prefix))

    def _make_layers(num_layers, layer_fn, *, prefix):
        layers = nn.ModuleList(
            [layer_fn(prefix=f"{prefix}.{index}") for index in range(num_layers)]
        )
        return 0, num_layers, layers

    rank = SimpleNamespace(is_last_rank=True)
    monkeypatch.setattr(lfm2_p, "get_pp_group", lambda: rank)
    monkeypatch.setattr(lfm2_p, "make_layers", _make_layers)
    monkeypatch.setattr(
        lfm2_p,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        lfm2_p,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        lfm2_p,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=object(),
        quant_config=object(),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    model = Lfm2PModel(
        vllm_config=vllm_config,
        prefix="model",
        attention_layer_type=_AttentionLayer,
        conv_layer_type=_ConvLayer,
    )

    assert len(model.layers) == 2
    assert observed == [
        ("conv", 0, "model.layers.0"),
        ("attention", 1, "model.layers.1"),
    ]


class _Projection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _value: torch.Tensor):
        return self.output, None


class _Scale(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor, residual=None):
        if residual is None:
            return value * self.amount
        return value * self.amount, value + residual


class _Rotary(nn.Module):
    def __init__(self, seen: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.seen = seen

    def forward(self, _positions, q, k):
        self.seen["rotary_q"] = q.clone()
        self.seen["rotary_k"] = k.clone()
        return q, k


class _AttentionKernel(nn.Module):
    def forward(self, q, _k, _v):
        return q


def test_lfm2_attention_hooks_normalized_qk_before_rope() -> None:
    attention = Lfm2PAttention.__new__(Lfm2PAttention)
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 2
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    attention.qkv_proj = _Projection(qkv)
    attention.q_layernorm = _Scale(2)
    attention.k_layernorm = _Scale(3)
    seen: dict[str, torch.Tensor] = {}
    attention.rotary_emb = _Rotary(seen)
    attention.attn = _AttentionKernel()
    attention.out_proj = _Projection(torch.full((1, 4), 99.0))
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
    assert torch.equal(captured["q"], qkv[:, :4].view(1, 2, 2) * 2)
    assert torch.equal(captured["k"], qkv[:, 4:6].view(1, 1, 2) * 3)
    assert torch.equal(captured["v"], qkv[:, 6:].view(1, 1, 2))
    assert torch.equal(seen["rotary_q"], captured["q"])
    assert torch.equal(seen["rotary_k"], captured["k"])
    assert torch.equal(captured["z"], seen["rotary_q"].view(1, 4))


class _ShortConv(nn.Module):
    def forward(self, hidden_states, output):
        output.copy_(hidden_states + 3)


class _Add(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1


def test_lfm2_short_conv_hooks_bound_operator_input_and_output() -> None:
    layer = Lfm2PShortConvDecoderLayer.__new__(
        Lfm2PShortConvDecoderLayer
    )
    nn.Module.__init__(layer)
    layer.operator_norm = _Scale(2)
    layer.short_conv = _ShortConv()
    layer.ffn_norm = _Scale(7)
    layer.feed_forward = _Add()
    captured: dict[str, torch.Tensor] = {}
    for name in (
        "resid_pre",
        "ln1",
        "conv_in",
        "conv_out",
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

    assert captured["conv_in"].item() == 2
    assert captured["conv_out"].item() == 5
    assert captured["resid_mid"].item() == 6
    assert output.item() == 36
    assert residual.item() == 6
