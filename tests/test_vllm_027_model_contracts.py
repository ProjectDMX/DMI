"""Focused CPU contracts for DMI's vLLM 0.27.1 model variants."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from collections.abc import Iterable

import pytest
import torch
from torch import nn

# Importing the adapter extends the official wheel's model package path and
# registers DMI's lazy model variants without replacing the wheel runtime.
import integration.vllm_adapter  # noqa: F401
from monitoring.hook_points import HookPoint


pytestmark = pytest.mark.framework_fork


GPT2_MODULES = (
    "vllm.model_executor.models.gpt2_p",
    "vllm.model_executor.models.gpt2_compare",
    "vllm.model_executor.models.gpt2_ref",
)
QWEN3_MODULES = (
    "vllm.model_executor.models.qwen3_p",
    "vllm.model_executor.models.qwen3_compare",
    "vllm.model_executor.models.qwen3_ref",
)


@pytest.mark.parametrize("module_name", GPT2_MODULES)
def test_gpt2_variants_transpose_only_hf_conv1d_weights(module_name: str) -> None:
    module = importlib.import_module(module_name)
    model = module.GPT2Model.__new__(module.GPT2Model)
    conv1d_weight = torch.arange(6).reshape(2, 3)
    ordinary_weight = torch.arange(8).reshape(2, 4)
    bias = torch.arange(3)

    converted = dict(
        model._transpose_conv1d(
            [
                ("h.0.attn.c_attn.weight", conv1d_weight),
                ("h.0.mlp.c_fc.weight", conv1d_weight),
                ("h.0.attn.c_attn.bias", bias),
                ("wte.weight", ordinary_weight),
            ]
        )
    )

    assert torch.equal(converted["h.0.attn.c_attn.weight"], conv1d_weight.t())
    assert torch.equal(converted["h.0.mlp.c_fc.weight"], conv1d_weight.t())
    assert converted["h.0.attn.c_attn.bias"] is bias
    assert converted["wte.weight"] is ordinary_weight


@pytest.mark.parametrize("module_name", GPT2_MODULES)
def test_gpt2_variants_use_027_auto_loader_contract(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(module_name)
    calls: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, model: object, *, skip_substrs: list[str]) -> None:
            calls["model"] = model
            calls["skip_substrs"] = skip_substrs

        def load_weights(
            self, weights: Iterable[tuple[str, torch.Tensor]]
        ) -> set[str]:
            materialized = list(weights)
            calls["weights"] = materialized
            return {name for name, _ in materialized}

    monkeypatch.setattr(module, "AutoWeightsLoader", FakeLoader)
    model = module.GPT2Model.__new__(module.GPT2Model)
    weight = torch.arange(6).reshape(2, 3)

    loaded = model.load_weights([("h.0.attn.c_proj.weight", weight)])

    assert calls["model"] is model
    assert calls["skip_substrs"] == [".attn.bias", ".attn.masked_bias"]
    assert loaded == {"h.0.attn.c_proj.weight"}
    assert torch.equal(calls["weights"][0][1], weight.t())  # type: ignore[index]


def _call_forwards_keyword(function: object, keyword: str) -> bool:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return any(
        item.arg == keyword
        and isinstance(item.value, ast.Name)
        and item.value.id == keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for item in node.keywords
    )


@pytest.mark.parametrize("module_name", QWEN3_MODULES)
def test_qwen3_variants_forward_per_layer_sliding_window(module_name: str) -> None:
    module = importlib.import_module(module_name)

    for model_type in (module.Qwen3Attention, module.Qwen3DecoderLayer):
        assert "per_layer_sliding_window" in inspect.signature(
            model_type.__init__
        ).parameters
        assert _call_forwards_keyword(
            model_type.__init__, "per_layer_sliding_window"
        )


def test_qwen2_moe_variant_uses_factory_runner_router_contract() -> None:
    module = importlib.import_module("vllm.model_executor.models.qwen2_moe_p")
    block = module.Qwen2MoeSparseMoeBlock.__new__(
        module.Qwen2MoeSparseMoeBlock
    )
    nn.Module.__init__(block)
    events: list[tuple[str, tuple[int, ...]]] = []

    class FakeGate(nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            events.append(("gate", tuple(hidden_states.shape)))
            logits = torch.arange(
                hidden_states.shape[0] * 4, dtype=torch.float32
            ).reshape(hidden_states.shape[0], 4)
            return logits, None

    class FakeRouter:
        def select_experts(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            topk_indices_dtype: torch.dtype | None = None,
            *,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert topk_indices_dtype is None
            assert input_ids is None
            events.append(("route", tuple(router_logits.shape)))
            rows = hidden_states.shape[0]
            return (
                torch.full((rows, 2), 0.5, dtype=torch.float32),
                torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2) % 4,
            )

    class FakeExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = FakeRouter()

        def forward(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor:
            events.append(("experts", tuple(router_logits.shape)))
            return hidden_states + 1

    block.gate = FakeGate()
    block.experts = FakeExperts()
    block.hook_router_logits = HookPoint()
    block.hook_topk_ids = HookPoint()
    block.hook_topk_weights = HookPoint()
    captured: dict[str, torch.Tensor] = {}
    block.hook_router_logits.register_forward_hook(
        lambda _module, _args, output: captured.setdefault("router", output)
    )
    block.hook_topk_ids.register_forward_hook(
        lambda _module, _args, output: captured.setdefault("ids", output)
    )
    block.hook_topk_weights.register_forward_hook(
        lambda _module, _args, output: captured.setdefault("weights", output)
    )

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = block(hidden_states)

    assert events == [
        ("gate", (6, 4)),
        ("route", (6, 4)),
        ("experts", (6, 4)),
    ]
    assert output.shape == hidden_states.shape
    assert torch.equal(output, hidden_states + 1)
    assert captured["router"].shape == (6, 4)
    assert captured["ids"].shape == (6, 2)
    assert captured["ids"].dtype is torch.int32
    assert captured["weights"].shape == (6, 2)
    assert captured["weights"].dtype is torch.float32
