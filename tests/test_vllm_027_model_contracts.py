"""Focused contracts for the model ports required by vLLM 0.27.1."""

from __future__ import annotations

import ast
from collections.abc import Iterable
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch


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
def test_gpt2_variants_use_v027_auto_loader_contract(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    calls: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, model: object, *, skip_substrs: list[str]) -> None:
            calls["model"] = model
            calls["skip_substrs"] = skip_substrs

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
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
def test_qwen3_variants_forward_per_layer_sliding_window(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)

    for model_type in (module.Qwen3Attention, module.Qwen3DecoderLayer):
        assert "per_layer_sliding_window" in inspect.signature(
            model_type.__init__
        ).parameters
        assert _call_forwards_keyword(
            model_type.__init__, "per_layer_sliding_window"
        )


def test_llama_ref_loader_passes_the_hf_to_vllm_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("vllm.model_executor.models.llama_ref")
    calls: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, model: object, *, skip_prefixes: list[str] | None):
            calls["model"] = model
            calls["skip_prefixes"] = skip_prefixes

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
            *,
            mapper: object,
        ) -> set[str]:
            calls["weights"] = list(weights)
            calls["mapper"] = mapper
            return {"loaded"}

    monkeypatch.setattr(module, "AutoWeightsLoader", FakeLoader)
    model = module.LlamaRefForCausalLM.__new__(module.LlamaRefForCausalLM)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(tie_word_embeddings=False),
    )
    weight = torch.zeros(2, 2)

    loaded = model.load_weights(
        [("model.layers.0.self_attn.q_proj.weight", weight)]
    )

    assert loaded == {"loaded"}
    assert calls["model"] is model
    assert calls["skip_prefixes"] is None
    assert calls["mapper"] is model.hf_to_vllm_mapper
    mapped_name, mapped_weight = next(
        iter(
            model.hf_to_vllm_mapper.apply(
                [("model.layers.0.self_attn.q_proj.weight", weight)]
            )
        )
    )
    assert mapped_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert mapped_weight is weight
    assert mapped_weight.shard_id == "q"
