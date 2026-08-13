"""CPU-only compatibility checks for supported vLLM worker APIs."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models import ModelRegistry
from vllm.v1.worker.gpu_worker import Worker

from integration.vllm_adapter import (
    DMXGPUWorker,
    _ARCH_REMAP,
    _DMI_MODEL_VARIANTS,
    _LLAMA_COMPAT_ARCHES,
)
from tests.compare_worker import _COMPARE_MODEL_VARIANTS


def test_llama_registry_aliases_use_the_hooked_variant():
    expected = {
        "AquilaModel",
        "AquilaForCausalLM",
        "CwmForCausalLM",
        "InternLMForCausalLM",
        "InternLM3ForCausalLM",
        "IQuestCoderForCausalLM",
        "LlamaForCausalLM",
        "LLaMAForCausalLM",
        "XverseForCausalLM",
    }

    assert _LLAMA_COMPAT_ARCHES == expected
    assert {_ARCH_REMAP[arch] for arch in expected} == {"LlamaPForCausalLM"}


def test_qwen2_uses_its_hooked_variant():
    assert _ARCH_REMAP["Qwen2ForCausalLM"] == "Qwen2PForCausalLM"


def test_bundled_variants_are_registered_with_official_vllm():
    supported = set(ModelRegistry.get_supported_archs())

    assert set(_DMI_MODEL_VARIANTS) <= supported
    for architecture in _DMI_MODEL_VARIANTS:
        model_cls = ModelRegistry._try_load_model_cls(architecture)
        assert model_cls is not None
        assert model_cls.__name__ == architecture


def test_compare_variants_are_lazily_registered_with_official_vllm():
    supported = set(ModelRegistry.get_supported_archs())

    assert set(_COMPARE_MODEL_VARIANTS) <= supported
    for architecture in _COMPARE_MODEL_VARIANTS:
        model_cls = ModelRegistry._try_load_model_cls(architecture)
        assert model_cls is not None
        assert model_cls.__name__ == architecture


def test_load_model_forwards_newer_vllm_keyword_arguments(monkeypatch):
    calls = []

    def fake_load_model(self, *args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(Worker, "load_model", fake_load_model)

    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        )
    )
    worker.adaptor = None

    worker.load_model(load_dummy_weights=True)

    assert calls == [((), {"load_dummy_weights": True})]
    assert worker.vllm_config.model_config.hf_config.architectures == [
        "LlamaPForCausalLM"
    ]


def test_v2_model_runner_fails_closed_before_device_initialization(monkeypatch):
    initialized = []
    monkeypatch.setattr(Worker, "init_device", lambda _self: initialized.append(True))

    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.use_v2_model_runner = True

    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        worker.init_device()

    assert initialized == []
