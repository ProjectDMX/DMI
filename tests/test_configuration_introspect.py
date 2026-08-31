"""Deriving descriptors from framework configs.

Descriptors are generated, not hand-typed. These tests pin the extraction
against the naming variants HF uses across model families, and against the
configs DMI should refuse.
"""
from __future__ import annotations

import json

import pytest

from dmi.configuration import (
    DescriptorError,
    describe_model,
    descriptor_from_hf_config,
    load_hf_config_document,
    resolve_descriptor,
)

pytestmark = pytest.mark.cpu

LLAMA = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16",
}

MOE = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "num_hidden_layers": 48,
    "intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}

GPT2 = {
    "model_type": "gpt2",
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "vocab_size": 50257,
}


def _write(tmp_path, payload, name="config.json", subdir="Meta-Llama-3-8B"):
    root = tmp_path / subdir
    root.mkdir(exist_ok=True)
    target = root / name
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


class TestExtraction:
    def test_dense_config(self, tmp_path):
        descriptor = describe_model(_write(tmp_path, LLAMA))
        topology = descriptor.topology
        assert topology.num_layers == 32
        assert topology.hidden_size == 4096
        assert topology.num_attention_heads == 32
        assert topology.num_kv_heads == 8
        assert topology.intermediate_size == 14336
        assert topology.vocab_size == 128256
        assert not topology.is_moe

    def test_moe_config(self, tmp_path):
        topology = describe_model(_write(tmp_path, MOE)).topology
        assert topology.is_moe
        assert topology.num_experts == 128
        assert topology.top_k == 8

    def test_gpt2_field_naming(self, tmp_path):
        # n_embd / n_head / n_layer instead of the llama spellings.
        topology = describe_model(_write(tmp_path, GPT2)).topology
        assert topology.num_layers == 12
        assert topology.hidden_size == 768
        assert topology.num_attention_heads == 12

    def test_kv_heads_default_to_attention_heads_for_mha(self, tmp_path):
        payload = {k: v for k, v in LLAMA.items() if k != "num_key_value_heads"}
        topology = describe_model(_write(tmp_path, payload)).topology
        assert topology.num_kv_heads == topology.num_attention_heads

    def test_head_dim_omitted_when_it_is_the_obvious_quotient(self, tmp_path):
        descriptor = describe_model(_write(tmp_path, LLAMA))
        assert descriptor.topology.head_dim is None
        assert descriptor.topology.effective_head_dim == 128

    def test_head_dim_kept_when_it_is_not(self, tmp_path):
        # 2048/32 == 64, but the config says 128; that must survive.
        descriptor = describe_model(_write(tmp_path, MOE))
        assert descriptor.topology.head_dim == 128

    def test_no_transformers_needed_for_a_config_file(self, tmp_path, monkeypatch):
        import builtins

        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] == "transformers":
                raise ImportError("blocked")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert describe_model(_write(tmp_path, LLAMA)).topology.num_layers == 32


class TestIdentity:
    def test_named_after_the_model_directory(self, tmp_path):
        descriptor = describe_model(_write(tmp_path, LLAMA, subdir="Qwen3-8B"))
        assert descriptor.model.id == "qwen3-8b"
        assert descriptor.model.name == "Qwen3-8B"

    def test_directory_and_config_path_agree(self, tmp_path):
        config = _write(tmp_path, LLAMA, subdir="Qwen3-8B")
        assert describe_model(config).model.id == describe_model(config.parent).model.id

    def test_explicit_name_wins(self, tmp_path):
        descriptor = describe_model(_write(tmp_path, LLAMA), name="Llama 3 8B")
        assert descriptor.model.name == "Llama 3 8B"

    def test_non_standard_filename_keeps_its_stem(self, tmp_path):
        target = _write(tmp_path, LLAMA, name="llama-custom.json")
        assert describe_model(target).model.id == "llama-custom"


class TestRejections:
    def test_encoder_decoder_rejected(self, tmp_path):
        payload = {**LLAMA, "is_encoder_decoder": True}
        with pytest.raises(DescriptorError, match="encoder-decoder"):
            describe_model(_write(tmp_path, payload))

    def test_missing_attention_geometry_rejected(self, tmp_path):
        payload = {"model_type": "llama", "num_hidden_layers": 32}
        with pytest.raises(DescriptorError, match="hidden_size"):
            describe_model(_write(tmp_path, payload))

    def test_missing_layer_count_rejected(self, tmp_path):
        payload = {k: v for k, v in LLAMA.items() if k != "num_hidden_layers"}
        with pytest.raises(DescriptorError, match="layer count"):
            describe_model(_write(tmp_path, payload))

    def test_inconsistent_topology_reported(self, tmp_path):
        # More KV heads than attention heads is not a valid GQA config.
        payload = {**LLAMA, "num_key_value_heads": 64}
        with pytest.raises(DescriptorError, match="inconsistent"):
            describe_model(_write(tmp_path, payload))

    def test_malformed_json_reported(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(DescriptorError, match="not valid JSON"):
            load_hf_config_document(target)

    def test_pointing_at_a_non_config_file_is_explained(self, tmp_path):
        stray = tmp_path / "notes.txt"
        stray.write_text("hello", encoding="utf-8")
        with pytest.raises(DescriptorError, match="not a model config"):
            describe_model(stray)

    def test_unknown_model_id_without_transformers_says_so(self, tmp_path, monkeypatch):
        import builtins

        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] == "transformers":
                raise ImportError("blocked")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(DescriptorError, match="transformers"):
            describe_model("Some/Model-That-Is-Not-Local")


class TestResolveDescriptor:
    def test_descriptor_yaml_still_works(self):
        descriptor = resolve_descriptor("examples/model_descriptors/llama3-8b.yaml")
        assert descriptor.model.id == "llama3-8b"

    def test_framework_config_and_descriptor_agree(self, tmp_path):
        """The shipped example must match what describe-model produces."""
        generated = describe_model(_write(tmp_path, LLAMA))
        shipped = resolve_descriptor("examples/model_descriptors/llama3-8b.yaml")
        assert generated.topology == shipped.topology

    def test_model_directory_accepted(self, tmp_path):
        config = _write(tmp_path, LLAMA)
        assert resolve_descriptor(config.parent).topology.num_layers == 32

    def test_config_json_accepted(self, tmp_path):
        assert resolve_descriptor(_write(tmp_path, LLAMA)).topology.num_layers == 32


class TestFromConfigObject:
    def test_accepts_any_duck_typed_config(self):
        class Config:
            hidden_size = 4096
            num_attention_heads = 32
            num_key_value_heads = 8
            num_hidden_layers = 32
            intermediate_size = 14336

        descriptor = descriptor_from_hf_config(Config(), "my/model")
        assert descriptor.model.id == "model"
        assert descriptor.topology.num_layers == 32
