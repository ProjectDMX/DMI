"""Model descriptor loading, validation, and translation into DMI naming."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dmi.configuration import (
    DescriptorError,
    UnsupportedConfigVersion,
    descriptor_to_dict,
    load_descriptor,
    parse_descriptor,
    save_descriptor,
    to_model_shape_config,
)

pytestmark = pytest.mark.cpu

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model_descriptors"
DATA = Path(__file__).resolve().parent / "data"

VALID = {
    "schema_version": 1,
    "model": {"id": "m", "name": "M", "architecture": "decoder_transformer"},
    "topology": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "intermediate_size": 14336,
    },
}


def _without(section, key):
    document = {k: dict(v) if isinstance(v, dict) else v for k, v in VALID.items()}
    document[section].pop(key)
    return document


class TestParseDescriptor:
    def test_valid_descriptor(self):
        descriptor = parse_descriptor(VALID)
        assert descriptor.model.id == "m"
        assert descriptor.topology.num_layers == 32
        assert descriptor.last_layer == 31

    def test_unsupported_architecture(self):
        document = {**VALID, "model": {**VALID["model"], "architecture": "rnn"}}
        with pytest.raises(DescriptorError, match="Unsupported architecture"):
            parse_descriptor(document)

    def test_unsupported_schema_version(self):
        with pytest.raises(UnsupportedConfigVersion):
            parse_descriptor({**VALID, "schema_version": 99})

    @pytest.mark.parametrize("key", ["id", "name", "architecture"])
    def test_missing_identity_field(self, key):
        with pytest.raises(DescriptorError, match=f"model.{key}"):
            parse_descriptor(_without("model", key))

    @pytest.mark.parametrize(
        "key", ["num_layers", "hidden_size", "num_attention_heads", "num_kv_heads"]
    )
    def test_missing_required_topology_field(self, key):
        with pytest.raises(DescriptorError, match="missing required"):
            parse_descriptor(_without("topology", key))

    def test_missing_topology_section(self):
        with pytest.raises(DescriptorError, match="topology"):
            parse_descriptor({k: v for k, v in VALID.items() if k != "topology"})

    def test_unknown_topology_field_is_a_typo_not_a_silent_default(self):
        document = {**VALID, "topology": {**VALID["topology"], "num_kv_head": 8}}
        with pytest.raises(DescriptorError, match="Unknown field"):
            parse_descriptor(document)

    def test_non_mapping_rejected(self):
        with pytest.raises(DescriptorError):
            parse_descriptor(["not", "a", "mapping"])


class TestDescriptorIO:
    def test_round_trip_through_disk(self, tmp_path):
        original = parse_descriptor(VALID)
        target = tmp_path / "m.model.yaml"
        save_descriptor(original, target)
        assert descriptor_to_dict(load_descriptor(target)) == descriptor_to_dict(original)

    def test_dict_form_omits_unset_optional_fields(self):
        document = descriptor_to_dict(parse_descriptor(VALID))
        assert "num_experts" not in document["topology"]
        assert "head_dim" not in document["topology"]

    def test_missing_file_reports_path(self, tmp_path):
        with pytest.raises(DescriptorError, match="Cannot read descriptor"):
            load_descriptor(tmp_path / "absent.yaml")

    def test_malformed_yaml_reports_clearly(self, tmp_path):
        target = tmp_path / "broken.yaml"
        target.write_text("model: [unclosed", encoding="utf-8")
        with pytest.raises(DescriptorError, match="not valid YAML"):
            load_descriptor(target)


class TestShippedExamples:
    def test_example_descriptor_loads(self):
        descriptor = load_descriptor(EXAMPLES / "llama3-8b.yaml")
        assert descriptor.topology.num_layers == 32
        assert descriptor.model.architecture == "decoder_transformer"

    def test_moe_fixture_actually_has_experts(self):
        descriptor = load_descriptor(DATA / "moe-decoder.model.yaml")
        assert descriptor.topology.is_moe
        assert descriptor.topology.top_k > 0


class TestModelShapeTranslation:
    def test_hf_naming_maps_onto_dmi_naming(self):
        descriptor = parse_descriptor(VALID)
        shape = to_model_shape_config(descriptor.topology)
        # ModelShapeConfig speaks hidden_dim/intermediate_dim; descriptors speak
        # HF. This mapping is the only place the two vocabularies meet.
        assert shape.hidden_dim == descriptor.topology.hidden_size
        assert shape.intermediate_dim == descriptor.topology.intermediate_size
        assert shape.num_heads == descriptor.topology.num_attention_heads
        assert shape.num_kv_heads == descriptor.topology.num_kv_heads
        assert shape.head_dim == 128

    def test_translation_drives_the_same_availability_rules(self):
        # select_hook_specs suppresses MoE hooks on num_experts == 0; a
        # descriptor for a dense model must produce a shape that does the same.
        shape = to_model_shape_config(parse_descriptor(VALID).topology)
        assert shape.num_experts == 0
        assert shape.top_k == 0
