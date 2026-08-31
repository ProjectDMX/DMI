"""Serialization contract: canonical form, round-tripping, and golden files.

The YAML is a versioned artifact users keep in git, so these tests guard the
on-disk shape as much as the parser.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml as pyyaml

from dmi.config import CaptureSchedule
from dmi.configuration import (
    ConfigurationError,
    DMIConfig,
    LayerSelection,
    ObservationConfig,
    RuntimePolicy,
    UnsupportedConfigVersion,
    config_to_dict,
    dump_config,
    load_config,
    normalize_config,
    parse_config,
    save_config,
)

pytestmark = pytest.mark.cpu

GOLDEN = Path(__file__).resolve().parent / "golden"

RICH = DMIConfig(
    observations=ObservationConfig(
        hooks=["q", "k", "v", "pattern", "resid_pre", "mlp_out"],
        layers=LayerSelection(8, 15),
    ),
    schedule=CaptureSchedule(step_stride=4, request_stride=1),
    policy=RuntimePolicy.BALANCED,
)


class TestRoundTrip:
    def test_parse_of_dump_equals_normalize(self):
        # The contract from the design doc, stated exactly.
        assert parse_config(pyyaml.safe_load(dump_config(RICH))) == normalize_config(RICH)

    def test_normalize_is_idempotent(self):
        once = normalize_config(RICH)
        assert normalize_config(once) == once

    def test_dump_is_stable_across_reparse(self):
        text = dump_config(RICH)
        assert dump_config(parse_config(pyyaml.safe_load(text))) == text

    def test_disk_round_trip(self, tmp_path):
        target = tmp_path / "capture.dmi.yaml"
        save_config(RICH, target)
        assert load_config(target) == normalize_config(RICH)

    def test_minimal_config_round_trips(self, tmp_path):
        config = DMIConfig(observations=ObservationConfig(hooks=["resid_pre"]))
        target = tmp_path / "min.yaml"
        save_config(config, target)
        assert load_config(target) == normalize_config(config)

    def test_advanced_schedule_fields_survive(self, tmp_path):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"]),
            schedule=CaptureSchedule(
                step_stride=3,
                step_offset=1,
                warmup_steps=8,
                request_stride=2,
                request_offset=1,
                warmup_requests=4,
            ),
        )
        target = tmp_path / "adv.yaml"
        save_config(config, target)
        assert load_config(target).schedule == config.schedule


class TestCanonicalForm:
    def test_hooks_are_sorted_into_catalog_order(self):
        document = config_to_dict(
            DMIConfig(observations=ObservationConfig(hooks=["v", "q", "pattern", "k"]))
        )
        assert document["observations"]["hooks"] == ["pattern", "q", "k", "v"]

    def test_duplicate_hooks_collapse(self):
        document = config_to_dict(
            DMIConfig(observations=ObservationConfig(hooks=["q", "q", "k", "q"]))
        )
        assert document["observations"]["hooks"] == ["q", "k"]

    def test_unknown_hooks_are_preserved_not_dropped(self):
        # Normalization is not validation. Dropping an unknown name here would
        # hide the error the validator is supposed to report.
        document = config_to_dict(
            DMIConfig(observations=ObservationConfig(hooks=["zzz", "q"]))
        )
        assert document["observations"]["hooks"] == ["q", "zzz"]

    def test_layers_omitted_when_all_layers(self):
        document = config_to_dict(DMIConfig(observations=ObservationConfig(hooks=["q"])))
        assert "layers" not in document["observations"]

    def test_policy_omitted_when_unset(self):
        assert "policy" not in config_to_dict(DMIConfig())

    def test_default_advanced_schedule_fields_omitted(self):
        document = config_to_dict(DMIConfig())
        assert set(document["schedule"]) == {
            "step_stride",
            "request_stride",
            "capture_prefill",
            "capture_decode",
        }


class TestParseErrors:
    def test_unsupported_version(self):
        with pytest.raises(UnsupportedConfigVersion):
            parse_config({"version": 2, "observations": {"hooks": ["q"]}})

    def test_comma_string_hooks_points_at_the_adapter(self):
        # The legacy flat form is supported, but through the compatibility
        # adapter -- not by quietly accepting it here.
        with pytest.raises(ConfigurationError, match="compatibility"):
            parse_config({"version": 1, "observations": {"hooks": "q,k,v"}})

    def test_unknown_schedule_field_rejected(self):
        with pytest.raises(ConfigurationError, match="Unknown field"):
            parse_config({"version": 1, "schedule": {"step_strid": 4}})

    def test_invalid_schedule_value_rejected(self):
        with pytest.raises(ConfigurationError, match="Invalid schedule"):
            parse_config({"version": 1, "schedule": {"step_stride": 0}})

    def test_incomplete_layer_range_rejected(self):
        with pytest.raises(ConfigurationError, match="missing"):
            parse_config({"version": 1, "observations": {"layers": {"start": 3}}})

    def test_inverted_layer_range_rejected(self):
        with pytest.raises(ConfigurationError, match="Invalid layer range"):
            parse_config(
                {"version": 1, "observations": {"layers": {"start": 9, "end": 2}}}
            )

    def test_unknown_policy_objective_rejected(self):
        with pytest.raises(ConfigurationError, match="Unknown policy objective"):
            parse_config({"version": 1, "policy": {"objective": "fastest"}})

    def test_empty_document_rejected(self):
        with pytest.raises(ConfigurationError):
            parse_config(None)

    def test_missing_file_reports_path(self, tmp_path):
        with pytest.raises(ConfigurationError, match="Cannot read configuration"):
            load_config(tmp_path / "absent.yaml")


class TestGoldenFiles:
    """Guard the configuration contract against accidental change.

    A diff here means the on-disk format moved. That may be intentional -- but
    it must be deliberate, and it must come with a version bump if it breaks
    existing files.
    """

    @pytest.mark.parametrize(
        "name", ["qwen3-basic", "qwen3-attention", "qwen3-moe"]
    )
    def test_golden_file_is_canonical(self, name):
        path = GOLDEN / f"{name}.yaml"
        assert dump_config(load_config(path)) == path.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "name", ["qwen3-basic", "qwen3-attention", "qwen3-moe"]
    )
    def test_golden_file_round_trips(self, name):
        config = load_config(GOLDEN / f"{name}.yaml")
        assert parse_config(pyyaml.safe_load(dump_config(config))) == config

    def test_golden_attention_content(self):
        config = load_config(GOLDEN / "qwen3-attention.yaml")
        assert config.observations.layers == LayerSelection(8, 15)
        assert config.schedule.step_stride == 4
        assert config.policy is RuntimePolicy.BALANCED

    def test_golden_moe_content(self):
        config = load_config(GOLDEN / "qwen3-moe.yaml")
        assert "router_logits" in config.observations.hooks
        assert config.schedule.capture_decode is False
        assert config.schedule.warmup_steps == 16
        assert config.policy is RuntimePolicy.PERFORMANCE
