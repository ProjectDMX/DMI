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


# ---------------------------------------------------------------------------
# Strictness: every section refuses keys this build does not understand
# ---------------------------------------------------------------------------


def test_a_misspelled_layers_key_is_refused_not_ignored():
    """The dangerous typo: ignoring it would silently capture every layer."""
    with pytest.raises(ConfigurationError, match="Unknown field.*layer"):
        parse_config(
            {
                "version": 1,
                "observations": {
                    "hooks": ["q"],
                    "layer": {"start": 8, "end": 15},
                },
            }
        )


def test_a_misspelled_hooks_key_is_refused():
    with pytest.raises(ConfigurationError, match="Unknown field.*hook"):
        parse_config({"version": 1, "observations": {"hook": ["q"]}})


def test_a_misspelled_top_level_key_is_refused():
    with pytest.raises(ConfigurationError, match="Unknown field.*polcy"):
        parse_config(
            {
                "version": 1,
                "observations": {"hooks": ["q"]},
                "polcy": {"objective": "performance"},
            }
        )


def test_the_unknown_field_error_lists_what_is_accepted():
    with pytest.raises(ConfigurationError) as excinfo:
        parse_config({"version": 1, "observations": {"nope": 1}})

    message = str(excinfo.value)
    assert "hooks" in message and "layers" in message


def test_every_section_is_equally_strict():
    """schedule was strict from the start; the others now match it."""
    for document, where in [
        ({"version": 1, "bogus": 1}, "top level"),
        ({"version": 1, "observations": {"bogus": 1}}, "observations"),
        ({"version": 1, "schedule": {"bogus": 1}}, "schedule"),
    ]:
        with pytest.raises(ConfigurationError, match="Unknown field"):
            parse_config(document)


def test_a_layers_key_set_to_null_is_still_accepted():
    """The UI sends layers: null for "all layers"; that is a known key."""
    config = parse_config(
        {"version": 1, "observations": {"hooks": ["q"], "layers": None}}
    )

    assert config.observations.layers is None


# ---------------------------------------------------------------------------
# version is required, not defaulted
# ---------------------------------------------------------------------------


def test_a_document_without_a_version_is_refused():
    """Defaulting is the guess that version dispatch exists to avoid."""
    with pytest.raises(ConfigurationError, match="missing 'version'"):
        parse_config({"observations": {"hooks": ["q"]}, "schedule": {}})


def test_the_missing_version_error_names_the_version_to_add():
    with pytest.raises(ConfigurationError, match="version: 1"):
        parse_config({"observations": {"hooks": ["q"]}})


@pytest.mark.parametrize("bad", ["1", 1.0, True, None, [1]])
def test_a_non_integer_version_is_malformed_not_unsupported(bad):
    """A quoted YAML version is a typo, not a version from the future."""
    with pytest.raises(ConfigurationError, match="must be an integer"):
        parse_config({"version": bad, "observations": {"hooks": ["q"]}})


def test_a_future_version_is_still_refused_as_unsupported():
    with pytest.raises(UnsupportedConfigVersion):
        parse_config({"version": 2, "observations": {"hooks": ["q"]}})


def test_every_dumped_document_carries_a_version():
    """dump must never produce a file its own parser would reject."""
    text = dump_config(
        DMIConfig(observations=ObservationConfig(hooks=["resid_pre"]))
    )

    assert "version:" in text
    assert parse_config(pyyaml.safe_load(text)).version == 1


# ---------------------------------------------------------------------------
# save_config error contract matches load_config
# ---------------------------------------------------------------------------


def test_saving_into_a_missing_directory_reports_a_configuration_error(tmp_path):
    config = DMIConfig(observations=ObservationConfig(hooks=["resid_pre"]))

    with pytest.raises(ConfigurationError, match="Cannot write configuration"):
        save_config(config, tmp_path / "no-such-dir" / "out.dmi.yaml")


def test_save_and_load_report_filesystem_trouble_the_same_way(tmp_path):
    """Both halves of the API raise ConfigurationError, not raw OSError."""
    missing = tmp_path / "absent" / "out.dmi.yaml"
    config = DMIConfig(observations=ObservationConfig(hooks=["resid_pre"]))

    with pytest.raises(ConfigurationError):
        save_config(config, missing)
    with pytest.raises(ConfigurationError):
        load_config(missing)
