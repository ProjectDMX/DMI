"""Schema-level contracts for DMI-configurator's configuration model.

Pure dataclass validation: no YAML, no model, no runtime.
"""
from __future__ import annotations

import pytest

from dmi.configuration import (
    CONFIG_VERSION,
    DMIConfig,
    LayerSelection,
    ModelTopology,
    ObservationConfig,
    RuntimePolicy,
)

pytestmark = pytest.mark.cpu


class TestLayerSelection:
    def test_range_is_inclusive_on_both_ends(self):
        # The UI labels this range "Layers 8-15"; the label must not lie.
        selection = LayerSelection(8, 15)
        assert selection.count == 8
        assert selection.contains(8)
        assert selection.contains(15)
        assert not selection.contains(16)

    def test_single_layer_range(self):
        selection = LayerSelection(3, 3)
        assert selection.count == 1
        assert selection.contains(3)

    def test_global_hooks_are_never_contained(self):
        # Global specs carry layer_no == -1 and must not be swept up by a range.
        assert not LayerSelection(0, 31).contains(-1)

    @pytest.mark.parametrize("start,end", [(-1, 4), (5, 1)])
    def test_invalid_ranges_rejected(self, start, end):
        with pytest.raises(ValueError):
            LayerSelection(start, end)


class TestModelTopology:
    def test_head_dim_derived_when_absent(self):
        topology = ModelTopology(
            num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8
        )
        assert topology.effective_head_dim == 128

    def test_explicit_head_dim_wins(self):
        topology = ModelTopology(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=96,
        )
        assert topology.effective_head_dim == 96

    def test_moe_detected_from_expert_count(self):
        dense = ModelTopology(
            num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8
        )
        moe = ModelTopology(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            num_experts=128,
            top_k=8,
        )
        assert not dense.is_moe
        assert moe.is_moe

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_layers": 0},
            {"hidden_size": 0},
            {"num_attention_heads": 0},
            {"num_kv_heads": 0},
            {"num_kv_heads": 64},  # GQA cannot have more KV heads than heads
            {"intermediate_size": -1},
            {"top_k": 4},  # top_k without experts is incoherent
        ],
    )
    def test_invalid_topology_rejected(self, kwargs):
        base = dict(
            num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            ModelTopology(**base)


class TestDMIConfig:
    def test_defaults_are_empty_but_well_formed(self):
        config = DMIConfig()
        assert config.version == CONFIG_VERSION
        assert config.observations.hooks == []
        assert config.observations.layers is None
        assert config.policy is None
        assert config.schedule.step_stride == 1

    def test_policy_values_are_stable_strings(self):
        # These strings are the on-disk contract; changing one breaks old files.
        assert RuntimePolicy.COMPLETENESS.value == "completeness"
        assert RuntimePolicy.BALANCED.value == "balanced"
        assert RuntimePolicy.PERFORMANCE.value == "performance"

    def test_observations_hold_structured_layers_not_strings(self):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"], layers=LayerSelection(8, 15))
        )
        assert isinstance(config.observations.layers, LayerSelection)
        assert config.observations.layers.start == 8
