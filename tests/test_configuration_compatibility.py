"""Compatibility with the existing hook-selection interface, and compilation.

Covers the bridge in both directions plus ``compile_config``, which is where a
human configuration becomes DMI execution configuration.
"""
from __future__ import annotations

import pytest

from dmi.config import CaptureSchedule
from dmi.configuration import (
    CompiledDMIConfig,
    ConfigValidationError,
    DMIConfig,
    LayerSelection,
    ModelContext,
    ObservationConfig,
    RuntimePolicy,
    compile_config,
    from_legacy_hook_selection,
    load_descriptor,
    to_legacy_hook_selection,
    to_model_shape_config,
)
from dmi.hooks.selection import filter_by_layers, hook_belongs_to_layers
from dmi.hooks.specs import (
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_Q,
    HOOK_TYPE_ROUTER_LOGITS,
    HookSpec,
)

pytestmark = pytest.mark.cpu

NUM_LAYERS = 32


class _Point:
    """Stand-in for a HookPoint: the filters only touch ``enabled``."""

    def __init__(self):
        self.enabled = True


def _specs(hook_types=(HOOK_TYPE_Q, HOOK_TYPE_K), layers=NUM_LAYERS, globals_=()):
    specs = [
        HookSpec(hook_type=hook_type, module=_Point(), layer_no=layer)
        for layer in range(layers)
        for hook_type in hook_types
    ]
    specs.extend(
        HookSpec(hook_type=hook_type, module=_Point(), layer_no=-1)
        for hook_type in globals_
    )
    return specs


class TestLegacySelectionString:
    def test_structured_to_string(self):
        observations = ObservationConfig(hooks=["q", "k", "v", "pattern"])
        assert to_legacy_hook_selection(observations) == "q,k,v,pattern"

    def test_duplicates_collapse_in_order(self):
        observations = ObservationConfig(hooks=["q", "k", "q"])
        assert to_legacy_hook_selection(observations) == "q,k"

    def test_layers_are_not_encoded_into_the_string(self):
        # A selection string cannot express layers; inventing syntax such as
        # "pattern@8-15" is exactly what the design rules out.
        observations = ObservationConfig(hooks=["pattern"], layers=LayerSelection(8, 15))
        assert to_legacy_hook_selection(observations) == "pattern"

    def test_empty_selection_rejected(self):
        with pytest.raises(ValueError):
            to_legacy_hook_selection(ObservationConfig(hooks=[]))

    def test_string_to_structured(self):
        observations = from_legacy_hook_selection("q,k,v")
        assert set(observations.hooks) == {"q", "k", "v"}
        assert observations.layers is None

    def test_presets_expand_through_the_existing_selector(self):
        # Resolved by dmi.hooks.selection, not by a table copied into the
        # configuration layer.
        observations = from_legacy_hook_selection("hf-only")
        assert set(observations.hooks) == {
            "resid_pre",
            "pattern",
            "final_ln",
            "final_logits",
        }

    def test_unknown_token_rejected(self):
        with pytest.raises(ValueError, match="Unknown hook selection"):
            from_legacy_hook_selection("not-a-preset")

    def test_string_round_trip_preserves_the_hook_set(self):
        original = ObservationConfig(hooks=["q", "k", "v", "pattern"])
        restored = from_legacy_hook_selection(to_legacy_hook_selection(original))
        assert set(restored.hooks) == set(original.hooks)


class TestLayerFilter:
    def test_keeps_only_layers_in_the_inclusive_range(self):
        specs = _specs(hook_types=(HOOK_TYPE_Q,))
        kept = filter_by_layers(specs, 8, 15)
        assert [spec.layer_no for spec in kept] == list(range(8, 16))

    def test_global_hooks_are_never_dropped(self):
        specs = _specs(hook_types=(HOOK_TYPE_Q,), globals_=(HOOK_TYPE_FINAL_LOGITS,))
        kept = filter_by_layers(specs, 8, 15)
        assert any(spec.layer_no == -1 for spec in kept)
        assert hook_belongs_to_layers(
            HookSpec(hook_type=HOOK_TYPE_FINAL_LOGITS, module=None, layer_no=-1), 8, 15
        )

    def test_dropped_specs_are_disabled_not_just_filtered(self):
        # Matches filter_by_pp_rank / filter_by_tp_rank: a dropped hook must
        # not keep firing.
        specs = _specs(hook_types=(HOOK_TYPE_Q,))
        filter_by_layers(specs, 8, 15)
        assert all(not spec.module.enabled for spec in specs if spec.layer_no < 8)
        assert all(spec.module.enabled for spec in specs if 8 <= spec.layer_no <= 15)

    def test_single_layer_range(self):
        kept = filter_by_layers(_specs(hook_types=(HOOK_TYPE_Q,)), 5, 5)
        assert [spec.layer_no for spec in kept] == [5]

    def test_inverted_range_rejected(self):
        with pytest.raises(ValueError, match="exceeds end"):
            filter_by_layers(_specs(), 15, 8)

    def test_unbound_spec_rejected_like_the_sibling_filters(self):
        specs = [HookSpec(hook_type=HOOK_TYPE_Q, module=None, layer_no=0)]
        with pytest.raises(RuntimeError, match="bound executable HookSpecs"):
            filter_by_layers(specs, 5, 9)


class TestCompileConfig:
    def test_selects_by_hook_type_then_by_layer(self):
        specs = _specs(
            hook_types=(HOOK_TYPE_Q, HOOK_TYPE_K), globals_=(HOOK_TYPE_FINAL_LOGITS,)
        )
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"], layers=LayerSelection(8, 15))
        )
        compiled = compile_config(config, ModelContext(specs=specs))

        assert isinstance(compiled, CompiledDMIConfig)
        assert {spec.hook_type for spec in compiled.hook_specs} == {HOOK_TYPE_Q}
        assert compiled.selected_layers == list(range(8, 16))

    def test_global_hooks_survive_a_layer_range(self):
        specs = _specs(hook_types=(HOOK_TYPE_Q,), globals_=(HOOK_TYPE_FINAL_LOGITS,))
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["q", "final_logits"], layers=LayerSelection(8, 15)
            )
        )
        compiled = compile_config(config, ModelContext(specs=specs))
        assert len(compiled.hook_specs) == 9  # 8 layers of q, plus final_logits
        assert any(spec.layer_no == -1 for spec in compiled.hook_specs)

    def test_no_layer_range_keeps_every_layer(self):
        specs = _specs(hook_types=(HOOK_TYPE_Q,))
        config = DMIConfig(observations=ObservationConfig(hooks=["q"]))
        compiled = compile_config(config, ModelContext(specs=specs))
        assert len(compiled.hook_specs) == NUM_LAYERS

    def test_schedule_and_policy_pass_through_untouched(self):
        schedule = CaptureSchedule(step_stride=4, warmup_steps=8)
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"]),
            schedule=schedule,
            policy=RuntimePolicy.PERFORMANCE,
        )
        compiled = compile_config(
            config, ModelContext(specs=_specs(hook_types=(HOOK_TYPE_Q,)))
        )
        assert compiled.schedule is schedule
        assert compiled.policy is RuntimePolicy.PERFORMANCE

    def test_model_shape_suppresses_unavailable_hooks_at_compile_time(self):
        # A dense model simply does not carry a router_logits spec; asking
        # for one is refused rather than silently compiling to less.
        descriptor = load_descriptor(
            "examples/model_descriptors/llama3-8b.yaml"
        )
        specs = _specs(hook_types=(HOOK_TYPE_Q, HOOK_TYPE_ROUTER_LOGITS))
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q", "router_logits"])
        )
        with pytest.raises(ConfigValidationError, match="router_logits"):
            compile_config(
                config,
                ModelContext(specs=specs, shape=to_model_shape_config(descriptor.topology)),
            )


class TestEndToEndIntegration:
    def test_descriptor_to_catalog_to_config_to_specs(self):
        """descriptor -> availability -> configuration -> legacy selector."""
        descriptor = load_descriptor("examples/model_descriptors/llama3-8b.yaml")

        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["q", "k", "mlp_out", "final_logits"],
                layers=LayerSelection(8, 15),
            ),
            schedule=CaptureSchedule(step_stride=4),
        )

        specs = _specs(
            hook_types=(HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_MLP_OUT),
            layers=descriptor.topology.num_layers,
            globals_=(HOOK_TYPE_FINAL_LOGITS,),
        )
        compiled = compile_config(
            config,
            ModelContext(specs=specs, shape=to_model_shape_config(descriptor.topology)),
        )

        # 3 per-layer hooks over 8 layers, plus one global.
        assert len(compiled.hook_specs) == 3 * 8 + 1
        assert compiled.selected_layers == list(range(8, 16))
        assert compiled.schedule.step_stride == 4
