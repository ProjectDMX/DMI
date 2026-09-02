"""Validation against a model descriptor: availability, bounds, and schedule.

Every issue is addressed to a control path so the UI can attach it to the
checkbox or field that caused it.
"""
from __future__ import annotations

import pytest

from dmi.config import CaptureSchedule
from dmi.configuration import (
    ConfigValidationError,
    DMIConfig,
    LayerSelection,
    ObservationConfig,
    catalog_payload,
    describe_hooks,
    ensure_valid,
    grouped_hooks,
    hook_ids,
    is_valid,
    parse_descriptor,
    per_layer_hook_ids,
    validate_config,
)
from dmi.configuration import catalog_adapter
from dmi.configuration.architecture import architecture_payload
from dmi.hooks.catalog import GROUP_OTHER, HOOK_DEFS, PP_ANY, SHAPE_HIDDEN

pytestmark = pytest.mark.cpu


def _descriptor(**topology):
    base = dict(
        num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8
    )
    base.update(topology)
    return parse_descriptor(
        {
            "schema_version": 1,
            "model": {"id": "m", "name": "M", "architecture": "decoder_transformer"},
            "topology": base,
        }
    )


DENSE = _descriptor(intermediate_size=14336)
MOE = _descriptor(intermediate_size=768, num_experts=128, top_k=8)


def _fields(issues):
    return {issue.field for issue in issues}


def _errors(issues):
    return [issue for issue in issues if issue.is_error]


class TestValidConfigurations:
    def test_minimal_valid_config(self):
        config = DMIConfig(observations=ObservationConfig(hooks=["resid_pre"]))
        assert validate_config(config, DENSE) == []
        assert is_valid(config, DENSE)

    def test_attention_capture_over_a_layer_range(self):
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["q", "k", "v", "pattern"], layers=LayerSelection(8, 15)
            ),
            schedule=CaptureSchedule(step_stride=4),
        )
        assert is_valid(config, DENSE)

    def test_moe_hooks_valid_on_a_moe_model(self):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["router_logits", "topk_ids"])
        )
        assert is_valid(config, MOE)

    def test_full_layer_range_is_in_bounds(self):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"], layers=LayerSelection(0, 31))
        )
        assert is_valid(config, DENSE)


class TestHookValidation:
    def test_empty_selection_is_an_error(self):
        issues = validate_config(DMIConfig(), DENSE)
        assert "observations.hooks" in _fields(issues)
        assert not is_valid(DMIConfig(), DENSE)

    def test_unknown_hook_names_the_control(self):
        config = DMIConfig(observations=ObservationConfig(hooks=["nope"]))
        issues = _errors(validate_config(config, DENSE))
        assert issues[0].field == "observations.hooks.nope"
        assert "Unknown observation" in issues[0].message

    def test_moe_hook_unavailable_on_dense_model_explains_why(self):
        config = DMIConfig(observations=ObservationConfig(hooks=["router_logits"]))
        issues = _errors(validate_config(config, DENSE))
        assert issues[0].field == "observations.hooks.router_logits"
        assert "no experts" in issues[0].message

    def test_mlp_post_unavailable_without_intermediate_size(self):
        descriptor = _descriptor(intermediate_size=0)
        config = DMIConfig(observations=ObservationConfig(hooks=["mlp_post"]))
        assert not is_valid(config, descriptor)

    def test_duplicate_hook_warns_but_does_not_block(self):
        config = DMIConfig(observations=ObservationConfig(hooks=["q", "q"]))
        issues = validate_config(config, DENSE)
        assert any(i.severity == "warning" for i in issues)
        assert is_valid(config, DENSE)

    def test_without_descriptor_only_model_independent_checks_run(self):
        # No descriptor means no availability data, so a MoE hook cannot be
        # judged -- but an unknown name still can.
        config = DMIConfig(observations=ObservationConfig(hooks=["router_logits"]))
        assert is_valid(config, None)
        unknown = DMIConfig(observations=ObservationConfig(hooks=["nope"]))
        assert not is_valid(unknown, None)


class TestLayerValidation:
    def test_range_beyond_last_layer_is_an_error(self):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"], layers=LayerSelection(0, 99))
        )
        issues = _errors(validate_config(config, DENSE))
        assert issues[0].field == "observations.layers"
        assert "last layer is 31" in issues[0].message

    def test_range_starting_beyond_the_model_is_an_error(self):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"], layers=LayerSelection(40, 45))
        )
        assert not is_valid(config, DENSE)

    def test_range_with_only_global_hooks_warns_it_has_no_effect(self):
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["final_logits"], layers=LayerSelection(0, 3)
            )
        )
        issues = validate_config(config, DENSE)
        assert is_valid(config, DENSE)
        assert any(
            i.severity == "warning" and i.field == "observations.layers" for i in issues
        )

    def test_no_range_means_all_layers_and_never_warns(self):
        config = DMIConfig(observations=ObservationConfig(hooks=["final_logits"]))
        assert validate_config(config, DENSE) == []


class TestScheduleValidation:
    def test_both_phases_off_captures_nothing(self):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"]),
            schedule=CaptureSchedule(capture_prefill=False, capture_decode=False),
        )
        issues = _errors(validate_config(config, DENSE))
        assert issues[0].field == "schedule.phase"

    def test_mutated_stride_is_caught_even_though_ctor_would_reject_it(self):
        schedule = CaptureSchedule()
        schedule.step_stride = 0  # bypasses __post_init__
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"]), schedule=schedule
        )
        assert "schedule.step_stride" in _fields(_errors(validate_config(config, DENSE)))


class TestEnsureValid:
    def test_raises_with_every_error_attached(self):
        config = DMIConfig(
            observations=ObservationConfig(hooks=["nope", "router_logits"])
        )
        with pytest.raises(ConfigValidationError) as caught:
            ensure_valid(config, DENSE)
        assert len(caught.value.issues) == 2

    def test_passes_silently_when_valid(self):
        ensure_valid(DMIConfig(observations=ObservationConfig(hooks=["q"])), DENSE)


class TestArchitectureFallbackScope:
    """An unassigned hook must be drawn at its own scope.

    The renderer places every layer-scoped node inside the "Transformer layer
    x N" group, so a *global* hook landing in the fallback with layer scope
    would look as though it were captured once per layer, and as though the
    layer-range control applied to it.
    """

    def _with_extra_hook(self, monkeypatch, *, per_layer):
        from dmi.configuration import catalog_adapter
        from dmi.hooks.catalog import GROUP_OTHER, HOOK_DEFS, PP_ANY, SHAPE_HIDDEN

        newcomer = (
            98,
            "hook_unplaced",
            "unplaced",
            per_layer,
            GROUP_OTHER,
            False,
            SHAPE_HIDDEN,
            PP_ANY,
        )
        monkeypatch.setattr(
            catalog_adapter, "HOOK_DEFS", HOOK_DEFS + (newcomer,)
        )
        from dmi.configuration.architecture import architecture_payload

        return architecture_payload(_descriptor())["nodes"]

    def _node_holding(self, nodes, hook_id):
        for node in nodes:
            if any(hook["id"] == hook_id for hook in node["hooks"]):
                return node
        raise AssertionError(f"{hook_id!r} is not reachable from any node")

    def test_an_unassigned_per_layer_hook_is_layer_scoped(self, monkeypatch):
        nodes = self._with_extra_hook(monkeypatch, per_layer=True)

        node = self._node_holding(nodes, "unplaced")
        assert node["scope"] == "layer"

    def test_an_unassigned_global_hook_is_global_scoped(self, monkeypatch):
        nodes = self._with_extra_hook(monkeypatch, per_layer=False)

        node = self._node_holding(nodes, "unplaced")
        assert node["scope"] == "global"

    def test_fallback_hooks_of_both_scopes_are_separated(self, monkeypatch):
        """Layer and global leftovers cannot share one node."""
        from dmi.configuration import catalog_adapter
        from dmi.hooks.catalog import GROUP_OTHER, HOOK_DEFS, PP_ANY, SHAPE_HIDDEN
        from dmi.configuration.architecture import architecture_payload

        extras = (
            (98, "hook_a", "unplaced_layer", True, GROUP_OTHER, False, SHAPE_HIDDEN, PP_ANY),
            (97, "hook_b", "unplaced_global", False, GROUP_OTHER, False, SHAPE_HIDDEN, PP_ANY),
        )
        monkeypatch.setattr(catalog_adapter, "HOOK_DEFS", HOOK_DEFS + extras)

        nodes = architecture_payload(_descriptor())["nodes"]

        layer_node = self._node_holding(nodes, "unplaced_layer")
        global_node = self._node_holding(nodes, "unplaced_global")
        assert layer_node["id"] != global_node["id"]
        assert layer_node["scope"] == "layer"
        assert global_node["scope"] == "global"

    def test_every_node_scope_is_one_of_the_two_known_values(self, monkeypatch):
        nodes = self._with_extra_hook(monkeypatch, per_layer=False)

        assert {node["scope"] for node in nodes} <= {"layer", "global"}


class TestCatalogAdapter:
    def test_every_catalog_hook_is_described(self):
        assert {info.id for info in describe_hooks()} == set(hook_ids())

    def test_availability_mirrors_selector_suppression(self):
        unavailable = {i.id for i in describe_hooks(DENSE.topology) if not i.available}
        assert unavailable == {"router_logits", "topk_ids", "topk_weights"}
        assert not [i for i in describe_hooks(MOE.topology) if not i.available]

    def test_moe_hooks_are_regrouped_away_from_catalog_group_other(self):
        groups = grouped_hooks(MOE.topology)
        assert {i.id for i in groups["moe"]} == {
            "router_logits",
            "topk_ids",
            "topk_weights",
        }

    def test_per_layer_set_excludes_global_hooks(self):
        per_layer = per_layer_hook_ids()
        assert "q" in per_layer
        assert "final_logits" not in per_layer
        assert "token_ids" not in per_layer

    def test_catalog_payload_is_json_shaped(self):
        payload = catalog_payload(DENSE.topology)
        assert [group["id"] for group in payload["groups"]]
        for group in payload["groups"]:
            for hook in group["hooks"]:
                assert set(hook) == {
                    "id",
                    "label",
                    "group",
                    "per_layer",
                    "available",
                    "reason",
                }


class TestArchitectureLayout:
    """The diagram must stay reachable from the catalog.

    ``architecture.py`` promises that "extending ``HOOK_DEFS`` can never make
    an observation unreachable from the UI". That promise rests entirely on the
    trailing "Other observations" node, which no fixture exercises: today every
    catalog hook is assigned to a real block, so the fallback branch never runs
    and could rot unnoticed until the day someone adds a hook.
    """

    def test_every_catalog_hook_is_reachable_from_some_node(self):
        nodes = architecture_payload(DENSE)["nodes"]
        placed = {hook["id"] for node in nodes for hook in node["hooks"]}
        assert placed == set(hook_ids())

    def test_a_hook_with_no_assigned_block_lands_in_other_observations(
        self, monkeypatch
    ):
        # A hook the node list has never heard of -- exactly what adding an
        # entry to HOOK_DEFS without touching architecture.py produces.
        newcomer = (
            99,
            "hook_brand_new",
            "brand_new",
            True,
            GROUP_OTHER,
            False,
            SHAPE_HIDDEN,
            PP_ANY,
        )
        monkeypatch.setattr(
            catalog_adapter, "HOOK_DEFS", HOOK_DEFS + (newcomer,)
        )

        nodes = architecture_payload(DENSE)["nodes"]
        other = [node for node in nodes if node["id"] == "other"]
        assert other, "an unassigned hook must not disappear from the diagram"
        assert [hook["id"] for hook in other[0]["hooks"]] == ["brand_new"]
        # and it is still reachable overall, which is the actual guarantee
        placed = {hook["id"] for node in nodes for hook in node["hooks"]}
        assert "brand_new" in placed
