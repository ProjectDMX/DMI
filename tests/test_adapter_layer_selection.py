"""Layer-range filtering through ``BackendAdapter.attach_model``.

The configurator lets a user author a layer range, but a range only means
something if it survives the trip into the runtime.  These tests pin the
wiring: ``attach_model(..., layers=...)`` applies the range, disables the
hook points it drops, and leaves global observations alone.

No GPU, no compiled backend: ``install_ring_hooks`` is attribute assignment
and accepts ``ring_payload=None``, so this belongs in the CPU gate.
"""
from __future__ import annotations

import pytest

from dmi.adapters.base import BackendAdapter
from dmi.configuration import DMIConfig, LayerSelection, ObservationConfig
from dmi.configuration.compiler import attach_config
from dmi.hooks.point import HookPoint
from dmi.hooks.specs import (
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HookSpec,
    ModelShapeConfig,
)

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTransport:
    def __init__(self) -> None:
        self.null_offload = False
        self.force_eager = False
        self._active_specs: list = []
        self._using_forward_hooks = False
        self._ring_payload = None
        self._model_cfg = None

    def set_model_cfg(self, cfg):
        self._model_cfg = cfg


class FakeEngine:
    def __init__(self) -> None:
        self._ring_transport = FakeTransport()
        self._ring_engine = None


class FakeModel:
    """A model that self-describes per-layer and global hooks."""

    def __init__(self, num_layers: int = 8) -> None:
        self.specs: list[HookSpec] = []
        for layer_no in range(num_layers):
            for hook_type in (HOOK_TYPE_RESID_PRE, HOOK_TYPE_Q):
                self.specs.append(
                    HookSpec(
                        hook_type=hook_type,
                        module=HookPoint(),
                        layer_no=layer_no,
                    )
                )
        # Global hooks carry layer_no == -1.
        for hook_type in (HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_FINAL_LOGITS):
            self.specs.append(
                HookSpec(hook_type=hook_type, module=HookPoint(), layer_no=-1)
            )

    def get_hook_specs(self) -> list[HookSpec]:
        return list(self.specs)


class StubAdapter(BackendAdapter):
    """Single-rank adapter with a fixed model shape."""

    def __init__(self, engine, model_id="test_model", shape=None):
        super().__init__(engine, model_id)
        self._shape = shape or _make_shape()

    def detect_model_shape(self, model):
        return self._shape

    def detect_parallel_ranks(self):
        return (0, 0, 0, 0)

    def is_pp_first(self):
        return True

    def is_pp_last(self):
        return True

    def build_step_context(self, *raw):
        raise NotImplementedError

    def on_capacity_exceeded(self, ctx):
        raise NotImplementedError


def _make_shape() -> ModelShapeConfig:
    import torch

    return ModelShapeConfig(
        hidden_dim=512,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        dtype=torch.float16,
        vocab_size=32000,
        intermediate_dim=1376,
    )


def _make_adapter(num_layers: int = 8):
    engine = FakeEngine()
    adapter = StubAdapter(engine)
    return adapter, FakeModel(num_layers=num_layers)


def _layers_of(specs) -> set[int]:
    return {spec.layer_no for spec in specs if spec.layer_no >= 0}


# ---------------------------------------------------------------------------
# attach_model(layers=...)
# ---------------------------------------------------------------------------


def test_no_layers_argument_keeps_every_layer():
    adapter, model = _make_adapter(num_layers=8)

    adapter.attach_model(model, "resid_pre,q")

    assert _layers_of(adapter.active_specs) == set(range(8))


def test_layer_range_keeps_only_the_selected_layers():
    adapter, model = _make_adapter(num_layers=8)

    adapter.attach_model(model, "resid_pre,q", layers=LayerSelection(2, 4))

    assert _layers_of(adapter.active_specs) == {2, 3, 4}


def test_layer_range_is_inclusive_of_both_bounds():
    adapter, model = _make_adapter(num_layers=8)

    adapter.attach_model(model, "resid_pre", layers=LayerSelection(3, 3))

    assert _layers_of(adapter.active_specs) == {3}


def test_layer_range_never_drops_global_observations():
    adapter, model = _make_adapter(num_layers=8)

    adapter.attach_model(
        model, "resid_pre,token_ids,final_logits", layers=LayerSelection(0, 1)
    )

    kept = {spec.hook_type for spec in adapter.active_specs}
    assert HOOK_TYPE_TOKEN_IDS in kept
    assert HOOK_TYPE_FINAL_LOGITS in kept


def test_dropped_layers_have_their_hook_points_disabled():
    """A filtered spec must be inert, not merely absent from active_specs.

    ``HookPoint.enabled`` defaults to True, so a filter that only shortened
    the list would leave the dropped hooks firing.
    """
    adapter, model = _make_adapter(num_layers=6)

    adapter.attach_model(model, "resid_pre,q", layers=LayerSelection(1, 2))

    kept = {id(spec) for spec in adapter.active_specs}
    for spec in model.specs:
        if spec.layer_no < 0:
            continue
        expected = id(spec) in kept
        assert spec.module.enabled is expected, (
            f"layer {spec.layer_no} hook_type {spec.hook_type} "
            f"enabled={spec.module.enabled}, expected {expected}"
        )


def test_layer_filter_composes_with_hook_selection():
    """Selection decides which kinds; the range decides where."""
    adapter, model = _make_adapter(num_layers=8)

    adapter.attach_model(model, "q", layers=LayerSelection(5, 6))

    assert {spec.hook_type for spec in adapter.active_specs} == {HOOK_TYPE_Q}
    assert _layers_of(adapter.active_specs) == {5, 6}


def test_installed_specs_are_published_to_the_transport():
    adapter, model = _make_adapter(num_layers=4)

    adapter.attach_model(model, "resid_pre", layers=LayerSelection(0, 1))

    assert adapter.transport._active_specs == adapter.active_specs
    assert adapter.transport._using_forward_hooks is True


def test_inverted_range_is_rejected_at_construction():
    with pytest.raises(ValueError, match="layer end must be >= start"):
        LayerSelection(5, 2)


# ---------------------------------------------------------------------------
# attach_config: DMIConfig -> attach_model
# ---------------------------------------------------------------------------


def test_attach_config_applies_hooks_and_layers_from_a_config():
    adapter, model = _make_adapter(num_layers=8)
    config = DMIConfig(
        observations=ObservationConfig(
            hooks=["resid_pre", "q"], layers=LayerSelection(4, 6)
        )
    )

    attach_config(adapter, model, config)

    assert _layers_of(adapter.active_specs) == {4, 5, 6}
    assert {spec.hook_type for spec in adapter.active_specs} == {
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_Q,
    }


def test_attach_config_without_a_layer_range_keeps_every_layer():
    adapter, model = _make_adapter(num_layers=5)
    config = DMIConfig(
        observations=ObservationConfig(hooks=["resid_pre"], layers=None)
    )

    attach_config(adapter, model, config)

    assert _layers_of(adapter.active_specs) == set(range(5))


def test_attach_config_disables_hooks_the_config_did_not_select():
    """The regression the docstring in attach_config warns about."""
    adapter, model = _make_adapter(num_layers=4)
    config = DMIConfig(
        observations=ObservationConfig(hooks=["resid_pre"], layers=None)
    )

    attach_config(adapter, model, config)

    for spec in model.specs:
        if spec.hook_type == HOOK_TYPE_Q:
            assert spec.module.enabled is False
