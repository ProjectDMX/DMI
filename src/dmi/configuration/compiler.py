"""Compile a ``DMIConfig`` into the objects the DMI runtime consumes.

This is the half of the pipeline that knows about the runtime and nothing
about YAML::

    config.yaml -> load_config() -> DMIConfig -> compile_config()
                -> CompiledDMIConfig -> DMI runtime

``compile_config`` needs a *bound* spec list, not just a descriptor:
``select_hook_specs`` filters specs the model produced via
``model.get_hook_specs()``; it does not build them from a config. That is why
:class:`ModelContext` carries specs alongside the shape configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import CaptureSchedule
from .compatibility import to_legacy_hook_selection
from .schema import DMIConfig, RuntimePolicy


@dataclass
class ModelContext:
    """What compilation needs from a live model.

    ``specs`` comes from ``model.get_hook_specs()``; ``shape`` is the
    ``ModelShapeConfig`` the adapter detected, and drives the same
    availability suppression the configurator showed at design time.
    """

    specs: list = field(default_factory=list)
    shape: Optional[Any] = None


@dataclass
class CompiledDMIConfig:
    """Execution-ready configuration.

    Deliberately not a dictionary: these are the objects DMI already takes.
    """

    hook_specs: list
    schedule: CaptureSchedule
    policy: Optional[RuntimePolicy] = None

    @property
    def selected_layers(self) -> list[int]:
        """Distinct per-layer indices present after compilation, sorted."""
        return sorted({spec.layer_no for spec in self.hook_specs if spec.layer_no >= 0})


def compile_config(config: DMIConfig, model_context: ModelContext) -> CompiledDMIConfig:
    """Resolve a user configuration against a live model's hooks.

    Two filtering stages, in order:

    1. ``select_hook_specs`` -- which *kinds* of observation, via the existing
       selection string interface, including its availability suppression.
    2. ``filter_by_layers`` -- *where*, if a layer range was given.

    PP/TP filtering is intentionally not applied here. It depends on rank
    placement the adapter owns, and ``attach_model`` already applies it after
    selection.
    """
    from ..hooks.selection import filter_by_layers, select_hook_specs

    specs = select_hook_specs(
        model_context.specs,
        to_legacy_hook_selection(config.observations),
        cfg=model_context.shape,
    )

    layers = config.observations.layers
    if layers is not None:
        specs = filter_by_layers(specs, layers.start, layers.end)

    return CompiledDMIConfig(
        hook_specs=specs,
        schedule=config.schedule,
        policy=config.policy,
    )


__all__ = ["ModelContext", "CompiledDMIConfig", "compile_config"]
