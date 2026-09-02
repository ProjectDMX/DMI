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
    2. ``hook_belongs_to_layers`` -- *where*, if a layer range was given.

    Both stages are pure: unlike ``filter_by_layers`` (which additionally
    disables the HookPoints it drops, and therefore belongs to
    ``attach_model``), nothing here touches the model, so this stays safe to
    call while authoring against a live model.

    PP/TP filtering is intentionally not applied here. It depends on rank
    placement the adapter owns, and ``attach_model`` already applies it after
    selection.
    """
    from ..hooks.selection import hook_belongs_to_layers, select_hook_specs

    specs = select_hook_specs(
        model_context.specs,
        to_legacy_hook_selection(config.observations),
        cfg=model_context.shape,
    )

    layers = config.observations.layers
    if layers is not None:
        specs = [
            spec for spec in specs
            if hook_belongs_to_layers(spec, layers.start, layers.end)
        ]

    return CompiledDMIConfig(
        hook_specs=specs,
        schedule=config.schedule,
        policy=config.policy,
    )


def attach_config(adapter, model, config: DMIConfig) -> None:
    """Install hooks on ``model`` according to ``config``.

    This is the *executing* counterpart to :func:`compile_config`. The two are
    deliberately different operations on the same configuration:

    ``compile_config``
        Answers "what would this configuration select?" without touching the
        model. Pure, rank-agnostic, and safe to call while authoring -- which
        is why it skips PP/TP filtering.

    ``attach_config``
        Actually installs the hooks, through the adapter's own
        ``attach_model``. That path additionally applies the PP/TP filters and
        establishes the enabled/disabled state across *every* spec, so it --
        not a pre-filtered spec list -- has to own selection.

    Handing ``attach_model`` a spec list from ``compile_config`` would look
    tempting and be wrong: ``HookPoint.enabled`` defaults to ``True``, and only
    ``apply_hook_selection`` walks the unselected specs to turn them off. A
    pre-filtered list would leave every deselected hook live.
    """
    # Pass `layers` only when there is a range to apply. Adapters are a public
    # extension point (see the v1 integration API), and an adapter that
    # overrides attach_model without the keyword would otherwise raise
    # TypeError for *every* configuration -- including the majority that set
    # no layer range and need nothing from it. A config that does set a range
    # still fails loudly on such an adapter, which is correct: the range would
    # not be honoured, and silently dropping it is the outcome this whole
    # change exists to prevent.
    kwargs = {}
    if config.observations.layers is not None:
        kwargs["layers"] = config.observations.layers

    adapter.attach_model(
        model,
        to_legacy_hook_selection(config.observations),
        **kwargs,
    )


__all__ = [
    "ModelContext",
    "CompiledDMIConfig",
    "compile_config",
    "attach_config",
]
