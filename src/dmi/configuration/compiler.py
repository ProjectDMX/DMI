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
from .errors import ConfigurationError
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

    _reject_requested_hooks_that_cannot_fire(config, specs)

    return CompiledDMIConfig(
        hook_specs=specs,
        schedule=config.schedule,
        policy=config.policy,
    )


def _reject_requested_hooks_that_cannot_fire(
    config: DMIConfig, specs: list
) -> None:
    """Refuse a selection the live model cannot satisfy.

    Topology-only availability (what the descriptor claims) can disagree with
    what the attached model actually exposes: a decoder Llama carries no
    ``pos_embed`` or ``mlp_post`` spec even when the descriptor marks them
    available. Selection then returns zero specs for the request and nothing
    anywhere complains -- 'valid' would certify captures that never fire.
    The live spec list is the executable truth, so it decides: a requested
    hook type absent from ``specs`` raises, naming the hook.
    """
    from ..hooks.catalog import HOOK_DEFS
    from .errors import ConfigValidationError
    from .validation import SEVERITY_ERROR, Issue

    id_by_short = {short: hid for hid, _act, short, *_rest in HOOK_DEFS}
    surviving = {spec.hook_type for spec in specs}
    missing = [
        short
        for short in dict.fromkeys(config.observations.hooks)
        if short in id_by_short and id_by_short[short] not in surviving
    ]
    if missing:
        raise ConfigValidationError(
            [
                Issue(
                    SEVERITY_ERROR,
                    f"observations.hooks.{short}",
                    f"{short!r} does not match any hook the attached model "
                    "exposes, so it can never fire. Remove it or select an "
                    "observation the model produces.",
                )
                for short in missing
            ]
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

    if kwargs and not _accepts_layers(adapter):
        # The pinned vLLM integration is exactly this shape. A bare TypeError
        # from inside attach_model names nothing; say here what the keyword
        # means, which artifact carries the gap, and what resolves it.
        raise ConfigurationError(
            f"{type(adapter).__name__}.attach_model does not accept a "
            "`layers` keyword, so this configuration's layer range "
            f"{config.observations.layers.start}-"
            f"{config.observations.layers.end} cannot be applied: a range "
            "silently dropped would supersede captures with the unfiltered "
            "pack. Update the adapter integration to a revision that accepts "
            "attach_config(..., LayerSelection(...)), or clear "
            "observations.layers in the configuration. (vLLM: "
            "https://github.com/ProjectDMX/DMI-vLLM-Integration/issues/20)"
        )

    adapter.attach_model(
        model,
        to_legacy_hook_selection(config.observations),
        **kwargs,
    )


def _accepts_layers(adapter) -> bool:
    """Can this adapter's ``attach_model`` take a ``layers`` keyword?"""
    import inspect

    try:
        parameters = inspect.signature(adapter.attach_model).parameters
    except (TypeError, ValueError):
        return False
    if "layers" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


__all__ = [
    "ModelContext",
    "CompiledDMIConfig",
    "compile_config",
    "attach_config",
]
