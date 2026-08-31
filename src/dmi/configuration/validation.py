"""Validate a user configuration against a model descriptor.

Every issue carries the dotted path of the control that produced it, so the UI
can attach "Router logits -- unavailable for this model" to that checkbox
instead of showing one opaque error at the top of the page.

Availability rules are not restated here: they come from
:mod:`dmi.configuration.catalog_adapter`, which in turn mirrors
``dmi.hooks.selection.select_hook_specs``. One rule, one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .catalog_adapter import describe_hooks, per_layer_hook_ids
from .errors import ConfigValidationError
from .schema import CONFIG_VERSION, DMIConfig, ModelDescriptor

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One validation finding, addressed to a specific control."""

    severity: str
    field: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == SEVERITY_ERROR

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
        }


def _validate_version(config: DMIConfig) -> Iterable[Issue]:
    if config.version != CONFIG_VERSION:
        yield Issue(
            SEVERITY_ERROR,
            "version",
            f"Unsupported configuration version {config.version!r} "
            f"(this build reads version {CONFIG_VERSION}).",
        )


def _validate_hooks(
    config: DMIConfig, descriptor: Optional[ModelDescriptor]
) -> Iterable[Issue]:
    topology = descriptor.topology if descriptor is not None else None
    catalog = {info.id: info for info in describe_hooks(topology)}

    selected = config.observations.hooks
    if not selected:
        yield Issue(
            SEVERITY_ERROR,
            "observations.hooks",
            "Select at least one observation.",
        )

    seen: set[str] = set()
    for hook_id in selected:
        if hook_id in seen:
            yield Issue(
                SEVERITY_WARNING,
                f"observations.hooks.{hook_id}",
                f"{hook_id!r} is selected more than once.",
            )
            continue
        seen.add(hook_id)

        info = catalog.get(hook_id)
        if info is None:
            yield Issue(
                SEVERITY_ERROR,
                f"observations.hooks.{hook_id}",
                f"Unknown observation {hook_id!r}. Available: "
                f"{', '.join(sorted(catalog))}.",
            )
        elif not info.available:
            yield Issue(
                SEVERITY_ERROR,
                f"observations.hooks.{hook_id}",
                f"{info.label} is unavailable for this model: {info.reason}.",
            )


def _validate_layers(
    config: DMIConfig, descriptor: Optional[ModelDescriptor]
) -> Iterable[Issue]:
    layers = config.observations.layers
    if layers is None:
        return

    if descriptor is not None:
        last = descriptor.last_layer
        if layers.start > last:
            yield Issue(
                SEVERITY_ERROR,
                "observations.layers",
                f"Layer range starts at {layers.start}, but the model has "
                f"{descriptor.topology.num_layers} layers (0-{last}).",
            )
        elif layers.end > last:
            yield Issue(
                SEVERITY_ERROR,
                "observations.layers",
                f"Layer range ends at {layers.end}, but the model's last "
                f"layer is {last}.",
            )

    per_layer = per_layer_hook_ids()
    if not any(hook in per_layer for hook in config.observations.hooks):
        yield Issue(
            SEVERITY_WARNING,
            "observations.layers",
            "A layer range is set but no per-layer observation is selected, "
            "so the range has no effect.",
        )


def _validate_schedule(config: DMIConfig) -> Iterable[Issue]:
    schedule = config.schedule
    if not (schedule.capture_prefill or schedule.capture_decode):
        yield Issue(
            SEVERITY_ERROR,
            "schedule.phase",
            "Enable at least one of prefill or decode, or nothing is captured.",
        )
    # CaptureSchedule.__post_init__ already rejects out-of-range values on
    # construction; these guard configurations built by mutating fields after
    # the fact.
    for name in ("step_stride", "request_stride"):
        if getattr(schedule, name) < 1:
            yield Issue(
                SEVERITY_ERROR,
                f"schedule.{name}",
                f"{name} must be >= 1.",
            )
    for name in ("step_offset", "request_offset", "warmup_steps", "warmup_requests"):
        if getattr(schedule, name) < 0:
            yield Issue(SEVERITY_ERROR, f"schedule.{name}", f"{name} must be >= 0.")


def validate_config(
    config: DMIConfig, descriptor: Optional[ModelDescriptor] = None
) -> list[Issue]:
    """Return every issue with ``config``, errors and warnings alike.

    ``descriptor=None`` checks only what is model-independent: hook names,
    schedule sanity, version. Layer bounds and per-model availability need a
    descriptor.
    """
    issues: list[Issue] = []
    issues.extend(_validate_version(config))
    issues.extend(_validate_hooks(config, descriptor))
    issues.extend(_validate_layers(config, descriptor))
    issues.extend(_validate_schedule(config))
    return issues


def is_valid(config: DMIConfig, descriptor: Optional[ModelDescriptor] = None) -> bool:
    """True when nothing blocks this configuration from running."""
    return not any(issue.is_error for issue in validate_config(config, descriptor))


def ensure_valid(
    config: DMIConfig, descriptor: Optional[ModelDescriptor] = None
) -> None:
    """Raise ``ConfigValidationError`` if any error-severity issue is present."""
    errors = [issue for issue in validate_config(config, descriptor) if issue.is_error]
    if errors:
        raise ConfigValidationError(errors)


__all__ = [
    "SEVERITY_ERROR",
    "SEVERITY_WARNING",
    "Issue",
    "validate_config",
    "is_valid",
    "ensure_valid",
]
