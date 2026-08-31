"""Serialize and parse DMI user configurations.

Serialization only -- no validation, no runtime. ``load_config`` returns a
``DMIConfig`` whether or not it is legal for any particular model; call
:func:`dmi.configuration.validation.validate_config` for that.

Note on the module name: this file is ``dmi/configuration/yaml.py`` and it
imports the third-party ``yaml``. Python 3's absolute imports make that
unambiguous, but ``from dmi.configuration import yaml`` gets this module, not
PyYAML.

Canonical form
--------------
Dumping normalizes first: hooks are deduplicated and ordered by the catalog,
so two configurations that differ only in the order the user clicked
checkboxes produce byte-identical YAML. The round-trip contract is therefore::

    parse(dump(config)) == normalize(config)

with ``normalize`` idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import CaptureSchedule
from .catalog_adapter import hook_ids
from .errors import ConfigurationError, UnsupportedConfigVersion
from .schema import (
    CONFIG_VERSION,
    DMIConfig,
    LayerSelection,
    ObservationConfig,
    RuntimePolicy,
)

# Always written, so a generated file shows the controls the UI exposes even
# when they hold default values.
_CORE_SCHEDULE_FIELDS = (
    "step_stride",
    "request_stride",
    "capture_prefill",
    "capture_decode",
)

# Written only when set to something other than the default, keeping the
# common file short. These are the "Advanced scheduling" controls.
_ADVANCED_SCHEDULE_FIELDS = (
    "step_offset",
    "warmup_steps",
    "request_offset",
    "warmup_requests",
)


def normalize_config(config: DMIConfig) -> DMIConfig:
    """Return ``config`` in canonical form.

    Deduplicates hooks and sorts them into catalog order. Unknown hook names
    are preserved, sorted, and placed after the known ones -- normalization is
    not validation's job, and silently dropping them would hide the error.
    """
    order = {hook: index for index, hook in enumerate(hook_ids())}
    unique = dict.fromkeys(config.observations.hooks)
    known = sorted((h for h in unique if h in order), key=order.__getitem__)
    unknown = sorted(h for h in unique if h not in order)

    return DMIConfig(
        observations=ObservationConfig(
            hooks=known + unknown,
            layers=config.observations.layers,
        ),
        schedule=config.schedule,
        policy=config.policy,
        version=config.version,
    )


def config_to_dict(config: DMIConfig) -> dict:
    """Canonical document form of a configuration."""
    canonical = normalize_config(config)

    observations: dict[str, Any] = {}
    if canonical.observations.layers is not None:
        observations["layers"] = {
            "start": canonical.observations.layers.start,
            "end": canonical.observations.layers.end,
        }
    observations["hooks"] = list(canonical.observations.hooks)

    defaults = CaptureSchedule()
    schedule: dict[str, Any] = {
        name: getattr(canonical.schedule, name) for name in _CORE_SCHEDULE_FIELDS
    }
    for name in _ADVANCED_SCHEDULE_FIELDS:
        value = getattr(canonical.schedule, name)
        if value != getattr(defaults, name):
            schedule[name] = value

    document: dict[str, Any] = {
        "version": canonical.version,
        "observations": observations,
        "schedule": schedule,
    }
    if canonical.policy is not None:
        document["policy"] = {"objective": canonical.policy.value}
    return document


def parse_config(data: Any) -> DMIConfig:
    """Build a ``DMIConfig`` from an already-parsed document."""
    if data is None:
        raise ConfigurationError("Configuration document is empty.")
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration must be a mapping, got {type(data).__name__}."
        )

    version = data.get("version", CONFIG_VERSION)
    # Dispatch on version so files outlive the code that wrote them. When a
    # v2 arrives, branch here rather than teaching one parser both shapes.
    if version != CONFIG_VERSION:
        raise UnsupportedConfigVersion(
            f"Configuration version {version!r} is not supported by this build "
            f"(expected {CONFIG_VERSION})."
        )
    return _parse_v1(data, version)


def _parse_v1(data: dict, version: int) -> DMIConfig:
    observations_raw = data.get("observations") or {}
    if not isinstance(observations_raw, dict):
        raise ConfigurationError("'observations' must be a mapping.")

    hooks_raw = observations_raw.get("hooks") or []
    if isinstance(hooks_raw, str):
        raise ConfigurationError(
            "'observations.hooks' must be a list, not a comma-separated "
            "string. Use dmi.configuration.compatibility."
            "from_legacy_hook_selection() to convert one."
        )
    if not isinstance(hooks_raw, list):
        raise ConfigurationError("'observations.hooks' must be a list.")
    hooks = [str(hook) for hook in hooks_raw]

    layers = None
    layers_raw = observations_raw.get("layers")
    if layers_raw is not None:
        if not isinstance(layers_raw, dict):
            raise ConfigurationError("'observations.layers' must be a mapping.")
        missing = [key for key in ("start", "end") if layers_raw.get(key) is None]
        if missing:
            raise ConfigurationError(
                f"'observations.layers' is missing: {', '.join(missing)}."
            )
        try:
            layers = LayerSelection(
                start=int(layers_raw["start"]), end=int(layers_raw["end"])
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid layer range: {exc}") from exc

    schedule_raw = data.get("schedule") or {}
    if not isinstance(schedule_raw, dict):
        raise ConfigurationError("'schedule' must be a mapping.")
    known_schedule = set(CaptureSchedule.__dataclass_fields__)
    unknown = sorted(set(schedule_raw) - known_schedule)
    if unknown:
        raise ConfigurationError(
            f"Unknown field(s) in 'schedule': {', '.join(unknown)}. "
            f"Known fields: {', '.join(sorted(known_schedule))}."
        )
    try:
        schedule = CaptureSchedule(**schedule_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid schedule: {exc}") from exc

    policy = None
    policy_raw = data.get("policy")
    if policy_raw is not None:
        if not isinstance(policy_raw, dict):
            raise ConfigurationError("'policy' must be a mapping.")
        objective = policy_raw.get("objective")
        if objective is not None:
            try:
                policy = RuntimePolicy(objective)
            except ValueError as exc:
                supported = ", ".join(item.value for item in RuntimePolicy)
                raise ConfigurationError(
                    f"Unknown policy objective {objective!r}. "
                    f"Supported: {supported}."
                ) from exc

    return DMIConfig(
        observations=ObservationConfig(hooks=hooks, layers=layers),
        schedule=schedule,
        policy=policy,
        version=int(version),
    )


def dump_config(config: DMIConfig) -> str:
    """Serialize a configuration to canonical YAML text."""
    return yaml.safe_dump(
        config_to_dict(config), sort_keys=False, default_flow_style=False
    )


def load_config(path: str | Path) -> DMIConfig:
    """Read a configuration from disk."""
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration {target}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Configuration {target} is not valid YAML: {exc}"
        ) from exc
    return parse_config(data)


def save_config(config: DMIConfig, path: str | Path) -> None:
    """Write a configuration to disk in canonical form."""
    Path(path).write_text(dump_config(config), encoding="utf-8")


__all__ = [
    "normalize_config",
    "config_to_dict",
    "parse_config",
    "dump_config",
    "load_config",
    "save_config",
]
