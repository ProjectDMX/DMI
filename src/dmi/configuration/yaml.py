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

import os
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


_KNOWN_TOP_LEVEL = ("version", "observations", "schedule", "policy")
_KNOWN_OBSERVATION_FIELDS = ("hooks", "layers")
_KNOWN_LAYERS_FIELDS = ("start", "end")
_KNOWN_POLICY_FIELDS = ("objective",)

# CaptureSchedule fields by the exact type the boundary requires. Booleans
# and floats are not integers here: YAML `true` must not become layer 1 and
# `2.9` must not become layer 2.
_INT_SCHEDULE_FIELDS = (
    "step_stride",
    "step_offset",
    "warmup_steps",
    "request_stride",
    "request_offset",
    "warmup_requests",
)
_BOOL_SCHEDULE_FIELDS = ("capture_prefill", "capture_decode")


def _reject_unknown(present, known, where: str) -> None:
    """Refuse keys this build does not understand.

    Every section is strict, not just ``schedule``. A silently ignored key is
    the worst outcome for a configuration file: ``observations.layer`` (missing
    the plural) would parse as "no layer range" and capture every layer, which
    is a large, quiet payload increase from a file that still reads correctly.
    """
    unknown = sorted(set(present) - set(known))
    if unknown:
        raise ConfigurationError(
            f"Unknown field(s) in {where}: {', '.join(unknown)}. "
            f"Known fields: {', '.join(sorted(known))}."
        )


def _exact_int(value: Any, where: str) -> int:
    """An integer, and nothing int()-shaped: no floats, no bools, no strings."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{where} must be an integer, got {type(value).__name__} "
            f"({value!r})."
        )
    return value


def _exact_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(
            f"{where} must be a boolean, got {type(value).__name__} "
            f"({value!r})."
        )
    return value


def _typed_schedule(schedule_raw: dict) -> CaptureSchedule:
    """Validate every scalar at the boundary, then construct."""
    typed: dict[str, Any] = {}
    for name, value in schedule_raw.items():
        if name in _INT_SCHEDULE_FIELDS:
            typed[name] = _exact_int(value, f"schedule.{name}")
        elif name in _BOOL_SCHEDULE_FIELDS:
            typed[name] = _exact_bool(value, f"schedule.{name}")
    try:
        return CaptureSchedule(**typed)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid schedule: {exc}") from exc


def parse_config(data: Any) -> DMIConfig:
    """Build a ``DMIConfig`` from an already-parsed document."""
    if data is None:
        raise ConfigurationError("Configuration document is empty.")
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration must be a mapping, got {type(data).__name__}."
        )

    _reject_unknown(data, _KNOWN_TOP_LEVEL, "the configuration")

    # Required, not defaulted. Research configuration files outlive the code
    # that wrote them, so an unversioned document is refused rather than
    # assumed to match this build -- defaulting is exactly the guess that
    # version dispatch exists to avoid.
    if "version" not in data:
        raise ConfigurationError(
            f"Configuration is missing 'version'. Add 'version: "
            f"{CONFIG_VERSION}'."
        )
    version = data["version"]
    # Reject a non-integer version as malformed rather than unsupported: YAML
    # will happily hand back the string "1" from a quoted value, and
    # "version '1' is not supported" would send the reader looking for a
    # build that reads it instead of at the quotes.
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigurationError(
            f"Configuration 'version' must be an integer, got "
            f"{type(version).__name__} ({version!r})."
        )
    # Dispatch on version so files outlive the code that wrote them. When a
    # v2 arrives, branch here rather than teaching one parser both shapes.
    if version != CONFIG_VERSION:
        raise UnsupportedConfigVersion(
            f"Configuration version {version!r} is not supported by this build "
            f"(expected {CONFIG_VERSION})."
        )
    return _parse_v1(data, version)


def _section(data: dict, key: str, where: str) -> dict:
    """Read a required-to-be-mapping section, defaulting only true absence.

    An explicit ``is None`` check, not ``or {}``: a falsy-but-wrong value
    (``observations: []``, ``schedule: ""``) must be refused as malformed, not
    silently defaulted into a configuration that never says what it means.
    """
    section = data.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ConfigurationError(f"{where} must be a mapping.")
    return section


def _parse_v1(data: dict, version: int) -> DMIConfig:
    observations_raw = _section(data, "observations", "'observations'")
    _reject_unknown(observations_raw, _KNOWN_OBSERVATION_FIELDS, "'observations'")

    hooks_raw = observations_raw.get("hooks")
    if hooks_raw is None:
        hooks_raw = []
    if isinstance(hooks_raw, str):
        raise ConfigurationError(
            "'observations.hooks' must be a list, not a comma-separated "
            "string. Use dmi.configuration.compatibility."
            "from_legacy_hook_selection() to convert one."
        )
    if not isinstance(hooks_raw, list):
        raise ConfigurationError("'observations.hooks' must be a list.")
    # str() would invent hook names out of authoring mistakes: [q, 1, true,
    # null] became ["q", "1", "True", "None"], four "unknown hook" issues
    # deep in validation instead of one message at the boundary that says
    # which element is wrong. A typed list[str] rejects non-strings.
    for index, hook in enumerate(hooks_raw):
        if not isinstance(hook, str):
            raise ConfigurationError(
                f"'observations.hooks[{index}]' must be a string, got "
                f"{type(hook).__name__} ({hook!r})."
            )
    hooks = list(hooks_raw)

    layers = None
    layers_raw = observations_raw.get("layers")
    if layers_raw is not None:
        if not isinstance(layers_raw, dict):
            raise ConfigurationError("'observations.layers' must be a mapping.")
        _reject_unknown(layers_raw, _KNOWN_LAYERS_FIELDS, "'observations.layers'")
        missing = [key for key in ("start", "end") if layers_raw.get(key) is None]
        if missing:
            raise ConfigurationError(
                f"'observations.layers' is missing: {', '.join(missing)}."
            )
        try:
            layers = LayerSelection(
                start=_exact_int(layers_raw["start"], "observations.layers.start"),
                end=_exact_int(layers_raw["end"], "observations.layers.end"),
            )
        except ValueError as exc:
            raise ConfigurationError(f"Invalid layer range: {exc}") from exc
    schedule_raw = _section(data, "schedule", "'schedule'")
    _reject_unknown(
        schedule_raw, CaptureSchedule.__dataclass_fields__, "'schedule'"
    )
    schedule = _typed_schedule(schedule_raw)

    policy = None
    policy_raw = data.get("policy")
    if policy_raw is not None:
        if not isinstance(policy_raw, dict):
            raise ConfigurationError("'policy' must be a mapping.")
        _reject_unknown(policy_raw, _KNOWN_POLICY_FIELDS, "'policy'")
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


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys.

    PyYAML's default is last-wins and silent. In a capture configuration that
    is a scope change nobody sees: a hand edit or a merge conflict leaving two
    ``observations.hooks`` keys quietly discards one of them. A configuration
    file is a contract, so an ambiguous document is an error, not a merge.
    """


def _no_duplicate_keys(loader: yaml.SafeLoader, node, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found a duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


def load_yaml_document(text: str, where: str = "Configuration") -> Any:
    """Parse one YAML document the way DMI configurations are parsed.

    Shared by the file path and the HTTP boundary so both refuse the same
    documents: safe tags only, and no duplicate mapping keys.
    """
    if not isinstance(text, str):
        raise ConfigurationError(
            f"{where} must be a YAML string, got {type(text).__name__}."
        )
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{where} is not valid YAML: {exc}") from exc


def load_config(path: str | Path) -> DMIConfig:
    """Read a configuration from disk."""
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration {target}: {exc}") from exc
    return parse_config(load_yaml_document(raw, f"Configuration {target}"))


def save_config(config: DMIConfig, path: str | Path) -> None:
    """Write a configuration to disk in canonical form.

    Atomic: the payload is written to a sibling temp file and renamed, so a
    mid-write failure (ENOSPC, permission) leaves the PREVIOUS configuration
    on disk instead of a truncated file that reads as broken on the next
    launch. Wraps ``OSError`` the way :func:`load_config` does: both halves
    of this API report filesystem trouble as ``ConfigurationError``.
    """
    target = Path(path)
    temp = target.with_name(target.name + ".tmp")
    try:
        temp.write_text(dump_config(config), encoding="utf-8")
        os.replace(temp, target)
    except OSError as exc:
        temp.unlink(missing_ok=True)
        raise ConfigurationError(
            f"Cannot write configuration {target}: {exc}"
        ) from exc


__all__ = [
    "normalize_config",
    "load_yaml_document",
    "config_to_dict",
    "parse_config",
    "dump_config",
    "load_config",
    "save_config",
]
