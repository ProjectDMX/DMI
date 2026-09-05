"""Load and validate model descriptors.

A descriptor answers "what is this model?" and is the only place Hugging Face
field naming meets DMI's internal ``ModelShapeConfig`` naming. Keeping that
translation here means the rest of the configuration layer sees exactly one
vocabulary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from .errors import DescriptorError, UnsupportedConfigVersion
from .schema import (
    DESCRIPTOR_SCHEMA_VERSION,
    SUPPORTED_ARCHITECTURES,
    ModelDescriptor,
    ModelIdentity,
    ModelTopology,
)

# Descriptor field -> ModelShapeConfig field. The two documents disagree on
# naming by design: descriptors are authored from HF configs, DMI core is not.
_TOPOLOGY_TO_SHAPE = {
    "hidden_size": "hidden_dim",
    "num_attention_heads": "num_heads",
    "num_kv_heads": "num_kv_heads",
    "intermediate_size": "intermediate_dim",
    "num_experts": "num_experts",
    "top_k": "top_k",
    "vocab_size": "vocab_size",
}

_REQUIRED_TOPOLOGY = (
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_kv_heads",
)

_KNOWN_MODEL_FIELDS = ("id", "name", "architecture")
_KNOWN_DESCRIPTOR_ROOT = ("schema_version", "model", "topology")


def _reject_unknown(present, known, where: str) -> None:
    """Refuse keys the descriptor grammar does not define, per section.

    The same rule the config parser applies everywhere: a silently ignored
    key is the worst outcome for an authored file. Non-string keys are
    reported as their repr rather than crashing the join.
    """
    unknown = sorted(repr(key) for key in set(present) - set(known))
    if unknown:
        raise DescriptorError(
            f"Unknown field(s) in {where}: {', '.join(unknown)}. "
            f"Known fields: {', '.join(sorted(map(repr, known)))}."
        )


def _require_mapping(value: Any, where: str) -> dict:
    if value is None:
        raise DescriptorError(f"Descriptor is missing the {where!r} section.")
    if not isinstance(value, dict):
        raise DescriptorError(
            f"Descriptor section {where!r} must be a mapping, got "
            f"{type(value).__name__}."
        )
    return value


def parse_descriptor(data: Any) -> ModelDescriptor:
    """Build a ``ModelDescriptor`` from an already-parsed document."""
    document = _require_mapping(data, "descriptor")

    _reject_unknown(document, _KNOWN_DESCRIPTOR_ROOT, "the descriptor")

    version = document.get("schema_version", DESCRIPTOR_SCHEMA_VERSION)
    # Exact integer only, mirroring the config boundary: YAML `true` compares
    # equal to 1 and `1.0` equals 1, so a bare `!=` would accept both and a
    # future version file with a mistyped key would silently parse as v1.
    if isinstance(version, bool) or not isinstance(version, int):
        raise DescriptorError(
            f"Descriptor 'schema_version' must be an integer, got "
            f"{type(version).__name__} ({version!r})."
        )
    if version != DESCRIPTOR_SCHEMA_VERSION:
        raise UnsupportedConfigVersion(
            f"Descriptor schema_version {version!r} is not supported by this "
            f"build (expected {DESCRIPTOR_SCHEMA_VERSION})."
        )

    model = _require_mapping(document.get("model"), "model")
    _reject_unknown(model, _KNOWN_MODEL_FIELDS, "'model'")
    for key in ("id", "name", "architecture"):
        if not model.get(key):
            raise DescriptorError(f"Descriptor field 'model.{key}' is required.")

    architecture = model["architecture"]
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise DescriptorError(
            f"Unsupported architecture {architecture!r}. Supported: "
            f"{', '.join(SUPPORTED_ARCHITECTURES)}."
        )

    topology = _require_mapping(document.get("topology"), "topology")
    missing = [key for key in _REQUIRED_TOPOLOGY if topology.get(key) is None]
    if missing:
        raise DescriptorError(
            f"Descriptor 'topology' is missing required field(s): "
            f"{', '.join(missing)}."
        )

    _reject_unknown(topology, set(ModelTopology.__dataclass_fields__), "topology")

    try:
        parsed_topology = ModelTopology(**topology)
    except (TypeError, ValueError) as exc:
        raise DescriptorError(f"Invalid topology: {exc}") from exc

    try:
        identity = ModelIdentity(
            id=str(model["id"]),
            name=str(model["name"]),
            architecture=str(architecture),
        )
    except ValueError as exc:
        # The config boundary raises ConfigurationError for exactly these;
        # the descriptor boundary must not leak a bare ValueError past the
        # documented DescriptorError hierarchy.
        raise DescriptorError(f"Invalid model identity: {exc}") from exc

    return ModelDescriptor(
        model=identity,
        topology=parsed_topology,
        schema_version=version,
    )


def load_descriptor(path: str | Path) -> ModelDescriptor:
    """Read and validate a descriptor from disk."""
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise DescriptorError(f"Cannot read descriptor {target}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise DescriptorError(f"Descriptor {target} is not valid YAML: {exc}") from exc
    return parse_descriptor(data)


def descriptor_to_dict(descriptor: ModelDescriptor) -> dict:
    """Serialize a descriptor back to its document form, omitting defaults."""
    topology: dict[str, Any] = {
        "num_layers": descriptor.topology.num_layers,
        "hidden_size": descriptor.topology.hidden_size,
        "num_attention_heads": descriptor.topology.num_attention_heads,
        "num_kv_heads": descriptor.topology.num_kv_heads,
    }
    for key in ("intermediate_size", "num_experts", "top_k", "vocab_size"):
        value = getattr(descriptor.topology, key)
        if value:
            topology[key] = value
    if descriptor.topology.head_dim is not None:
        topology["head_dim"] = descriptor.topology.head_dim

    return {
        "schema_version": descriptor.schema_version,
        "model": {
            "id": descriptor.model.id,
            "name": descriptor.model.name,
            "architecture": descriptor.model.architecture,
        },
        "topology": topology,
    }


def save_descriptor(descriptor: ModelDescriptor, path: str | Path) -> None:
    """Write a descriptor to disk, atomically, like ``save_config``.

    Same contract as the config half of the package: filesystem trouble is
    a ``DescriptorError``, and the write never truncates the previous file
    on failure.
    """
    target = Path(path)
    temp = target.with_name(target.name + ".tmp")
    try:
        temp.write_text(
            yaml.safe_dump(descriptor_to_dict(descriptor), sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temp, target)
    except OSError as exc:
        temp.unlink(missing_ok=True)
        raise DescriptorError(f"Cannot write descriptor {target}: {exc}") from exc


def to_model_shape_config(
    topology: ModelTopology,
    dtype: Optional[Any] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
):
    """Translate descriptor topology into DMI's ``ModelShapeConfig``.

    Imported lazily: ``ModelShapeConfig`` pulls in torch, and the descriptor
    and validation paths deliberately do not need it.
    """
    from ..hooks.specs import ModelShapeConfig  # local: torch import

    if dtype is None:
        import torch

        dtype = torch.float16

    mapped = {
        shape_field: getattr(topology, descriptor_field)
        for descriptor_field, shape_field in _TOPOLOGY_TO_SHAPE.items()
    }
    return ModelShapeConfig(
        head_dim=topology.effective_head_dim,
        dtype=dtype,
        tp_size=tp_size,
        tp_rank=tp_rank,
        **mapped,
    )


__all__ = [
    "parse_descriptor",
    "load_descriptor",
    "descriptor_to_dict",
    "save_descriptor",
    "to_model_shape_config",
]
