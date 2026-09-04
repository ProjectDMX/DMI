"""Derive model descriptors from framework configuration.

Descriptors should not be typed by hand. The framework already knows the
model, and DMI already knows how to read a Hugging-Face-shaped config:
:func:`dmi.api.v1.model_shape.make_model_shape_from_hf_config` extracts every
topology field except the layer count. This module adds the layer count and
the identity fields, and reuses that extractor rather than duplicating it.

Reading a ``config.json`` needs no ``transformers`` install: the extractor uses
``getattr``, so a parsed JSON object wrapped in a namespace works directly.
``transformers`` is imported lazily, and only when resolving a bare model id.

A descriptor file still exists on purpose. The configurator runs where the
model is not loaded -- pick layers on a laptop, run the capture on a cluster --
so the descriptor is the portable, design-time record. The runtime never reads
it: ``compile_config`` takes its shape from the adapter's live
``detect_model_shape(model)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from .errors import DescriptorError
from .manifest import load_descriptor
from .schema import (
    DESCRIPTOR_SCHEMA_VERSION,
    ModelDescriptor,
    ModelIdentity,
    ModelTopology,
)

# HF spells the layer count differently across model families, the same way it
# spells hidden size as hidden_size or n_embd.
_LAYER_COUNT_FIELDS = ("num_hidden_layers", "n_layer", "num_layers")

_DESCRIPTOR_SUFFIXES = (".yaml", ".yml")


def _first_attr(config: Any, names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _slug(model_id: str) -> str:
    """``Qwen/Qwen3-8B`` -> ``qwen3-8b``."""
    tail = model_id.rstrip("/").split("/")[-1]
    return tail.lower().replace("_", "-").replace(" ", "-")


# Model families that are NOT causal decoder-only transformers, by the
# ``model_type`` the HF config carries. ``is_encoder_decoder == False`` does
# not imply a decoder: BERT is bidirectional, ViT has no causal LM head at
# all, and both were previously labeled decoder_transformer and rendered with
# decoder observations. Matching is prefix-based so the family variants
# (`bert`, `bert-generation`, `vit_mae`, ...) are covered without listing
# every one.
_KNOWN_NON_DECODER_MODEL_TYPES = (
    "bert",
    "albert",
    "distilbert",
    "roberta",
    "electra",
    "vit",
    "deit",
    "beit",
    "clip",
    "siglip",
    "swin",
    "convnext",
    "resnet",
    "timm",
    "whisper",
    "wav2vec2",
    "hubert",
    "speecht5",
    "t5",
    "longt5",
)


def _reject_known_non_decoder(hf_config: Any, model_id: str) -> None:
    """Refuse config shapes DMI-configurator v1 cannot observe correctly.

    Only ``decoder_transformer`` is supported. A model_type known to name
    something else -- masked LMs, encoders, vision backbones, audio models --
    is refused even though its attention geometry would parse, because a
    descriptor built from it would promise observations the model cannot
    produce in the shapes DMI expects.
    """
    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str):
        normalized = model_type.lower()
        for family in _KNOWN_NON_DECODER_MODEL_TYPES:
            if normalized == family or normalized.startswith(family + "_"):
                raise DescriptorError(
                    f"{model_id!r} is a {model_type!r} model, which is not a "
                    "causal decoder-only transformer. DMI-configurator v1 "
                    "supports decoder_transformer only."
                )
    architectures = getattr(hf_config, "architectures", None) or []
    for architecture in architectures:
        # HF names carry no underscores ("BertForMaskedLM"), so lower and
        # match the unbroken suffixes.
        name = str(architecture).lower()
        if name.endswith(("formaskedlm", "forconditionalgeneration")):
            raise DescriptorError(
                f"{model_id!r} declares {architecture!r}, which is not a "
                "causal decoder-only transformer. DMI-configurator v1 "
                "supports decoder_transformer only."
            )


def descriptor_from_hf_config(
    hf_config: Any,
    model_id: str,
    name: Optional[str] = None,
) -> ModelDescriptor:
    """Build a descriptor from a Hugging-Face-shaped config object.

    The object need not be a Transformers class; only the standard attributes
    are read, so a ``SimpleNamespace`` over a parsed ``config.json`` works.
    """
    # Lazy: the extractor's module imports torch, and this package promises
    # torch-free descriptor and validation paths (estimate.py and manifest.py
    # defer their torch imports for the same reason).
    from ..api.v1.model_shape import make_model_shape_from_hf_config

    if getattr(hf_config, "is_encoder_decoder", False):
        raise DescriptorError(
            f"{model_id!r} is an encoder-decoder model. DMI-configurator v1 "
            f"supports decoder_transformer only."
        )
    _reject_known_non_decoder(hf_config, model_id)

    shape = make_model_shape_from_hf_config(hf_config)
    if shape is None:
        raise DescriptorError(
            f"Could not read attention geometry from the config for "
            f"{model_id!r}: hidden_size and num_attention_heads are required."
        )

    num_layers = _first_attr(hf_config, _LAYER_COUNT_FIELDS)
    if num_layers is None:
        raise DescriptorError(
            f"Config for {model_id!r} has no layer count "
            f"({' / '.join(_LAYER_COUNT_FIELDS)})."
        )

    # Emit head_dim only when it is not the obvious hidden/heads quotient, so
    # generated descriptors stay short and the interesting cases stand out.
    implied_head_dim = int(shape.hidden_dim) // int(shape.num_heads)
    head_dim = shape.head_dim if shape.head_dim != implied_head_dim else None

    try:
        topology = ModelTopology(
            num_layers=int(num_layers),
            hidden_size=int(shape.hidden_dim),
            num_attention_heads=int(shape.num_heads),
            num_kv_heads=int(shape.num_kv_heads),
            intermediate_size=int(shape.intermediate_dim),
            num_experts=int(shape.num_experts),
            top_k=int(shape.top_k),
            head_dim=head_dim,
            vocab_size=int(shape.vocab_size),
        )
    except ValueError as exc:
        raise DescriptorError(f"Config for {model_id!r} is inconsistent: {exc}") from exc

    try:
        identity = ModelIdentity(
            id=_slug(model_id),
            name=name or model_id.rstrip("/").split("/")[-1],
            architecture="decoder_transformer",
        )
    except ValueError as exc:
        # _slug of a degenerate id ("/" and friends) leaves nothing usable.
        raise DescriptorError(
            f"Could not derive a usable model id from {model_id!r}: {exc}"
        ) from exc

    return ModelDescriptor(
        model=identity,
        topology=topology,
        schema_version=DESCRIPTOR_SCHEMA_VERSION,
    )


def load_hf_config_document(path: str | Path) -> Any:
    """Read a ``config.json`` into an attribute-accessible object."""
    target = Path(path)
    if target.is_dir():
        target = target / "config.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DescriptorError(f"Cannot read model config {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DescriptorError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DescriptorError(f"{target} must contain a JSON object.")
    return SimpleNamespace(**data)


def _load_hf_config_by_id(model_id: str) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise DescriptorError(
            f"Resolving the model id {model_id!r} needs `transformers` "
            f"installed. Either install it, or point at the model's "
            f"config.json directly."
        ) from exc
    try:
        return AutoConfig.from_pretrained(model_id)
    except Exception as exc:  # network, auth, unknown model -- all opaque here
        raise DescriptorError(
            f"Could not load a config for {model_id!r}: {exc}"
        ) from exc


def _identity_source(target: Path) -> str:
    """Name a model after its directory, never after ``config.json``.

    ``~/models/Qwen3-8B/config.json`` and ``~/models/Qwen3-8B`` should both
    yield ``qwen3-8b``, not ``config.json``.
    """
    if target.is_dir():
        candidate = target.resolve().name
    elif target.suffix == ".json" and target.stem == "config":
        candidate = target.resolve().parent.name
    else:
        candidate = target.stem
    return candidate or target.resolve().name


def describe_model(source: str | Path, name: Optional[str] = None) -> ModelDescriptor:
    """Build a descriptor from a model directory, ``config.json``, or model id."""
    target = Path(source)
    if target.is_dir() or target.suffix == ".json":
        return descriptor_from_hf_config(
            load_hf_config_document(target), _identity_source(target), name
        )
    if target.exists():
        raise DescriptorError(
            f"{target} is not a model config. Pass a config.json, a model "
            f"directory, or a Hugging Face model id."
        )
    return descriptor_from_hf_config(_load_hf_config_by_id(str(source)), str(source), name)


def resolve_descriptor(source: str | Path) -> ModelDescriptor:
    """Load a descriptor from any supported source.

    Accepts a DMI descriptor YAML, a model directory, a ``config.json``, or a
    Hugging Face model id. Framework configs are the everyday path; a
    descriptor file is the override for models the extractor cannot read.
    """
    target = Path(source)
    if target.suffix in _DESCRIPTOR_SUFFIXES:
        # A YAML suffix is a file path, never a Hugging Face model id. Say the
        # file is missing instead of falling through to id resolution, which
        # would blame a missing `transformers` install for a shell typo.
        if not target.exists():
            raise DescriptorError(f"Descriptor {target} does not exist.")
        return load_descriptor(target)
    return describe_model(source)


__all__ = [
    "descriptor_from_hf_config",
    "load_hf_config_document",
    "describe_model",
    "resolve_descriptor",
]
