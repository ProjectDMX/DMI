"""Canonical typed representation of a DMI user configuration.

This module is the contract every other configuration layer is written
against. It holds no YAML knowledge and no runtime knowledge: parsing lives in
:mod:`dmi.configuration.yaml`, and the bridge into live ``HookSpec`` objects
lives in :mod:`dmi.configuration.compiler`.

Two documents are modelled here:

``ModelDescriptor``
    What a model *is* -- identity plus topology. Authored per model, usually
    from a Hugging Face config, and reused across many captures.

``DMIConfig``
    What the user wants DMI *to do* -- observations, schedule, policy. This is
    the artifact ``DMI-configurator`` produces.

They stay separate files on disk (``qwen3-8b.model.yaml`` and
``attention-debug.dmi.yaml``) because the descriptor outlives any single
capture, and the runtime already receives model information from the
framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config import CaptureSchedule

# Bump only on a breaking change to the on-disk shape. The loader dispatches on
# these, so configuration files outlive the code that generated them.
CONFIG_VERSION = 1
DESCRIPTOR_SCHEMA_VERSION = 1

# Architecture types the configurator knows how to render. Encoder-decoder,
# vision, and explicit MoE topologies are future work; a MoE decoder is already
# expressible through ``num_experts``/``top_k``.
SUPPORTED_ARCHITECTURES = ("decoder_transformer",)


class RuntimePolicy(Enum):
    """What DMI should protect when observation and inference contend.

    Carried through the configuration model and serialized, but *not* yet
    connected to runtime behaviour -- see the policy phasing in
    ``docs/dmi-configurator-plan.md``. Callers must not present this as
    changing capture behaviour until phase C lands.
    """

    COMPLETENESS = "completeness"
    BALANCED = "balanced"
    PERFORMANCE = "performance"


@dataclass(frozen=True)
class LayerSelection:
    """An inclusive range of layer indices.

    ``LayerSelection(8, 15)`` selects eight layers, 8 through 15. Inclusive
    because that is how the range reads in the UI ("Layers 8-15") and how users
    describe it; the exclusive alternative would make the label lie.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"layer start must be >= 0, got {self.start}.")
        if self.end < self.start:
            raise ValueError(
                f"layer end must be >= start, got start={self.start} "
                f"end={self.end}."
            )

    def contains(self, layer_no: int) -> bool:
        """True when ``layer_no`` falls inside the range.

        Global hooks carry ``layer_no == -1`` and are never contained; the
        layer filter treats them separately rather than dropping them.
        """
        return self.start <= layer_no <= self.end

    @property
    def count(self) -> int:
        return self.end - self.start + 1


@dataclass
class ObservationConfig:
    """Which hooks to capture, and over which layers.

    ``layers=None`` means every layer. The range constrains per-layer hooks
    only; global hooks such as ``final_logits`` are unaffected by it.
    """

    hooks: list[str] = field(default_factory=list)
    layers: Optional[LayerSelection] = None


@dataclass
class ModelIdentity:
    """Who the model is.

    ``id`` is used as a filename stem -- the configurator writes
    ``<id>.dmi.yaml`` beside the model. It must therefore be a single safe
    path segment. Descriptors derived from a framework config always are
    (``introspect._slug`` reduces ``Qwen/Qwen3-8B`` to ``qwen3-8b``), but a
    hand-written descriptor is not checked by anything else, and a separator
    there would move the write outside the directory the user named on the
    command line -- the one guarantee ``dmi.ui.app`` makes about where it
    writes.
    """

    id: str
    name: str
    architecture: str

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("model id must not be empty.")
        if self.id in (".", "..") or any(sep in self.id for sep in ("/", "\\")):
            raise ValueError(
                f"model id must be a single path segment, got {self.id!r}. "
                f"It names the configuration file written beside the model."
            )


@dataclass
class ModelTopology:
    """Model geometry, in Hugging Face field naming.

    HF naming rather than DMI's internal ``ModelShapeConfig`` naming
    (``hidden_dim``, ``intermediate_dim``) because descriptors are authored
    from HF configs. The translation is isolated in
    :mod:`dmi.configuration.manifest`.

    ``num_layers`` has no counterpart in ``ModelShapeConfig`` at all -- DMI core
    never needs a layer count, since it consumes a spec list the adapter builds
    by walking the model. The configurator needs it to render the architecture
    and to bounds-check layer ranges.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int = 0
    num_experts: int = 0
    top_k: int = 0
    head_dim: Optional[int] = None
    vocab_size: int = 0

    def __post_init__(self) -> None:
        # Exact integer types first: YAML floats and bools pass every
        # magnitude comparison here and then explode downstream -- a
        # num_layers of 1.5 raises TypeError inside range(), a True reads as
        # layer count 1.
        integral = (
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_kv_heads",
            "intermediate_size",
            "num_experts",
            "top_k",
            "vocab_size",
        )
        for name in integral:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__} "
                    f"({value!r})."
                )
        if self.head_dim is not None:
            if isinstance(self.head_dim, bool) or not isinstance(self.head_dim, int):
                raise TypeError(
                    f"head_dim must be an integer, got "
                    f"{type(self.head_dim).__name__} ({self.head_dim!r})."
                )
            if self.head_dim < 1:
                raise ValueError(
                    f"head_dim must be >= 1, got {self.head_dim}."
                )
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}.")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}.")
        if self.num_attention_heads < 1:
            raise ValueError(
                f"num_attention_heads must be >= 1, got {self.num_attention_heads}."
            )
        if self.num_kv_heads < 1:
            raise ValueError(
                f"num_kv_heads must be >= 1, got {self.num_kv_heads}."
            )
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        for name in ("intermediate_size", "num_experts", "top_k", "vocab_size"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}.")
        if self.top_k and not self.num_experts:
            raise ValueError("top_k is set but num_experts is 0.")

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0


@dataclass
class ModelDescriptor:
    """A loaded, validated model descriptor."""

    model: ModelIdentity
    topology: ModelTopology
    schema_version: int = DESCRIPTOR_SCHEMA_VERSION

    @property
    def last_layer(self) -> int:
        return self.topology.num_layers - 1


@dataclass
class DMIConfig:
    """A complete DMI user configuration.

    No ``runtime`` block in v1. The existing runtime configuration surface is
    ``RingConfig`` -- a native struct of transport-level parameters (see
    ``docs/config.md``) -- and exposing a subset of it here would imply a
    support contract that does not exist yet.
    """

    observations: ObservationConfig = field(default_factory=ObservationConfig)
    schedule: CaptureSchedule = field(default_factory=CaptureSchedule)
    policy: Optional[RuntimePolicy] = None
    version: int = CONFIG_VERSION


__all__ = [
    "CONFIG_VERSION",
    "DESCRIPTOR_SCHEMA_VERSION",
    "SUPPORTED_ARCHITECTURES",
    "RuntimePolicy",
    "LayerSelection",
    "ObservationConfig",
    "ModelIdentity",
    "ModelTopology",
    "ModelDescriptor",
    "DMIConfig",
]
