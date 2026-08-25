"""Stable public DMI integration API, version 1.

This module is an additive facade over DMI core. It deliberately does not
import a framework integration, register framework presets, or create runtime
resources.
"""

from __future__ import annotations

from typing import Any

import torch

from ...adapters.base import (
    BackendAdapter,
    BackendAdaptor,
    StepPlan,
    StepReservation,
)
from ...storage.clickhouse import CHClickhouseDriverReadOnly
from ...config import CaptureSchedule, MonitoringConfig
from ...engine import (
    HostEngineConfig,
    MonitoringEngine,
    RingCapacities,
    SinkStats,
)
from ...hooks.dispatch import install_ring_hooks
from ...hooks.point import HookPoint
from ...hooks import specs as _specs
from ...hooks.specs import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_ATTN_SCORES,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_PATTERN,
    HOOK_TYPE_POS_EMBED,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookRowBasis,
    HookSpec,
    ModelShapeConfig,
    hook_row_basis,
)
from ...storage.internals import (
    IncompleteInternalError,
    InternalRequirement,
    InternalRequirements,
    LazyInternal,
    make_lazy_internal as _make_lazy_internal,
)
from ...hooks import selection as _selection
from ...hooks.selection import (
    hook_belongs_to_pp_rank,
    hook_belongs_to_tp_rank,
    register_preset,
    select_hook_specs,
)
from ...adapters.types import StepContext

from .model_shape import make_model_shape_from_hf_config


DMI_INTEGRATION_API_VERSION = 1


# Public names for behavior that predates the versioned facade.
compute_hook_shape = _specs.compute_hook_shape
align_up = _specs.align_up_py
ALL_HOOK_TYPES = _selection._ALL_HOOK_TYPES
ATTENTION_WEIGHT_HOOK_TYPES = _specs._ATTN_WT_TYPES


def deactivate_ring_transport() -> None:
    """Clear the process-global active ring transport."""

    from ...transport import ring as _ring_transport

    _ring_transport.deactivate()


def is_preset_registered(name: str) -> bool:
    """Return whether a hook-selection preset or individual hook exists."""

    return name in _selection._HOOK_SELECTIONS


def configure_hook_padding_strip(
    hook_point: HookPoint,
    row_count_tensor: torch.Tensor | None,
    row_bytes: int = 0,
) -> None:
    """Configure an existing HookPoint's producer padding-strip mode.

    ``row_count_tensor=None`` selects the static producer. A tensor with a
    positive ``row_bytes`` selects prefix stripping; a tensor with a
    non-positive ``row_bytes`` selects chunked stripping.
    """

    hook_point._strip_tensor = row_count_tensor
    hook_point._strip_row_bytes = row_bytes


def make_lazy_internal(
    model_id: str,
    reader: CHClickhouseDriverReadOnly | None = None,
    requirements: InternalRequirements | None = None,
    request_ids: tuple[str, ...] | list[str] | None = None,
    token_ranges: (
        dict[
            str,
            tuple[tuple[int, int], ...] | list[tuple[int, int]],
        ]
        | None
    ) = None,
) -> LazyInternal:
    """Create a lazy captured-internals handle with public v1 annotations."""

    return _make_lazy_internal(
        model_id,
        reader=reader,
        requirements=requirements,
        request_ids=request_ids,
        token_ranges=token_ranges,
    )


_NATIVE_EXPORTS = frozenset(
    {
        "ClickHouseClientConfig",
        "DMXHostEngine",
        "EnqueuePolicy",
        "OnClosedPolicy",
        "OnFullPolicy",
        "QueueConfig",
        "RingConfig",
        "StageConfig",
        "ThreadFailure",
    }
)


def __getattr__(name: str) -> Any:
    if name in _NATIVE_EXPORTS:
        from ...transport import native as _native_engine

        value = getattr(_native_engine._load_extension(), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS)


__all__ = [
    "DMI_INTEGRATION_API_VERSION",
    "BackendAdapter",
    "BackendAdaptor",
    "StepPlan",
    "StepReservation",
    "StepContext",
    "MonitoringEngine",
    "RingCapacities",
    "SinkStats",
    "MonitoringConfig",
    "CaptureSchedule",
    "HostEngineConfig",
    "deactivate_ring_transport",
    "HookPoint",
    "HookSpec",
    "HookRowBasis",
    "ModelShapeConfig",
    "hook_row_basis",
    "compute_hook_shape",
    "align_up",
    "make_model_shape_from_hf_config",
    "install_ring_hooks",
    "configure_hook_padding_strip",
    "ALL_HOOK_TYPES",
    "ATTENTION_WEIGHT_HOOK_TYPES",
    "HOOK_TYPE_RESID_PRE",
    "HOOK_TYPE_LN1",
    "HOOK_TYPE_ATTN_OUT",
    "HOOK_TYPE_RESID_MID",
    "HOOK_TYPE_ATTN_SCORES",
    "HOOK_TYPE_PATTERN",
    "HOOK_TYPE_Q",
    "HOOK_TYPE_K",
    "HOOK_TYPE_V",
    "HOOK_TYPE_Z",
    "HOOK_TYPE_LN2",
    "HOOK_TYPE_MLP_IN",
    "HOOK_TYPE_MLP_OUT",
    "HOOK_TYPE_MLP_POST",
    "HOOK_TYPE_RESID_FINAL",
    "HOOK_TYPE_EMBED",
    "HOOK_TYPE_POS_EMBED",
    "HOOK_TYPE_FINAL_LN",
    "HOOK_TYPE_TOKEN_IDS",
    "HOOK_TYPE_FINAL_LOGITS",
    "HOOK_TYPE_ROUTER_LOGITS",
    "HOOK_TYPE_TOPK_IDS",
    "HOOK_TYPE_TOPK_WEIGHTS",
    "register_preset",
    "is_preset_registered",
    "select_hook_specs",
    "hook_belongs_to_pp_rank",
    "hook_belongs_to_tp_rank",
    "RingConfig",
    "ClickHouseClientConfig",
    "StageConfig",
    "QueueConfig",
    "EnqueuePolicy",
    "OnFullPolicy",
    "OnClosedPolicy",
    "DMXHostEngine",
    "ThreadFailure",
    "CHClickhouseDriverReadOnly",
    "InternalRequirement",
    "InternalRequirements",
    "IncompleteInternalError",
    "LazyInternal",
    "make_lazy_internal",
]
