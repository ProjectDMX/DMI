"""Stable DMI integration API, version 1.

This module is an additive facade over DMI core. It deliberately does not
import a framework integration, register framework presets, or create runtime
resources.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
from monitoring.config import CaptureSchedule, MonitoringConfig
from monitoring.engine import HostEngineConfig, MonitoringEngine, RingCapacities
from monitoring.hook_points import HookPoint
from monitoring.internal_mapper import (
    IncompleteInternalError,
    InternalRequirement,
    InternalRequirements,
    LazyInternal,
    make_lazy_internal as _make_lazy_internal,
)
from monitoring.model_shape import ModelShapeConfig
from monitoring.step_context import StepContext


DMI_INTEGRATION_API_VERSION = 1


def is_preset_registered(name: str) -> bool:
    """Return whether a hook-selection preset or individual hook exists."""

    from monitoring import selection

    return name in selection._HOOK_SELECTIONS


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


def make_model_shape_from_hf_config(
    hf_config: Any,
    dtype: torch.dtype | None = None,
) -> ModelShapeConfig | None:
    """Build a model-shape description from a Hugging-Face-shaped config."""

    from .model_shape import make_model_shape_from_hf_config as build

    return build(hf_config, dtype=dtype)


_HOST_NATIVE_EXPORTS = frozenset(
    {
        "ClickHouseClientConfig",
        "DMXHostEngine",
        "EnqueuePolicy",
        "OnClosedPolicy",
        "OnFullPolicy",
        "QueueConfig",
        "StageConfig",
        "ThreadFailure",
    }
)
_RING_NATIVE_EXPORTS = frozenset({"RingConfig"})
_RING_EXPORTS = {
    "ATTENTION_WEIGHT_HOOK_TYPES": "_ATTN_WT_TYPES",
    "HookRowBasis": "HookRowBasis",
    "HookSpec": "HookSpec",
    "align_up": "align_up_py",
    "compute_hook_shape": "_compute_hook_shape",
    "deactivate_ring_transport": "deactivate",
    "hook_row_basis": "hook_row_basis",
    "install_ring_hooks": "install_ring_hooks",
    **{
        name: name
        for name in (
            "HOOK_TYPE_ATTN_OUT",
            "HOOK_TYPE_ATTN_SCORES",
            "HOOK_TYPE_EMBED",
            "HOOK_TYPE_FINAL_LN",
            "HOOK_TYPE_FINAL_LOGITS",
            "HOOK_TYPE_K",
            "HOOK_TYPE_LN1",
            "HOOK_TYPE_LN2",
            "HOOK_TYPE_MLP_IN",
            "HOOK_TYPE_MLP_OUT",
            "HOOK_TYPE_MLP_POST",
            "HOOK_TYPE_PATTERN",
            "HOOK_TYPE_POS_EMBED",
            "HOOK_TYPE_Q",
            "HOOK_TYPE_RESID_FINAL",
            "HOOK_TYPE_RESID_MID",
            "HOOK_TYPE_RESID_PRE",
            "HOOK_TYPE_ROUTER_LOGITS",
            "HOOK_TYPE_TOKEN_IDS",
            "HOOK_TYPE_TOPK_IDS",
            "HOOK_TYPE_TOPK_WEIGHTS",
            "HOOK_TYPE_V",
            "HOOK_TYPE_Z",
        )
    },
}
_SELECTION_EXPORTS = {
    "ALL_HOOK_TYPES": "_ALL_HOOK_TYPES",
    "hook_belongs_to_pp_rank": "hook_belongs_to_pp_rank",
    "hook_belongs_to_tp_rank": "hook_belongs_to_tp_rank",
    "register_preset": "register_preset",
    "select_hook_specs": "select_hook_specs",
}
_ADAPTOR_EXPORTS = frozenset({"BackendAdaptor", "StepPlan", "StepReservation"})
_LAZY_EXPORTS = (
    _HOST_NATIVE_EXPORTS
    | _RING_NATIVE_EXPORTS
    | _RING_EXPORTS.keys()
    | _SELECTION_EXPORTS.keys()
    | _ADAPTOR_EXPORTS
)


def __getattr__(name: str) -> Any:
    if name in _HOST_NATIVE_EXPORTS:
        from monitoring import _native_engine

        value = getattr(_native_engine, name)
    elif name in _RING_NATIVE_EXPORTS:
        from monitoring import _native_engine

        value = getattr(_native_engine, name)
    elif name in _RING_EXPORTS:
        value = getattr(
            import_module("monitoring.ring_transport"), _RING_EXPORTS[name]
        )
    elif name in _SELECTION_EXPORTS:
        value = getattr(import_module("monitoring.selection"), _SELECTION_EXPORTS[name])
    elif name in _ADAPTOR_EXPORTS:
        value = getattr(import_module("monitoring.adaptor_base"), name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_EXPORTS)


__all__ = [
    "DMI_INTEGRATION_API_VERSION",
    "BackendAdaptor",
    "StepPlan",
    "StepReservation",
    "StepContext",
    "MonitoringEngine",
    "RingCapacities",
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
