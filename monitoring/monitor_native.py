"""Lightweight helpers exposing native monitoring ops to Python."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from . import _native_engine

_EXTENSION = None


def _get_extension():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = _native_engine._load_extension()
    return _EXTENSION


def monitor_activation(tensor: torch.Tensor, handle: Any) -> torch.Tensor:
    """Call the native monitor op in-line and return the original tensor."""
    if handle is None:
        return tensor
    ext = _get_extension()
    return ext.monitor_activation(tensor, handle)


def parse_shadow_block(
    metadata: torch.Tensor,
    slot_ids: Sequence[int],
    hook_names: Sequence[str],
):
    """Parse graph shadow metadata into a native backend SoA spec."""

    if len(slot_ids) != len(hook_names):
        raise ValueError("slot_ids and hook_names must have the same length")
    ext = _get_extension()
    slot_list = [int(idx) for idx in slot_ids]
    name_list = [str(name) for name in hook_names]
    return ext.parse_shadow_block(metadata, slot_list, name_list)


def create_graph_delegate(backend):
    """Instantiate the native Graph delegate."""

    if backend is None:
        raise ValueError("backend must not be None")
    ext = _get_extension()
    return ext.GraphNativeDelegate(backend)


__all__ = ["monitor_activation", "parse_shadow_block", "create_graph_delegate"]
