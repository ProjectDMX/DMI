"""Lightweight helpers exposing native monitoring ops to Python."""

from __future__ import annotations

from typing import Any

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


__all__ = ["monitor_activation"]
