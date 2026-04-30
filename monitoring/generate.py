"""Backwards-compat shim for the HF generate entry points.

Phase 2a of the unified-adaptor refactor moved the implementations to
``integration/hf_adapter.py``.  This file re-exports the public surface
so existing call sites keep working.  Phase 5 deletes this shim once
external callers migrate to ``from integration.hf_adapter import ...``.
"""
from __future__ import annotations

from integration.hf_adapter import (
    HFAdaptor,
    GreedyGenerateTimings,
    generate_with_monitoring,
    generate_greedy,
    _make_model_shape,
    _make_model_shape_from_hf_config,
    _prepare_profile_times,
    print_prepare_profile,
)

__all__ = [
    "HFAdaptor",
    "GreedyGenerateTimings",
    "generate_with_monitoring",
    "generate_greedy",
    "_make_model_shape",
    "_make_model_shape_from_hf_config",
    "_prepare_profile_times",
    "print_prepare_profile",
]
