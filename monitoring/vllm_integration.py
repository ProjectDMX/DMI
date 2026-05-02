"""Backwards-compat shim for the vLLM monitored worker.

Phase 3a moved the implementations to ``integration/vllm_adapter.py``.
This module re-exports the public surface so existing call sites keep
working, including:

  * ``worker_cls="monitoring.vllm_integration.DMXGPUWorker"`` strings
    used in test runners and benchmark scripts.
  * Direct imports like ``from monitoring.vllm_integration import
    DMXGPUWorker`` (e.g. ``tests/compare_worker.py``).
  * ``from monitoring.vllm_integration import normalize_vllm_request_id``
    (``tests/vllm_rowcnt_comparator.py``).

Importing this module also triggers the ``register_preset("vllm-full", ...)``
call at the top of ``integration/vllm_adapter.py`` -- so the
``"vllm-full"`` preset is available before any caller resolves it.

Phase 5 deletes this shim once external callers migrate to
``integration.vllm_adapter``.
"""
from __future__ import annotations

from integration.vllm_adapter import (
    DMXGPUWorker,
    VLLMAdaptor,
    normalize_vllm_request_id,
)

__all__ = [
    "DMXGPUWorker",
    "VLLMAdaptor",
    "normalize_vllm_request_id",
]
