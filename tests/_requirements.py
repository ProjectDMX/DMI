"""Resource skip-guards for the test suite.

Every GPU / E2E / native test should fail *closed with a precise reason* when a
prerequisite is missing, never with an import error or an opaque crash. Each
helper here returns a ``pytest.mark.skipif`` marker, so it can be used either as
a decorator::

    from tests._requirements import require_cuda

    @require_cuda()
    def test_kernel():
        ...

or composed into a module-level mark list alongside a category marker::

    pytestmark = [pytest.mark.gpu, require_cuda()]

This module must stay importable on a CPU-only box with no CUDA, ClickHouse,
vLLM, model weights, or native build toolchain present. All heavy imports
(``torch`` in particular) are deferred into the helper bodies so that merely
importing this file costs nothing.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import socket

import pytest

__all__ = [
    "require_cuda",
    "require_gpus",
    "require_clickhouse",
    "require_vllm",
    "require_model_cache",
    "require_nvcc",
]


def _cuda_device_count() -> int:
    """CUDA device count, or 0 if torch/CUDA is unavailable. Never raises."""
    try:
        import torch
    except Exception:
        return 0
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def require_cuda():
    """Skip unless at least one CUDA device is visible."""
    return pytest.mark.skipif(
        _cuda_device_count() < 1, reason="no CUDA device available"
    )


def require_gpus(n: int):
    """Skip unless at least ``n`` CUDA devices are visible."""
    have = _cuda_device_count()
    return pytest.mark.skipif(
        have < n, reason=f"needs >= {n} CUDA device(s), found {have}"
    )


def require_clickhouse(host: str | None = None, port: int | None = None):
    """Skip unless a ClickHouse TCP port is reachable.

    Host/port default to the ``DMX_DB_HOST`` / ``DMX_DB_PORT`` env vars (and
    finally ``127.0.0.1:9000``), matching the runners' connection defaults.
    """
    host = host or os.environ.get("DMX_DB_HOST", "127.0.0.1")
    port = int(port if port is not None else os.environ.get("DMX_DB_PORT", "9000"))
    reachable = False
    try:
        with socket.create_connection((host, port), timeout=1.0):
            reachable = True
    except OSError:
        reachable = False
    return pytest.mark.skipif(
        not reachable, reason=f"ClickHouse unreachable at {host}:{port}"
    )


def require_vllm():
    """Skip unless the vLLM runtime is importable."""
    available = importlib.util.find_spec("vllm") is not None
    return pytest.mark.skipif(not available, reason="vLLM not importable")


def _model_in_cache(model: str) -> bool:
    """Best-effort check that ``model`` is a local path or a cached HF repo."""
    # Explicit local path (a checkpoint dir).
    if os.path.sep in model and os.path.exists(model):
        return True
    # HuggingFace hub cache layout: ``models--<org>--<name>``.
    cache_root = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(
            os.environ.get(
                "HF_HOME", os.path.expanduser("~/.cache/huggingface")
            ),
            "hub",
        ),
    )
    repo_dir = "models--" + model.replace("/", "--")
    return os.path.isdir(os.path.join(cache_root, repo_dir))


def require_model_cache(model: str):
    """Skip unless ``model`` (HF repo id or local path) is already on disk.

    Keeps the default/offline suites from triggering a network download.
    """
    return pytest.mark.skipif(
        not _model_in_cache(model),
        reason=f"model {model!r} not found in local cache",
    )


def require_nvcc():
    """Skip unless the CUDA compiler ``nvcc`` is on PATH (native ring tests)."""
    return pytest.mark.skipif(
        shutil.which("nvcc") is None, reason="nvcc not found on PATH"
    )
