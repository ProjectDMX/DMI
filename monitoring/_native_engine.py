"""Wrapper that builds and loads the native monitoring engine extension."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional

import torch

try:
    from torch.utils.cpp_extension import load as load_extension
except ImportError:  # pragma: no cover - torch without cpp extension support
    load_extension = None  # type: ignore[assignment]

_EXTENSION_NAME = "monitoring_native_backend"
_EXTENSION_MODULE: Optional[Any] = None


def _load_extension() -> Any:
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    # Try to import a pre-built module first.
    try:
        _EXTENSION_MODULE = importlib.import_module(_EXTENSION_NAME)
        return _EXTENSION_MODULE
    except ImportError:
        pass

    if load_extension is None:
        raise ImportError("torch.utils.cpp_extension is not available")

    source_dir = Path(__file__).resolve().parent / "csrc"
    source_file = source_dir / "native_engine.cpp"
    if not source_file.exists():
        raise ImportError("native engine source not found")

    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3"]

    try:
        _EXTENSION_MODULE = load_extension(
            name=_EXTENSION_NAME,
            sources=[str(source_file)],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
    except (RuntimeError, OSError) as exc:  # pragma: no cover - compile failure path
        raise ImportError("failed to build monitoring native backend") from exc

    return _EXTENSION_MODULE


def create_engine(
    queue_size: int,
    cache_dtype: Optional[torch.dtype],
    delay_steps: int,
):
    module = _load_extension()
    return module.create_engine(queue_size, cache_dtype, delay_steps)


__all__ = ["create_engine"]
