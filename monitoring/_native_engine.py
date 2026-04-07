"""Wrapper that builds and loads the native monitoring engine extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import importlib
import importlib.util
import importlib.machinery
import glob
import sys

import torch

_EXTENSION_NAME = "monitoring_native_backend"
_EXTENSION_MODULE: Optional[Any] = None


BASE_DIR = Path(__file__).resolve().parent


_NATIVE_EXPORTS = (
    "StageConfig",
    "DMXHostEngine",
    "ClickHouseClientConfig",
    "ThreadFailure",
    "QueueConfig",
    "EnqueuePolicy",
    "OnFullPolicy",
    "OnClosedPolicy",
    "consume_backend_futures_cpp",
    "RingConfig",
    "RingEngine",
    "ring_set_active_engine",
    "ring_clear_active_engine",
    "ring_set_cpu_direct",
)


def _load_extension() -> Any:
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE
    cached = sys.modules.get(_EXTENSION_NAME)
    if cached is not None:
        _EXTENSION_MODULE = cached
        return _EXTENSION_MODULE

    # Prefer the standard import path so Python picks the extension matching
    # the current ABI and caches it in sys.modules exactly once.
    try:
        _EXTENSION_MODULE = importlib.import_module(_EXTENSION_NAME)
        return _EXTENSION_MODULE
    except Exception:
        pass

    # JIT build is intentionally disabled for reproducibility/stability.
    # Only load an already-built extension from this repository tree.
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    seen_paths: set[Path] = set()
    candidates: list[Path] = []
    for base_dir in (repo_root, pkg_dir):
        for suffix in suffixes:
            so_path = (base_dir / f"{_EXTENSION_NAME}{suffix}").resolve()
            if so_path.exists() and so_path not in seen_paths:
                seen_paths.add(so_path)
                candidates.append(so_path)
        for so_path_str in glob.glob(str(base_dir / f"{_EXTENSION_NAME}*.so")):
            so_path = Path(so_path_str).resolve()
            if so_path.exists() and so_path not in seen_paths:
                seen_paths.add(so_path)
                candidates.append(so_path)

    for so_path in candidates:
        try:
            spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, so_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[_EXTENSION_NAME] = module
                spec.loader.exec_module(module)  # type: ignore[arg-type]
                _EXTENSION_MODULE = module
                return _EXTENSION_MODULE
        except Exception:
            sys.modules.pop(_EXTENSION_NAME, None)
            # Continue searching other local candidates.
            pass

    raise ImportError(
        "monitoring native backend .so not found in repository. "
        "Build it first with `make -C monitoring`."
    )


def create_engine(
    queue_size: int,
    cache_dtype: Optional[torch.dtype],
    delay_steps: int,
    *,
    pinpool_bins_kb: list[int] | tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192),
    pinpool_max_mb: int = 512,
    host_copy_threads: int = 0,
    host_copy_queue_size: int = 512,
):
    module = _load_extension()
    return module.create_engine(
        queue_size,
        cache_dtype,
        delay_steps,
        pinpool_bins_kb,
        pinpool_max_mb,
        host_copy_threads,
        host_copy_queue_size,
    )


def __getattr__(name: str) -> Any:
    if name in _NATIVE_EXPORTS:
        return getattr(_load_extension(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_NATIVE_EXPORTS)))


__all__ = ["create_engine", *_NATIVE_EXPORTS]
