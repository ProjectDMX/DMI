"""Wrapper that builds and loads the native monitoring engine extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import glob

_EXTENSION_NAME = "monitoring_native_backend"
_EXTENSION_MODULE: Optional[Any] = None


BASE_DIR = Path(__file__).resolve().parent


_NATIVE_EXPORTS = (
    "StageConfig",
    "DMXHostEngine",
    "ClickHouseClientConfig",
    "QueueConfig",
    "EnqueuePolicy",
    "OnFullPolicy",
    "OnClosedPolicy",
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

    # JIT build is intentionally disabled for reproducibility/stability.
    # Only load an already-built extension from this repository tree.
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    candidates = []
    candidates.extend(glob.glob(str(pkg_dir / f"{_EXTENSION_NAME}*.so")))
    candidates.extend(glob.glob(str(repo_root / f"{_EXTENSION_NAME}*.so")))

    for so_path in candidates:
        try:
            spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, so_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[arg-type]
                _EXTENSION_MODULE = module
                return _EXTENSION_MODULE
        except Exception:
            # Continue searching other local candidates.
            pass

    raise ImportError(
        "monitoring native backend .so not found in repository. "
        "Build it first with `make -C monitoring`."
    )


def __getattr__(name: str) -> Any:
    if name in _NATIVE_EXPORTS:
        return getattr(_load_extension(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_NATIVE_EXPORTS)))


__all__ = [*_NATIVE_EXPORTS]
