"""Wrapper that builds and loads the native monitoring engine extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.machinery
import importlib.util
import glob

_FULL_EXTENSION_NAME = "monitoring_native_backend"
_HOST_EXTENSION_NAME = "monitoring_host_backend"
_EXTENSION_MODULES: dict[str, Any] = {}

_HOST_EXPORTS = (
    "StageConfig",
    "DMXHostEngine",
    "ClickHouseClientConfig",
    "QueueConfig",
    "EnqueuePolicy",
    "OnFullPolicy",
    "OnClosedPolicy",
    "ThreadFailure",
)

_RING_EXPORTS = (
    "RingConfig",
    "RingEngine",
    "ring_set_active_engine",
    "ring_clear_active_engine",
)

_NATIVE_EXPORTS = (*_HOST_EXPORTS, *_RING_EXPORTS)


def _load_named_extension(extension_name: str) -> Any:
    cached = _EXTENSION_MODULES.get(extension_name)
    if cached is not None:
        return cached

    # JIT build is intentionally disabled for reproducibility/stability.
    # Only load an already-built extension from this repository tree.
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    candidates = []
    seen = set()
    search_dirs = (pkg_dir, repo_root)
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for search_dir in search_dirs:
            path = search_dir / f"{extension_name}{suffix}"
            if path.exists() and path not in seen:
                candidates.append(str(path))
                seen.add(path)
    for search_dir in search_dirs:
        for so_path in glob.glob(str(search_dir / f"{extension_name}*.so")):
            path = Path(so_path)
            if path not in seen:
                candidates.append(str(path))
                seen.add(path)

    failures = []
    for so_path in candidates:
        try:
            spec = importlib.util.spec_from_file_location(extension_name, so_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[arg-type]
                _EXTENSION_MODULES[extension_name] = module
                return module
        except Exception as exc:
            failures.append(f"{so_path}: {exc}")

    detail = f" Last load error: {failures[-1]}" if failures else ""
    raise ImportError(f"{extension_name} extension not found or loadable.{detail}")


def _load_extension() -> Any:
    try:
        return _load_named_extension(_FULL_EXTENSION_NAME)
    except ImportError as exc:
        raise ImportError(
            "DMI full native backend is unavailable. "
            "Build it with `make -C monitoring`."
        ) from exc


def _load_host_extension() -> Any:
    try:
        return _load_named_extension(_FULL_EXTENSION_NAME)
    except ImportError:
        pass
    try:
        return _load_named_extension(_HOST_EXTENSION_NAME)
    except ImportError as exc:
        raise ImportError(
            "DMI host native backend is unavailable. "
            "Build it with `make -C monitoring host`."
        ) from exc


def __getattr__(name: str) -> Any:
    if name in _HOST_EXPORTS:
        return getattr(_load_host_extension(), name)
    if name in _RING_EXPORTS:
        return getattr(_load_extension(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_NATIVE_EXPORTS)))


__all__ = [*_NATIVE_EXPORTS]
