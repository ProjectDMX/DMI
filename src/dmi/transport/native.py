"""Lazy loaders for DMI's compiled native backends."""

from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any


_FULL_EXTENSION_NAME = "_native_backend"
_HOST_EXTENSION_NAME = "_host_backend"
_EXTENSION_MODULES: dict[str, Any] = {}

_HOST_EXPORTS = frozenset(
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
_RING_EXPORTS = frozenset(
    {
        "RingConfig",
        "RingEngine",
        "ring_set_active_engine",
        "ring_clear_active_engine",
    }
)
_NATIVE_EXPORTS = _HOST_EXPORTS | _RING_EXPORTS


def _search_dirs() -> tuple[Path, ...]:
    transport_dir = Path(__file__).resolve().parent
    package_root = transport_dir.parent
    source_root = package_root.parent
    repo_root = source_root.parent if source_root.name == "src" else source_root
    native_dir = repo_root / "native"
    return package_root, transport_dir, native_dir, native_dir / "build", repo_root


def _load_named_extension(extension_name: str) -> Any:
    cached = _EXTENSION_MODULES.get(extension_name)
    if cached is not None:
        return cached

    candidates: list[Path] = []
    seen: set[Path] = set()
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for search_dir in _search_dirs():
            path = search_dir / f"{extension_name}{suffix}"
            if path.exists() and path not in seen:
                candidates.append(path)
                seen.add(path)
    for search_dir in _search_dirs():
        for so_path in glob.glob(str(search_dir / f"{extension_name}*.so")):
            path = Path(so_path)
            if path not in seen:
                candidates.append(path)
                seen.add(path)

    failures = []
    for path in candidates:
        try:
            spec = importlib.util.spec_from_file_location(extension_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[arg-type]
                _EXTENSION_MODULES[extension_name] = module
                return module
        except Exception as exc:
            failures.append(f"{path}: {exc}")

    detail = f" Last load error: {failures[-1]}" if failures else ""
    raise ImportError(f"{extension_name} extension not found or loadable.{detail}")


def _load_extension() -> Any:
    try:
        return _load_named_extension(_FULL_EXTENSION_NAME)
    except ImportError as exc:
        raise ImportError(
            "DMI full native backend is unavailable. Build it with `make -C native`."
        ) from exc


def _load_host_extension() -> Any:
    try:
        return _load_named_extension(_FULL_EXTENSION_NAME)
    except ImportError:
        try:
            return _load_named_extension(_HOST_EXTENSION_NAME)
        except ImportError as exc:
            raise ImportError(
                "DMI host backend is unavailable. Build it with `make -C native host`."
            ) from exc


def __getattr__(name: str) -> Any:
    if name in _HOST_EXPORTS:
        return getattr(_load_host_extension(), name)
    if name in _RING_EXPORTS:
        return getattr(_load_extension(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS)


__all__ = sorted(_NATIVE_EXPORTS)
