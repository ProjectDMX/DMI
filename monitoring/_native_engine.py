"""Wrapper that builds and loads the native monitoring engine extension."""

from __future__ import annotations

import importlib
from pathlib import Path
import os
import shlex
from typing import Any, Optional
import importlib.util
import glob

import torch

try:
    from torch.utils.cpp_extension import load as load_extension
except ImportError:  # pragma: no cover - torch without cpp extension support
    load_extension = None  # type: ignore[assignment]

_EXTENSION_NAME = "monitoring_native_backend"
_EXTENSION_MODULE: Optional[Any] = None


_NATIVE_EXPORTS = (
    "StageConfig",
    "DMXHostEngine",
    "ClickHouseClientConfig",
    "ThreadFailure",
)


def _load_extension() -> Any:
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    force_build = bool(int(os.environ.get("MON_NATIVE_FORCE_BUILD", "0")))

    # 1) Prefer a locally built .so in this repo to avoid importing a stale
    #    system-wide module with the same name.
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    candidates = [] if force_build else []
    if not force_build:
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
            # Fall back to next strategy
            pass

    # 2) Try to import an installed module by name (may be stale).
    if not force_build:
        try:
            _EXTENSION_MODULE = importlib.import_module(_EXTENSION_NAME)
            return _EXTENSION_MODULE
        except ImportError:
            pass

    if load_extension is None:
        raise ImportError("torch.utils.cpp_extension is not available")

    source_dir = Path(__file__).resolve().parent / "csrc"

    use_unified = bool(int(os.environ.get("MON_NATIVE_UNIFIED", "0")))
    if use_unified:
        unified = source_dir / "unified.cpp"
        if not unified.exists():
            raise ImportError("unified.cpp not found in csrc")
        sources = [str(unified)]
    else:
        sources = sorted(str(p) for p in source_dir.glob("*.cpp") if p.name != "unified.cpp")
        if not sources:
            raise ImportError("native engine source files not found")

    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3"]
    extra_ldflags = []
    if bool(int(os.environ.get("MON_NATIVE_LTO", "0"))):
        extra_cflags.append("-flto")
        extra_ldflags.append("-flto")


    # ---- ClickHouse C++ client (matches Makefile env knobs) ----
    # CLICKHOUSE_INCLUDE: path to headers (e.g. /usr/local/include)
    # CLICKHOUSE_LIB_DIR: path to libs    (e.g. /usr/local/lib)
    # CLICKHOUSE_LIBS:    link flags      (default: -lclickhouse-cpp-lib)
    ch_include = os.environ.get("CLICKHOUSE_INCLUDE", "").strip()
    if ch_include:
        extra_cflags.append(f"-I{ch_include}")
        extra_cuda_cflags.append(f"-I{ch_include}")

    ch_lib_dir = os.environ.get("CLICKHOUSE_LIB_DIR", "").strip()
    if ch_lib_dir:
        extra_ldflags.extend([f"-L{ch_lib_dir}", f"-Wl,-rpath,{ch_lib_dir}"])

    ch_libs_env = os.environ.get("CLICKHOUSE_LIBS")
    ch_libs = "-lclickhouse-cpp-lib" if ch_libs_env is None else ch_libs_env
    ch_libs = ch_libs.strip()
    if ch_libs:
        extra_ldflags.extend(shlex.split(ch_libs))

    try:
        _EXTENSION_MODULE = load_extension(
            name=_EXTENSION_NAME,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
            extra_ldflags=extra_ldflags,
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


def __getattr__(name: str) -> Any:
    if name in _NATIVE_EXPORTS:
        return getattr(_load_extension(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_NATIVE_EXPORTS)))


__all__ = ["create_engine", *_NATIVE_EXPORTS]
