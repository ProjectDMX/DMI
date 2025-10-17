"""Wrapper that builds and loads the native monitoring engine extension."""

from __future__ import annotations

import importlib
from pathlib import Path
import os
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


def _load_extension() -> Any:
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    # 1) Prefer a locally built .so in this repo to avoid importing a stale
    #    system-wide module with the same name.
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
            # Fall back to next strategy
            pass

    # 2) Try to import an installed module by name (may be stale).
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
        sources = sorted(str(p) for p in source_dir.glob("*.cpp"))
        if not sources:
            raise ImportError("native engine source files not found")

    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3"]
    extra_ldflags = []
    if bool(int(os.environ.get("MON_NATIVE_LTO", "0"))):
        extra_cflags.append("-flto")
        extra_ldflags.append("-flto")

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


__all__ = ["create_engine"]
