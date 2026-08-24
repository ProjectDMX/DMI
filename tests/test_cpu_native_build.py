from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.cpu


def test_host_build_plan_has_no_cuda_toolchain_or_libraries():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-C", "native", "-B", "-n", "host"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "_host_backend" in output
    if sys.platform == "darwin":
        assert "-undefined,dynamic_lookup" in output
    for forbidden in ("nvcc", "-lcuda", "-lcudart", "-lc10_cuda", "-ltorch_cuda"):
        assert forbidden not in output


def test_host_export_falls_back_to_cpu_backend(monkeypatch):
    from dmi.transport import native

    sentinel = object()
    calls = []

    def load_named(name):
        calls.append(name)
        if name == "_native_backend":
            raise ImportError("full backend absent")
        return SimpleNamespace(DMXHostEngine=sentinel)

    monkeypatch.setattr(native, "_load_named_extension", load_named)
    monkeypatch.setattr(native, "_EXTENSION_MODULES", {})

    assert native.DMXHostEngine is sentinel
    assert calls == ["_native_backend", "_host_backend"]


def test_ring_export_requires_full_backend(monkeypatch):
    from dmi.transport import native

    calls = []

    def load_named(name):
        calls.append(name)
        raise ImportError("backend absent")

    monkeypatch.setattr(native, "_load_named_extension", load_named)
    monkeypatch.setattr(native, "_EXTENSION_MODULES", {})

    with pytest.raises(ImportError, match="full native backend"):
        native.RingEngine
    assert calls == ["_native_backend"]


def test_v1_host_export_does_not_load_ring_backend():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dmi.api.v1; "
                "assert 'dmi.transport.ring' not in sys.modules, "
                "'ring backend was loaded by v1 host-export path'"
            ),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_v1_model_shape_contract_does_not_load_ring_backend():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dmi.api.v1; "
                "from types import SimpleNamespace; "
                "shape = dmi.api.v1.make_model_shape_from_hf_config("
                "    SimpleNamespace(hidden_size=64, num_attention_heads=8)); "
                "assert isinstance(shape, dmi.api.v1.ModelShapeConfig); "
                "assert 'dmi.transport.ring' not in sys.modules, "
                "'ring backend was loaded by v1 model-shape path'"
            ),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
