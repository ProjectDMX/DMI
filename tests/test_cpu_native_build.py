from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import get_type_hints

import pytest


pytestmark = pytest.mark.cpu


def test_host_build_plan_has_no_cuda_toolchain_or_libraries():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-C", "monitoring", "-B", "-n", "host"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "monitoring_host_backend" in output
    if sys.platform == "darwin":
        assert "-undefined,dynamic_lookup" in output
    for forbidden in ("nvcc", "-lcuda", "-lcudart", "-lc10_cuda", "-ltorch_cuda"):
        assert forbidden not in output


def test_host_export_falls_back_to_cpu_backend(monkeypatch):
    from monitoring import _native_engine

    sentinel = object()
    calls = []

    def load_named(name):
        calls.append(name)
        if name == "monitoring_native_backend":
            raise ImportError("full backend absent")
        return SimpleNamespace(DMXHostEngine=sentinel)

    monkeypatch.setattr(_native_engine, "_load_named_extension", load_named)
    monkeypatch.setattr(_native_engine, "_EXTENSION_MODULES", {})

    assert _native_engine.DMXHostEngine is sentinel
    assert calls == ["monitoring_native_backend", "monitoring_host_backend"]


def test_ring_export_requires_full_backend(monkeypatch):
    from monitoring import _native_engine

    calls = []

    def load_named(name):
        calls.append(name)
        raise ImportError("backend absent")

    monkeypatch.setattr(_native_engine, "_load_named_extension", load_named)
    monkeypatch.setattr(_native_engine, "_EXTENSION_MODULES", {})

    with pytest.raises(ImportError, match="full native backend"):
        _native_engine.RingEngine
    assert calls == ["monitoring_native_backend"]


def test_v1_host_export_does_not_load_ring_backend(monkeypatch):
    from monitoring import _native_engine

    sentinel = object()
    host_module = SimpleNamespace(DMXHostEngine=sentinel)
    monkeypatch.setattr(_native_engine, "_load_host_extension", lambda: host_module)
    sys.modules.pop("monitoring.integration_api.v1", None)

    api = importlib.import_module("monitoring.integration_api.v1")

    assert api.DMXHostEngine is sentinel
    assert "monitoring.ring_transport" not in sys.modules


def test_v1_model_shape_contract_does_not_load_ring_backend():
    sys.modules.pop("monitoring.integration_api.v1", None)

    api = importlib.import_module("monitoring.integration_api.v1")
    hints = get_type_hints(api.make_model_shape_from_hf_config)
    shape = api.make_model_shape_from_hf_config(
        SimpleNamespace(hidden_size=64, num_attention_heads=8)
    )

    assert hints["return"] == api.ModelShapeConfig | None
    assert isinstance(shape, api.ModelShapeConfig)
    assert "monitoring.ring_transport" not in sys.modules
