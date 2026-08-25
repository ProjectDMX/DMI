from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import get_type_hints

import pytest


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.native_backend
def test_clickhouse_stage_has_bounded_batching_defaults():
    from dmi.transport.native import ClickHouseClientConfig, StageConfig

    stage = StageConfig.clickhouse_insert(ClickHouseClientConfig())
    queue = stage.input_queue

    assert queue.min_batch_items is None
    assert queue.min_batch_size == 16 * 1024**2
    assert queue.max_linger_s == pytest.approx(0.05)
    assert queue.max_batch_items == 10_000
    assert queue.max_batch_size is None
    assert queue.high_watermark_items == 20_000
    assert queue.high_watermark_size == 512 * 1024**2


@pytest.mark.native_backend
def test_clickhouse_client_exposes_socket_timeouts_and_worker_metrics():
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    config = ClickHouseClientConfig()
    assert config.connect_timeout_ms == 5000
    assert config.receive_timeout_ms == 0
    assert config.send_timeout_ms == 0

    engine = DMXHostEngine(StageConfig.clickhouse_insert(config, parallelism=3))
    metrics = engine.clickhouse_metrics()
    assert metrics.expected_workers == 3
    assert metrics.ready_workers == 0
    assert metrics.peak_active_inserts == 0
    assert [worker.worker_index for worker in metrics.workers] == [0, 1, 2]


@pytest.mark.native_backend
def test_engine_metrics_follow_mutated_stage_parallelism():
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    stage = StageConfig.clickhouse_insert(ClickHouseClientConfig(), parallelism=1)
    stage.parallelism = 3

    metrics = DMXHostEngine(stage).clickhouse_metrics()

    assert metrics.expected_workers == 3
    assert [worker.worker_index for worker in metrics.workers] == [0, 1, 2]


@pytest.mark.cpu
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


@pytest.mark.cpu
def test_v1_host_export_does_not_load_ring_backend(monkeypatch):
    import dmi.api.v1 as api
    from dmi.transport import native

    sentinel = object()
    host_module = SimpleNamespace(DMXHostEngine=sentinel)
    ring_before = sys.modules.get("dmi.transport.ring")
    cached = api.__dict__.pop("DMXHostEngine", None)
    monkeypatch.setattr(native, "_load_host_extension", lambda: host_module)
    try:
        assert api.DMXHostEngine is sentinel
        assert sys.modules.get("dmi.transport.ring") is ring_before
    finally:
        api.__dict__.pop("DMXHostEngine", None)
        if cached is not None:
            api.__dict__["DMXHostEngine"] = cached


@pytest.mark.cpu
def test_v1_model_shape_contract_does_not_load_ring_backend():
    import dmi.api.v1 as api

    ring_before = sys.modules.get("dmi.transport.ring")
    hints = get_type_hints(api.make_model_shape_from_hf_config)
    shape = api.make_model_shape_from_hf_config(
        SimpleNamespace(hidden_size=64, num_attention_heads=8)
    )

    assert hints["return"] == api.ModelShapeConfig | None
    assert isinstance(shape, api.ModelShapeConfig)
    assert sys.modules.get("dmi.transport.ring") is ring_before
