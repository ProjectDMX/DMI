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
def test_schema_driven_stage_is_additive_and_uses_bounded_batching_defaults():
    from dmi.api.v1 import RecordCellType, RecordColumn, RecordLayout, RecordSchema
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    schema = RecordSchema(
        (
            RecordLayout(
                name="event",
                table="event_records",
                columns=(RecordColumn("event_id", RecordCellType.INT64),),
                primary_key=("event_id",),
                order_by=("event_id",),
            ),
        )
    )
    config = ClickHouseClientConfig()
    assert not hasattr(config, "record_schema")

    stage = StageConfig.clickhouse_records(config, schema)
    queue = stage.input_queue
    assert stage.name == "clickhouse_records"
    assert queue.min_batch_items is None
    assert queue.min_batch_size == 16 * 1024**2
    assert queue.max_linger_s == pytest.approx(0.05)
    assert queue.max_batch_items == 10_000
    assert queue.high_watermark_items == 20_000
    assert queue.high_watermark_size == 512 * 1024**2

    engine = DMXHostEngine(stage)
    assert callable(engine.submit_record)
    assert callable(engine.flush_and_wait)


@pytest.mark.native_backend
def test_record_host_schema_identity_is_exact_but_layout_order_is_irrelevant():
    from dmi.api.v1 import RecordCellType, RecordColumn, RecordLayout, RecordSchema
    from dmi.transport import native
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    event_layout = RecordLayout(
        name="event",
        table="event_records",
        columns=(
            RecordColumn("run", RecordCellType.STRING),
            RecordColumn("event_id", RecordCellType.INT64),
            RecordColumn("score", RecordCellType.FLOAT64),
        ),
        primary_key=("run", "event_id"),
        order_by=("run", "event_id"),
    )
    tensor_layout = RecordLayout(
        name="tensor",
        table="tensor_records",
        columns=(
            RecordColumn("run", RecordCellType.STRING),
            RecordColumn(
                "payload",
                RecordCellType.TENSOR,
                dtype_column="payload_dtype",
                shape_column="payload_shape",
                bytes_column="payload_bytes",
            ),
        ),
        primary_key=("run",),
        order_by=("run",),
    )
    schema = RecordSchema((event_layout, tensor_layout), index_granularity=1024)
    reordered_schema = RecordSchema(
        (tensor_layout, event_layout), index_granularity=1024
    )

    def event_variant(
        *,
        name=event_layout.name,
        table=event_layout.table,
        columns=event_layout.columns,
        primary_key=event_layout.primary_key,
        order_by=event_layout.order_by,
    ):
        return RecordLayout(
            name=name,
            table=table,
            columns=columns,
            primary_key=primary_key,
            order_by=order_by,
        )

    def tensor_variant(*, dtype="payload_dtype", shape="payload_shape", bytes_="payload_bytes"):
        return RecordLayout(
            name="tensor",
            table="tensor_records",
            columns=(
                RecordColumn("run", RecordCellType.STRING),
                RecordColumn(
                    "payload",
                    RecordCellType.TENSOR,
                    dtype_column=dtype,
                    shape_column=shape,
                    bytes_column=bytes_,
                ),
            ),
            primary_key=("run",),
            order_by=("run",),
        )

    identity_mismatches = {
        "layout set": RecordSchema((event_layout,), index_granularity=1024),
        "layout name": RecordSchema(
            (event_variant(name="other_event"), tensor_layout),
            index_granularity=1024,
        ),
        "target table": RecordSchema(
            (event_variant(table="other_event_records"), tensor_layout),
            index_granularity=1024,
        ),
        "logical column order": RecordSchema(
            (
                event_variant(
                    columns=(
                        event_layout.columns[0],
                        event_layout.columns[2],
                        event_layout.columns[1],
                    )
                ),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "logical column name": RecordSchema(
            (
                event_variant(
                    columns=(
                        event_layout.columns[0],
                        event_layout.columns[1],
                        RecordColumn("metric", RecordCellType.FLOAT64),
                    )
                ),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "logical column type": RecordSchema(
            (
                event_variant(
                    columns=(
                        event_layout.columns[0],
                        event_layout.columns[1],
                        RecordColumn("score", RecordCellType.INT64),
                    )
                ),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "tensor dtype column": RecordSchema(
            (event_layout, tensor_variant(dtype="other_dtype")),
            index_granularity=1024,
        ),
        "tensor shape column": RecordSchema(
            (event_layout, tensor_variant(shape="other_shape")),
            index_granularity=1024,
        ),
        "tensor bytes column": RecordSchema(
            (event_layout, tensor_variant(bytes_="other_bytes")),
            index_granularity=1024,
        ),
        "primary key order": RecordSchema(
            (
                event_variant(primary_key=("event_id", "run")),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "ordering key order": RecordSchema(
            (
                event_variant(order_by=("event_id", "run")),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "index granularity": RecordSchema(
            (event_layout, tensor_layout), index_granularity=2048
        ),
    }

    host = DMXHostEngine(
        StageConfig.clickhouse_records(ClickHouseClientConfig(), schema)
    )
    backend = native._load_host_extension()
    backend._validate_record_host_schema(host, reordered_schema)
    for field, mismatched_schema in identity_mismatches.items():
        try:
            backend._validate_record_host_schema(host, mismatched_schema)
        except ValueError as error:
            assert "does not match" in str(error)
        else:
            pytest.fail(f"schema identity omitted {field}")

    legacy_host = DMXHostEngine(
        StageConfig.clickhouse_insert(ClickHouseClientConfig())
    )
    with pytest.raises(RuntimeError, match="schema-driven record stage"):
        backend._validate_record_host_schema(legacy_host, schema)


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
