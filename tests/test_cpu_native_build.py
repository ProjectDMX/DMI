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


def _torch_compile_lines(target: str) -> list[str]:
    """The compile commands a dry run of ``target`` would execute that pull in
    torch's headers -- the ones carrying TORCH_EXTENSION_NAME."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        # PYTHON is the interpreter running this test: its torch is the one
        # the build would target, and the Makefile's default `python` need
        # not exist.
        ["make", "-C", "native", "-B", "-n", target, f"PYTHON={sys.executable}"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot plan `{target}` here: {result.stderr[-300:]}")
    return [line for line in result.stdout.splitlines()
            if "-DTORCH_EXTENSION_NAME=" in line and " -c " in f" {line} "]


@pytest.mark.parametrize("target", [
    # host plans on any machine, so CI's cpu job checks it.
    pytest.param("host", marks=pytest.mark.cpu),
    # The full backend needs the CUDA toolchain even to be PLANNED, which a
    # cpu runner does not have. It is a gpu test, not a cpu test that skips:
    # the cpu gate rightly fails any skip that is not absent hardware.
    pytest.param("all", marks=pytest.mark.gpu),
])
def test_every_torch_including_compile_requests_cxx20(target):
    """PyTorch's headers refuse anything older than C++20.

    ATen.h opens with ``#error C++20 or later compatible compiler is required``,
    and the pinned range here (torch>=2.8,<3) resolves to a release that
    enforces it. The backend and host targets were still compiled with
    -std=c++17, so a fresh install could not build either one -- and CI never
    noticed, because it only dry-runs `host` and the torch-free drivers never
    reach ATen. This pins the flag on the plan actually executed rather than on
    the Makefile's text, so it holds however the flags are assembled.

    Only ``host`` is marked cpu. Planning ``all`` resolves the CUDA toolkit and
    fails without one, so it runs under the gpu marker instead.
    """
    lines = _torch_compile_lines(target)
    assert lines, f"no torch-including compile found in the `{target}` plan"
    stale = [line for line in lines
             if not any(f"-std={std}" in line
                        for std in ("c++20", "c++23", "c++26", "gnu++20", "gnu++23"))]
    assert not stale, (
        f"{len(stale)} torch-including compile(s) below C++20 in `{target}`:\n"
        + stale[0][:300])


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
