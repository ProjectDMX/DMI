"""Native metadata-only output, asynchronous completion, and real CUDA transport."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
import torch

from dmi.api.v1 import (
    DropConfig, MonitoringConfig, MonitoringEngine, PayloadSlice,
    RecordCellType, RecordColumn, RecordDescriptor, RecordLayout, RecordSchema,
    OutputStorage,
)
from tests._requirements import require_native_backend

pytestmark = [pytest.mark.native_backend, require_native_backend()]


def schema():
    return RecordSchema((
        RecordLayout(
            name="tensor", table="tensor_rows",
            columns=(RecordColumn("identity", RecordCellType.STRING),
                     RecordColumn("row_index", RecordCellType.INT64),
                     RecordColumn("tensor", RecordCellType.TENSOR,
                                  dtype_column="dtype", shape_column="shape", bytes_column="bytes")),
            primary_key=("identity",), order_by=("identity",),
        ),
        RecordLayout(
            name="scalar", table="scalar_rows",
            columns=(RecordColumn("identity", RecordCellType.STRING),
                     RecordColumn("value", RecordCellType.FLOAT64)),
            primary_key=("identity",), order_by=("identity",),
        ),
    ))


def sink(root, rank=0, timing=False, ring=False):
    from dmi.transport.native import DropRecordSink
    return DropRecordSink(schema(), str(root), rank, timing, ring)


def submit_tensor(target, identity="a", shape=(2, 3)):
    tensor = torch.arange(6, dtype=torch.float32)
    descriptor = RecordDescriptor("tensor", ((identity, 17,
        PayloadSlice(dtype=torch.float32, shape=shape)),))
    target.submit(descriptor, tensor.view(torch.uint8))


def events(root, rank=0):
    return [json.loads(line) for line in
            (Path(root) / f"rank_{rank:05d}" / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("timing,ring", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_flags_and_fixed_origin(tmp_path, timing, ring):
    target = sink(tmp_path, rank=9, timing=timing, ring=ring)
    submit_tensor(target, "setup")
    for iteration in (41, 42):  # A resumed run need not start at iteration 1.
        target.iteration_start(iteration)
        submit_tensor(target, str(iteration))
        target.iteration_end(iteration)
        target.ring_metrics(iteration, {"payload_reserved_bytes": 32})
    target.close()
    rows = events(tmp_path, 9)
    assert all(row["rank"] == 9 for row in rows)
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    records = [row for row in rows if row["type"] == "record"]
    assert len(records) == 3
    assert "cpu_arrival_ns" not in records[0]  # No invented origin at setup.
    starts = [row for row in rows if row["type"] == "iteration_start"]
    ends = [row for row in rows if row["type"] == "iteration_end"]
    if timing:
        assert starts[0]["time_ns"] == 0
        assert starts[1]["time_ns"] >= ends[0]["time_ns"] > 0
        for start, record, end in zip(starts, records[1:], ends):
            assert start["time_ns"] <= record["cpu_arrival_ns"] <= end["time_ns"]
            assert end["duration_ns"] == end["time_ns"] - start["time_ns"]
    else:
        assert starts == ends == []
        assert all("time_ns" not in row and "cpu_arrival_ns" not in row for row in rows)
    assert sum(row["type"] == "ring_metrics" for row in rows) == (2 if ring else 0)


def test_schema_metadata_dynamic_empty_scalar_and_append(tmp_path):
    target = sink(tmp_path)
    submit_tensor(target, 'quote"\n雪', shape=(-1, 3))
    target.submit(RecordDescriptor("tensor", (("empty", 18,
        PayloadSlice(dtype=torch.float32, shape=(-1, 3))),)), torch.empty(0, dtype=torch.uint8))
    target.submit(RecordDescriptor("scalar", (("loss",
        PayloadSlice(dtype=torch.float32, shape=(), storage=OutputStorage.SCALAR_FLOAT)),)),
        torch.tensor([2.5], dtype=torch.float32).view(torch.uint8))
    assert target.flush_and_wait(5)
    first = events(tmp_path)
    assert first[1]["rows"] == [{"identity": 'quote"\n雪', "row_index": 17,
                                  "dtype": "torch.float32", "shape": [2, 3]}]
    assert first[2]["rows"][0]["shape"] == [0, 3]
    assert first[3]["rows"] == [{"identity": "loss", "value": 2.5}]
    assert all("bytes" not in row for event in first for row in event.get("rows", []))
    target.close()
    reopened = sink(tmp_path)
    submit_tensor(reopened, "later")
    reopened.close()
    assert events(tmp_path)[:len(first)] == first


def test_flush_accepts_timeout_longer_than_24_hours(tmp_path):
    target = sink(tmp_path)
    try:
        submit_tensor(target)
        assert target.flush_and_wait(48 * 60 * 60)
        assert events(tmp_path)[1]["rows"][0]["identity"] == "a"
    finally:
        target.close()


def test_one_writer_per_rank_and_concurrent_producers_drain(tmp_path):
    target = sink(tmp_path, rank=3)
    with pytest.raises(RuntimeError, match="another writer"):
        sink(tmp_path, rank=3)
    other = sink(tmp_path, rank=4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: submit_tensor(target, str(i)), range(200)))
    submit_tensor(other, "other")
    target.close()
    other.close()
    records = [row for row in events(tmp_path, 3) if row["type"] == "record"]
    assert len(records) == 200
    assert {row["rows"][0]["identity"] for row in records} == {str(i) for i in range(200)}
    assert events(tmp_path, 4)[1]["rows"][0]["identity"] == "other"
    with pytest.raises(RuntimeError, match="after close"):
        submit_tensor(target)


def test_write_failure_surfaces_at_barrier_and_close(tmp_path):
    directory = tmp_path / "rank_00000"
    directory.mkdir()
    (directory / "events.jsonl").symlink_to("/dev/full")
    target = sink(tmp_path)
    with pytest.raises(RuntimeError):
        target.flush_and_wait(5)
    with pytest.raises(RuntimeError):
        target.close()


def test_disabled_engine_callbacks_do_not_sample(monkeypatch):
    from types import SimpleNamespace
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled measurement path called")
    engine = MonitoringEngine(enable_ring_transport=False)
    engine._drop_sink = SimpleNamespace(timing_enabled=False, ring_metrics_enabled=False,
        iteration_start=forbidden, iteration_end=forbidden, ring_metrics=forbidden)
    engine._ring_engine = SimpleNamespace(ring_metrics=forbidden)
    engine.record_iteration_start(1)
    engine.record_iteration_end(1)
    engine._drop_sink = None
    engine._ring_engine = None


@pytest.mark.gpu
@pytest.mark.parametrize("timing,ring", [(False, False), (True, False), (False, True), (True, True)])
def test_real_cuda_drop_shutdown(tmp_path, timing, ring):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from dmi.api.v1 import RingConfig, ProducerPlanBuilder, TransportSpec, HookPointV1, HookSpecV1
    class Format:
        schema = schema()
        def encode(self, metadata, entry):
            return RecordDescriptor("tensor", ((metadata, 0,
                PayloadSlice(dtype=entry.dtype, shape=entry.output_shape)),), output_id=entry.output_id)
    cfg = RingConfig()
    cfg.payload_ring_bytes = cfg.pinned_staging_bytes = 1024 * 1024
    cfg.task_ring_entries = 64
    cfg.drain_flush_byte_threshold = 0
    cfg.drain_flush_payload_ratio = 0
    engine = MonitoringEngine(config=MonitoringConfig(storage_backend="drop",
        drop=DropConfig(str(tmp_path), rank=7, timing_enabled=timing, ring_metrics_enabled=ring)),
        ring_config=cfg, record_mode_v1=True)
    runtime = engine.create_record_runtime(Format())
    class HookRuntime:
        metadata = ""
        def should_emit(self, hook):
            return True
        def prepare_output(self, *, output_id, output_spec, output, **kwargs):
            entry = ProducerPlanBuilder().record_output(
                output_id=output_id, output_spec=output_spec, output=output)
            return runtime.emit_output(entry, self.metadata, output)
    hook_runtime = HookRuntime()
    hook = HookPointV1(HookSpecV1("drop_test", (TransportSpec("source"),)))
    runtime.bind_hook(hook, hook_runtime=hook_runtime)
    source = torch.arange(24, dtype=torch.float32, device="cuda").reshape(4, 6)
    try:
        for iteration in (5, 6):
            engine.record_iteration_start(iteration)
            hook_runtime.metadata = str(iteration)
            hook(source)
            engine.record_iteration_end(iteration)
        engine.close()  # No explicit flush: shutdown must deliver the pending records.
    finally:
        engine.close()
    rows = events(tmp_path, 7)
    records = [row for row in rows if row["type"] == "record"]
    assert [row["rows"][0]["identity"] for row in records] == ["5", "6"]
    assert all(row["rows"][0]["shape"] == [4, 6] for row in records)
    snapshots = [row for row in rows if row["type"] == "ring_metrics"]
    assert len(snapshots) == (2 if ring else 0)
    if ring:
        assert snapshots[0]["payload_reserved_bytes"] >= 96
        assert snapshots[0]["payload_high_water_bytes"] >= 96
        assert snapshots[0]["task_high_water"] >= 1
    if timing:
        assert next(row for row in rows if row["type"] == "iteration_start")["time_ns"] == 0
        assert all(row["cpu_arrival_ns"] > 0 for row in records)
