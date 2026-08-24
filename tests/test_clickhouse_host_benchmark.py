from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
import torch

from benchmark.bench_clickhouse_host import (
    BenchmarkConfig,
    TrialMeasurement,
    _server_time,
    _query_log_metrics,
    _submit_and_drain,
    build_client_settings,
    configure_stage,
    generate_payload_pool,
    parse_byte_size,
    quote_identifier,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    ("value", "expected"),
    [("4096", 4096), ("64KiB", 65536), ("2 MiB", 2 * 1024**2), ("1GB", 10**9)],
)
def test_parse_byte_size(value, expected):
    assert parse_byte_size(value) == expected


def test_config_rejects_payload_misaligned_to_dtype():
    with pytest.raises(ValueError, match="multiple of 2"):
        BenchmarkConfig(payload_bytes=3, dtype="float16")


def test_config_rejects_batch_larger_than_queue():
    with pytest.raises(ValueError, match="queue_capacity_bytes"):
        BenchmarkConfig(
            payload_bytes=16,
            min_batch_bytes=16,
            max_batch_bytes=32,
            queue_capacity_bytes=16,
        )


def test_config_rejects_normal_pattern_for_integer_dtype():
    with pytest.raises(ValueError, match="floating-point"):
        BenchmarkConfig(dtype="int8", pattern="normal")


@pytest.mark.parametrize("port", (0, 65536))
def test_config_rejects_invalid_native_tcp_port(port):
    with pytest.raises(ValueError, match="port"):
        BenchmarkConfig(port=port)


def test_config_uses_a_unique_benchmark_table_by_default():
    first = BenchmarkConfig().table
    second = BenchmarkConfig().table

    assert first.startswith("dmi_host_bench_")
    assert first != second


def test_payload_pool_is_fixed_size_contiguous_and_deterministic():
    cfg = BenchmarkConfig(
        rows=4,
        payload_bytes=64,
        dtype="float32",
        pattern="normal",
        pool_size=2,
        seed=7,
        min_batch_bytes=64,
        max_batch_bytes=128,
        queue_capacity_bytes=256,
    )

    first = generate_payload_pool(cfg)
    second = generate_payload_pool(cfg)

    assert len(first) == 2
    assert all(t.nbytes == 64 and t.is_contiguous() for t in first)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_stage_configuration_sets_batching_and_backpressure():
    queue = SimpleNamespace()
    policy = SimpleNamespace()
    stage = SimpleNamespace(input_queue=queue, ingress_policy=policy)
    cfg = BenchmarkConfig(
        rows=10,
        payload_bytes=16,
        min_batch_bytes=64,
        max_batch_bytes=128,
        queue_capacity_bytes=256,
        max_linger_ms=25,
        max_batch_items=50,
        queue_capacity_items=100,
    )

    configure_stage(stage, cfg)

    assert queue.min_batch_items is None
    assert queue.min_batch_size == 64
    assert queue.max_linger_s == 0.025
    assert queue.max_batch_items == 50
    assert queue.max_batch_size == 128
    assert queue.high_watermark_items == 100
    assert queue.high_watermark_size == 256
    assert policy.block is True


def test_async_insert_settings_wait_for_durability():
    assert build_client_settings(False) == {}
    assert build_client_settings(True) == {
        "async_insert": 1,
        "wait_for_async_insert": 1,
    }


def test_server_time_uses_clickhouse_clock():
    class Client:
        def execute(self, query):
            assert query == "SELECT now64(6)"
            return [(datetime(2026, 8, 24, 15, 0),)]

    assert _server_time(Client()) == datetime(2026, 8, 24, 15, 0)


def test_empty_query_log_uses_json_safe_null_aggregates():
    class Client:
        def execute(self, query, params=None):
            if query == "SYSTEM FLUSH LOGS":
                return []
            return [(0, 0, 0, float("nan"), float("nan"))]

    metrics, warning = _query_log_metrics(
        Client(),
        BenchmarkConfig(),
        datetime.now(),
    )

    assert warning is None
    assert metrics["average_duration_ms"] is None
    assert metrics["p95_duration_ms"] is None


@pytest.mark.parametrize(
    ("async_insert", "query_kind"),
    [(False, "Insert"), (True, "AsyncInsertFlush")],
)
def test_query_log_selects_one_insert_event_kind(async_insert, query_kind):
    class Client:
        def __init__(self):
            self.params = None

        def execute(self, query, params=None):
            if query == "SYSTEM FLUSH LOGS":
                return []
            self.params = params
            return [(1, 25, 1024, 2.0, 2.0)]

    client = Client()
    _query_log_metrics(
        client,
        BenchmarkConfig(async_insert=async_insert),
        datetime.now(),
    )

    assert client.params["query_kind"] == query_kind


def test_trial_measurement_reports_enqueue_and_drain_separately():
    trial = TrialMeasurement(
        rows=4,
        payload_bytes=1024,
        enqueue_seconds=2.0,
        total_seconds=5.0,
        enqueue_latencies_ns=(1_000_000, 2_000_000, 3_000_000, 4_000_000),
    )

    report = trial.as_dict()

    assert report["enqueue"]["rows_per_second"] == 2.0
    assert report["total"]["rows_per_second"] == 0.8
    assert report["drain_seconds"] == 3.0
    assert report["enqueue"]["latency_ms"]["p50"] == 2.5
    assert report["enqueue"]["latency_ms"]["p95"] == pytest.approx(3.85)
    assert report["enqueue"]["latency_samples"] == 4


def test_submit_and_drain_uses_graceful_lifecycle():
    class Engine:
        def __init__(self):
            self.calls = []

        def start(self):
            self.calls.append("start")

        def submit_direct(self, *args):
            self.calls.append(("submit", args))

        def close_input(self):
            self.calls.append("close_input")

        def join(self, timeout):
            self.calls.append(("join", timeout))
            return True

        def raise_if_failed(self):
            self.calls.append("raise_if_failed")

        def request_abort(self):
            self.calls.append("request_abort")

    ticks = iter((0, 10, 15, 20, 30, 40, 70))
    engine = Engine()
    payload = SimpleNamespace(nbytes=16)

    result = _submit_and_drain(
        engine,
        [payload],
        rows=2,
        model_id="run",
        timeout_seconds=5.0,
        clock=lambda: next(ticks),
    )

    assert [call[0] for call in engine.calls if isinstance(call, tuple)] == [
        "submit",
        "submit",
        "join",
    ]
    assert "close_input" in engine.calls
    assert "raise_if_failed" in engine.calls
    assert "request_abort" not in engine.calls
    assert result.enqueue_seconds == 40 / 1_000_000_000
    assert result.total_seconds == 70 / 1_000_000_000


def test_identifier_quoting_rejects_sql_fragments():
    assert quote_identifier("bench_2026") == "`bench_2026`"
    with pytest.raises(ValueError, match="identifier"):
        quote_identifier("bench; DROP TABLE x")
