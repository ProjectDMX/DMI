from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
import torch

import benchmarks.bench_clickhouse_host as benchmark
from benchmarks.bench_clickhouse_host import (
    BenchmarkConfig,
    ServerTelemetrySampler,
    TrialMeasurement,
    _ensure_table,
    _parts_metrics,
    _server_time,
    _query_log_metrics,
    _submit_and_drain,
    build_client_settings,
    configure_stage,
    generate_payload_pool,
    parse_parallelism_sweep,
    parse_byte_size,
    quote_identifier,
    run_sweep,
    summarize_scaling_trials,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_linger_ms", float("nan")),
        ("max_linger_ms", float("inf")),
        ("drain_timeout_seconds", float("nan")),
        ("drain_timeout_seconds", float("inf")),
    ],
)
def test_config_rejects_non_finite_timings(field, value):
    with pytest.raises(ValueError, match="finite"):
        BenchmarkConfig(**{field: value})


def test_config_rejects_non_positive_index_granularity():
    with pytest.raises(ValueError, match="index_granularity"):
        BenchmarkConfig(index_granularity=0)


def test_config_rejects_invalid_socket_and_sampling_timeouts():
    with pytest.raises(ValueError, match="socket_timeout_seconds"):
        BenchmarkConfig(socket_timeout_seconds=0)
    with pytest.raises(ValueError, match="server_sample_interval_ms"):
        BenchmarkConfig(server_sample_interval_ms=-1)


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
    assert policy.timeout_s == cfg.socket_timeout_seconds


def test_async_insert_settings_wait_for_durability():
    assert build_client_settings(False) == {"async_insert": 0}
    assert build_client_settings(True) == {
        "async_insert": 1,
        "wait_for_async_insert": 1,
    }


def test_single_value_sweep_uses_requested_parallelism(monkeypatch, capsys):
    seen = []

    def runner(config):
        seen.append(config.parallelism)
        return {"parallelism": config.parallelism}

    monkeypatch.setattr(benchmark, "run", runner)

    assert benchmark.main(["--parallelism-sweep", "8"]) == 0
    assert seen == [8]
    assert '"parallelism": 8' in capsys.readouterr().out


def test_server_time_uses_clickhouse_clock():
    class Client:
        def execute(self, query):
            assert query == "SELECT now64(6)"
            return [(datetime(2026, 8, 24, 15, 0),)]

    assert _server_time(Client()) == datetime(2026, 8, 24, 15, 0)


def test_table_setup_uses_configured_index_granularity():
    class Client:
        query = None

        def execute(self, query):
            self.query = query

    client = Client()
    _ensure_table(client, BenchmarkConfig(index_granularity=4096))

    assert "SETTINGS index_granularity = 4096" in client.query


def test_parts_metrics_report_on_disk_primary_key_size():
    class Client:
        query = None

        def execute(self, query, params):
            self.query = query
            return [(1, 2, 3, 4, 5, 6)]

    client = Client()
    metrics = _parts_metrics(client, BenchmarkConfig())

    assert "primary_key_size" in client.query
    assert "primary_key_bytes_in_memory" not in client.query
    assert metrics["primary_key_bytes"] == 6


def test_run_disconnects_client_when_table_setup_fails(monkeypatch):
    class Client:
        disconnected = False

        def execute(self, query):
            return []

        def disconnect(self):
            self.disconnected = True

    client = Client()
    monkeypatch.setattr(benchmark, "generate_payload_pool", lambda config: [])
    monkeypatch.setattr(benchmark, "_connect", lambda config: client)

    def fail_setup(client, config):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(benchmark, "_ensure_table", fail_setup)

    with pytest.raises(RuntimeError, match="setup failed"):
        benchmark.run(BenchmarkConfig())

    assert client.disconnected is True


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


def test_query_log_preserves_measurement_start_microseconds():
    class Client:
        def __init__(self):
            self.query = None
            self.params = None

        def execute(self, query, params=None):
            if query == "SYSTEM FLUSH LOGS":
                return []
            self.query = query
            self.params = params
            return [(0, 0, 0, float("nan"), float("nan"))]

    client = Client()
    _query_log_metrics(
        client,
        BenchmarkConfig(),
        datetime(2026, 8, 24, 15, 53, 50, 408123),
    )

    assert "toDateTime64(%(started_at)s, 6)" in client.query
    assert client.params["started_at"] == "2026-08-24 15:53:50.408123"


def test_trial_measurement_reports_enqueue_and_drain_separately():
    trial = TrialMeasurement(
        rows=4,
        payload_bytes=1024,
        enqueue_seconds=2.0,
        total_seconds=5.0,
        enqueue_latencies_ns=(1_000_000, 2_000_000, 3_000_000, 4_000_000),
        startup_seconds=0.25,
        client_metrics={"peak_active_inserts": 3},
    )

    report = trial.as_dict()

    assert report["enqueue"]["rows_per_second"] == 2.0
    assert report["total"]["rows_per_second"] == 0.8
    assert report["drain_seconds"] == 3.0
    assert report["enqueue"]["latency_ms"]["p50"] == 2.5
    assert report["enqueue"]["latency_ms"]["p95"] == pytest.approx(3.85)
    assert report["enqueue"]["latency_samples"] == 4
    assert report["startup_seconds"] == 0.25
    assert report["client"]["peak_active_inserts"] == 3


def test_submit_and_drain_uses_graceful_lifecycle():
    class Engine:
        def __init__(self):
            self.calls = []

        def start(self):
            self.calls.append("start")

        def wait_until_ready(self, timeout):
            self.calls.append(("wait_until_ready", timeout))
            return True

        def submit_direct(self, *args):
            self.calls.append(("submit", args))

        def close_input(self):
            self.calls.append("close_input")

        def join(self, timeout):
            self.calls.append(("join", timeout))
            return True

        def raise_if_failed(self):
            self.calls.append("raise_if_failed")

        def clickhouse_metrics(self):
            return SimpleNamespace(
                expected_workers=2,
                ready_workers=2,
                active_inserts=0,
                peak_active_inserts=2,
                batches=2,
                rows=2,
                logical_bytes=32,
                insert_seconds=0.5,
                workers=[],
            )

        def request_abort(self):
            self.calls.append("request_abort")

    class Sampler:
        def start(self):
            engine.calls.append("sampler_start")

        def stop(self):
            engine.calls.append("sampler_stop")

    ticks = iter((0, 10, 20, 30, 40, 50, 60, 70, 90))
    engine = Engine()
    payload = SimpleNamespace(nbytes=16)

    result = _submit_and_drain(
        engine,
        [payload],
        rows=2,
        model_id="run",
        timeout_seconds=5.0,
        clock=lambda: next(ticks),
        deadline_clock=lambda: 0.0,
        sampler=Sampler(),
    )

    assert [call[0] for call in engine.calls if isinstance(call, tuple)] == [
        "wait_until_ready",
        "submit",
        "submit",
        "join",
    ]
    assert "close_input" in engine.calls
    assert "raise_if_failed" in engine.calls
    assert "request_abort" not in engine.calls
    assert engine.calls.index("sampler_start") > engine.calls.index(("wait_until_ready", 5.0))
    assert engine.calls.index("sampler_stop") > engine.calls.index(("join", 5.0))
    assert result.startup_seconds == 10 / 1_000_000_000
    assert result.enqueue_seconds == 50 / 1_000_000_000
    assert result.total_seconds == 70 / 1_000_000_000
    assert result.client_metrics["peak_active_inserts"] == 2


def test_submit_and_drain_does_not_restart_timeout_during_abort():
    class Engine:
        def __init__(self):
            self.join_timeouts = []

        def start(self):
            pass

        def wait_until_ready(self, timeout):
            return True

        def submit_direct(self, *args):
            pass

        def close_input(self):
            pass

        def join(self, timeout):
            self.join_timeouts.append(timeout)
            return False

        def request_abort(self):
            pass

    deadlines = iter((0.0, 0.0, 0.0, 0.0, 5.0))
    engine = Engine()

    with pytest.raises(TimeoutError, match="drain exceeded"):
        _submit_and_drain(
            engine,
            [SimpleNamespace(nbytes=16)],
            rows=1,
            model_id="run",
            timeout_seconds=5.0,
            clock=iter(range(20)).__next__,
            deadline_clock=lambda: next(deadlines),
        )

    assert engine.join_timeouts == [5.0, 0.0]


def test_parallelism_sweep_is_unique_and_positive():
    assert parse_parallelism_sweep("1,2,4,2,8") == (1, 2, 4, 8)
    with pytest.raises(ValueError, match="positive"):
        parse_parallelism_sweep("1,0,4")


def test_scaling_summary_reports_speedup_and_variance():
    trials = [
        {"parallelism": 1, "throughput": 1.0},
        {"parallelism": 1, "throughput": 1.2},
        {"parallelism": 2, "throughput": 1.8},
        {"parallelism": 2, "throughput": 2.0},
    ]

    summary = summarize_scaling_trials(trials, plateau_threshold_percent=5.0)

    assert summary[0]["parallelism"] == 1
    assert summary[0]["median_gib_per_second"] == pytest.approx(1.1)
    assert summary[1]["speedup_vs_one"] == pytest.approx(1.9 / 1.1)
    assert summary[1]["gain_vs_previous_percent"] == pytest.approx((1.9 / 1.1 - 1) * 100)
    assert summary[1]["plateau"] is False


def test_scaling_summary_does_not_invent_one_worker_baseline():
    summary = summarize_scaling_trials(
        [
            {"parallelism": 2, "throughput": 2.0},
            {"parallelism": 4, "throughput": 3.0},
        ]
    )

    assert summary[0]["baseline_parallelism"] == 2
    assert summary[0]["speedup_vs_one"] is None
    assert summary[1]["speedup_vs_baseline"] == pytest.approx(1.5)


def test_run_sweep_keeps_raw_trials_and_uses_unique_tables():
    calls = []

    def runner(config):
        calls.append(config)
        return {
            "measurement": {
                "total": {"logical_gib_per_second": float(config.parallelism)}
            }
        }

    report = run_sweep(
        BenchmarkConfig(rows=1, payload_bytes=16, min_batch_bytes=16,
                        max_batch_bytes=16, queue_capacity_bytes=32,
                        warmup_rows=0),
        parallelisms=(1, 2),
        trials=2,
        runner=runner,
    )

    assert len(report["trials"]) == 4
    assert {trial["parallelism"] for trial in report["trials"]} == {1, 2}
    assert len({config.table for config in calls}) == 4
    assert report["summary"][1]["speedup_vs_one"] == 2.0


def test_server_sampler_tracks_insert_concurrency_and_metric_peaks():
    class Client:
        def __init__(self):
            self.calls = 0

        def execute(self, query, params=None):
            self.calls += 1
            if "system.processes" in query:
                return [(2,)]
            if "system.asynchronous_metrics" in query:
                return [("OSUserTimeNormalized", 0.5)]
            return [("Merge", 3), ("Query", 5)]

        def disconnect(self):
            pass

    sampler = ServerTelemetrySampler(lambda: Client(), interval_ms=1)
    sampler.sample_once()
    snapshot = sampler.snapshot()

    assert snapshot["samples"] == 1
    assert snapshot["peak_active_inserts"] == 2
    assert snapshot["metric_peaks"] == {
        "Merge": 3.0,
        "Query": 5.0,
        "async.OSUserTimeNormalized": 0.5,
    }


def test_identifier_quoting_rejects_sql_fragments():
    assert quote_identifier("bench_2026") == "`bench_2026`"
    with pytest.raises(ValueError, match="identifier"):
        quote_identifier("bench; DROP TABLE x")
