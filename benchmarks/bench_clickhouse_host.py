"""Host-only benchmark for DMI's native ClickHouse sink."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import resource
import statistics
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Sequence


_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}
_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_byte_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]*)\s*", value)
    if match is None or match.group(2).lower() not in _BYTE_UNITS:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}")
    return int(match.group(1)) * _BYTE_UNITS[match.group(2).lower()]


def quote_identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid ClickHouse identifier: {value!r}")
    return f"`{value}`"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    rows: int = 10_000
    payload_bytes: int = 64 * 1024
    dtype: str = "float16"
    pattern: str = "normal"
    pool_size: int = 64
    seed: int = 17
    latency_samples: int = 10_000
    warmup_rows: int = 512
    parallelism: int = 4
    min_batch_bytes: int = 16 * 1024**2
    max_batch_bytes: int = 64 * 1024**2
    max_batch_items: int = 10_000
    max_linger_ms: float = 50.0
    queue_capacity_bytes: int = 512 * 1024**2
    queue_capacity_items: int = 20_000
    index_granularity: int = 8192
    compression: str = "lz4"
    async_insert: bool = False
    drain_timeout_seconds: float = 300.0
    socket_timeout_seconds: float = 30.0
    server_sample_interval_ms: int = 50
    host: str = "localhost"
    port: int = 9000
    user: str = "default"
    password: str = ""
    database: str = "default"
    table: str = field(
        default_factory=lambda: f"dmi_host_bench_{uuid.uuid4().hex[:12]}"
    )
    secure: bool = False
    create_database: bool = False
    keep_table: bool = False

    def __post_init__(self) -> None:
        positive = {
            "rows": self.rows,
            "payload_bytes": self.payload_bytes,
            "pool_size": self.pool_size,
            "latency_samples": self.latency_samples,
            "parallelism": self.parallelism,
            "min_batch_bytes": self.min_batch_bytes,
            "max_batch_bytes": self.max_batch_bytes,
            "max_batch_items": self.max_batch_items,
            "queue_capacity_bytes": self.queue_capacity_bytes,
            "queue_capacity_items": self.queue_capacity_items,
            "index_granularity": self.index_granularity,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_rows < 0:
            raise ValueError("warmup_rows must be non-negative")
        if not math.isfinite(self.max_linger_ms) or self.max_linger_ms < 0:
            raise ValueError("max_linger_ms must be a finite non-negative number")
        if not math.isfinite(self.drain_timeout_seconds) or self.drain_timeout_seconds <= 0:
            raise ValueError("drain_timeout_seconds must be a finite positive number")
        if not math.isfinite(self.socket_timeout_seconds) or self.socket_timeout_seconds <= 0:
            raise ValueError("socket_timeout_seconds must be a finite positive number")
        if self.server_sample_interval_ms < 0:
            raise ValueError("server_sample_interval_ms must be non-negative")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        element_size = _DTYPE_BYTES[self.dtype]
        if self.payload_bytes % element_size:
            raise ValueError(f"payload_bytes must be a multiple of {element_size} for {self.dtype}")
        if self.pattern not in {"zeros", "normal", "random"}:
            raise ValueError("pattern must be zeros, normal, or random")
        floating_dtype = self.dtype.startswith("float") or self.dtype == "bfloat16"
        if self.pattern == "normal" and not floating_dtype:
            raise ValueError("normal pattern requires a floating-point dtype")
        if self.compression not in {"none", "lz4", "zstd"}:
            raise ValueError("compression must be none, lz4, or zstd")
        if self.payload_bytes > self.max_batch_bytes:
            raise ValueError("payload_bytes must not exceed max_batch_bytes")
        if self.min_batch_bytes > self.max_batch_bytes:
            raise ValueError("min_batch_bytes must not exceed max_batch_bytes")
        if self.max_batch_bytes > self.queue_capacity_bytes:
            raise ValueError("max_batch_bytes must not exceed queue_capacity_bytes")
        if self.max_batch_items > self.queue_capacity_items:
            raise ValueError("max_batch_items must not exceed queue_capacity_items")
        quote_identifier(self.database)
        quote_identifier(self.table)


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True, slots=True)
class TrialMeasurement:
    rows: int
    payload_bytes: int
    enqueue_seconds: float
    total_seconds: float
    enqueue_latencies_ns: tuple[int, ...]
    process_cpu_seconds: float = 0.0
    process_lifetime_peak_rss_bytes: int = 0
    startup_seconds: float = 0.0
    client_metrics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        logical_bytes = self.rows * self.payload_bytes

        def rates(seconds: float) -> dict[str, float]:
            return {
                "seconds": seconds,
                "rows_per_second": self.rows / seconds,
                "logical_gib_per_second": logical_bytes / seconds / 1024**3,
            }

        enqueue = rates(self.enqueue_seconds)
        enqueue["latency_ms"] = {
            name: _percentile(self.enqueue_latencies_ns, q) / 1_000_000
            for name, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
        }
        enqueue["latency_samples"] = len(self.enqueue_latencies_ns)
        return {
            "rows": self.rows,
            "logical_payload_bytes": logical_bytes,
            "startup_seconds": self.startup_seconds,
            "enqueue": enqueue,
            "drain_seconds": max(0.0, self.total_seconds - self.enqueue_seconds),
            "total": rates(self.total_seconds),
            "host_process": {
                "cpu_seconds": self.process_cpu_seconds,
                "effective_cpu_cores": self.process_cpu_seconds / self.total_seconds,
                "process_lifetime_peak_rss_bytes": self.process_lifetime_peak_rss_bytes,
            },
            "client": self.client_metrics,
        }


def build_client_settings(async_insert: bool) -> dict[str, int]:
    if not async_insert:
        return {"async_insert": 0}
    return {"async_insert": 1, "wait_for_async_insert": 1}


def parse_parallelism_sweep(value: str) -> tuple[int, ...]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("parallelism values must be integers") from exc
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("parallelism values must be positive")
    return tuple(dict.fromkeys(parsed))


def _client_metrics_as_dict(metrics: Any) -> dict[str, Any]:
    workers = [
        {
            "worker_index": int(worker.worker_index),
            "batches": int(worker.batches),
            "rows": int(worker.rows),
            "logical_bytes": int(worker.logical_bytes),
            "insert_seconds": float(worker.insert_seconds),
        }
        for worker in metrics.workers
    ]
    return {
        "expected_workers": int(metrics.expected_workers),
        "ready_workers": int(metrics.ready_workers),
        "active_inserts": int(metrics.active_inserts),
        "peak_active_inserts": int(metrics.peak_active_inserts),
        "batches": int(metrics.batches),
        "rows": int(metrics.rows),
        "logical_bytes": int(metrics.logical_bytes),
        "insert_seconds": float(metrics.insert_seconds),
        "workers": workers,
    }


class ServerTelemetrySampler:
    _METRICS_QUERY = """
        SELECT metric, value
        FROM system.metrics
        WHERE metric IN ('Query', 'Merge', 'BackgroundMergesAndMutationsPoolTask',
                         'TCPConnection')
    """
    _ASYNC_METRICS_QUERY = """
        SELECT metric,
               if(isFinite(value), value, arraySum(mapValues(key_values))) AS scalar_value
        FROM system.asynchronous_metrics
        WHERE metric IN ('OSUserTimeNormalized', 'OSSystemTimeNormalized',
                         'OSIOWaitTimeNormalized', 'MemoryResident', 'LoadAverage1',
                         'MaxPartCountForPartition', 'LongestRunningMerge',
                         'BlockReadBytes', 'BlockWriteBytes',
                         'NetworkReceiveBytes', 'NetworkSendBytes')
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        interval_ms: int = 50,
        database: str | None = None,
        table: str | None = None,
    ):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self._client_factory = client_factory
        self._interval_seconds = interval_ms / 1000.0
        if database is not None and table is not None:
            qualified = f"{quote_identifier(database)}.{quote_identifier(table)}"
            self._process_query = (
                "SELECT count() FROM system.processes"
                " WHERE query_kind = 'Insert'"
                " AND current_database = %(database)s"
                " AND position(query, %(qualified_table)s) > 0"
            )
            self._process_query_params: dict[str, Any] = {
                "database": database,
                "qualified_table": qualified,
            }
        else:
            self._process_query = (
                "SELECT count() FROM system.processes WHERE query_kind = 'Insert'"
            )
            self._process_query_params = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._lock = threading.Lock()
        self._samples = 0
        self._peak_active_inserts = 0
        self._metric_values: dict[str, list[float]] = {}
        self._errors: list[str] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_once()
        if self._stop.is_set():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="dmi-clickhouse-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if not self._errors:
            self.sample_once()
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self._interval_seconds)

    def sample_once(self) -> None:
        try:
            if self._client is None:
                self._client = self._client_factory()
            if self._process_query_params:
                active_result = self._client.execute(
                    self._process_query, self._process_query_params
                )
            else:
                active_result = self._client.execute(self._process_query)
            active = int(active_result[0][0])
            metrics = self._client.execute(self._METRICS_QUERY)
            async_metrics = self._client.execute(self._ASYNC_METRICS_QUERY)
            with self._lock:
                self._samples += 1
                self._peak_active_inserts = max(self._peak_active_inserts, active)
                for name, value in metrics:
                    self._metric_values.setdefault(str(name), []).append(float(value))
                for name, value in async_metrics:
                    key = f"async.{name}"
                    self._metric_values.setdefault(key, []).append(float(value))
        except Exception as exc:
            with self._lock:
                if not self._errors:
                    self._errors.append(str(exc))
            self._stop.set()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "samples": self._samples,
                "peak_active_inserts": self._peak_active_inserts,
                "metric_peaks": {
                    name: max(values) for name, values in self._metric_values.items()
                },
                "metric_means": {
                    name: statistics.fmean(values)
                    for name, values in self._metric_values.items()
                },
                "metric_deltas": {
                    name: values[-1] - values[0]
                    for name, values in self._metric_values.items()
                },
                "errors": list(self._errors),
            }


def generate_payload_pool(config: BenchmarkConfig) -> list[Any]:
    import torch

    dtype = getattr(torch, config.dtype)
    elements = config.payload_bytes // _DTYPE_BYTES[config.dtype]
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    payloads = []
    for _ in range(config.pool_size):
        if config.pattern == "zeros":
            tensor = torch.zeros(elements, dtype=dtype)
        elif config.pattern == "normal":
            if not dtype.is_floating_point:
                raise ValueError("normal pattern requires a floating-point dtype")
            tensor = torch.randn(elements, dtype=dtype, generator=generator)
        elif dtype.is_floating_point:
            tensor = torch.rand(elements, dtype=dtype, generator=generator)
        else:
            tensor = torch.randint(0, 127, (elements,), dtype=dtype, generator=generator)
        payloads.append(tensor.contiguous())
    return payloads


def configure_stage(stage: Any, config: BenchmarkConfig) -> None:
    queue = stage.input_queue
    queue.min_batch_items = None
    queue.min_batch_size = config.min_batch_bytes
    queue.max_linger_s = config.max_linger_ms / 1000.0
    queue.max_batch_items = config.max_batch_items
    queue.max_batch_size = config.max_batch_bytes
    queue.high_watermark_items = config.queue_capacity_items
    queue.high_watermark_size = config.queue_capacity_bytes
    stage.ingress_policy.block = True
    stage.ingress_policy.timeout_s = min(
        config.socket_timeout_seconds,
        config.drain_timeout_seconds,
    )


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _build_engine(config: BenchmarkConfig) -> Any:
    from dmi.transport.native import (
        ClickHouseClientConfig,
        DMXHostEngine,
        StageConfig,
    )

    clickhouse = ClickHouseClientConfig()
    clickhouse.host = config.host
    clickhouse.port = config.port
    clickhouse.username = config.user
    clickhouse.password = config.password
    clickhouse.database = config.database
    clickhouse.table = config.table
    clickhouse.secure = config.secure
    clickhouse.create_database_if_missing = config.create_database
    clickhouse.drop_existing_database = False
    clickhouse.client_side_compress = config.compression
    clickhouse.client_settings = build_client_settings(config.async_insert)
    clickhouse.index_granularity = config.index_granularity
    timeout_ms = round(config.socket_timeout_seconds * 1000)
    clickhouse.connect_timeout_ms = min(5000, timeout_ms)
    clickhouse.send_timeout_ms = timeout_ms
    clickhouse.receive_timeout_ms = timeout_ms

    stage = StageConfig.clickhouse_insert(
        clickhouse,
        parallelism=config.parallelism,
        name="clickhouse_insert",
    )
    configure_stage(stage, config)
    return DMXHostEngine(stage)


def _submit_and_drain(
    engine: Any,
    payloads: Sequence[Any],
    rows: int,
    model_id: str,
    timeout_seconds: float,
    max_latency_samples: int = 10_000,
    clock: Callable[[], int] = time.perf_counter_ns,
    deadline_clock: Callable[[], float] = time.monotonic,
    sampler: ServerTelemetrySampler | None = None,
) -> TrialMeasurement:
    startup_start = clock()
    sampler_started = False
    deadline = deadline_clock() + timeout_seconds
    try:
        engine.start()
        if not engine.wait_until_ready(timeout_seconds):
            engine.raise_if_failed()
            raise TimeoutError(f"ClickHouse workers were not ready after {timeout_seconds:.1f}s")
        startup_end = clock()
        deadline = deadline_clock() + timeout_seconds
        cpu_start = time.process_time()
        if sampler is not None:
            sampler.start()
            sampler_started = True
        total_start = clock()
        latencies = []
        sample_stride = max(1, math.ceil(rows / max_latency_samples))
        for row in range(rows):
            if deadline_clock() >= deadline:
                raise TimeoutError(f"ClickHouse benchmark exceeded {timeout_seconds:.1f}s")
            started = clock() if row % sample_stride == 0 else None
            engine.submit_direct(
                model_id,
                0,
                f"{row:012d}",
                "synthetic_activation",
                0,
                0,
                1,
                payloads[row % len(payloads)],
            )
            if started is not None:
                latencies.append(clock() - started)
        enqueue_end = clock()
        engine.close_input()
        remaining = max(0.0, deadline - deadline_clock())
        if not engine.join(remaining):
            raise TimeoutError(f"ClickHouse drain exceeded {timeout_seconds:.1f}s")
        engine.raise_if_failed()
        total_end = clock()
        if sampler is not None:
            sampler.stop()
            sampler_started = False
        cpu_end = time.process_time()
        process_lifetime_peak_rss = _peak_rss_bytes()
        client_metrics = _client_metrics_as_dict(engine.clickhouse_metrics())
    except BaseException:
        engine.request_abort()
        remaining = max(0.0, deadline - deadline_clock())
        try:
            engine.join(remaining)
        except BaseException:
            pass
        raise
    finally:
        if sampler_started:
            sampler.stop()

    return TrialMeasurement(
        rows=rows,
        payload_bytes=int(payloads[0].nbytes),
        enqueue_seconds=(enqueue_end - total_start) / 1_000_000_000,
        total_seconds=(total_end - total_start) / 1_000_000_000,
        enqueue_latencies_ns=tuple(latencies),
        process_cpu_seconds=cpu_end - cpu_start,
        process_lifetime_peak_rss_bytes=process_lifetime_peak_rss,
        startup_seconds=(startup_end - startup_start) / 1_000_000_000,
        client_metrics=client_metrics,
    )


def _connect(config: BenchmarkConfig, timeout_seconds: float | None = None) -> Any:
    try:
        from clickhouse_driver import Client
    except ImportError as exc:
        raise RuntimeError("clickhouse-driver is required for benchmark verification") from exc

    socket_timeout = timeout_seconds or config.socket_timeout_seconds
    connection = {
        "host": config.host,
        "port": config.port,
        "user": config.user,
        "password": config.password,
        "secure": config.secure,
        "connect_timeout": min(5.0, socket_timeout),
        "send_receive_timeout": socket_timeout,
    }
    if config.create_database:
        bootstrap = Client(database="default", **connection)
        bootstrap.execute(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(config.database)}")
        bootstrap.disconnect()
    return Client(database=config.database, **connection)


def _verification(client: Any, config: BenchmarkConfig, model_id: str) -> dict[str, int]:
    table = f"{quote_identifier(config.database)}.{quote_identifier(config.table)}"
    rows, payload_bytes = client.execute(
        f"SELECT count(), sum(length(bytes)) FROM {table} WHERE model_id = %(model_id)s",
        {"model_id": model_id},
    )[0]
    expected_bytes = config.rows * config.payload_bytes
    result = {
        "rows": int(rows),
        "payload_bytes": int(payload_bytes or 0),
        "expected_rows": config.rows,
        "expected_payload_bytes": expected_bytes,
    }
    if result["rows"] != config.rows or result["payload_bytes"] != expected_bytes:
        raise RuntimeError(f"ClickHouse verification failed: {result}")
    return result


def _parts_metrics(client: Any, config: BenchmarkConfig) -> dict[str, int]:
    values = client.execute(
        """
        SELECT count(), sum(rows), sum(bytes_on_disk), sum(data_compressed_bytes),
               sum(data_uncompressed_bytes), sum(primary_key_size)
        FROM system.parts
        WHERE active AND database = %(database)s AND table = %(table)s
        """,
        {"database": config.database, "table": config.table},
    )[0]
    names = (
        "active_parts",
        "part_rows",
        "bytes_on_disk",
        "data_compressed_bytes",
        "data_uncompressed_bytes",
        "primary_key_bytes",
    )
    return {name: int(value or 0) for name, value in zip(names, values)}


def _server_time(client: Any) -> datetime:
    return client.execute("SELECT now64(6)")[0][0]


def _query_log_metrics(
    client: Any,
    config: BenchmarkConfig,
    started_at: datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        client.execute("SYSTEM FLUSH LOGS")
        values = client.execute(
            """
            SELECT count(), sum(written_rows), sum(written_bytes),
                   avg(query_duration_ms), quantileExact(0.95)(query_duration_ms)
            FROM system.query_log
            WHERE event_time_microseconds >= toDateTime64(%(started_at)s, 6)
              AND type = 'QueryFinish'
              AND query_kind = %(query_kind)s
              AND written_rows > 0
              AND has(tables, %(table)s)
            """,
            {
                "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "query_kind": "AsyncInsertFlush" if config.async_insert else "Insert",
                "table": f"{config.database}.{config.table}",
            },
        )[0]
        insert_queries = int(values[0] or 0)
        written_rows = int(values[1] or 0)
        written_bytes = int(values[2] or 0)

        def optional_float(value: Any) -> float | None:
            number = float(value)
            return number if math.isfinite(number) else None

        result = {
            "insert_queries": insert_queries,
            "written_rows": written_rows,
            "written_bytes": written_bytes,
            "average_duration_ms": optional_float(values[3]) if insert_queries else None,
            "p95_duration_ms": optional_float(values[4]) if insert_queries else None,
        }
        result["average_rows_per_insert"] = (
            written_rows / insert_queries if insert_queries else 0.0
        )
        result["average_bytes_per_insert"] = (
            written_bytes / insert_queries if insert_queries else 0.0
        )
        return result, None
    except Exception as exc:
        return None, f"query_log metrics unavailable: {exc}"


def _safe_config(config: BenchmarkConfig) -> dict[str, Any]:
    values = asdict(config)
    values["password"] = "***" if config.password else ""
    return values


def _ensure_table(client: Any, config: BenchmarkConfig) -> None:
    """Create the trial table before native workers enter the timed path."""
    table = f"{quote_identifier(config.database)}.{quote_identifier(config.table)}"
    cols = ", ".join([
        f"{quote_identifier('model_id')} String",
        f"{quote_identifier('request_id')} String",
        f"{quote_identifier('act_name')} String",
        f"{quote_identifier('layer_no')} Int32",
        f"{quote_identifier('shard_rank')} Int32",
        f"{quote_identifier('start_token_idx')} Int32",
        f"{quote_identifier('end_token_idx')} Int32",
        f"{quote_identifier('dtype')} String",
        f"{quote_identifier('shape')} Array(Int64)",
        f"{quote_identifier('bytes')} String",
    ])
    pk_cols = ", ".join(
        quote_identifier(c)
        for c in ("model_id", "request_id", "act_name", "layer_no",
                  "shard_rank", "start_token_idx", "end_token_idx")
    )
    client.execute(
        f"CREATE TABLE IF NOT EXISTS {table} ({cols})"
        f" ENGINE = MergeTree"
        f" PRIMARY KEY ({pk_cols})"
        f" ORDER BY ({pk_cols})"
        f" SETTINGS index_granularity = {config.index_granularity}"
    )


def run(config: BenchmarkConfig) -> dict[str, Any]:
    payloads = generate_payload_pool(config)
    client = _connect(config)
    table = f"{quote_identifier(config.database)}.{quote_identifier(config.table)}"
    measured_model_id = f"host-bench-{uuid.uuid4().hex}"
    warnings: list[str] = []
    try:
        _ensure_table(client, config)
        server_version = str(client.execute("SELECT version()")[0][0])
        if config.warmup_rows:
            warmup = _build_engine(config)
            _submit_and_drain(
                warmup,
                payloads,
                config.warmup_rows,
                f"warmup-{uuid.uuid4().hex}",
                config.drain_timeout_seconds,
                config.latency_samples,
            )
            client.execute(f"TRUNCATE TABLE {table}")

        started_at = _server_time(client)
        sampler = None
        if config.server_sample_interval_ms:
            sampler = ServerTelemetrySampler(
                lambda: _connect(config, min(2.0, config.socket_timeout_seconds)),
                interval_ms=config.server_sample_interval_ms,
                database=config.database,
                table=config.table,
            )
        measurement = _submit_and_drain(
            _build_engine(config),
            payloads,
            config.rows,
            measured_model_id,
            config.drain_timeout_seconds,
            config.latency_samples,
            sampler=sampler,
        )
        server_telemetry = sampler.snapshot() if sampler is not None else None
        if server_telemetry and server_telemetry["errors"]:
            warnings.append(f"server telemetry unavailable: {server_telemetry['errors'][0]}")
        verification = _verification(client, config, measured_model_id)
        parts = _parts_metrics(client, config)
        query_log, warning = _query_log_metrics(client, config, started_at)
        if warning:
            warnings.append(warning)
        return {
            "benchmark": "dmi_clickhouse_host",
            "server_version": server_version,
            "config": _safe_config(config),
            "effective_client_settings": build_client_settings(config.async_insert),
            "measurement": measurement.as_dict(),
            "verification": verification,
            "parts": parts,
            "query_log": query_log,
            "server_telemetry": server_telemetry,
            "warnings": warnings,
        }
    finally:
        if not config.keep_table:
            try:
                client.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception as exc:
                warnings.append(f"table cleanup failed: {exc}")
        client.disconnect()


def summarize_scaling_trials(
    trials: Sequence[dict[str, Any]],
    plateau_threshold_percent: float = 5.0,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[float]] = {}
    for trial in trials:
        grouped.setdefault(int(trial["parallelism"]), []).append(float(trial["throughput"]))
    if not grouped:
        return []

    parallelisms = sorted(grouped)
    medians = {value: statistics.median(grouped[value]) for value in parallelisms}
    baseline_parallelism = 1 if 1 in medians else parallelisms[0]
    baseline = medians[baseline_parallelism]
    summary = []
    previous = None
    for parallelism in parallelisms:
        values = grouped[parallelism]
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        median = medians[parallelism]
        gain = None if previous is None else (median / previous - 1.0) * 100.0
        summary.append({
            "parallelism": parallelism,
            "trials": len(values),
            "median_gib_per_second": median,
            "mean_gib_per_second": mean,
            "stdev_gib_per_second": stdev,
            "coefficient_of_variation_percent": stdev / mean * 100.0 if mean else None,
            "min_gib_per_second": min(values),
            "max_gib_per_second": max(values),
            "baseline_parallelism": baseline_parallelism,
            "speedup_vs_baseline": median / baseline if baseline else None,
            "speedup_vs_one": median / baseline if baseline and baseline_parallelism == 1 else None,
            "gain_vs_previous_percent": gain,
            "plateau": None if gain is None else gain < plateau_threshold_percent,
        })
        previous = median
    return summary


def run_sweep(
    config: BenchmarkConfig,
    parallelisms: Sequence[int],
    trials: int,
    *,
    plateau_threshold_percent: float = 5.0,
    runner: Callable[[BenchmarkConfig], dict[str, Any]] = run,
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    values = tuple(dict.fromkeys(int(value) for value in parallelisms))
    if not values or any(value <= 0 for value in values):
        raise ValueError("parallelism values must be positive")

    schedule = [(parallelism, trial) for trial in range(trials) for parallelism in values]
    random.Random(config.seed).shuffle(schedule)
    raw = []
    for run_index, (parallelism, trial_index) in enumerate(schedule):
        table = f"{config.table}_p{parallelism}_t{trial_index}_{uuid.uuid4().hex[:6]}"
        trial_config = replace(config, parallelism=parallelism, table=table)
        report = runner(trial_config)
        throughput = float(report["measurement"]["total"]["logical_gib_per_second"])
        raw.append({
            "run_index": run_index,
            "trial_index": trial_index,
            "parallelism": parallelism,
            "throughput": throughput,
            "report": report,
        })

    return {
        "benchmark": "dmi_clickhouse_host_scaling",
        "base_config": _safe_config(config),
        "parallelisms": list(values),
        "trials_per_parallelism": trials,
        "plateau_threshold_percent": plateau_threshold_percent,
        "trials": raw,
        "summary": summarize_scaling_trials(raw, plateau_threshold_percent),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--payload-bytes", type=parse_byte_size, default=64 * 1024)
    parser.add_argument("--dtype", choices=tuple(_DTYPE_BYTES), default="float16")
    parser.add_argument("--pattern", choices=("zeros", "normal", "random"), default="normal")
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--latency-samples", type=int, default=10_000)
    parser.add_argument("--warmup-rows", type=int, default=512)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--parallelism-sweep")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--plateau-threshold-percent", type=float, default=5.0)
    parser.add_argument("--min-batch-bytes", type=parse_byte_size, default=16 * 1024**2)
    parser.add_argument("--max-batch-bytes", type=parse_byte_size, default=64 * 1024**2)
    parser.add_argument("--max-batch-items", type=int, default=10_000)
    parser.add_argument("--max-linger-ms", type=float, default=50.0)
    parser.add_argument("--queue-capacity-bytes", type=parse_byte_size, default=512 * 1024**2)
    parser.add_argument("--queue-capacity-items", type=int, default=20_000)
    parser.add_argument("--index-granularity", type=int, default=8192)
    parser.add_argument("--compression", choices=("none", "lz4", "zstd"), default="lz4")
    parser.add_argument("--async-insert", action="store_true")
    parser.add_argument("--drain-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--socket-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--server-sample-interval-ms", type=int, default=50)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--user", default="default")
    parser.add_argument("--password", default=os.environ.get("DMI_CLICKHOUSE_PASSWORD", ""))
    parser.add_argument("--database", default="default")
    parser.add_argument("--table-prefix", default="dmi_host_bench")
    parser.add_argument("--secure", action="store_true")
    parser.add_argument("--create-database", action="store_true")
    parser.add_argument("--keep-table", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    quote_identifier(args.table_prefix)
    table = f"{args.table_prefix}_{uuid.uuid4().hex[:12]}"
    return BenchmarkConfig(
        rows=args.rows,
        payload_bytes=args.payload_bytes,
        dtype=args.dtype,
        pattern=args.pattern,
        pool_size=args.pool_size,
        seed=args.seed,
        latency_samples=args.latency_samples,
        warmup_rows=args.warmup_rows,
        parallelism=args.parallelism,
        min_batch_bytes=args.min_batch_bytes,
        max_batch_bytes=args.max_batch_bytes,
        max_batch_items=args.max_batch_items,
        max_linger_ms=args.max_linger_ms,
        queue_capacity_bytes=args.queue_capacity_bytes,
        queue_capacity_items=args.queue_capacity_items,
        index_granularity=args.index_granularity,
        compression=args.compression,
        async_insert=args.async_insert,
        drain_timeout_seconds=args.drain_timeout_seconds,
        socket_timeout_seconds=args.socket_timeout_seconds,
        server_sample_interval_ms=args.server_sample_interval_ms,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        table=table,
        secure=args.secure,
        create_database=args.create_database,
        keep_table=args.keep_table,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    parallelisms = (
        parse_parallelism_sweep(args.parallelism_sweep)
        if args.parallelism_sweep
        else (config.parallelism,)
    )
    if len(parallelisms) == 1:
        config = replace(config, parallelism=parallelisms[0])
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    if args.dry_run:
        report: dict[str, Any] = {
            "benchmark": "dmi_clickhouse_host",
            "dry_run": True,
            "config": _safe_config(config),
            "parallelisms": list(parallelisms),
            "trials_per_parallelism": args.trials,
            "logical_payload_bytes": config.rows * config.payload_bytes,
            "payload_pool_bytes": config.pool_size * config.payload_bytes,
        }
    elif len(parallelisms) > 1 or args.trials > 1:
        report = run_sweep(
            config,
            parallelisms,
            args.trials,
            plateau_threshold_percent=args.plateau_threshold_percent,
        )
    else:
        report = run(config)
    output = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(output)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as destination:
            destination.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
