"""Host-only benchmark for DMI's native ClickHouse sink."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
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
    compression: str = "lz4"
    async_insert: bool = False
    drain_timeout_seconds: float = 300.0
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
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_rows < 0:
            raise ValueError("warmup_rows must be non-negative")
        if self.max_linger_ms < 0:
            raise ValueError("max_linger_ms must be non-negative")
        if self.drain_timeout_seconds <= 0:
            raise ValueError("drain_timeout_seconds must be positive")
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
    peak_rss_before_bytes: int = 0
    peak_rss_after_bytes: int = 0

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
            "enqueue": enqueue,
            "drain_seconds": max(0.0, self.total_seconds - self.enqueue_seconds),
            "total": rates(self.total_seconds),
            "host_process": {
                "cpu_seconds": self.process_cpu_seconds,
                "effective_cpu_cores": self.process_cpu_seconds / self.total_seconds,
                "peak_rss_bytes": self.peak_rss_after_bytes,
                "peak_rss_growth_bytes": max(
                    0, self.peak_rss_after_bytes - self.peak_rss_before_bytes
                ),
            },
        }


def build_client_settings(async_insert: bool) -> dict[str, int]:
    if not async_insert:
        return {}
    return {"async_insert": 1, "wait_for_async_insert": 1}


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
    clickhouse.index_granularity = 8192

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
) -> TrialMeasurement:
    engine.start()
    cpu_start = time.process_time()
    peak_rss_before = _peak_rss_bytes()
    total_start = clock()
    latencies = []
    sample_stride = max(1, math.ceil(rows / max_latency_samples))
    try:
        for row in range(rows):
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
        if not engine.join(timeout_seconds):
            raise TimeoutError(f"ClickHouse drain exceeded {timeout_seconds:.1f}s")
        engine.raise_if_failed()
        total_end = clock()
        cpu_end = time.process_time()
        peak_rss_after = _peak_rss_bytes()
    except BaseException:
        engine.request_abort()
        engine.join(timeout_seconds)
        raise

    return TrialMeasurement(
        rows=rows,
        payload_bytes=int(payloads[0].nbytes),
        enqueue_seconds=(enqueue_end - total_start) / 1_000_000_000,
        total_seconds=(total_end - total_start) / 1_000_000_000,
        enqueue_latencies_ns=tuple(latencies),
        process_cpu_seconds=cpu_end - cpu_start,
        peak_rss_before_bytes=peak_rss_before,
        peak_rss_after_bytes=peak_rss_after,
    )


def _connect(config: BenchmarkConfig) -> Any:
    try:
        from clickhouse_driver import Client
    except ImportError as exc:
        raise RuntimeError("clickhouse-driver is required for benchmark verification") from exc

    connection = {
        "host": config.host,
        "port": config.port,
        "user": config.user,
        "password": config.password,
        "secure": config.secure,
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


def run(config: BenchmarkConfig) -> dict[str, Any]:
    payloads = generate_payload_pool(config)
    client = _connect(config)
    table = f"{quote_identifier(config.database)}.{quote_identifier(config.table)}"
    measured_model_id = f"host-bench-{uuid.uuid4().hex}"
    warnings: list[str] = []
    try:
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
        measurement = _submit_and_drain(
            _build_engine(config),
            payloads,
            config.rows,
            measured_model_id,
            config.drain_timeout_seconds,
            config.latency_samples,
        )
        verification = _verification(client, config, measured_model_id)
        parts = _parts_metrics(client, config)
        query_log, warning = _query_log_metrics(client, config, started_at)
        if warning:
            warnings.append(warning)
        return {
            "benchmark": "dmi_clickhouse_host",
            "server_version": server_version,
            "config": _safe_config(config),
            "measurement": measurement.as_dict(),
            "verification": verification,
            "parts": parts,
            "query_log": query_log,
            "warnings": warnings,
        }
    finally:
        if not config.keep_table:
            try:
                client.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception as exc:
                warnings.append(f"table cleanup failed: {exc}")
        client.disconnect()


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
    parser.add_argument("--min-batch-bytes", type=parse_byte_size, default=16 * 1024**2)
    parser.add_argument("--max-batch-bytes", type=parse_byte_size, default=64 * 1024**2)
    parser.add_argument("--max-batch-items", type=int, default=10_000)
    parser.add_argument("--max-linger-ms", type=float, default=50.0)
    parser.add_argument("--queue-capacity-bytes", type=parse_byte_size, default=512 * 1024**2)
    parser.add_argument("--queue-capacity-items", type=int, default=20_000)
    parser.add_argument("--compression", choices=("none", "lz4", "zstd"), default="lz4")
    parser.add_argument("--async-insert", action="store_true")
    parser.add_argument("--drain-timeout-seconds", type=float, default=300.0)
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
        compression=args.compression,
        async_insert=args.async_insert,
        drain_timeout_seconds=args.drain_timeout_seconds,
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
    if args.dry_run:
        report: dict[str, Any] = {
            "benchmark": "dmi_clickhouse_host",
            "dry_run": True,
            "config": _safe_config(config),
            "logical_payload_bytes": config.rows * config.payload_bytes,
            "payload_pool_bytes": config.pool_size * config.payload_bytes,
        }
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
