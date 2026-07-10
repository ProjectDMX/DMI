#!/usr/bin/env python3
"""Measure chunk-cap and lifecycle-hint protection for a critical D2H copy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


MIB = 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def describe(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(vals),
        "mean": statistics.mean(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "max": max(vals),
    }


def wait_until(predicate, timeout_s: float, message: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.0005)
    raise TimeoutError(message)


def sleep_precise(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    if seconds > 0.0005:
        time.sleep(seconds - 0.0003)
    while time.perf_counter() < deadline:
        pass


def critical_copy(torch_mod: Any, device_buf: Any, host_buf: Any, stream: Any) -> tuple[float, float]:
    start_event = torch_mod.cuda.Event(enable_timing=True)
    end_event = torch_mod.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    with torch_mod.cuda.stream(stream):
        start_event.record(stream)
        host_buf.copy_(device_buf, non_blocking=True)
        end_event.record(stream)
    stream.synchronize()
    host_ms = (time.perf_counter() - host_start) * 1000.0
    event_ms = float(start_event.elapsed_time(end_event))
    return host_ms, event_ms


def run_condition(args: argparse.Namespace, condition: str) -> list[dict[str, Any]]:
    import torch

    from monitoring import _native_engine as native
    from monitoring.governor import PCIeGovernor, PCIeGovernorConfig, PCIeHint
    from monitoring.ring_transport import RingTransport, activate, deactivate

    torch.cuda.set_device(args.device)
    ring_bytes = args.ring_mb * MIB
    chunk_bytes = args.dmi_chunk_mb * MIB
    backlog_bytes = args.dmi_backlog_mb * MIB
    critical_bytes = args.critical_mb * MIB
    num_chunks = backlog_bytes // chunk_bytes
    if backlog_bytes % chunk_bytes or num_chunks <= 0:
        raise ValueError("dmi_backlog_mb must be a positive multiple of dmi_chunk_mb")

    cfg = native.RingConfig()
    cfg.payload_ring_bytes = ring_bytes
    cfg.pinned_staging_bytes = ring_bytes
    cfg.task_ring_entries = max(128, num_chunks * 4)
    cfg.drain_poll_timeout_us = 50
    cfg.drain_flush_byte_threshold = 1
    cfg.drain_flush_timeout_us = 0

    engine = native.RingEngine(cfg, None)
    engine.init()
    engine.start()
    transport = RingTransport(engine)
    activate(transport)

    governor = None
    if condition == "hint_and_cap":
        governor = PCIeGovernor(
            engine,
            PCIeGovernorConfig(
                enabled=True,
                max_defer_us=args.max_defer_us,
                hard_watermark_ratio=args.hard_watermark_ratio,
                max_d2h_chunk_bytes=args.max_d2h_chunk_mb * MIB,
            ),
        )
    elif condition == "cap_only":
        engine.set_drain_control(0, args.max_d2h_chunk_mb * MIB, 0)
    else:
        engine.set_drain_control(0, 0, 0)

    source = torch.empty(chunk_bytes, dtype=torch.uint8, device=f"cuda:{args.device}")
    critical_device = torch.empty(
        critical_bytes, dtype=torch.uint8, device=f"cuda:{args.device}"
    )
    critical_host = torch.empty(critical_bytes, dtype=torch.uint8, pin_memory=True)
    critical_stream = torch.cuda.Stream(device=args.device)
    torch.cuda.synchronize(args.device)

    for _ in range(3):
        critical_copy(torch, critical_device, critical_host, critical_stream)

    rows: list[dict[str, Any]] = []
    try:
        for repetition in range(1, args.repetitions + 1):
            stats_before = engine.link_stats()
            d2h_before = int(stats_before["d2h_bytes"])
            batches_before = int(stats_before["d2h_batches"])
            event_us_before = int(stats_before["d2h_probe_event_us"])
            host_us_before = int(stats_before["d2h_probe_host_us"])

            if condition != "idle":
                hold_until = time.monotonic_ns() + 1_000_000_000
                engine.set_defer_until_ns(hold_until)
                result = int(engine.prepare_step(backlog_bytes, num_chunks))
                if result != 0:
                    raise RuntimeError(f"prepare_step failed: {result}")
                for hook_id in range(num_chunks):
                    torch.ops.ring.producer(
                        transport._ring_payload,
                        source,
                        0,
                        hook_id,
                    )
                torch.cuda.synchronize(args.device)

                if governor is None:
                    engine.set_defer_until_ns(0)
                else:
                    governor.on_hint(PCIeHint(direction="D2H", source="kv_store"))
                sleep_precise(args.dmi_head_start_ms / 1000.0)

            critical_host_ms, critical_event_ms = critical_copy(
                torch,
                critical_device,
                critical_host,
                critical_stream,
            )

            if governor is not None:
                governor.on_hint(
                    PCIeHint(
                        direction="D2H",
                        source="kv_store",
                        valid_until_ns=time.monotonic_ns(),
                    )
                )

            if condition != "idle":
                target = d2h_before + backlog_bytes
                wait_until(
                    lambda: int(engine.link_stats()["d2h_bytes"]) >= target,
                    args.drain_timeout_s,
                    f"DMI backlog did not drain for {condition} repetition {repetition}",
                )

            stats_after = engine.link_stats()
            d2h_delta = int(stats_after["d2h_bytes"]) - d2h_before
            batches_delta = int(stats_after["d2h_batches"]) - batches_before
            event_us_delta = int(stats_after["d2h_probe_event_us"]) - event_us_before
            host_us_delta = int(stats_after["d2h_probe_host_us"]) - host_us_before
            rows.append(
                {
                    "condition": condition,
                    "repetition": repetition,
                    "critical_host_ms": critical_host_ms,
                    "critical_event_ms": critical_event_ms,
                    "critical_mb": args.critical_mb,
                    "dmi_bytes": d2h_delta,
                    "dmi_batches": batches_delta,
                    "dmi_event_us": event_us_delta,
                    "dmi_host_us": host_us_delta,
                    "dmi_event_gbps": (
                        d2h_delta * 8.0 / (event_us_delta * 1000.0)
                        if event_us_delta > 0
                        else 0.0
                    ),
                    "dmi_backlog_mb": args.dmi_backlog_mb if condition != "idle" else 0,
                }
            )
            time.sleep(0.005)
    finally:
        if governor is not None:
            governor.disable()
        deactivate()
        engine.stop()

    return rows


def delta(lhs: float | None, rhs: float | None) -> dict[str, float | None]:
    if lhs is None or rhs is None:
        return {"lhs": lhs, "rhs": rhs, "delta_abs": None, "delta_pct": None}
    absolute = lhs - rhs
    return {
        "lhs": lhs,
        "rhs": rhs,
        "delta_abs": absolute,
        "delta_pct": absolute / rhs * 100.0 if rhs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--ring-mb", type=int, default=1024)
    parser.add_argument("--dmi-backlog-mb", type=int, default=512)
    parser.add_argument("--dmi-chunk-mb", type=int, default=16)
    parser.add_argument("--critical-mb", type=int, default=32)
    parser.add_argument("--dmi-head-start-ms", type=float, default=1.0)
    parser.add_argument("--max-defer-us", type=int, default=5000)
    parser.add_argument("--max-d2h-chunk-mb", type=int, default=32)
    parser.add_argument("--hard-watermark-ratio", type=float, default=0.80)
    parser.add_argument("--drain-timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    if args.dmi_backlog_mb >= args.ring_mb * args.hard_watermark_ratio:
        raise ValueError("DMI backlog must stay below the governor hard watermark")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    conditions = ("idle", "gov_off", "cap_only", "hint_and_cap")
    for condition in conditions:
        rows.extend(run_condition(args, condition))

    jsonl_path = args.out_dir / "contention_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    aggregates: dict[str, Any] = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        aggregates[condition] = {
            "critical_host_ms": describe(row["critical_host_ms"] for row in selected),
            "critical_event_ms": describe(row["critical_event_ms"] for row in selected),
            "dmi_batches": describe(row["dmi_batches"] for row in selected),
            "dmi_event_gbps": describe(row["dmi_event_gbps"] for row in selected),
        }

    def compare_conditions(lhs: str, rhs: str) -> dict[str, Any]:
        left = aggregates[lhs]["critical_host_ms"]
        right = aggregates[rhs]["critical_host_ms"]
        return {
            "mean": delta(left["mean"], right["mean"]),
            "p95": delta(left["p95"], right["p95"]),
        }

    summary = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "aggregate": aggregates,
        "comparisons": {
            "uncontrolled_slowdown_vs_idle": compare_conditions("gov_off", "idle"),
            "cap_only_vs_off": compare_conditions("cap_only", "gov_off"),
            "hint_and_cap_vs_cap_only": compare_conditions(
                "hint_and_cap", "cap_only"
            ),
            "hint_and_cap_vs_off": compare_conditions("hint_and_cap", "gov_off"),
            "hint_and_cap_gap_vs_idle": compare_conditions("hint_and_cap", "idle"),
        },
    }
    summary_path = args.out_dir / "contention_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
