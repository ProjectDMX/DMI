#!/usr/bin/env python3
"""Summarize paired scheduler OFF/ON overhead trials."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from common import percentile, read_jsonl, write_json


METRIC_NAMES = (
    "latency_mean_s",
    "latency_p50_s",
    "latency_p95_s",
    "latency_p99_s",
    "ttft_mean_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "ttft_p99_s",
    "request_throughput_rps",
    "output_throughput_tps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def phase_elapsed(markers: list[dict[str, Any]], phase: str) -> float | None:
    for row in markers:
        if row.get("event_type") == "phase_end" and row.get("phase") == phase:
            value = row.get("elapsed_s")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def summarize_trial(run_root: Path) -> dict[str, Any]:
    run_dir = run_root / "no_move"
    rows = [
        row
        for row in read_jsonl(run_dir / "client_results.jsonl")
        if row.get("agent_id") == "replay"
    ]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    latencies = [float(row["latency_s"]) for row in ok_rows]
    ttfts = [float(row["ttft_s"]) for row in ok_rows]
    output_tokens = sum(int(row.get("completion_tokens", 0)) for row in ok_rows)
    elapsed = phase_elapsed(read_jsonl(run_dir / "phase_markers.jsonl"), "replay")

    metrics = {
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_p99_s": percentile(ttfts, 99),
        "request_throughput_rps": len(ok_rows) / elapsed if elapsed else None,
        "output_throughput_tps": output_tokens / elapsed if elapsed else None,
    }
    return {
        "run_root": str(run_root),
        "requests": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "phase_elapsed_s": elapsed,
        "output_tokens": output_tokens,
        "metrics": metrics,
    }


def mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "values": values,
    }


def pct_delta(new: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (new - baseline) * 100.0 / baseline


def main() -> None:
    args = parse_args()
    trials: dict[str, dict[str, Any]] = {"off": {}, "on": {}}
    for mode in ("off", "on"):
        for repetition in range(1, 4):
            key = f"rep{repetition}"
            trials[mode][key] = summarize_trial(
                args.root / "runs" / f"{mode}_{key}"
            )

    aggregate: dict[str, Any] = {}
    for mode, mode_trials in trials.items():
        aggregate[mode] = {
            name: mean_std(
                [
                    float(trial["metrics"][name])
                    for trial in mode_trials.values()
                    if isinstance(trial["metrics"].get(name), (int, float))
                ]
            )
            for name in METRIC_NAMES
        }

    paired: dict[str, Any] = {}
    for name in METRIC_NAMES:
        rows = []
        for repetition in range(1, 4):
            key = f"rep{repetition}"
            off = trials["off"][key]["metrics"][name]
            on = trials["on"][key]["metrics"][name]
            if not isinstance(off, (int, float)) or not isinstance(on, (int, float)):
                continue
            rows.append(
                {
                    "repetition": repetition,
                    "off": off,
                    "on": on,
                    "delta_abs": on - off,
                    "delta_pct": pct_delta(on, off),
                }
            )
        paired[name] = {
            "pairs": rows,
            "delta_abs": mean_std([float(row["delta_abs"]) for row in rows]),
            "delta_pct": mean_std(
                [float(row["delta_pct"]) for row in rows if row["delta_pct"] is not None]
            ),
        }

    summary = {
        "experiment_root": str(args.root),
        "design": {
            "repetitions": 3,
            "order": ["off_rep1", "on_rep1", "on_rep2", "off_rep2", "off_rep3", "on_rep3"],
            "comparison": "same vLLM+DMI trace; only PCIe governor enable changes",
            "kv_offload_enabled": False,
            "analyzed_phase": "replay",
        },
        "trials": trials,
        "aggregate": aggregate,
        "paired": paired,
    }
    write_json(args.out, summary)
    print(json.dumps({"aggregate": aggregate, "paired": paired}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
