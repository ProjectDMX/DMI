#!/usr/bin/env python3
"""Summarize the DMI PCIe governor KV-store storm experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STORE_RE = re.compile(
    r"Stored (\d+) out of total (\d+) tokens\. size: ([0-9.]+) GB, "
    r"cost ([0-9.]+) ms, throughput: ([0-9.]+) GB/s"
    r"(?:; offload_time: ([0-9.]+) ms, put_time: ([0-9.]+) ms)?"
)
MP_STORE_RE = re.compile(r"Stored (\d+) tokens in ([0-9.]+) seconds")
MP_HINT_AUDIT_RE = re.compile(
    r"\[DMI PCIeHint\] source=lmcache_mp_store event=end "
    r"duration_ms=([0-9.]+)(?: stores=(\d+))?"
)
HINT_ACTION_RE = re.compile(
    r"PCIeHint\(direction='D2H', source='([^']+)', est_bytes=\d+, "
    r"valid_until_ns=(\d+)\) action=(start|renew|end) now_ns=(\d+) "
    r"source_deadline_ns=(\d+) hint_deadline_ns=(\d+)"
)
HINT_RE = re.compile(
    r"PCIeHint\(direction='D2H', source='([^']+)', est_bytes=\d+, "
    r"valid_until_ns=(\d+)\) hint_deadline_ns=(\d+)"
)
STEP_RE = re.compile(
    r"step pressure=([0-9.]+) baseline_gbps=([0-9.]+).*"
    r"feedback_deadline_ns=(\d+)"
)
TRIAL_RE = re.compile(r"(.+)_rep(\d+)$")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def describe(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "mean": statistics.mean(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": max(vals),
    }


def phase_window(run_dir: Path) -> tuple[int | None, int | None, float | None]:
    begin = None
    end = None
    elapsed = None
    for row in read_jsonl(run_dir / "phase_markers.jsonl"):
        if row.get("phase") != "cold_store":
            continue
        if row.get("event_type") == "phase_begin":
            begin = int(row["ts_ns"])
        elif row.get("event_type") == "phase_end":
            end = int(row["ts_ns"])
            elapsed = float(row.get("elapsed_s", 0.0)) or None
    return begin, end, elapsed


def summarize_client(run_dir: Path, warmup_requests: int) -> dict[str, Any]:
    all_rows = sorted(
        read_jsonl(run_dir / "client_results.jsonl"),
        key=lambda row: (
            float(row.get("planned_timestamp_s", 0.0)),
            float(row.get("send_ts", 0.0)),
        ),
    )
    warmup = min(max(0, warmup_requests), len(all_rows))
    rows = all_rows[warmup:]
    ok = [row for row in rows if row.get("status") == "ok"]
    latency = [
        float(row["latency_s"]) for row in ok if row.get("latency_s") is not None
    ]
    ttft = [float(row["ttft_s"]) for row in ok if row.get("ttft_s") is not None]
    prompt_tokens = [
        float(row["prompt_tokens"])
        for row in ok
        if row.get("prompt_tokens") is not None
    ]
    measured_elapsed_s = None
    if ok:
        begin = min(float(row.get("send_ts", 0.0)) for row in ok)
        end = max(float(row.get("done_ts", begin)) for row in ok)
        measured_elapsed_s = max(0.0, end - begin)
    throughput = (
        len(ok) / measured_elapsed_s
        if measured_elapsed_s is not None and measured_elapsed_s > 0
        else None
    )
    return {
        "requests_total": len(all_rows),
        "warmup_requests": warmup,
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "errors_total": sum(row.get("status") != "ok" for row in all_rows),
        "measured_elapsed_s": measured_elapsed_s,
        "request_throughput_rps": throughput,
        "latency_s": describe(latency),
        "ttft_s": describe(ttft),
        "prompt_tokens": describe(prompt_tokens),
    }


def summarize_lmcache(
    log_text: str,
    phase_elapsed_s: float | None,
    *,
    mp_log_text: str = "",
    kv_bytes_per_token: int = 0,
    warmup_requests: int = 0,
) -> dict[str, Any]:
    all_rows = [
        {
            "stored_tokens": int(match.group(1)),
            "total_tokens": int(match.group(2)),
            "size_gb": float(match.group(3)),
            "cost_ms": float(match.group(4)),
            "throughput_gb_s": float(match.group(5)),
            # Layerwise logging reports only total cost. It is the closest
            # operation-level D2H timing available for that mode.
            "offload_ms": (
                float(match.group(6))
                if match.group(6) is not None
                else float(match.group(4))
            ),
            "put_ms": (float(match.group(7)) if match.group(7) is not None else None),
        }
        for match in STORE_RE.finditer(log_text)
    ]
    mode = "inprocess"
    mapping_valid = True
    completion_rows: list[dict[str, Any]] = all_rows
    mp_enqueue_ms: list[float] = []
    if not all_rows and mp_log_text and kv_bytes_per_token > 0:
        mode = "mp"
        stores = [
            {
                "stored_tokens": int(match.group(1)),
                "total_tokens": int(match.group(1)),
                "size_gb": int(match.group(1)) * kv_bytes_per_token / (1024**3),
                "enqueue_ms": float(match.group(2)) * 1000.0,
            }
            for match in MP_STORE_RE.finditer(mp_log_text)
        ]
        all_rows = stores
        audits = [
            (float(match.group(1)), int(match.group(2)) if match.group(2) else None)
            for match in MP_HINT_AUDIT_RE.finditer(log_text)
        ]
        if audits and all(count is None for _, count in audits):
            mapping_valid = len(audits) == len(stores)
            audits = [(duration_ms, 1) for duration_ms, _ in audits]
        else:
            mapping_valid = bool(audits) and all(
                count is not None and count > 0 for _, count in audits
            )
            mapping_valid = mapping_valid and sum(
                int(count or 0) for _, count in audits
            ) == len(stores)

        completion_rows = []
        if mapping_valid:
            offset = 0
            for duration_ms, count_value in audits:
                count = int(count_value or 0)
                episode_stores = stores[offset : offset + count]
                offset += count
                size_gb = sum(float(row["size_gb"]) for row in episode_stores)
                seconds = duration_ms / 1000.0
                completion_rows.append(
                    {
                        "store_count": count,
                        "stored_tokens": sum(
                            int(row["stored_tokens"]) for row in episode_stores
                        ),
                        "size_gb": size_gb,
                        "cost_ms": duration_ms,
                        "offload_ms": duration_ms,
                        "throughput_gb_s": (size_gb / seconds if seconds > 0 else 0.0),
                        "put_ms": None,
                    }
                )

        warmup_left = min(max(0, warmup_requests), len(stores))
        measured_completion_rows = []
        for row in completion_rows:
            count = int(row.get("store_count", 1))
            if warmup_left > 0:
                warmup_left = max(0, warmup_left - count)
                continue
            measured_completion_rows.append(row)
        completion_rows = measured_completion_rows
        mp_enqueue_ms = [
            float(row["enqueue_ms"])
            for row in stores[min(max(0, warmup_requests), len(stores)) :]
        ]

    warmup = min(max(0, warmup_requests), len(all_rows))
    rows = all_rows[warmup:]
    if mode != "mp":
        completion_rows = rows
    total_gb = sum(row["size_gb"] for row in rows)
    aggregate_gb_s = None
    if phase_elapsed_s and phase_elapsed_s > 0:
        aggregate_gb_s = total_gb / phase_elapsed_s
    return {
        "mode": mode,
        "store_calls_total": len(all_rows),
        "warmup_store_calls": warmup,
        "store_calls": len(rows),
        "stored_tokens": sum(row["stored_tokens"] for row in rows),
        "total_gb": total_gb,
        "phase_aggregate_gb_s": aggregate_gb_s,
        "completion_episodes": len(completion_rows),
        "completion_mapping_valid": mapping_valid,
        "cost_ms": describe(row["cost_ms"] for row in completion_rows),
        "offload_ms": describe(row["offload_ms"] for row in completion_rows),
        "throughput_gb_s": describe(row["throughput_gb_s"] for row in completion_rows),
        "put_ms": describe(row["put_ms"] for row in completion_rows),
        "mp_enqueue_ms": describe(mp_enqueue_ms),
    }


def summarize_governor(log_text: str, max_defer_us: int) -> dict[str, Any]:
    starts: dict[str, int] = {}
    deadlines: dict[str, int] = {}
    durations_ms: list[float] = []
    started = 0
    renewed = 0
    ended = 0
    expired_without_end = 0
    action_matches = list(HINT_ACTION_RE.finditer(log_text))
    if action_matches:
        for match in action_matches:
            source, _valid, action, now_text, deadline_text, _global = match.groups()
            now = int(now_text)
            deadline = int(deadline_text)
            if action == "start":
                if source in starts:
                    expired_without_end += 1
                starts[source] = now
                deadlines[source] = deadline
                started += 1
            elif action == "renew":
                if source not in starts:
                    starts[source] = now
                    started += 1
                else:
                    renewed += 1
                deadlines[source] = deadline
            elif action == "end" and source in starts:
                start = starts.pop(source)
                deadlines.pop(source, None)
                durations_ms.append(max(0.0, (now - start) / 1_000_000.0))
                ended += 1
    else:
        # Backward-compatible parser for logs written before explicit hint
        # actions were added. This is exact for the benchmark's one-source
        # lifecycle and prevents a post-watchdog start from becoming a renewal.
        max_defer_ns = max(0, max_defer_us) * 1000
        for match in HINT_RE.finditer(log_text):
            source, valid_text, deadline_text = match.groups()
            valid = int(valid_text)
            deadline = int(deadline_text)
            if valid == 0 and deadline > 0:
                now = max(0, deadline - max_defer_ns)
                active = source in starts and deadlines.get(source, 0) > now
                if active:
                    renewed += 1
                else:
                    if source in starts:
                        expired_without_end += 1
                    starts[source] = now
                    started += 1
                deadlines[source] = deadline
            elif valid > 0 and source in starts:
                start = starts.pop(source)
                deadlines.pop(source, None)
                durations_ms.append(max(0.0, (valid - start) / 1_000_000.0))
                ended += 1

    pressures: list[float] = []
    baselines: list[float] = []
    active_steps = 0
    activation_edges = 0
    was_active = False
    for match in STEP_RE.finditer(log_text):
        pressure, baseline, deadline = match.groups()
        pressures.append(float(pressure))
        baselines.append(float(baseline))
        active = int(deadline) > 0
        active_steps += int(active)
        activation_edges += int(active and not was_active)
        was_active = active

    return {
        "hints_started": started,
        "hints_renewed": renewed,
        "hints_ended": ended,
        "hints_expired_without_end": expired_without_end,
        "hints_unterminated": len(starts),
        "hint_coverage_ms": describe(durations_ms),
        "feedback_samples": len(pressures),
        "feedback_active_samples": active_steps,
        "feedback_activation_edges": activation_edges,
        "pressure": describe(pressures),
        "baseline_gbps": describe(baselines),
    }


def trial_identity(path: Path) -> tuple[str, int]:
    match = TRIAL_RE.fullmatch(path.parent.name)
    if not match:
        return path.parent.name, 0
    return match.group(1), int(match.group(2))


def summarize_trial(run_dir: Path, warmup_requests: int) -> dict[str, Any]:
    condition, repetition = trial_identity(run_dir)
    config = read_json(run_dir / "experiment_config.json")
    _begin_ns, _end_ns, phase_elapsed_s = phase_window(run_dir)
    log_path = run_dir / "logs" / "vllm.log"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    mp_log_path = run_dir / "logs" / "lmcache_server.log"
    mp_log_text = (
        mp_log_path.read_text(encoding="utf-8", errors="replace")
        if mp_log_path.exists()
        else ""
    )
    trace_summary = read_json(run_dir / "trace_summary.json")
    kv_bytes_per_token = int(
        trace_summary.get("config", {}).get("kv_bytes_per_token", 0)
    )
    max_defer_us = int(config.get("pcie_governor_max_defer_us", 5000))
    return {
        "condition": condition,
        "repetition": repetition,
        "run_dir": str(run_dir),
        "phase_elapsed_s": phase_elapsed_s,
        "config": config,
        "client": summarize_client(run_dir, warmup_requests),
        "lmcache_store": summarize_lmcache(
            log_text,
            phase_elapsed_s,
            mp_log_text=mp_log_text,
            kv_bytes_per_token=kv_bytes_per_token,
            warmup_requests=warmup_requests,
        ),
        "governor": summarize_governor(log_text, max_defer_us),
    }


METRICS = {
    "client_latency_mean_s": ("client", "latency_s", "mean"),
    "client_latency_p95_s": ("client", "latency_s", "p95"),
    "client_ttft_mean_s": ("client", "ttft_s", "mean"),
    "client_ttft_p95_s": ("client", "ttft_s", "p95"),
    "request_throughput_rps": ("client", "request_throughput_rps"),
    "lmcache_offload_mean_ms": ("lmcache_store", "offload_ms", "mean"),
    "lmcache_offload_p95_ms": ("lmcache_store", "offload_ms", "p95"),
    "lmcache_throughput_mean_gb_s": (
        "lmcache_store",
        "throughput_gb_s",
        "mean",
    ),
}


def nested_value(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if value is not None else None


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[str(trial["condition"])].append(trial)
    out: dict[str, Any] = {}
    for condition, condition_trials in grouped.items():
        out[condition] = {
            name: describe(
                value
                for trial in condition_trials
                if (value := nested_value(trial, path)) is not None
            )
            for name, path in METRICS.items()
        }
    return out


def comparisons(aggregates: dict[str, Any]) -> dict[str, Any]:
    pairs = {
        "governor_on_minus_off": ("gov_on", "gov_off"),
        "governor_on_minus_dmi_off": ("gov_on", "dmi_off"),
    }
    out: dict[str, Any] = {}
    for label, (lhs, rhs) in pairs.items():
        if lhs not in aggregates or rhs not in aggregates:
            continue
        metrics: dict[str, Any] = {}
        for name in METRICS:
            left = aggregates[lhs][name]["mean"]
            right = aggregates[rhs][name]["mean"]
            if left is None or right is None:
                continue
            delta = left - right
            metrics[name] = {
                "lhs": left,
                "rhs": right,
                "delta_abs": delta,
                "delta_pct": delta / right * 100.0 if right else None,
            }
        out[label] = metrics
    return out


def validate_trials(
    trials: list[dict[str, Any]],
    *,
    min_repetitions: int,
    min_measured_requests: int,
    require_governor_engaged: bool,
) -> list[str]:
    errors: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[str(trial["condition"])].append(trial)

    if not trials:
        return ["no trial directories found"]

    for condition, condition_trials in sorted(grouped.items()):
        repetitions = {int(trial["repetition"]) for trial in condition_trials}
        if len(repetitions) < min_repetitions:
            errors.append(
                f"{condition}: found {len(repetitions)} repetitions, "
                f"require at least {min_repetitions}"
            )
        for trial in condition_trials:
            label = f"{condition}_rep{trial['repetition']}"
            client = trial["client"]
            if int(client["errors_total"]) > 0:
                errors.append(f"{label}: {client['errors_total']} client errors")
            if int(client["requests"]) < min_measured_requests:
                errors.append(
                    f"{label}: only {client['requests']} measured requests after "
                    f"warmup; require {min_measured_requests}"
                )

            config = trial["config"]
            expected = {
                "dmi_off": (False, False),
                "gov_off": (True, False),
                "gov_on": (True, True),
            }.get(condition)
            if expected is not None:
                actual = (
                    bool(config.get("dmi_enabled")),
                    bool(config.get("pcie_governor_enabled")),
                )
                if actual != expected:
                    errors.append(
                        f"{label}: expected dmi/governor={expected}, got {actual}"
                    )

            lmcache = trial["lmcache_store"]
            if lmcache["mode"] == "mp" and not lmcache["completion_mapping_valid"]:
                errors.append(f"{label}: MP store-to-episode mapping is ambiguous")

    if require_governor_engaged:
        for trial in grouped.get("gov_on", []):
            label = f"gov_on_rep{trial['repetition']}"
            governor = trial["governor"]
            started = int(governor["hints_started"])
            ended = int(governor["hints_ended"])
            if started == 0:
                errors.append(f"{label}: governor accepted no lifecycle starts")
            if ended != started:
                errors.append(
                    f"{label}: lifecycle starts/ends mismatch ({started}/{ended})"
                )
            if int(governor["hints_expired_without_end"]) > 0:
                errors.append(f"{label}: lifecycle hint expired before end")
            if int(governor["hints_unterminated"]) > 0:
                errors.append(f"{label}: lifecycle hint remained unterminated")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--min-repetitions", type=int, default=0)
    parser.add_argument("--min-measured-requests", type=int, default=0)
    parser.add_argument("--require-governor-engaged", action="store_true")
    args = parser.parse_args()

    run_dirs = sorted((args.run_root / "runs").glob("*_rep*/cold_store"))
    trials = [summarize_trial(path, args.warmup_requests) for path in run_dirs]
    aggregates = aggregate(trials)
    validation_errors = validate_trials(
        trials,
        min_repetitions=max(0, args.min_repetitions),
        min_measured_requests=max(0, args.min_measured_requests),
        require_governor_engaged=args.require_governor_engaged,
    )
    result = {
        "run_root": str(args.run_root.resolve()),
        "warmup_requests": max(0, args.warmup_requests),
        "trials": trials,
        "aggregate": aggregates,
        "comparisons": comparisons(aggregates),
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
        },
    }
    out_path = args.run_root / "scheduler_effectiveness_summary.json"
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if validation_errors:
        print("benchmark validation failed:", file=sys.stderr)
        for error in validation_errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
