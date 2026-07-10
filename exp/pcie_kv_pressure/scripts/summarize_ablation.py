#!/usr/bin/env python3
"""Summarize DMI + vLLM + LMCache no-move vs move ablations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import read_jsonl, stats, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--out", default="")
    return p.parse_args()


def phase_of_client_row(row: dict[str, Any]) -> str:
    value = row.get("agent_id")
    if isinstance(value, str) and value:
        return value
    event_id = str(row.get("event_id", "unknown"))
    return event_id.split("_", 1)[0]


def summarize_client(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "client_results.jsonl")
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_phase[phase_of_client_row(row)].append(row)
    out: dict[str, Any] = {"total": summarize_client_rows(rows), "by_phase": {}}
    for phase, items in sorted(by_phase.items()):
        out["by_phase"][phase] = summarize_client_rows(items)
    return out


def summarize_client_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    return {
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "latency_s": stats(
            [
                row.get("latency_s")
                for row in ok
                if isinstance(row.get("latency_s"), (int, float))
            ]
        ),
        "ttft_s": stats(
            [
                row.get("ttft_s")
                for row in ok
                if isinstance(row.get("ttft_s"), (int, float))
            ]
        ),
        "prompt_tokens": stats(
            [
                row.get("prompt_tokens")
                for row in ok
                if isinstance(row.get("prompt_tokens"), (int, float))
            ]
        ),
        "completion_tokens": stats(
            [
                row.get("completion_tokens")
                for row in ok
                if isinstance(row.get("completion_tokens"), (int, float))
            ]
        ),
    }


def summarize_kv_wait(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "kv_wait_trace.jsonl")
    event_counts = Counter(str(row.get("event_type")) for row in rows)
    retrieve_done = [row for row in rows if row.get("event_type") == "kv_retrieve_done"]
    engine_done = [
        row for row in rows if row.get("event_type") == "kv_retrieve_engine_done"
    ]
    allowed = [row for row in rows if row.get("event_type") == "kv_load_allowed"]
    return {
        "rows": len(rows),
        "event_counts": dict(event_counts),
        "unique_retrieve_reqs": len(
            {row.get("req_id") for row in retrieve_done if row.get("req_id")}
        ),
        "unique_allowed_reqs": len(
            {row.get("req_id") for row in allowed if row.get("req_id")}
        ),
        "adapter_retrieve_ms": stats(
            [
                row.get("adapter_retrieve_ms")
                for row in retrieve_done
                if isinstance(row.get("adapter_retrieve_ms"), (int, float))
            ]
        ),
        "engine_retrieve_ms": stats(
            [
                row.get("engine_retrieve_ms")
                for row in engine_done
                if isinstance(row.get("engine_retrieve_ms"), (int, float))
            ]
        ),
        "to_gpu_ms": stats(
            [
                row.get("to_gpu_ms")
                for row in engine_done
                if isinstance(row.get("to_gpu_ms"), (int, float))
            ]
        ),
        "kv_size_gb": stats(
            [
                row.get("kv_size_gb")
                for row in engine_done
                if isinstance(row.get("kv_size_gb"), (int, float))
            ]
        ),
        "retrieved_tokens": stats(
            [
                row.get("retrieved_tokens")
                for row in retrieve_done
                if isinstance(row.get("retrieved_tokens"), (int, float))
            ]
        ),
        "need_to_load_tokens": stats(
            [
                row.get("need_to_load_tokens")
                for row in allowed
                if isinstance(row.get("need_to_load_tokens"), (int, float))
            ]
        ),
    }


def summarize_markers(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "phase_markers.jsonl")
    return {
        "rows": len(rows),
        "events": [
            row
            for row in rows
            if row.get("event_type") in {"phase_begin", "phase_end", "replay_error"}
        ],
    }


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "manifest": read_manifest(run_dir),
        "client": summarize_client(run_dir),
        "kv_wait": summarize_kv_wait(run_dir),
        "phase_markers": summarize_markers(run_dir),
    }


def pct_delta(new: float | None, base: float | None) -> float | None:
    if base in (None, 0) or new is None:
        return None
    return (new - base) / base * 100.0


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def compare(
    no_move: dict[str, Any] | None, move: dict[str, Any] | None
) -> dict[str, Any]:
    if not no_move or not move:
        return {}
    out: dict[str, Any] = {}
    metrics = {
        "replay_ttft_p50_s": ("client", "by_phase", "replay", "ttft_s", "p50"),
        "replay_ttft_p95_s": ("client", "by_phase", "replay", "ttft_s", "p95"),
        "replay_ttft_p99_s": ("client", "by_phase", "replay", "ttft_s", "p99"),
        "replay_latency_p99_s": ("client", "by_phase", "replay", "latency_s", "p99"),
    }
    for name, path in metrics.items():
        base = get_nested(no_move, path)
        new = get_nested(move, path)
        out[name] = {"no_move": base, "move": new, "delta_pct": pct_delta(new, base)}
    out["kv_retrieve_reqs"] = {
        "no_move": get_nested(no_move, ("kv_wait", "unique_retrieve_reqs")),
        "move": get_nested(move, ("kv_wait", "unique_retrieve_reqs")),
    }
    return out


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    conditions = {}
    for name in ("no_move", "move"):
        run_dir = run_root / name
        if run_dir.exists():
            conditions[name] = summarize_run(run_dir)
    summary = {
        "run_root": str(run_root),
        "conditions": conditions,
        "comparison": compare(conditions.get("no_move"), conditions.get("move")),
    }
    out = Path(args.out) if args.out else run_root / "ablation_summary.json"
    write_json(out, summary)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
