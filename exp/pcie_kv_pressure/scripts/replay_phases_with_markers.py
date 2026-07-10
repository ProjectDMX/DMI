#!/usr/bin/env python3
"""Replay RealReentry phases and emit structured phase markers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from common import append_jsonl, now_ns, read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hiddencache-dir", type=Path, default=Path("../hiddencache"))
    p.add_argument("--trace", required=True)
    p.add_argument("--prefix-bank", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--markers", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=("open-loop", "closed-loop"), default="open-loop")
    p.add_argument("--endpoint", choices=("completions", "chat"), default="chat")
    p.add_argument("--max-workers", type=int, default=128)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--start-delay-s", type=float, default=1.0)
    p.add_argument("--inter-phase-sleep-s", type=float, default=5.0)
    p.add_argument("--phase-order", default="warm,evict,replay")
    p.add_argument("--require-nonempty-phases", action="store_true")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--ignore-eos", action="store_true")
    return p.parse_args()


def install_hiddencache_imports(hiddencache_dir: Path) -> None:
    workload_dir = hiddencache_dir.resolve() / "workload_generator"
    sys.path.insert(0, str(workload_dir))


def marker(path: Path, *, run_phase: str, event_type: str, **fields: Any) -> None:
    row = {
        "source": "dmi_phase_replay",
        "event_type": event_type,
        "phase": run_phase,
        "ts_ns": now_ns(),
        "ts": time.time(),
    }
    row.update(fields)
    append_jsonl(path, row)


def normalize_phase_timestamps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    min_ts = min(float(event.get("timestamp_s", 0.0)) for event in events)
    out: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        row["original_timestamp_s"] = row.get("timestamp_s")
        row["timestamp_s"] = round(float(row.get("timestamp_s", 0.0)) - min_ts, 6)
        out.append(row)
    return sorted(out, key=lambda row: (row["timestamp_s"], row["event_id"]))


def main() -> None:
    args = parse_args()
    install_hiddencache_imports(args.hiddencache_dir)

    from replay_trace import JsonlWriter, load_prefix_bank, run_closed_loop, run_open_loop

    events = read_jsonl(args.trace)
    prefixes = load_prefix_bank(args.prefix_bank)
    phase_order = [phase.strip() for phase in args.phase_order.split(",") if phase.strip()]
    marker_path = Path(args.markers)
    writer = JsonlWriter(args.results)
    try:
        marker(marker_path, run_phase="all", event_type="replay_begin", phase_order=phase_order)
        for phase in phase_order:
            phase_events = [event for event in events if event.get("phase") == phase]
            if not phase_events:
                marker(marker_path, run_phase=phase, event_type="phase_skip", count=0)
                if args.require_nonempty_phases:
                    raise RuntimeError(f"requested phase has no events: {phase}")
                continue
            normalized = normalize_phase_timestamps(phase_events)
            marker(
                marker_path,
                run_phase=phase,
                event_type="phase_begin",
                count=len(normalized),
                replay_mode=args.mode,
                endpoint=args.endpoint,
            )
            start = time.time()
            replay_args = SimpleNamespace(**vars(args))
            if args.mode == "open-loop":
                run_open_loop(normalized, prefixes, replay_args, writer)
            else:
                run_closed_loop(normalized, prefixes, replay_args, writer)
            elapsed = time.time() - start
            marker(marker_path, run_phase=phase, event_type="phase_end", count=len(normalized), elapsed_s=elapsed)
            if phase != phase_order[-1] and args.inter_phase_sleep_s > 0:
                marker(
                    marker_path,
                    run_phase=phase,
                    event_type="inter_phase_sleep_begin",
                    sleep_s=args.inter_phase_sleep_s,
                )
                time.sleep(args.inter_phase_sleep_s)
                marker(marker_path, run_phase=phase, event_type="inter_phase_sleep_end")
        marker(marker_path, run_phase="all", event_type="replay_end")
    except Exception as exc:
        marker(marker_path, run_phase="all", event_type="replay_error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
