#!/usr/bin/env python3
"""Compare DMI overhead under no-move and KV-move conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import write_json
from summarize_ablation import get_nested, pct_delta, summarize_run


METRICS = {
    "replay_ttft_p50_s": ("client", "by_phase", "replay", "ttft_s", "p50"),
    "replay_ttft_p95_s": ("client", "by_phase", "replay", "ttft_s", "p95"),
    "replay_ttft_p99_s": ("client", "by_phase", "replay", "ttft_s", "p99"),
    "replay_latency_p50_s": ("client", "by_phase", "replay", "latency_s", "p50"),
    "replay_latency_p95_s": ("client", "by_phase", "replay", "latency_s", "p95"),
    "replay_latency_p99_s": ("client", "by_phase", "replay", "latency_s", "p99"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--dmi-root", required=True)
    p.add_argument("--out", default="")
    return p.parse_args()


def summarize_root(root: Path) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for name in ("no_move", "move"):
        run_dir = root / name
        if run_dir.exists():
            conditions[name] = summarize_run(run_dir)
    return {"run_root": str(root), "conditions": conditions}


def compare_metric(base: dict[str, Any], dmi: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    base_value = get_nested(base, path)
    dmi_value = get_nested(dmi, path)
    delta = None
    if isinstance(base_value, (int, float)) and isinstance(dmi_value, (int, float)):
        delta = dmi_value - base_value
    return {
        "baseline": base_value,
        "dmi": dmi_value,
        "delta_abs": delta,
        "delta_pct": pct_delta(dmi_value, base_value),
    }


def compare_conditions(baseline: dict[str, Any], dmi: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in ("no_move", "move"):
        base_cond = baseline["conditions"].get(condition)
        dmi_cond = dmi["conditions"].get(condition)
        if not base_cond or not dmi_cond:
            continue
        metrics = {
            name: compare_metric(base_cond, dmi_cond, path)
            for name, path in METRICS.items()
        }
        out[condition] = {
            "metrics": metrics,
            "kv_retrieve_reqs": {
                "baseline": get_nested(base_cond, ("kv_wait", "unique_retrieve_reqs")),
                "dmi": get_nested(dmi_cond, ("kv_wait", "unique_retrieve_reqs")),
            },
            "kv_wait_rows": {
                "baseline": get_nested(base_cond, ("kv_wait", "rows")),
                "dmi": get_nested(dmi_cond, ("kv_wait", "rows")),
            },
        }

    if "no_move" in out and "move" in out:
        amp: dict[str, Any] = {}
        for name in METRICS:
            no_delta = out["no_move"]["metrics"][name].get("delta_abs")
            move_delta = out["move"]["metrics"][name].get("delta_abs")
            amp_delta = None
            if isinstance(no_delta, (int, float)) and isinstance(move_delta, (int, float)):
                amp_delta = move_delta - no_delta
            amp[name] = {
                "dmi_overhead_no_move_abs": no_delta,
                "dmi_overhead_move_abs": move_delta,
                "extra_overhead_under_kv_move_abs": amp_delta,
            }
        out["overhead_amplification"] = amp
    return out


def main() -> None:
    args = parse_args()
    baseline = summarize_root(Path(args.baseline_root))
    dmi = summarize_root(Path(args.dmi_root))
    summary = {
        "baseline": baseline,
        "dmi": dmi,
        "comparison": compare_conditions(baseline, dmi),
    }
    out = Path(args.out) if args.out else Path(args.dmi_root) / "dmi_overhead_summary.json"
    write_json(out, summary)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
