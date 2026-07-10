#!/usr/bin/env python3
"""Aggregate repeated DMI-overhead comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import write_json
from summarize_dmi_overhead import compare_conditions, summarize_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="LABEL:BASELINE_ROOT:DMI_ROOT",
        help="Repeated comparison pair. Use the same LABEL for repeated trials.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_case(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise SystemExit(f"bad --case {value!r}; expected LABEL:BASELINE_ROOT:DMI_ROOT")
    return parts[0], Path(parts[1]), Path(parts[2])


def mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "values": values,
    }


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    kv_retrieve: list[dict[str, Any]] = []
    for trial in trials:
        cond = trial.get("comparison", {}).get("no_move", {})
        if cond.get("kv_retrieve_reqs") is not None:
            kv_retrieve.append(cond["kv_retrieve_reqs"])
        metrics = cond.get("metrics", {})
        for metric_name, row in metrics.items():
            for field in ("baseline", "dmi", "delta_abs", "delta_pct"):
                value = row.get(field)
                if isinstance(value, (int, float)):
                    by_metric[metric_name][field].append(float(value))

    return {
        "n_trials": len(trials),
        "kv_retrieve_reqs": kv_retrieve,
        "metrics": {
            metric: {field: mean_std(values) for field, values in sorted(fields.items())}
            for metric, fields in sorted(by_metric.items())
        },
    }


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trial_details: list[dict[str, Any]] = []

    for raw_case in args.case:
        label, baseline_root, dmi_root = parse_case(raw_case)
        baseline = summarize_root(baseline_root)
        dmi = summarize_root(dmi_root)
        comparison = compare_conditions(baseline, dmi)
        trial = {
            "label": label,
            "baseline_root": str(baseline_root),
            "dmi_root": str(dmi_root),
            "comparison": comparison,
        }
        grouped[label].append(trial)
        trial_details.append(trial)

    summary = {
        "trials": trial_details,
        "aggregate": {label: aggregate_trials(trials) for label, trials in sorted(grouped.items())},
    }
    write_json(Path(args.out), summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
