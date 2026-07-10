#!/usr/bin/env python3
"""Summarize one PCIe KV pressure run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import read_jsonl, stats, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default="")
    return p.parse_args()


def summarize_synthetic(run_dir: Path) -> dict[str, Any]:
    rows = [
        r
        for r in read_jsonl(run_dir / "synthetic_pcie_hog.jsonl")
        if r.get("event_type") == "copy_sample"
    ]
    dry_rows = [
        r
        for r in read_jsonl(run_dir / "synthetic_pcie_hog.jsonl")
        if r.get("event_type") == "dry_run_sample"
    ]
    by_gpu: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("gpu_index"))].append(row)
    for gpu, items in grouped.items():
        by_gpu[gpu] = {
            "samples": len(items),
            "h2d_mb_s": stats(
                [
                    r.get("h2d_mb_s")
                    for r in items
                    if isinstance(r.get("h2d_mb_s"), (int, float))
                ]
            ),
            "d2h_mb_s": stats(
                [
                    r.get("d2h_mb_s")
                    for r in items
                    if isinstance(r.get("d2h_mb_s"), (int, float))
                ]
            ),
            "h2d_gb_total": sum(float(r.get("h2d_bytes") or 0) for r in items)
            / 1024
            / 1024
            / 1024,
            "d2h_gb_total": sum(float(r.get("d2h_bytes") or 0) for r in items)
            / 1024
            / 1024
            / 1024,
        }
    return {
        "copy_samples": len(rows),
        "dry_run_samples": len(dry_rows),
        "by_gpu": by_gpu,
    }


def summarize_client(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "client_results.jsonl")
    lat = [
        float(r["latency_s"])
        for r in rows
        if isinstance(r.get("latency_s"), (int, float))
    ]
    ttft = [
        float(r["ttft_s"]) for r in rows if isinstance(r.get("ttft_s"), (int, float))
    ]
    return {"requests": len(rows), "latency_s": stats(lat), "ttft_s": stats(ttft)}


def sample_matches(
    name: str, *, any_terms: tuple[str, ...], all_terms: tuple[str, ...] = ()
) -> bool:
    lowered = name.lower()
    return all(term in lowered for term in all_terms) and any(
        term in lowered for term in any_terms
    )


def flatten_metric_rows(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            for sample in row.get("samples", []) or []:
                metric = sample.get("metric")
                value = sample.get("value")
                if isinstance(metric, str) and isinstance(value, (int, float)):
                    out.append(
                        {
                            "path": path.name,
                            "source": row.get("source"),
                            "metric": metric,
                            "value": float(value),
                            "t_rel_ms": row.get("t_rel_ms"),
                            "labels": sample.get("labels", {}),
                        }
                    )
    return out


def summarize_metrics(run_dir: Path) -> dict[str, Any]:
    paths = [
        run_dir / "vllm_metrics.jsonl",
        run_dir / "lmcache_metrics.jsonl",
        run_dir / "dmi_metrics.jsonl",
        run_dir / "metrics.jsonl",
    ]
    rows = flatten_metric_rows(paths)
    groups = {
        "dmi_drain_bandwidth_like": [],
        "capture_to_consumable_latency_like": [],
        "ring_occupancy_like": [],
    }
    for row in rows:
        name = row["metric"]
        lower = name.lower()
        is_dmi = "dmi" in lower or "dmx" in lower
        if (
            is_dmi
            and "drain" in lower
            and any(t in lower for t in ("byte", "bandwidth", "throughput"))
        ):
            groups["dmi_drain_bandwidth_like"].append(row)
        if "capture" in lower and any(
            t in lower for t in ("consumable", "consumer", "latency")
        ):
            groups["capture_to_consumable_latency_like"].append(row)
        if "ring" in lower and any(t in lower for t in ("occupancy", "fill", "util")):
            groups["ring_occupancy_like"].append(row)

    summary: dict[str, Any] = {"scraped_samples": len(rows)}
    for key, items in groups.items():
        by_metric: dict[str, list[float]] = defaultdict(list)
        for item in items:
            by_metric[str(item["metric"])].append(float(item["value"]))
        summary[key] = {
            metric: {
                "samples": len(values),
                "last": values[-1] if values else None,
                "stats": stats(values),
            }
            for metric, values in sorted(by_metric.items())
        }
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "summary.json"
    summary = {
        "run_dir": str(run_dir),
        "synthetic": summarize_synthetic(run_dir),
        "serving_client": summarize_client(run_dir),
        "prometheus_metrics": summarize_metrics(run_dir),
    }
    write_json(out, summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
