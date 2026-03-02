import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summaries(results_dir: str) -> dict:
    """Load all summary JSON files and extract timing data."""
    data = {}
    for f in Path(results_dir).glob("summary_*.json"):
        with open(f, "r") as handle:
            summary = json.load(handle)
        bs = summary["batch_size"]
        data[bs] = {
            "hf_total": summary["hf"]["total_seconds"],
            "nodb_main": summary["monitoring_nodb"]["main_seconds"],
            "nodb_total": summary["monitoring_nodb"]["total_seconds"],
            "db_main": summary["monitoring"]["main_seconds"] if summary["monitoring"] else None,
            "db_total": summary["monitoring"]["total_seconds"] if summary["monitoring"] else None,
        }
    return data


def plot_benchmark(data: dict, output_path: str):
    """Create bar chart comparing backends across batch sizes."""
    batch_sizes = sorted(data.keys())
    x = np.arange(len(batch_sizes))

    hf_total = [data[bs]["hf_total"] for bs in batch_sizes]
    nodb_main = [data[bs]["nodb_main"] for bs in batch_sizes]
    nodb_total = [data[bs]["nodb_total"] for bs in batch_sizes]
    db_main = [data[bs]["db_main"] for bs in batch_sizes]
    db_total = [data[bs]["db_total"] for bs in batch_sizes]

    n_bars = 5
    width = 0.12
    offsets = np.arange(n_bars) - (n_bars - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x + offsets[0] * width, hf_total, width, label="HF total", color="#1f77b4")
    ax.bar(x + offsets[1] * width, nodb_main, width, label="No-DB main", color="#2ca02c")
    ax.bar(x + offsets[2] * width, nodb_total, width, label="No-DB total", color="#98df8a")
    ax.bar(x + offsets[3] * width, db_main, width, label="DB main", color="#d62728")
    ax.bar(x + offsets[4] * width, db_total, width, label="DB total", color="#ff9896")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Benchmark: HF vs Monitoring (No-DB vs DB)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


def main():
    results_dir = "benchmark/results"
    output_path = "benchmark/figures/benchmark_comparison.png"

    data = load_summaries(results_dir)
    if not data:
        print(f"No summary files found in {results_dir}")
        return

    plot_benchmark(data, output_path)


if __name__ == "__main__":
    main()
