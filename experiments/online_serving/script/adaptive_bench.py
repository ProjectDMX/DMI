#!/usr/bin/env python3
"""
Adaptive inflection-point benchmark.

Phase 1: Binary search with ONE probe dataset to find the rate where
          median TTFT crosses a threshold (default 1000ms).
Phase 2: Run ALL 6 datasets at rates densely around the inflection.

Usage (called from sbatch scripts):
    python3 adaptive_bench.py \
        --base-url http://localhost:8010 \
        --model <MODEL_PATH> \
        --tag qwen4b \
        --result-dir ttft_results \
        --low-rate 64 --high-rate 256 \
        --threshold 1000 \
        --resolution 4 \
        --duration 30 \
        --cooldown 30 \
        --result-prefix ""
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

# ── dataset definitions ──────────────────────────────────────────────
DATASETS = [
    ("sampled_datasets/sharegpt_seed42_n500_n30.json",  "sharegpt_s42"),
    ("sampled_datasets/sharegpt_seed123_n500_n30.json", "sharegpt_s123"),
    ("sampled_datasets/sharegpt_seed456_n500_n30.json", "sharegpt_s456"),
    ("sampled_datasets/wildchat_seed42_n500_n30.json",  "wildchat_s42"),
    ("sampled_datasets/wildchat_seed123_n500_n30.json", "wildchat_s123"),
    ("sampled_datasets/wildchat_seed456_n500_n30.json", "wildchat_s456"),
]

PROBE_DS = DATASETS[0]  # sharegpt_s42 — usually the most stable


def run_one(args, rate, ds_path, ds_tag, warmup=50):
    """Run benchmark at a single rate+dataset, return parsed JSON dict."""
    np_count = max(1, int(math.ceil(rate * args.duration)))
    fname = f"{args.result_prefix}{args.tag}_{ds_tag}_rate{rate}.json"
    outpath = os.path.join(args.result_dir, fname)

    cmd = [
        sys.executable, "run_bench.py",
        "--dataset-name", "sharegpt",
        "--dataset-path", ds_path,
        "--backend", "openai",
        "--base-url", args.base_url,
        "--model", args.model,
        "--sharegpt-output-len", "128",
        "--request-rate", str(rate),
        "--num-prompts", str(np_count),
        "--num-warmups", str(warmup),
        "--save-result",
        "--result-dir", args.result_dir,
        "--result-filename", fname,
    ]

    print(f"  ▶ rate={rate}  np={np_count}  → {fname}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.0f}s", flush=True)

    with open(outpath) as f:
        return json.load(f)


def get_ttft(result):
    return result.get("median_ttft_ms", float("inf"))


def snap_rate(r, fractional=False):
    """Round rate to a 'nice' value."""
    if fractional:
        # For vLLM-Hook: snap to 0.25 increments
        return round(r * 4) / 4
    else:
        # For baseline/DMI: snap to integer
        return max(1, round(r))


def binary_search(args, lo, hi, threshold, fractional=False):
    """
    Binary search for the inflection rate on the probe dataset.

    If --multiplier is set, the threshold is computed dynamically:
      threshold = TTFT(lo_rate) × multiplier
    This captures the "knee" where latency starts rising, rather than
    hunting for an absolute ms value that may already be deep in
    saturation.

    Returns (inflection_rate, all_probe_results_dict).
    """
    ds_path, ds_tag = PROBE_DS
    results = {}  # rate -> median_ttft

    # Run lo boundary first
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 1: Binary search  lo={lo}  hi={hi}", flush=True)

    time.sleep(args.cooldown)
    lo_result = run_one(args, lo, ds_path, ds_tag)
    results[lo] = get_ttft(lo_result)
    print(f"    lo={lo} → TTFT={results[lo]:.1f}ms", flush=True)

    # Compute threshold: use multiplier if set, otherwise absolute
    if args.multiplier is not None:
        threshold = results[lo] * args.multiplier
        print(f"    threshold = {results[lo]:.1f}ms × {args.multiplier} = {threshold:.1f}ms", flush=True)
    else:
        print(f"    threshold = {threshold:.1f}ms (absolute)", flush=True)

    print(f"{'='*60}", flush=True)

    time.sleep(args.cooldown)
    hi_result = run_one(args, hi, ds_path, ds_tag)
    results[hi] = get_ttft(hi_result)
    print(f"    hi={hi} → TTFT={results[hi]:.1f}ms", flush=True)

    if results[lo] >= threshold:
        print(f"  ⚠ lo rate already above threshold! Inflection is below {lo}", flush=True)
        return lo, results
    if results[hi] < threshold:
        print(f"  ⚠ hi rate still below threshold! Inflection is above {hi}", flush=True)
        return hi, results

    # Binary search
    iteration = 0
    while (hi - lo) > args.resolution:
        mid = snap_rate((lo + hi) / 2, fractional)
        if mid == lo or mid == hi:
            break
        iteration += 1
        print(f"\n  Iteration {iteration}: trying rate={mid} (bracket [{lo}, {hi}])", flush=True)

        time.sleep(args.cooldown)
        mid_result = run_one(args, mid, ds_path, ds_tag)
        results[mid] = get_ttft(mid_result)
        print(f"    rate={mid} → TTFT={results[mid]:.1f}ms", flush=True)

        if results[mid] >= threshold:
            hi = mid
        else:
            lo = mid

    inflection = hi  # first rate above threshold
    print(f"\n  ✓ Inflection at rate≈{inflection} (bracket [{lo}, {hi}])", flush=True)
    return inflection, results


def choose_sweep_rates(inflection, probe_results, args, fractional=False):
    """
    Pick ~5-7 rates around the inflection for the full 6-dataset sweep.
    Include 2-3 rates below and 2-3 above the inflection.
    Skip rates we already have good data for in plot_dir.
    """
    # Build candidate rates around inflection
    if fractional:
        step = max(0.25, args.resolution / 2)
        candidates = set()
        r = inflection
        for _ in range(4):
            r -= step
            if r > 0:
                candidates.add(snap_rate(r, True))
        r = inflection
        for _ in range(3):
            candidates.add(snap_rate(r, True))
            r += step
    else:
        step = max(2, args.resolution // 2)
        candidates = set()
        r = inflection
        for _ in range(4):
            r -= step
            if r > 0:
                candidates.add(snap_rate(r, False))
        r = inflection
        for _ in range(3):
            candidates.add(snap_rate(r, False))
            r += step

    # Always include inflection itself and boundary probes
    candidates.add(inflection)
    for rate in probe_results:
        candidates.add(rate)

    rates = sorted(candidates)
    print(f"\n  Sweep rates: {rates}", flush=True)
    return rates


def full_sweep(args, rates):
    """Run all 6 datasets at each rate."""
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 2: Full sweep at {len(rates)} rates × {len(DATASETS)} datasets", flush=True)
    print(f"{'='*60}", flush=True)

    for ds_path, ds_tag in DATASETS:
        print(f"\n--- Dataset: {ds_tag} ---", flush=True)
        for rate in rates:
            fname = f"{args.result_prefix}{args.tag}_{ds_tag}_rate{rate}.json"
            outpath = os.path.join(args.result_dir, fname)
            time.sleep(args.cooldown)
            run_one(args, rate, ds_path, ds_tag)


def main():
    parser = argparse.ArgumentParser(description="Adaptive inflection-point benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True, help="e.g. qwen4b, dmil_llama8b")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--result-prefix", default="", help="e.g. 'dmil_' for DMI files")
    parser.add_argument("--low-rate", type=float, required=True, help="Known-good rate (low TTFT)")
    parser.add_argument("--high-rate", type=float, required=True, help="Known-bad rate (high TTFT)")
    parser.add_argument("--threshold", type=float, default=1000,
                        help="Absolute TTFT threshold in ms (default: 1000). "
                             "Ignored if --multiplier is set.")
    parser.add_argument("--multiplier", type=float, default=None,
                        help="Relative threshold: inflection = lo_rate_TTFT × multiplier. "
                             "E.g. --multiplier 5 finds where TTFT is 5× the low-rate baseline. "
                             "Overrides --threshold when set.")
    parser.add_argument("--resolution", type=float, default=4,
                        help="Stop binary search when bracket < this (default: 4 req/s)")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--cooldown", type=int, default=30)
    parser.add_argument("--fractional", action="store_true",
                        help="Allow fractional rates (for vLLM-Hook)")

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    inflection, probe_results = binary_search(
        args, args.low_rate, args.high_rate, args.threshold, args.fractional
    )

    rates = choose_sweep_rates(inflection, probe_results, args, args.fractional)
    full_sweep(args, rates)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Inflection ≈ rate {inflection}", flush=True)
    print(f"Results in {args.result_dir}/", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
