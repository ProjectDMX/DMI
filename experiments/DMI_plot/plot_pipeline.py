#!/usr/bin/env python3
"""
Plotting Pipeline for vLLM Benchmark Results

Usage:
    python plot_pipeline.py --base_dir /path/to/base_dir --output_dir ./output

Directory structure:
    base_dir/
        baseline1/              # 子目录名称 = baseline名称
            qwen4b_sharegpt_s42_rate1.json
            qwen4b_sharegpt_s123_rate1.json
            ...
        baseline2/
            ...
        baseline_labels.json    # 可选：自定义显示名称

Optional: baseline_labels.json 格式:
    {
        "baseline1": "vLLM Baseline",
        "baseline2": "vLLM with Hooks"
    }

Output:
    - ttft_comparison.png/pdf: TTFT across all baselines
    - tpot_comparison.png/pdf: TPOT across all baselines

Each figure has 6 subplots (2x3):
    Row 1: ShareGPT - [Qwen4B, Llama8B, Qwen14B]
    Row 2: WildChat - [Qwen4B, Llama8B, Qwen14B]
"""

import json
import os
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_filename(fname):
    """
    Parse filename to extract model, dataset, seed, and rate.

    Expected format: {model}_{dataset}_s{seed}_rate{rate}.json
    Examples:
        - qwen4b_sharegpt_s42_rate16.json
        - llama8b_wildchat_s123_rate8.json

    Returns:
        (model, dataset, seed, rate) or None if parsing fails
    """
    if not fname.endswith('.json'):
        return None

    fname = fname.replace('.json', '')
    parts = fname.split('_')

    # Try to parse: model_dataset_sXXX_rateYYY
    if len(parts) >= 4:
        model = parts[0]

        # Find seed and rate positions
        seed_idx = None
        rate_idx = None
        for i, part in enumerate(parts):
            if part.startswith('s') and seed_idx is None:
                try:
                    seed = int(part[1:])
                    seed_idx = i
                except ValueError:
                    continue
            if part.startswith('rate'):
                try:
                    rate = int(part[4:])
                    rate_idx = i
                except ValueError:
                    continue

        if seed_idx is not None and rate_idx is not None:
            # Dataset is everything between model and seed
            dataset = '_'.join(parts[1:seed_idx])
            return (model, dataset, seed, rate)

    return None


def load_data_from_baseline(baseline_dir):
    """
    Load all JSON files from a baseline directory and aggregate by (model, dataset, rate).

    Returns:
        dict: {(model, dataset, rate): [run1_data, run2_data, ...]}
    """
    data = defaultdict(list)

    if not os.path.exists(baseline_dir):
        print(f"Warning: Baseline directory does not exist: {baseline_dir}")
        return data

    for fname in os.listdir(baseline_dir):
        if not fname.endswith('.json'):
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            continue

        model, dataset, seed, rate = parsed

        fpath = os.path.join(baseline_dir, fname)
        try:
            with open(fpath) as f:
                result = json.load(f)

            data[(model, dataset, rate)].append({
                'mean_ttft_ms': result.get('mean_ttft_ms', 0),
                'p99_ttft_ms': result.get('p99_ttft_ms', 0),
                'mean_tpot_ms': result.get('mean_tpot_ms', 0),
                'p99_tpot_ms': result.get('p99_tpot_ms', 0),
                'completed': result.get('completed', 0),
                'failed': result.get('failed', 0),
            })
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

    return data


def aggregate_data(data):
    """
    Aggregate data across seeds by taking the mean.

    Returns:
        dict: {(model, dataset, rate): {'mean_ttft_ms': ..., 'mean_tpot_ms': ..., ...}}
    """
    aggregated = {}

    for key, runs in data.items():
        if len(runs) == 0:
            continue

        aggregated[key] = {
            'mean_ttft_ms': np.mean([r['mean_ttft_ms'] for r in runs]),
            'p99_ttft_ms': np.mean([r['p99_ttft_ms'] for r in runs]),
            'mean_tpot_ms': np.mean([r['mean_tpot_ms'] for r in runs]),
            'p99_tpot_ms': np.mean([r['p99_tpot_ms'] for r in runs]),
            'completed': int(np.sum([r['completed'] for r in runs])),
            'failed': int(np.sum([r['failed'] for r in runs])),
            'num_seeds': len(runs)
        }

    return aggregated


def load_all_baselines(base_dir):
    """
    Load data from all baseline directories in base_dir.

    Returns:
        dict: {baseline_name: aggregated_data}
    """
    all_baselines = {}

    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        return all_baselines

    # Find all subdirectories
    subdirs = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]

    if len(subdirs) == 0:
        print(f"Warning: No subdirectories found in {base_dir}")
        return all_baselines

    for subdir in sorted(subdirs):
        baseline_path = os.path.join(base_dir, subdir)
        print(f"Loading baseline: {subdir}")

        raw_data = load_data_from_baseline(baseline_path)
        aggregated_data = aggregate_data(raw_data)

        all_baselines[subdir] = aggregated_data
        print(f"  Loaded {len(aggregated_data)} (model, dataset, rate) combinations")

    return all_baselines


def load_baseline_labels(base_dir):
    """
    Load optional baseline display labels from baseline_labels.json.

    Args:
        base_dir: directory containing baseline_labels.json

    Returns:
        dict: {baseline_dir_name: display_label} or empty dict
    """
    labels_file = os.path.join(base_dir, 'baseline_labels.json')

    if not os.path.exists(labels_file):
        return {}

    try:
        with open(labels_file) as f:
            labels = json.load(f)
        print(f"\n✓ Loaded custom baseline labels from baseline_labels.json")
        return labels
    except Exception as e:
        print(f"Warning: Failed to load baseline_labels.json: {e}")
        return {}


def plot_comparison(all_baselines, output_dir, metric='ttft', baseline_labels=None):
    """
    Create a 2x3 comparison plot for a given metric.

    Args:
        all_baselines: dict of {baseline_name: aggregated_data}
        output_dir: where to save the plots
        metric: 'ttft' or 'tpot'
        baseline_labels: dict of {baseline_dir_name: display_label} (optional)
    """
    # Initialize baseline_labels
    baseline_labels = baseline_labels or {}

    # Configuration
    models = ['qwen4b', 'llama8b', 'qwen14b']
    datasets = ['sharegpt', 'wildchat']
    model_labels = {
        'qwen4b': 'Qwen-4B',
        'llama8b': 'Llama-8B',
        'qwen14b': 'Qwen-14B'
    }

    # Color scheme for baselines
    colors = ['#2E86AB', '#F18F01', '#06A77D', '#C73E1D', '#A23B72', '#D4AF37', '#6A4C93']
    baseline_colors = {}
    for i, baseline_name in enumerate(sorted(all_baselines.keys())):
        baseline_colors[baseline_name] = colors[i % len(colors)]

    # Determine metric key
    if metric == 'ttft':
        mean_key = 'mean_ttft_ms'
        p99_key = 'p99_ttft_ms'
        ylabel = 'Time To First Token (s)'
        title_prefix = 'TTFT'
    elif metric == 'tpot':
        mean_key = 'mean_tpot_ms'
        p99_key = 'p99_tpot_ms'
        ylabel = 'Time Per Output Token (s)'
        title_prefix = 'TPOT'
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix} Comparison Across Baselines', fontsize=16, fontweight='bold', y=0.995)

    # Plot each (dataset, model) combination
    for row, dataset in enumerate(datasets):
        for col, model in enumerate(models):
            ax = axes[row, col]

            # Collect data for this (model, dataset) across all baselines
            for baseline_name, baseline_data in sorted(all_baselines.items()):
                # Extract rates and values for this model/dataset
                rates = []
                mean_values = []

                for (m, d, r), values in baseline_data.items():
                    if m == model and d == dataset:
                        rates.append(r)
                        mean_values.append(values[mean_key] / 1000)  # Convert ms to s

                if len(rates) > 0:
                    # Sort by rate
                    sorted_pairs = sorted(zip(rates, mean_values))
                    rates, mean_values = zip(*sorted_pairs)

                    # Get display label (use custom label if available, else use dir name)
                    display_label = baseline_labels.get(baseline_name, baseline_name)

                    # Plot
                    ax.plot(rates, mean_values, 'o-', linewidth=2, markersize=7,
                           label=display_label, color=baseline_colors[baseline_name])

            # Configure subplot
            ax.set_xlabel('Request Rate (req/s)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{dataset.capitalize()} - {model_labels[model]}',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Use log scale for x-axis with power-of-2 ticks only
            ax.set_xscale('log', base=2)
            pow2_ticks = [2**i for i in range(0, 9)]  # 1,2,4,8,16,32,64,128,256
            ax.set_xticks(pow2_ticks)
            ax.set_xticklabels([str(t) for t in pow2_ticks])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # Place legend outside the subplots at the top
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=min(len(labels), 5),
               fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, f'{metric}_comparison.png')
    output_path_pdf = os.path.join(output_dir, f'{metric}_comparison.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"\nSaved {metric.upper()} comparison to:")
    print(f"  {output_path_png}")
    print(f"  {output_path_pdf}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot vLLM benchmark comparison across baselines')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing baseline subdirectories')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for plots (default: ./output)')

    args = parser.parse_args()

    print(f"Loading data from: {args.base_dir}")
    all_baselines = load_all_baselines(args.base_dir)

    if len(all_baselines) == 0:
        print("Error: No baselines found!")
        return

    # Load custom baseline labels if available
    baseline_labels = load_baseline_labels(args.base_dir)

    print(f"\nFound {len(all_baselines)} baselines:")
    for baseline_name in sorted(all_baselines.keys()):
        display_name = baseline_labels.get(baseline_name, baseline_name)
        if baseline_name in baseline_labels:
            print(f"  - {baseline_name} → \"{display_name}\"")
        else:
            print(f"  - {baseline_name}")

    # Plot TTFT comparison
    print("\nGenerating TTFT comparison plot...")
    plot_comparison(all_baselines, args.output_dir, metric='ttft', baseline_labels=baseline_labels)

    # Plot TPOT comparison
    print("\nGenerating TPOT comparison plot...")
    plot_comparison(all_baselines, args.output_dir, metric='tpot', baseline_labels=baseline_labels)

    print("\nDone!")


if __name__ == '__main__':
    main()
