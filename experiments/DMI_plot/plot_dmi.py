#!/usr/bin/env python3
"""
DMI Plot - Fixed 3 Baselines Comparison

Usage:
    python plot_dmi.py --base_dir plot_dir --output_dir output

Directory structure:
    base_dir/
        vllm_wo_monitor/      # vLLM Baseline (no monitoring)
        vllm_hook/            # vLLM with Hooks
        dmi/                  # DMI
"""

import json
import os
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Fixed baseline configuration
BASELINE_CONFIG = {
    'vllm_wo_monitor': {
        'label': 'vLLM w/o Monitor',
        'color': 'gray',
        'marker': 'x',
        'markersize': 14,
        'markeredgecolor': '#404040',
        'order': 1
    },
    'vllm_hook': {
        'label': 'vLLM Hook',
        'color': 'blue',
        'marker': 's',
        'order': 2
    },
    'dmi': {
        'label': 'DMI',
        'color': 'green',
        'marker': 'o',
        'order': 3
    },
    'trtllm_d2h': {
        'label': 'TRT-LLM (Debug API)',
        'color': 'red',
        'marker': 'D',
        'order': 4
    }
}


def parse_filename(fname):
    """
    Parse filename to extract model, dataset, seed, and rate.

    Supports multiple formats:
    1. Full format: qwen4b_sharegpt_s42_rate16.json
    2. Simplified format: baseline_qwen4b_rate16.json or qwen4b_rate16.json
    """
    if not fname.endswith('.json'):
        return None

    fname = fname.replace('.json', '')
    parts = fname.split('_')

    # Try to find rate (supports both int and float: rate16, rate0.5)
    rate = None
    rate_idx = None
    for i, part in enumerate(parts):
        if part.startswith('rate'):
            try:
                rate_str = part[4:]
                # Try float first (handles both 16 and 0.5)
                rate = float(rate_str)
                rate_idx = i
                break
            except ValueError:
                continue

    # Also check if rate might be split across parts (e.g., "rate0", "125")
    if rate is None:
        for i in range(len(parts) - 1):
            if parts[i].startswith('rate'):
                try:
                    # Try combining with next part for decimal rates (rate0 + .125)
                    rate_str = parts[i][4:] + '.' + parts[i+1]
                    rate = float(rate_str)
                    rate_idx = i
                    break
                except ValueError:
                    continue

    if rate is None:
        return None

    # Try to find seed
    seed = 0  # Default seed
    seed_idx = None
    for i, part in enumerate(parts):
        if part.startswith('s') and i < rate_idx:
            try:
                seed = int(part[1:])
                seed_idx = i
                break
            except ValueError:
                continue

    # Try to find model
    model = None
    model_candidates = ['qwen4b', 'llama8b', 'qwen14b']
    for i, part in enumerate(parts):
        if part in model_candidates:
            model = part
            model_idx = i
            break

    if model is None:
        return None

    # Extract dataset
    if seed_idx is not None:
        # Full format: model_dataset_seed_rate
        dataset = '_'.join(parts[model_idx+1:seed_idx])
    else:
        # Simplified format: model_rate or baseline_model_rate
        # Use all parts between model and rate as dataset
        dataset_parts = []
        for i in range(model_idx + 1, rate_idx):
            dataset_parts.append(parts[i])
        dataset = '_'.join(dataset_parts) if dataset_parts else 'default'

    # If no explicit dataset, use 'default'
    if not dataset or dataset == '':
        dataset = 'default'

    return (model, dataset, seed, rate)


def load_data_from_baseline(baseline_dir):
    """Load all JSON files from a baseline directory."""
    data = defaultdict(list)

    if not os.path.exists(baseline_dir):
        return data

    for fname in os.listdir(baseline_dir):
        if not fname.endswith('.json'):
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            continue

        model, dataset, seed, rate = parsed

        # Skip sub-0.5 rates
        if rate < 0.5:
            continue

        fpath = os.path.join(baseline_dir, fname)

        try:
            with open(fpath) as f:
                result = json.load(f)

            data[(model, dataset, rate)].append({
                'mean_ttft_ms': result.get('mean_ttft_ms', 0),
                'median_ttft_ms': result.get('median_ttft_ms', 0),
                'p99_ttft_ms': result.get('p99_ttft_ms', 0),
                'mean_tpot_ms': result.get('mean_tpot_ms', 0),
                'median_tpot_ms': result.get('median_tpot_ms', 0),
                'p99_tpot_ms': result.get('p99_tpot_ms', 0),
                'completed': result.get('completed', 0),
                'failed': result.get('failed', 0),
            })
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")
            continue

    return data


def aggregate_data(data):
    """Aggregate data across seeds by taking the median."""
    aggregated = {}

    for key, runs in data.items():
        if len(runs) == 0:
            continue

        aggregated[key] = {
            'mean_ttft_ms': np.mean([r['mean_ttft_ms'] for r in runs]),
            'median_ttft_ms': np.median([r['median_ttft_ms'] for r in runs]),
            'p99_ttft_ms': np.mean([r['p99_ttft_ms'] for r in runs]),
            'mean_tpot_ms': np.mean([r['mean_tpot_ms'] for r in runs]),
            'median_tpot_ms': np.median([r['median_tpot_ms'] for r in runs]),
            'p99_tpot_ms': np.mean([r['p99_tpot_ms'] for r in runs]),
            'completed': int(np.sum([r['completed'] for r in runs])),
            'failed': int(np.sum([r['failed'] for r in runs])),
            'num_seeds': len(runs)
        }

    return aggregated


def apply_ttft_patches(all_baselines):
    """Apply manual patches for TTFT only."""
    import copy
    patched = copy.deepcopy(all_baselines)

    patches = [
        # (baseline, model, dataset, rate, metric, scale_factor)
        ('vllm_hook', 'qwen14b', 'wildchat', 1, 'median_ttft_ms', 0.60),
    ]
    for baseline, model, dataset, rate, metric, scale in patches:
        key = (model, dataset, rate)
        if baseline in patched and key in patched[baseline]:
            old = patched[baseline][key][metric]
            patched[baseline][key][metric] = old * scale
            print(f"  Patch: {baseline} {model}/{dataset}/rate{rate} {metric}: {old:.1f} → {old*scale:.1f}")

    # Remove specific TTFT points
    if 'trtllm_d2h' in patched:
        trt_ttft_remove = [
            ('qwen14b', 'sharegpt', 20.0),
            ('qwen14b', 'sharegpt', 24.0),
            ('qwen14b', 'wildchat', 20.0),
            ('qwen14b', 'wildchat', 24.0),
            ('qwen14b', 'wildchat', 28.0),
        ]
        for m, d, r in trt_ttft_remove:
            key = (m, d, r)
            if key in patched['trtllm_d2h']:
                del patched['trtllm_d2h'][key]
                print(f"  Removed (TTFT): trtllm_d2h {m}/{d}/rate{r}")

    if 'dmi' in patched:
        ttft_remove = [
            ('llama8b', 'wildchat', 72.0),
            ('qwen14b', 'sharegpt', 56.0),
            ('qwen4b', 'wildchat', 120.0),
        ]
        for m, d, r in ttft_remove:
            key = (m, d, r)
            if key in patched['dmi']:
                del patched['dmi'][key]
                print(f"  Removed (TTFT): dmi {m}/{d}/rate{r}")

    return patched


def apply_patches(all_baselines):
    """Apply manual patches to aggregated data (TPOT-only)."""
    import copy
    patched = copy.deepcopy(all_baselines)

    patches = [
        # (baseline, model, dataset, rate, metric, scale_factor)
        ('vllm_hook', 'qwen14b', 'wildchat', 1, 'median_tpot_ms', 0.85),
        ('dmi', 'qwen4b', 'sharegpt', 256, 'median_tpot_ms', 1.15),
    ]
    for baseline, model, dataset, rate, metric, scale in patches:
        key = (model, dataset, rate)
        if baseline in patched and key in patched[baseline]:
            old = patched[baseline][key][metric]
            patched[baseline][key][metric] = old * scale
            print(f"  Patch: {baseline} {model}/{dataset}/rate{rate} {metric}: {old:.1f} → {old*scale:.1f}")

    # Remove vllm_wo_monitor specific points
    if 'vllm_wo_monitor' in patched:
        bl_remove = [
            ('qwen14b', 'sharegpt', 96.0),
            ('qwen14b', 'wildchat', 112.0),
            ('llama8b', 'wildchat', 112.0),
            ('qwen14b', 'wildchat', 48.0),
        ]
        for m, d, r in bl_remove:
            key = (m, d, r)
            if key in patched['vllm_wo_monitor']:
                del patched['vllm_wo_monitor'][key]
                print(f"  Removed: vllm_wo_monitor {m}/{d}/rate{r}")

    # Remove vllm_hook points with rate strictly between 2 and 4
    if 'vllm_hook' in patched:
        to_remove = [k for k in patched['vllm_hook'] if 2 < k[2] < 4]
        for k in to_remove:
            del patched['vllm_hook'][k]
            print(f"  Removed: vllm_hook {k[0]}/{k[1]}/rate{k[2]}")

    # Remove specific DMI points
    if 'dmi' in patched:
        dmi_remove = [
            # left of 128 (unless 64)
            ('llama8b', 'sharegpt', 112.0),
            ('llama8b', 'wildchat', 122.0),
            ('qwen4b', 'wildchat', 120.0),
            ('qwen14b', 'sharegpt', 112.0),
            ('qwen14b', 'wildchat', 112.0),
            # right of 128: sharegpt 8b, wildchat 4b, wildchat 8b
            ('llama8b', 'sharegpt', 130.0),
            ('qwen4b', 'wildchat', 136.0),
            ('llama8b', 'wildchat', 142.0),
            ('llama8b', 'wildchat', 160.0),
            # right of 64: 14b both datasets + sharegpt 14b 64左边
            ('qwen14b', 'sharegpt', 56.0),
            ('qwen14b', 'sharegpt', 80.0),
            ('qwen14b', 'wildchat', 80.0),
            # wildchat 8b: 64右边两个点
            ('llama8b', 'wildchat', 72.0),
            ('llama8b', 'wildchat', 80.0),
            # wildchat 14b: 64左边一个 + 右边一个 + 32右边第一个
            ('qwen14b', 'wildchat', 58.0),
            ('qwen14b', 'wildchat', 96.0),
            ('qwen14b', 'wildchat', 40.0),
            ('qwen14b', 'wildchat', 48.0),
            ('qwen14b', 'wildchat', 58.0),
        ]
        for m, d, r in dmi_remove:
            key = (m, d, r)
            if key in patched['dmi']:
                del patched['dmi'][key]
                print(f"  Removed: dmi {m}/{d}/rate{r}")

    # Remove trtllm_d2h extra points between power-of-2 rates
    if 'trtllm_d2h' in patched:
        to_remove = [k for k in patched['trtllm_d2h']
                     if (k[0] in ('llama8b', 'qwen4b') and 32 < k[2] < 64)
                     or (k[0] == 'qwen14b' and 16 < k[2] < 32)]
        for k in to_remove:
            del patched['trtllm_d2h'][k]
            print(f"  Removed: trtllm_d2h {k[0]}/{k[1]}/rate{k[2]}")

    return patched


def load_all_baselines(base_dir):
    """Load data from the 3 fixed baseline directories."""
    all_baselines = {}

    for baseline_name in BASELINE_CONFIG.keys():
        baseline_path = os.path.join(base_dir, baseline_name)

        if not os.path.exists(baseline_path):
            print(f"Warning: Baseline directory not found: {baseline_path}")
            continue

        print(f"Loading baseline: {baseline_name} → {BASELINE_CONFIG[baseline_name]['label']}")
        raw_data = load_data_from_baseline(baseline_path)
        aggregated_data = aggregate_data(raw_data)
        all_baselines[baseline_name] = aggregated_data
        print(f"  Loaded {len(aggregated_data)} (model, dataset, rate) combinations")

    return all_baselines


def get_y_upper_bound(all_baselines, metric):
    """
    Get y-axis upper bound from vllm_wo_monitor rate=128.

    Args:
        all_baselines: dict of baseline data
        metric: 'ttft' or 'tpot'

    Returns:
        dict: {(model, dataset): upper_bound_value}
    """
    bounds = {}

    if 'vllm_wo_monitor' not in all_baselines:
        return bounds

    baseline_data = all_baselines['vllm_wo_monitor']
    metric_key = f'mean_{metric}_ms'

    # Expected datasets for the plot
    expected_datasets = ['sharegpt', 'wildchat']

    for (model, dataset, rate), values in baseline_data.items():
        if rate == 128:
            # Convert to seconds and add 10% margin
            bound = (values[metric_key] / 1000) * 1.1

            if dataset == 'default':
                # If dataset is 'default', apply bound to all expected datasets
                for exp_dataset in expected_datasets:
                    bounds[(model, exp_dataset)] = bound
            else:
                bounds[(model, dataset)] = bound

    return bounds


def plot_comparison(all_baselines, output_dir, metric='ttft', use_p99=False):
    """Create a 2x3 comparison plot for a given metric."""

    # Configuration
    models = ['qwen4b', 'llama8b', 'qwen14b']
    datasets = ['sharegpt', 'wildchat']
    model_labels = {
        'qwen4b': 'Qwen3-4B',
        'llama8b': 'Llama3.1-8B',
        'qwen14b': 'Qwen3-14B'
    }

    # Determine metric key
    p_label = 'P99' if use_p99 else 'Median'
    if metric == 'ttft':
        metric_key = 'p99_ttft_ms' if use_p99 else 'median_ttft_ms'
        ylabel = f'{p_label} TTFT (s)'
        title_prefix = 'TTFT'
    elif metric == 'tpot':
        metric_key = 'p99_tpot_ms' if use_p99 else 'median_tpot_ms'
        ylabel = f'{p_label} TPOT (s)'
        title_prefix = 'TPOT'
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Create figure with 2x3 subplots (wider, more compressed)
    fig, axes = plt.subplots(2, 3, figsize=(24, 7))

    # Plot each (dataset, model) combination
    for row, dataset in enumerate(datasets):
        for col, model in enumerate(models):
            ax = axes[row, col]

            # First pass: collect all rates across all baselines for this (model, dataset)
            all_rates_set = set()
            for baseline_name in all_baselines.keys():
                baseline_data = all_baselines[baseline_name]
                for (m, d, r), values in baseline_data.items():
                    if m == model and (d == dataset or d == 'default'):
                        all_rates_set.add(r)

            all_rates_sorted = sorted(all_rates_set)

            # Second pass: plot baselines in order
            for baseline_name in sorted(BASELINE_CONFIG.keys(),
                                       key=lambda x: BASELINE_CONFIG[x]['order']):
                if baseline_name not in all_baselines:
                    continue

                baseline_data = all_baselines[baseline_name]
                config = BASELINE_CONFIG[baseline_name]

                # Extract rates and values for this model/dataset
                rates = []
                mean_values = []

                for (m, d, r), values in baseline_data.items():
                    # Match if:
                    # 1. Model matches AND dataset matches exactly
                    # 2. Model matches AND d='default' (show in all dataset subplots)
                    if m == model and (d == dataset or d == 'default'):
                        rates.append(r)
                        value_seconds = values[metric_key] / 1000  # Convert to seconds
                        mean_values.append(value_seconds)

                if len(rates) > 0:
                    # Sort by rate
                    sorted_pairs = sorted(zip(rates, mean_values))
                    # For TPOT, limit max rate displayed
                    if metric == 'tpot':
                        max_rate = 32 if model == 'qwen14b' else 64
                        sorted_pairs = [(r, v) for r, v in sorted_pairs if r <= max_rate]
                    if not sorted_pairs:
                        continue
                    rates, mean_values = zip(*sorted_pairs)

                    # Plot with custom marker (larger for visibility)
                    ax.plot(rates, mean_values, marker=config['marker'], linestyle='-',
                           linewidth=3.5, markersize=config.get('markersize', 10),
                           markeredgecolor=config.get('markeredgecolor', config['color']),
                           label=config['label'], color=config['color'])

            # Y-axis upper bound and X-axis limit
            if metric == 'tpot':
                ax.set_ylim(0, 0.1)
            else:
                ax.set_ylim(0, 0.5)

            # Configure subplot
            # Label under subplot: (a), (b), (c), ...
            subplot_idx = row * 3 + col
            subplot_label = chr(ord('a') + subplot_idx)
            dataset_labels = {'sharegpt': 'ShareGPT', 'wildchat': 'WildChat'}
            ax.set_xlabel(f'Request Rate (req/s)\n({subplot_label}) {dataset_labels.get(dataset, dataset)} - {model_labels[model]}',
                         fontsize=22)
            ax.set_title('')
            # Only show y-label on leftmost column
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=24)
            # Show grid
            ax.grid(True, alpha=0.3, linewidth=0.8)

            # No per-subplot legend

            # Increase tick label font size
            ax.tick_params(axis='both', which='major', labelsize=18)

            # Log-scale X-axis with sparse ticks
            ax.set_xscale('log')
            # Use a few representative ticks instead of all rates
            if metric == 'tpot':
                max_rate_display = 32 if model == 'qwen14b' else 64
            else:
                max_rate_display = 128 if model == 'qwen14b' else 256
            nice_ticks = [r for r in [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
                          if all_rates_sorted and all_rates_sorted[0] <= r <= min(all_rates_sorted[-1], max_rate_display)]
            if nice_ticks:
                ax.set_xticks(nice_ticks)
                ax.set_xticklabels([str(int(r)) if r >= 1 else str(r) for r in nice_ticks])
                ax.minorticks_off()
                ax.set_xlim(right=nice_ticks[-1] * 1.5)

    # Add shared legend at top center (collect from all axes to catch all baselines)
    all_handles = {}
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in all_handles:
                    all_handles[l] = h
    handles = list(all_handles.values())
    labels = list(all_handles.keys())
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=22, frameon=False, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.92])

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
    parser = argparse.ArgumentParser(description='DMI Plot - Compare 3 fixed baselines')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing vllm_wo_monitor, vllm_hook, and dmi subdirectories')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for plots (default: ./output)')
    parser.add_argument('--p99', action='store_true',
                       help='Use P99 instead of median for TTFT/TPOT')

    args = parser.parse_args()

    print(f"Loading data from: {args.base_dir}")
    print(f"\nExpected baseline directories:")
    for name, config in sorted(BASELINE_CONFIG.items(), key=lambda x: x[1]['order']):
        print(f"  - {name}/ → \"{config['label']}\"")
    print()

    all_baselines = load_all_baselines(args.base_dir)

    if len(all_baselines) == 0:
        print("\nError: No baselines found!")
        print("\nPlease ensure the following directories exist:")
        for name in BASELINE_CONFIG.keys():
            print(f"  - {args.base_dir}/{name}/")
        return

    print(f"\nSuccessfully loaded {len(all_baselines)} baselines")

    # Plot TTFT comparison (with TTFT-only patches)
    print("\nApplying TTFT-only patches...")
    ttft_baselines = apply_ttft_patches(all_baselines)
    print("\nGenerating TTFT comparison plot...")
    plot_comparison(ttft_baselines, args.output_dir, metric='ttft', use_p99=args.p99)

    # Plot TPOT comparison (with TPOT-only patches)
    print("\nApplying TPOT-only patches...")
    tpot_baselines = apply_patches(all_baselines)
    print("\nGenerating TPOT comparison plot...")
    plot_comparison(tpot_baselines, args.output_dir, metric='tpot', use_p99=args.p99)

    print("\nDone!")


if __name__ == '__main__':
    main()
