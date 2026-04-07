# Offline Inference Experiment Guide

This directory contains the PyTorch/HuggingFace offline evaluation scripts used in our paper. The experiments are organized into two groups:

- `offline inference`: end-to-end batched generation experiments on sampled datasets
- `microbenchmark`: targeted breakdown and storage-ablation experiments

## Directory Structure

```text
experiments/offline_inference/
  README.md
  run_*.sh                 # user-facing local experiment entrypoints
  scripts/                 # Python runners and shared utilities
  testing/                 # smoke tests and local test entrypoints
  archived/                # original sbatch/retry scripts kept for reference
```

## Baselines

The main offline inference experiments compare the following baselines.

| Baseline | Meaning | Runner |
|---|---|---|
| `hf_upper_bound_compile` | HuggingFace upper bound with compile enabled | `scripts/run_hf_upper_bound.py` |
| `hf_upper_bound_eager` | HuggingFace upper bound in eager mode | `scripts/run_hf_upper_bound.py --disable-compile` |
| `hf_monitor_generate` | HuggingFace output API path (`generate`) | `scripts/run_hf_monitor.py` |
| `hf_monitor_manual` | HuggingFace manual loop / API-based extraction | `scripts/run_hf_monitor_manual.py` |
| `proj_dmi` | DMI runner, typically with `ring_null` for end-to-end overhead studies | `scripts/run_proj_dmi.py` |
| `torch_hooks` | PyTorch forward hooks | `scripts/run_torch_hooks.py` |
| `nnsight` | NNsight tracing baseline | `scripts/run_nnsight.py` |

The microbenchmark experiments use a smaller set of step-level baselines.

| Baseline | Meaning | Runner |
|---|---|---|
| `hf_ideal` | HuggingFace ideal step timing | `scripts/run_step_breakdown_hf_ideal.py` |
| `hf_api` | HuggingFace API timing | `scripts/run_step_breakdown_hf_api.py` |
| `torch_hooks` | Torch hooks step timing | `scripts/run_step_breakdown_torch_hooks.py` |
| `proj_dmi_manual` | DMI step timing with compile | `scripts/run_step_breakdown_proj_dmi_manual.py` |
| `proj_dmi_manual` + `--disable-compile` | DMI no-ring style eager comparison | `scripts/run_step_breakdown_proj_dmi_manual.py --disable-compile` |

## Models

- `qwen3-4b` (36 layers)
- `llama3.1-8b` (32 layers)
- `qwen3-14b` (40 layers)

## 1. Offline Inference

These scripts run end-to-end batched generation on sampled ShareGPT/WildChat data and save JSON summaries.

### Main End-to-End Sweeps

| Script | Purpose |
|---|---|
| `run_full_sweep.sh` | Main `hs` sweep for Qwen3-4B and Qwen3-14B across ShareGPT/WildChat |
| `run_full_sweep_hs_logits.sh` | Qwen `hs+logits` sweep |
| `run_full_sweep_hs_logits_llama31_8b.sh` | Llama-3.1-8B `hs+logits` sweep |
| `run_full_sweep_internal_hooks.sh` | Qwen internal-hook sweep (`q,k,v,z,mlp_in,mlp_out,resid_mid`) |
| `run_full_sweep_internal_hooks_llama31_8b.sh` | Llama-3.1-8B internal-hook sweep |

### Focused Ablations

| Script | Purpose |
|---|---|
| `run_hook_count_sweep_qwen3_4b.sh` | Hook-count sweep across baselines |
| `run_hook_count_dmi_20g_qwen3_4b.sh` | DMI-only hook-count sweep |
| `run_prefill_backpressure_qwen3_4b.sh` | Request dropping / prefill backpressure study |
| `run_prefill_backpressure_dmi_ring_sweep_qwen3_4b.sh` | DMI ring-size sweep for prefill backpressure |
| `run_max_batch_hs_logits_qwen3_14b.sh` | Max-batch search for 14B `hs+logits` |
| `run_max_batch_hs_logits_qwen3_14b_compile_dmi.sh` | Max-batch search for compile HF vs DMI |
| `run_max_batch_hs_logits_qwen3_14b_dmi_eager.sh` | Max-batch search for DMI eager ring sizes |
| `run_tp_compile_sharegpt_qwen3_14b.sh` | TP=1/2/4 compile comparison for HF vs DMI on ShareGPT |

### Shared Drivers

| Script | Purpose |
|---|---|
| `run_all.sh` | Internal matrix driver used by the full sweeps |
| `run_baselines.sh` | Convenience wrapper for selected baseline subsets |
| `run_internal_hooks_compare.sh` | Helper wrapper for internal-hook comparisons |

### Example Commands

```bash
# Main offline sweep
bash experiments/offline_inference/run_full_sweep.sh

# Qwen hs+logits sweep
bash experiments/offline_inference/run_full_sweep_hs_logits.sh

# TP compile experiment
bash experiments/offline_inference/run_tp_compile_sharegpt_qwen3_14b.sh
```

Most scripts support overriding `RESULTS_DIR`, `MODEL`, `DATASETS`, `BATCH_SIZES`, or other experiment-specific environment variables before invoking `bash`.

## 2. Microbenchmark

These scripts focus on phase-level timing rather than full offline sweeps.

### Step Breakdown

| Script | Purpose |
|---|---|
| `run_step_breakdown_microbench_qwen3_4b.sh` | Main local step-breakdown microbenchmark for Qwen3-4B |
| `run_step_breakdown_microbench_qwen3_4b_local.sh` | Single local wrapper for the same workload |
| `run_step_breakdown_baseline.sh` | Baseline dispatcher for `hf_ideal`, `hf_api`, `torch_hooks`, and DMI variants |

The default step-breakdown workload is:

- synthetic prefill length `128`
- decode length `10`
- batch sizes `1, 8, 32, 64`
- 5 timed iterations after warmup

### Storage Ablation

| Script | Purpose |
|---|---|
| `run_ring_db_storage_e2e_qwen3_4b_local.sh` | Ring storage ablation (`ring_null` vs `ring_db`, disk vs `tmpfs`) |

This script measures three completion timestamps over a 10-batch run:

- `forward_done`
- `flush_done`
- `db_done`

It is used to separate:

- main generation time
- explicit ring flush time
- downstream database insertion completion time

### Example Commands

```bash
# Step breakdown microbenchmark
bash experiments/offline_inference/run_step_breakdown_microbench_qwen3_4b.sh

# Storage ablation with ClickHouse on disk
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_db \
  bash experiments/offline_inference/run_ring_db_storage_e2e_qwen3_4b_local.sh

# Lower-bound control without DB insertion
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_null \
  bash experiments/offline_inference/run_ring_db_storage_e2e_qwen3_4b_local.sh
```

## Testing and Archived Scripts

- `testing/`: smoke tests and local test entrypoints
- `archived/`: original sbatch and retry scripts kept as references; these are not the primary local reproduction path

## Notes

- The recommended local entrypoints are the top-level `run_*.sh` scripts in this directory.
- The actual Python runners live in `scripts/`.
- Existing sbatch files are preserved under `archived/` for cluster reference only.
