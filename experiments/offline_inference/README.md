# Offline Inference/Microbenchmarks Experiment Guide

This directory contains the PyTorch/HuggingFace offline evaluation scripts used in our paper. The experiments are organized into two groups:

- `offline inference`: end-to-end batched generation experiments on sampled datasets
- `microbenchmark`: targeted breakdown and ablation experiments

## Directory Structure

```text
experiments/offline_inference/
  README.md
  run_*.sh                 # user-facing local experiment entrypoints
  scripts/                 # Python runners and internal helper scripts
  testing/                 # smoke tests and local test entrypoints
  archived/                # original sbatch/retry scripts kept for reference
```

## Baselines

The main offline inference experiments compare the following baselines.

| Baseline | Description | Runner |
|---|---|---|
| **Hugging Face (compiled baseline)** | Unmonitored HuggingFace baseline with compile enabled | `scripts/run_hf_upper_bound.py` |
| **Hugging Face (eager baseline)** | Unmonitored HuggingFace baseline in eager mode | `scripts/run_hf_upper_bound.py --disable-compile` |
| **DMI** | DMI monitoring integration with Hugging Face | `scripts/run_proj_dmi.py` |
| **Hugging Face (generate)** | HuggingFace with offloading using output API path (`generate`) | `scripts/run_hf_monitor.py` |
| **Hugging Face (manual)** | HuggingFace with offloading using manual loop | `scripts/run_hf_monitor_manual.py` |
| **Torch Hooks** | PyTorch forward hooks | `scripts/run_torch_hooks.py` |
| **NNsight** | NNsight tracing baseline | `scripts/run_nnsight.py` |

The microbenchmark experiments use a smaller set of step-level baselines.

| Baseline | Meaning | Runner |
|---|---|---|
| **Hugging Face (baseline)** | Unmonitored HuggingFace baseline step timing | `scripts/run_step_breakdown_hf_ideal.py` |
| **DMI (full)** | DMI full pipeline step timing with compile | `scripts/run_step_breakdown_proj_dmi_manual.py` |
| **DMI (no ring)** | DMI without ring step timing (fall back to eager) | `scripts/run_step_breakdown_proj_dmi_manual.py --disable-compile` |
| **Hugging Face (API)** | HuggingFace with offloading using output API path step timing | `scripts/run_step_breakdown_hf_api.py` |
| **Torch Hooks** | Torch hooks step timing | `scripts/run_step_breakdown_torch_hooks.py` |


## Models

- `qwen3-4b` (36 layers)
- `llama3.1-8b` (32 layers)
- `qwen3-14b` (40 layers)

## 1. Offline Inference

These scripts run end-to-end batched generation on sampled ShareGPT/WildChat data and save JSON summaries.

### Main End-to-End Sweeps

| Script | Purpose |
|---|---|
| `run_full_sweep_hs_logits.sh` | Qwen `hs+logits` sweep |
| `run_full_sweep_hs_logits_llama31_8b.sh` | Llama-3.1-8B `hs+logits` sweep |
| `run_full_sweep_internal_hooks.sh` | Qwen internal-hook sweep (`q,k,v,z,mlp_in,mlp_out,resid_mid`) |
| `run_full_sweep_internal_hooks_llama31_8b.sh` | Llama-3.1-8B internal-hook sweep |

### Focused Ablations

| Script | Purpose |
|---|---|
| `run_hook_count_sweep_qwen3_4b.sh` | Hook-count sweep across baselines, using the 20G DMI configuration used in the final figure |
| `run_prefill_backpressure_qwen3_4b.sh` | Request dropping / prefill backpressure study, including the DMI ring-size and hook-selection sweep used for the final figure |
| `run_tp_compile_sharegpt_qwen3_14b.sh` | TP=1/2/4 compile comparison for HF vs DMI on ShareGPT |

### Example Commands

```bash
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
| `run_step_breakdown_microbench_qwen3_4b_local.sh` | Main local step-breakdown microbenchmark for Qwen3-4B |

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
bash experiments/offline_inference/run_step_breakdown_microbench_qwen3_4b_local.sh

# Storage ablation with ClickHouse on disk
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_db \
  bash experiments/offline_inference/run_ring_db_storage_e2e_qwen3_4b_local.sh

# Lower-bound control without DB insertion
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_null \
  bash experiments/offline_inference/run_ring_db_storage_e2e_qwen3_4b_local.sh
```

## Testing and Archived Scripts

- `testing/`: smoke tests and local test entrypoints
- `archived/`: original sbatch/retry scripts and older superseded entrypoints kept as references; these are not the primary local reproduction path
- `scripts/`: Python runners plus internal shell helpers such as the sweep matrix drivers and step-breakdown dispatcher

## Notes

- The recommended local entrypoints are the top-level `run_*.sh` scripts in this directory.
- The actual Python runners live in `scripts/`.
- Existing sbatch files and older superseded scripts are preserved under `archived/` for reference only.
