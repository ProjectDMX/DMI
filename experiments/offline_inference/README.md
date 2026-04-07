# Offline Inference/Microbenchmarks Experiment Guide

This directory contains the PyTorch/HuggingFace offline evaluation scripts used in our paper. The experiments are organized into two groups:

- `offline inference`: end-to-end batched generation experiments on sampled datasets
- `microbenchmark`: targeted breakdown and ablation experiments

## Directory Structure

```text
experiments/offline_inference/
  README.md
  offline_inference_*.sh   # user-facing end-to-end experiment entrypoints
  microbenchmark_*.sh      # user-facing microbenchmark entrypoints
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

| Baseline | Description | Runner |
|---|---|---|
| **Hugging Face (baseline)** | Unmonitored HuggingFace baseline step timing | `scripts/run_step_breakdown_hf_ideal.py` |
| **DMI (full)** | DMI full pipeline step timing with compile | `scripts/run_step_breakdown_proj_dmi_manual.py` |
| **DMI (no ring)** | DMI without ring step timing (fall back to eager) | `scripts/run_step_breakdown_proj_dmi_manual.py --disable-compile` |
| **Hugging Face (API)** | HuggingFace with offloading using output API step timing | `scripts/run_step_breakdown_hf_api.py` |
| **Torch Hooks** | PyTorch forward hooks step timing | `scripts/run_step_breakdown_torch_hooks.py` |


## Models

- `qwen3-4b` (36 layers)
- `llama3.1-8b` (32 layers)
- `qwen3-14b` (40 layers)

## 1. Offline Inference

These scripts run end-to-end batched generation on sampled ShareGPT/WildChat data and save JSON summaries.

### Main End-to-End Experiments

| Script | Purpose |
|---|---|
| `offline_inference_qwen_hs_logits.sh` | Qwen end-to-end comparison for the `hs+logits` setting |
| `offline_inference_llama31_8b_hs_logits.sh` | Llama-3.1-8B end-to-end comparison for the `hs+logits` setting |
| `offline_inference_qwen_internal_hooks.sh` | Qwen end-to-end comparison for the internal-hook setting (`q,k,v,z,mlp_in,mlp_out,resid_mid`) |
| `offline_inference_llama31_8b_internal_hooks.sh` | Llama-3.1-8B end-to-end comparison for the internal-hook setting |

### Example Commands

```bash
# Qwen hs+logits end-to-end experiment
bash experiments/offline_inference/offline_inference_qwen_hs_logits.sh

# Llama hs+logits end-to-end experiment
bash experiments/offline_inference/offline_inference_llama31_8b_hs_logits.sh
```

Most scripts support overriding `RESULTS_DIR`, `MODEL`, `DATASETS`, `BATCH_SIZES`, or other experiment-specific environment variables before invoking `bash`.

## 2. Microbenchmark

These scripts focus on phase-level timing and targeted ablations rather than the main end-to-end runs.

### Hook Count

| Script | Purpose |
|---|---|
| `microbenchmark_hook_count_qwen3_4b.sh` | Hook-count comparison across baselines, using the 20G DMI configuration used in the final figure |

### Request Dropping / Prefill Backpressure

| Script | Purpose |
|---|---|
| `microbenchmark_prefill_backpressure_qwen3_4b.sh` | Request dropping / prefill backpressure study, including the DMI ring-size and hook-selection configurations used for the final figure |

### Tensor Parallelism

| Script | Purpose |
|---|---|
| `microbenchmark_tp_compile_qwen3_14b.sh` | TP=1/2/4 compile comparison for HF vs DMI on ShareGPT |

### Step Breakdown

| Script | Purpose |
|---|---|
| `microbenchmark_step_breakdown_qwen3_4b.sh` | Main local step-breakdown microbenchmark for Qwen3-4B |

The default step-breakdown workload is:

- synthetic prefill length `128`
- decode length `10`
- batch sizes `1, 8, 32, 64`
- 5 timed iterations after warmup

### Storage Ablation

| Script | Purpose |
|---|---|
| `microbenchmark_storage_ablation_qwen3_4b.sh` | Ring storage ablation (`ring_null` vs `ring_db`, disk vs `tmpfs`) |

This script measures three completion timestamps over a 10-batch run:

- `forward_done`
- `flush_done`
- `db_done`

It is used to separate:

- main generation time
- explicit ring flush time
- downstream database insertion completion time

Important: `STORAGE_LABEL=disk` or `STORAGE_LABEL=tmpfs` only changes the experiment label written to the result files. It does **not** reconfigure ClickHouse storage automatically. To actually compare disk-backed storage against `tmpfs`, you must manually change the ClickHouse data path, restart ClickHouse, and then rerun the script under the corresponding label.

### Max-Batch Memory Microbenchmark

| Script | Purpose |
|---|---|
| `microbenchmark_max_batch_memory_qwen3_14b.sh` | Unified local max-batch-size search used for the Qwen3-14B memory microbenchmark, keeping only the final baselines and the DMI 2G configuration |

### Example Commands

```bash
# Step breakdown microbenchmark
bash experiments/offline_inference/microbenchmark_step_breakdown_qwen3_4b.sh

# Hook-count microbenchmark
bash experiments/offline_inference/microbenchmark_hook_count_qwen3_4b.sh

# Request dropping / prefill backpressure
bash experiments/offline_inference/microbenchmark_prefill_backpressure_qwen3_4b.sh

# TP microbenchmark
bash experiments/offline_inference/microbenchmark_tp_compile_qwen3_14b.sh

# Max-batch memory microbenchmark
bash experiments/offline_inference/microbenchmark_max_batch_memory_qwen3_14b.sh

# Storage ablation with ClickHouse on disk
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_db \
  bash experiments/offline_inference/microbenchmark_storage_ablation_qwen3_4b.sh

# Lower-bound control without DB insertion
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_null \
  bash experiments/offline_inference/microbenchmark_storage_ablation_qwen3_4b.sh
```

## Testing and Archived Scripts

- `testing/`: smoke tests and local test entrypoints
- `archived/`: original sbatch/retry scripts and older superseded entrypoints kept as references; these are not the primary local reproduction path
- `scripts/`: Python runners plus internal shell helpers such as the end-to-end matrix drivers and step-breakdown dispatcher

## Notes

- The recommended local entrypoints are the top-level `offline_inference_*.sh` and `microbenchmark_*.sh` scripts in this directory.
- The actual Python runners live in `scripts/`.
- Existing sbatch files and older superseded scripts are preserved under `archived/` for reference only.
