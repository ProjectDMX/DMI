# DMI Experiment Reproduction Guide

This branch contains everything needed to reproduce the 4 baselines from the DMI monitoring paper evaluation.

## Baselines

| # | Baseline | Description | Conda Env | Key Requirement |
|---|----------|-------------|-----------|-----------------|
| 1 | **vLLM (baseline)** | Unmodified vLLM 0.17.0 | `vllm-exp` | Stock vLLM, CUDA graphs enabled |
| 2 | **DMI** | DMI monitoring integration | `vllm-exp` | This repo (DMXGPUWorker + ring transport) |
| 3 | **vLLM-Hook** | IBM vLLM-Hook with PyTorch forward hooks | `hook-exp` | `--enforce-eager` (no CUDA graphs) |
| 4 | **TRT-LLM (Debug API)** | TensorRT-LLM with per-step D2H extraction | `trtllm-exp` + `vllm-exp` | Pre-built engines + Python patches |

## Models

- Qwen3-4B (36 layers)
- Llama-3.1-8B-Instruct (32 layers)
- Qwen3-14B (40 layers)

## Directory Structure

```
experiments/online_serving/
  README.md                   # This file
  vLLM-Hook/                  # Submodule: ProjectDMX/vLLM-Hook-baseline
  TensorRT-LLM/               # Submodule: ProjectDMX/TensorRT-LLM-baseline
  sampled_datasets/            # 6 JSON files: {sharegpt,wildchat}_seed{42,123,456}_n500_n30.json
  results/                     # Benchmark output (git-ignored)
  DMI_plot/                    # Plotting scripts
  envs/                        # pip freeze outputs for each environment
  script/
    setup_env.sh               # Create conda envs + download models
    run_vllm_baseline.sh       # Run vLLM baseline at specified rates
    run_dmi.sh                 # Run DMI at specified rates
    run_vllm_hook.sh           # Run vLLM-Hook at specified rates
    run_trtllm_d2h.sh          # Run TRT-LLM D2H at specified rates
    build_trtllm_engines.sh    # Build TRT-LLM engines with debug output
    run_bench.py               # Benchmark client (wraps vllm.benchmarks.serve)
    adaptive_bench.py          # Adaptive rate binary search (reference)
    sample_datasets.py         # Dataset sampling script
    sbatch/                    # Original SLURM sbatch files (reference)
```

## Quick Start

### 1. Clone with submodules

```bash
git clone --recurse-submodules -b DMI_experiments git@github.com:ProjectDMX/DMI.git
cd DMI
```

### 2. Set up environments

`setup_env.sh` prints the commands but `conda activate` must be run manually
in your shell. Run each step, then activate as instructed:

```bash
# Create env for vLLM baseline & DMI
bash experiments/online_serving/script/setup_env.sh baseline
conda activate vllm-exp

# Create env for vLLM-Hook
bash experiments/online_serving/script/setup_env.sh hook
conda activate hook-exp

# Create env for TRT-LLM (also applies patches automatically)
bash experiments/online_serving/script/setup_env.sh trtllm
conda activate trtllm-exp

# Download HuggingFace models (requires internet)
conda activate vllm-exp
bash experiments/online_serving/script/setup_env.sh models

# Generate datasets (or use pre-generated ones in experiments/online_serving/sampled_datasets/)
bash experiments/online_serving/script/setup_env.sh datasets
```

See `experiments/online_serving/envs/*.requirements.txt` for exact package versions.

### 3. Run benchmarks

Activate the appropriate conda env first, then run from the repo root.
Each script starts a server, runs benchmarks across 6 datasets at the specified
rates, and saves JSON results to `experiments/online_serving/results/`.

```bash
# vLLM baseline
conda activate vllm-exp
./experiments/online_serving/script/run_vllm_baseline.sh --model qwen4b --rates "1 2 4 8 16 32 64 128 256"

# DMI (same env as baseline)
conda activate vllm-exp
./experiments/online_serving/script/run_dmi.sh --model qwen4b --rates "1 2 4 8 16 32 64 128 256"

# vLLM-Hook
conda activate hook-exp
./experiments/online_serving/script/run_vllm_hook.sh --model qwen4b --rates "1 2 4 8 16 32 64"

# TRT-LLM D2H (requires engines — see step 4)
# Needs two envs: trtllm-exp for server, vllm-exp for benchmark client
ENV_PYTHON=$(conda run -n trtllm-exp which python) \
BENCH_PYTHON=$(conda run -n vllm-exp which python) \
./experiments/online_serving/script/run_trtllm_d2h.sh --model qwen4b --rates "1 2 4 8 16 32 64"
```

You can override the Python binary for any script:
`ENV_PYTHON=/path/to/python ./experiments/online_serving/script/run_vllm_baseline.sh ...`

### 4. TRT-LLM engine setup

If you used `setup_env.sh trtllm`, patches are already applied. Otherwise apply manually:

```bash
conda activate trtllm-exp
TRTLLM_SRC=experiments/online_serving/TensorRT-LLM/tensorrt_llm
TRTLLM_DST=$(python -c "import tensorrt_llm; print(tensorrt_llm.__path__[0])")

# Build-time (needed before engine compilation):
cp $TRTLLM_SRC/models/modeling_utils.py $TRTLLM_DST/models/

# Runtime (needed for serve with D2H):
cp $TRTLLM_SRC/llmapi/llm.py $TRTLLM_DST/llmapi/
cp $TRTLLM_SRC/sampling_params.py $TRTLLM_DST/
```

Build engines (requires GPU, ~30 min per model):

```bash
conda activate trtllm-exp
./experiments/online_serving/script/build_trtllm_engines.sh --model qwen4b
./experiments/online_serving/script/build_trtllm_engines.sh --model llama8b
./experiments/online_serving/script/build_trtllm_engines.sh --model qwen14b
```

### 5. Plot results

Results are saved in `experiments/online_serving/results/` with subdirectories matching the
expected plot layout (`vllm_wo_monitor/`, `vllm_hook/`, `dmi/`, `trtllm_d2h/`).

```bash
conda activate vllm-exp

# 4-baseline comparison plot (TTFT + TPOT)
python experiments/online_serving/DMI_plot/plot_dmi.py \
    --base_dir experiments/online_serving/results/ \
    --output_dir experiments/online_serving/DMI_plot/output/

# Generic pipeline plot (auto-detects baselines)
python experiments/online_serving/DMI_plot/plot_pipeline.py \
    --base_dir experiments/online_serving/results/ \
    --output_dir experiments/online_serving/DMI_plot/output/
```

## How Each Baseline Works

### vLLM Baseline

Unmodified vLLM 0.17.0. No hidden state extraction. CUDA graphs enabled.
This is the performance reference — any monitoring overhead is measured against this.

### DMI

DMI hooks into vLLM via a custom worker class (`DMXGPUWorker`). During each decode
step, DMI's ring-buffer transport asynchronously copies per-layer hidden states from
GPU to a pinned CPU buffer, overlapping with computation. CUDA graphs remain enabled.

### vLLM-Hook

Uses PyTorch's `register_forward_hook()` to intercept each decoder layer's output
during forward pass. On every decode step, each layer's hook fires a Python callback
that calls `hidden_states.cpu()` (synchronous GPU→CPU copy).

**Key limitation:** Must use `--enforce-eager` because CUDA graphs replay the entire
forward as a single recorded kernel launch — Python hooks are never triggered during
replay. This means no CUDA graph acceleration, which is a major source of overhead.

**Modifications** (`experiments/online_serving/vLLM-Hook/`, branch `dmi_experiment_mods`):

- `probe_hidden_states_worker.py` (new): Custom vLLM Worker that installs forward
  hooks on all decoder layers to extract hidden states via GPU→CPU copy
- `probe_hookqk_worker.py` (modified): Original worker wrote Q/K tensors to disk
  on every hook call. Rewritten to only do `.cpu()` copy and immediately discard,
  so the benchmark measures pure D2H overhead without I/O
- `__init__.py` (modified): Registers the new worker in the plugin registry

### TRT-LLM (Debug API)

TensorRT compiles the model into an optimized compute graph, fusing layers and
eliminating intermediate tensors. Normally, per-layer hidden states don't exist at
runtime — they're optimized away.

TRT-LLM's Debug API (`enable_debug_output=True` + `register_network_output()`) prevents
this optimization by marking specific tensors as engine outputs. At runtime, the C++
Executor's batch manager has built-in D2H support (`allocAdditionalOutputs` →
`copyAdditionalOutputs`) that copies these outputs from GPU to CPU on every decode step.

**Patches** (`experiments/online_serving/TensorRT-LLM/`, branch `dmi_experiment_mods`, base commit `51f5ef3`):

The Python API doesn't fully expose the D2H configuration, so 3 files in the
`tensorrt_llm` pip package are patched (applied automatically by `setup_env.sh trtllm`):

| File | Change | Why |
|------|--------|-----|
| `models/modeling_utils.py` | +4 lines: `register_network_output()` in decoder loop | Without this, TRT fuses away hidden states — nothing to extract |
| `llmapi/llm.py` | +15 lines: read `TRTLLM_EXTRACT_NLAYERS` env var | `ExecutorConfig.additional_model_outputs` is not exposed by the constructor; this patch sets it internally |
| `sampling_params.py` | 1 line: `gather_context=False` → `True` | Hardcoded default causes token index mismatch assertion |

### Comparison

| | vLLM-Hook | TRT-LLM Debug API |
|---|---|---|
| D2H mechanism | Python forward hook + `.cpu()` | C++ batch manager (built-in) |
| CUDA graphs | Incompatible (must use eager) | Compatible |
| Overhead source | Eager mode + Python callback + D2H | D2H only |
| Modification style | External plugin (no vLLM changes) | Patch pip package (no external API) |

## Benchmark Parameters

- **Datasets**: ShareGPT + WildChat, 500 samples each, 3 seeds (42, 123, 456)
- **Output length**: 128 tokens
- **Duration**: 30 seconds per rate
- **Warmup**: 50 requests
- **Metrics**: TTFT (Time To First Token), TPOT (Time Per Output Token), ITL, throughput

## Environment Variables

All scripts support these overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV_PYTHON` | `python` | Python binary for the server |
| `BENCH_PYTHON` | `$ENV_PYTHON` | Python binary for benchmark client (TRT-LLM only) |
| `WORK_DIR` | repo root | Root directory (where `experiments/online_serving/` lives) |
| `HF_HOME` | `$WORK_DIR/hf_cache` | HuggingFace cache directory |
| `MPIRUN` | `which mpirun` | MPI launcher (TRT-LLM only) |
