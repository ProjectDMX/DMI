# DMI Experiment Reproduction Guide

This branch contains everything needed to reproduce the 4 baselines from the DMI monitoring paper evaluation.

## Baselines

| # | Baseline | Description | Key Requirement |
|---|----------|-------------|-----------------|
| 1 | **vLLM (baseline)** | Unmodified vLLM 0.17.0 | Stock vLLM, CUDA graphs enabled |
| 2 | **DMI** | DMI monitoring integration | This repo (DMXGPUWorker + ring transport) |
| 3 | **vLLM-Hook** | IBM vLLM-Hook with PyTorch forward hooks | `--enforce-eager` (no CUDA graphs) |
| 4 | **TRT-LLM (Debug API)** | TensorRT-LLM with per-step D2H extraction | Pre-built engines + Python patches |

## Models

- Qwen3-4B (36 layers)
- Llama-3.1-8B-Instruct (32 layers)
- Qwen3-14B (40 layers)

## Directory Structure

```
experiments/
  README.md                   # This file
  vLLM-Hook/                  # Submodule: ProjectDMX/vLLM-Hook-baseline (branch: dmi_experiment_mods)
  TensorRT-LLM/               # Submodule: ProjectDMX/TensorRT-LLM-baseline (branch: dmi_experiment_mods)
  sampled_datasets/            # 6 JSON files: {sharegpt,wildchat}_seed{42,123,456}_n500_n30.json
  DMI_plot/                    # Plotting scripts (plot_dmi.py, plot_pipeline.py)
  script/
    run_vllm_baseline.sh       # Run vLLM baseline at specified rates
    run_dmi.sh                 # Run DMI at specified rates
    run_vllm_hook.sh           # Run vLLM-Hook at specified rates
    run_trtllm_d2h.sh          # Run TRT-LLM D2H at specified rates
    build_trtllm_engines.sh    # Build TRT-LLM engines with debug output
    setup_env.sh               # Environment setup instructions
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

See `experiments/script/setup_env.sh` for detailed instructions.

### 3. Run benchmarks

Each `run_*.sh` script starts a server, runs benchmarks across 6 datasets at the specified rates, and saves JSON results.

```bash
cd experiments/script

# vLLM baseline
./run_vllm_baseline.sh --model qwen4b --rates "1 2 4 8 16 32 64 128 256"

# DMI
./run_dmi.sh --model qwen4b --rates "1 2 4 8 16 32 64 128 256"

# vLLM-Hook (much slower, use lower rates)
./run_vllm_hook.sh --model qwen4b --rates "1 2 4 8 16 32 64"

# TRT-LLM D2H (requires engines, see below)
./run_trtllm_d2h.sh --model qwen4b --rates "1 2 4 8 16 32 64"
```

### 4. TRT-LLM setup

#### Apply patches

The `experiments/TensorRT-LLM/` submodule contains the patched TensorRT-LLM source.
Copy the 3 modified files to your TRT-LLM pip installation:

```bash
TRTLLM_SRC=experiments/TensorRT-LLM/tensorrt_llm
TRTLLM_DST=$(python -c "import tensorrt_llm; print(tensorrt_llm.__path__[0])")

# Build-time (needed before engine compilation):
cp $TRTLLM_SRC/models/modeling_utils.py $TRTLLM_DST/models/

# Runtime (needed for serve with D2H):
cp $TRTLLM_SRC/llmapi/llm.py $TRTLLM_DST/llmapi/
cp $TRTLLM_SRC/sampling_params.py $TRTLLM_DST/
```

#### Build engines

```bash
./experiments/script/build_trtllm_engines.sh --model qwen4b
./experiments/script/build_trtllm_engines.sh --model llama8b
./experiments/script/build_trtllm_engines.sh --model qwen14b
```

### 5. Plot results

```bash
python experiments/DMI_plot/plot_pipeline.py \
    --base_dir results/ \
    --output_dir experiments/DMI_plot/output/
```

## TRT-LLM Patches Explained

3 files are patched in TensorRT-LLM (base: commit `51f5ef3`):

| File | Change | Purpose |
|------|--------|---------|
| `models/modeling_utils.py` | +4 lines: `register_network_output()` per layer | Marks hidden_states as engine outputs at build time |
| `llmapi/llm.py` | +15 lines: read `TRTLLM_EXTRACT_NLAYERS` env var | Configures C++ Executor to do per-step D2H copies |
| `sampling_params.py` | 1 line: `gather_context=False` -> `True` | Fixes token index mismatch assertion |

## vLLM-Hook Modifications

The `experiments/vLLM-Hook/` submodule (branch `dmi_experiment_mods`) contains:

- `probe_hidden_states_worker.py` (new): Extracts all-layer hidden states via PyTorch forward hooks
- `probe_hookqk_worker.py` (modified): Rewritten to do GPU->CPU copy without disk accumulation
- `__init__.py` (modified): Registers ProbeHiddenStatesWorker in plugin registry

## Benchmark Parameters

- **Datasets**: ShareGPT + WildChat, 500 samples each, 3 seeds (42, 123, 456)
- **Output length**: 128 tokens
- **Duration**: 30 seconds per rate
- **Warmup**: 50 requests
- **Metrics**: TTFT (Time To First Token), TPOT (Time Per Output Token), ITL, throughput
