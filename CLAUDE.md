# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DMI (Deep Model Inspector) — high-performance LLM internal observability system. Captures arbitrary internal model states (activations, attention, Q/K/V, logits) during inference via a CUDA-graph-compatible Ring² transport pipeline, with async export to ClickHouse.

Integrates with HuggingFace Transformers and vLLM (v0.17+).

## Build

```bash
# 1. ClickHouse C++ client (one-time)
cmake -S libs/clickhouse-cpp -B libs/clickhouse-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build libs/clickhouse-cpp/build -j

# 2. Native backend (.so)
make -C monitoring -j

# Or with options:
make -C monitoring -j UNIFIED=1 LTO=1    # single-TU + LTO for max perf
make -C monitoring -j SM_ARCH=native     # auto-detect GPU arch
```

Output: `monitoring_native_backend*.so` in repo root.

Requires: `CUDA_MODULE_LOADING=EAGER` at runtime (cuStreamWaitValue32 deadlocks with lazy loading on CUDA 12.2+).

## Architecture

Three-component pipeline:

1. **HookPoint** (`monitoring/hook_points.py`) — `nn.Module` inserted into model forward graph. Calls `torch.ops.ring.producer(x_cont, hook_type, hook_id)` — a custom CUDA op registered via `TORCH_LIBRARY` in `csrc/ring/ring_torch_op.cpp`. Marked as `_EffectType.ORDERED` to prevent Inductor DCE.

2. **Ring²** (`monitoring/csrc/ring/`) — GPU-side staging:
   - **Payload ring** (`payload_ring.cuh`): circular device-memory byte buffer for tensor data
   - **Meta ring** (`task_ring.cuh`): managed-memory FIFO of 64B TaskEntry descriptors
   - **Producer kernel** (`producer.cu`): multi-block size-tiered D2D copy + metadata publish
   - **Drain thread** (`drain_thread.cpp`): polls meta ring, batched D2H to pinned staging
   - **P2P thread** (`p2p_thread.cpp`): pinned→pageable copy, per-request slicing, SubmitFn callback

3. **Data Exporter** (`monitoring/engine.py`, `csrc/engine_core.cpp`) — host-side pipeline to ClickHouse via `DMXHostEngine`.

Metadata flows through `TensorMetaFifo` (`csrc/ring/tensor_meta.h`) — pushed from Python before forward (`pre_push_all_metas`), popped by P2P thread. FIFO ordering implicitly matches producer kernel firing order.

## Key code paths

- **HF path**: `monitoring/generate.py` → `generate_with_monitoring()` wraps HF `generate()`, installs `_prepare_wrapper` before each forward step
- **vLLM path**: `monitoring/vllm_integration.py` → `DMXGPUWorker` subclasses vLLM Worker, overrides `init_device/load_model/execute_model`
- **Hook installation**: `monitoring/ring_transport.py` → `install_ring_hooks()` + `apply_hook_selection()` filter hooks by preset ("full", "vllm-full", "hidden-states", "logits", "attention")
- **Shape computation**: `ring_transport._compute_hook_shape()` — analytical, no warmup needed
- **Capacity check**: `ring_engine_py.cu::prepare_step()` — returns RING_OK/RING_FLUSHED/CPU_DIRECT

## Tests

```bash
# E2E correctness (HF, requires ClickHouse running)
python -m tests.test_e2e_correctness_vs_hf

# vLLM bitwise identical test
E2E_MODEL=gpt2 E2E_ENFORCE_EAGER=1 python -m tests.test_vllm_identical

# Benchmarks
python -m benchmark.bench_ring_transport --model qwen3 --batch 64 --modes baseline,ring_null,ring_db
```

## C++ / CUDA conventions

- Ring namespace: `ring::` for device code, `ring_py::` for Python-facing wrappers
- `RingState` is a POD struct passed by value into kernels (capture-safe)
- Managed memory placement: task entries → CPU-preferred (drain polls), head counters → GPU-preferred (producer writes)
- `__threadfence()` before publishing `ready_seq` in task entries
- Producer kernel uses last-block-arrives pattern (`atomicAdd` on `g_block_done_counter`)
- `null_mode`: `__device__ bool g_ring_null_mode` makes producer kernel early-return; toggled via `cudaMemcpyToSymbol`

## Paper context (SOSP submission)

DMI paper is submitted to SOSP. Key numbers: 0.4-6.8% offline overhead, ~6% online TPOT overhead. Baselines: PyTorch Hooks (46.9%), NNsight (62.3%), vLLM Hook (10-15x), TRT-LLM Debug API (~2x TPOT). Without Ring², decode overhead jumps from 5.05% to 63.67%.

Current hook filtering is **static** (decided before CUDA graph capture). Dynamic reconfiguration via `cudaGraphNodeSetEnabled` is being explored on `feature/dmi_kernel_node_toggle` branch. See `~/.claude/projects/-home-yibo-DMI/memory/project_dmi_sosp.md` for full paper context.

## Zaratan cluster (experiment environment)

Experiments run on UMD Zaratan HPC cluster. The existing DMI environment can be reused without rebuilding.

```
Partition:   gpu-h100 (gres=gpu:h100:1, cpus=16, mem=64G)
Scratch:     /scratch/zt1/project/zaoxing-prj/user/ynn1999
Project:     ${SCRATCH}/DMI/DMI
Python:      ${SCRATCH}/envs/vllm-h100/bin/python3.10  (conda vllm-exp, vllm==0.17.0)
DMI site:    ${SCRATCH}/dmi-env/lib/python3.10/site-packages
HF cache:    ${SCRATCH}/hf_cache  (offline, HF_HUB_OFFLINE=1)
```

**Required env vars** (see `experiments/online_serving/script/sbatch/*.sbatch`):
```bash
ENV_PYTHON=$(pwd)/envs/vllm-h100/bin/python3.10
SITE=$(pwd)/dmi-env/lib/python3.10/site-packages
DMI=$(pwd)/DMI
export PYTHONPATH=$DMI/integration/vllm:$DMI:$DMI/transformers/src:$SITE
export LD_LIBRARY_PATH=$(pwd)/dmi-env/lib:$(pwd)/DMI/libs/clickhouse-cpp/build/clickhouse${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_MODULE_LOADING=EAGER
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1
```

**Experiment infrastructure**:
- Offline microbenchmark: `experiments/offline_inference/microbenchmark_step_breakdown_qwen3_4b.sh` — per-step compute/transfer/total breakdown, runs `scripts/run_step_breakdown_microbench.py` with baselines: hf_ideal, hf_api, torch_hooks, proj_dmi_manual
- Offline E2E: `experiments/offline_inference/scripts/run_all.sh` — sweeps baselines × datasets × batch sizes × repeats, outputs JSON + CSV summary
- Online serving: `experiments/online_serving/script/sbatch/adapt_dmi_*.sbatch` — launches vLLM server with DMXGPUWorker + adaptive_bench.py
- sbatch template: `#SBATCH --partition=gpu-h100 --gres=gpu:h100:1 --cpus-per-task=16 --mem=64G`
- Models available offline: Qwen3-4B, Qwen3-14B, Llama-3.1-8B

## Model integration

Hooked model variants live in `integration/vllm/` (fork) with `get_hook_specs()` method. Architecture remapping in `vllm_integration.py`:
- `GPT2LMHeadModel` → `GPT2PLMHeadModel`
- `Qwen3ForCausalLM` → `Qwen3PForCausalLM`
- `LlamaForCausalLM` → `LlamaPForCausalLM`
