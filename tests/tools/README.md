# tests/tools — manual analysis & release-sweep scripts

These are **manual** entry points: debugging aids, transport/correctness sweeps,
and release-candidate regression wrappers. They are intentionally **not** part of
the pytest regression gates — `pyproject.toml` lists `tests/tools` under
`norecursedirs`, so pytest never discovers anything here.

Run them by hand, from the **repository root**, when you want a full sweep or are
debugging a specific backend. They require a GPU (most also need ClickHouse and
the vLLM runtime); they are not CPU-safe.

| Script | What it does |
|---|---|
| `run_regression.sh` | Full release sweep: CPU unit tests + HF/vLLM transport correctness across models/modes/TP. Calls the `run_tp_compare_*` wrappers. |
| `run_tp_compare_hf.sh` | Single HF transport-correctness run (`.copy_()` buffers vs ClickHouse) for one model/mode/TP. |
| `run_tp_compare_vllm.sh` | Single vLLM transport-correctness run for one model/mode/TP. |
| `run_qwen2_moe_vllm_pipeline.sh` | Qwen2-MoE / EP vLLM ref → monitored → compare pipeline. |
| `identical_vllm.sh` | Wrapper around the vLLM bitwise-identical pytest check. |
| `verify_vllm.sh` | vLLM row-count + identical verification sweep across ring sizes. |
| `verify_hf.sh` | HF E2E correctness sweep across ring sizes. |
| `smoke_vllm_model.py` | Baseline/monitored public-output smoke for a new vLLM version or model; no ClickHouse required. |
| `run_vllm_release_matrix.py` | Version-pinned focused/public/storage matrix with two-GPU idle gates and retained JSON/log evidence. |

Example:

```bash
# from the repo root
LD_PRELOAD=/path/to/libstdc++.so.6 CUDA_VISIBLE_DEVICES=0,1 \
  bash tests/tools/run_regression.sh
```

The formal API-only differential gate combines the curated corpus under
`tests/blackbox/cases/` with reproducible generated prompts, runs baseline and
monitored vLLM in separate processes, and compares prompt tokens, generated
tokens, text, and stop metadata:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q -s tests/test_vllm_blackbox.py

# Exercise both eager and CUDA-graph public API paths.
DMI_BLACKBOX_CUDAGRAPH=1 CUDA_VISIBLE_DEVICES=0 \
  pytest -q -s tests/test_vllm_blackbox.py

# Reproduce or broaden a generated corpus.
DMI_BLACKBOX_SEED=20260812 DMI_BLACKBOX_GENERATED_CASES=20 \
  CUDA_VISIBLE_DEVICES=0 pytest -q -s tests/test_vllm_blackbox.py

# A model that requires two-way tensor parallelism.
DMI_BLACKBOX_MODEL=qwen2_moe DMI_BLACKBOX_TP_SIZE=2 \
  DMI_BLACKBOX_GPU_MEMORY_UTILIZATION=0.85 \
  CUDA_VISIBLE_DEVICES=0,1 pytest -q -s tests/test_vllm_blackbox.py
```

Tiny fixtures whose context window is below the default 512-token test budget
can set `DMI_BLACKBOX_MAX_MODEL_LEN` to the exact smaller runtime limit.
Repositories that keep a loadable model below their root can additionally set
`DMI_BLACKBOX_MODEL_SUBFOLDER`; the runner resolves the current cached snapshot
without storing a machine-specific snapshot hash.

The Gemma 3 expansion cell is reproducible as:

```bash
DMI_BLACKBOX_MODEL=gemma3 DMI_BLACKBOX_MODEL_SUBFOLDER=hf \
  DMI_BLACKBOX_MAX_MODEL_LEN=128 DMI_BLACKBOX_CUDAGRAPH=1 \
  CUDA_VISIBLE_DEVICES=0 pytest -q -s tests/test_vllm_blackbox.py

E2E_GPUS=0 E2E_MODEL_SUBFOLDER=hf E2E_MAX_MODEL_LEN=128 \
  E2E_MAX_NUM_BATCHED_TOKENS=128 \
  bash tests/tools/run_tp_compare_vllm.sh gemma3 cudagraph 1
```

For a completed full-hook capture, validate completeness independently of the
model implementation:

```bash
python tests/tools/check_vllm_storage.py \
  --contract tests/fixtures/vllm/gemma3_2m_storage_contract.json \
  --model-id <unique-capture-id>
```

Phi-3 release validation uses the official checkpoint for public parity and the
tiny fixture for the independent full-hook value oracle:

```bash
DMI_BLACKBOX_MODEL=microsoft/Phi-3.5-mini-instruct \
  DMI_BLACKBOX_CUDAGRAPH=1 CUDA_VISIBLE_DEVICES=0 \
  pytest -q -s tests/test_vllm_blackbox.py

E2E_GPUS=0 E2E_MAX_MODEL_LEN=128 E2E_MAX_NUM_BATCHED_TOKENS=128 \
  bash tests/tools/run_tp_compare_vllm.sh phi3 cudagraph 1
```

Mistral uses the official 7B checkpoint for public parity and a graph-safe small
fixture for the full-hook value oracle. Keep the fixture's memory and token
bounds: an extremely small Mistral can make vLLM derive excessive KV metadata,
and attention heads below the target FlexAttention minimum cannot compile.

```bash
DMI_BLACKBOX_MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
  DMI_BLACKBOX_CUDAGRAPH=1 DMI_BLACKBOX_GPU_MEMORY_UTILIZATION=0.9 \
  DMI_BLACKBOX_MAX_MODEL_LEN=512 CUDA_VISIBLE_DEVICES=0 \
  pytest -q -s tests/test_vllm_blackbox.py

E2E_GPUS=0 E2E_GPU_MEM_UTIL=0.2 E2E_MAX_MODEL_LEN=128 \
  E2E_MAX_NUM_BATCHED_TOKENS=128 \
  bash tests/tools/run_tp_compare_vllm.sh mistral cudagraph 1
```

Set `DMI_BLACKBOX_ARTIFACT_DIR` to retain each mode's generated cases and raw
baseline/monitored JSON instead of relying on pytest's temporary directory.

The public decision-logprob oracle keeps 0.25 nat cross-run drift and branch-gap
ceilings by default. The exact `gpt2` release cell uses 0.5 nat ceilings for
both, calibrated from independent baseline and monitored processes on vLLM
0.27.1. This does not relax its other first-divergence requirements: both branch
tokens must occur in both public top-logprob maps and each selected token must
be within `1e-6` of the public maximum. Non-GPT-2 cells retain 0.25 nat.

For manual diagnosis, run both modes against the same case manifest and compare
their JSON outputs:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/tools/smoke_vllm_model.py \
  --mode baseline --model qwen2 \
  --cases tests/blackbox/cases/transparency.json \
  --output /tmp/qwen2-baseline.json
CUDA_VISIBLE_DEVICES=0 python tests/tools/smoke_vllm_model.py \
  --mode monitored --model qwen2 \
  --cases tests/blackbox/cases/transparency.json \
  --output /tmp/qwen2-monitored.json
```

The vLLM runners set `VLLM_USE_V2_MODEL_RUNNER=0` before importing vLLM.
The 0.27.1 port is V1-runner-only and fails closed if an embedding process
selects V2.

Before starting a shared-machine multi-GPU sweep, verify the selected physical
cards are idle for consecutive samples:

```bash
python tests/tools/check_gpu_idle.py --gpus 0,1 --samples 3 --interval 2

# Once two physical cards are idle, run the complete vLLM 0.27.1 matrix from
# the intended Python environment. The output directory must not already exist.
python tests/tools/run_vllm_release_matrix.py \
  --gpus 0,1 --phase all --artifact-dir /tmp/dmi-vllm-0271-evidence
```

For the practical four-H100 SOTA gate, pre-cache the pinned model artifacts,
start ClickHouse, and run the scheduler-neutral wrapper from a clean committed
worktree:

```bash
bash tests/tools/run_h100_tp4_matrix.sh \
  0,1,2,3 /path/to/new/dmi-vllm-0271-h100-artifacts
```

The gate first proves DMI's generic TP4 path with `Qwen/Qwen3-0.6B` using a
public eager/graph differential and eager/graph storage comparisons. It then
runs the seven SOTA checkpoints that fit at TP4 or below, using a public
eager differential and one eager storage comparison per checkpoint. CUDA-graph
storage is sampled only by the small TP4 representative to keep this acceptance
gate bounded.
GLM-5.2 and Kimi K3 are deliberately omitted because their pinned cells require
TP32; this phase does not substitute a different architecture or report those
two checkpoints as runtime-tested.

> Native CUDA ring tests live separately under `tests/ring/` (built via its
> `Makefile`, marker `ring_native`, needs `nvcc`) and are likewise excluded from
> default pytest discovery.

As the configurable E2E matrix (`tests/e2e_matrix`) lands, these hardcoded
wrappers are expected to be superseded by matrix invocations.
