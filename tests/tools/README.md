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
The 0.25.1 port is V1-runner-only and fails closed if an embedding process
selects V2.

Before starting a shared-machine multi-GPU sweep, verify the selected physical
cards are idle for consecutive samples:

```bash
python tests/tools/check_gpu_idle.py --gpus 0,1 --samples 3 --interval 2
```

> Native CUDA ring tests live separately under `tests/ring/` (built via its
> `Makefile`, marker `ring_native`, needs `nvcc`) and are likewise excluded from
> default pytest discovery.

As the configurable E2E matrix (`tests/e2e_matrix`) lands, these hardcoded
wrappers are expected to be superseded by matrix invocations.
