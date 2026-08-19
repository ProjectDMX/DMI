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

Example:

```bash
# from the repo root
LD_PRELOAD=/path/to/libstdc++.so.6 CUDA_VISIBLE_DEVICES=0,1 \
  bash tests/tools/run_regression.sh
```

> Native CUDA ring tests live separately under `tests/ring/` (built via its
> `Makefile`, marker `ring_native`, needs `nvcc`) and are likewise excluded from
> default pytest discovery.

As the configurable E2E matrix (`tests/e2e_matrix`) lands, these hardcoded
wrappers are expected to be superseded by matrix invocations.
