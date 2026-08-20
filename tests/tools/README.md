# tests/tools — manual analysis & release-sweep scripts

These are **manual** entry points: debugging aids, transport/correctness sweeps,
and release-candidate regression wrappers. They are intentionally **not** part of
the pytest regression gates — `pyproject.toml` lists `tests/tools` under
`norecursedirs`, so pytest never discovers anything here.

Run them by hand, from the **repository root**, when you want a full sweep or are
debugging a specific backend. They require a GPU and most also need ClickHouse;
they are not CPU-safe.

| Script | What it does |
|---|---|
| `run_regression.sh` | Full root release sweep: CPU unit tests + HF transport correctness across models/modes/TP. |
| `run_tp_compare_hf.sh` | Single HF transport-correctness run (`.copy_()` buffers vs ClickHouse) for one model/mode/TP. |
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
