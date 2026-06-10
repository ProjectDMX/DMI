# Testing

The test suite is split into explicit categories by **pytest markers** so each
test declares the resources it needs. The default suite is CPU-only; everything
that needs a GPU, ClickHouse, vLLM, model weights, or the native CUDA build is
marked and opt-in.

## The four canonical commands

| Suite | When | Command |
|---|---|---|
| **CPU default** | every PR / push | `python -m pytest -m "not gpu and not e2e and not manual" -q` |
| **Single-GPU smoke** | per PR (GPU runner) | `python -m pytest -m "gpu and not multi_gpu and not slow" -q` |
| **Multi-GPU / TP** | per PR (multi-GPU runner) | `python -m pytest -m "multi_gpu" -q` |
| **Full / nightly** | nightly schedule | `python -m pytest -m "slow or nightly" -q` |

The CPU default suite is the acceptance gate for every PR: it must pass with no
CUDA device, no ClickHouse, no vLLM runtime, and no downloaded model weights.

> The native backend `.so` still has to be **built** for the CPU suite, because
> importing `monitoring` loads it at import time (JIT is disabled for
> reproducibility). Building needs `nvcc` but not a GPU at runtime. See
> [install.md](install.md) §5.

## Marker taxonomy

Markers are registered in [`pyproject.toml`](../pyproject.toml). CPU is the
**default unmarked** suite — a test with no resource marker is assumed CPU-safe;
GPU/E2E/etc. must be marked explicitly.

| Marker | Meaning |
|---|---|
| `cpu` | Pure-CPU contract/unit test; the default suite. No CUDA / ClickHouse / vLLM / weights / native build needed at runtime. |
| `gpu` | Requires a CUDA device. |
| `multi_gpu` | Requires ≥ 2 CUDA devices (TP / EP / routing). |
| `e2e` | End-to-end pipeline through the native backend + host engine. |
| `clickhouse` | Requires a reachable ClickHouse instance. |
| `vllm` | Requires the vLLM runtime importable. |
| `hf` | Requires HuggingFace weights / model cache. |
| `ring_native` | Native CUDA ring tests built via `tests/ring/Makefile` (needs `nvcc`). |
| `slow` | > ~30 s (full per-hook sweep, large E2E sweeps). Skipped unless selected. |
| `nightly` | Scheduled full-sweep tests; run via `-m "slow or nightly"`. |
| `numeric` | Per-hook numeric-difference study (drift vs the unhooked baseline). |
| `manual` | Investigation / tooling, **not** a regression gate; not collected by default. |

A test may carry several markers (e.g. `gpu`, `vllm`, `clickhouse`, `e2e`).
Selection composes them with boolean expressions:

```bash
python -m pytest -m "gpu and not multi_gpu and not slow" -q
python -m pytest -m "vllm and clickhouse" -q
```

`manual` tools and the `tests/tools` / `tests/ring` directories are excluded
from default collection (`addopts = -ra -m 'not manual'` plus `norecursedirs`).

## Skip-guards

GPU / E2E tests fail **closed with a reason** instead of erroring on a missing
prerequisite, via the helpers in [`tests/_requirements.py`](../tests/_requirements.py):

| Helper | Skips when |
|---|---|
| `require_cuda()` | no CUDA device visible |
| `require_gpus(n)` | fewer than `n` CUDA devices |
| `require_clickhouse(host, port)` | the ClickHouse TCP port is unreachable |
| `require_vllm()` | the vLLM runtime is not importable |
| `require_model_cache(model)` | the model is not in the local HF cache / path |
| `require_nvcc()` | `nvcc` is not on `PATH` |

Use them as decorators or in a module-level `pytestmark` list:

```python
import pytest
from tests._requirements import require_cuda, require_clickhouse

pytestmark = [pytest.mark.gpu, require_cuda()]

@require_clickhouse()
def test_rows_land_in_clickhouse():
    ...
```

Relevant env vars (defaults match the runners): `DMX_DB_HOST` / `DMX_DB_PORT`
for the ClickHouse probe, `HF_HOME` / `HF_HUB_CACHE` for the weight-cache check.

## Continuous integration

[`.github/workflows/tests.yml`](../.github/workflows/tests.yml) wires the four
commands into four jobs, sharing the [`setup-dmi`](../.github/actions/setup-dmi/action.yml)
composite action (Python deps + Transformers fork + native backend build):

| Job | Trigger | Runner | Command |
|---|---|---|---|
| `cpu` | push / PR | `[self-hosted, linux, dmi-cpu]` | CPU default |
| `gpu-smoke` | push / PR (after `cpu`) | `[self-hosted, linux, gpu]` | single-GPU smoke |
| `multi-gpu` | push / PR (after `cpu`) | `[self-hosted, linux, multi-gpu]` | multi-GPU / TP |
| `nightly` | `schedule` (07:00 UTC) / manual | `[self-hosted, linux, gpu]` | `slow or nightly` |

The GPU jobs run on self-hosted runners with CUDA devices, labelled by
capability. Because the suites use the skip-guards above, a runner missing
ClickHouse or model weights **skips** the affected tests with a reason rather
than failing the job. Trigger an off-schedule full sweep with the
**workflow_dispatch** button (the `nightly` job also runs on manual dispatch).
