# Repository Guidelines

## Project Structure & Module Organization
- `vllm/`: Python source (engines, workers, schedulers, model executors, entrypoints).
- `csrc/` + `cmake/`: CUDA/C++ kernels and build scripts.
- `tests/`: Pytest test suite (mirrors `vllm/` layout).
- `benchmarks/`: Throughput/latency benchmarks and helpers.
- `docs/`, `examples/`, `docker/`, `tools/`: Documentation, samples, images, utilities.

## Build, Test, and Development Commands
- Setup (editable install): `pip install -e .` then `pip install -r requirements/{dev,test,lint}.txt`.
- Run tests: `pytest tests/` (e.g., `pytest -m "not distributed"`, or `pytest tests/engine/`).
- Type-check: `tools/mypy.sh` (CI uses stricter settings; local runs are lenient).
- Lint/format: `pre-commit install`; then hooks run on commit. Manual: `ruff check vllm/`, `ruff format vllm/`, `isort vllm/`, `yapf -ir vllm/`.
- Build CUDA extensions (dev): `python setup.py clean --all && python setup.py build_ext --inplace`.

## Coding Style & Naming Conventions
- Python: 4-space indent, line length 80 (`ruff`), type hints required where practical (`mypy`).
- Formatting: `ruff format` + `yapf` (see `.pre-commit-config.yaml` and `pyproject.toml`).
- Imports: `isort`-compatible ordering; avoid wildcard imports outside explicit patterns.
- C++/CUDA: follow `.clang-format`; keep kernels minimal and documented.
- Filenames and modules should be descriptive (e.g., `vllm/engine/llm_engine.py`, tests as `tests/engine/test_llm_engine.py`).

## Testing Guidelines
- Framework: `pytest` with markers defined in `pyproject.toml` (e.g., `distributed`, `core_model`).
- Add unit tests alongside feature areas; prefer small, deterministic cases.
- Keep GPU/distributed tests opt-in; default local runs should pass with `-m "not distributed"`.
- Ensure new public behavior has coverage; keep fixtures under `tests/**/fixtures/`.

## Commit & Pull Request Guidelines
- DCO sign-off is required: add `Signed-off-by: Your Name <email>` to every commit (hook auto-adds on commit-msg).
- Before pushing: run tests and `pre-commit` locally; ensure type checks pass.
- PRs should include: clear description, rationale, linked issues, tests, and docs updates if APIs/configs change.
- Keep changes focused; avoid unrelated refactors.

## Configuration & Security Notes
- Config changes must include defaults and docstrings (validated by `tools/validate_config.py`).
- Avoid new `pickle/cloudpickle` usage; hooks enforce this.
- When adding kernels or low-level paths, guard for CUDA/ROCm/CPU and document env variables.

