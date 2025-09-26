# Repository Guidelines

## Project Structure & Module Organization
- `vllm/`: core Python package (engines, entrypoints, CLI, v1, ops). CLI `vllm` is defined here.
- `csrc/`: C++/CUDA kernels built via CMake; artifacts land in `vllm/` as `*.abi3.so`.
- `tests/`: pytest suite mirroring package layout.
- `docs/`, `examples/`, `benchmarks/`, `tools/`, `docker/`, `cmake/`, `CMakeLists.txt`: documentation, samples, performance, developer tools, containers, and native build files.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements/dev.txt` then `pip install -e .`
- Install hooks: `pre-commit install`; run all: `pre-commit run -a`
- Run tests: `pytest -q` (examples: `pytest -m "core_model or cpu_model"`)
- Serve locally: `vllm serve --model <hf_repo_or_path> --port 8000`

## Coding Style & Naming Conventions
- Python: 4-space indent; 80-char soft limit (Ruff). `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Format/lint via pre-commit: YAPF, Ruff (lint/format), isort, typos; C++/CUDA via clang-format.
- Typing: add type hints and run `pre-commit run mypy-local -a`.
- Imports: prefer relative within `vllm`; do not import `triton` directly (use project wrappers).

## Testing Guidelines
- Place tests under `tests/` mirroring source paths: `tests/<area>/test_<module>.py`.
- Use pytest markers (see `pyproject.toml`): `core_model`, `cpu_model`, `distributed`, `optional`, etc.
- Keep PR tests CPU-friendly; gate heavy/distributed/model tests behind markers.
- Include regression and edge-case coverage; add fixtures under `tests/**/fixtures/` when helpful.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, concise subject; include scope where useful.
- Sign every commit (DCO): append `Signed-off-by: Your Name <email>`.
- Before opening a PR: run `pre-commit run -a` and `pytest` locally.
- PRs should include: clear description, linked issues, perf/accuracy impact, test plan, and doc updates when applicable.
- Include SPDX headers in new source files to match repository convention.

## Security & Configuration Tips
- Do not commit secrets or tokens; pass via environment/CLI. Follow `SECURITY.md` for disclosures.
- Large models: reference by name/path only in examples/tests; never vendor weights.

