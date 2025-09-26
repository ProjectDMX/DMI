# Repository Guidelines

## 项目结构与模块组织
- `vllm/`：核心包（引擎、入口、V1、插件）；CLI 入口 `vllm`。
- `DEV_DOC/`：设计文档；监控方案见 `mvp_monitoring_design.md`。
- `csrc/`：C++/CUDA 内核（CMake）；产物为 `vllm/*.abi3.so`。
- `tests/`：pytest 测试，目录与源码对应。
- 其他：`docs/`、`examples/`、`benchmarks/`、`tools/`、`docker/`。

## 构建、测试与本地运行
- 环境准备：
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements/dev.txt
  pip install -e .
  pre-commit install
  ```
- 代码检查：`pre-commit run -a`
- 运行测试：`pytest -q` 或 `pytest -m "core_model or cpu_model"`
- 本地服务：`vllm serve --model <hf_repo_or_path> --port 8000`

## 编码风格与命名约定
- Python 4 空格缩进；Ruff 限 80 列；模块/函数用 `snake_case`，类用 `PascalCase`，常量 `UPPER_SNAKE_CASE`。
- 统一使用 pre-commit：YAPF、Ruff（lint/format）、isort、typos；C/CUDA 用 clang-format。
- 类型：补全类型注解，执行 `pre-commit run mypy-local -a`。
- 导入约束：避免直接 `import triton`（仓库钩子会检查）。

## 测试规范
- 测试文件：`tests/<area>/test_<module>.py`；可用标记（见 `pyproject.toml`）：`core_model`、`cpu_model`、`distributed`、`optional` 等。
- 重/分布式/依赖大模型测试请用标记隔离，默认仅跑轻量用例。

## 提交与 Pull Request
- 提交信息用祈使句；所有提交需 DCO：`Signed-off-by: 姓名 <邮箱>`（commit-msg 钩子会自动追加）。
- 提 PR 前本地通过：`pre-commit run -a` 与 `pytest`；PR 需含变更描述、关联 issue、性能/准确性影响与测试计划；新增源码请加 SPDX 头。

## Monitoring 版开发要点
- 依据 `DEV_DOC/mvp_monitoring_design.md`：在 Decoder Layer 边界挂钩、镜像 paged KV、提取 Top-K 注意力；优先异步、低开销、非阻塞。
- 建议新建 `vllm/v1/monitoring/` 模块，配置项加入 `vllm/config.py`（需默认值与 docstring，确保 `tools/validate_config.py` 通过）。
- 新增 CLI 选项放在 `vllm/entrypoints/openai/cli_args.py`，沿用 `v1` 执行流；必要时在 `serve` 子命令暴露开关。

## 安全与配置
- 不提交模型权重与密钥；安全披露遵循 `SECURITY.md`。
