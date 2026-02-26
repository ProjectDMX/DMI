# Issue #26: 移除 Python Backend，仅保留 C++ Engine（阶段实现）

日期：2026-02-26

## 背景

在 issue #26 中，我们决定将 MonitoringEngine 收敛为 **C++/native-only** 架构，彻底移除 Python fallback 执行链，避免双路径维护和语义漂移。

## 本次实现范围

1. `monitoring/engine.py`
- 删除 Python backend 分支（`_PythonBackend` / `_StepQueue` / `_SimpleStepQueue` / `_process_task_sync`）。
- `MonitoringEngine` 初始化改为 fail-fast：
  - `async_enabled=False` 直接报错；
  - native backend 初始化失败直接报错；
  - 不再 silent fallback 到 Python。
- `submit/end_step/resolve_all/close` 仅保留 native backend 路径。

2. `monitoring/task.py`
- `CacheFuture` 收敛为 native token 模式。
- 移除 `threading.Event + set_result/set_exception` 逻辑。
- 未绑定 backend/token 时，`result()/wait()` 报错。

3. `monitoring/hook_points.py`
- 删除 `engine.submit(task)` fallback 分支。
- 当挂载 `MonitoringEngine` 但 native callback 不可用时，立即抛错（不再退回 Python submit）。
- 清理由 fallback 分支遗留的 hook 统计字段。

4. 测试重构
- 移除旧的 `async_enabled=False` 语义依赖。
- 新增/改写测试覆盖：
  - `CacheFuture` 未绑定错误行为；
  - native 绑定后的结果读取；
  - engine fail-fast 语义（`async_enabled=False` 与 backend 缺失）；
  - request-id DB step 逻辑在 mock native backend 下继续验证。

## 兼容性说明（Breaking）

- `MonitoringEngine(async_enabled=False)` 不再可用。
- 未加载 native backend 时不再退回 Python fallback，而是直接失败。
- `CacheFuture` 不再支持 Python 侧手动 `set_result/set_exception`。

## 目标状态

- Runtime 仅保留 C++ backend 路径。
- Python fallback 执行链和对应死代码已删除。
