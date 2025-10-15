# MonitoringEngine 实现说明（代码级）

本文从实现角度详细说明 MonitoringEngine 的整体结构、关键数据结构、提交流程（多条路径）、同步与清理、统计与可观测性，以及当前的性能画像与未来优化方向，便于代码评审与二次开发。

## 目录
- 组件/数据结构
- 执行流（三种原生路径 + Python 回退）
- CUDA 流与同步
- 结果与清理
- 统计与可观测性
- 性能画像（当前）
- 已知取舍与后续工作

## 组件/数据结构

- Python 层（`monitoring/engine.py`）
  - `MonitoringEngine`：对外入口，收集每步任务并委派给后端；
    - 路径选择：原生构建器（append_hook + seal_step）、SoA 批量（submit_step_soa）、快路径追加（add_task + seal_step）、Python 回退；
    - 统计（`MON_ENGINE_STATS=1`）：`py_serialize_ms/py_submit_ms/py_bind_ms/py_resolve_ms/max_tasks_per_step`；
    - 环境开关：`MON_NATIVE_BUILDER`（默认 1）、`MON_NATIVE_BATCH`（默认 0）。

- Hook 层（`transformers/.../hook_points.py`）
  - `HookPoint.save_hook`：每个 Hook 的回调逻辑；按优先级选择：
    1) 完全后端化（默认）：`append_hook(step_id, hook_name, tensor, remove_batch_dim, pos_slice, device)`；
    2) SoA 批量：聚合列，在 `end_step` 一次性 `submit_step_soa`；
    3) 快路径追加：每 hook `add_task`，在 `end_step` `seal_step`；
    4) 纯 Python：构造 `MonitoringTask`，`engine.submit(task)`；
  - Hook 侧统计（`MON_ENGINE_STATS` 或 `MON_HOOK_STATS`）。

- 原生后端（`monitoring/csrc/native_engine.cpp`）
  - `SliceSpec`：模式 + 参数，支持 0/1/2/3 整型模式或字符串模式；
  - `TaskSpec`：输入张量、切片维度/规则、可选目标设备与 dtype；
  - `TaskEntry`：`TaskSpec + token`；
  - `StepWork`：步级容器 + 可选 `cudaEvent_t`；
  - `ResultSlot`：结果/错误/ready/future 同步的条件变量槽。
  - 数据结构：
    - `open_steps_`：append_hook/add_task 构建中的步；
    - `sealed_steps_`：等待分发的步；
    - `queue_`：工作线程队列；
    - `slots_`：`token → ResultSlot`；
    - 计数与统计：`pending_tasks_`、`next_token_`、`stats_*`。

## 执行流（三种原生路径 + Python 回退）

记号：S 表示 `start_step()`，E 表示 `end_step()`；`stream` 为当前生产者流（PyTorch 当前 CUDA 流）。

### A. 完全后端化（默认）

1) S：引擎仅推进 `step_id`。
2) 每个 Hook：调用 `append_hook(step_id, hook_name, tensor, remove_batch, pos_slice, device)`：
   - C++ 推断 `pos_dim`（q/k/v/z/result→-3，否则 -2），计算 `can_slice`，解析 `pos_slice`，生成 `TaskSpec` 追加到 `open_steps_[step_id]`；
   - 仅 `pending_tasks_++`，不分配 token/slot。
3) E：调用 `seal_step(step_id, stream_handle)`：
   - 从 `open_steps_` 取出 `StepWork`，若提供了 `stream_handle`，记录 CUDA event；
   - 步入 `sealed_steps_`，按 delay 策略触发分发；
   - 工作线程处理 `StepWork` 时，懒分配 token/slot → `run_task(TaskSpec)` → `store_result(token, tensor)`。

### B. SoA 批量

1) S：推进 `step_id`；
2) 每个 Hook：将 `tensor/flags/slice_mode(idx)/indices 等` 追加到 Python 端 SoA 列容器；
3) E：调用 `submit_step_soa(step_id, spec_dict, stream_handle)`，一次性构造 `StepWork`，创建所有 slot/token 并分发；
4) 工作线程按 `TaskSpec` 处理。

### C. 快路径追加（每 hook 一次 add_task）

1) 每个 Hook：`add_task(step_id, tuple)`，返回 token 并立即创建 slot；
2) E：`seal_step(step_id, stream_handle)` 分发。

### D. 纯 Python 回退

1) 每个 Hook：构造 `MonitoringTask`，`engine.submit(task)` 收集到桶中；
2) E：构建 tuple payload 列表并调用原生 `submit_step`；
3) 绑定 token 到 `CacheFuture`；
4) Python 后端（无原生扩展时）则用一个工作线程在 `cache_stream` 上同步执行处理。

## CUDA 流与同步

- 生产者流：`torch.cuda.current_stream()`；
- 与缓存流的同步：
  - `submit_step(_soa)`：在生产者流上记录 CUDA event，缓存流等待该 event；
  - `seal_step`：同上；
  - 工作线程在 `cache_stream_` 上 set current，结束前恢复之前流。

## 结果与清理

- `ResultSlot`：`ready/has_error/consumed/tensor` + `cv`；
- `future_*`：暴露 `ready/wait/result`；
- `clear_completed_results`：扫描 slots 中 ready 的条目并删除，避免 OOM；
- `close()`：
  - 排空 `sealed_steps_` 并分发；
  - 通知工作线程退出；
  - 清理残留所有 slot。

## 统计与可观测性

- Python（`engine.py`）：`py_serialize_ms/py_submit_ms/py_bind_ms/py_resolve_ms/max_tasks_per_step`；
- 原生（`get_stats`）：`total_steps/total_tasks/submit_us/process_us`；
- Hook（`hook_points.py`）：`build_us/submit_us/cache_set_us` 与同步路径的 `sync_*`；
- 基准脚本（`benchmark/tests/profile_decode.py`）：在 `MON_ENGINE_STATS=1` 时，无论同步/异步都会打印 `[Hook/Stats]`。

## 性能画像（当前 BS=64, decode_steps=64, 收集 hidden/attn）

- 同步（`hf_modified_hook`）：主时钟 1.05–1.10s；
- 异步（add_task 快路径）：约 1.07s；
- 异步（SoA 批量）：约 1.01–1.03s；
- 异步（完全后端化构建器）：约 0.97–1.02s；

主要剩余开销：Hook 回调本身的 Python 调度与极少量参数准备（`build_us≈90–110ms`）。提交/解析/后台处理已降至几十毫秒总量级。

## 已知取舍与后续工作

- Hook 回调仍在 Python 中触发（PyTorch 框架级）。要进一步压缩 `build_us`：
  - 减少 Hook 点数量或按层聚合（线性收益）；
  - 将 Hook 回调本身替换为原生回调（pybind 注册的原生函数），由 C++ 直接聚合（需要在 TL Hook 注册处做更深入改动）；
  - 将 SoA `indices` 改为 LongTensor（减少 pybind 小对象）；

- 已提供稳定的三条原生路径，建议默认 `MON_NATIVE_BUILDER=1`；遇到问题可切到 `MON_NATIVE_BATCH=1` 的 SoA 批量或 `add_task + seal_step` 的快路径。

