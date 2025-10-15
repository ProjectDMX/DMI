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

## 执行流（四种路径：Native Callback + 三种原生路径 + Python 回退）

记号：S 表示 `start_step()`，E 表示 `end_step()`；`stream` 为当前生产者流（PyTorch 当前 CUDA 流）。

### A. Native Callback（推荐，最优性能）

**环境变量**: `MON_NATIVE_CALLBACK=1`（启用）

1) **注册阶段**（模型初始化时）：
   - 调用 `create_hook_callback(hook_name, remove_batch, pos_slice, device)` 返回 C++ lambda；
   - 将 C++ lambda 注册给 PyTorch：`hook_point.register_forward_hook(cpp_function)`；
   - `HookConfig` 预分配并缓存在 `hook_configs_[hook_name]`。

2) S：引擎调用 `begin_step(step_id)` 更新 `current_step_id_` 原子变量。

3) **每次前向时**（94k 次回调）：
   - PyTorch 直接调用 C++ lambda（无 Python 解释器开销）；
   - C++ lambda 执行：
     ```cpp
     {
         py::gil_scoped_release release;  // 释放 GIL
         if (tensor.requires_grad()) tensor = tensor.detach();
         engine->append_hook_current_step(*cfg_ptr, std::move(tensor));
     }
     ```
   - `append_hook_current_step()` 读取 `current_step_id_`，直接追加到 `open_steps_[step_id]`；
   - 仅 `pending_tasks_++`，不分配 token/slot。

4) E：调用 `seal_step(step_id, stream_handle)`：
   - 从 `open_steps_` 取出 `StepWork`，记录 CUDA event；
   - 步入 `sealed_steps_`，按 delay 策略触发分发；
   - 工作线程处理时懒分配 token/slot → `run_task()` → `store_result()`。

**性能特点**：
- ✅ 消除 94k 次 Python→C++ 边界跨越
- ✅ GIL 释放，CUDA 操作与主线程并行
- ✅ HookConfig 预分配，零重复解析
- ✅ 当前最快实现：0.8s（vs baseline 0.42s）

---

### B. 完全后端化构建器（MON_NATIVE_BUILDER=1）

1) S：引擎仅推进 `step_id`。
2) 每个 Hook：调用 `append_hook(step_id, hook_name, tensor, remove_batch, pos_slice, device)`：
   - C++ 推断 `pos_dim`（q/k/v/z/result→-3，否则 -2），计算 `can_slice`，解析 `pos_slice`，生成 `TaskSpec` 追加到 `open_steps_[step_id]`；
   - 仅 `pending_tasks_++`，不分配 token/slot。
3) E：调用 `seal_step(step_id, stream_handle)`：
   - 从 `open_steps_` 取出 `StepWork`，若提供了 `stream_handle`，记录 CUDA event；
   - 步入 `sealed_steps_`，按 delay 策略触发分发；
   - 工作线程处理 `StepWork` 时，懒分配 token/slot → `run_task(TaskSpec)` → `store_result(token, tensor)`。

### C. SoA 批量（MON_NATIVE_BATCH=1）

1) S：推进 `step_id`；
2) 每个 Hook：将 `tensor/flags/slice_mode(idx)/indices 等` 追加到 Python 端 SoA 列容器；
3) E：调用 `submit_step_soa(step_id, spec_dict, stream_handle)`，一次性构造 `StepWork`，创建所有 slot/token 并分发；
4) 工作线程按 `TaskSpec` 处理。

### D. 快路径追加（每 hook 一次 add_task）

1) 每个 Hook：`add_task(step_id, tuple)`，返回 token 并立即创建 slot；
2) E：`seal_step(step_id, stream_handle)` 分发。

### E. 纯 Python 回退

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
- 原生（`get_stats`）：`total_steps/total_tasks/submit_us/process_us/callback_us`；
  - **新增 `callback_us`**：Native Callback 路径的回调总耗时（包含 GIL 释放的 C++ 执行时间）；
- Hook（`hook_points.py`）：`build_us/submit_us/cache_set_us` 与同步路径的 `sync_*`；
- 基准脚本（`benchmark/tests/profile_decode.py`）：在 `MON_ENGINE_STATS=1` 时，无论同步/异步都会打印 `[Hook/Stats]` 和 `[Native/Stats]`。

## 性能画像（当前 BS=64, decode_steps=64, 收集 hidden/attn）

- **Baseline（无 hook）**：0.42s
- **同步（`hf_modified_hook`）**：1.05–1.10s (+154%)
- **异步优化历程**：
  - 初始异步：1.53s (+264%)
  - add_task 快路径：1.07s (+155%)
  - SoA 批量：1.01–1.03s (+140-145%)
  - 完全后端化构建器：0.97–1.02s (+131-143%)
  - **Native Callback（当前）**：**0.8s (+90%)** ✨

**剩余 0.38s (48%) 差距主要来源**：
- PyTorch Hook 调度（框架级）：30-50ms
- Tensor 引用计数与内存管理：20-30ms
- GPU 带宽竞争（前向+抓取）：100-150ms
- CUDA 流切换与同步：20-30ms
- 其他（调度/GC/等）：50-100ms

Native Callback 已将回调开销从 `build_us≈107ms` 降至 `callback_us≈8ms`（**13x 提升**）。进一步优化需要架构级重构（批量 hook、双 GPU 等）。

## 已知取舍与后续工作

### 已完成 ✅
- ✅ **Native Callback**：Hook 回调完全在 C++ 执行，消除 Python→C++ 边界（0.8s）
- ✅ **GIL 释放**：CUDA 操作不持有 GIL，与主线程并行
- ✅ **HookConfig 预分配**：零重复解析开销

### 未来优化方向

#### 优先级 1: 批量 Hook 聚合
- **目标**：减少 PyTorch hook 调用次数从 94k → 几百次
- **方法**：每层一个聚合 hook，而非每个 tensor 一个 hook
- **预期收益**：30-50ms → 5-10ms

#### 优先级 2: 双 GPU 异步采集
- **目标**：消除 GPU 带宽竞争（100-150ms）
- **方法**：GPU 0 纯前向，GPU 1 异步抓取（NVLink 传输）
- **预期收益**：接近 baseline (0.42s)

#### 优先级 3: 流式处理（减少内存占用）
- **目标**：避免保存完整 tensor，降低 OOM 风险
- **方法**：立即计算统计量，只保存标量结果
- **收益**：降低峰值内存占用

### 路径选择建议
- **推荐**：`MON_NATIVE_CALLBACK=1`（默认，最优性能）
- **备选**：`MON_NATIVE_BUILDER=1`（完全后端化构建器）
- **调试**：`MON_NATIVE_BATCH=1`（SoA 批量）或 `add_task + seal_step`（快路径）

**详细文档**：参见 `docs/Native_Callback_Implementation_CN.md`

