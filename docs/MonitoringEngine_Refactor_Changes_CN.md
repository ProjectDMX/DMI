# MonitoringEngine 重构变更记录（阶段性总结）

本文记录自异步后端引入以来，为解决 OOM、降低开销、提升异步性能而进行的所有主要改动，涵盖 Python 端与 C++ 原生后端的实现演进与开关控制。

## 目录
- 背景与目标
- 关键问题与定位
- 代码级改动清单
- 环境变量与运行方式
- 性能阶段性结果
- 已解决问题与仍在进行

## 背景与目标

- 在 Hook 收集激活（hidden/attn）场景下，引擎需做到：
  - 主计算流不被后台缓存阻塞（完美异步）；
  - GPU 内存不泄漏（OOM 修复）；
  - Python⇄C++ 边界开销尽可能低；
  - 在相同收集配置下，异步应优于同步。

## 关键问题与定位

- 早期问题：
  - C++ 扩展仍只接受 dict，导致 tuple 传参报错（ValueError: dictionary update sequence element ...）。
  - 结果槽未及时清理，长跑后内存累积导致 OOM。
  - 每个 hook 都执行一次 engine.submit()，累计 47,190 次 Python 调用，submit_us≈1.04s，为最大瓶颈。

- 定位工具：
  - Python 端引擎统计（`MON_ENGINE_STATS`）：细分 `py_serialize_ms/py_submit_ms/py_bind_ms/py_resolve_ms`；
  - C++ 端统计：`steps/tasks/submit_us/process_us`；
  - Hook 侧统计：`build_us/submit_us/cache_set_us` 与 per-hook Top-N。

## 代码级改动清单

### C++ 原生后端（`monitoring/csrc/native_engine.cpp`）

- 支持 tuple 任务协议，移除 dict 依赖：
  - 解析 `TaskSpec`（tensor、slice_dim、flags、slice、target_device）。
  - 解析 `SliceSpec`（Identity/Int/Range/Array），新增整型 mode 快速路径（0/1/2/3）。

- 引入三条提交路径与一个构建器：
  - `submit_step(step_id, list_of_tuples, stream)`：批量 tuple；
  - `add_task(step_id, tuple)` + `seal_step(step_id, stream)`：逐条追加、步级封包；
  - `submit_step_soa(step_id, spec_dict, stream)`：SoA 列式一次性提交，减少 Python 小对象；
  - `append_hook(step_id, hook_name, tensor, remove_batch_dim, pos_slice, target_device)`：完全后端化构建器，由 C++ 推断 `pos_dim/can_slice` 并解析 `pos_slice`，在 `seal_step` 时统一分发与分配 token。

- 执行与同步：
  - `StepWork` 保存步任务与 `cudaEvent_t`；
  - `dispatch_step`/`worker_loop` 后台线程在独立 `cache_stream_` 上执行；
  - `run_task` 非阻塞设备转移/类型转换、按 `SliceSpec` 应用切片；
  - lazy 分配 token/ResultSlot（builder 路径避免追加时分配）。

- 结果与清理：
  - `future_ready/wait/result`；
  - `clear_completed_results` 清理已 ready 的 slot；
  - `close()` 排空队列、回收所有 slot，避免 OOM。

- 统计：
  - `get_stats()` 暴露 `total_steps/total_tasks/submit_us/process_us`；
  - `append_hook/add_task/seal_step/submit_step(_soa)` 更新统计。

### Python 引擎层（`monitoring/engine.py`）

- 细分统计（`MON_ENGINE_STATS=1`）：
  - `py_serialize_ms`：构建 tuple payload 列表；
  - `py_submit_ms`：调用原生提交接口；
  - `py_bind_ms`：token 绑定 Future；
  - `py_resolve_ms`：`resolve_all + clear`；
  - `max_tasks_per_step`：步峰值任务数。

- 三条原生路径切换（含回退）：
  - 完全后端化构建器（默认）：`MON_NATIVE_BUILDER=1`；`end_step` 仅调用 `seal_step`；
  - SoA 批量：`MON_NATIVE_BATCH=1`；`end_step` 调 `submit_step_soa`；
  - 快路径追加：无上述开关时，走 `add_task + seal_step`；
  - 纯 Python 路径保留：无原生后端或禁用异步时，走 Python fallback。

- 其它：
  - `_serialize_task` 采用扁平 tuple；
  - `clear_completed_results/close/resolve_all` 调整，确保清理时机；
  - `monitoring/_native_engine.py` 优先加载当前仓库内 `.so`，避免加载到旧模块。

### Hook 侧（`transformers/.../hook_points.py`）

- 优先使用完全后端化构建器：
  - `native_backend.append_hook(step_id, hook_name, tensor, remove_batch_dim, pos_slice, device)`；
  - `cache[hook_name]` 放占位（基准不读取）。

- SoA 批量收集（可选）：
  - 仅在 `MON_NATIVE_BATCH=1` 启用，按列聚合，在 `end_step` 一次性 `submit_step_soa`；

- 快路径追加（可选回退）：
  - 每个 hook 直接 `add_task`，在 `end_step` `seal_step`。

- 统计（`MON_ENGINE_STATS=1` 或 `MON_HOOK_STATS=1`）：
  - `build_us/submit_us/cache_set_us`；`sync_move_us/sync_slice_us/sync_cache_set_us`（同步路径）；
  - per-hook Top10；
  - 基准脚本在结尾统一打印 `[Hook/Stats]`（即使是同步 baseline）。

### 任务与切片编码（`monitoring/task.py`）

- 扁平 tuple：`(tensor, slice_dim, remove_batch, can_slice, slice_tuple, target_device)`；
- 两种 slice 编码：
  - 字符串版（兼容路径）与整型版（0/1/2/3，供原生快速路径）；
- 新增 `BackendFuture`：轻量 Future，直接调用原生 `future_*`，避免 `threading.Event` 分配。

## 环境变量与运行方式

- `MON_ENGINE_STATS=1`：启用 Python/C++/Hook 全链路统计；
- `MON_ENGINE_SLICE_STATS=1`：统计 slice 模式分布；
- `MON_NATIVE_BUILDER=1`：启用完全后端化构建器（默认开启）；
- `MON_NATIVE_BATCH=1`：启用 SoA 批量提交（与 builder 互斥，builder 优先）。

## 性能阶段性结果（BS=64，decode_steps=64，收集 hidden/attn，GPU）

- 同步（`hf_modified_hook`）：主时钟约 1.05–1.10s。
- 异步（原 submit 每 hook 一次）：主时钟约 1.30–1.65s（已废弃路径）。
- 异步（add_task 快路径）：主时钟约 1.07s，优于同步。
- 异步（SoA 批量）：主时钟约 1.01–1.03s，稳定优于同步。
- 异步（完全后端化构建器）：主时钟约 0.97–1.02s（与 SoA 接近，偶有更优）。

主要剩余开销：Hook 回调本身的 Python 调度/轻量参数准备（`build_us≈90–110ms`）。

## 已解决问题与仍在进行

- 已解决：
  - 原生扩展仅 dict 导致的 ValueError；
  - 结果槽泄漏导致的 OOM（`clear_completed_results/close`）；
  - 每 hook 一次提交导致的 1s 级 Python 累积开销；
  - 提交前重复序列化与字符串解析的额外成本（tuple/整型 slice/SoA）。

- 仍在进行/可选优化：
  - 进一步将 Hook 回调替换为原生回调（彻底减少 Python 调用次数）；
  - 将 indices 改为 LongTensor 直传，减少 pybind 小对象；
  - 按需减少 Hook 点数量或做层级聚合（线性降低次数）。

