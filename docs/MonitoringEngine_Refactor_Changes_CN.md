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

---

## 近期更新（代码组织 + 构建 + CPU Offload）

本节记录在原有文档基础上的新增与调整（与行为保持一致，聚焦工程化与可选特性）。

### 1) 头文件瘦身 + PImpl（行为不变）
- 将原先单文件内联实现改为头/源分离，采用 PImpl 隐藏实现细节：
  - 对外头文件：`monitoring/csrc/native_engine.h` 仅暴露 API；
  - 内部实现：`monitoring/csrc/native_engine_internal.h`（内部结构体、私有数据、工具函数声明）。
- 目的：显著减少头文件依赖与重编译面，增强可维护性；不改变任何外部接口与行为。

### 2) 源码按职责拆分（不过度细分）
- `monitoring/csrc/native_engine.cpp`：对外类薄封装、`create_hook_callback` 注册薄层；
- `monitoring/csrc/engine_core.cpp`：队列/后台线程、事件与流同步、`dispatch_step/process_step`、结果存取与周期清理；
- `monitoring/csrc/api_submit.cpp`：`submit_step/_soa/add_task/seal_step/resolve_all/future_*` 与统计更新；
- `monitoring/csrc/hooks.cpp`：`append_hook/append_hook_current_step/deduce_pos_dim`；
- `monitoring/csrc/slice.cpp`：`parse_slice_py/parse_slice_tuple` 与切片工具；
- `monitoring/csrc/bindings.cpp`：`PYBIND11_MODULE` 与 `create_engine` 工厂。

### 3) 构建与加载优化（可选，不改逻辑）
- Makefile：
  - `UNIFIED=1 make`：单 TU 构建（`csrc/unified.cpp`），最大化内联与优化；
  - `LTO=1 make`：启用链接时优化 `-flto`；
  - 默认多 TU 构建保持不变。
- 动态加载器（`monitoring/_native_engine.py`）：
  - `MON_NATIVE_UNIFIED=1`：仅编译 `unified.cpp`；
  - `MON_NATIVE_LTO=1`：传递 `-flto` 给编译/链接；
  - 默认按目录下所有 `.cpp` 源构建。

### 4) 后台流异步 D2H 到 CPU（可选开关）
- 目的：将 Hook 产物在后台缓存流上异步拷贝到 CPU，减轻显存压力/便于 CPU 侧消费；
- 行为：
  - 在 `run_task` 完成“去 batch/切片/降精度”后，如需 Offload：
    - 若 `MON_NATIVE_PINNED=1`（默认）：分配 CPU pinned 内存并 `non_blocking` 复制；
    - 否则：`tensor.to(cpu, non_blocking=True, copy=True)`；
  - 在 `process_step` 中确认 GPU→CPU 拷贝完成（同步当前缓存流）后再 `store_result`，保证 Python 读到的是落地数据；
  - Future 语义与接口保持不变。
- 开关：
  - `MON_NATIVE_TO_CPU=1`：对所有任务执行 D2H；
  - 任务级：传入 `target_device='cpu'` 仅对指定 Hook 开启；
  - `MON_NATIVE_PINNED=1/0`：是否使用 pinned 内存（默认 1）。
- 取舍与说明：
  - 主计算流不被阻塞（拷贝在独立缓存流上）；
  - 可能与前向产生带宽竞争（显存/PCIe/NVLink）；
  - 小量碎片拷贝建议合并或设尺寸阈值（后续可做）；
  - 未来可将“同步等待”升级为基于事件的就绪判定，进一步减少后台线程等待开销。

---

## 2025/Q4 更新：Pinned 复用池 + 全局回调 + 诊断（重要）

本阶段重点：解决 CPU pinned 内存用尽导致的 `cudaHostAlloc` 风暴、每步挂/卸钩子的 CPU 气泡、以及定位 Python/C++ 侧的“空白段”。

### A. Pinned 内存复用池（稳定 D2H 吞吐）

- 背景问题：每任务 `at::empty_like(..., pinned_memory=true)` 会在运行期频繁触发 `cudaHostAlloc`（页锁定），导致后段显著变慢。
- 方案落地（engine_core）：
  - 引入尺寸分级的 pinned 池（按 bytes 分桶），acquire→D2H→步末单次 `cudaStreamSynchronize`→pinned→pageable 的 host memcpy→立即归还池块。
  - 小块阈值（默认 64KB）以下一律 pageable，避免碎片 pinned 与 HostAlloc 抖动。
  - 统计：`pool_hits/pool_misses/pool_fallbacks/pool_high_watermark_bytes`、`host_memcpy_mb`。
- 相关环境变量：
  - `MON_NATIVE_PINPOOL=1`（默认在 `MON_NATIVE_TO_CPU=1 && MON_NATIVE_PINNED=1` 时启用）
  - `MON_NATIVE_PINPOOL_BINS_KB=256,512,1024,2048,4096,8192`
  - `MON_NATIVE_PINPOOL_SLOTS_PER_BIN=8`
  - `MON_NATIVE_PINPOOL_MAX_MB=512`
  - `MON_NATIVE_PIN_THRESH_BYTES=65536`

### B. 单步同步 + 主机 memcpy 并行（可选）

- 将 per-task 同步改为“每步一次同步”，先批量发起 D2H（非阻塞），步末统一 `cudaStreamSynchronize(cache_stream)`；
- Host 侧 pinned→pageable 的 memcpy 支持线程池并行（默认关闭）：
  - `MON_NATIVE_HOST_COPY_THREADS=4`（并发数），`MON_NATIVE_HOST_COPY_QUEUE_SIZE=512`（有界队列，队满回退串行）。
  - 统计：`host_copy_queue_depth/host_copy_total_tasks/host_copy_total_mb`。

### C. 全局回调（一次注册）+ 每步仅切换启用集 + 批量收集 Futures（避免 per‑hook GIL）

- 旧痛点：`TL::EnableHooks[fwd]` / `TL::ResetHooks` 每步挂/卸导致 CPU 气泡；原生路径若在回调内直接写 Python dict，会引入大量 GIL 获取。
- 新路径：
  - 一次性为所有 HookPoint 注册 C++ 回调（`create_global_hook_callback_sig`，is_permanent=True），回调仅“登记任务”与记录 `(name, token)`，不触发 GIL/不写 Python dict；
  - 每步：
    - 用 `set_enabled_hooks([...])` 下发启用名集合（由 `names_filter` 决定采集哪些）；
    - 前向结束后，用 `collect_step_futures_into(step_id, cache_dict)` 一次性把 `{name → BackendFuture(native_backend, token)}` 写入 cache（一次 GIL，批量写入）；
    - 若本步无临时 hooks（fwd/bwd 为空），跳过 hooks 上下文，避免 `TL::ResetHooks` 每步遍历。
  - Slice 解析：Python 侧只做“签名元组”编码，C++ 侧 `parse_slice_tuple` 解析，避免每步 Python 解析 slice。

### D. 诊断与 NVTX 覆盖（定位“空白段”）

- C++：
  - `MonEng::finalize_results`（步末 host memcpy 与归还）、`MonEng::pending_notify`（每次唤醒）、`MonEng::clear_results`（清理扫描耗时）、`MonEng::resolve_wait`（resolve 等待区间）。
  - 统计：`pending_notifies`、`clear_calls/clear_ms_total/clear_scanned_total/clear_ready_total`。
- Python：
  - `MonEng::PyStartStep/MonEng::PyEndStep`（步边界）、`CacheDict::clear`（脚本侧清理）、`TL::BuildCallback[...]`、`TL::RegisterHook[...]`、`TL::EnableHooks[fwd]/[bwd]`、`TL::ResetHooks`（可见每步 add/remove 开销，已通过永久注册规避）。
- 强制重建扩展：`MON_NATIVE_FORCE_BUILD=1`（绕过旧 .so，现编现载）。

### E. 已修复的问题清单（关键）

1) Pinned 用尽 + `cudaHostAlloc` 风暴 → 引入 pinned 池 + 小块 pageable 回退（已修复）。
2) per‑task 同步导致串行化 → 改为“每步一次同步”（已修复）。
3) Host‑copy 线程池的线程生命周期 → 析构/close 中显式 `stop + join`（已修复）。
4) 小块路径仍 pinned → 小块统一 pageable（已修复）。
5) `ptr_to_block_id_` 映射提前删除导致回收不稳 → 改为在 `release_pool_block` 时按 block_id 清理（已修复）。
6) 周期清理被注释 → 每 8 步自动 `clear_completed_results_internal()`（已恢复）。
7) Python 回调写 cache 的 GIL 热点 → 回调不写 cache；步后一次性收集 Futures（已修复）。
8) 每步 `EnableHooks/ResetHooks` CPU 气泡 → 永久注册回调 + 无临时 hooks 时跳过上下文（已修复）。

### F. 使用建议与示例命令

- 推荐 profile 命令：
  ```bash
  MON_NATIVE_FORCE_BUILD=1 MON_ENGINE_STATS=1 TL_ENABLE_NVTX=1 MON_NATIVE_CALLBACK=1 \
  nsys profile --output=results/nsight_async_perm --force-overwrite=true \
    --trace=cuda,nvtx,osrt --sample=cpu --sampling-period=10000000 --cpuctxsw=process-tree \
    --cuda-memory-usage=true \
    python benchmark/tests/hf_modified_async_only.py --batch-size 64 --steps 1 --warmup 1 \
      --collect-hidden --collect-attention --no-profile
  ```
- 常用调参：
  - `MON_NATIVE_HOST_COPY_THREADS=0/1/2/4`：CPU/内存总线干扰可调；
  - `MON_NATIVE_PINPOOL_MAX_MB=512~1024`：避免池早退；
  - `MON_NATIVE_PIN_THRESH_BYTES=65536~131072`：小块 pageable；
  - `--engine-delay-steps 1`：错峰 D2H（需要时）。

### G. 仍存的注意点与后续 TODO

- D2H 与前向的带宽竞争：属于物理资源争用，可通过 `delay_steps/线程池并发/阈值` 调整，或后续引入“最低优先级缓存流”和“步内 in‑flight 限流”（计划中）。
- 若需要 per‑step 切片/设备/移除 batch 动态变更：将补充 `set_hook_slice(name, slice_tuple)` 等 setter 在步前更新 HookConfig（接口预留）。

---

## 变更总览（文件级，便于追踪）

- C++ 原生后端：
  - engine_core.cpp：pinned 池 + 单步同步 + host‑copy 线程池 + 统计/NVTX + 析构清理。
  - native_engine.cpp/hooks.cpp：全局回调、启用集、步后收集 Futures、签名元组式 slice 解析。
  - bindings.cpp/native_engine.h/native_engine_internal.h：接口与数据结构声明。
- Python/HF 集成：
  - transformers/.../hook_points.py：永久回调注册、每步 set_enabled_hooks、步后 collect_step_futures_into、NVTX 覆盖。
  - monitoring/engine.py：Python start/end step 的 NVTX、Python 侧 clear 的 NVTX。
  - monitoring/_native_engine.py：`MON_NATIVE_FORCE_BUILD=1` 强制现编现载。
  - benchmark/tests/hf_modified_async_only.py：`CacheDict::clear` NVTX。
