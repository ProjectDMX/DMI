# Monitoring Engine 异步缓存方案记录

## 背景与目标

- 项目需要对自定义的 `HookedGPT2Model` 进行性能剖析，关注“带 Hook 系统但不开启缓存”以及“开启 Hook 并收集激活”两种场景的额外开销。
- 现有的 `run_with_cache` 在前向过程中同步执行 `pos_slice.apply`，会频繁触发 `aten::slice`，导致主 CUDA stream 阻塞，profile 中出现大量 slice 开销。
- 希望保留全部 Hook 缓存能力，同时把切片/缓存等重量操作从主推理路径中剥离出来，为后续扩展（GPU→CPU DMA、磁盘写入等）铺路。

## 已完成工作概述

1. 在 `benchmark/tests/profile_decode.py` 中新增两个 baseline：
   - `hf_modified`：使用 `HookedGPT2Model` 但不调用 `run_with_cache`；复用原 `lm_head` 计算 logits，用于度量“仅挂载 Hook 系统”的开销。
   - `hf_modified_hook`：使用 `run_with_cache` 收集激活，打开 `--collect-hidden` / `--collect-attention` 时缓存会包含全部 Hook 点。
2. 通过复用同一个 `lm_head` 保证 logits 与原 Hugging Face GPT-2 对齐。
3. 在 profile 中观察到，`hf_modified_hook` 触发大量 `aten::slice`，源自 `run_with_cache` 内部同步切片。
4. 引入 `monitoring` 包，提供 `MonitoringTask` / `CacheFuture` / `MonitoringEngine` 骨架，以及在 `HookedRootModule.get_caching_hooks` 中的初步接入（默认仍走同步路径，只有显式启用 engine 时才会返回异步 future）。
5. 基准脚本加入异步 Hook 基线，并同时打印主计算流耗时和异步任务完全结束耗时，方便对比开销差异。
6. 优化（第一阶段已落地）：
   - 步级事件聚合：新增 `MonitoringEngine.start_step()`，后台流按 step 等待一次事件，移除 per-task 事件/record_stream。
   - 仅在 `requires_grad=True` 时 `detach()`，推理默认不再 detach。
   - `profile_*` 中移除了强制的逐步 `cuda.synchronize()`，`main_duration` 更贴近用户端推理时延。
   - 预留 `cache_dtype`（可选）用于后台降精存储，减少带宽（未在 CLI 暴露）。

## 现存痛点

- 切片/拷贝在主流同步执行，直接阻塞 decode 主流程。
- 每步 decode 需要处理 12 层多个 Hook，`aten::slice` 积累成明显的性能热点。
- 未来还要叠加 DMA、写盘等异步能力，现有结构耦合度高，扩展困难。

## 核心方案概述

构建一个解耦的 **Monitoring Engine**：

1. **Hook 中只登记任务**
   - `save_hook` 不再立刻切片，而是：
     1. 在 no_grad 场景沿用张量（仅在需要时做 `requires_grad` 防护）。
     2. 创建 `MonitoringTask`（包含 tensor、pos_slice、hook 名等元信息）。
     3. 调用 `monitoring_engine.submit(task)`，返回 `CacheFuture` 存入 cache。

2. **Monitoring Engine 异步处理**
   - Engine 内维护：
     - StepWork 队列（自建 ring buffer）。
     - 专用低优先级 CUDA stream。
     - 后台线程 `MonitoringWorker`。
   - Worker 处理流程：
     1. 取出 StepWork（整步任务），遇到哨兵退出。
     2. `with torch.cuda.stream(cache_stream): cache_stream.wait_stream(producer_stream)`。
     3. 遍历任务，按需切片/降精/迁移，串行写回 `CacheFuture`。

3. **CacheFuture / Cache API**
   - cache 条目保存的是 `CacheFuture`，提供：
     - `ready()`：是否完成。
     - `result()`：阻塞等待并返回张量。
   - Cache 新增辅助方法：
     - `resolve(name=None)`：解析指定或全部激活。
     - `to_dict()`：一次性获取同步字典。
   - `run_with_cache` 返回 `(model_out, cache)`；调用方可在需要时通过 `cache.resolve_all()` 等接口同步。

4. **生命周期管理**
   - Engine 提供 `start_step()` / `end_step()` / `resolve_all()` / `close()`：
     - `start_step()` 在主流进入 decode / prefill 步时调用，递增步号。
     - `end_step()` 执行一次 `cache_stream.wait_stream(生产流)` 并将该步所有任务打包入队。
     - `resolve_all()` 将剩余步补齐入队并阻塞等待后台完成。
     - `close()` 推送哨兵到队列，回收线程与 CUDA stream。
   - `run_with_cache(async_enabled=False)` 仍保持旧的同步行为，以兼容既有代码。

## 代码模块规划

- `monitoring/engine.py`
  - `MonitoringEngine`、`MonitoringWorker`、事件回调等。
- `monitoring/task.py`
  - `MonitoringTask`、`CacheFuture` 及扩展点。
- Hooked 模块：
  - `HookedRootModule` 增加 `self.monitoring_engine`，在 `get_caching_hooks` 中调用 Engine。

## 后续扩展方向

- 新增 Handler：
  - `TransferHandler`：GPU→CPU DMA（non_blocking + pinned memory）。
  - `CompressionHandler` / `DiskHandler`：后台压缩或写盘。
  - 自定义统计、日志收集插件。
- 增加监控接口，输出队列长度、平均处理时间等指标。
- 在 decode 基准中测试异步模式，验证 main stream 上的 `aten::slice` 是否显著减少，并评估整体吞吐影响。

## 开发步骤清单（按顺序）

1. 搭建 `monitoring` 目录与核心骨架类。
2. 改造 `HookedRootModule.get_caching_hooks`，接入 Engine；保留同步 fallback。
3. 更新 `run_with_cache`，增加 `async_mode` 开关，返回异步 `cache`。
4. 编写最小单元测试：同步 vs 异步结果一致；无 CUDA 环境 fallback 正常。
5. 在 `profile_decode.py` / `profile_inference.py` 中开启异步模式，确认主流性能改善。
6. 预留配置入口，未来根据命令行或环境变量开启 DMA/写盘等附加功能。

## 最近迭代（已落地）

- 步级同步：新增 `MonitoringEngine.start_step()` / `end_step()`，按步聚合依赖，后台流一次等待。
- wait_stream 优先：在支持的环境下以 `cache_stream.wait_stream(producer_stream)` 代替事件，减少 CUDA runtime 开销。
- 后台流最低优先级：使用 `torch.cuda.Stream(priority=max_pri)` 创建 cache stream，降低对主流的资源干扰。
- 队列限流（可选）：新增 `--engine-queue-size` CLI，限制待处理任务数量，避免瞬时抢带宽。
- 可选降精（谨慎）：新增 `--cache-dtype {none,fp32,fp16,bf16}`，仅影响缓存存储。当前 decode 配置下建议保留 `none`，否则转换开销可能大于带宽收益。
- **StepWork + RingBuffer**：主线程按步打包任务，后台线程一次性处理整步，显著降低 Python/GIL 调度开销。

## 进行中 / 下一步

- 评估是否以可选策略方式重新提供“大 buffer 合并拷贝”（默认关闭，避免额外 D2D）。
- 继续削减主线程任务登记成本，必要时考虑 C++/CUDA Hook 输出。
- 利用 `--engine-delay-steps` 做步级延迟，缓解后台与主流带宽竞争。
- 增强监控指标，暴露 StepQueue 深度、平均处理时延等数据。

## 异步处理流程详解

1. **Hook 登记任务（主 stream）**
   - `HookedRootModule.get_caching_hooks` 生成的 `save_hook` 只做必要的张量守护（grad check）。
   - 构造 `MonitoringTask`（含 tensor、hook 名、`pos_slice` 等），交给 `monitoring_engine.submit`。
   - `submit` 返回 `CacheFuture`，存入 cache（`cache[hook_name] = future`）。

2. **Engine 排队**
   - `MonitoringEngine` 按步收集任务列表，`end_step()` 时打包成 `StepWork`。
   - `StepWork` 通过 ring buffer 队列交给后台线程；CPU fallback 时仍可直接返回结果。

3. **后台处理（专用 CUDA stream）**
   - Worker 进入 `with torch.cuda.stream(cache_stream)`，调用 `cache_stream.wait_stream(producer_stream)` 完成步级同步。
   - 遍历 StepWork 中的任务，根据 `pos_slice` 决定是否执行切片（`Slice.identity` 会跳过）。
   - 可选执行降精/跨设备迁移等操作，最后将结果写回 `CacheFuture`。
   - 完成整步后 `queue.task_done()` 并清空 StepWork，释放引用。

4. **使用缓存**
   - cache 项是 `CacheFuture`，调用 `future.ready()` 检查状态，`future.result()` 阻塞直到数据可用。
   - `Cache.resolve_all()`/`resolve(name)` 会遍历 future 并返回实际张量，供后续分析。

5. **生命周期管理**
   - `run_with_cache(async_mode=True)` 返回 `(model_out, cache)`；调用方可在任意时刻 `cache.resolve_all()`。
   - 需要结束时执行 `engine.close()`，向队列推送哨兵 `None`，等待 worker 释放 CUDA stream/线程资源。

---

此文档用于记录 Hook 异步化的背景、设计与开发计划，供后续刷新记忆或继续开发时参考。
