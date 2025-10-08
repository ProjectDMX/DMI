# Monitoring Engine 异步缓存方案记录

## 背景

- 当前 Python 版 MonitoringEngine 已实现步级打包、后台 CUDA 流和延迟队列，主流耗时仍受 Python/GIL 与 queue 开销影响。
- 目标是在保持 `HookedGPT2Model` 现有 API 的同时，将所有耗时操作（任务登记、同步、切片、降精、CUDA stream 管理）迁移到 C++ 后端，使主计算流性能逼近 `hf_modified`，仅剩必要的带宽竞争成本。

## 设计范围

- 本轮专注单设备单 GPU 场景；多设备、多 GPU、DMA/磁盘等扩展留待后续。
- Python 继续提供 CLI/环境变量解析、与模型 hook 的集成、fallback 选择与异常转译。

## Python Wrapper 职责划定

- 负责在 `monitoring/engine.py` 中构造 metadata、收集每步任务，并在 `end_step()` 调用单次 `submit_step(step_id, tasks, delay_k)`。
- 维护 `CacheFuture` 薄封装：仅保存 `task_id` 和 C++ 引擎句柄，`ready/result` 直接调用 native 查询接口。
- 处理 `async_enabled=False` 或 C++ 扩展缺席时的同步 fallback（沿用旧逻辑）。
- 暴露调试/统计接口、CLI 开关及向上层传播异常。

## C++ Native Engine 设计

### 核心职责

- 负责队列、线程、CUDA stream、事件同步、sealing 逻辑以及所有张量操作（clone、slice、dtype 转换、设备迁移）。
- 在提交时持有生产者 stream 的事件，后台 stream 通过 `cudaStreamWaitEvent` 建立步级依赖，主流不再执行 `wait_stream`。

### 任务生命周期

1. Python `end_step()` 聚合生成 `StepSubmission`：包含 `step_id`、`delay_steps`、该步任务列表及处理所需 metadata。
2. C++ `submit_step` 在持有 GIL 的上下文中解析 metadata 后释放 GIL，将任务写入无锁 SPSC 队列；若队列满，则同步执行 `process_task` 并立即返回结果，保证正确性。
3. Worker 线程循环从队列取 `StepWork`，在专用低优先级 CUDA stream 上等待事件并依次处理任务：
   - `at::slice`/`at::narrow` 等实现位移
   - `at::clone` 确保与主流写路径隔离（不做 zero-copy）
   - `at::to` 完成 dtype/设备转换（非阻塞）
4. 处理结果写入 `ResultSlot`（tensor 或异常字符串），设置完成标记，唤醒相应 future。

### 队列与背压

- 采用 lock-free SPSC 环形队列；容量由 `--engine-queue-size` 或默认值控制。
- `try_push` 失败策略：同步 fallback 处理当前 `StepWork`，并记录计数器供调试；避免丢弃或无限阻塞主流。
- 结果槽与任务对象使用对象池复用，降低分配开销。

### CacheFuture 接口

- `CacheFuture` 保存 `task_id`，`ready()` 查询 native 状态，`result(timeout)` 阻塞等待并返回托管 tensor；若 native 记录异常，则在 Python 端重新抛出 `RuntimeError`。
- `resolve_all()` 调用 `await_all()`，确保所有 step 数据落盘后才返回。

### GIL 与线程

- `submit_step` 在转换 Python metadata 后调用 `py::gil_scoped_release`；worker 始终在无 GIL 状态下运行。
- 线程创建、销毁、CUDA stream 分配均在 C++ 中进行，Python 仅负责生命周期钩子。

### 延迟 K 步

- C++ 维护 `std::deque<PendingStep>`，`submit_step` 将 step 标记为 sealed；当 sealed 队列长度超过 `delay_steps` 时将最旧 step 投喂 worker。
- 所有任务在提交时即记录事件并入池，即便延迟处理也不会影响主流生命周期。

### 调试与指标

- 提供 `struct EngineStats`：包含队列深度、同步 fallback 次数、平均处理时延等；Python wrapper 在 `MON_ENGINE_DEBUG` 时打印。
- 支持 `dump_pending_steps()` 诊断卡顿。

## 实施阶段

1. **Phase 0 – 接口冻结**
   - 梳理 Python wrapper 的 public API，补齐 `CacheFuture` 行为测试，记录延迟/队列指标基线；移除文档中旧的“零拷贝”描述。
2. **Phase 1 – Native MVP**
   - 在 `monitoring/csrc/` 实现最小 C++ 引擎：任务结构、SPSC 队列、事件同步、clone+slice+dtype；提供 `submit_step`/`await_result` API，并接入 Python wrapper。
   - 保留同步 fallback，编写基础单元测试验证结果一致性。
3. **Phase 2 – 功能补齐**
   - 实现 delay-K、对象池、统计信息、异常传播、`--engine-queue-size` 背压策略。
   - 支持批提交以减少 Python↔C++ 边界次数，暴露调试计数器。
4. **Phase 3 – 默认切换**
   - 默认启用 C++ 引擎，缺失编译产物或 CPU-only 时回退 Python 并打印提示。
   - 引入 deprecated warning，计划在后续版本移除 Python 实现。
5. **Phase 4 – 验证与回归**
   - 性能基准：比较 `hf_modified_hook_async` 与 `hf_modified` 的 `main_duration`；确认差距主要来源于激活带宽。
   - 扩展测试矩阵：压力测试、异常注入、不同 dtype；在 CI 中加入 native 编译和 pytest 覆盖。

## 测试与验证计划

- 单元测试：
  - Python fallback vs C++ 结果一致 (`torch.testing.assert_close`).
  - 队列满触发同步处理，结果仍正确。
  - 异常传播（模拟 ATen 抛错）。
- 集成测试：
  - `profile_decode.py` / `profile_inference.py` 在异步模式下跑通并生成指标。
  - 长时间压力测试确保无内存泄漏（ASan/Valgrind 手工执行）。
- 性能回归：
  - 每次提交记录 `main_duration`、后台耗时、队列深度、fallback 次数。

## 风险与缓解

- **构建依赖**：通过 `setup.py` + `torch.utils.cpp_extension.CUDAExtension` 编译，CI 中启用 ccache；若编译失败自动 fallback 并提示。
- **异常处理遗漏**：C++ 强制捕获 `c10::Error` 并存入结果槽；Python 端统一抛出。
- **同步 fallback 频繁**：通过指标监控，必要时扩大队列或优化批提交。

## 交付预期

- Native 引擎上线后，主流额外开销预计从 ~1.8s 降至 ~1.1s（64×64 decode 样例），与 `hf_modified` 的差距主要由必需的激活读取带宽决定。
- Python 路径继续作为安全兜底；后续计划在稳定后移除。

## 暂缓事项

- 多 GPU/多设备、跨设备 DMA、磁盘写入、压缩等高级功能暂不纳入本轮；待单 GPU native 后端稳定后另行规划。
