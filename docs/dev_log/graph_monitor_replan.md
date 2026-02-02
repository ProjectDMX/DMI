# CUDA Graph Monitoring 重构计划

> 目的：回到“避免 Hook 阶段 tensor pointer 处理，同时兼容 CUDA Graph”的最初目标，对现有实现与文档脱节的问题重新规划。本文聚焦方向、差异与落地步骤，便于后续执行与对齐。

## 1. 背景与现状

- 旧文档与实现（`MonitoringEngine` + `append_hook/submit_step`）仍然描述 *Hook 即刻捕获 tensor*、塞入 C++ 队列后由后台线程处理的路径。
- 最近代码改动（SlotWriter、staging manager、Graph sink）依旧在 Hook 回调内接触 `at::Tensor`，只是把 pointer 写入 staging buffer，并通过 sink kernel 延长生命周期，未真正实现“Hook 不触碰 tensor”。
- 结果：文档与实现双重错位，且与原始诉求（彻底消除 Hook pointer 传递）相违背。

## 2. 新方向（目标态）

> 针对 CUDA Graph，我们区分 **capture**（录制）与 **replay**（重放）：Hook 在 capture 期间仍会读取 tensor，用于插入 `record/sink` 节点；replay 期间则完全不再执行 Python hook，达到零 CPU/零 pointer 的效果。

1. **Capture 期间：插入 Graph 原生节点**  
   - Hook 在 capture 时执行一次，用来自定义 `record_kernel`（写 slot metadata）与 `sink_kernel`（延长生命周期），以及 Graph 内 D2H 节点。  
   - 除了记录必要的 pointer/shape/stride 外，不做任何 Python 逻辑；这些节点被固化在图里。

2. **Replay 期间：零 CPU 干预**  
   - Python hook 不再运行；Graph replay 自动执行 record/sink/D2H，GPU 自主完成 metadata 写入和生命周期管理。  
   - 主线程只需 `g.replay()` + 记录 CUDA event，后台线程轮询 event 后直接读取 pinned host buffer。

3. **Step 结束统一读取/复制**  
   - 每个 step 仅在 graph replay 完成后读取一次 host buffer，解析所有 slot 的 pointer/shape/stride，再由 backend 决定是否发起 D2D/D2H 操作或在 GPU 上后处理。  
   - 不再在 hook/end_step 中逐个处理 tensor，彻底避免 per-hook overhead。

## 3. 与现有实现的差异

| 项目 | Legacy 实现 | 目标方案（Graph-Safe） |
| --- | --- | --- |
| Hook 行为 | 每次前向即刻拿 tensor，构造任务 | Capture 时插入 `record/sink` 节点；Replay 时零 hook |
| 监控入口 | `MonitoringEngine.submit_step` | Graph 内写 buffer + Step 末读取 host buffer |
| 元数据记录 | Hook 构建 `TaskSpec` | Graph record kernel 写 slot metadata |
| 生命周期控制 | 引擎消费进度决定 | Sink kernel 锁定到 Graph 末尾 |
| D2H 策略 | 后台 `cache_stream_` 异步 | Graph 内 D2H + pinned host buffer |
| 文档基线 | `MonitoringEngine_Implementation_CN` | 需新增 Graph-Safe 章节并标注 Legacy |

## 4. 研发计划

1. **文档对齐**  
   - 在 `docs/backup/docs/MonitoringEngine_Implementation_CN.md` 中增补“Graph-Safe 监控”章节，并将现有 `MonitoringEngine` 流程标记为 Legacy。  
   - 输出单独的设计稿（本文）说明目标架构、阶段目标与里程碑。

2. **最小可行方案 (MVP)**  
   - 原型：在一个小模型上，让每层激活写入专用 buffer（或通过 PyTorch graph rewrite 捕获 activations），Step 末尾由监控端读取。  
   - 确认：Hook 不访问 tensor；CUDA Graph 仍能 capture/replay；后端能拿到正确数据。

3. **监控引擎改造**  
   - 将 `MonitoringEngine` 的职责从 “hook 收集任务” 改为 “step 末尾统一收集/提交”。  
   - 清理 `append_hook/submit_step` 等不再使用的路径，提供新的 API：`register_capture_slot()`、`collect_step_metadata()`。

4. **Pipeline 互操作**  
   - 设计 capture 调度（哪些 HookPoint 需要 `record_kernel`）与 slot 映射；Graph capture 阶段自动生成 record/sink/D2H 节点。  
   - `end_step` 仅负责把 “本 step host buffer 可读” 的事件抛给 backend，后者解析 slot metadata。

5. **逐步迁移现有代码**  
   - SlotWriter / staging manager：评估是否仍有价值（若新缓冲机制替代它们，则移除或归档）。  
   - Benchmark pipeline：重新添加两种模式（Legacy vs Graph-Safe）以便对比和灰度。

## 5. 说明 / 已解决问题

- **如何避免修改模型内核？** 使用 `register_forward_hook` 在 capture 阶段调用 `torch.ops.monitors.record/sink`，只在图中追加节点，模型算子本身不变。  
- **Graph capture 如何写缓冲？** 直接在 `record_kernel` 中将 `data_ptr/shape/stride` 写到预分配的 GPU buffer，并在 capture 中调用 `host_buffer.copy_(gpu_buffer, non_blocking=True)` 生成 D2H 节点。  
- **后端读取时机？** 建议按 step 粒度：`g.replay()` → 记录 event → backend poll → 读取 pinned host buffer；延迟读取会导致 sink 锁定更多显存。

## 6. 下一步

1. 完成此计划文档的评审，确认方向一致。  
2. 更新主实现文档，标注旧路径；为新架构开章节。  
3. 制作最小实验：在 HF GPT-2 上验证 capture 插入 record/sink + replay 零 hook + host buffer 读取闭环。  
4. 基于实验结果，制定详细开发任务列表（Graph pipeline MVP → 监控引擎重构 → vLLM 场景适配）。

> 注：第 2 步（文档同步）已完成，以下实现步骤以新的 Graph-Safe 基线为准。

## 7. 落地实施步骤

1. **C++ 自定义算子落地**  
   - 使用 `torch.utils.cpp_extension.load` 做 JIT 编译，便于 MVP 阶段快速迭代；正式版再落地构建脚本。  
   - `TensorMetadata` 结构体强制 128B 对齐并加 `static_assert`；`record` 算子仅支持 CUDA Tensor。  
   - `sink(Tensor[])` 需要容忍空列表（直接 `return;`），防止极端配置下触发非法 kernel launch。

2. **GraphMonitor Python 入口**  
   - 提供 `GraphMonitor(model, max_slots)`，在 capture 阶段注册 hook 并调用 C++ ops。  
   - `finalize_capture()` 负责插入 `sink` 和 Graph 内的 `host_buffer.copy_`；`on_step_end()` 记录 CUDA event。  
   - 新增 `get_slot_mapping()`，返回 `slot_id -> layer_name`，供后端在 `collect_results()` 阶段解码。

3. **最小闭环实验（MVP）**  
   - 在 HF GPT-2 上 capture+replay，确认 host buffer metadata 与真实 Tensor 对齐，同时验证 Batch Size > 1。  
   - 在 Replay 结束、后台读取 metadata 后，校验引用的显存内容（或借助显存快照）以确认 lifecycle。  
   - 实验记录写回本文件，说明通过/失败以及修复措施。  
   - ✅ `tests_monitoring/test_graph_monitor.py` 已实现 Tiny MLP 的 CUDA Graph capture/replay 测试，用 slot metadata 校验 batch=2、shape/stride/device，一致则视为 MVP 打通。

4. **MonitoringEngine 重构**  
   - 新增 `GraphSafeEngine`，内部持有 `GraphMonitor` 与 slot map，提供 `prepare_for_model/start_step/end_step/collect_results`。  
   - `collect_results()` 使用 slot map 将 `slot_id` 映射回层名，并在 Python 端附加 step_id（GPU buffer 只存物理属性）。  
   - Legacy `MonitoringEngine` 保留，外部通过 flag/env 选择两种实现。  
   - ✅ `monitoring/graph_engine.py` 实现 `GraphSafeEngine` 与 `GraphSlotResult`，并在 `tests_monitoring/test_graph_monitor.py::test_graph_safe_engine_collect_results` 验证 start→replay→collect 流程。

5. **Backend 解析路径调整**  
   - `consume_graph_slots(metadata, step_id)` 新接口消费 Graph 模式数据；内部按旧有逻辑决定是否复制或直接 GPU 处理。  
   - 保持 delay_steps 等策略兼容；Step ID 由 `GraphSafeEngine` 在 `collect_results()` 阶段附加。  
   - 编写单测覆盖空 slot、部分 slot、多 step 等场景。  
   - ✅ `monitoring/graph_consumer.py` 提供 `GraphSlotConsumer`，并在 `tests_monitoring/test_graph_consumer.py` 覆盖 delay/空/多 step 情况；`GraphSafeEngine.consume_with()` 支持直接把收集到的 slots 交给消费者。

6. **Pipeline 对接（HF/vLLM）**  
   - Benchmark/推理脚本新增 `--monitoring-mode {legacy,graph}`，当选择 graph 时加载 `GraphSafeEngine` 并执行 capture/replay 流程。  
   - Nsight (Systems/Compute) 验证：Replay 阶段不再出现 Python Hook 的 NVTX；仅保留 record/sink/D2H 节点。  
   - 在性能测试中跳过前 3–5 个 warmup step，避免首次 replay 抖动影响统计。  
   - ✅ `benchmark/tests/profile_decode.py` 支持 `--monitoring-mode graph`，基准路径使用 `GraphSafeEngine + GraphSlotConsumer`，并在所有监控 drain 点调用新 helper。

7. **Benchmark & Idle Gap 调优**  
   - 记录 `hf_modified` vs `hf_modified_hook_graph` 的 tokens/s、idle gap，并分析 D2H copy 是否成为瓶颈。  
   - 若 idle gap 仍大，检查 host buffer 大小与 pinned 属性，必要时拆分多个 buffer 或调整复制频率。  
   - 将分析结果写入 `results/*` 与本文档，形成决策闭环。  
   - ✅ `benchmark/tests/profile_decode.py` 在 `--monitoring-mode graph` 下新增 `HFGraphDecodeRunner`，使用 CUDA Graph capture/replay + `GraphSafeEngine` 采集元数据；`masking_utils.eager_mask` 也改为避免 torch.where 中的 host->device 复制，使 capture 成功；warmup 结束后会 reset graph capture 以把 capture latency 计入主时钟，并额外提供 `hf_modified_graph` 实验用于纯 forward（无监控）的 capture/replay 基线。

8. **旧路径迁移与清理**  
   - SlotWriter / staging manager 等仅服务于 Legacy 的模块根据需要归档或删除。  
   - 在主实现文档中新增 “Legacy 组件清单” 表格，指向仍在维护的旧代码。  
   - `pytest tests_monitoring -k graph` 等回归通过后再提交合并请求。

9. **GraphSafeEngine Tensor Copy Delegation 计划**  
   - **目标**：在 graph 模式下不仅采集 metadata，还能像 legacy 模式一样触发每个 tensor 的 D2H/D2D 拷贝，并在 Nsight 中可见。  
   - **步骤**：  
     1. 为 `GraphSafeEngine` 增加可选 `backend_delegate`，默认指向 Python `_PythonBackend` 或 native backend；在 `collect_results()` 解析 `GraphSlotResult` 后，将其转换成 `MonitoringTask` 并调用 delegate 的 `submit_step`。  
     2. 扩展 `GraphMonitor`/GraphSafeEngine，使得 metadata 中保留实际 tensor `torch.Tensor` 的引用直到 delegate 拷贝完成；当前 sink kernel + event 机制继续确保在拷贝完成前内存不会复用。  
     3. 更新 `GraphSlotConsumer`，在 delegate 模式下把 step id、delay 配置传递给 backend，保持与 legacy 行为一致。  
     4. 编写 e2e 验证脚本：在 graph 模式下启用 delegate，跑 benchmark 并使用 Nsight 检查 D2H/D2D copy 是否出现，与 legacy 输出对比。  
     5. 完成后更新文档，说明 graph 模式支持 metadata-only 或 delegate-copy 两种模式，以及如何配置切换。
