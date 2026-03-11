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

## 10. C++ Native Backend 行为调查

> 参考 `monitoring/csrc/native_engine*.{h,cpp}`、`monitoring/task.py`：该实现仍是 graph delegate 需要对接的核心后端。

- **任务构造**：  
  - Python hook 通过 `MonitoringTask.native_payload` 或 native builder (`append_hook_current_step`) 构建 `TaskSpec`。  
  - `submit_step`/`submit_step_soa` 将任务与 `step_id` 封装进 `StepWork`；`seal_step` 会附带 step event（`cudaEvent_t`）用于和主 forward stream 同步。  
  - `delay_steps_` 逻辑在 `seal_step` 中实现，确保 step 可根据策略延迟入队。

- **Worker 线程** (`monitoring/csrc/engine_core.cpp`)：  
  - 独立线程（`MonEngWorker`）消费 `StepWork`，在单独的 `cache_stream_` 上运行所有任务。  
  - 每个任务经 `run_task` 执行 remove_batch、slice、dtype cast、`target_device` 迁移；当需要 GPU→CPU 时在 `cache_stream_` 异步 `copy_`。  
  - Step 层面仅同步一次 `cudaStreamSynchronize(cache_stream_)`，之后再将结果写入 `ResultSlot`。

- **内存/拷贝策略**：  
  - `MON_NATIVE_TO_CPU`/`MON_NATIVE_PINNED` 控制是否默认搬到 CPU 以及是否使用 pinned 内存。  
  - `MON_NATIVE_PINPOOL*` 配置 struct-of-arrays 的 pinned pool，`acquire_pinned_block`/`release_pinned_block` 避免频繁 malloc。  
  - `MON_NATIVE_HOST_COPY_THREADS` 启用 host 复制线程池，把 pinned tensor 异步 memcpy 到 pageable。

- **SoA / builder**：  
  - `submit_step_soa` 支持 struct-of-arrays (`spec["tensors"]`, `"slice_dims"`, `"remove_batch"` 等)，`native_batch` 路径就是为了喂这个接口。  
  - builder 回调 (`create_global_hook_callback_sig`) 在 Capturing 阶段一次性注册 slice/device；运行时只 append tensor，提高 hook 端效率。

- **Capture schedule / gating**：  
  - `set_capture_schedule`, `begin_request`, `begin_step` 控制 `capture_enabled_`。hook 回调首先判断该原子量，未命中 schedule 时直接返回。  
  - Graph delegate 仍需沿用这些 API，保证 capture 频率与 Legacy 一致。

- **Future / 清理**：  
  - 每个任务会注册 `ResultSlot`，通过 `future_wait/result` 暴露给 Python；`resolve_all`、`clear_completed_results` 处理 backlog 并在关闭时清理线程。  
  - graph delegate 要求仍使用这些接口，以便现有 `BackendFuture`、`MonitoringEngine` 逻辑无缝复用。

## 11. Graph → Native Backend（同步版）实施计划

> 新目标：把 shadow block 的解析 / SoA 填充 / 任务提交全部下沉到 C++，Python 只负责把 metadata buffer 和 step 事件交过去。

1. **C++ ShadowBlock Parser**  
   - 在 `monitoring/csrc` 侧新增 helper（或扩展 `graphmonitor_ops`），直接接收 metadata tensor（shadow block）并输出 native backend 可消费的 `TaskSpec`/SoA 结构。  
   - 解析逻辑：C++ 遍历 slot 行、创建 alias view、推断 slice_dim/pos_dim、填写 remove_batch/can_slice 等字段，完全避免 Python 端逐项 `alias_tensor` 的开销。

2. **C++ Graph Delegate**  
   - 新建 `GraphNativeDelegate`（C++ 实现 + pybind 暴露），接口为 `submit_and_resolve(step_id, metadata_tensor, stream_handle)`。  
   - 内部流程：`parse_shadow_block` → `submit_step_soa` → `seal_step(stream_handle)` → `resolve_all(true)`；过程中直接把 alias tensor append 到 native backend 内部队列。  
   - 若解析/提交失败，抛出异常让 Python fallback 到 metadata-only 模式。

3. **GraphSafeEngine → Delegate 连接**  
   - Python 仅需把 metadata view、ready step id、对应 `cudaStream` 句柄传入 delegate。  
   - 删除/精简 Python 侧 `_build_native_batch`、`_alias_tensor` 等逻辑，转而调用 `GraphNativeDelegate` 的 C++ 路径。  
   - `end_step()` 仍记录 sink event；stream handle 透传给 delegate 供 `seal_step` 使用。

4. **Benchmark/CLI**
   - `profile_decode.py` 增加 `--graph-copy-mode {sync,disabled}`，当选择 sync 时实例化 C++ delegate 并传给 `GraphSafeEngine`。  
   - Benchmark 验证 Graph 模式 + delegate 能看到真实的 D2H/D2D copy，并对比 legacy async 的吞吐。

5. **验证/测试**
   - C++ 单元：ShadowBlock parser 的 shape/stride/alias 生命周期测试；`submit_and_resolve` 异常路径。  
   - Python 端集成：`tests_monitoring/test_graph_delegate.py` 更新为调用 C++ delegate；Nsight trace 证明 copy 在 Graph replay 之后同步完成。  
   - 文档更新当前限制：仅支持同步 copy，异步/backpressure 参考第 12 节。

> 完成上述步骤后，GraphSafeEngine 的 graph 模式即可完全依赖 C++ 解析与 native backend 队列，消除 Python 热路径。

### 11.1 单 Graph 同步版落地步骤（当前阶段）

1. **GraphMonitor 事件与 metadata 队列**
   - `on_step_end()` 不再覆盖单一 `_step_event`；改为为每个 step 创建独立的 `cudaEvent_t`，并将 `(step_id, event)` 追加到 deque。  
   - 同步阻塞模式下，如果 step t 结束后立刻 `wait=True`，仍会等待该 event 完成；若暂时未等待，也必须保留 event，直到下一次消费。

2. **GraphSafeEngine 收集逻辑**
   - `collect_results(wait=False)` 遍历队首事件，只有当队首 ready 时才解析该 step 的 metadata；一旦消费，就 pop 出队并记录 snapshot、stream handle。
   - `collect_results(wait=True)` 在队首事件未 ready 时阻塞，保证“步 t 结束后可立即同步复制”。  
   - `drain_ready_results()` / `resolve_all()` 基于上述队列实现，确保不会遗漏任一步骤。

3. **Decode 循环集成**
   - 在 `HFGraphDecodeRunner` 的 replay 路径里，step 结束后调用 `process_monitoring_results(wait=True)`，同步等待当前 step 的 copy 完成，再进入下一 step。  
   - Warmup、prefill 与 benchmark 代码也按此方式调用，保证单 Graph 同步版每步都能稳定复制；后续切换到异步/双 graph 时，只需调整 `wait` 模式或调用频率。

4. **日志与验证**
   - 保留 `[GraphDebug] delegate submit…` 输出，确认每一步都被提交。  
   - Nsight 中应看到 decode 每步 forward 后紧跟 D2H，无大段 `cudaStreamSynchronize`；`graph_native_backend.get_stats()` 的 `total_steps` 应与 decode 步数一致。

> 依据单 Graph 的限制：“下一步 replay 会覆盖上一步 tensor”；因此同步版必须在每步 forward/`end_step` 后立即消费事件并复制，不能指望保存旧 tensor。此实现为后续双 graph / staging buffer 的异步方案打底。

## 12. Future Work：双 Graph / 双缓冲并行

> 目的：解决“Graph replay 写回固定地址”导致的 forward 与 D2H copy 互斥问题，让 step t 的 copy 与 step t+1 的 forward 真正并发。

- **问题现状**：单 Graph capture 情况下，每次 replay 都会把激活写回 capture 时的同一显存地址；即便 host 侧 alias 了 tensor，也无法阻止下一步 kernel 立即覆盖这块内存，因此 GraphSafeEngine 只能在复制完成后再启动下一次 replay。
- **候选方案：双 Graph 轮换**  
  - 分别 capture `GraphExec[0]` 与 `GraphExec[1]`，在 capture 过程中让 allocator 在 clean state 下依次分配两套 buffer；依赖 `_capture_anchors + sink` 让两套 graph 的激活在 replay 完成前不会互相复用。  
  - runtime 以 step parity 选择 graph：奇数步 replay graph 0，同时复制 graph 1 的结果；偶数步反之。这样 copy stream 永远读的是“上一套 graph”的 buffer，避免与当前 forward 写同一地址。
- **替代方案：双 staging buffer**  
  - 保持单 graph，但在 capture 中每个监控点先 copy 到 staging buffer（A/B 双份），GraphSafeEngine/后端只访问 staging buffer。  
  - 需要在 graph capture 里插入 `record -> copy_to_buffer[current]` 节点，并让 sink kernel 引用 staging buffer，以保证 buffer 生命周期覆盖到 host copy 结束。
- **实现要点**  
  1. capture 前清理 allocator（empty cache + 预热）确保两套 graph 拿到可预测的地址。  
  2. 每套 graph capture 完成后立即固定 `_capture_anchors`，避免内存被复用；GraphSafeEngine 要维护 per-graph sink event。  
  3. backend 需要知道当前消费的是哪一套 buffer，以便在 `seal_step(step_id, stream_handle)` 时等待正确的 event。  
  4. 在文档与 benchmark CLI 中暴露配置（`--graph-double-buffer`），供未来实验开启。
- **风险/成本**：显存消耗翻倍、capture 时间增加；需要验证对大模型能否容忍。若不可行，可考虑混合策略（prefill 阶段单 graph，decode 阶段双 graph）或仅对热点 hook 使用双缓冲。

> 在完成 delegate 回接后，这一章节将作为下一阶段性能优化的探索方向，帮助我们重新获得 legacy 异步 copy 的吞吐优势。

## 13. Future Work：基于内存伪装的双 Graph 乒乓架构（零拷贝）

> 方案一 Design A – Dual-Graph Ping-Pong Orchestration（Zero-Copy）  
> 目标：通过“原地生产”思路，让监控 tensor 在 capture/replay 阶段直接写入预留 slot，彻底消除 D2D 拷贝和 allocator 碎片。

1. **内存拓扑**  
   - **瞬态共享池（Transient Shared Pool）**：存放所有无需保留的中间量（LN、MatMul buffer 等）；Graph α/β 共用同一个 mempool，因计算拓扑一致可实现 100% 复用。  
   - **持久化乒乓槽（Persistent Ping-Pong Slots）**：为需要监控的 tensor 额外预分配两组物理地址 `Slotset A/B`，完全与共享池隔离。

2. **执行流**  
   - **Step 2t / Graph α**：瞬态变量写入共享池；被监控的 tensor 直接产出在 `Slotset A`。Replay 结束后，pool 数据可立即复用，`Slotset A` 保留给 backend 异步读取。  
   - **Step 2t+1 / Graph β**：共享池复用原地址（覆盖写）；监控 tensor 写入 `Slotset B`。此时 backend 可并行消费 `Slotset A`，实现 compute / I/O overlap。  
   - Graph capture 前需切分好 mempool 与 slot，Graph replay 时按 step parity 切换 slot，sink/event 机制保证 Slotset 生命周期。

3. **优势**  
   - **True Zero-Copy**：监控数据“出生即在 slot”，HBM 不再承受额外 D2D；只需 sink/event 保障生命周期。  
   - **Allocator 无碎片**：共享池里不再混入持久化 tensor，Graph α/β 可以轮流完全复用该 pool。  
   - **完美并发**：当 Graph α 在共享池上运行时，backend 正读 `Slotset B`；下一步反之，forward 与 copy 无需互相等待。

> 实现难点在于：需要对模型 forward 层级的算子做“输出地址劫持”，确保所有被监控 tensor 的 storage 绑定到 slot，而其它临时激活始终走共享池。待 delegate 模式稳定后，可评估对核心模块（Attention/MLP）改写或自定义 kernel 的代价，逐步验证该方案的可行性。

## 14. Future Work：基于融合内核的异步流水线架构（D2D）

> 方案二 Design B – Fused Post-Hoc Copy Pipeline（非侵入式快照）  
> 思路：保持模型现有的内存布局，通过图末尾的融合 kernel 将监控 tensor 聚合到 staging buffer，再异步回传。

1. **内存拓扑**  
   - **单体激活池（Monolithic Activation Pool）**：模型所有中间量（含监控/非监控）仍由 PyTorch allocator 统一管理，step 结束即视为脏数据，可被下次 replay 覆盖。  
   - **暂存环形缓冲区（Staging Ring Buffer）**：独立显存区域，用作“快照”容器；按帧（step）循环使用，供后台 D2H/D2D 消费。

2. **执行流**  
   - **阶段 I – Compute**：单个 CUDA Graph 完成 forward，所有 tensor 写入激活池，无需定制 allocator。  
   - **阶段 II – Fused Gather**：在 Graph 末尾加入自定义 gather kernel，读取激活池中散布的目标 tensor，按 Struct-of-Arrays 方式打包到 staging buffer 当前帧；kernel 需等待 compute stream 完结后启动，以避免读写冲突。  
   - **阶段 III – Async D2H/D2D**：通过独立 stream 将 staging buffer 的内容搬到 CPU 或其它设备，实现 compute / I/O 的流水并行；下一 step 可立即启动，唯一共享资源是 HBM 带宽。

3. **特性评估**  
   - **兼容性**：无需修改模型算子或输出地址，只需在 Graph capture 阶段追加一个 gather kernel；对于 Transformer/HF 等通用模型改动较小。  
   - **可扩展性**：staging buffer 只存真正需要的 tensor，显存成本与监控量成正比；ring buffer 可配置帧数以支撑多个未完成的异步 copy。  
   - **性能权衡**：引入额外 D2D 读写，HBM 带宽成瓶颈；当监控数据量大时，Latency 会线性上升。可通过压缩/采样/分批 gather 缓解。

> 该方案适用于“希望保持模型实现无侵入，却需要一定程度 async copy”的情境。后续可探索：gather kernel 如何与 `GraphMonitor` slot metadata 对齐、如何将 gather 阶段与 SoA aggregation 融合、以及如何在 Nsight 中观察到明确的 copy stream。
