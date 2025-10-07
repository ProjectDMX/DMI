# 异步 Hook 抓取激活（Monitoring Engine）会议说明（面向非专项）

目的：在不影响主推理路径（prefill/decode）的前提下，抓取 GPT‑2 的中间激活（hidden states / attention），用于后续分析与可视化。

核心要求：
- 用户关心“主计算流完成”的时间（main_duration），而不是异步写盘/搬运等全部结束的总时间（total_duration）。
- 保留全部 hook 缓存，不牺牲正确性。

我们做了什么（自下而上）
- 监控引擎 MonitoringEngine（HF_Prometheus/monitoring/engine.py）
  - 专用后台线程 + 低优先级 CUDA 流（减少与主流资源争用）。
  - 步级边界：start_step()/end_step() 封装每一步 decode；主流在 end_step 后用 `cache_stream.wait_stream(producer_stream)` 建立一次性依赖（替换 per‑task 事件）。
  - 任务队列 + 节流：--engine-queue-size 控制 in‑flight；避免瞬时带宽冲击。
  - StepWork + RingBuffer：每步打包全部任务一次入队，后台线程一次性处理整步，降低 Python/GIL 开销。
  - K 步延迟：--engine-delay-steps（环形缓冲），允许后台缓几步再处理，进一步让主流“像没开 hook 一样”。
  - 解决生命周期：对引用的张量保持强引用；仅在 no_grad 下工作（推理场景），减少 detach/record_stream 使用频率。

- Hook 集成与基线
  - 同步版 `hf_modified_hook`：原地 slice/clone 复制激活（用于对照）。
  - 异步版 `hf_modified_hook_async`：仅“登记任务”，主流继续前进；复制/整理在后台流进行。profile 脚本打印 main_duration 与 total_duration 两个时间。
  - 其他基线：`hf_modified`（无缓存）、`hf_hook`（HF 原生 hook）。

我们发现的问题与修复
1) detach() 与 record_stream() 的过度使用
   - 现象：trace 中出现明显的 `aten::detach`，`aten::record_stream` 时间累积（几十毫秒级）。
   - 原因：在 no_grad 推理下，detach() 并不能带来额外安全性；逐条 record_stream 会让分配器延迟复用，且维护成本高。
   - 修复：no_grad 默认移除 detach；record_stream 从“每个条目”降到“步级/必要时”，多数路径移除；正确性由步级屏障与强引用保证。

2) 事件（Event）开销与细粒度同步
   - 现象：`cudaEventRecordWithFlags`/`cudaStreamWaitEvent`/`cudaStreamIsCapturing` 在异步路径合计几十毫秒（每任务都打一遍）。
   - 修复：改为 `cache_stream.wait_stream(producer_stream)` 的步级屏障，不再为每个任务单独打事件；开销基本消除。

3) slice/clone 等 CPU 侧整理开销
   - 现象（同步版）：`aten::slice` 等重排算子长期为 Top‑1 开销来源。
   - 措施：将切片/整理延后到后台流；异步版通过 StepWork 批处理减少了 launch/调度。

4) cache_dtype（fp16/bf16）引入额外开销
   - 结论：在当前 decode 设置下，类型转换的代价抵消甚至超过带宽收益，导致更慢。建议默认 `--cache-dtype none`，待更激进的合并拷贝/压缩后再评估。

5) delay_steps>0 的偶发卡死（仍在处理）
   - 现象：`--engine-delay-steps 1` 时运行卡住（GPU 利用率 0%，显存占用高）。`resolve_all()` 阻塞在队列 `join()`。
   - 已做：
     - prefill/decode wrapper 加 `try/finally`，确保 `end_step()` 一定执行；
     - `resolve_all()` 同时flush已 seal 与未 seal 的 bucket；
     - 分块合并拷贝（避免一次性超大 buffer 触发分配器卡住）。
   - 仍需：在 `resolve_all()` 加 watchdog 与详细 dump，定位具体卡住的 step/bucket/队列状态；并在 decode 封装点增加“开始/结束步”的 debug 线索（已支持 MON_ENGINE_DEBUG=1）。

指标与结果（样例：64×64 decode，单步/两步与多轮对比）
- 打印样例（第 6 轮）：
  - `hf_modified`: 2.757s（1485 tok/s）
  - `hf_modified_hook`: 5.0265s（815 tok/s）
  - `hf_modified_hook_async`: main 4.5728s / total 4.5805s（896/894 tok/s）

主要开销对比（旧 trace 与最新实现对比）
- 同步版相对 `hf_modified`：Top‑1 仍然是 `aten::slice` 与 `aten::as_strided`（同步路径无法规避）。
- 异步版（StepWork 之前）：事件 + 大量 D2D 拷贝成为主瓶颈。
- 异步版（StepWork 之后）：事件与 D2D 拷贝消失，主开销转为 Python 调度（队列/锁）与必要的 `addmm/matmul/layer_norm` 抬头。

结论：我们已经把 GPU 上的额外拷贝消除，下一步重点是继续削减主线程登记成本（批提交、无锁结构、必要时 C++ 扩展），让 main_duration 贴近 `hf_modified_hook` 甚至进一步下降。

下一步计划（从易到难）
1) 单步批量提交 + 更轻量的队列（StepWork 已上线，后续继续打磨无锁结构、减少锁争用）。
2) 如需连续缓冲，再评估“大 buffer 合并拷贝”作为开关选项（默认关闭，避免额外 D2D）。
3) 更智能的节流：结合 in-flight 限制与任务大小排序，保障主流算子稳定。
4) 降精/压缩（再次评估）：在需要时引入轻量量化以进一步减小后台工作负载。
5) CUDA Graphs（可选）：捕获主前向图，进一步压低 launch/调度开销；hook 的异步逻辑保持图外。

如何复现
- 无延迟异步（对比 main vs total）：
  - `python HF_Prometheus/benchmark/tests/profile_decode.py --batch-size 64 --decode-steps 64 --steps 1 --profile-dir HF_Prometheus/results/HF_modified_decode_async_6 --engine-delay-steps 0 --engine-queue-size 128 --cache-dtype none --collect-hidden --collect-attention`
- 带延迟=1（诊断）：
  - `MON_ENGINE_DEBUG=1 python HF_Prometheus/benchmark/tests/profile_decode.py --batch-size 64 --decode-steps 64 --steps 1 --profile-dir HF_Prometheus/results/HF_modified_decode_async_6 --engine-delay-steps 1 --engine-queue-size 128 --cache-dtype none --collect-hidden --collect-attention > HF_Prometheus/logs/debug.out`

可视化材料
- Notebook：`HF_Prometheus/notebooks/overhead_analysis.ipynb`（已修正为过滤 user_annotation/Trace）。
- 脚本版：`HF_Prometheus/notebooks/overhead_analysis.py`（支持保存 PNG）。

结论（给老板的话）
- “同步抓激活”带来的整理开销依旧存在，是对照基线。
- “异步抓激活”现已去掉 per-task 事件与大规模 D2D 拷贝；主路径慢主要是 Python 调度成本，我们已通过 StepWork/RingBuffer 把这块降到最低，并继续优化。
- 接下来通过批量提交、更轻量的队列、可选的后台压缩，主路径会越来越接近 `hf_modified`。
- 可视化脚本/Notebook 已就绪，随时可以展示差异；卡死问题也在新架构下得到进一步缓解。
