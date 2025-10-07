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
  - K 步延迟：--engine-delay-steps（ring‑buffer 思路），允许后台缓几步再处理，进一步让主流“像没开 hook 一样”。
  - 合并拷贝（分块）：每步尽量预分配大 buffer，将多个小拷贝合并，减少 kernel/调度开销（可被 MON_ENGINE_MAX_COALESCE_MB 控制分块大小）。
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
   - 措施：将切片/整理延后到后台流，并引入“合并拷贝”减少许多小 copy/launch 调度。

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

主要开销对比（基于 1..4 与 6 轮 trace 聚合）
- 同步版相对 `hf_modified`：
  - Top‑1 持续为 `aten::slice`；其次为 `aten::as_strided` 与少量 `addmm/matmul/layer_norm`。
- 异步版早期（1..4 轮）：
  - 事件相关（`cudaEventRecordWithFlags`/`cudaStreamIsCapturing`/`cudaStreamWaitEvent`）明显；`record_stream` 与少量 `detach` 也占据开销。
- 异步版近期（第 6 轮）：
  - 事件开销消失，取而代之的是 D2D 拷贝（`cudaMemcpyAsync`/`Memcpy DtoD`）成为主要来源；其次是 `addmm/matmul/layer_norm`（受带宽争用影响）。

这意味着：我们把“同步/事件/细粒度开销”挪到了后台，但后台拷贝的显存带宽竞争开始成为主路径的主要干扰。因此下一步应侧重减小 D2D 拷贝体量和并发对主流的影响。

下一步计划（从易到难）
1) 进一步“更不打扰”主流：
   - 限制后台同时在途拷贝数量；
   - 维持低优先级流；必要时按 token 大小优先，减少长拷贝对主路径的影响。
2) 合并拷贝与大 buffer：
   - 一步一个大 buffer + 偏移表，用更少的大块拷贝替代大量小拷贝；
   - 动态分块大小（避免 allocator 压力）。
3) K 步延迟的 ring buffer：
   - 将后台任务批处理到 K 步后再统一落地，最大化隐藏对主流的影响（以显存换干扰度）。
4) 降精/压缩（再次评估）：
   - 当合并拷贝成熟后，重测 fp16/bf16 的净收益；必要时引入轻量量化以进一步减小 D2D 体量。
5) CUDA Graphs（可选）：
   - 捕获主前向图，进一步压低 launch/调度开销；hook 的异步逻辑保持图外。

如何复现
- 无延迟异步（对比 main vs total）：
  - `python HF_Prometheus/benchmark/tests/profile_decode.py --batch-size 64 --decode-steps 64 --steps 1 --profile-dir HF_Prometheus/results/HF_modified_decode_async_6 --engine-delay-steps 0 --engine-queue-size 128 --cache-dtype none --collect-hidden --collect-attention`
- 带延迟=1（诊断）：
  - `MON_ENGINE_DEBUG=1 MON_ENGINE_MAX_COALESCE_MB=128 python HF_Prometheus/benchmark/tests/profile_decode.py --batch-size 64 --decode-steps 64 --steps 1 --profile-dir HF_Prometheus/results/HF_modified_decode_async_6 --engine-delay-steps 1 --engine-queue-size 128 --cache-dtype none --collect-hidden --collect-attention > HF_Prometheus/logs/debug.out`

可视化材料
- Notebook：`HF_Prometheus/notebooks/overhead_analysis.ipynb`（已修正为过滤 user_annotation/Trace）。
- 脚本版：`HF_Prometheus/notebooks/overhead_analysis.py`（支持保存 PNG）。

结论（给老板的话）
- “同步抓激活”让主路径多了大量 `slice/clone` 等整理开销，直接拖慢 1 倍左右；
- “异步抓激活”成功把同步/事件开销移出主路径，但目前主要瓶颈变成了后台 D2D 拷贝对主路径带宽/SM 的干扰；
- 接下来通过“步级合并拷贝 + 限流 + K 步延迟 +（成熟后）降精/压缩”，主路径将更加接近“不开 hook”的 `hf_modified`，用户感知的推理时间会显著降低；
- 现有方案已具备实验对比基础和可视化支撑，卡死问题定位中，已提供调试日志与守护器思路，预计可在下一轮修正。

