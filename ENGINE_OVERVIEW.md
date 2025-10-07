# 异步 Hook 监控引擎：演进、问题与方案（对外分享版）

本文面向不熟悉底层细节的读者，梳理我们从“同步抓取激活”到“异步流水化”的全过程：做了什么、为什么、遇到哪些问题、怎么解决，以及目前的效果与后续计划。

## 背景与目标

- 需求：在推理（no_grad）过程中抓取 GPT-2 中间激活（hidden states、attention 等），用于可视化/分析/验证。
- 目标：让“用户看到的推理时间”（主计算流）尽量接近不抓激活的基线，同时保证抓取的正确性；后台工作尽量隐藏、批量、低优先级、可延迟。

## 初始状态与基线

- 基准脚本：
  - `benchmark/tests/profile_inference.py`
  - `benchmark/tests/profile_decode.py`
- 基线（labels）：
  - `hf_modified`：不抓激活；仅一次前向，最贴近用户侧性能参考。
  - `hf_modified_hook`：同步抓激活；在一次前向里插入 hook，slice/拷贝/整理激活。
  - `hf_modified_hook_async`：异步抓激活；主流仅“登记任务”，重活尽量移到后台流。
- 第一版观测（典型 64×64，steps=1）：
  - `hf_modified` ≈ 2.76s；`hf_modified_hook` ≈ 5.02s；同步抓激活几乎“翻倍”。
  - `hf_modified_hook_async` 初稿仍接近同步，说明我们只是“并行”了工作，但主路径仍被 slice 和事件同步拖慢。

## 我们发现的问题（瓶颈画像）

- 大头之一：`aten::slice` 在同步路径上大量出现（百毫秒量级），主流被阻塞。
- `detach` 与 `record_stream` 的开销可见（累积百毫秒），每个 hook 都做一次并不必要。
- 细粒度事件：对每个条目都 `cudaEventRecord/WaitEvent`，累积几十毫秒。
- 即使移走 slice，主路径仍有固定开销：
  - 常规算子小幅抬头（`addmm/matmul/layer_norm` 等每类 +10~20ms），来自“边算边抓”引入的额外读写/带宽竞争；
  - 少量张量整理（`reshape/view/empty/clone/copy_`）不可完全避免。

## 核心方案：Monitoring Engine（异步监控引擎）

- 模块与类：
  - `monitoring/engine.py`：`MonitoringEngine`（后台管理/队列/流/批处理）
  - `monitoring/task.py`：`MonitoringTask`（任务）、`CacheFuture`（结果占位）
- Hook 行为：
  - 前向瞬间“登记任务”（张量引用 + 元数据），不做昂贵操作；
  - 任务入引擎队列，后台流按 step 聚合后处理；
  - `CacheFuture` 在需要时 `result()`，主路径无需等待。

## 分阶段落地（做了什么 + 为什么）

1) 压缩同步点到“步级”
- 做法：
  - 用 `cache_stream.wait_stream(producer_stream)` 在 step 结束时建立依赖（`end_step`），取代 per-task 事件；
  - 移除前后 `cuda.synchronize()`，`main_duration` 只覆盖前向计算；
- 效果：
  - `aten::slice` 主路径基本消失（移到后台）；
  - 事件同步从每条到每步，运行时开销显著降低；

2) 减少无谓的 CPU 操作
- 做法：
  - 仅当 `tensor.requires_grad=True` 时 `detach()`；推理默认不 detach；
  - 移除 per-task 的 `record_stream`；
- 效果：
  - `detach/record_stream` 两类热点大幅减少或消失。

3) 后台“更不打扰”主流
- 做法：
  - 后台流使用最低优先级（`torch.cuda.Stream(priority=max_pri)`）；
  - CLI: `--engine-queue-size` 限制待处理任务，避免瞬时抢带宽；
  - CLI: `--engine-delay-steps K`，K 步延迟（ring buffer），进一步让后台远离主前向（可选）。
- 效果：
  - 主路径算子（`addmm/matmul/layer_norm`）的抬头幅度降低；

4) 合并拷贝（coalesced copy，初版）
- 做法：
  - 同一 step 的任务批量处理；一次或分块分配大 buffer，将各条目拷贝到连续切片，future 指向切片 view；
  - 限制单块大小（默认 256MB，`MON_ENGINE_MAX_COALESCE_MB` 可调），避免大分配卡顿；
- 效果：
  - `empty/clone/copy_/reshape` 等数量下降，减少 launch/调度成本。

5) 只计主流时间（用户视角）
- 做法：
  - `profile_*` 中 `main_duration` 只计 `fn()`（前向）；`total_duration` 另行统计后台 resolve。
- 效果：
  - 输出指标更符合“用户端感知的推理时延”。

## 典型结果（64×64，steps=1）

- 之前：`hf_modified_hook` ≈ 5.02s；`hf_modified_hook_async` ≈ 4.99s；`hf_modified` ≈ 2.76s。
- 现在：`hf_modified_hook_async main_duration` ≈ 4.57s；`hf_modified_hook` ≈ 5.03s；`hf_modified` ≈ 2.76s。
- 变化点：
  - slice 与 per-task 事件已基本移除出主路径；
  - 剩余差距主要是抓激活“不可避免的”额外读写/带宽竞争和少量整理开销。

## 我们遇到的 Bug 与修复

- 症状：delay_steps=1 时卡住（GPU 0% 利用、显存高），Ctrl-C 打断；日志只见大量 submit，没有 end_step 或 enqueue。
- 根因：
  - 某些路径没有执行 `end_step`，任务只“封存”没入队；
  - `resolve_all()` 只 flush sealed steps，未 flush leftover steps；
  - worker 批处理时 `task_done()` 计数不匹配，`queue.join()` 永久等待。
- 修复：
  - 在异步封装中用 `try/finally` 保证 `end_step()` 总被调用；
  - `resolve_all()` 同时 flush sealed 和 leftover 的 step（都有 debug 日志）；
  - worker 端批处理后统一 `task_done()`，并对哨兵项 `task_done()` 后退出；
  - 合并拷贝按块分配，避免一次性大分配引发 allocator 等待；
  - 提供调试：`MON_ENGINE_DEBUG=1` 打印 submit/start_step/end_step/enqueue/worker/chunk 等。

## 关于 `cache_dtype`

- 在当前 decode 设置（小 batch、短序列）下，`fp16/bf16` 转换开销往往大于带宽收益，反而更慢；
- 结论：默认 `--cache-dtype none`；仅在“大体积搬运”场景下再尝试降精。

## 如何使用（建议）

- 纯主流时延：`--cache-dtype none`，`--engine-queue-size 128~256`，`--engine-delay-steps 0~1`。
- 观察异步行为：`MON_ENGINE_DEBUG=1`；必要时 `MON_ENGINE_MAX_COALESCE_MB=128` 控制单块大小。
- 推荐按两步跑：
  1) `delay_steps=0` 验证主线；
  2) 开 `delay_steps=1`，看 main_duration 是否再收敛，若卡住查看 debug 日志的 end_step/enqueue/worker/chunk。

## 仍然存在的差距（原因与预期）

- 与 `hf_modified` 的差距来自“抓激活的固定成本”：
  - 前向中每层在计算瞬间被 tap 出来，必然有额外的内存读写与少量整理；
  - 即使 slice 与 per-task 同步消除，`addmm/matmul/layer_norm` 等仍会小幅抬头，这是抓激活的代价，而不是异步本身的问题。

## 后续路线（更多优化）

- 更激进的合并拷贝：自定义 kernel 或更少的高效大拷贝，进一步降低 copy_/launch 开销；
- 更严的节流策略：按 in-flight/大小/优先级调度，保障主流算子稳定；
- 更长的 K 步延迟（ring buffer），彻底把后台工作与主流错峰；
- CUDA Graphs 捕获主前向（保持 hook 排队在图外），降低主流调度；
- 多 GPU：后台流转移到第二块 GPU（NVLink），完全隔离 compute 与 copy。

---

## 附录：关键文件与开关

- 引擎代码：
  - `monitoring/engine.py`（核心逻辑：队列、后台流、步级批处理、合并拷贝）
  - `monitoring/task.py`（任务与 Future）
- Hook 适配：
  - `transformers/src/transformers/models/gpt2_p/hook_points.py`（只在需要时 `detach()`）
- 基准脚本：
  - `benchmark/tests/profile_inference.py`
  - `benchmark/tests/profile_decode.py`
- 有用的 CLI：
  - `--cache-dtype {none,fp32,fp16,bf16}`（建议 none）
  - `--engine-queue-size N`（建议 64~256）
  - `--engine-delay-steps K`（建议 0~1 起步）
- 调试与控制：
  - `MON_ENGINE_DEBUG=1` 打开详细日志
  - `MON_ENGINE_MAX_COALESCE_MB=128` 控制单块 coalesce 上限

