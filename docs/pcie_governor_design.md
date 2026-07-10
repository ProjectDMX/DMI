# PCIe Governor — 设计文档

DMI 观测流量的 serving-first 控制器。**核心设定：链路上有两级流量——
serving-critical（KV offload/reload、token 回传、NCCL 等）是高层（Tier-0），
DMI 观测流量是低层（Tier-1）。governor 做的事只有一件：在高层流量出现时，
让 DMI 的 D2H 暂停/避让，用 ring 缓冲吸收这段时间的捕获。** 不做 per-hook
优先级、不做细粒度保真度策略——那会要求用户给 hook 排优先级，伤易用性。

一个 CPU 侧组件；感知 = 前馈 hint + 反馈探针；执行 = 只动 drain 节奏与缓冲，
不动捕获路径，graph 拓扑永不变。

## 0. 非目标（先划清边界）

- **不做 hook 级 tier/优先级**：用户不需要标注哪个 hook 重要；
- **不做 governor 驱动的保真度阶梯**：持续过载时数据取舍交给**既有**用户策略
  （completeness = 接受 flush 停顿；best-effort = 既有的 drop-recent /
  keep-by-pattern 按请求丢），governor 不发明新的丢数据机制；
- 不声称"调度链路"：PCIe 没有 user-space 优先级，我们只能自我让路（yield）。

## 1. 契约

- **Tier-0 优先**：已宣告的 serving-critical D2H 传输期间，DMI 不发起 D2H
  （ring 能吸收的前提下）；
- **ε 有界**：DMI 造成的 serving 劣化 ≤ ε（默认 5% step 延迟）——包括避让
  期间也不许因 ring 满而堵 serving；
- **可审计**：每次避让/恢复记录原因（hint 或探针读数）；避让期间观测延迟
  （capture → consumable）会升高，作为指标随数据导出。

**避让的升级序列（全部是传输层动作，不碰捕获）**：

1. **defer**：推迟 drain flush，捕获照常进 ring（ring = 减震器）；
2. **pacing**：恢复时只发小批 D2H，避免一次大 flush 抢回链路；
3. **hard watermark override**：ring 占用过水位线时自动取消 defer，照常 flush
   capped batch——绝不为礼貌把自己憋到堵 serving；
4. **持续过载**：超出缓冲能力时按用户既有策略处置（completeness 接受停顿 /
   best-effort 按请求丢），governor 只负责触发，不新增机制。

## 2. 对手流量的本性（设计的前提，先于一切机制）

| 流量 | 形态 | 时长量级 | 对 step 关键路径 | 能否预知 |
|---|---|---|---|---|
| KV load（H2D，prefix 命中） | burst | 数 ms~数十 ms（prompt 长度 × ~百 KB/token） | **同步**：prefill 等它 | 能（引擎发起） |
| KV save/offload（D2H） | burst | 同上；抢占/换出时整请求 KV | 多为**异步**（落后才反压） | 能（引擎发起） |
| 内存压力下流式换入换出 | **持续** | 秒级以上（Rev-A 引 Jiang et al. 的场景） | 持续挤压 | 部分 |
| TP/PP over PCIe（无 NVLink） | **持续嗡嗡** | 每层 2 次 all-reduce，整个 forward 期间 | 是，且共享 GPU 上行口 | 无意义（永远在） |
| logits/token 回传、输入 metadata | 微小 | µs 级 | — | — |
| 其他进程/租户 | burst 或持续 | 未知 | — | **不能** |

**核心事实：burst 域和持续域需要完全不同的机制。**

- **burst（ms 级）**：等你测到它，它已经结束——任何反馈检测（探针 EWMA
  100ms 窗、NVML 20ms 采样）都存在时间混叠，物理上测不到。但 burst 恰好
  **引擎自己提前知道**（hint）；不能预知的（外部进程）只能靠**限界单次伤害**
  （flush 分成可配置 batch，例如 16-64MB，碰撞窗口有界）。
- **持续（秒级）**：hint 无意义（没有"事件"可宣告），但反馈探针正确——压力
  稳定存在，EWMA 收敛，让路决策有效。这正是 Rev-A 问的场景。

**因此感知分工：hint 管 burst；探针管持续域 + 事后审计。探针不承担 burst
的实时检测——那是它做不到的事。**

### 2.0 先想清楚：什么测得到，什么测不到

**测不到**：PCIe 无 per-stream 计数器（看不到"谁"在用）；copy engine 排队不可见；
别人的流量只能间接推断。

**测得到**：①自己每次 D2H 的 cudaEvent 耗时 ②host 侧 enqueue→sync 完成耗时
（可给出近似排队/串行化上界，不是精确 copy-engine queue delay）
③NVML 链路吞吐（TX/RX 分方向，~20ms 粒度）④ring/staging 占用及斜率
⑤`prepare_step` 停顿时长（对 serving 的**真实**干扰）。

### 2.1 S0：引擎前馈 hint（第一优先信号）

serving 打 PCIe 的流量可枚举：KV offload/reload（最大头）、logits/token 回传（小）、
每步输入 metadata H2D（小）、NCCL P2P（无 NVLink 时）、权重/LoRA 加载（非常态）。
**真正要让的是 KV save 的 D2H——与 DMI drain 同向，直接抢。** 引擎在发起前
自己就知道，不必等测出来：

```
PCIeHint { direction: D2H|H2D, est_bytes: N,
           source: str, valid_until: ts|next_step }
```

**hint 来源（现有代码可挂）**：
- `kv_connector.py:62` `pre_forward()` → `start_load_kv()`：KV load，H2D；
- `kv_connector.py:79` `post_forward()` → `wait_for_save()`：KV save，**D2H，最该让**；
- `lmcache_mp_connector.py:244`：STORE/RETRIEVE 方向 + block ids → 可估 bytes；
- vLLM CPU-offload worker 的 `Transfer` 记录（真实 bytes/耗时）→ 事后校准可信度。

**使用规则**：v1 只让 DMI D2H 对同向 D2H hint 避让；H2D hint 只做审计，不改变
drain。KV save/store 的 D2H 是标准 defer 对象。worker 主路径估不出准确 bytes
时，trusted D2H source 允许 `est_bytes=0` 且仍生效；`hint_min_bytes` 只在
`est_bytes > 0` 时用于跳过明确的小传输。
hint 是前馈不是真值——它不知道 copy 何时真正进 copy engine、走不走 PCIe、
有没有别的进程。**hint 负责提前让，探针负责持续域 + 事后纠错。** 生命周期：
connector 在真实 D2H operation 开始/结束时发 begin/end；`max_defer_us` 是 end
丢失时的 stale-hint watchdog，不是正常保护窗口。正常恢复由 end 立即触发；避让
期间 hard watermark 和 force flush 仍可排水。v1 不让 governor 消费
`criticality` / `rank` / `request_ids`：关键路径性由 `(source, direction)` 集中
推导；connector 在本地把多个请求聚合为 source active/inactive。

**为什么前馈优先**：纯探测是"发现慢了再让"，检测要 ≥1 个 flush + EWMA 窗
（几十到上百 ms），KV 尖峰本身就这个量级——测出来时冲突已经发生。hint 把
避让提前到冲突之前，且自带归因和量。

#### 2.1.1 LMCache 生命周期问题与修复

最初实现统一在 worker 的 `wait_for_save()` 前后发 hint，并把默认
`max_defer_us=5000` 当成实际保护时长。实测 LMCache connector 的 worker-side
hint bracket 约 100–130ms；5ms 到期后 DMI 会在 KV 尚未完成时恢复，绝大部分
critical window 没被保护。这个 bracket 包含 metadata/分配/排队等工作，不等于
纯 PCIe D2H 时间，但其返回点保证 serving 不再等待该 store。

更重要的是，`wait_for_save()` 不是所有模式的统一开始/结束边界：

- non-layerwise：真实 store 在 `wait_for_save()` 内同步执行，函数返回是 D2H
  完成边界；
- layerwise：D2H 从第一个实际 `save_kv_layer()` 开始，`wait_for_save()` 只完成
  尾部，旧 hint 开始得太晚；
- LMCacheMP：`wait_for_save()` 只提交异步 STORE，返回时 D2H 可能尚未完成；完成
  边界是 `CUDAMessagingFuture.query()` 观察到 CUDA IPC event 完成。

修复采用 connector-managed `D2HHintLease`：source 从 inactive→active 发 begin，
长操作低频 renew，最后一个未完成 store 结束时发 end。worker 只给不管理自身
lifecycle 的 connector 保留 coarse `wait_for_save()` fallback。默认
`max_defer_us=1_000_000` 只作为 stale watchdog；正常 operation 由 end 提前恢复。

### 2.2 S1：自有传输当探针（**只负责持续域 + 事后审计，不做 burst 实时检测**）

drain 本来就按批发 `cudaMemcpyAsync` D2H（`drain_thread.cpp: enqueue_d2h`）。
按 **flush batch** 套一对 cudaEvent（不是按 ring wrap chunk；一个 batch 内
可能因 wrap 被拆成 2-3 段 memcpy）：

- `achieved_bw = batch_bytes / (t_end − t_start)`
- `approx_queue_delay = host_elapsed − cuda_event_elapsed`

C++ 只导出单调计数器；空载标定、EWMA 和
`P = 1 − achieved_bw/bw_baseline` 全在 Python governor 的 step 差分里算，
便于 §2.5 反复调参。

**角色一（持续域估计器）**：流式换入换出、TP/PP-over-PCIe、多租户挤压——
压力持续存在时 EWMA 收敛且有意义，驱动反馈避让。**明确不承担 burst 检测**：
burst 2–20ms vs 探针机会主义采样 + 100ms 窗，时间混叠，命中率低且平均后
信号消失；等收敛冲突早已结束。

反馈还有一个根本盲区：低优先级 DMI 如果已经排在前面，自己仍可维持约
200Gbit/s，而后到的 LMCache 才是排队受害者；DMI 自探针看到的 pressure 接近
零，无法识别这种 priority inversion。因此 LMCache burst 必须以 lifecycle hint
为主，self-probe 只覆盖 DMI 自己也变慢的持续竞争和事后审计。

**角色二（事后审计）**：每批实测带宽天然形成**碰撞日志**（哪批慢了 = 撞上
谁了），用于：①统计 hint 覆盖率与 lead-time（撞上的块有没有对应 hint）；
②校准各 source 可信度；③论文里的冲突字节量指标直接从这来。审计是回顾性的，
不进实时控制回路。

守卫：batch < 4 MB 不作样本；持续域避让后的恢复判断缺数据时发 4 MB 只读合成
探针；`approx_queue_delay` 含自串行化，需 self-only 校准（2.5）。

### 2.3 S2：NVML 交叉验证（辅助，20–50 ms 轮询）

`ext_bytes ≈ nvml_total − 自己的账`。用途：区分"堵的是我"（调节奏即可）与
"堵的是别人"（避让）。**方向性**：PCIe 全双工，KV 写回是 D2H（同向真竞争），
权重/KV 载入是 H2D（反向，影响弱）；TX/RX 分开读，只计同向。

### 2.4 S3：直接干扰事件（契约地面真值）

`prepare_step` 返回 `STEP_RING_FLUSHED` = 我们真实停顿了 serving；在
`ring_engine_py.cu:prepare_step` 测停顿时长，直接扣 ε 预算。ring 占用斜率
是先行指标。

### 2.5 感知验证实验（做控制器之前先做）

1. **估计器精度**：注入已知强度的 D2H/H2D 对抗流量（方波/斜坡/随机），
   P 估计 vs 真值，分方向；
2. **SNR 标定**：扫 flush 尺寸 × 对抗强度，定最小可靠探针尺寸；
3. **self-only 校准**：无外部流量时 `approx_queue_delay` 分布 → 自串行化底噪；
4. **基线漂移**：长跑观测 bw_baseline 漂移，定空闲重标定周期；
5. **hint 质量**：hint 到达 vs Transfer 实际发生的 lead-time 分布、
   est_bytes 误差——决定 defer 窗口大小。

### 2.6 已知局限（论文老实写）

- 只能自我让路；措辞 "yield"，不是 "schedule the link"；
- 单 rank 探针只见自己链路段；共享 switch 的上行竞争留 v2
  （`nvmlDeviceGetTopologyCommonAncestor`）；
- NVML 粒度/权限问题：S2 是交叉验证不是依赖；
- **PCIe-TP 机器（无 NVLink）**：forward 期间链路被 all-reduce 持续占用，
  永不空闲——"空窗回填"退化，DMI 在这类机器上天然更贵。定位：DMI 主目标是
  NVLink-TP / 单卡 serving；PCIe-TP 场景由持续域反馈路径处理（让路 =
  接受更低 drain 吞吐），不承诺 burst 级精细避让。

## 3. 控制律

**总原则：hint 前馈，探针反馈。** 高层流量宣告意图，低层 DMI 主动让路，
被动测量纠正错误。

**前馈路径（提前让）**：

```
if hint.direction == D2H and (hint.source in coarse_trusted_sources or
                              hint.est_bytes > threshold):
    if ring_occupancy < hard_watermark:  defer_drain(until=hint.valid_until)
    else:                                 allow_capped_flush()  # 保自己不堵 serving
else:                                     normal_drain()
```

- **可抢占分块 flush**：normal flush 按可配置小 batch（先扫 16/32/64MB）
  发送；hint 到达时已提交 batch 不取消，下一轮 `should_flush()` 会看到 defer，
  因而冲突窗口被 batch size 限界；
- `coarse_trusted_sources` v1 至少包含 `kv_store`；这些 source 允许
  `est_bytes=0` 仍触发 defer，避免 worker 层估不出 bytes 时主路径 no-op；
- **空窗回填（增强）**：decode 节奏稳定，hint 流 + step 边界给出短期链路日历，
  drain 把 flush 排进已知空窗（KV save 完成后、下步输入 H2D 前）——从 backoff
  升级为 self-scheduling。

**反馈路径（只管持续域：流式换入换出 / TP-PP-over-PCIe / 多租户）**：
avoid/normal 两态 + 滞回——P > P_hi（0.35）连续 2 窗 → 进入避让态（写入
有界 feedback deadline，例如 `now + 2 * feedback_window`，后续干净前滚动续期）；
连续 20 个干净窗（P < 0.15 且零停顿）→ 恢复。
决策窗 100ms；动作只在 flush 边界与 step 边界生效。**不试图对 burst 做反馈
——时间混叠，见 §2；burst 的伤害由分块抢占限界，未 hint 的 burst 碰撞计入
ε 预算并进事后审计。**

**两路仲裁**：都说堵 → 取更保守；hint 说堵探针说不堵 → 扣该 source 可信度；
探针说堵无 hint → 反馈路径接管。

**ring 水位（硬约束，压倒一切）**：占用 <50% 放心让；>80% 时 defer 失效，
drain 线程照常 flush capped batch——绝不拿自己的停顿换礼貌。中间档继续
defer/pacing；v1 不靠 pageable spill 缓解 PCIe 冲突，因为真正吸收 burst 的是
GPU payload ring 空间。
持续过载超出缓冲 → 触发用户既有策略（completeness/best-effort），见 §1 第 4 级。

**量级验算（ring 兜得住 burst 吗）**：hidden-state 预设 ≈ 256KB/token/step
（32 层 × 8KB），batch 64 → ~16MB/step，TPOT 10ms → 捕获率 ~1.6GB/s；
2GB ring 可吸收 ~1.2s 的完全避让。**KV burst（ms~几十 ms）随便兜；只有持续域
才会真正填满 ring**——与"反馈只管持续域"的分工自洽。让路的代价是观测延迟
临时上升 ≈ burst 时长（对 ≤1-token freshness claim：P50 不受影响，burst 期
P99 会破，论文分开报）。

## 4. 执行器（全部传输层，全部已存在或小改）

| 动作 | 机制 | 代码 |
|---|---|---|
| defer / 恢复 drain | 运行时 `defer_until_ns` 抑制普通阈值 | `drain_thread.cpp: should_flush`（3 个 atomic 控制字段） |
| 可抢占分块 flush | normal loop 按可配置 batch cap 选批，下一轮重新评估 defer | `drain_thread.cpp: should_flush` / batch 选择逻辑 |
| 扩大吸收余量 | 启动时调大 payload ring / pinned staging；pageable spill 只作为 v2 host 队列保护 | `RingConfig.payload_ring_bytes` / `pinned_staging_bytes` |
| hard watermark override | 超水位时忽略 defer、继续 capped flush | `drain_thread.cpp: should_flush` |
| 过载移交 | 触发既有 completeness / best-effort 策略 | policy manager 现有路径，不新增 |

**不需要的**（相比上一版删除）：per-hook strip 向量、hook 子集降档、采样档位
——捕获路径完全不被 governor 触碰。

## 5. 集成

```
monitoring/governor.py                  # 状态机 + hint/feedback 策略，~150 行
monitoring/csrc/ring/drain_thread.cpp   # + batch cudaEvent 计数器、3 atomic 控制、分块检查点
monitoring/csrc/ring/ring_engine_py.cu  # + prepare_step 停顿计时
monitoring/csrc/bindings.cpp           # 暴露 link_stats()/set_drain_control()/set_defer_until_ns()
integration/vllm_adapter.py             # kv_connector hint 挂钩 → governor.on_hint()
```

多 GPU：每 rank 一个 governor（ring/drain 本就 rank-local），v1 无跨 rank 协调。

## 6. 评估计划

1. **对抗共跑**：vLLM serving + KV offload（或合成 hog）+ DMI 全量捕获。
   governor 关：P99 劣化量化；开：P99 守 ε + 避让时间线 vs 注入压力。
2. **估计器精度**：注入已知带宽 vs S1/S2 估计（2.5 正式化）。
3. **前馈 vs 反馈 ablation（核心图）**：hint-only / probe-only / both 三条线；
   指标 = 冲突字节量（DMI D2H 与 KV D2H 重叠量）+ serving P99 + 观测延迟。
   预期：hint-only 快但有漏、probe-only 全覆盖但慢半拍、both 双优。
4. **让路的代价要老实报**：避让期间 capture→consumable 延迟升高多少
   （P50/P99），ring 尺寸 vs 可吸收突发时长的换算表。
5. **抗震荡**：方波负载验证滞回，测恢复延迟。

## 7. 排期（对齐 8/31 决策门；相比 hook-tier 版少 ~1.5 周）

1. 第 1–2 周：drain 探针统计 + 停顿计时 + governor 骨架（defer/恢复）；
   **先跑 2.5 感知验证**；合成 hog 端到端 demo。
2. 第 3 周：kv_connector hint 挂钩 + 分块抢占；空窗回填只做 v2 预留。
3. 第 4–5 周：对抗评估矩阵 + ablation + 调参；冻结。

## 8. 真实实现计划 / 源码改动计划

本节按当前代码形态落地。约束不变：governor **只调 DMI drain 的 D2H**
路径，不改 producer/hook 捕获路径，不改 CUDA graph 拓扑；serving/KV/NCCL
是 Tier-0，DMI drain 是 Tier-1。PCIe 不能硬限速，所有动作都必须表达成
admission / pause / defer / pacing / hard watermark override。

### 8.1 当前源码事实

- `monitoring/csrc/ring/drain_thread.cpp/.h`：
  - `DrainThread::should_flush()` 只读构造期 `cfg_.drain_flush`，运行时不可调；
  - `do_full_flush()` 会尽量 flush 所有 pending entry；
  - `enqueue_d2h()` 已按 GPU ring wrap 和 staging ring wrap 分块，但这些是
    memcpy wrap chunk，不是 governor batch；当前 batch 选择不受可配置上限控制；
  - `force_flush_and_wait()` 是现有强制 flush 入口，Python binding 已释放 GIL。
- `monitoring/csrc/ring/ring_engine_py.cu/.h`：
  - `prepare_step()` 是 serving 前的容量闸门；只有 ring 放不下时才同步主 stream
    并 `force_flush_and_wait()`，返回 `STEP_RING_FLUSHED` / `STEP_OVERSIZED`；
  - 这里最适合记录“DMI 真实堵到 serving”的停顿时间。
- `monitoring/csrc/bindings.cpp`：
  - 已暴露 `RingConfig`、`RingEngine.prepare_step()`、`flush_and_wait()`、
    `available_capacity()` 等，但还没有 governor 相关 API。
- `monitoring/adaptor_base.py` 与 `monitoring/ring_transport.py`：
  - `BackendAdaptor.before_forward()` 统一调用 `prepare_step(total_bytes, n_hooks)`；
  - `RingTransport` 持有 `_ring_engine`，是 Python governor 挂到传输层的自然位置。
- `integration/vllm_adapter.py`：
  - 创建 `MonitoringEngine`、`RingConfig` 和 `VLLMAdaptor`；
  - 已有 `additional_config` / env 配置入口，可加 `dmx_pcie_governor_*` 开关。
- vLLM / LMCache connector：
  - `integration/vllm/vllm/v1/worker/gpu/kv_connector.py` 的
    `ActiveKVConnector.pre_forward()` 调 `start_load_kv()`，是 KV load H2D hint 点；
    `post_forward()` 调 `wait_for_save()`，是 KV save/store D2H hint 点；
  - `integration/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
    包装 `start_load_kv()`、`save_kv_layer()`、`wait_for_save()`；
  - `lmcache_mp_connector.py` 的 `LMCacheMPRequestMetadata.direction` 已区分
    `STORE` / `RETRIEVE`，`LoadStoreOp` 带 token/block 范围，可估 bytes；
  - `offloading_connector.py` 的 `OffloadingConnectorWorker.start_kv_transfers()`、
    `prepare_store_kv()` 知道 `TransferSpec`，`get_finished()` 已记录真实
    `transfer_size` / `transfer_time` / `transfer_type`，可做事后校准。

### 8.2 新增/修改的 API 和数据结构

**C++ 传输层数据结构**

v1 不新增 C++ struct，不在 C++ 放策略状态。`DrainThread` 只增加 3 个
atomic 控制字段：

```cpp
std::atomic<uint64_t> defer_until_ns_{0};        // 0 = normal
std::atomic<uint64_t> max_d2h_chunk_bytes_{0};   // 0 = unlimited
std::atomic<uint64_t> hard_watermark_bytes_{0};  // 0 = disabled
```

含义：

- `defer_until_ns > now`：普通 `entry_threshold` / `byte_threshold` /
  `timeout_us` 不触发 flush；
- `hard_watermark_bytes > 0 && payload_reserved_bytes >= hard_watermark`：
  忽略 defer，照常 flush，避免 DMI 把 serving 堵在 `prepare_step()`；
- `max_d2h_chunk_bytes > 0`：每次 flush batch 的 D2H 字节上限，用于把单次
  碰撞窗口限在 16/32/64MB 量级。

`DrainFlushConfig.byte_threshold` / `timeout_us` 仍是正常模式配置，不复制到
runtime config。`force_flush_and_wait()` 仍是强制 flush all 的动作，不镜像成
runtime mode。

`DrainThread` / `RingEnginePy` 暴露三个 C++/pybind API：

- `set_drain_control(defer_until_ns, max_d2h_chunk_bytes, hard_watermark_bytes)`：
  初始化/关闭用；
- `set_defer_until_ns(defer_until_ns)`：热路径用，单 u64 store；
- `link_stats() -> dict`

`link_stats()` 只返回单调计数器和当前容量快照，不返回 EWMA / pressure /
baseline：

- `d2h_bytes`, `d2h_batches`
- `d2h_probe_bytes`, `d2h_probe_event_us`, `d2h_probe_host_us`
  （仅统计 >=4MB 的 batch 样本）
- `stall_us_total`, `stall_count`
- `pending_bytes`, `pending_entries`
- `payload_reserved_bytes = cpu_payload_head_ - cpu_payload_tail_committed_`
  （与 `prepare_step()` 容量闸门同一口径）
- `staging_used_bytes`

所有计数器 atomic 累加，无 `reset_link_stats()`；Python governor 用相邻
snapshot 做差分。`pending_*` / occupancy 可在现有管理锁下做短快照。

**Python governor 数据结构**

新增 `monitoring/governor.py`：

- `PCIeHint`
  - `direction: Literal["D2H", "H2D"]`
  - `est_bytes: int`
  - `source: str`
  - `valid_until_ns: int`
- `PCIeGovernorConfig`
  - `enabled`（总开关，默认 false）
  - `hint_min_bytes`
  - `max_defer_us`（默认 1,000,000us；stale-hint watchdog，不是正常 operation
    时长）
  - `hard_watermark_ratio`
  - `max_d2h_chunk_bytes`
  - `baseline_gbps`（可选；实验固定基线用）
  - `p_hi`, `p_lo`, `feedback_window_ms`
  - `clean_windows_to_resume`
- `PCIeGovernor`
  - `on_hint(hint: PCIeHint) -> None`
  - `on_step() -> None`
  - `snapshot() -> dict`

`on_hint()` 是前馈热路径：收到有效 hint 后立即更新 hint deadline，并调用
`set_defer_until_ns()`，目标是 hint 到 C++ atomic 的延迟为 Python 函数调用量级。
`on_step()` 在 `prepare_step()` 后调用一次：读取 `link_stats()` 差分，更新
EWMA / baseline / hysteresis，只负责反馈状态机和过期重算。governor 内部维护
`hint_deadline_ns` 与 `feedback_deadline_ns`，任何一方变化都把
`max(hint_deadline_ns, feedback_deadline_ns)` 写入 `defer_until_ns`；结束 hint
（`valid_until_ns=now`）只清 hint deadline，不会误杀反馈避让。

线程契约：vendored vLLM v1 的 `pre_forward()` / forward / `post_forward()` /
`before_forward()` 在同一个 worker 线程串行调用；LMCacheMP 的后台 D2H 由已有
future/CUDA event 表示，worker 在 `get_finished()` 中轮询并更新 lease。v1
governor 内部不加锁，唯一跨线程接口是 C++ 的 3 个 atomic。`snapshot()` 只用于
审计导出。
不单独做 `poll()`；decode step 已经比 100ms 决策窗密，step 边界就是天然
poll 点。

baseline 初始化：若配置 `baseline_gbps`，直接使用固定基线；否则用 probe
`event_bw` 的 running max（慢衰减）作为初始 v1 基线。§2.5 实验 4 后再替换成
更精细的空闲重标定。

时钟契约：`defer_until_ns` 使用 Linux `CLOCK_MONOTONIC` 纳秒。Python 侧用
`time.monotonic_ns()`；C++ 侧用 `std::chrono::steady_clock` 并在 binding smoke
test 断言两边 now 差值小于容差。

同文件内提供模块级 `set_current(governor)` / `get_current()`，给 vendored vLLM
connector 做 optional import。vLLM worker 一进程一卡，v1 不需要单独
`pcie_governor_registry.py` 或 `(device, rank)` 弱引用表。

**总开关契约（实验对照用）**

`dmx_pcie_governor_enabled` / `DMX_PCIE_GOVERNOR_ENABLED` 是唯一总开关，
默认 false。关闭时必须满足：

- 不创建 `PCIeGovernor` 实例；
- `RingTransport.governor is None`；
- `monitoring.governor.set_current(None)`，connector 侧 `get_current()` 返回
  `None`，hint 代码只做 no-op；
- `RingEngine.set_drain_control(0, 0, 0)`，C++ drain 回到现状路径；
- `BackendAdaptor.before_forward()` 不调用 governor；
- `link_stats()` 可以存在但不被 governor 轮询，不影响对照 run。

配置优先级：`additional_config["dmx_pcie_governor_enabled"]` >
`DMX_PCIE_GOVERNOR_ENABLED` > 默认 false。v1 只要求启动时开关；运行时热切换
不是实验必需项，避免引入状态清理复杂度。

### 8.3 C++ drain 执行器改动

1. **运行时 drain control**
   - `DrainThread::should_flush()` 读取 3 个 atomic。
   - watermark 度量固定为
     `payload_reserved_bytes = cpu_payload_head_ - cpu_payload_tail_committed_`，
     与 `prepare_step()` 容量闸门同一口径；不要用 `pending_bytes_` 做水位，
     因为它只统计已 scan 的 entry，会漏掉 reserved-未写入部分。
   - `defer_until_ns > now` 时，普通 `entry_threshold` / `byte_threshold` /
     `timeout_us` 不触发 flush。
   - 检查顺序固定：现有 task/payload 满硬检查 → `payload_reserved_bytes`
     超过 `hard_watermark_bytes` → deferred return false → 现有阈值。
   - `defer_until_ns = 0`、`max_d2h_chunk_bytes = 0`、`hard_watermark_bytes = 0`
     时必须与现状行为等价。

2. **flush batch helper**
   - 把 `do_full_flush()` 内部拆成 helper：
     - `select_flush_batch(max_bytes, force_all) -> (flush_count, flush_bytes)`
     - `flush_batch(flush_count, flush_bytes)`
   - `force_flush_and_wait()` 继续 flush all，保持现有语义；
   - force path 对 governor 完全免疫：不检查 `defer_until_ns`，不套
     `max_d2h_chunk_bytes` cap，不因 hard watermark 改变行为。`prepare_step`
     慢路径调用 force flush 时 serving 已经同步等待 ring 腾空间，任何 defer
     都会直接违反 ε 契约；stop/final flush 同理。
   - 不新增 `minimal_flush_and_wait()`。hard watermark 下的“最小必要 flush”
     由 `should_flush()` 内生触发：defer 被覆盖，drain 线程按 capped batch
     正常推进，低于水位后自然回到 defer。

3. **normal-loop batch cap**
   - normal loop 每轮本来只做一次 batch 选择 + flush，然后回到
     `should_flush()` 重新评估；因此 batch 边界抢占不需要新的内层检查。
   - 在 normal-loop batch 选择处把上限设为
     `min(staging_.capacity(), max_d2h_chunk_bytes)`（`max_d2h_chunk_bytes=0`
     表示只受 staging capacity 限制），且只在完整 `TaskEntry` 边界切；
     不把已选 entry 拷一半。
   - `enqueue_d2h()` 只负责处理 ring/staging wrap；wrap chunk 不是 governor
     batch，不在这里检查 runtime mode。
   - 注意只在 batch 边界抢占；已提交的 `cudaMemcpyAsync` 不取消。

4. **passive D2H feedback**
   - 每个 flush batch 记录一对 cudaEvent 和两个 host timestamp。不要按
     ring/staging wrap chunk 建 event；wrap 只是 batch 内部的 memcpy 拆分。
   - `t0_host` 固定在 staging free wait 完成之后、`enqueue_d2h()` 之前；
     `t1_host` 固定在 `sync_stream()` 返回之后。不要把 staging 背压混进
     `approx_queue_delay`。
   - 所有 D2H batch 累加 `d2h_bytes` / `d2h_batches`。
   - 样本 >=4MB 时再累加 `d2h_probe_bytes` / `d2h_probe_event_us` /
     `d2h_probe_host_us`，Python 用差分计算
     `event_bw` 和 `approx_queue_delay = host_elapsed_us - event_elapsed_us`。
   - baseline、EWMA、pressure、clean window 都在 Python governor 内做。这些统计
     只用于持续压力反馈和审计，不尝试实时捕捉 ms 级 burst。

### 8.4 `prepare_step` 停顿计时

在 `monitoring/csrc/ring/ring_engine_py.cu::prepare_step()`：

- 只在 slow path 计时：
  - `STEP_OVERSIZED` 分支：主 stream sync + `force_flush_and_wait()` 的总耗时；
  - `STEP_RING_FLUSHED` 分支：主 stream sync + `force_flush_and_wait()` 的总耗时。
- 把 `stall_us_total` / `stall_count` 作为 RingEngine 侧单调计数器，随
  `link_stats()` 一起返回。
- Python `BackendAdaptor.before_forward()` 在 `prepare_step()` 后调用
  `transport.governor.on_step()`；governor 通过 `link_stats()` 差分读取本 step
  是否发生停顿。
- `before_forward()` 当前只有 `n_hooks > 0` 才调用 `prepare_step()`；如果无捕获
  则也不调用 `on_step()`。这是良性的，因为无捕获就无 DMI drain 流量；但
  feedback-yield 写入的 deadline 必须有界（例如 `now + 2 * feedback_window`，
  每次 `on_step()` 滚动续期），不能写无限期 defer。

### 8.5 Python 集成改动

1. `monitoring/ring_transport.py`
   - `RingTransport.__init__` 增加 `self.governor: Optional[PCIeGovernor] = None`。
   - 增加 `set_governor(governor)`；不加 `on_pcie_hint()` /
     `poll_governor()` 这类死转发。
   - 不改 HookPoint / producer 分支；capture 路径完全不感知 governor。

2. `monitoring/engine.py`
   - `enable_ring_transport()` 创建 `RingTransport` 后解析总开关。
   - 如果启用 governor：构造 `PCIeGovernor(ring_engine, cfg)`，
     `transport.set_governor(governor)`，并调 `monitoring.governor.set_current(governor)`。
   - 如果关闭 governor：`transport.set_governor(None)`，
     `monitoring.governor.set_current(None)`，并调用
     `ring_engine.set_drain_control(0, 0, 0)` 清零 C++ 控制面。
   - `enable_ring_transport()` 的 teardown-重建路径和 `close()` 都必须先
     `monitoring.governor.set_current(None)`，再 stop 旧 `ring_engine`，避免
     connector 拿到指向已 stop engine 的 stale governor。

3. `monitoring/adaptor_base.py`
   - `before_forward()` 保持现有 `prepare_step(total_bytes, n_hooks)` 顺序；
   - `prepare_step()` 后仅在 `transport.governor is not None` 时调用
     `transport.governor.on_step()`。
   - 不改变 `force_eager` 语义；`STEP_OVERSIZED` 仍走现有 CPU-direct safety net。

4. `integration/vllm_adapter.py`
   - 从 `additional_config` / env 增加：
     - `dmx_pcie_governor_enabled` / `DMX_PCIE_GOVERNOR_ENABLED`
     - `dmx_pcie_governor_hint_min_mb`
     - `dmx_pcie_governor_max_defer_us`
     - `dmx_pcie_governor_hard_watermark_ratio`
     - `dmx_pcie_governor_max_chunk_mb`
     - `dmx_pcie_governor_baseline_gbps`
   - 创建 `MonitoringEngine` 后把 config 传入 engine 或直接在 adapter 初始化
     governor。推荐放在 `MonitoringEngine`，因为 HF 也能复用。

### 8.6 vLLM / LMCache hint 接入

v1 原则：hint 接入必须是 best-effort、可选、失败静默，不影响 vLLM 原语义。
公共 bridge 位于 `vllm/distributed/kv_transfer/dmi_pcie_hint.py`：optional import
`monitoring.governor.emit_hint`，并提供无锁 `D2HHintLease`。所有异常 fail-open。

1. `integration/vllm/vllm/v1/worker/{gpu/kv_connector.py,kv_connector_model_runner_mixin.py}`
   - 若 connector 声明 `manages_dmi_pcie_hints=True`，worker 只调用 connector，
     不再发重复 coarse hint；
   - 其他 connector 保留 `wait_for_save()` 前后 `source="kv_store"` fallback；
   - H2D load 不驱动 v1 DMI defer。

2. `integration/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
   - runtime 自动读取 adapter 的 `use_layerwise`，不要求用户改变 LMCache 配置；
   - non-layerwise：metadata 确认有 store 后，在 `wait_for_save()` 前 begin，函数
     返回或异常时 end；
   - layerwise：第一个实际可保存请求的 `save_kv_layer()` 前 begin，后续 layer
     低频 renew，最终 `wait_for_save()` 后 end；
   - 无 store 的 step 不发 hint；`use_native=true/false` 都经过同一 wrapper。

3. `integration/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`
   - 提交第一个 STORE batch 时 `source="lmcache_mp_store"` begin；
   - `wait_for_save()` 返回不结束 lease；
   - `get_finished()` 利用已有 `store_futures`/`CUDAMessagingFuture.query()` 续租，
     只有 pending future 集合为空时 end；多个并发 store 不会提前恢复；
   - 只追踪 CUDA D2H 完成，不等待后续磁盘/远端持久化。

4. `integration/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
   - `OffloadingConnectorWorker.start_kv_transfers()` 对 load jobs 发 H2D hint；
   - `prepare_store_kv()` 对 store jobs 发 D2H hint，但注意当前代码会把 store
     延迟到下一 engine step 的开头提交，hint 的 `valid_until` 应覆盖这个延迟；
   - `get_finished()` 已拿到 `transfer_size` / `transfer_time` / `transfer_type`，
     v1 先写入 connector 自身日志或通过 `snapshot()` 审计，不新增
     `record_tier0_transfer()` 入口；source 可信度留 v2。

### 8.7 最小 v1

v1 目标是证明 serving-first 控制有效，不追求完整 PCIe 拓扑感知。

必须完成：

1. C++ drain control：`defer_until_ns`、`max_d2h_chunk_bytes`、
   `hard_watermark_bytes` 三个 atomic。
2. `link_stats()`：D2H batch/probe 单调计数器、pending/ring/staging occupancy、
   prepare_step stall 单调计数器。
3. `monitoring/governor.py`：hint 优先 + feedback 滞回状态机；EWMA / baseline /
   pressure 全在 Python。
4. vLLM worker 的 coarse fallback + LMCache sync/layerwise/MP lifecycle lease。
5. 单 rank / 单 GPU 路径；多 GPU 只是每 rank 独立实例，无跨 rank 协调。
6. 单元测试：
   - `tests/test_pcie_governor_state.py`：hint 进入 yield、过期恢复、feedback
     进入/退出 yield、hard watermark 覆盖 yield；
   - `PCIeGovernor` 构造时注入 duck-typed engine（只要求 `link_stats()` /
     `set_defer_until_ns()`），state test 用纯 Python fake 跑，不碰 CUDA；
   - `tests/test_adapter_protocol.py` 扩一例：`before_forward()` 调 governor 但
     不改变 `force_eager`；
   - C++ binding smoke test：`set_drain_control()` / `set_defer_until_ns()` /
     `link_stats()` 可调用，且 Python/C++ monotonic now 差值小于容差；
   - governor 关闭或 3 个控制字段全 0 时，drain 行为与当前实现等价；
   - 总开关关闭时不创建 governor、`get_current()` 返回 `None`、connector hint
     no-op、`before_forward()` 不调用 governor。
   - `test_dmi_pcie_hint.py`：lease begin/renew/end、non-layerwise/layerwise 时序、
     no-store/异常清理、MP future completion/并发、managed connector 去重。

明确不做：

- 不改 hook selection / strip / producer；
- 不新增 drop 策略；
- 不依赖 NVML；
- 不承诺抢占已提交的 memcpy；
- 不新增 `minimal_flush_and_wait()` 阻塞 API；
- 不在 C++ 维护 EWMA / baseline / pressure；
- 不改任何 `exp/` 或 `experiments/` 下文件。

### 8.8 可选 v2

- NVML TX/RX 交叉验证：单独 `monitoring/nvml_pcie_probe.py`，无权限时自动禁用；
- source credibility：按 `kv_store` / `lmcache_store` / `offloading_store` 维护
  hint lead-time、est_bytes 误差、missed collision；
- offloading connector 的真实 `transfer_size` / `transfer_time` 回灌，用于
  source credibility 和 est_bytes 校准；
- 空窗回填：根据 step 边界、hint valid window 和 decode cadence，把 normal
  drain 排进 KV save 完成后的窗口；
- 跨 rank / 共享 switch：用 NVML topology common ancestor 做同 switch 分组，
  但 v1 不阻塞于此；
- 更精确 queue delay：若需要绝对 copy-engine 排队时间，再引入 CUPTI 或专门
  polling thread；不把它放入 v1。

### 8.9 实施顺序

1. 先做 C++ `set_drain_control()` 和单调 `link_stats()`，保证 3 个控制字段全 0
   时行为等价于当前 drain。
2. 接 Python 总开关 disabled path：不创建 governor、清零 C++ 控制、connector
   `get_current()` 返回 `None`。
3. 接 `set_defer_until_ns()` 热路径、normal-loop batch cap 和 hard watermark
   override；确认 force flush 不受 defer/cap 影响，已提交 memcpy 不被取消。
4. 接 `prepare_step` stall 审计，确认不会增加 fast path 开销。
5. 接 Python `PCIeGovernor`，先用 synthetic hint 驱动 defer / hard watermark
   override。
6. 接 vLLM coarse fallback 和 LMCache connector-managed lifecycle lease。
7. 分别用 non-layerwise、layerwise、LMCacheMP 实机验证 begin/end 相对真实 D2H
   的顺序。
8. 最后调参：hard watermark、watchdog、chunk size、feedback EWMA 阈值。

### 8.10 2026-07-09 实现验证

- 单元测试：新增 11 个 lifecycle case，覆盖 lease begin/renew/end、managed
  connector 去重、non-layerwise/layerwise/no-store/异常路径、MP pending/并发
  future；原 LMCache connector 23 个测试保持通过。
- 真实 non-layerwise：Qwen3-0.6B + DMI + LMCache CPU，1024-token cold store；
  日志顺序为 `lmcache_store begin` → 6.32ms offload → end，end 后 governor
  deadline 立即归零。
- 真实 layerwise：相同 workload；begin 出现在 LMCache GPU buffer 初始化和第一层
  D2H 前，19.86ms layerwise store 完成后 end，不再是尾部对齐。
- 真实 LMCacheMP：独立 LMCache server + `LMCacheMPConnector` + DMI；begin 后
  server 完成 1024-token CUDA store（13ms），worker 观察到 CUDA future 完成后
  end。`wait_for_save()` 返回本身不清 lease。
