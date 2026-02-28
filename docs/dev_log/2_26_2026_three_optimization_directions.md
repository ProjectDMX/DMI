# 三个优化方向：Pipeline 提速、Forward/D2H 并行、选择性监控

> 日期：2026-02-26
> 基于 GPT-2 small (12 layers), fp32, batch=64, decode_steps=64, StaticCache
> 监控模式：custom ops + torch.compile(mode="reduce-overhead")
> 183 HookPoint slots，MON_NATIVE_HOST_COPY_THREADS=10

## 当前 Pipeline 时序

```
每个 decode step 的时间线（串行，同步 copy 路径）：

[forward 1.6ms] → [shadow D2H + parse 0.6ms] → [tensor D2H 2.3ms] → [H2H 5.7ms]
                                                                       ↑ pinned→pageable
总开销：~10.2ms/step（forward 占比仅 16%）
```

| 阶段 | 耗时 | 描述 |
|---|---|---|
| Forward | 1.6ms | torch.compile 管理的 CUDA Graph replay |
| Shadow D2H + Parse | 0.6ms | metadata buffer GPU→pinned→C++ 解析 |
| Tensor D2H | 2.3ms | 183 个 tensor GPU→pinned（PCIe） |
| H2H | 5.7ms | 183 个 tensor pinned→pageable（CPU 内存） |

---

## 方向一：Pipeline 后半段提速（H2H 优化 + D2H/H2H 并行）

### 1.1 问题分析

H2H（pinned→pageable）是当前最大瓶颈，占 step 总开销的 56%。

根因：
- 每个 tensor 单独 `malloc()` + `memcpy()` + `release_pool_block()`
- 183 次 malloc 的 allocator 开销显著
- 小 tensor（~200KB）无法饱和 DDR4 带宽

### 1.2 已验证的 H2H 路径

| 路径 | 耗时/step | 相对速度 | 说明 |
|---|---|---|---|
| Per-tensor Thread Pool (10 threads) | 2.2ms | 1.0x (baseline) | 多线程并行 memcpy，当前最优 |
| Gather 单线程 | 9.5ms | 0.23x | 1次 malloc 但单线程 sequential memcpy |
| Gather + 并行 memcpy | 3.0ms | 0.73x | 1次 malloc + thread pool，但 chunk 分配不均 |

详细分析见 `2_26_2026_compile_mode_hook_and_h2h_analysis.md`。

### 1.3 推荐优化：Ring/Ping-Pong Pageable Buffer

**目标**：消除热路径 malloc + 实现 D2H 与 H2H 的 step 间流水线。

**当前（串行）**：
```
Step N:   [D2H 2.3ms][H2H 2.2ms]
Step N+1:                         [D2H 2.3ms][H2H 2.2ms]
Step 开销：4.5ms（D2H + H2H 串行）
```

**优化后（流水线）**：
```
Step N:   [D2H→pinned_A 2.3ms][H2H: pinned_A→pageable_buf[0] 2.2ms]
Step N+1:                      [D2H→pinned_B 2.3ms][H2H: pinned_B→buf[1] 2.2ms]
Step 开销：~2.3ms（被 D2H 限制，H2H 完全隐藏）
```

**实现要点**：
- 预分配 K 个 pageable buffer（~18MB each），step 轮流写入
- Thread pool 多线程 memcpy 仍可用，只是目标从 per-tensor malloc 变为 ring buffer offset
- 需要引用计数或 fence 机制确保 consumer 消费完才能回收 slot
- 如果 consumer 慢（delay_steps > ring size），需要 backpressure

**预期收益**：
- H2H 零 malloc（启动时预分配一次）
- D2H + H2H 从串行 4.5ms → 流水线 ~2.3ms
- Step 监控开销：从 ~4.5ms 降至 ~2.5ms

### 1.4 更激进的优化：GPU-side Gather（Design B 的局部实现）

不做完整的 Design B staging ring buffer，仅在 GPU 上做 gather：

```
当前：183 × cudaMemcpyAsync（GPU→pinned），每次 ~200KB
优化：1 × GPU gather kernel（scatter→contiguous buffer）+ 1 × 大块 DMA（~18MB）
```

收益：单次大 DMA 可接近 PCIe 理论带宽（12 GB/s），但增加 D2D gather 开销。
适用场景：如果 D2H 碎片化（182 次小拷贝的 DMA setup 开销）成为瓶颈。

### 1.5 方向一优先级

1. **Ring Buffer**（中等工作量，收益确定）：消除 malloc + D2H/H2H 流水线
2. **GPU Gather**（较大工作量，Design B 的子集）：从根本上解决 D2H 碎片化
3. **Gather + 并行 memcpy 调优**（小工作量）：按字节均分 chunk

---

## 方向二：Forward 与 D2H 并行

### 2.1 核心问题

当前每个 step 的 forward 和 D2H 是串行的：

```
Step N:   [forward 1.6ms][D2H + H2H 4.5ms]
Step N+1:                                   [forward 1.6ms][D2H + H2H 4.5ms]
```

forward 执行完后，被监控的 tensor 占用的 GPU 内存不能被下一步 forward 覆盖，否则 D2H 拷贝到的数据就是脏的。在 CUDA Graph 下这个问题更严重——replay 会写入与 capture 时完全相同的物理地址。

### 2.2 Design A：Dual-Graph Ping-Pong（零拷贝）

**思路**：预分配两套物理内存（Slotset A/B），交替录制两张 CUDA Graph，奇偶步切换。

```
Step 2t:   [Graph α → Slotset A][backend 读 Slotset B]
Step 2t+1: [Graph β → Slotset B][backend 读 Slotset A]
```

| 项目 | 评估 |
|---|---|
| 性能 | 最优——forward 与 copy 完全并行，零 D2D 开销 |
| torch.compile 兼容性 | **不兼容** |
| 实现复杂度 | 非常高——需要控制 allocator mempool 分区 |
| 显存开销 | 2× 监控数据量 |

**为什么不兼容 torch.compile**：

`torch.compile` 对每个 input signature 缓存一张 graph。Design A 需要同一 signature 交替使用两张 graph，写入不同物理地址。torch.compile 不提供：
- 按 step parity 选择不同 graph 执行
- 为同一编译函数维护两套 tensor allocation
- 控制 allocator 的 mempool 分区

**可能的绕行方案**：

1. **GPU-side step counter**：在 record kernel 内根据 counter 选写 buffer A/B。但 counter 只影响 metadata buffer，不影响激活值的物理地址（由 allocator 决定），无法解决核心问题。

2. **`torch.cond` 条件分支**：`torch.cond(step%2==0, forward_A, forward_B, (x,))`。但 torch.cond 在 reduce-overhead 模式下的 CUDA Graph 支持不成熟。

**结论**：如果要做 Design A，必须使用手动 CUDA Graph 管理，放弃 torch.compile。

### 2.3 Design B：Fused Gather Pipeline（D2D 快照）

**思路**：保持单张 graph，在 graph 末尾插入 gather kernel，把散布的监控 tensor 聚合到 staging ring buffer，再异步 D2H。

```
Step N:   [forward + gather→staging[0] 1.8ms][async D2H staging[0]→host]
Step N+1: [forward + gather→staging[1] 1.8ms]       [async D2H staging[0]→host]
                                                      ↑ 独立 stream，与 forward 并行
```

| 项目 | 评估 |
|---|---|
| 性能 | forward 与 D2H 并行，但 gather D2D 增加 ~0.2ms |
| torch.compile 兼容性 | **兼容** |
| 实现复杂度 | 中等——需要新 custom op `gather_to_staging` |
| 显存开销 | staging ring buffer × 帧数（~36MB/帧） |

**为什么兼容 torch.compile**：

```
torch.compile 的编译边界天然对齐 Design B：
  ├─ [compiled region] forward + record() + gather_to_staging()  ← 全在 graph 内
  └─ [eager region]    async D2H + CPU processing                ← graph 外
```

关键可行性条件：
1. **gather kernel 是 custom op**：Dynamo 追踪进编译图，replay 时自动执行
2. **地址稳定性**：reduce-overhead 模式下所有中间激活地址 replay 间不变，gather 的源地址有效
3. **帧切换**：CPU 在两次 compiled_forward 之间更新 device-side frame counter（1次 H2D scalar copy）
4. **Async D2H 在 graph 外**：独立 stream 复制 staging buffer，下一步 forward 可立即开始

**需要新增的 custom op**：

```cpp
// graph_monitor_ops.cu
void gather_to_staging(
    const at::Tensor& metadata_buffer,   // slot 地址信息（record kernel 写入的）
    const at::Tensor& staging_buffer,    // 目标 staging ring 的当前帧
    int64_t num_slots,
    int64_t frame_id
);

// Meta dispatch (for Dynamo tracing)
TORCH_LIBRARY_IMPL(graphmonitor_ops, Meta, m) {
  m.impl("gather_to_staging", [](const at::Tensor&, const at::Tensor&, int64_t, int64_t) {});
}
```

**gather kernel 实现思路**：
- 读 metadata buffer 中每个 slot 的 `data_ptr`、`nbytes`
- 对于每个 slot，从 `data_ptr` 拷贝 `nbytes` 数据到 `staging_buffer + offset[slot]`
- 所有 slot 的 offset 在 capture 时确定（因为 shape 固定）
- 可以用多个 CUDA thread blocks 并行拷贝不同 slot

### 2.4 Design A vs Design B 对比

| | Design A (Dual Graph) | Design B (Fused Gather) |
|---|---|---|
| Forward/Copy 并行 | 完全并行 | 完全并行（gather 串在 forward 尾部） |
| 额外 GPU 开销 | 无 | gather D2D ~0.2ms |
| torch.compile | 不兼容 | 兼容 |
| 实现复杂度 | 非常高 | 中等 |
| 显存 | 2× 激活 | staging ring × 帧数 |
| D2H 效率 | 仍然 182 次小拷贝 | 1 次大块 DMA（~18MB） |

### 2.5 推荐路线

在 custom ops + torch.compile 路径下：

1. **首选 Design B**——与 torch.compile 天然兼容，gather kernel 解决 D2H 碎片化问题，staging ring buffer 实现 forward/D2H 完全并行。
2. **Design A 仅作为手动 Graph 路径的备选**——如果需要极致性能且能接受手动 CUDA Graph 管理的复杂度。

### 2.6 预期收益（Design B）

```
当前（串行）：
  Step = [forward 1.6ms] + [shadow D2H 0.6ms] + [tensor D2H 2.3ms] + [H2H 2.2ms]
       = 6.7ms/step

Design B（并行）：
  Step = max([forward 1.6ms + gather 0.2ms], [async D2H 1.2ms*]) + [H2H 流水线化]
       ≈ 1.8ms/step（被 forward+gather 限制）

  * 单次大块 DMA 18MB / 12 GB/s = 1.5ms，比 forward+gather 短，完全隐藏
```

如果结合方向一的 Ring Buffer H2H 优化，H2H 也能流水线化，总 step 开销接近纯 forward 时间。

---

## 方向三：选择性监控

### 3.1 Hook 级别：用户指定需要的 hook

**需求**：用户不需要监控所有 183 个 HookPoint，应该能选择性开启。

**当前状态**：
- `profile_decode.py` 已有 `--hook-selection` 参数（full/attention/mlp/minimal）
- `MonitoringConfig.HookSelection` 已有基础设施
- `module_filter` 机制已支持按名称/模块类型过滤

**需要做的**：

#### 3.1.1 API 设计

```python
# 方案 A：白名单（推荐）
monitor = GraphMonitor(
    model,
    module_filter=lambda name, mod: name in user_selected_hooks,
)

# 方案 B：正则匹配
monitor = GraphMonitor(
    model,
    hook_patterns=["blocks.*.hook_resid_*", "blocks.*.hook_attn_out"],
)

# 方案 C：按类别
monitor = GraphMonitor(
    model,
    hook_categories=["residual", "attention_pattern"],  # 预定义类别
)
```

#### 3.1.2 性能影响

hook 数量与 D2H/H2H 时间近似线性关系：

| Hook 数量 | 估算数据量/step | D2H 时间 | H2H 时间 |
|---|---|---|---|
| 183 (全部) | ~18MB | 2.3ms | 2.2ms |
| 91 (hidden states only) | ~9MB | ~1.2ms | ~1.1ms |
| 36 (attention only) | ~3.6MB | ~0.5ms | ~0.5ms |
| 3 (minimal) | ~0.6MB | ~0.08ms | ~0.08ms |

#### 3.1.3 实现要点

- `GraphMonitor._register_hooks` 已支持 `module_filter`
- 需要确保 `_mon_slot_*` 的 inline attrs 与 filter 一致（只对被选中的 HookPoint 设置）
- native backend 的 `HookConfig` 已经按名称索引，无需改动
- gather kernel（如果实现 Design B）的 slot 偏移量需要根据实际 hook 数量动态计算

### 3.2 Request 级别：Batch 内选择性监控

**需求**：一个 batch 内只监控部分 request，忽略其余的（特别是已结束的 request 的 padding）。

**当前状态**：
- `CaptureSchedule` 已支持 `request_stride`/`request_offset`/`warmup_requests`
- native backend 的 `begin_request()` / `should_capture_request()` 已实现
- 但这是 request 粒度的整体开关，不是 batch 内的 per-request 选择

#### 3.2.1 问题分析

HF GPT-2 的 batch 维度结构：
```
tensor shape: [batch_size, seq_len, hidden_dim]
                ↑ 每个 batch 位置对应一个 request

Batch = [request_0, request_1, request_2, ..., request_63]
         active     active     finished(padding)  active
```

当前行为：D2H 拷贝整个 tensor，包括已结束 request 的 padding 数据。

#### 3.2.2 方案设计

**方案 A：D2H 后 CPU 侧裁剪（最简单）**

```python
# 在 consumer/delegate 层面
active_mask = [True, True, False, ..., True]  # batch 内哪些 request 是 active 的
for slot_result in step_results:
    tensor = slot_result.tensor  # [batch, ...]
    tensor = tensor[active_mask]  # 只保留 active 的
```

- 优点：无需改动 GPU pipeline，实现简单
- 缺点：仍然拷贝了无用数据（bandwidth 浪费）
- 适用场景：padding 比例不高时（< 30%）

**方案 B：GPU-side mask gather**

```
在 gather kernel 中加入 active_mask:
  gather_to_staging(metadata, staging, num_slots, frame_id, active_mask)
  → 只拷贝 active request 对应的行
```

- 优点：从源头减少数据量
- 缺点：staging buffer 的 layout 变为不规则（每 step 的 active 数量不同）
- 适用场景：Design B 实现后自然扩展

**方案 C：Per-request shadow buffer**

```
为每个 request 维护独立的 shadow buffer slot 区域：
  request_0: slot[0..182]
  request_1: slot[183..365]
  ...
record kernel 根据 request_id 选写不同区域
```

- 优点：完全隔离，可独立 D2H
- 缺点：slot 数量 × batch_size，显存开销大
- 适用场景：需要异步消费不同 request 的结果时

#### 3.2.3 Padding 检测

HF 的 batch 管理方式：
- 使用 `attention_mask` 标记有效 token（1=valid, 0=padding）
- 在 vLLM 中，continuous batching 会动态替换已结束的 request

检测方法：
```python
# 方案 1：外部传入 active mask
engine.begin_step(step_id, active_requests=[0, 1, 3, ...])

# 方案 2：从 attention_mask 推断
active_mask = attention_mask.any(dim=-1)  # [batch] bool tensor
```

#### 3.2.4 推荐实现路径

1. **短期**：方案 A（CPU 侧裁剪）——改动最小，只需在 consumer 层加 mask
2. **中期**：方案 B（GPU mask gather）——与 Design B 的 gather kernel 一起实现
3. **长期**：方案 C（per-request buffer）——如果需要支持 vLLM continuous batching 的异步消费

### 3.3 方向三优先级

1. **Hook 白名单 API**（小工作量）：已有基础设施，只需暴露更好的用户接口
2. **CPU 侧 active request 裁剪**（小工作量）：在 consumer 层面加 mask
3. **GPU mask gather**（中等工作量）：依赖 Design B gather kernel

---

## 总结：三个方向的依赖关系与优先级

```
                            ┌─────────────────────────┐
                            │  方向三：选择性监控       │
                            │  (1) Hook 白名单 API     │ ← 独立，可立即做
                            │  (2) CPU 侧 request 裁剪 │ ← 独立，可立即做
                            │  (3) GPU mask gather     │ ← 依赖 Design B
                            └────────────┬────────────┘
                                         │
                            ┌─────────────▼────────────┐
                            │  方向二：Forward/D2H 并行  │
                            │  Design B: gather kernel  │ ← 核心工作
                            │  + staging ring buffer    │
                            │  + async D2H stream       │
                            └────────────┬────────────┘
                                         │
                            ┌─────────────▼────────────┐
                            │  方向一：H2H 优化          │
                            │  Ring/Ping-Pong pageable  │ ← 独立于方向二
                            │  buffer（消除 malloc）     │
                            └──────────────────────────┘
```

### 推荐执行顺序

| 阶段 | 任务 | 预期收益 | 工作量 |
|---|---|---|---|
| **Phase 1** | Hook 白名单 API + CPU request 裁剪 | 按需减少 50-90% 数据量 | 1-2 天 |
| **Phase 2** | Ring/Ping-Pong pageable buffer | D2H+H2H 从 4.5ms→2.3ms | 3-5 天 |
| **Phase 3** | Design B gather kernel + staging ring | Forward/D2H 并行，step→~1.8ms | 2-3 周 |
| **Phase 4** | GPU mask gather（扩展 Design B） | 减少无用 padding 的 GPU→CPU 带宽 | 1 周 |

### 最终目标

```
当前：  [forward 1.6ms][D2H 2.3ms][H2H 2.2ms] = 6.1ms/step（不含 shadow parse）
Phase 2: [forward 1.6ms][D2H+H2H 流水线 2.3ms] = 3.9ms/step
Phase 3: [forward+gather 1.8ms] ← D2H 完全隐藏 → ~1.8ms/step（接近纯 forward）
```

---

## 附录：关键文件索引

| 文件 | 内容 |
|---|---|
| `docs/dev_log/graph_monitor_replan.md` | 总体架构计划，Design A/B 详细描述 |
| `docs/dev_log/custom_op_torch_compile_feasibility.md` | torch.compile 兼容性分析 |
| `docs/dev_log/2_23_2026_d2h_h2h_bandwidth_analysis.md` | D2H/H2H 带宽分析 |
| `docs/dev_log/2_26_2026_compile_mode_hook_and_h2h_analysis.md` | Hook bug 修复 + H2H 路径对比 |
| `monitoring/csrc/graph_monitor_ops.cu` | record/sink custom ops |
| `monitoring/csrc/engine_core.cpp` | native backend worker + H2H thread pool |
| `monitoring/graph_monitor.py` | GraphMonitor（hook 注册 + metadata 收集） |
| `benchmark/tests/profile_decode.py` | 基准测试 + TorchCompileDecodeRunner |
