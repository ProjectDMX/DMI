# Forward / D2H 并行方案设计与评估

> 日期：2026-02-26
> 模型：GPT-2 small (12 layers, hidden=768, heads=12, head_dim=64)
> 配置：batch=64, decode seq_len=1, fp32, max_cache_len=81
> 183 HookPoint hooks (after filter), ~18MB monitored data per step

---

# 正式方案：Design B — torch.compile + Gather + Staging Ring + Copy Stream

## 架构总览

```
torch.compile(mode="reduce-overhead") 管理的 CUDA Graph:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  [forward kernels]  →  [hooks collect anchor refs]                  │
  │       1.6ms                    ~0ms                                │
  │                         →  [gather_to_staging(anchors, ...)]       │
  │                                  0.1ms                             │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │ event
                                     ▼
  Copy stream (graph 外):       [D2H: staging → pinned_host]
                                     1.5ms (单次 DMA, 18MB)
                                     │
                                     ▼
  CPU thread pool:              [H2H: pinned → pageable_ring]
                                     ~1ms (多线程 memcpy)
                                     │
                                     ▼
  Consumer:                     [torch.frombuffer() → dict of tensors]
                                     ~0.2ms (零拷贝视图)
```

**Step cadence: 1.7ms**（forward 1.6ms + gather 0.1ms，D2H/H2H 完全隐藏）
vs 当前 6.7ms → **3.9× 加速**

### 核心简化：不再需要 shadow buffer

之前的架构：`record()` per hook → shadow buffer → `sink()` → shadow parser → tensor D2H
现在的架构：hooks 收集 tensor 引用 → `gather_to_staging()` 直接拷贝到 staging

**被淘汰的组件**：
- `record()` custom op — 不再需要写 metadata 到 shadow buffer
- `sink()` custom op — gather_to_staging 直接接收 tensor 参数，同时起到 sink 保活的作用
- `alias_tensor()` custom op — 不再需要
- Shadow buffer (GPU) — metadata 被静态 offset table 替代
- C++ shadow parser — 不再需要每步解析 metadata

**新增/替代的组件**：
- `gather_to_staging()` custom op — 接收 tensor 参数列表（替代 sink 保活 + 数据拷贝一步完成）
- 静态 offset table — 注册时一次性计算，CUDA Graph 下 shape 固定
- `active_mask` tensor — 用于 selective monitoring / padding skip

## 为什么选 Design B

| 考量 | Design A (手动双 Graph) | **Design B (torch.compile + gather)** |
|---|---|---|
| Step cadence | 1.6ms（无 gather）/ 1.7ms（有 gather） | **1.7ms** |
| 差距 | 0~0.1ms | — |
| torch.compile 兼容 | 不兼容（手动 Graph） | **完全兼容** |
| 模型移植 | 每个模型重写 capture 逻辑 | **改一行 torch.compile()** |
| 动态 shape (prefill) | 需要多对 graph | **自动管理** |
| 显存开销 | 86MB~5GB（hold intermediate） | **36~72MB（staging ring）** |
| 大模型可扩展性 | 差（显存 ∝ 全部 intermediate） | **好（显存 ∝ 监控数据量）** |

Design A 的 0.1ms 优势（省了 gather D2D）不值得牺牲可移植性。在大模型上 forward >> 0.1ms，差距消失。

## 关键洞察：为什么不需要 D2H-in-graph

183 次 `cudaMemcpyAsync` 的 2.3ms 延迟中，1.8ms 是 **CPU 侧 per-call overhead**（用户态↔内核态切换 + driver 加锁 + DMA descriptor 编程）。

- **CUDA Graph 内**：录进 graph 后 CPU overhead 消失 → 183 次 DMA = 1.5ms（纯带宽）
- **但 Design B 不需要这样做**：gather 把 183 tensor 合成 1 个 staging buffer → graph 外只需 **1 次** `cudaMemcpyAsync` → CPU overhead = 10μs（可忽略）→ D2H = 1.5ms

两种方式 D2H 都是 1.5ms，但 gather 方案更简单（不需要多流 graph capture）。

| D2H 方式 | 延迟 | 在 graph 内？ | 复杂度 |
|---|---|---|---|
| 183 × async (graph 外) | 2.3ms | 否 | 低，但慢 |
| 183 × async (graph 内多流) | 1.5ms | 是 | 高（多流 capture） |
| **gather + 1 × async (graph 外)** | **1.5ms** | **否** | **低** |

## 完整 Buffer 链

```
GPU:
  staging_ring[0..K-1]   [K × ~18MB]              gather 写连续数据
  slot_offsets           [num_slots] int64          静态，注册时一次性计算
  slot_sizes             [num_slots] int64          静态
  active_mask            [num_slots] bool           动态（padding/selective skip）
  frame_counter          [1] int32                  CPU 每步更新
                         ↓ PCIe D2H (copy stream, 单次 DMA)
Host (pinned):
  pinned_ring[0..K-1]    [K × ~18MB]              D2H 目标
                         ↓ memcpy (CPU thread pool)
Host (pageable):
  pageable_ring[0..K-1]  [K × ~18MB]              consumer 读取
                         ↓ torch.frombuffer() 零拷贝
  dict[str, Tensor]      183 个 tensor 视图          consumer 消费
```

最小配置 K=2（ping-pong），K=3~4 留 backpressure 余量。

### 显存 / 内存预算

| Buffer | K=2 | K=4 | 说明 |
|---|---|---|---|
| GPU staging ring | 36 MB | 72 MB | 仅监控数据量，不随 model size 增长 |
| Pinned host ring | 36 MB | 72 MB | cudaHostAlloc，启动时一次性分配 |
| Pageable host ring | 36 MB | 72 MB | 普通 malloc，consumer 消费后回收 |
| **总计** | **108 MB** | **216 MB** | |

## 完整时序（稳态）

```
Compute stream:
  [fwd+gth₀ 1.7ms][fwd+gth₁ 1.7ms][fwd+gth₂ 1.7ms][fwd+gth₃ 1.7ms]
   → staging[0]     → staging[1]     → staging[0]     → staging[1]

Copy stream:
                   [D2H₀ 1.5ms]     [D2H₁ 1.5ms]     [D2H₂ 1.5ms]
                    stg[0]→pin[0]     stg[1]→pin[1]     stg[0]→pin[0]

CPU thread pool:
                                     [H2H₀ ~1ms]       [H2H₁ ~1ms]
                                      pin[0]→page[0]     pin[1]→page[1]

Consumer:
                                                        [consume₀]
```

- **Compute**: 1.7ms/step（forward 1.6ms + gather 0.1ms）
- **D2H**: 1.5ms < 1.7ms → 完全隐藏在下一步 compute 里
- **H2H**: ~1ms < 1.7ms → 完全隐藏
- **Pipeline latency**: 3 steps（compute → D2H → H2H），但 throughput = 1.7ms/step

## 帧切换机制

torch.compile(reduce-overhead) 将输入 tensor 复制到 static buffer → 不能通过传入不同 staging_frame tensor 来切换帧。

**解决方案：GPU-side frame counter**

```python
frame_counter = torch.zeros(1, dtype=torch.int32, device="cuda")  # 固定地址

def forward_with_gather(x, cache, pos):
    output = model(x, use_cache=True, past_key_values=cache,
                   cache_position=pos, return_dict=True)
    logits = lm_head(output.last_hidden_state)

    # hooks 在 forward 中收集了 anchor 引用列表
    # gather_to_staging 接收 tensor 参数（替代 sink 保活 + 数据拷贝）
    torch.ops.graphmonitor_ops.gather_to_staging(
        monitor.get_anchors(),     # list[Tensor] — 183 个 hook 输出
        staging_ring, frame_counter,
        slot_offsets, slot_sizes,
        active_mask,               # 动态 mask: padding/selective skip
    )
    return logits

compiled_forward = torch.compile(forward_with_gather, mode="reduce-overhead")

copy_stream = torch.cuda.Stream()
for step in range(N):
    cur_frame = step % NUM_FRAMES
    prev_frame = (step - 1) % NUM_FRAMES

    # 更新帧号（H2D scalar, ~1μs）
    frame_counter.fill_(cur_frame)

    torch.compiler.cudagraph_mark_step_begin()
    logits = compiled_forward(token, cache, cache_position)

    # graph 外：单次 D2H on copy stream
    if step > 0:
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            pinned_ring[prev_frame].copy_(staging_ring[prev_frame], non_blocking=True)

# --- CPU-side tensor reconstruction (after D2H completes) ---
def reconstruct_tensors(cpu_buffer, slot_names, slot_offsets, slot_metas):
    """从连续 buffer 重建 dict[str, Tensor]，零拷贝视图，~0.2ms for 183 slots"""
    result = {}
    for name, offset, (shape, dtype) in zip(slot_names, slot_offsets, slot_metas):
        t = torch.frombuffer(cpu_buffer, dtype=dtype,
                             count=math.prod(shape), offset=offset).view(shape)
        result[name] = t
    return result
```

gather kernel 内部（直接读 tensor 参数的 data_ptr，不再需要 shadow buffer）：
```cpp
__global__ void gather_to_staging_kernel(
    const void* const* src_ptrs,     // 每个 anchor tensor 的 data_ptr
    uint8_t* staging_ring_base,      // 整个 ring buffer 的起始地址
    const int32_t* frame_counter,    // GPU-side，CPU 每步更新
    const int64_t* slot_offsets,
    const int64_t* slot_sizes,
    const bool* active_mask,         // per-slot: false = skip (padding/inactive)
    int64_t frame_stride,            // 每帧的字节大小
    int num_slots
) {
    int slot = blockIdx.x;
    if (slot >= num_slots) return;
    if (!active_mask[slot]) return;  // padding or selective skip
    if (src_ptrs[slot] == nullptr) return;

    int frame = *frame_counter;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_ptrs[slot]);
    uint8_t* dst = staging_ring_base + frame * frame_stride + slot_offsets[slot];
    int64_t nbytes = slot_sizes[slot];

    // 4-byte aligned cooperative memcpy
    const uint32_t* src4 = reinterpret_cast<const uint32_t*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    int64_t n4 = nbytes / 4;
    for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        dst4[i] = src4[i];
    }
}
```

## gather_to_staging Op 注册

```cpp
// graph_monitor_ops.cu

// ---- CUDA implementation ----
void gather_to_staging_cuda(
    const at::TensorList anchors,          // 183 个 hook 输出 tensor（替代 sink 保活）
    at::Tensor& staging_ring,              // [num_frames, frame_bytes] uint8
    const at::Tensor& frame_counter,       // [1] int32, device
    const at::Tensor& slot_offsets,        // [num_slots] int64
    const at::Tensor& slot_sizes,          // [num_slots] int64
    const at::Tensor& active_mask          // [num_slots] bool, 动态 mask
) {
    int64_t num_slots = anchors.size();
    int64_t frame_stride = staging_ring.size(1);

    // 构建 src_ptrs 数组（device memory）
    std::vector<const void*> h_ptrs(num_slots);
    for (int64_t i = 0; i < num_slots; ++i) {
        h_ptrs[i] = anchors[i].defined() ? anchors[i].data_ptr() : nullptr;
    }
    // TODO: 优化为持久化 device buffer，避免每步 H2D
    at::Tensor d_ptrs = at::from_blob(h_ptrs.data(), {num_slots},
        at::TensorOptions().dtype(at::kLong)).to(anchors[0].device());

    gather_to_staging_kernel<<<num_slots, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const void* const*>(d_ptrs.data_ptr()),
        staging_ring.data_ptr<uint8_t>(),
        frame_counter.data_ptr<int32_t>(),
        slot_offsets.data_ptr<int64_t>(),
        slot_sizes.data_ptr<int64_t>(),
        active_mask.data_ptr<bool>(),
        frame_stride,
        num_slots
    );
}

// ---- Schema ----
TORCH_LIBRARY(graphmonitor_ops, m) {
    // record() 和 sink() 不再需要 — gather_to_staging 替代两者
    m.def("gather_to_staging(Tensor[] anchors, Tensor(a!) staging, Tensor counter, "
          "Tensor offsets, Tensor sizes, Tensor mask) -> ()");
}

TORCH_LIBRARY_IMPL(graphmonitor_ops, CUDA, m) {
    m.impl("gather_to_staging", gather_to_staging_cuda);
}

// ---- Meta dispatch (for Dynamo tracing) ----
TORCH_LIBRARY_IMPL(graphmonitor_ops, Meta, m) {
    m.impl("gather_to_staging",
           [](const at::TensorList, at::Tensor&, const at::Tensor&,
              const at::Tensor&, const at::Tensor&, const at::Tensor&) {});
}
```

> **注意**：`src_ptrs` 数组在 CUDA Graph 下需要特殊处理。由于 CUDA Graph replay
> 时 anchor tensor 的 data_ptr 是固定的（Graph 烧入地址），src_ptrs 在首次 capture
> 后不会变化，可以在 capture 时一次性构建并缓存到 device memory。后续 replay 直接
> 使用缓存的 device buffer，无需每步 H2D。

## 跨硬件性能预测

| 硬件 | Forward | Gather | D2H | Step Cadence | 瓶颈 |
|---|---|---|---|---|---|
| RTX 3090 (PCIe 3.0) | 1.6ms | 0.04ms | 1.5ms | **1.7ms** | forward+gather |
| RTX 4090 (PCIe 4.0) | ~1.2ms | 0.04ms | 0.72ms | **~1.3ms** | forward+gather |
| A100 (PCIe 4.0) | ~1.0ms | 0.02ms | 0.72ms | **~1.0ms** | forward |
| B200 (PCIe 5.0) | ~0.8ms | 0.01ms | 0.36ms | **~0.8ms** | forward |
| Llama-7B on A100 | ~15ms | 0.05ms | 1.5ms | **~15ms** | forward |

**所有配置下 D2H 都被 forward 完全隐藏。** 模型越大、硬件越新，gather/D2H 的占比越小。

## 实施路线

```
Phase 1: gather_to_staging kernel + 架构简化（1-2 周）
  ├─ 实现 gather_to_staging CUDA kernel（接收 Tensor[] anchors）
  ├─ TORCH_LIBRARY 注册 + Meta dispatch（Dynamo 兼容）
  ├─ 删除 record() / sink() / alias_tensor() / shadow buffer / shadow parser
  ├─ Python 端 GraphMonitor 改为：hooks 收集 anchor refs → gather_to_staging()
  ├─ 注册时一次性计算 slot_offsets / slot_sizes（StaticCache → shape 固定）
  ├─ 单元测试：correctness（gather 后 staging 内容 == 原始 tensor）
  └─ 基准测试：gather kernel 延迟 vs 预测（~0.1ms）

Phase 2: Staging ring + copy stream pipeline（1-2 周）
  ├─ 分配 staging_ring (GPU) + pinned_ring (host)
  ├─ frame_counter 机制（GPU scalar, CPU 更新）
  ├─ Copy stream D2H 调度（event-based synchronization）
  ├─ CPU 侧 torch.frombuffer() 重建 tensor 视图（~0.2ms）
  ├─ 集成到 TorchCompileDecodeRunner
  └─ 端到端基准：verify step cadence ≈ 1.7ms

Phase 3: H2H ring pageable buffer（1 周，可与 Phase 1-2 并行）
  ├─ 预分配 pageable_ring，消除 per-step malloc
  ├─ Thread pool memcpy: pinned_ring[i] → pageable_ring[i]
  └─ Consumer 从 pageable_ring 以 torch.frombuffer() 视图读取

Phase 4: 集成到 GraphSafeEngine / native backend（1 周）
  ├─ GraphSafeEngine 增加 staging mode
  ├─ Native backend delegate 从 pageable_ring 消费
  ├─ active_mask 动态更新（padding skip / selective monitoring）
  └─ 更新 benchmark CLI：--graph-copy-mode staging
```

---
---

# 以下为方案探索过程（Design A vs B 详细分析）

## 0. 当前瓶颈

```
当前（串行 sync copy）：
  Step N:   [forward 1.6ms][shadow D2H 0.6ms][tensor D2H 2.3ms][H2H 2.2ms]
  Step N+1:                                                                  [forward ...
  Step 开销：~6.7ms（forward 仅占 24%）
```

核心问题：forward 结束后，所有后续阶段（shadow parse、tensor D2H、H2H）都必须在下一步 forward 之前完成，因为**单张 CUDA Graph replay 会覆盖上一步的所有中间激活地址**。

---

## 1. Design A-Hybrid：torch.compile(default) + 手动双 Graph + Hold Intermediate

### 1.1 原理

两张 CUDA Graph（α/β）共享同一个 memory pool，但通过在 capture 阶段 hold 住所有 intermediate tensor 的引用，强制两张 graph 的**全部地址空间不重叠**。

```
Capture α：
  所有 intermediate 不 free → α 占据 pool 中 {I_α ∪ O_α}

Capture β（α 的全部地址仍被 hold）：
  allocator 被迫在 pool 其他区域分配 → β 占据 {I_β ∪ O_β}
  {I_α ∪ O_α} ∩ {I_β ∪ O_β} = ∅  ← 保证

释放 held references（graph 已录好）
```

交替 replay 时：
- replay α 只写 {I_α ∪ O_α}，不碰 β 的任何地址
- replay β 只写 {I_β ∪ O_β}，不碰 α 的任何地址
- **D2H of step N 与 forward of step N+1 可完全并行**

### 1.2 实现

```python
compiled_forward = torch.compile(model.forward, mode="default")

# Warmup Inductor
for _ in range(3):
    compiled_forward(static_input, past_key_values=cache, cache_position=pos)

pool = torch.cuda.graph_pool_handle()

# ---- Capture α，hold all intermediates ----
held_a = []
hooks = [m.register_forward_hook(lambda m, i, o, h=held_a: h.append(o))
         for m in model.modules()]
graph_a = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph_a, pool=pool):
    static_out_a = compiled_forward(static_input, ...)
for h in hooks: h.remove()

# ---- Capture β（α 全部地址被 held_a 占据）----
graph_b = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph_b, pool=pool):
    static_out_b = compiled_forward(static_input, ...)
del held_a  # 释放，地址已烧入 graph

# ---- Ping-pong decode ----
copy_stream = torch.cuda.Stream()
for step in range(N):
    static_input.copy_(token)
    cache_position.fill_(step)
    if step % 2 == 0:
        graph_a.replay()
        # D2H α's monitored tensors on copy_stream
    else:
        graph_b.replay()
        # D2H β's monitored tensors on copy_stream
```

### 1.3 时序分析

D2H 仍然是 183 次 scattered cudaMemcpyAsync（每次 ~100KB），总计 ~2.3ms。

```
Compute stream: [fwd_α 1.6ms][fwd_β 1.6ms][fwd_α 1.6ms][fwd_β 1.6ms]
Copy stream:                 [D2H_α 2.3ms-------->]     [D2H_β 2.3ms-------->]
                                            ↑ 超出 fwd_β 0.7ms
```

**问题**：每 1.6ms 产出一个 step 的 D2H 任务，但每个任务需要 2.3ms。Copy stream 每步落后 0.7ms，不可持续。

**两种解决方案**：

#### 方案 A-1：接受 D2H 瓶颈，step cadence = 2.3ms

```
Compute: [fwd_α 1.6ms]       [fwd_β 1.6ms]       [fwd_α 1.6ms]
Copy:                [D2H_α 2.3ms]       [D2H_β 2.3ms]
                     ↑ compute 等 copy 完成再开始下一步

Step cadence: 2.3ms（被 D2H 限制）
```

vs 当前串行 6.7ms → **2.9× 加速**，无需新 kernel。

#### 方案 A-2：加 GPU gather kernel，step cadence = 1.7ms

在每张 graph 末尾加 gather kernel，把 183 个散布的 tensor 聚合到连续 staging buffer，单次大 DMA 传输。

```
Compute: [fwd_α 1.6ms][gth_α 0.1ms][fwd_β 1.6ms][gth_β 0.1ms]
Copy:                                [D2H_α 1.5ms]              [D2H_β 1.5ms]
                                     ↑ 单次 18MB DMA

Step cadence: 1.7ms（D2H 1.5ms < forward+gather 1.7ms → 完全隐藏）
```

vs 当前串行 6.7ms → **3.9× 加速**。

### 1.4 显存分析

Hold intermediate 导致 graph 内部不做 intermediate 复用，每张 graph 需要所有 intermediate 同时存在的空间。

**GPT-2 small 每层 intermediate（近似）**：

| Tensor | Shape | 大小 |
|---|---|---|
| LayerNorm 1 输出 | [64, 1, 768] | 192 KB |
| Q / K / V 各一个 | [64, 12, 1, 64] × 3 | 576 KB |
| Attention scores | [64, 12, 1, 81] | 243 KB |
| Softmax 输出 | [64, 12, 1, 81] | 243 KB |
| Attention weighted V | [64, 12, 1, 64] | 192 KB |
| Attention projection | [64, 1, 768] | 192 KB |
| LayerNorm 2 输出 | [64, 1, 768] | 192 KB |
| MLP fc1 (4×hidden) | [64, 1, 3072] | 768 KB |
| GELU 输出 | [64, 1, 3072] | 768 KB |
| MLP fc2 | [64, 1, 768] | 192 KB |
| Residual additions | ≈ | 384 KB |
| **每层合计** | | **~3.9 MB** |

| 项目 | 正常 (intermediate 复用) | Hold intermediate |
|---|---|---|
| 12 层 intermediate | ~10 MB (peak 2-3 层) | 12 × 3.9 = **47 MB** |
| + embedding / final LN / lm_head | ~2 MB | ~2 MB |
| **单张 graph 总计** | **~12 MB** | **~49 MB** |
| **两张 graph** | — | **~98 MB** |
| **额外显存开销** | baseline | **+86 MB** |

GPT-2 small + batch=64 + fp32：86MB 额外 → 完全可接受（GPU 显存 24-80GB）。

**模型规模扩展**：

| 模型 | batch | dtype | 每层 intermediate | 两张 graph 额外显存 |
|---|---|---|---|---|
| GPT-2 small (12L) | 64 | fp32 | 3.9 MB | ~86 MB |
| GPT-2 medium (24L) | 64 | fp32 | 3.9 MB | ~176 MB |
| Llama-7B (32L, h=4096) | 1 | fp16 | ~0.1 MB | ~6 MB |
| Llama-7B (32L, h=4096) | 64 | fp16 | ~8 MB | ~500 MB |
| Llama-70B (80L, h=8192) | 64 | fp16 | ~32 MB | ~5 GB |

大模型 + 大 batch 下显存代价较高，需要权衡。

### 1.5 注意事项

1. **Hold hooks 注册时机**：必须在 `torch.compile()` 之后、CUDA Graph capture 之前注册。在 compile 之前注册会导致 Dynamo graph break（Python list append 是 side effect）。

2. **hold hooks 不影响 graph replay**：capture 完成后立即 remove hooks。Replay 不执行 Python hooks，只回放录制的 kernel 序列。

3. **两张 graph 的 model weights 共享**：weights 是只读的，两张 graph 都读同一份，没有冲突。

4. **StaticCache 共享**：两张 graph 都写入同一个 StaticCache（通过 `index_copy_` 更新 KV cache）。每步只 replay 一张 graph，不会冲突。

5. **监控 hook (record/sink)**：作为 custom op 被 Inductor 编译进 graph。两张 graph 各自的 record kernel 写入各自的 shadow buffer（需要为 α/β 分配两个 shadow buffer）。

---

## 2. Design B：Single Graph + Fused Gather + Staging Ring Buffer

### 2.1 原理

保持单张 CUDA Graph，在 graph 末尾加 gather kernel 把散布的监控 tensor 聚合到连续 staging buffer。Staging buffer 使用 ring（≥2 帧），D2H 在独立 stream 上异步进行。

```
单张 graph → forward 每次覆盖相同地址
但 gather kernel 在覆盖前已经把数据拷到 staging buffer
staging[N%K] 不会被 forward 覆盖（独立内存）
```

### 2.2 实现（与 torch.compile 兼容）

```python
# gather_to_staging 作为 custom op，Dynamo 追踪进 graph
torch.ops.graphmonitor_ops.gather_to_staging(
    shadow_buffer,     # metadata: 每个 slot 的 data_ptr + size
    staging_buffer,    # 目标连续 buffer 的当前帧
    num_slots,
)

compiled_forward = torch.compile(forward_with_gather, mode="reduce-overhead")

copy_stream = torch.cuda.Stream()
for step in range(N):
    frame = step % NUM_FRAMES
    torch.compiler.cudagraph_mark_step_begin()
    # graph replay: forward + gather → staging[frame]
    output = compiled_forward(token, cache, pos, staging[frame])

    # async D2H on copy stream（staging[frame] 不会被下一步覆盖）
    with torch.cuda.stream(copy_stream):
        host_buffer.copy_(staging[frame], non_blocking=True)
```

### 2.3 Gather Kernel 延迟分析

**任务**：读 183 个散布的 GPU tensor（各 ~100KB），写到连续 staging buffer。

```
数据量：
  Read:  183 × ~100KB = ~18 MB（scattered）
  Write: 18 MB（sequential to staging）
  Total HBM traffic: ~36 MB

GPU HBM 带宽：
  RTX 3090: 936 GB/s
  RTX 4090: 1008 GB/s
  A100:     2039 GB/s

理论延迟（RTX 3090）：
  36 MB / 936 GB/s = 38 μs

Scattered read penalty（非连续读取）：
  183 个不同地址，每个 ~100KB
  L2 cache miss 后走 HBM → 每个 read ~0.1-0.5μs latency
  但 100KB 远大于 cache line → 主要是 bandwidth-bound
  Penalty: ~1.5-2×

实际估算：
  38 μs × 1.5 ~ 2 = 57 ~ 76 μs
  + kernel launch overhead: ~5-10 μs
  总计：~70-100 μs ≈ 0.1 ms
```

**结论**：gather kernel 延迟 ~0.1ms，占 forward 时间的 6%，开销极小。

### 2.4 D2H 延迟（单次大块 vs 183 次散布）

| 方式 | 数据量 | 延迟 | 原因 |
|---|---|---|---|
| 183 × cudaMemcpyAsync | ~18 MB | 2.3 ms | 183 次 DMA setup overhead (~10μs/次) |
| 1 × cudaMemcpyAsync (staging) | ~18 MB | **1.5 ms** | 单次 DMA，18MB / 12 GB/s (PCIe 3.0) |
| 理论最优 | 18 MB | 1.5 ms | PCIe 3.0 x16 bandwidth limit |

Gather + 单次 DMA 节省 0.8ms（消除 DMA setup overhead）。

### 2.5 时序分析

```
Compute: [fwd 1.6ms][gth 0.1ms][fwd 1.6ms][gth 0.1ms][fwd 1.6ms]
Copy:                           [D2H 1.5ms]            [D2H 1.5ms]
                                ↑ 1.5ms < 1.7ms → 完全隐藏

Step cadence: 1.7ms
```

### 2.6 显存分析

| 项目 | 大小 | 说明 |
|---|---|---|
| Staging ring (2 帧) | 2 × 18 MB = **36 MB** | 最小配置 |
| Staging ring (4 帧) | 4 × 18 MB = **72 MB** | 留 backpressure 余量 |

vs Design A-Hybrid 的 86 MB 额外显存：**Design B 显存开销更小或相当**。

### 2.7 需要新增的 custom op

```cpp
// graph_monitor_ops.cu
//
// gather_to_staging: 从 shadow buffer 中读取每个 slot 的 data_ptr 和 size，
// 从源地址拷贝 tensor 数据到 staging buffer 的对应 offset。
//
// slot_offsets: 预计算的每个 slot 在 staging buffer 中的字节偏移
//              （capture 时 shape 固定 → offset 固定）
//
// 每个 thread block 处理一个 slot 的 memcpy。

__global__ void gather_to_staging_kernel(
    const TensorMetadata* shadow,
    uint8_t* staging,
    const int64_t* slot_offsets,   // [num_slots] 预计算偏移
    const int64_t* slot_sizes,     // [num_slots] 每个 slot 的字节数
    int num_slots
) {
    int slot = blockIdx.x;
    if (slot >= num_slots) return;
    if (shadow[slot].data_ptr == 0) return;  // stale slot, skip

    const uint8_t* src = reinterpret_cast<const uint8_t*>(shadow[slot].data_ptr);
    uint8_t* dst = staging + slot_offsets[slot];
    int64_t nbytes = slot_sizes[slot];

    // Cooperative memcpy within thread block
    for (int64_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Meta dispatch for Dynamo tracing
TORCH_LIBRARY_IMPL(graphmonitor_ops, Meta, m) {
    m.impl("gather_to_staging", [](const at::Tensor&, at::Tensor&,
                                    const at::Tensor&, const at::Tensor&, int64_t) {});
}
```

---

## 3. Head-to-Head 对比

### 3.1 延迟

| 方案 | Step Cadence | 加速比 (vs 6.7ms) | 需要 gather kernel |
|---|---|---|---|
| 当前（串行 sync） | 6.7 ms | 1.0× | — |
| **A-Hybrid 无 gather** | **2.3 ms** | **2.9×** | 否 |
| A-Hybrid + gather | 1.7 ms | 3.9× | 是 |
| **Design B + gather** | **1.7 ms** | **3.9×** | 是 |

### 3.2 显存

| 方案 | 额外显存 (GPT-2 b=64) | 额外显存 (Llama-7B b=64) |
|---|---|---|
| A-Hybrid (两张 graph, hold) | 86 MB | ~500 MB |
| Design B (staging ring ×2) | 36 MB | ~64 MB* |
| Design B (staging ring ×4) | 72 MB | ~128 MB* |

*Llama-7B 的监控数据量取决于 hook 数量和 tensor 大小，此处为估算。

### 3.3 实现复杂度

| 方面 | A-Hybrid | Design B |
|---|---|---|
| CUDA Graph 管理 | 手动（两张 graph + pool + hold） | torch.compile 自动（或手动单张） |
| torch.compile 模式 | `mode="default"`（手动套 graph） | `mode="reduce-overhead"`（自动） |
| 新 custom op | 无（可选 gather） | `gather_to_staging`（必须） |
| Shadow buffer | 需要两份（α/β 各一个） | 一份（单 graph） |
| 帧管理 | 无需（α/β 天然 ping-pong） | Ring buffer + frame counter |
| KV Cache 管理 | 手动 static copy | torch.compile 自动处理 |
| Warmup 流程 | Inductor warmup + 双 graph capture | torch.compile 自动 warmup |

### 3.4 灵活性

| 方面 | A-Hybrid | Design B |
|---|---|---|
| 动态 shape (prefill) | 需要多对 graph | torch.compile 自动管理 |
| Hook 数量变化 | 需要重新 capture 两张 graph | 自动适应（gather offset 重算） |
| 与其他 torch.compile 特性兼容 | 受限（手动 graph） | 完全兼容 |

---

## 4. 关键洞察

### 4.1 Design A-Hybrid 无 gather 已是显著改进

```
当前：  6.7ms/step
A-Hybrid 无 gather：2.3ms/step（2.9× 加速）
```

不需要任何新 kernel。只需要：
1. torch.compile(default) + 手动 CUDA Graph capture
2. Capture 时 hold intermediate
3. 交替 replay + 独立 copy stream D2H

这是**最小实现成本**的并行化方案。

### 4.2 Gather Kernel 是共同的"最后一公里"优化

无论 Design A 还是 Design B，gather kernel 都将 step cadence 从 2.3ms 推到 1.7ms。区别在于：

- **Design A + gather**：gather 在 graph 外手动调用（或录入两张 graph 各自末尾）
- **Design B + gather**：gather 在单张 graph 末尾，staging ring 做帧管理

如果最终目标是 1.7ms，两个方案都需要 gather，此时 **Design B 更简单**（单 graph + staging ring vs 双 graph + hold + pool 管理）。

### 4.3 选择建议

| 场景 | 推荐 |
|---|---|
| 快速出成果，不写新 kernel | **A-Hybrid 无 gather** (2.3ms) |
| 追求最优 latency，接受手动 Graph | A-Hybrid + gather (1.7ms) |
| 追求最优 latency，保持 torch.compile | **Design B** (1.7ms) |
| 大模型，显存敏感 | **Design B**（staging 36MB vs hold 500MB+） |
| 需要支持动态 shape / prefill | **Design B** |

---

## 5. 推荐路线图

```
Phase 1: A-Hybrid 无 gather（最小改动）
  → torch.compile(default) + 手动双 Graph + hold intermediate
  → 2.3ms/step（2.9× 加速）
  → 验证地址隔离、D2H 并行、正确性
  → 预计工作量：1 周

Phase 2: 实现 gather_to_staging kernel
  → 无论后续走 A 还是 B，gather 都是必要的
  → 消除 D2H 碎片化（2.3ms → 1.5ms）
  → 预计工作量：1 周

Phase 3: 选择最终架构
  → 如果手动 Graph 管理可接受 → A-Hybrid + gather (1.7ms)
  → 如果需要 torch.compile 自动管理 → Design B (1.7ms)
  → 根据实际模型规模和显存约束决定

Phase 4: H2H 流水线化（Ring pageable buffer）
  → 与 Phase 1-3 独立，可并行开发
  → 消除 H2H 的 per-step malloc + D2H/H2H 流水线
```

---

## 附录：D2H 不能无限并行的原因

即使 Design A 的两张 graph 地址不重叠，copy stream 的 D2H 仍然是串行的（同一个 PCIe link）：

```
如果 D2H > forward：
  Compute: [fwd_α 1.6ms][fwd_β 1.6ms][fwd_α 1.6ms]
  Copy:             [D2H_α 2.3ms------>][D2H_β 2.3ms------>]

  Copy stream 每步落后 0.7ms → K 步后落后 0.7K ms
  → 要么降速等 copy，要么 buffer K 个 pending D2H
  → Step cadence = max(forward, D2H) = 2.3ms
```

这就是为什么 gather kernel（将 D2H 从 2.3ms 降到 1.5ms < 1.6ms forward）是突破 2.3ms 瓶颈的关键。
