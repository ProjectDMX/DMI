# Design C: Dual-Graph via torch.compile + Shadow Buffer 跨 Graph 保护

> 日期：2026-02-27
> 替代之前的 Design B (staging ring) 作为正式方案
> 基准模型：GPT-2 small (12L, fp32, batch=64, 183 hooks)
> 目标环境：vLLM + Llama 70B + B200

---

## 1. 动机

### 1.1 Design B (Staging Ring) 的根本问题

Design B 要求在 GPU 上预分配 staging ring buffer，大小 = 所有被监控 tensor 的总字节数。在 CUDA Graph 下 shape 固定（StaticCache），staging 大小 = 实际数据量，看似合理。

但在生产环境中：
- `max_seq_len` 可达 4096~131072
- Attention scores shape: `[batch, heads, 1, max_seq_len]`
- Llama 70B, batch=64, heads=64, max_seq_len=4096, fp16:
  单层 attn scores = `64 × 64 × 1 × 4096 × 2` = **32 MB**
- 80 layers × 32MB = **2.5 GB** staging buffer（仅 attention scores）

Staging buffer 是**额外**显存开销，叠加在模型本身的显存之上。且 staging buffer 按 max_seq_len 分配，即使实际 seq_len 很短也不会缩小。

### 1.2 Design A (手动双 Graph) 的移植性问题

Design A 手动管理两张 CUDA Graph（capture + pool + hold intermediate），不兼容 `torch.compile`。每换一个模型都需要重写 capture 逻辑。

### 1.3 本方案目标

1. Forward 与 D2H **完全异步**（D2H 隐藏在下一步 forward 背后）
2. **零额外 GPU buffer**（不需要 staging ring）
3. **保持 torch.compile 兼容**（`mode="reduce-overhead"` 自动管理 CUDA Graph）
4. 利用**已有的 shadow buffer + parse 机制**实现一次性地址发现，**sink() C++ 真实引用**实现录制阶段跨 graph 地址隔离

---

## 2. 核心机制

### 2.1 `if flag` → torch.compile 自动创建双 Graph

```python
@torch.compile(mode="reduce-overhead")
def compiled_forward(x, cache, pos, flag: int):
    if flag == 0:
        pass  # Dynamo guard: flag == 0
    else:
        pass  # Dynamo guard: flag == 1
    # 模型前向（两个分支代码相同，但 Dynamo 分别 trace）
    output = model(x, use_cache=True, past_key_values=cache, cache_position=pos)
    logits = lm_head(output.last_hidden_state)
    # hooks 在 forward 过程中已调用 record() + sink()
    return logits
```

**原理**：
- `flag` 是 Python int，Dynamo 对 int 输入做 **guard specialization**
- `flag=0` 和 `flag=1` 触发两次独立的 trace → 两个独立的 FX graph → 两个独立的 Inductor 编译
- `mode="reduce-overhead"` 对每个编译结果独立录制 CUDA Graph
- CUDAGraph Tree Manager 通过 `fn_cache[int_key]` 分发到对应的 graph recording

**结果**：同一个 `compiled_forward` 函数，自动维护两张 CUDA Graph（graph_A 和 graph_B），通过 `flag` 值切换。

### 2.2 双 Graph 地址隔离

两张 Graph 在同一个 CUDA memory pool 上录制，但获得**不重叠的地址集**。

**关键洞察**：地址隔离只需要在**录制阶段**保证，replay 阶段天然安全（地址已烧入 graph）。

```
Recording graph_A (flag=0):
  allocator 分配地址集 {A} 给 graph_A 的所有中间 tensor
  sink() 在 graph 内保活被监控 tensor → 地址集 {A_mon} ⊂ {A} 不会被层间复用
  sink() 的 C++ 实现同时将真实 at::Tensor 引用存入全局容器
    → 录制结束后，这些引用保持 allocator refcount > 0

Recording graph_B (flag=1):
  allocator 向 pool 请求显存
  → {A_mon} 的 refcount > 0（C++ 容器持有真实引用）
  → allocator 被迫在 pool 其他区域分配
  → graph_B 的被监控 tensor {B_mon} ⊂ {B}

保证：{A_mon} ∩ {B_mon} = ∅

录制完成后:
  清空 C++ 容器 → 释放引用
  地址已烧入两张 graph，不再需要动态保活
```

Replay 时地址已烧入 graph，不会改变。graph_A 只写 {A}，graph_B 只写 {B}。**无需任何运行时保护。**

### 2.3 两阶段保护模型

```
┌─ 录制阶段（Warmup）─────────────────────────────────────────────┐
│                                                                  │
│  Phase 1: Graph 内保护（sink）                                    │
│    sink(monitored_tensors) → dummy kernel 参数依赖                │
│    → graph 内地址不被层间复用                                      │
│    → C++ 侧同时将真实 at::Tensor push 到全局容器                   │
│                                                                  │
│  Phase 2: 跨 Graph 保护（C++ 真实引用）                            │
│    graph_A 录制结束 → compiled_forward 返回                       │
│    → Python 层无引用，但 C++ 全局容器持有真实 at::Tensor            │
│    → allocator 看到 refcount > 0 → 地址不释放                     │
│    → graph_B 录制时被迫分配不重叠的地址                             │
│                                                                  │
│  录制全部完成后: 清空 C++ 容器（地址已烧入，不需要了）                │
└──────────────────────────────────────────────────────────────────┘

┌─ Replay 阶段（稳态）────────────────────────────────────────────┐
│                                                                  │
│  天然安全，无需任何保护:                                            │
│    - 两张 graph 地址已烧入，replay 只写各自的固定地址集              │
│    - 无新 allocation → 不存在地址被抢占的风险                       │
│    - D2H 直接从已知的固定地址读取                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.4 Shadow Buffer 的角色（地址发现，非保护）

Shadow buffer 在本方案中的作用是**地址发现**，不是引用保护：

- record() 在 graph 内写 metadata（data_ptr, shape, stride, dtype）到 GPU flat buffer
- Warmup 后**一次性** parse → 得到每个 hook 的固定 data_ptr + tensor 形状
- 缓存结果，稳态 decode loop 直接使用（不再每步 parse）

**注意**：`from_blob(data_ptr, ..., noop_deleter)` 创建的 alias tensor 不会增加 CUDA caching allocator 的引用计数。它只是一个"幽灵 Tensor"——知道地址在哪，但 allocator 完全不认账。因此 from_blob **不能用于保护地址**，只用于告诉 D2H "去哪里拷贝数据"。

**为什么不直接 return 监控 tensor**：

| 方案 | 问题 |
|---|---|
| return 183 tensors | Dynamo 追踪 183 个额外返回值；Tree Manager track 全部；改变函数签名 |
| return list/tuple | Dynamo 对 dynamic list return 支持有限；graph break 风险 |
| **shadow buffer（一次性 parse）** | **graph 机制之外的侧通道；不影响 Dynamo/Inductor/Tree Manager** |

---

## 3. 架构

### 3.1 组件总览

```
保留（已有）：
  ├─ record() custom op        写 metadata 到 shadow buffer
  ├─ sink() custom op          dummy kernel，保活 tensor（graph 内）
  ├─ shadow buffer (GPU)       TensorMetadata[num_slots] × 2 帧
  ├─ shadow parser (C++)       parse_shadow_block() → alias_tensor()
  └─ alias_tensor()            from_blob() 零拷贝 tensor alias

新增：
  ├─ flag 参数 (int)           step % 2，驱动 Dynamo 双 graph specialization
  ├─ 双帧 shadow buffer        GPU buffer 容量 2 × num_slots
  ├─ monitor._current_frame    Python int (0/1)，hook 中计算 actual_slot
  └─ copy stream D2H 调度      graph 外异步 D2H

不需要：
  ├─ gather_to_staging kernel  不需要 D2D gather
  ├─ staging ring buffer       不需要额外 GPU buffer
  ├─ GPU frame_counter         用 Python int specialization 替代
  └─ torch.frombuffer()        用 alias_tensor(from_blob) 替代
```

### 3.2 Shadow Buffer 双帧复用

```
GPU shadow buffer 布局: [2 * num_slots, 128] uint8

  ┌─── Frame 0 (flag=0) ────────────┐
  │ slot 0:   TensorMetadata (128B)  │  ← graph_A 的 hook 0
  │ slot 1:   TensorMetadata (128B)  │  ← graph_A 的 hook 1
  │ ...                              │
  │ slot N-1: TensorMetadata (128B)  │  ← graph_A 的 hook N-1
  ├─── Frame 1 (flag=1) ────────────┤
  │ slot N:   TensorMetadata (128B)  │  ← graph_B 的 hook 0
  │ slot N+1: TensorMetadata (128B)  │  ← graph_B 的 hook 1
  │ ...                              │
  │ slot 2N-1: TensorMetadata (128B) │  ← graph_B 的 hook N-1
  └──────────────────────────────────┘

  GPT-2:  N=183, buffer = 366 × 128B = 46.8 KB
  Llama 70B: N≈1200, buffer = 2400 × 128B = 300 KB
```

**hook 中的 slot 计算**：

```python
def _make_compile_hook(self, slot_id: int):
    ops = self._ops
    gpu_buffer = self._gpu_buffer
    num_slots = self._num_slots
    monitor = self  # capture self for _current_frame access

    def hook(module, inputs, output):
        tensor = self._extract_tensor(output)
        if tensor is None or not tensor.is_cuda:
            return
        # Dynamo traces self._current_frame as Python int → guard specialization
        # graph_A: actual_slot = slot_id + 0 = slot_id (constant folded)
        # graph_B: actual_slot = slot_id + num_slots   (constant folded)
        actual_slot = slot_id + monitor._current_frame * num_slots
        ops.record(tensor, gpu_buffer, actual_slot)
        ops.sink([tensor])

    return hook
```

Dynamo 对 `monitor._current_frame` 做 guard：
- 第一次调用 `_current_frame=0` → trace 时 `actual_slot = slot_id + 0` → 常量折叠
- 第二次调用 `_current_frame=1` → guard 失败 → 重新 trace → `actual_slot = slot_id + num_slots`
- 后续调用：根据 `_current_frame` 值命中对应缓存的 graph

### 3.3 sink() 改动：录制时持有真实引用

record() 无需改动。sink() 需要在**录制阶段**将真实 `at::Tensor` 引用存入 C++ 全局容器：

```cpp
// C++ 侧：录制阶段的引用容器
static std::vector<at::Tensor> held_tensors_[2];  // [flag] 索引

void sink_op(const std::vector<at::Tensor>& tensors, int64_t flag) {
    if (tensors.empty()) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    for (const auto& t : tensors) {
        if (!t.defined()) continue;
        // GPU-side: dummy kernel 保活（已有逻辑）
        sink_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<uint64_t>(t.data_ptr()));

        // C++ 侧: 录制时持有真实引用（增加真正的 refcount）
        // 只在录制态（eager 或 capture）时生效
        // replay 时此 C++ 代码不执行（CUDA Graph 只回放 GPU kernel）
        held_tensors_[flag].push_back(t);
    }
}

// Warmup 完成后调用：释放所有持有的引用
void clear_held_tensors() {
    held_tensors_[0].clear();
    held_tensors_[1].clear();
}
```

**关键**：replay 时 CUDA Graph 只回放 GPU kernel（sink_kernel），不执行 C++ 宿主代码。所以 `held_tensors_` 只在录制阶段增长，replay 时不会无限膨胀。

其他改动：shadow buffer 分配大小 `2 * num_slots`（原来是 `num_slots`）。

### 3.4 Warmup 流程（双 Graph 录制 + 一次性地址发现）

```python
# --- Warmup: 触发 torch.compile 录制两张 Graph ---

# 录制 graph_A (flag=0)
# sink() 的 C++ 实现将真实 at::Tensor push 到 held_tensors_[0]
monitor.set_frame(0)
torch.compiler.cudagraph_mark_step_begin()
_ = compiled_forward(static_input, cache, pos, flag=0)
# 此时 held_tensors_[0] 持有 graph_A 所有被监控 tensor 的真实引用
# → allocator 的 refcount > 0 → 这些地址不会被释放

# 录制 graph_B (flag=1)
# allocator 看到 graph_A 的监控地址仍然 alive → 被迫分配新地址
monitor.set_frame(1)
torch.compiler.cudagraph_mark_step_begin()
_ = compiled_forward(static_input, cache, pos, flag=1)
# held_tensors_[1] 持有 graph_B 的引用（但 graph_B 的录制已经避开了 graph_A 的地址）

# --- 录制完成：释放 C++ 容器，地址已烧入 graph ---
torch.ops.graphmonitor_ops.clear_held_tensors()

# --- 一次性 parse shadow buffer：发现固定的 D2H 地址 ---
torch.cuda.current_stream().synchronize()
cached_refs = [
    monitor.parse_shadow_buffer(frame=0),  # graph_A 的 data_ptrs + shapes
    monitor.parse_shadow_buffer(frame=1),  # graph_B 的 data_ptrs + shapes
]
# 这些 from_blob alias 只用于"知道去哪里 D2H"，不提供任何引用保护
# 验证地址隔离（可选）
# assert set(ptrs_A) & set(ptrs_B) == set()
```

### 3.5 稳态 Decode Loop

**核心简化**：CUDA Graph 下 shadow buffer 内容每步不变（replay 只重放同样的 kernel 写同样的值）。地址在录制时已固定且隔离。稳态无需 sync，无需 parse，无需动态保护。

```python
copy_stream = torch.cuda.Stream()
d2h_events = [torch.cuda.Event(), torch.cuda.Event()]
pinned = [pre_alloc_pinned(0), pre_alloc_pinned(1)]

for step in range(num_steps):
    flag = step % 2

    # ---- 1. 等待同一帧上一轮的 D2H 完成 ----
    if step >= 2:
        d2h_events[flag].synchronize()   # 只等 D2H，不等 forward
        consume(pinned[flag])

    # ---- 2. Forward（自动选择 graph_A 或 graph_B）----
    monitor.set_frame(flag)
    torch.compiler.cudagraph_mark_step_begin()
    logits = compiled_forward(token, cache, cache_position, flag)

    # ---- 3. 异步 D2H on copy stream ----
    # 使用 warmup 时缓存的 cached_refs[flag]（固定地址）
    copy_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(copy_stream):
        d2h_to_pinned(cached_refs[flag], pinned[flag])
        d2h_events[flag].record(copy_stream)

    # ---- 4. 下一步可立即开始 forward（不等 D2H）----
    # graph_{1-flag} 使用不同地址集（录制时保证），不会覆盖 flag 的数据
```

**关键不变量**：
- 两张 graph 的被监控地址集在**录制时**已保证不重叠（{A_mon} ∩ {B_mon} = ∅）
- Replay 时地址已烧入，不存在动态分配/释放 → 天然安全
- D2H 读取固定地址上的数据，graph_{1-flag} 的 replay 不会碰这些地址
- `d2h_events[flag].synchronize()` 只在同一帧被复用时才等待（间隔 2 步）

---

## 4. 时序分析

### 4.1 GPT-2 small (18MB, 183 hooks, PCIe 3.0)

各阶段延迟：
| 阶段 | 延迟 | 说明 |
|---|---|---|
| Forward (graph replay) | 1.6ms | torch.compile reduce-overhead |
| Tensor D2H (183 × scattered async) | 2.3ms | 183 × cudaMemcpyAsync on copy stream |
| Shadow parse (warmup only) | ~0.4ms | 一次性执行，不在稳态关键路径上 |

**关键简化**：Shadow buffer 在 CUDA Graph replay 下内容不变（每步 replay 写入相同的地址/形状元数据）。因此 shadow parse 在 warmup 阶段执行**一次**，缓存结果（`cached_refs`），稳态 decode loop 直接使用固定地址。

**Pipeline 时序**：

稳态中只有 forward 和 D2H 两个操作在两条 stream 上交替：

```
Compute stream:   [fwd_A 1.6ms]             [fwd_B 1.6ms]             [fwd_A 1.6ms]
                   t=0→1.6                    t=3.9→5.5                 t=6.2→7.8

Copy stream:                  [D2H_A 2.3ms]              [D2H_B 2.3ms]
                               t=1.6→3.9                   t=5.5→7.8

CPU sync points:  step 2 需等待 D2H_A(step 0) 完成 → compute idle 3.2→3.9 (0.7ms stall)
```

**分析**：
- Copy stream 每步 2.3ms（D2H），compute stream 每步 1.6ms（forward）
- D2H > forward → copy stream 是瓶颈 → compute stream 每步有 0.7ms 空闲
- 同一帧复用间隔 = 2 × step_cadence ≥ D2H → step_cadence ≥ D2H

**Step cadence = max(forward, D2H) = max(1.6ms, 2.3ms) = 2.3ms**

vs 当前串行 6.7ms → **2.9× 加速**

### 4.2 GPT-2 + Gather D2H（消除 scattered overhead）

Scattered D2H 的 2.3ms 中 1.8ms 是 CPU launch overhead（183 × ~10μs）。用 graph 外 gather kernel 合并为单次 DMA：

```
Gather D2D: ~0.1ms (183 scattered → 1 contiguous buffer)
单次 DMA:  ~1.5ms (18MB / 12 GB/s)
总 D2H:   ~1.6ms
```

此时 D2H ≈ forward → 两条 stream 完美平衡：

```
Compute stream:   [fwd_A 1.6ms][fwd_B 1.6ms][fwd_A 1.6ms][fwd_B 1.6ms]
                   t=0→1.6      t=1.6→3.2    t=3.2→4.8    t=4.8→6.4

Copy stream:                   [D2H_A 1.6ms][D2H_B 1.6ms][D2H_A 1.6ms]
                                t=1.6→3.2    t=3.2→4.8    t=4.8→6.4
```

**Step cadence = max(1.6ms, 1.6ms) = 1.6ms → 4.2× 加速**

瓶颈转移到纯 PCIe 带宽 — 这是设计目标。

### 4.3 Llama 70B + B200 (PCIe 5.0, ~64 GB/s)

假设监控全部 ~1200 hooks（所有层全部 hook type）：

| 项目 | 值 | 说明 |
|---|---|---|
| Forward | ~15ms | 80 layers, tensor parallel |
| 被监控数据量 | ~2-3 GB | 取决于 max_seq_len |
| Tensor D2H (scattered) | ~40-50ms | 2.5 GB / 64 GB/s ≈ 39ms + launch overhead |

```
Step cadence = max(forward, D2H) = max(15ms, ~45ms) = ~45ms

vs 串行: 15 + 45 = ~60ms
加速: 60 / 45 = 1.33×
```

**瓶颈纯粹在 PCIe 带宽上** — 这正是设计目标。forward 完全隐藏在 D2H 背后，step cadence 由 `数据量 / PCIe带宽` 决定。所有 CPU 开销（shadow parse、tensor 构建等）都在 warmup 阶段或 D2H 等待间隙中完成。

---

## 5. 显存分析

### 5.1 双 Graph 的显存开销

双 Graph 要求被监控 tensor（sink 保活的）在两张 graph 中不重叠。这意味着 GPU memory pool 需要同时容纳两份被监控 tensor 的地址空间。

**关键区分**：sink() 保活的是**被监控 tensor**（非全部 intermediate）。未被监控的中间 tensor 正常参与层间复用。

```
单 Graph 正常执行:
  allocator 复用 intermediate → peak ≈ max(per_layer_intermediate) × small_factor

单 Graph + sink 保活:
  被监控 tensor 不可复用 → peak ≈ normal_peak + sum(monitored_tensors)

双 Graph + sink 保活:
  两份被监控 tensor 不重叠 → peak ≈ normal_peak + 2 × sum(monitored_tensors)
  额外显存 = sum(monitored_tensors)  (相对单 graph + sink)
```

| 模型 | sum(monitored) | 额外显存 (vs 单 graph) |
|---|---|---|
| GPT-2 (12L, fp32, b=64, 183 hooks) | 18 MB | **18 MB** |
| Llama 70B (80L, fp16, b=64, seq=4096, ~1200 hooks) | ~2.5 GB | **~2.5 GB** |

**对比 Design B staging ring (K=2)**：

| 方案 | GPT-2 额外显存 | Llama 70B 额外显存 |
|---|---|---|
| Design B (staging ×2) | 36 MB | ~5 GB |
| **Design C (双 graph)** | **18 MB** | **~2.5 GB** |

Design C 额外显存 = Design B 的一半（staging ×2 = 2 份数据，双 graph 额外 = 1 份）。

### 5.2 Shadow Buffer 显存

| 模型 | Shadow buffer (双帧) |
|---|---|
| GPT-2 (183 hooks) | 46.8 KB |
| Llama 70B (1200 hooks) | 300 KB |

可忽略。

### 5.3 Pinned Host Memory

每帧需要一个 pinned host buffer 作为 D2H 目标。

| 方案 | 说明 |
|---|---|
| Per-tensor pinned pool | 已有实现，183 个小 block，pool 复用 |
| Pre-allocated pinned buffer | 一次性 cudaHostAlloc，按 offset 写入 |

推荐：保留已有的 pinned pool 机制，后续可优化为预分配。

---

## 6. D2H 优化路径

Design C 的 baseline D2H 是 **scattered**（183 × cudaMemcpyAsync），有 CPU overhead。以下是逐步优化路径：

### 6.1 Baseline: Scattered D2H（无新 kernel）

```
183 × cudaMemcpyAsync，每次 ~100KB
CPU overhead: 183 × ~10μs = 1.8ms
纯带宽: 18MB / 12 GB/s = 1.5ms
总计: ~2.3ms (GPT-2, PCIe 3.0)
```

Step cadence: max(1.6, 2.3) = **2.3ms**（见 4.1 节）。D2H 是瓶颈，compute stream 每步空闲 0.7ms。无需新 kernel，最小实现成本。

### 6.2 Graph 外 Gather + 单次 DMA

Warmup 时一次性 shadow parse 后，已知所有 data_ptr 和 size（固定不变）。可在 graph 外启动 gather kernel：

```python
# warmup 时缓存的 layout（固定地址和 offset）
data_ptrs, sizes, offsets = monitor.get_monitored_layout(frame=flag)

# graph 外启动 gather kernel（非 CUDA Graph，普通 kernel launch）
with torch.cuda.stream(copy_stream):
    gpu_gather(data_ptrs, gather_buffer, offsets, sizes)  # D2D ~0.1ms
    pinned_buffer.copy_(gather_buffer, non_blocking=True)  # D2H ~1.5ms
```

| 项目 | Scattered | Gathered |
|---|---|---|
| D2H 延迟 | 2.3ms | 1.6ms (gather 0.1ms + DMA 1.5ms) |
| 需要额外 GPU buffer | 否 | 是（gather buffer ~18MB） |
| 需要新 kernel | 否 | 是（gather kernel，graph 外） |
| Step cadence (GPT-2) | **2.3ms** | **1.6ms** |

Gather 后 D2H ≈ forward → 两条 stream 完美平衡，消除 compute idle。

Gather buffer 按**实际数据大小**分配（一次性，在 warmup 后已知 shape），不是按 max_seq_len。因为 CUDA Graph 下 shape 固定，实际 = captured shape。

---

## 7. 方案对比

### 7.1 延迟

| 方案 | Step Cadence (GPT-2) | Step Cadence (Llama 70B 全量) | 需要新 kernel |
|---|---|---|---|
| 当前（串行 sync） | 6.7ms | ~60ms | — |
| Design A (手动双 graph) | 2.3ms | ~45ms | 否 |
| Design B (staging ring) | 1.7ms | ~45ms | gather kernel |
| **Design C (torch.compile 双 graph)** | **2.3ms** | **~45ms** | **否** |
| Design C + gather | **1.6ms** | ~45ms | gather kernel (graph 外) |

GPT-2 上 Design C + gather (1.6ms) 接近 Design B (1.7ms)。大模型上所有方案都被 D2H 限制（~45ms），差异消失。

### 7.2 显存

| 方案 | GPT-2 额外 | Llama 70B 额外 |
|---|---|---|
| Design A (hold ALL intermediate) | 86 MB | ~5 GB |
| Design B (staging ring ×2) | 36 MB | ~5 GB |
| **Design C (双 graph monitored)** | **18 MB** | **~2.5 GB** |

Design C 显存最优：只为**被监控 tensor** 额外分配一份（vs Design B 的两份）。

### 7.3 实现复杂度

| 方面 | Design A | Design B | **Design C** |
|---|---|---|---|
| torch.compile 兼容 | 不兼容 | 兼容 | **兼容** |
| CUDA Graph 管理 | 手动 capture + pool | 自动 | **自动** |
| 新 custom op | 无 | gather_to_staging | **无** |
| Shadow buffer | 需要 | 不需要 (被 staging 替代) | **需要（已有）** |
| 模型移植成本 | 每模型重写 | 改一行 torch.compile | **改一行 torch.compile** |
| 帧管理 | 手动 α/β 切换 | Ring + GPU frame_counter | **flag 参数** |

### 7.4 选择建议

| 场景 | 推荐 |
|---|---|
| 快速出成果，不写新 kernel | **Design C** (2.3ms，仅需改 Python 层 + sink C++ 小改) |
| GPT-2 级别小模型，追求极致 latency | **Design C + gather** (1.6ms，接近 Design B 的 1.7ms) |
| 大模型生产环境 (vLLM + Llama 70B) | **Design C** (所有方案都被 D2H 限制，Design C 实现最简) |
| 需要手动 CUDA Graph 控制 | Design A |

---

## 8. 风险与待验证

### 8.1 Dynamo Guard Specialization 行为

**风险**：Dynamo 可能对 `if flag == 0: pass else: pass`（空分支）做优化，合并两条路径。

**缓解**：在分支中包含有实际效果的代码（如 `monitor._current_frame` 访问导致的 guard 差异）。最保险的做法：

```python
if flag == 0:
    torch.ops.graphmonitor_ops.sink([])  # 空 sink，仅为区分 trace
else:
    torch.ops.graphmonitor_ops.sink([])
```

或者完全依赖 `monitor._current_frame` 的 guard 差异（hook 中访问 `_current_frame` 已足够触发不同 guard）。

**验证方法**：`torch._dynamo.explain(compiled_forward, ...)` 检查是否生成两个独立的 guard set。

### 8.2 CUDAGraph Tree Manager Replay 时的地址安全

**风险**：Tree Manager 在 graph 切换时是否可能"回收" pool 区域，导致 D2H 读到被覆盖的数据？

**分析**：Replay 阶段不存在此风险。原因：
1. Tree Manager 在 replay 时不做新分配/释放——所有地址在录制时已烧入 graph
2. 两张 graph 轮流 replay，各自只写自己录制时的固定地址集
3. D2H 在 copy stream 上读取上一帧的地址，此时该帧的 graph 不在执行（间隔 1 步）

**录制时的地址隔离已在 2.2 节保证**：sink() 的 C++ 容器持有真实引用 → allocator refcount > 0 → graph_B 录制被迫分配不重叠地址。录制完成后清空容器（地址已烧入）。

**验证方法**：在 PoC 中对比两张 graph 录制时的 data_ptr 集合，确认无交集。在稳态 decode 中验证 D2H 数据正确性（与 eager mode 对比）。

### 8.3 from_blob Alias 的角色界定

**说明**：`from_blob(data_ptr, ..., noop_deleter)` 创建的 tensor 是"幽灵 Tensor"——不参与 CUDA caching allocator 引用追踪。这是**设计预期**，不是风险。

本方案中 from_blob alias 仅用于**地址发现**（告诉 D2H "去哪里拷贝"），不用于引用保护。地址保护完全由录制阶段的 C++ 真实引用（2.2 节）和 replay 阶段的天然安全性（8.2 节）保证。

**潜在问题**：如果 Tree Manager 在某些异常路径（如 OOM recovery、graph 重新录制）回收 pool 区域，from_blob alias 指向的地址可能失效。

**缓解**：在正常 decode loop 中不触发这些异常路径。如需更健壮，可在 D2H 前校验地址有效性（一次 `cudaPointerGetAttributes` 调用，~1μs）。

### 8.4 Hook 闭包中 `self._current_frame` 的 Dynamo 兼容性

**风险**：Dynamo 可能对 GraphMonitor 对象（非 nn.Module）的属性访问有限制。

**缓解**：
1. 将 `_current_frame` 设为 nn.Module 的 buffer 或 Python int attr
2. 或者直接将 frame 信息编码到 buffer tensor 对象中（传不同 buffer 对象）
3. 最坏情况：手动维护两套 hook（frame=0 和 frame=1 各一套），注册/反注册

**验证方法**：PoC 中实际 trace 并检查 graph break。

---

## 9. 实施路线

```
Phase 1: PoC 验证（3-5 天）
  ├─ 验证 Dynamo int specialization → 两张独立 CUDA Graph
  ├─ 验证 hook 中 _current_frame guard 行为
  ├─ 验证双帧 shadow buffer 录制/解析正确性
  ├─ 验证 sink() C++ held_tensors_ → graph_B 录制地址隔离 → clear → 一次性 parse
  ├─ 验证稳态 decode D2H 数据正确性（vs eager baseline，多步后无数据损坏）
  └─ 用 GPT-2 small 跑端到端 benchmark

Phase 2: 集成到 GraphMonitor（1 周）
  ├─ GraphMonitor.__init__: 分配双帧 shadow buffer
  ├─ GraphMonitor._current_frame: frame 管理
  ├─ GraphMonitor._make_compile_hook: actual_slot 计算
  ├─ GraphMonitor.parse_shadow_buffer(frame): 按帧解析
  ├─ GraphMonitor.warmup_dual_graph(): 封装 warmup 流程
  └─ 单元测试: 地址隔离、D2H 正确性

Phase 3: Decode Runner 集成（1 周）
  ├─ TorchCompileDecodeRunner: 稳态 decode loop
  ├─ copy_stream 调度 + refs 生命周期管理
  ├─ consumer 集成（delegate / callback）
  ├─ benchmark CLI: --dual-graph flag
  └─ 端到端 benchmark: step cadence, throughput

Phase 4: D2H 优化（可选，1 周）
  ├─ Graph 外 gather kernel（消除 scattered D2H 的 CPU overhead，2.3ms → 1.6ms）
  ├─ Pre-allocated pinned host buffer（消除 per-tensor malloc）
  └─ 验证 step cadence = max(forward, D2H) ≈ 1.6ms（两条 stream 完美平衡）
```

---

## 附录 A: 为什么地址保护只需要在录制阶段

**问题**：双 Graph 方案中，graph_A 和 graph_B 的被监控 tensor 地址必须不重叠。如何保证？

**录制阶段（危险）**：

```
Recording graph_A:
  allocator 分配地址集 {A} → sink() 的 dummy kernel 保活（graph 内）
  compiled_forward() 返回 → Python 层丢失所有中间 tensor 引用
  from_blob() alias？→ noop_deleter → allocator 不认 → 不增加 refcount     ✗

Recording graph_B（如果 {A_mon} 没有保护）:
  allocator 看到 {A_mon} 的 refcount = 0 → "可用"
  graph_B 分配到 {A_mon} 相同的地址                                         ✗
  → 两张 graph 的被监控 tensor 地址重叠 → D2H 数据互相覆盖
```

**解决方案**：sink() 的 C++ 实现在录制时将真实 `at::Tensor` 引用 push 到全局容器（`held_tensors_[flag]`）。真实 at::Tensor 共享 Storage → allocator 看到 refcount > 0 → 地址不释放。graph_B 录制时被迫分配不重叠地址。录制完成后清空容器（地址已烧入 graph）。

**Replay 阶段（天然安全）**：

```
Replay graph_A:
  CUDA Graph 回放录制时的 kernel sequence → 写入烧入的固定地址集 {A}
  不做新 allocation → 不存在地址被抢占的风险

Replay graph_B:
  写入固定地址集 {B}，{A} ∩ {B} = ∅（录制时已保证）
  → 两张 graph 互不干扰，D2H 读取上一帧数据始终安全
```

**结论**：保护是一次性的（录制阶段 C++ 持有真实引用），不是持续性的。Replay 阶段无需任何动态保护机制。

## 附录 B: 与 Design B 的兼容演化路径

如果未来需要 Design B 的 gather 性能（1.7ms），可以在 Design C 基础上增量添加：

1. 保留 dual graph + shadow buffer（跨 graph 保护）
2. 在 graph 外（shadow parse 之后、D2H 之前）插入 gather kernel
3. Gather buffer 按实际大小一次性分配（不是 staging ring，不需要帧管理）
4. 单次 DMA 替代 scattered D2H

这样 Design C 是 Design B 的子集，可以渐进式升级。
