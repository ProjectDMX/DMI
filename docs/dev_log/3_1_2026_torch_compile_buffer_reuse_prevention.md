# torch.compile Buffer Reuse 防护机制

**Date**: 2026-03-01
**Status**: 已实现，24/24 测试通过
**Depends on**: Design C dual_compile (2_27_2026_dual_graph_shadow_buffer_design.md)

## 1. 问题背景

我们的 monitoring 系统通过 `record()` op 在 forward 中逐 hook 记录每个中间 tensor 的 `data_ptr`，然后通过 `alias_tensor()` 用 `from_blob` 在固定地址上构造 ghost tensor 做 D2H。这要求：**从 record 写入 data_ptr 到 D2H 读取数据的整个生命周期内，该地址的内容不能被覆盖。**

但 `torch.compile` + CUDA Graph 的多层优化会积极地复用 buffer 内存，导致 record 记录的地址在 forward 结束前就被分配给了其他 tensor，D2H 读到的是错误数据。

## 2. torch.compile 编译管线

```
Python forward code
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: TorchDynamo (Graph Tracing)           │
│  ─ 追踪 Python 代码 → FX Graph (symbolic)       │
│  ─ 识别 graph breaks, guards                    │
│  ─ 不做内存决策                                  │
└─────────────────────────────────────────────────┘
        │ FX Graph
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: AOTAutograd + Functionalization       │
│  ─ 分解高级 op → 原始 ATen op                    │
│  ─ Functionalization: in-place op → functional  │
│    (x.add_(y) → x = x.add(y))                  │
│  ─ 生成前向+反向图                               │
│  ─ 引入更多中间 tensor (functional 副本)          │
└─────────────────────────────────────────────────┘
        │ Functionalized FX Graph
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: TorchInductor                         │
│                                                 │
│  3a. Lowering (IR 构建)                          │
│      ─ FX op → Inductor IR node                 │
│        (Pointwise, Reduction, FallbackKernel)   │
│      ─ 注册 custom lowering (register_lowering) │
│      ─ 构建 buffer 对象 + 数据依赖图              │
│      ─ 标记 never_reuse_buffers                 │
│                                                 │
│  3b. Scheduling (调度 + 融合)                    │
│      ─ 排序 IR node, 计算 read/write 依赖        │
│      ─ 融合兼容 op → FusedSchedulerNode          │
│      ─ 决定 inplace reuse (Path 1)              │
│                                                 │
│  3c. Memory Planning (内存规划)                   │
│      ─ 计算 buffer live range [alloc, last_use]  │
│      ─ 不重叠的 buffer 共享内存 (Path 2)          │
│      ─ 生成 alloc/free 代码                      │
│                                                 │
│  3d. Code Generation (代码生成)                   │
│      ─ 输出 Python/Triton 代码                   │
│      ─ 包含 torch.empty(), del, kernel calls     │
└─────────────────────────────────────────────────┘
        │ compiled Python function
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 4: CUDA Graph Capture + Replay           │
│  ─ 首次执行: capture (私有内存池分配)             │
│  ─ 后续执行: replay (固定地址序列, 无 malloc)     │
│  ─ 私有池内 caching allocator 可回收 freed 块    │
└─────────────────────────────────────────────────┘
```

## 3. 四种 Buffer 复用机制

### 3.1 Rematerialization（重算替代存储）

**发生阶段**: Inductor Lowering / Scheduling

**机制**: Inductor 判断某个中间结果不值得存储（内存开销 > 重算开销），直接丢掉 buffer，需要时重新计算。被 rematerialize 的 tensor 根本没有 buffer，自然没有稳定地址。

**典型场景**: 融合 kernel 内部的中间值（如 `a + b` 的结果直接喂给下一个 op，不单独分配 buffer）。

**关键代码路径**:
- `torch/_inductor/scheduler.py`: 融合决策
- `torch/_inductor/ir.py`: buffer 是否 `realize()` 决定是否物化

### 3.2 Path 1: Inplace Reuse（就地复用）

**发生阶段**: Inductor Scheduling (`decide_inplace_update()`)

**机制**: 当一个 buffer 只剩一个 downstream consumer，且 consumer 的输出和该 buffer size/stride 匹配时，Inductor 直接让输出复用输入的内存地址。

**生成代码**:
```python
buf5 = buf0  # inplace reuse, no new allocation
del buf0     # 只是 alias，不影响实际内存
```

**决策条件** (`scheduler.py:471-600`):
1. `config.inplace_buffers` 启用
2. `can_reuse(input_buf)` 返回 True
3. input_buf 只有一个 remaining user
4. 该 user 标记 `can_inplace=True`
5. input/output size 匹配

**关键函数**: `wrapper.py` 中的 `can_reuse()`:
```python
def can_reuse(self, input_buffer, output_buffer=None):
    name = input_buffer.get_name()
    return not (
        name in V.graph.removed_buffers
        or name in V.graph.graph_inputs       # 除非 donated
        or name in V.graph.constants
        or name in V.graph.torchbind_constants
        or name in V.graph.never_reuse_buffers # ← 我们的 defense
        or name in self.freed
    )
```

### 3.3 Path 2: FreeIfNotReused（释放后被其他 buffer 占用）

**发生阶段**: Inductor Memory Planning

**机制**: Inductor 计算每个 buffer 的 live range `[alloc_time, last_use_time]`。两个 live range 不重叠的 buffer 可以共享同一块内存（类似寄存器分配）。

**生成代码**:
```python
buf0 = empty(...)     # 分配给 tensor A
# ... 使用 buf0 ...
del buf0              # tensor A 生命周期结束
# ... 后续 ...
buf5 = empty(...)     # 分配给 tensor B — 可能拿到 buf0 的地址！
```

**关键数据结构** (`memory_planning.py`):
```python
@dataclass
class LiveRange:
    begin: float  # 分配时间
    end: float    # 最后使用时间

class TemporalSplit:
    """时间上不重叠的 buffer 可以共享内存"""
    def _allocate(self, block):
        overlapping = [s for s in self.allocations
                       if s.live_range.overlaps(block.live_range)]
        if len(overlapping) == 0:  # 不重叠 → 复用！
            self.allocations.append(block)
            return True
```

注意：Path 2 的 `can_reuse()` 检查和 Path 1 **共享同一个函数**，`never_reuse_buffers` 同时阻断两条路径。

### 3.4 CUDA Graph Caching Allocator 回收

**发生阶段**: CUDA Graph Capture / Replay（运行时）

**机制**: CUDA Graph capture 使用**私有内存池**。Inductor 生成的 `del buf0` 在 capture 时执行，将内存归还私有池。同一次 capture 中后续的 `torch.empty()` 可能从池中拿到同一个地址。Replay 时严格重放相同的分配/释放序列，地址复用被"固化"。

**关键区别**: 这是**运行时**行为，发生在 Inductor 编译完成之后。即使 Inductor 不做任何 buffer 复用优化，caching allocator 仍然可以独立地回收和重新分配内存。

**代码路径**: `torch/_inductor/cudagraph_trees.py`, CUDA runtime `cudaMalloc`/`cudaFree`

## 4. 我们的四层防护

### 4.1 防 Rematerialization: `tensor.realize()`

**位置**: `monitoring/graph_ops.py` — custom lowering

```python
@register_lowering(torch.ops.graphmonitor_ops.record.default, ...)
def record_lowering(tensor, buffer, slot_id):
    tensor.realize()   # ← 强制物化为 named buffer
    buffer.realize()
    ...
```

**原理**: 在 Inductor IR 中，tensor 默认是 lazy 的（一个计算表达式）。`realize()` 强制将其物化为一个 concrete named buffer（如 `buf0`），写入内存。Inductor 不会对已经 realize 的 buffer 做 rematerialization。

**没有 `realize()` 会怎样**: tensor 可能只是一个 `Pointwise` node，Inductor 可以选择在需要时重算而不分配 buffer。那 `never_reuse_buffers.add(name)` 连 name 都拿不到。

### 4.2 防 Path 1 + 防 DCE: `Tensor(a!)` schema 注解

**位置**: `monitoring/csrc/graph_monitor_ops.cu` — op schema

```cpp
m.def("record(Tensor tensor, Tensor(a!) buffer, int slot_id) -> ()");
//                           ^^^^^^^^^^^
//                     buffer 参数标记为 mutated
```

**原理**:
- `Tensor(a!)` 告诉 Functionalization / Inductor 这个 op 会 mutate buffer
- **防 Path 1**: `decide_inplace_update()` 不会把 buffer 的内存分配给其他 tensor（因为 buffer 被 mutate，不能被覆盖）
- **防 DCE**: Inductor 不会删除有 side effect（mutation）的 op。没有这个注解，`record()` 作为 void op 会被 Inductor 当作 dead code 消除

**注意**: 这里保护的是 **buffer**（shadow metadata buffer），不是 input tensor。Input tensor 的保护靠 4.3。

### 4.3 防 Path 1 + Path 2: `never_reuse_buffers`

**位置**: `monitoring/graph_ops.py` — custom lowering

```python
def record_lowering(tensor, buffer, slot_id):
    tensor.realize()
    buffer.realize()
    V.graph.never_reuse_buffers.add(tensor.data.get_name())  # ← 关键
    ir.FallbackKernel.create(
        torch.ops.graphmonitor_ops.record.default,
        tensor, buffer, slot_id,
    )
    return ()
```

**原理**: 将 input tensor 的 buffer name 加入 `never_reuse_buffers` 集合。`can_reuse()` 对该 buffer 返回 False，Inductor 的 Path 1（inplace reuse）和 Path 2（temporal reuse）都不会复用该 buffer 的内存。

**实测验证**: 没有 `never_reuse_buffers` 时（即使有 `Tensor(a!)` + anchor），生成代码仍出现 `buf11 = buf0; del buf0  # reuse`，correctness test FAIL。加上后 PASS。

**参考**: PyTorch 自身的 `resize_storage_bytes_` lowering 使用了相同模式。

### 4.4 防 Caching Allocator 回收: `anchor` op

**位置**:
- C++ op: `monitoring/csrc/graph_monitor_ops.cu`
- 调用点: `modeling_gpt2.py` GPT2Model.forward 末尾

```cpp
// C++ 端
void anchor_op(at::Tensor& buffer, const std::vector<at::Tensor>& tensors) {
    for (const auto& tensor : tensors) {
        sink_kernel<<<1,1,0,stream>>>(
            reinterpret_cast<uint64_t>(tensor.data_ptr()));
    }
}
```

```python
# Python 端 (forward 末尾)
if _anch:
    torch.ops.graphmonitor_ops.anchor(_mon, _anch)
```

**原理**:
- `_anch` 列表在 forward 开始时创建，每个 `_mon_record` 调用将 tensor append 进去
- Forward 末尾调用 `anchor(_mon, _anch)`，对所有 tensor 执行 `sink_kernel`
- `sink_kernel` 读取每个 tensor 的 `data_ptr` → 在 CUDA Graph 中建立 GPU 级数据依赖
- CUDA Graph capture 记录了这些读取 → replay 时所有地址必须有效 → caching allocator 不会在 graph 内部回收这些地址
- schema 中 `Tensor(a!) buffer` 防止 Inductor DCE 掉这个 op

**为什么 never_reuse_buffers 不够**: `never_reuse_buffers` 阻止 Inductor **显式地**把 buf0 的内存分配给 buf5。但 Inductor 仍会在 tensor 最后一个 consumer 之后生成 `del buf0`。Caching allocator 收到 free 后把内存放回私有池，后续 `torch.empty()` **隐式地**从池中拿到同一个地址。这不受 `never_reuse_buffers` 控制。

## 5. 总结

| # | 机制 | 位置 | 阶段 | 防护目标 | 缺一不可？ |
|---|------|------|------|---------|-----------|
| 1 | `tensor.realize()` | `graph_ops.py` lowering | Inductor Lowering | 禁 rematerialization | ✅ 没有它 → 无 buffer name → 4.3 失效 |
| 2 | `Tensor(a!)` on buffer | `graph_monitor_ops.cu` schema | Functionalization + Inductor | 禁 buffer 被 inplace 覆盖 + 防 DCE | ✅ 没有它 → record 被优化掉 |
| 3 | `never_reuse_buffers` | `graph_ops.py` lowering | Inductor Memory Planning | 禁 input tensor 被 Path 1/2 复用 | ✅ 实测: 没有它 → FAIL |
| 4 | `anchor` op | `graph_monitor_ops.cu` + `modeling_gpt2.py` | CUDA Graph Capture/Replay | 禁 caching allocator 回收 | ✅ 理论需要 (未单独验证) |

**依赖关系**: 1 → 3（realize 产生 name，name 才能加入 never_reuse_buffers），2 独立，4 独立。

```
torch.compile 编译期                           CUDA Graph 运行期
┌─────────────────────────────┐              ┌──────────────────────┐
│  realize() ──→ named buffer │              │                      │
│       │                     │              │  anchor() GPU 读取    │
│       ▼                     │              │  所有 tensor data_ptr │
│  never_reuse_buffers        │              │       │               │
│  (阻断 Path 1 + Path 2)    │              │       ▼               │
│       │                     │              │  CUDA Graph 记录依赖  │
│       ▼                     │              │  → 内存不可回收       │
│  Tensor(a!) on buffer       │              │                      │
│  (阻断 DCE + inplace)      │              └──────────────────────┘
└─────────────────────────────┘
```

## 6. Monitoring 对 Kernel Fusion 的影响

### 6.1 问题：`realize()` 打断 Fusion

Inductor 的核心优化之一是 **kernel fusion**：将多个连续 op 融合为一个 Triton kernel，中间结果只在 registers/shared memory 中存在，不写回 HBM。

例如 GPT2Block 中：
```python
hidden_states = self.ln_1(hidden_states)   # LayerNorm
attn_input = hidden_states                 # 直接传给 attention
q, k, v = self.c_attn(attn_input)          # QKV projection
```

**不 monitor 时**，Inductor 可能融合 LayerNorm + QKV projection：
```
fused_kernel:
    ln_out = layer_norm(x)     ← register 内，不写 HBM
    q = ln_out @ W_q           ← 直接消费
    k = ln_out @ W_k
    v = ln_out @ W_v
    → 写回 q, k, v 到 HBM
```
`ln_out` 从未存在于 HBM，省去一次 HBM write + 后续 read。

**monitor 时**，`record()` 的 custom lowering 调用 `tensor.realize()`，强制 `ln_out` 物化：
```
kernel_1:  layer_norm
    buf0 = layer_norm(x)       ← 写入 HBM (realize 强制)
    record(buf0, shadow, slot) ← 记录 data_ptr

kernel_2:  qkv_projection
    q = buf0 @ W_q             ← 从 HBM 读回 buf0
    k = buf0 @ W_k
    v = buf0 @ W_v
```

每个被 monitor 的 tensor 都会被强制物化到 HBM，打断原本可能的 fusion。

### 6.2 性能代价

每个 hook 点的 `realize()` 引入：
- **1 次额外 HBM write**: fused kernel 被拆成两个，中间结果必须写回
- **1 次额外 HBM read**: 下游 kernel 必须从 HBM 重新读取
- **1 个额外 kernel launch**: fusion 被打断 → 更多独立 kernel
- **1 次 `record_metadata_kernel` launch**: 1 thread, 128B 写入

对于 GPT-2 (12 层, 每层 ~15 hook points, 共 183 hooks)：最多 183 次 fusion break。

### 6.3 Selective Monitoring 的双重意义

Selective monitoring (183 → 12 hooks) 不只是减少 D2H 数据量：

| 优化维度 | 183 hooks | 12 hooks | 说明 |
|---------|-----------|----------|------|
| D2H 数据量 | 183 × tensor_size | 12 × tensor_size | 减少 DMA 传输 |
| Fusion breaks | 最多 183 | 最多 12 | **减少 HBM 往返 + kernel launch** |
| record kernels | 183 | 12 | 减少 GPU kernel 调度 |
| anchor sink kernels | 183 | 12 | 减少 forward 末尾开销 |

Fusion break 的影响在大模型上尤其显著：HBM bandwidth 是 LLM inference 的主要瓶颈（memory-bound），每次额外的 HBM 读写都直接拖慢 forward。

### 6.4 Record Elimination 不能恢复 Fusion

注意：Phase 2 的 record elimination（`disable_record()` → `_mon_buf=None` → Dynamo re-trace）虽然消除了 record kernel，但 **不影响 fusion break**，因为：
- Phase 1（with record）的 CUDA Graph 已经 capture 了物化后的 kernel 序列
- Phase 2（without record）重新 trace，此时没有 `record()` → 没有 `realize()` → Inductor 可以自由 fusion
- **但 Phase 1 和 Phase 2 的地址必须一致**（alias_tensor 靠 Phase 1 的地址做 D2H）
- 所以 Phase 1 的物化布局已经决定了最终的内存布局

实际上 Phase 2 graph 确实可能有不同的 fusion 策略，但这不影响 D2H 正确性，因为 D2H 读的是 Phase 1 graph replay 写入的数据。两组 graph 交替执行（ping-pong），各自独立。
