# D2D Staging Ping-Pong 可行性研究

**Date**: 2026-03-01
**Status**: 研究完成，方案可行但暂不实施
**Depends on**: Design C dual_compile (2_27_2026_dual_graph_shadow_buffer_design.md)

## 1. 动机：选择性 Ping-Pong

### 1.1 当前方案的问题

Design C 使用 `cudagraph_trees=False` 实现双帧 ping-pong。`cudagraph_trees=False` 关闭了 CUDAGraphTreeManager 的 checkpoint/restore 机制，每个 CUDA Graph 获得独立的内存池。这保证了两帧 graph（frame 0 和 frame 1）中所有 tensor 的地址互不重叠——包括**被监控的**和**未被监控的**。

但我们真正需要的只是被监控 tensor 的地址隔离。未被监控的 tensor 用相同地址完全没问题——它们不需要 D2H，没有数据竞争风险。

理想情况：

```
cudagraph_trees=True (当前不可行):
  graph_0 和 graph_1 中:
    - unmonitored tensors: 相同地址 ✅（checkpoint/restore，节省显存）
    - monitored tensors: 不同地址 ✅（ping-pong D2H 安全）

cudagraph_trees=False (当前方案):
  graph_0 和 graph_1 中:
    - unmonitored tensors: 不同地址 ❌（浪费显存）
    - monitored tensors: 不同地址 ✅
```

### 1.2 显存影响

对 GPT-2（183 hooks, batch=64, fp32）：
- 被监控 tensor 总量 ≈ 18 MB
- 全部 intermediate tensor 总量 ≈ 86 MB
- `cudagraph_trees=False` 额外显存 = 86 MB（所有 intermediate 两份）
- 理想情况额外显存 = 18 MB（只有 monitored 两份）

对大模型（Llama 70B）这个差距更大。

### 1.3 研究目标

**能否在 `cudagraph_trees=True` 下实现只对 monitored tensor 做 ping-pong？**

---

## 2. CUDAGraphTreeManager 内存共享机制

### 2.1 Checkpoint/Restore 如何工作

`cudagraph_trees=True` 时，PyTorch 的 `CUDAGraphTreeManager` 管理所有 CUDA Graph。关键机制：

```
Recording graph_0:
  allocator 分配地址集 {A} → graph_0 执行完成
  checkpoint(): 保存所有 tensor 的地址 + 内容快照

Recording graph_1:
  restore(): 恢复到 checkpoint 状态（地址集 {A} 全部恢复）
  graph_1 执行 → 所有输出写到相同的地址集 {A}

Replay:
  graph_0 replay → 写 {A}
  graph_1 replay → 写 {A}  ← 地址完全相同！
```

**核心机制**：`checkpoint()` / `restore()` 确保 sibling graph 从完全相同的内存状态开始录制 → 输出到相同地址。

### 2.2 源码验证

`torch/_inductor/cudagraph_trees.py`:

```python
class CUDAGraphNode:
    def _record(self, model, inputs):
        # ...
        if self.parent is not None:
            # Sibling 节点：restore 到父节点的 checkpoint
            self.parent._restore()  # ← 恢复所有地址
        # ... 然后开始录制
```

`torch/cuda/graphs.py`:

```python
class CUDAGraph:
    def _record(self):
        # checkpoint 保存 pool 状态 (MempoolId → snapshot)
        # restore 恢复 pool 状态到 checkpoint
```

**结论**：CUDAGraphTreeManager 的 sibling 共享是**架构级**的，不是配置选项。无法对个别 tensor 禁用。

---

## 3. 尝试方案一：cudaMemPool_t（C++ 层）

### 3.1 思路

CUDA runtime 提供 `cudaMemPool_t` API，可以创建独立的内存池。如果能让 Inductor 把 monitored tensor 的输出分配到独立 pool，就能绕过 checkpoint/restore 的地址共享。

### 3.2 C++ 可用性

`cudaMemPool_t` 在 C++ CUDA runtime 中完全可用：

```c++
cudaMemPool_t pool;
cudaMemPoolCreate(&pool, &poolProps);
cudaMallocFromPoolAsync(&ptr, size, pool, stream);
```

### 3.3 问题：Inductor 不支持指定分配池

Inductor 的内存分配路径：

```
Inductor CodeGen → torch.empty(..., device='cuda') → caching allocator → default pool
```

Inductor 没有 API 让 custom lowering 指定 "这个 buffer 应该分配到 pool X"。整个 memory planning 阶段只处理 buffer 的 live range 和 size，不涉及 pool 选择。

要实现这个功能需要修改 Inductor 的：
1. IR 层（给 buffer 加 pool 标注）
2. Memory planning（按 pool 分组）
3. CodeGen（生成 `allocate_from_pool(pool_id)` 代码）

这是 PyTorch Inductor 的核心架构改动，不现实。

---

## 4. 尝试方案二：torch.cuda.MemPool + use_mem_pool

### 4.1 发现

PyTorch 提供了 Python 层的 MemPool API：

```python
pool = torch.cuda.MemPool()
with torch.cuda.use_mem_pool(pool):
    # 此 context 内的 CUDA 分配使用 pool
    x = torch.randn(100, device='cuda')  # → 分配到 pool
```

### 4.2 测试结果

#### 测试 1：use_mem_pool 在 compiled function 内部

```python
pool = torch.cuda.MemPool()

@torch.compile(mode="reduce-overhead")
def fn(x, flag):
    if flag == 0:
        with torch.cuda.use_mem_pool(pool):
            monitored = x @ w  # 希望分配到 pool
        unmonitored = monitored + bias  # 回到 default pool
    # ...
```

**结果**：`use_mem_pool` 是 Python context manager，Dynamo 遇到它时触发 **graph break**（`Unsupported: 'skip function use_mem_pool'`）。编译函数被拆成多个子图，失去 CUDA Graph 的整体优化。

#### 测试 2：use_mem_pool 在 compiled function 外部

```python
pool = torch.cuda.MemPool()

with torch.cuda.use_mem_pool(pool):
    # 整个 compiled function 在 pool 内执行
    out = compiled_fn(x, flag=0)
```

**结果**：CUDAGraphTreeManager 的 `check_memory_pool()` 检查所有输出 tensor 是否分配在 tree 自己的 pool 里。非 tree-pool 的输出被**拒绝**（`RuntimeError: check_memory_pool`）。

#### 测试 3：graph break 后的后续子图

观察到 graph break 后的后续子图（在 `use_mem_pool` context 之外）的 default-pool 分配也产生了不同地址。这是因为子图独立编译，memory planning 独立执行——但这不是 "选择性 ping-pong"，而是 graph break 导致的意外行为。

### 4.3 结论

`torch.cuda.MemPool` 无法用于 `torch.compile` + CUDA Graph 场景：
- 内部使用：graph break
- 外部使用：check_memory_pool 拒绝

---

## 5. 尝试方案三：手动 CUDA Graph 录制

### 5.1 思路

放弃 `torch.compile` 自动 CUDA Graph 管理，手动录制 CUDA Graph：

```python
# 手动录制 graph_0
g0 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g0, pool=shared_pool):
    out = model(x, flag=0)

# 手动录制 graph_1
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1, pool=shared_pool):
    out = model(x, flag=1)
```

### 5.2 共享 pool 结果

手动录制时，如果两个 graph 共享同一个 pool：
- graph_0 录制时分配地址集 {A}
- graph_1 录制时，{A} 已经被 graph_0 "占用"（capture 期间 pool 不回收）
- graph_1 被迫分配不同地址集 {B}

**所有 tensor 地址都不同**——包括 unmonitored tensor。这和 `cudagraph_trees=False` 效果一样。

### 5.3 手动 checkpoint/restore + 独立 monitored pool

尝试手动实现 checkpoint/restore 机制，同时让 monitored tensor 使用独立 pool：

```python
# 录制 graph_0
with torch.cuda.graph(g0, pool=shared_pool):
    # unmonitored: 分配在 shared_pool
    # monitored: 切到 separate_pool → graph break 或不可行
    out = model(x, flag=0)
```

**问题**：无法在 graph capture 过程中切换 allocator pool。capture 期间所有分配走同一个 pool（传给 `torch.cuda.graph()` 的那个）。

### 5.4 结论

手动 CUDA Graph 录制无法实现选择性 ping-pong。要么所有 tensor 共享地址（checkpoint/restore），要么所有 tensor 不同地址（独立 pool）。没有中间状态。

---

## 6. 根本原因分析

### 6.1 为什么选择性 ping-pong 在 PyTorch 下不可行

```
CUDA Graph 录制 → 所有分配走同一个 pool
                → pool 是 graph 级粒度，不是 tensor 级粒度
                → 无法对个别 tensor 指定不同的分配策略

checkpoint/restore → pool 级快照/恢复
                   → 要么恢复所有 tensor（共享地址），要么不恢复（不同地址）
                   → 无法只恢复部分 tensor

Inductor memory planning → buffer 级 live range 分析
                         → 无 pool 标注 → 所有 buffer 共享同一个 allocation namespace
                         → 无法指定 "这个 buffer 用 pool X"
```

**三个层面**（CUDA runtime、PyTorch caching allocator、Inductor memory planner）都不支持 per-tensor pool 选择。这不是配置问题，而是架构设计。

### 6.2 可行的 ping-pong 方式

| 方式 | 地址隔离粒度 | 显存开销 | 当前状态 |
|------|------------|---------|---------|
| `cudagraph_trees=False`（当前） | 所有 tensor | 所有 intermediate × 2 | ✅ 已实现 |
| D2D staging（提议） | 只有 monitored | monitored × 2 (staging buffer) | 📋 方案设计 |
| Per-tensor pool（理想） | 只有 monitored | monitored × 2 | ❌ PyTorch 不支持 |

---

## 7. D2D Staging Ping-Pong 方案

### 7.1 核心思路

不依赖 CUDA Graph 的地址隔离，而是在 forward 中显式 `copy_()` monitored tensor 到预分配的 staging buffer：

```python
# 预分配 staging buffer（module buffer，固定地址）
# 每个 monitored hook 有两个 staging slot（frame 0 和 frame 1）
self.staging_0 = torch.empty_like(expected_shape)  # frame 0
self.staging_1 = torch.empty_like(expected_shape)  # frame 1

# forward 中（替代 record()）
if frame == 0:
    self.staging_0.copy_(monitored_tensor)  # D2D copy, ~0.13μs
else:
    self.staging_1.copy_(monitored_tensor)

# D2H 从 staging buffer 读取（固定地址）
pinned.copy_(self.staging_0, non_blocking=True)  # DMA from staging
```

### 7.2 关键优势

1. **兼容 `cudagraph_trees=True`**：staging buffer 是 module buffer，地址固定。CUDA Graph 的 checkpoint/restore 不影响它们（module buffer 不参与 graph-internal allocation）。

2. **显存精确**：staging buffer = 2 × sum(monitored_tensor_sizes)。只为 monitored tensor 分配，不浪费。

3. **消除所有复杂机制**：

| 当前 Design C 组件 | Staging 方案是否需要 | 原因 |
|---|---|---|
| `record()` custom op | ❌ 不需要 | `copy_()` 替代 |
| Custom Inductor lowering | ❌ 不需要 | 无需 `realize()` / `never_reuse_buffers` |
| `anchor()` op | ❌ 不需要 | staging buffer 是 module buffer，生命周期自动管理 |
| `alias_tensor()` (from_blob) | ❌ 不需要 | D2H 直接从 staging buffer 读取 |
| Shadow metadata buffer | ❌ 不需要 | 不需要地址发现（staging 地址固定） |
| Phase 1 warmup（地址发现） | ❌ 不需要 | staging 地址在分配时已知 |
| Per-slot event barrier (`wait_d2h`) | ❌ 不需要 | staging buffer 双帧隔离，forward 和 D2H 读写不同 frame |
| `sink()` / `sink_hold()` | ❌ 不需要 | 无需保活原始 tensor |

### 7.3 开销分析

**D2D copy 开销**：
- 每个 hook 一次 `copy_()`，走 DMA engine
- GPT-2 hidden state: 64 × 768 × 4B = 192KB → ~0.13μs per copy
- 12 hooks (selective): 12 × 0.13μs ≈ **1.5μs**
- 183 hooks (full): 183 × 0.13μs ≈ **24μs**

**对比**：当前 `record()` metadata write = 183 × 128B → ~0.003μs per hook，但需要 realize() + never_reuse_buffers + anchor 等复杂机制。Staging 方案用 ~24μs D2D copy 换取了极大的架构简化。

### 7.4 Dynamo/CUDA Graph 兼容性

`copy_()` 是标准 ATen op：
- Dynamo 正常 trace
- Inductor 正常 lower（作为 in-place mutation）
- CUDA Graph capture 正常录制 DMA kernel

staging buffer 是 `nn.Buffer` 或 module attribute：
- 不参与 Inductor memory planning（已存在的 persistent tensor）
- CUDA Graph capture 录制写入固定地址的 DMA
- checkpoint/restore 不影响（persistent tensor 不在 graph-internal allocation 中）

### 7.5 Warmup 流程简化

```
当前 Design C（4 graphs）:
  Phase 1: _mon_buf=tensor, _off=0   → graph A (with record, frame 0)
  Phase 1: _mon_buf=tensor, _off=N   → graph B (with record, frame 1)
  parse_metadata → create_aliases
  Phase 2: _mon_buf=None, _off=0     → graph C (no record, frame 0)
  Phase 2: _mon_buf=None, _off=N     → graph D (no record, frame 1)

D2D Staging（2-3 graphs）:
  _staging=buffers, frame=0  → graph A (with copy_(), frame 0)
  _staging=buffers, frame=1  → graph B (with copy_(), frame 1)
  [可选: _staging=None       → graph C (no copy, zero overhead)]
```

Phase 1（地址发现）完全消除。Phase 2（production graph without record）可选——如果 copy_() 开销可接受（~1.5μs for 12 hooks），可以只用 2 个 graph。

---

## 8. 用户级 Feature 兼容性分析

### 8.1 Zero-Overhead Forward（零开销 forward）

**当前实现**: `_mon_buf = None` → Dynamo re-trace → Phase 2 图没有任何 record 内核。

**Staging 方案**: `_staging = None`（或等效标志） → Dynamo re-trace → 生成无 `copy_()` 的生产图。

**机制**：完全一样。Dynamo guard 在 attribute 值变化时 re-trace，生成不含 monitoring 代码的独立 CUDA Graph。

**结论**：**完全兼容**。可选实现——如果 ~1.5μs（12 hooks）的 copy_() 开销可接受，甚至可以省掉这个优化。

### 8.2 Hook Selection（选择性监控）

**当前实现**:
- 静态选择：`module_filter` 在构造时决定哪些 module 被监控
- 动态选择：`select_hooks(["hook_resid_post"])` → `update_d2h_mask(active_slots)` 控制 D2H

**Staging 方案**:
- 静态选择：构造时只给选中的 hook 分配 staging buffer，只 trace 这些 `copy_()`。完全兼容。
- 动态 D2H mask：`copy_()` 对所有分配了 staging 的 hook 都执行（baked into CUDA Graph），但 D2H 只读活跃 slot 的 staging buffer。开销：12 hooks 的 D2D copy ≈ 1.5μs，可接受。

**选择粒度**：
| 层级 | 当前方案 | Staging 方案 |
|------|---------|-------------|
| 构造时静态选择 | `module_filter` | 同上，只分配选中 hook 的 staging buffer |
| 运行时 hook 选择 | `select_hooks()` → D2H mask | 同上，D2H mask 控制哪些 staging 被读取 |
| 运行时 copy_() 选择 | N/A（record 在 graph 中全跑） | 同理（copy_() 在 graph 中全跑） |

**结论**：**完全兼容**。hook selection 的选择点在 D2H 阶段，不在 forward 阶段。

### 8.3 Skip Steps（跳步监控）

**当前实现**:
- `monitor_interval=I`，每 I 步监控一次
- `_mon_frame` 在 monitored step 的 end_step 翻转
- Non-monitored step 跑 Phase 2 graph（零 record 开销）
- D2H deferred 到下一步 start_step（GPU-level forward/D2H overlap）

**Staging 方案**:
需要 3 种 Dynamo specialization 的 CUDA Graph：

| Graph | 内容 | 用途 |
|-------|------|------|
| Frame 0 | `copy_()` 到 staging[0] | Monitored step, frame=0 |
| Frame 1 | `copy_()` 到 staging[1] | Monitored step, frame=1 |
| No-copy | 无 staging `copy_()` | Non-monitored step, 零开销 |

时序分析（interval=2）：

```
step 0 (f0, mon):  run graph_frame0 → copy_() to staging[0]
step 1 (f1, skip): run graph_nocopy → zero overhead
                   submit D2H from staging[0]
step 2 (f1, mon):  run graph_frame1 → copy_() to staging[1]
step 3 (f0, skip): run graph_nocopy → zero overhead
                   submit D2H from staging[1]
...
```

**对比当前方案**：
- 当前：4 graphs（Phase 1 × 2 + Phase 2 × 2）
- Staging：3 graphs（frame 0 + frame 1 + no-copy）

**结论**：**完全兼容**，且比当前方案少 1 个 graph。

### 8.4 Per-Request Selection（按请求选择）

**当前实现**:
- `select_hooks(["hook_resid_post"], requests={2, 15, 46})`
- D2H 通过指针算术只拷贝 batch 中指定 request 的数据
- Build 阶段（off critical path）：tensor slice → `data_ptr()` 提取 → int64 tensor 打包
- Launch 阶段（on critical path）：`batch_d2h_ptrs` C++ op

**Staging 方案**:
staging buffer shape = `[B, *rest]`（与原始 tensor 相同），D2H 时同样可以通过指针算术读取特定 request：

```python
# staging buffer 也是 [B, H] shape
req_ptr = staging_ptr + r * stride[0] * element_size
# D2H 只拷贝 active request
```

**I/O Coalescing**：同样适用。staging buffer 是连续 tensor，stride 结构与原始 tensor 一致。

**额外优势**：staging buffer 地址在分配时已知，不需要 Phase 1 warmup 的 shadow parse + alias_tensor 地址发现流程。Per-request slicing 的 build 阶段更简单（直接算 staging buffer 的偏移量）。

**结论**：**完全兼容**，且 build 阶段简化。

### 8.5 兼容性总结

| Feature | 兼容性 | 说明 |
|---------|--------|------|
| Zero-overhead forward | ✅ 完全兼容 | `_staging=None` → Dynamo re-trace → no copy_() |
| Hook selection (static) | ✅ 完全兼容 | 构造时只分配选中 hook 的 staging |
| Hook selection (dynamic) | ✅ 完全兼容 | D2H mask 控制读取哪些 staging |
| Skip steps | ✅ 完全兼容 | 3 graphs (frame 0 + frame 1 + no-copy) vs 当前 4 graphs |
| Per-request slicing | ✅ 完全兼容 | 指针算术读 staging buffer，build 阶段更简单 |
| Per-hook event barrier | ✅ 不再需要 | 双帧 staging 天然隔离 forward 写和 D2H 读 |

---

## 9. 方案对比

### 9.1 架构复杂度

| 维度 | 当前 Design C | D2D Staging |
|------|-------------|-------------|
| Custom C++ ops | 7 个 (record, sink, anchor, alias_tensor, wait_d2h, batch_d2h, batch_d2h_ptrs) | 2 个 (batch_d2h, batch_d2h_ptrs) |
| Custom Inductor lowering | 需要 (realize + never_reuse_buffers) | 不需要 |
| CUDA Graph 数量 | 4 (Phase 1×2 + Phase 2×2) | 2-3 (frame 0 + frame 1 + [no-copy]) |
| Warmup 流程 | Phase 1 (record) → parse → Phase 2 (no record) | 直接 trace copy_() graphs |
| cudagraph_trees | False (必须) | True (可选) |
| Forward 代码侵入 | `_mon_record()` + `_anch` + anchor() (18 call sites) | `staging.copy_()` (同等 call sites) |
| Buffer reuse 防护 | 4 层 (realize + Tensor(a!) + never_reuse + anchor) | 0 层 (不需要) |

### 9.2 Forward 开销

| 方案 | Forward 内 monitoring 开销 | 说明 |
|------|--------------------------|------|
| Design C Phase 2 (production) | ~0ms | record eliminated，CUDA Graph 无 monitoring kernel |
| D2D Staging (12 hooks) | ~1.5μs | 12 × copy_() DMA，可选 no-copy graph 消除 |
| D2D Staging (183 hooks) | ~24μs | 183 × copy_() DMA |
| D2D Staging + no-copy graph | ~0ms | non-monitored step 零开销 |

### 9.3 D2H 开销

两种方案的 D2H 阶段完全相同：
- `batch_d2h` / `batch_d2h_ptrs` 从固定地址读取（Design C 用 alias_tensor 地址，Staging 用 staging buffer 地址）
- Per-request slicing 通过指针算术
- I/O coalescing 逻辑不变

### 9.4 显存开销

| 方案 | 额外显存 (GPT-2, 12 hooks selective) | 额外显存 (GPT-2, 183 hooks full) |
|------|--------------------------------------|----------------------------------|
| Design C (cudagraph_trees=False) | ~86 MB (所有 intermediate × 2) | ~86 MB |
| D2D Staging (cudagraph_trees=True) | ~4.5 MB (12 × 192KB × 2 frames) | ~70 MB (183 × 192KB × 2 frames) |
| D2D Staging (selective, cudagraph_trees=True) | **~4.5 MB** | ~4.5 MB (只分配 12 hooks) |

**Selective monitoring 下 staging 方案显存开销极小**：只为实际监控的 12 个 hook 分配 staging buffer。

---

## 10. 不实施的原因

尽管 D2D Staging 方案在架构上更简洁、兼容性更好，当前暂不实施，原因：

1. **当前方案已经工作**：Design C 已通过所有测试（24/24），selective monitoring 下开销仅 2%。
2. **`cudagraph_trees=False` 对当前模型反而更快**：+17.6% vs True（32,989 vs 28,048 tok/s），消除了 TreeManager 开销。
3. **Staging 增加 D2D copy 开销**：虽然极小（~1.5μs），但 Design C Phase 2 是真正的零开销。
4. **代码改动量大**：需要重写 forward 中的 monitoring 代码、GraphMonitor、GraphSafeEngine。

### 10.1 何时考虑实施

- 显存成为瓶颈（大模型、长序列、大 batch）
- 需要 `cudagraph_trees=True` 的其他优化（如 tree manager 的 graph reuse）
- 新模型集成时希望更简单的 API（不需要 custom Inductor lowering）

---

## 11. 文件索引

- 本文：`docs/dev_log/3_1_2026_d2d_staging_pingpong_feasibility.md`
- Design C 设计：`docs/dev_log/2_27_2026_dual_graph_shadow_buffer_design.md`
- Buffer reuse 防护：`docs/dev_log/3_1_2026_torch_compile_buffer_reuse_prevention.md`
- Benchmark 分析：`docs/dev_log/2_28_2026_benchmark_overhead_analysis.md`
- Skip-step 设计：`docs/dev_log/2_28_2026_barrier_skip_steps_design.md`
- Per-request slicing：`docs/dev_log/3_1_2026_per_request_slicing_design.md`
- CUDAGraphTreeManager 源码：`torch/_inductor/cudagraph_trees.py`
- MemPool API：`torch/cuda/memory.py` (`MemPool`, `use_mem_pool`)
