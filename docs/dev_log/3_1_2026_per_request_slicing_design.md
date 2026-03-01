# Per-Request Slicing & I/O Coalescing Design

**Date**: 2026-03-01
**Status**: Implemented, all tests pass (9/9)
**Depends on**: BatchedD2HDescriptor (2_28_2026_benchmark_overhead_analysis.md)

## 核心 Insight

CUDA 虚拟内存是连续的。对于 shape `[B, *rest]` stride `[s0, ...]` 的 batch tensor，任意 request `r` 的子 tensor 地址可通过纯指针算术计算：

```
req_ptr   = base_ptr + r * stride[0] * element_size
req_shape = shape[1:]
req_stride = stride[1:]
req_bytes = stride[0] * element_size
```

不需要任何 GPU 计算，不需要 PyTorch dispatcher，只需整数运算。

## 动机：数据量对比

| 场景 | hooks | requests | 单 hook/req 大小 | 总 D2H/step |
|---|---|---|---|---|
| 全量 (current) | 183 | 64 | 192KB | ~35MB |
| hook 筛选 only | 12 | 64 | 192KB | ~2.3MB |
| hook + request 筛选 | 12 | 3 | 3KB | **~108KB** |

从 35MB → 108KB，**~300× 缩减**。

## 使用方式

```python
# 初始化（照常）
engine = GraphSafeEngine(graph_mode="dual_compile", ...)
engine.prepare_for_model(model)
# ... warmup + finalize_dual_frame ...

# ---- Layer 1: Hook Selection ----

# 只监控 per-layer residual stream（183 hooks → 12）
engine.select_hooks(["hook_resid_post"])

# 多 pattern 组合
engine.select_hooks(["hook_resid_post", "hook_pattern"])

# ---- Layer 2: Per-Request Slicing ----

# 所有选中 hook 共享同一组 request
engine.select_hooks(["hook_resid_post"], requests={2, 15, 46})

# 不同 hook 看不同 request（dict 形式）
engine.select_hooks({
    "hook_resid_post": {2, 15, 46},        # residual 只看 3 个 request
    "hook_pattern":    set(range(0, 10)),   # attention 看前 10 个
})

# 某个 hook 全 batch，其他看子集
engine.select_hooks({
    "hook_resid_post": None,       # 全 batch（64 个）
    "hook_pattern":    {2, 15},    # 只看 2 个
})

# ---- 动态更新 ----

# request 变化时随时调用（off critical path）
engine.select_hooks(["hook_resid_post"], requests={2, 46})  # request 15 退出

# 恢复全量监控
engine.select_hooks(None)

# ---- Decode loop 不变 ----
for step in range(decode_steps):
    engine.start_step()
    # ... forward ...
    engine.end_step()
```

`select_hooks` 在 off critical path 上执行，随时可调用，不影响 step cadence。

## 实现架构：控制面/数据面解耦

### Build 阶段（控制面，off critical path）

`update_requests()` / `select_hooks()` 调用时执行：

1. **Tensor slice bounds check**：用 `alias[start:end+1]` 验证 request 索引合法性
2. **提取裸指针**：`src_slice.data_ptr()` → 纯 int64 整数
3. **Coalescing**：相邻 request 自动合并为连续 range（减少 DMA 次数）
4. **打包**：`torch.tensor(src_ptrs, dtype=torch.int64)` → 3 个 CPU int64 tensor

### Launch 阶段（数据面，on critical path）

`end_step()` 中执行：

- **Per-request 模式**：`ops.batch_d2h_ptrs(src_ptrs, dst_ptrs, sizes, n)` — O(1) Python→C++ 传参
- **Full-batch 模式**：`ops.batch_d2h(dst_list, src_list)` — 原路径

### 两种模式对比

| 维度 | Full-batch (batch_d2h) | Per-request (batch_d2h_ptrs) |
|---|---|---|
| Python→C++ 传参 | O(N) — N 个 at::Tensor 序列化 | O(1) — 3 个 int64 tensor |
| launch 延迟 (400 DMA) | ~1.0ms | ~0.6ms |
| launch 延迟 (4000 DMA) | ~10ms | ~6ms |
| 安全性 | PyTorch bounds check | Build 阶段 tensor slice check |

## C++ Op: `batch_d2h_ptrs`

```cpp
// graph_monitor_ops.cu
void batch_d2h_ptrs_op(Tensor src_ptrs, Tensor dst_ptrs, Tensor byte_sizes, int n) {
    // src_ptrs, dst_ptrs: CPU int64 tensor 存裸地址
    // 内部: for i in 0..n: cudaMemcpyAsync(dst[i], src[i], size[i], D2H, stream)
}
```

注册为 `CompositeImplicitAutograd`（CPU tensor 输入，CUDA stream 上发射 DMA）。

## I/O Coalescing（已实现）

### 物理建模

```
合并条件: Gap_bytes / Bandwidth_PCIe < Overhead_launch
Threshold = 1.5μs × 25 GB/s = 37,500 bytes ≈ 37.5 KB
```

对于 GPT-2 hidden state (768 × fp32 = 3KB/request)：Max_Gap = floor(37.5KB / 3KB) = 12 requests。
对于 Llama 70B hidden state (8192 × bf16 = 16KB/request)：Max_Gap = floor(37.5KB / 16KB) = 2 requests。

### 贪心合并算法

`_coalesce_requests()` 在 `_rebuild_ptrs()` 中自动调用：

```python
def _coalesce_requests(sorted_reqs, alias):
    stride_bytes = alias.stride(0) * alias.element_size()
    threshold = 37500  # 1.5μs × 25 GB/s
    segments = []
    seg_start = seg_end = sorted_reqs[0]
    for r in sorted_reqs[1:]:
        gap = (r - seg_end - 1) * stride_bytes
        if gap <= threshold:
            seg_end = r  # 合并
        else:
            segments.append((seg_start, seg_end))
            seg_start = seg_end = r
    segments.append((seg_start, seg_end))
    return segments  # List[(start, end)] 闭区间
```

合并后对每个 range 做 `alias[start:end+1]` 切片 → 单次 DMA 覆盖整个 range。

**注意**：coalescing 会搬运 range 内的"gap" request（无用数据），以减少 DMA 次数。消费端应只读 active request 位置。

### DMA 发射数量分析

| 场景 | DMA 次数 | CPU launch 时间 | 风险 |
|---|---|---|---|
| 12 hooks × 3 req | 36 | ~0.054ms | 无 |
| 12 hooks × 10 req | 120 | ~0.18ms | 无 |
| 12 hooks × 50 req | 600 | ~0.9ms | 边界 |
| 80 hooks × 10 req (Llama 70B) | 800 | ~1.2ms | 需 coalescing |
| 183 hooks × 10 req | 1830 | ~2.7ms | 需 hook selection |

## Pinned Buffer 策略

复用现有 full-batch pinned buffer，对 destination 做相同偏移：

```
dst_ptr = pinned_base_ptr + r * stride[0] * elem_size
```

`finalize_dual_frame()` 分配逻辑不变。D2H 后 pinned buffer 中 active request 位置被填充，其余位置 stale。消费端用 `results[slot_id][request_idx]` 索引。

## 消费端 API

采用 **Option A (Full-batch view)**：`collect_dual_frame_results()` 返回 `{slot_id: Tensor[B, H]}`，与现有 API 完全兼容。消费端已知 active requests 集合，直接 index 即可。

## 实现 Checklist

- [x] C++ op `batch_d2h_ptrs`：CPU int64 tensor 输入，cudaMemcpyAsync 循环
- [x] `BatchedD2HDescriptor.update_requests(slot_requests)`：per-slot request set
- [x] `_coalesce_requests()`：贪心合并算法（threshold = 37.5KB）
- [x] Build 阶段 tensor slice bounds check + `data_ptr()` 提取
- [x] Launch 阶段 `batch_d2h_ptrs` (per-request) vs `batch_d2h` (full-batch) 自动切换
- [x] `GraphSafeEngine.select_hooks()` 扩展：List + requests kwarg / Dict per-hook form
- [x] `GraphSafeEngine.update_d2h_requests()` passthrough
- [x] 集成测试：shared requests, per-hook request sets, coalescing, restore (9/9 pass)
- [ ] Benchmark：对比 full-batch vs per-request D2H overhead

## 实施层级

```
Layer 1: Hook Selection (select_hooks)           ← 已实现
         183 hooks → 12 hooks
         D2H: 35MB → 2.3MB, DMA: 183 → 12

Layer 2: Per-Request Slicing (+ requests)         ← 已实现
         Build: tensor slice bounds check → data_ptr extraction
         Launch: batch_d2h_ptrs (O(1) Python→C++ crossing)
         D2H: 2.3MB → ~108KB (N=3)

Layer 3: I/O Coalescing                           ← 已实现（内置于 Layer 2）
         贪心合并算法, threshold = 37.5KB
         自动减少碎片化 DMA 次数
```

## 文件索引

- 核心实现: `monitoring/graph_engine.py` (`BatchedD2HDescriptor`, `_coalesce_requests`)
- C++ ops: `monitoring/csrc/graph_monitor_ops.cu` (`batch_d2h_ptrs_op`)
- 集成测试: `tests_monitoring/test_design_c_integration.py` (`test_select_hooks`, `test_per_request_slicing`)
- Benchmark 分析: `docs/dev_log/2_28_2026_benchmark_overhead_analysis.md`
- Design C 设计: `docs/dev_log/2_27_2026_dual_graph_shadow_buffer_design.md`
