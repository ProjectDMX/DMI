# Benchmark Overhead Analysis & Optimization Roadmap

**Date**: 2026-02-28
**Status**: Profiling complete, optimization directions identified
**Depends on**: Design C dual_compile (2_27_2026_dual_graph_shadow_buffer_design.md)

## Benchmark 说明

Profile 工具：`benchmark/tests/profile_decode.py`，运行命令：

```bash
CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include PYTHONPATH=. \
  nsys profile --wait=primary --output=results/nsys/profile_decode_dual_compile \
  --force-overwrite=true --trace=cuda,nvtx,osrt --sample=cpu \
  python benchmark/tests/profile_decode.py --monitoring-mode dual_compile \
  --profile-dir=results/profile_traces/nsys_run \
  --batch-size 64 --decode-steps 64 --collect-hidden --collect-attention \
  --steps 1 --warmup 1 --no-profile
```

### Benchmark 定义

| Label | Model | Compile | Monitoring | D2H | 用途 |
|---|---|---|---|---|---|
| `transformer_lens` | TransformerLens | No | No | — | 第三方库 baseline |
| `transformer_lens_cache` | TL + run_with_cache | No | No | — | TL 全量缓存开销 |
| `huggingface` | GPT2LMHeadModel | No (eager) | No | — | 原版 HF eager baseline |
| `hf_torch_compile` | GPT2LMHeadModel | Yes (reduce-overhead) | No | — | **vanilla HF 上限** |
| `hf_modified_overhead` | HookedGPT2Model | Yes (reduce-overhead) | ops.record() inline | **No** | **仅 GPU record 开销** |
| `hf_modified_hook_async` | HookedGPT2Model | Yes (dual_compile) | ops.record() + D2H | **Yes** | **完整 monitoring pipeline** |
| `huggingface_hook_cpu` | GPT2LMHeadModel | No | Python hooks | CPU copy | 原版 HF + forward hooks + CPU copy |

关键区别：
- `hf_torch_compile` vs `hf_modified_overhead`：隔离 HookPoint 结构 + `ops.record()` 的 GPU 开销
- `hf_modified_overhead` vs `hf_modified_hook_async`：隔离 D2H pipeline（start_step/end_step/event sync）的开销

## 最新 Profile 结果

**配置**：GPT-2 124M, batch=64, fp32, 183 HookPoint hooks, decode_steps=64, monitor_interval=1, d2h_repeat=1

| Benchmark | Duration (s) | tok/s | ms/step |
|---|---|---|---|
| `transformer_lens` | 1.1427 | 3,584 | 17.85 |
| `transformer_lens_cache` | 1.7215 | 2,379 | 26.90 |
| `huggingface` | 0.6087 | 6,729 | 9.51 |
| `hf_torch_compile` | 0.1444 | 28,363 | **2.26** |
| `hf_modified_overhead` | 0.1578 | 25,953 | **2.47** |
| `hf_modified_hook_async` | 0.1777 | 23,054 | **2.78** |
| `huggingface_hook_cpu` | 1.1074 | 3,699 | 17.30 |

## 开销分解

```
hf_torch_compile         2.26 ms/step  ─── vanilla compile baseline
                           │
                          +0.21 ms  (9.3%)   ← ops.record() × 183 hooks
                           │
hf_modified_overhead     2.47 ms/step  ─── compile + record, no D2H
                           │
                          +0.31 ms  (12.6%)  ← D2H pipeline
                           │
hf_modified_hook_async   2.78 ms/step  ─── 完整 monitoring
```

**总监控开销：0.52ms/step (23.1%)**，对 183 个 hook 全量监控。

### 开销来源分析

#### 1. ops.record() 开销 = 0.21ms/step

- 183 个 inline `ops.record()` 被 Inductor 编译进 CUDA Graph
- 每个 record 是一个小 CUDA kernel（写 128B metadata 到 GPU buffer）
- 在 GPU timeline 上与计算 kernel 串行排列
- 183 × ~1.1μs/kernel ≈ 0.2ms，与观测一致
- **这是不可压缩的固有开销**（除非减少 hook 数量或将多个 record 合并成一个 kernel）

#### 2. D2H pipeline 开销 = 0.31ms/step

来源拆解：
1. **Python for-loop launch** (~0.15ms)：`end_step()` 中 183 个 `pinned_buf.copy_(alias, non_blocking=True)` 在 Python 层逐个发起
2. **Event sync** (~0.10ms)：`start_step()` 中 `d2h_events[frame].synchronize()` 等待上一次同帧 D2H 完成
3. **Stream sync + set_frame** (~0.06ms)：`copy_stream.wait_stream(forward_stream)` + `set_frame()` 的 Python 开销

实际 D2H 传输本身是 async 的（copy_stream），大部分与下一步 forward 重叠。但 event sync 和 Python launch 在主线程上阻塞。

## 历史 Profile 对比

### 不同配置下的结果

| 配置 | hf_torch_compile | hf_modified_hook_async | 监控开销 |
|---|---|---|---|
| batch=64, steps=64, hidden+attn, d2h_repeat=1 (latest) | 28,363 tok/s | 23,054 tok/s | 23.1% |
| batch=64, steps=100, hidden only, d2h_repeat=1 (earlier) | — | 21,476 tok/s | — |
| batch=64, steps=64, hidden+attn, d2h_repeat=3 | — | 17,602 tok/s | — |

d2h_repeat=3 模拟更重的 D2H 负载（每次 D2H 重复 3 次），throughput 从 23k → 17.6k tok/s (24% 下降)，说明 D2H bandwidth 是瓶颈之一。

### Eager vs Compile 对比

| Mode | tok/s | 加速比 |
|---|---|---|
| `huggingface` (eager, no monitor) | 6,729 | 1.0× |
| `huggingface_hook_cpu` (eager, hooks + CPU copy) | 3,699 | 0.55× |
| `hf_torch_compile` (compile, no monitor) | 28,363 | 4.2× |
| `hf_modified_hook_async` (compile, full monitor) | 23,054 | 3.4× |

torch.compile 带来 **4.2×** 加速（消除 Python dispatch + CUDA Graph），全量 monitoring 后仍有 **3.4×** 加速。

## 优化方向

### Direction 1: Batched D2H Launch (estimated: 0.31ms → ~0.05ms)

**目标**：消除 Python for-loop 逐个 launch copy 的开销。

**方案**：写一个 batched memcpy CUDA kernel，一次 launch 拷贝所有 183 个 alias tensors 到 pinned host buffers：

```
// 伪代码
__global__ void batched_d2h(void** src_ptrs, void** dst_ptrs, size_t* sizes, int n) {
    int i = blockIdx.x;
    if (i < n) memcpy(dst_ptrs[i], src_ptrs[i], sizes[i]);
}
```

- 1 次 kernel launch 替代 183 次 Python `copy_()` 调用
- 省去 Python for-loop 的 ~0.15ms overhead
- 同时减少 CUDA driver overhead（183 次 → 1 次 cudaMemcpyAsync dispatch）
- **预计 D2H 开销从 0.31ms → 0.05ms**

实现路径：在 `graph_monitor_ops.cu` 中新增 `batched_d2h_async` op。

### Direction 2: Record Kernel Fusion (estimated: 0.21ms → ~0.05ms)

**目标**：减少 ops.record() 的 kernel launch overhead。

**方案 A：Per-layer fused record**
- 当前 GPT2Block 内有 ~15 个 record() 调用
- 合并为每层 1 个 fused_record kernel，写所有 15 个 slot 的 metadata
- 12 layers × 1 kernel = 12 次 launch（替代 183 次）

**方案 B：Warmup 后消除 record（推荐）**

核心 insight：`cudagraph_trees=False` 下，tensor 地址在 CUDA Graph 捕获后固定不变。`record()` 每次 replay 写的都是完全相同的 metadata。`parse_frame_metadata()` + `create_frame_aliases()` 在 warmup 后已经提取了所有地址信息，生产阶段 alias D2H 不依赖 record()。

**实现**：warmup 后设 `_mon_buf = None`，Dynamo re-trace 出不含 record kernel 的 graph。

关键问题：`_mon_frame_offset` 原本在 `if _mon is not None:` 内部读取。`_mon_buf=None` 后 Dynamo 看不到 `_off`，不会 guard → 只生成一个 graph → 失去双帧地址隔离。

**修复**：将 `_off` 的读取提到 `if` 外面：

```python
# Before (frame guard inside dead branch)
_mon = getattr(self, "_mon_buf", None)
if _mon is not None:
    _off = getattr(self, "_mon_frame_offset", 0)
    _mon_record(tensor, _mon, self._mon_slot_hook_xxx + _off)

# After (frame guard always visible to Dynamo)
_off = getattr(self, "_mon_frame_offset", 0)
_mon = getattr(self, "_mon_buf", None)
if _mon is not None:
    _mon_record(tensor, _mon, self._mon_slot_hook_xxx + _off)
```

Warmup 流程（4 graphs）：

```
Phase 1 — metadata discovery:
  _mon_buf = tensor, _off = 0  → Dynamo trace graph A (with record)
  _mon_buf = tensor, _off = N  → Dynamo trace graph B (with record)
  parse_metadata → create_aliases（一次性）

Phase 2 — production graph:
  set _mon_buf = None on all modules
  _mon_buf = None, _off = 0   → Dynamo trace graph C (no record, 独立内存池)
  _mon_buf = None, _off = N   → Dynamo trace graph D (no record, 独立内存池)

Steady state: 只用 graph C/D，alias D2H 照常工作
```

**效果**：0.21ms → 0ms。代价：warmup 多 trace 2 个 graph（一次性）。

**风险**：依赖于 `cudagraph_trees=False` 的地址稳定性保证。如果未来地址策略变化，需要回退到方案 A。

### Direction 3: Selective Monitoring / Hook Whitelist

**目标**：减少 hook 数量，降低 record + D2H 的线性开销。

**方案**：
- 新增 `hook_whitelist` API：只 monitor 指定的 hook 子集
- `per_request_mask`：每次推理请求可指定 active hook set
- 从 183 hooks 降到 ~12（per-layer residual stream only），开销线性下降到 ~1/15

**适用场景**：
- 生产环境只需 per-layer hidden state（12 hooks）
- 调试时开全量 hook（183 hooks）
- Per-request 动态切换

### Direction 4: Skip-Step Monitoring (已实现)

`--monitor-interval I`：每 I 步 monitor 一次。非 monitor 步跳过 D2H。

- I=2：D2H 有 2× forward 时间完成，减少 event sync stall
- I=4：基本消除 D2H stall，throughput 接近 `hf_modified_overhead`

详见 `2_28_2026_barrier_skip_steps_design.md`。

### 优化优先级

| 方向 | 预期收益 | 复杂度 | 优先级 |
|---|---|---|---|
| Selective monitoring (Dir 3) | 高（线性减少） | 低 | **P0** |
| Batched D2H launch (Dir 1) | 中（0.31→0.05ms） | 中 | P1 |
| Record fusion/elimination (Dir 2) | 中（0.21→0.05ms） | 高 | P2 |
| Skip-step (Dir 4) | 已实现 | — | Done |

**Selective monitoring 优先**：因为它同时减少 record 和 D2H 的开销（线性关系），且实现简单（只需在 `_register_hooks` 中过滤）。从 183 hooks → 12 hooks，预计总开销从 0.52ms → ~0.03ms（<2% overhead）。

## 文件索引

- Profile 结果 JSON: `results/profile_traces/nsys_run/timing_results.json`
- nsys trace: `results/nsys/profile_decode_dual_compile.nsys-rep`
- Benchmark 脚本: `benchmark/tests/profile_decode.py`
- 相关设计: `docs/dev_log/2_27_2026_dual_graph_shadow_buffer_design.md`
- Barrier 设计: `docs/dev_log/2_28_2026_barrier_skip_steps_design.md`
