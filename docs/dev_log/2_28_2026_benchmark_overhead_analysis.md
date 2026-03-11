# Benchmark Overhead Analysis & Optimization Roadmap

**Date**: 2026-02-28 (updated 2026-03-01)
**Status**: Selective monitoring 实现，D2H 开销 0.44ms → 0.02ms (~1%)
**Depends on**: Design C dual_compile (2_27_2026_dual_graph_shadow_buffer_design.md)

## Benchmark 说明

Profile 工具：`benchmark/tests/profile_decode.py`，运行命令：

```bash
CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include PYTHONPATH=. \
  python benchmark/tests/profile_decode.py --monitoring-mode dual_compile \
  --profile-dir=results/profile_traces/nsys_run \
  --batch-size 64 --decode-steps 64 --collect-hidden --collect-attention \
  --steps 1 --warmup 1 --no-profile
```

### Benchmark 定义

| Label | Model | Compile | cudagraph_trees | Monitoring | D2H | 用途 |
|---|---|---|---|---|---|---|
| `transformer_lens` | TransformerLens | No | — | No | — | 第三方库 baseline |
| `transformer_lens_cache` | TL + run_with_cache | No | — | No | — | TL 全量缓存开销 |
| `huggingface` | GPT2LMHeadModel | No (eager) | — | No | — | 原版 HF eager baseline |
| `hf_torch_compile` | GPT2LMHeadModel | Yes | True (default) | No | — | vanilla HF compile baseline |
| `hooked_compile` | HookedGPT2Model | Yes | **False** | No | — | **fair compile baseline** |
| `hf_modified_overhead` | HookedGPT2Model | Yes (dual_compile) | **False** | record eliminated | **No** | **仅 forward 开销** |
| `hf_modified_hook_async` | HookedGPT2Model | Yes (dual_compile) | **False** | record eliminated + D2H | **Yes (183 hooks)** | **完整 monitoring pipeline** |
| `hf_modified_hook_selective` | HookedGPT2Model | Yes (dual_compile) | **False** | record eliminated + D2H | **Yes (12 hooks)** | **selective monitoring** |
| `huggingface_hook_cpu` | GPT2LMHeadModel | No | — | Python hooks | sync CPU copy | naive hook baseline |

关键区别：
- `hooked_compile` vs `hf_modified_overhead`：隔离 record elimination 后 monitoring 框架自身开销（≈0）
- `hooked_compile` vs `hf_modified_hook_async`：隔离全量 D2H pipeline 的总开销（183 hooks）
- `hooked_compile` vs `hf_modified_hook_selective`：隔离 selective D2H pipeline 的总开销（12 hooks）
- `hf_modified_overhead` vs `hf_modified_hook_async`：隔离 D2H pipeline（start_step/end_step/event sync）
- `hf_modified_hook_selective` vs `hf_modified_hook_async`：隔离 hook 数量的影响（12 vs 183）
- `hf_torch_compile` vs `hooked_compile`：隔离 `cudagraph_trees=False` + HookedGPT2Model 的影响

**注意**：`hf_torch_compile` 使用 `cudagraph_trees=True`（默认），其余三个 monitoring 相关 benchmark 均使用 `cudagraph_trees=False`（dual_compile 设置）。`cudagraph_trees=False` 消除了 CUDAGraphTreeManager 开销，对这个模型反而更快。

## Profile 结果

### Run 4 (2026-03-01) — Selective Monitoring (latest)

**配置**：GPT-2 124M, batch=64, fp32, 183 HookPoint hooks (12 selective), decode_steps=64

| Benchmark | Duration (s) | tok/s | ms/step |
|---|---|---|---|
| `transformer_lens` | 0.7576 | 5,407 | 11.84 |
| `transformer_lens_cache` | 1.2679 | 3,231 | 19.81 |
| `huggingface` | 0.4127 | 9,925 | 6.45 |
| `hf_torch_compile` | 0.1322 | 30,984 | **2.07** |
| `hooked_compile` | 0.1173 | 34,911 | **1.83** |
| `hf_modified_overhead` | 0.1185 | 34,561 | **1.85** |
| `hf_modified_hook_selective` | 0.1197 | 34,227 | **1.87** |
| `hf_modified_hook_async` | 0.1465 | 27,951 | **2.29** |
| `huggingface_hook_cpu` | 0.7920 | 5,172 | 12.38 |

### 开销分解

```
hooked_compile              1.83 ms/step  ─── fair compile baseline
                              │
                             +0.02 ms  (~1%)   ← monitoring 框架开销 (record eliminated)
                              │
hf_modified_overhead        1.85 ms/step  ─── compile + monitoring, no D2H
                              │
                             +0.02 ms  (~1%)   ← D2H pipeline (12 hooks, selective) ✅
                              │
hf_modified_hook_selective  1.87 ms/step  ─── selective monitoring (12 hooks)
                              │
                             +0.42 ms  (22.5%)  ← D2H pipeline (183→12 的差量)
                              │
hf_modified_hook_async      2.29 ms/step  ─── 完整 monitoring (183 hooks)
```

**Selective monitoring 将 D2H 开销从 0.44ms → 0.02ms**（降低 95%）。
**总监控开销（selective）：0.04ms/step (~2%)**，基本实现零开销目标。

### Run 3 (2026-03-01) — BatchedD2HDescriptor

**配置**：GPT-2 124M, batch=64, fp32, 183 HookPoint hooks, decode_steps=64

| Benchmark | Duration (s) | tok/s | ms/step |
|---|---|---|---|
| `transformer_lens` | 0.7685 | 5,330 | 12.01 |
| `transformer_lens_cache` | 1.2628 | 3,244 | 19.73 |
| `huggingface` | 0.4231 | 9,680 | 6.61 |
| `hf_torch_compile` | 0.1302 | 31,447 | **2.04** |
| `hooked_compile` | 0.1176 | 34,817 | **1.84** |
| `hf_modified_overhead` | 0.1190 | 34,427 | **1.86** |
| `hf_modified_hook_async` | 0.1502 | 27,274 | **2.35** |
| `huggingface_hook_cpu` | 0.7998 | 5,121 | 12.50 |

D2H pipeline 开销 0.49ms/step (26.3%)。

### 开销来源分析

#### 1. ops.record() 开销 = ~0ms ✅ (已消除)

Record elimination 方案已实现。详见下方"已完成优化"。

#### 2. D2H pipeline 开销 = 0.49ms/step ← **当前瓶颈**

来源拆解（估计）：
1. **C++ loop of cudaMemcpyAsync** (~0.12ms)：183 次 DMA descriptor setup 在 C++ 层
2. **Event sync stall** (~0.20ms)：`start_step()` 中 GPU barrier 等待上一帧 D2H 完成
3. **Stream sync + set_frame + Python overhead** (~0.17ms)：`copy_stream.wait_event()` + `set_frame()` + event record

### Run 2 (2026-03-01) — record elimination + fair baseline (before BatchedD2HDescriptor)

| Benchmark | tok/s | ms/step |
|---|---|---|
| `hooked_compile` | 32,989 | 1.94 |
| `hf_modified_overhead` | 32,702 | 1.96 |
| `hf_modified_hook_async` | 24,352 | 2.63 |

D2H 开销 = 0.67ms/step。

### Run 1 (2026-02-28) — before record elimination

| Benchmark | tok/s | ms/step |
|---|---|---|
| `hf_torch_compile` | 28,363 | 2.26 |
| `hf_modified_overhead` (with record) | 25,953 | 2.47 |
| `hf_modified_hook_async` (with record) | 23,054 | 2.78 |

Record elimination 前总开销 0.52ms：record 0.21ms + D2H 0.31ms。

## Eager vs Compile 对比

| Mode | tok/s | 加速比 vs eager |
|---|---|---|
| `huggingface` (eager, no monitor) | 9,925 | 1.0× |
| `huggingface_hook_cpu` (eager, hooks + sync CPU copy) | 5,172 | 0.52× |
| `hooked_compile` (compile, no monitor) | 34,911 | 3.5× |
| `hf_modified_hook_selective` (compile, 12 hooks) | 34,227 | **3.4×** |
| `hf_modified_hook_async` (compile, 183 hooks) | 27,951 | 2.8× |

torch.compile + `cudagraph_trees=False` 带来 **3.5×** 加速。
Selective monitoring (12 hooks) 仍有 **3.4×** 加速（仅 -1% vs no monitor）。
全量 monitoring (183 hooks) 仍有 **2.8×** 加速。

## 已完成优化

### ✅ Record Elimination (0.21ms → ~0ms)

**实现日期**：2026-02-28

核心 insight：`cudagraph_trees=False` 下，tensor 地址在 CUDA Graph 捕获后固定不变。`record()` 每次 replay 写完全相同的 metadata → redundant。

**实现**：warmup 后 `_mon_buf = None` → Dynamo re-trace 出不含 record kernel 的 production graph。

关键修复：`_off = getattr(self, "_mon_frame_offset", 0)` 必须在 `_mon = getattr(self, "_mon_buf", None)` **之前**读取，否则 Dynamo 在 `_mon_buf=None` 时看不到 `_off`，不 guard → 只生成一个 graph → 失去双帧地址隔离。

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

**验证**：`hooked_compile` (34,817) ≈ `hf_modified_overhead` (34,427)，差异 <1%（noise）。

### ✅ BatchedD2HDescriptor + Selective Mask (0.67ms → 0.49ms)

**实现日期**：2026-03-01

`BatchedD2HDescriptor` 在 `finalize_dual_frame()` 时预构建 per-frame tensor 列表，`launch()` 通过单次 C++ `batch_d2h` 调用发起全部 `cudaMemcpyAsync`。支持 `update_mask(active_slots)` 动态调整 D2H 的 hook 子集。

##### SM Kernel vs DMA — 实测对比

| 方案 | 实现 | tok/s | ms/step | D2H overhead | 结论 |
|---|---|---|---|---|---|
| Python loop `copy_()` | 183 次 Python→CUDA roundtrip | ~23,000 | ~2.78 | ~0.84ms | baseline |
| SM kernel `batched_d2h_sm` | 1 kernel, SM 直写 host | **12,205** | **5.25** | **3.41ms** | ❌ DMA 的 7× |
| C++ `batch_d2h` + descriptor | 1 Python→C++ call, DMA engine | ~24,000 | ~2.67 | ~0.73ms | ✅ DMA 不占 SM |

SM kernel D2H overhead 3.41ms vs DMA 0.73ms = **7× 慢**。

##### 为什么 SM kernel 比 DMA 慢

**原因 1：DMA Copy Engine 是独立硬件，不占 SM**

`cudaMemcpyAsync` 用 GPU 上的 Copy Engine（CE）——与 SM 完全独立的硬件单元。CE 和 SM 可以真正并行：

```
主 stream:   [forward kernel]  [forward kernel]  [forward kernel]
copy_stream: [CE: D2H copy]   [CE: D2H copy]    [CE: D2H copy]
              ↑ 不占 SM，零竞争
```

SM kernel 占用计算单元：

```
主 stream:   [forward kernel........]  ← SM 被抢，变慢
copy_stream: [SM: D2H write to host]   ← 占了一部分 SM
```

GPU 只有 ~40 个 SM（RTX 3090/4090），183 个 block × 256 thread = ~47K threads ≈ 4-5 waves，forward 期间被抢走大量 SM。

**原因 2：SM 写 host 的带宽远低于 DMA**

SM 写 host memory 走 PCIe Write-Combining 路径——每次写操作有 ~1μs PCIe 延迟，warp 在等待 PCIe 确认时 stall。DMA engine 是专为 bulk PCIe transfer 设计的硬件，能饱和 PCIe 带宽（~25 GB/s Gen4）。SM 能达到的 host 写带宽通常只有 DMA 的 **1/3 到 1/5**。

对我们的 ~4.5MB 数据：
- DMA engine: ~0.2ms transfer（且不占 SM）
- SM kernel: ~0.5-1ms（且占 SM，拖慢 forward）

**原因 3：SM kernel 省的是 launch overhead，但这不是瓶颈**

SM kernel 的优势是 1 次 launch 替代 183 次 `cudaMemcpyAsync`。但 183 次 `cudaMemcpyAsync` 从 C++ 调用只需 ~0.12ms（纯 driver 开销），而 DMA transfer 本身在专用硬件上不阻塞任何东西。省下 0.12ms launch overhead 却付出 SM 竞争 + 低带宽的代价，净亏。

**SM kernel 适用场景**：大量极小 tensor（< 1KB），此时 DMA descriptor setup per-copy 的开销占比大。我们的 hidden state 是 192KB，DMA 完胜。

`batched_d2h_sm` kernel 保留在 `graph_monitor_ops.cu` 备用（未来小 tensor 场景可能有用）。

##### 灵活性保证

| 场景 | 操作 | 开销 | 需要重录 CUDA Graph？ |
|---|---|---|---|
| 初始化 | `_rebuild(all)` | 一次性 | No |
| Request 退出 (padding 不 monitor) | `update_mask(active_slots)` | ~0.01ms | No |
| 新 request 进入 | `update_mask(new_slots)` | ~0.01ms | No |
| 每步 `end_step` | `launch(frame)` | 1 Python call | No |

核心保证：forward CUDA Graph 和 D2H 完全解耦。mask 变化只影响 tensor 列表内容，不影响 CUDA Graph。

### ✅ Skip-Step Monitoring

`--monitor-interval I`：每 I 步 monitor 一次。详见 `2_28_2026_barrier_skip_steps_design.md`。

### ✅ Selective Monitoring + Per-Request Slicing (0.44ms → 0.02ms)

**实现日期**：2026-03-01

**Hook Selection** (`select_hooks`)：按 hook name 子串匹配筛选 D2H 的 hook 子集。

```python
engine.select_hooks(["hook_resid_post"])                    # 12 hooks (per-layer)
engine.select_hooks(["hook_resid_post", "hook_pattern"])    # 24 hooks
engine.select_hooks(None)                                   # 恢复全量
```

**Per-Request Slicing** (`select_hooks` + `requests`)：利用 CUDA 虚拟内存连续性，通过指针算术只 D2H batch 中的指定 request。

```python
engine.select_hooks(["hook_resid_post"], requests={2, 15, 46})  # 12 hooks × 3 req
engine.select_hooks({                                            # per-hook 独立 request set
    "hook_resid_post": {2, 15, 46},
    "hook_pattern":    set(range(0, 10)),
})
```

**I/O Coalescing**：自动合并相邻 request 的 DMA（阈值 37.5KB = 1.5μs × 25GB/s）。

**控制面/数据面解耦**：
- Build 阶段（off critical path）：tensor slice bounds check → `data_ptr()` 提取 → int64 tensor 打包
- Launch 阶段（on critical path）：`batch_d2h_ptrs` C++ op，O(1) Python→C++ 传参

**实测结果**：

| 配置 | ms/step | D2H overhead | vs baseline |
|---|---|---|---|
| 183 hooks 全量 | 2.29 | 0.44ms (24%) | -20% throughput |
| 12 hooks selective | 1.87 | **0.02ms (~1%)** | **-1% throughput** |
| No monitoring | 1.83 | 0ms | baseline |

详细设计：`docs/dev_log/3_1_2026_per_request_slicing_design.md`

## 优化历程总结

| 优化 | 日期 | 开销变化 | 状态 |
|---|---|---|---|
| Record Elimination | 02-28 | record 0.21ms → ~0ms | ✅ Done |
| BatchedD2HDescriptor | 03-01 | D2H 0.67ms → 0.49ms | ✅ Done |
| Skip-Step Monitoring | 02-28 | event stall 可调 | ✅ Done |
| Selective Monitoring | 03-01 | D2H 0.44ms → 0.02ms | ✅ Done |
| Per-Request Slicing | 03-01 | D2H 数据量 ~300× 缩减 | ✅ Done |
| I/O Coalescing | 03-01 | 自动减少碎片 DMA | ✅ Done |

**最终状态**：Selective monitoring 下总监控开销 **~0.04ms/step (~2%)**，基本实现零开销目标。

## 文件索引

- Profile 结果 JSON: `results/profile_traces/nsys_run/timing_results.json`
- Benchmark 脚本: `benchmark/tests/profile_decode.py`
- 相关设计: `docs/dev_log/2_27_2026_dual_graph_shadow_buffer_design.md`
- Barrier 设计: `docs/dev_log/2_28_2026_barrier_skip_steps_design.md`
- Per-Request Slicing 设计: `docs/dev_log/3_1_2026_per_request_slicing_design.md`
