# Benchmark Overhead Analysis & Optimization Roadmap

**Date**: 2026-02-28 (updated 2026-03-01)
**Status**: Record elimination + BatchedD2HDescriptor implemented, D2H event sync is next
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
| `hf_modified_hook_async` | HookedGPT2Model | Yes (dual_compile) | **False** | record eliminated + D2H | **Yes** | **完整 monitoring pipeline** |
| `huggingface_hook_cpu` | GPT2LMHeadModel | No | — | Python hooks | sync CPU copy | naive hook baseline |

关键区别：
- `hooked_compile` vs `hf_modified_overhead`：隔离 record elimination 后 monitoring 框架自身开销（≈0）
- `hooked_compile` vs `hf_modified_hook_async`：隔离 D2H pipeline 的总开销
- `hf_modified_overhead` vs `hf_modified_hook_async`：隔离 D2H pipeline（start_step/end_step/event sync）
- `hf_torch_compile` vs `hooked_compile`：隔离 `cudagraph_trees=False` + HookedGPT2Model 的影响

**注意**：`hf_torch_compile` 使用 `cudagraph_trees=True`（默认），其余三个 monitoring 相关 benchmark 均使用 `cudagraph_trees=False`（dual_compile 设置）。`cudagraph_trees=False` 消除了 CUDAGraphTreeManager 开销，对这个模型反而更快。

## Profile 结果

### Run 3 (2026-03-01) — BatchedD2HDescriptor (latest)

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

### 开销分解

```
hooked_compile           1.84 ms/step  ─── fair compile baseline (cudagraph_trees=False)
                           │
                          +0.02 ms  (~0%)   ← monitoring 框架开销 (record eliminated)
                           │
hf_modified_overhead     1.86 ms/step  ─── compile + monitoring, no D2H
                           │
                          +0.49 ms  (26.3%)  ← D2H pipeline
                           │
hf_modified_hook_async   2.35 ms/step  ─── 完整 monitoring
```

**D2H pipeline 开销从 0.67ms → 0.49ms** (-27%)，得益于 `BatchedD2HDescriptor` 消除冗余 tensor 列表构建。
**总监控开销：0.51ms/step (27.7%)**，全部来自 D2H pipeline。

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
| `huggingface` (eager, no monitor) | 9,680 | 1.0× |
| `huggingface_hook_cpu` (eager, hooks + sync CPU copy) | 5,121 | 0.53× |
| `hooked_compile` (compile, no monitor) | 34,817 | 3.6× |
| `hf_modified_hook_async` (compile, full monitor) | 27,274 | 2.8× |

torch.compile + `cudagraph_trees=False` 带来 **3.6×** 加速，全量 monitoring 后仍有 **2.8×** 加速。

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

## 待优化：D2H Pipeline (0.49ms/step)

### 剩余开销拆解

| 来源 | 估计 | 说明 |
|---|---|---|
| C++ cudaMemcpyAsync loop | ~0.12ms | 183 次 DMA descriptor setup |
| Event sync stall | ~0.20ms | GPU barrier 等待上一帧 D2H 完成 |
| Python overhead | ~0.17ms | stream wait_event + set_frame + event record |

### 优化方向

#### Direction 1: Selective Monitoring / Hook Whitelist

**目标**：减少 hook 数量，降低 D2H 的线性开销。

**方案**：
- 新增 `hook_whitelist` API：只 monitor 指定的 hook 子集
- `per_request_mask`：通过 `update_d2h_mask(active_slots)` 动态调整（已实现 API）
- 从 183 hooks 降到 ~12（per-layer residual stream only），D2H 数据量和 launch 次数下降 ~15×
- 183 → 12 hooks：D2H 开销预计从 0.49ms → ~0.05ms

**适用场景**：
- 生产环境只需 per-layer hidden state（12 hooks）
- 调试时开全量 hook（183 hooks）
- Request 退出时 padding 部分不 monitor

#### Direction 2: Forward/D2H Overlap via Skip-Step

已有 `monitor_interval` 机制。I=2 时 D2H 有 2× forward 时间完成 → event sync stall 大幅减少。

### 优化优先级

| 方向 | 预期收益 | 复杂度 | 优先级 |
|---|---|---|---|
| Selective monitoring (Dir 1) | 高（0.49ms→~0.05ms） | 低 | **P0** |
| Skip-step overlap (Dir 2) | 已实现 | — | Done |

## 文件索引

- Profile 结果 JSON: `results/profile_traces/nsys_run/timing_results.json`
- Benchmark 脚本: `benchmark/tests/profile_decode.py`
- 相关设计: `docs/dev_log/2_27_2026_dual_graph_shadow_buffer_design.md`
- Barrier 设计: `docs/dev_log/2_28_2026_barrier_skip_steps_design.md`
