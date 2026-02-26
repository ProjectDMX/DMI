# Compile Mode Hook 修复与 H2H 优化分析 (2026-02-26)

## 1. Hook 注册 Bug 与修复

### 问题

`GraphMonitor._register_hooks` 在 compile 模式下有两条路径：

- **路径 A (inline attrs)**：对有 `monitor_activation` 的 HookPoint 模块，在 parent 模块上设置 `_mon_buf` / `_mon_slot_*` 属性，**不注册 forward hook**
- **路径 B (forward hook)**：对其他模块注册 `_make_compile_hook`

路径 A 的设计意图是让模型 forward 里的 `_mon_record()` 直接调用 custom ops。但问题是两条路径互斥——走了路径 A 就不注册 forward hook。

### 影响

1. **Prefill 阶段**：`monitor_activation()` 已被 `_mon_record()` 替换，`HookPoint.forward()` 不再被调用 → forward hook 不触发
2. **Decode 阶段**：`_mon_record()` 需要 `_mon_buf` 属性才能工作 → 如果 inline attrs 被删除则不触发

### 修复方案

两者并存：对 HookPoint 模块**同时**设置 inline attrs **和**注册 forward hook。

```python
# _register_hooks 中：
handle = module.register_forward_hook(self._make_hook(slot_id))  # 所有模块都注册
self._handles.append(handle)

# HookPoint 额外设置 inline attrs
if self._graph_mode == "compile" and hasattr(module, "monitor_activation"):
    parent._mon_buf = self._gpu_buffer
    setattr(parent, f"_mon_slot_{attr_name}", slot_id)
```

### 相关背景：`monitor_activation()` vs `_mon_record()`

| | `monitor_activation()` | `_mon_record()` |
|---|---|---|
| 位置 | `HookPoint` 方法 | 模型 forward 里的函数 |
| 机制 | `super().__call__()` 触发 forward hook | 直接调 `record()` + `sink()` custom ops |
| Dynamo 兼容 | 否（`__call__` + side effects） | 是（Meta dispatch） |
| 依赖 | forward hooks 注册在 HookPoint 上 | `_mon_buf` / `_mon_slot_*` 属性在 parent 上 |

当前模型 forward 已将所有 `monitor_activation()` 替换为 `_mon_record()`（在 transformers submodule 的 `modeling_gpt2.py` 中）。

---

## 2. Module Filter：HookPoint vs 全模块

### 问题

`module_filter=lambda name, module: True` 对所有 `named_modules()` 注册 hook。

GPT-2 共 344 个模块：
- **183 HookPoint**：`hook_q`, `hook_k`, `hook_v`, `hook_attn_scores`, `hook_pattern`, `hook_z`, `hook_result`, `hook_resid_pre/mid/post`, `hook_ln1/ln2`, `hook_mlp_in/out`, `hook_embed`, `hook_pos_embed`, `hook_final_ln`
- **161 非 HookPoint**：`nn.Linear(Conv1D)`, `nn.LayerNorm`, `nn.GELU`, `nn.Dropout`, `GPT2Block`, `GPT2Attention`, `GPT2MLP`, `nn.Embedding`, `nn.ModuleList`

非 HookPoint 模块的 hook 全是冗余或无用的：
- `nn.LayerNorm` 输出 = 对应 `hook_ln1/ln2` 记录的 tensor
- `nn.Dropout` 推理时是 identity
- 容器模块（`GPT2Block` 等）输出 = 最后一个子模块输出

### 修复

```python
module_filter=lambda name, module: hasattr(module, "monitor_activation")
```

效果：hook 数量 344 → 183，D2H 数据量减半（4.6ms → 2.3ms）。

---

## 3. H2H (Pinned → Pageable) 优化对比

### 三种已实现路径

#### 路径 1：Per-tensor Thread Pool（`MON_NATIVE_HOST_COPY_THREADS=10`）

```
对每个 pinned tensor:
  thread pool worker: malloc() + memcpy() + release pool block
10 线程并行
```

**实测**：`process_us = 282,532`（128 steps），**2.2ms/step**

优点：多线程并行 memcpy 饱和内存带宽
缺点：182 次 malloc，线程调度开销

#### 路径 2：Gather 单线程（`MON_NATIVE_GATHER_H2H=1`，无并行 memcpy）

```
1× malloc(~18MB contiguous)
182× sequential memcpy（单线程）
182× from_blob（zero-copy view）
```

**实测**：`process_us = 1,210,703`（128 steps），**9.5ms/step**

优点：1 次 malloc，目标连续
缺点：单线程 memcpy 无法饱和带宽，比 thread pool 慢 4.3x

#### 路径 3：Gather + 并行 memcpy（实验性，已撤销）

```
1× malloc(~18MB contiguous)
将 gather_list 按 chunk 分配给 thread pool workers 并行 memcpy
182× from_blob（zero-copy view）
```

**实测**：`process_us = 381,969`（128 steps），**3.0ms/step**

比单线程 gather 快 3.2x，但比纯 thread pool 路径慢 1.4x。原因待分析（可能是 chunk 分配不均、worker 竞争等）。

### 对比总结

| 路径 | process_us/step | 相对速度 |
|---|---|---|
| Thread Pool (10 threads) | 2.2ms | 1.0x (baseline) |
| Gather + 并行 memcpy | 3.0ms | 0.73x |
| Gather 单线程 | 9.5ms | 0.23x |

### 未探索的优化方向

1. **Gather + 并行 memcpy 调优**：当前 chunk 按 tensor 数量均分，改为按字节均分可能更均衡
2. **Early pre-allocation**：shadow buffer 到达后立即 malloc pageable buffer，与 D2H 并行——但 gather 已经是 1 次 malloc（~50μs），并行化收益可忽略
3. **GPU-side gather (Design B)**：在 GPU 上 pack 所有 tensor 到连续 buffer，单次大 DMA 传输——从根本上解决 D2H 碎片化问题，但实现复杂度高
4. **Ring/Ping-Pong Pageable Buffer**：消除热路径上的所有 malloc，并实现 D2H 与 H2H 的 step 间流水线

#### Ring/Ping-Pong Buffer 方案详解

**当前问题**：每个 step 的 D2H 和 H2H 串行执行，且每次 H2H 都需要 malloc pageable 内存。

```
Step N:   [D2H: GPU→pinned 2.3ms][H2H: pinned→pageable 2.2ms]
Step N+1:                          idle                         [D2H: 2.3ms][H2H: 2.2ms]
```

**方案**：预分配 2 个（ping-pong）或 K 个（ring）大 pageable buffer（~18MB each），step 轮流写入不同 buffer。

```
Step N:   [D2H→pinned_A 2.3ms][H2H: pinned_A→pageable_buf[0] 2.2ms]
Step N+1:                      [D2H→pinned_B 2.3ms]           [H2H: pinned_B→pageable_buf[1] 2.2ms]
Step N+2:                                      (consumer 消费完 buf[0]) [D2H→pinned_A ...][H2H→buf[0] ...]
```

**收益**：
- **零 malloc**：pageable buffer 启动时预分配一次，热路径无 allocator 调用
- **D2H/H2H 流水线**：step N 的 H2H 和 step N+1 的 D2H 可以并行（不同 pinned blocks + 不同 pageable slots）
- Thread pool 的多线程 memcpy 仍可用——只是目标从 per-tensor malloc 变为 ring buffer 内的 offset

**约束**：
- Ring buffer slot 在下游 consumer 消费完结果前不能被覆盖
- 需要引用计数或 fence 机制确保 `from_blob` views 在 slot 回收前失效
- 如果 consumer 消费慢（delay_steps > ring size），需要 backpressure 或增大 ring

**预期效果**：
- H2H 时间：接近当前 thread pool 路径（2.2ms），但省去 per-step malloc (~0.5-1ms)
- 有效吞吐：D2H + H2H 从串行 4.5ms 降至流水线 ~2.3ms（被 D2H 瓶颈限制）
- 总 step 开销：从 ~4.5ms 降至 ~2.5ms（D2H 2.3ms + 流水线化的 H2H 重叠部分）

---

## 4. 当前状态

### 保留的修改（未提交）

- `graph_monitor.py`：inline attrs + forward hook 并存修复
- `profile_decode.py`：`module_filter` 只匹配 HookPoint
- `test_graph_monitor.py`：删除 inline attrs 专用测试

### 已撤销的修改

- `engine_core.cpp`：并行 memcpy gather 路径（实验性，性能不如纯 thread pool）
- `native_engine_internal.h`：`gather_tasks_` 字段

### 推荐配置

当前最优：`MON_NATIVE_HOST_COPY_THREADS=10`（不开 gather），2.2ms/step H2H。
