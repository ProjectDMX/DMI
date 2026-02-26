# Custom Op + torch.compile 可行性评估报告

> 日期：2026-02-23
> 背景：评估在当前同步 copy 路径下将监控路径改为 custom op + `torch.compile(mode="reduce-overhead")` 的可行性，以及在启用异步方案（Design A 双 Graph 乒乓 / Design B Fused Gather Pipeline）后的兼容性。

## 1. 当前同步 Copy 路径改 Custom Op 的可行性

**结论：可行，改动量中等，但有几个必须解决的关键点。**

### 1.1 架构对比

```
当前架构 (手动 CUDA Graph):
  register_forward_hook → record() kernel → 手动 graph.capture() → graph.replay()
  → on_step_end() → event.record() → wait → metadata D2H → delegate → sync copy

torch.compile 架构:
  register_forward_hook → record() kernel  ←─ Dynamo 追踪，自动纳入编译图
  → compiled_forward() 返回              ←─ reduce-overhead 自动管理 graph replay
  → 读 metadata → delegate → sync copy   ←─ 与当前一致
```

核心简化：**消除 `HFGraphDecodeRunner` 全部手动 capture/replay 代码**，`torch.compile` 接管 CUDA Graph 的生命周期。

### 1.2 必须解决的三个问题

#### 问题 A：`record` / `sink` 缺少 Meta dispatch 实现

当前 `monitoring/csrc/graph_monitor_ops.cu` 只注册了 CUDA dispatch key。`torch.compile` 的 Dynamo 追踪阶段使用 `FakeTensor`（Meta dispatch key），没有 Meta 实现会导致 **trace failure**。

修复方法（简单）：

```cpp
// graph_monitor_ops.cu 新增
TORCH_LIBRARY_IMPL(graphmonitor_ops, Meta, m) {
  m.impl("record", [](const at::Tensor&, const at::Tensor&, int64_t) {
    // void return — no-op for tracing
  });
  m.impl("sink", [](const std::vector<at::Tensor>&) {});
}
```

#### 问题 B：Hook 函数不是 Dynamo-safe 的

当前 `_make_hook`（`monitoring/graph_monitor.py:86-95`）有两处会导致 **graph break**：

1. `torch.cuda.is_current_stream_capturing()` — runtime query，Dynamo 无法静态推断
2. `_extract_tensor()` 中的递归 isinstance 检查 — 对 FakeTensor 可能不稳定

在 `torch.compile` 模式下，hook 需要简化——去掉 `is_current_stream_capturing()` 分支和手动 anchor 收集，但 `sink` 的**语义仍然需要保留**（见问题 C）：

```python
def hook(module, inputs, output):
    # 直接假设 output 是 tensor，或用 output[0] 等确定性提取
    torch.ops.graphmonitor_ops.record(output, self._gpu_buffer, slot_id)
    torch.ops.graphmonitor_ops.sink([output])  # 每个 hook 直接 sink，不再手动收集
```

不再需要 `is_current_stream_capturing()` 和 `_capture_anchors`（手动收集/批量 sink 的流程），因为 `torch.compile` 内部管理 capture/replay 时序。但 `sink` 本身作为 custom op 仍需被 Dynamo 追踪进编译图（见下）。

#### 问题 C：`sink()` 在 `torch.compile` 下仍然必要

`sink()` 的职责是阻止 CUDA Graph allocator 复用被监控 tensor 的内存。**这个需求在 `torch.compile(mode="reduce-overhead")` 下依然存在**，因为：

- `reduce-overhead` 内部也使用 CUDA Graph，allocator 同样会复用已释放 tensor 的地址
- `record` kernel 只把 `data_ptr` 当作 uint64 数字写入 shadow buffer，**不构成对原始 tensor 内存的依赖**
- 如果没有 `sink`，graph replay 后 shadow buffer 记录的地址可能指向已被后续层覆盖的内存

**改造方式**：不再在 `finalize_capture()` 中批量 sink，而是让每个 hook 内的 `sink()` 调用作为 custom op 被 Dynamo 追踪进编译图。需要：

1. 给 `sink` 添加 Meta dispatch（同问题 A）
2. 在 hook 中直接调用 `sink([output])`，Dynamo 会将其纳入编译后的 graph
3. 验证 Inductor 不会优化掉 `sink` 的 `asm volatile`——由于 `sink` 是 opaque custom op（非 Triton kernel），Inductor 不会尝试融合或消除它，`asm volatile` 在 NVCC 编译的 kernel 内仍然有效

**替代方案**：如果逐 hook sink 的开销不可接受（每个监控点额外一次 1-thread kernel launch），可以将 `record` 和 `sink` 合并为一个 op `record_and_anchor`，在同一个 kernel 内完成 metadata 写入和假依赖注入。

### 1.3 同步 Copy 路径的改造方案

```python
# ====== 编译前：注册 hooks ======
monitor = GraphMonitor(model, max_slots=4096)  # hook 内只调用 record()

# ====== 编译 ======
compiled_forward = torch.compile(model.forward, mode="reduce-overhead")

# ====== Decode 循环 ======
for step in range(num_steps):
    # reduce-overhead 自动管理 capture/replay
    output = compiled_forward(token, past_kv=past, ...)

    # 此刻 graph replay 已完成，metadata buffer 已由 record kernel 写入
    # CPU 侧直接读取 pinned metadata
    monitor.on_step_end(step_id=step)
    monitor.wait_for_step()          # sync: 等 D2H copy 完成
    step_id, snapshot = monitor.pop_ready_step(wait=True)

    # 与当前完全一致：decode → delegate → native backend
    results = engine.collect_results(wait=True)
    engine._submit_to_delegate(results)
```

### 1.4 工作量评估

| 改动项 | 工作量 |
|--------|--------|
| 添加 Meta dispatch | < 1 小时 |
| 简化 hook（去除 capture check） | 半天 |
| 删除 `HFGraphDecodeRunner` 手动 capture/replay | 1 天 |
| 更新 benchmark 脚本使用 `torch.compile` | 1 天 |
| 验证 + 修 edge cases | 1-2 天 |
| **总计** | **约 1 周** |

### 1.5 KV Cache 问题的影响

上次发现的 KV Cache `resize_()` 问题——在 `torch.compile(mode="reduce-overhead")` 下：

- `torch.compile` 遇到动态 shape（如 KV Cache 增长）时会 **fallback 到 eager mode 或重新 capture**
- 对 **decode 阶段**（固定 shape）：每步输入 shape 相同，`reduce-overhead` 能稳定复用同一张 graph，**没有 resize 问题**
- 对 **prefill 阶段**（变长 sequence）：会触发多次 capture 或 fallback，性能下降但**功能正确**
- 这比手动 CUDA Graph + resize 的"乱码"问题好得多——`torch.compile` 至少保证正确性

## 2. 异步方案下改为 Custom Op + `torch.compile` 的可能性

### 2.1 Design A（双 Graph 乒乓 / 零拷贝）：可行性低

```
核心冲突：torch.compile 对每个 input signature 缓存 一张 graph。
          Design A 要求同一 input signature 交替使用两张 graph，写入不同物理地址。
```

`torch.compile` 不提供以下能力：

- 按 step parity 选择不同的 graph 执行
- 为同一编译函数维护两套 tensor allocation
- 控制 allocator 的 mempool 分区

#### 可能的绕行方案

**方案 A1：GPU-side step counter**

```python
# 在 record kernel 内选 buffer
record_to_pingpong(tensor, buffer_A, buffer_B, step_counter_device, slot_id)
# kernel 读 step_counter 决定写 A 或 B
```

问题：

- `buffer_A` 和 `buffer_B` 只是 metadata buffer，不是激活值本身
- 真正需要"乒乓"的是**激活值的物理地址**——这由 allocator 决定，`torch.compile` 不暴露此控制
- 即使 metadata 可以乒乓，backend 读取上一步的 tensor 时，该内存已被当前步覆盖

**方案 A2：`torch.cond` 条件分支**

```python
def forward_A(x): ...  # 写入 slot set A
def forward_B(x): ...  # 写入 slot set B
output = torch.cond(step % 2 == 0, forward_A, forward_B, (x,))
```

问题：

- `torch.cond` 在 `reduce-overhead` 下的 CUDA Graph 支持尚不成熟
- 两个分支会产生两张 graph，但调度逻辑本身不在 graph 内

**结论**：Design A 的核心需求——控制两套激活值的物理内存地址和 graph 交替——与 `torch.compile` 的抽象层级根本不匹配。如果要做 Design A，**必须保留手动 CUDA Graph 管理**。

### 2.2 Design B（Fused Gather Pipeline / D2D）：可行性中高

```
Design B 的核心流程：
  forward(compute) → gather_kernel(D2D to staging) → async D2H(separate stream)

这恰好与 torch.compile 的边界天然对齐：
  ├─ [compiled region] forward + record + gather  ← 全部在 graph 内
  └─ [eager region]    async D2H + CPU processing  ← graph 外
```

#### 为什么 Design B 与 `torch.compile` 兼容

**1. Gather kernel 可以是 custom op：**

```python
# 新增 custom op
torch.ops.graphmonitor_ops.gather_to_staging(
    metadata_buffer,    # slot 地址信息
    staging_buffer,     # 目标 staging ring buffer 的当前帧
    num_slots,
)
# Dynamo 追踪 → graph 内执行 → replay 时自动执行
```

**2. 地址固定性保证：** `reduce-overhead` 模式下，所有中间激活的地址在 replay 间不变 → gather kernel 的源地址（capture 时写入 metadata 的 `data_ptr`）在 replay 时仍然有效。

**3. Staging ring buffer 帧切换：** 可以通过一个 device-side frame counter 在 kernel 内部选择写入哪一帧：

```cpp
// gather kernel 内部
int frame = atomicAdd(frame_counter, 0) % NUM_FRAMES;  // 读当前帧
staging[frame][slot_id] = *source_ptr;  // 写入对应帧的 staging
```

帧切换由 CPU 在两次 `compiled_forward()` 调用之间更新 device-side counter（一次 H2D scalar copy）。

**4. Async D2H 完全在 graph 外：**

```python
output = compiled_forward(token, past_kv=past)
# graph replay 完毕，staging_buffer[current_frame] 已写入

# 在独立 stream 上异步复制
with torch.cuda.stream(copy_stream):
    host_buffer.copy_(staging_buffer[current_frame], non_blocking=True)
    copy_event.record(copy_stream)

# 下一步的 compiled_forward 可以立即开始
# 因为它写入 staging_buffer[next_frame]，不冲突
```

**5. 与现有 delegate 对接：** gather 完成后，staging buffer 中的数据布局与当前 shadow block 一致（或可设计为一致），`parse_shadow_block` → `submit_step_soa` 的路径基本不变。

#### Design B + `torch.compile` 的限制

| 限制 | 说明 |
|------|------|
| 额外 D2D 带宽 | gather kernel 需要把散布的激活值聚合到连续 buffer，HBM 带宽开销正比于监控数据量 |
| Ring buffer 帧数 | 需要 ≥2 帧才能实现 overlap，显存成本 = `监控数据量 × 帧数` |
| Frame counter 同步 | 每步需要一次 H2D scalar copy 更新帧号，延迟极低但增加了 CPU-GPU 同步点 |
| Gather kernel 设计 | 需要知道每个 slot 的 `data_ptr`、`nbytes`——这些在 capture 时固定，可以 bake 进 kernel 参数 |

## 3. 总结矩阵

| | 同步 Copy (当前) | Design A (双 Graph 乒乓) | Design B (Fused Gather) |
|---|---|---|---|
| **Custom Op 兼容** | 已有 `record`/`sink` | 需要新 op `record_pingpong` | 需要新 op `gather_to_staging` |
| **torch.compile 兼容** | **高** — 加 Meta impl + 简化 hook 即可 | **低** — 需要控制两套 graph 和 allocator | **中高** — gather 自然在 graph 内，D2H 在 graph 外 |
| **改动量** | ~1 周 | 不建议走此路线 | ~2-3 周（含 gather kernel 开发） |
| **性能特征** | forward 与 copy 串行 | forward 与 copy 完全并行 | forward 与 D2H 并行，gather 串在 forward 尾部 |
| **显存开销** | 仅 metadata buffer | 2× 激活值 + metadata | staging ring × 帧数 |

## 4. 建议路线

1. **立即可做**：给 `record`/`sink` 加 Meta dispatch，简化 hook，用 `torch.compile(mode="reduce-overhead")` 替代手动 CUDA Graph。在 Llama-2/Mistral 上验证同步路径。

2. **性能不满意时**：实现 Design B 的 `gather_to_staging` custom op。因为 `torch.compile` 下地址固定，gather kernel 可以在 capture 时 bake 所有源地址，replay 时直接按固定地址批量 memcpy——这比当前逐 slot 的 `alias_tensor` + native backend copy 更高效。

3. **Design A 仅在手动 Graph 路径下考虑**——它与 `torch.compile` 抽象不兼容，但如果需要极致性能且愿意承担手动管理的复杂度，可以作为独立分支保留。

## 5. 实施结果与实验数据（2026-02-23 实施）

### 5.1 已完成的代码改动

| 文件 | 改动 |
|------|------|
| `monitoring/csrc/graph_monitor_ops.cu` | 新增 `TORCH_LIBRARY_IMPL(graphmonitor_ops, Meta, m)` — `record`/`sink` 空 Meta dispatch |
| `monitoring/graph_monitor.py` | 新增 `graph_mode="compile"` 参数；`_make_compile_hook` 直接 `sink([tensor])`；`finalize_capture` compile 模式下 no-op |
| `monitoring/graph_engine.py` | 新增 `graph_mode` 参数，透传给 `GraphMonitor` |
| `benchmark/tests/profile_decode.py` | 新增 `TorchCompileDecodeRunner`；`--monitoring-mode compile` 选项 |
| `tests_monitoring/test_graph_monitor.py` | 新增 compile 模式 metadata 正确性测试 + engine 端到端测试 |

### 5.2 遇到的问题：CUDA Graph Output 被覆盖

`torch.compile(mode="reduce-overhead")` 将编译后的函数包装为 CUDA Graph。所有输出 tensor 是 graph 拥有的**静态 buffer**——地址固定，每次 replay 写入同一块内存。

在自回归 decode 中，step N 的输出 `past_key_values` 被作为 step N+1 的输入传回。但 step N+1 的 graph replay 一开始就覆盖了这些静态 buffer，导致 read-after-write hazard：

```
compiled_forward(token, past_1)
  → graph replay → 输出写入静态 buffer [0xA, 0xB]
  → 返回 past_2（指向 0xA, 0xB 的 view）

compiled_forward(token, past_2)        ← past_2 仍指向 0xA, 0xB
  → graph replay → 覆盖 0xA, 0xB       ← past_2 内容被破坏
  → RuntimeError: accessing tensor output of CUDAGraphs
                   that has been overwritten by a subsequent run
```

**对比手动 CUDA Graph**（`HFGraphDecodeRunner`）：每步显式 `static_past.copy_(past)` 把数据拷入 graph 拥有的独立 buffer，输入输出物理分离，不存在此问题。

**修复**：每步 forward 后 clone `past_key_values`，将数据从 graph 静态 buffer 拷到独立内存。配合 `torch.compiler.cudagraph_mark_step_begin()` 标记步骤边界。

```python
def run(self, token, past):
    torch.compiler.cudagraph_mark_step_begin()
    logits, new_past = self._compiled_forward(token, past)
    cloned_past = tuple((k.clone(), v.clone()) for k, v in new_past)
    return logits, cloned_past
```

### 5.3 KV Cache 动态 shape 与 CUDA Graph 重录

`reduce-overhead` 模式下，每当 kv_seq_len 发生变化时，CUDA Graph Tree Manager 会为新 shape 录制一张新 graph。实验显示前 3 步为编译/重录（秒级），之后 Dynamo 将 seq_len 标记为 dynamic shape，不再重编译。

对比实验（GPT-2, batch=4, 10 decode steps）:

```
                        编译阶段 (step 0-2)    稳态 (step 3+)
Vanilla (无 hook)        7-10 秒/步             ~6 ms/步
With 163 hooks           3-4 秒/步              ~6 ms/步
```

**结论**：163 个 `record`+`sink` custom op hook 在 steady state 下对性能几乎无影响。

注意：CUDA Graph Tree 会为每个不同的 kv_seq_len 录制独立 graph。64 步 decode = 64 张 graph。PyTorch 警告 `51 distinct sizes`，建议 padding 输入或设置 `cudagraph_skip_dynamic_graphs=True`。

### 5.4 Benchmark 数据（GPT-2, batch=64, 64 decode steps, fp32, collect_hidden+attention）

```
监控模式: compile + sync copy
Native backend stats:
  total_steps=128, total_tasks=28220
  host_memcpy_mb=6149.3
  pool_hits=27896, pool_misses=324

hf_modified_hook_async (compile mode):
  main_duration=1.82s
  tokens/s=2246
```

### 5.5 已知限制

1. **clone 开销**：每步需要 clone 全部 KV cache（24 层 × 2 × [batch, heads, seq, dim]），额外 D2D copy
2. **graph 数量**：每个新 kv_seq_len 触发一次 graph 录制，decode 64 步 = 64 张 graph；长序列生成需要 padding 策略
3. **编译延迟**：首次运行需要 Inductor 编译（~8 秒），后续通过 `/tmp/torchinductor_*` 缓存加速

## 6. 相关文件

- 现有 custom ops：`monitoring/csrc/graph_monitor_ops.cu`
- GraphMonitor（hooks）：`monitoring/graph_monitor.py`
- GraphSafeEngine（sync path）：`monitoring/graph_engine.py`
- GraphNativeDelegate（C++ delegate）：`monitoring/csrc/graph_native_delegate.cpp`
- Shadow block parser：`monitoring/csrc/graph_shadow_parser.cpp`
- Benchmark runner：`benchmark/tests/profile_decode.py`
- 重构总计划：`docs/dev_log/graph_monitor_replan.md`
