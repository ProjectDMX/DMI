# Barrier & Skip-Step Monitoring Design

**Date**: 2026-02-28
**Status**: Design discussion (not yet implemented)
**Depends on**: Design C dual_compile (2_27_2026_dual_graph_shadow_buffer_design.md)

## Background

Design C Phase 2 实现了 dual_compile 双帧 ping-pong 流水线。当前行为：每步都 monitor，frame 按 `step_id % 2` 交替。

nsys profiling 显示 forward/D2H 已实现 GPU 并行（从 5.7ms → 3.37ms/step），但当 D2H 时间 > forward 时间时，barrier 仍会造成等待。本文讨论 **skip-step monitoring**：不是每步都 monitor，通过跳步给 D2H 更多时间完成，减少或消除 barrier 开销。

## 问题分析

### 当前流水线 (monitor_interval=1)

```
step 0 (f0): forward → end_step: defer D2H
step 1 (f1): start_step: D2H f0 starts (overlap w/ fwd 1)
             forward → end_step: defer D2H
step 2 (f0): start_step: D2H f1 starts (overlap w/ fwd 2)
             barrier: wait D2H f0 (started at step 1)
             forward → ...
```

Barrier at step t+2 等待 step t 的 D2H。从 D2H 提交到 barrier 检查，只经过 ~1 个 forward 的时间。如果 D2H > 1×fwd，每步都有 barrier 开销。

### 实测数据 (GPT-2, batch=64, 183 hooks)

| 阶段 | 耗时 |
|------|------|
| Forward | ~1.8ms |
| D2H (183 alias copies) | ~3-5ms |
| D2H / Forward 比值 | ~2-3× |

## Skip-Step 核心思想

**不是每步都 monitor**。设 `monitor_interval = I`，每 I 步 monitor 一次。非 monitor 步的 forward 仍然执行（推理不能跳），但不触发 D2H。

关键收益：D2H 有更多步的时间完成 → barrier 等待减少或消除。

## Frame 分配：`_mon_frame` Flip 逻辑

### 错误方案：所有 monitored steps 用同一个 frame

```
step 0 (f0, mon): D2H f0
step 1 (f1, skip)
step 2 (f0, mon): D2H f0 ← 覆盖了 step 0 正在 D2H 的数据！数据竞争！
```

### 错误方案：`step_id % 2`

```
step 0 (f0, mon): D2H f0
step 1 (f1, skip)
step 2 (f0, mon): D2H f0 ← step 2 % 2 = 0 = f0，同上问题
```

### 正确方案：`_mon_frame` 在 monitored step 的 end_step 翻转

状态变量 `_mon_frame` 初始为 0。**所有 step 都用 `_mon_frame` 作为 frame**。
Monitored step 结束时翻转 `_mon_frame = 1 - _mon_frame`。

```python
self._mon_frame = 0  # init

# 每步 start_step:
frame = self._mon_frame
set_frame(frame)

# monitored step 的 end_step:
defer_d2h(self._mon_frame)
self._mon_frame = 1 - self._mon_frame  # FLIP
```

### Trace: interval=2

```
_mon_frame = 0

step 0 (f0, mon):  frame=0. end: defer D2H f0, flip → _mon_frame=1
step 1 (f1, skip): frame=1. start: trigger D2H f0. end: nothing.
step 2 (f1, mon):  frame=1. end: defer D2H f1, flip → _mon_frame=0
step 3 (f0, skip): frame=0. start: trigger D2H f1, barrier on f0. end: nothing.
step 4 (f0, mon):  frame=0. end: defer D2H f0, flip → _mon_frame=1
step 5 (f1, skip): frame=1. start: trigger D2H f0, barrier on f1. end: nothing.
...

Frame 序列: f0, f1, f1, f0, f0, f1, f1, f0, ...
```

关键性质：
- **Monitored steps 交替 frame**：mon#0=f0, mon#1=f1, mon#2=f0, ...
- **Non-monitored steps 用翻转后的 frame**（= 下一个 monitored step 的 frame），不干扰正在 D2H 的 frame
- **interval=1 时退化为 `step_id % 2`**：每步都是 monitored，每步都 flip → 0,1,0,1... 完全向后兼容

## D2H 提交时机

D2H 不在 monitored step 本身提交，而是 **defer 到下一个 step 的 `start_step`**。原因：

```
CPU timeline:
  end_step(t, mon):     defer only (不提交GPU work)
  start_step(t+1):      record pre_fwd_event
                         submit D2H to copy_stream (wait pre_fwd_event)
                         submit forward(t+1) to fwd_stream

GPU timeline:
  copy_stream:  wait pre_fwd_event → D2H copies
  fwd_stream:   forward(t+1)
  两个 stream 在 GPU 上并行执行
```

如果在 end_step(t) 中提交 D2H，此时 forward(t) 刚完成，D2H 跑在 copy_stream 但 fwd_stream 空闲（forward(t+1) 还没提交）。**没有 overlap**。

Defer 到 start_step(t+1) 后，D2H 和 forward(t+1) 几乎同时提交到 GPU，实现真正的流并行。

## Barrier 逻辑

### 何时需要 barrier

当 `start_step(t)` 的 frame 有正在进行的 D2H（`_d2h_in_flight[frame] = True`），需要等 D2H 完成才能让 forward 写入该 frame。

```python
if self._d2h_in_flight[frame]:
    fwd_stream.wait_event(self._d2h_events[frame])  # GPU-side wait, CPU不阻塞
    self._d2h_in_flight[frame] = False
```

### 事件顺序约束

`pre_fwd_event` 必须在 barrier **之前** 录制：

```python
# 正确顺序:
pre_fwd_event.record(fwd_stream)     # ① 标记 forward(t-1) 完成点
submit_d2h_to_copy_stream()          # ② copy_stream.wait_event(pre_fwd_event)
fwd_stream.wait_event(d2h_event)     # ③ barrier

# 如果 ③ 在 ① 之前:
# pre_fwd_event 会包含 barrier 等待时间
# → copy_stream 间接等待自己的 D2H 完成 → 死锁或串行化
```

### Barrier 耗时分析

D2H of frame f 提交时间 = start_step(t+1)，其中 step t 是 monitored step using frame f。
下次使用 frame f = start_step(t + 2*I)（跳过 I 步到下一个 monitored step using 另一个 frame，再 I 步回到 frame f）。

GPU 时间差（D2H 提交到 barrier 检查）≈ `2*I - 1` 个 forward。

```
Zero-barrier 条件: 2*I - 1 >= D2H/fwd
即: I >= (D2H/fwd + 1) / 2
```

| D2H / fwd | 最小 interval (zero barrier) | Barrier at I=1 | Barrier at I=2 |
|-----------|------------------------------|----------------|----------------|
| 1×        | 1                            | 0              | 0              |
| 2×        | 1.5 → 2                      | ~1×fwd         | 0              |
| 3×        | 2                            | ~2×fwd         | ~1×fwd (边界)  |
| 4×        | 2.5 → 3                      | ~3×fwd         | ~1×fwd         |
| 6×        | 3.5 → 4                      | ~5×fwd         | ~3×fwd         |

注：barrier 在边界条件（2*I-1 = D2H/fwd）时是否为零取决于 GPU 调度精度和 stream 并行效率。实际中建议 interval 取上界。

## 完整 Pipeline 伪代码

```python
class GraphSafeEngine:
    def __init__(self, ..., monitor_interval=1):
        # 新增状态
        self._monitor_interval = monitor_interval
        self._mon_frame = 0                       # ping-pong flag
        self._pending_d2h = False
        self._pending_d2h_frame = 0
        self._d2h_in_flight = {0: False, 1: False}

    def start_step(self):
        self._current_step_id += 1
        step = self._current_step_id

        if not (self._graph_mode == "dual_compile" and self._dual_frame_ready):
            return  # 非 dual_compile 走原有逻辑

        # ---- frame assignment ----
        frame = self._mon_frame
        self._monitor.set_frame(frame)

        fwd_stream = torch.cuda.current_stream(self._device)

        # ① Record pre_fwd_event BEFORE barrier
        #    标记 fwd_stream 当前位置 = forward(t-1) 完成点
        self._pre_fwd_event.record(fwd_stream)

        # ② Trigger deferred D2H (from previous monitored step)
        #    submit to copy_stream → overlap with upcoming forward
        if self._pending_d2h:
            d2h_frame = self._pending_d2h_frame
            self._copy_stream.wait_event(self._pre_fwd_event)
            with torch.cuda.stream(self._copy_stream):
                for sid, alias in self._frame_aliases[d2h_frame].items():
                    self._pinned_buffers[d2h_frame][sid].copy_(
                        alias, non_blocking=True
                    )
                self._d2h_events[d2h_frame].record(self._copy_stream)
            self._d2h_in_flight[d2h_frame] = True
            self._pending_d2h = False

        # ③ Barrier: protect frame data from being overwritten during D2H
        if self._d2h_in_flight[frame]:
            fwd_stream.wait_event(self._d2h_events[frame])
            self._d2h_in_flight[frame] = False

        # forward(t) 由 caller 执行 (compiled_forward)

    def end_step(self):
        step = self._current_step_id

        if not (self._graph_mode == "dual_compile" and self._dual_frame_ready):
            # ... 原有 end_step 逻辑 ...
            return

        is_monitored = (step % self._monitor_interval) == 0

        if is_monitored:
            # Defer D2H to next step's start_step for overlap
            self._pending_d2h = True
            self._pending_d2h_frame = self._mon_frame
            # FLIP: next step (and all subsequent non-monitored steps)
            # will use the other frame
            self._mon_frame = 1 - self._mon_frame
        # else: nothing — non-monitored steps don't trigger D2H

    def collect_dual_frame_results(self, *, wait=False):
        """Collect the most recently completed D2H frame."""
        if not self._dual_frame_ready:
            return None
        # 找最近完成的 D2H frame
        for frame in (0, 1):
            if self._d2h_in_flight[frame]:
                if wait:
                    self._d2h_events[frame].synchronize()
                elif not self._d2h_events[frame].query():
                    continue
                self._d2h_in_flight[frame] = False
                return {
                    sid: buf.clone()
                    for sid, buf in self._pinned_buffers[frame].items()
                }
        return None
```

## 与 Dynamo Guard 的兼容性

`_mon_frame_offset` 取值仍然只有两种：`0` 和 `num_slots`。Dynamo guard 会为这两个值各 trace 一次，产生两个 CUDA Graph（graph A 和 graph B）。

Skip-step 不改变 guard 值的集合，只改变切换时机：
- interval=1: A B A B A B ... (每步切换)
- interval=2: A B B A A B B A ... (每两步切换)
- interval=3: A B B B A A A B B B ... (每三步切换)

所有情况下，GPU replay 的是同样的两个 graph，只是 replay 顺序不同。Dynamo 兼容性无问题。

## 实测建议

1. 先跑 nsys 确认当前 D2H/fwd 比值（目前估计 2-3×）
2. 设 `monitor_interval=2` 测试，预期 barrier 从 ~1-2×fwd 降到 ~0-1×fwd
3. 设 `monitor_interval=3` 应该能完全消除 barrier
4. 观察吞吐量变化：跳步意味着部分 step 没有 activation 数据，需要权衡监控精度

## 与当前代码的差异

| 方面 | 当前 (interval=1) | Skip-step (interval>1) |
|------|-------------------|----------------------|
| Frame 分配 | `step_id % 2` | `_mon_frame` flip at monitored steps |
| D2H 触发 | 每步 end_step | 只在 monitored step 后的第一个 step |
| Barrier 检查 | `step >= 3` 硬编码 | `_d2h_in_flight[frame]` flag |
| D2H 提交位置 | `_dual_frame_end_step` (end_step) | `start_step` (for overlap) |
| Flip 时机 | 隐式 (`step % 2`) | 显式 (`_mon_frame = 1 - _mon_frame`) |
| interval=1 行为 | — | 完全等价（backward compat） |

## Per-Hook Barrier：在 CUDA Graph 内部分布式 wait

### 动机

单一 barrier（forward 开头等所有 D2H 完成）浪费时间：D2H 按 slot 0→182 顺序 copy，forward hook 也按 0→182 顺序执行。如果 barrier 分布到每个 hook（或每个 layer），前面的 hook 不需要等后面 slot 的 D2H。

当 D2H_remaining ≤ fwd 时，per-hook barrier 可以把 stall 从 `D2H_remaining` 降到 **0**（D2H 尾部被 forward 计算完全掩盖）。

### 关键发现：`cudaEventWaitExternal`

在 CUDA Graph capture 期间，对外部 event（其他 stream 上 record 的）调用 `cudaStreamWaitEvent` 必须传 `cudaEventWaitExternal` (0x2) flag：

| 调用方式 | Capture 行为 | Replay 行为 |
|---------|-------------|-------------|
| `flags=0`, 外部 event | **非法，报错** | N/A |
| `cudaEventWaitExternal`, 外部 event | 捕获为 event wait node | **每次 replay 检查 event 当前状态** |
| 未 record / 已 complete 的 event | — | 立即通过 (no-op) |

PyTorch 自己的 `CUDAEvent::block()` 做了同样的 capture 状态检测。

### PoC 实现

新增 4 个 custom ops (`graph_monitor_ops.cu`):

```c++
init_d2h_events(int num_events)         // 预创建 cudaEventDisableTiming events
wait_d2h(Tensor(a!) buf, int slot_id)   // cudaStreamWaitEvent + cudaEventWaitExternal
record_d2h_event(int slot_id)           // cudaEventRecord on current stream
destroy_d2h_events()                    // cleanup
```

`wait_d2h` 的核心逻辑：

```c++
void wait_d2h_op(const at::Tensor& buf, int64_t slot_id) {
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus status;
    cudaStreamIsCapturing(stream, &status);
    unsigned int flags = (status != cudaStreamCaptureStatusNone)
        ? cudaEventWaitExternal : 0;
    cudaStreamWaitEvent(stream, g_d2h_events[slot_id], flags);
}
```

Forward 中在 `record()` 前调用：

```python
ops.wait_d2h(buf, slot + offset)    # barrier: 等 D2H 完成
ops.record(tensor, buf, slot + offset)  # 写 metadata
```

Meta dispatch 为 no-op → Dynamo 正常 trace。`Tensor(a!)` 防 DCE。

### PoC 验证结果 (4/4 PASS)

测试文件：`tests_monitoring/test_wait_d2h_poc.py`

| 测试 | 结果 | 说明 |
|------|------|------|
| `test_wait_d2h_compile_and_capture` | PASS | Dynamo → Inductor → CUDA Graph capture 成功 |
| `test_wait_d2h_no_hang` | PASS | 未 record 的 event，100 次 replay 共 7.8ms，零 stall |
| `test_wait_d2h_blocks_on_pending` | PASS | 挂大量 D2H 后 replay → **295.7× 减速**，wait 确实生效 |
| `test_per_slot_d2h_pipeline` | PASS | 双帧 20 步 per-slot event pipeline，D2H 数据全部正确 |

关键验证：
- `cudaEventWaitExternal` 在 CUDA Graph capture 中被正确接受
- Graph replay 检查 event 当前状态（非 capture 时状态）
- 未 record 的 event 立即通过（skip monitoring 自然支持）
- Per-slot event record + wait 与 alias D2H pipeline 正确集成

### 实现方案对比

| 方案 | Stall (D2H_rem=1×fwd) | 实现复杂度 | overhead |
|------|----------------------|-----------|----------|
| 单一 barrier (当前) | 1×fwd | 已实现 | 0 |
| Per-layer barrier (12 events) | ~0.08×fwd | 中 | ~1.2μs |
| Per-hook barrier (183 events) | ~0 | 高 | ~18μs |
| interval+1 (无代码改动) | 0 | 零 | 0 (降低监控频率) |

Per-layer barrier 是性价比最高的方案：12 个 event + barrier point 就能接近 zero stall，不牺牲监控频率。

### 潜在风险

1. **Inductor kernel 重排**：`Tensor(a!)` 标注建立数据依赖链 `wait_d2h(buf) → record(buf) → wait_d2h(buf)`，应阻止重排，需实测确认
2. **D2H 与 forward hook 顺序一致性**：当前 dict 遍历 = slot 分配顺序 = `named_modules()` 顺序，一致；但需确保 Inductor 不改变执行顺序
3. **Event 生命周期**：events 必须在 CUDA Graph 销毁前存活；在 `finalize_dual_frame` 创建，`engine.close()` 销毁

## 后续

- [x] PoC 验证 `cudaEventWaitExternal` 在 torch.compile CUDA Graph 中工作
- [x] 实现 `_mon_frame` + `_monitor_interval` + `_pending_d2h` 逻辑
- [x] 统一 interval=1 和 interval>1 的代码路径（用同一套 flip 逻辑替代 `step_id % 2`）
- [ ] 添加 `--monitor-interval` CLI 参数到 benchmark
- [ ] 集成测试：interval=1,2,3 各跑一遍，验证 D2H 数据正确性
- [ ] nsys 验证不同 interval 下的 barrier 行为
- [ ] 评估 per-layer barrier 集成（在 GPT2Block.forward 入口加 wait_d2h）
- [ ] 对比 per-layer barrier vs interval 调参的实际吞吐差异
