# 方案 V2：Native Engine Partial Seal（仅 Native 内部分块排水）

## 1. Feature 说明
本方案在 Native Engine 内部实现 **partial seal / chunked drain**，目标是降低单个 step（尤其 prefill）在 step 内积压大量 hook tensor 导致的 GPU 内存尖峰与尾部清空开销。

**关键约束（已决策）**：
- 只改 Native 内部调度与排水，不改上层调用语义（`monitoring/hook_points.py` + `monitoring/engine.py`）。
- 保持“一次 forward -> 一次 `_register_db_step` -> 一次 host submit”的节奏不变。
- Host Engine API 保持不变（`submit(keys, start_token_idxs, cache_dicts)`）。

---

## 2. 要解决的问题

当前路径是“step 末尾统一 seal”：
- hook 在 step 内持续把 tensor 放进 open step。
- 只有 `end_step()` 才整体 seal + dispatch。
- prefill 时单步 hook 数很大，容易出现：
  - step 内 GPU 引用积压；
  - 运行末尾残留 pending futures 较多，需要 `resolve_all()` 清空；
  - 内存/延迟抖动明显。

V2 希望做到：
- step 尚未结束时，已积累到阈值的任务可先分块封口并异步排水；
- 但对上层调用方/Host 看起来仍是同一个 step、同一次 forward 提交。

### 2.1 流程示意（改造前 vs 改造后）

**改造前（无 partial seal）流程**：

```text
[Hook callback]
      |
      v
[append TaskEntry -> open_steps_[step_id]]
      |
      v
[继续 forward，重复 append，open 持续堆积]
      |
      v
[end_step(step_id)]
      |
      v
[seal 整个 step -> dispatch_step]
      |
      v
[worker process_step -> future ready]
      |
      v
[collect_step_futures_into + register_db_step + host submit]
```

**改造后（partial seal）流程**：

```text
[Hook callback]
      |
      v
[append TaskEntry -> open_steps_[step_id]]
      |
      v
[open_bytes >= chunk_bytes ?] --否--> [继续 forward]
      |
     是
      |
      v
[cut chunk from open_steps]
      |
      v
[dispatch_step(chunk)] -> [worker process_step] -> [future ready(部分已完成)]
      |
      v
[forward 继续]
      |
      v
[end_step(step_id)]
      |
      v
[seal tail chunk + dispatch]
      |
      v
[collect_step_futures_into + register_db_step + host submit(仍一次)]
```

**不同策略如何控制“水位”**（流程视角）：

```text
dispatch_step(chunk)
      |
      v
[cap_enabled ?]
  | 否
  v
[best-effort: 按原队列策略 enqueue/inline]
  |
  +---> [队列满则 inline process，未满则异步队列]

  | 是
  v
[计算 allowed_inflight]
  - hard_cap = total_mem * cap_ratio
  - effective_free = driver_free + reclaimable(reserved-allocated) - driver_guard
  - allowed = min(hard_cap-allocated, effective_free)
      |
      v
[inflight + chunk_bytes > allowed ?]
  | 是 -> [inline process_step（立即排水/回压）]
  | 否 -> [enqueue + inflight_bytes += chunk_bytes]
```

### 2.2 控制面差异图（旧：仅 step 级；新：可控 open-step）

```text
旧方案（仅 step seal）:

Hook -> open_steps(持续累积) -> end_step seal -> dispatch -> process
                 ^                               ^
                 |                               |
           无法在这里控水位                 只能在这里做拥塞控制
           （step内不可排）                （已到step末尾）


新方案（partial seal）:

Hook -> open_steps(累积) --(chunk阈值触发)--> cut chunk -> dispatch -> process
                 ^                 ^                  ^
                 |                 |                  |
            可观测open bytes   可在step内提前排水     仍保留原有队列/inline控制
            可做step内控制      （不必等end_step）
```

要点：
- 旧方案：congestion control 只能作用在 `dispatch_step`（step 末尾），无法限制 step 内峰值增长。
- 新方案：通过 chunk 触发，把控制点前移到 open-step 阶段；既能保留原队列控制，也能在 step 内排水。
- 这就是“现在可以控制 open step 里的积压”的本质。

---

## 3. 我们的决策

### 决策 A：不改 Host/上层外部 API
- 不改 `monitoring/engine.py` 的 `_register_db_step` 与 `_submit_pending_db_step` 调用关系。
- 不改 `monitoring/hook_points.py` 的 `collect_step_futures_into(step_id, cache_dict)` 调用时机（仍在 forward 结束后）。
- Host 仍消费 `BackendFuture`，不感知 chunk 细节。

### 决策 B：partial seal 只在 Native 内部实现
- 一个逻辑 step（`step_id`）内部拆成多个 chunk。
- chunk 到阈值就 seal+dispatch（内部行为），剩余任务在 `seal_step(step_id)` 时做 final seal。
- 对外仍只有一个 step_id，不引入 step_id 子编号到上层接口。

### 决策 C：优先“快速排水”，不改变 DB 粒度
- DB 粒度仍是 forward 粒度（一次 `_register_db_step`）。
- partial seal 仅用于降低中间积压与尾部堵塞，不改变行级语义。

### 决策 D：加开关，支持无 cap 的 best-effort
- partial seal 与 congestion cap 解耦，不强绑定。
- 开关统一走 `MonitoringConfig`，不使用环境变量控制该 feature。
- cap 采用 **GPU 总显存百分比** 控制（例如 `0.8`），语义是“目标水位不超过总显存的 80%”。
- 运行模式分三类：
  1) `partial_seal=off`：回到当前行为（step 末尾统一 seal）。
  2) `partial_seal=on, cap=off`：chunk 排水 + best-effort（不做内存预算限流）。
  3) `partial_seal=on, cap=on`：chunk 排水 + 内存预算限流。
- 默认建议：先 `partial_seal=on, cap=off`，先验证正确性和收益，再打开 cap。

---

## 4. 实现概览（高层）

### 4.1 数据流（逻辑）
1. hook 回调进入 Native，向当前 `open_steps_[step_id]` 追加任务。  
2. 若 `open_step_bytes >= chunk_bytes_threshold`，将这批任务切成一个 sealed chunk，立即进入 dispatch 路径。  
3. forward 结束时调用 `seal_step(step_id)`，把最后未达阈值的尾块封口并 dispatch。  
4. `collect_step_futures_into(step_id, cache_dict)` 仍在 forward 末尾统一收集 name->future。  
5. 上层一次 `_register_db_step`，随后一次 host submit。  

### 4.2 对外兼容性
- `BackendFuture` 类型不变。
- Host stage `ProcessFuture` 调用 `future.result()` 方式不变。
- `DMXHostEngine.submit` 入参形态不变。
- 说明：Host Engine/Future 已是 C++ 实现；这里的“上层”仅指调用编排仍在 Python 侧。

---

## 5. 代码改动点（全部列出，供 review）

> 说明：以下是计划改动点；优先保持 public API 不变。

### 5.0 `monitoring/config.py` + `monitoring/engine.py`（配置入口）
新增 runtime 配置，统一由 `MonitoringConfig` 下发：

```python
@dataclass
class NativePartialSealConfig:
    enabled: bool = True
    chunk_bytes: int = 64 * 1024 * 1024
    cap_enabled: bool = False
    cap_ratio: float = 0.8       # 0~1，按总显存比例限流
    driver_guard_mb: int = 1024  # 预留给 driver/其他分配的安全余量（MB）

    def __post_init__(self):
        if not (0.0 < self.cap_ratio <= 1.0):
            raise ValueError("cap_ratio must be in (0, 1].")
        if self.driver_guard_mb < 0:
            raise ValueError("driver_guard_mb must be >= 0.")

@dataclass
class MonitoringConfig:
    hooks: HookSelection = ...
    schedule: CaptureSchedule = ...
    native_partial_seal: NativePartialSealConfig = field(default_factory=NativePartialSealConfig)
```

`MonitoringEngine` 初始化 native backend 后调用：

```python
def _apply_native_runtime_config(self):
    cfg = self.config.native_partial_seal
    self._native_backend.set_partial_seal_config(
        bool(cfg.enabled),
        int(cfg.chunk_bytes),
        bool(cfg.cap_enabled),
        float(cfg.cap_ratio),
        int(cfg.driver_guard_mb),
    )
```

说明：该 feature 的开关/参数不再依赖 `MON_*` 环境变量。

### 5.1 `monitoring/csrc/native_engine_internal.h`
新增/调整内部结构与状态（示意）：

```cpp
struct StepWork {
  int64_t step_id;
  std::vector<TaskEntry> tasks;
  cudaEvent_t event{nullptr};
  int64_t bytes{0};          // 新增：当前 work 的估算字节
  bool final_chunk{false};   // 新增：是否该 step 的最终 chunk
};

// open step 聚合状态（示意）
struct OpenStepState {
  int64_t step_id{0};
  std::vector<TaskEntry> tasks;
  int64_t bytes{0};
};

std::unordered_map<int64_t, OpenStepState> open_steps_;

// 来自 MonitoringConfig 的运行时参数
int64_t partial_seal_chunk_bytes_{64 * 1024 * 1024}; // 默认 64MB（示例）
bool partial_seal_enabled_{true};

// congestion cap（可选，允许关闭走 best-effort）
bool congestion_cap_enabled_{false};     // 默认先关
double congestion_cap_ratio_{0.8};       // 0~1，按总显存比例
int64_t congestion_cap_bytes_{0};        // 运行时计算：total_mem * ratio
int64_t driver_guard_bytes_{1024ll * 1024ll * 1024ll}; // driver_guard_mb 转换后
std::atomic<int64_t> inflight_bytes_{0};
```

同时在 native API 增加配置入口（示意）：

```cpp
// native_engine.h
void set_partial_seal_config(bool enabled,
                             int64_t chunk_bytes,
                             bool cap_enabled,
                             double cap_ratio,
                             int64_t driver_guard_mb);
```

`bindings.cpp` 暴露同名 pybind 方法，由 `MonitoringEngine` 调用一次下发。

### 5.2 `monitoring/csrc/hooks.cpp`
在任务 append 路径上增加“阈值触发分块封口”逻辑（`append_hook_current_step` / `add_task_from_config` / `append_hook` 共用 helper）：

```cpp
void append_task_and_maybe_partial_seal(step_id, TaskEntry entry) {
  lock(staging_mutex_);
  auto& st = open_steps_[step_id];
  st.tasks.push_back(std::move(entry));
  st.bytes += estimate_task_bytes(st.tasks.back()); // tensor.nbytes()

  if (partial_seal_enabled_ && st.bytes >= partial_seal_chunk_bytes_) {
    StepWork chunk = take_all_tasks_as_chunk(st);  // st.tasks 清空, st.bytes=0
    unlock(staging_mutex_);
    dispatch_step(std::move(chunk));               // 立即排水
    return;
  }
  unlock(staging_mutex_);
}
```

**锁开销关注（先记录，不作为本轮实现目标）**：
- 该路径按设计会“每次 hook 回调进入一次短临界区”，即每 hook 都有一次 `staging_mutex_` 的 lock/unlock。
- 在 full hooks + 大 batch 场景下，这可能成为 CPU 热点（锁竞争/缓存抖动）。
- 本轮实现先保持该直观方案，优先保证正确性与语义稳定；后续再评估优化（如 TLS 局部聚合后批量 flush、分片锁等）。

### 5.3 `monitoring/csrc/api_submit.cpp`
调整 `seal_step(step_id, ...)` 语义：只负责把当前 open step 的“尾块”封口，而不是假设整个 step 只 seal 一次。

```cpp
void seal_step(step_id, stream_handle) {
  StepWork tail;
  bool has_tail = pop_open_step_tail(step_id, &tail);
  if (has_tail) {
    tail.final_chunk = true;
    attach_event_if_needed(tail, stream_handle);
    dispatch_step(std::move(tail));
  } else {
    // 空 step 仍保序（保持现有行为）
    sync_event_if_needed(stream_handle);
  }
}
```

`resolve_all()` 继续保证：
- 清空 open steps（包括未到阈值的尾块）；
- 清空 sealed/queue；
- 等待 `pending_tasks_ == 0`。

### 5.4 `monitoring/csrc/engine_core.cpp`
在 `dispatch_step/process_step` 路径上支持更高频的 chunk dispatch（逻辑不变，次数增加）：

```cpp
void dispatch_step(StepWork&& work) {
  // 沿用现有队列/inline策略；区别仅在 work 变成 chunk 级
  // 若 cap 关闭，直接走 best-effort（仅受原队列策略约束）
  if (!congestion_cap_enabled_) {
    enqueue_or_inline_by_existing_policy(std::move(work));
    return;
  }

  // cap 打开：刷新预算（同时考虑 allocator 可复用显存）
  // driver: total/free 来自 cudaMemGetInfo
  // alloc:  allocated/reserved 来自 CUDACachingAllocator 统计（若可用）
  maybe_refresh_memory_stats();
  int64_t hard_cap = int64_t(total_mem_bytes * congestion_cap_ratio_);
  int64_t reclaimable = std::max<int64_t>(reserved_bytes - allocated_bytes, 0);
  int64_t effective_free = std::max<int64_t>(driver_free_bytes + reclaimable - driver_guard_bytes_, 0);
  int64_t allowed_inflight = std::max<int64_t>(
      std::min<int64_t>(hard_cap - allocated_bytes, effective_free), 0);

  // cap 打开时，再叠加内存预算判断
  if (inflight_bytes_.load(std::memory_order_relaxed) + work.bytes > allowed_inflight) {
    process_step(std::move(work));   // inline 快速排水
  } else {
    inflight_bytes_.fetch_add(work.bytes, std::memory_order_relaxed);
    queue_.push_back(std::move(work));
    queue_cv_.notify_one();
  }
}
```

并在 `process_step` 结束时回收 `inflight_bytes_`（仅 cap 模式需要）。

### 5.5 （可选）`monitoring/csrc/native_engine.cpp` / `monitoring/csrc/bindings.cpp`
如需 debug 能力，可选新增 `debug_state()` 输出：
- `open_steps`
- `pending_tasks`
- `queue_size`
- `current_step_id`
- `partial_seal_dispatch_count`

> 该项可选，不影响主流程。

### 5.6 明确不改的文件（本方案范围外）
- `monitoring/engine.py`（不改 DB 注册/提交节奏）
- `monitoring/hook_points.py`（不改 forward 后 collect/register 的时机）
- `monitoring/csrc/dmx_host_engine.h`、`monitoring/csrc/future_process.cpp`（host API/消费逻辑不改）

---

## 6. 伪代码：端到端行为（保持外部语义不变）

```python
# 上层调用侧（不改语义）
engine.start_step(...)
model_out, cache = model.run_with_cache(...)
# forward 内部：native 可能多次 partial seal + dispatch（上层不感知 chunk 细节）
engine._register_db_step(cache, input_ids, past_key_values)  # 一次
engine.end_step()                                             # 一次（seal 尾块）
# 内部仍由 host_engine.submit([key], [start_idx], [cache]) 一次提交
```

```cpp
// Native 侧（新增 chunk 化）
on_hook_tensor(step_id, tensor):
  append task into open_steps_[step_id]
  if open_bytes >= chunk_threshold:
    chunk = cut_chunk(open_steps_[step_id])
    dispatch_step(chunk)              // partial seal

seal_step(step_id):
  if open_steps_[step_id] has tail:
    dispatch_step(tail_chunk_final)   // final seal
```

---

## 7. 开销分析

### 新增开销
1. **更频繁的 dispatch/queue 操作**  
   - chunk 越小，dispatch 次数越多；队列锁竞争会增加。

2. **step 内字节统计**  
   - 每次 append 读取 `tensor.nbytes()` 并累加，CPU 开销小但频率高。

3. **事件管理（若每 chunk 挂 event）**  
   - 需控制 event 创建/同步次数，避免过细粒度导致额外开销。

4. **内存统计采样**
   - cap 开启时会读取 driver + allocator 统计，增加少量 CPU 开销。
   - 相比 D2H/序列化/DB IO 通常很小，不是主开销来源。

### 预期收益
1. **降低 step 内峰值积压**  
   - 不必等到 step 末尾才统一排水。

2. **减少尾部 `resolve_all()` 压力**  
   - 运行中已持续排水，结束时 backlog 更小。

3. **更稳定的延迟曲线**  
   - 大 step 被拆散，减少“单次大抖动”。

### 参数取舍建议
- `chunk_bytes_threshold` 默认建议：32MB~128MB（先用 64MB 起步）。
- 太小：调度开销上升；太大：接近原行为，收益有限。
- `cap` 建议默认关闭，先跑 best-effort；确认稳定后再开启。
- `cap_ratio` 建议默认 `0.8`，可在 `0.7~0.9` 间调优。
- `driver_guard_mb` 建议默认 `1024`（1GB），显存紧张机器可降到 `512`。
- 参数位置：`MonitoringConfig.native_partial_seal`（而不是环境变量）。

---

## 8. 风险与限制

1. **不改变 forward 粒度 DB 语义**  
   - 如果未来要做 chunk 粒度 DB 写入，需要单独方案（不在本 V2 范围）。

2. **step_name_tokens 映射必须保持完整**  
   - 即使任务提前 dispatch，也必须保证 `collect_step_futures_into(step_id, ...)` 能收齐该 step 的 future 引用。

3. **高并发下锁竞争**  
   - `staging_mutex_` 可能成为热点，需要在实现里减少持锁区（先搬运后解锁再 dispatch）。

4. **allocator 统计可用性**
   - 若某些构建/版本下 allocator 详细统计不可用，需要降级为仅用 driver 指标（更保守）。

---

## 9. 验证计划（review 后实施）

1. **功能正确性**
- 与当前基线对比：生成文本一致（同 seed、同参数）。
- DB 行数/主键范围一致（仍按 forward 粒度推进 start/end token）。

2. **稳定性**
- 长序列 + 大 batch 下验证无悬挂 future、`resolve_all()` 可退出。
- `close()` 前后 `pending_tasks==0`。

3. **性能**
- 记录 `end_step` wall time、尾部 drain 时间、GPU 峰值占用。
- 对比不同 chunk 阈值（32/64/128MB）。
- 对比 `cap=off` vs `cap=on` 两组行为，确认开销与收益边界。

4. **配置开关验证**
- `partial_seal=off` 时行为回归当前基线。
- `partial_seal=on, cap=off` 时应为 best-effort，不触发 cap 相关路径。
- `partial_seal=on, cap=on` 时 cap 生效且不会破坏正确性。

---

## 10. 总结

V2 的核心是：**只在 Native 内部分块排水，外部 API 和上层/Host 语义保持不变**。  
这样可以在不破坏现有 pipeline 的前提下，优先解决 step 内积压和尾部清空压力，适合作为可控、低风险的下一步重构。
