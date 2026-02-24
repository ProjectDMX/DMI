# 提案：基于 GPU 内存的 Native Engine 拥塞控制

## 问题
当前拥塞控制依赖固定 queue size：满了就同步 `process_step`，不满就异步入队。这个策略非常依赖手工调参，无法与真实 GPU 内存压力对齐，容易出现要么 OOM、要么过度保守的问题。

## 目标
用**GPU 内存预算驱动的动态回压**替代（或增强）固定 queue size，使入队逻辑与实际 GPU 内存压力挂钩。

---

## 核心思路
维护一个**in‑flight 字节计数**（队列中 + 正在处理的 GPU tensor 总字节数），并动态计算一个**内存预算**（基于 `cudaMemGetInfo` 的 free/total）。

当 `inflight + step_bytes` 超过预算时：
- 方案 A：阻塞等待（直到低水位）
- 方案 B：直接 inline `process_step`（当前的降级行为）

**首版选择方案 B**，改动小、行为可预测。

---

## 代码组织（避免改动集中在主文件）
为了避免 `engine_core.cpp` 过于臃肿，新增一组 helper 放到 utils 文件里：
- 新文件建议：`monitoring/csrc/engine_utils.h` / `monitoring/csrc/engine_utils.cpp`
- 负责：
  - `refresh_budget()`
  - `calc_step_bytes()`
  - `should_backpressure()`

`engine_core.cpp` 只保留调用逻辑，不塞大量实现细节。

---

## 具体改动点（文件 + 伪代码）

### 1) 维护 in‑flight 计数
**文件**：`monitoring/csrc/native_engine_internal.h` / `monitoring/csrc/engine_core.cpp`

新增字段：
```cpp
// native_engine_internal.h
std::atomic<int64_t> inflight_gpu_bytes_{0};
std::atomic<int64_t> inflight_pinned_bytes_{0}; // 可选

int64_t mem_budget_bytes_{0};
int64_t mem_budget_low_watermark_bytes_{0};
float mem_budget_ratio_{0.6f}; // 默认 0.6
```

### 2) 计算每个 step 的字节数
**文件**：`monitoring/csrc/engine_utils.*`

```cpp
int64_t calc_step_bytes(const StepWork& work) {
    int64_t step_bytes = 0;
    for (auto& entry : work.tasks) {
        if (entry.spec.tensor.defined() && entry.spec.tensor.is_cuda()) {
            step_bytes += entry.spec.tensor.nbytes();
        }
    }
    return step_bytes;
}
```

### 3) 动态预算（cudaMemGetInfo）
**文件**：`monitoring/csrc/engine_utils.*`

```cpp
void refresh_budget() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    int64_t dyn_budget = static_cast<int64_t>(free_bytes * mem_budget_ratio_);
    if (mem_budget_bytes_ > 0) dyn_budget = std::min(dyn_budget, mem_budget_bytes_);
    mem_budget_high_ = dyn_budget;
    mem_budget_low_ = static_cast<int64_t>(dyn_budget * 0.8); // hysteresis
}
```

### 4) 入队前回压判断
**文件**：`monitoring/csrc/engine_core.cpp`（`dispatch_step`）

```cpp
if (mem_budget_enabled) {
    int64_t projected = inflight_gpu_bytes_ + work.step_bytes;
    if (projected > mem_budget_high_) {
        // 方案 B：inline process（当前行为）
        process_step(std::move(work));
        return;
    }
}

// 正常入队
inflight_gpu_bytes_.fetch_add(work.step_bytes);
queue_.push_back(std::move(work));
```

在 `process_step` 结束后释放：
```cpp
inflight_gpu_bytes_.fetch_sub(work.step_bytes);
```

### 5) 配置项 / 环境变量
**文件**：`monitoring/csrc/engine_core.cpp`

新增 env：
- `MON_NATIVE_MEM_BUDGET_ENABLE`（默认 1）
- `MON_NATIVE_MEM_BUDGET_RATIO`（默认 0.6）
- `MON_NATIVE_MEM_BUDGET_MB`（硬上限）
- `MON_NATIVE_MEM_BUDGET_LOW_WATERMARK`（可选）

伪代码：
```cpp
if (const char* v = std::getenv("MON_NATIVE_MEM_BUDGET_ENABLE")) enable_budget_ = (*v != '0');
if (const char* v = std::getenv("MON_NATIVE_MEM_BUDGET_RATIO")) mem_budget_ratio_ = std::atof(v);
if (const char* v = std::getenv("MON_NATIVE_MEM_BUDGET_MB")) mem_budget_bytes_ = std::atoll(v) * 1024 * 1024;
```

---

## 开销分析

### 运行时新增开销
1) **每步统计 step_bytes**
   - 复杂度：O(#tasks)
   - 代价：只遍历 `work.tasks` 并读取 `tensor.nbytes()`，通常可忽略。

2) **cudaMemGetInfo 调用**
   - 如果每步调用：有轻微开销（微秒级），但在高步频模型上会被放大。
   - 建议：每 N 步刷新一次（例如 N=8/16），减少开销。

3) **原子计数更新**
   - `inflight_gpu_bytes_` 的 `fetch_add/fetch_sub` 为原子操作，开销极低。

4) **回压路径 inline process**
   - 当触发时，会将负载转为同步处理，带来主线程延迟增加。
   - 这是主动降速，换取稳定性与避免 OOM。

### 性能影响预期
- 正常负载：吞吐影响极小。
- 高压力时：主线程延迟会上升（有意识的 backpressure）。

---

## 行为总结
- 正常负载：继续异步入队，不影响吞吐。
- 高压时：触发 inline `process_step`，避免内存爆炸。
- 动态预算：根据 GPU 可用内存自适应，不再猜 queue size。

---

## 风险 / 取舍
- `cudaMemGetInfo` 是瞬时值，且受 allocator 缓存影响，需要保守系数。
- 高压时 inline 可能提升主线程延迟。
- `step_bytes` 仅统计输入 tensor，不含内部临时内存。

---

## 验证计划
1) 高负载场景下对比 OOM/稳定性。
2) 对比吞吐与延迟变化。
3) 记录 `inflight_gpu_bytes` 与预算阈值，确认回压触发时机。

