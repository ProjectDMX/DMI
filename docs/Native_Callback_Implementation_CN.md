# Native Callback 实现说明

## 概述

Native Callback 是监控引擎的最新优化，通过将 PyTorch Hook 回调从 Python 层完全下沉到 C++ 层，消除了 94,000+ 次 Python→C++ 边界跨越，显著降低了主线程开销。

**性能进展**：
- 初始异步版本：1.53s
- add_task 优化：1.09s
- SoA 批量优化：1.03s
- append_hook 优化：0.97s
- **Native Callback**：**0.8s** ✨ (当前)

距离 no-hook baseline (0.42s) 还有 **0.38s (48%)** 差距。

---

## 核心设计

### 1. 架构概览

```
PyTorch Forward Pass
    ↓
PyTorch Hook Registry (注册时)
    ↓
create_hook_callback() → py::cpp_function (C++ lambda)
    ↓
Forward 触发 Hook (运行时)
    ↓
C++ lambda 直接执行 (no Python!)
    ↓
append_hook_current_step() → open_steps_[step_id]
    ↓
seal_step() → 分发到后台线程处理
```

**关键突破**：Hook 回调不再进入 Python 解释器，直接在 C++ 中完成。

---

## 核心实现

### 2. `create_hook_callback()` - Hook 注册

位置: `monitoring/csrc/native_engine.cpp`（薄封装）

```cpp
py::object create_hook_callback(const std::string& hook_name,
                                bool remove_batch_dim,
                                py::object pos_slice,
                                py::object target_device) {
    // 1. 构建并缓存 HookConfig（避免每次回调重新解析）
    auto config = std::make_unique<HookConfig>();
    config->name = hook_name;
    config->pos_dim = deduce_pos_dim(hook_name);  // q/k/v/z→-3, 其他→-2
    config->remove_batch_dim = remove_batch_dim;
    config->slice = parse_slice_py(std::move(pos_slice));

    // 2. 存入预分配池（裸指针引用，零拷贝）
    HookConfig* cfg_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(hook_config_mutex_);
        auto& entry = hook_configs_[hook_name];
        if (!entry) entry = std::make_unique<HookConfig>();
        *entry = std::move(*config);
        cfg_ptr = entry.get();
    }

    // 3. 返回 C++ lambda（直接注册给 PyTorch）
    auto engine = shared_from_this();
    return py::cpp_function(
        [engine, cfg_ptr](py::args args, py::kwargs /*kwargs*/) -> py::object {
            if (args.size() == 0) {
                throw std::runtime_error("Native callback expected tensor argument");
            }
            at::Tensor tensor = args[0].cast<at::Tensor>();
            auto t0 = std::chrono::steady_clock::now();
            {
                // ✅ 释放 GIL，让 CUDA 操作不阻塞主线程
                py::gil_scoped_release release;
                if (tensor.requires_grad()) {
                    tensor = tensor.detach();
                }
                engine->append_hook_current_step(*cfg_ptr, std::move(tensor));
            }
            auto t1 = std::chrono::steady_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            engine->record_callback_duration(us);
            return py::none();
        });
}
```

**关键优化**：
- ✅ **HookConfig 预分配池**：每个 hook_name 只解析一次配置，后续直接引用
- ✅ **py::cpp_function**：返回 C++ 可调用对象，PyTorch 直接调用 C++ lambda
- ✅ **GIL 释放**：`py::gil_scoped_release` 让 CUDA 操作与主线程并行
- ✅ **统计回调耗时**：`record_callback_duration(us)` 累积到 `stats_callback_us_`

---

### 3. `append_hook_current_step()` - Hook 聚合

位置: `monitoring/csrc/hooks.cpp`

```cpp
void append_hook_current_step(const HookConfig& cfg, at::Tensor tensor) {
    TaskSpec spec;
    spec.tensor = std::move(tensor);
    spec.slice_dim = cfg.pos_dim;
    spec.remove_batch_dim = cfg.remove_batch_dim;
    spec.slice = cfg.slice;
    spec.target_device = cfg.target_device;

    // 计算 can_slice（是否满足切片条件）
    int64_t dim = spec.slice_dim;
    int64_t tensor_dims = spec.tensor.dim();
    spec.can_slice = (dim >= 0) ? (tensor_dims > dim) : (tensor_dims >= -dim);

    // 读取当前 step_id（原子变量，线程安全）
    int64_t step_id = current_step_id_.load(std::memory_order_acquire);

    // 追加到 open_steps_（无需 Python 跨语言开销）
    std::lock_guard<std::mutex> lock(staging_mutex_);
    StepWork& work = open_steps_[step_id];
    work.step_id = step_id;
    TaskEntry entry;
    entry.spec = std::move(spec);
    entry.token = 0;  // 延迟分配（seal_step 时分配）
    work.tasks.emplace_back(std::move(entry));
    pending_tasks_.fetch_add(1, std::memory_order_relaxed);
}
```

**关键特性**：
- ✅ **无 pybind 开销**：完全在 C++ 内部操作，不跨语言边界
- ✅ **延迟分配 token**：避免在 hot path 创建 ResultSlot
- ✅ **原子读取 step_id**：`std::memory_order_acquire` 保证内存顺序

---

### 4. Python 集成

**位置**: `monitoring/engine.py:77-80, 152-153`

```python
# 初始化时
native_backend.begin_step(int(self._current_step_id))

# start_step 时更新 step_id
def start_step(self):
    self._current_step_id += 1
    if backend is not None:
        backend.begin_step(int(self._current_step_id))
```

**位置**: `transformers/src/transformers/models/gpt2_p/hook_points.py` (集成示例)

```python
# 注册时使用 native callback
if using_native_callback:
    hook_fn = native_backend.create_hook_callback(
        hook_name,
        remove_batch_dim,
        pos_slice,
        device if device is not None else None
    )
    handle = hook_point.register_forward_hook(hook_fn, prepend=prepend)
    cache[hook_name] = None  # 占位
```

---

## 性能分析

### 开销分解（当前 0.8s vs 0.42s baseline）

| 组件 | 耗时 | 说明 |
|------|------|------|
| **No-hook baseline** | **0.42s** | 完全无监控的纯前向时间 |
| Native callback 主线程 | 0.8s | 当前实现 |
| → PyTorch hook dispatch | ~30-50ms | PyTorch 框架级 hook 调度（94k 次函数调用） |
| → C++ callback 执行 | ~5-10ms | C++ lambda 执行（GIL 已释放） |
| → Tensor 引用计数 | ~20-30ms | 94k 次 tensor detach/move |
| → 带宽竞争 | ~100-150ms | GPU 同时执行前向 + 抓取激活 |
| → CUDA 流切换 | ~20-30ms | 主流与缓存流的切换开销 |
| → 其他（调度/同步）| ~50-100ms | 剩余未归类开销 |

**剩余 0.38s 差距主要来源**：
1. **PyTorch Hook 机制本身**（30-50ms）：即使是 C++ callback，PyTorch 仍需调度 94k 次函数
2. **Tensor 引用计数**（20-30ms）：每次 detach/move 都有原子操作
3. **GPU 带宽竞争**（100-150ms）：前向计算 + 激活抓取同时进行
4. **CUDA 流开销**（20-30ms）：流同步与切换
5. **其他开销**（50-100ms）：内存分配、GC 压力等

---

## 环境变量

```bash
# 启用 native callback（默认已启用）
MON_NATIVE_CALLBACK=1

# 查看回调统计
MON_ENGINE_STATS=1

# 调试模式
MON_DEBUG=1
```

---

## 性能对比

```bash
# Baseline（无 hook）
hf_modified: 0.42s

# 同步 hook
hf_modified_hook: 1.06s (+154%)

# 异步优化历程
- 初始异步: 1.53s (+264%)
- add_task: 1.09s (+160%)
- SoA: 1.03s (+145%)
- append_hook: 0.97s (+131%)
- Native Callback: 0.8s (+90%) ✨ 当前

# 统计输出
[Native/Stats] {'total_steps': 1, 'total_tasks': 47190,
                'submit_us': 15000, 'process_us': 13000,
                'callback_us': 8000}  # callback_us 从 107ms → 8ms
```

---

## 已知限制与未来优化

### 当前限制
1. **PyTorch Hook 调度开销**（30-50ms）：框架级限制，无法通过 callback 优化消除
2. **GPU 带宽竞争**（100-150ms）：前向计算与激活抓取共享带宽
3. **Tensor 生命周期管理**：94k 次 tensor 引用累积在 `open_steps_`，可能导致内存压力

### 未来优化方向

#### 优先级 1: 批量 Hook 聚合
**目标**：减少 PyTorch hook 调用次数从 94k → 几百次

```cpp
// 当前：每层每 token 位置一个 hook (94k 次)
hook_q, hook_k, hook_v, hook_z, hook_resid_pre, ...

// 优化：每层一个聚合 hook (几十次)
layer_hook(outputs_dict) {
    for (auto& [name, tensor] : outputs_dict) {
        append_hook_current_step(cfg[name], tensor);
    }
}
```

**预期收益**：30-50ms → 5-10ms

---

#### 优先级 2: 完全异步采集（双 GPU 方案）
**目标**：消除 GPU 带宽竞争

```
GPU 0: 纯前向计算（不抓激活）
   ↓ NVLink
GPU 1: 异步抓取激活并处理
```

**预期收益**：100-150ms → <10ms

---

#### 优先级 3: 流式处理（减少内存占用）
**目标**：不保存完整 tensor，立即处理并只保存统计量

```cpp
// 当前：保存完整 tensor 直到 process_step
spec.tensor = std::move(tensor);  // 可能 3MB × 47k = 100+GB

// 优化：立即计算统计量
compute_statistics(tensor);  // 只保存标量（mean/std/...）
```

**收益**：降低内存占用，避免 OOM

---

## 调试技巧

### 1. 查看回调统计
```bash
MON_ENGINE_STATS=1 python benchmark/tests/profile_decode.py
```

输出示例：
```
[Native/Stats] {'callback_us': 8000, 'submit_us': 15000, ...}
```

### 2. 验证 GIL 释放效果
使用 `py-spy` 或 `cProfile` 查看主线程是否被 GIL 阻塞：
```bash
py-spy top --pid <PID> --gil
```

### 3. GPU Timeline 分析
使用 Nsight Systems 查看 GPU 流调度：
```bash
nsys profile --trace=cuda,nvtx python benchmark/tests/profile_decode.py
```

---

## 总结

Native Callback 通过以下手段将异步性能从 0.97s 提升到 0.8s：

1. ✅ **消除 Python→C++ 边界**：Hook 回调直接在 C++ 执行
2. ✅ **GIL 释放**：CUDA 操作不持有 GIL，与主线程并行
3. ✅ **HookConfig 预分配**：避免每次回调重新解析配置
4. ✅ **统计可观测性**：`callback_us` 精确追踪回调开销

距离 no-hook baseline 还有 0.38s (48%) 差距，主要来自 PyTorch Hook 机制本身和 GPU 带宽竞争，需要更激进的架构重构（批量 hook、双 GPU 等）才能进一步缩小。

---

**相关文档**：
- `ENGINE_OVERVIEW.md` - 整体架构与演进历程
- `MonitoringEngine_Implementation_CN.md` - 实现细节
- `MONITORING_ENGINE_PLAN.md` - 未来规划
