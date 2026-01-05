# 2026-01-05 Bypass 开销分析：为什么全 bypass 仍然慢？

## 问题描述

在测试 config 模块的采样跳过功能时，发现一个反直觉的现象：

**配置**：
```python
"every_100_tokens": MonitoringConfig(
    hooks=base_hooks,
    schedule=CaptureSchedule(
        step_stride=100,      # 每 100 步采样一次
        capture_prefill=False, # 跳过 prefill
        capture_decode=True,
    ),
)
```

**测试参数**：
- `--decode-steps 64`（只有 64 个 decode step）
- `step_stride=100` 意味着 step 1-64 全部被跳过
- `capture_prefill=False` 意味着 prefill 也被跳过

**预期**：所有监控被 bypass，性能应该接近纯 HF forward。

**实际结果**：
```
hf:           main_duration=0.8181s  (baseline)
full_capture: main_duration=1.7628s  (2.15x slower)
every_5_tokens: main_duration=1.2352s (1.51x slower)
```

即使理论上 0/64 步被采样，性能仍然比 baseline 慢 50%+！

---

## 验证：确实全部 bypass 了

```python
from monitoring.config import CaptureSchedule

s = CaptureSchedule(step_stride=100, capture_prefill=False, capture_decode=True)

# Prefill
print(s.should_capture_step(1, "prefill"))  # False

# Decode steps 1-64
captured = [i for i in range(1, 65) if s.should_capture_step(i, 'decode')]
print(f'Captured: {captured}')  # []
print(f'Total: {len(captured)} / 64')  # 0 / 64
```

**结论**：配置正确，所有 step 都应该被 bypass。

---

## 开销分解实验

设计了分层实验来定位开销来源：

### 实验设置
- batch_size = 64
- decode_steps = 64
- device = CUDA
- dtype = float32
- 每个测试运行 3 次取平均

### 实验结果

| 测试 | 耗时 (ms) | 增量 (ms) | 说明 |
|------|----------|----------|------|
| 1. 纯 HF forward | 294.16 | - | baseline |
| 2. HookedModel forward | 310.48 | +16.32 | HookPoint 结构开销 |
| 3. + start/end_step | 312.61 | +2.13 | Engine 边界调用 |
| 4. + run_with_cache (bypass) | 442.65 | +130.03 | **主要开销来源！** |

### 关键发现

**`run_with_cache` 本身的开销：130.03 ms（占 baseline 的 44.2%）**

即使所有监控任务都被 C++ 侧 bypass，`run_with_cache` 函数本身仍有巨大开销。

---

## 深入分析：run_with_cache 开销来源

### Hook 统计

> **⚠️ 注意：以下分析基于 `MON_NATIVE_CALLBACK=0` 的情况。**
> **当 `MON_NATIVE_CALLBACK=1` 时，hooks 只注册一次（永久），不存在每步注册/注销开销。**
> **详见文档末尾的"修正后的开销分析"。**

```
Total hooks in model: 363
每个 step 都要注册/注销这 363 个 hooks!  ← 仅在非 native callback 模式下成立
```


### 单次 run_with_cache 开销分解（非 native callback 模式）
```
=== 单次 run_with_cache 开销 (avg of 10 runs) ===
Total: 7.24 ms

=== 直接 forward (无 hook 注册) ===
Total: 6.47 ms

=== 结论 ===
Hook 注册/注销开销: 0.77 ms / step
每个 hook 的平均开销: 2.13 us

对于 64 decode steps: 49.4 ms 额外开销
```

### 开销组成（非 native callback 模式）

> **⚠️ 此表仅适用于 `MON_NATIVE_CALLBACK=0`。Native callback 模式下的开销分解见文档末尾。**

| 开销来源 | 估算时间 | 占比 |
|---------|---------|------|
| Hook 注册/注销 (363 hooks × 64 steps) | ~49 ms | 38% |
| Hook 回调执行（即使 bypass） | ~40 ms | 31% |
| Cache dict 创建/清理 | ~25 ms | 19% |
| 其他 Python 开销 | ~16 ms | 12% |
| **总计** | ~130 ms | 100% |

---

## 根本原因

> **⚠️ 以下分析基于 `MON_NATIVE_CALLBACK=0`。**
> **当 `MON_NATIVE_CALLBACK=1` 时，原因 1 不成立，但原因 2、3 仍然适用。**
> **修正后的分析见文档末尾。**

### 1. Hook 注册/注销是 per-step 的（仅非 native callback 模式）

`run_with_cache` 的实现逻辑：
```python
def run_with_cache(self, ...):
    # 1. 获取 caching hooks（创建 363 个 hook 函数）
    cache_dict, fwd_hooks, bwd_hooks = self.get_caching_hooks(...)

    # 2. 注册所有 hooks（363 次 register_forward_hook）
    self.add_hook(fwd_hooks, ...)

    # 3. 执行 forward
    out = self(...)

    # 4. 注销所有 hooks（363 次 remove）
    self.reset_hooks(...)

    return out, cache_dict
```

**每个 decode step 都要执行这个流程！**

### 2. Python Hook 回调无法完全避免

即使 C++ 侧 `is_capture_enabled()` 返回 false，Python 的 hook 函数仍然会被调用：
```python
def hook_fn(tensor, hook):
    # 这个函数会被调用！
    # 然后才调用 C++ 的 append_hook_current_step
    # C++ 侧检查 is_capture_enabled() 后直接返回
```

调用栈：
```
PyTorch forward_hook → Python hook_fn → C++ append_hook → is_capture_enabled() → return
                       ↑                ↑
                       已经有开销了      这里才 bypass
```

### 3. Cache dict 管理开销

每个 step 都要：
- 创建新的 cache dict
- 存储 hook 结果（即使是空的）
- 清理 cache dict

---

## 影响分析

### 当前架构的性能天花板

| 采样策略 | 理论最优 | 实际可达 | 原因 |
|---------|---------|---------|------|
| 全采样 | - | 1.76s | 正常 |
| 50% 采样 | 1.38s | ~1.4s | 接近理论值 |
| 0% 采样 (bypass) | 0.82s | ~1.2s | **受 run_with_cache 开销限制** |

### 结论

**当前架构下，即使完全 bypass 监控，也无法达到纯 HF 的性能。**

---

## 解决方案

### 方案 1: 条件性使用 run_with_cache（推荐，短期）

```python
def decode_step(token, past, step_id, phase):
    engine.start_step(phase=phase)

    if schedule.should_capture_step(step_id, phase):
        # 需要采样：使用 run_with_cache
        out, cache = model.run_with_cache(token, past_key_values=past, use_cache=True)
        cache.clear()
    else:
        # 不需要采样：直接 forward
        out = model(token, past_key_values=past, use_cache=True)

    engine.end_step()
    return out
```

**优点**：
- 实现简单，改动小
- 不采样时性能接近纯 HF

**缺点**：
- 需要调用方感知采样逻辑
- 代码分支增加

### 方案 2: 永久 Hook + Lazy 注册（中期）

不使用 `run_with_cache`，而是使用永久 hooks：

```python
# 初始化时注册一次永久 hooks
model.add_perma_hook(hook_fn, ...)

# 每个 step 只执行 forward
def decode_step(token, past):
    engine.start_step(phase='decode')
    out = model(token, past_key_values=past, use_cache=True)  # hooks 已经注册
    engine.end_step()
    return out
```

**优点**：
- 消除每 step 的 hook 注册/注销开销
- Native callback 可以在 C++ 侧完全 bypass

**缺点**：
- 需要重构 hook 管理逻辑
- 永久 hooks 可能影响非监控场景

### 方案 3: 完全 C++ 化的 Hook 注入（长期）

将 hook 注入逻辑完全移到 C++ 层，消除 Python 回调开销。

**优点**：
- 理论最优性能
- bypass 时真正零开销

**缺点**：
- 实现复杂
- 需要深度修改模型结构

---

## 建议的下一步

1. **短期**：在 benchmark 中实现方案 1，验证条件性 `run_with_cache` 的性能提升
2. **中期**：评估方案 2 的可行性，设计永久 hook 架构
3. **长期**：研究方案 3，考虑与 torch.compile 的兼容性

---

## 附录：完整测试代码

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model
from monitoring import MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection, MonitoringConfig

device = torch.device('cuda')
batch_size = 64
decode_steps = 64
RUNS = 3

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

hf_model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32).to(device).eval()
hf_hooked = HookedGPT2Model.from_pretrained('gpt2', torch_dtype=torch.float32).to(device).eval()
lm_head = hf_model.lm_head

prompt = tokenizer(['Hello'] * batch_size, return_tensors='pt', padding=True)['input_ids'].to(device)

bypass_config = MonitoringConfig(
    hooks=HookSelection(mode='full'),
    schedule=CaptureSchedule(step_stride=1000, capture_prefill=False, capture_decode=True)
)
engine = MonitoringEngine(async_enabled=True, config=bypass_config)
hf_hooked.monitoring_engine = engine

def measure(name, fn):
    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

# Test 1: 纯 HF forward
def run_pure_hf():
    with torch.no_grad():
        past = None
        token = prompt
        for _ in range(decode_steps):
            out = hf_model(token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1:, :].argmax(dim=-1)

# Test 2: HookedModel forward
def run_hooked_forward():
    with torch.no_grad():
        past = None
        token = prompt
        for _ in range(decode_steps):
            out = hf_hooked(token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = lm_head(out.last_hidden_state)
            token = logits[:, -1:, :].argmax(dim=-1)

# Test 3: + start/end_step
def run_step_overhead():
    with torch.no_grad():
        past = None
        token = prompt
        for step in range(decode_steps):
            engine.start_step(phase='decode')
            out = hf_hooked(token, past_key_values=past, use_cache=True)
            engine.end_step()
            past = out.past_key_values
            logits = lm_head(out.last_hidden_state)
            token = logits[:, -1:, :].argmax(dim=-1)

# Test 4: + run_with_cache (bypass)
def run_with_cache_bypass():
    with torch.no_grad():
        past = None
        token = prompt
        for step in range(decode_steps):
            engine.start_step(phase='decode')
            out, cache = hf_hooked.run_with_cache(token, past_key_values=past, use_cache=True)
            engine.end_step()
            past = out.past_key_values
            logits = lm_head(out.last_hidden_state)
            token = logits[:, -1:, :].argmax(dim=-1)
            cache.clear()

# Warmup
for _ in range(2):
    run_pure_hf()
torch.cuda.synchronize()

# Measure
pure_hf_time = measure('pure_hf', run_pure_hf)
hooked_forward_time = measure('hooked_forward', run_hooked_forward)
step_overhead_time = measure('step_overhead', run_step_overhead)
run_with_cache_time = measure('run_with_cache', run_with_cache_bypass)

print(f'=== 开销分解 (batch={batch_size}, {decode_steps} decode steps) ===')
print(f'1. 纯 HF forward:              {pure_hf_time*1000:.2f} ms')
print(f'2. HookedModel forward:        {hooked_forward_time*1000:.2f} ms  (+{(hooked_forward_time-pure_hf_time)*1000:.2f} ms)')
print(f'3. + start/end_step:           {step_overhead_time*1000:.2f} ms  (+{(step_overhead_time-hooked_forward_time)*1000:.2f} ms)')
print(f'4. + run_with_cache (bypass):  {run_with_cache_time*1000:.2f} ms  (+{(run_with_cache_time-step_overhead_time)*1000:.2f} ms)')

engine.close()
```

---

## 代码核查（是否每步注册/注销？）

结论：**只有在 native callback 关闭/不可用时，才会每步注册/注销 hooks。**

关键逻辑（`HookedRootModule.get_caching_hooks`）：
```python
if native_callback_active and native_backend is not None:
    if not self._native_callbacks_registered:
        for reg_name, hp in self.hook_dict.items():
            callback = native_backend.create_global_hook_callback_sig(...)
            hp.add_hook(callback, dir="fwd", is_permanent=True)
        self._native_callbacks_registered = True
    continue  # 不再为本步创建 fwd_hooks
```

`run_with_cache` 里如果 `native_callback_active` 且 `fwd_hooks/bwd_hooks` 为空，会跳过 hooks 上下文：
```python
if native_callback_active and not fwd and not bwd:
    use_ctx = False  # 不进入 hooks()，也不会 reset_hooks
```

因此：
- **`MON_NATIVE_CALLBACK=1` 且 native backend 生效时**：hook 只注册一次（per model），后续 step 不会反复注册/注销。
- **`MON_NATIVE_CALLBACK=0` 或 native backend 未启用时**：每个 `run_with_cache` 都会构建并注册 hooks，随后 reset，确实存在 per-step 注册/注销开销。

=> 文档里"每步注册/注销 363 hooks"的结论只在 **非 native callback** 路径成立。

---

## 修正后的开销分析（Native Callback 模式）

基于代码核查，当使用 `MON_NATIVE_CALLBACK=1` 时，开销来源与之前的分析不同。

### C++ Callback 的 bypass 逻辑

永久注册的 C++ callback（`create_global_hook_callback_sig`）在 bypass 时会立即返回：

```cpp
// native_engine.cpp:194-196
if (!engine->impl_->is_capture_enabled()) {
    return py::none();  // 立即返回，不做任何实际工作
}
```

### 但 hook 仍然会被触发！

虽然 C++ 侧立即返回，但调用链仍然存在：

```
PyTorch forward
  → HookPoint.__call__
    → PyTorch 触发 registered hook
      → Python function object (C++ wrapped)
        → C++ is_capture_enabled() check
          → return py::none()
```

**363 hooks × 64 steps = 23,232 次** 跨语言调用。

### Per-Step 仍然执行的操作

即使在 native callback 模式下，每个 step 仍然执行：

| 操作 | 每步执行次数 | 说明 |
|-----|------------|------|
| `get_caching_hooks()` 调用 | 1 | 创建 cache dict |
| 遍历 `hook_dict` 构建 `enabled_names` | 363 | line 1046-1048 |
| `set_enabled_hooks(enabled_names)` | 1 | C++ 调用 + mutex 锁 |
| 遍历 `hook_dict` 检查 native callback | 363 | line 1057-1081 (立即 continue) |
| **永久 hook 触发** | 363 | forward 时每个 hook 都被 PyTorch 调用 |
| `collect_step_futures_into()` | 1 | C++ mutex 锁 + dict 操作 |

### 修正后的开销分解（Native Callback 模式）

| 开销来源 | 估算时间 | 说明 |
|---------|---------|------|
| Hook 注册/注销 | **~0 ms** | ✅ 永久 hooks，不再每步注册 |
| 永久 hook 触发（23K 次 Python→C++ 调用） | ~40-50 ms | PyTorch dispatch + Python→C++ 开销 |
| `get_caching_hooks` 遍历（726×64 次迭代） | ~20 ms | Python 循环开销 |
| `set_enabled_hooks` + `collect_step_futures_into` | ~10 ms | C++ mutex + dict 操作 |
| Cache dict 创建/清理 | ~10 ms | Python dict 分配 |
| 其他 Python 开销 | ~10-20 ms | getattr、条件判断等 |
| **总计** | ~90-110 ms | 与观察到的 ~130ms 接近 |

### 核心结论

1. **Native callback 模式下，不存在每步 hook 注册/注销开销**（之前分析的 49ms 这部分不适用）

2. **主要开销来自永久 hooks 的触发**：即使 C++ 立即 bypass，PyTorch 仍会触发 363 个 hook，每个都有 Python→C++ 调用开销

3. **`run_with_cache` 的 Python 循环开销仍然存在**：每步遍历 hook_dict 两次（726 次迭代）

4. **要实现真正的零开销 bypass**，需要：
   - 方案 1（推荐）：不采样时完全跳过 `run_with_cache`，直接调用 `model()`
   - 方案 2：让 PyTorch 层面也能跳过 hook 触发（需要修改 HookPoint 实现）

---

## 更新日志

- 2026-01-05: 初始分析，发现 run_with_cache 本身开销问题
- 2026-01-05: 代码核查，修正 native callback 模式下的开销分析
