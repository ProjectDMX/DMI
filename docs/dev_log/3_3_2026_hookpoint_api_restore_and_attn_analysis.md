# HookPoint API 兼容性修复 & Attention 实现分析

**日期**: 2026-03-03
**分支**: feature/native-monitoring
**上下文**: 合并 `performant-monitor` 到 `dmx/HF_Prometheus` 后 11 个测试失败

---

## 1. 问题根因

合并后 GPT-2 forward 代码为支持 CUDA Graph 监控，将所有 `self.hook_xxx(tensor)` 替换为 `_mon_record()` 内联写 GPU shadow buffer，但：

1. **缺少 `else` 分支** — 非监控模式下（`_mon_buf is None`），HookPoint 完全不被调用
2. **`mod_dict` 未归一化** — `_normalize_hook_names()` 只处理了 `hook_dict`，`mod_dict` 保留 `transformer.` 前缀
3. **`test_graph_delegate.py` 属性名过期** — `_host_buffer` 已重构为 `_host_template` / `_latest_snapshot`
4. **`monitored_forward` 递归** — `run_with_cache` → `self()` → `monitored_forward` → `run_with_cache` 无限递归

## 2. 修复内容

### 2.1 Fix 1: 添加 `else` 分支调用 `monitor_activation()`

**文件**: `transformers/src/transformers/models/gpt2_p/modeling_gpt2.py`

在所有 20 个 `_mon_record` 调用点添加 `else` 分支：

```python
# 修改前
if _mon is not None:
    _mon_record(tensor, _mon, slot_id + _off, _anch)
# tensor 不经过 HookPoint → run_with_cache 返回空 cache

# 修改后
if _mon is not None:
    _mon_record(tensor, _mon, slot_id + _off, _anch)
else:
    tensor = self.hook_xxx.monitor_activation(tensor)
```

**为什么用 `monitor_activation()` 而非 `self.hook_xxx(tensor)`:**

| 方法 | Python hooks (run_with_cache) | Native 监控 (_monitor_handle) |
|------|:---:|:---:|
| `self.hook_xxx(tensor)` → `HookPoint.__call__` | ✅ 触发 register_forward_hook | ❌ 不检查 _monitor_handle |
| `self.hook_xxx.monitor_activation(tensor)` | ✅ 通过 `super().__call__()` | ✅ 检查并调用 |

`prepare_for_model()` 默认使用 inline monitoring handle（`MON_INLINE_HOOK=1`），设置 `_monitor_handle` 后直接 return，不注册 forward hook。只有 `monitor_activation()` 会检查 `_monitor_handle`。

`monitor_activation` 源码：
```python
def monitor_activation(self, tensor):
    has_python_hooks = bool(self.fwd_hooks)
    if has_python_hooks:
        tensor = super().__call__(tensor)  # 触发 PyTorch forward hooks
    handle = getattr(self, "_monitor_handle", None)
    if handle is None:
        return tensor
    tensor = monitor_native.monitor_activation(tensor, handle)  # native 监控
    return tensor
```

无监控时 `_monitor_handle is None` 且 `fwd_hooks` 为空 → 直接返回 tensor（等同 identity），无副作用。

**20 个调用点分布:**
- `eager_attention_forward`: 2（hook_attn_scores, hook_pattern）
- `GPT2Attention.forward`: 8（hook_k ×2, hook_v ×2, hook_q, hook_z, hook_result, non-eager hook_pattern）
- `GPT2Block.forward`: 8（hook_resid_pre, hook_ln1, hook_attn_out, hook_resid_mid, hook_ln2, hook_mlp_in, hook_mlp_out, hook_resid_post）
- `GPT2Model.forward`: 3（hook_embed, hook_pos_embed, hook_final_ln）

### 2.2 Fix 2: `mod_dict` 归一化

**文件**: `transformers/src/transformers/models/gpt2_p/modeling_gpt2.py`

`HookedGPT2Model._normalize_hook_names()` 和 `HookedGPT2LMHeadModel._normalize_hook_names()`:

```python
# 修改前：只归一化 hook_dict
normalized: dict[str, HookPoint] = {}
for name, hook_point in list(self.hook_dict.items()):
    ...
self.hook_dict = normalized
# mod_dict 仍有 transformer.blocks.0.attn.hook_q → KeyError

# 修改后：同时归一化 mod_dict
normalized_hooks: dict[str, HookPoint] = {}
normalized_mods: dict[str, nn.Module] = {}
for name, hook_point in list(self.hook_dict.items()):
    ...
    normalized_hooks[name] = hook_point
    normalized_mods[name] = hook_point
for old_name, mod in list(self.mod_dict.items()):
    short = old_name[len("transformer."):] if old_name.startswith("transformer.") else old_name
    if short.startswith("h."):
        continue
    if short not in normalized_mods:
        normalized_mods[short] = mod
self.hook_dict = normalized_hooks
self.mod_dict = normalized_mods
```

### 2.3 Fix 3: `test_graph_delegate.py`

- `monitor._host_buffer` → `monitor._latest_snapshot`（`metadata_view()` 读取的是 `_latest_snapshot`，不是 `_host_template`）
- 补充 `test_delegate_submits_tasks` 中缺失的 `metadata = monitor.metadata_view()` 变量定义

### 2.4 Fix 4: `monitored_forward` 重入保护

**文件**: `monitoring/generate.py`

```python
def monitored_forward(*f_args, **f_kwargs):
    # 检测重入：run_with_cache → self() → monitored_forward 回调
    fwd_fn = f_kwargs.pop("forward_fn", None)
    if fwd_fn is not None:
        return fwd_fn(*f_args, **f_kwargs)  # 直接调用原始 forward，断开递归
    # ... 正常路径
```

调用链：
```
generate() → model() → monitored_forward()
  → engine.start_step()
  → model.run_with_cache(forward_fn=orig_forward, **kwargs)
    → run_with_cache 注册 hooks，然后 self(**model_kwargs)
      → monitored_forward(forward_fn=orig_forward, ...)  ← 重入
        → 检测到 forward_fn，直接调用 orig_forward()   ← 断开递归
  → engine.end_step()
```

## 3. 测试结果

| | 修复前 | 修复后 |
|---|---|---|
| 通过 | 98 | 108 |
| 失败 | 11 | 1 |

剩余 1 个失败：`test_generate_and_forward_collect_cpp_futures_and_consume_results` — native backend `add_task()` 创建的 `BackendFuture` 在 `end_step()` 后不 resolve。属于 C++ backend pipeline 的预存问题，与 HookPoint API 无关。

## 4. sdpa vs eager Attention 实现分析

### 4.1 HF 的 `_attn_implementation` 自动设置

```python
GPT2Config()                                    → _attn_implementation = None
                                                    ↓ Model.__init__()
                                                  autoset → "sdpa"

GPT2Config(attn_implementation="eager")          → _attn_implementation = "eager" ✅
config.attn_implementation = "eager"  # 事后设置  → _attn_implementation = None   ❌ 无效
config._attn_implementation = "eager" # 私有属性  → _attn_implementation = "eager" ✅
```

**关键**：`PreTrainedModel.__init__()` 中 `_autoset_attn_implementation()` 在 `_attn_implementation is None` 时自动设为 `"sdpa"`。事后设置 public 属性不影响 private 属性。

### 4.2 对 hook 的影响

```
GPT2Attention.forward():
    using_eager = (config._attn_implementation == "eager")

    if using_eager:
        attention_interface = eager_attention_forward  ← hook_attn_scores/pattern 在这里
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]  ← 融合 kernel，无中间张量
```

| 配置 | hook_dict | 实际可 cache | 差异 |
|------|-----------|-------------|------|
| `_attn_implementation = "eager"` | 185 | **185** | 0 |
| `_attn_implementation = "sdpa"` (默认) | 185 | 161 | -24 (12层 × hook_attn_scores + hook_pattern) |

缺失的 24 个 hook 不是 bug — sdpa/FlashAttention 在 kernel 内部分块计算 attention，中间的 scores/pattern 从未作为独立张量存在于显存。

### 4.3 性能对比

#### 纯 attention 算子（B=4, H=12, D=64, fp16）

| seq_len | eager | sdpa | 加速比 | attention matrix |
|---:|---:|---:|---:|---:|
| 8 | 0.057ms | 0.012ms | 4.6x | 0.0 MB |
| 256 | 0.059ms | 0.013ms | 4.6x | 6 MB |
| 1024 | 0.691ms | 0.085ms | **8.1x** | 96 MB |
| 4096 | 10.8ms | 1.3ms | **8.3x** | 1.5 GB |

FlashAttention 优势来自：
- **内存**: eager 物化完整 `[B, H, N, N]` attention matrix（O(N²)），FlashAttention 分块计算（O(N)）
- **带宽**: eager 反复读写 HBM，FlashAttention 在 SRAM 内完成

#### 整模型 + torch.compile（2层小模型, seq_len=8）

| 配置 | 速度 | hook 数 |
|------|---:|---:|
| sdpa + compile | 0.18 ms | 161 |
| eager + compile | 0.19 ms | 185 |
| 差异 | +3.3% | +24 |

差异仅 3.3% 因为 seq_len=8 时 attention matrix 只有 8×8=64 元素，attention 在整个 forward 里占比极小。在实际推理场景（seq_len=2048+）下差距会显著增大。

### 4.4 结论

- `eager + torch.compile` 完全兼容，Inductor 会自动融合 eager 算子
- 如需完整 185 个 hook（含 attn_scores/pattern），使用 `GPT2Config(attn_implementation="eager")` 或 `config._attn_implementation = "eager"`
- 如不需要 attn_scores/pattern，保持 sdpa 默认（161 hooks）性能更优
- 现有 benchmark 和测试代码已正确使用 `attn_implementation="eager"`

## 5. 修改的文件清单

| 文件 | 修改 |
|------|------|
| `transformers/src/transformers/models/gpt2_p/modeling_gpt2.py` | 20 处 `else` 分支 + 2 处 `_normalize_hook_names` |
| `monitoring/generate.py` | `monitored_forward` 重入保护 |
| `tests/test_graph_delegate.py` | `_host_buffer` → `_latest_snapshot` + 补 `metadata` 变量 |
