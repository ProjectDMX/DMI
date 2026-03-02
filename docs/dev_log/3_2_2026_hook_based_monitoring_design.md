# Hook-Based Monitoring Design

**Date**: 2026-03-02
**Status**: Validated (4/4 PoC tests pass)
**Replaces**: Inline `_mon_record()` approach in `modeling_gpt2.py`

## 1. Motivation

当前 inline 方案在模型代码中嵌入了大量监控逻辑：

```python
# 当前：GPT2Block.forward() 中 ~40 行监控专用代码
_off = getattr(self, "_mon_frame_offset", 0)
_mon = getattr(self, "_mon_buf", None)
if _mon is not None:
    _mon_record(hidden_states, _mon, self._mon_slot_hook_ln1 + _off, _mon_anchors)
```

问题：
- **高侵入性**：18 个 `_mon_record` call site、4 个函数签名加 `_mon_anchors` 参数
- **模型耦合**：模型代码直接读取 `_mon_buf`、`_mon_slot_xxx`、`_mon_frame_offset`
- **不可移植**：适配新模型需要理解 slot 命名规则并手动插入条件块
- **anchor op 开销**：每个 forward 末尾调用 `anchor(buf, tensors)`，需要收集所有 tensor

实验发现：**forward hook 在 torch.compile 下可以被 Dynamo 正确 trace**，record() 会进入 FX graph，
Inductor lowering 的 `never_reuse_buffers` 足以防止 buffer reuse，**不需要 anchor op**。

## 2. 实验结果

测试文件：`tests_monitoring/test_hook_based_dual_compile.py`

| 测试 | 验证内容 | 结果 |
|---|---|---|
| test_hook_traced_by_dynamo | Hook 内的 record() 是否进入 FX graph | ✓ PASS |
| test_hook_dual_frame_isolation | `_mon_frame_offset` guard → 双帧地址隔离 | ✓ PASS (0 overlap) |
| test_hook_tensor_correctness | `never_reuse_buffers` 防止中间 tensor 数据被覆写 | ✓ PASS (含非末层 tensor) |
| test_hook_disable_record | `_mon_buf=None` → Dynamo re-trace → 无 record kernel | ✓ PASS |

关键发现：
1. Dynamo 会 trace forward hook 中的代码，生成 FX graph 节点
2. Hook 中读取 `module._mon_frame_offset` → Dynamo 创建 guard → 不同值生成不同 CUDA Graph
3. `never_reuse_buffers` lowering 足够防止 buffer reuse，**不需要 anchor op**
4. `_mon_buf=None` 触发 Dynamo guard 失败 → re-trace 出不含 record 的 graph

## 3. 架构概览

### 3.1 当前架构 (inline)

```
模型代码 (modeling_gpt2.py)          GraphMonitor
┌──────────────────────────┐      ┌──────────────────┐
│ forward():               │      │ _register_hooks: │
│   _off = getattr(...)    │←─────│   set _mon_buf   │
│   _mon = getattr(...)    │      │   set _mon_slot  │
│   if _mon:               │      │   set _mon_off   │
│     _mon_record(t,_mon,  │      │   (no hooks for  │
│       slot+_off,_anch)   │      │    dual_compile) │
│   ...                    │      └──────────────────┘
│   anchor(_mon, _anch)    │
└──────────────────────────┘
```

### 3.2 新架构 (hook-based)

```
模型代码 (modeling_gpt2.py)          GraphMonitor
┌──────────────────────────┐      ┌────────────────────────┐
│ forward():               │      │ _register_hooks:       │
│   x = self.ln_1(x)      │      │   for each HookPoint:  │
│   x = self.hook_ln1(x)  │──┐   │     register_fwd_hook  │
│   ...                    │  │   │     set _mon_buf       │
│   (无监控代码)            │  │   │     set _mon_frame_off │
└──────────────────────────┘  │   │                        │
                              │   │ hook closure:          │
                              └──→│   buf = mod._mon_buf   │
                                  │   if buf is None: ret  │
                                  │   off = mod._mon_off   │
                                  │   record(out,buf,s+off)│
                                  └────────────────────────┘
```

模型代码只调用 identity 模块，**不包含任何监控逻辑**。

## 4. HookPoint 调用位置 (18 个监控点)

### 4.1 GPT2Model.forward (3 个)

```python
# BEFORE (inline)                           # AFTER (hook-based)
inputs_embeds = self.wte(input_ids)         inputs_embeds = self.wte(input_ids)
if _mon is not None:                        inputs_embeds = self.hook_embed(inputs_embeds)
    _mon_record(inputs_embeds, ...)

position_embeds = self.wpe(position_ids)    position_embeds = self.wpe(position_ids)
if _mon is not None:                        position_embeds = self.hook_pos_embed(position_embeds)
    _mon_record(position_embeds, ...)

hidden_states = self.ln_f(hidden_states)    hidden_states = self.ln_f(hidden_states)
if _mon is not None:                        hidden_states = self.hook_final_ln(hidden_states)
    _mon_record(hidden_states, ...)
if _anch:
    torch.ops.graphmonitor_ops.anchor(...)  # 删除
```

同时删除：
- `_off = getattr(self, "_mon_frame_offset", 0)`
- `_mon = getattr(self, "_mon_buf", None)`
- `_anch = [] if _mon is not None else None`
- `_mon_anchors=_anch` 传给 block

### 4.2 GPT2Block.forward (8 个)

```python
# BEFORE                                    # AFTER
def forward(self, ...,                      def forward(self, ...,
            _mon_anchors=None, **kwargs):               **kwargs):       # 删除 _mon_anchors
    _off = getattr(...)                                                  # 删除
    _mon = getattr(...)                                                  # 删除
    residual = hidden_states                    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)    hidden_states = self.ln_1(hidden_states)
    if _mon is not None:                        hidden_states = self.hook_ln1(hidden_states)
        _mon_record(...)
    attn_input = hidden_states                  attn_input = self.hook_resid_pre(hidden_states)
    if _mon is not None:
        _mon_record(...)
    out, w = self.attn(attn_input, ...,         out, w = self.attn(attn_input, ...,
        _mon_anchors=_mon_anchors)                  **kwargs)            # 删除 _mon_anchors
    if _mon is not None:                        out = self.hook_attn_out(out)
        _mon_record(...)
    hidden_states = out + residual              hidden_states = out + residual
    if _mon is not None:                        hidden_states = self.hook_resid_mid(hidden_states)
        _mon_record(...)
    residual = hidden_states                    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)    hidden_states = self.ln_2(hidden_states)
    if _mon is not None:                        hidden_states = self.hook_ln2(hidden_states)
        _mon_record(...)
    mlp_input = hidden_states                   mlp_input = self.hook_mlp_in(hidden_states)
    if _mon is not None:
        _mon_record(...)
    ffn = self.mlp(mlp_input)                   ffn = self.mlp(mlp_input)
    if _mon is not None:                        ffn = self.hook_mlp_out(ffn)
        _mon_record(...)
    hidden_states = residual + ffn              hidden_states = residual + ffn
    if _mon is not None:                        hidden_states = self.hook_resid_post(hidden_states)
        _mon_record(...)
```

### 4.3 GPT2Attention.forward (5 个)

```python
# BEFORE                                    # AFTER
def forward(self, ...,                      def forward(self, ...,
            _mon_anchors=None, **kwargs):               **kwargs):
    _off = getattr(...)                                                  # 删除
    _mon = getattr(...)                                                  # 删除
    ...
    key_states = key_states.view(shape_kv)      key_states = key_states.view(shape_kv)
    if _mon is not None:                        key_states = self.hook_k(key_states)
        _mon_record(...)
    key_states = key_states.transpose(1, 2)     key_states = key_states.transpose(1, 2)
    value_states = value_states.view(shape_kv)  value_states = value_states.view(shape_kv)
    if _mon is not None:                        value_states = self.hook_v(value_states)
        _mon_record(...)
    value_states = value_states.transpose(1,2)  value_states = value_states.transpose(1, 2)
    ...
    query_states = query_states.view(shape_q)   query_states = query_states.view(shape_q)
    if _mon is not None:                        query_states = self.hook_q(query_states)
        _mon_record(...)
    query_states = query_states.transpose(1,2)  query_states = query_states.transpose(1, 2)
    ...
    out, w = attention_interface(self, ...,      out, w = attention_interface(self, ...,
        _mon_anchors=_mon_anchors, ...)             ...)                 # 删除 _mon_anchors
    ...
    if _mon is not None:                        out = self.hook_z(out)
        _mon_record(out, ..., hook_z)
    out = out.reshape(...).contiguous()         out = out.reshape(...).contiguous()
    out = self.c_proj(out)                      out = self.c_proj(out)
    out = self.resid_dropout(out)               out = self.resid_dropout(out)
    if _mon is not None:                        out = self.hook_result(out)
        _mon_record(out, ..., hook_result)
```

注意：cross-attention 分支中的 hook_k/hook_v 同样替换。同一个 HookPoint 模块在两个分支中都调用。

### 4.4 eager_attention_forward (2 个)

```python
# BEFORE                                    # AFTER
def eager_attention_forward(                def eager_attention_forward(
    module, q, k, v, mask,                      module, q, k, v, mask,
    _mon_anchors=None, **kwargs):               **kwargs):          # 删除 _mon_anchors
    ...
    _off = getattr(module, ...)                                      # 删除
    _mon = getattr(module, ...)                                      # 删除
    if _mon is not None and ...:                attn_weights = module.hook_attn_scores(attn_weights)
        _mon_record(attn_weights, ...)
    attn_weights = softmax(attn_weights)        attn_weights = softmax(attn_weights)
    if _mon is not None and ...:                attn_weights = module.hook_pattern(attn_weights)
        _mon_record(attn_weights, ...)
```

### 4.5 汇总

| 位置 | Hook 名称 | 监控目标 |
|---|---|---|
| GPT2Model | hook_embed | wte 输出 |
| GPT2Model | hook_pos_embed | wpe 输出 |
| GPT2Model | hook_final_ln | ln_f 输出 |
| GPT2Block | hook_ln1 | ln_1 输出 |
| GPT2Block | hook_resid_pre | attention 输入 (= ln_1 输出) |
| GPT2Block | hook_attn_out | attention 输出 (residual add 前) |
| GPT2Block | hook_resid_mid | 第一次 residual add 后 |
| GPT2Block | hook_ln2 | ln_2 输出 |
| GPT2Block | hook_mlp_in | MLP 输入 (= ln_2 输出) |
| GPT2Block | hook_mlp_out | MLP 输出 (residual add 前) |
| GPT2Block | hook_resid_post | 第二次 residual add 后 |
| GPT2Attention | hook_q | Q view 后, transpose 前 |
| GPT2Attention | hook_k | K view 后, transpose 前 |
| GPT2Attention | hook_v | V view 后, transpose 前 |
| GPT2Attention | hook_z | attention output (reshape 前) |
| GPT2Attention | hook_result | c_proj + dropout 后 |
| eager_attn | hook_attn_scores | QK^T + mask 后, softmax 前 |
| eager_attn | hook_pattern | softmax 后 |

Per layer 15 hooks × N layers + 3 global = 15N + 3 (GPT-2: 183 hooks)

## 5. GraphMonitor 修改

### 5.1 `_register_hooks` (核心变更)

```python
def _register_hooks(self, model: nn.Module) -> None:
    slot_id = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        if module in self._module_to_slot:
            continue
        if self._module_filter is not None and not self._module_filter(name, module):
            continue
        if slot_id >= self._max_slots:
            break
        self._module_to_slot[module] = slot_id
        self._slot_mapping[slot_id] = SlotInfo(slot_id=slot_id, module_name=name)

        # 所有模式都注册 hook（包括 dual_compile）
        handle = module.register_forward_hook(self._make_hook(slot_id))
        self._handles.append(handle)

        # dual_compile: 在 HookPoint 模块上设置 attrs（不是 parent）
        if self._graph_mode == "dual_compile":
            module._mon_buf = self._gpu_buffer
            module._mon_frame_offset = 0
            self._frame_parents.append(module)
            if module not in self._inline_attrs:
                self._inline_attrs[module] = []
            self._inline_attrs[module].extend(["_mon_buf", "_mon_frame_offset"])

        slot_id += 1
    self._num_monitored_slots = slot_id
```

关键变更：
- 删除 `if self._graph_mode != "dual_compile"` 条件 → 所有模式注册 hook
- dual_compile: attrs 设在**被监控模块自身**（HookPoint），不是 parent
- 删除所有 `_mon_slot_xxx` 设置（hook 闭包直接捕获 slot_id）
- 不再需要 `parent_map` 构建

### 5.2 Hook 实现

```python
def _make_hook(self, slot_id: int) -> Callable[..., None]:
    if self._graph_mode == "dual_compile":
        return self._make_dual_compile_hook(slot_id)
    elif self._graph_mode == "compile":
        return self._make_compile_hook(slot_id)
    return self._make_manual_hook(slot_id)

def _make_dual_compile_hook(self, slot_id: int) -> Callable[..., None]:
    """Hook for dual_compile mode.

    Reads _mon_buf and _mon_frame_offset from the module.
    Dynamo creates guards on these values → separate CUDA graphs per frame.
    """
    def hook(module: nn.Module, inputs, output) -> None:
        buf = module._mon_buf
        if buf is None:
            return
        tensor = GraphMonitor._extract_tensor(output)
        if tensor is None or not tensor.is_cuda:
            return
        offset = module._mon_frame_offset
        torch.ops.graphmonitor_ops.record(tensor, buf, slot_id + offset)
    return hook
```

Dynamo guard 机制：
- `module._mon_buf`: `None` vs `Tensor` → 决定是否有 record kernel
- `module._mon_frame_offset`: `0` vs `num_slots` → 决定写入 shadow buffer 的哪个 frame

### 5.3 `set_frame` / `disable_record` / `enable_record`

逻辑不变，只是 `_frame_parents` 现在存的是 HookPoint 模块（不是 parent）：

```python
def set_frame(self, frame: int) -> None:
    offset = frame * self._num_monitored_slots
    for mod in self._frame_parents:  # 现在是 HookPoint 列表
        mod._mon_frame_offset = offset

def disable_record(self) -> None:
    for mod, attrs in self._inline_attrs.items():  # 现在是 HookPoint
        if hasattr(mod, "_mon_buf"):
            mod._mon_buf = None

def enable_record(self) -> None:
    for mod, attrs in self._inline_attrs.items():
        mod._mon_buf = self._gpu_buffer
```

### 5.4 `close()` 清理

不变。`_inline_attrs` 和 `_frame_parents` 存的是 HookPoint，清理逻辑相同。

## 6. 删除的组件

### 6.1 模型代码 (modeling_gpt2.py)

| 删除项 | 行数 |
|---|---|
| `_mon_record()` 函数定义 | 11 行 |
| `_mon_anchors=None` 参数 (4 个函数签名) | 4 行 |
| `_off = getattr(...)` / `_mon = getattr(...)` 读取 | 8 行 |
| `if _mon is not None: _mon_record(...)` 条件块 | 18 块 |
| `_anch = [] if _mon is not None else None` | 1 行 |
| `if _anch: torch.ops.graphmonitor_ops.anchor(...)` | 1 行 |
| `_mon_anchors=_anch` / `_mon_anchors=_mon_anchors` 传递 | 3 行 |
| **合计删除** | **~46 行** |

新增：18 行 `x = self.hook_xxx(x)` identity call。

净减少：~28 行监控专用代码。

### 6.2 GraphMonitor (graph_monitor.py)

| 删除项 | 说明 |
|---|---|
| `parent_map` 构建逻辑 | 不再需要找 parent |
| `_mon_slot_xxx` 设置 | hook 闭包直接捕获 slot_id |
| `if self._graph_mode != "dual_compile"` 条件 | 所有模式统一注册 hook |

### 6.3 C++ ops (graph_monitor_ops.cu)

| 组件 | 状态 | 原因 |
|---|---|---|
| `anchor_op` | **可删除** | never_reuse_buffers 足够 |
| `sink_op` | **可删除** | 已被 Inductor DCE，从未生效 |
| `sink_hold_op` | **可删除** | torch.compile 下无效 |
| `clear_held_tensors_op` | **可删除** | 依赖 sink_hold |
| `held_tensors_count_op` | **可删除** | 依赖 sink_hold |
| `record_op` | **保留** | 核心：写 metadata 到 shadow buffer |
| `alias_tensor` | **保留** | D2H 地址发现 |
| `batch_d2h` / `batch_d2h_ptrs` | **保留** | D2H 数据传输 |
| `batched_d2h_sm` | **保留** | SM kernel D2H (备用) |
| `wait_d2h` / `init_d2h_events` / etc | **保留** | per-hook barrier (备用) |

TORCH_LIBRARY 注册也相应删除 anchor/sink/sink_hold 的 def 和 impl。

### 6.4 Inductor lowering (graph_ops.py)

`_register_record_lowering()` **保留不变**。这是防止 buffer reuse 的关键机制：

```python
@register_lowering(torch.ops.graphmonitor_ops.record.default)
def record_lowering(tensor, buffer, slot_id):
    tensor.realize()
    buffer.realize()
    V.graph.never_reuse_buffers.add(tensor.data.get_name())
    ir.FallbackKernel.create(...)
```

## 7. 功能对照表

### 7.1 所有现有功能保留

| 功能 | 当前实现 | Hook-based 实现 | 变化 |
|---|---|---|---|
| **record()** | inline `_mon_record` → `ops.record` | hook → `ops.record` | 调用位置变 |
| **anchor()** | `_anch` list → `ops.anchor` at forward end | **删除** | never_reuse_buffers 足够 |
| **never_reuse_buffers** | Inductor lowering on record() | 同 | 不变 |
| **Dual-frame** | `_mon_frame_offset` on parent, inline 读取 | `_mon_frame_offset` on HookPoint, hook 读取 | attr 位置变 |
| **Address isolation** | `cudagraph_trees=False` + Dynamo guard | 同 | 不变 |
| **Record elimination** | `_mon_buf=None` on parent → inline skip | `_mon_buf=None` on HookPoint → hook skip → re-trace | attr 位置变 |
| **4-graph warmup** | Phase 1 ×2 + Phase 2 ×2 | 同 | 不变 |
| **D2H pipeline** | alias_tensor + copy_stream + events | 同 | 不变 |
| **BatchedD2H** | `batch_d2h` / `batch_d2h_ptrs` | 同 | 不变 |
| **Selective hooks** | `select_hooks()` → `update_d2h_mask()` | 同 | 不变 |
| **Per-request selection** | `update_d2h_requests()` → `batch_d2h_ptrs` | 同 | 不变 |
| **Skip-step** | `monitor_interval`, flip at monitored step | 同 | 不变 |
| **Forward/D2H overlap** | `pre_fwd_event` + `copy_stream.wait_event` | 同 | 不变 |
| **compile 模式** | Hook + inline attrs on parent | Hook (attrs 可选) | 简化 |
| **manual 模式** | Hook + capture_anchors | Hook + capture_anchors | 不变 |

### 7.2 所有 18 个 Hook 保留

所有 HookPoint 监控点完整保留，slot 分配和命名与当前一致。
`module_filter=lambda name, mod: hasattr(mod, "monitor_activation")` 行为不变。

### 7.3 GraphSafeEngine API 不变

`GraphSafeEngine` 的所有公开 API 完全不变：

- `prepare_for_model()`, `close()`
- `set_frame()`, `disable_record()`, `enable_record()`
- `finalize_dual_frame()`, `collect_dual_frame_results()`
- `start_step()`, `end_step()`
- `select_hooks()`, `update_d2h_mask()`, `update_d2h_requests()`
- `collect_results()`, `drain_ready_results()`

## 8. 为什么不需要 anchor

`anchor(Tensor(a!) buffer, Tensor[] tensors)` 的两个作用：

1. **防 DCE**：`Tensor(a!)` 注解让 Inductor 不会删除 anchor 节点
2. **GPU data dependency**：`sink_kernel` 读取每个 tensor 的 data_ptr

Hook-based 下这两个需求都已满足：

1. **record() 自身不会被 DCE**：record 的 schema `record(Tensor, Tensor(a!) buffer, int) -> ()` 中
   `Tensor(a!) buffer` 已经防止 DCE（buffer 被标记为 mutation）
2. **never_reuse_buffers 防止 buffer reuse**：Inductor lowering 把每个 recorded tensor
   加入 `V.graph.never_reuse_buffers` → Inductor 的 `can_reuse()` 检查返回 False →
   该 buffer 在整个 graph 生命周期内不被回收

实验验证：`test_hook_tensor_correctness` 中 `hook_after_l1`（非末层 intermediate tensor）数据完全正确，
说明 never_reuse_buffers 在没有 anchor 的情况下已经足够。

## 9. torch.compile 下 hook 的工作原理

```
torch.compile(model.forward, mode="reduce-overhead")
  │
  ├─ Dynamo trace:
  │    model.forward(proxy_input)
  │      → self.ln_1(x)              # Dynamo trace into LayerNorm
  │      → self.hook_ln1(x)          # Dynamo trace into HookPoint.__call__
  │        → HookPoint.forward(x)    # identity, returns x
  │        → forward_hook(mod, inp, out)  # Dynamo trace INTO the hook
  │          → mod._mon_buf           # Guard: Tensor vs None
  │          → mod._mon_frame_offset  # Guard: 0 vs num_slots
  │          → ops.record(out, buf, slot+off)  # FX node created
  │
  ├─ Inductor compile:
  │    record_lowering(tensor, buffer, slot_id):
  │      → V.graph.never_reuse_buffers.add(tensor)  # 防 buffer reuse
  │      → FallbackKernel.create(record, ...)        # 保留 record kernel
  │
  └─ CUDA Graph capture:
       record_metadata_kernel recorded into graph
       → replay 时自动执行，写 metadata 到 shadow buffer
```

Guard 状态机：

| `_mon_buf` | `_mon_frame_offset` | Dynamo 行为 |
|---|---|---|
| `gpu_buffer` (Tensor) | `0` | Graph A: record to frame 0 |
| `gpu_buffer` (Tensor) | `num_slots` | Graph B: record to frame 1 |
| `None` | `0` | Graph C: no record (Phase 2, frame 0) |
| `None` | `num_slots` | Graph D: no record (Phase 2, frame 1) |

4 个 guard 组合 → 4 个 CUDA Graph = 当前 4-graph warmup 完全一致。

## 10. 实施步骤

### Step 1: 修改 `graph_monitor.py`

1. `_register_hooks`: 删除 `!= "dual_compile"` 条件，所有模式注册 hook
2. `_register_hooks`: dual_compile 设 attrs 在被监控模块（非 parent）
3. 删除 `parent_map` 构建和 `_mon_slot_xxx` 设置
4. 新增 `_make_dual_compile_hook()` 方法
5. `_make_hook()` 增加 dual_compile 分派

### Step 2: 修改 `modeling_gpt2.py`

1. 删除 `_mon_record()` 函数
2. 删除所有 `_mon_anchors` 参数和传递
3. 删除所有 `_off = getattr(...)` / `_mon = getattr(...)` 读取
4. 删除所有 `if _mon is not None: _mon_record(...)` 条件块
5. 删除 `_anch` 创建和 `anchor()` 调用
6. 在对应位置加 `x = self.hook_xxx(x)` 调用

### Step 3: 清理 C++ ops

1. 删除 `anchor_op` 实现和注册
2. 删除 `sink_op` 实现和注册（已无调用方）
3. 删除 `sink_hold_op` / `clear_held_tensors_op` / `held_tensors_count_op`
4. 保留 `record_op`, `alias_tensor`, `batch_d2h*`, `wait_d2h*`

### Step 4: 清理 `graph_ops.py`

删除 `anchor` 在 lowering 中的任何引用（当前没有 anchor lowering，只有 record lowering）。

### Step 5: 更新测试

1. `test_design_c_integration.py`:
   - `test_dual_compile_monitor_setup`: 修改 assert — dual_compile 现在有 hooks (`len(handles) > 0`)
   - 其余测试逻辑不变（warmup、address isolation、D2H correctness、record elimination）
2. 保留 `test_hook_based_dual_compile.py` 作为 hook 机制的独立验证
3. 运行全部现有测试确认无回归

### Step 6: 更新 benchmark

`profile_decode.py` 中 `DualCompileDecodeRunner` 的 warmup 流程不变。
模型 forward 内无监控代码 → 不需要改 benchmark。

## 11. vLLM 移植路径

Hook-based 方案使 vLLM 集成大幅简化：

### 最小化侵入 (Tier 1)

对于只需要 residual stream + layer output 的场景：
- **零模型代码改动**
- 直接 hook 现有 nn.Module（LayerNorm, Attention, MLP）
- `module_filter=lambda name, mod: isinstance(mod, (nn.LayerNorm, Attention, MLP))`

### TransformerLens 粒度 (Tier 2)

对于需要 Q/K/V、attention scores 等内部监控的场景：
- 在 Attention 类中加 5-6 个 HookPoint identity call
- 每行代码：`x = self.hook_xxx(x)`
- 无 `_mon_record`、无 `_mon_anchors`、无 anchor、无 slot 概念

### 适配新模型的工作量对比

| 步骤 | 当前 inline | Hook-based (Tier 2) | Hook-based (Tier 1) |
|---|---|---|---|
| 定义 HookPoint 模块 | 需要 | 需要 | 不需要 |
| import _mon_record | 需要 | 不需要 | 不需要 |
| 加 _mon_anchors 参数 | 需要 (所有 forward 签名) | 不需要 | 不需要 |
| 读 _off / _mon | 需要 (每个 forward 顶部) | 不需要 | 不需要 |
| 插入监控代码 | `if _mon: _mon_record(...)` ×N | `x = self.hook(x)` ×N | 不需要 |
| 调 anchor() | 需要 (顶层 forward 末尾) | 不需要 | 不需要 |
| 理解 slot 命名 | 需要 | 不需要 | 不需要 |
| **总工作量** | 大 | 小 | **零** |
