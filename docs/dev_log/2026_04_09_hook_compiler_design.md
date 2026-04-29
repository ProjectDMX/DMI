# DMI Hook Compiler — 设计文档

Issue: #36 (Hook Interface & Model Compiler — Auto-inject HookPoints into model files)

## 1. 背景与问题

给新模型添加 DMI 监控 hook 目前是纯手工操作，每个模型约需修改 187 行代码。
需要在多个位置做改动，且每换一个模型 × 每换一个框架（HF / vLLM）都得重复一遍。

目标：提供一个编译器工具，让用户以极低成本为任意模型生成带 hook 的变体文件。

## 2. 方案概述

采用 **Python DSL + 两阶段编译** 的方式：

```
dmi extract <model_file> → spec 文件（自动生成的极简骨架）
用户在骨架里加 H(...)       → dmi compile <spec_file> → hooked 模型文件
```

### 2.1 为什么用 Python DSL 而不是 YAML / JSON / XML

| 维度 | Python DSL | YAML |
|------|-----------|------|
| 用户看到的 | 简化版 forward 代码 | 抽象的 pattern 字符串 |
| 指定位置 | 直接放 `H()` 在对应行 | 写 `after: "torch.matmul(*)"` 然后祈祷 match 对 |
| 有歧义时 | 看上下文就知道是哪个语句 | 要加 `occurrence: 2` |
| IDE 支持 | 语法高亮、跳转全有 | 纯字符串，无法跳转 |
| 学习成本 | 会 Python 就会 | 要学 pattern 语法 |

核心 insight：forward 的结构本身就是 Python，用 Python 描述 Python 最直接。

## 3. DSL 设计

### 3.1 Spec 文件格式

`dmi extract` 自动从原始模型文件生成极简骨架。骨架只保留**数据流骨架**——
tensor 从哪来、经过哪些变换、到哪去。去掉所有 shape 计算、assert、logging、
type hint 细节。

```python
# Auto-generated from: modeling_qwen3.py
# Add H("name") where you want hooks, then run: dmi compile

from dmi.dsl import H, spec

@spec(source="modeling_qwen3.py")
class Qwen3Attention:
    def forward(self, hidden_states, position_embeddings):
        query_states = self.q_norm(self.q_proj(hidden_states).view(...))
        key_states = self.k_norm(self.k_proj(hidden_states).view(...))
        value_states = self.v_proj(hidden_states).view(...)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, ...)
        attn_output = self.o_proj(attn_output)

class Qwen3DecoderLayer:
    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

class Qwen3Model:
    def forward(self, input_ids, position_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, ...)
        hidden_states = self.norm(hidden_states)
```

### 3.2 用户编辑：添加 `H()` 标记

用户在想要插入 hook 的位置添加 `H()` 调用：

```python
from dmi.dsl import H, spec

@spec(source="modeling_qwen3.py")
class Qwen3Attention:
    def forward(self, hidden_states, position_embeddings):
        query_states = self.q_norm(self.q_proj(hidden_states).view(...))
        H("q", query_states)
        key_states = self.k_norm(self.k_proj(hidden_states).view(...))
        H("k", key_states)
        value_states = self.v_proj(hidden_states).view(...)
        H("v", value_states)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, ...)
        H("z", attn_output)
        attn_output = self.o_proj(attn_output)

class Qwen3DecoderLayer:
    def forward(self, hidden_states, position_embeddings):
        H("resid_pre", hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        H("ln1", hidden_states)
        attn_output, _ = self.self_attn(hidden_states, ...)
        H("attn_out", attn_output)
        hidden_states = residual + attn_output
        H("resid_mid", hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        H("ln2", hidden_states)
        H("mlp_in", hidden_states)
        hidden_states = self.mlp(hidden_states)
        H("mlp_out", hidden_states)
        hidden_states = residual + hidden_states

class Qwen3Model:
    def forward(self, input_ids, position_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        H("embed", inputs_embeds)
        for layer in self.layers:
            hidden_states = layer(hidden_states, ...)
        H("resid_final", hidden_states)
        hidden_states = self.norm(hidden_states)
        H("final_ln", hidden_states)

class Qwen3ForCausalLM:
    def forward(self, input_ids):
        H("token_ids", input_ids, dtype=torch.long)
        outputs = self.model(input_ids, ...)
        logits = self.lm_head(hidden_states)
        H("final_logits", logits)
```

### 3.3 `H()` 完整签名

```python
def H(
    name: str,                    # hook 名称，必填
    target: variable = None,      # hook 哪个变量
    dtype: torch.dtype = None,    # 覆盖 dtype（如 token_ids 用 int64）
):
```

**关键规则：多赋值目标时 `target` 必填。**

```python
hidden_states = self.mlp(hidden_states)
H("mlp_out")                            # OK — 单目标，自动推断为 hidden_states

hidden_states, residual = self.input_layernorm(hidden_states, residual)
H("ln1")                                # 编译器报错：多赋值目标，必须指定变量
H("ln1", hidden_states)                 # OK — 显式指定 hook hidden_states
```

编译器在 parse spec 时即可做此校验——往回看一条语句，如果是 tuple unpack 且
`H()` 没传第二个参数，直接报错提示用户补上。

### 3.4 HF 与 vLLM 分别生成

同一个模型（如 Qwen3）在 HF 和 vLLM 中 forward 结构不同（控制流、函数签名、
返回值、fused kernel 用法），因此**同一模型必须有两份 spec**。这是无法避免的，
因为 hook 的插入位置取决于具体的代码结构。

```bash
dmi extract modeling_qwen3.py -o qwen3_hf_spec.py
dmi extract qwen3.py -o qwen3_vllm_spec.py
```

## 4. 编译器需要生成的改动（Ring Path Only）

经审查实际调用链（`bench_ring_transport.py` → `_install_monitoring_forward`
→ `install_ring_hooks`），确认 ring transport 路径不经过 `HookedRootModule` /
`setup()` / `hook_dict` / `_normalize_hook_names()` 这套 legacy 机制。

`prepare_monitoring()` 在 ring 模式下直接 return（hook_points.py:824）：
```python
if getattr(engine, "_using_ring_transport", False):
    native_using = False
if not native_callback_active or native_backend is None:
    return  # ring 模式 — 空操作
```

ring path 的 hook 安装走：
```
_install_monitoring_forward(model)
  → model.get_hook_specs()        # 获取有序 spec 列表
  → install_ring_hooks(specs)     # 在每个 HookPoint 上设 _ring_hook_type/_ring_hook_id
  → HookPoint.forward()           # 运行时调 torch.ops.ring.producer
```

因此编译器**只需要生成以下改动**：

### 4.1 Import 添加

```python
from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HookSpec,
    HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_Q, HOOK_TYPE_K,
    HOOK_TYPE_V, HOOK_TYPE_Z, HOOK_TYPE_ATTN_OUT, ...
)
```

### 4.2 各子类 `__init__` — 声明 HookPoint 实例

根据 spec 中每个类里的 `H()` 调用，在对应类的 `__init__` 末尾添加：

```python
self.hook_q = HookPoint()
self.hook_k = HookPoint()
...
```

### 4.3 各子类 `forward()` — 插入 hook 调用

HF 和 vLLM 使用不同的插入模式：

```python
# HF: hook 在数据流里，返回值赋回变量
hidden_states = self.hook_ln1(hidden_states)

# vLLM: hook 是旁路观测，不影响数据流
self.hook_ln1(hidden_states)
```

编译器需要根据 `@spec(framework=...)` 参数决定生成哪种模式。

### 4.4 顶层模型类 `get_hook_specs()` — 按 forward 执行顺序

此函数是 ring transport 的入口，**顺序必须严格匹配实际 forward 中 hook 的触发顺序**。
编译器从 spec 中 `H()` 的出现顺序（跨类展开后）自动生成。

示例输出（HF Qwen3）：
```python
def get_hook_specs(self) -> list:
    from monitoring.ring_transport import HookSpec, HOOK_TYPE_TOKEN_IDS, ...
    specs = []
    specs.append(HookSpec(HOOK_TYPE_TOKEN_IDS, self.token_ids, dtype=torch.long))
    specs.append(HookSpec(HOOK_TYPE_EMBED, self.model.hook_embed))
    for i, layer in enumerate(self.model.layers):
        specs.append(HookSpec(HOOK_TYPE_RESID_PRE, layer.hook_resid_pre, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_LN1, layer.hook_ln1, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_Q, layer.self_attn.hook_q, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_K, layer.self_attn.hook_k, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_V, layer.self_attn.hook_v, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_Z, layer.self_attn.hook_z, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_ATTN_OUT, layer.hook_attn_out, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_RESID_MID, layer.hook_resid_mid, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_LN2, layer.hook_ln2, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_MLP_IN, layer.hook_mlp_in, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_MLP_POST, layer.mlp.hook_post, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_MLP_OUT, layer.hook_mlp_out, layer_no=i))
    specs.append(HookSpec(HOOK_TYPE_RESID_FINAL, self.model.hook_resid_final))
    specs.append(HookSpec(HOOK_TYPE_FINAL_LN, self.model.hook_final_ln))
    specs.append(HookSpec(HOOK_TYPE_FINAL_LOGITS, self.final_logits))
    return specs
```

### 4.5 顶层类 forward — virtual hooks

`token_ids` 和 `final_logits` 不在子模块的 forward 里，而是在顶层 CausalLM 的
forward 中：

```python
# Qwen3ForCausalLM.forward
if input_ids is not None:
    input_ids = self.token_ids(input_ids)     # token_ids hook
...
logits = self.lm_head(hidden_states)
logits = self.final_logits(logits)            # final_logits hook
```

这两个 HookPoint 也声明在顶层类的 `__init__` 中。

### 4.6 特殊情况：独立函数中的 hook

HF 的 `eager_attention_forward` 是模块级函数（不是类方法），通过 `hasattr` 检查
module 上是否有 hook：

```python
def eager_attention_forward(module, query, key, value, ...):
    ...
    if hasattr(module, "hook_attn_scores"):
        attn_weights = module.hook_attn_scores(attn_weights)
    ...
    if hasattr(module, "hook_pattern"):
        attn_weights = module.hook_pattern(attn_weights)
```

编译器第一版可以将此作为特殊 case 处理，或要求用户在 spec 中手动标注需要修改独立函数。

### 4.7 不需要生成的（legacy path，ring 不用）

以下均为 legacy native backend 路径的产物，ring transport 不需要：

- ~~`HookedRootModule` 继承~~
- ~~`setup()` 调用~~
- ~~`hook_dict` / `mod_dict` 构建~~
- ~~`_normalize_hook_names()`~~
- ~~`prepare_monitoring()`~~
- ~~`run_with_cache()` / `add_caching_hooks()`~~

## 5. Fused Kernel 与 Hookable 边界

`dmi extract` 工作在 Python AST 层面，fused kernel 对它来说就是一个普通的
`self.xxx(...)` 调用。这**天然解决了 hookable 边界的问题**——骨架里能看到的就是
能 hook 的，看不到的（kernel 内部）就是 hook 不了的。用户不需要自己判断哪些是
fused 的。

### 5.1 vLLM：fused kernel 不透明，只能 hook 前后

```python
# vLLM 骨架：fused RMSNorm — 进不去内部
hidden_states, residual = self.input_layernorm(hidden_states, residual)
H("ln1", hidden_states)          # ← 只能 hook 输出

# vLLM 骨架：Flash Attention — 内部不可见
attn_output = self.attn(q, k, v)
H("z", attn_output)              # ← 能 hook q/k/v（之前）和 attn_output（之后）
                                  #    但看不到 attn_scores / softmax
```

### 5.2 HF：eager 模式展开内部，可以 hook 更多

同一个模型的 HF 版本如果使用 eager attention，内部操作在 Python 层可见：

```python
# HF 骨架：eager attention — 内部展开
attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
H("attn_scores", attn_weights)   # ← 能 hook
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
H("pattern", attn_weights)       # ← 能 hook
attn_output = torch.matmul(attn_weights, value_states)
H("z", attn_output)
```

这和现有手写版本的行为完全一致：vLLM 的 `gpt2_p.py` 没有 `hook_attn_scores` /
`hook_pattern`，而 HF 的 `modeling_gpt2.py` 有。同一模型的 HF spec 天然比
vLLM spec 有更多可 hook 的点。

### 5.3 对用户的意义

用户拿到骨架后，看到的代码就是他能 hook 的全部。不需要翻文档查哪些操作是 fused
的——**如果骨架里没展开，就说明 hook 不进去**。这使得 spec 文件本身就是该框架下
hookable 位置的完整参考。

## 6. `dmi extract` — AST 骨架提取器

### 6.1 提取规则

编译器用 Python `ast` 模块解析原始模型文件，对每个类的 `forward()` 方法：

| 保留 | 删除 |
|------|------|
| `x = self.xxx(...)` — 子模块调用 | shape 计算、view/reshape |
| `x = func(a, b)` — 顶层函数调用 | assert、logging、warning |
| `x = a + b` — 简单算术（残差连接） | 复杂的条件分支 |
| `for layer in self.layers` — 循环 | type hint 细节 |
| `return ...` | 缓存管理 boilerplate |
| 多赋值 `a, b = f(x, y)` | mask 构建逻辑 |

### 6.2 复杂参数的简化

```python
# 原始
attn_output, attn_weights = attention_interface(
    self, query_states, key_states, value_states, attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
)

# 骨架
attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, ...)
```

保留前几个关键位置参数（通常是 tensor 参数），其余用 `...` 代替。

## 7. `dmi compile` — 编译流程

1. **解析 spec 文件 AST** — 找到所有 `H()` 调用及其上下文语句
2. **解析原始模型文件 AST**
3. **对每个 `H()`** — 用其前后语句作为 anchor 在原始 AST 中定位插入点
   （AST 结构匹配，不是字符串 grep）
4. **生成 hooked 文件**：
   - `__init__` 中添加 `self.hook_xxx = HookPoint()`
   - `forward` 中插入 hook 调用
   - 文件顶部添加 import
   - 顶层类添加 `get_hook_specs()`
5. **校验**：检查生成的 hook 数量、名称、顺序与 spec 一致

### 7.1 AST 匹配策略

spec 骨架中的语句是原始 forward 的简化版，编译器需要做**模糊 AST 匹配**：

- 匹配赋值目标（左值）：`query_states = ...` 在 spec 和原始中目标相同
- 匹配调用模式（右值核心）：`self.q_proj(...)` 匹配包含该调用的语句
- 按顺序匹配：spec 中语句的相对顺序与原始 forward 一致

当多条语句匹配同一 pattern 时（如两个 `residual + hidden_states`），按出现顺序
1:1 对应。

## 8. 用户工作流

```bash
# Step 1: 提取骨架
dmi extract path/to/modeling_qwen3.py -o qwen3_hf_spec.py

# Step 2: 编辑（唯一的手动步骤 — 加 H()）
vim qwen3_hf_spec.py

# Step 3: 编译
dmi compile qwen3_hf_spec.py -o modeling_qwen3_p.py

# Step 4: 验证（对比手写版本）
diff modeling_qwen3_p.py integration/transformers/.../qwen3_p/modeling_qwen3.py
```

## 9. 测试策略

现有手写 hooked 文件（GPT-2 + Qwen3 × HF + vLLM = 4 个文件）作为 ground truth。
不能做字面 diff，因为手写版本包含编译器不会生成的 legacy 代码（`HookedRootModule`、
`setup()`、`_normalize_hook_names()` 等）。测试分两层：

### 9.1 层 1：结构化 AST diff（CI 快速校验）

不比较完整文件，而是提取并比较关键结构：

```python
def test_compiled_matches_handwritten():
    compiled = parse_hooked_file("compiled_qwen3_p.py")
    handwritten = parse_hooked_file("handwritten_qwen3_p.py")

    # 1. 每个类的 HookPoint 声明集合一致
    assert compiled.hook_declarations == handwritten.hook_declarations

    # 2. 每个 forward 中 hook 调用的位置和目标变量一致
    assert compiled.hook_calls == handwritten.hook_calls

    # 3. get_hook_specs() 的顺序和类型一致
    assert compiled.hook_specs_order == handwritten.hook_specs_order
```

比较的三个维度：
- **声明一致**：每个类的 `__init__` 中声明了哪些 `HookPoint()`
- **调用一致**：每个 `forward()` 中 hook 调用的位置（相对于哪条语句）和目标变量
- **顺序一致**：`get_hook_specs()` 返回的 `(hook_type, layer_no)` 序列

### 9.2 层 2：功能等价测试（完整验证）

加载两个版本的模型，跑同一输入，比较 hook 实际触发的结果：

```python
def test_functional_equivalence():
    compiled_model = load("compiled_qwen3_p.py")
    handwritten_model = load("handwritten_qwen3_p.py")

    # 共享同一份权重
    compiled_model.load_state_dict(handwritten_model.state_dict(), strict=False)

    # hook spec 顺序和类型一致
    specs_c = compiled_model.get_hook_specs()
    specs_h = handwritten_model.get_hook_specs()
    assert [(s.hook_type, s.layer_no) for s in specs_c] == \
           [(s.hook_type, s.layer_no) for s in specs_h]

    # 同一输入，每个 hook 拿到的 tensor 完全一致
    input_ids = torch.randint(0, 1000, (1, 32)).cuda()
    captures_c = collect_hook_outputs(compiled_model, input_ids)
    captures_h = collect_hook_outputs(handwritten_model, input_ids)
    for name in captures_c:
        assert torch.equal(captures_c[name], captures_h[name]), f"mismatch at {name}"
```

### 9.3 Ground truth 文件

| 模型 | 框架 | 手写 hooked 文件路径 |
|------|------|---------------------|
| GPT-2 | HF | `integration/transformers/src/transformers/models/gpt2_p/modeling_gpt2.py` |
| GPT-2 | vLLM | `integration/vllm/vllm/model_executor/models/gpt2_p.py` |
| Qwen3 | HF | `integration/transformers/src/transformers/models/qwen3_p/modeling_qwen3.py` |
| Qwen3 | vLLM | `integration/vllm/vllm/model_executor/models/qwen3_p.py` |

4 个文件 × 2 层测试 = 编译器的完整验证矩阵。

## 10. 实现优先级

1. **AST skeleton extractor** (`dmi extract`) — 用户的第一个触点
2. **Compiler core** (`dmi compile`) — 处理 4.1–4.5 所有改动
3. **验证** — 层 1 AST diff + 层 2 功能等价，对 4 个 ground truth 文件全部通过
4. **独立函数 hook**（4.6）— 处理 `eager_attention_forward` 等特殊 case

## 11. Phase 2：Generalization 边界与后续扩展

### 11.1 当前编译器的实际泛化范围

当前实现并不是“任意模型 Python 文件都能直接编译出 hooked 版本”的通用程序变换器，
而是一个针对 **Transformer 常见 `forward()` 结构** 的专用 AST 编译器。

它目前最稳定支持的是这类情况：

- hook 插入点位于 **类的 `forward()` 方法内部**
- 相关 `forward()` 定义就在 **当前 source file** 中
- 数据流骨架能被 `extract` 正常提取为赋值 / `for` / `if` 语句
- per-layer 结构能从 `make_layers(...)` 或 `nn.ModuleList(...)` 中识别出来
- `get_hook_specs()` 可以从“root model + layer stack + 子模块组合关系”直接推出

换句话说，当前编译器已经具备：

- **跨模型泛化**：例如 GPT-2、Qwen3
- **跨框架泛化**：例如 HF、vLLM

但这是一种**同构结构上的泛化**，不是对任意实现风格的完全泛化。

### 11.2 当前实现为什么会不断长 heuristic

当前编译器虽然已经能覆盖 GPT-2 / Qwen3、HF / vLLM 等多条路径，但它的核心实现路线
并不是“严格语义上的 compiler”，而更像一个：

> **spec-guided AST/text hybrid rewriter**

也就是说，它当前主要是按下面这条链路工作：

1. 从 spec 中提取 `H(...)` 和其相邻语句
2. 把这些锚点信息降成：
   - 普通语句文本
   - `for` / `if` 的 header 文本
   - 少量 block kind / branch / ordinal 元信息
3. 再回到真实源码，用 AST + 文本匹配把这些锚点“猜”回去
4. 最后按源码行插入 hook 调用，必要时做局部 rewrite

这条路线的最大问题是：

- 它会在中间丢失结构信息
- 丢失后只能靠 matcher / fallback / rewrite pass 补回来
- 每支持一种新的代码形态，通常就会多出一条 heuristic

这也是为什么当前代码里会逐步出现：

- 语句模糊匹配
- control-flow header 模糊匹配
- best-effort target 校验
- 找不到 anchor 时的表达式拆分 rewrite
- layer loop 前后分类推断

这些都不是“错误实现”，但它们会持续积累 **heuristic debt**。如果继续沿这条路线扩展，
编译器会越来越像一个 patch library，而不是一个可以稳定演进的 compiler。

### 11.3 当前不能直接覆盖的几类模型实现

下面这些情况会超出 Phase 1 编译器的能力边界：

1. **跨文件继承的 `forward()`**

   例如 vLLM Qwen3 中，`Qwen3Model` / `Qwen3MLP` 的关键 `forward()` 逻辑来自
   基类（Qwen2），而不在 `qwen3.py` 当前文件里。单文件编译器看不到完整 AST，
   因此无法在这些位置注入 hook。

2. **hook 点位于独立函数，而不是类方法**

   例如 HF GPT-2 / Qwen3 里的 `attn_scores` / `pattern` 实际发生在
   `eager_attention_forward` 这类独立函数中。当前编译器只处理 class method，
   不处理函数级 hook。

3. **ground truth 依赖 wrapper / root model 聚合**

   HF 的手写版本通常在更外层 wrapper（如 `GPT2LMHeadModel`、
   `Qwen3ForCausalLM`）上统一生成 `get_hook_specs()`，把：

   - `token_ids`
   - `embed`
   - per-layer hooks
   - `resid_final`
   - `final_ln`
   - `final_logits`

   串成完整 FIFO 顺序。当前编译器更偏向“按 spec root class 生成一个局部
   `get_hook_specs()`”，还没有完整建模 wrapper 层。

4. **需要改写链式表达式 / helper function 才能对齐 hand-written GT**

   有些 hand-written hooked 文件并不只是“在原语句后面插一行 hook”，而是先把一条
   链式表达式拆成两三条语句，再在中间插 hook。当前编译器虽然已经支持少量通用 rewrite，
   但整体上仍以“在现有语句边界插入 hook”为主。

### 11.4 这意味着什么

因此，Phase 1 的正确定位应该是：

- **它不是通用 compiler**
- **它是一个对常见 Transformer `forward()` 结构具有较强泛化能力的专用 compiler**

这也是为什么当前测试矩阵虽然已经从单一模型扩展到多模型 / 多框架，
但仍然不能据此宣称“任何模型都可以直接用”。

更准确的表述应是：

> 对于 hook 点位于当前文件 `forward()`、层结构可识别、且无需跨文件/函数重写的模型，
> 编译器可以较低成本生成 hooked 版本。

### 11.5 Phase 2 不应继续沿着 heuristic 路线扩展

Phase 2 的目标，不应该是：

- 继续为更多模型添加 matcher
- 继续为更多代码形态添加字符串规则
- 继续在主流程里塞更多“找不到就 fallback”的 rewrite

这条路线短期能补 case，长期会让实现越来越像 monkey patch 系统：

- correctness 越来越难证明
- 新 heuristic 容易和旧 heuristic 相互干扰
- 测试矩阵会越来越依赖具体模型，而不是结构能力
- 代码会继续增长，但增长的是补丁复杂度，不是架构能力

因此，Phase 2 应明确转向 **结构化重设计**。

### 11.6 Phase 2 的重设计方向

Phase 2 的核心工作不是“加更多特判”，而是把当前 compiler 从
“AST/text hybrid rewriter” 收敛成 “基于结构化 IR 的 source-to-source compiler”。

建议的方向如下：

1. **把 anchor 从字符串升级成结构化表示**

   当前很多复杂度来自：

   - `anchor_before` 先被 `ast.unparse()` 成文本
   - 后续再通过 `_find_line()` / header match 猜回 source

   这一步会丢失语义信息。Phase 2 应该直接保留结构化 anchor，例如：

   - `StmtAnchor(stmt_index=...)`
   - `AfterBlock(kind="for", ordinal=...)`
   - `BlockEntry(kind="if", ordinal=..., branch="body" | "orelse")`

   也就是说，spec 解析阶段应该保留“结构位置语义”，而不是尽早降级成字符串。

2. **为 spec/source 建统一的 `forward IR`**

   对 spec 和 source 都提取成同一种中间表示，例如：

   - 顺序语句节点
   - `for` block 节点
   - `if.body` / `if.orelse` 节点
   - 每个节点带 stable fingerprint / ordinal / child index

   然后匹配发生在 IR 层，而不是在线文本或源码行层。

3. **主路径尽量改成 AST transform，而不是按行插入**

   当前很多 heuristic 都来自：

   - 找源码行
   - 处理 multiline stmt
   - 算缩进
   - 再把字符串插回源码

   如果目标节点已经存在于 AST / IR 中，那么插 hook 应该优先变成：

   - 在 body 中插入 AST stmt
   - 或在表达式节点上做结构化 rewrite

   最后统一 `ast.unparse()` 成源码。这样很多定位和缩进 heuristic 会自然消失。

4. **把 rewrite 变成显式的独立 pass**

   像“把 `out = f(expr)` 改写成 `x = expr; hook(x); out = f(x)`”这类逻辑，
   不应该再作为主流程里的隐式 fallback。更合理的做法是：

   - 单独定义 rewrite pass
   - 每个 pass 明确声明支持的输入模式和输出模式
   - 主编译流程只编排 pass，不内嵌越来越多特殊情况

5. **让 `get_hook_specs()` 生成也基于结构模型，而不是后验推断**

   例如：

   - root hooks
   - per-layer hooks
   - post-layer hooks
   - wrapper/root-level hooks

   这些最好来自统一的结构树或 composition IR，而不是再额外做一轮
   “按 loop 前后位置推断 pre/post”。

### 11.7 Phase 2 的结构能力目标

在完成上面的架构收敛后，Phase 2 应补齐真正限制泛化能力的三个结构能力：

1. **多源文件 / 继承解析**

   能沿 MRO 或 import 路径找到基类 `forward()`，把“当前类 + 基类定义”拼成完整可编译视图。

2. **函数级 hook 编译**

   不仅支持 class method，也支持对 `eager_attention_forward` 这类顶层函数插 hook。
   这样才能覆盖 `attn_scores` / `pattern` 等现有 GT 中的重要点位。

3. **wrapper/root-level `get_hook_specs()` 生成**

   能从内层 model、layer stack、lm head、token_ids hook 等多个类的组合关系，
   自动生成和 hand-written GT 更接近的完整 FIFO hook spec 列表。

### 11.8 对测试矩阵的影响

在 Phase 1 中：

- HF GPT-2 / HF Qwen3 可以逐步补成“主要结构对齐”
- vLLM Qwen3 这类跨文件继承 case 不能强行纳入“必须完全通过”的矩阵

当前阶段的实际测试覆盖状态：

- **已覆盖**
  - GPT-2 vLLM
  - GPT-2 HF
  - Qwen3 HF
- **未覆盖**
  - Qwen3 vLLM（`forward()` 关键逻辑来自基类 / 其他 source file，单文件编译器暂时无法完整处理）

因此，当前测试可以证明：

- compiler 已经能跨 **vLLM / HF** 两个框架工作
- compiler 已经能跨 **GPT-2 / Qwen3** 两个模型工作
- 但完整的“4 个 ground truth 全覆盖”目标尚未达成

到 Phase 2 完成后，才应把下面这些 case 视为真正的“完整验证矩阵”：

- GPT-2 HF
- GPT-2 vLLM
- Qwen3 HF
- Qwen3 vLLM
- 以及后续新增的 Llama / 其他模型
