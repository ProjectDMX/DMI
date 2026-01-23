# generate_with_cache wrapper 方案

## 背景与目标
我们希望保留 Hugging Face `generate()` 的完整行为（beam/sample/processor/streamer 等），
同时让每一步前向都走我们的监控管线。当前 `run_with_cache()` 不能直接替换 `forward`，
因为它返回 `(model_out, cache_dict)`，且内部会调用 `self(*args, **kwargs)`。
因此需要一个 wrapper：
1) 调用 `run_with_cache()` 时显式传入 `forward_fn` 避免递归；
2) 只返回 `model_out`，保持 `generate()` 的返回约束。

## 问题描述
- `generate()` 期望 `forward()` 返回标准 `ModelOutput`（含 logits、past_key_values 等）。
- `run_with_cache()` 当前返回 `(model_out, cache_dict)`，并且内部调用 `self(...)`。
  如果把 `forward` 替换成 `run_with_cache` 会递归。
- 我们需要捕获每步激活并可选落库，但不希望重写整个 HF 的生成栈。

## 决策
优先包装 HF `generate()`，而不是重写 generate。这样可以保持与 HF 解码策略一致，
降低维护成本。监控逻辑只在 forward 边界注入。

补充决策：默认采用**监控常驻**。在单脚本内不切换到非监控模式，因此 wrapper
安装后不恢复 `forward`，实例生命周期内始终走监控。如需关闭监控，建议重建模型
或显式卸载 wrapper（同时处理 compile 缓存）。

## 方案概述
1) 扩展 `run_with_cache()`，支持可选 `forward_fn` 参数。
   - 不传时保持原行为。
   - 传入时使用 `forward_fn(*args, **kwargs)` 执行真实前向，避免递归。
2) 新增模块 `monitoring/generate.py`，提供 `generate_with_monitoring(model, *args, **kwargs)`：
   - 默认**常驻包裹** `model.forward`：
     - `engine.start_step(phase=...)`
     - `model.run_with_cache(..., forward_fn=orig_forward, ...)`
     - `engine.end_step()`
     - 仅返回 `model_out`
   - 包裹期间调用 HF `generate()`（或 `GenerationMixin.generate`）。

## 伪代码（wrapper）
```python
import functools
import inspect

def generate_with_monitoring(model, *args, **kwargs):
    engine = getattr(model, "monitoring_engine", None)
    orig_forward = model.forward

    @functools.wraps(orig_forward)
    def monitored_forward(*f_args, **f_kwargs):
        phase = "prefill" if f_kwargs.get("past_key_values") is None else "decode"
        if engine is not None:
            engine.start_step(phase=phase)
        try:
            model_out, _cache = model.run_with_cache(
                *f_args,
                forward_fn=orig_forward,  # 避免递归
                **f_kwargs,
            )
            return model_out
        finally:
            if engine is not None:
                engine.end_step()

    monitored_forward.__signature__ = inspect.signature(orig_forward)

    try:
        model.forward = monitored_forward
        return model.generate(*args, **kwargs)
    finally:
        # 监控常驻：默认不恢复 forward
        pass
```

## 伪代码（run_with_cache 扩展）
```python
def run_with_cache(self, *model_args, forward_fn=None, **model_kwargs):
    if forward_fn is None:
        forward_fn = lambda *a, **k: self(*a, **k)

    # 构建 hooks / cache
    model_out = forward_fn(*model_args, **model_kwargs)
    # 收集 futures + register db step
    return model_out, cache_dict
```

## 计划改动（暂不实现）
- `monitoring/hook_points.py`：`run_with_cache` 增加 `forward_fn` 支持。
- `monitoring/generate.py`：新增 `generate_with_monitoring` helper。

## 备注与风险
- 直接替换 `model.forward` 不是线程安全的；并发场景需 context manager 或锁。
- wrapper 通过外层 `__call__` 进入，因此 PyTorch module-level hooks 仍会执行
  一次；内部调用 `orig_forward` 只是避免再次进入 `__call__`。若某些外部框架
  依赖 `__call__` 包裹实际计算，需额外验证。当前仅保证单机/无外部包装器场景
  （例如不启用 FSDP/Accelerate 等）。
- 必须调用 `engine.end_step()` 才会 seal step 并触发 DB 提交。
- phase 判断在 `prefill_chunk_size` / 静态 cache 路径可能误判；初版使用
  `past_key_values` 作为近似，后续可结合 `cache_position` 或显式 override。
- `prepare_inputs_for_generation` 会通过 `inspect.signature(self.forward)` 判断参数
  支持度（如 `position_ids`）；因此 wrapper 必须保留原始签名（`__signature__` +
  `functools.wraps`），否则会改变 HF 输入准备逻辑。
- `torch.compile` 可能将 wrapper 固化进 `self._compiled_call`。当前方案默认监控
  常驻，因此该固化可接受；若改为临时包裹，必须显式禁用 compile 或保存/恢复
  `_compiled_call/_last_compile_config`。
- HF 某些高级解码（assisted decoding / candidate rescore 等）会触发额外 forward，
  wrapper 会把它们也计入 step，导致 step 数与 token 生成不一致。当前仅保证常规
  greedy/sample 路径；若要支持高级策略需额外处理或说明限制。
- `prepare_inputs_for_generation` 可能在 `inputs_embeds` 路径将 `input_ids` 设为
  `None`；而 `_register_db_step` 依赖 `input_ids.shape` 计算 token_len。当前仅
  支持 `input_ids` 路径；如需支持 `inputs_embeds`，应在注册时 fallback 到
  `inputs_embeds.shape[1]`。

## Demo 伪代码（生成 + 监控）
```python
import os
import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import MonitoringEngine, MonitoringConfig
from monitoring.config import CaptureSchedule, HookSelection
from monitoring.generate import generate_with_monitoring  # 新模块

os.environ["MON_NATIVE_TO_CPU"] = "1"
os.environ["MON_NATIVE_CALLBACK"] = "1"
os.environ["MON_NATIVE_BUILDER"] = "1"
os.environ["MON_NATIVE_BATCH"] = "0"

cfg = MonitoringConfig(
    hooks=HookSelection(mode="full"),
    schedule=CaptureSchedule(
        step_stride=1,
        step_offset=0,
        warmup_steps=0,
        capture_prefill=True,
        capture_decode=True,
        request_stride=1,
        request_offset=0,
        warmup_requests=0,
    ),
)

engine = MonitoringEngine(
    async_enabled=True,
    config=cfg,
    model_id="gpt2",
    db_config=host_cfg,  # 如果接入 DB，传 HostEngineConfig；否则不传
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = HookedGPT2LMHeadModel.from_pretrained(
    "gpt2",
    torch_dtype=torch.float32,
).cuda().eval()

model.monitoring_engine = engine
engine.prepare_for_model(model)

input_ids = tokenizer("The future of AI is", return_tensors="pt").input_ids.cuda()

output_ids = generate_with_monitoring(
    model,
    input_ids=input_ids,
    max_new_tokens=32,
    do_sample=False,
)

# 如需关闭监控，建议重建模型实例（避免 compile 缓存泄漏）
# model = HookedGPT2LMHeadModel.from_pretrained(...)

engine.close()
```
