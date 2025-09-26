# Hook 机制开发文档（最小侵入，MVP）

## 背景与目标（MVP）
- 不改动 vLLM 核心推理/内核，优先在 DecoderLayer 的“边界”挂钩：`x_in / attn_out / mlp_out / x_out`；decode 步额外采集 Q（供 Top‑K）。
- Hook 必须轻量、可控、上下文可启停，避免对吞吐造成明显影响（目标 < 5%）。

## TransformerLens 的做法（要点）
- HookPoint：一个 nn.Module，forward 恒等，内部维护 fwd/bwd hooks 列表；提供 `add_hook/remove_hooks`，并在根模块收集为 `hook_dict` 以支持按名称开关。
- HookedRootModule：
  - 启动时 `setup()` 遍历 `named_modules()` 记录 modules 和 HookPoint 名称映射。
  - `hooks()/run_with_hooks()` 上下文管理器临时注册并在退出时移除，防止 hook 成为全局状态残留。
  - 统一 hook 签名：用户函数签名 `fn(tensor, *, hook)`，底层用 `register_forward_hook` / `register_full_backward_hook` 适配。

参考文件：`transformer_lens/hook_points.py`（HookPoint 与 HookedRootModule 的实现）。

## 我们的采用方案（vLLM 监控版）
- 不侵入模型定义：优先使用 PyTorch 的 `register_forward_hook` 绑定到 HF 模型中的可见子模块，例如：
  - `blocks.[i].attn.q_proj`（采集 decode 步 Q）；
  - `blocks.[i].attn.out_proj`、`blocks.[i].mlp.down_proj`（边界激活采集点：`attn_out/mlp_out/x_out`）。
- 统一 Hook 管理：实现轻量 `HookManager`（仅管理 handle 与名称过滤），提供：
  - `add_hook(name_or_pred, fn)`、`remove_all()`、`context()`；
  - 统一用户函数签名 `fn(tensor, name, module, ctx) -> Optional[tensor]`（返回值可用于“干预”，MVP 仅观测不修改）。
- 性能与内存：
  - Hook 内不做 CPU 拷贝，仅登记张量引用 + 元数据；优先写入 GPUStager（GPU 端合并），再由 D2H 异步批量转运至 CPU pinned ring。
  - 仅在 decode 步启用 Q 采集；支持 `sample_rate` 与层子集过滤；MVP 不做量化压缩。

## 接口草案
```python
class HookManager:
    def __init__(self, model):
        self.model = model
        self.handles = []
    def add_hook(self, name_or_pred, fn): ...
    def remove_all(self): ...
    @contextmanager
    def context(self, hooks: list[tuple[str|Callable[[str], bool], Callable]]):
        try:
            for name, fn in hooks: self.add_hook(name, fn)
            yield
        finally:
            self.remove_all()
```

Hook 函数模板：
```python
def save_q_hook(tensor, name, module, ctx):
    # 仅 decode 步采集；tensor 形状约为 [B,H,d_head] 或 [B,L,H,d_head]
    if ctx.step_type == 'decode':
        ctx.emit_q(layer=ctx.layer_of(name), q=tensor)
    return None  # 不修改前向
```

## 实施步骤（建议）
- 列出模型 `named_modules()`，确认各层 `q_proj/out_proj/down_proj` 的名称模式；在创建引擎后按层过滤注册 hook。
- 与 MonitoringEngine/GPUStager 对接：hook 仅入队“引用/摘要+元数据”，由 GPUStager 合并触发；失败时快速降级（关闭某些 hook）。
- 测试：
  - 单层单步正确性（Q 形状与头数一致）；
  - 端到端对吞吐的影响（Lite 默认仅少量层+采样率）。

## 注意事项
- Hook 是全局状态，务必用上下文或生命周期管理器移除。
- 严格避开 fused/FA 内核内部；仅取层级边界张量，兼容编译优化。
- 复杂模型（GQA/MLA）下 Q 的形状与布局不同，需参考当前模型配置决定 reshape 方式。
- KV 相关信息通过 vLLM 的 KV 元数据视图获取（只读），不直接复制 K/V。

## 实施状态（代码位置）
- 配置与 CLI：Hook 选择与采样率由 `MonitoringConfig` 提供；默认关闭。
  - 配置定义：vllm/config/__init__.py:3215
  - CLI 参数组：vllm/engine/arg_utils.py:853, vllm/engine/arg_utils.py:857
- 实际 Hook 注入：尚未在代码中启用（计划在 `vllm/v1/monitoring/` 中实现 HookManager 并在引擎构造时按配置注册）。
