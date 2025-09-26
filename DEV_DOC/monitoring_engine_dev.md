# 监控引擎（Monitoring Engine）开发文档（MVP）

## 目标与范围（MVP）
- 低开销、非阻塞地调度采集：层边界激活（`x_in/attn_out/mlp_out/x_out` 可裁剪）、注意力 Top‑K。
- 仅读取 vLLM 的 KV 元数据（`block_table/slot_mapping/cache_seqlens`）用于索引解释；不做 KV 镜像或落盘。
- 统一配置、预算评估、异步调度与背压；对接 GPU→CPU 通道与“实时订阅流”（不落盘）。

## 关键职责
- 策略：层集合（或步进 every N 层）、采样率（每 N 个 token）、Top‑K 的 K 值、背压降级策略。
- 预算：按模型与设置估算 D2H 带宽与缓冲占用（参考：7B decode 激活≈3.0MB/步，Top‑K≈0.03MB/步），以及 GPU 端 staging 占用。
- 调度：GPU 端 staging 合并（2–3 组×64–128 MB）→ 多 CUDA copy streams → CPU pinned ring（2–4 槽）→ 实时流推送（gRPC/WebSocket/IPC）。
- 指标：tokens/s、D2H 带宽、staging 命中/触发、ring 水位、降级事件、吞吐损失（对照基线）。

## 生命周期与并发
- 进程内组件：
  - Hook 回调在解码/预填充“层边界/新 token”处登记“设备指针 + 元数据”，提交给 GPUStager。
  - Stager 合并到 staging 批；CUDA events 标记批次可读；D2H 线程在独立 copy streams 上发起 DMA。
  - 一个调度线程协调 staging/D2H/ring；订阅端从 ring 拉取并推送到外部。

## 对外接口（草案）
```python
class MonitoringEngine:
    def __init__(self, cfg: MonitoringConfig, sink: MonitoringSink): ...
    def start(self): ...
    def stop(self): ...

    # 层边界/新 token 回调（轻量）：仅登记引用与元数据
    def on_layer_boundary(self, ctx: LayerCtx, tensors: LayerTensors): ...
    def on_new_token(self, req: ReqMeta, step: int, q_ptr=None): ...

    # 背压/降级
    def maybe_apply_backpressure(self, stats: PipeStats) -> DegradeAction: ...
```

## 与其他模块交互
- KV 元数据视图：提供 `block_table/slot_mapping/cache_seqlens` 的只读访问，用于将 Top‑K 的 `idx` 映射回原位置信息。
- Top‑K 采集器：在 decode 获取 Q 后触发块状扫描；返回 `(idx, logits[, lse])`。
- 数据通道：协调 GPUStager（合并与触发）、异步 D2H、多流与 CPU pinned ring；向实时流 sink 提交包。

## 背压策略（从轻到重）
- 降层（layer subset）→ 降采样（每 N token）→ 仅 Top‑K → 暂停采集。

## 完成定义（DoD）
- 可通过配置启动/关闭；在 7B/4090 上 decode 吞吐损失 < 5%。
- 指标完整上报；背压可触发且恢复；最小基准脚本验证 Top‑K 正确性与无丢帧（ring 窗口内）。

## 实施状态（代码位置）
- 入口配置与 CLI：
  - `MonitoringConfig` 定义：vllm/config/__init__.py:3215
  - 注入 VllmConfig：vllm/config/__init__.py:3412
  - CLI 参数组与开关：vllm/engine/arg_utils.py:853, vllm/engine/arg_utils.py:857, vllm/engine/arg_utils.py:863
  - 配置构造与注入：vllm/engine/arg_utils.py:1430, vllm/engine/arg_utils.py:1456
- 引擎内监控组件：待接入（MVP 阶段将新增 `vllm/v1/monitoring/` 模块，提供 MonitoringEngine + GPUStager/D2H/CpuRing/RealtimeSink 的骨架与 no‑op 实现）。
