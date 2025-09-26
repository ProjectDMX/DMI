# 注意力 Top-K 采集器开发文档

## 目标与约束（MVP Phase 1）
- 不改动注意力内核；decode 路径优先支持 Top-K。
- 使用当前步的 Q 对 paged-KV 的 K 做块状扫描，返回 `(indices, logits[, lse])`。

## 算法与形状
- 输入：
  - Q_t: `[num_heads, d_head]`（当前 token）
  - K: paged KV，经 `block_table` 映射为若干块（block）
  - K（每块）形状：`[num_heads, block_size, d_head]`
- 过程：
  1) 逐块计算 `scores = Q_t @ K_block^T`（按头并行）；
  2) 取块内 top-k（k=K_block）；
  3) 归并为全局 top-K（K=cfg.topk_k）并得到全局位置（seq_id/pos 或绝对 idx）。
- 输出：
  - `topk_idx: [num_heads, K] (int32)`；`topk_logits: [num_heads, K] (fp16/bf16)`；
  - 可选 `lse`（若从 FA decode 获得）。

## 计算与并发
- 将计算放在独立 CUDA stream，使用 event 等待 Q 可用；
- 结果优先写入 GPUStager（与激活同批合并），统一触发 D2H；
- 与主推理流解耦，限制每步预算（如 < 0.1ms/h）。

## 接口（草案）
```python
class TopKCollector:
    def __init__(self, cfg: TopKConfig, kv_meta: KVMetaView): ...  # 仅元数据视图（block_table/slot_mapping/cache_seqlens）
    def collect(self, req: ReqMeta, layer: int, q_ptr, heads: int) -> TopKResult
```

## 精度与性能权衡
- MVP 仅 decode；prefill 可选关闭或降低 K。
- 块选择可采用“候选块”启发式（最近块优先）以降耗；
- 允许近似：保留 leftover_mass 作为诊断指标。

## DoD
- 7B/4090、K=4、H=32 下每步 < 0.5ms（目标），无阻塞主流；
- 返回索引与 logits 与参考实现一致；错误与越界均可检测并 fail-safe；
- 与 vLLM KV 元数据的一致性校验（idx→原始位置映射正确）。

## 实施状态（代码位置）
- 配置入口：`topk_k`、采样与层过滤来源于 `MonitoringConfig`（默认禁用）。
  - 配置定义：vllm/config/__init__.py:3215
  - CLI 参数组：vllm/engine/arg_utils.py:853
- 实现：Top‑K 采集器尚未落地；计划在 `vllm/v1/monitoring/` 下实现 `TopKCollector`，在 decode 步通过 Q 触发，结果写入 GPUStager（与激活同批）。
