# 接口与配置开发文档（MVP）

## 实时事件
- 事件：`on_token`
- 消息体（按策略裁剪）：
  - `meta`: `request_id, token_id, layer, head, shape, dtype, position_ids, query_start_loc, cache_seqlens`
  - `activations?`: `x_in / attn_out / mlp_out / x_out`
  - `attn_topk?`: `{indices[int32,K], logits[fp16,K], lse?:fp32}`
  - `kv_ref?`: `{block_table_ref, slot_mapping_ref, cache_seqlens_ref}`（仅引用/元数据）
- 传输：推荐 gRPC streaming；可替代为 WebSocket 或本地 IPC（三选一实现其一）。

## 轻量查询（内存态窗口）
- 提供最近 N 秒数据窗口的拉取接口（进程内），供调试/对齐使用；不涉及落盘。

## 配置（在 vllm/config.py 新增 MonitoringConfig）
```python
@dataclass
class MonitoringConfig:
    enabled: bool = False                   # 开关
    layers: str = "all"                     # all|comma-list e.g. "0,4,8"|every:2
    sample_rate: int = 1                    # 每 N token 采样一次
    topk_k: int = 4                         # Top-K K 值
    gpu_staging_bytes: int = 3*64*1024**2   # GPU 端 staging 总容量（约 192MB）
    staging_trigger_bytes: int = 16*1024**2 # 合并触发大小阈值（典型 8–32MB）
    staging_trigger_ms: int = 3             # 合并触发时间阈值（ms）
    cpu_ring_bytes: int = 2*1024**3         # pinned ring 总容量
    copy_streams: int = 2                   # D2H 复制流数量
    max_d2h_bw_gbps: float = 20.0           # D2H 限速（GB/s 近似）
    backpressure_high_watermark: float = 0.7# 触发降级阈值（0~1）
    transport: str = "grpc"                 # grpc|ws|ipc
```
- 为每个字段补充 docstring，保证 `tools/validate_config.py` 通过。

## CLI 与入口（vllm/entrypoints/openai/cli_args.py）
- 新增参数：
  - `--monitoring`（bool 开关）
  - `--mon-layers <all|every:2|0,4,8>`、`--mon-sample-rate N`、`--mon-topk K`
  - `--mon-gpu-staging-bytes`、`--mon-staging-trigger-bytes`、`--mon-staging-trigger-ms`
  - `--mon-cpu-ring-bytes`、`--mon-copy-streams`、`--mon-max-d2h-bw`
  - `--mon-transport {grpc,ws,ipc}`
- `ServeSubcommand` 读取后注入 `MonitoringConfig`，传入引擎配置。

## DoD
- 提供稳定的事件模型与配置；
- CLI 能启动/关闭监控并正确传递配置；
- 校验与默认值完整，错误提示清晰。

## 实施状态（代码位置）
- 配置类型：`MonitoringConfig` 已加入并包含 GPU staging/DMA、采样、Top‑K、ring 等参数。
  - 定义：vllm/config/__init__.py:3215
  - 注入 VllmConfig：vllm/config/__init__.py:3412
- CLI 参数：`--monitoring` 与 `--mon-*` 前缀参数自动对齐 `MonitoringConfig` 字段。
  - 参数组注册：vllm/engine/arg_utils.py:853
  - 布尔开关：vllm/engine/arg_utils.py:857
  - 其它 `--mon-*` 字段：vllm/engine/arg_utils.py:863
  - AsyncEngineArgs 字段（承接 CLI 输入）：vllm/engine/arg_utils.py:463
- 配置贯通：在 `create_engine_config()` 中构造并注入 `monitoring_config`。
  - 构造：vllm/engine/arg_utils.py:1430
  - 注入：vllm/engine/arg_utils.py:1456

## 快速验证
- 查看帮助：`vllm serve --help` 中应出现 “MonitoringConfig” 参数组。
- 最小启用（不改变推理路径）：
  - 示例：`vllm serve --model <hf_repo> --monitoring --mon-layers all --mon-sample-rate 10 --mon-topk-k 4`
