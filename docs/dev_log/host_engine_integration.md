# host_engine_integration

## 新 host_engine API（来自 `HF_Prometheus/dmx_host/test_prefill.py`）
使用 `PipelinedEngine.submit(keys, start_token_idx, cache_dicts)`：
- **keys**: `List[tuple]`，每个 request 的唯一标识（通常是 `(model_id, request_id)`）
- **start_token_idx**: `List[int]`，每个 request 当前 step 的起始 token 位置
- **cache_dicts**: `List[dict]`，每个 request 的 `cache_dict`（`str -> CacheFuture`）

对应的输入处理器：
`HF_Prometheus/dmx_host/dmx_interface.py::input_handler_v1`

```python
def input_handler_v1(list_of_tuple_keys, list_of_start_token_idx, list_of_cache_dict):
    for tuple_key, start_idx, cache_dict in zip(...):
        for k, v in cache_dict.items():
            StageOneItemForFuture((*tuple_key, start_idx, k, v))
```

## 关键输入要求与坑

### 1) cache_dict 的 value 必须是 **Future**
`stage_one_parsing_and_wait` 会直接调用 `tensor_future.result()`：
```python
tensor = tensor_future.result()
```
**坑**：如果 value 是纯 tensor 或 None，会报错或被跳过。  
**要求**：value 必须是 `CacheFuture/BackendFuture`（或至少实现 `.result()`）。

### 2) logits 必须在 cache_dict 里（由 backend offload 注入）
当前 pipeline **不会**从 `outputs` 读取 logits，只处理 `cache_dict`。  
因此 `final_logits` 必须作为 `cache_dict` 的一个 entry（value 仍需是 Future）。

### 3) start_token_idx 语义（必须是 step 起始位置）
`stage_one_parsing_and_wait` 会用 tensor 形状计算 `end_token_idx`：
```python
delta_token_len = get_delta_token_len(tensor.shape, act_name)
end_token_idx = start_token_idx + delta_token_len
```
规则在 `HF_Prometheus/dmx_host/dmx_interface.py`：
- 普通激活使用 `tensor.shape[1]`
- `attn.hook_attn_scores` / `attn.hook_pattern` 使用 `tensor.shape[2]`

### 4) act_name 格式影响 layer_no
`parse_internal_id` 只识别 `"blocks.X.*"`：  
- `blocks.3.hook_q` -> `layer_no=3`  
- 其他前缀（如 `h.3.*`）会变成 `layer_no=-1`

**坑**：同时存在 `blocks.*` 和 `h.*` 会导致 DB 里重复写入。  
建议只保留一种命名风格。

### 5) cache_dict 里如果包含 token_ids
`token_ids` 会被当作普通 activation 写入 DB（`act_name="token_ids"`）。  
如果你不想写入 token_ids，需要在提交前过滤。

### 6) keys 的长度必须是 2
`input_handler_v1` 把 `tuple_key` 展开后组装成 5 元组：  
`(model_id, request_id, start_token_idx, act_name, tensor_future)`  
因此 `tuple_key` **必须是 `(model_id, request_id)`**，长度为 2。  
否则在 `stage_one_parsing_and_wait` 解包会报错：
```python
model_id, request_id, start_token_idx, act_name, tensor_future = row
```

### 7) tensor 必须在 CPU 且可转 numpy
`stage_one_parsing_and_wait` 会执行：
```python
metadata, data = torch_encode(tensor)  # 内部用 numpy
```
因此要求：
- tensor 必须是 **CPU**（GPU tensor 会直接报错）
- 最好是 **contiguous**（`flatten().numpy()` 会触发复制）

**建议**：开启 `MON_NATIVE_TO_CPU=1`，否则 `future.result()` 返回 GPU tensor 会失败。

### 8) 提前清理风险
`future.result()` 需要 token 仍在 native slot 内：  
如果开启自动清理或主线程调用 `resolve_all()/clear_completed_results()`，  
可能导致 `Unknown future token`。  
建议默认 `MON_NATIVE_AUTOCLEAR=0` 并让 host_engine 尽快消费。

## cache_dict 兼容性决策（实现记录）
为满足 `host_engine.submit(keys, start_token_idx, cache_dicts)` 接口：
- `cache_dict` 的 value 必须是 `CacheFuture/BackendFuture`
- 需要额外包含两个条目：
  - `token_ids`：本 step 的输入 token
  - `final_logits`：本 step 的 logits

当前实现策略：
1) 新增 `HookedGPT2LMHeadModel`（不影响原 `HookedGPT2Model`）
2) 在 LMHead 模型中增加两个 HookPoint：
   - `token_ids`（在 forward 开头接收 `input_ids`）
   - `final_logits`（在 `lm_head` 输出后接收 logits）
3) 这样 `run_with_cache` 会自动把这两项作为 Future 写入 `cache_dict`

文件位置：
- `HF_Prometheus/transformers/src/transformers/models/gpt2_p/modeling_gpt2.py`
  - `HookedGPT2LMHeadModel`
  - `token_ids = HookPoint()`
  - `final_logits = HookPoint()`

好处：
- 不需要在上层手动 `.cpu()` 或手动计算 `@ embedding`
- 与 host_engine 新 API 的 future 约束一致

## 集成计划（单请求串行，自动追踪 request_id）
前提：只考虑 **C++ 后端**（Python 后端已弃用）。  
目标：用户不显式调用 submit / request 上下文，MonitoringEngine 自动处理。

### 1) 初始化阶段
- 在 `MonitoringEngine.__init__` 增加 `model_id` 参数（用于生成 key）
- 增加可选 `db_config`（启用 host_engine）
- 初始化 host_engine（`dmx_host.engine.PipelinedEngine`）并 `start()`
- 初始化 engine 内部 request 追踪状态：
  - `_auto_request_id`（int，起始 0）
  - `_auto_start_token_idx`（int，起始 0）
  - `_auto_active_request_key`（tuple `(model_id, request_id)`）

### 2) run_with_cache 内部自动注册 step payload
- 在 `HookedRootModule.run_with_cache` 内部：
  - 读取 `input_ids`（来自 `model_kwargs["input_ids"]` 或 `model_args[0]`）
  - 计算 `token_len = input_ids.shape[1]`
  - 读取 `past_key_values`：
    - 若 `past_key_values is None`，视为新 request：
      - `_auto_request_id += 1`
      - `_auto_start_token_idx = 0`
      - `_auto_active_request_key = (model_id, f"{_auto_request_id}")`
    - 否则沿用当前 request
  - 将 `cache_dict` + `key` + `start_token_idx` 注册到 engine：
    - `engine._pending_db_step = (key, start_idx, cache_dict)`
  - 更新 `_auto_start_token_idx += token_len`

### 3) end_step 自动 submit
- 在 `MonitoringEngine.end_step()` 最后：
  - **仅在 C++ backend 路径**且 `_pending_db_step` 存在时：
    - `host_engine.submit([key], [start_idx], [cache_dict])`
  - 清空 `_pending_db_step`

### 4) 关闭流程
- `MonitoringEngine.close()` 时：
  - 调 `host_engine.stop()` / `terminate_host_engine(...)`

### 5) 关键隐藏点（用户无需显式调用）
用户代码只需要：
```python
engine.start_step()
outputs, cache_dict = model.run_with_cache(...)
engine.end_step()
```
request_id / start_token_idx / submit 全部由 engine 内部完成。

### 6) 限制
- 仅适用于 **单请求串行** 推理  
- 不支持多请求混 batch 或交错 decode  

## 需要改写的代码段（仅 C++ 后端）

### A) `HF_Prometheus/monitoring/engine.py`
1) `MonitoringEngine.__init__`  
- 增加 `model_id` 与 `db_config` 参数  
- 初始化 host_engine (`dmx_host.engine.PipelinedEngine`) 并 `start()`  
- 初始化自动追踪字段：`_auto_request_id/_auto_start_token_idx/_auto_active_request_key`  
- 新增 `_pending_db_step` 用于 end_step 提交

2) `MonitoringEngine.end_step`  
- 在 **native backend** 路径完成 `submit_step/seal_step` 之后，调用：  
  `self._host_engine.submit([key], [start_idx], [cache_dict])`  
- 清空 `_pending_db_step`  
- 注意：**不要**在 end_step 内部调用 `clear_completed_results()`（避免 future token 提前失效）

3) `MonitoringEngine.close`  
- 增加 `host_engine.stop()` / `host_engine.abort()` 的收尾逻辑  
- 避免在 stop 之前调用 `clear_completed_results()`（让 host_engine 完成消费）

### B) `HF_Prometheus/monitoring/hook_points.py`
1) `HookedRootModule.run_with_cache`  
- 在 `cache_dict` 创建后、`return model_out` 前：  
  - 读取 `input_ids` / `past_key_values`  
  - 计算 `token_len = input_ids.shape[1]`  
  - 若 `past_key_values is None`，新建 request 并重置 `_auto_start_token_idx`  
  - 将 `cache_dict + key + start_token_idx` 写入 `engine._pending_db_step`  
  - `_auto_start_token_idx += token_len`

2) names_filter 策略  
- host_engine 只识别 `blocks.X.*`：  
  - 当 DB 模式启用时，建议在 run_with_cache 内部 **过滤掉** `h.*` alias  
  - 避免 DB 中重复写入（`blocks.*` + `h.*` 同时存在）

### C) `HF_Prometheus/transformers/src/transformers/models/gpt2_p/modeling_gpt2.py`
1) `HookedGPT2LMHeadModel`  
- 确保 `token_ids` 与 `final_logits` 已存在并进入 cache_dict  
- 若启用 names_filter，需要确保这两个 hook 名被保留  
- 注意：该模型与 `HookedGPT2Model` 并存，不影响原 benchmark

### D) 运行前置条件（写入文档/README）
- 必须开启 native to CPU：`MON_NATIVE_TO_CPU=1`  
- 必须关闭 auto clear：`MON_NATIVE_AUTOCLEAR=0`  
- 推荐使用 native callback 路径生成 futures：  
  - `MON_NATIVE_CALLBACK=1`  
  - 避免 `MON_NATIVE_BATCH=1` + `native_builder` 组合导致 cache_dict 为 `None`


MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 MON_NATIVE_AUTOCLEAR=0 python -m benchmark.tests.host_engine_min_integration