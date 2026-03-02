# 每请求 `request_id` 追踪方案（保持当前 cache stream，不改 native D2H 粒度）

## 1. 背景与目标

当前实现是“一个 batch 一个 `request_id`”，host stage1 拿到的 `BackendFuture` 对应的是整批 tensor。  
目标是改成“一个 batch 内每条样本都有独立 `request_id`”，同时先走最简单方案：

- 继续保持 native 侧按 hook 产出整批 tensor（不在 cache stream 里拆）。
- 在 host engine **stage1** 里做按 batch 维切分。
- stage2/ClickHouse 仍按行写入，不改主流程。

## 2. 非目标（本版不做）

- 不在 native/cache stream 里做 per-request D2H 拆分。
- 不处理复杂 beam 重排语义（先按 `num_beams=1` 场景）。

---

## 3. 提议接口（基于你给的 submit 形状）

你给的方向是对的，建议接口为（修正模板括号）：

```cpp
void submit(
    const std::string& model_id,
    int32_t shard_rank,
    const std::vector<std::vector<std::string>>& request_ids,
    const std::vector<std::vector<std::pair<int32_t, int32_t>>>& token_range_per_request,
    const std::vector<std::map<std::string, monitoring::BackendFuture>>& cache_dicts
);
```

约束：

- 外层长度一致：`request_ids.size() == token_range_per_request.size() == cache_dicts.size() == N`
- 对每个 `i`：`request_ids[i].size() == token_range_per_request[i].size() == batch_size_i`
- `cache_dicts[i]` 是该 step 的 `hook_name -> future`
- `attention_mask` 在 prefill 路径必须是 **`[B, L]` 的 padding mask**（0/1 或 bool）；不接受 `[B,1,L,L]` 的 causal mask 作为长度来源

说明：

- 外层 `N` 允许一次 submit 多个 step（与现有批提交习惯兼容）。
- `model_id` + `shard_rank` 放接口顶层，避免每条 item 重复携带。
- `shard_rank` 在本 feature 中仅作为 API 预留字段（为后续 TP/分布式做准备）：
  - 本次实现不引入任何基于 `shard_rank` 的业务逻辑；
  - 仅要求接口保留该参数并在数据结构中透传（可不参与当前处理分支）。
- V1 实现固定按 `N=1` 提交（每次一个 step）；保留 `N` 仅为接口前向兼容。

---

## 4. 端到端流程（新）

### 4.1 Python MonitoringEngine 维护 request 元数据

在 `monitoring/engine.py` 中维护 per-batch 请求上下文（示意）：

```python
# 仅示意
active_batch_request_ids: list[str] | None
active_batch_start_idx_per_request: list[int] | None   # 每条请求当前 start idx
active_batch_finished_per_request: list[bool] | None    # 每条请求是否已结束
auto_batch_group_id: int                                # 单调递增组 id，request_id 用 "{gid}:{i}"
state_lock: threading.Lock                              # 保护 active_batch_* 读写
```

在 `_register_db_step(cache_dict, input_ids, attention_mask, past_key_values)`：

1. 识别 batch size = `input_ids.shape[0]`
2. 判定 `is_prefill = (past_key_values is None)`，并先做初始化/重置：
   - 若 `is_prefill=True`，**无条件重置并初始化**（避免新一轮生成复用旧状态）
   - 若 `is_prefill=False` 且 `active_batch_request_ids is None`，做 decode 路径兜底初始化并打 warning
   - 若已有状态但长度与当前 `B` 不一致，也重置并打 warning（防御性处理）
   - 为 batch 内每条样本分配 request id（如 `"{auto_batch_group_id}:{i}"`）
   - 初始化 `active_batch_start_idx_per_request`（都从 0 开始）
   - 初始化 `active_batch_finished_per_request`（都为 `False`）
3. 计算每条请求本 step 的 `(start, end)`：
   - prefill：`delta_i = attention_mask[i].sum()`（仅接受 `attention_mask.dim()==2`）
   - decode：若 `finished_i=True`，则 `delta_i=0`；否则 `delta_i=1`
   - `start_i = active_batch_start_idx_per_request[i]`
   - `end_i = start_i + delta_i`
   - 更新 `active_batch_start_idx_per_request[i] = end_i`
4. 更新 `finished` 状态（decode）：
   - 基础版可用 `input_ids[:, -1]` 与 `eos/pad` 的规则更新
   - 后续可升级为直接对齐 HF `unfinished_sequences`
5. 过滤 `cache_dict`（去 alias / 非 future）
6. 挂起待提交流水：
   - `pending = (model_id, shard_rank, request_ids, token_ranges, filtered_cache_dict)`

在 `_submit_pending_db_step()`：

```python
host_engine.submit(
    model_id,
    shard_rank,
    [request_ids],          # 外层 N=1
    [token_ranges],         # 外层 N=1
    [cache_dict],           # 外层 N=1
)
```

### 4.2 Host input handler 组装 stage1 item

`input_handler_v2(...)` 逻辑：

- 对每个 step-item 的每个 hook future，生成一条 `FutureProcessRowV2`
- Row 里包含：
  - `model_id`
  - `shard_rank`
  - `request_ids`（整批）
  - `token_ranges`（整批）
  - `act_name`
  - `BackendFuture`

### 4.3 Stage1：一次 `future.result()`，再按 request 切分

在 `future_process.cpp`：

1. `tensor = backend_future.result(...)`（只调用一次）
2. 校验 `tensor.size(0) == request_ids.size()`（至少在支持路径要求）
3. 按 dim0 切分每个 request：

```cpp
for (int j = 0; j < B; ++j) {
    // select 返回 view；这里做独立 contiguous 副本，避免长期持有整批 storage
    auto t_j = tensor.select(0, j).unsqueeze(0).contiguous();
    auto [start, end] = token_ranges[j];
    if (start >= end) {
        continue;                           // 空区间（finished/pad）不入库
    }
    // 组 ClickHouseRow: model_id, request_id, act_name, layer_no, start, end, t_j
    next_q->enqueue(...)
}
```

注意：

- 这里是 CPU 上的切分（结果已是 CPU tensor），不增加 GPU D2H 次数。
- 不要把一个 future 复制给多个 request 去 `result()`，token 会被消费。
- `token_range` 表示该 request 在**逻辑序列坐标**上的有效区间；即使 payload tensor 仍含 pad 位，也以 `token_range` 作为有效 token 语义。

### 4.4 Stage2：ClickHouse 插入不变

`clickhouse_insert` 仍吃 row，不需要改核心流程。  
本 feature 不实现 `shard_rank` 相关逻辑（仅保留接口占位）。
后续若要区分多卡，可选：

- 新增列 `shard_rank`
- 或把 `request_id` 写成 `"{shard_rank}:{request_id}"`（兼容旧表）

---

## 5. 为什么这个方案最稳

1. 不动 native/cache stream 主路径，回归风险低。  
2. 不引入 N 次 GPU 小拷贝，避免把 D2H 吞吐打碎。  
3. 先拿到每请求可追踪语义，后续再做性能优化（pad/drop、GPU 侧预切）有基线可比。  

---

## 6. 需要改动的代码文件

### 6.1 Python: `monitoring/engine.py`

#### A) 新增状态字段（类成员）

```python
# 新增（示意）
self._state_lock = threading.Lock()
self._active_batch_request_ids: Optional[list[str]] = None
self._active_batch_start_idx_per_request: Optional[list[int]] = None
self._active_batch_finished_per_request: Optional[list[bool]] = None
self._auto_batch_group_id: int = 0
```

#### B) `_register_db_step(...)` 改签名与核心逻辑

```python
def _register_db_step(self, cache_dict, input_ids, attention_mask, past_key_values):
    if not host_enabled or not capture_enabled:
        return

    with self._state_lock:
        B = int(input_ids.shape[0])
        is_prefill = (past_key_values is None)

        # 1) 新一轮 prefill: 无条件重置；decode: 仅在状态缺失时兜底初始化
        need_reset = (
            is_prefill
            or self._active_batch_request_ids is None
            or len(self._active_batch_request_ids) != B
        )
        if need_reset:
            gid = self._auto_batch_group_id
            self._auto_batch_group_id += 1
            self._active_batch_request_ids = [f"{gid}:{i}" for i in range(B)]
            self._active_batch_start_idx_per_request = [0] * B
            self._active_batch_finished_per_request = [False] * B
            if not is_prefill:
                logger.warning("db_step initialized/reset in decode path; fallback init applied")

        req_ids = self._active_batch_request_ids
        starts = self._active_batch_start_idx_per_request
        fins = self._active_batch_finished_per_request

        # 2) 计算 token_range_per_request
        token_ranges: list[tuple[int, int]] = []
        if is_prefill:
            if attention_mask is None or attention_mask.dim() != 2:
                raise ValueError("prefill requires 2D padding attention_mask [B, L]")
            lens = attention_mask.sum(dim=1).tolist()
            for i in range(B):
                s = int(starts[i]); e = s + int(lens[i])
                token_ranges.append((s, e)); starts[i] = e
        else:
            for i in range(B):
                s = int(starts[i])
                if fins[i]:
                    token_ranges.append((s, s))
                else:
                    e = s + 1
                    token_ranges.append((s, e)); starts[i] = e

        # 3) 更新 finished（基础版）
        if not is_prefill:
            last_ids = input_ids[:, -1]
            for i in range(B):
                if not fins[i] and int(last_ids[i]) in eos_or_pad_ids:
                    fins[i] = True

        filtered = filter_cache_dict(cache_dict)
        if not filtered:
            return

        self._pending_db_step = (
            self._model_id,
            0,                 # shard_rank 占位
            req_ids,
            token_ranges,
            filtered,
        )
```

#### C) `_submit_pending_db_step()` 改调用

```python
def _submit_pending_db_step(self):
    with self._state_lock:
        if self._pending_db_step is None:
            return
        model_id, shard_rank, req_ids, token_ranges, cache_dict = self._pending_db_step
        self._host_engine.submit(
            model_id,
            int(shard_rank),
            [req_ids],          # N=1 (V1 固定)
            [token_ranges],     # N=1 (V1 固定)
            [cache_dict],       # N=1 (V1 固定)
        )
        self._pending_db_step = None
```

### 6.2 Python: `monitoring/hook_points.py`

`run_with_cache(...)` 里调用 `_register_db_step(...)` 时，补传 `attention_mask`：

```python
attention_mask = model_kwargs.get("attention_mask") if model_kwargs else None
engine._register_db_step(cache_dict, input_ids, attention_mask, past_key_values)
```

### 6.3 C++: `monitoring/csrc/dmx_host_engine.h`

`DMXHostEngine::submit(...)` 改签名：

```cpp
void submit(
    const std::string& model_id,
    int32_t shard_rank,
    const std::vector<std::vector<std::string>>& request_ids,
    const std::vector<std::vector<std::pair<int32_t,int32_t>>>& token_range_per_request,
    const std::vector<std::map<std::string, monitoring::BackendFuture>>& cache_dicts) {
  submit_items(input_handler_v2(model_id, shard_rank, request_ids, token_range_per_request, cache_dicts));
}
```

### 6.4 C++: `monitoring/csrc/dmx_host_utils.h/.cpp`

#### A) 新增 row 类型（示意）

```cpp
using FutureProcessValueV2 = std::variant<
    std::string,                                   // model_id
    int32_t,                                       // shard_rank
    std::vector<std::string>,                      // request_ids
    std::vector<std::pair<int32_t,int32_t>>,       // token_ranges
    std::string,                                   // act_name
    monitoring::BackendFuture                      // future
>;
using FutureProcessRowV2 = std::vector<FutureProcessValueV2>;
```

#### B) 新增 `input_handler_v2(...)`

```cpp
std::vector<dmx_host_queue_item> input_handler_v2(...) {
  // 校验外层和内层长度
  // 对每个 step 的每个 hook future 组装一条 FutureProcessRowV2
  // push 到 outputs
}
```

### 6.5 C++: `monitoring/csrc/future_process.cpp`

`ProcessFuture(...)` 支持 V2：

```cpp
// 1) 取 row_v2 字段
// 2) tensor = backend_future.result(...)
// 3) 校验 tensor.size(0) == request_ids.size()
// 4) for j in [0..B):
//      auto [start,end] = token_ranges[j];
//      if (start >= end) continue;   // 空区间跳过
//      auto t_j = tensor.select(0, j).unsqueeze(0).contiguous();
//      组 ClickHouseRow 并 enqueue
```

### 6.6 C++: `monitoring/csrc/bindings.cpp`

pybind `DMXHostEngine.submit` 改参数：

```cpp
.def("submit",
     &DMXHostEngine::submit,
     py::arg("model_id"),
     py::arg("shard_rank"),
     py::arg("request_ids"),
     py::arg("token_range_per_request"),
     py::arg("cache_dicts"),
     py::call_guard<py::gil_scoped_release>())
```

### 6.7 可选：`monitoring/csrc/clickhouse_client.*`

本 feature 不实现 `shard_rank` 逻辑，故当前可不改。  
后续若要落库区分多卡，再扩列或写入 request_id 前缀。

---

## 7. 兼容与校验建议

最小校验：

- `request_ids` / `token_ranges` 长度一致
- `token_ranges[j].first <= token_ranges[j].second`
- `tensor.dim() >= 1`
- `tensor.size(0) == request_ids.size()`（不一致时打印错误并丢弃该条，避免写错数据）
- prefill 必须提供 `attention_mask.dim()==2` 的 padding mask；否则直接报错并中止该 step 注册

运行约束（V1）：

- 优先支持 `num_beams=1`
- `do_sample` 可开关，不影响 request 粒度映射
- `token_range` 定义为逻辑有效 token 区间（不含 pad 语义）；分析端按该区间解释有效长度

---

## 8. 后续可选优化（不在本次）

1. 将 decode `finished` 判定从启发式升级为直接对齐 HF `unfinished_sequences`。  
2. 对已 finished request 在 Python 侧就不再下发该 request 的 token_range（进一步减轻 stage1 工作量）。  
3. 仅对高价值 hook（如 `final_logits`）做更细粒度优化。  
4. 评估 GPU 侧预切分（仅在收益显著时启用）。  

---

## 9. 示例（每步传给 host engine 的 `request_ids` 与 `token_range_per_request`）

假设一个 batch 有 3 条请求：

- `request_ids = ["r0", "r1", "r2"]`
- prefill 真实长度（由 `attention_mask` 得到）：`[10, 6, 3]`
- `model_id = "qwen3"`
- `shard_rank = 0`

并约定：

- 每步 submit 外层都按 `N=1` 传入。
- 对已 finished 的 request，用空区间 `(start, start)` 表示。
- stage1 规则：仅当 `start < end` 时入库；`start == end` 直接跳过。

### Step P (prefill)

```text
submit(
  model_id="qwen3",
  shard_rank=0,
  request_ids=[["r0","r1","r2"]],
  token_range_per_request=[[(0,10),(0,6),(0,3)]],
  cache_dicts=[{hook_name -> future, ...}]
)
```

### Step D1 (decode 第1步，r2 在本步结束)

```text
submit(
  model_id="qwen3",
  shard_rank=0,
  request_ids=[["r0","r1","r2"]],
  token_range_per_request=[[(10,11),(6,7),(3,4)]],
  cache_dicts=[{hook_name -> future, ...}]
)
```

### Step D2 (decode 第2步，r2 已 finished)

```text
submit(
  model_id="qwen3",
  shard_rank=0,
  request_ids=[["r0","r1","r2"]],
  token_range_per_request=[[(11,12),(7,8),(4,4)]],  # r2 空区间
  cache_dicts=[{hook_name -> future, ...}]
)
```

stage1 对该步行为：

- r0: `(11,12)` 入库
- r1: `(7,8)` 入库
- r2: `(4,4)` 跳过

### Step D3 (decode 第3步，假设 r1 也 finished)

```text
submit(
  model_id="qwen3",
  shard_rank=0,
  request_ids=[["r0","r1","r2"]],
  token_range_per_request=[[(12,13),(8,8),(4,4)]],  # r1/r2 空区间
  cache_dicts=[{hook_name -> future, ...}]
)
```

最终只入库 r0。这样接口始终保持固定长度，stage1 逻辑简单且稳定。
