# 推理透明性 **MVP 开发文档**（聚焦版）

> **与旧稿相比的关键变化**
> - ✅ **复用 vLLM 自带 KVCache 管理**（block_table / slot_mapping / cache_seqlens 等）——**不再**自研 KV 管理器，只做“镜像/抽样”与元数据读取。
> - ✅ **仅做 MVP**：**GPU→CPU 非阻塞传输 + 内存态缓存 + 实时流**；**CPU→磁盘/数据库**的落盘与索引**暂不纳入范围**。
> - ✅ 砍掉与 MVP 无关内容（治理/安全/多租户等）。
> - ✅ 设计压缩为**最小侵入**：只在层级边界挂载 Hook，避免破坏 fused/FA 路径。

---

## 0) 目标与边界

**目标**：在 vLLM 推理（prefill + decode）过程中，**低开销、非阻塞**地采集：
- 关键**激活**：`x_in / attn_out / mlp_out / x_out`（隐藏状态流）
- **Top-K 注意力**：“当前 token 在看谁” → 每层每头 `K` 个位置与打分（`(idx, logit)`）
- **KV 镜像元数据**（只读）：引用 vLLM 的页式 KV（增量 token 的 `k_t,v_t` 视作“可引用对象”，不主动搬运至磁盘）

**非目标（MVP）**：训练/反向；完整 `T×T` 注意力矩阵；跨节点聚合；**任何落盘/索引/数据库**。

**硬件&模型基线**：单卡 4090（24 GB）；LLaMA-7B（L=32, d=4096, H=32, d_head=128，SwiGLU，**不考虑 GQA**）。

---

## 1) 系统概览（最小闭环）

```
┌───────────────────────────────┐
│           vLLM Runtime        │
│  ┌─────────────────────────┐  │
│  │ Decoder Layer (×L)      │  │
│  │  ┌ Hook @ in/attn/mlp/out│  │
│  │  └─> Emit tensors/meta  │  │
│  └─────────────────────────┘  │
│      │                │        │
│      │ Top-K Extract  │        │
│      ▼                │        │
│  (Q vs paged-K)       │        │
│      │                │        │
│  ┌─────────────────────────┐  │
│  │ Monitoring Engine (MVP) │  │
│  │  • 采样与背压            │  │
│  │  • GPU 端 staging 缓冲   │  │
│  │  • D2H 调度(多流)        │  │
│  │  • CPU pinned ring 缓存  │  │
│  └─────────────────────────┘  │
└─────────────┬─────────────────┘
              │ 实时流 (gRPC/WebSocket/IPC)
              ▼
        订阅者（可视化/调试/对齐）
```

**数据面（Data Plane）**
1. **Hook**：在每层**边界**抓 `x_in / attn_out / mlp_out / x_out`；必要时可切换为“仅 `x_out`”。
2. **Top-K**：对当前步 `Q` 与 paged-KV 的 `K`做块状扫描→每层每头输出 `K` 个 `(idx:int32, logit:fp16)`；decode 路径可额外返回 FA 的 `lse`（行归一化常数）。
3. **传输**：先写入**GPU 端 staging 缓冲**（合并小张量）→ CUDA 多 copy stream（DMA）→ **CPU pinned ring**（2–4 槽；1–5 GB，可配）。不落盘；订阅者实时消费。

**控制面（Control Plane）**
- **Monitoring Engine**：采样/限流、传输调度、背压（降层→降采样→仅 Top-K）；基础指标（tokens/s、D2H 带宽、ring 水位）。
- **KV**：**仅读取 vLLM 的 KV 元数据**，用于定位/解释，不复制到磁盘。

---

## 2) MVP Feature List（对外能力）

### 2.1 激活（Activations）
- **必选**：
  - `x_in`（layer input / hidden state）
  - `attn_out`（注意力分支输出，投影后）
  - `mlp_out`（MLP 分支输出，down_proj 后）
  - `x_out`（layer output / 下一层输入）
- **可配置裁剪**：仅 `x_out`；或 `x_in + x_out`（最轻）。

### 2.2 注意力“看谁”（Top-K）
- 每层每头 `K` 个最重历史位置：`(idx:int32, logit:fp16)`；默认 `K=4`。
- **可选**：`lse`（decode 从 FA 返回，用于概率重构/数值校验）。

### 2.3 KV 相关（只读/引用）
- 读取 vLLM 的 `block_table / slot_mapping / cache_seqlens` 等**元数据**，作为 Top-K 索引的**解释层**（例如把 `idx` 映射回原序列位置）。
- 不做 KV 的 CPU 镜像或落盘。

### 2.4 实时接口
- **事件**：`on_token(request_id, token_id)`
- **载荷**（按采样策略）：激活张量（可裁剪）、Top-K（及 `lse`）、KV 元数据引用、基本指标。

---

## 3) 性能与带宽预算（7B，单卡，decode）

- **激活（“去重版”）**：约 **~3.0 MB/步**（仅 `x_in/attn_out/mlp_out/x_out`；bf16）。  
- **Top-K**：H=32，K=4 → **~1 KB/层/步** → L=32 ≈ **~32 KB/步**（可忽略）。  
- **D2H 预算**：≈ `3.0 MB × TPS + 0.03 MB × TPS`；PCIe 4×16 实测可 20–28 GB/s，**几百 TPS** 仍充裕。  
- **CPU pinned ring**：建议 **1–5 GB**（0.3–1.5 s 的数据窗口），订阅端按需消费。  
- **VRAM 占用**：不新增常驻显存（仅瞬时 staging）；与 vLLM KV 页式管理共存。

---

## 4) 模块设计（仅 MVP 所需）

### 4.1 Hook 层（最小侵入）
- 位置：DecoderLayer **入口/分支/出口**张量（不进入内核细节）。
- 机制：前向钩子 + `no_grad` + 轻量元数据（layer/head/token_id/shape/dtype）。
- 可配置：层集合、采样频率（每 N 个 token）、激活裁剪方案。

### 4.2 Top-K Extractor
- 输入：当前步 `Q`、paged-KV 的 `K`（由 vLLM 提供）。
- 算法：**块状扫描 + 块内 topk + 全局归并**；支持候选块优化（可后续）。
- 输出：每层每头 `K` 个 `(idx, logit)`；可携带 `lse`（decode）。
- 运行：独立 CUDA stream，与主前向重叠执行；开销低且可关闭。

### 4.3 GPU 端 staging（启用 DMA 与合并）
- 在 GPU 上预留若干 staging 缓冲（2–3 组，典型：三缓冲），用于将多路小张量（层边界激活、Top‑K 结果）拼接为较大连续块，减少 D2H 调用次数与提升带宽利用。
- 通过 CUDA events 标记批次可读，D2H 线程在独立 copy streams 上执行 `cudaMemcpyAsync` 触发 DMA。
- 触发策略：大小阈值（如 ≥ 8–32 MB）或时间阈值（如 ≤ 2–5 ms）二选一/组合；可按批合并不同层的同类载荷。

### 4.4 传输与缓冲（不落盘）
- CUDA **多 copy stream** + **非阻塞** D2H（从 GPU staging 复制）；
- **CPU pinned ring**（2–4 槽；1–5 GB）：生产者（D2H）→消费者（订阅流）；
- **背压策略**：`ring` 高水位 → **降层** → **降采样** → **仅 Top‑K**；并上报指标。

### 4.5 Monitoring Engine（调度/配置/指标）
- **配置项**：`layers / sample_rate / K / enable_QOS / ring_size / max_d2h_bw`；
- **调度**：在 GPU 端进行合并（staging 2–3 组×64–128 MB），批量触发 D2H；
- **指标**：tokens/s、D2H 带宽、ring 水位、降级事件、吞吐损失（对照基线）。

---

## 5) 对外接口（最小集）

### 5.1 订阅流（推荐）
- **协议**：gRPC streaming / WebSocket / 本地 IPC（三选一即可）。
- **Topic**：`/mvp/on_token`。
- **消息体**：
  - `meta`: `request_id, token_id, layer, head, shape, dtype, position_ids, query_start_loc, seq_lens`  
  - `activations`: `x_in / attn_out / mlp_out / x_out`（按配置出现）  
  - `attn_topk`: `indices[int32, K], logits[fp16, K]`（及可选 `lse`）  
  - `kv_ref`: `{block_table_ref, slot_mapping_ref, cache_seqlens_ref}`（仅引用/元数据）

### 5.2 轻量查询（内存态视图）
- 提供**最近 N 秒**窗口的内存态数据快照拉取（避免落盘依赖）。
- 过滤条件：`request_id / layer / head / token_range / feature_set`。

---

## 6) 开发计划（从易到难）

### Phase A — **MVP Lite（两周目标建议）**
**交付内容**
1. Hook at layer 边界：`x_out`（默认）+ 可选 `x_in/attn_out/mlp_out`；
2. Top-K（decode）：H×K 的 `(idx, logit)` + 可选 `lse`；
3. GPU 端 staging（三缓冲，64–128 MB 可配）→ D2H（多流）→ CPU pinned ring（2 槽起）；
4. 订阅接口（gRPC/WebSocket 二选一）；
5. 指标：tokens/s、D2H 带宽、ring 水位、降级事件；
6. 验收：吞吐损失 < **5%**、Top-K 排序与抽样真值一致、无数据丢失（窗口内）。

### Phase B — **增强（可选迭代）**
- Prefill Top-K 抽样（按窗口/层采样）  
- 激活裁剪策略（仅 `x_in+x_out` / 仅 `x_out`）与 INT8 压缩可选  
- GPU staging（3×64–128 MB）与聚合触发（时间/大小）  
- 更细指标与可视化 Demo（Attention Explorer 原型）

> **注**：CPU→磁盘/索引**延后到未来**，不影响 MVP 闭环。

---

## 7) 风险与验证

- **Hook 与编译/FA 内核的相容性**：仅在层边界取张量，不触碰 fused kernel；保留开关。
- **D2H 与 PCIe 抢占**：多流 + 限速阈值，持续监控吞吐损失；背压优雅降级。
- **Pinned 内存占用**：设定上限（如 ≤ 8 GB 或物理内存 25%），超限主动降级。
- **Top-K 正确性**：与抽样计算的注意力排名做 A/B 校验（小批量离线比对）。

---

## 8) 快速参数建议（默认值）

| 项 | 默认 | 说明 |
|---|---:|---|
| 采样层 | `all` | MVP下可配 `every 2/4 layers` |
| 激活裁剪 | `x_out` | 可切 `x_in+x_out` |
| Top-K | `K=4` | H=32 时每步 ~32 KB |
| 订阅 | `gRPC` | 单路流，回压可控 |
| pinned ring | `2 GB × 2 槽` | ≈ 0.6–1.2 s 窗口 |
| GPU staging | `3 × 64–128 MB` | 合并与 DMA 触发 |
| D2H 限速 | `20 GB/s` | 接近 PCIe4×16 可持续 |
| 背压阈值 | `>70%` | 逐级降级触发 |

---

**附：形状约定（7B, bf16）**
- `x_*`: `[B_active, 1, d_model]`（decode 单步聚合到 batch 维）  
- `attn_topk`: `[L, H, K]` 的 `(idx, logit)` 对  
- `lse`: `[L, H]`（可选）  
- `meta`: `request_id, token_id, position_ids, query_start_loc, seq_lens`

---

## 9) 实施现状与入口（Phase 0）
- 配置与 CLI 已接通（默认关闭，不改变推理路径）：
  - `MonitoringConfig` 定义：vllm/config/__init__.py:3215
  - 注入到 `VllmConfig`：vllm/config/__init__.py:3412
  - CLI 参数组登记（`--monitoring` 与 `--mon-*`）：vllm/engine/arg_utils.py:853, vllm/engine/arg_utils.py:857, vllm/engine/arg_utils.py:863
  - 在引擎配置创建时构造并挂载：vllm/engine/arg_utils.py:1430, vllm/engine/arg_utils.py:1456

后续 Phase A：在不影响主流的前提下，逐步接入 Hook → Top‑K → GPUStager/D2H → RealtimeSink。
