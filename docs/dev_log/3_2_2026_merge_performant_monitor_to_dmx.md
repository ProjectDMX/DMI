# Merge Log: performant-monitor → ProjectDMX/DMI:HF_Prometheus

**Date**: 2026-03-02
**Branch**: `merge/performant-monitor`
**Base**: `HF_Prometheus` (fast-forwarded to `performant-monitor` HEAD `a5f016698`)
**Target**: `dmx/HF_Prometheus` (65 commits ahead)

## Merge Strategy

Our side (performant-monitor, 24 commits): inline record + dual graph + shadow buffer + D2H pipeline
Their side (dmx, 65 commits): Python backend removal, C++ host engine refactor, ClickHouse pipeline, Qwen3 support

## Conflict Resolution Log

### 1. `.gitignore` — 合并两边
- 我们: `monitoring/.torch_extensions/`
- 对方: `benchmark/figures/`, `*.nsys-rep`, `node_modules/`, `package-lock.json`
- **处理**: 全部保留

### 2. `Quick_Start.ipynb` — 用对方
- 我们: 只改了 notebook metadata（kernel name TL→proj-dmx, python 3.10.18→3.10.19）
- 对方: 加了 inference 示例 + DB 集成的新 cell
- **处理**: `git checkout --theirs`

### 3. `benchmark/tests/profile_decode.py` — 用我们
- 我们: 完整的 hook_selection + monitoring_bypass + CaptureSchedule + GraphSafeEngine runner
- 对方: 简化版 MonitoringEngine 构造
- **处理**: `git checkout --ours`

### 4. `monitoring/__init__.py` — 合并两边 exports
- 我们: `GraphSafeEngine`, `GraphSlotConsumer`, `GraphSlotResult`, `GraphMonitor`, `SlotInfo`
- 对方: `HostEngineConfig`, `AdvanceConfig`, `NativePartialSealConfig`
- **处理**: 全部保留

### 5. `monitoring/csrc/api_submit.cpp` — 合并
- 我们: 加了 `hook_calls`/`hook_enqueued` stats + `if (enable_pinpool_)` 守卫
- 对方: pool stats 无条件输出（对方已移除 `enable_pinpool_` 字段）
- **处理**: 保留我们的两个新 stats，去掉 `if (enable_pinpool_)` 守卫（对方代码中该字段不存在）

### 6. `monitoring/csrc/bindings.cpp` — 合并两边绑定
- 冲突1（includes）: 我们 `graph_native_delegate.h` + 对方 `clickhouse_client.h`/`dmx_host_engine.h`/`future_process.h` → 全部保留
- 冲突2+3（bindings）:
  - 我们: `GraphNativeDelegate` class + `monitor_activation()` + `parse_shadow_block()` + `create_graph_delegate()`
  - 对方: `BackendFuture` class + `create_engine()`（扩展版7参数） + ClickHouse/DMXHostEngine 全套绑定
  - `create_engine` 用对方版本（参数超集，向后兼容）
  - **处理**: 全部保留，删掉我们的 `create_engine`（3参数版）

### 7. `monitoring/csrc/engine_core.cpp` — 合并
- 冲突1（includes）: 我们 `<cstdlib>`/`<cstring>` + 对方 `<cstdio>`/`<limits>`/CUDA allocator stats → 全部保留
- 冲突2（D2H offload 逻辑）:
  - 我们: `want_cpu` + `use_pinned_` + `enable_pinpool_` 三层条件判断 + try-catch pool block 安全保护
  - 对方: 简化版，无条件 D2H，去掉了 `enable_pinpool_`/`use_pinned_`/`move_to_cpu_` flag（struct 中已不存在）
  - **处理**: 用对方简化版（兼容其重构后的 struct）+ 加回我们的 try-catch 保护 pool block 不泄漏
  - **注意**: 这是 native engine 老路径（per-hook D2H），不影响我们的 graph-safe 快速路径（graph_engine.py）

### 8. `monitoring/csrc/native_engine.cpp` — 用对方
- 一处冲突：`create_global_hook_callback_sig` 内的 `add_task_from_config` + `record_step_name_token` 调用
  - 我们: `add_task_from_config(cfg, tensor)` 返回 token, `record_step_name_token(step_id, name, token)` 3参数
  - 对方: `add_task_from_config(cfg, tensor, step_id)` 返回 pair(token, task_size), `record_step_name_token(step_id, name, token, task_size)` 4参数
- **处理**: 用对方版本，这是对方重构后的函数签名，我们的旧签名会编译失败

### 9. `monitoring/csrc/native_engine_internal.h` — 合并
- 一处冲突：方法声明区
  - 我们: `add_task_from_config(cfg, tensor)` 旧签名 + 4个新方法（`process_native_hook`, `create_inline_hook_ticket`, `monitor_inline`）
  - 对方: `add_task_from_config(cfg, tensor, step_id)` 新签名（返回 pair）
- **处理**: 用对方的 `add_task_from_config` 新签名 + 保留我们的 4 个新方法声明
- **原因**: graph-safe 快速路径最终通过 GraphNativeDelegate 交给 native engine，这些方法是中间环节

### 10. `monitoring/csrc/unified.cpp` — 合并两边 includes
- 我们: `graph_native_delegate.cpp` + `graph_shadow_parser.cpp`
- 对方: `dmx_host_utils.cpp` + `future_process.cpp` + `clickhouse_client.cpp`
- **处理**: 全部保留（单翻译单元编译，各自的 cpp 互不冲突）

### 11. `monitoring/engine.py` — 合并 imports
- 只有一处冲突（import 行），对方大量修改被 git 自动合并
- 我们: `_encode_slice_native`, `Slice` (transformer_lens), `_QUEUE_SENTINEL`
- 对方: `AdvanceConfig`, `HostEngineConfig`
- **处理**: 保留对方的 `AdvanceConfig` import + 保留我们的 `_encode_slice_native` 和 `Slice` import，删除 `_QUEUE_SENTINEL`（对方已移除 Python fallback backend）
- **注意**: 我们的 graph-safe 路径在 `graph_engine.py`（独立文件），与 `engine.py` 是上下游关系

### 12. `transformers` (submodule) — 用我们的
- 我们 (`6e013cd3d`): inline `_mon_record` + `_mon_anchors` + `_mon_buf`/`_mon_frame_offset`（graph-safe pipeline 依赖）
- 对方 (`e4c0d530d`): 把 inline `_mon_record` 全部改回 `self.hook_xxx(x)` HookPoint 调用 + Qwen3 + hook 精简 + resid_pre fix
- **处理**: 用我们的版本（`git checkout --ours`），否则 graph-safe pipeline 直接坏掉
- **后续 TODO**:
  1. Cherry-pick 对方的 `resid_pre` fix（`e4c0d530d`）：`hook_resid_pre` 应在 `ln_1` 之前捕获
  2. Cherry-pick 对方的 hook 精简（`3d8aa4b89`）：`_normalize_hook_names` 去掉 `h.` alias + `transformer.` 前缀
  3. Cherry-pick 对方的 Qwen3 支持（`be8bf81ea` + `d251009fd`）
  4. **重构 `graph_monitor.py`**：hook 发现从 `named_modules()` 改为 `model.hook_dict.items()`
     - 好处：名字规范化（无 `transformer.` 前缀）、无冗余 alias、DB 名字与 slot 名字一致
     - 需要同步修改 `_mon_slot_xxx` 属性的设置逻辑

### 13. `tests/test_*.py` × 7 — 接受位置移动
- 对方把 `tests_monitoring/` 重命名为 `tests/`
- 我们的 7 个测试文件自动移到 `tests/` 下
- **处理**: `git add` 接受新位置

## Auto-merged 后的手动修复

### `monitoring/csrc/hooks.cpp` — 签名不一致修复
- 虽然 git auto-merge 没报冲突，但 `process_native_hook` 里的调用使用旧签名：
  - `add_task_from_config(cfg, tensor)` → 改为 `add_task_from_config(cfg, tensor, step_id)` + pair 返回
  - `record_step_name_token(step_id, name, token)` → 改为 `record_step_name_token(step_id, name, token, task_size)`
- **原因**: 对方重构了这两个函数的签名，auto-merge 只合并了定义但没更新我们的调用点

## Auto-merged (no conflict)
- `monitoring/Makefile` — 对方加了 clickhouse/pipelined_engine 编译，我们加了 NVTX 和 LDFLAGS
- `monitoring/csrc/native_engine.h` — 双方都加了新方法声明
- 对方新增的大量文件（ClickHouse, DMXHostEngine, Qwen3, benchmark scripts 等）
