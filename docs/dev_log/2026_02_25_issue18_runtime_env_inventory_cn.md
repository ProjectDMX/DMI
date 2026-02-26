# Issue #18 阶段二前置：Runtime 环境变量盘点（中文）

日期：2026-02-25  
分支基线：`HF_Prometheus` @ `f8123edc3`

## 1. 目的与范围

本文只回答一件事：**当前还在生效的 runtime 环境变量有哪些、各自控制什么行为**。  
不包含编译期变量的迁移方案（编译期开关单列在附录）。

## 2. 当前 Runtime 环境变量总表

### 2.1 Python 引擎/Hook 侧

| 变量 | 默认值 | 读取时机 | 作用 | 代码位置 |
|---|---:|---|---|---|
| `MON_ENGINE_DEBUG` | `0` | `MonitoringEngine.__init__`；`close()` 再读一次 | 打印 Python 侧调试日志（submit/start/end/close 等） | `monitoring/engine.py:81`, `monitoring/engine.py:747` |
| `MON_NATIVE_BATCH` | `0` | `MonitoringEngine.__init__` | 启用 SoA 聚合路径，`end_step` 走 `submit_step_soa` | `monitoring/engine.py:170`, `monitoring/engine.py:592` |
| `MON_NATIVE_BUILDER` | `1` | `MonitoringEngine.__init__` | 启用 C++ builder/append 路径分流 | `monitoring/engine.py:172`, `monitoring/hook_points.py:962` |
| `MON_NATIVE_CALLBACK` | `1` | `MonitoringEngine.__init__` | 启用全局 native callback 路径（减少 Python hook 开销） | `monitoring/engine.py:174`, `monitoring/hook_points.py:966` |
| `MON_ENGINE_STATS` | `0` | `MonitoringEngine.__init__`；`hook_points` 模块导入时 | 引擎端统计输出；同时可触发 hook 统计 | `monitoring/engine.py:178`, `monitoring/hook_points.py:34` |
| `MON_HOOK_STATS` | `0` | `hook_points` 模块导入时 | 仅 hook 侧统计（不依赖 engine stats） | `monitoring/hook_points.py:34-35` |
| `MON_ENGINE_SLICE_STATS` | `0` | `task` 模块导入时 | 统计 slice 模式分布（identity/int/slice/array） | `monitoring/task.py:20` |
| `TL_ENABLE_NVTX` | `0` | 混合：部分导入时，部分运行时多处重读 | 控制 Python 侧 NVTX range 标记 | `monitoring/hook_points.py:33`, `monitoring/engine.py:236/566/853` |

### 2.2 Native C++ 运行时（D2H/内存/拷贝）

| 变量 | 默认值 | 读取时机 | 作用 | 代码位置 |
|---|---:|---|---|---|
| `MON_NATIVE_TO_CPU` | `0`(off) | Native engine 构造时 | 开启 GPU->CPU offload | `monitoring/csrc/engine_core.cpp:27-29` |
| `MON_NATIVE_PINNED` | `1` | Native engine 构造时 | D2H 时优先使用 pinned memory | `monitoring/csrc/engine_core.cpp:30-32` |
| `MON_NATIVE_PINPOOL` | 自动（当 `TO_CPU=1` 且 `PINNED=1` 时默认开） | Native engine 构造时 | pinned pool 总开关 | `monitoring/csrc/engine_core.cpp:42-46` |
| `MON_NATIVE_PINPOOL_BINS_KB` | `256,512,1024,2048,4096,8192` | Native engine 构造时 | 自定义 pool bin 桶大小（KB） | `monitoring/csrc/engine_core.cpp:47-67` |
| `MON_NATIVE_PINPOOL_MAX_MB` | `512` | Native engine 构造时 | pool 总容量上限（MB） | `monitoring/csrc/engine_core.cpp:68-71` + `monitoring/csrc/native_engine_internal.h:233` |
| `MON_NATIVE_HOST_COPY_THREADS` | `0` | Native engine 构造时 | host copy 线程池并行度（`>0` 启用线程池） | `monitoring/csrc/engine_core.cpp:75-88` |
| `MON_NATIVE_HOST_COPY_QUEUE_SIZE` | `512` | Native engine 构造时（仅在线程池启用时） | host copy 队列深度 | `monitoring/csrc/engine_core.cpp:80-83`, `monitoring/csrc/native_engine_internal.h:277` |
| `MON_NATIVE_PINNED_INDEX` | `0` | 每次 array slice 时读取 | array 索引构造是否走 pinned staging | `monitoring/csrc/engine_core.cpp:646-648` |

## 3. 已在 Stage-1 删除的无效 Runtime 变量

以下变量已移除（原因：读到后不影响行为）：

- `MON_NATIVE_AUTOCLEAR`
- `MON_NATIVE_STEP_STATS`
- `MON_NATIVE_PINPOOL_SLOTS_PER_BIN`
- `MON_NATIVE_PIN_THRESH_BYTES`

参考：`docs/dev_log/2026_02_25_issue18_env_audit_stage1.md`

## 4. 当前实现的注意点（对后续重构有影响）

1. 读取时机不一致：
- `MON_HOOK_STATS`、`MON_ENGINE_SLICE_STATS` 在模块导入时冻结；
- 部分 `TL_ENABLE_NVTX` 在运行时重读；
- 大多数 `MON_NATIVE_*` 在 native engine 构造时读取一次。

2. 同一变量跨层生效路径不同：
- `MON_ENGINE_STATS` 同时影响 engine 侧统计和 hook 侧统计开关来源。

3. 路径分流复杂度仍高：
- `MON_NATIVE_BATCH` / `MON_NATIVE_BUILDER` / `MON_NATIVE_CALLBACK` 组合决定四种不同数据路径。

## 5. 阶段二建议（简版）

1. 先收敛“读取时机”：尽量改成 engine 构造期单点确定。  
2. 把长期默认值路径固化，减少组合开关。  
3. 把确实需要在线调参的少量参数迁入 `MonitoringConfig`。  
4. 最后再做 `MON_NATIVE_*` 运行时代码零读取的验收 grep。

## 5.1 已确认决策：清除 `MON_NATIVE_BATCH`

结论：`MON_NATIVE_BATCH` 进入“清除”范围，不再作为长期 runtime 开关保留。  
要求是**删变量 + 删相关代码路径**，不能只移除 env 读取。

需要一并清理的实现面（后续 PR 落地）：

1. Python 分支与聚合缓存
- `monitoring/engine.py` 中 `_native_batch_enabled` 与 `self._native_batch` 相关逻辑。
- `end_step()` 中 `submit_step_soa` 分支（SoA 分支整体删除）。

2. Hook 侧 SoA 聚合路径
- `monitoring/hook_points.py` 中 `native_batch_active` 分支及 SoA 聚合字典构建逻辑。

3. Native 接口与绑定
- `monitoring/csrc/native_engine.h/.cpp` 与 `monitoring/csrc/bindings.cpp` 中 `submit_step_soa` 暴露接口。
- `monitoring/csrc/api_submit.cpp` 中 `submit_step_soa` 实现。

4. 测试与脚本
- 删除/更新所有设置 `MON_NATIVE_BATCH` 的测试、example、benchmark 启动脚本。
- 回归验证覆盖默认路径（`builder + callback`）与非 native fallback 路径。

## 5.2 已确认决策：清除 `MON_NATIVE_BUILDER` 与 `MON_NATIVE_CALLBACK`（行为固定为 `true`）

结论：`MON_NATIVE_BUILDER` 与 `MON_NATIVE_CALLBACK` 也进入“清除”范围。  
最终行为固定为：

- `native_builder_enabled = true`
- `native_callback_enabled = true`

要求是**删变量 + 删分支**，不保留“可关闭”路径。

需要一并清理的实现面（后续 PR 落地）：

1. Python 引擎配置读取
- 删除 `monitoring/engine.py` 中 `_native_builder_enabled` / `_native_callback_enabled` 的 env 读取。
- 改为常量语义（构造后固定 true），并清理相关条件判断。

2. Hook 分支收敛
- 删除 `monitoring/hook_points.py` 中 `native_builder_python`、`native_using and ... not native_builder_python` 等分流分支。
- 保留并强化默认主路径：`native_callback_active`（全局 callback + collect futures）。
- 纯 Python fallback（无 native backend）保留。

3. 旧接口路径评估与移除
- 评估并清理仅用于 `builder/callback` 关闭场景的路径（如 Python 侧 `add_task` 回填分支）。
- 保留当前主路径实际依赖的 native 接口。

4. 测试与脚本
- 删除/更新所有设置 `MON_NATIVE_BUILDER`、`MON_NATIVE_CALLBACK` 的测试、example、benchmark 启动脚本。
- 测试矩阵从“组合开关”改为“默认主路径 + 无 native fallback”。

## 5.3 已确认决策：清除 `MON_ENGINE_DEBUG`（仅保留 C++ backend）

结论：由于后续明确只使用 C++ native backend，`MON_ENGINE_DEBUG` 不再保留。  
该变量只影响 Python 侧 `print` 调试日志，不影响核心功能。

需要一并清理的实现面（后续 PR 落地）：

1. Python 引擎调试分支
- 删除 `monitoring/engine.py` 中 `MON_ENGINE_DEBUG` 读取（包括 `__init__` 与 `close()` 里的重复读取）。
- 删除 `_debug` 相关打印分支。

2. Python fallback 相关日志/路径
- 在“仅 C++ backend”策略下，清理 `_PythonBackend` 调试路径与不再需要的 fallback 分支。
- 对于关键错误（如 host submit 失败）改为固定告警/异常，不依赖 debug 开关。

3. 测试与脚本
- 删除设置 `MON_ENGINE_DEBUG` 的调试脚本入口（或改为配置级 debug，非 env）。

## 5.4 已确认决策：清除 `MON_HOOK_STATS`、`MON_ENGINE_SLICE_STATS`；保留 `MON_ENGINE_STATS`、`TL_ENABLE_NVTX` 但收敛为 `MonitoringConfig.debug`

结论（基于“仅 C++ backend”后续方向）：

- 直接清除：`MON_HOOK_STATS`、`MON_ENGINE_SLICE_STATS`
- 保留能力但去 env：`MON_ENGINE_STATS`、`TL_ENABLE_NVTX`，收敛为 `MonitoringConfig.debug` 单一开关

理由：

1. `MON_HOOK_STATS` 与 `MON_ENGINE_SLICE_STATS` 都是 Python 侧统计入口，且在模块导入期读取，行为隐式且不稳定；在只保留 C++ backend 的方向下价值低于维护成本。  
2. `MON_ENGINE_STATS` 与 `TL_ENABLE_NVTX` 仍有调试价值，但更适合作为显式配置项，而不是 runtime env 分散读取。

需要一并清理的实现面（后续 PR 落地）：

1. 删除两个待清理变量的读取与分支
- 删除 `monitoring/hook_points.py` 中 `MON_HOOK_STATS` 的读取与统计开关分支。
- 删除 `monitoring/task.py` 中 `MON_ENGINE_SLICE_STATS` 的读取与统计累积分支。

2. 收敛 debug 配置
- 在配置层使用 `MonitoringConfig.debug: bool` 作为唯一调试开关：
  - `debug=true`：等价开启 stats + NVTX
  - `debug=false`：等价关闭 stats + NVTX
- 由 `MonitoringEngine` 构造期一次性读取 `MonitoringConfig.debug`，避免运行时多处 `os.getenv`。

3. 代码清洁目标
- 删除与 `MON_HOOK_STATS`、`MON_ENGINE_SLICE_STATS` 相关的死代码与无效测试设置。
- 验收标准：runtime 代码中不再出现这两个变量名（`rg` 全仓为 0）。

## 5.5 已确认决策：清除 `MON_NATIVE_TO_CPU`、`MON_NATIVE_PINNED`、`MON_NATIVE_PINPOOL`（行为固定为 `true`）

结论：

- 这三个 runtime env 变量都进入“清除”范围：
  - `MON_NATIVE_TO_CPU`
  - `MON_NATIVE_PINNED`
  - `MON_NATIVE_PINPOOL`
- 目标行为固定为：
  - `to_cpu_enabled = true`
  - `pinned_enabled = true`
  - `pinned_pool_enabled = true`

要求是**删变量 + 删分支 + 删回退路径**，不保留“关闭 offload / pinned / pinpool”的运行时开关。

需要一并清理的实现面（后续 PR 落地）：

1. Native 构造期 env 读取与默认值逻辑
- 删除 `monitoring/csrc/engine_core.cpp` 中对上述三变量的 `getenv` 读取。
- 删除 `MON_NATIVE_PINPOOL` “自动默认”推导逻辑（`to_cpu && pinned`）并改为固定开启。

2. 相关条件分支与回退路径
- 清理仅在 `to_cpu/pinned/pinpool` 关闭时才会走到的分支。
- 确保 D2H 路径统一落到“pinned + pinpool”主路径，避免保留死分支。

3. 脚本与测试
- 删除测试、example、benchmark 中对上述三变量的设置。
- 回归验证主路径稳定性（包含大张量传输与高并发 copy 场景）。

4. 验收标准
- runtime 代码中不再出现这三个变量名（`rg` 全仓为 0）。

## 5.6 已确认决策：保留 `MON_NATIVE_PINPOOL_BINS_KB`、`MON_NATIVE_PINPOOL_MAX_MB` 功能，但迁入 Config（默认值不变）

结论：

- 这两个参数的功能保留，不删除：
  - `MON_NATIVE_PINPOOL_BINS_KB`
  - `MON_NATIVE_PINPOOL_MAX_MB`
- 但不再作为 runtime env 读取入口，迁移为显式配置项 `MonitoringConfig.advance.*`。
- 默认值维持当前实现：
  - `pinpool_bins_kb = [256, 512, 1024, 2048, 4096, 8192]`
  - `pinpool_max_mb = 512`

需要一并清理的实现面（后续 PR 落地）：

1. 配置层
- 新增对应配置字段并提供上述默认值。
- 在 `MonitoringEngine` 构造阶段将配置透传给 native backend（单点设置，避免分散读取）。

2. Native 层
- 删除 `engine_core.cpp` 中这两个变量的 `getenv` 解析。
- 保留现有 pool bin / max cap 逻辑本身，只改参数来源。

3. 脚本与测试
- 删除 benchmark/example/test 中对这两个 env 的设置方式（若存在）。
- 改为通过 config 传参；新增至少一组“自定义 bins/max”回归用例。

4. 验收标准
- runtime 代码中不再出现 `MON_NATIVE_PINPOOL_BINS_KB`、`MON_NATIVE_PINPOOL_MAX_MB`（`rg` 全仓为 0）。

## 5.7 已确认决策：`MON_NATIVE_HOST_COPY_THREADS` 迁入 Config（默认值 `0`）

结论：

- `MON_NATIVE_HOST_COPY_THREADS` 功能保留，但从 runtime env 迁移到配置项。
- 默认值保持当前语义：`host_copy_threads = 0`（即默认不启用 host-copy 线程池）。

需要一并清理的实现面（后续 PR 落地）：

1. 配置层
- 新增/补齐 `host_copy_threads` 配置字段，默认 `0`。
- 在 `MonitoringEngine` 构造期透传到 native backend，避免 C++ 层直接读取该 env。

2. Native 层
- 删除 `engine_core.cpp` 中 `MON_NATIVE_HOST_COPY_THREADS` 的 `getenv` 读取与解析入口。
- 保留线程池实现与行为，仅改参数来源。

3. 脚本与测试
- 删除 benchmark/example/test 中通过 env 配置 `MON_NATIVE_HOST_COPY_THREADS` 的方式（若存在）。
- 增加配置驱动的回归：`0`（关闭）与 `>0`（开启）两组场景。

4. 验收标准
- runtime 代码中不再出现 `MON_NATIVE_HOST_COPY_THREADS`（`rg` 全仓为 0）。

## 5.8 已确认决策：清除 `MON_NATIVE_PINNED_INDEX`

结论：

- `MON_NATIVE_PINNED_INDEX` 进入“清除”范围，不再保留为 runtime env 开关。
- 该变量只影响 `SliceMode::Array` 的索引构造路径（pinned staging vs 直接 device index），与当前主采样路径（step/request stride）无直接耦合。

要求是**删变量 + 删分支**，保留单一路径实现（`SliceMode::Array` 统一使用直接 device index 构造）。

需要一并清理的实现面（后续 PR 落地）：

1. Native `apply_slice` 分支收敛
- 删除 `monitoring/csrc/engine_core.cpp` 中 `MON_NATIVE_PINNED_INDEX` 的 `getenv` 读取。
- 删除 `pin_index` 条件分支，保留一条 `index_select` 实现路径。

2. 脚本与测试
- 清理所有设置 `MON_NATIVE_PINNED_INDEX` 的脚本/测试（若存在）。
- 补充 array-slice 回归用例，确保功能与结果一致（仅性能路径简化）。

3. 验收标准
- runtime 代码中不再出现 `MON_NATIVE_PINNED_INDEX`（`rg` 全仓为 0）。

---

## 6. PR 模板（English）

```markdown
## Summary
This PR removes legacy runtime env toggles and consolidates runtime controls into `MonitoringConfig`.

## Final Design
- `MonitoringConfig.debug: bool` is the only debug switch.
  - `debug=true`: enable both engine stats and NVTX.
  - `debug=false`: disable both.
- `AdvanceConfig` is introduced under `MonitoringConfig` for runtime tuning:
  - `pinpool_bins_kb` (default: `[256, 512, 1024, 2048, 4096, 8192]`)
  - `pinpool_max_mb` (default: `512`)
  - `host_copy_threads` (default: `0`)
  - `host_copy_queue_size` (default: `512`)

## Removed Runtime Env Vars
- `MON_NATIVE_BATCH`
- `MON_NATIVE_BUILDER`
- `MON_NATIVE_CALLBACK`
- `MON_ENGINE_DEBUG`
- `MON_HOOK_STATS`
- `MON_ENGINE_SLICE_STATS`
- `MON_NATIVE_TO_CPU`
- `MON_NATIVE_PINNED`
- `MON_NATIVE_PINPOOL`
- `MON_NATIVE_PINNED_INDEX`

## Migrated from Env to Config
- `MON_ENGINE_STATS` + `TL_ENABLE_NVTX` -> `MonitoringConfig.debug`
- `MON_NATIVE_PINPOOL_BINS_KB` -> `MonitoringConfig.advance.pinpool_bins_kb`
- `MON_NATIVE_PINPOOL_MAX_MB` -> `MonitoringConfig.advance.pinpool_max_mb`
- `MON_NATIVE_HOST_COPY_THREADS` -> `MonitoringConfig.advance.host_copy_threads`
- `MON_NATIVE_HOST_COPY_QUEUE_SIZE` -> `MonitoringConfig.advance.host_copy_queue_size`

## Fixed Runtime Behavior
- Native builder path is always enabled.
- Native callback path is always enabled.
- D2H offload is always enabled.
- Pinned memory path is always enabled.
- Pin pool is always enabled.

## Cleanup Guarantees
- Remove all associated env reads and dead branches across Python/C++ paths.
- Update benchmark/example/test scripts to use config, not env.
- Verify removed env names are absent from runtime code via repository grep.

## Additional Changes (for Review)
- Host pipeline log noise is now debug-gated:
  - `worker X in future_process introduced ... tensor copies due to non-contiguous`
  - only prints when `process_future(debug=true)`.
- `StageConfig.process_future(...)` adds optional `debug: bool = false`.
  - backward compatible by default (silent unless debug is enabled).
- Host pipeline builders in benchmark/example/validation scripts now pass debug from `MonitoringConfig.debug`.
- README and Quick Start examples are updated to remove old runtime env usage and reflect config-driven runtime/debug setup.

## Issue Link
Part of #18
```
