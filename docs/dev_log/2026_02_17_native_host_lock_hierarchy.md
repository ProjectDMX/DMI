# Native Engine + Host Engine 锁层级与 `resolve_all/result` 调用链梳理

> 目标：把当前实现里**所有关键锁**、锁层级、以及 `resolve_all` / `result` 的调用链和锁行为写清楚，便于排查“`resolve_all` 能过、`stop` 会卡”的差异。

## 1. 范围

- Native backend: `monitoring/csrc/native_engine_internal.h` + `api_submit.cpp` + `engine_core.cpp` + `hooks.cpp`
- Host engine (DMXHostEngine): `monitoring/csrc/pipelined_engine.hpp` + `batching_queue.hpp` + `future_process.cpp`
- Python orchestration: `monitoring/engine.py`

---

## 2. 锁清单（Native）

### 2.1 全局/成员锁

- `staging_mutex_`  
  - 保护：`open_steps_`, `sealed_steps_`, `step_name_tokens_`  
  - 定义：`monitoring/csrc/native_engine_internal.h:294-297`, `monitoring/csrc/native_engine_internal.h:214`
- `queue_mutex_` + `queue_cv_`  
  - 保护：native 工作队列 `queue_`、`stop_`  
  - 定义：`monitoring/csrc/native_engine_internal.h:315-320`
- `slots_mutex_`  
  - 保护：`slots_`（token -> `ResultSlot`）  
  - 定义：`monitoring/csrc/native_engine_internal.h:321-322`
- `pending_mutex_` + `pending_cv_`  
  - 用于 `resolve_all` barrier 等待 `pending_tasks_ == 0`  
  - 定义：`monitoring/csrc/native_engine_internal.h:324-326`
- `hook_config_mutex_`  
  - 保护：`hook_configs_`  
  - 定义：`monitoring/csrc/native_engine_internal.h:312-313`
- `enabled_mutex_`  
  - 保护：`enabled_hooks_`  
  - 定义：`monitoring/csrc/native_engine_internal.h:206-207`
- `pool_mutex_` / `ptr_mutex_`  
  - 保护：pinned pool 元数据与 `ptr_to_block_id_` 映射  
  - 定义：`monitoring/csrc/native_engine_internal.h:240-243`
- `host_copy_pool_->queue_mutex_` + `queue_cv_`  
  - 保护：host-copy 线程池队列  
  - 定义：`monitoring/csrc/native_engine_internal.h:277-279`

### 2.2 每个 Future 的局部锁

- `ResultSlot::mutex` + `ResultSlot::cv`  
  - 每个 token 一把锁，保护 `ready/has_error/consumed/tensor/error`  
  - 定义：`monitoring/csrc/native_engine_internal.h:101-109`

---

## 3. 锁清单（Host）

### 3.1 PipelinedEngine 级别锁

- `state_mu_`  
  - 生命周期状态：`started_`, `input_closed_` 等  
  - 定义：`monitoring/csrc/pipelined_engine.hpp:1204-1207`
- `done_mu_` + `done_cv_`  
  - join 等待：`stage_running_`, `total_running_`  
  - 定义：`monitoring/csrc/pipelined_engine.hpp:1210-1213`
- `fail_mu_`  
  - 保护线程失败信息 `failures_`  
  - 定义：`monitoring/csrc/pipelined_engine.hpp:1216-1217`
- `warn_mu_`  
  - 保护 warning 去重集合  
  - 定义：`monitoring/csrc/pipelined_engine.hpp:1220-1221`
- `prof_mu_`  
  - 保护 profiling 累计数据  
  - 定义：`monitoring/csrc/pipelined_engine.hpp:1224`

### 3.2 每个 stage 队列锁（WatermarkBatchingQueue）

- 每个队列对象内部只有一把 `mu_`，配合 `cv_can_enqueue_` / `cv_can_dequeue_`  
  - 定义：`monitoring/csrc/batching_queue.hpp:1144-1146`
- 典型路径：`enqueue` / `dequeue_batch` / `close` 全部在 `mu_` 内完成状态检查和 cvar wait/notify  
  - 参考：`monitoring/csrc/batching_queue.hpp:414-427`, `monitoring/csrc/batching_queue.hpp:546`, `monitoring/csrc/batching_queue.hpp:721`

---

## 4. 锁层级（Hierarchy）总结

## 4.1 Native 侧（观测到的层级）

1. `staging_mutex_ -> queue_mutex_`（仅特定路径）  
   - `close()` 中在持有 `staging_mutex_` 时调用 `dispatch_step(...)`，`dispatch_step` 可能再取 `queue_mutex_`  
   - 位置：`monitoring/csrc/api_submit.cpp:565-569` + `monitoring/csrc/engine_core.cpp:334`, `monitoring/csrc/engine_core.cpp:370`

2. `slots_mutex_ -> ResultSlot::mutex`（逻辑上分两段，不是同时长期持有）  
   - 先 `get_slot()` 取 shared_ptr（`slots_mutex_`），再对 slot 上锁等待/读取  
   - 位置：`monitoring/csrc/engine_core.cpp:170-177`, `monitoring/csrc/api_submit.cpp:490`, `monitoring/csrc/api_submit.cpp:513`

3. `pool_mutex_ -> ptr_mutex_`（顺序固定）  
   - pool block 状态更新后，再更新/删除 ptr map  
   - 位置：`monitoring/csrc/engine_core.cpp:650`, `monitoring/csrc/engine_core.cpp:701`, `monitoring/csrc/engine_core.cpp:713`, `monitoring/csrc/engine_core.cpp:724`

4. `pending_mutex_` 仅用于 barrier wait（不与上面几把锁形成固定嵌套）
   - 位置：`monitoring/csrc/api_submit.cpp:469-472`

## 4.2 Host 侧（观测到的层级）

1. `state_mu_ -> done_mu_`（start 路径会出现）
   - `start()` 先持有 `state_mu_`，内部重置 running 计数时取 `done_mu_`  
   - 位置：`monitoring/csrc/pipelined_engine.hpp:376-394`

2. `done_mu_` 用于 join 条件等待，不与队列 `mu_` 强嵌套
   - `join()` wait `stage_running_[idx]==0`  
   - 位置：`monitoring/csrc/pipelined_engine.hpp:546-555`

3. `fail_mu_` / `warn_mu_` / `prof_mu_` 基本独立，不参与主数据流阻塞

## 4.3 跨子系统锁关系

- Host stage0 线程在 `ProcessFutureStage` 里调用 `backend_future.result(...)`，会进入 Native 的 `slots_mutex_` / `ResultSlot::mutex` 逻辑。
- Host engine 的 `done_mu_` 与 Native 的 `slots_mutex_` 没有直接嵌套路径（更像“等待链”，不是经典锁反转死锁）。

---

## 5. `resolve_all` 调用链与锁链

## 5.1 调用链

1. Python: `MonitoringEngine.resolve_all()`  
   - `monitoring/engine.py:592-636`
2. C++: `NativeMonitoringEngine::resolve_all()`  
   - `monitoring/csrc/native_engine.cpp:300`
3. C++ impl: `Impl::resolve_all()`  
   - `monitoring/csrc/api_submit.cpp:435-474`

## 5.2 锁逻辑链

1. `staging_mutex_`：搬运 `open_steps_` 到 ready（并清空）  
   - `monitoring/csrc/api_submit.cpp:441-450`
2. 无锁 dispatch ready（`dispatch_step`）  
   - `monitoring/csrc/api_submit.cpp:452-454`
3. 再次 `staging_mutex_`：搬运 `sealed_steps_` 到 ready（并清空）  
   - `monitoring/csrc/api_submit.cpp:458-463`
4. 无锁 dispatch ready  
   - `monitoring/csrc/api_submit.cpp:465-467`
5. `pending_mutex_ + pending_cv_.wait(...)` 等待 `pending_tasks_ == 0`  
   - `monitoring/csrc/api_submit.cpp:469-472`

## 5.3 关键结论

- `resolve_all()` **不调用** `future_result()` / `result()`。
- 它只做两件事：**flush staging** + **等 pending 清零**。

---

## 6. `result`（Host stage0 消费 future）调用链与锁链

## 6.1 调用链

1. end_step 后提交 DB payload  
   - `MonitoringEngine._submit_pending_db_step()` 调 `host_engine.submit(...)`  
   - `monitoring/engine.py:438-455`
2. Host stage0 执行 `ProcessFutureStage::ProcessFuture`  
   - `monitoring/csrc/future_process.cpp:44-119`
3. 对每个 future 调 `backend_future.result(..., called_from_cpp=true)`  
   - `monitoring/csrc/future_process.cpp:93`
4. 进入 native `Impl::future_result(...)`  
   - `monitoring/csrc/api_submit.cpp:500-552`

## 6.2 锁逻辑链（单 token）

1. `get_slot(token)`：`slots_mutex_` 查表  
   - `monitoring/csrc/engine_core.cpp:170-177`
2. `slot->mutex` 上锁并等待 `slot->cv`（ready）  
   - `monitoring/csrc/api_submit.cpp:513-529`
3. 成功后根据状态返回；必要时 `remove_slot(token)` 再次走 `slots_mutex_`  
   - `monitoring/csrc/api_submit.cpp:534-551`, `monitoring/csrc/engine_core.cpp:179-182`

## 6.3 谁负责唤醒 `slot->cv`

- Native worker 在 `store_result` / `store_exception` 中：
  - 持有 `slot->mutex` 写 ready/结果
  - `slot->cv.notify_all()`
  - `pending_tasks_--` 并 `pending_cv_.notify_all()`
  - 位置：`monitoring/csrc/engine_core.cpp:828-843`, `monitoring/csrc/engine_core.cpp:845-859`

---

## 7. `stop` 调用链与锁链（对比 `resolve_all`）

## 7.1 调用链

1. `MonitoringEngine.close()` -> `self._host_engine.stop()`  
   - `monitoring/engine.py:723-727`
2. `DMXHostEngine.stop(graceful=true)` -> `close_input()` + `join()`  
   - `monitoring/csrc/pipelined_engine.hpp:517-555`

## 7.2 锁逻辑

1. `close_input()`：`state_mu_` + 关闭 stage0 input queue  
2. `join()`：按 stage 等 `stage_running_[i] == 0`（`done_mu_ + done_cv_`）

注意：

- `stop()` 不会等待 native `pending_tasks_ == 0`。
- 它等待的是 host stage 线程退出；而 stage0 线程退出前可能卡在 `future_result` 的 `slot->cv.wait(...)`。

---

## 8. 对“`resolve_all` 能过但 `stop` 卡”的解释（锁视角）

1. 两者等待对象不同：
   - `resolve_all`：native pending barrier（`pending_cv`）
   - `stop`：host stage 线程退出（`done_cv`）
2. 即使 `open_steps_/sealed_steps_` 为 0，也仍可能有大量 pending 在 native queue/worker 中。
3. 这时 `stop` 可能卡在 stage0 的 `result()`；而 `resolve_all` 先等 native pending 清零，再进入 `stop` 时 stage0 更容易退出。

---

## 9. 补充：Host stage 队列锁说明

- 每个 stage 队列（`WatermarkBatchingQueue`）内部单锁模型（`mu_`），enqueue/dequeue/close 都在这一把锁上协调；没有再嵌套 Native 锁。
- Host `join(done_mu_)` 与 Queue `mu_` 是分离的，`done_cv.wait` 会释放 `done_mu_`，不长期占锁。

---

## 10. 可直接用于对照的关键代码点

- `resolve_all` wait pending：`monitoring/csrc/api_submit.cpp:469-472`
- `future_result` wait slot ready：`monitoring/csrc/api_submit.cpp:513-529`
- `store_result` 唤醒 slot/pending：`monitoring/csrc/engine_core.cpp:836-840`
- host stop/join：`monitoring/csrc/pipelined_engine.hpp:517-555`
- host stage0 调 result：`monitoring/csrc/future_process.cpp:93`

