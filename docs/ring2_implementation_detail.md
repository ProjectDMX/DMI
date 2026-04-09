# Ring2 完整实现详解

---

## 一、物理结构

Ring2 由三个预分配的内存区域组成，在 `AllocatedRing` (`ring_alloc.h`) 构造时一次性分配：

```
┌──────────────────────────────────────────────────────────────┐
│  GPU Device Memory                                            │
│                                                               │
│  ┌─ Payload Ring ─────────────────────────────────────────┐   │
│  │  cudaMalloc, 256 MiB (默认)                             │   │
│  │  纯 device memory，D2D 写入 + D2H 读出                   │   │
│  │  circular byte buffer，uint4 向量化拷贝                   │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─ Task Entries ─────────────────────────────────────────┐   │
│  │  cudaMallocManaged, CPU-preferred (cudaMemAdvise)       │   │
│  │  1024 个 TaskEntry × 64B = 64 KB                        │   │
│  │  GPU producer 写入 (PCIe posted writes, fire-and-forget) │   │
│  │  CPU drain thread 读取 (本地 DRAM read, 无 PCIe 流量)    │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─ Head Counters ────────────────────────────────────────┐   │
│  │  cudaMallocManaged, GPU-preferred (cudaMemAdvise)       │   │
│  │  task_head (uint64) + payload_head (uint64)             │   │
│  │  GPU producer L2/HBM 速度读写                            │   │
│  │  CPU drain thread 写入 via PCIe (低频)                   │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Host Pinned Memory                                           │
│                                                               │
│  ┌─ Pinned Staging Ring ──────────────────────────────────┐   │
│  │  cudaHostAlloc, 256 MiB (默认 = payload_ring_bytes)      │   │
│  │  drain thread 写 head, p2p thread 读 tail                │   │
│  │  D2H 目标缓冲区                                          │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Host Pageable Memory                                         │
│                                                               │
│  ┌─ TensorMetaFifo ──────────────────────────────────────┐   │
│  │  std::deque<TensorMeta> + std::deque<StepContext*>      │   │
│  │  Python 线程 push (forward 前), P2P 线程 pop             │   │
│  │  单 mutex 保护                                           │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Managed memory 放置策略**：

```cpp
// ring_alloc.h — GPU-preferred: producer kernel 在 L2/HBM 速度读写 head
cudaMemAdvise(task_head,    sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, gpu_loc);
cudaMemAdvise(payload_head, sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, gpu_loc);

// CPU-preferred: drain thread 在本地 DRAM 轮询 task entries，无 PCIe 流量
cudaMemAdvise(task_entries, entries_sz, cudaMemAdviseSetPreferredLocation, cpu_loc);
// 但 GPU 仍需写入（SetAccessedBy），通过 PCIe posted writes（fire-and-forget）
cudaMemAdvise(task_entries, entries_sz, cudaMemAdviseSetAccessedBy, gpu_loc);
```

---

## 二、RingState — 传入 kernel 的 POD 描述符

```cpp
// ring_state.h
struct RingState {
    TaskEntry*  task_entries;   // managed memory, 1024 slots
    uint64_t    task_cap;       // = 1024

    uint64_t*   task_head;      // managed memory (GPU-preferred), producer writes

    uint8_t*    payload_buf;    // device memory, 256 MiB
    uint64_t    payload_cap;    // = 256 MiB

    uint64_t*   payload_head;   // managed memory (GPU-preferred), producer writes
};
```

**值传递**进 kernel（6 个指针/整数，48B），CUDA graph capture-safe — 所有地址在分配后固定不变。

---

## 三、TaskEntry — 64 字节，cache-line 对齐

```cpp
// task_entry.h
struct alignas(64) TaskEntry {
    uint64_t ready_seq;           //  8B  序列守卫（SENTINEL 直到发布）
    uint64_t tensor_total_bytes;  //  8B  原始 tensor 字节数
    uint64_t payload_off1;        //  8B  payload ring 第一段偏移
    uint64_t payload_len1;        //  8B  第一段实际数据长度
    uint64_t payload_off2;        //  8B  第二段偏移（wrap 时）
    uint64_t payload_len2;        //  8B  第二段长度（0 表示无 wrap）
    uint8_t  _padding[16];        // 16B  填充到 64B
};
// static_assert(sizeof(TaskEntry) == 64)
```

`ready_seq = SENTINEL (0xFFFF...FFFF)` 表示未发布。`ready_seq = seq_no` 表示已发布，所有字段有效。

---

## 四、Producer Kernel — 核心数据路径

```
触发链: HookPoint.forward()
  → torch.ops.ring.producer(x_cont, hook_type, hook_id)
  → ring_producer_impl() [C++, 仅 capture 时执行]
  → hook_no_notify()
  → launch_producer()
  → producer_kernel<<<grid, 256>>>()
```

**Size-tiered grid 选择**（capture 时确定，baked into graph）：

| tensor 大小 | grid blocks | 总线程数 |
|------------|-------------|---------|
| <= 64 KB | 1 | 256 |
| <= 4 MB | 4 | 1,024 |
| <= 32 MB | 16 | 4,096 |
| > 32 MB | 64 | 16,384 |

**Kernel 内部三阶段**：

```
阶段 1: D2D Copy（所有线程参与）
  ├─ 读 *ring.task_head 和 *ring.payload_head（managed memory, L2 cached）
  ├─ payload_compute_spans(head, cap, alloc_bytes) → TwoSpan{off1,len1,off2,len2}
  ├─ Span 1: grid-stride uint4 向量化拷贝 activation → payload_buf[off1..off1+len1)
  └─ Span 2: 如果 wrap，拷贝剩余到 payload_buf[off2..off2+len2)

阶段 2: __threadfence()
  └─ 确保 D2D 写入对 CPU (drain thread) 和其他 GPU 单元全局可见

阶段 3: Last-block-arrives 发布元数据（仅最后完成的 block 的 thread 0）
  ├─ __syncthreads()
  ├─ atomicAdd(&g_block_done_counter, 1) → 检查是否是最后一个 block
  ├─ 构造 TaskEntry{tensor_total_bytes, off1, len1, off2, len2}
  ├─ task_publish(): __threadfence() → 写 ready_seq = task_head（原子发布）
  ├─ *ring.task_head = task_head + 1
  ├─ *ring.payload_head += alloc_bytes
  └─ g_block_done_counter = 0（为下一个 kernel 重置）
```

**Wrap-around 处理**：payload ring 是循环的。当分配跨越缓冲区末尾时，`payload_compute_spans` 返回两段：

```
                    off1          cap
payload_buf: [.....|XXXXXX|............]
                         ↓ wrap
payload_buf: [YYYY|......|XXXXXX|......]
              off2        off1
```

D2D copy 分两轮执行，TaskEntry 记录两段的 offset/length。

---

## 五、Pre-forward 容量检查 — prepare_step()

每步 forward 前由 Python `_prepare_wrapper` 调用一次：

```cpp
int RingEnginePy::prepare_step(uint64_t step_total_bytes, uint32_t num_hooks) {
    current_hook_idx = 0;  // 重置 hook 计数器

    effective_cap = min(payload_cap, staging_cap);

    // Case B: 单步超出容量 → cpu_direct fallback
    if (step_total_bytes > effective_cap || num_hooks > task_cap) {
        cudaStreamSynchronize(main_stream);
        drain.force_flush_and_wait();
        return STEP_CPU_DIRECT;  // = 2
    }

    // Case A 快速路径: 空间足够，无任何同步
    payload_avail = payload_cap - (cpu_payload_head - cpu_payload_tail_committed);
    task_avail    = task_cap - (cpu_task_head - cpu_task_tail_committed);

    if (step_total_bytes <= payload_avail && num_hooks <= task_avail) {
        drain.reserve(step_total_bytes, num_hooks);  // 推进 CPU-side head
        return STEP_RING_OK;  // = 0，~几十ns，无 CUDA 操作
    }

    // Case A 慢路径: ring 被之前的步骤占满 → sync + flush → 腾出空间
    cudaStreamSynchronize(main_stream);
    drain.force_flush_and_wait();
    drain.reserve(step_total_bytes, num_hooks);
    return STEP_RING_FLUSHED;  // = 1
}
```

**双重记账**：CPU-side 维护 shadow head/tail（`cpu_payload_head`, `cpu_payload_tail_committed`），和 GPU-side 的 `*ring.payload_head` 独立。`reserve()` 只推进 CPU-side head，确保下一个 `prepare_step` 看到正确的 available space。

---

## 六、Drain Thread — GPU→Host 搬运

独立 C++ 线程，运行 `drain_thread.cpp::loop()`：

```
while (running) {
    1. scan_ready():
       while (task_cpu_ready(entries, cap, visible_head)):
           // CPU DRAM 读 task_entries[visible_head % cap].ready_seq
           // 这是 managed memory CPU-preferred，本地读，无 PCIe
           if ready_seq == visible_head:
               push to scanned_ deque
               pending_entries++, pending_bytes += alloc_bytes
               visible_head++

    2. should_flush():
       检查 threshold（entry_count, byte_count, timeout_us）

    3. flush_state_update():
       释放 task slots: task_release_cpu(entries, cap, tail) → 写 SENTINEL
       推进 cpu_payload_tail

    4. enqueue_d2h():
       cudaMemcpyAsync(pinned_staging + stg_cursor,
                       payload_buf + gpu_cursor,
                       chunk, D2H, drain_stream)
       // 可能多段（payload ring + staging ring 都可能 wrap）

    5. cudaStreamSynchronize(drain_stream)  // 等 D2H 完成

    6. cpu_payload_tail_committed = cpu_payload_tail  // 安全更新

    7. submit_to_p2p():
       构造 DrainTask{data_ptr1, data_len1, data_ptr2, data_len2, alloc_bytes}
       push to task_queue_ → notify p2p thread

    8. trim_scanned(): pop 已处理的 entries

    9. cv_.wait_for(drain_poll_timeout_us)  // 睡 100us 或被 notify 唤醒
}
```

**force_flush_and_wait()**：Python 线程调用（GIL released），信号 drain thread 立刻 flush 所有 pending entries，阻塞直到完成。用于 `prepare_step` 的 Case A 慢路径。

---

## 七、P2P Thread — 元数据配对 + Per-request 切片

```
while (true) {
    n = drain.wait_for_tasks();  // 阻塞等待 drain thread 提交任务
    drain.pop_tasks(n, local);

    for each DrainTask:
        1. memcpy(pageable_buf, pinned_staging, tensor_bytes)  // pinned → pageable
        2. drain.notify_staging_freed_bytes(alloc_bytes)        // 释放 staging 空间

        3. fifo.pop(meta)  // 从 TensorMetaFifo 取对应的元数据
           // meta = {hook_name, shape, dtype, last_in_step}

        4. 如果是新 step 的第一个 hook: fifo.pop_context()
           // ctx = {model_id, shard_rank, requests[{req_id, start_token, end_token}]}

        5. tensor = byte_buf.view(dtype).reshape(shape)  // 重构 tensor

        6. for each request in ctx.requests:
               slice = slice_for_request(tensor, batch_idx, start, end, is_attn)
               submit_fn_(model_id, rank, req_id, act_name, layer, start, end, slice)
               // → ClickHouse insert queue

        7. if meta.last_in_step: delete ctx  // 释放 step context
}
```

**FIFO 对齐保障**：
- Python `pre_push_all_metas()` 按 `_active_specs` 顺序 push `TensorMeta`
- Producer kernel 按相同顺序执行（CUDA graph 内顺序固定）
- Drain thread 按 TaskEntry 到达顺序处理
- P2P thread 按 FIFO 顺序 pop → **1:1 对应，无需显式 ID 匹配**

---

## 八、TensorMetaFifo — 解耦元数据关联

```cpp
// tensor_meta.h
struct TensorMeta {
    std::string hook_name;
    std::vector<int64_t> shape;
    int dtype;          // at::ScalarType
    bool last_in_step;  // P2P thread 用来释放 StepContext
};

struct StepContext {
    std::string model_id;
    int32_t shard_rank;
    std::vector<RequestMeta> requests;  // {req_id, start_token, end_token}
};

class TensorMetaFifo {
    std::deque<TensorMeta> q_;
    std::deque<StepContext*> ctx_q_;  // 堆分配，ownership 转给 P2P thread
    std::mutex mu_;

    void push_step(StepContext* ctx, vector<TensorMeta>& metas);  // 一次 lock 推 ctx + 所有 metas
    bool pop(TensorMeta& out);     // P2P thread 逐个 pop
    StepContext* pop_context();     // P2P thread 在每步第一个 hook 时 pop
};
```

**设计理由**（论文 Section 4.2 "Decoupled metadata association"）：

Producer kernel 在 GPU 热路径上只写 **最小信息**（payload offset + length + ready flag = 64B）。完整语义信息（hook name, shape, dtype, request ID, token range）在 Python 层预计算后通过 TensorMetaFifo 传递，完全不经过 GPU。两条流在 P2P thread 汇合重建完整的 tensor record。

---

## 九、Null Mode — 全局禁用

```cuda
// producer.cu
__device__ bool g_ring_null_mode = false;

void set_ring_null_mode(bool enabled) {
    cudaMemcpyToSymbol(g_ring_null_mode, &enabled, sizeof(bool));
}

__global__ void producer_kernel(...) {
    if (g_ring_null_mode) return;  // 所有线程立即退出
    // ... 正常 D2D + metadata publish
}
```

- Warmup 期间开启：kernel 被 capture 进 graph 但不写数据
- Warmup 后关闭：replay 时 kernel 正常执行
- `cudaMemcpyToSymbol` 是同步的，写入对后续 `cudaGraphLaunch` 可见

---

## 十、端到端数据流总图

```
Python Thread                   GPU (main stream)                 Drain Thread              P2P Thread
─────────────                   ─────────────────                 ────────────              ──────────
_prepare_wrapper()
  ├ prepare_step(bytes, N)  ──→ [检查容量，reserve]
  ├ pre_push_all_metas()    ──→
  │   push TensorMeta ×N   ─────────────────────────────────────────────────────→ FIFO
  │   push StepContext      ─────────────────────────────────────────────────────→ ctx_q
  │
  └ return to HF/vLLM

model.forward()
  └ HookPoint.forward(x)
      x_cont = x.contiguous()
      ring.producer(x_cont)  ──→ producer_kernel<<<grid,256>>>
                                   ├ D2D: x → payload_ring
                                   ├ publish TaskEntry[head]
                                   └ advance heads
                                                                  scan_ready()
                                                                    └ poll TaskEntry[tail]
                                                                      (CPU DRAM read)
                                                                  should_flush() → yes
                                                                  enqueue_d2h()
                                                                    └ cudaMemcpyAsync
                                                                      payload→pinned
                                                                  sync drain_stream
                                                                  submit_to_p2p()
                                                                    └ push DrainTask ──────→ pop DrainTask
                                                                                            memcpy pinned→pageable
                                                                                            pop TensorMeta from FIFO
                                                                                            pop StepContext (if first)
                                                                                            reshape + slice per request
                                                                                            submit_fn → ClickHouse
```
