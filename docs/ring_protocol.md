# Ring Offload Protocol

Condition-tensor-gated, deadlock-free publish/consume protocol for
the CUDA-graph-safe LLM internal-state offload system.  One tensor =
one task entry (no chunking).

---

## Overview

The offload pipeline uses two preallocated GPU rings plus a condition
tensor to move LLM internal tensors from device to host:

| Component | Role |
|------|------|
| **Task ring** (control ring) | Fixed-size FIFO of `TaskEntry` structs (64B each); carries metadata and payload descriptors |
| **Payload ring** (byte ring) | Circular byte buffer; holds the raw tensor bytes (D2D from model tensors) |
| **Condition tensor** | `uint32[]` on GPU; gates producer kernels via `cuStreamWaitValue32` |

All are preallocated before graph capture and remain at fixed device addresses.

**Requires `CUDA_MODULE_LOADING=EAGER`** (set before process start).
`cuStreamWaitValue32` deadlocks with CUDA 12.2+ lazy module loading.

---

## Data Structures

### TaskEntry (64 bytes, 64-byte aligned)

One CPU cache line.  Managed memory preferred on CPU (drain thread polls
via fast local DRAM read; GPU writes via PCIe posted writes).

```
offset  0  ready_seq           uint64  sequence guard (SENTINEL until published)
offset  8  tensor_total_bytes  uint64  raw tensor bytes
offset 16  payload_off1        uint64  first payload span offset in payload_buf
offset 24  payload_len1        uint64  first payload span length
offset 32  payload_off2        uint64  second payload span offset (0 if single span)
offset 40  payload_len2        uint64  second payload span length (0 if single span)
offset 48  device_src_ptr      ptr     large tensor bypass: device source pointer (null for normal)
offset 56  _padding[8]         uint8   padding to 64 bytes
```

Large tensor bypass: `device_src_ptr != nullptr`.  Drain D2H's directly
from this pointer.  Padded allocation size derived: `align_up(tensor_total_bytes, 16)`.

### Condition Tensor Values

| Value | Name | Written by | Meaning |
|---|---|---|---|
| 0 | `COND_RESET` | kernel (normal) | Done, ready for next forward |
| 1 | `COND_PENDING` | kernel (large bypass) | Kernel done, drain must D2H + ack to 0 |
| 3 | `COND_GRANT_TASK_ONLY` | drain / prepare_forward | Large tensor: task slot available |
| 4 | `COND_GRANT_FULL` | drain / prepare_forward | Normal tensor: task slot + ring space |

### PayloadRingState (logical view)

```
payload_buf[payload_ring_bytes]  — circular byte buffer in device memory
payload_head                     — uint64, managed memory (GPU-preferred), producer writes
payload_tail                     — CPU-only shadow (cpu_payload_tail_), NOT in GPU memory
```

Free bytes = `payload_ring_bytes - (payload_head - payload_tail)`.
Physical offset = `logical_position % payload_ring_bytes`.

Tail pointers are CPU-only shadows in the drain thread.  The producer
kernel never reads tail — it is gated by the condition tensor instead.

---

## Publish Protocol (producer kernel)

The producer kernel never spins.  Backpressure is handled by
`cuStreamWaitValue32` on the condition tensor BEFORE the kernel is
launched (host side, in `hook_no_notify`).

### Normal path (tensor fits in ring):

1. **Host: cuStreamWaitValue32** — `wait(d_condition[hook_idx] >= 4)`
2. **Kernel: compute payload spans** — `payload_compute_spans(phead, pcap, alloc_bytes)`
3. **Kernel: D2D copy** — vectorized `uint4` copy into `payload_buf` spans
4. **Kernel: publish TaskEntry** — `__threadfence()` then `ready_seq = seq_no`
5. **Kernel: advance heads** — `task_head++`, `payload_head += alloc_bytes`
6. **Kernel: reset condition** — `d_condition[hook_idx] = COND_RESET (0)`

### Large bypass path (tensor exceeds ring):

1. **Host: cuStreamWaitValue32** — `wait(d_condition[hook_idx] >= 3)`
2. **Kernel: set condition** — `d_condition[hook_idx] = COND_PENDING (1)` **BEFORE task_publish** (race prevention: drain must see PENDING before the entry)
3. **Kernel: publish TaskEntry** — `device_src_ptr = src`, no D2D, no payload
4. **Kernel: advance task_head** — `task_head++` (no payload_head advance)
5. **Host: cuStreamWaitValue32** — `wait(d_condition[hook_idx] == 0)` (drain ack)

---

## Consume Protocol (drain thread)

The drain thread runs outside the CUDA graph.  Uses `mgmt_mu_` lock for
state management.  Stream ops (D2H, H2D) happen outside the lock.

### prepare_forward (Python thread, before each forward):

1. **Lock mgmt_mu_** — reclassify old entries, reset per-forward state
2. **Compute conditions** — grant hooks based on available resources (committed payload tail)
3. **Enqueue H2D** — condition delta `[0, N)` on main stream (no sync)
4. **Unlock** — return, forward starts

### Drain loop (scan + flush):

1. **Lock mgmt_mu_** — `scan_ready()` polls task_entries (CPU DRAM read), classify old/current
2. **should_flush()** — forced flush when `scanned_complete_ == granted_count_ && blocked_count_ > 0`
3. **flush_state_update** — release task slots, advance tails, `grant_next_hooks()`
4. **Unlock** — stream ops outside lock
5. **D2H payload** — `cudaMemcpyAsync(staging ← payload_buf, drain_stream)`
6. **H2D conditions** — delta only `[N, N+K)` on drain stream (after D2H, ordered)
7. **Sync** — one `cudaStreamSynchronize` for both
8. **OCC re-check** — lock, update `committed_tail`, re-grant if `prepare_forward` leaked in
9. **Push tasks** — `submit_to_p2p()` + `trim_scanned()` (separate to avoid mgmt_mu_ self-deadlock)

### Large tensor handling (inline in scan_ready):

Flush pending normal entries first (FIFO ordering), then:
1. `cudaMemcpyAsync` D2H from `device_src_ptr` to pageable buffer
2. Ack: `h_condition_[hook_idx] = COND_RESET`, H2D, sync
3. Push bypass DrainTask to p2p thread

---

## CUDA Graph Constraints

The producer post-op must be **capture-safe**:

| Requirement | How enforced |
|-------------|--------------|
| No allocation during capture | All buffers (rings, condition tensor) preallocated via `init_hooks()` |
| No host synchronisation | No `cudaDeviceSynchronize` / `cudaStreamSynchronize` in captured code |
| Stable pointers | Ring buffer + condition tensor addresses fixed after allocation |
| `cuStreamWaitValue32` captured | Wait address + threshold baked into graph nodes |
| `CUDA_MODULE_LOADING=EAGER` | Prevents lazy loading deadlock with captured wait nodes |

The drain loop and host pipeline run **outside** the captured graph.

### Drain thread wake-up in CUDA graph mode

In the non-graph (eager) path, each producer kernel is followed by a
`cudaLaunchHostFunc` that calls `drain_thread.notify()` to wake the drain
thread.  This cannot be used in CUDA graphs because `cudaLaunchHostFunc` is
captured as a **host node**, causing ~18 μs GPU→CPU→GPU round-trip per hook
per graph replay (see `ring_overhead.md` for measured data).

Instead, the CUDA graph path uses `hook_no_notify()` (producer kernel only,
no hostfunc).  The drain thread uses two mechanisms for streaming drain:

1. **`notify_drain()`** — called from Python in `_prepare_wrapper()` before
   each forward pass.  Wakes the drain thread for the *previous* step's data.
   Non-blocking (~100 ns): sets a flag and signals the drain thread's CV.

2. **`cv_.wait_for` timeout (500 µs)** — safety net for the last step of
   generation (no subsequent `_prepare_wrapper` call) and gaps between
   `generate()` calls.  Task entries are in managed memory preferred on CPU,
   so polling `task_cpu_ready()` is a local DRAM read — no PCIe traffic.

A final flush loop at `stop()` calls `drain_ready()` → `cudaStreamSynchronize`
→ `poll_completed()` until all task entries are processed, handling any
stragglers.

### Effectful op and CUDA graph capture

`ring::producer` is registered as an effectful op (`_register_effectful_op`
with `_EffectType.ORDERED`) to prevent dead-code elimination by inductor.
This wraps the op in a `with_effects` HOP that threads an effect token
(zero-element tensor) as input→output.

With `torch.compile(mode="reduce-overhead")`, the cudagraph tree runtime
logs `"skipping cudagraphs due to mutated inputs"` on the first warmup
invocation because the effect token isn't yet a cudagraph-recorded tensor.
This is **harmless** — the runtime falls back to eager only for warmup,
then uses CUDA graphs for all subsequent calls once the token is recognized
as a cudagraph-managed tensor.

---

## Sequence Guard Invariant

`ready_seq = READY_SEQ_SENTINEL` (`0xFFFF…FFFF`) means the slot has never been published (or was just released).

`ready_seq = seq_no` (any other value) means the producer has published the entry with that sequence number and all other fields are valid.

Since `seq_no` is a monotonically increasing 64-bit counter, collision with SENTINEL would require ~5.8×10^11 years at 10^9 slots/second.  No wraparound guard is necessary.

---

## Inductor DCE of Q/K/V Producer Calls (CUDA Graphs)

**Status**: Root cause found, fix identified (2026-03-10)

### Root Cause

Q/K/V tensors are **non-contiguous views** of the QKV projection output (e.g.
shape `[1, 4, 12, 64]`, strides `(9216, 2304, 64, 1)`).  In `HookPoint.forward()`:

```python
x_cont = x.contiguous()          # creates NEW buffer for non-contiguous x
torch.ops.ring.producer(x_cont)  # void, no return
return x                          # original x returned, x_cont unused elsewhere
```

Inductor sees `x_cont` feeds only a void op with no downstream dependency →
eliminates both the `.contiguous()` copy and the `ring::producer` call.

All other hooks are inherently contiguous → `.contiguous()` is a no-op →
`x_cont IS x` → buffer has downstream model consumers → inductor cannot DCE.

### Mechanism

The DCE happens at **compile time**, before any CUDA graph capture:

1. `torch.compile` traces the decode function and hands the FX graph to inductor
2. Inductor sees Q/K/V `.contiguous()` creates an isolated buffer
3. That buffer's only consumer is the void `ring::producer` → DCE eliminates both
4. The compiled decode function no longer contains Q/K/V producer calls
5. CUDA graph warmup/capture/replay all run compiled code — no Q/K/V

Prefill is a separate compilation (different input shape, Q/K/V are contiguous
at `q_len > 1`) that retains all 185 hooks.

### Fix

Return `x_cont` instead of `x` from `HookPoint.forward()`:

```python
x_cont = x.contiguous()
torch.ops.ring.producer(x_cont, hook_type, hook_id)
return x_cont   # x_cont is live → inductor cannot DCE
```

For contiguous hooks: no change (`x_cont IS x`).  For Q/K/V: adds a copy
into the data path but preserves the producer call in the compiled graph.

### Evidence (GPT-2, batch=4, 8 steps, 12 layers)

Both host (`ring_producer_impl`) and device (`producer_kernel`) counters agree:

```
Eager mode:    all types 96 or 8 → total=1480 (correct)
CUDA graph:    Q=12 K=12 V=12 (prefill only), others unchanged → total=1228
Missing:       252 = 1480 - 1228 = 3 × 12 × 7
```
