# Ring Transport Configuration

All parameters are set on `RingConfig` before constructing `RingEngine`.
Changing them does not require re-capturing the CUDA graph.

Source: `monitoring/csrc/ring/ring_config.h`
Python binding: `monitoring/csrc/bindings.cpp` → `RingConfig`

---

## GPU Ring Buffers

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_ring_entries` | `uint64_t` | 1024 | Number of fixed-size TaskEntry slots in the GPU task/control ring. Power of 2 recommended. |
| `payload_ring_bytes` | `uint64_t` | 256 MiB | Total size of the GPU circular payload byte buffer (HBM). |
| `chunk_bytes` | `uint64_t` | 64 MiB | Max bytes per chunk. One logical tensor is split into chunks of at most this size. Must be a multiple of `PAYLOAD_ALIGN` (16) and ≤ `payload_ring_bytes`. |

## Producer Backpressure

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wait_policy` | `int` | 0 (INFINITE) | `0` = spin until space available (risks deadlock if consumer dies). `1` = TIMEOUT_DROP: abandon task after no consumer progress for `no_progress_timeout_cycles`. |
| `no_progress_timeout_cycles` | `uint64_t` | 1,000,000,000 | GPU clock cycles without `consumer_heartbeat` change before dropping. ~400 ms at 2.5 GHz. Only used when `wait_policy=1`. |
| `drop_reporting` | `int` | 1 (DROP_TASK) | `0` = increment internal counter only. `1` = publish a DROP marker TaskEntry so the CPU pipeline can account for the missing tensor. |

## Drain Thread

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drain_poll_timeout_us` | `uint64_t` | 0 | Drain thread sleep timeout in µs. `0` = infinite wait (only wakes on explicit notify or stop). |
| `drain_notify_on_forward` | `bool` | true | Whether Python calls `notify_drain()` before each forward pass. This only **wakes** the drain thread to scan; it does NOT force a flush. A flush only happens if a drain flush threshold is met (see below). If false AND `drain_poll_timeout_us=0`, the drain thread sleeps until `stop()` — ring must hold all data or producer deadlocks. |

### Drain Flush Thresholds

Controls when the drain thread flushes scanned entries to the pinned staging ring.
Force flush at 100% capacity is always active (prevents deadlock).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drain_flush_task_ratio` | `float` | 0.0 | Flush when scanned entries ≥ this fraction of `task_ring_entries`. 0 = disabled. |
| `drain_flush_payload_ratio` | `float` | 0.0 | Flush when scanned payload bytes ≥ this fraction of `payload_ring_bytes`. 0 = disabled. |
| `drain_flush_entry_threshold` | `uint64_t` | 0 | Flush after N entries ready. 0 = disabled. |
| `drain_flush_byte_threshold` | `uint64_t` | 0 | Flush after N payload bytes ready. 0 = disabled. |

With all thresholds at 0 (default), the drain thread only flushes when the ring is 100% full or at `stop()` time. This minimizes CUDA API calls but increases latency.

## Pinned Staging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pinned_staging_bytes` | `uint64_t` | 0 | Pinned staging ring size (host memory, `cudaHostAlloc`). `0` = defaults to `payload_ring_bytes`. The bypass guard (`tensor_total_padded_bytes > staging_capacity`) prevents deadlock regardless of the ratio. |

## Large Tensor Bypass

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bypass_budget_bytes` | `uint64_t` | 256 MiB | Max total bytes of bypass tensors queued between drain and p2p threads. One tensor is always allowed even if it exceeds this budget (prevents single-large-tensor deadlock). |

Tensors with `tensor_total_padded_bytes > staging_capacity` skip the pinned staging ring entirely. The drain thread D2H-copies directly into an ATen-allocated pageable tensor, then passes it to the p2p thread.

## P2P Thread / Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clone_slices` | `bool` | false | Clone per-request slices before submitting. When true (and batch > 1), each slice is an independent tensor so the full assembled tensor can be freed immediately. When false, slices are views that keep the full tensor alive until consumed. |
| `insert_queue_max_bytes` | `uint64_t` | 512 MiB | ClickHouse insert queue byte limit. P2P thread blocks when queue is full. |
| `insert_queue_max_items` | `uint64_t` | 4096 | ClickHouse insert queue item count limit. |

## Constants (not configurable)

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `PAYLOAD_ALIGN` | 16 bytes | `ring_config.h` | Payload allocation alignment. Every reservation is rounded up to this for vectorized uint4 D2D copies. `chunk_bytes` and `payload_ring_bytes` must be multiples of this. |
| `READY_SEQ_SENTINEL` | `UINT64_MAX` | `task_entry.h` | Sentinel value for `TaskEntry::ready_seq` (slot not yet published). |
| `TaskEntry` size | 128 bytes | `task_entry.h` | Fixed slot size, `alignas(128)` for cache-line isolation. |

## Python Usage

```python
from monitoring._native_engine import RingConfig, RingEngine

cfg = RingConfig()
cfg.task_ring_entries = 512
cfg.payload_ring_bytes = 128 * 1024 * 1024
cfg.chunk_bytes = 32 * 1024 * 1024
cfg.drain_poll_timeout_us = 100
cfg.drain_notify_on_forward = True
cfg.drain_flush_entry_threshold = 64
cfg.bypass_budget_bytes = 128 * 1024 * 1024
cfg.clone_slices = False
cfg.insert_queue_max_items = 2048

engine = RingEngine(cfg, host_engine)
engine.init(stream_handle)
engine.start()
# ... generate ...
engine.stop()
```

---

## E2E Test Environment Variables

All E2E ring parameters are set via `E2E_*` environment variables
(see `tests/test_e2e_correctness_vs_hf.py::_make_ring_cfg`).

| Env Var | Default | Maps to |
|---------|---------|---------|
| `E2E_RING_TASK_ENTRIES` | 16384 | `task_ring_entries` |
| `E2E_RING_PAYLOAD_BYTES` | 4 GiB | `payload_ring_bytes` |
| `E2E_RING_CHUNK_BYTES` | 4 MiB | `chunk_bytes` |
| `E2E_RING_PINNED_BYTES` | 4 GiB | `pinned_staging_bytes` |
| `E2E_DRAIN_POLL_TIMEOUT_US` | 0 | `drain_poll_timeout_us` |
| `E2E_DRAIN_NOTIFY_ON_FORWARD` | 1 | `drain_notify_on_forward` |
| `E2E_DRAIN_FLUSH_TASK_RATIO` | 0.0 | `drain_flush_task_ratio` |
| `E2E_DRAIN_FLUSH_PAYLOAD_RATIO` | 0.0 | `drain_flush_payload_ratio` |
| `E2E_DRAIN_FLUSH_ENTRY_THRESHOLD` | 0 | `drain_flush_entry_threshold` |
| `E2E_DRAIN_FLUSH_BYTE_THRESHOLD` | 0 | `drain_flush_byte_threshold` |
| `E2E_BYPASS_BUDGET_BYTES` | 256 MiB | `bypass_budget_bytes` |
| `E2E_CLONE_SLICES` | 0 | `clone_slices` |
| `E2E_INSERT_QUEUE_MAX_BYTES` | 512 MiB | `insert_queue_max_bytes` |
| `E2E_INSERT_QUEUE_MAX_ITEMS` | 4096 | `insert_queue_max_items` |
| `E2E_CH_PARALLELISM` | 10 | ClickHouse insert parallelism |
| `E2E_CH_QUEUE_MAX_ITEMS` | 1024 | ClickHouse queue item limit |
| `E2E_CH_QUEUE_MAX_BYTES` | 2 GiB | ClickHouse queue byte limit |

## Benchmark CLI Arguments

All benchmark ring parameters are set via CLI flags
(see `benchmark/bench_ring_transport.py`).

```
Ring engine — GPU buffers:
  --ring-task-entries N       Task ring slot count (default: 65536)
  --ring-payload-mb N         GPU payload ring size in MiB (default: 4096)
  --ring-chunk-kb N           Max chunk size in KiB (default: 4096)
  --ring-pinned-mb N          Pinned staging ring size in MiB (default: 4096)

Ring engine — drain thread:
  --drain-poll-timeout-us N   Drain thread poll timeout in µs (default: 0)
  --no-drain-notify           Disable notify_drain() before each forward
  --drain-flush-task-ratio F  Flush at F fraction of task ring (default: 0.0)
  --drain-flush-payload-ratio F  Flush at F fraction of payload ring (default: 0.0)
  --drain-flush-entry-threshold N  Flush after N entries (default: 0)
  --drain-flush-byte-threshold N   Flush after N bytes (default: 0)

Ring engine — bypass / p2p:
  --bypass-budget-mb N        Large tensor bypass budget in MiB (default: 256)
  --clone-slices              Clone per-request slices before submit

ClickHouse stage:
  --ch-parallelism N          Insert thread parallelism (default: 10)
  --ch-queue-max-items N      Insert queue item limit (default: 1024)
  --ch-queue-max-size-mb N    Insert queue byte limit in MiB (default: 2048)
```
