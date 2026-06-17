# Ring Transport Configuration

All parameters are set on `RingConfig` before constructing `RingEngine`.
Changing them does not require re-capturing the CUDA graph.

Source: `monitoring/csrc/ring/ring_config.h`
Python binding: `monitoring/csrc/bindings.cpp` -> `RingConfig`

---

## GPU Ring Buffers

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_ring_entries` | `uint64_t` | 1024 | Number of fixed-size TaskEntry slots in the GPU task/control ring. Power of 2 recommended. |
| `payload_ring_bytes` | `uint64_t` | 256 MiB | Total size of the GPU circular payload byte buffer (HBM). Must be a multiple of `PAYLOAD_ALIGN` (16). |

## Pinned Staging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pinned_staging_bytes` | `uint64_t` | 0 | Pinned staging ring size (host memory, `cudaHostAlloc`). `0` = defaults to `payload_ring_bytes`. |

## Drain Thread

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drain_poll_timeout_us` | `uint64_t` | 100 | Drain thread sleep timeout in microseconds. Must be > 0. |

### Drain Flush Thresholds (`drain_flush`)

Controls when the drain thread flushes scanned entries to the pinned staging
ring. Force flush at 100% capacity is always active (prevents deadlock).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drain_flush_task_ratio` | `float` | 0.0 | Flush when scanned entries >= this fraction of `task_ring_entries`. 0 = disabled. |
| `drain_flush_payload_ratio` | `float` | 0.0 | Flush when scanned payload bytes >= this fraction of `payload_ring_bytes`. 0 = disabled. |
| `drain_flush_entry_threshold` | `uint64_t` | 0 | Flush after N entries ready. 0 = disabled. |
| `drain_flush_byte_threshold` | `uint64_t` | 0 | Flush after N payload bytes ready. 0 = disabled. |
| `drain_flush_timeout_us` | `uint64_t` | 100000 | If a complete tensor has been pending for longer than this many microseconds, flush unconditionally. 0 = disabled. |

With timeout-based flushing disabled explicitly (`drain_flush_timeout_us = 0`)
and all other flush thresholds at 0, the drain thread only flushes when the ring
is 100% full or at `stop()` time. This minimizes CUDA API calls but increases
latency.

## P2P Thread / Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clone_slices` | `bool` | false | Clone per-request slices before submitting to the host engine. When true (and batch > 1), each slice is an independent tensor so the full assembled tensor can be freed immediately. When false, slices are views that keep the full tensor alive until consumed. |
| `insert_queue_max_bytes` | `uint64_t` | 4 GiB | ClickHouse insert queue byte limit. The p2p thread blocks when the queue is full. |
| `insert_queue_max_items` | `uint64_t` | 65536 | ClickHouse insert queue item-count limit. |

## Constants (not configurable)

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `PAYLOAD_ALIGN` | 16 bytes | `ring_config.h` | Payload allocation alignment. Every reservation is rounded up to this for vectorized uint4 D2D copies. `payload_ring_bytes` must be a multiple of this. |
| `READY_SEQ_SENTINEL` | `UINT64_MAX` | `task_entry.h` | Sentinel value for `TaskEntry::ready_seq` (slot not yet published). |
| `TaskEntry` size | 128 bytes | `task_entry.h` | Fixed slot size, `alignas(128)` for cache-line isolation. |

## Python Usage

```python
from monitoring._native_engine import RingConfig, RingEngine

cfg = RingConfig()
cfg.task_ring_entries = 512
cfg.payload_ring_bytes = 128 * 1024 * 1024
cfg.pinned_staging_bytes = 128 * 1024 * 1024
cfg.drain_poll_timeout_us = 100
cfg.drain_flush_entry_threshold = 64
cfg.drain_flush_timeout_us = 1000
cfg.clone_slices = False
cfg.insert_queue_max_items = 4096

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
| `E2E_RING_PINNED_BYTES` | 4 GiB | `pinned_staging_bytes` |
| `E2E_DRAIN_POLL_TIMEOUT_US` | 100 | `drain_poll_timeout_us` |
| `E2E_DRAIN_FLUSH_TASK_RATIO` | 0.0 | `drain_flush_task_ratio` |
| `E2E_DRAIN_FLUSH_PAYLOAD_RATIO` | 0.0 | `drain_flush_payload_ratio` |
| `E2E_DRAIN_FLUSH_ENTRY_THRESHOLD` | 0 | `drain_flush_entry_threshold` |
| `E2E_DRAIN_FLUSH_BYTE_THRESHOLD` | 0 | `drain_flush_byte_threshold` |
| `E2E_DRAIN_FLUSH_TIMEOUT_US` | 100000 | `drain_flush_timeout_us` |
| `E2E_CLONE_SLICES` | 0 | `clone_slices` |
| `E2E_INSERT_QUEUE_MAX_BYTES` | 512 MiB | `insert_queue_max_bytes` |
| `E2E_INSERT_QUEUE_MAX_ITEMS` | 4096 | `insert_queue_max_items` |
| `E2E_CH_PARALLELISM` | 10 | ClickHouse insert parallelism |
| `E2E_CH_QUEUE_MAX_ITEMS` | 1024 | ClickHouse queue item limit |
| `E2E_CH_QUEUE_MAX_BYTES` | 2 GiB | ClickHouse queue byte limit |

## Benchmark CLI Arguments

All benchmark ring parameters are set via CLI flags
(see `benchmark/bench_hf_transport.py`).

```
Ring engine -- GPU buffers:
  --ring-task-entries N       Task ring slot count (default: 65536)
  --ring-payload-mb N         GPU payload ring size in MiB (default: 4096)
  --ring-pinned-mb N          Pinned staging ring size in MiB (default: 4096)

Ring engine -- drain thread:
  --drain-poll-timeout-us N        Drain thread poll timeout in microseconds (default: 100)
  --drain-flush-task-ratio F       Flush at F fraction of task ring (default: 0.0)
  --drain-flush-payload-ratio F    Flush at F fraction of payload ring (default: 0.0)
  --drain-flush-entry-threshold N  Flush after N entries (default: 0)
  --drain-flush-byte-threshold N   Flush after N bytes (default: 0)
  --drain-flush-timeout-us N       Per-tensor flush timeout in microseconds (default: 100000)

Ring engine -- p2p / output:
  --clone-slices              Clone per-request slices before submit

ClickHouse stage:
  --ch-parallelism N          Insert thread parallelism (default: 10)
  --ch-queue-max-items N      Insert queue item limit (default: 1024)
  --ch-queue-max-size-mb N    Insert queue byte limit in MiB (default: 2048)
```
