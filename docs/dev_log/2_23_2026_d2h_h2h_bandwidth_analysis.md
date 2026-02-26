# D2H / H2H Bandwidth Analysis (2026-02-23)

## Setup

- Model: GPT-2 small (12 layers), fp32
- Batch=64, decode_steps=64, StaticCache
- 343 total slots (all named_modules), ~182 valid per step
- 10 host copy threads, pinned pool enabled
- Profiled with nsys

## Observed Timing

| Phase | Duration | Data Volume | Effective BW |
|---|---|---|---|
| GPU → pinned (D2H) | 4.6ms | ~36MB | 7.8 GB/s |
| pinned → pageable (H2H) | 6.6ms | ~36MB | 5.5 GB/s |
| **Total** | **11.2ms** | | |

## Data Volume Estimate

Per hook tensor (decode step):
- Hidden state hooks: `[64, 1, 768]` fp32 = 192KB
- QKV/Z/result hooks: `[64, 12, 1, 64]` fp32 = 192KB
- Attention score/pattern: `[64, 12, 1, 81]` fp32 = 249KB (max_cache_len=81)

Per layer: 8 hidden × 192KB + 5 QKV × 192KB + 2 attn × 249KB ≈ 3MB
12 layers + 3 global hooks ≈ **~36MB/step**

## Why Both Are Far Below Theoretical Bandwidth

### D2H: 7.8 GB/s vs PCIe 3.0 12 GB/s (65%)

182 individual `cudaMemcpyAsync` calls, each ~200KB:
- Per-call DMA setup overhead: ~5-10us
- 182 × 10us overhead = **1.8ms** of pure overhead
- Actual transfer: 36MB / 12 GB/s = **3ms**
- Total: ~4.8ms (matches observed 4.6ms)

### H2H: 5.5 GB/s vs DDR4 30+ GB/s (18%)

182 individual copies, each involving:
- `malloc()` for pageable destination (~5-10us each)
- Tensor construction overhead
- Small memcpy (~200KB, can't saturate memory bus)
- Thread pool dispatch + condition variable sync
- 182 × (malloc + ctor + memcpy + sync) dominates over raw bandwidth

## Optimization: H2H Gather

Current (per-tensor × 182):
```
D2H: gpu_tensor[i] → pinned_block[i]         ×182 (4.6ms)
H2H: pinned_block[i] → malloc() + memcpy     ×182 (6.6ms)
```

Proposed (batch gather after resolve):
```
D2H: gpu_tensor[i] → pinned_block[i]         ×182 (4.6ms, unchanged)
     resolve_all() waits for completion
H2H: 1× malloc(~36MB contiguous buffer)
     182× memcpy(pinned_block[i] → buf+offset[i])   sequential write
     182× torch::from_blob(buf+offset, sizes)        zero-copy view
     release all pinned blocks
```

Expected improvement:
- 1 malloc vs 182 → saves ~1-2ms allocator overhead
- Sequential write to contiguous memory → CPU prefetch effective
- Zero-copy tensor views (from_blob) → no per-tensor construction
- Estimated H2H: 36MB / 30 GB/s ≈ **1.2ms** (vs 6.6ms)

## Slot Count Analysis

343 slots come from ALL `named_modules()`, not just HookPoint:
- 183 HookPoints (15 per layer × 12 + 3 global)
- 160 non-HookPoint (Conv1D, LayerNorm, Dropout, MLP containers, etc.)

`_register_hooks` has no `isinstance(module, HookPoint)` filter.
Adding one would reduce 343 → 183 slots, eliminating:
- 160 useless forward hooks + record() calls during graph capture
- ~80 stale-pointer exceptions per step (fewer wasted thread pool cycles)
- Does NOT reduce data volume much (non-HookPoint outputs are similar size)

## Next Steps

1. **H2H gather** in `engine_core.cpp`: post-resolve batch copy to contiguous buffer
2. **HookPoint filter** in `graph_monitor.py._register_hooks`: skip non-HookPoint modules
3. **GPU gather** (future, Design B): pack tensors on GPU before D2H for single large transfer
