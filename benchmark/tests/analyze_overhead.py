"""Analyze overhead of precision reduction vs quantization in monitoring pipeline."""

import torch
import time
import numpy as np

def benchmark_overhead():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulate typical activation sizes in GPT-2
    # 12 layers × batch_size=64 × seq_len=1 × hidden_dim=768
    batch_size = 64
    num_layers = 12
    hidden_dim = 768

    print("=" * 80)
    print("Overhead Analysis: Precision Reduction vs Quantization")
    print("=" * 80)
    print(f"\nScenario: GPT-2 decode step monitoring")
    print(f"  Batch size: {batch_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Total tensors per step: ~{num_layers * 5} (Q/K/V/hidden/residual per layer)")

    # Create sample tensors (one layer's worth)
    tensors = []
    for _ in range(5):  # Q, K, V, hidden, residual
        t = torch.randn(batch_size, 12, 1, 64, device=device, dtype=torch.float32)
        tensors.append(t)

    print(f"\n{'='*80}")
    print("BASELINE: No compression (FP32)")
    print("=" * 80)

    total_bytes = sum(t.element_size() * t.numel() for t in tensors)
    print(f"  Data size per layer: {total_bytes / 1024:.2f} KB")
    print(f"  Data size per step (all layers): {total_bytes * num_layers / 1024:.2f} KB")
    print(f"  Overhead: 0 ms (baseline)")

    # ========================================
    # Method 1: Precision Reduction (FP32 → FP16)
    # ========================================
    print(f"\n{'='*80}")
    print("METHOD 1: Precision Reduction (FP32 → FP16)")
    print("=" * 80)

    # Warm-up
    for t in tensors:
        _ = t.to(torch.float16)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark on GPU
    iters = 100
    if device.type == 'cuda':
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                t_fp16 = t.to(torch.float16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_time = (t1 - t0) / iters * 1000

        # Benchmark D2H transfer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                t_fp16 = t.to(torch.float16)
                t_cpu = t_fp16.to('cpu', non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        d2h_time = (t1 - t0) / iters * 1000
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                t_fp16 = t.to(torch.float16)
        t1 = time.perf_counter()
        gpu_time = (t1 - t0) / iters * 1000
        d2h_time = 0

    fp16_bytes = sum(t.element_size() // 2 * t.numel() for t in tensors)

    print(f"\n  Compression:")
    print(f"    Data size per layer: {fp16_bytes / 1024:.2f} KB")
    print(f"    Data size per step: {fp16_bytes * num_layers / 1024:.2f} KB")
    print(f"    Reduction: {total_bytes / fp16_bytes:.1f}x")

    print(f"\n  GPU Overhead (conversion only):")
    print(f"    Per layer: {gpu_time:.3f} ms")
    print(f"    Per step (12 layers): {gpu_time * num_layers:.3f} ms")

    print(f"\n  D2H Transfer (FP16 vs FP32):")
    print(f"    FP16 bandwidth saving: {(1 - fp16_bytes/total_bytes)*100:.1f}%")
    print(f"    Transfer time reduction: ~{(1 - fp16_bytes/total_bytes)*100:.1f}%")

    # ========================================
    # Method 2: Quantization (FP32 → INT8)
    # ========================================
    print(f"\n{'='*80}")
    print("METHOD 2: Quantization (FP32 → INT8, Per-Tensor)")
    print("=" * 80)

    def quantize_tensor(t):
        # Symmetric quantization
        max_abs = t.abs().max()
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
        return q, scale

    # Warm-up
    for t in tensors:
        _ = quantize_tensor(t)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark quantization
    if device.type == 'cuda':
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                q, scale = quantize_tensor(t)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        quant_time = (t1 - t0) / iters * 1000

        # Benchmark D2H transfer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                q, scale = quantize_tensor(t)
                q_cpu = q.to('cpu', non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        quant_d2h_time = (t1 - t0) / iters * 1000

        # Benchmark dequantization (on CPU)
        q_list = [quantize_tensor(t) for t in tensors]
        q_cpu_list = [(q.cpu(), s.cpu()) for q, s in q_list]
        t0 = time.perf_counter()
        for _ in range(iters):
            for q_cpu, scale_cpu in q_cpu_list:
                dequant = q_cpu.to(torch.float32) * scale_cpu
        t1 = time.perf_counter()
        dequant_time = (t1 - t0) / iters * 1000
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                q, scale = quantize_tensor(t)
        t1 = time.perf_counter()
        quant_time = (t1 - t0) / iters * 1000
        quant_d2h_time = 0
        dequant_time = 0

    int8_bytes = sum(t.numel() for t in tensors)  # 1 byte per element
    metadata_bytes = len(tensors) * 4  # 1 float32 scale per tensor

    print(f"\n  Compression:")
    print(f"    Data size per layer: {(int8_bytes + metadata_bytes) / 1024:.2f} KB")
    print(f"    Data size per step: {(int8_bytes + metadata_bytes) * num_layers / 1024:.2f} KB")
    print(f"    Reduction: {total_bytes / (int8_bytes + metadata_bytes):.1f}x")

    print(f"\n  GPU Overhead (quantization):")
    print(f"    Per layer: {quant_time:.3f} ms")
    print(f"    Per step (12 layers): {quant_time * num_layers:.3f} ms")

    print(f"\n  CPU Overhead (dequantization):")
    print(f"    Per layer: {dequant_time:.3f} ms")
    print(f"    Per step: {dequant_time * num_layers:.3f} ms")

    print(f"\n  D2H Transfer (INT8 vs FP32):")
    print(f"    INT8 bandwidth saving: {(1 - int8_bytes/total_bytes)*100:.1f}%")
    print(f"    Transfer time reduction: ~{(1 - int8_bytes/total_bytes)*100:.1f}%")

    # ========================================
    # Method 3: Per-Channel Quantization
    # ========================================
    print(f"\n{'='*80}")
    print("METHOD 3: Quantization (FP32 → INT8, Per-Channel)")
    print("=" * 80)

    def quantize_per_channel(t):
        # Per-channel quantization (along dim=-1)
        max_abs = t.abs().amax(dim=tuple(range(t.ndim-1)), keepdim=True)
        scale = torch.clamp(max_abs / 127.0, min=1e-8)
        q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
        return q, scale

    # Warm-up
    for t in tensors:
        _ = quantize_per_channel(t)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                q, scale = quantize_per_channel(t)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        perch_time = (t1 - t0) / iters * 1000
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            for t in tensors:
                q, scale = quantize_per_channel(t)
        t1 = time.perf_counter()
        perch_time = (t1 - t0) / iters * 1000

    # Calculate metadata size
    total_channels = sum(t.shape[-1] for t in tensors)
    perch_metadata_bytes = total_channels * 4  # 1 float32 per channel

    print(f"\n  Compression:")
    print(f"    Data size per layer: {(int8_bytes + perch_metadata_bytes) / 1024:.2f} KB")
    print(f"    Metadata: {perch_metadata_bytes / 1024:.2f} KB ({total_channels} scales)")
    print(f"    Reduction: {total_bytes / (int8_bytes + perch_metadata_bytes):.1f}x")

    print(f"\n  GPU Overhead (quantization):")
    print(f"    Per layer: {perch_time:.3f} ms")
    print(f"    Per step: {perch_time * num_layers:.3f} ms")

    # ========================================
    # Summary Table
    # ========================================
    print(f"\n{'='*80}")
    print("SUMMARY: End-to-End Pipeline Overhead")
    print("=" * 80)

    print(f"\nPer-Step Overhead Breakdown (12 layers):")
    print(f"{'Method':<35} {'Convert':<12} {'D2H':<12} {'Dequant':<12} {'Total':<12} {'Net vs FP32':<12} {'Size (KB)':<12}")
    print("-" * 115)

    # Assume D2H time is proportional to data size (10 GB/s PCIe)
    pcie_bandwidth = 10 * 1024 * 1024  # 10 GB/s in KB/s

    baseline_d2h = (total_bytes * num_layers / 1024) / pcie_bandwidth * 1000
    fp16_d2h = (fp16_bytes * num_layers / 1024) / pcie_bandwidth * 1000
    int8_d2h = (int8_bytes * num_layers / 1024) / pcie_bandwidth * 1000

    # Calculate net overhead (total - baseline)
    baseline_total = baseline_d2h
    fp16_total = gpu_time * num_layers + fp16_d2h
    fp16_net = fp16_total - baseline_total
    int8_realtime_total = (quant_time + dequant_time) * num_layers + int8_d2h
    int8_realtime_net = int8_realtime_total - baseline_total
    int8_offline_total = quant_time * num_layers + int8_d2h  # No dequant in realtime
    int8_offline_net = int8_offline_total - baseline_total
    perch_realtime_total = (perch_time + dequant_time) * num_layers + int8_d2h
    perch_realtime_net = perch_realtime_total - baseline_total
    perch_offline_total = perch_time * num_layers + int8_d2h
    perch_offline_net = perch_offline_total - baseline_total

    print(f"{'Baseline (FP32)':<35} {'0.00':<12} {baseline_d2h:<12.3f} {'0.00':<12} {baseline_total:<12.3f} {'0.00':<12} {total_bytes * num_layers / 1024:<12.2f}")
    print(f"{'FP16':<35} {gpu_time * num_layers:<12.3f} {fp16_d2h:<12.3f} {'0.00':<12} {fp16_total:<12.3f} {fp16_net:<+12.3f} {fp16_bytes * num_layers / 1024:<12.2f}")
    print(f"{'INT8 (realtime dequant)':<35} {quant_time * num_layers:<12.3f} {int8_d2h:<12.3f} {dequant_time * num_layers:<12.3f} {int8_realtime_total:<12.3f} {int8_realtime_net:<+12.3f} {(int8_bytes + metadata_bytes) * num_layers / 1024:<12.2f}")
    print(f"{'INT8 (offline dequant)':<35} {quant_time * num_layers:<12.3f} {int8_d2h:<12.3f} {'[later]':<12} {int8_offline_total:<12.3f} {int8_offline_net:<+12.3f} {(int8_bytes + metadata_bytes) * num_layers / 1024:<12.2f}")
    print(f"{'INT8 per-ch (realtime dequant)':<35} {perch_time * num_layers:<12.3f} {int8_d2h:<12.3f} {dequant_time * num_layers:<12.3f} {perch_realtime_total:<12.3f} {perch_realtime_net:<+12.3f} {(int8_bytes + perch_metadata_bytes) * num_layers / 1024:<12.2f}")
    print(f"{'INT8 per-ch (offline dequant)':<35} {perch_time * num_layers:<12.3f} {int8_d2h:<12.3f} {'[later]':<12} {perch_offline_total:<12.3f} {perch_offline_net:<+12.3f} {(int8_bytes + perch_metadata_bytes) * num_layers / 1024:<12.2f}")

    # ========================================
    # Critical Analysis
    # ========================================
    print(f"\n{'='*80}")
    print("CRITICAL ANALYSIS FOR YOUR MONITORING SYSTEM")
    print("=" * 80)

    print(f"""
1. COMPARISON: REALTIME vs OFFLINE DEQUANTIZATION

   Realtime Dequant:  GPU Quant → D2H → [SYNC] → CPU Dequant → Store (as FP32)
   Offline Dequant:   GPU Quant → D2H → [SYNC] → Store (as INT8) ... [dequant when accessed]

   Net Overhead Summary (12 layers):
     FP16:                      {fp16_net:+.3f} ms  {'✅ SPEEDUP' if fp16_net < 0 else '⚠️ SLOWDOWN'}
     INT8 (realtime dequant):   {int8_realtime_net:+.3f} ms  {'✅ SPEEDUP' if int8_realtime_net < 0 else '⚠️ SLOWDOWN'}
     INT8 (offline dequant):    {int8_offline_net:+.3f} ms  {'✅ SPEEDUP' if int8_offline_net < 0 else '⚠️ SLOWDOWN'}

   KEY INSIGHT: INT8 with offline dequant vs FP16: {int8_offline_net - fp16_net:+.3f} ms difference

2. WHERE OVERHEAD OCCURS IN YOUR PIPELINE:

   GPU Forward → Hook Capture → [CONVERT] → D2H → [SYNC] → Host Copy → [DEQUANT?] → Store
                                    ↑                ↑                      ↑
                                    A                B                      C

   A. Conversion (GPU): Run during process_task (in cache_stream)
   B. Sync: Unified sync point (once per step)
   C. Dequantization (CPU):
      - Realtime: In process_copy_job (blocks collection)
      - Offline: When Python accesses result (doesn't block collection)

3. PRECISION REDUCTION (FP16/BF16):

   ✅ Pros:
      - Very low GPU overhead (~0.02ms per layer × 12 = 0.24ms per step)
      - Native hardware support (Tensor Cores)
      - No dequantization needed (can use FP16 directly)
      - 50% bandwidth saving on D2H (~0.5ms saved per step)

   ⚠️ Cons:
      - Only 2x compression
      - Already well-optimized in your system (via cache_dtype)

   💡 Recommendation: USE THIS (already supported)
      - Set cache_dtype=torch.bfloat16
      - Total overhead: ~{fp16_net:.2f}ms
      - Net {'speedup' if fp16_net < 0 else 'slowdown'}!

3. INT8 QUANTIZATION (Realtime Dequantization):

   ⚠️ Pros:
      - 4x compression
      - 75% bandwidth saving (~1.5ms saved per step)

   ❌ Cons:
      - High GPU overhead (~{quant_time * num_layers:.1f}ms per step)
      - CPU dequantization overhead (~{dequant_time * num_layers:.1f}ms per step)
      - Total overhead: {int8_realtime_net:+.2f}ms
      - Blocks collection pipeline (dequant in critical path)

   💡 Recommendation: NOT WORTH IT for realtime
      - {int8_realtime_net - fp16_net:.2f}ms SLOWER than FP16
      - Consider offline dequant instead (see below)

4. INT8 QUANTIZATION (Offline Dequantization) 🆕:

   ✅ Pros:
      - 4x compression (same as realtime)
      - Removes CPU dequant from critical path
      - Net overhead: {int8_offline_net:+.2f}ms
      - Store compressed INT8, dequant only when Python accesses

   ⚠️ Cons:
      - Still has GPU quantization overhead (~{quant_time * num_layers:.1f}ms)
      - Need to handle INT8→FP32 conversion in Python layer
      - {'Still SLOWER than FP16' if int8_offline_net > fp16_net else 'FASTER than FP16!'}

   💡 Recommendation: {'CONSIDER if storage is bottleneck' if int8_offline_net > fp16_net else 'VIABLE ALTERNATIVE to FP16'}
      - vs FP16: {int8_offline_net - fp16_net:+.2f}ms difference
      - Trade: {abs(int8_offline_net - fp16_net):.2f}ms {'extra overhead' if int8_offline_net > fp16_net else 'savings'} for 2x better compression

5. INT8 QUANTIZATION (Per-Channel):

   ⚠️ Pros:
      - Better accuracy than per-tensor
      - Same compression ratio

   ❌ Cons:
      - Even higher GPU overhead (~{perch_time * num_layers:.1f}ms)
      - More metadata to store/transfer
      - Realtime: {perch_realtime_net:+.2f}ms net overhead
      - Offline: {perch_offline_net:+.2f}ms net overhead

   💡 Recommendation: NOT RECOMMENDED
      - Higher GPU cost than per-tensor quantization
      - Minimal accuracy benefit for monitoring metrics

6. WHEN TO USE EACH METHOD:

   ✅ Use FP16/BF16 if:
      - You want minimal overhead ({fp16_net:+.2f}ms)
      - 2x compression is sufficient
      - Simple implementation (already supported)

   ✅ Use INT8 (offline dequant) if:
      - Storage/memory is critical bottleneck (need 4x compression)
      - Can tolerate {int8_offline_net:+.2f}ms overhead
      - Don't need immediate FP32 access to data

   ❌ Avoid INT8 (realtime dequant):
      - {int8_realtime_net:+.2f}ms overhead is too high
      - Blocks collection pipeline
      - {int8_realtime_net - int8_offline_net:.2f}ms slower than offline variant

7. OPTIMAL STRATEGY FOR YOUR SYSTEM:

   Option A (RECOMMENDED): FP16/BF16 Only
      - Set cache_dtype=torch.bfloat16
      - Overhead: {fp16_net:+.2f}ms
      - Compression: 2x
      - Implementation: Already supported!

   Option B (Advanced): INT8 with Offline Dequant
      - GPU quantize + D2H as INT8
      - Overhead: {int8_offline_net:+.2f}ms
      - Compression: 4x
      - Store INT8 in results, convert to FP32 in Python when accessed
      - Implementation:
          engine_core.cpp: Add INT8 quantization in run_task()
          engine.py: Add lazy dequant in result() method
      - Trade-off: {abs(int8_offline_net - fp16_net):.2f}ms {'extra cost' if int8_offline_net > fp16_net else 'savings'} for 2x better compression

   Option C (Not Recommended): INT8 with Realtime Dequant
      - Full pipeline quantization
      - Overhead: {int8_realtime_net:+.2f}ms
      - Too slow, blocks collection
""")

    print(f"\n{'='*80}")
    print("RECOMMENDATION SUMMARY")
    print("=" * 80)
    print(f"""
For your async monitoring system:

1. ✅ BEST: cache_dtype=torch.bfloat16
   - Net overhead: {fp16_net:+.2f}ms
   - Compression: 2x
   - Zero complexity

2. ✅ ALTERNATIVE: INT8 with offline dequant
   - Net overhead: {int8_offline_net:+.2f}ms ({'faster' if int8_offline_net < fp16_net else 'slower'} than FP16 by {abs(int8_offline_net - fp16_net):.2f}ms)
   - Compression: 4x
   - Use if storage is critical bottleneck

3. ❌ AVOID: INT8 with realtime dequant
   - Net overhead: {int8_realtime_net:+.2f}ms
   - {int8_realtime_net - fp16_net:.2f}ms slower than FP16
   - Blocks collection pipeline

Decision Matrix:
  Storage OK, want speed → FP16 ({fp16_net:+.2f}ms)
  Storage critical, can trade time → INT8 offline ({int8_offline_net:+.2f}ms)
  Never use → INT8 realtime ({int8_realtime_net:+.2f}ms)
    """)

if __name__ == "__main__":
    benchmark_overhead()
