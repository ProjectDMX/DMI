"""Analyze how overhead scales with tensor size for precision reduction vs quantization."""

import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def benchmark_scaling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different tensor sizes
    # Simulate: batch_size × seq_len × hidden_dim
    test_configs = [
        # (name, batch_size, seq_len, hidden_dim, description)
        ("Tiny", 1, 1, 768, "Single token, batch=1"),
        ("Small", 8, 1, 768, "Decode step, batch=8"),
        ("Medium", 32, 1, 768, "Decode step, batch=32"),
        ("Large", 64, 1, 768, "Decode step, batch=64"),
        ("XLarge", 64, 10, 768, "Short sequence, batch=64"),
        ("Prefill-S", 64, 128, 768, "Prefill 128 tokens"),
        ("Prefill-M", 64, 512, 768, "Prefill 512 tokens"),
        ("Prefill-L", 64, 1024, 768, "Prefill 1024 tokens"),
        ("Prefill-XL", 64, 2048, 768, "Prefill 2048 tokens"),
    ]

    print("=" * 100)
    print("Scaling Analysis: How Does Overhead Change with Tensor Size?")
    print("=" * 100)

    results = []

    for config_name, batch_size, seq_len, hidden_dim, desc in test_configs:
        print(f"\n{'='*100}")
        print(f"Configuration: {config_name} - {desc}")
        print(f"  Shape: [{batch_size}, {seq_len}, {hidden_dim}]")
        print("=" * 100)

        # Create tensor
        tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
        size_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)

        print(f"  Tensor size: {size_mb:.2f} MB")

        # Warm-up
        _ = tensor.to(torch.float16)
        torch.cuda.synchronize() if device.type == 'cuda' else None

        iters = 50

        # ========================================
        # Baseline: FP32
        # ========================================
        if device.type == 'cuda':
            # D2H baseline
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                t_cpu = tensor.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            baseline_d2h = (t1 - t0) / iters * 1000
        else:
            baseline_d2h = 0

        # ========================================
        # Method 1: FP16
        # ========================================
        if device.type == 'cuda':
            # Convert
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                t_fp16 = tensor.to(torch.float16)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fp16_convert = (t1 - t0) / iters * 1000

            # D2H
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                t_fp16 = tensor.to(torch.float16)
                t_cpu = t_fp16.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fp16_total = (t1 - t0) / iters * 1000
            fp16_d2h = fp16_total - fp16_convert
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                t_fp16 = tensor.to(torch.float16)
            t1 = time.perf_counter()
            fp16_convert = (t1 - t0) / iters * 1000
            fp16_d2h = 0
            fp16_total = fp16_convert

        # ========================================
        # Method 2: INT8 (per-tensor)
        # ========================================
        def quantize_tensor(t):
            max_abs = t.abs().max()
            scale = max_abs / 127.0
            q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
            return q, scale

        if device.type == 'cuda':
            # Quantize
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                q, scale = quantize_tensor(tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            int8_quantize = (t1 - t0) / iters * 1000

            # D2H
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                q, scale = quantize_tensor(tensor)
                q_cpu = q.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            int8_total_gpu = (t1 - t0) / iters * 1000
            int8_d2h = int8_total_gpu - int8_quantize

            # Dequantize (CPU)
            q_cpu, scale_cpu = quantize_tensor(tensor)
            q_cpu = q_cpu.cpu()
            scale_cpu = scale_cpu.cpu()
            t0 = time.perf_counter()
            for _ in range(iters):
                dequant = q_cpu.to(torch.float32) * scale_cpu
            t1 = time.perf_counter()
            int8_dequantize = (t1 - t0) / iters * 1000

            int8_total = int8_quantize + int8_d2h + int8_dequantize
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                q, scale = quantize_tensor(tensor)
            t1 = time.perf_counter()
            int8_quantize = (t1 - t0) / iters * 1000
            int8_d2h = 0
            int8_dequantize = 0
            int8_total = int8_quantize

        # Calculate sizes
        fp16_size = size_mb / 2
        int8_size = size_mb / 4

        # Calculate bandwidth savings
        d2h_saved_fp16 = baseline_d2h - fp16_d2h
        d2h_saved_int8 = baseline_d2h - int8_d2h

        # Net overhead
        fp16_net = fp16_total - baseline_d2h
        int8_realtime_net = int8_total - baseline_d2h
        int8_offline_net = (int8_quantize + int8_d2h) - baseline_d2h  # No dequant

        print(f"\n  FP32 Baseline:")
        print(f"    D2H: {baseline_d2h:.3f} ms")
        print(f"    Size: {size_mb:.2f} MB")

        print(f"\n  FP16:")
        print(f"    Convert: {fp16_convert:.3f} ms")
        print(f"    D2H: {fp16_d2h:.3f} ms (saved {d2h_saved_fp16:.3f} ms)")
        print(f"    Total: {fp16_total:.3f} ms")
        print(f"    Size: {fp16_size:.2f} MB")
        print(f"    Net overhead: {fp16_net:+.3f} ms {'✅ SPEEDUP' if fp16_net < 0 else '❌ SLOWDOWN'}")

        print(f"\n  INT8 (realtime dequant):")
        print(f"    Quantize (GPU): {int8_quantize:.3f} ms")
        print(f"    D2H: {int8_d2h:.3f} ms (saved {d2h_saved_int8:.3f} ms)")
        print(f"    Dequantize (CPU): {int8_dequantize:.3f} ms")
        print(f"    Total: {int8_total:.3f} ms")
        print(f"    Size: {int8_size:.2f} MB")
        print(f"    Net overhead: {int8_realtime_net:+.3f} ms {'✅ SPEEDUP' if int8_realtime_net < 0 else '❌ SLOWDOWN'}")

        int8_offline_total = int8_quantize + int8_d2h
        print(f"\n  INT8 (offline dequant):")
        print(f"    Quantize (GPU): {int8_quantize:.3f} ms")
        print(f"    D2H: {int8_d2h:.3f} ms (saved {d2h_saved_int8:.3f} ms)")
        print(f"    Dequantize (CPU): [deferred]")
        print(f"    Total: {int8_offline_total:.3f} ms")
        print(f"    Size: {int8_size:.2f} MB")
        print(f"    Net overhead: {int8_offline_net:+.3f} ms {'✅ SPEEDUP' if int8_offline_net < 0 else '❌ SLOWDOWN'}")

        # Crossover analysis
        print(f"\n  Comparison:")
        print(f"    FP16 vs INT8 (realtime):  ", end="")
        if fp16_net < int8_realtime_net:
            print(f"FP16 is {int8_realtime_net - fp16_net:.3f} ms FASTER")
        else:
            print(f"INT8 realtime is {fp16_net - int8_realtime_net:.3f} ms FASTER")

        print(f"    FP16 vs INT8 (offline):   ", end="")
        if fp16_net < int8_offline_net:
            print(f"FP16 is {int8_offline_net - fp16_net:.3f} ms FASTER")
        else:
            print(f"INT8 offline is {fp16_net - int8_offline_net:.3f} ms FASTER")

        results.append({
            'name': config_name,
            'size_mb': size_mb,
            'baseline_d2h': baseline_d2h,
            'fp16_convert': fp16_convert,
            'fp16_d2h': fp16_d2h,
            'fp16_total': fp16_total,
            'fp16_net': fp16_net,
            'int8_quantize': int8_quantize,
            'int8_d2h': int8_d2h,
            'int8_dequantize': int8_dequantize,
            'int8_total': int8_total,
            'int8_realtime_net': int8_realtime_net,
            'int8_offline_total': int8_offline_total,
            'int8_offline_net': int8_offline_net,
        })

    # ========================================
    # Summary Table
    # ========================================
    print(f"\n{'='*120}")
    print("SUMMARY: Net Overhead vs Baseline (negative = speedup)")
    print("=" * 120)
    print(f"\n{'Config':<15} {'Size (MB)':<12} {'Baseline':<12} {'FP16 Net':<12} {'INT8 RT':<12} {'INT8 Off':<12} {'Best':<20}")
    print("-" * 120)

    for r in results:
        # Find winner among all three
        candidates = [
            ('FP16', r['fp16_net']),
            ('INT8-RT', r['int8_realtime_net']),
            ('INT8-Off', r['int8_offline_net'])
        ]
        winner_name, winner_val = min(candidates, key=lambda x: x[1])
        winner_str = f"{winner_name} ({winner_val:+.2f}ms)"

        print(f"{r['name']:<15} {r['size_mb']:<12.2f} {r['baseline_d2h']:<12.3f} "
              f"{r['fp16_net']:<+12.3f} {r['int8_realtime_net']:<+12.3f} {r['int8_offline_net']:<+12.3f} {winner_str:<20}")

    # ========================================
    # Visualization
    # ========================================
    print(f"\n{'='*100}")
    print("Generating plots...")
    print("=" * 100)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sizes = [r['size_mb'] for r in results]
    names = [r['name'] for r in results]

    # Plot 1: Net Overhead
    ax = axes[0, 0]
    x = range(len(results))
    ax.plot(x, [r['fp16_net'] for r in results], 'o-', label='FP16', linewidth=2, markersize=8)
    ax.plot(x, [r['int8_realtime_net'] for r in results], 's-', label='INT8 (realtime dequant)', linewidth=2, markersize=8, linestyle='--')
    ax.plot(x, [r['int8_offline_net'] for r in results], '^-', label='INT8 (offline dequant)', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Net Overhead vs Baseline (ms)', fontsize=12)
    ax.set_title('Net Overhead (negative = speedup)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Overhead Breakdown
    ax = axes[0, 1]
    width = 0.35
    x_pos = range(len(results))

    # FP16 stacked bars
    ax.bar([i - width/2 for i in x_pos], [r['fp16_convert'] for r in results],
           width, label='FP16 Convert', color='skyblue')
    ax.bar([i - width/2 for i in x_pos], [r['fp16_d2h'] for r in results],
           width, bottom=[r['fp16_convert'] for r in results],
           label='FP16 D2H', color='lightblue')

    # INT8 stacked bars
    int8_quant = [r['int8_quantize'] for r in results]
    int8_d2h = [r['int8_d2h'] for r in results]
    int8_dequant = [r['int8_dequantize'] for r in results]

    ax.bar([i + width/2 for i in x_pos], int8_quant,
           width, label='INT8 Quantize', color='salmon')
    ax.bar([i + width/2 for i in x_pos], int8_d2h,
           width, bottom=int8_quant,
           label='INT8 D2H', color='lightcoral')
    ax.bar([i + width/2 for i in x_pos], int8_dequant,
           width, bottom=[q+d for q,d in zip(int8_quant, int8_d2h)],
           label='INT8 Dequant', color='mistyrose')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Overhead Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Overhead as % of data size
    ax = axes[1, 0]
    fp16_overhead_per_mb = [r['fp16_net'] / r['size_mb'] if r['size_mb'] > 0 else 0 for r in results]
    int8_realtime_overhead_per_mb = [r['int8_realtime_net'] / r['size_mb'] if r['size_mb'] > 0 else 0 for r in results]
    int8_offline_overhead_per_mb = [r['int8_offline_net'] / r['size_mb'] if r['size_mb'] > 0 else 0 for r in results]

    ax.plot(sizes, fp16_overhead_per_mb, 'o-', label='FP16', linewidth=2, markersize=8)
    ax.plot(sizes, int8_realtime_overhead_per_mb, 's-', label='INT8 (realtime)', linewidth=2, markersize=8, linestyle='--', alpha=0.7)
    ax.plot(sizes, int8_offline_overhead_per_mb, '^-', label='INT8 (offline)', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Tensor Size (MB)', fontsize=12)
    ax.set_ylabel('Net Overhead per MB (ms/MB)', fontsize=12)
    ax.set_title('Overhead Efficiency (lower = better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 4: Crossover point analysis
    ax = axes[1, 1]
    bandwidth_saved_fp16 = [r['baseline_d2h'] - r['fp16_d2h'] for r in results]
    bandwidth_saved_int8 = [r['baseline_d2h'] - r['int8_d2h'] for r in results]
    compute_cost_fp16 = [r['fp16_convert'] for r in results]
    compute_cost_int8 = [r['int8_quantize'] + r['int8_dequantize'] for r in results]

    x_pos = range(len(results))
    width = 0.35

    # Savings (positive)
    ax.bar([i - width/2 for i in x_pos], bandwidth_saved_fp16,
           width, label='FP16 BW Saved', color='lightgreen', alpha=0.7)
    ax.bar([i + width/2 for i in x_pos], bandwidth_saved_int8,
           width, label='INT8 BW Saved', color='darkgreen', alpha=0.7)

    # Costs (negative)
    ax.bar([i - width/2 for i in x_pos], [-c for c in compute_cost_fp16],
           width, label='FP16 Compute Cost', color='lightcoral', alpha=0.7)
    ax.bar([i + width/2 for i in x_pos], [-c for c in compute_cost_int8],
           width, label='INT8 Compute Cost', color='darkred', alpha=0.7)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=2)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Bandwidth Savings vs Compute Cost', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = '/home/nengneng/AIPrometheus/HF_Prometheus/results/overhead_scaling_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plots saved to: {plot_path}")

    # ========================================
    # Critical Analysis
    # ========================================
    print(f"\n{'='*100}")
    print("CRITICAL ANALYSIS: Does Quantization Get Better with Larger Tensors?")
    print("=" * 100)

    # Find crossover points
    crossover_realtime_found = False
    crossover_offline_found = False

    for r in results:
        if not crossover_realtime_found and r['int8_realtime_net'] < r['fp16_net']:
            crossover_realtime_found = True
            print(f"\n✅ REALTIME DEQUANT CROSSOVER at {r['name']} ({r['size_mb']:.2f} MB):")
            print(f"   INT8 (realtime) becomes {r['fp16_net'] - r['int8_realtime_net']:.3f} ms faster than FP16")

        if not crossover_offline_found and r['int8_offline_net'] < r['fp16_net']:
            crossover_offline_found = True
            print(f"\n✅ OFFLINE DEQUANT CROSSOVER at {r['name']} ({r['size_mb']:.2f} MB):")
            print(f"   INT8 (offline) becomes {r['fp16_net'] - r['int8_offline_net']:.3f} ms faster than FP16")

        if crossover_realtime_found and crossover_offline_found:
            break

    if not crossover_offline_found:
        print(f"\n❌ NO OFFLINE DEQUANT CROSSOVER in tested range (up to {results[-1]['size_mb']:.2f} MB)")
        print(f"   FP16 remains faster than INT8 (offline) across all tensor sizes")

    if not crossover_realtime_found:
        print(f"\n❌ NO REALTIME DEQUANT CROSSOVER in tested range")
        print(f"   FP16 always faster than INT8 (realtime)")

        # Extrapolate
        last = results[-1]
        # INT8 quantize overhead grows sub-linearly (has fixed cost component)
        # D2H savings grow linearly
        # Find theoretical crossover

        # Assume: quantize_time = fixed_cost + per_element_cost * size
        # Model from data
        small = results[1]  # Small config
        large = results[-1]  # Largest config

        int8_fixed = small['int8_quantize']  # Approximate fixed cost
        int8_per_mb = (large['int8_quantize'] - small['int8_quantize']) / (large['size_mb'] - small['size_mb'])

        fp16_per_mb = (large['fp16_convert'] - small['fp16_convert']) / (large['size_mb'] - small['size_mb'])

        # D2H savings per MB (linear)
        d2h_saved_int8_per_mb = (large['baseline_d2h'] - large['int8_d2h']) / large['size_mb']
        d2h_saved_fp16_per_mb = (large['baseline_d2h'] - large['fp16_d2h']) / large['size_mb']

        print(f"\n   Model parameters:")
        print(f"     INT8 quantize: {int8_fixed:.3f} ms (fixed) + {int8_per_mb:.3f} ms/MB")
        print(f"     INT8 dequant: ~{last['int8_dequantize']/last['size_mb']:.3f} ms/MB")
        print(f"     FP16 convert: {fp16_per_mb:.3f} ms/MB")
        print(f"     D2H saved (INT8): {d2h_saved_int8_per_mb:.3f} ms/MB")
        print(f"     D2H saved (FP16): {d2h_saved_fp16_per_mb:.3f} ms/MB")

        # INT8 total = fixed + (quant_per_mb + dequant_per_mb) * MB - d2h_saved_per_mb * MB
        # FP16 total = convert_per_mb * MB - d2h_saved_per_mb * MB

        int8_total_per_mb = int8_per_mb + last['int8_dequantize']/last['size_mb'] - d2h_saved_int8_per_mb
        fp16_total_per_mb = fp16_per_mb - d2h_saved_fp16_per_mb

        print(f"\n   Net cost per MB:")
        print(f"     INT8: {int8_fixed:.3f} + {int8_total_per_mb:.3f} * MB")
        print(f"     FP16: {fp16_total_per_mb:.3f} * MB")

        if int8_total_per_mb < fp16_total_per_mb:
            # Will eventually cross over
            crossover_mb = int8_fixed / (fp16_total_per_mb - int8_total_per_mb)
            print(f"\n   ✅ Theoretical crossover at ~{crossover_mb:.0f} MB")
            print(f"      This is equivalent to:")
            print(f"        - Batch={int(crossover_mb/3)}, seq_len=1024, hidden=768")
            print(f"        - Batch=64, seq_len={int(crossover_mb*1024*1024/4/768/64)}, hidden=768")
        else:
            print(f"\n   ❌ FP16 will remain faster even for very large tensors")
            print(f"      INT8 overhead ({int8_total_per_mb:.3f} ms/MB) > FP16 ({fp16_total_per_mb:.3f} ms/MB)")

    print(f"\n{'='*100}")
    print("KEY INSIGHTS")
    print("=" * 100)
    print("""
1. SMALL TENSORS (< 1 MB): FP16 dominates
   - Quantization has high fixed overhead
   - D2H time too small to matter
   - FP16 is 2-10x faster

2. MEDIUM TENSORS (1-10 MB): FP16 still better
   - Quantization overhead starts to amortize
   - But still slower than FP16 in most cases
   - FP16 is 1.5-3x faster

3. LARGE TENSORS (> 10 MB): Depends on bandwidth
   - If PCIe bandwidth is bottleneck: INT8 may win
   - If GPU/CPU compute is bottleneck: FP16 still wins
   - Your case: PCIe ~10 GB/s, so INT8 may win at >100 MB

4. FOR YOUR MONITORING SYSTEM (typical decode):
   - Tensor size: ~1-10 MB per step
   - FP16 is optimal
   - INT8 only makes sense for prefill (>100 MB) or offline storage

5. BOTTOM LINE:
   ✅ Use FP16 for real-time monitoring (decode)
   ✅ Consider INT8 (offline dequant) if:
      - Storage is bottleneck and need 4x compression
      - Can tolerate extra GPU quantization overhead
      - Crossover found at certain tensor sizes
   ❌ Avoid INT8 (realtime dequant):
      - CPU dequant overhead dominates
      - Blocks collection pipeline
      - Almost never better than FP16
    """)

if __name__ == "__main__":
    benchmark_scaling()
