"""Simple benchmark to measure TransformerLens cache impact on inference performance."""

import torch
import time
import gc
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory() -> float:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 / 1024


def create_input_data(batch_size: int, seq_length: int, device: str = "cuda") -> torch.Tensor:
    """Create random input tokens."""
    vocab_size = 50257  # GPT-2 vocab size
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)


def benchmark_without_cache(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    num_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark normal inference without cache."""
    clear_gpu_memory()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids[:1])  # Small warmup
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    memory_before = get_gpu_memory()
    
    # Actual benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    memory_after = get_gpu_memory()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    batch_size, seq_length = input_ids.shape
    total_tokens = batch_size * seq_length * num_iterations
    
    return {
        "mode": "no_cache",
        "throughput_tokens_per_sec": total_tokens / elapsed_time,
        "latency_ms_per_batch": (elapsed_time * 1000) / num_iterations,
        "memory_mb": memory_after - memory_before,
        "peak_memory_mb": peak_memory,
        "total_time_sec": elapsed_time
    }


def benchmark_with_cache(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    num_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark inference with activation cache."""
    clear_gpu_memory()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _, _ = model.run_with_cache(input_ids[:1])  # Small warmup
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    memory_before = get_gpu_memory()
    
    # Actual benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output, cache = model.run_with_cache(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Clear cache to avoid accumulation
            del cache
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    memory_after = get_gpu_memory()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    batch_size, seq_length = input_ids.shape
    total_tokens = batch_size * seq_length * num_iterations
    
    return {
        "mode": "with_cache",
        "throughput_tokens_per_sec": total_tokens / elapsed_time,
        "latency_ms_per_batch": (elapsed_time * 1000) / num_iterations,
        "memory_mb": memory_after - memory_before,
        "peak_memory_mb": peak_memory,
        "total_time_sec": elapsed_time
    }


def print_comparison(results_no_cache: Dict, results_cache: Dict, batch_size: int, seq_length: int):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print(f"Batch Size: {batch_size}, Sequence Length: {seq_length}")
    print(f"{'='*60}")
    
    # Performance metrics
    print(f"\n{'Metric':<25} {'No Cache':>15} {'With Cache':>15} {'Overhead':>15}")
    print("-" * 70)
    
    # Throughput
    throughput_no_cache = results_no_cache["throughput_tokens_per_sec"]
    throughput_cache = results_cache["throughput_tokens_per_sec"]
    throughput_overhead = ((throughput_no_cache - throughput_cache) / throughput_no_cache) * 100
    print(f"{'Throughput (tokens/sec)':<25} {throughput_no_cache:>15.1f} {throughput_cache:>15.1f} {throughput_overhead:>14.1f}%")
    
    # Latency
    latency_no_cache = results_no_cache["latency_ms_per_batch"]
    latency_cache = results_cache["latency_ms_per_batch"]
    latency_overhead = ((latency_cache - latency_no_cache) / latency_no_cache) * 100
    print(f"{'Latency (ms/batch)':<25} {latency_no_cache:>15.2f} {latency_cache:>15.2f} {latency_overhead:>14.1f}%")
    
    # Memory
    memory_no_cache = results_no_cache["peak_memory_mb"]
    memory_cache = results_cache["peak_memory_mb"]
    memory_overhead = memory_cache - memory_no_cache
    print(f"{'Peak Memory (MB)':<25} {memory_no_cache:>15.1f} {memory_cache:>15.1f} {memory_overhead:>14.1f} MB")
    
    # Summary
    print(f"\n{'Summary':^70}")
    print("-" * 70)
    print(f"Cache overhead: {throughput_overhead:.1f}% slower, {memory_overhead:.1f} MB extra memory")
    

def run_benchmark(
    model_name: str = "gpt2",
    batch_sizes: List[int] = None,
    seq_lengths: List[int] = None,
    num_iterations: int = 10,
    device: str = "cuda"
):
    """Run the cache comparison benchmark."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]
    if seq_lengths is None:
        seq_lengths = [128, 256, 512]
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"TransformerLens Cache Impact Benchmark")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Iterations per test: {num_iterations}")
    
    # Load model
    print(f"\nLoading model: {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    print(f"Model loaded successfully")
    
    # Run benchmarks
    all_results = []
    
    for seq_length in seq_lengths:
        for batch_size in batch_sizes:
            try:
                print(f"\nTesting batch_size={batch_size}, seq_length={seq_length}...")
                
                # Create input data
                input_ids = create_input_data(batch_size, seq_length, device)
                
                # Run benchmarks
                results_no_cache = benchmark_without_cache(model, input_ids, num_iterations)
                results_cache = benchmark_with_cache(model, input_ids, num_iterations)
                
                # Store results
                all_results.append({
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "no_cache": results_no_cache,
                    "with_cache": results_cache
                })
                
                # Print comparison
                print_comparison(results_no_cache, results_cache, batch_size, seq_length)
                
            except Exception as e:
                print(f"Error with batch_size={batch_size}, seq_length={seq_length}: {e}")
                if "out of memory" in str(e).lower():
                    print("Skipping larger batch sizes due to memory constraints")
                    break
    
    # Print overall summary
    print(f"\n{'='*60}")
    print(f"Overall Summary")
    print(f"{'='*60}")
    
    if all_results:
        avg_throughput_overhead = []
        avg_memory_overhead = []
        
        for result in all_results:
            no_cache = result["no_cache"]["throughput_tokens_per_sec"]
            with_cache = result["with_cache"]["throughput_tokens_per_sec"]
            overhead = ((no_cache - with_cache) / no_cache) * 100
            avg_throughput_overhead.append(overhead)
            
            mem_diff = result["with_cache"]["peak_memory_mb"] - result["no_cache"]["peak_memory_mb"]
            avg_memory_overhead.append(mem_diff)
        
        print(f"Average throughput overhead: {sum(avg_throughput_overhead)/len(avg_throughput_overhead):.1f}%")
        print(f"Average memory overhead: {sum(avg_memory_overhead)/len(avg_memory_overhead):.1f} MB")
        
        # Find best and worst cases
        worst_case = max(all_results, key=lambda x: (x["no_cache"]["throughput_tokens_per_sec"] - x["with_cache"]["throughput_tokens_per_sec"]) / x["no_cache"]["throughput_tokens_per_sec"])
        best_case = min(all_results, key=lambda x: (x["no_cache"]["throughput_tokens_per_sec"] - x["with_cache"]["throughput_tokens_per_sec"]) / x["no_cache"]["throughput_tokens_per_sec"])
        
        worst_overhead = ((worst_case["no_cache"]["throughput_tokens_per_sec"] - worst_case["with_cache"]["throughput_tokens_per_sec"]) / worst_case["no_cache"]["throughput_tokens_per_sec"]) * 100
        best_overhead = ((best_case["no_cache"]["throughput_tokens_per_sec"] - best_case["with_cache"]["throughput_tokens_per_sec"]) / best_case["no_cache"]["throughput_tokens_per_sec"]) * 100
        
        print(f"\nWorst case: batch={worst_case['batch_size']}, seq={worst_case['seq_length']} -> {worst_overhead:.1f}% overhead")
        print(f"Best case: batch={best_case['batch_size']}, seq={best_case['seq_length']} -> {best_overhead:.1f}% overhead")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark TransformerLens cache impact on inference")
    parser.add_argument("--model", default="gpt2", help="Model to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512],
                       help="Sequence lengths to test")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations per test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to run on")
    
    args = parser.parse_args()
    
    results = run_benchmark(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_iterations=args.iterations,
        device=args.device
    )
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()