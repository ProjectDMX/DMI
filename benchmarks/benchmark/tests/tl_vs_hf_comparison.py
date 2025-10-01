"""Benchmark comparing TransformerLens vs HuggingFace implementation performance."""

import torch
import time
import gc
import json
from datetime import datetime
from typing import Dict, List, Optional
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def create_input_data(batch_size: int, seq_length: int, vocab_size: int = 50257, device: str = "cuda") -> torch.Tensor:
    """Create random input tokens."""
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)


def benchmark_huggingface(
    model_name: str,
    input_ids: torch.Tensor,
    num_iterations: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark HuggingFace native model."""
    clear_gpu_memory()
    
    # Load model
    print("  Loading HuggingFace model...")
    load_start = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    hf_model.eval()
    load_time = time.perf_counter() - load_start
    print(f"  HuggingFace model loaded in {load_time:.2f} seconds")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = hf_model(input_ids[:1])
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    memory_before = get_gpu_memory()
    
    # Actual benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = hf_model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    memory_after = get_gpu_memory()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    batch_size, seq_length = input_ids.shape
    total_tokens = batch_size * seq_length * num_iterations
    
    # Clean up
    del hf_model
    clear_gpu_memory()
    
    return {
        "implementation": "HuggingFace",
        "load_time_sec": load_time,
        "throughput_tokens_per_sec": total_tokens / elapsed_time,
        "latency_ms_per_batch": (elapsed_time * 1000) / num_iterations,
        "memory_mb": memory_after - memory_before,
        "peak_memory_mb": peak_memory,
        "total_time_sec": elapsed_time
    }


def benchmark_transformerlens_no_cache(
    model_name: str,
    input_ids: torch.Tensor,
    num_iterations: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark TransformerLens without cache."""
    clear_gpu_memory()
    
    # Load model
    print("  Loading TransformerLens model...")
    load_start = time.perf_counter()
    tl_model = HookedTransformer.from_pretrained(model_name, device=device)
    tl_model.eval()
    load_time = time.perf_counter() - load_start
    print(f"  TransformerLens model loaded in {load_time:.2f} seconds")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = tl_model(input_ids[:1])
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    memory_before = get_gpu_memory()
    
    # Actual benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = tl_model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    memory_after = get_gpu_memory()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    batch_size, seq_length = input_ids.shape
    total_tokens = batch_size * seq_length * num_iterations
    
    # Clean up
    del tl_model
    clear_gpu_memory()
    
    return {
        "implementation": "TransformerLens (no cache)",
        "load_time_sec": load_time,
        "throughput_tokens_per_sec": total_tokens / elapsed_time,
        "latency_ms_per_batch": (elapsed_time * 1000) / num_iterations,
        "memory_mb": memory_after - memory_before,
        "peak_memory_mb": peak_memory,
        "total_time_sec": elapsed_time
    }


def benchmark_transformerlens_with_cache(
    model_name: str,
    input_ids: torch.Tensor,
    num_iterations: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark TransformerLens with cache."""
    clear_gpu_memory()
    
    # Load model
    print("  Loading TransformerLens model (for cache test)...")
    load_start = time.perf_counter()
    tl_model = HookedTransformer.from_pretrained(model_name, device=device)
    tl_model.eval()
    load_time = time.perf_counter() - load_start
    print(f"  TransformerLens model loaded in {load_time:.2f} seconds")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _, _ = tl_model.run_with_cache(input_ids[:1])
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    memory_before = get_gpu_memory()
    
    # Actual benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output, cache = tl_model.run_with_cache(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            del cache  # Free cache memory immediately
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    memory_after = get_gpu_memory()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    batch_size, seq_length = input_ids.shape
    total_tokens = batch_size * seq_length * num_iterations
    
    # Clean up
    del tl_model
    clear_gpu_memory()
    
    return {
        "implementation": "TransformerLens (with cache)",
        "load_time_sec": load_time,
        "throughput_tokens_per_sec": total_tokens / elapsed_time,
        "latency_ms_per_batch": (elapsed_time * 1000) / num_iterations,
        "memory_mb": memory_after - memory_before,
        "peak_memory_mb": peak_memory,
        "total_time_sec": elapsed_time
    }


def print_comparison_table(results: List[Dict], batch_size: int, seq_length: int):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"Batch Size: {batch_size}, Sequence Length: {seq_length}")
    print(f"{'='*80}")
    
    # Find HuggingFace baseline
    hf_result = next((r for r in results if "HuggingFace" in r["implementation"]), None)
    if not hf_result:
        print("Error: No HuggingFace baseline found")
        return
    
    # Print header
    print(f"\n{'Implementation':<30} {'Load Time':>12} {'Throughput':>15} {'Latency':>12} {'Memory':>12}")
    print(f"{'':30} {'(seconds)':>12} {'(tokens/sec)':>15} {'(ms/batch)':>12} {'(MB)':>12}")
    print("-" * 95)
    
    # Print results
    for result in results:
        impl = result["implementation"]
        load_time = result["load_time_sec"]
        throughput = result["throughput_tokens_per_sec"]
        latency = result["latency_ms_per_batch"]
        memory = result["peak_memory_mb"]
        
        print(f"{impl:<30} {load_time:>12.2f} {throughput:>15.1f} {latency:>12.2f} {memory:>12.1f}")
    
    # Print overhead analysis
    print(f"\n{'Overhead Analysis (vs HuggingFace)':^95}")
    print("-" * 95)
    
    hf_throughput = hf_result["throughput_tokens_per_sec"]
    hf_latency = hf_result["latency_ms_per_batch"]
    hf_memory = hf_result["peak_memory_mb"]
    hf_load_time = hf_result["load_time_sec"]
    
    for result in results:
        if "HuggingFace" in result["implementation"]:
            continue
            
        impl = result["implementation"]
        throughput_overhead = ((hf_throughput - result["throughput_tokens_per_sec"]) / hf_throughput) * 100
        latency_overhead = ((result["latency_ms_per_batch"] - hf_latency) / hf_latency) * 100
        memory_overhead = result["peak_memory_mb"] - hf_memory
        load_time_overhead = ((result["load_time_sec"] - hf_load_time) / hf_load_time) * 100
        
        print(f"\n{impl}:")
        print(f"  Load time: {load_time_overhead:+.1f}% ({result['load_time_sec'] - hf_load_time:+.2f}s)")
        print(f"  Throughput: {-throughput_overhead:+.1f}% slower")
        print(f"  Latency: {latency_overhead:+.1f}% higher")
        print(f"  Memory: {memory_overhead:+.1f} MB more")


def save_results_to_json(results: List[Dict], model_name: str, output_dir: str = "results"):
    """Save benchmark results to JSON file."""
    output_path = Path(output_dir) / "tl_vs_hf"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"comparison_{model_name}_{timestamp}.json"
    
    # Prepare data for saving
    save_data = {
        "metadata": {
            "model": model_name,
            "timestamp": timestamp,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        },
        "results": results
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return filename


def run_comparison(
    model_name: str = "gpt2",
    batch_sizes: List[int] = None,
    seq_lengths: List[int] = None,
    num_iterations: int = 10,
    device: str = "cuda",
    save_results: bool = True
):
    """Run the full comparison benchmark."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    if seq_lengths is None:
        seq_lengths = [128, 256]
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print(f"\n{'='*80}")
    print(f"TransformerLens vs HuggingFace Performance Comparison")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Iterations per test: {num_iterations}")
    
    all_results = []
    
    for seq_length in seq_lengths:
        for batch_size in batch_sizes:
            print(f"\n{'='*80}")
            print(f"Testing batch_size={batch_size}, seq_length={seq_length}")
            print(f"{'='*80}")
            
            try:
                # Create input data
                input_ids = create_input_data(batch_size, seq_length, device=device)
                
                results = []
                
                # Test HuggingFace
                print("\n1. Testing HuggingFace implementation...")
                try:
                    hf_result = benchmark_huggingface(model_name, input_ids, num_iterations, device)
                    results.append(hf_result)
                except Exception as e:
                    print(f"  Error testing HuggingFace: {e}")
                
                # Test TransformerLens without cache
                print("\n2. Testing TransformerLens (no cache)...")
                try:
                    tl_result = benchmark_transformerlens_no_cache(model_name, input_ids, num_iterations, device)
                    results.append(tl_result)
                except Exception as e:
                    print(f"  Error testing TransformerLens: {e}")
                
                # Test TransformerLens with cache
                print("\n3. Testing TransformerLens (with cache)...")
                try:
                    tl_cache_result = benchmark_transformerlens_with_cache(model_name, input_ids, num_iterations, device)
                    results.append(tl_cache_result)
                except Exception as e:
                    print(f"  Error testing TransformerLens with cache: {e}")
                
                # Store results with batch_size and seq_length info
                for result in results:
                    result["batch_size"] = batch_size
                    result["sequence_length"] = seq_length
                
                # Print comparison
                if results:
                    print_comparison_table(results, batch_size, seq_length)
                    all_results.extend(results)
                
            except Exception as e:
                print(f"Error with batch_size={batch_size}, seq_length={seq_length}: {e}")
                if "out of memory" in str(e).lower():
                    print("Skipping larger batch sizes due to memory constraints")
                    break
    
    # Print overall summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"Overall Summary")
        print(f"{'='*80}")
        
        # Group by implementation
        impl_groups = {}
        for result in all_results:
            impl = result["implementation"]
            if impl not in impl_groups:
                impl_groups[impl] = []
            impl_groups[impl].append(result)
        
        # Calculate averages
        print(f"\n{'Implementation':<30} {'Avg Throughput':>20} {'Avg Latency':>15} {'Avg Memory':>15}")
        print(f"{'':30} {'(tokens/sec)':>20} {'(ms/batch)':>15} {'(MB)':>15}")
        print("-" * 85)
        
        for impl, results in impl_groups.items():
            avg_throughput = sum(r["throughput_tokens_per_sec"] for r in results) / len(results)
            avg_latency = sum(r["latency_ms_per_batch"] for r in results) / len(results)
            avg_memory = sum(r["peak_memory_mb"] for r in results) / len(results)
            
            print(f"{impl:<30} {avg_throughput:>20.1f} {avg_latency:>15.2f} {avg_memory:>15.1f}")
        
        # Key findings
        print(f"\n{'Key Findings':^85}")
        print("-" * 85)
        
        hf_results = impl_groups.get("HuggingFace", [])
        tl_no_cache = impl_groups.get("TransformerLens (no cache)", [])
        tl_with_cache = impl_groups.get("TransformerLens (with cache)", [])
        
        if hf_results and tl_no_cache:
            hf_avg_throughput = sum(r["throughput_tokens_per_sec"] for r in hf_results) / len(hf_results)
            tl_avg_throughput = sum(r["throughput_tokens_per_sec"] for r in tl_no_cache) / len(tl_no_cache)
            overhead = ((hf_avg_throughput - tl_avg_throughput) / hf_avg_throughput) * 100
            print(f"• TransformerLens (no cache) is {overhead:.1f}% slower than HuggingFace")
        
        if tl_no_cache and tl_with_cache:
            tl_nc_throughput = sum(r["throughput_tokens_per_sec"] for r in tl_no_cache) / len(tl_no_cache)
            tl_c_throughput = sum(r["throughput_tokens_per_sec"] for r in tl_with_cache) / len(tl_with_cache)
            cache_overhead = ((tl_nc_throughput - tl_c_throughput) / tl_nc_throughput) * 100
            print(f"• Enabling cache adds {cache_overhead:.1f}% additional overhead")
        
        if hf_results:
            avg_load_time = sum(r["load_time_sec"] for r in hf_results) / len(hf_results)
            print(f"• HuggingFace average model load time: {avg_load_time:.2f} seconds")
        
        if tl_no_cache:
            avg_load_time = sum(r["load_time_sec"] for r in tl_no_cache) / len(tl_no_cache)
            print(f"• TransformerLens average model load time: {avg_load_time:.2f} seconds")
    
    # Save results if requested
    if save_results and all_results:
        save_results_to_json(all_results, model_name)
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare TransformerLens vs HuggingFace performance")
    parser.add_argument("--model", default="gpt2", help="Model to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8],
                       help="Batch sizes to test")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256],
                       help="Sequence lengths to test")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations per test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--no-save", action="store_true",
                       help="Do not save results to JSON file")
    
    args = parser.parse_args()
    
    # Determine whether to save results
    save_results = not args.no_save
    
    if save_results:
        print("Note: Results will be saved to results/tl_vs_hf/ directory")
    else:
        print("Note: Results will NOT be saved (--no-save flag is set)")
    
    results = run_comparison(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_iterations=args.iterations,
        device=args.device,
        save_results=save_results
    )
    
    print("\nBenchmark complete!")
    
    if save_results and results:
        print("Results have been saved successfully.")
    elif not save_results:
        print("Results were not saved (--no-save flag was used).")
    elif not results:
        print("No results to save (tests may have failed).")


if __name__ == "__main__":
    main()