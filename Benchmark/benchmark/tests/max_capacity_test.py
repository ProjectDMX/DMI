"""Test maximum batch size and sequence length capacity for different implementations."""

import torch
import gc
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved": torch.cuda.memory_reserved() / 1024 / 1024,
        "total": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    }


def test_configuration(
    model, 
    batch_size: int, 
    seq_length: int,
    device: str = "cuda",
    use_cache: bool = False,
    vocab_size: int = 50257
) -> Tuple[bool, float, float]:
    """
    Test if a configuration works and measure memory usage.
    
    Returns:
        Tuple of (success, memory_used_mb, throughput_tokens_per_sec)
    """
    try:
        clear_gpu_memory()
        
        # Create input with correct vocab size
        input_ids = torch.randint(0, min(vocab_size, 50257), (batch_size, seq_length), device=device)
        
        # Measure initial memory
        mem_before = get_gpu_memory_info()["allocated"]
        
        # Test inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        
        with torch.no_grad():
            if use_cache and hasattr(model, 'run_with_cache'):
                output, cache = model.run_with_cache(input_ids)
                del cache
            else:
                output = model(input_ids)
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            elapsed_ms = start_time.elapsed_time(end_time)
        else:
            elapsed_ms = 100  # Dummy value for CPU
        
        # Measure memory after
        mem_after = get_gpu_memory_info()["allocated"]
        memory_used = mem_after - mem_before
        
        # Calculate throughput
        total_tokens = batch_size * seq_length
        throughput = (total_tokens / elapsed_ms) * 1000 if elapsed_ms > 0 else 0
        
        del output
        del input_ids
        clear_gpu_memory()
        
        return True, memory_used, throughput
        
    except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
        clear_gpu_memory()
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            return False, 0, 0
        # For other errors, also return False but don't raise
        print(f"    Error during test: {str(e)[:100]}")
        return False, 0, 0


def find_max_batch_size(
    model,
    seq_length: int,
    min_batch: int = 1,
    max_batch: int = 512,
    device: str = "cuda",
    use_cache: bool = False,
    vocab_size: int = 50257
) -> Dict[str, any]:
    """Find maximum batch size using binary search."""
    print(f"  Finding max batch size for seq_length={seq_length}...")
    
    left, right = min_batch, max_batch
    best_batch = min_batch
    best_memory = 0
    best_throughput = 0
    
    while left <= right:
        mid = (left + right) // 2
        success, memory, throughput = test_configuration(model, mid, seq_length, device, use_cache, vocab_size)
        
        if success:
            best_batch = mid
            best_memory = memory
            best_throughput = throughput
            left = mid + 1
            print(f"    Batch {mid}: ✓ (memory: {memory:.1f}MB, throughput: {throughput:.0f} tok/s)")
        else:
            right = mid - 1
            print(f"    Batch {mid}: ✗ (OOM)")
    
    return {
        "max_batch_size": best_batch,
        "sequence_length": seq_length,
        "memory_mb": best_memory,
        "throughput_tokens_per_sec": best_throughput,
        "tokens_per_mb": (best_batch * seq_length / best_memory) if best_memory > 0 else 0
    }


def find_max_sequence_length(
    model,
    batch_size: int,
    min_seq: int = 32,
    max_seq: int = 4096,
    device: str = "cuda",
    use_cache: bool = False,
    vocab_size: int = 50257
) -> Dict[str, any]:
    """Find maximum sequence length using binary search."""
    print(f"  Finding max sequence length for batch_size={batch_size}...")
    
    left, right = min_seq, max_seq
    best_seq = min_seq
    best_memory = 0
    best_throughput = 0
    
    # Round to nearest power of 2 or multiple of 64 for more realistic testing
    def round_seq(x):
        return ((x + 63) // 64) * 64
    
    while left <= right:
        mid = round_seq((left + right) // 2)
        success, memory, throughput = test_configuration(model, batch_size, mid, device, use_cache, vocab_size)
        
        if success:
            best_seq = mid
            best_memory = memory
            best_throughput = throughput
            left = mid + 1
            print(f"    Seq {mid}: ✓ (memory: {memory:.1f}MB, throughput: {throughput:.0f} tok/s)")
        else:
            right = mid - 1
            print(f"    Seq {mid}: ✗ (OOM)")
    
    return {
        "batch_size": batch_size,
        "max_sequence_length": best_seq,
        "memory_mb": best_memory,
        "throughput_tokens_per_sec": best_throughput,
        "tokens_per_mb": (batch_size * best_seq / best_memory) if best_memory > 0 else 0
    }


def test_implementation_capacity(
    implementation: str,
    model_name: str,
    test_batch_sizes: list,
    test_seq_lengths: list,
    device: str = "cuda"
) -> Dict[str, any]:
    """Test capacity for a specific implementation."""
    print(f"\n{'='*60}")
    print(f"Testing: {implementation}")
    print(f"{'='*60}")
    
    # Clear any previous CUDA errors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Try to clear any previous errors
        try:
            torch.cuda.synchronize()
        except:
            pass
    
    clear_gpu_memory()
    
    # Load model
    print(f"Loading model...")
    vocab_size = 50257  # Default GPT-2 vocab size
    
    try:
        if implementation == "HuggingFace":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=device
            )
            use_cache = False
            # Try to get vocab size from model config
            if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                vocab_size = model.config.vocab_size
        elif implementation == "TransformerLens (no cache)":
            model = HookedTransformer.from_pretrained(model_name, device=device)
            use_cache = False
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'd_vocab'):
                vocab_size = model.cfg.d_vocab
        elif implementation == "TransformerLens (with cache)":
            model = HookedTransformer.from_pretrained(model_name, device=device)
            use_cache = True
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'd_vocab'):
                vocab_size = model.cfg.d_vocab
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return {
            "implementation": implementation,
            "max_batch_tests": [],
            "max_seq_tests": [],
            "error": str(e)
        }
    
    results = {
        "implementation": implementation,
        "max_batch_tests": [],
        "max_seq_tests": []
    }
    
    # Test max batch size for different sequence lengths
    print("\nMax Batch Size Tests:")
    print(f"Using vocab_size: {vocab_size}")
    for seq_len in test_seq_lengths:
        try:
            result = find_max_batch_size(model, seq_len, device=device, use_cache=use_cache, vocab_size=vocab_size)
            results["max_batch_tests"].append(result)
        except Exception as e:
            print(f"  Error testing seq_len={seq_len}: {e}")
            continue
    
    # Test max sequence length for different batch sizes
    print("\nMax Sequence Length Tests:")
    for batch_size in test_batch_sizes:
        try:
            result = find_max_sequence_length(model, batch_size, device=device, use_cache=use_cache, vocab_size=vocab_size)
            results["max_seq_tests"].append(result)
        except Exception as e:
            print(f"  Error testing batch_size={batch_size}: {e}")
            continue
    
    # Clean up
    del model
    clear_gpu_memory()
    
    return results


def print_results_table(all_results: list):
    """Print formatted results tables."""
    print("\n" + "="*80)
    print("MAXIMUM CAPACITY SUMMARY")
    print("="*80)
    
    # Max batch size table
    print("\n1. Maximum Batch Sizes (by sequence length):")
    print("-"*80)
    print(f"{'Implementation':<30} {'Seq=128':<12} {'Seq=256':<12} {'Seq=512':<12} {'Seq=1024':<12}")
    print("-"*80)
    
    for result in all_results:
        impl = result["implementation"]
        if len(impl) > 30:
            impl = impl[:27] + "..."
        row = f"{impl:<30}"
        
        for test_seq in [128, 256, 512, 1024]:
            test = next((t for t in result["max_batch_tests"] 
                        if t["sequence_length"] == test_seq), None)
            if test:
                row += f" {test['max_batch_size']:<12}"
            else:
                row += f" {'N/A':<12}"
        print(row)
    
    # Max sequence length table
    print("\n2. Maximum Sequence Lengths (by batch size):")
    print("-"*80)
    print(f"{'Implementation':<30} {'Batch=1':<12} {'Batch=4':<12} {'Batch=8':<12} {'Batch=16':<12}")
    print("-"*80)
    
    for result in all_results:
        impl = result["implementation"]
        if len(impl) > 30:
            impl = impl[:27] + "..."
        row = f"{impl:<30}"
        
        for test_batch in [1, 4, 8, 16]:
            test = next((t for t in result["max_seq_tests"] 
                        if t["batch_size"] == test_batch), None)
            if test:
                row += f" {test['max_sequence_length']:<12}"
            else:
                row += f" {'N/A':<12}"
        print(row)
    
    # Memory efficiency table
    print("\n3. Memory Efficiency (tokens per MB):")
    print("-"*80)
    print(f"{'Implementation':<30} {'Avg Efficiency':<20} {'Best Case':<20} {'Worst Case':<20}")
    print("-"*80)
    
    for result in all_results:
        impl = result["implementation"]
        if len(impl) > 30:
            impl = impl[:27] + "..."
        
        all_efficiencies = []
        for test in result["max_batch_tests"] + result["max_seq_tests"]:
            if test.get("tokens_per_mb", 0) > 0:
                all_efficiencies.append(test["tokens_per_mb"])
        
        if all_efficiencies:
            avg_eff = sum(all_efficiencies) / len(all_efficiencies)
            best_eff = max(all_efficiencies)
            worst_eff = min(all_efficiencies)
            print(f"{impl:<30} {avg_eff:<20.1f} {best_eff:<20.1f} {worst_eff:<20.1f}")
        else:
            print(f"{impl:<30} {'N/A':<20} {'N/A':<20} {'N/A':<20}")


def save_results(results: list, model_name: str, output_dir: str = "results"):
    """Save results to JSON file."""
    output_path = Path(output_dir) / "max_capacity"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"capacity_{model_name}_{timestamp}.json"
    
    save_data = {
        "metadata": {
            "model": model_name,
            "timestamp": timestamp,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "total_gpu_memory_mb": get_gpu_memory_info()["total"]
        },
        "results": results
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return filename


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test maximum capacity for different implementations")
    parser.add_argument("--model", default="gpt2", help="Model to test")
    parser.add_argument("--test-batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16],
                       help="Batch sizes to test for max sequence length")
    parser.add_argument("--test-seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024],
                       help="Sequence lengths to test for max batch size")
    parser.add_argument("--implementations", nargs="+", 
                       default=["HuggingFace", "TransformerLens (no cache)", "TransformerLens (with cache)"],
                       help="Implementations to test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--no-save", action="store_true",
                       help="Do not save results to file")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"\n{'='*80}")
    print(f"Maximum Capacity Test")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    if args.device == "cuda":
        gpu_info = get_gpu_memory_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {gpu_info['total']:.0f} MB")
    
    all_results = []
    
    # Test each implementation
    for impl in args.implementations:
        try:
            results = test_implementation_capacity(
                impl,
                args.model,
                args.test_batch_sizes,
                args.test_seq_lengths,
                args.device
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error testing {impl}: {e}")
            continue
    
    # Print summary
    if all_results:
        print_results_table(all_results)
        
        # Save results
        if not args.no_save:
            save_results(all_results, args.model)
    else:
        print("\nNo results collected.")
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()