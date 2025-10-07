"""Batch inference benchmark to test throughput at different batch sizes."""

import torch
import time
from typing import Optional
from transformer_lens import HookedTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from benchmark.core.base_benchmark import BaseBenchmark
from benchmark.core.metrics import BenchmarkResult
from benchmark.core.utils import (
    create_sample_data, 
    warmup_model, 
    clear_gpu_memory,
    get_gpu_memory,
    calculate_max_batch_size
)


class BatchInferenceBenchmark(BaseBenchmark):
    """Benchmark for testing batch inference performance."""
    
    def run_single_test(
        self,
        model: HookedTransformer,
        batch_size: int,
        sequence_length: int,
        model_name: str,
        num_iterations: int = 10,
        test_max_batch: bool = False,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run batch inference benchmark.
        
        Args:
            model: The model to test
            batch_size: Batch size
            sequence_length: Sequence length
            model_name: Name of the model
            num_iterations: Number of iterations to average over
            test_max_batch: Whether to test maximum batch size
        
        Returns:
            BenchmarkResult with performance metrics
        """
        clear_gpu_memory()
        
        # Test maximum batch size if requested
        if test_max_batch and batch_size == self.batch_sizes[0]:
            max_batch = calculate_max_batch_size(
                model, 
                sequence_length,
                max_batch_size=128
            )
            if self.verbose:
                print(f"  Maximum batch size for seq_len={sequence_length}: {max_batch}")
        
        # Create test data
        input_ids = create_sample_data(batch_size, sequence_length, device=self.device)
        
        # Warmup
        warmup_model(model, input_ids, num_iterations=3)
        
        # Start metrics collection
        self.metrics_collector.start_benchmark(
            model_name=model_name,
            test_name="batch_inference",
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        # Record initial memory
        initial_memory, _ = get_gpu_memory()
        
        # Run inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(input_ids)
                # Force synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # Calculate metrics
        total_tokens = batch_size * sequence_length * num_iterations
        
        # Manually set metrics since we're doing multiple iterations
        result = self.metrics_collector.current_result
        result.throughput_tokens_per_sec = total_tokens / elapsed_time
        result.latency_ms_per_batch = (elapsed_time * 1000) / num_iterations
        result.latency_ms_per_token = (elapsed_time * 1000) / total_tokens
        
        # Memory metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            result.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Add additional metrics
        self.metrics_collector.add_metric("num_iterations", num_iterations)
        self.metrics_collector.add_metric("initial_memory_mb", initial_memory)
        self.metrics_collector.add_metric("memory_increase_mb", result.gpu_memory_mb - initial_memory)
        
        # No hooks in basic inference
        self.metrics_collector.set_hook_config(
            num_hooks=0,
            hooks_enabled=False,
            cache_all=False
        )
        
        # Finalize without calling end_benchmark (since we calculated manually)
        self.metrics_collector.results.append(result)
        self.metrics_collector.current_result = None
        
        return result


def main():
    """Run batch inference benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Inference Benchmark")
    parser.add_argument("--models", nargs="+", default=["gpt2"], 
                       help="Models to test")
    parser.add_argument("--batch-sizes", nargs="+", type=int, 
                       default=[1, 2, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                       default=[128, 256, 512],
                       help="Sequence lengths to test")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations per test")
    parser.add_argument("--output-dir", default="results/batch_inference",
                       help="Output directory")
    parser.add_argument("--test-max-batch", action="store_true",
                       help="Test maximum batch size")
    
    args = parser.parse_args()
    
    benchmark = BatchInferenceBenchmark(
        model_names=args.models,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.seq_lengths,
        output_dir=args.output_dir,
        verbose=True
    )
    
    print("="*50)
    print("Running Batch Inference Benchmark")
    print("="*50)
    
    results = benchmark.run(
        num_iterations=args.iterations,
        test_max_batch=args.test_max_batch
    )
    
    print("\n" + "="*50)
    print("Benchmark Complete!")
    print(f"Results saved to: {args.output_dir}")
    
    # Print best configurations
    if results:
        best_throughput = max(results, key=lambda x: x.throughput_tokens_per_sec if x.error is None else 0)
        print(f"\nBest throughput: {best_throughput.throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"  Model: {best_throughput.model_name}")
        print(f"  Batch size: {best_throughput.batch_size}")
        print(f"  Sequence length: {best_throughput.sequence_length}")
    
    benchmark.cleanup()


if __name__ == "__main__":
    main()