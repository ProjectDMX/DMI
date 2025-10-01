"""Base class for all benchmarks."""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
import transformer_lens
from transformer_lens import HookedTransformer

from .metrics import MetricsCollector, BenchmarkResult
from .utils import clear_gpu_memory, warmup_model, get_gpu_memory


class BaseBenchmark(ABC):
    """Base class for TransformerLens benchmarks."""
    
    def __init__(
        self,
        model_names: List[str] = None,
        batch_sizes: List[int] = None,
        sequence_lengths: List[int] = None,
        device: str = "cuda",
        output_dir: str = "results",
        verbose: bool = True
    ):
        """
        Initialize benchmark.
        
        Args:
            model_names: List of model names to test
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            device: Device to run on
            output_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.model_names = model_names or ["gpt2"]
        self.batch_sizes = batch_sizes or [1, 2, 4, 8]
        self.sequence_lengths = sequence_lengths or [128, 256, 512]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.metrics_collector = MetricsCollector()
        self.models = {}
        
        if self.verbose:
            print(f"Initialized benchmark on {self.device}")
            if self.device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Total GPU Memory: {total_memory:.2f} GB")
    
    def load_model(self, model_name: str) -> HookedTransformer:
        """Load a model, using cache if available."""
        if model_name not in self.models:
            if self.verbose:
                print(f"Loading model: {model_name}")
            
            clear_gpu_memory()
            model = HookedTransformer.from_pretrained(
                model_name,
                device=self.device
            )
            self.models[model_name] = model
            
            if self.verbose:
                allocated, _ = get_gpu_memory()
                print(f"Model loaded. GPU memory used: {allocated:.2f} MB")
        
        return self.models[model_name]
    
    @abstractmethod
    def run_single_test(
        self,
        model: HookedTransformer,
        batch_size: int,
        sequence_length: int,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a single benchmark test.
        
        Args:
            model: The model to test
            batch_size: Batch size
            sequence_length: Sequence length
            **kwargs: Additional test-specific parameters
        
        Returns:
            BenchmarkResult object
        """
        pass
    
    def run(self, **kwargs) -> List[BenchmarkResult]:
        """
        Run the complete benchmark suite.
        
        Args:
            **kwargs: Additional parameters to pass to run_single_test
        
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for model_name in self.model_names:
            try:
                model = self.load_model(model_name)
                
                for seq_len in self.sequence_lengths:
                    for batch_size in self.batch_sizes:
                        if self.verbose:
                            print(f"\nTesting {model_name} with batch_size={batch_size}, seq_len={seq_len}")
                        
                        try:
                            result = self.run_single_test(
                                model=model,
                                batch_size=batch_size,
                                sequence_length=seq_len,
                                model_name=model_name,
                                **kwargs
                            )
                            results.append(result)
                            
                            if self.verbose:
                                print(f"  Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
                                print(f"  Latency: {result.latency_ms_per_token:.2f} ms/token")
                                print(f"  GPU Memory: {result.gpu_memory_mb:.2f} MB")
                        
                        except Exception as e:
                            if self.verbose:
                                print(f"  Error: {str(e)}")
                            
                            # Create error result
                            error_result = BenchmarkResult(
                                model_name=model_name,
                                test_name=self.__class__.__name__,
                                batch_size=batch_size,
                                sequence_length=seq_len,
                                error=str(e)
                            )
                            results.append(error_result)
                            
                            # Clear memory and continue
                            clear_gpu_memory()
            
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load model {model_name}: {str(e)}")
                continue
        
        # Save results
        if results:
            self.metrics_collector.results = results
            self.metrics_collector.save_results(self.output_dir)
            
            if self.verbose:
                summary = self.metrics_collector.get_summary()
                print("\n" + "="*50)
                print("Benchmark Summary:")
                print(f"Total tests run: {summary.get('total_tests', 0)}")
                print(f"Models tested: {summary.get('models_tested', [])}")
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        self.models.clear()
        clear_gpu_memory()