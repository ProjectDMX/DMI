"""Metrics collection and management for benchmarks."""

import time
import torch
import psutil
import GPUtil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import csv
from pathlib import Path
import numpy as np


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    model_name: str
    test_name: str
    batch_size: int
    sequence_length: int
    
    # Performance metrics
    throughput_tokens_per_sec: float = 0.0
    latency_ms_per_batch: float = 0.0
    latency_ms_per_token: float = 0.0
    
    # Memory metrics
    gpu_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    
    # Hook configuration
    num_hooks: int = 0
    hooks_enabled: bool = False
    cache_all_activations: bool = False
    cached_activation_names: List[str] = field(default_factory=list)
    
    # Additional info
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    gpu_name: str = ""
    error: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MetricsCollector:
    """Collects and manages benchmark metrics."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.current_result: Optional[BenchmarkResult] = None
        self.start_time: Optional[float] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get GPU info if available
        self.gpu_name = self._get_gpu_name()
    
    def _get_gpu_name(self) -> str:
        """Get GPU name if available."""
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].name
            except:
                pass
            return torch.cuda.get_device_name(0)
        return "CPU"
    
    def start_benchmark(self, model_name: str, test_name: str, 
                        batch_size: int, sequence_length: int) -> None:
        """Start a new benchmark measurement."""
        self.current_result = BenchmarkResult(
            model_name=model_name,
            test_name=test_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            gpu_name=self.gpu_name
        )
        
        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        self.start_time = time.perf_counter()
    
    def end_benchmark(self, tokens_processed: int) -> BenchmarkResult:
        """End benchmark and calculate metrics."""
        if self.current_result is None or self.start_time is None:
            raise ValueError("No benchmark in progress")
        
        # Calculate time metrics
        elapsed_time = time.perf_counter() - self.start_time
        self.current_result.latency_ms_per_batch = elapsed_time * 1000
        self.current_result.throughput_tokens_per_sec = tokens_processed / elapsed_time
        self.current_result.latency_ms_per_token = (elapsed_time * 1000) / tokens_processed
        
        # Collect memory metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.current_result.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.current_result.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # CPU memory
        process = psutil.Process()
        self.current_result.cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Store result
        self.results.append(self.current_result)
        result = self.current_result
        self.current_result = None
        self.start_time = None
        
        return result
    
    def set_hook_config(self, num_hooks: int, hooks_enabled: bool,
                        cache_all: bool, cached_names: List[str] = None) -> None:
        """Set hook configuration for current benchmark."""
        if self.current_result is None:
            raise ValueError("No benchmark in progress")
        
        self.current_result.num_hooks = num_hooks
        self.current_result.hooks_enabled = hooks_enabled
        self.current_result.cache_all_activations = cache_all
        if cached_names:
            self.current_result.cached_activation_names = cached_names
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add additional metric to current benchmark."""
        if self.current_result is None:
            raise ValueError("No benchmark in progress")
        self.current_result.additional_metrics[key] = value
    
    def save_results(self, output_dir: Path) -> None:
        """Save all results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_dir / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # Save as CSV
        if self.results:
            csv_path = output_dir / f"results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            keys = self.results[0].to_dict().keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())
        
        print(f"Results saved to {output_dir}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all results."""
        if not self.results:
            return {}
        
        summary = {
            "total_tests": len(self.results),
            "models_tested": list(set(r.model_name for r in self.results)),
            "batch_sizes": list(set(r.batch_size for r in self.results)),
            "sequence_lengths": list(set(r.sequence_length for r in self.results)),
        }
        
        # Calculate aggregates by test type
        test_groups = {}
        for result in self.results:
            key = f"{result.test_name}_{result.model_name}"
            if key not in test_groups:
                test_groups[key] = []
            test_groups[key].append(result)
        
        summary["test_summaries"] = {}
        for key, results in test_groups.items():
            throughputs = [r.throughput_tokens_per_sec for r in results]
            latencies = [r.latency_ms_per_token for r in results]
            memories = [r.gpu_memory_mb for r in results]
            
            summary["test_summaries"][key] = {
                "avg_throughput": np.mean(throughputs),
                "max_throughput": np.max(throughputs),
                "min_latency": np.min(latencies),
                "avg_memory_mb": np.mean(memories),
            }
        
        return summary