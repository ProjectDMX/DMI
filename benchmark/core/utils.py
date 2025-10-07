"""Utility functions for benchmarking."""

import torch
import time
import gc
from typing import Tuple, List, Optional, Callable, Any
from contextlib import contextmanager
import numpy as np


def get_gpu_memory() -> Tuple[float, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Tuple of (allocated_mb, reserved_mb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    return allocated, reserved


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.perf_counter()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    try:
        yield lambda: time.perf_counter() - start
    finally:
        torch.cuda.synchronize() if torch.cuda.is_available() else None


def create_sample_data(
    batch_size: int,
    sequence_length: int,
    vocab_size: int = 50257,  # GPT-2 vocab size
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create sample token data for benchmarking.
    
    Args:
        batch_size: Number of sequences in batch
        sequence_length: Length of each sequence
        vocab_size: Size of vocabulary
        device: Device to create tensor on
    
    Returns:
        Tensor of shape (batch_size, sequence_length)
    """
    # Create realistic-looking token distribution
    # Most tokens are common (low IDs), with occasional rare tokens
    common_tokens = torch.randint(0, min(1000, vocab_size), 
                                  (batch_size, int(sequence_length * 0.8)))
    rare_tokens = torch.randint(1000, vocab_size, 
                                (batch_size, int(sequence_length * 0.2)))
    
    tokens = torch.cat([common_tokens, rare_tokens], dim=1)
    
    # Shuffle along sequence dimension
    for i in range(batch_size):
        tokens[i] = tokens[i][torch.randperm(sequence_length)]
    
    return tokens[:, :sequence_length].to(device)


def create_text_samples(
    batch_size: int,
    approximate_length: int = 100
) -> List[str]:
    """
    Create sample text for benchmarking.
    
    Args:
        batch_size: Number of text samples
        approximate_length: Approximate number of tokens per sample
    
    Returns:
        List of text strings
    """
    # Sample sentences that roughly correspond to token counts
    templates = [
        "The quick brown fox jumps over the lazy dog. " * (approximate_length // 10),
        "In the heart of the bustling city, where skyscrapers touch the clouds, " * (approximate_length // 15),
        "Scientists have discovered a new species of butterfly in the Amazon rainforest. " * (approximate_length // 12),
        "The annual technology conference brings together innovators from around the world. " * (approximate_length // 13),
        "Climate change continues to be one of the most pressing issues of our time. " * (approximate_length // 14),
    ]
    
    samples = []
    for i in range(batch_size):
        template = templates[i % len(templates)]
        # Add some variation
        variation = f" Sample {i}. " + template
        samples.append(variation[:approximate_length * 4])  # Rough char to token ratio
    
    return samples


def warmup_model(model, input_ids: torch.Tensor, num_iterations: int = 3):
    """
    Warm up the model to ensure consistent timing.
    
    Args:
        model: The model to warm up
        input_ids: Sample input tensor
        num_iterations: Number of warmup iterations
    """
    with torch.no_grad():
        for _ in range(num_iterations):
            if input_ids.shape[0] > 1:
                # Use smaller batch for warmup to save memory
                warmup_input = input_ids[:1]
            else:
                warmup_input = input_ids
            
            _ = model(warmup_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def calculate_max_batch_size(
    model,
    sequence_length: int,
    start_batch_size: int = 1,
    max_batch_size: int = 256,
    safety_margin: float = 0.9
) -> int:
    """
    Find the maximum batch size that fits in GPU memory.
    
    Args:
        model: The model to test
        sequence_length: Sequence length to test with
        start_batch_size: Starting batch size
        max_batch_size: Maximum batch size to try
        safety_margin: Use only this fraction of available memory
    
    Returns:
        Maximum viable batch size
    """
    if not torch.cuda.is_available():
        return min(4, max_batch_size)  # Conservative for CPU
    
    device = next(model.parameters()).device
    clear_gpu_memory()
    
    # Binary search for max batch size
    low = start_batch_size
    high = max_batch_size
    best_batch_size = start_batch_size
    
    while low <= high:
        mid = (low + high) // 2
        
        try:
            clear_gpu_memory()
            test_input = create_sample_data(mid, sequence_length, device=device)
            
            with torch.no_grad():
                _ = model(test_input)
            
            # Check memory usage
            allocated, _ = get_gpu_memory()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            
            if allocated / total_memory < safety_margin:
                best_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
                
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            high = mid - 1
            clear_gpu_memory()
    
    clear_gpu_memory()
    return best_batch_size


def run_with_timeout(func: Callable, timeout: float, *args, **kwargs) -> Any:
    """
    Run a function with a timeout.
    
    Note: This is a simple implementation. For production use,
    consider using multiprocessing or threading with proper timeout handling.
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function timed out after {timeout} seconds")
    
    # Set the timeout handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))
    
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    return result