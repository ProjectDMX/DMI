#!/usr/bin/env python3
"""Test GPU memory cleanup after D2H transfer."""

import torch
import gc
import os

# Enable CPU transfer
os.environ['MON_NATIVE_TO_CPU'] = '1'
os.environ['MON_NATIVE_CALLBACK'] = '1'
os.environ['MON_NATIVE_BATCH'] = '1'

from monitoring import MonitoringEngine

def test_memory_leak():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available, skipping test")
        return

    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Initial GPU memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Create engine
    engine = MonitoringEngine(
        async_enabled=True,
        cache_dtype=torch.float16,
        queue_size=0,
        delay_steps=0,
    )

    # Create a large tensor
    batch_size = 64
    seq_len = 128
    hidden_size = 768

    tensor = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    tensor_size_mb = tensor.numel() * tensor.element_size() / 1024**2

    print(f"\nCreated tensor: {list(tensor.shape)}, {tensor_size_mb:.2f} MB")
    print(f"  GPU Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Submit tensor with transformations (remove_batch_dim + slice + dtype)
    engine.start_step()

    # Simulate hook submission with all transformations
    from monitoring._native_engine import NativeMonitoringEngine
    backend = engine._native_backend

    # Add task with remove_batch_dim=True, slice, dtype conversion
    task_tuple = (
        tensor,           # tensor
        -2,               # slice_dim
        True,             # remove_batch_dim
        True,             # can_slice
        (1, 0, 10, 1),    # slice: (mode=Range, start=0, stop=10, step=1)
        torch.float16,    # target dtype (will be converted)
    )

    token = backend.add_task(0, task_tuple)

    mem_before_seal = torch.cuda.memory_allocated() / 1024**2
    print(f"\nAfter add_task:")
    print(f"  GPU Allocated: {mem_before_seal:.2f} MB")

    # Seal step (triggers processing)
    stream_handle = torch.cuda.current_stream().cuda_stream
    engine.end_step()

    # Wait for D2H to complete
    torch.cuda.current_stream().synchronize()
    engine.resolve_all()
    torch.cuda.synchronize()

    mem_after_resolve = torch.cuda.memory_allocated() / 1024**2
    print(f"\nAfter resolve_all:")
    print(f"  GPU Allocated: {mem_after_resolve:.2f} MB")

    # Get result (should be on CPU)
    result = backend.future_result(token, None)
    print(f"\nResult device: {result.device}, dtype: {result.dtype}, shape: {list(result.shape)}")

    # Clear result
    del result
    del tensor
    gc.collect()
    torch.cuda.synchronize()

    mem_after_del = torch.cuda.memory_allocated() / 1024**2
    print(f"\nAfter del result + gc:")
    print(f"  GPU Allocated: {mem_after_del:.2f} MB")

    # Force cleanup
    backend.clear_completed_results()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem_final = torch.cuda.memory_allocated() / 1024**2
    print(f"\nAfter clear_completed_results + empty_cache:")
    print(f"  GPU Allocated: {mem_final:.2f} MB")

    # Check for leak
    print(f"\n{'='*60}")
    print(f"Memory leak check:")
    print(f"  Expected to free: ~{tensor_size_mb:.2f} MB (original tensor)")
    print(f"  Actually freed:   {mem_before_seal - mem_final:.2f} MB")

    if mem_final > mem_before_seal * 0.1:  # More than 10% remaining
        print(f"  ⚠️  WARNING: Potential memory leak detected!")
        print(f"     Still allocated: {mem_final:.2f} MB")
    else:
        print(f"  ✅ Memory properly cleaned")

    engine.close()

    # Peak memory
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\nPeak GPU memory: {peak_mem:.2f} MB")

if __name__ == '__main__':
    test_memory_leak()
