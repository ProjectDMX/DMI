#!/usr/bin/env python3
"""Test script to verify C++ backend is used with debug logging."""

import sys
import torch
from pathlib import Path

print("=" * 60)
print("Step 1: Pre-compile C++ backend with debug logging")
print("=" * 60)

from torch.utils.cpp_extension import load

source_dir = Path(__file__).resolve().parent / "monitoring" / "csrc"
source_file = source_dir / "native_engine.cpp"

print(f"Source file: {source_file}")
print(f"Exists: {source_file.exists()}")

# Compile with MON_DEBUG flag
module = load(
    name='monitoring_native_backend',
    sources=[str(source_file)],
    extra_cflags=['-O3', '-DMON_DEBUG=1'],
    extra_cuda_cflags=['-O3', '-DMON_DEBUG=1'],
    verbose=True,
)

print(f"\n✅ Module compiled: {module}")
print(f"   Module attributes: {dir(module)}")

print("\n" + "=" * 60)
print("Step 2: Test C++ module directly")
print("=" * 60)

# Test direct usage
engine = module.create_engine(
    queue_size=128,
    cache_dtype=torch.float16,
    delay_steps=0
)

print(f"✅ Engine created: {engine}")
stats = engine.get_stats()
print(f"✅ Stats: {stats}")

print("\n" + "=" * 60)
print("Step 3: Test via Python wrapper")
print("=" * 60)

# Now import the wrapper (should find the pre-compiled module)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitoring import MonitoringEngine

wrapper_engine = MonitoringEngine(
    queue_size=128,
    cache_dtype=torch.float16,
    async_enabled=True,
    delay_steps=0
)

print(f"\n✅ Wrapper engine created")
print(f"   Using native backend: {wrapper_engine._using_native_backend}")
print(f"   Native backend object: {wrapper_engine._native_backend}")

print("\n" + "=" * 60)
print("Step 4: Submit test tasks")
print("=" * 60)

# Create a simple test tensor on CUDA
device = torch.device("cuda:0")
test_tensor = torch.randn(4, 8, 64, device=device)

print(f"Test tensor shape: {test_tensor.shape}")

# Create a dummy task
from monitoring.task import MonitoringTask

task = MonitoringTask(
    name="test_activation",
    tensor=test_tensor,
    step_id=1
)

# Submit through wrapper
print("\nSubmitting step 1 with 1 task...")
futures = wrapper_engine.submit_step(1, [task])
print(f"✅ Submitted, got {len(futures)} futures")

# Wait and get result
print("\nWaiting for result...")
result = futures[0].result(timeout=5.0)
print(f"✅ Result shape: {result.shape}")
print(f"   Result device: {result.device}")

print("\n" + "=" * 60)
print("Step 5: Check stats")
print("=" * 60)

stats = wrapper_engine.get_stats()
print("Final stats:")
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("Step 6: Cleanup")
print("=" * 60)

wrapper_engine.close()
print("✅ Engine closed")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print("\nNote: Check stderr above for [NativeEngine] debug logs")
