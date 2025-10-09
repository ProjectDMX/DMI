#!/usr/bin/env python3
"""Simple test to verify C++ backend usage with existing compiled module."""

import sys
import torch
from pathlib import Path

print("=" * 60)
print("Testing C++ Backend via Python Wrapper")
print("=" * 60)

# Import the wrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitoring import MonitoringEngine
from monitoring.task import MonitoringTask

# Create engine with async enabled
print("\n1. Creating MonitoringEngine with async_enabled=True...")
engine = MonitoringEngine(
    queue_size=128,
    cache_dtype=torch.float16,
    async_enabled=True,
    delay_steps=0
)

print(f"   Using native backend: {engine._using_native_backend}")
print(f"   Native backend object: {engine._native_backend}")
print(f"   Python backend object: {engine._python_backend}")

if not engine._using_native_backend:
    print("\n⚠️  WARNING: Using Python backend fallback")
    print("   This means C++ module is not being used")
else:
    print("\n✅ Successfully using C++ native backend!")

# Create a test tensor on CUDA
print("\n2. Creating test tensor on CUDA...")
device = torch.device("cuda:0")
test_tensor = torch.randn(4, 8, 64, device=device)
print(f"   Tensor shape: {test_tensor.shape}")

# Create and submit task
print("\n3. Submitting test task...")
task = MonitoringTask(
    name="test_activation",
    tensor=test_tensor,
    step_id=1
)

futures = engine.submit_step(1, [task])
print(f"   Got {len(futures)} futures")

# Get result
print("\n4. Waiting for result...")
result = futures[0].result(timeout=5.0)
print(f"   Result shape: {result.shape}")
print(f"   Result device: {result.device}")

# Check stats
print("\n5. Engine stats:")
stats = engine.get_stats()
for key, value in sorted(stats.items()):
    print(f"   {key}: {value}")

# Cleanup
print("\n6. Closing engine...")
engine.close()

print("\n" + "=" * 60)
print("✅ Test completed!")
print("=" * 60)
print("\nNote: If using C++ backend, check stderr for [NativeEngine] logs")
