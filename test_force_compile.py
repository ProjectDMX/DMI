#!/usr/bin/env python3
"""Force recompile and test C++ backend."""

import sys
import shutil
import torch
from pathlib import Path

# Clean cache to force recompile
cache_dir = Path.home() / ".cache" / "torch_extensions"
backend_dirs = list(cache_dir.glob("*/monitoring_native_backend"))
print(f"Found {len(backend_dirs)} cached backend directories")
for d in backend_dirs:
    print(f"  Removing: {d}")
    shutil.rmtree(d, ignore_errors=True)

print("\n" + "=" * 60)
print("Testing C++ Backend with Forced Recompile")
print("=" * 60)

# Now import - should trigger recompile with debug flags
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("\n1. Loading native backend (will recompile)...")
from monitoring._native_engine import create_engine

print("\n2. Creating engine...")
engine = create_engine(
    queue_size=128,
    cache_dtype=torch.float16,
    delay_steps=0
)

print(f"   Engine: {engine}")

print("\n3. Getting stats...")
stats = engine.get_stats()
for key, value in sorted(stats.items()):
    print(f"   {key}: {value}")

print("\n4. Testing via MonitoringEngine wrapper...")
from monitoring import MonitoringEngine
from monitoring.task import MonitoringTask

wrapper = MonitoringEngine(
    queue_size=128,
    cache_dtype=torch.float16,
    async_enabled=True,
    delay_steps=0
)

print(f"   Using native: {wrapper._using_native_backend}")
print(f"   Backend: {wrapper._native_backend}")

# Submit a test step
device = torch.device("cuda:0")
test_tensor = torch.randn(4, 8, 64, device=device)

task = MonitoringTask(
    name="test",
    tensor=test_tensor,
    step_id=1
)

print("\n5. Submitting step...")
wrapper.start_step(1)
future = wrapper.enqueue(task)
result = wrapper.finalize_step(wait=True)

print(f"   Result: {result}")

print("\n6. Final stats:")
final_stats = wrapper.get_stats()
for key, value in sorted(final_stats.items()):
    print(f"   {key}: {value}")

wrapper.close()

print("\n" + "=" * 60)
print("✅ Test completed!")
print("=" * 60)
print("\nCheck stderr above for [NativeEngine] debug logs")
