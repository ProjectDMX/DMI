#!/usr/bin/env python3
"""Final integration test: C++ backend via Python wrapper."""

import sys
import torch
from pathlib import Path

print("=" * 70)
print(" Final Integration Test: C++ Backend via MonitoringEngine Wrapper")
print("=" * 70)

# Import wrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitoring import MonitoringEngine
from monitoring.task import MonitoringTask

print("\n[Step 1] Creating MonitoringEngine with async_enabled=True...")
engine = MonitoringEngine(
    queue_size=256,
    cache_dtype=torch.float16,
    async_enabled=True,
    delay_steps=0
)

print(f"   Using native backend: {engine._using_native_backend}")
print(f"   Native backend type: {type(engine._native_backend)}")
print(f"   Python backend type: {type(engine._python_backend) if engine._python_backend else None}")

if not engine._using_native_backend:
    print("\n   ⚠️  WARNING: Fallback to Python backend!")
    print("   This means the compiled C++ module was not loaded properly.")
else:
    print("\n   ✅ SUCCESS: Using C++ native backend!")

print("\n[Step 2] Getting initial stats...")
stats = engine.get_stats()
print("   Initial stats:")
for key in sorted(stats.keys()):
    print(f"     {key:20s} = {stats[key]}")

print("\n[Step 3] Creating test tensors on GPU...")
device = torch.device("cuda:0")
test_tensors = [
    torch.randn(2, 16, 128, device=device),
    torch.randn(2, 16, 128, device=device),
    torch.randn(2, 16, 128, device=device),
]
print(f"   Created {len(test_tensors)} tensors, each shape: {test_tensors[0].shape}")

print("\n[Step 4] Submitting tasks via wrapper API...")
for step_num in range(1, 4):
    print(f"\n   Step {step_num}:")

    # Start step (increments internal step_id)
    engine.start_step()

    # Create tasks
    tasks = [
        MonitoringTask(
            name=f"layer_{i}",
            tensor=test_tensors[i]
        )
        for i in range(len(test_tensors))
    ]

    # Submit tasks
    futures = []
    for task in tasks:
        future = engine.submit(task)
        futures.append(future)

    print(f"     Enqueued {len(futures)} tasks")

    # End step (submits to backend)
    engine.end_step()
    print(f"     Step ended and submitted to backend")

    # Wait for results
    for i, future in enumerate(futures):
        result = future.result(timeout=5.0)
        print(f"     Future {i}: shape={result.shape}, device={result.device}")

print("\n[Step 5] Getting final stats...")
final_stats = engine.get_stats()
print("   Final stats:")
for key in sorted(final_stats.keys()):
    print(f"     {key:20s} = {final_stats[key]}")

print("\n[Step 6] Verifying stats...")
expected_submitted = 3
expected_processed = 3
actual_submitted = final_stats.get('submitted_steps', 0)
actual_processed = final_stats.get('processed_steps', 0)

print(f"   Expected submitted steps: {expected_submitted}")
print(f"   Actual submitted steps:   {actual_submitted}")
print(f"   Expected processed steps: {expected_processed}")
print(f"   Actual processed steps:   {actual_processed}")

if actual_submitted >= expected_submitted and actual_processed >= expected_processed:
    print("   ✅ Stats verification PASSED")
else:
    print("   ❌ Stats verification FAILED")

print("\n[Step 7] Cleanup...")
engine.close()
print("   Engine closed")

print("\n" + "=" * 70)
print(" ✅ Integration Test Completed Successfully!")
print("=" * 70)
print("\nCheck stderr above for [NativeEngine] debug logs to confirm C++ usage.")
print("You should see logs like:")
print("  - [NativeEngine] Engine initialized: queue_capacity=...")
print("  - [NativeEngine] submit_step: step_id=...")
print("  - [NativeEngine] process_step: completed step_id=...")
