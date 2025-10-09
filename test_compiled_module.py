#!/usr/bin/env python3
"""Test the compiled monitoring_native_backend module directly."""

import sys
import torch
from pathlib import Path

print("=" * 60)
print("Testing Compiled C++ Backend Module")
print("=" * 60)

# Add current directory to path so we can import the .so file
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("\n1. Importing monitoring_native_backend...")
try:
    import monitoring_native_backend
    print(f"   ✅ Module imported: {monitoring_native_backend}")
    print(f"   Has create_engine: {hasattr(monitoring_native_backend, 'create_engine')}")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Creating engine directly...")
try:
    engine = monitoring_native_backend.create_engine(
        queue_size=128,
        cache_dtype=torch.float16,
        delay_steps=0
    )
    print(f"   ✅ Engine created: {engine}")
except Exception as e:
    print(f"   ❌ Creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Getting stats...")
try:
    stats = engine.get_stats()
    print("   Stats:")
    for key, value in sorted(stats.items()):
        print(f"     {key}: {value}")
except Exception as e:
    print(f"   ❌ get_stats failed: {e}")

print("\n4. Testing submit_step...")
try:
    device = torch.device("cuda:0")
    test_tensor = torch.randn(4, 8, 64, device=device)
    print(f"   Test tensor shape: {test_tensor.shape}")

    # Create a dummy task (dict with ALL required fields from C++)
    tasks = [{
        'tensor': test_tensor,
        'slice_dim': -2,
        'remove_batch_dim': False,
        'can_slice': True,
        'slice': {
            'mode': 'identity',  # SliceMode::Identity
            'int_value': 0,
            'start': None,
            'stop': None,
            'step': None,
            'indices': []
        },
        'target_device': torch.device('cpu')
    }]

    # Get CUDA stream handle
    stream = torch.cuda.current_stream(device)
    stream_handle = stream.cuda_stream

    print(f"   Submitting step 1 with {len(tasks)} tasks...")
    tokens = engine.submit_step(1, tasks, stream_handle)
    print(f"   ✅ Got {len(tokens)} tokens: {tokens}")

    # Check stats again
    stats = engine.get_stats()
    print("\n   Updated stats:")
    for key, value in sorted(stats.items()):
        print(f"     {key}: {value}")

except Exception as e:
    print(f"   ❌ submit_step failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Direct C++ module test completed!")
print("=" * 60)
print("\nIf you see [NativeEngine] logs above, debug logging is working!")
