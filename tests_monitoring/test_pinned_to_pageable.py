"""
Test suite for pinned→pageable memory conversion in MonitoringEngine.

Verifies that:
1. When pinpool=0 (disabled), pinned tensors are converted to pageable
2. When pinpool=1 (enabled), pinned tensors are converted to pageable
3. Final results are always pageable (not pinned)
4. VmPin memory doesn't leak over multiple steps
"""

import os

import pytest
import torch

from monitoring import MonitoringEngine
from monitoring.task import MonitoringTask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_to_pageable_no_pool():
    """Test that results are pageable when pinpool is disabled."""
    # Configure environment
    os.environ["MON_NATIVE_TO_CPU"] = "1"
    os.environ["MON_NATIVE_PINNED"] = "1"
    os.environ["MON_NATIVE_PINPOOL"] = "0"  # Disable pool

    engine = MonitoringEngine(async_enabled=True)

    # Create CUDA tensor
    tensor = torch.randn(128, 512, device="cuda", dtype=torch.float32)

    engine.start_step()
    task = MonitoringTask(
        name="test.activation",
        tensor=tensor,
        target_device=torch.device("cpu"),  # Request CPU offload
        metadata={"remove_batch_dim": False, "can_slice": False},
    )

    future = engine.submit(task)
    engine.end_step()

    try:
        # Wait for async processing
        result = future.result(timeout=5.0)
        engine.resolve_all()

        # Assert: result is on CPU and pageable
        assert result.device.type == "cpu", f"Result should be on CPU, got {result.device}"
        assert not result.is_pinned(), f"Result should be pageable, but is_pinned={result.is_pinned()}"

        # Verify data correctness
        assert result.shape == tensor.shape, f"Shape mismatch: {result.shape} vs {tensor.shape}"
        assert torch.allclose(result, tensor.cpu(), atol=1e-5), "Data mismatch"

        print(f"✓ Pool disabled: result is pageable (is_pinned={result.is_pinned()})")
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_to_pageable_with_pool():
    """Test that results are pageable when pinpool is enabled."""
    # Configure environment
    os.environ["MON_NATIVE_TO_CPU"] = "1"
    os.environ["MON_NATIVE_PINNED"] = "1"
    os.environ["MON_NATIVE_PINPOOL"] = "1"  # Enable pool

    engine = MonitoringEngine(async_enabled=True)

    tensor = torch.randn(256, 768, device="cuda", dtype=torch.float32)

    engine.start_step()
    task = MonitoringTask(
        name="test.pooled",
        tensor=tensor,
        target_device=torch.device("cpu"),
        metadata={"remove_batch_dim": False, "can_slice": False},
    )

    future = engine.submit(task)
    engine.end_step()

    try:
        result = future.result(timeout=5.0)
        engine.resolve_all()

        # Key assertion: even with pool, final result should be pageable
        assert result.device.type == "cpu", f"Result should be on CPU, got {result.device}"
        assert not result.is_pinned(), f"Pool-backed results must be converted to pageable (is_pinned={result.is_pinned()})"
        assert torch.allclose(result, tensor.cpu(), atol=1e-5), "Data mismatch"

        # Check pool statistics (if available)
        try:
            backend = engine._native_backend
            if backend is not None:
                stats = backend.get_stats()
                print(f"✓ Pool enabled: result is pageable (is_pinned={result.is_pinned()})")
                if "pool_hits" in stats:
                    print(f"  Pool hits: {stats.get('pool_hits', 0)}")
                if "host_memcpy_mb" in stats:
                    print(f"  Host memcpy: {stats.get('host_memcpy_mb', 0):.2f} MB")
        except Exception:
            pass
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_pinned_memory_leak():
    """Test that pinned memory doesn't accumulate over multiple steps."""
    # Configure environment
    os.environ["MON_NATIVE_TO_CPU"] = "1"
    os.environ["MON_NATIVE_PINNED"] = "1"
    os.environ["MON_NATIVE_PINPOOL"] = "1"

    def get_vmpin_kb():
        """Parse /proc/self/status for VmPin (Linux only)."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmPin:"):
                        return int(line.split()[1])  # KB
        except (FileNotFoundError, ValueError):
            pytest.skip("VmPin monitoring requires Linux /proc filesystem")
        return 0

    engine = MonitoringEngine(async_enabled=True)

    # Warmup
    for _ in range(3):
        tensor = torch.randn(100, 200, device="cuda")
        engine.start_step()
        task = MonitoringTask(
            name="warmup",
            tensor=tensor,
            target_device=torch.device("cpu"),
            metadata={"remove_batch_dim": False, "can_slice": False},
        )
        future = engine.submit(task)
        engine.end_step()
        result = future.result(timeout=5.0)
        assert not result.is_pinned(), f"Warmup result should be pageable"

    engine.resolve_all()
    torch.cuda.synchronize()

    # Record baseline VmPin
    vmpin_before = get_vmpin_kb()
    print(f"VmPin baseline: {vmpin_before} KB ({vmpin_before / 1024:.2f} MB)")

    # Run multiple steps
    results = []
    for i in range(20):
        tensor = torch.randn(500, 512, device="cuda")
        engine.start_step()
        task = MonitoringTask(
            name=f"step_{i}",
            tensor=tensor,
            target_device=torch.device("cpu"),
            metadata={"remove_batch_dim": False, "can_slice": False},
        )
        future = engine.submit(task)
        engine.end_step()
        result = future.result(timeout=5.0)
        assert not result.is_pinned(), f"Step {i}: result is pinned (should be pageable)"
        results.append(result)

    engine.resolve_all()
    torch.cuda.synchronize()

    # Check VmPin didn't grow significantly
    vmpin_after = get_vmpin_kb()
    vmpin_growth_mb = (vmpin_after - vmpin_before) / 1024.0

    print(f"VmPin after: {vmpin_after} KB ({vmpin_after / 1024:.2f} MB)")
    print(f"VmPin growth: {vmpin_growth_mb:.2f} MB")

    # Allow small growth (pool preallocation), but not continuous growth
    assert vmpin_growth_mb < 50.0, f"VmPin leaked {vmpin_growth_mb:.2f} MB (should be < 50 MB)"

    print(f"✓ No memory leak: VmPin growth = {vmpin_growth_mb:.2f} MB (< 50 MB threshold)")

    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pool_statistics():
    """Compare pool hits/misses when pool is enabled vs disabled."""

    def run_with_pool(enable_pool: bool):
        os.environ["MON_NATIVE_TO_CPU"] = "1"
        os.environ["MON_NATIVE_PINNED"] = "1"
        os.environ["MON_NATIVE_PINPOOL"] = "1" if enable_pool else "0"

        engine = MonitoringEngine(async_enabled=True)

        for i in range(10):
            tensor = torch.randn(128, 256, device="cuda")
            engine.start_step()
            task = MonitoringTask(
                name=f"test_{i}",
                tensor=tensor,
                target_device=torch.device("cpu"),
                metadata={"remove_batch_dim": False, "can_slice": False},
            )
            future = engine.submit(task)
            engine.end_step()
            result = future.result(timeout=5.0)
            assert not result.is_pinned(), f"Iteration {i}: result should be pageable"

        engine.resolve_all()
        stats = {}
        try:
            backend = engine._native_backend
            if backend is not None:
                stats = backend.get_stats()
        except Exception:
            pass
        engine.close()
        return stats

    # Pool enabled
    stats_pooled = run_with_pool(True)
    print(f"With pool: {stats_pooled}")

    # Pool disabled
    stats_no_pool = run_with_pool(False)
    print(f"No pool: {stats_no_pool}")

    # Verify: pool enabled should have hits statistics
    if "pool_hits" in stats_pooled:
        assert stats_pooled["pool_hits"] > 0, "Pool should have hits when enabled"
        print(f"✓ Pool hits: {stats_pooled['pool_hits']}")

    # Both cases should have host memcpy (pinned→pageable conversion)
    if "host_memcpy_mb" in stats_pooled:
        assert stats_pooled["host_memcpy_mb"] > 0, "Should have host memcpy with pool"
        print(f"✓ Host memcpy (pool): {stats_pooled['host_memcpy_mb']:.2f} MB")

    if "host_memcpy_mb" in stats_no_pool:
        assert stats_no_pool["host_memcpy_mb"] > 0, "Should have host memcpy without pool"
        print(f"✓ Host memcpy (no pool): {stats_no_pool['host_memcpy_mb']:.2f} MB")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_disabled_returns_pageable():
    """Test that when pinned memory is disabled, results are still pageable."""
    # Configure: no pinned memory
    os.environ["MON_NATIVE_TO_CPU"] = "1"
    os.environ["MON_NATIVE_PINNED"] = "0"  # Disable pinned

    engine = MonitoringEngine(async_enabled=True)

    tensor = torch.randn(64, 128, device="cuda", dtype=torch.float32)

    engine.start_step()
    task = MonitoringTask(
        name="test.no_pinned",
        tensor=tensor,
        target_device=torch.device("cpu"),
        metadata={"remove_batch_dim": False, "can_slice": False},
    )

    future = engine.submit(task)
    engine.end_step()

    try:
        result = future.result(timeout=5.0)
        engine.resolve_all()

        # Should be pageable (obviously, since we didn't use pinned)
        assert result.device.type == "cpu"
        assert not result.is_pinned(), "Result should be pageable when pinned disabled"
        assert torch.allclose(result, tensor.cpu(), atol=1e-5)

        print(f"✓ Pinned disabled: result is pageable (is_pinned={result.is_pinned()})")
    finally:
        engine.close()


if __name__ == "__main__":
    # Allow running individual tests
    import sys
    if len(sys.argv) > 1:
        pytest.main([__file__, "-v", "-s", "-k", sys.argv[1]])
    else:
        pytest.main([__file__, "-v", "-s"])
