"""Tests for fixed pinned-offload behavior in MonitoringEngine."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from monitoring import MonitoringEngine
from monitoring.config import AdvanceConfig, MonitoringConfig
from monitoring.task import MonitoringTask


def _submit_offload(engine: MonitoringEngine, tensor: torch.Tensor, name: str) -> torch.Tensor:
    engine.start_step()
    future = engine.submit(
        MonitoringTask(
            name=name,
            tensor=tensor,
            target_device=torch.device("cpu"),
            metadata={"remove_batch_dim": False, "can_slice": False},
        )
    )
    engine.end_step()
    result = future.result(timeout=10.0)
    engine.resolve_all()
    return result


def _read_vmpin_kb() -> int:
    status = Path("/proc/self/status")
    if not status.exists():
        pytest.skip("VmPin monitoring requires Linux /proc filesystem")
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmPin:"):
            return int(line.split()[1])
    return 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_offload_results_are_pageable() -> None:
    """Results should always end up as pageable CPU tensors."""

    engine = MonitoringEngine(async_enabled=True)
    tensor = torch.randn(128, 512, device="cuda", dtype=torch.float32)

    try:
        result = _submit_offload(engine, tensor, "test.activation")

        assert result.device.type == "cpu"
        assert not result.is_pinned()
        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor.cpu(), atol=1e-5)
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pool_stats_and_host_memcpy_are_non_zero() -> None:
    """Pinned-pool and host memcpy statistics should be active in fixed-on path."""

    cfg = MonitoringConfig(advance=AdvanceConfig(host_copy_threads=2, host_copy_queue_size=128))
    engine = MonitoringEngine(async_enabled=True, config=cfg)

    try:
        for i in range(12):
            tensor = torch.randn(192, 384, device="cuda", dtype=torch.float32)
            result = _submit_offload(engine, tensor, f"test.stats.{i}")
            assert result.device.type == "cpu"
            assert not result.is_pinned()
            assert torch.allclose(result, tensor.cpu(), atol=1e-5)

        backend = engine._native_backend
        assert backend is not None
        stats = backend.get_stats()
        pool_hits = int(stats.get("pool_hits", 0))
        pool_misses = int(stats.get("pool_misses", 0))
        host_memcpy_mb = float(stats.get("host_memcpy_mb", 0.0))

        assert pool_hits + pool_misses > 0
        assert host_memcpy_mb > 0.0
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_host_copy_threads_zero_vs_parallel_consistent() -> None:
    """Both host-copy paths should offload correctly and produce identical results."""

    base = torch.arange(64 * 128, dtype=torch.float32, device="cuda").reshape(64, 128)

    def run_once(host_copy_threads: int) -> list[torch.Tensor]:
        cfg = MonitoringConfig(
            advance=AdvanceConfig(
                host_copy_threads=host_copy_threads,
                host_copy_queue_size=128,
            )
        )
        engine = MonitoringEngine(async_enabled=True, config=cfg)
        outputs: list[torch.Tensor] = []
        try:
            for i in range(5):
                tensor = (base + float(i)).clone()
                result = _submit_offload(engine, tensor, f"test.threads.{host_copy_threads}.{i}")
                assert result.device.type == "cpu"
                assert not result.is_pinned()
                outputs.append(result)
        finally:
            engine.close()
        return outputs

    serial_outputs = run_once(host_copy_threads=0)
    parallel_outputs = run_once(host_copy_threads=2)

    assert len(serial_outputs) == len(parallel_outputs) == 5
    for serial, parallel in zip(serial_outputs, parallel_outputs):
        assert torch.equal(serial, parallel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vmpin_growth_bounded_under_long_run() -> None:
    """VmPin should stay bounded under long-run fixed pinned offload path."""

    engine = MonitoringEngine(async_enabled=True)
    shape = (256, 512)

    try:
        for i in range(4):
            warm = torch.randn(*shape, device="cuda", dtype=torch.float32)
            warm_res = _submit_offload(engine, warm, f"test.vmpin.warmup.{i}")
            assert not warm_res.is_pinned()

        torch.cuda.synchronize()
        vmpin_before_kb = _read_vmpin_kb()

        for i in range(24):
            tensor = torch.randn(*shape, device="cuda", dtype=torch.float32)
            result = _submit_offload(engine, tensor, f"test.vmpin.run.{i}")
            assert result.device.type == "cpu"
            assert not result.is_pinned()

        torch.cuda.synchronize()
        vmpin_after_kb = _read_vmpin_kb()
        vmpin_growth_mb = (vmpin_after_kb - vmpin_before_kb) / 1024.0

        # Allow moderate one-time growth, but reject unbounded increase.
        assert vmpin_growth_mb < 64.0, f"VmPin growth too high: {vmpin_growth_mb:.2f} MB"
    finally:
        engine.close()
