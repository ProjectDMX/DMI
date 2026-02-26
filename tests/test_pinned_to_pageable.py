"""Tests for fixed pinned-offload behavior in MonitoringEngine."""

from __future__ import annotations

import pytest
import torch

from monitoring import MonitoringEngine
from monitoring.config import AdvanceConfig, MonitoringConfig
from monitoring.task import MonitoringTask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_offload_results_are_pageable() -> None:
    """Results should always end up as pageable CPU tensors."""

    engine = MonitoringEngine(async_enabled=True)
    tensor = torch.randn(128, 512, device="cuda", dtype=torch.float32)

    engine.start_step()
    task = MonitoringTask(
        name="test.activation",
        tensor=tensor,
        target_device=torch.device("cpu"),
        metadata={"remove_batch_dim": False, "can_slice": False},
    )
    future = engine.submit(task)
    engine.end_step()

    try:
        result = future.result(timeout=5.0)
        engine.resolve_all()

        assert result.device.type == "cpu"
        assert not result.is_pinned()
        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor.cpu(), atol=1e-5)
    finally:
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_advance_config_is_applied() -> None:
    """Advanced runtime config should be accepted and produce valid stats."""

    cfg = MonitoringConfig(
        advance=AdvanceConfig(
            pinpool_bins_kb=(512, 1024, 2048),
            pinpool_max_mb=128,
            host_copy_threads=2,
            host_copy_queue_size=128,
        )
    )
    engine = MonitoringEngine(async_enabled=True, config=cfg)

    try:
        tensor = torch.randn(64, 256, device="cuda", dtype=torch.float32)
        engine.start_step()
        future = engine.submit(
            MonitoringTask(
                name="test.advance",
                tensor=tensor,
                target_device=torch.device("cpu"),
                metadata={"remove_batch_dim": False, "can_slice": False},
            )
        )
        engine.end_step()
        result = future.result(timeout=5.0)
        assert result.device.type == "cpu"
        assert not result.is_pinned()

        backend = engine._native_backend
        assert backend is not None
        stats = backend.get_stats()
        assert "pool_hits" in stats
        assert "pool_misses" in stats
        assert "pool_high_watermark_bytes" in stats
        assert "host_memcpy_mb" in stats
    finally:
        engine.close()
