import threading
import time

import pytest
import torch

from monitoring import MonitoringEngine, MonitoringTask
from monitoring.task import CacheFuture

try:
    from transformer_lens.utils import Slice
except Exception:  # pragma: no cover
    Slice = None


def test_cache_future_waits_for_result():
    tensor = torch.ones(2)
    task = MonitoringTask(name="test", tensor=tensor)
    future = CacheFuture(task)

    def producer():
        time.sleep(0.05)
        future.set_result(tensor * 2)

    thread = threading.Thread(target=producer)
    thread.start()

    assert not future.ready()
    result = future.result(timeout=1.0)
    assert future.ready()
    assert torch.equal(result, tensor * 2)

    thread.join()


def test_monitoring_engine_sync_fallback():
    engine = MonitoringEngine(async_enabled=False)
    tensor = torch.arange(6.0).view(1, 6)
    task = MonitoringTask(
        name="blocks.0.hook_resid_pre",
        tensor=tensor,
        metadata={"remove_batch_dim": True, "can_slice": True},
    )

    future = engine.submit(task)
    engine.end_step()
    assert future.ready()
    result = future.result()
    assert torch.equal(result, tensor[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for async monitoring test")
def test_monitoring_engine_async_cuda():
    engine = MonitoringEngine(async_enabled=True, cache_dtype=torch.float16)

    tensor = torch.arange(12, dtype=torch.float32, device="cuda").view(1, 3, 4)
    slice_obj = Slice(None) if Slice is not None else None

    engine.start_step()
    task = MonitoringTask(
        name="blocks.0.hook_resid_mid",
        tensor=tensor,
        pos_slice=slice_obj,
        slice_dim=-2,
        metadata={"remove_batch_dim": True, "can_slice": True},
        target_device=tensor.device,
    )

    future = engine.submit(task)
    engine.end_step()
    try:
        assert not future.ready()
        result = future.result(timeout=5.0)
        engine.resolve_all()
        assert future.ready()
        assert result.dtype == torch.float16
        assert torch.allclose(result.float(), tensor[0].float(), atol=1e-3, rtol=1e-3)
    finally:
        engine.close()
