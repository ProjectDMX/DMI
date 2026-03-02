import pytest
import torch

from monitoring import MonitoringEngine, MonitoringTask
from monitoring.task import CacheFuture


class _DummyNativeBackend:
    def __init__(self):
        self._next_token = 1
        self._results = {}
        self.partial_seal_args = None

    def begin_step(self, step_id: int, phase_code: int) -> None:
        return None

    def begin_request(self, request_id: int) -> None:
        return None

    def set_capture_schedule(self, *args):
        return None

    def set_partial_seal_config(self, *args):
        self.partial_seal_args = args

    def submit_step(self, step_id, task_specs, stream_handle):
        tokens = []
        for spec in task_specs:
            token = self._next_token
            self._next_token += 1
            tensor = spec[0]
            self._results[token] = tensor.detach().cpu()
            tokens.append(token)
        return tokens

    def seal_step(self, step_id, stream_handle):
        return None

    def resolve_all(self):
        return None

    def clear_completed_results(self):
        return None

    def close(self):
        return None

    def future_ready(self, token: int) -> bool:
        return token in self._results

    def future_wait(self, token: int, timeout=None) -> bool:
        return token in self._results

    def future_result(self, token: int, timeout=None):
        return self._results.pop(token)


def _patch_native_loader(monkeypatch):
    def _loader(*args, **kwargs):
        return _DummyNativeBackend()

    monkeypatch.setattr("monitoring.engine._load_native_backend", _loader)


def test_cache_future_requires_backend_binding():
    task = MonitoringTask(name="test", tensor=torch.ones(2))
    future = CacheFuture(task)

    assert future.ready() is False
    with pytest.raises(RuntimeError, match="not bound"):
        future.result(timeout=0.1)
    with pytest.raises(RuntimeError, match="not bound"):
        future.wait(timeout=0.1)


def test_cache_future_reads_from_native_backend():
    task = MonitoringTask(name="test", tensor=torch.ones(2))
    future = CacheFuture(task)

    backend = _DummyNativeBackend()
    backend._results[7] = torch.tensor([3.0, 4.0])
    future.bind_backend(backend, 7)

    assert future.ready() is True
    out = future.result(timeout=1.0)
    assert torch.equal(out, torch.tensor([3.0, 4.0]))
    # Second call should use cached tensor.
    out2 = future.result(timeout=1.0)
    assert torch.equal(out2, out)


def test_monitoring_engine_rejects_async_disabled():
    with pytest.raises(RuntimeError, match="async_enabled=False"):
        MonitoringEngine(async_enabled=False)


def test_monitoring_engine_requires_native_backend(monkeypatch):
    monkeypatch.setattr("monitoring.engine._load_native_backend", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Failed to initialize native monitoring backend"):
        MonitoringEngine(async_enabled=True)


def test_monitoring_engine_submit_rejects_cpu_tensor(monkeypatch):
    _patch_native_loader(monkeypatch)
    engine = MonitoringEngine(async_enabled=True)
    try:
        task = MonitoringTask(name="blocks.0.hook_resid_pre", tensor=torch.ones(2))
        with pytest.raises(RuntimeError, match="must be CUDA"):
            engine.submit(task)
    finally:
        engine.close()
