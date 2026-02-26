import time

import pytest
import torch

from monitoring import MonitoringEngine, MonitoringTask
from monitoring.task import BackendFuture


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_native_drained(engine: MonitoringEngine) -> None:
    backend = getattr(engine, "_native_backend", None)
    assert backend is not None
    if hasattr(backend, "debug_state"):
        dbg = backend.debug_state()
        assert int(dbg.get("pending_tasks", -1)) == 0
        assert int(dbg.get("queue_size", -1)) == 0
        assert int(dbg.get("open_steps", -1)) == 0
        assert int(dbg.get("sealed_steps", -1)) == 0
        return
    stats = backend.get_stats()
    assert int(stats.get("inflight_bytes", -1)) == 0
    if "host_copy_queue_depth" in stats:
        assert int(stats.get("host_copy_queue_depth", -1)) == 0


def _wait_native_drain_without_resolve(engine: MonitoringEngine, timeout_s: float = 120.0) -> None:
    backend = getattr(engine, "_native_backend", None)
    assert backend is not None

    deadline = time.perf_counter() + timeout_s
    last_state = None
    use_debug_state = hasattr(backend, "debug_state")
    while time.perf_counter() < deadline:
        if use_debug_state:
            last_state = backend.debug_state()
            pending = int(last_state.get("pending_tasks", -1))
            qsz = int(last_state.get("queue_size", -1))
            open_steps = int(last_state.get("open_steps", -1))
            sealed_steps = int(last_state.get("sealed_steps", -1))
            if pending == 0 and qsz == 0 and open_steps == 0 and sealed_steps == 0:
                return
        else:
            last_state = backend.get_stats()
            inflight = int(last_state.get("inflight_bytes", -1))
            host_q = int(last_state.get("host_copy_queue_depth", 0))
            if inflight == 0 and host_q == 0:
                return
        time.sleep(0.05)

    pytest.fail(f"native backend did not drain without resolve_all: {last_state}")


def test_cpp_future_result_with_end_step_flow() -> None:
    engine = MonitoringEngine(async_enabled=True)
    device = torch.device("cuda")

    backend = getattr(engine, "_native_backend", None)
    assert backend is not None

    try:
        # Stress-scale to thousands of tasks.
        num_steps = 64
        tasks_per_step = 32
        for step_idx in range(num_steps):
            tensors = [
                torch.randn(1, 3, 8, device=device, dtype=torch.float32)
                for _ in range(tasks_per_step)
            ]
            futures = []

            engine.start_step()
            for task_idx, tensor in enumerate(tensors):
                task = MonitoringTask(
                    name=f"blocks.{task_idx}.hook_resid_post",
                    tensor=tensor,
                    metadata={"remove_batch_dim": False, "can_slice": False},
                )
                futures.append(engine.submit(task))
            engine.end_step()

            # Proactively consume half of futures through native C++ API.
            for task_idx in range(0, len(futures), 2):
                token = futures[task_idx]._token
                assert token is not None
                out = backend.future_result(int(token), 30.0)
                assert out.device.type == "cpu"
                assert torch.allclose(
                    out, tensors[task_idx].cpu(), atol=1e-6, rtol=1e-6
                ), f"step={step_idx} task={task_idx}"

            # Consume the rest through normal CacheFuture API.
            for task_idx in range(1, len(futures), 2):
                out = futures[task_idx].result(timeout=30.0)
                assert out.device.type == "cpu"
                assert torch.allclose(
                    out, tensors[task_idx].cpu(), atol=1e-6, rtol=1e-6
                ), f"step={step_idx} task={task_idx}"

        _wait_native_drain_without_resolve(engine)
        _assert_native_drained(engine)
    finally:
        engine.close()


def test_cpp_backendfuture_result_does_not_block_engine_close() -> None:
    engine = MonitoringEngine(async_enabled=True)
    device = torch.device("cuda")

    backend = getattr(engine, "_native_backend", None)
    assert backend is not None

    try:
        pending_cpp_futures = []
        pending_expected = []

        # Produce and consume across many steps to stress token lifecycle.
        # Stress-scale to thousands of explicit C++ BackendFuture.result calls.
        for step_idx in range(2048):
            tensor = torch.randn(1, 4, 16, device=device, dtype=torch.float32)
            engine.start_step()
            task = MonitoringTask(
                name=f"blocks.{step_idx % 12}.hook_resid_mid",
                tensor=tensor,
                metadata={"remove_batch_dim": False, "can_slice": False},
            )
            cache_future = engine.submit(task)
            engine.end_step()

            token = cache_future._token
            assert token is not None
            pending_cpp_futures.append(BackendFuture(backend, int(token)))
            pending_expected.append(tensor.cpu())

            # Keep some backlog and consume part of it.
            if len(pending_cpp_futures) >= 16:
                for _ in range(8):
                    cpp_future = pending_cpp_futures.pop(0)
                    expected = pending_expected.pop(0)
                    out = cpp_future.result(30.0, True)
                    assert out.device.type == "cpu"
                    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)

        while pending_cpp_futures:
            cpp_future = pending_cpp_futures.pop(0)
            expected = pending_expected.pop(0)
            out = cpp_future.result(30.0, True)
            assert out.device.type == "cpu"
            assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)

        _wait_native_drain_without_resolve(engine)
        _assert_native_drained(engine)
    finally:
        engine.close()
