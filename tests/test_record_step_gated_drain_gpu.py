"""Step-gated drain: a record ring that flushes only what each step grants.

With ``RingConfig.drain_policy = "step_gated"`` the drain never flushes on its
own. ``RecordRuntime.allow_flush_step()``, called once per step after the step's
producers, records an event on the current stream and grants the drain the
ring records reserved since the previous call; the drain waits on the event,
then flushes them. Forced flushes (oversized records, ``flush_and_wait``) work
as with the default policy.

Build: make -C native SM_ARCH=... PYTHON=...
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]

RING_BYTES = 64 * 1024
RECORD_ELEMENTS = 256  # float32: 1 KiB per record
SETTLE_S = 0.3  # time in which an autonomous drain would have flushed


class _Target:
    """Capture target recording the step number of each delivered record."""

    def __init__(self):
        self.steps: list[int] = []
        self.lock = threading.Lock()

    def _attach(self):
        pass

    def _detach(self):
        pass

    def _submit_capture(self, metadata_json, payload):
        with self.lock:
            self.steps.append(int(json.loads(metadata_json)["step_number"]))

    def _flush_capture(self, timeout_s):
        return True

    def _rethrow_capture(self):
        pass

    def delivered(self) -> list[int]:
        with self.lock:
            return list(self.steps)

    def wait_for(self, count: int, timeout_s: float = 10.0) -> list[int]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            got = self.delivered()
            if len(got) >= count:
                return got
            time.sleep(0.005)
        return self.delivered()


class _HookRuntime:
    def __init__(self, runtime):
        self._runtime = runtime
        self.metadata = None

    def should_emit(self, hook):
        return True

    def prepare_output(self, *, hook, output_index, output_id, output_spec,
                       output):
        from dmi.api.v1 import ProducerPlanBuilder

        entry = ProducerPlanBuilder().record_output(
            output_id=output_id, output_spec=output_spec, output=output)
        return self._runtime.emit_output(entry, self.metadata, output)


def _metadata(step: int, tensor: torch.Tensor):
    from dmi.storage.capture import CaptureMetadata

    return CaptureMetadata(
        capture_id=f"gated-{step:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{step}",
        sequence_id="n", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v",
        hook_name="capture_tensor", layer_number=0, producer_rank=0,
        step_number=step, token_start=step, token_end=step + 1,
        batch_position=0, dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


class _GatedRing:
    def __init__(self, policy: str = "step_gated"):
        from dmi.api.v1 import (
            HookPointV1, HookSpecV1, MonitoringEngine, RingConfig,
            TransportSpec,
        )
        from dmi.storage.capture import CaptureRecordFormat
        from dmi.transport import native

        config = RingConfig()
        config.task_ring_entries = 64
        config.payload_ring_bytes = RING_BYTES
        config.pinned_staging_bytes = RING_BYTES
        config.drain_policy = policy
        self.target = _Target()
        self.engine = MonitoringEngine(model_id="gated-gpu",
                                       ring_config=config)
        sink = native.ReferencePythonCaptureSink(
            self.target, CaptureRecordFormat.LAYOUT_NAME)
        self.runtime = self.engine.create_record_runtime(
            CaptureRecordFormat(), record_sink=sink)
        self.hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        self.hook_runtime = _HookRuntime(self.runtime)
        self.runtime.bind_hook(self.hook, hook_runtime=self.hook_runtime)

    def emit(self, step: int, elements: int = RECORD_ELEMENTS) -> None:
        tensor = torch.full((elements,), float(step), device="cuda")
        self.hook_runtime.metadata = _metadata(step, tensor)
        self.hook(tensor)

    def close(self):
        self.engine.close()


@pytest.fixture
def ring():
    r = _GatedRing()
    yield r
    r.close()


def test_nothing_is_flushed_without_a_grant(ring):
    ring.runtime.begin_step()
    for step in range(3):
        ring.emit(step)
    torch.cuda.synchronize()
    time.sleep(SETTLE_S)
    assert ring.target.delivered() == []
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(3) == [0, 1, 2]


def test_a_grant_covers_exactly_its_step(ring):
    ring.runtime.begin_step()
    for step in range(3):
        ring.emit(step)
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(3) == [0, 1, 2]

    ring.runtime.begin_step()
    for step in range(3, 5):
        ring.emit(step)
    torch.cuda.synchronize()
    time.sleep(SETTLE_S)
    assert ring.target.delivered() == [0, 1, 2]  # step 2 not granted yet
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(5) == [0, 1, 2, 3, 4]


def test_grant_waits_for_the_gpu_without_host_sync(ring):
    # Warm-up: the first emit initializes the eager record path.
    ring.runtime.begin_step()
    ring.emit(100)
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(1) == [100]

    a = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    ring.runtime.begin_step()
    for _ in range(100):  # GPU work queued ahead of the producers
        a = a @ a
        a = a / a.norm()
    ring.emit(0)
    ring.emit(1)
    t0 = time.perf_counter()
    ring.runtime.allow_flush_step()
    returned_ms = (time.perf_counter() - t0) * 1000
    # Neither the producers nor the grant waited for the queued work.
    assert not torch.cuda.current_stream().query()
    assert returned_ms < 50, returned_ms
    assert ring.target.delivered() == [100]
    assert ring.target.wait_for(3, timeout_s=30.0) == [100, 0, 1]


def test_grants_are_delivered_in_order(ring):
    step = 0
    for _ in range(4):
        ring.runtime.begin_step()
        for _ in range(3):
            ring.emit(step)
            step += 1
        ring.runtime.allow_flush_step()
    assert ring.target.wait_for(12) == list(range(12))


def test_empty_step_records_no_grant(ring):
    ring.runtime.begin_step()
    ring.runtime.allow_flush_step()  # nothing reserved: no event, no error
    ring.runtime.begin_step()
    ring.emit(0)
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(1) == [0]


def test_beginning_a_step_with_ungranted_records_raises(ring):
    ring.runtime.begin_step()
    ring.emit(0)
    with pytest.raises(RuntimeError, match="not granted by allow_flush_step"):
        ring.runtime.begin_step()
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(1) == [0]


def test_forced_flush_delivers_ungranted_records_once(ring):
    ring.runtime.begin_step()
    ring.emit(0)
    ring.emit(1)
    ring.engine.flush_and_wait(timeout_s=30.0)
    assert ring.target.delivered() == [0, 1]
    ring.runtime.allow_flush_step()  # its records are already delivered
    ring.runtime.begin_step()
    ring.emit(2)
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(3) == [0, 1, 2]
    time.sleep(SETTLE_S)
    assert ring.target.delivered() == [0, 1, 2]  # no duplicates


def test_oversized_records_bypass_the_ring_in_order(ring):
    oversized = 2 * RING_BYTES // 4  # float32 elements: twice the ring
    ring.runtime.begin_step()
    ring.emit(0)
    ring.emit(1, elements=oversized)  # forced flush, then CPU-direct
    ring.emit(2)
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(3) == [0, 1, 2]

    ring.runtime.begin_step()
    ring.emit(3, elements=oversized)  # the step's only record: no ring record
    ring.runtime.allow_flush_step()
    assert ring.target.wait_for(4) == [0, 1, 2, 3]


def test_allow_flush_requires_the_step_gated_policy():
    r = _GatedRing(policy="autonomous")
    try:
        with pytest.raises(RuntimeError, match="step_gated"):
            r.runtime.allow_flush_step()
    finally:
        r.close()


def test_unknown_policy_is_refused():
    with pytest.raises((ValueError, RuntimeError), match="drain_policy"):
        _GatedRing(policy="sometimes")


@pytest.mark.parametrize("policy", ["autonomous", "step_gated"])
def test_engine_builds_a_ring_only_on_use(policy):
    from dmi.api.v1 import MonitoringEngine, RingConfig

    def config():
        c = RingConfig()
        c.task_ring_entries = 64
        c.payload_ring_bytes = RING_BYTES
        c.pinned_staging_bytes = RING_BYTES
        c.drain_policy = policy
        return c

    # Recording: only the record ring is ever built.
    r = _GatedRing(policy)
    try:
        assert r.engine._ring_transport is not None
        assert r.engine._record_mode
    finally:
        r.close()

    # No ring at construction; a legacy entry point builds the legacy ring,
    # which refuses the step-gated policy.
    engine = MonitoringEngine(model_id="gated-gpu", ring_config=config())
    try:
        assert engine._ring_transport is None
        if policy == "step_gated":
            with pytest.raises(ValueError, match="requires a record ring"):
                engine.ring_capacities()
        else:
            assert engine.ring_capacities().task_entries == 64
            assert engine._ring_transport is not None
            assert not engine._record_mode
    finally:
        engine.close()
