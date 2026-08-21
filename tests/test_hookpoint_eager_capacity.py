"""Boundary tests for HookPoint's eager capacity safety net.

The doubles model payload slack and pinned-staging capacity independently so
each test owns its complete state and can be run in any order.
"""
from __future__ import annotations

import pytest

try:
    from monitoring import hook_points, ring_transport
    _NATIVE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on build environment
    hook_points = None
    ring_transport = None
    _NATIVE_IMPORT_ERROR = exc


pytestmark = [
    pytest.mark.native_backend,
    pytest.mark.skipif(
        _NATIVE_IMPORT_ERROR is not None,
        reason=f"DMI native backend required: {_NATIVE_IMPORT_ERROR}",
    ),
]


class _FakeCudaTensor:
    is_cuda = True

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self.cpu_calls = 0

    def contiguous(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return ("cpu", self.nbytes)


class _CapacityEngine:
    def __init__(
        self,
        *,
        available: int,
        payload: int,
        staging: int,
        available_after_flush: int | None = None,
    ) -> None:
        self.available = available
        self.payload = payload
        self.staging = staging
        self.available_after_flush = (
            payload if available_after_flush is None else available_after_flush
        )
        self.calls = []

    def available_capacity(self) -> int:
        self.calls.append(("available_capacity",))
        return self.available

    def payload_cap(self) -> int:
        self.calls.append(("payload_cap",))
        return self.payload

    def staging_cap(self) -> int:
        self.calls.append(("staging_cap",))
        return self.staging

    def reserve_one(self, nbytes: int) -> None:
        self.calls.append(("reserve_one", nbytes))
        if nbytes > self.available:
            raise RuntimeError("reservation exceeds available capacity")
        self.available -= nbytes

    def flush_and_wait(self) -> None:
        self.calls.append(("flush_and_wait",))
        self.available = self.available_after_flush


class _Transport:
    def __init__(self, engine: _CapacityEngine) -> None:
        self._ring_engine = engine
        self.force_eager = True
        self.direct = []

    def submit_cpu_direct(self, tensor, hook_type: int, hook_id: int) -> None:
        self.direct.append((tensor, hook_type, hook_id))


@pytest.fixture
def dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hook_points,
        "_dispatch_producer",
        lambda *args: calls.append(args),
    )
    return calls


@pytest.fixture(autouse=True)
def restore_active_transport():
    previous = ring_transport._active_transport
    try:
        yield
    finally:
        ring_transport._active_transport = previous


def _run(nbytes: int, engine: _CapacityEngine):
    transport = _Transport(engine)
    ring_transport._active_transport = transport
    hp = hook_points.HookPoint()
    hp._ring_hook_type = 7
    hp._ring_hook_id = 3
    hp._ring_payload = object()
    tensor = _FakeCudaTensor(nbytes)
    result = hp(tensor)
    assert result is tensor
    return tensor, transport


@pytest.mark.parametrize("nbytes", [0, 1, 64])
def test_tensor_fitting_current_slack_reserves_then_dispatches(dispatch, nbytes):
    engine = _CapacityEngine(available=64, payload=128, staging=128)

    _tensor, transport = _run(nbytes, engine)

    assert ("reserve_one", nbytes) in engine.calls
    assert ("flush_and_wait",) not in engine.calls
    assert len(dispatch) == 1
    assert transport.direct == []


def test_exact_effective_capacity_is_accepted(dispatch):
    engine = _CapacityEngine(available=64, payload=64, staging=64)

    _tensor, transport = _run(64, engine)

    assert engine.calls[-1] == ("reserve_one", 64)
    assert len(dispatch) == 1
    assert transport.direct == []


def test_flushes_before_reserving_when_only_current_slack_is_insufficient(dispatch):
    engine = _CapacityEngine(
        available=15,
        payload=64,
        staging=64,
        available_after_flush=64,
    )

    _tensor, transport = _run(16, engine)

    assert engine.calls.index(("flush_and_wait",)) < engine.calls.index(
        ("reserve_one", 16)
    )
    assert len(dispatch) == 1
    assert transport.direct == []


def test_tensor_larger_than_payload_uses_cpu_direct_without_reservation(dispatch):
    engine = _CapacityEngine(available=8, payload=32, staging=64)

    tensor, transport = _run(33, engine)

    assert ("flush_and_wait",) in engine.calls
    assert not any(call[0] == "reserve_one" for call in engine.calls)
    assert dispatch == []
    assert tensor.cpu_calls == 1
    assert transport.direct == [(('cpu', 33), 7, 3)]


@pytest.mark.parametrize(
    ("nbytes", "payload", "staging"),
    [
        (33, 64, 32),
        (65, 128, 64),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason="known bug: eager fallback ignores pinned-staging capacity",
)
def test_tensor_larger_than_staging_uses_cpu_direct(
    dispatch, nbytes, payload, staging
):
    engine = _CapacityEngine(
        available=payload,
        payload=payload,
        staging=staging,
    )

    tensor, transport = _run(nbytes, engine)

    assert not any(call[0] == "reserve_one" for call in engine.calls)
    assert dispatch == []
    assert tensor.cpu_calls == 1
    assert transport.direct == [(('cpu', nbytes), 7, 3)]


@pytest.mark.xfail(
    strict=True,
    reason="known bug: eager fallback reserves without rechecking after flush",
)
def test_failed_post_flush_capacity_check_falls_back_to_cpu_direct(dispatch):
    engine = _CapacityEngine(
        available=8,
        payload=64,
        staging=64,
        available_after_flush=16,
    )

    tensor, transport = _run(32, engine)

    assert not any(call[0] == "reserve_one" for call in engine.calls)
    assert dispatch == []
    assert tensor.cpu_calls == 1
    assert transport.direct == [(('cpu', 32), 7, 3)]


def test_disabled_hook_is_a_true_noop(dispatch):
    engine = _CapacityEngine(available=64, payload=64, staging=64)
    transport = _Transport(engine)
    ring_transport._active_transport = transport
    hp = hook_points.HookPoint()
    hp.enabled = False
    hp._ring_hook_type = 7
    hp._ring_hook_id = 3
    hp._ring_payload = object()
    tensor = _FakeCudaTensor(16)

    assert hp(tensor) is tensor
    assert engine.calls == []
    assert dispatch == []
    assert transport.direct == []
