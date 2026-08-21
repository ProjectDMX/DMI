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


# The producer op receives (ring_payload, tensor, strip_tensor, strip_row_bytes,
# hook_type, hook_id).  Pinning the identity values here lets tests assert the
# whole tuple instead of just the call count -- a swapped hook_type/hook_id or a
# wrong payload handle mis-associates activations with hooks and raises nothing.
HOOK_TYPE = 7
HOOK_ID = 3
PAYLOAD = object()
# HookPoint defaults for a hook with no strip configured (hook_points.py:174).
NO_STRIP = (None, 0)


def expected_dispatch(tensor):
    return (PAYLOAD, tensor, *NO_STRIP, HOOK_TYPE, HOOK_ID)


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
        tasks: int = 1024,
    ) -> None:
        self.available = available
        self.payload = payload
        self.staging = staging
        self.available_after_flush = (
            payload if available_after_flush is None else available_after_flush
        )
        self.tasks = tasks
        self.tasks_used = 0
        self.overdrafts = []
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

    def task_cap(self) -> int:
        self.calls.append(("task_cap",))
        return self.tasks

    def reserve_one(self, nbytes: int) -> None:
        # DrainThread::reserve just advances the head counters; it never
        # raises.  An over-reservation is silent ring corruption in the real
        # engine, so record it here rather than turning it into an exception
        # the production path would never see.
        self.calls.append(("reserve_one", nbytes))
        if nbytes > self.available:
            self.overdrafts.append(nbytes)
        self.available -= nbytes
        self.tasks_used += 1

    def flush_and_wait(self) -> None:
        self.calls.append(("flush_and_wait",))
        self.available = self.available_after_flush
        self.tasks_used = 0


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
    hp._ring_hook_type = HOOK_TYPE
    hp._ring_hook_id = HOOK_ID
    hp._ring_payload = PAYLOAD
    tensor = _FakeCudaTensor(nbytes)
    result = hp(tensor)
    assert result is tensor
    return tensor, transport


@pytest.mark.parametrize("nbytes", [0, 1, 64])
def test_tensor_fitting_current_slack_reserves_then_dispatches(dispatch, nbytes):
    engine = _CapacityEngine(available=64, payload=128, staging=128)

    tensor, transport = _run(nbytes, engine)

    assert ("reserve_one", nbytes) in engine.calls
    assert ("flush_and_wait",) not in engine.calls
    assert dispatch == [expected_dispatch(tensor)]
    assert transport.direct == []


def test_exact_effective_capacity_is_accepted(dispatch):
    engine = _CapacityEngine(available=64, payload=64, staging=64)

    tensor, transport = _run(64, engine)

    assert engine.calls[-1] == ("reserve_one", 64)
    assert dispatch == [expected_dispatch(tensor)]
    assert transport.direct == []


def test_flushes_before_reserving_when_only_current_slack_is_insufficient(dispatch):
    engine = _CapacityEngine(
        available=15,
        payload=64,
        staging=64,
        available_after_flush=64,
    )

    tensor, transport = _run(16, engine)

    assert engine.calls.index(("flush_and_wait",)) < engine.calls.index(
        ("reserve_one", 16)
    )
    assert dispatch == [expected_dispatch(tensor)]
    assert transport.direct == []


def test_tensor_larger_than_payload_uses_cpu_direct_without_reservation(dispatch):
    engine = _CapacityEngine(available=8, payload=32, staging=64)

    tensor, transport = _run(33, engine)

    assert ("flush_and_wait",) in engine.calls
    assert not any(call[0] == "reserve_one" for call in engine.calls)
    assert dispatch == []
    assert tensor.cpu_calls == 1
    assert transport.direct == [(('cpu', 33), HOOK_TYPE, HOOK_ID)]


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
    assert transport.direct == [(('cpu', nbytes), HOOK_TYPE, HOOK_ID)]


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
    assert transport.direct == [(('cpu', 32), HOOK_TYPE, HOOK_ID)]


def test_disabled_hook_is_a_true_noop(dispatch):
    engine = _CapacityEngine(available=64, payload=64, staging=64)
    transport = _Transport(engine)
    ring_transport._active_transport = transport
    hp = hook_points.HookPoint()
    hp.enabled = False
    hp._ring_hook_type = HOOK_TYPE
    hp._ring_hook_id = HOOK_ID
    hp._ring_payload = PAYLOAD
    tensor = _FakeCudaTensor(16)

    assert hp(tensor) is tensor
    assert engine.calls == []
    assert dispatch == []
    assert transport.direct == []


# ---------------------------------------------------------------------------
# Early-return branches of HookPoint.forward.  None of these touch the ring
# engine, so a capacity call appearing here is itself the regression.
# ---------------------------------------------------------------------------


def test_non_eager_transport_skips_capacity_checks_entirely(dispatch):
    engine = _CapacityEngine(available=0, payload=0, staging=0)
    transport = _Transport(engine)
    transport.force_eager = False
    ring_transport._active_transport = transport
    hp = hook_points.HookPoint()
    hp._ring_hook_type = HOOK_TYPE
    hp._ring_hook_id = HOOK_ID
    hp._ring_payload = PAYLOAD
    tensor = _FakeCudaTensor(1 << 30)

    assert hp(tensor) is tensor
    assert engine.calls == []
    assert len(dispatch) == 1
    assert transport.direct == []


def test_no_active_transport_still_dispatches(dispatch):
    ring_transport._active_transport = None
    hp = hook_points.HookPoint()
    hp._ring_hook_type = HOOK_TYPE
    hp._ring_hook_id = HOOK_ID
    hp._ring_payload = PAYLOAD
    tensor = _FakeCudaTensor(1 << 30)

    assert hp(tensor) is tensor
    assert len(dispatch) == 1


def test_uninstalled_hook_is_a_noop(dispatch):
    engine = _CapacityEngine(available=64, payload=64, staging=64)
    ring_transport._active_transport = _Transport(engine)
    hp = hook_points.HookPoint()
    tensor = _FakeCudaTensor(16)

    assert hp(tensor) is tensor
    assert engine.calls == []
    assert dispatch == []


def test_cpu_tensor_is_a_noop(dispatch):
    engine = _CapacityEngine(available=64, payload=64, staging=64)
    ring_transport._active_transport = _Transport(engine)
    hp = hook_points.HookPoint()
    hp._ring_hook_type = HOOK_TYPE
    hp._ring_hook_id = HOOK_ID
    hp._ring_payload = PAYLOAD
    tensor = _FakeCudaTensor(16)
    tensor.is_cuda = False

    assert hp(tensor) is tensor
    assert engine.calls == []
    assert dispatch == []


# ---------------------------------------------------------------------------
# Task-ring capacity.  prepare_step() checks payload AND task entries
# (`num_hooks > tcap` -> STEP_OVERSIZED, `num_hooks <= task_avail` -> fast
# path), and reserve_one() claims one task entry per call.  The eager net --
# which exists to service the steps prepare_step rejected -- only ever looks
# at payload bytes.
# ---------------------------------------------------------------------------


def _run_repeatedly(nbytes: int, engine: _CapacityEngine, times: int):
    transport = _Transport(engine)
    ring_transport._active_transport = transport
    for _ in range(times):
        hp = hook_points.HookPoint()
        hp._ring_hook_type = HOOK_TYPE
        hp._ring_hook_id = HOOK_ID
        hp._ring_payload = PAYLOAD
        hp(_FakeCudaTensor(nbytes))
    return transport


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug: the eager net never consults task_cap(), so successive "
        "hooks keep calling reserve_one() past the task ring's capacity.  "
        "task_cap() is bound to Python right next to available_capacity() and "
        "reserve_one() as part of the documented safety-net surface."
    ),
)
def test_task_capacity_is_respected_across_successive_hooks(dispatch):
    engine = _CapacityEngine(available=1024, payload=1024, staging=1024, tasks=2)

    _run_repeatedly(8, engine, times=3)

    assert ("task_cap",) in engine.calls
    assert engine.tasks_used <= 2


def test_payload_overdraft_never_happens_on_the_reserve_paths(dispatch):
    """Both reserving branches must leave the ring accounting consistent."""
    engine = _CapacityEngine(available=64, payload=64, staging=64)

    # Six 16-byte hooks against a 64-byte ring: the first four consume the
    # initial slack, the fifth takes the flush-then-reserve branch.
    _run_repeatedly(16, engine, times=6)

    assert ("flush_and_wait",) in engine.calls
    assert engine.overdrafts == []
