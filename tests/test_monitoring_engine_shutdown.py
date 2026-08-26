"""Checked completion remains separate from best-effort engine close."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmi.engine import MonitoringEngine

pytestmark = pytest.mark.cpu


def test_flush_and_wait_preserves_transport_async_error():
    calls = []

    class _Transport:
        def flush_records_and_wait(self, timeout_s):
            calls.append(timeout_s)
            raise RuntimeError("record insert failed")

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = _Transport()
    engine._record_mode = True

    with pytest.raises(RuntimeError, match="record insert failed"):
        engine.flush_and_wait(4.5)
    assert calls == [4.5]


def test_ring_transport_maps_native_timeout_to_timeout_error():
    from dmi.transport.ring import RingTransport

    transport = RingTransport.__new__(RingTransport)
    transport._ring_engine = SimpleNamespace(
        flush_records_and_wait=lambda timeout_s: False
    )

    with pytest.raises(TimeoutError, match="record ring completion"):
        transport.flush_records_and_wait(0.25)


def test_flush_and_wait_rejects_nonpositive_deadline():
    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = SimpleNamespace()
    engine._record_mode = True
    with pytest.raises(ValueError, match="must be positive"):
        engine.flush_and_wait(0)


def test_flush_and_wait_uses_remaining_deadline_for_host_durability(monkeypatch):
    calls = []
    monotonic_times = iter((100.0, 101.75, 102.0))
    monkeypatch.setattr(
        "dmi.engine.time.monotonic", lambda: next(monotonic_times)
    )

    class _Transport:
        def flush_records_and_wait(self, timeout_s):
            calls.append(("ring", timeout_s))
            return True

    class _Host:
        def flush_and_wait(self, timeout_s):
            calls.append(("host", timeout_s))
            return True

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = _Transport()
    engine._host_engine = _Host()
    engine._record_mode = True

    engine.flush_and_wait(5.0)

    assert calls[0] == ("ring", 5.0)
    assert calls[1][0] == "host"
    assert calls[1][1] == pytest.approx(3.25)


def test_flush_and_wait_rejects_host_success_after_caller_deadline(monkeypatch):
    monotonic_times = iter((100.0, 101.0, 105.01))
    monkeypatch.setattr(
        "dmi.engine.time.monotonic", lambda: next(monotonic_times)
    )

    class _Transport:
        def flush_records_and_wait(self, timeout_s):
            return True

    class _Host:
        def flush_and_wait(self, timeout_s):
            assert timeout_s == pytest.approx(4.0)
            return True

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = _Transport()
    engine._host_engine = _Host()
    engine._record_mode = True

    with pytest.raises(TimeoutError, match="durable record flush timed out"):
        engine.flush_and_wait(5.0)


def test_flush_and_wait_raises_when_host_reports_timeout():
    class _Transport:
        def flush_records_and_wait(self, timeout_s):
            return True

    class _Host:
        def flush_and_wait(self, timeout_s):
            return False

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = _Transport()
    engine._host_engine = _Host()
    engine._record_mode = True

    with pytest.raises(TimeoutError, match="durable record flush timed out"):
        engine.flush_and_wait(1.0)


def test_flush_and_wait_preserves_host_exception():
    failure = RuntimeError("host insert failed")

    class _Transport:
        def flush_records_and_wait(self, timeout_s):
            return True

    class _Host:
        def flush_and_wait(self, timeout_s):
            raise failure

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = _Transport()
    engine._host_engine = _Host()
    engine._record_mode = True

    with pytest.raises(RuntimeError) as raised:
        engine.flush_and_wait(1.0)
    assert raised.value is failure


def test_close_remains_best_effort_when_native_stop_fails(monkeypatch):
    class _Ring:
        def stop(self):
            raise RuntimeError("stop failed")

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = SimpleNamespace(null_offload=False, force_eager=False)
    engine._ring_engine = _Ring()

    monkeypatch.setattr("dmi.engine._ring_module", lambda: SimpleNamespace(deactivate=lambda: None))
    engine.close()
    assert engine._ring_transport is None
    assert engine._ring_engine is None
