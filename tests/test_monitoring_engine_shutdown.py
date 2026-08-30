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

    with pytest.raises(TimeoutError, match="durable record completion"):
        transport.flush_records_and_wait(0.25)


def test_flush_and_wait_rejects_nonpositive_deadline():
    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = SimpleNamespace()
    engine._record_mode = True
    with pytest.raises(ValueError, match="must be positive"):
        engine.flush_and_wait(0)


def test_flush_and_wait_uses_one_native_deadline_and_does_not_flush_host_twice():
    calls = []

    class _Transport:
        def flush_records_and_wait(self, timeout_s):
            calls.append(("record_sink", timeout_s))

    class _Host:
        def flush_and_wait(self, timeout_s):  # pragma: no cover - must not run
            calls.append(("legacy_host", timeout_s))

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = _Transport()
    engine._host_engine = _Host()
    engine._record_mode = True

    engine.flush_and_wait(5.0)

    assert calls == [("record_sink", 5.0)]


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


def test_close_keeps_record_ring_when_native_worker_may_be_alive():
    class _Ring:
        def stop(self):
            raise RuntimeError("stop failed before join")

    engine = MonitoringEngine(enable_ring_transport=False)
    transport = SimpleNamespace(null_offload=False, force_eager=False)
    ring = _Ring()
    engine._ring_transport = transport
    engine._ring_engine = ring
    engine._record_mode = True

    engine.close()

    assert engine._ring_transport is transport
    assert engine._ring_engine is ring
