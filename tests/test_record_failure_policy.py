"""The record runtime's failure policy and capture status, from the engine.

The native ring does the work (tests/native/ring: the consumer's discard
under disable_capture, the per-step stall budget). This suite pins the
Python surface around it with the native modules faked: what
create_record_runtime accepts and passes to the native ring, what it refuses
before touching the live ring, how a step is marked, and what
capture_status() and close() report.
"""

from __future__ import annotations

import logging

import pytest

from dmi.engine import MonitoringEngine
from tests.test_engine_runtime_api import (
    _FakeRingEngine,
    _engine_with_fake_ring,
    _explicit_sink_format,
    _record_ring_fakes,
)

pytestmark = pytest.mark.cpu


class _StatusRing(_FakeRingEngine):
    """A record ring that reports the native capture status."""

    def __init__(self, status=None):
        super().__init__()
        self.status = status or {
            "failure_policy": "raise", "failed": False, "failure": "",
            "discarded_descriptors": 0, "discarded_payloads": 0,
            "step_stall_budget_ms": 0, "stall_budget_exhaustions": 0,
            "reserve_wait_ns": 0, "max_step_wait_ns": 0,
        }
        self.steps = 0

    def capture_status(self):
        return dict(self.status)

    def begin_record_step(self):
        self.steps += 1


def _record_engine(monkeypatch, ring, **runtime_options):
    """An engine switched to `ring` through create_record_runtime."""
    engine, _old_transport, _old_ring = _engine_with_fake_ring()
    engine._ring_config = object()
    engine._host_engine = object()
    created = []

    def create_record(config, target, **options):
        created.append(options)
        return ring

    _record_ring_fakes(monkeypatch, create_record=create_record,
                       activate=lambda transport: None, deactivated=[])
    runtime = engine.create_record_runtime(
        _explicit_sink_format(), **runtime_options)
    return engine, runtime, created


# --- create_record_runtime ----------------------------------------------------


def test_the_default_policy_is_raise_with_no_stall_budget(monkeypatch):
    _engine, _runtime, created = _record_engine(monkeypatch, _StatusRing())
    assert created == [{"failure_policy": "raise", "step_stall_budget_ms": 0}]


def test_disable_capture_and_a_budget_reach_the_native_ring(monkeypatch):
    _engine, _runtime, created = _record_engine(
        monkeypatch, _StatusRing(), failure_policy="disable_capture",
        step_stall_budget_ms=2000)
    assert created == [{"failure_policy": "disable_capture",
                        "step_stall_budget_ms": 2000}]


@pytest.mark.parametrize("options, match", [
    ({"failure_policy": "ignore"}, "failure_policy"),
    ({"failure_policy": None}, "failure_policy"),
    ({"step_stall_budget_ms": 0}, "step_stall_budget_ms"),
    ({"step_stall_budget_ms": -5}, "step_stall_budget_ms"),
    ({"step_stall_budget_ms": 2.5}, "step_stall_budget_ms"),
    ({"step_stall_budget_ms": True}, "step_stall_budget_ms"),
])
def test_bad_options_are_refused_before_the_live_ring_is_touched(
        monkeypatch, options, match):
    engine, _old_transport, old_ring = _engine_with_fake_ring()
    engine._ring_config = object()
    engine._host_engine = object()
    _record_ring_fakes(
        monkeypatch,
        create_record=lambda *a, **k: pytest.fail("ring must not be built"),
        activate=lambda transport: None, deactivated=[])
    with pytest.raises(ValueError, match=match):
        engine.create_record_runtime(_explicit_sink_format(), **options)
    assert old_ring.stop_calls == 0
    assert engine._record_mode is False


def test_disable_capture_needs_a_bounded_sink_admission(tmp_path):
    """Once the budget latches, the forward still waits for the one sink
    admission in progress; unbounded, that wait is the stall the policy
    exists to prevent."""
    from dmi.config import MonitoringConfig
    from dmi.storage.native_capture import NativeSinkConfig

    config = MonitoringConfig(
        storage_backend="persistent",
        capture_sink_config=NativeSinkConfig(
            spool_root=str(tmp_path), admission_timeout_s=None))
    engine = MonitoringEngine(config=config, model_id="policy",
                              enable_ring_transport=False)
    with pytest.raises(ValueError, match="admission_timeout_s"):
        engine.create_record_runtime(
            _explicit_sink_format(), failure_policy="disable_capture")


# --- steps -----------------------------------------------------------------------


def test_begin_step_starts_a_native_stall_budget_step(monkeypatch):
    ring = _StatusRing()
    _engine, runtime, _created = _record_engine(monkeypatch, ring)
    runtime._transport.begin_record_step = ring.begin_record_step
    runtime.begin_step()
    runtime.begin_step()
    assert ring.steps == 2


# --- capture_status --------------------------------------------------------------


def test_capture_status_without_a_record_runtime():
    engine = MonitoringEngine(enable_ring_transport=False)
    assert engine.capture_status() == {
        "record_mode": False, "capture_active": False,
        "failure_policy": None, "failure": None,
        "discarded_descriptors": 0, "discarded_payloads": 0,
        "step_stall_budget_ms": None, "stall_budget_exhaustions": 0,
        "reserve_wait_s": 0.0, "max_step_wait_s": 0.0,
        "sink": None, "storage": None,
    }


def test_capture_status_reports_a_disabled_capture(monkeypatch):
    ring = _StatusRing({
        "failure_policy": "disable_capture", "failed": True,
        "failure": "NativePackSink: sink refused durable admission: timed_out",
        "discarded_descriptors": 7, "discarded_payloads": 9,
        "step_stall_budget_ms": 2000, "stall_budget_exhaustions": 1,
        "reserve_wait_ns": 2_500_000_000, "max_step_wait_ns": 2_100_000_000,
    })
    engine, _runtime, _created = _record_engine(
        monkeypatch, ring, failure_policy="disable_capture",
        step_stall_budget_ms=2000)
    status = engine.capture_status()
    assert status == {
        "record_mode": True, "capture_active": False,
        "failure_policy": "disable_capture",
        "failure": "NativePackSink: sink refused durable admission: timed_out",
        "discarded_descriptors": 7, "discarded_payloads": 9,
        "step_stall_budget_ms": 2000, "stall_budget_exhaustions": 1,
        "reserve_wait_s": 2.5, "max_step_wait_s": 2.1,
        "sink": None, "storage": None,
    }


def test_capture_status_includes_the_sink_and_storage_snapshots(monkeypatch):
    engine, _runtime, _created = _record_engine(monkeypatch, _StatusRing())

    class _Sink:
        def snapshot(self):
            return {"persisted_records": 3, "timed_out_records": 0}

    class _Storage:
        def snapshot(self):
            return {"indexed_packs": 1}

    engine._record_sink = _Sink()
    engine._capture_storage = _Storage()
    status = engine.capture_status()
    assert status["capture_active"] is True
    assert status["failure"] is None
    assert status["step_stall_budget_ms"] is None
    assert status["sink"] == {"persisted_records": 3, "timed_out_records": 0}
    assert status["storage"] == {"indexed_packs": 1}


def test_close_reports_a_capture_that_stopped(monkeypatch, caplog):
    ring = _StatusRing({
        "failure_policy": "disable_capture", "failed": True,
        "failure": "record capture stall budget exhausted",
        "discarded_descriptors": 4, "discarded_payloads": 5,
        "step_stall_budget_ms": 50, "stall_budget_exhaustions": 1,
        "reserve_wait_ns": 0, "max_step_wait_ns": 0,
    })
    engine, _runtime, _created = _record_engine(
        monkeypatch, ring, failure_policy="disable_capture",
        step_stall_budget_ms=50)
    with caplog.at_level(logging.WARNING, logger="dmi.engine"):
        engine.close()
    assert ring.stop_calls == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("stall budget exhausted" in message and "5 payloads" in message
               for message in messages), messages
