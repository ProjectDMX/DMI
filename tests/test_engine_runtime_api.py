"""Focused tests for MonitoringEngine's narrow ring runtime surface."""

import os
import subprocess
import sys
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from dmi.engine import MonitoringEngine, RingCapacities, SinkStats

pytestmark = pytest.mark.cpu


class _FakeRingEngine:
    def __init__(
        self, transport=None, *, fail_null_mode=False, suppressed=0,
        fail_stop=False,
    ):
        self.transport = transport
        self.fail_null_mode = fail_null_mode
        self.suppressed = suppressed
        self.fail_stop = fail_stop
        self.null_mode_calls = []
        self.stop_calls = 0
        self.init_calls = 0
        self.start_calls = 0

    def payload_cap(self):
        return 1024

    def staging_cap(self):
        return 768

    def task_cap(self):
        return 32

    def set_null_mode(self, enabled):
        self.null_mode_calls.append(
            (
                enabled,
                None if self.transport is None else self.transport.null_offload,
                None if self.transport is None else self.transport.force_eager,
            )
        )
        if self.fail_null_mode:
            raise RuntimeError("native transition failed")

    def payload_tensor(self):
        return object()

    def init(self):
        self.init_calls += 1

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("ring stop failed")

    def suppressed_submit_failures(self):
        return self.suppressed


class _FakeQueueStats:
    def __init__(
        self,
        *,
        dropped=0,
        full_errors=0,
        closed_errors=0,
        too_large_errors=0,
        retries=0,
    ):
        self.dropped = dropped
        self.full_errors = full_errors
        self.closed_errors = closed_errors
        self.too_large_errors = too_large_errors
        self.retries = retries


class _FakeProfiling:
    def __init__(self, queue_stats):
        self.queue_by_stage = [queue_stats]


class _FakeHostEngine:
    def __init__(self, *, failure=None, profiling=None):
        self.failure = failure
        self._profiling = profiling
        self.close_input_calls = 0
        self.stop_calls = 0

    def close_input(self):
        self.close_input_calls += 1

    def stop(self):
        self.stop_calls += 1

    def raise_if_failed(self):
        if self.failure is not None:
            raise self.failure

    def profiling(self):
        return self._profiling


def _engine_with_fake_ring(
    *,
    null_offload=False,
    force_eager=False,
    fail_null_mode=False,
    suppressed=0,
    fail_stop=False,
):
    engine = MonitoringEngine(enable_ring_transport=False)
    transport = SimpleNamespace(
        null_offload=null_offload,
        force_eager=force_eager,
    )
    ring_engine = _FakeRingEngine(
        transport,
        fail_null_mode=fail_null_mode,
        suppressed=suppressed,
        fail_stop=fail_stop,
    )
    engine._ring_transport = transport
    engine._ring_engine = ring_engine
    return engine, transport, ring_engine


def test_ring_capacities_is_frozen_snapshot_with_effective_limit():
    engine, _transport, _ring_engine = _engine_with_fake_ring()

    capacities = engine.ring_capacities()

    assert capacities == RingCapacities(
        payload_bytes=1024,
        staging_bytes=768,
        task_entries=32,
    )
    assert capacities.effective_bytes == 768
    assert not hasattr(capacities, "__dict__")
    with pytest.raises(FrozenInstanceError):
        capacities.payload_bytes = 1


def test_capture_toggle_changes_metadata_flag_after_native_transition():
    engine, transport, ring_engine = _engine_with_fake_ring(force_eager=True)
    assert engine.capture_enabled is True

    engine.set_capture_enabled(False)

    assert transport.null_offload is True
    assert transport.force_eager is False
    assert engine.capture_enabled is False
    assert ring_engine.null_mode_calls == [(True, False, True)]

    engine.set_capture_enabled(True)

    assert transport.null_offload is False
    assert transport.force_eager is False
    assert engine.capture_enabled is True
    assert ring_engine.null_mode_calls[-1] == (False, True, False)


def test_capture_toggle_is_a_noop_when_state_already_matches():
    engine, _transport, ring_engine = _engine_with_fake_ring()

    engine.set_capture_enabled(True)

    assert ring_engine.null_mode_calls == []


def test_capture_toggle_leaves_python_flags_unchanged_on_native_failure():
    engine, transport, ring_engine = _engine_with_fake_ring(
        force_eager=True,
        fail_null_mode=True,
    )

    with pytest.raises(RuntimeError, match="native transition failed"):
        engine.set_capture_enabled(False)

    assert ring_engine.null_mode_calls == [(True, False, True)]
    assert transport.null_offload is False
    assert transport.force_eager is True
    assert engine.capture_enabled is True


def test_close_restores_device_global_null_mode_before_ring_stop():
    engine, transport, ring_engine = _engine_with_fake_ring(null_offload=True)

    engine.close()

    assert ring_engine.null_mode_calls == [(False, True, False)]
    assert ring_engine.stop_calls == 1
    assert transport.null_offload is False
    assert engine.capture_enabled is False


def test_close_raises_host_engine_failure_with_sink_stats():
    engine, _transport, ring_engine = _engine_with_fake_ring(suppressed=7)
    host_engine = _FakeHostEngine(
        failure=RuntimeError("insert failed"),
        profiling=_FakeProfiling(
            _FakeQueueStats(dropped=3, full_errors=2, retries=1)
        ),
    )
    engine._host_engine = host_engine

    with pytest.raises(
        RuntimeError, match="DMX host sink failed during teardown"
    ) as excinfo:
        engine.close()

    message = str(excinfo.value)
    assert "dropped=3" in message
    assert "suppressed=7" in message
    assert "full_errors=2" in message
    assert "retries=1" in message
    assert excinfo.value.__cause__ is host_engine.failure
    assert host_engine.close_input_calls == 1
    assert host_engine.stop_calls == 1
    assert ring_engine.stop_calls == 1
    assert engine._host_engine is None
    assert engine._ring_engine is None


def test_close_raises_on_suppressed_submits_without_a_worker_failure():
    """The counters must not be gated behind an unrelated worker failure.

    ``log_submit_failure_once`` prints at most one warning per process, so a
    silent ``close()`` here would hide unbounded row loss.
    """
    engine, _transport, _ring_engine = _engine_with_fake_ring(suppressed=4)
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats())
    )

    with pytest.raises(RuntimeError, match=r"lost 4 row\(s\)") as excinfo:
        engine.close()

    assert "suppressed=4" in str(excinfo.value)


def test_close_warns_on_dropped_rows_without_a_worker_failure():
    # Drops are reachable through a configured OnFullPolicy.DROP, so they warn
    # rather than raise -- but they must not be silent.
    engine, _transport, _ring_engine = _engine_with_fake_ring()
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats(dropped=2, full_errors=2))
    )

    with pytest.warns(RuntimeWarning, match=r"dropped 2 row\(s\)"):
        engine.close()

    assert engine._host_engine is None


def test_close_is_silent_when_nothing_was_lost():
    engine, _transport, _ring_engine = _engine_with_fake_ring()
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats(retries=5))
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine.close()


def test_close_keeps_an_earlier_teardown_error_in_the_traceback():
    engine, _transport, _ring_engine = _engine_with_fake_ring(fail_stop=True)
    engine._host_engine = _FakeHostEngine(
        failure=RuntimeError("insert failed"),
        profiling=_FakeProfiling(_FakeQueueStats()),
    )

    with pytest.raises(
        RuntimeError, match="DMX host sink failed during teardown"
    ) as excinfo:
        engine.close()

    context = excinfo.value.__context__
    assert isinstance(context, RuntimeError)
    assert "ring stop failed" in str(context)


def test_close_reports_a_ring_failure_when_the_sink_is_healthy():
    engine, _transport, _ring_engine = _engine_with_fake_ring(fail_stop=True)
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats())
    )

    with pytest.raises(RuntimeError, match="ring stop failed"):
        engine.close()


def test_close_stays_idempotent_after_raising():
    engine, _transport, _ring_engine = _engine_with_fake_ring(suppressed=1)
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats())
    )

    with pytest.raises(RuntimeError):
        engine.close()

    engine.close()  # handles were cleared before the raise


def test_a_second_close_does_not_erase_the_final_sink_stats():
    """close() is idempotent and callers invoke it from finally blocks, so a
    repeat call must not overwrite the forensic record with zeros."""
    engine, _transport, _ring_engine = _engine_with_fake_ring(suppressed=4)
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats(dropped=5, retries=2))
    )

    with pytest.raises(RuntimeError):
        engine.close()
    first = engine.sink_stats()
    assert (first.dropped, first.suppressed, first.retries) == (5, 4, 2)

    engine.close()  # no engines left to read
    assert engine.sink_stats() == first


def test_a_second_close_does_not_re_warn(recwarn):
    engine, _transport, _ring_engine = _engine_with_fake_ring()
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats(dropped=3))
    )

    with pytest.warns(RuntimeWarning):
        engine.close()
    recwarn.clear()
    engine.close()
    assert len(recwarn) == 0


def test_sink_stats_is_readable_before_and_after_close():
    engine, _transport, _ring_engine = _engine_with_fake_ring(suppressed=6)
    engine._host_engine = _FakeHostEngine(
        profiling=_FakeProfiling(_FakeQueueStats(dropped=1, retries=3))
    )

    live = engine.sink_stats()
    assert (live.dropped, live.suppressed, live.retries) == (1, 6, 3)
    assert live.lost_rows == 7

    with pytest.raises(RuntimeError):
        engine.close()

    # The engines are gone, but the final snapshot is still available.
    assert engine.sink_stats() == live


def test_sink_stats_handles_a_backend_whose_profiling_returns_none():
    """profiling() returns None when the engine carries no stats.

    The ring counter must still be read, and the queue counters must report
    zero rather than blowing up.
    """
    engine, _transport, _ring_engine = _engine_with_fake_ring(suppressed=2)
    engine._host_engine = _FakeHostEngine(profiling=None)

    stats = engine.sink_stats()

    assert stats == SinkStats(suppressed=2)
    assert stats.lost_rows == 2


def test_sink_stats_reads_the_pybind_snapshot_shape():
    """queue_by_stage is a list (from std::array<QueueStats, NumStages>) whose
    entries expose exactly the six read-only counters bound in bindings.cpp."""

    class _PybindQueueStats:
        __slots__ = (
            "enqueued", "dropped", "full_errors",
            "closed_errors", "too_large_errors", "retries",
        )

        def __init__(self):
            self.enqueued, self.dropped, self.full_errors = 100, 3, 2
            self.closed_errors, self.too_large_errors, self.retries = 1, 4, 5

    class _PybindSnapshot:
        def __init__(self):
            self.ingest = object()
            self.queue_by_stage = [_PybindQueueStats()]
            self.stage_by_stage = [object()]

    engine, _transport, _ring_engine = _engine_with_fake_ring(suppressed=7)
    engine._host_engine = _FakeHostEngine(profiling=_PybindSnapshot())

    assert engine.sink_stats() == SinkStats(
        dropped=3,
        suppressed=7,
        full_errors=2,
        closed_errors=1,
        too_large_errors=4,
        retries=5,
    )


def test_sink_stats_tolerates_a_backend_without_the_counters():
    engine, _transport, _ring_engine = _engine_with_fake_ring()
    engine._ring_engine = SimpleNamespace()  # no suppressed_submit_failures()
    engine._host_engine = SimpleNamespace()  # no profiling()

    assert engine.sink_stats() == SinkStats()


def test_replacing_disabled_ring_restores_native_null_mode(monkeypatch):
    engine, _transport, old_ring = _engine_with_fake_ring(null_offload=True)
    new_ring = _FakeRingEngine()
    activated = []
    deactivated = []

    class _FakeTransport:
        def __init__(self, ring_engine):
            self._ring_engine = ring_engine
            self._ring_payload = ring_engine.payload_tensor()
            self.null_offload = False
            self.force_eager = False

        def set_model_cfg(self, _model_shape):
            pass

    fake_transport_module = ModuleType("dmi.transport.ring")
    fake_transport_module.RingTransport = _FakeTransport
    fake_transport_module.activate = activated.append
    fake_transport_module.deactivate = lambda: deactivated.append(True)

    fake_native_module = ModuleType("dmi.transport.native")
    fake_native_module.RingEngine = lambda _config, _host: new_ring
    fake_native_module.DMXHostEngine = type("DMXHostEngine", (), {})

    monkeypatch.setitem(
        sys.modules, "dmi.transport.ring", fake_transport_module
    )
    monkeypatch.setitem(
        sys.modules, "dmi.transport.native", fake_native_module
    )
    engine.enable_ring_transport(object())

    assert old_ring.null_mode_calls == [(False, True, False)]
    assert old_ring.stop_calls == 1
    assert deactivated == [True]
    assert new_ring.init_calls == 1
    assert new_ring.start_calls == 1
    assert activated == [engine._ring_transport]
    assert engine.capture_enabled is True

    engine.close()
    assert new_ring.stop_calls == 1
    assert deactivated == [True, True]


def test_ring_runtime_api_requires_an_enabled_transport():
    engine = MonitoringEngine(enable_ring_transport=False)

    assert engine.capture_enabled is False
    with pytest.raises(RuntimeError, match="Ring transport is not enabled"):
        engine.ring_capacities()
    with pytest.raises(RuntimeError, match="Ring transport is not enabled"):
        engine.set_capture_enabled(True)


def test_plain_dmi_import_does_not_load_native_or_transport_modules():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dmi; "
                "assert 'dmi.transport.native' not in sys.modules; "
                "assert 'dmi.transport.ring' not in sys.modules"
            ),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
