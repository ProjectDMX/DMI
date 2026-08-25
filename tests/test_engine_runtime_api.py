"""Focused tests for MonitoringEngine's narrow ring runtime surface."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

import dmi.engine as engine_module
from dmi.engine import MonitoringEngine, RingCapacities

pytestmark = pytest.mark.cpu


class _FakeRingEngine:
    def __init__(self, transport=None, *, fail_null_mode=False, fail_stop=False):
        self.transport = transport
        self.fail_null_mode = fail_null_mode
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
            raise RuntimeError("ring drain failed")


def _engine_with_fake_ring(
    *, null_offload=False, force_eager=False, fail_null_mode=False, fail_stop=False
):
    engine = MonitoringEngine(enable_ring_transport=False)
    transport = SimpleNamespace(
        null_offload=null_offload,
        force_eager=force_eager,
    )
    ring_engine = _FakeRingEngine(
        transport,
        fail_null_mode=fail_null_mode,
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


def test_default_ring_flushes_small_workloads_with_bounded_linger(monkeypatch):
    class _RingConfig:
        pass

    monkeypatch.setattr(
        engine_module,
        "_native_module",
        lambda: SimpleNamespace(RingConfig=_RingConfig),
    )

    config = MonitoringEngine._make_default_ring_config(
        payload_mb=64,
        pinned_mb=32,
        task_entries=128,
    )

    assert config.drain_flush_timeout_us == 100_000


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


def test_close_surfaces_host_worker_failure_after_cleanup(monkeypatch):
    engine, _transport, ring_engine = _engine_with_fake_ring()
    deactivated = []

    class _FailingHostEngine:
        def __init__(self):
            self.calls = []

        def close_input(self):
            self.calls.append("close_input")

        def stop(self):
            self.calls.append("stop")

        def raise_if_failed(self):
            self.calls.append("raise_if_failed")
            raise RuntimeError("clickhouse worker failed")

    host_engine = _FailingHostEngine()
    engine._host_engine = host_engine
    fake_transport_module = ModuleType("dmi.transport.ring")
    fake_transport_module.deactivate = lambda: deactivated.append(True)
    monkeypatch.setitem(sys.modules, "dmi.transport.ring", fake_transport_module)

    with pytest.raises(RuntimeError, match="clickhouse worker failed"):
        engine.close()

    assert ring_engine.stop_calls == 1
    assert deactivated == [True]
    assert host_engine.calls == ["close_input", "stop", "raise_if_failed"]
    assert engine._ring_engine is None
    assert engine._ring_transport is None
    assert engine._host_engine is None


def test_close_preserves_ring_failure_while_cleaning_up_host(monkeypatch):
    engine, _transport, ring_engine = _engine_with_fake_ring(fail_stop=True)
    deactivated = []

    class _HostEngine:
        def __init__(self):
            self.calls = []

        def close_input(self):
            self.calls.append("close_input")

        def stop(self):
            self.calls.append("stop")

        def raise_if_failed(self):
            self.calls.append("raise_if_failed")

    host_engine = _HostEngine()
    engine._host_engine = host_engine
    fake_transport_module = ModuleType("dmi.transport.ring")
    fake_transport_module.deactivate = lambda: deactivated.append(True)
    monkeypatch.setitem(sys.modules, "dmi.transport.ring", fake_transport_module)

    with pytest.raises(RuntimeError, match="ring drain failed"):
        engine.close()

    assert ring_engine.stop_calls == 1
    assert deactivated == [True]
    assert host_engine.calls == ["close_input", "stop", "raise_if_failed"]
    assert engine._ring_engine is None
    assert engine._ring_transport is None
    assert engine._host_engine is None


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
