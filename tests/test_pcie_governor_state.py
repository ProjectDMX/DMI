from __future__ import annotations

import pytest

from monitoring.governor import (
    PCIeGovernor,
    PCIeGovernorConfig,
    PCIeHint,
    emit_hint,
    get_current,
    set_current,
)


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000


class FakeEngine:
    def __init__(self, payload_cap: int = 1024 * 1024 * 1024) -> None:
        self._payload_cap = payload_cap
        self.drain_control_calls: list[tuple[int, int, int]] = []
        self.defer_writes: list[int] = []
        self.stats = {
            "d2h_bytes": 0,
            "d2h_batches": 0,
            "d2h_probe_bytes": 0,
            "d2h_probe_event_us": 0,
            "d2h_probe_host_us": 0,
            "stall_us_total": 0,
            "stall_count": 0,
            "pending_bytes": 0,
            "pending_entries": 0,
            "payload_reserved_bytes": 0,
            "staging_used_bytes": 0,
            "payload_cap": payload_cap,
            "task_cap": 1024,
            "staging_cap": payload_cap,
        }
        self.link_stats_calls = 0

    def payload_cap(self) -> int:
        return self._payload_cap

    def set_drain_control(
        self,
        defer_until_ns: int,
        max_d2h_chunk_bytes: int,
        hard_watermark_bytes: int,
    ) -> None:
        self.drain_control_calls.append(
            (defer_until_ns, max_d2h_chunk_bytes, hard_watermark_bytes)
        )

    def set_defer_until_ns(self, defer_until_ns: int) -> None:
        self.defer_writes.append(defer_until_ns)

    def link_stats(self) -> dict[str, int]:
        self.link_stats_calls += 1
        return dict(self.stats)


def _governor(**kwargs) -> tuple[PCIeGovernor, FakeEngine, FakeClock]:
    engine = FakeEngine()
    clock = FakeClock()
    cfg = PCIeGovernorConfig(enabled=True, **kwargs)
    return PCIeGovernor(engine, cfg, clock_ns=clock), engine, clock


def _add_probe(engine: FakeEngine, *, bandwidth_gbps: float) -> None:
    bytes_sample = 10 * 1024 * 1024
    engine.stats["d2h_probe_bytes"] += bytes_sample
    engine.stats["d2h_probe_event_us"] += int(
        (bytes_sample * 8) / (bandwidth_gbps * 1000)
    )


def test_hint_enters_defer_and_resume_hint_clears_only_hint_deadline():
    governor, engine, clock = _governor(max_defer_us=10_000)

    governor.on_hint(PCIeHint(direction="D2H", source="kv_store"))

    assert engine.defer_writes[-1] > clock()
    assert governor.snapshot()["hints_accepted"] == 1

    governor.on_hint(
        PCIeHint(
            direction="D2H",
            source="kv_store",
            valid_until_ns=clock(),
        )
    )

    assert engine.defer_writes[-1] == 0
    assert governor.snapshot()["hint_deadline_ns"] == 0


def test_small_untrusted_hint_is_ignored():
    governor, engine, _ = _governor(hint_min_bytes=4 * 1024 * 1024)

    governor.on_hint(
        PCIeHint(direction="D2H", source="unknown", est_bytes=1024)
    )

    assert engine.defer_writes == []
    assert governor.snapshot()["hints_ignored"] == 1


def test_h2d_hint_is_audit_only_and_does_not_defer_drain():
    governor, engine, _ = _governor()

    governor.on_hint(PCIeHint(direction="H2D", source="kv_load"))

    assert engine.defer_writes == []
    assert governor.snapshot()["hints_ignored"] == 1


def test_default_max_defer_is_a_stale_hint_watchdog():
    assert PCIeGovernorConfig().max_defer_us == 1_000_000


def test_lmcache_mp_store_is_a_trusted_d2h_source():
    governor, engine, clock = _governor(max_defer_us=10_000)

    governor.on_hint(PCIeHint(direction="D2H", source="lmcache_mp_store"))

    assert engine.defer_writes[-1] > clock()
    assert governor.snapshot()["hints_accepted"] == 1


def test_ending_one_source_keeps_other_source_deferred():
    governor, engine, clock = _governor(max_defer_us=10_000)
    governor.on_hint(PCIeHint(direction="D2H", source="kv_store"))
    first_deadline = engine.defer_writes[-1]

    clock.advance_ms(1)
    governor.on_hint(PCIeHint(direction="D2H", source="lmcache_store"))
    second_deadline = engine.defer_writes[-1]
    assert second_deadline > first_deadline

    governor.on_hint(
        PCIeHint(
            direction="D2H",
            source="kv_store",
            valid_until_ns=clock(),
        )
    )
    assert governor.snapshot()["hint_deadline_ns"] == second_deadline
    assert engine.defer_writes[-1] == second_deadline

    governor.on_hint(
        PCIeHint(
            direction="D2H",
            source="lmcache_store",
            valid_until_ns=clock(),
        )
    )
    assert engine.defer_writes[-1] == 0


def test_feedback_uses_wall_clock_windows_before_yielding_and_cleaning():
    governor, engine, clock = _governor(
        baseline_gbps=100.0,
        pressure_ewma_alpha=1.0,
        p_hi=0.35,
        p_lo=0.15,
        feedback_window_ms=100,
        hot_windows_to_yield=2,
        clean_windows_to_resume=2,
    )

    governor.on_step()
    assert engine.link_stats_calls == 1

    _add_probe(engine, bandwidth_gbps=50.0)
    clock.advance_ms(10)
    governor.on_step()
    assert engine.link_stats_calls == 1
    assert engine.defer_writes == []

    clock.advance_ms(90)
    governor.on_step()
    assert governor.snapshot()["hot_windows"] == 1
    assert engine.defer_writes == []

    _add_probe(engine, bandwidth_gbps=50.0)
    clock.advance_ms(100)
    governor.on_step()

    assert engine.defer_writes[-1] > clock()
    assert governor.snapshot()["feedback_yields"] == 1
    assert governor.snapshot()["feedback_active"] is True

    for _ in range(2):
        _add_probe(engine, bandwidth_gbps=100.0)
        clock.advance_ms(100)
        governor.on_step()

    assert engine.defer_writes[-1] == 0
    assert governor.snapshot()["feedback_deadline_ns"] == 0
    assert governor.snapshot()["feedback_active"] is False


def test_mid_band_window_breaks_consecutive_hot_sequence():
    governor, engine, clock = _governor(
        baseline_gbps=100.0,
        pressure_ewma_alpha=1.0,
        p_hi=0.35,
        p_lo=0.15,
        feedback_window_ms=100,
        hot_windows_to_yield=2,
    )
    governor.on_step()

    _add_probe(engine, bandwidth_gbps=50.0)
    clock.advance_ms(100)
    governor.on_step()
    assert governor.snapshot()["hot_windows"] == 1

    _add_probe(engine, bandwidth_gbps=80.0)
    clock.advance_ms(100)
    governor.on_step()
    assert governor.snapshot()["hot_windows"] == 0

    _add_probe(engine, bandwidth_gbps=50.0)
    clock.advance_ms(100)
    governor.on_step()
    assert governor.snapshot()["hot_windows"] == 1
    assert engine.defer_writes == []


def test_init_sets_static_chunk_cap_and_hard_watermark():
    engine = FakeEngine(payload_cap=1000)
    cfg = PCIeGovernorConfig(
        enabled=True,
        max_d2h_chunk_bytes=64,
        hard_watermark_ratio=0.75,
    )

    PCIeGovernor(engine, cfg)

    assert engine.drain_control_calls == [(0, 64, 750)]


def test_current_governor_helper_is_noop_when_disabled_and_emits_when_set():
    set_current(None)
    assert get_current() is None
    assert emit_hint(direction="D2H", source="kv_store") is False

    governor, engine, clock = _governor()
    set_current(governor)
    try:
        assert emit_hint(direction="D2H", source="kv_store") is True
        assert engine.defer_writes[-1] > clock()
    finally:
        set_current(None)


def test_hard_watermark_falls_back_to_link_stats_payload_cap():
    class StatsOnlyEngine:
        def __init__(self) -> None:
            self.controls = []

        def set_drain_control(self, *control) -> None:
            self.controls.append(control)

        def set_defer_until_ns(self, _deadline: int) -> None:
            pass

        def link_stats(self) -> dict[str, int]:
            return {"payload_cap": 1000}

    engine = StatsOnlyEngine()
    PCIeGovernor(
        engine,
        PCIeGovernorConfig(enabled=True, hard_watermark_ratio=0.75),
    )

    assert engine.controls == [(0, 32 * 1024 * 1024, 750)]


def test_enable_ring_transport_stops_engine_when_governor_init_fails(monkeypatch):
    from monitoring import _native_engine, governor as governor_module
    from monitoring import ring_transport
    from monitoring.engine import MonitoringEngine

    class FakeNativeRingEngine:
        def __init__(self) -> None:
            self.init_calls = 0
            self.start_calls = 0
            self.stop_calls = 0

        def init(self) -> None:
            self.init_calls += 1

        def start(self) -> None:
            self.start_calls += 1

        def stop(self) -> None:
            self.stop_calls += 1

        def payload_tensor(self):
            return object()

    native_engine = FakeNativeRingEngine()
    monkeypatch.setattr(
        _native_engine,
        "RingEngine",
        lambda _config, _host: native_engine,
    )

    def raise_on_governor_init(*_args, **_kwargs):
        raise RuntimeError("synthetic governor init failure")

    monkeypatch.setattr(
        governor_module,
        "PCIeGovernor",
        raise_on_governor_init,
    )

    engine = MonitoringEngine.__new__(MonitoringEngine)
    engine._host_engine = None
    engine._ring_engine = None
    engine._ring_transport = None
    engine._pcie_governor_config = PCIeGovernorConfig(enabled=True)
    set_current(None)
    ring_transport.deactivate()

    with pytest.raises(RuntimeError, match="synthetic governor init failure"):
        engine.enable_ring_transport(object())

    assert native_engine.init_calls == 1
    assert native_engine.start_calls == 1
    assert native_engine.stop_calls == 1
    assert engine._ring_engine is None
    assert engine._ring_transport is None
    assert get_current() is None
    assert ring_transport.get_active() is None


def test_enable_ring_transport_cleans_up_when_native_start_fails(monkeypatch):
    from monitoring import _native_engine
    from monitoring.engine import MonitoringEngine

    class StartFailingRingEngine:
        def __init__(self) -> None:
            self.init_calls = 0
            self.start_calls = 0
            self.stop_calls = 0

        def init(self) -> None:
            self.init_calls += 1

        def start(self) -> None:
            self.start_calls += 1
            raise RuntimeError("synthetic native start failure")

        def stop(self) -> None:
            self.stop_calls += 1

    native_engine = StartFailingRingEngine()
    monkeypatch.setattr(
        _native_engine,
        "RingEngine",
        lambda _config, _host: native_engine,
    )

    engine = MonitoringEngine.__new__(MonitoringEngine)
    engine._host_engine = None
    engine._ring_engine = None
    engine._ring_transport = None
    engine._pcie_governor_config = PCIeGovernorConfig(enabled=True)

    with pytest.raises(RuntimeError, match="synthetic native start failure"):
        engine.enable_ring_transport(object())

    assert native_engine.init_calls == 1
    assert native_engine.start_calls == 1
    assert native_engine.stop_calls == 1
    assert engine._ring_engine is None
    assert engine._ring_transport is None
