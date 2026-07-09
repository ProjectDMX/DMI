from __future__ import annotations

import time

from monitoring.governor import (
    PCIeGovernor,
    PCIeGovernorConfig,
    PCIeHint,
    emit_hint,
    get_current,
    set_current,
)


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
        }

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
        return dict(self.stats)


def _governor(**kwargs) -> tuple[PCIeGovernor, FakeEngine]:
    engine = FakeEngine()
    cfg = PCIeGovernorConfig(enabled=True, **kwargs)
    return PCIeGovernor(engine, cfg), engine


def test_hint_enters_defer_and_resume_hint_clears_only_hint_deadline():
    governor, engine = _governor(max_defer_us=10_000)

    governor.on_hint(PCIeHint(direction="D2H", source="kv_store"))

    assert engine.defer_writes[-1] > time.monotonic_ns()
    assert governor.snapshot()["hints_accepted"] == 1

    governor.on_hint(
        PCIeHint(
            direction="D2H",
            source="kv_store",
            valid_until_ns=time.monotonic_ns(),
        )
    )

    assert engine.defer_writes[-1] == 0
    assert governor.snapshot()["hint_deadline_ns"] == 0


def test_small_untrusted_hint_is_ignored():
    governor, engine = _governor(hint_min_bytes=4 * 1024 * 1024)

    governor.on_hint(
        PCIeHint(direction="D2H", source="unknown", est_bytes=1024)
    )

    assert engine.defer_writes == []
    assert governor.snapshot()["hints_ignored"] == 1


def test_feedback_pressure_enters_and_clean_windows_resume():
    governor, engine = _governor(
        baseline_gbps=100.0,
        pressure_ewma_alpha=1.0,
        p_hi=0.35,
        p_lo=0.15,
        hot_windows_to_yield=1,
        clean_windows_to_resume=2,
    )

    governor.on_step()

    bytes_sample = 10 * 1024 * 1024
    engine.stats["d2h_probe_bytes"] += bytes_sample
    engine.stats["d2h_probe_event_us"] += int((bytes_sample * 8) / (50.0 * 1000))
    governor.on_step()

    assert engine.defer_writes[-1] > time.monotonic_ns()
    assert governor.snapshot()["feedback_yields"] == 1

    for _ in range(2):
        engine.stats["d2h_probe_bytes"] += bytes_sample
        engine.stats["d2h_probe_event_us"] += int((bytes_sample * 8) / (100.0 * 1000))
        governor.on_step()

    assert engine.defer_writes[-1] == 0
    assert governor.snapshot()["feedback_deadline_ns"] == 0


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

    governor, engine = _governor()
    set_current(governor)
    try:
        assert emit_hint(direction="D2H", source="kv_store") is True
        assert engine.defer_writes[-1] > time.monotonic_ns()
    finally:
        set_current(None)
