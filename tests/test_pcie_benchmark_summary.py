from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "exp"
    / "pcie_kv_pressure"
    / "scripts"
    / "summarize_scheduler_effectiveness.py"
)
SPEC = importlib.util.spec_from_file_location("pcie_effectiveness_summary", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _hint_line(
    *, source: str, valid: int, action: str, now: int, source_deadline: int
) -> str:
    return (
        "[DMI PCIeGovernor] hint "
        "hint=PCIeHint(direction='D2H', "
        f"source='{source}', est_bytes=0, valid_until_ns={valid}) "
        f"action={action} now_ns={now} "
        f"source_deadline_ns={source_deadline} "
        f"hint_deadline_ns={source_deadline}\n"
    )


def test_governor_summary_uses_explicit_lifecycle_actions():
    log = "".join(
        [
            _hint_line(
                source="lmcache_store",
                valid=0,
                action="start",
                now=10_000_000,
                source_deadline=1_010_000_000,
            ),
            _hint_line(
                source="lmcache_store",
                valid=0,
                action="renew",
                now=110_000_000,
                source_deadline=1_110_000_000,
            ),
            _hint_line(
                source="lmcache_store",
                valid=130_000_000,
                action="end",
                now=130_000_000,
                source_deadline=0,
            ),
        ]
    )

    result = summary.summarize_governor(log, max_defer_us=1_000_000)

    assert result["hints_started"] == 1
    assert result["hints_renewed"] == 1
    assert result["hints_ended"] == 1
    assert result["hint_coverage_ms"]["mean"] == 120.0


def test_legacy_summary_treats_post_watchdog_hint_as_fresh_start():
    log = "\n".join(
        [
            "PCIeHint(direction='D2H', source='lmcache_store', "
            "est_bytes=0, valid_until_ns=0) hint_deadline_ns=10000000",
            "PCIeHint(direction='D2H', source='lmcache_store', "
            "est_bytes=0, valid_until_ns=0) hint_deadline_ns=30000000",
            "PCIeHint(direction='D2H', source='lmcache_store', "
            "est_bytes=0, valid_until_ns=28000000) hint_deadline_ns=0",
        ]
    )

    result = summary.summarize_governor(log, max_defer_us=5_000)

    assert result["hints_started"] == 2
    assert result["hints_renewed"] == 0
    assert result["hints_expired_without_end"] == 1
    assert result["hint_coverage_ms"]["mean"] == 3.0


def test_mp_summary_aggregates_overlapping_stores_as_one_episode():
    mp_log = "\n".join(
        [
            "Stored 100 tokens in 0.001 seconds",
            "Stored 200 tokens in 0.002 seconds",
        ]
    )
    audit_log = (
        "[DMI PCIeHint] source=lmcache_mp_store event=end duration_ms=100.0 stores=2"
    )

    result = summary.summarize_lmcache(
        audit_log,
        phase_elapsed_s=1.0,
        mp_log_text=mp_log,
        kv_bytes_per_token=1024,
    )

    assert result["store_calls"] == 2
    assert result["completion_episodes"] == 1
    assert result["completion_mapping_valid"] is True
    assert result["offload_ms"]["mean"] == 100.0


def test_mp_summary_rejects_ambiguous_legacy_episode_mapping():
    mp_log = "\n".join(
        [
            "Stored 100 tokens in 0.001 seconds",
            "Stored 200 tokens in 0.002 seconds",
        ]
    )
    audit_log = "[DMI PCIeHint] source=lmcache_mp_store event=end duration_ms=100.0"

    result = summary.summarize_lmcache(
        audit_log,
        phase_elapsed_s=1.0,
        mp_log_text=mp_log,
        kv_bytes_per_token=1024,
    )

    assert result["completion_mapping_valid"] is False
    assert result["completion_episodes"] == 0
    assert result["offload_ms"]["count"] == 0
