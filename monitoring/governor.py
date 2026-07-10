"""PCIe-aware, serving-first drain governor for DMI ring offload."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class PCIeHint:
    """Best-effort declaration that serving-critical traffic is about to run."""

    direction: str
    source: str
    est_bytes: int = 0
    valid_until_ns: int = 0


@dataclass
class PCIeGovernorConfig:
    """Runtime policy for the rank-local DMI drain governor."""

    enabled: bool = False
    hint_min_bytes: int = 4 * 1024 * 1024
    # Safety expiry for a missing end hint, not the normal yield duration.
    max_defer_us: int = 1_000_000
    hard_watermark_ratio: float = 0.80
    max_d2h_chunk_bytes: int = 32 * 1024 * 1024
    baseline_gbps: float = 0.0
    p_hi: float = 0.35
    p_lo: float = 0.15
    feedback_window_ms: int = 100
    clean_windows_to_resume: int = 20
    hot_windows_to_yield: int = 2
    pressure_ewma_alpha: float = 0.20
    baseline_decay: float = 0.999
    debug: bool = False


_current: Optional["PCIeGovernor"] = None


def set_current(governor: Optional["PCIeGovernor"]) -> None:
    """Set the process-local governor used by optional connector hints."""

    global _current
    _current = governor


def get_current() -> Optional["PCIeGovernor"]:
    """Return the process-local governor, if enabled for this worker."""

    return _current


def emit_hint(
    *,
    direction: str,
    source: str,
    est_bytes: int = 0,
    valid_until_ns: int = 0,
) -> bool:
    """Send a hint to the current governor, if one exists.

    Connector paths use this helper so that disabled or unavailable governor
    support is a silent no-op.
    """

    governor = get_current()
    if governor is None:
        return False
    governor.on_hint(
        PCIeHint(
            direction=direction,
            source=source,
            est_bytes=int(est_bytes or 0),
            valid_until_ns=int(valid_until_ns or 0),
        )
    )
    return True


class PCIeGovernor:
    """Rank-local policy engine for DMI D2H drain yielding.

    The governor has no CUDA dependency. The injected engine only needs the
    small duck-typed surface exposed by RingEnginePy:
    ``set_drain_control()``, ``set_defer_until_ns()``, and ``link_stats()``.
    """

    _COARSE_TRUSTED_D2H_SOURCES = frozenset(
        {
            "kv_store",
            "lmcache_store",
            "lmcache_mp_store",
            "offloading_store",
        }
    )

    def __init__(
        self,
        engine: Any,
        config: PCIeGovernorConfig,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.engine = engine
        self.config = config
        self._clock_ns = clock_ns
        self._state_lock = threading.RLock()

        self._hint_deadline_ns = 0
        self._hint_deadlines_ns: dict[str, int] = {}
        self._feedback_deadline_ns = 0
        self._feedback_active = False
        self._last_written_defer_ns = 0
        self._last_stats: Optional[dict[str, int]] = None
        self._window_started_ns = 0
        self._disabled = False

        self._baseline_gbps = float(config.baseline_gbps or 0.0)
        self._ewma_pressure = 0.0
        self._clean_windows = 0
        self._hot_windows = 0

        self.hints_seen = 0
        self.hints_accepted = 0
        self.hints_ignored = 0
        self.feedback_yields = 0
        self.defer_writes = 0

        self._max_defer_ns = int(config.max_defer_us) * 1_000
        self._feedback_window_ns = max(
            1, int(config.feedback_window_ms) * 1_000_000
        )
        debug_env = os.environ.get("DMX_PCIE_GOVERNOR_DEBUG")
        self._debug = bool(
            config.debug
            or debug_env not in (None, "", "0", "false", "False")
        )

        hard_watermark_bytes = self._compute_hard_watermark_bytes()
        self.engine.set_drain_control(
            0,
            int(config.max_d2h_chunk_bytes or 0),
            hard_watermark_bytes,
        )

    def on_hint(self, hint: PCIeHint) -> None:
        with self._state_lock:
            self._on_hint_locked(hint)

    def _on_hint_locked(self, hint: PCIeHint) -> None:
        """Accept a serving-side hint and immediately update C++ defer state."""

        if self._disabled:
            return

        now = self._clock_ns()
        source = str(hint.source or "")
        ending = hint.valid_until_ns > 0 and hint.valid_until_ns <= now
        self._expire_hint_deadlines(now)
        self.hints_seen += 1

        if not self._hint_is_relevant(hint) and not (
            ending and source in self._hint_deadlines_ns
        ):
            self.hints_ignored += 1
            self._debug_log("ignore_hint", hint=hint)
            return

        self.hints_accepted += 1
        if ending:
            self._hint_deadlines_ns.pop(source, None)
        else:
            requested = (
                int(hint.valid_until_ns)
                if hint.valid_until_ns > now
                else now + self._max_defer_ns
            )
            self._hint_deadlines_ns[source] = min(
                requested, now + self._max_defer_ns
            )

        self._refresh_hint_deadline()
        self._write_defer(now)
        self._debug_log("hint", hint=hint, hint_deadline_ns=self._hint_deadline_ns)

    def on_step(self) -> None:
        with self._state_lock:
            self._on_step_locked()

    def _on_step_locked(self) -> None:
        """Run feedback control once at the serving step boundary."""

        if self._disabled:
            return

        now = self._clock_ns()
        self._expire_hint_deadlines(now)
        if self._feedback_deadline_ns <= now:
            self._feedback_deadline_ns = 0

        if (
            self._last_stats is not None
            and now - self._window_started_ns < self._feedback_window_ns
        ):
            self._write_defer(now)
            return

        stats = self.engine.link_stats()
        stats = {str(k): int(v) for k, v in stats.items()}

        if self._last_stats is None:
            self._last_stats = stats
            self._window_started_ns = now
            self._write_defer(now)
            return

        probe_bytes = self._delta(stats, "d2h_probe_bytes")
        probe_event_us = self._delta(stats, "d2h_probe_event_us")
        stall_count = self._delta(stats, "stall_count")

        pressure_observed = False
        pressure = self._ewma_pressure
        if probe_bytes > 0 and probe_event_us > 0:
            bw_gbps = (probe_bytes * 8.0) / (probe_event_us * 1000.0)
            self._update_baseline(bw_gbps)
            if self._baseline_gbps > 0:
                raw_pressure = max(0.0, 1.0 - (bw_gbps / self._baseline_gbps))
                alpha = min(max(float(self.config.pressure_ewma_alpha), 0.0), 1.0)
                pressure = alpha * raw_pressure + (1.0 - alpha) * self._ewma_pressure
                self._ewma_pressure = pressure
                pressure_observed = True

        is_hot = pressure_observed and pressure > float(self.config.p_hi)
        is_clean = (
            pressure_observed
            and pressure < float(self.config.p_lo)
            and stall_count == 0
        )

        if not self._feedback_active:
            self._clean_windows = 0
            self._hot_windows = self._hot_windows + 1 if is_hot else 0
            if self._hot_windows >= max(1, int(self.config.hot_windows_to_yield)):
                self._feedback_active = True
                self._feedback_deadline_ns = now + 2 * self._feedback_window_ns
                self.feedback_yields += 1
                self._hot_windows = 0
        else:
            self._hot_windows = 0
            if is_hot:
                self._clean_windows = 0
                self._feedback_deadline_ns = now + 2 * self._feedback_window_ns
                self.feedback_yields += 1
            elif is_clean:
                self._clean_windows += 1
            else:
                self._clean_windows = 0

            if self._clean_windows >= max(
                1, int(self.config.clean_windows_to_resume)
            ):
                self._feedback_active = False
                self._feedback_deadline_ns = 0
                self._clean_windows = 0

        self._last_stats = stats
        self._window_started_ns = now
        self._write_defer(now)
        self._debug_log(
            "step",
            pressure=round(self._ewma_pressure, 4),
            baseline_gbps=round(self._baseline_gbps, 4),
            hint_deadline_ns=self._hint_deadline_ns,
            feedback_deadline_ns=self._feedback_deadline_ns,
        )

    def disable(self) -> None:
        with self._state_lock:
            self._disable_locked()

    def _disable_locked(self) -> None:
        """Fail open and restore the pre-governor drain controls."""

        if self._disabled:
            return
        self._disabled = True
        self._hint_deadlines_ns.clear()
        self._hint_deadline_ns = 0
        self._feedback_deadline_ns = 0
        self._feedback_active = False
        self._last_written_defer_ns = 0
        try:
            self.engine.set_drain_control(0, 0, 0)
        except Exception:
            try:
                self.engine.set_defer_until_ns(0)
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        """Return an audit snapshot of governor state."""

        return {
            "enabled": bool(self.config.enabled) and not self._disabled,
            "hint_deadline_ns": self._hint_deadline_ns,
            "feedback_deadline_ns": self._feedback_deadline_ns,
            "feedback_active": self._feedback_active,
            "defer_until_ns": self._last_written_defer_ns,
            "baseline_gbps": self._baseline_gbps,
            "ewma_pressure": self._ewma_pressure,
            "clean_windows": self._clean_windows,
            "hot_windows": self._hot_windows,
            "window_started_ns": self._window_started_ns,
            "hints_seen": self.hints_seen,
            "hints_accepted": self.hints_accepted,
            "hints_ignored": self.hints_ignored,
            "feedback_yields": self.feedback_yields,
            "defer_writes": self.defer_writes,
        }

    def _compute_hard_watermark_bytes(self) -> int:
        ratio = float(self.config.hard_watermark_ratio or 0.0)
        if ratio <= 0:
            return 0
        try:
            cap = int(self.engine.payload_cap())
        except Exception:
            stats = self.engine.link_stats()
            cap = int(stats.get("payload_cap", 0))
        if cap <= 0:
            return 0
        return max(0, int(cap * min(ratio, 1.0)))

    def _hint_is_relevant(self, hint: PCIeHint) -> bool:
        source = str(hint.source or "")
        direction = str(hint.direction or "").upper()
        est_bytes = int(hint.est_bytes or 0)

        if direction != "D2H":
            return False
        if est_bytes > 0 and est_bytes < int(self.config.hint_min_bytes or 0):
            return False
        if source in self._COARSE_TRUSTED_D2H_SOURCES:
            return True
        return est_bytes > 0 and est_bytes >= int(self.config.hint_min_bytes or 0)

    def _expire_hint_deadlines(self, now_ns: int) -> None:
        expired = [
            source
            for source, deadline in self._hint_deadlines_ns.items()
            if deadline <= now_ns
        ]
        for source in expired:
            self._hint_deadlines_ns.pop(source, None)
        self._refresh_hint_deadline()

    def _refresh_hint_deadline(self) -> None:
        self._hint_deadline_ns = max(self._hint_deadlines_ns.values(), default=0)

    def _update_baseline(self, bw_gbps: float) -> None:
        fixed = float(self.config.baseline_gbps or 0.0)
        if fixed > 0:
            self._baseline_gbps = fixed
            return
        if self._baseline_gbps <= 0:
            self._baseline_gbps = bw_gbps
            return
        decayed = self._baseline_gbps * float(self.config.baseline_decay)
        self._baseline_gbps = max(decayed, bw_gbps)

    def _delta(self, stats: dict[str, int], key: str) -> int:
        if self._last_stats is None:
            return 0
        return max(0, int(stats.get(key, 0)) - int(self._last_stats.get(key, 0)))

    def _write_defer(self, now_ns: int) -> None:
        hint_deadline = self._hint_deadline_ns if self._hint_deadline_ns > now_ns else 0
        feedback_deadline = (
            self._feedback_deadline_ns if self._feedback_deadline_ns > now_ns else 0
        )
        defer_until_ns = max(hint_deadline, feedback_deadline)
        if defer_until_ns == self._last_written_defer_ns:
            return
        self.engine.set_defer_until_ns(int(defer_until_ns))
        self._last_written_defer_ns = int(defer_until_ns)
        self.defer_writes += 1

    def _debug_log(self, event: str, **fields: Any) -> None:
        if not self._debug:
            return
        payload = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[DMI PCIeGovernor] {event} {payload}", file=sys.stderr, flush=True)


__all__ = [
    "PCIeGovernor",
    "PCIeGovernorConfig",
    "PCIeHint",
    "emit_hint",
    "get_current",
    "set_current",
]
