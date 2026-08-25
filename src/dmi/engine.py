"""DMI engine wrapper backed by the native C++/CUDA engine."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import warnings
from typing import Any, Optional, Sequence

from .config import MonitoringConfig


DEFAULT_DRAIN_FLUSH_TIMEOUT_US = 0


def _native_module() -> Any:
    """Load the native-extension facade only when the engine needs it."""
    return importlib.import_module("dmi.transport.native")


def _ring_module() -> Any:
    """Load the ring transport only when the engine needs it."""
    return importlib.import_module("dmi.transport.ring")


@dataclass(frozen=True, slots=True)
class SinkStats:
    """Immutable snapshot of the host sink's loss and backpressure counters.

    ``dropped`` and ``suppressed`` are the two counters that mean rows were
    produced but never reached ClickHouse:

    * ``dropped`` -- the insert queue discarded rows, either because the stage
      was configured with ``OnFullPolicy.DROP`` or because the engine was
      already stopping when they were submitted.
    * ``suppressed`` -- the native P2P thread caught an exception out of
      ``submit_direct()`` and dropped the row.  This is never a configured
      behavior; a nonzero value always means something went wrong.

    The remaining fields are diagnostics that explain *why*, and may overlap
    with the two above.
    """

    dropped: int = 0
    suppressed: int = 0
    full_errors: int = 0
    closed_errors: int = 0
    too_large_errors: int = 0
    retries: int = 0

    @property
    def lost_rows(self) -> int:
        """Rows that provably never reached the sink."""
        return self.dropped + self.suppressed

    def __str__(self) -> str:
        return (
            f"dropped={self.dropped} suppressed={self.suppressed} "
            f"full_errors={self.full_errors} "
            f"closed_errors={self.closed_errors} "
            f"too_large_errors={self.too_large_errors} "
            f"retries={self.retries}"
        )


def _read_sink_stats(host_engine: Any, ring_engine: Any) -> SinkStats:
    """Best-effort read of the sink counters; never raises."""

    fields: dict[str, int] = {}

    profiling = None
    if host_engine is not None:
        try:
            profiling = host_engine.profiling()
        except Exception:
            profiling = None
    if profiling is not None:
        try:
            queue_stats = profiling.queue_by_stage[0]
        except Exception:
            queue_stats = None
        if queue_stats is not None:
            for name in (
                "dropped",
                "full_errors",
                "closed_errors",
                "too_large_errors",
                "retries",
            ):
                try:
                    fields[name] = int(getattr(queue_stats, name))
                except Exception:
                    pass

    if ring_engine is not None:
        try:
            fields["suppressed"] = int(ring_engine.suppressed_submit_failures())
        except Exception:
            pass

    return SinkStats(**fields)


def _host_engine_teardown_error(stats: SinkStats, exc: Exception) -> RuntimeError:
    error = RuntimeError(f"DMX host sink failed during teardown ({stats})")
    error.__cause__ = exc
    return error


def _lost_rows_error(stats: SinkStats) -> RuntimeError:
    return RuntimeError(
        f"DMX host sink lost {stats.lost_rows} row(s) during this run; "
        f"captured internals are incomplete ({stats})"
    )


@dataclass(frozen=True, slots=True)
class RingCapacities:
    """Immutable snapshot of the active ring transport's capacities."""

    payload_bytes: int
    staging_bytes: int
    task_entries: int

    @property
    def effective_bytes(self) -> int:
        """Usable per-step byte capacity across payload and staging rings."""
        return min(self.payload_bytes, self.staging_bytes)


@dataclass
class HostEngineConfig:
    """Configuration wrapper for the native DMXHostEngine pipeline.

    The DMXHostEngine is a single-stage ClickHouse insert pipeline that
    receives pre-assembled rows from the ring transport drain thread.

    Notes:
      - Requires the native monitoring extension (CUDA + pybind11).
      - Expects exactly **one** stage (clickhouse_insert).
    """

    stages: Sequence[Any]
    start_on_init: bool = True


class MonitoringEngine:
    """High-level wrapper that routes monitoring tasks to the native backend.

    Canonical surface that adapters depend on:
      * ``__init__(config, model_id, host_engine|db_config)``
      * ``enable_ring_transport(ring_config, model_shape=None) -> RingTransport``
        (enabled by default from ``__init__`` with a default RingConfig)
      * ``next_auto_group_id() -> int``  -- engine-scoped counter for HF;
        vLLM passes its own scheduler-assigned request IDs.
      * ``close()``
      * ``model.monitoring_engine = engine`` -- the convention adapters
        look for to discover the active engine.

    Per-framework state (no_strip_left_pad, batch tracking, etc.) lives on the
    concrete framework adapter, not here. Callers wanting NVTX
    ranges call ``dmi.hooks.set_monitoring_debug(True)``
    directly.
    """

    def __init__(
        self,
        *,
        config: Optional[MonitoringConfig] = None,
        model_id: Optional[str] = None,
        host_engine: Optional[Any] = None,
        db_config: Optional[HostEngineConfig] = None,
        enable_ring_transport: bool = True,
        ring_config: Optional[Any] = None,
        ring_payload_mb: int = 4096,
        ring_pinned_mb: int = 4096,
        ring_task_entries: int = 65536,
    ) -> None:
        self.config = config
        self._model_id = model_id
        self._auto_batch_group_id = 0
        # Final sink counters, captured during close() so sink_stats() keeps
        # working after the engines are gone.
        self._final_sink_stats: Optional[SinkStats] = None

        # Host-side DB engine (optional; C++ backend only)
        self._host_engine: Optional[Any] = None

        self._ring_transport: Optional[Any] = None

        if host_engine is not None and db_config is not None:
            raise ValueError("Provide either host_engine or db_config, not both")

        if host_engine is not None or db_config is not None:
            if self._model_id is None:
                raise ValueError("model_id is required when host_engine integration is enabled")
            self._host_engine = host_engine
            if self._host_engine is None and db_config is not None:
                try:
                    _native_engine = _native_module()
                    DMXHostEngine = _native_engine.DMXHostEngine  # type: ignore[attr-defined]
                except Exception as exc:
                    raise RuntimeError("Failed to import native DMXHostEngine") from exc
                stages = tuple(db_config.stages)
                if len(stages) != 1:
                    raise ValueError("db_config.stages must contain exactly 1 StageConfig object (clickhouse_insert)")

                try:
                    self._host_engine = DMXHostEngine(stages[0])  # type: ignore[call-arg]
                except Exception as exc:
                    raise RuntimeError("Failed to construct DMXHostEngine") from exc
            if self._host_engine is not None:
                try:
                    if db_config is None or db_config.start_on_init:
                        self._host_engine.start()
                except Exception as exc:
                    raise RuntimeError("Failed to start host_engine") from exc

        if enable_ring_transport or ring_config is not None:
            if ring_config is None:
                ring_config = self._make_default_ring_config(
                    payload_mb=ring_payload_mb,
                    pinned_mb=ring_pinned_mb,
                    task_entries=ring_task_entries,
                )
            self.enable_ring_transport(ring_config)


    # ------------------------------------------------------------------
    # Ring transport API

    def ring_capacities(self) -> RingCapacities:
        """Return a stable snapshot of the active ring's capacity limits.

        This deliberately exposes values rather than the native ``RingEngine``
        object so integrations cannot mutate ring lifecycle or reservation
        state.
        """
        ring_engine = getattr(self, "_ring_engine", None)
        if ring_engine is None:
            raise RuntimeError("Ring transport is not enabled")
        return RingCapacities(
            payload_bytes=int(ring_engine.payload_cap()),
            staging_bytes=int(ring_engine.staging_cap()),
            task_entries=int(ring_engine.task_cap()),
        )

    @property
    def capture_enabled(self) -> bool:
        """Whether the active transport is accepting capture metadata."""
        transport = self._ring_transport
        return transport is not None and not bool(transport.null_offload)

    def set_capture_enabled(self, enabled: bool) -> None:
        """Enable or suppress capture at a lifecycle quiescent boundary.

        Callers must ensure that no forward pass can overlap this method (for
        example, immediately before and after framework warmup).  The native
        null-mode transition performs its own CUDA synchronization.  Metadata
        suppression changes only after that transition succeeds, so an error
        leaves the Python-visible state unchanged.
        """
        transport = self._ring_transport
        ring_engine = getattr(self, "_ring_engine", None)
        if transport is None or ring_engine is None:
            raise RuntimeError("Ring transport is not enabled")

        target_null_mode = not bool(enabled)
        if bool(transport.null_offload) == target_null_mode:
            return
        ring_engine.set_null_mode(target_null_mode)
        transport.null_offload = target_null_mode
        # The eager safety-net bypasses the native producer and does not read
        # null_offload.  Never carry a prior oversized-step decision across a
        # lifecycle toggle; the next committed step recomputes it.
        transport.force_eager = False

    @staticmethod
    def _make_default_ring_config(
        *,
        payload_mb: int,
        pinned_mb: int,
        task_entries: int,
    ) -> Any:
        """Build a default RingConfig for the ring-only monitoring path."""
        _native_engine = _native_module()

        ring_config = _native_engine.RingConfig()
        ring_config.payload_ring_bytes = int(payload_mb) * 1024 * 1024
        ring_config.pinned_staging_bytes = int(pinned_mb) * 1024 * 1024
        ring_config.task_ring_entries = int(task_entries)
        ring_config.drain_flush_timeout_us = DEFAULT_DRAIN_FLUSH_TIMEOUT_US
        return ring_config

    def enable_ring_transport(
        self, ring_config: Any, model_shape: Optional[Any] = None
    ) -> Any:
        """Switch to ring-based D2H transport.

        Creates a RingEngine with the C++ host engine as the submit target so
        tensor reconstruction, slicing, and DB submission all happen in C++
        without the GIL.

        Args:
            ring_config:  A _native_engine.RingConfig instance.
            model_shape:  Optional ModelShapeConfig for analytical shape computation.
                          When provided, the new CUDA-graph-compatible forward-hook
                          path is activated.  If None, shape is auto-detected from
                          model.config in _install_monitoring_forward.

        Returns:
            The ``RingTransport`` instance (also stored as
            ``self._ring_transport``).  Returned so adapters can hold a
            direct reference instead of reaching through the engine.
        """
        _rt = _ring_module()
        _native_engine = _native_module()

        if self._ring_transport is not None:
            # Native null mode is device-global rather than RingEngine-local.
            # Restore its default before destroying a disabled transport so a
            # replacement starts capture-enabled without an extra startup sync.
            if not self.capture_enabled:
                self.set_capture_enabled(True)
            try:
                ring_engine = getattr(self, "_ring_engine", None)
                if ring_engine is not None:
                    ring_engine.stop()
            except Exception:
                pass
            try:
                _rt.deactivate()
            except Exception:
                pass
            self._ring_transport = None
            self._ring_engine = None

        # Pass the DMXHostEngine C++ object directly; RingEngine builds a
        # SubmitFn that calls submit_direct without touching Python/GIL.
        # Pass None for null/benchmark mode (no DB writes).
        host_cpp = None
        if self._host_engine is not None and isinstance(
            self._host_engine, _native_engine.DMXHostEngine
        ):
            host_cpp = self._host_engine

        ring_engine = _native_engine.RingEngine(ring_config, host_cpp)

        ring_engine.init()
        ring_engine.start()

        transport = _rt.RingTransport(ring_engine)
        if model_shape is not None:
            transport.set_model_cfg(model_shape)
        self._ring_engine = ring_engine
        self._ring_transport = transport

        _rt.activate(transport)
        return transport

    # ------------------------------------------------------------------
    def next_auto_group_id(self) -> int:
        """Claim a unique batch-group ID for an HF generate() call.

        Engine-scoped counter so each top-level monitored generate()
        receives a distinct group prefix; per-request IDs are then minted
        as f"{group}:{i}" by the HF adapter.  vLLM does not use this
        (vLLM passes its own scheduler-assigned request IDs).
        """
        gid = int(self._auto_batch_group_id)
        self._auto_batch_group_id += 1
        return gid

    def sink_stats(self) -> SinkStats:
        """Snapshot the host sink's loss and backpressure counters.

        Safe to call at any point in the run, and after ``close()`` -- the
        final counts are cached during teardown, so the snapshot survives the
        engines being torn down.
        """

        if self._final_sink_stats is not None:
            return self._final_sink_stats
        return _read_sink_stats(
            self._host_engine, getattr(self, "_ring_engine", None)
        )

    def close(self) -> None:
        """Tear down backend resources.

        Raises ``RuntimeError`` if teardown failed, if the sink recorded a
        worker failure, or if rows were produced but never reached the sink.
        Both engine handles are cleared before raising, so ``close()`` stays
        idempotent and a second call is a no-op.
        """

        teardown_error: Exception | None = None
        ring_engine = getattr(self, "_ring_engine", None)

        if self._ring_transport is not None:
            # Best-effort reset of the device-global native null flag.  This is
            # needed only after callers explicitly disabled capture; the normal
            # HF path pays no extra synchronization cost.
            if not self.capture_enabled:
                try:
                    self.set_capture_enabled(True)
                except Exception:
                    pass
            try:
                if ring_engine is not None:
                    ring_engine.stop()
            except Exception as exc:
                teardown_error = teardown_error or exc
            try:
                _rt = _ring_module()
                _rt.deactivate()
            except Exception as exc:
                teardown_error = teardown_error or exc
            self._ring_transport = None
            self._ring_engine = None

        host_engine = self._host_engine
        if host_engine is not None:
            try:
                host_engine.close_input()
            except Exception as exc:
                teardown_error = teardown_error or exc
            try:
                host_engine.stop()
            except Exception as exc:
                teardown_error = teardown_error or exc

        # Read the counters after both engines are stopped and joined, so the
        # snapshot is final rather than a moving target.  Only the close() that
        # actually had engines reads and reports them: a repeat call must
        # neither overwrite the snapshot with zeros nor re-report it.
        stats = None
        if host_engine is not None or ring_engine is not None:
            stats = _read_sink_stats(host_engine, ring_engine)
            self._final_sink_stats = stats

        if host_engine is not None:
            # stats is always set here: host_engine being non-None satisfies the
            # condition above.  Spelled out so a future edit cannot quietly
            # pass None into the error builder.
            assert stats is not None
            try:
                host_engine.raise_if_failed()
            except Exception as exc:
                sink_error = _host_engine_teardown_error(stats, exc)
                # Prefer the sink failure -- it is the more actionable one --
                # but keep an earlier teardown error visible in the traceback
                # instead of discarding it.
                sink_error.__context__ = teardown_error
                teardown_error = sink_error
            self._host_engine = None

        if teardown_error is not None:
            raise teardown_error

        if stats is None:
            return  # already closed; nothing new to report

        # A clean shutdown is not the same as a complete capture: rows the
        # queue dropped or the P2P thread failed to submit are silent
        # otherwise (the native warning fires at most once per process).
        if stats.suppressed:
            raise _lost_rows_error(stats)
        if stats.dropped:
            # Reachable through a configured OnFullPolicy.DROP, so warn rather
            # than raise -- the caller may have opted into shedding load.
            warnings.warn(
                f"DMX host sink dropped {stats.dropped} row(s); captured "
                f"internals are incomplete ({stats})",
                RuntimeWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Backend loader


__all__ = ["MonitoringEngine", "RingCapacities"]
