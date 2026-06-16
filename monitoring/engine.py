"""Monitoring engine wrapper backed only by the native C++/CUDA engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .config import MonitoringConfig


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
    adapter (HFAdaptor / VLLMAdaptor), not here.  Callers wanting NVTX
    ranges call ``monitoring.hook_points.set_monitoring_debug(True)``
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
                    from . import _native_engine

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
            # Only DMXHostEngine has a lifecycle here. A Python callable sink
            # (e.g. hallu_monitor ProbeWorker) is started/stopped by its owner.
            if isinstance(self._host_engine, _native_engine.DMXHostEngine):
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

    @staticmethod
    def _make_default_ring_config(
        *,
        payload_mb: int,
        pinned_mb: int,
        task_entries: int,
    ) -> Any:
        """Build a default RingConfig for the ring-only monitoring path."""
        from . import _native_engine  # type: ignore[attr-defined]

        ring_config = _native_engine.RingConfig()
        ring_config.payload_ring_bytes = int(payload_mb) * 1024 * 1024
        ring_config.pinned_staging_bytes = int(pinned_mb) * 1024 * 1024
        ring_config.task_ring_entries = int(task_entries)
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
        from . import ring_transport as _rt
        from . import _native_engine  # type: ignore[attr-defined]

        if self._ring_transport is not None:
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
        if isinstance(self._host_engine, _native_engine.DMXHostEngine):
            host_cpp = self._host_engine
        elif callable(self._host_engine):
            # Python callable sink (e.g. hallu_monitor ProbeWorker): RingEngine's
            # binding routes host tensors to it in memory, bypassing ClickHouse.
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

    def close(self) -> None:
        """Tear down backend resources."""

        if self._ring_transport is not None:
            try:
                ring_engine = getattr(self, "_ring_engine", None)
                if ring_engine is not None:
                    ring_engine.stop()
            except Exception:
                pass
            try:
                from . import ring_transport as _rt
                _rt.deactivate()
            except Exception:
                pass
            self._ring_transport = None
            self._ring_engine = None

        # Only DMXHostEngine owns a ClickHouse pipeline to tear down here; a
        # callable sink (ProbeWorker) is stopped by its owner.
        if isinstance(self._host_engine, _native_engine.DMXHostEngine):
            try:
                self._host_engine.close_input()
                self._host_engine.stop()
            except Exception:
                pass
        self._host_engine = None


# ---------------------------------------------------------------------------
# Backend loader


__all__ = ["MonitoringEngine"]
