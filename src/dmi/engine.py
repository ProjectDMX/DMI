"""DMI engine wrapper backed by the native C++/CUDA engine."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import time
from typing import Any, Optional, Sequence, TYPE_CHECKING, TypeVar

from .config import MonitoringConfig

if TYPE_CHECKING:
    from .records import RecordFormat, RecordRuntime


MetadataT = TypeVar("MetadataT")

_LOG = logging.getLogger(__name__)

DEFAULT_DRAIN_FLUSH_TIMEOUT_US = 0

# What a record runtime does when its sink refuses a record or stalls the
# forward past the step budget. "raise" (the default) raises in the forward
# that publishes the next record; "disable_capture" stops capture and keeps
# the forward running. Both report the failure at flush_and_wait, in
# capture_status() and at close.
RECORD_FAILURE_POLICIES = ("raise", "disable_capture")


def _native_module() -> Any:
    """Load the native-extension facade only when the engine needs it."""
    return importlib.import_module("dmi.transport.native")


def _ring_module() -> Any:
    """Load the ring transport only when the engine needs it."""
    return importlib.import_module("dmi.transport.ring")


def effective_ring_bytes(payload_bytes: int, staging_bytes: int) -> int:
    """Usable per-step byte capacity across payload and staging rings.

    The single spelling of the ceiling the native prepare_step/reserve_record
    enforce: the drain assembles each flush batch per whole entry and breaks
    when the entry does not fit staging, so the real limit is the smaller of
    the two rings, never payload alone.
    """
    return min(int(payload_bytes), int(staging_bytes))


@dataclass(frozen=True, slots=True)
class RingCapacities:
    """Immutable snapshot of the active ring transport's capacities."""

    payload_bytes: int
    staging_bytes: int
    task_entries: int

    @property
    def effective_bytes(self) -> int:
        """Usable per-step byte capacity across payload and staging rings."""
        return effective_ring_bytes(self.payload_bytes, self.staging_bytes)


@dataclass
class HostEngineConfig:
    """Configuration wrapper for the native DMXHostEngine pipeline.

    The DMXHostEngine is a single-stage ClickHouse insert pipeline that
    receives pre-assembled rows from the ring transport drain thread.

    Notes:
      - Requires the CPU host extension or the full CUDA extension.
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


        # Host-side DB engine (optional; C++ backend only)
        self._host_engine: Optional[Any] = None

        self._ring_transport: Optional[Any] = None
        self._ring_config: Optional[Any] = None
        self._record_mode = False
        # The native sink the record runtime writes to, when the engine
        # holds one (the persistent path's pack sink, or an explicit one).
        self._record_sink: Optional[Any] = None

        if host_engine is not None and db_config is not None:
            raise ValueError("Provide either host_engine or db_config, not both")

        # The canonical name: a deprecated one ("native", "capture") is
        # replaced here, for a MonitoringConfig and any config-like object
        # alike, so everything below reads only the current three choices
        # and "auto".
        from .config import canonical_storage_backend

        # What the caller wrote, kept only to quote it back in a refusal.
        self._storage_backend_requested = getattr(
            config, "storage_backend", "auto")
        self._storage_backend: str = canonical_storage_backend(
            self._storage_backend_requested)
        # A None config is the ctor's documented no-configuration mode;
        # say so here rather than behind a getattr default.
        self._capture_sink_config = (
            config.capture_sink_config if config is not None else None)
        if self._capture_sink_config is not None:
            from .storage.native_capture import NativeSinkConfig

            if not isinstance(self._capture_sink_config, NativeSinkConfig):
                raise TypeError(
                    "config.capture_sink_config must be a NativeSinkConfig")
        self._capture_storage_config = getattr(
            config, "capture_storage_config", None)
        if self._capture_storage_config is not None:
            from .storage.native_capture import NativeCaptureStorageConfig

            if not isinstance(
                self._capture_storage_config, NativeCaptureStorageConfig
            ):
                raise TypeError("config.capture_storage_config must be a "
                                "NativeCaptureStorageConfig")
        # The running storage service, while a record runtime is attached.
        self._capture_storage: Optional[Any] = None
        host_configured = host_engine is not None or db_config is not None
        if self._storage_backend == "in-memory" and not host_configured:
            raise ValueError(
                f"config.storage_backend={self._backend_label()} delivers "
                "records through "
                "the C++ ClickHouseRecordSink until its consumer interface "
                "exists, which needs a host engine: pass host_engine= or "
                "db_config="
            )
        if self._storage_backend == "none" and ring_config is not None:
            raise ValueError(
                "config.storage_backend='none' turns capture off, so the "
                "engine allocates no ring, but a ring_config was given. Pick "
                "'in-memory' or 'persistent' to capture, or drop ring_config"
            )
        if self._storage_backend in ("persistent", "none") and host_configured:
            raise ValueError(
                f"config.storage_backend={self._backend_label()} does not "
                "use the C++ ClickHouse host, but host_engine/db_config was "
                "given. Configured together, the host engine starts, connects "
                "and is then never fed, because a record runtime is handed "
                "either the host or an explicit record_sink and never both"
            )

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
                    raise ValueError(
                        "db_config.stages must contain exactly 1 StageConfig object"
                    )

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

        # Capture off means no ring at all: none of its pinned staging and
        # payload memory is allocated.
        if self._storage_backend == "none":
            enable_ring_transport = False
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

    def validate_capture_bounds(self, max_record_bytes: int) -> None:
        """Refuse this engine's capture bounds if a record cannot be stored.

        Adapters call it at attach with the largest record (one captured
        row's payload) they will emit, so a bound too small for it is a
        ``ConfigurationError`` there instead of a refusal on the record
        worker, which latches the runtime, in the middle of a forward. See
        :func:`dmi.storage.native_capture.validate_capture_bounds`.

        Checks the persistent path's ``capture_sink_config`` (and
        ``capture_storage_config``); with no sink config there is nothing
        to check.
        """
        sink_config = self._capture_sink_config
        if sink_config is None:
            if type(max_record_bytes) is not int or max_record_bytes <= 0:
                raise ValueError("max_record_bytes must be a positive int")
            return
        from .storage.native_capture import validate_capture_bounds

        validate_capture_bounds(
            sink_config, max_record_bytes,
            storage_config=self._capture_storage_config)

    def _backend_label(self) -> str:
        """The storage backend as the caller wrote it, with its current name
        when that was a deprecated one: ``'native' (now 'in-memory')``."""
        requested = getattr(self, "_storage_backend_requested",
                            self._storage_backend)
        if requested == self._storage_backend:
            return repr(self._storage_backend)
        return f"{requested!r} (now {self._storage_backend!r})"

    def _reject_a_sink_the_config_did_not_ask_for(
        self, record_sink: Optional[Any]
    ) -> None:
        """Hold the runtime to the storage path the config declared.

        The two paths are chosen in two different places -- a host engine at
        construction, a ``record_sink`` here -- so nothing but the caller's
        memory connected them. Declaring the backend is what lets a mismatch
        be an error: "I configured the object-store path and got ClickHouse
        rows" is otherwise a silent outcome, and the engine is the only place
        that can see both halves.

        ``"auto"`` keeps the original inference, so every caller written
        before the field existed is unaffected.
        """
        backend = getattr(self, "_storage_backend", "auto")
        if backend == "auto":
            return
        if backend == "persistent" and record_sink is None:
            raise ValueError(
                f"config.storage_backend={self._backend_label()} selects the "
                "object-store "
                "path, whose default writer is the native pack sink — pass "
                "capture_sink_config=NativeSinkConfig(...) in the config, or "
                "a record_sink explicitly (the reference sink is the "
                "documented rollback)"
            )
        if backend in ("in-memory", "none") and record_sink is not None:
            raise ValueError(
                f"config.storage_backend={self._backend_label()} does not use "
                "an explicit "
                "record_sink; passing one would send records to a backend the "
                "configuration did not select"
            )

    def create_record_runtime(
        self,
        record_format: "RecordFormat[MetadataT]",
        *,
        record_sink: Optional[Any] = None,
        failure_policy: str = "raise",
        step_stall_budget_ms: Optional[int] = None,
    ) -> "RecordRuntime[MetadataT]":
        """Create an opt-in, non-owning runtime for encoded records.

        ``record_sink=None`` preserves the native ClickHouse host path.  An
        explicit native ``RecordSink`` selects a separate backend for this
        runtime; the two paths are never active at the same time.

        ``failure_policy`` decides what a sink refusal (a dropped, timed-out
        or oversized record, or a sink failure) does to the forward:
        ``"raise"`` raises from the next record reservation or descriptor
        push; ``"disable_capture"`` stops capture, discards what follows
        and keeps the forward running. Either way ``flush_and_wait`` raises
        the failure, ``capture_status()`` reports it and ``close`` logs it.

        ``step_stall_budget_ms`` caps how long the record reservations of
        one step (see ``RecordRuntime.begin_step``) may wait for the sink to
        free ring space; past it the policy applies. ``None`` waits without
        bound. Under ``"disable_capture"`` the forward's stall per step is
        then at most the budget plus one sink admission timeout.
        """

        if getattr(self, "_storage_backend", "auto") == "none":
            raise RuntimeError(
                "config.storage_backend='none' turns capture off: this engine "
                "has no ring and creates no record runtime. Pick 'in-memory' "
                "or 'persistent' to capture"
            )
        self._validate_record_failure_options(
            failure_policy, step_stall_budget_ms)
        transport = self._ring_transport
        ring_config = self._ring_config
        if transport is None or ring_config is None:
            raise RuntimeError("Ring transport is not enabled")
        if self._record_mode:
            raise RuntimeError("A record runtime is already active")

        from .records import RecordFormat, RecordSchema

        if not isinstance(record_format, RecordFormat):
            raise TypeError("record_format must implement RecordFormat")
        record_schema = record_format.schema
        if not isinstance(record_schema, RecordSchema):
            raise TypeError("record_format.schema must be a RecordSchema")

        # The storage service starts BEFORE the sink it drains opens the
        # spool: its start sweeps a crashed sink's stale .open files, which
        # is safe only while nothing writes there. An explicit record_sink
        # may already hold the spool open, so it is left unswept.
        storage = self._start_capture_storage(sweep_spool=record_sink is None)
        try:
            runtime = self._attach_record_runtime(
                record_format, record_schema, record_sink,
                failure_policy=failure_policy,
                step_stall_budget_ms=step_stall_budget_ms or 0)
        except BaseException:
            if storage is not None:
                self._capture_storage = None
                storage.stop()
            raise
        return runtime

    def _validate_record_failure_options(
        self, failure_policy: Any, step_stall_budget_ms: Any
    ) -> None:
        if failure_policy not in RECORD_FAILURE_POLICIES:
            raise ValueError(
                f"failure_policy must be one of {RECORD_FAILURE_POLICIES}, "
                f"got {failure_policy!r}")
        if step_stall_budget_ms is not None and (
            type(step_stall_budget_ms) is not int or step_stall_budget_ms <= 0
        ):
            raise ValueError(
                "step_stall_budget_ms must be a positive int, or None to wait "
                "without bound")
        sink_config = self._capture_sink_config
        if (
            failure_policy == "disable_capture"
            and self._storage_backend == "persistent"
            and sink_config is not None
            and sink_config.overload == "block"
            and sink_config.admission_timeout_s is None
        ):
            # Once capture latches, the forward still waits for the one sink
            # admission in progress. Without a timeout that wait has no bound.
            raise ValueError(
                "failure_policy='disable_capture' needs a bounded sink "
                "admission: set capture_sink_config.admission_timeout_s, or "
                "overload='drop_newest'")

    def _start_capture_storage(self, *, sweep_spool: bool) -> Optional[Any]:
        config = self._capture_storage_config
        if config is None or self._storage_backend != "persistent":
            return None
        from .storage.native_capture import NativeCaptureStorage

        sink_config = self._capture_sink_config
        storage = NativeCaptureStorage(
            config,
            spool_root=sink_config.spool_root,
            spool_max_bytes=sink_config.spool_max_bytes,
            sweep_spool=sweep_spool,
        )
        storage.start()
        self._capture_storage = storage
        return storage

    def _attach_record_runtime(
        self,
        record_format: "RecordFormat[MetadataT]",
        record_schema: Any,
        record_sink: Optional[Any],
        *,
        failure_policy: str,
        step_stall_budget_ms: int,
    ) -> "RecordRuntime[MetadataT]":
        from .records import RecordRuntime

        ring_config = self._ring_config
        # D4's flip: the persistent backend's default writer is the native
        # pack sink, built from the config's bounds. An explicit
        # record_sink overrides it — the reference sink is the documented
        # rollback — and the ClickHouse host path is untouched.
        if (
            record_sink is None
            and self._storage_backend == "persistent"
            and self._capture_sink_config is not None
        ):
            from .storage.capture.native_sink import create_native_pack_sink

            record_sink = create_native_pack_sink(
                self._capture_sink_config
            ).native_sink

        _native_engine = _native_module()
        if record_sink is not None and not isinstance(
            record_sink, _native_engine.RecordSink
        ):
            raise TypeError("record_sink must be a native RecordSink")
        self._reject_a_sink_the_config_did_not_ask_for(record_sink)
        if record_sink is None and self._host_engine is not None:
            _native_engine._load_extension()._validate_record_host_schema(
                self._host_engine,
                record_schema,
            )
        sink_lease = (
            None if record_sink is None else record_sink._acquire_engine()
        )
        record_engine = None
        record_transport = None
        runtime = None
        switched = False
        try:
            _rt = _ring_module()
            if not self.capture_enabled:
                self.set_capture_enabled(True)
            ring_engine = getattr(self, "_ring_engine", None)
            if ring_engine is not None:
                ring_engine.stop()
            _rt.deactivate()
            self._ring_transport = None
            self._ring_engine = None
            switched = True

            sink_or_host = self._host_engine if record_sink is None else sink_lease
            record_engine = _native_engine.RingEngine.create_record(
                ring_config,
                sink_or_host,
                failure_policy=failure_policy,
                step_stall_budget_ms=step_stall_budget_ms,
            )
            record_engine.init()
            record_engine.start()
            record_transport = _rt.RingTransport(record_engine)
            runtime = RecordRuntime(record_transport, record_format)
            self._ring_engine = record_engine
            self._ring_transport = record_transport
            self._record_mode = True
            self._record_sink = record_sink
            _rt.activate(record_transport)
            return runtime
        except BaseException:
            if switched:
                try:
                    _rt.deactivate()
                except Exception:
                    pass
            if record_engine is not None:
                try:
                    record_engine.stop()
                except Exception:
                    pass
            if switched:
                self._ring_transport = None
                self._ring_engine = None
                self._record_mode = False
                self._record_sink = None
            # Drop every Python owner of the new native engine.  Its native
            # lease joins the record worker before releasing the sink.
            runtime = None
            record_transport = None
            record_engine = None
            raise

    def flush_and_wait(self, timeout_s: float = 600.0) -> None:
        """Complete the record ring and its configured sink durability boundary."""

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        transport = self._ring_transport
        if transport is None:
            raise RuntimeError("Ring transport is not enabled")
        if not self._record_mode:
            raise RuntimeError("Record runtime is not active")
        deadline = time.monotonic() + float(timeout_s)
        transport.flush_records_and_wait(float(timeout_s))
        # The sink's boundary is a staged pack; with the storage service it
        # is a pack in the catalog. Its flush runs one cycle even at zero.
        storage = self._capture_storage
        if storage is not None:
            storage.flush(max(0.0, deadline - time.monotonic()))

    def capture_status(self) -> dict[str, Any]:
        """The record runtime's capture state, for monitoring and RPCs.

        ``capture_active`` is False once a failure latched: under
        ``"disable_capture"`` the forward keeps running and ``failure``
        says why capture stopped; ``discarded_*`` count what was dropped
        after it. ``reserve_wait_s`` and ``max_step_wait_s`` are the time
        record reservations waited for the sink (in total, and in the
        worst step). ``sink`` and ``storage`` are the native pack sink's
        and storage service's snapshots, when the engine holds them.
        Without a record runtime every field is empty.
        """
        status: dict[str, Any] = {
            "record_mode": False, "capture_active": False,
            "failure_policy": None, "failure": None,
            "discarded_descriptors": 0, "discarded_payloads": 0,
            "step_stall_budget_ms": None, "stall_budget_exhaustions": 0,
            "reserve_wait_s": 0.0, "max_step_wait_s": 0.0,
            "sink": None, "storage": None,
        }
        ring_engine = getattr(self, "_ring_engine", None)
        if not self._record_mode or ring_engine is None:
            return status
        native = dict(ring_engine.capture_status())
        status.update(
            record_mode=True,
            capture_active=not native["failed"],
            failure_policy=native["failure_policy"],
            failure=native["failure"] or None,
            discarded_descriptors=int(native["discarded_descriptors"]),
            discarded_payloads=int(native["discarded_payloads"]),
            step_stall_budget_ms=int(native["step_stall_budget_ms"]) or None,
            stall_budget_exhaustions=int(native["stall_budget_exhaustions"]),
            reserve_wait_s=int(native["reserve_wait_ns"]) / 1e9,
            max_step_wait_s=int(native["max_step_wait_ns"]) / 1e9,
        )
        snapshot = getattr(self._record_sink, "snapshot", None)
        if callable(snapshot):
            status["sink"] = dict(snapshot())
        if self._capture_storage is not None:
            status["storage"] = dict(self._capture_storage.snapshot())
        return status

    def _report_capture_failure(self) -> None:
        """Log, once, why capture stopped, as a record ring is retired."""
        try:
            status = self.capture_status()
        except Exception as exc:
            _LOG.warning("capture status unavailable at close: %s", exc)
            return
        if not status["record_mode"] or status["capture_active"]:
            return
        _LOG.warning(
            "record capture stopped before close (failure_policy=%s): %s; "
            "%d descriptors and %d payloads discarded after it",
            status["failure_policy"], status["failure"],
            status["discarded_descriptors"], status["discarded_payloads"])

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

        Refused under ``storage_backend="none"``, which turns capture off.

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
        if getattr(self, "_storage_backend", "auto") == "none":
            raise RuntimeError(
                "config.storage_backend='none' turns capture off: this engine "
                "allocates no ring. Pick 'in-memory' or 'persistent' to capture"
            )
        _rt = _ring_module()
        _native_engine = _native_module()

        if self._ring_transport is not None:
            old_record_mode = self._record_mode
            # Native null mode is device-global rather than RingEngine-local.
            # Restore its default before destroying a disabled transport so a
            # replacement starts capture-enabled without an extra startup sync.
            if not self.capture_enabled:
                self.set_capture_enabled(True)
            # Replacing a record ring ends the runtime the storage service
            # drains, so retire it as close() does: seal the sink, then drain
            # and stop the service. Left running, it would keep the catalog
            # lease the next create_record_runtime's service must take.
            storage = self._capture_storage if old_record_mode else None
            drain_deadline = None if storage is None else (
                time.monotonic()
                + self._capture_storage_config.close_flush_timeout_s)
            if storage is not None:
                self._seal_capture_sink(drain_deadline)
            if old_record_mode:
                self._report_capture_failure()
            try:
                ring_engine = getattr(self, "_ring_engine", None)
                if ring_engine is not None:
                    ring_engine.stop()
            except Exception:
                if old_record_mode:
                    # A record sink remains leased while its worker may still
                    # call it. Preserve the transport so shutdown can retry.
                    raise
            try:
                _rt.deactivate()
            except Exception:
                pass
            self._ring_transport = None
            self._ring_engine = None
            self._record_mode = False
            self._record_sink = None
            if storage is not None:
                self._retire_capture_storage(storage, drain_deadline)

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
        self._ring_config = ring_config
        self._record_mode = False

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

    def _seal_capture_sink(self, deadline: float) -> None:
        """Flush the record sink before its ring stops.

        Stopping the ring releases the sink WITHOUT flushing it, so the
        records of its open pack would still be in memory while the service
        drains a spool that does not hold them yet.
        """
        try:
            self._ring_transport.flush_records_and_wait(
                max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            _LOG.warning("capture sink did not flush: %s", exc)

    def _retire_capture_storage(self, storage: Any, deadline: float) -> None:
        """Drain the storage service until ``deadline``, then stop it."""
        if self._capture_storage is not storage:
            return
        self._capture_storage = None
        # Best effort: once the sink is sealed, a pack that does not reach
        # the catalog here is still in the spool or the bucket, and the next
        # start uploads or reconciles it. flush_and_wait is the boundary
        # that reports.
        try:
            storage.flush(max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            _LOG.warning("capture storage did not drain: %s", exc)
        finally:
            storage.stop()

    def close(self) -> None:
        """Tear down backend resources."""

        storage = self._capture_storage
        # One budget for the whole capture drain: sealing the sink's open
        # pack, then getting everything staged into the catalog.
        drain_deadline = None if storage is None else (
            time.monotonic() + self._capture_storage_config.close_flush_timeout_s)
        if self._ring_transport is not None:
            record_mode = self._record_mode
            stopped = False
            # Best-effort reset of the device-global native null flag.  This is
            # needed only after callers explicitly disabled capture; the normal
            # HF path pays no extra synchronization cost.
            if not self.capture_enabled:
                try:
                    self.set_capture_enabled(True)
                except Exception:
                    pass
            if record_mode and storage is not None:
                self._seal_capture_sink(drain_deadline)
            if record_mode:
                self._report_capture_failure()
            try:
                ring_engine = getattr(self, "_ring_engine", None)
                if ring_engine is not None:
                    ring_engine.stop()
                stopped = True
            except Exception:
                pass
            # A record sink remains leased while a native worker may still be
            # alive. Leave the state intact so close can be retried.
            if record_mode and not stopped:
                return
            try:
                _rt = _ring_module()
                _rt.deactivate()
            except Exception:
                pass
            self._ring_transport = None
            self._ring_engine = None
            self._record_mode = False
            self._record_sink = None

        if storage is not None:
            self._retire_capture_storage(storage, drain_deadline)

        if self._host_engine is not None:
            try:
                self._host_engine.close_input()
                self._host_engine.stop()
            except Exception:
                pass
            self._host_engine = None


# ---------------------------------------------------------------------------
# Backend loader


__all__ = ["MonitoringEngine", "RECORD_FAILURE_POLICIES", "RingCapacities"]
