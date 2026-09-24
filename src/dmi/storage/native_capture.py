"""The native capture storage path after the spool, and its query side.

Under ``storage_backend="persistent"`` the native pack sink stages immutable
packs in a local spool (``capture_sink_config``). Setting
``capture_storage_config`` as well makes the engine run a
:class:`NativeCaptureStorage` in-process: one C++ thread that uploads each
staged pack to the object store and indexes it into the ClickHouse catalog,
where :class:`NativeCaptureReader` finds and reads it back::

    ring -> NativePackSink -> spool -> upload -> object store
                                          `-> index -> ClickHouse catalog
                                                            |
                          NativeCaptureReader.search/select/read

No Python runs on any of it; this module configures, starts and queries the
C++ implementation in ``_dmi_native_store``. The Python implementation in
:mod:`dmi.storage.capture` is the reference that path is validated against,
not a dependency of it.

Deployment shape: the catalog has ONE publisher lease per (``database``,
``table_prefix``), so run one capture process per catalog. A second engine
on the same catalog fails at ``create_record_runtime`` with the lease held.
"""

from __future__ import annotations

import json
import math
import os
import socket
import uuid
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Optional, Sequence


# The catalog columns an item carries as integers. Everything else stays the
# catalog's text, except ``shape``, which is parsed into a tuple.
_INTEGER_COLUMNS = frozenset({
    "layer_number", "producer_rank", "step_number", "token_start",
    "token_end", "batch_position", "captured_at_ns", "object_bytes",
    "pack_record_count", "payload_offset", "stored_length", "decoded_length",
})


def _load_native_store_extension() -> Any:
    from ..transport import native as _native_transport

    try:
        return _native_transport._load_named_extension("_dmi_native_store")
    except ImportError as exc:
        raise ImportError(
            "The native capture storage path is unavailable. Build it with "
            "`make -C native build/_dmi_native_store "
            "PYTHON=<venv>/bin/python`."
        ) from exc


# The sink's admission policies (native/csrc/sink/pack_sink.h Overload).
SINK_OVERLOAD_POLICIES = ("block", "drop_newest")


@dataclass(frozen=True, slots=True)
class NativeSinkConfig:
    """Bounds and admission policy for the native pack sink.

    The sink runs on the ring's record worker: every record the forward
    captures is admitted here, into a bounded queue ahead of pack assembly.
    ``overload`` decides what a full queue does. ``"block"`` waits for room
    for up to ``admission_timeout_s`` (``None`` waits without bound) and
    then refuses the record as timed out; the rows of one ring record (one
    envelope) share that one deadline. ``"drop_newest"`` refuses it at
    once, and ``admission_timeout_s`` is not used. Either refusal latches
    the record runtime (see ``create_record_runtime``'s failure policy),
    and so does a record lost after admission. A ``step_stall_budget_ms``
    needs a bounded admission, so it is refused with block and no timeout.

    The default, block with 2 s, absorbs a burst larger than the queue at
    the cost of stalling the record worker, and so the ring, while the sink
    catches up. The C++ ``SinkConfig`` keeps drop_newest with no timeout,
    the reference pipeline's default; this config is what the ring-fed sink
    is built from.

    A record larger than ``max_queue_bytes`` or ``max_pack_bytes`` can
    never be admitted; ``validate_capture_bounds`` refuses such a bound at
    attach, before any forward runs.
    """

    spool_root: str
    spool_max_bytes: int = 1 << 40
    num_workers: int = 1
    max_queue_records: int = 256
    max_queue_bytes: int = 16 * 1024 * 1024
    max_pack_bytes: int = 128 * 1024 * 1024
    max_pack_records: int = 10_000
    max_linger_ns: int = 1_000_000_000
    overload: str = "block"
    admission_timeout_s: Optional[float] = 2.0

    def __post_init__(self) -> None:
        if not self.spool_root:
            raise ValueError("spool_root is required")
        for name in (
            "spool_max_bytes",
            "num_workers",
            "max_queue_records",
            "max_queue_bytes",
            "max_pack_bytes",
            "max_pack_records",
            "max_linger_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.overload not in SINK_OVERLOAD_POLICIES:
            raise ValueError(
                f"overload must be one of {SINK_OVERLOAD_POLICIES}, "
                f"got {self.overload!r}")
        if self.admission_timeout_s is not None:
            _positive("admission_timeout_s", self.admission_timeout_s, float)


def _native_sink_config_setstate(self: NativeSinkConfig, state: Any) -> None:
    # A frozen slots dataclass pickles its field values as a list, and the
    # generated __setstate__ zips them with the current fields, leaving any
    # the pickle predates unset. A config pickled before the admission
    # fields existed (the old class, at the path that now re-exports this
    # one) gets their defaults instead.
    names = fields(NativeSinkConfig)
    if len(state) > len(names):
        raise TypeError(
            f"NativeSinkConfig state has {len(state)} values, expected at "
            f"most {len(names)}")
    for item, value in zip(names, state):
        object.__setattr__(self, item.name, value)
    for item in names[len(state):]:
        if item.default is MISSING:
            raise TypeError(
                f"NativeSinkConfig state is missing {item.name!r}")
        object.__setattr__(self, item.name, item.default)


NativeSinkConfig.__setstate__ = _native_sink_config_setstate  # type: ignore[method-assign]


def _positive(name: str, value: Any, kind: type) -> None:
    if type(value) is not kind and not (kind is float and type(value) is int):
        raise TypeError(f"{name} must be {kind.__name__}")
    # NaN passes every comparison, and inf overflows the native nanoseconds.
    if kind is float and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class NativeCaptureStorageConfig:
    """Where staged packs go: the object store and the ClickHouse catalog.

    The spool is not configured here; the engine drains the one
    ``capture_sink_config.spool_root`` names, so the two cannot disagree.
    """

    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str = field(repr=False)
    s3_secret_key: str = field(repr=False)
    s3_region: str = "us-east-1"
    s3_session_token: str = field(default="", repr=False)
    # Plain-HTTP endpoints (a local Garage or MinIO) must be opted into.
    s3_allow_insecure_http: bool = False
    # The name packs are indexed under; readers resolve it to this store.
    store_id: str = "s3"

    clickhouse_host: str = "127.0.0.1"
    clickhouse_port: int = 8123  # the HTTP interface
    # Every catalog request is bounded, so a server that stops answering
    # cannot hold a flush, a publish or the lease renewal indefinitely.
    clickhouse_connect_timeout_s: float = 10.0
    clickhouse_request_timeout_s: float = 60.0
    database: str = "default"
    table_prefix: str = "dmi"

    # The publisher lease holder. Empty picks "<host>:<pid>:<random>".
    holder: str = ""
    poll_interval_s: float = 0.5
    # Index packs a crashed process uploaded but never indexed. Object keys
    # name no catalog, so every pack under ``reconcile_prefix`` is indexed
    # into THIS catalog: the prefix (by default the whole bucket) must belong
    # to one catalog. Set reconcile_on_start=False to share a bucket.
    reconcile_on_start: bool = True
    reconcile_prefix: str = "v1/"
    # 0 disables the periodic pass; in-process index failures are retried
    # regardless.
    reconcile_interval_s: float = 0.0
    # engine.close()'s total budget for draining capture: sealing the sink's
    # open pack, then getting every staged pack into the catalog.
    close_flush_timeout_s: float = 60.0
    # Bytes of packs the uploader holds in flight at once. A staged pack
    # larger than this is never uploaded, so the sink's max_pack_bytes must
    # not exceed it (validate_capture_bounds). The native default.
    uploader_max_in_flight_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("s3_endpoint", "s3_bucket", "s3_access_key",
                     "s3_secret_key", "store_id", "database", "table_prefix",
                     "reconcile_prefix"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        # The native client reads a scheme-less endpoint as plain HTTP, and
        # refuses https with the insecure flag, at every request rather than
        # here -- so both are refused here.
        if self.s3_endpoint.startswith("http://"):
            if not self.s3_allow_insecure_http:
                raise ValueError(
                    "s3_endpoint is plain HTTP; set s3_allow_insecure_http=True "
                    "to send credentials over it")
        elif self.s3_endpoint.startswith("https://"):
            if self.s3_allow_insecure_http:
                raise ValueError(
                    "s3_allow_insecure_http admits plain http:// endpoints and "
                    "never downgrades TLS; leave it False for https://")
        else:
            raise ValueError("s3_endpoint must start with http:// or https://")
        if type(self.clickhouse_port) is not int or not 0 < self.clickhouse_port < 65536:
            raise ValueError("clickhouse_port must be in 1..65535")
        _positive("poll_interval_s", self.poll_interval_s, float)
        # The native wait is int(poll_interval_s * 1e9) ns; a zero wait spins.
        if self.poll_interval_s < 0.001:
            raise ValueError("poll_interval_s must be at least 0.001")
        _positive("clickhouse_connect_timeout_s",
                  self.clickhouse_connect_timeout_s, float)
        _positive("clickhouse_request_timeout_s",
                  self.clickhouse_request_timeout_s, float)
        _positive("close_flush_timeout_s", self.close_flush_timeout_s, float)
        if (type(self.uploader_max_in_flight_bytes) is not int
                or self.uploader_max_in_flight_bytes <= 0):
            raise ValueError("uploader_max_in_flight_bytes must be positive")
        if type(self.reconcile_interval_s) not in (int, float):
            raise TypeError("reconcile_interval_s must be float")
        if not math.isfinite(self.reconcile_interval_s):
            raise ValueError("reconcile_interval_s must be finite")
        if self.reconcile_interval_s < 0:
            raise ValueError("reconcile_interval_s must be non-negative")

    def _native_dict(self) -> dict[str, Any]:
        return {
            "s3_endpoint": self.s3_endpoint,
            "s3_bucket": self.s3_bucket,
            "s3_region": self.s3_region,
            "s3_access_key": self.s3_access_key,
            "s3_secret_key": self.s3_secret_key,
            "s3_session_token": self.s3_session_token,
            "s3_allow_insecure_http": self.s3_allow_insecure_http,
            "store_id": self.store_id,
            "clickhouse_host": self.clickhouse_host,
            "clickhouse_port": self.clickhouse_port,
            "clickhouse_connect_timeout_s": float(self.clickhouse_connect_timeout_s),
            "clickhouse_request_timeout_s": float(self.clickhouse_request_timeout_s),
            "database": self.database,
            "table_prefix": self.table_prefix,
            "uploader_max_in_flight_bytes": self.uploader_max_in_flight_bytes,
        }


# What an empty pack needs besides one record's payload: the 64-byte header
# and trailer, up to 63 bytes of payload alignment, and that record's footer
# row (its metadata as JSON, well under a kilobyte for real identifiers).
# Generous on purpose: undershooting it lets a record through that the pack
# worker then counts as oversized and drops.
PACK_FRAMING_RESERVE_BYTES = 64 * 1024


def validate_capture_bounds(
    sink_config: NativeSinkConfig,
    max_record_bytes: int,
    *,
    storage_config: Optional[NativeCaptureStorageConfig] = None,
) -> None:
    """Refuse capture bounds under which a record or pack cannot be stored.

    ``max_record_bytes`` is the largest single record (one captured row's
    payload) the caller will emit. Checked here, at attach, these refusals
    never reach the forward:

    - ``max_queue_bytes`` below it: the sink refuses the record outright.
    - ``max_pack_bytes`` below it plus ``PACK_FRAMING_RESERVE_BYTES``: the
      sink admits the record and the pack worker then drops it as oversized.
    - ``max_pack_bytes`` above ``uploader_max_in_flight_bytes``: a full pack
      is staged and never uploaded.

    Raises ``ConfigurationError`` naming the bound to raise.
    """
    if not isinstance(sink_config, NativeSinkConfig):
        raise TypeError("sink_config must be a NativeSinkConfig")
    if type(max_record_bytes) is not int or max_record_bytes <= 0:
        raise ValueError("max_record_bytes must be a positive int")
    from ..configuration.errors import ConfigurationError

    if sink_config.max_queue_bytes < max_record_bytes:
        raise ConfigurationError(
            f"a {max_record_bytes}-byte record exceeds the sink's "
            f"max_queue_bytes ({sink_config.max_queue_bytes}); raise "
            "max_queue_bytes to at least the largest record")
    needed = max_record_bytes + PACK_FRAMING_RESERVE_BYTES
    if sink_config.max_pack_bytes < needed:
        raise ConfigurationError(
            f"a {max_record_bytes}-byte record does not fit an empty pack of "
            f"max_pack_bytes ({sink_config.max_pack_bytes}); raise "
            f"max_pack_bytes to at least {needed} (the record plus "
            f"{PACK_FRAMING_RESERVE_BYTES} bytes of pack framing)")
    if (storage_config is not None and sink_config.max_pack_bytes
            > storage_config.uploader_max_in_flight_bytes):
        raise ConfigurationError(
            f"the sink's max_pack_bytes ({sink_config.max_pack_bytes}) "
            "exceeds the storage service's uploader_max_in_flight_bytes "
            f"({storage_config.uploader_max_in_flight_bytes}), so a full "
            "pack would never be uploaded; lower max_pack_bytes or raise "
            "uploader_max_in_flight_bytes")


class NativeCaptureStorage:
    """The in-process storage service: spool -> object store -> catalog."""

    def __init__(
        self,
        config: NativeCaptureStorageConfig,
        *,
        spool_root: str,
        spool_max_bytes: int,
        sweep_spool: bool,
    ) -> None:
        if not isinstance(config, NativeCaptureStorageConfig):
            raise TypeError("config must be a NativeCaptureStorageConfig")
        module = _load_native_store_extension()
        native = config._native_dict()
        native.update(
            spool_root=spool_root,
            spool_max_bytes=spool_max_bytes,
            holder=config.holder or (
                f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"),
            poll_interval_ns=int(config.poll_interval_s * 1e9),
            reconcile_on_start=config.reconcile_on_start,
            reconcile_prefix=config.reconcile_prefix,
            reconcile_interval_ns=int(config.reconcile_interval_s * 1e9),
            sweep_spool_on_start=sweep_spool,
        )
        self._config = config
        self._service = module.StorageService(native)

    def start(self) -> None:
        """Sweep the spool, ensure the catalog schema, take the lease."""
        self._service.start()

    def flush(self, timeout_s: float) -> None:
        """Wait until everything staged so far is in the catalog.

        Call after the sink's own flush. Raises TimeoutError, carrying the
        last upload or index error, if the spool has not drained in time.
        """
        if not self._service.flush(float(timeout_s)):
            snapshot = self._service.snapshot()
            raise TimeoutError(
                "timed out waiting for staged packs to reach the catalog "
                f"({snapshot['pending_index']} uploaded but unindexed); last "
                f"error: {snapshot['last_error'] or 'none'}")

    def stop(self) -> None:
        """Stop the background thread and release the lease. No flush."""
        self._service.stop()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._service.snapshot())

    def rethrow_if_failed(self) -> None:
        self._service.rethrow_if_failed()


@dataclass(frozen=True, slots=True)
class NativeCapturePage:
    items: tuple[dict[str, Any], ...]
    watermark: str
    next_cursor: Optional[str]


@dataclass(frozen=True, slots=True)
class NativeCaptureSelection:
    """A bounded, catalog-pinned set of captures to read."""

    selection_id: str
    capture_ids: tuple[str, ...]
    catalog_watermark: str
    filter_hash: str
    tenant_id: str

    def _native_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "capture_ids": list(self.capture_ids),
            "catalog_watermark": self.catalog_watermark,
            "filter_hash": self.filter_hash,
            "tenant_id": self.tenant_id,
        }


@dataclass(frozen=True, slots=True)
class NativeCapture:
    """One capture: its catalog descriptor and verified payload bytes."""

    descriptor: dict[str, Any]
    payload: bytes

    def tensor(self) -> Any:
        """The payload as a CPU torch tensor of the captured dtype and shape."""
        import torch

        dtype = getattr(torch, self.descriptor["dtype"])
        shape = self.descriptor["shape"]
        if math.prod(shape) == 0:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(self.payload), dtype=dtype).reshape(shape)


class NativeCaptureReader:
    """Search the catalog and read captures back through the native reader."""

    def __init__(
        self,
        config: NativeCaptureStorageConfig,
        *,
        max_coalesce_gap_bytes: int = 4096,
    ) -> None:
        if not isinstance(config, NativeCaptureStorageConfig):
            raise TypeError("config must be a NativeCaptureStorageConfig")
        module = _load_native_store_extension()
        native = config._native_dict()
        native["max_coalesce_gap_bytes"] = max_coalesce_gap_bytes
        self._columns: tuple[str, ...] = tuple(module.SEARCH_ITEM_COLUMNS)
        self._reader = module.CaptureReader(native)

    @staticmethod
    def _filters(
        *,
        tenant_id: Optional[str],
        experiment_id: Optional[str],
        run_id: Optional[str],
        session_id: Optional[str],
        model_id: Optional[str],
        hook_names: Optional[Sequence[str]],
        layer_numbers: Optional[Sequence[int]],
        captured_after_ns: Optional[int],
        captured_before_ns: Optional[int],
        limit: Optional[int],
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "tenant_id": tenant_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "session_id": session_id,
            "model_id": model_id,
            "hook_names": None if hook_names is None else list(hook_names),
            "layer_numbers": (
                None if layer_numbers is None else list(layer_numbers)),
            "captured_after_ns": captured_after_ns,
            "captured_before_ns": captured_before_ns,
            "limit": limit,
            "cursor": cursor,
        }

    def _item(self, row: Sequence[str]) -> dict[str, Any]:
        item: dict[str, Any] = dict(zip(self._columns, row, strict=True))
        for name in _INTEGER_COLUMNS:
            item[name] = int(item[name])
        item["shape"] = tuple(json.loads(item["shape"]))
        return item

    def search(
        self,
        *,
        tenant_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model_id: Optional[str] = None,
        hook_names: Optional[Sequence[str]] = None,
        layer_numbers: Optional[Sequence[int]] = None,
        captured_after_ns: Optional[int] = None,
        captured_before_ns: Optional[int] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> NativeCapturePage:
        """One page of descriptors, in catalog order; walk ``next_cursor``."""
        page = self._reader.search(self._filters(
            tenant_id=tenant_id, experiment_id=experiment_id, run_id=run_id,
            session_id=session_id, model_id=model_id, hook_names=hook_names,
            layer_numbers=layer_numbers, captured_after_ns=captured_after_ns,
            captured_before_ns=captured_before_ns, limit=limit, cursor=cursor))
        return NativeCapturePage(
            items=tuple(self._item(row) for row in page["items"]),
            watermark=page["watermark"],
            next_cursor=page["next_cursor"],
        )

    def select(
        self,
        *,
        tenant_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model_id: Optional[str] = None,
        hook_names: Optional[Sequence[str]] = None,
        layer_numbers: Optional[Sequence[int]] = None,
        captured_after_ns: Optional[int] = None,
        captured_before_ns: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> NativeCaptureSelection:
        """Pin one bounded page as a selection; refuses more than a page."""
        raw = self._reader.select(self._filters(
            tenant_id=tenant_id, experiment_id=experiment_id, run_id=run_id,
            session_id=session_id, model_id=model_id, hook_names=hook_names,
            layer_numbers=layer_numbers, captured_after_ns=captured_after_ns,
            captured_before_ns=captured_before_ns, limit=limit))
        return NativeCaptureSelection(
            selection_id=raw["selection_id"],
            capture_ids=tuple(raw["capture_ids"]),
            catalog_watermark=raw["catalog_watermark"],
            filter_hash=raw["filter_hash"],
            tenant_id=raw["tenant_id"],
        )

    def read(
        self,
        selection: NativeCaptureSelection,
        *,
        byte_limit: int,
        request_limit: int = 1024,
    ) -> tuple[NativeCapture, ...]:
        """Fetch a selection's payloads, verified against each pack footer.

        Descriptors resolve at the selection's watermark, so a pack indexed
        after ``select`` cannot change what this returns.
        """
        if not isinstance(selection, NativeCaptureSelection):
            raise TypeError("selection must be a NativeCaptureSelection")
        # Refused before resolve's catalog query rather than after it.
        for name, value in (("byte_limit", byte_limit),
                            ("request_limit", request_limit)):
            if type(value) is not int:
                raise TypeError(f"{name} must be int")
        if byte_limit < 0:
            raise ValueError("byte_limit must be non-negative")
        if request_limit <= 0:
            raise ValueError("request_limit must be positive")
        native = selection._native_dict()
        rows = self._reader.resolve(native)
        payloads = self._reader.hydrate(native, byte_limit, request_limit)
        if len(rows) != len(payloads):
            raise RuntimeError(  # pragma: no cover - native invariant
                "native reader returned mismatched descriptors and payloads")
        return tuple(
            NativeCapture(descriptor=self._item(row), payload=payload)
            for row, payload in zip(rows, payloads)
        )


__all__ = [
    "PACK_FRAMING_RESERVE_BYTES",
    "SINK_OVERLOAD_POLICIES",
    "NativeSinkConfig",
    "NativeCapture",
    "NativeCapturePage",
    "NativeCaptureReader",
    "NativeCaptureSelection",
    "NativeCaptureStorage",
    "NativeCaptureStorageConfig",
    "validate_capture_bounds",
]
