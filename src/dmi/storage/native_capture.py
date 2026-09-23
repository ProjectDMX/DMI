"""The native capture storage path after the spool, and its query side.

Under ``storage_backend="capture"`` the native pack sink stages immutable
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
from dataclasses import dataclass, field
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


def _positive(name: str, value: Any, kind: type) -> None:
    if type(value) is not kind and not (kind is float and type(value) is int):
        raise TypeError(f"{name} must be {kind.__name__}")
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
    # How long engine.close() waits for staged packs to reach the catalog.
    close_flush_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("s3_endpoint", "s3_bucket", "s3_access_key",
                     "s3_secret_key", "store_id", "database", "table_prefix",
                     "reconcile_prefix"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.s3_endpoint.startswith("http://") and not self.s3_allow_insecure_http:
            raise ValueError(
                "s3_endpoint is plain HTTP; set s3_allow_insecure_http=True "
                "to send credentials over it")
        if type(self.clickhouse_port) is not int or not 0 < self.clickhouse_port < 65536:
            raise ValueError("clickhouse_port must be in 1..65535")
        _positive("poll_interval_s", self.poll_interval_s, float)
        _positive("close_flush_timeout_s", self.close_flush_timeout_s, float)
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
            "database": self.database,
            "table_prefix": self.table_prefix,
        }


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
    "NativeCapture",
    "NativeCapturePage",
    "NativeCaptureReader",
    "NativeCaptureSelection",
    "NativeCaptureStorage",
    "NativeCaptureStorageConfig",
]
