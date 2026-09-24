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
import re
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


# A bare host name or IPv4 address, or a bracketed IPv6 address. Anything
# else would change what the URL the native client builds means.
_BARE_HOST = re.compile(r"[A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\]")

# The native catalog writer's publish timeout (WriterConfig
# publish_timeout_ns), which this config does not expose: a publish may run
# that long server-side before the server gives up on it.
_PUBLISH_TIMEOUT_S = 5.0


def _text(name: str, value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be str")
    # Credentials travel as HTTP headers; a line break would end the header.
    if any(c in value for c in "\r\n\x00"):
        raise ValueError(f"{name} must not contain CR, LF or NUL")


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

    clickhouse_host: str = "127.0.0.1"  # a bare host: no scheme, port or user
    clickhouse_port: int = 8123  # the HTTP interface (8443 for its TLS port)
    # "https" always verifies the server's certificate and name, against the
    # system roots plus clickhouse_ca_file (a PEM bundle) or
    # clickhouse_ca_path (an OpenSSL hashed directory) for a private CA.
    clickhouse_scheme: str = "http"
    # Sent as X-ClickHouse-User / X-ClickHouse-Key headers, never in a URL.
    # Empty: no credentials, which ClickHouse reads as its `default` user.
    clickhouse_user: str = ""
    clickhouse_password: str = field(default="", repr=False)
    clickhouse_ca_file: str = ""
    clickhouse_ca_path: str = ""
    # A password over plain http must be opted into, as for s3.
    clickhouse_allow_insecure_http: bool = False
    # An optional separate account for NativeCaptureReader, typically one
    # granted SELECT only. Empty: the reader uses clickhouse_user.
    clickhouse_reader_user: str = ""
    clickhouse_reader_password: str = field(default="", repr=False)
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
        # A publish runs server-side for up to the publish timeout. A client
        # that gives up first reports an outcome it does not know, and the
        # writer quarantines itself over a statement that may have committed.
        if self.clickhouse_request_timeout_s < 2 * _PUBLISH_TIMEOUT_S:
            raise ValueError(
                "clickhouse_request_timeout_s must be at least "
                f"{2 * _PUBLISH_TIMEOUT_S:g} s: twice the catalog's "
                f"{_PUBLISH_TIMEOUT_S:g} s publish timeout, so a publish is "
                "never abandoned while the server may still commit it")
        self._validate_clickhouse_connection()
        _positive("close_flush_timeout_s", self.close_flush_timeout_s, float)
        if type(self.reconcile_interval_s) not in (int, float):
            raise TypeError("reconcile_interval_s must be float")
        if not math.isfinite(self.reconcile_interval_s):
            raise ValueError("reconcile_interval_s must be finite")
        if self.reconcile_interval_s < 0:
            raise ValueError("reconcile_interval_s must be non-negative")

    def _validate_clickhouse_connection(self) -> None:
        """What the native client refuses, refused here with field names."""
        for name in ("clickhouse_scheme", "clickhouse_host", "clickhouse_user",
                     "clickhouse_password", "clickhouse_ca_file",
                     "clickhouse_ca_path", "clickhouse_reader_user",
                     "clickhouse_reader_password"):
            _text(name, getattr(self, name))
        if type(self.clickhouse_allow_insecure_http) is not bool:
            raise TypeError("clickhouse_allow_insecure_http must be bool")
        if self.clickhouse_scheme not in ("http", "https"):
            raise ValueError('clickhouse_scheme must be "http" or "https"')
        # The value is never repeated: the likeliest mistake here is a URL
        # with a password in it.
        if "@" in self.clickhouse_host:
            raise ValueError(
                "clickhouse_host must not carry userinfo (user:password@); "
                "set clickhouse_user and clickhouse_password instead")
        if _BARE_HOST.fullmatch(self.clickhouse_host) is None:
            raise ValueError(
                "clickhouse_host must be a bare host name or address; the "
                "scheme and port have their own fields")
        for user, password in (
                ("clickhouse_user", "clickhouse_password"),
                ("clickhouse_reader_user", "clickhouse_reader_password")):
            if getattr(self, password) and not getattr(self, user):
                raise ValueError(f"{password} needs {user}: name the account "
                                 "it belongs to")
        if self.clickhouse_scheme == "http":
            for name in ("clickhouse_ca_file", "clickhouse_ca_path"):
                if getattr(self, name):
                    raise ValueError(
                        f"{name} needs clickhouse_scheme='https'; over http "
                        "it would verify nothing")
            if ((self.clickhouse_password or self.clickhouse_reader_password)
                    and not self.clickhouse_allow_insecure_http):
                raise ValueError(
                    "a ClickHouse password over plain http is refused: set "
                    "clickhouse_scheme='https', or clickhouse_allow_insecure_http"
                    "=True to send it in the clear")
        elif self.clickhouse_allow_insecure_http:
            raise ValueError(
                "clickhouse_allow_insecure_http admits plain http and never "
                "downgrades TLS; leave it False for https")

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
            "clickhouse_scheme": self.clickhouse_scheme,
            "clickhouse_host": self.clickhouse_host,
            "clickhouse_port": self.clickhouse_port,
            "clickhouse_user": self.clickhouse_user,
            "clickhouse_password": self.clickhouse_password,
            "clickhouse_ca_file": self.clickhouse_ca_file,
            "clickhouse_ca_path": self.clickhouse_ca_path,
            "clickhouse_allow_insecure_http": self.clickhouse_allow_insecure_http,
            "clickhouse_connect_timeout_s": float(self.clickhouse_connect_timeout_s),
            "clickhouse_request_timeout_s": float(self.clickhouse_request_timeout_s),
            "database": self.database,
            "table_prefix": self.table_prefix,
        }

    def _native_reader_dict(self) -> dict[str, Any]:
        """The reader's native config: the reader account, when one is set."""
        native = self._native_dict()
        if self.clickhouse_reader_user:
            native["clickhouse_user"] = self.clickhouse_reader_user
            native["clickhouse_password"] = self.clickhouse_reader_password
        return native


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
        native = config._native_reader_dict()
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
    "NativeCapture",
    "NativeCapturePage",
    "NativeCaptureReader",
    "NativeCaptureSelection",
    "NativeCaptureStorage",
    "NativeCaptureStorageConfig",
]
