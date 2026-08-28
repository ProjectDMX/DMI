from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import BinaryIO, Mapping, Protocol, Sequence, runtime_checkable



_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}
_TEXT_LIMIT = 512
# A keyset cursor carries the whole catalog sort key, so it cannot fit the
# per-identifier text limit that bounds the fields it encodes.
_CURSOR_LIMIT = 2048
# Fields of CaptureQuery that address a page rather than describe the filters.
_NON_FILTER_QUERY_FIELDS = frozenset({"cursor", "limit"})
_MAX_RANK = 32


class CaptureStorageError(Exception):
    """Base error for capture-pack storage."""


class PackFormatError(CaptureStorageError):
    """A pack or catalog descriptor violates the format contract."""


class PackIntegrityError(CaptureStorageError):
    """Stored bytes do not match their integrity metadata."""


class PackConflictError(CaptureStorageError):
    """An immutable object key already contains different bytes."""


class DuplicateCaptureError(CaptureStorageError):
    """A logical selection contains a capture more than once."""


class HydrationLimitError(CaptureStorageError):
    """A hydration plan exceeds the caller's byte limit."""


class InvalidCursorError(CaptureStorageError):
    """A pagination cursor is malformed or belongs to a different query."""


def _validate_text(
    name: str, value: str | None, *, optional: bool = False, limit: int = _TEXT_LIMIT
) -> None:
    if value is None:
        if optional:
            return
        raise ValueError(f"{name} is required")
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > limit:
        raise ValueError(f"{name} must be non-empty UTF-8 within {limit} bytes")


@dataclass(frozen=True, slots=True)
class CaptureMetadata:
    capture_id: str
    tenant_id: str
    experiment_id: str
    run_id: str
    session_id: str
    request_id: str
    sequence_id: str
    model_id: str
    model_revision: str
    adapter_revision: str | None
    capture_policy_version: str
    hook_name: str
    layer_number: int
    producer_rank: int
    step_number: int
    token_start: int
    token_end: int
    batch_position: int
    dtype: str
    shape: tuple[int, ...]
    captured_at_ns: int

    def __post_init__(self) -> None:
        required = (
            "capture_id",
            "tenant_id",
            "experiment_id",
            "run_id",
            "session_id",
            "request_id",
            "sequence_id",
            "model_id",
            "model_revision",
            "capture_policy_version",
            "hook_name",
        )
        for name in required:
            _validate_text(name, getattr(self, name))
        _validate_text("adapter_revision", self.adapter_revision, optional=True)
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype!r}")
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))
        if len(self.shape) > _MAX_RANK:
            raise ValueError(f"shape rank must not exceed {_MAX_RANK}")
        if any(type(dim) is not int or dim < 0 or dim > 2**31 - 1 for dim in self.shape):
            raise ValueError("shape dimensions must be integers in [0, 2^31 - 1]")
        # Upper bounds mirror where these fields ultimately land: the catalog
        # stores producer_rank/batch_position as UInt32, the counters and
        # captured_at_ns as UInt64, and layer_number as Int32, while the pack
        # header packs created_at_ns (= captured_at_ns) into a uint64 struct
        # field. Rejecting out-of-range values here keeps one poison record
        # from failing the persistence thread or wedging catalog indexing.
        bounded = (
            ("producer_rank", 2**32 - 1),
            ("batch_position", 2**32 - 1),
            ("step_number", 2**64 - 1),
            ("token_start", 2**64 - 1),
            ("token_end", 2**64 - 1),
            ("captured_at_ns", 2**64 - 1),
        )
        for name, upper in bounded:
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > upper:
                raise ValueError(
                    f"{name} must be a non-negative integer <= {upper}"
                )
        if (
            type(self.layer_number) is not int
            or self.layer_number < -1
            or self.layer_number > 2**31 - 1
        ):
            raise ValueError("layer_number must be an integer in [-1, 2^31 - 1]")
        if self.token_end < self.token_start:
            raise ValueError("token_end must be >= token_start")

    @property
    def logical_bytes(self) -> int:
        return math.prod(self.shape) * _DTYPE_BYTES[self.dtype]

    def to_mapping(self) -> dict[str, object]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CaptureMetadata:
        fields = cls.__dataclass_fields__
        missing = [name for name in fields if name not in value]
        if missing:
            raise PackFormatError("capture metadata is missing: " + ", ".join(missing))
        selected = {name: value[name] for name in fields}
        shape = selected["shape"]
        if not isinstance(shape, list) or not all(type(dim) is int for dim in shape):
            raise PackFormatError("capture shape must be an integer list")
        selected["shape"] = tuple(shape)
        try:
            return cls(**selected)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise PackFormatError(f"invalid capture metadata: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    metadata: CaptureMetadata
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            object.__setattr__(self, "payload", bytes(self.payload))
        if len(self.payload) != self.metadata.logical_bytes:
            raise ValueError(
                "payload length does not match dtype and shape: "
                f"{len(self.payload)} != {self.metadata.logical_bytes}"
            )


@dataclass(frozen=True, slots=True)
class PackRef:
    pack_id: str
    store_id: str
    object_key: str
    object_bytes: int
    checksum: str
    record_count: int


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    size: int
    checksum: str


@dataclass(frozen=True, slots=True)
class StoredObject:
    object_key: str
    object_bytes: int


@dataclass(frozen=True, slots=True)
class ObjectPage:
    items: tuple[StoredObject, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class PayloadLocator:
    pack_id: str
    store_id: str
    object_key: str
    object_bytes: int
    pack_checksum: str
    pack_record_count: int
    offset: int
    stored_length: int
    decoded_length: int
    codec: str
    checksum: str

    @property
    def pack_ref(self) -> PackRef:
        return PackRef(
            pack_id=self.pack_id,
            store_id=self.store_id,
            object_key=self.object_key,
            object_bytes=self.object_bytes,
            checksum=self.pack_checksum,
            record_count=self.pack_record_count,
        )


@dataclass(frozen=True, slots=True)
class CaptureDescriptor:
    metadata: CaptureMetadata
    locator: PayloadLocator

    @property
    def capture_id(self) -> str:
        return self.metadata.capture_id


@dataclass(frozen=True, slots=True)
class CaptureQuery:
    tenant_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    model_id: str | None = None
    hook_names: tuple[str, ...] = ()
    layer_numbers: tuple[int, ...] = ()
    captured_after_ns: int | None = None
    captured_before_ns: int | None = None
    cursor: str | None = None
    limit: int = 1000

    def __post_init__(self) -> None:
        for name in ("tenant_id", "experiment_id", "run_id", "session_id", "model_id"):
            value = getattr(self, name)
            if value is not None:
                _validate_text(name, value)
        if self.cursor is not None:
            _validate_text("cursor", self.cursor, limit=_CURSOR_LIMIT)
        if not 1 <= self.limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if len(self.hook_names) > 128 or len(self.layer_numbers) > 1024:
            raise ValueError("query filters exceed their bounded cardinality")
        for hook_name in self.hook_names:
            _validate_text("hook_name", hook_name)
        if any(layer < -1 for layer in self.layer_numbers):
            raise ValueError("layer numbers must be >= -1")
        for name in ("captured_after_ns", "captured_before_ns"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.captured_after_ns is not None
            and self.captured_before_ns is not None
            and self.captured_before_ns < self.captured_after_ns
        ):
            raise ValueError("captured_before_ns must be >= captured_after_ns")

    @property
    def query_hash(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    @property
    def filter_hash(self) -> str:
        """Identity of the filters alone, excluding cursor and page size.

        ``query_hash`` covers the whole request, so it changes from page to
        page. Cursors and selections bind to this instead, which is what makes
        a paginated walk one identifiable query.
        """
        selected = {
            name: value
            for name, value in asdict(self).items()
            if name not in _NON_FILTER_QUERY_FIELDS
        }
        encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CapturePage:
    items: tuple[CaptureDescriptor, ...]
    next_cursor: str | None
    watermark: str


@dataclass(frozen=True, slots=True)
class CaptureSelection:
    selection_id: str
    capture_ids: tuple[str, ...]
    catalog_watermark: str
    filter_hash: str

    @classmethod
    def create(
        cls,
        descriptors: Sequence[CaptureDescriptor],
        *,
        catalog_watermark: str,
        filter_hash: str,
    ) -> CaptureSelection:
        ids = tuple(item.capture_id for item in descriptors)
        seen: set[str] = set()
        for capture_id in ids:
            if capture_id in seen:
                raise DuplicateCaptureError(
                    f"duplicate logical capture: {capture_id}"
                )
            seen.add(capture_id)
        identity = json.dumps(
            {
                "version": 2,
                "catalog_watermark": catalog_watermark,
                "filter_hash": filter_hash,
                "capture_ids": ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return cls(
            selection_id=sha256(identity).hexdigest(),
            capture_ids=ids,
            catalog_watermark=catalog_watermark,
            filter_hash=filter_hash,
        )


@dataclass(frozen=True, slots=True)
class HydrationEstimate:
    capture_count: int
    object_count: int
    request_count: int
    logical_bytes: int
    stored_bytes: int
    request_bytes: int

    @property
    def read_amplification(self) -> float:
        return self.request_bytes / self.stored_bytes if self.stored_bytes else 0.0


@dataclass(frozen=True, slots=True)
class HydratedCapture:
    descriptor: CaptureDescriptor
    payload: bytes

    @property
    def capture_id(self) -> str:
        return self.descriptor.capture_id


@runtime_checkable
class PackSource(Protocol):
    pack_id: str
    created_at_ns: int
    record_count: int
    checksum: str

    @property
    def object_bytes(self) -> int: ...
    def open(self) -> BinaryIO: ...


@runtime_checkable
class PackStore(Protocol):
    store_id: str

    def put(self, pack: PackSource, object_key: str) -> PackRef: ...
    def stat(self, ref: PackRef) -> ObjectInfo: ...
    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes: ...


@runtime_checkable
class CaptureCatalog(Protocol):
    def search(self, query: CaptureQuery) -> CapturePage: ...
    def get_by_ids(
        self, capture_ids: Sequence[str], *, watermark: str
    ) -> Sequence[CaptureDescriptor]: ...
