from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import re
import struct
from typing import Iterable
from uuid import UUID
import zlib

from .model import (
    CaptureDescriptor,
    CaptureMetadata,
    CaptureRecord,
    DuplicateCaptureError,
    PackFormatError,
    PackIntegrityError,
    PackRef,
    PackStore,
    PayloadLocator,
)


PACK_MAJOR_VERSION = 1
PACK_MINOR_VERSION = 0
PACK_ALIGNMENT = 64
MAX_FOOTER_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 1_000_000

_HEADER_MAGIC = b"DMIPACK\0"
_TRAILER_MAGIC = b"DMIFTR\0\0"
_HEADER = struct.Struct("<8sHHI16sQI20s")
_TRAILER = struct.Struct("<8sHHQQI32s")
_CRC_PATTERN = re.compile(r"^[0-9a-f]{8}$")


class PackCapacityError(ValueError):
    pass


class _PackStateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SealedPack:
    pack_id: str
    created_at_ns: int
    data: bytes
    record_count: int
    footer_offset: int
    checksum: str

    @property
    def object_bytes(self) -> int:
        return len(self.data)

    def open(self) -> BytesIO:
        return BytesIO(self.data)


@dataclass(frozen=True, slots=True)
class _IndexedRecord:
    metadata: CaptureMetadata
    offset: int
    stored_length: int
    decoded_length: int
    codec: str
    checksum: str


def _align(buffer: bytearray) -> None:
    padding = (-len(buffer)) % PACK_ALIGNMENT
    if padding:
        buffer.extend(b"\0" * padding)


def _aligned_length(length: int) -> int:
    return length + (-length) % PACK_ALIGNMENT


def _encode_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _record_mapping(item: _IndexedRecord) -> dict[str, object]:
    return {
        "metadata": item.metadata.to_mapping(),
        "offset": item.offset,
        "stored_length": item.stored_length,
        "decoded_length": item.decoded_length,
        "codec": item.codec,
        "checksum": item.checksum,
    }


def _crc32(data: bytes | memoryview) -> str:
    return f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"


def verify_payload(descriptor: CaptureDescriptor, payload: bytes | memoryview) -> None:
    if len(payload) != descriptor.locator.stored_length:
        raise PackIntegrityError(
            f"short record for {descriptor.capture_id}: "
            f"{len(payload)} != {descriptor.locator.stored_length}"
        )
    if descriptor.locator.codec != "none":
        raise PackFormatError(f"unsupported codec: {descriptor.locator.codec}")
    if descriptor.locator.stored_length != descriptor.locator.decoded_length:
        raise PackFormatError("none codec requires equal stored and decoded lengths")
    if _crc32(payload) != descriptor.locator.checksum:
        raise PackIntegrityError(f"record checksum mismatch: {descriptor.capture_id}")


class PackWriter:
    def __init__(
        self,
        *,
        pack_id: UUID | str,
        created_at_ns: int,
        max_pack_bytes: int,
        max_records: int = MAX_RECORDS,
    ) -> None:
        try:
            parsed_id = pack_id if isinstance(pack_id, UUID) else UUID(pack_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("pack_id must be a UUID") from exc
        if created_at_ns < 0:
            raise ValueError("created_at_ns must be non-negative")
        if max_pack_bytes < _HEADER.size + _TRAILER.size + 2:
            raise ValueError("max_pack_bytes is too small for a pack")
        if not 1 <= max_records <= MAX_RECORDS:
            raise ValueError(f"max_records must be between 1 and {MAX_RECORDS}")

        self._uuid = parsed_id
        self._pack_id = str(parsed_id)
        self._created_at_ns = created_at_ns
        self._max_pack_bytes = max_pack_bytes
        self._max_records = max_records
        self._buffer = bytearray(
            _HEADER.pack(
                _HEADER_MAGIC,
                PACK_MAJOR_VERSION,
                PACK_MINOR_VERSION,
                _HEADER.size,
                parsed_id.bytes,
                created_at_ns,
                0,
                b"\0" * 20,
            )
        )
        self._records: list[_IndexedRecord] = []
        empty_footer = _encode_json(
            {
                "format": "dmi-pack",
                "major_version": PACK_MAJOR_VERSION,
                "minor_version": PACK_MINOR_VERSION,
                "pack_id": self._pack_id,
                "created_at_ns": self._created_at_ns,
                "records": [],
            }
        )
        self._footer_prefix, self._footer_suffix = empty_footer.rsplit(b"[]", 1)
        self._record_json: list[bytes] = []
        self._record_json_bytes = 0
        self._capture_ids: set[str] = set()
        self._sealed = False

    @property
    def record_count(self) -> int:
        return len(self._records)

    def append(self, record: CaptureRecord) -> None:
        if self._sealed:
            raise _PackStateError("pack is already sealed")
        if len(self._records) >= self._max_records:
            raise PackCapacityError("pack record limit reached")
        if record.metadata.capture_id in self._capture_ids:
            raise DuplicateCaptureError(
                f"duplicate capture ID: {record.metadata.capture_id}"
            )
        offset = _aligned_length(len(self._buffer))
        indexed = _IndexedRecord(
            metadata=record.metadata,
            offset=offset,
            stored_length=len(record.payload),
            decoded_length=len(record.payload),
            codec="none",
            checksum=_crc32(record.payload),
        )
        encoded_record = _encode_json(_record_mapping(indexed))
        record_content_bytes = self._record_json_bytes + len(encoded_record)
        record_separators = len(self._record_json)
        footer_length = (
            len(self._footer_prefix)
            + 2
            + record_content_bytes
            + record_separators
            + len(self._footer_suffix)
        )
        projected = (
            _aligned_length(offset + len(record.payload))
            + footer_length
            + _TRAILER.size
        )
        if footer_length > MAX_FOOTER_BYTES or projected > self._max_pack_bytes:
            raise PackCapacityError("record would exceed max_pack_bytes")

        _align(self._buffer)
        self._buffer.extend(record.payload)
        self._records.append(indexed)
        self._record_json.append(encoded_record)
        self._record_json_bytes += len(encoded_record)
        self._capture_ids.add(record.metadata.capture_id)

    def seal(self) -> SealedPack:
        if self._sealed:
            raise _PackStateError("pack is already sealed")
        if not self._records:
            raise ValueError("cannot seal an empty pack")
        buffer = bytearray(self._buffer)
        _align(buffer)
        footer_offset = len(buffer)
        footer = (
            self._footer_prefix
            + b"["
            + b",".join(self._record_json)
            + b"]"
            + self._footer_suffix
        )
        if len(footer) > MAX_FOOTER_BYTES:
            raise ValueError("pack footer exceeds its size limit")
        buffer.extend(footer)
        # One hash pass: the object checksum covers body || trailer, so the
        # body hasher is extended with the trailer instead of re-reading the
        # whole pack a second time.
        hasher = sha256(buffer)
        body_checksum = hasher.digest()
        trailer = _TRAILER.pack(
            _TRAILER_MAGIC,
            PACK_MAJOR_VERSION,
            PACK_MINOR_VERSION,
            footer_offset,
            len(footer),
            zlib.crc32(footer) & 0xFFFFFFFF,
            body_checksum,
        )
        buffer.extend(trailer)
        if len(buffer) > self._max_pack_bytes:
            raise ValueError("sealed pack exceeds max_pack_bytes")
        self._sealed = True
        self._buffer = buffer
        data = bytes(buffer)
        hasher.update(trailer)
        return SealedPack(
            pack_id=self._pack_id,
            created_at_ns=self._created_at_ns,
            data=data,
            record_count=len(self._records),
            footer_offset=footer_offset,
            checksum=hasher.hexdigest(),
        )


class PackIndex:
    def __init__(self, ref: PackRef, records: Iterable[_IndexedRecord]) -> None:
        self.ref = ref
        self.pack_id = ref.pack_id
        self._records = tuple(records)

    @staticmethod
    def trailer_size() -> int:
        return _TRAILER.size

    @classmethod
    def from_store(cls, store: PackStore, ref: PackRef) -> PackIndex:
        if ref.store_id != store.store_id:
            raise ValueError("pack reference belongs to another store")
        if ref.object_bytes < _HEADER.size + _TRAILER.size + 2:
            raise PackFormatError("pack is truncated")
        trailer_offset = ref.object_bytes - _TRAILER.size
        trailer_bytes = store.read_range(ref, trailer_offset, _TRAILER.size)
        try:
            trailer = _TRAILER.unpack(trailer_bytes)
        except struct.error as exc:
            raise PackFormatError("pack trailer is truncated") from exc
        magic, major, minor, footer_offset, footer_length, footer_crc, _ = trailer
        if magic != _TRAILER_MAGIC:
            raise PackFormatError("pack has an invalid trailer")
        if major != PACK_MAJOR_VERSION or minor > PACK_MINOR_VERSION:
            raise PackFormatError(f"unsupported pack version: {major}.{minor}")
        if footer_length > MAX_FOOTER_BYTES:
            raise PackFormatError("pack footer exceeds its size limit")
        if footer_offset < _HEADER.size or footer_offset + footer_length != trailer_offset:
            raise PackFormatError("pack footer range is invalid")
        footer = store.read_range(ref, footer_offset, footer_length)
        if zlib.crc32(footer) & 0xFFFFFFFF != footer_crc:
            raise PackIntegrityError("footer checksum mismatch")
        decoded = _decode_footer(footer, major=major, minor=minor)
        try:
            pack_id = str(UUID(str(decoded.get("pack_id"))))
        except (ValueError, TypeError, AttributeError) as exc:
            raise PackFormatError("pack footer has an invalid pack ID") from exc
        if pack_id != ref.pack_id:
            raise PackFormatError("pack footer identity does not match its object key")
        records = _parse_records(decoded, footer_offset)
        if len(records) != ref.record_count:
            raise PackFormatError("pack record count does not match its object metadata")
        return cls(ref, records)

    def descriptors(self) -> tuple[CaptureDescriptor, ...]:
        return _descriptors(self.ref, self._records)


class PackReader:
    def __init__(
        self,
        data: bytes,
        *,
        pack_id: str,
        created_at_ns: int,
        records: Iterable[_IndexedRecord],
        object_checksum: str,
    ) -> None:
        self._data = data
        self.pack_id = pack_id
        self.created_at_ns = created_at_ns
        self._records = tuple(records)
        self._by_capture_id = {item.metadata.capture_id: item for item in self._records}
        self.object_checksum = object_checksum

    @classmethod
    def from_bytes(cls, value: bytes | bytearray | memoryview) -> PackReader:
        data = bytes(value)
        minimum = _HEADER.size + _TRAILER.size + 2
        if len(data) < minimum:
            raise PackFormatError("pack is truncated")

        try:
            header = _HEADER.unpack_from(data)
        except struct.error as exc:  # pragma: no cover - unreachable after the length check
            raise PackFormatError("pack header is truncated") from exc
        magic, major, minor, header_size, pack_bytes, created_at_ns, flags, reserved = header
        if magic != _HEADER_MAGIC:
            raise PackFormatError("invalid pack magic")
        if major != PACK_MAJOR_VERSION:
            raise PackFormatError(f"unsupported pack major version: {major}")
        if minor > PACK_MINOR_VERSION:
            raise PackFormatError(f"unsupported pack minor version: {minor}")
        if header_size != _HEADER.size or flags != 0 or reserved != b"\0" * 20:
            raise PackFormatError("invalid pack header fields")

        trailer_offset = len(data) - _TRAILER.size
        try:
            trailer = _TRAILER.unpack_from(data, trailer_offset)
        except struct.error as exc:  # pragma: no cover - unreachable after the length check
            raise PackFormatError("pack trailer is truncated") from exc
        (
            trailer_magic,
            trailer_major,
            trailer_minor,
            footer_offset,
            footer_length,
            footer_crc,
            body_hash,
        ) = trailer
        if trailer_magic != _TRAILER_MAGIC:
            raise PackFormatError("pack is truncated or has an invalid trailer")
        if (trailer_major, trailer_minor) != (major, minor):
            raise PackFormatError("pack header and trailer versions differ")
        if footer_length > MAX_FOOTER_BYTES:
            raise PackFormatError("pack footer exceeds its size limit")
        if footer_offset < _HEADER.size or footer_offset + footer_length != trailer_offset:
            raise PackFormatError("pack footer range is invalid")
        # memoryview avoids copying the whole body for the hash, and the
        # object checksum (body || trailer) reuses the body hasher instead of
        # re-reading the pack from offset zero.
        view = memoryview(data)
        hasher = sha256(view[:trailer_offset])
        if hasher.digest() != body_hash:
            raise PackIntegrityError("pack checksum mismatch")
        footer = data[footer_offset:trailer_offset]
        if zlib.crc32(footer) & 0xFFFFFFFF != footer_crc:
            raise PackIntegrityError("footer checksum mismatch")

        decoded = _decode_footer(footer, major=major, minor=minor)
        pack_id = str(UUID(bytes=pack_bytes))
        if decoded.get("pack_id") != pack_id or decoded.get("created_at_ns") != created_at_ns:
            raise PackFormatError("pack footer identity does not match the header")
        records = _parse_records(decoded, footer_offset)

        hasher.update(view[trailer_offset:])
        return cls(
            data,
            pack_id=pack_id,
            created_at_ns=created_at_ns,
            records=records,
            object_checksum=hasher.hexdigest(),
        )

    @staticmethod
    def _parse_record(raw: object, footer_offset: int) -> _IndexedRecord:
        if not isinstance(raw, dict):
            raise PackFormatError("pack record entry must be an object")
        try:
            metadata_raw = raw["metadata"]
            offset = raw["offset"]
            stored_length = raw["stored_length"]
            decoded_length = raw["decoded_length"]
            codec = raw["codec"]
            checksum = raw["checksum"]
        except KeyError as exc:
            raise PackFormatError(f"pack record is missing {exc.args[0]}") from exc
        if not isinstance(metadata_raw, dict):
            raise PackFormatError("pack record metadata must be an object")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (offset, stored_length, decoded_length)
        ):
            raise PackFormatError("pack record offsets and lengths must be integers")
        if offset < _HEADER.size or stored_length < 0 or decoded_length < 0:
            raise PackFormatError("pack record range is invalid")
        if offset + stored_length > footer_offset:
            raise PackFormatError("pack record extends into the footer")
        if codec != "none" or stored_length != decoded_length:
            raise PackFormatError("dmi-pack-v1 supports only uncompressed records")
        if not isinstance(checksum, str) or _CRC_PATTERN.fullmatch(checksum) is None:
            raise PackFormatError("pack record checksum is invalid")
        metadata = CaptureMetadata.from_mapping(metadata_raw)
        if metadata.logical_bytes != decoded_length:
            raise PackFormatError("record length does not match metadata dtype and shape")
        return _IndexedRecord(
            metadata=metadata,
            offset=offset,
            stored_length=stored_length,
            decoded_length=decoded_length,
            codec=codec,
            checksum=checksum,
        )

    def descriptors(self, *, store_id: str, object_key: str) -> tuple[CaptureDescriptor, ...]:
        return _descriptors(
            PackRef(
                pack_id=self.pack_id,
                store_id=store_id,
                object_key=object_key,
                object_bytes=len(self._data),
                checksum=self.object_checksum,
                record_count=len(self._records),
            ),
            self._records,
        )

    def read_payload(self, descriptor: CaptureDescriptor) -> bytes:
        if descriptor.locator.pack_id != self.pack_id:
            raise PackFormatError("descriptor belongs to another pack")
        indexed = self._by_capture_id.get(descriptor.capture_id)
        if indexed is None:
            raise PackFormatError("descriptor is not present in this pack")
        expected = (
            indexed.offset,
            indexed.stored_length,
            indexed.decoded_length,
            indexed.codec,
            indexed.checksum,
        )
        actual = (
            descriptor.locator.offset,
            descriptor.locator.stored_length,
            descriptor.locator.decoded_length,
            descriptor.locator.codec,
            descriptor.locator.checksum,
        )
        if actual != expected:
            raise PackFormatError("descriptor does not match the pack footer")
        payload = self._data[indexed.offset:indexed.offset + indexed.stored_length]
        verify_payload(descriptor, payload)
        return payload


def _decode_footer(footer: bytes, *, major: int, minor: int) -> dict[str, object]:
    try:
        decoded = json.loads(footer)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackFormatError("pack footer is not valid JSON") from exc
    if not isinstance(decoded, dict) or decoded.get("format") != "dmi-pack":
        raise PackFormatError("pack footer has an invalid format marker")
    if decoded.get("major_version") != major or decoded.get("minor_version") != minor:
        raise PackFormatError("pack footer version does not match the trailer")
    return decoded


def _parse_records(decoded: dict[str, object], footer_offset: int) -> tuple[_IndexedRecord, ...]:
    raw_records = decoded.get("records")
    if not isinstance(raw_records, list) or len(raw_records) > MAX_RECORDS:
        raise PackFormatError("pack footer has an invalid record list")
    records: list[_IndexedRecord] = []
    previous_end = _HEADER.size
    seen: set[str] = set()
    for raw in raw_records:
        record = PackReader._parse_record(raw, footer_offset)
        if record.metadata.capture_id in seen:
            raise PackFormatError(f"duplicate capture ID: {record.metadata.capture_id}")
        if record.offset < previous_end:
            raise PackFormatError("pack record ranges overlap or are out of order")
        previous_end = record.offset + record.stored_length
        seen.add(record.metadata.capture_id)
        records.append(record)
    return tuple(records)


def _descriptors(
    ref: PackRef, records: Iterable[_IndexedRecord]
) -> tuple[CaptureDescriptor, ...]:
    records = tuple(records)
    return tuple(
        CaptureDescriptor(
            metadata=item.metadata,
            locator=PayloadLocator(
                pack_id=ref.pack_id,
                store_id=ref.store_id,
                object_key=ref.object_key,
                object_bytes=ref.object_bytes,
                pack_checksum=ref.checksum,
                pack_record_count=ref.record_count,
                offset=item.offset,
                stored_length=item.stored_length,
                decoded_length=item.decoded_length,
                codec=item.codec,
                checksum=item.checksum,
            ),
        )
        for item in records
    )
