"""Byte-level format enforcement for dmi-pack v1.

Every test here corrupts a real sealed pack -- header, trailer, or footer JSON
-- and asserts the reader refuses it with a typed error rather than decoding
garbage. ``_reseal`` rewrites the footer and re-signs the footer CRC and body
hash, so each case fails on exactly the field it corrupts, not on a checksum.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import struct
from uuid import UUID
import zlib

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    PackFormatError,
    PackIndex,
    PackIntegrityError,
    PackReader,
    PackRef,
    PackWriter,
)
from dmi.storage.capture.pack import (
    _TRAILER,
    MAX_FOOTER_BYTES,
    MAX_RECORDS,
    PackCapacityError,
    verify_payload,
)


pytestmark = pytest.mark.cpu


PACK_ID = UUID("018f0000-0000-7000-8000-000000000001")


def _metadata(capture_id: str, *, step: int = 0) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=capture_id,
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id="request-a",
        sequence_id="sequence-a",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=3,
        producer_rank=0,
        step_number=step,
        token_start=step,
        token_end=step + 1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


def _record(capture_id: str, payload: bytes, *, step: int = 0) -> CaptureRecord:
    return CaptureRecord(metadata=_metadata(capture_id, step=step), payload=payload)


def _writer(**overrides) -> PackWriter:
    settings = dict(
        pack_id=PACK_ID,
        created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=1024 * 1024,
    )
    settings.update(overrides)
    return PackWriter(**settings)


def _sealed_pack(*records: CaptureRecord):
    writer = _writer()
    for record in records:
        writer.append(record)
    return writer.seal()


def _patched(data: bytes, offset: int, value: bytes) -> bytes:
    changed = bytearray(data)
    changed[offset : offset + len(value)] = value
    return bytes(changed)


def _reseal(data: bytes, *, footer: bytes | None = None, mutate=None) -> bytes:
    """Rewrite the footer of a sealed pack, re-signing CRC and body hash."""
    trailer_offset = len(data) - _TRAILER.size
    magic, major, minor, footer_offset, _, _, _ = _TRAILER.unpack_from(
        data, trailer_offset
    )
    if mutate is not None:
        decoded = json.loads(data[footer_offset:trailer_offset])
        mutate(decoded)
        footer = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()
    assert footer is not None
    body = data[:footer_offset] + footer
    trailer = _TRAILER.pack(
        magic,
        major,
        minor,
        footer_offset,
        len(footer),
        zlib.crc32(footer) & 0xFFFFFFFF,
        sha256(body).digest(),
    )
    return body + trailer


class _MemStore:
    store_id = "local"

    def __init__(self, data: bytes):
        self.data = data

    def read_range(self, ref, offset, length):
        return self.data[offset : offset + length]


def _mem_ref(data: bytes, *, checksum: str, record_count: int = 1) -> PackRef:
    return PackRef(
        pack_id=str(PACK_ID),
        store_id="local",
        object_key="packs/a.dmi-pack",
        object_bytes=len(data),
        checksum=checksum,
        record_count=record_count,
    )


# --- verify_payload -----------------------------------------------------------


def test_verify_payload_rejects_length_codec_and_decode_mismatches():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    descriptor = PackReader.from_bytes(sealed.data).descriptors(
        store_id="local", object_key="a"
    )[0]
    offset = descriptor.locator.offset
    payload = sealed.data[offset : offset + 8]

    with pytest.raises(PackIntegrityError, match="short record"):
        verify_payload(descriptor, payload[:-1])

    compressed = replace(descriptor, locator=replace(descriptor.locator, codec="zstd"))
    with pytest.raises(PackFormatError, match="unsupported codec"):
        verify_payload(compressed, payload)

    widened = replace(
        descriptor, locator=replace(descriptor.locator, decoded_length=16)
    )
    with pytest.raises(PackFormatError, match="equal stored and decoded"):
        verify_payload(widened, payload)


# --- PackWriter construction and lifecycle --------------------------------------


@pytest.mark.parametrize("bad_id", ("not-a-uuid", 123, None))
def test_pack_writer_requires_a_uuid_pack_id(bad_id):
    with pytest.raises(ValueError, match="pack_id"):
        _writer(pack_id=bad_id)


def test_pack_writer_validates_its_bounds():
    with pytest.raises(ValueError, match="created_at_ns"):
        _writer(created_at_ns=-1)
    with pytest.raises(ValueError, match="too small"):
        _writer(max_pack_bytes=64)
    with pytest.raises(ValueError, match="max_records"):
        _writer(max_records=0)
    with pytest.raises(ValueError, match="max_records"):
        _writer(max_records=MAX_RECORDS + 1)


def test_a_sealed_pack_accepts_no_further_operations():
    writer = _writer()
    writer.append(_record("capture-a", b"\x00" * 8))
    writer.seal()

    with pytest.raises(RuntimeError, match="already sealed"):
        writer.append(_record("capture-b", b"\x01" * 8, step=1))
    with pytest.raises(RuntimeError, match="already sealed"):
        writer.seal()


def test_pack_writer_enforces_max_records():
    writer = _writer(max_records=1)
    writer.append(_record("capture-a", b"\x00" * 8))

    with pytest.raises(PackCapacityError, match="record limit"):
        writer.append(_record("capture-b", b"\x01" * 8, step=1))


def test_pack_writer_refuses_to_seal_an_empty_pack():
    with pytest.raises(ValueError, match="empty pack"):
        _writer().seal()


def test_seal_rechecks_the_footer_limit(monkeypatch):
    writer = _writer()
    writer.append(_record("capture-a", b"\x00" * 8))
    monkeypatch.setattr("dmi.storage.capture.pack.MAX_FOOTER_BYTES", 8)

    with pytest.raises(ValueError, match="footer exceeds"):
        writer.seal()


def test_seal_rechecks_max_pack_bytes():
    writer = _writer()
    writer.append(_record("capture-a", b"\x00" * 8))
    # Shrink the budget after the append-time projection has passed, the way a
    # mis-accounted projection would look.
    writer._max_pack_bytes = 130

    with pytest.raises(ValueError, match="sealed pack exceeds"):
        writer.seal()


# --- PackReader.from_bytes: header and trailer fields ----------------------------


def test_from_bytes_rejects_a_short_buffer():
    with pytest.raises(PackFormatError, match="truncated"):
        PackReader.from_bytes(b"\x00" * 8)


def test_from_bytes_rejects_bad_header_fields():
    data = _sealed_pack(_record("capture-a", b"\x00" * 8)).data

    with pytest.raises(PackFormatError, match="invalid pack magic"):
        PackReader.from_bytes(_patched(data, 0, b"NOTAPACK"))
    with pytest.raises(PackFormatError, match="minor version"):
        PackReader.from_bytes(_patched(data, 10, struct.pack("<H", 9)))
    with pytest.raises(PackFormatError, match="header fields"):
        PackReader.from_bytes(_patched(data, 40, struct.pack("<I", 1)))


def test_from_bytes_rejects_bad_trailer_fields():
    data = _sealed_pack(_record("capture-a", b"\x00" * 8)).data
    trailer = len(data) - _TRAILER.size

    with pytest.raises(PackFormatError, match="invalid trailer"):
        PackReader.from_bytes(_patched(data, trailer, b"NOTAFTRX"))
    with pytest.raises(PackFormatError, match="versions differ"):
        PackReader.from_bytes(_patched(data, trailer + 10, struct.pack("<H", 1)))
    with pytest.raises(PackFormatError, match="footer exceeds"):
        PackReader.from_bytes(
            _patched(data, trailer + 20, struct.pack("<Q", MAX_FOOTER_BYTES + 1))
        )
    with pytest.raises(PackFormatError, match="footer range"):
        PackReader.from_bytes(_patched(data, trailer + 12, struct.pack("<Q", 0)))
    with pytest.raises(PackIntegrityError, match="footer checksum"):
        PackReader.from_bytes(_patched(data, trailer + 28, struct.pack("<I", 0)))


# --- PackReader.from_bytes: footer JSON matrix -----------------------------------


def _set(key, value):
    def mutate(decoded):
        decoded[key] = value

    return mutate


def _record_set(key, value):
    def mutate(decoded):
        decoded["records"][0][key] = value

    return mutate


def _record_del(key):
    def mutate(decoded):
        del decoded["records"][0][key]

    return mutate


def _record_shrink(decoded):
    decoded["records"][0]["stored_length"] = 4
    decoded["records"][0]["decoded_length"] = 4


@pytest.mark.parametrize(
    "mutate,match",
    (
        (_set("format", "zip"), "format marker"),
        (_set("major_version", 2), "version does not match the trailer"),
        (_set("records", 5), "invalid record list"),
        (_set("records", [5]), "entry must be an object"),
        (_record_del("offset"), "missing offset"),
        (_record_set("metadata", 5), "metadata must be an object"),
        (_record_set("offset", True), "must be integers"),
        (_record_set("stored_length", "8"), "must be integers"),
        (_record_set("offset", 0), "record range is invalid"),
        (_record_set("offset", 10**9), "extends into the footer"),
        (_record_set("codec", "zstd"), "uncompressed"),
        (_record_set("checksum", "nope"), "record checksum is invalid"),
        (_record_shrink, "does not match metadata"),
    ),
)
def test_footer_json_corruption_is_rejected(mutate, match):
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    with pytest.raises(PackFormatError, match=match):
        PackReader.from_bytes(_reseal(sealed.data, mutate=mutate))


def test_footer_must_be_a_json_object():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    with pytest.raises(PackFormatError, match="not valid JSON"):
        PackReader.from_bytes(_reseal(sealed.data, footer=b"{broken"))
    with pytest.raises(PackFormatError, match="format marker"):
        PackReader.from_bytes(_reseal(sealed.data, footer=b"[]"))


def test_footer_rejects_duplicate_and_out_of_order_records():
    sealed = _sealed_pack(
        _record("capture-a", b"\x00" * 8),
        _record("capture-b", b"\x01" * 8, step=1),
    )

    def duplicated(decoded):
        decoded["records"] = [decoded["records"][0], decoded["records"][0]]

    with pytest.raises(PackFormatError, match="duplicate capture ID"):
        PackReader.from_bytes(_reseal(sealed.data, mutate=duplicated))

    def reordered(decoded):
        decoded["records"] = decoded["records"][::-1]

    with pytest.raises(PackFormatError, match="overlap or are out of order"):
        PackReader.from_bytes(_reseal(sealed.data, mutate=reordered))


def test_footer_identity_must_match_the_header():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    def skewed(decoded):
        decoded["created_at_ns"] += 1

    with pytest.raises(PackFormatError, match="identity does not match the header"):
        PackReader.from_bytes(_reseal(sealed.data, mutate=skewed))


# --- PackIndex.from_store ---------------------------------------------------------


def test_from_store_rejects_foreign_refs_and_truncated_objects():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    store = _MemStore(sealed.data)
    ref = _mem_ref(sealed.data, checksum=sealed.checksum)

    with pytest.raises(ValueError, match="another store"):
        PackIndex.from_store(store, replace(ref, store_id="other"))
    with pytest.raises(PackFormatError, match="truncated"):
        PackIndex.from_store(store, replace(ref, object_bytes=16))


def test_from_store_validates_the_trailer():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    trailer = len(sealed.data) - _TRAILER.size

    def index(data: bytes):
        return PackIndex.from_store(
            _MemStore(data), _mem_ref(data, checksum=sealed.checksum)
        )

    with pytest.raises(PackFormatError, match="invalid trailer"):
        index(_patched(sealed.data, trailer, b"NOTAFTRX"))
    with pytest.raises(PackFormatError, match="unsupported pack version"):
        index(_patched(sealed.data, trailer + 8, struct.pack("<H", 99)))
    with pytest.raises(PackFormatError, match="footer exceeds"):
        index(
            _patched(
                sealed.data, trailer + 20, struct.pack("<Q", MAX_FOOTER_BYTES + 1)
            )
        )
    with pytest.raises(PackFormatError, match="footer range"):
        index(_patched(sealed.data, trailer + 12, struct.pack("<Q", 1)))


def test_from_store_validates_the_footer_identity():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    def index(data: bytes, record_count: int = 1):
        return PackIndex.from_store(
            _MemStore(data),
            _mem_ref(data, checksum=sealed.checksum, record_count=record_count),
        )

    with pytest.raises(PackFormatError, match="invalid pack ID"):
        index(_reseal(sealed.data, mutate=_set("pack_id", "not-a-uuid")))
    with pytest.raises(PackFormatError, match="does not match its object key"):
        index(
            _reseal(
                sealed.data,
                mutate=_set("pack_id", "018f0000-0000-7000-8000-00000000ffff"),
            )
        )
    with pytest.raises(PackFormatError, match="record count does not match"):
        index(sealed.data, record_count=5)


# --- PackReader.read_payload --------------------------------------------------------


def test_read_payload_rejects_foreign_and_drifted_descriptors():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    reader = PackReader.from_bytes(sealed.data)
    descriptor = reader.descriptors(store_id="local", object_key="a")[0]

    foreign = replace(
        descriptor,
        locator=replace(
            descriptor.locator, pack_id="018f0000-0000-7000-8000-00000000ffff"
        ),
    )
    with pytest.raises(PackFormatError, match="another pack"):
        reader.read_payload(foreign)

    absent = replace(
        descriptor, metadata=replace(descriptor.metadata, capture_id="capture-z")
    )
    with pytest.raises(PackFormatError, match="not present"):
        reader.read_payload(absent)

    drifted = replace(
        descriptor,
        locator=replace(descriptor.locator, offset=descriptor.locator.offset + 64),
    )
    with pytest.raises(PackFormatError, match="does not match the pack footer"):
        reader.read_payload(drifted)
