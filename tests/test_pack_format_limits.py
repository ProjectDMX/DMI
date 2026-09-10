"""The pack format at its ceilings — refused loudly, never misread.

Validation at `CaptureMetadata` construction is the boundary: out-of-range
counters, shapes and dtypes are refused before a poison record reaches a
pack, and the footer capacity check bounds the metadata×records product at
write time. These tests pin the boundaries and one fp8/uint16/uint32 round
trip through the reference writer and reader.
"""

import uuid

import pytest

from dmi.storage.capture.model import CaptureMetadata, CaptureRecord
from dmi.storage.capture.pack import (
    PackReader,
    PackWriter,
)

sys_path_guard = None  # (pythonpath is configured; no manual sys.path here)


def _meta(**overrides):
    fields = dict(
        capture_id="c", tenant_id="t", experiment_id="e", run_id="r",
        session_id="s", request_id="q", sequence_id="n", model_id="m",
        model_revision="mr", adapter_revision=None,
        capture_policy_version="v", hook_name="h", layer_number=0,
        producer_rank=0, step_number=0, token_start=0, token_end=1,
        batch_position=0, dtype="uint8", shape=(4,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    fields.update(overrides)
    return CaptureMetadata(**fields)


def test_rank_32_shape_round_trips():
    """Rank 32 is the ceiling and must round trip through a pack."""
    meta = _meta(shape=(1,) * 32, dtype="uint8")
    assert meta.logical_bytes == 1
    pack_id = str(uuid.uuid4())
    writer = PackWriter(pack_id=pack_id, created_at_ns=1, max_pack_bytes=1 << 20)
    record = CaptureRecord(metadata=meta, payload=b"\x2a")
    writer.append(record)
    sealed = writer.seal()
    packed = PackReader.from_bytes(sealed.data)
    (descriptor,) = packed.descriptors(store_id="s", object_key="k")
    assert descriptor.metadata.shape == (1,) * 32


def test_rank_33_is_refused():
    with pytest.raises(ValueError, match="rank must not exceed 32"):
        _meta(shape=(1,) * 33)


def test_dim_boundaries():
    with pytest.raises(ValueError, match=r"2\^31 - 1"):
        _meta(shape=(2**31,))
    meta = _meta(shape=(2**31 - 1,))
    assert meta.shape == (2**31 - 1,)


def test_producer_rank_is_uint32_bounded():
    meta = _meta(producer_rank=2**32 - 1)
    assert meta.producer_rank == 2**32 - 1
    with pytest.raises(ValueError, match="producer_rank"):
        _meta(producer_rank=2**32)


def test_fp8_and_wide_int_dtypes_round_trip():
    """fp8 and the wide integers are validated, packed and read back."""
    payloads = {
        "float8_e4m3fn": bytes([0x00, 0x80, 0x38, 0xB8, 0x7F]),
        "float8_e5m2": bytes([0x3C, 0x7C, 0x7E, 0x01, 0xC0]),
        "uint16": (1234).to_bytes(2, "little") + (5678).to_bytes(2, "little"),
        "uint32": (2**32 - 1).to_bytes(4, "little"),
    }
    for dtype, payload in payloads.items():
        elements = len(payload) // 1 if "8" in dtype.split("_")[-1][:1] or dtype in ("float8_e4m3fn", "float8_e5m2", "uint16", "uint32") else len(payload)
        if dtype == "uint16":
            elements = 2
        elif dtype == "uint32":
            elements = 1
        else:
            elements = len(payload)
        meta = _meta(dtype=dtype, shape=(elements,))
        assert meta.logical_bytes == len(payload)
        pack_id = str(uuid.uuid4())
        writer = PackWriter(
            pack_id=pack_id, created_at_ns=1, max_pack_bytes=1 << 20)
        writer.append(CaptureRecord(metadata=meta, payload=payload))
        sealed = writer.seal()
        packed = PackReader.from_bytes(sealed.data)
        (descriptor,) = packed.descriptors(store_id="s", object_key="k")
        assert descriptor.metadata.dtype == dtype
        assert descriptor.metadata.logical_bytes == len(payload)


def test_footer_capacity_is_loud_at_write():
    """The metadata×records product is bounded at write, not at read.

    The footer carries one JSON row per record under a 64 MiB limit; a
    run that exceeds it gets PackCapacityError from the appends that
    crossed the line, not a pack that cannot be read back.
    """
    import zlib

    pack_id = str(uuid.uuid4())
    # Text fields are bounded at 512 bytes; a max-size model_revision
    # inflates each footer row enough to cross the 64 MiB footer bound
    # within a bounded number of appends.
    meta = _meta(model_revision="x" * 512, shape=(1,))
    writer = PackWriter(
        pack_id=pack_id, created_at_ns=1, max_pack_bytes=1 << 30,
        max_records=1_000_000)
    crossed = 0
    with pytest.raises(Exception) as excinfo:
        for index in range(200_000):
            record = CaptureRecord(
                metadata=_meta(model_revision="x" * 512, shape=(1,),
                               step_number=index, capture_id=f"c{index}"),
                payload=b"\x00")
            writer.append(record)
            crossed = index
    assert crossed > 1000, "the bound should not trip on the first few rows"
    # The reference writer names its capacity refusal (PackCapacityError);
    # any storage-layer capacity exception is acceptable, but it must name
    # itself rather than being a bare overflow.
    assert excinfo.value.__class__.__name__ in (
        "PackCapacityError", "CapacityError"), excinfo.value
