"""A5b: the torch adapter — envelopes to packs without Python on the path.

Drives NativePackSink (built as _dmi_native_sink, torch CPU only) with
synthetic envelopes: real torch CPU tensors paired with metadata JSON in
the capture_pack_reference_v1 layout. Pinned: layout/cell validation,
dtype mapping for all ten dtypes, slice bounds and alignment, submission,
flush durability, failure surfacing, and the staged bytes readable by the
Python PackReader.

Build: make -C native build/_dmi_native_sink PYTHON=<venv>/bin/python
"""

from __future__ import annotations

import base64
import glob
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MATCHES = sorted((REPO_ROOT / "native" / "build").glob("_dmi_native_sink*.so"))

pytestmark = pytest.mark.cpu

# Module-level skip, not per-test: this module imports the built .so
# below, which must not execute at collection when the build is absent.
if not MATCHES:
    pytest.skip(
        "native/build/_dmi_native_sink*.so is not built; run "
        "`make -C native build/_dmi_native_sink PYTHON=<venv>/bin/python`",
        allow_module_level=True,
    )

import torch  # noqa: E402

sys.path.insert(0, str(MATCHES[0].parent))
import _dmi_native_sink as native_sink  # noqa: E402

from dmi.storage.capture import (  # noqa: E402
    CaptureMetadata,
    DurablePackSpool,
    PackReader,
)

# at::ScalarType numeric values (c10/core/ScalarType.h — stable ABI).
ATEN = {
    "bool": 11,
    "uint8": 0,
    "int8": 1,
    "int16": 2,
    "float16": 5,
    "bfloat16": 15,
    "int32": 3,
    "float32": 6,
    "int64": 4,
    "float64": 7,
}
TORCH_DTYPE = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "float32": torch.float32,
    "int64": torch.int64,
    "float64": torch.float64,
}
WIDTH = {"bool": 1, "uint8": 1, "int8": 1, "int16": 2, "float16": 2,
         "bfloat16": 2, "int32": 4, "float32": 4, "int64": 8, "float64": 8}

LAYOUT = "capture_pack_reference_v1"


def _meta(index: int, dtype: str = "float32", session: str = "s") -> dict:
    meta = CaptureMetadata(
        capture_id=f"adapter-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id=session, request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v", hook_name="h",
        layer_number=0, producer_rank=0, step_number=index,
        token_start=index, token_end=index + 1, batch_position=0,
        dtype=dtype, shape=(16,), captured_at_ns=1_700_000_000_000_000_000,
    )
    return json.loads(json.dumps(meta.to_mapping()))


def _row(meta_json: dict, offset: int, length: int | None, dtype: str,
         shape=(16,)) -> dict:
    return {
        "metadata_json": json.dumps(meta_json),
        "offset": offset,
        "length": length,
        "dtype": ATEN[dtype],
        "shape": list(shape),
    }


def _make_sink(tmp_path, **overrides):
    kwargs = {"spool_root": str(tmp_path), "layout": LAYOUT}
    kwargs.update(overrides)
    sink = native_sink.NativePackSink(**kwargs)
    lease = sink.attach()
    return sink, lease


def test_envelopes_become_packs(tmp_path):
    sink, lease = _make_sink(tmp_path)
    payload = torch.arange(32, dtype=torch.float32)
    sink.submit_envelope(
        LAYOUT,
        [_row(_meta(0), 0, 64, "float32"),
         _row(_meta(1), 64, 64, "float32")],
        payload,
    )
    assert sink.flush_and_wait(30.0)
    sink.rethrow_if_failed()
    snapshot = sink.snapshot()
    assert snapshot["admitted_records"] == 2
    assert snapshot["persisted_records"] == 2
    assert snapshot["failures"] == 0
    spool = DurablePackSpool(tmp_path, max_bytes=1 << 40)
    recovered = spool.recover()
    assert len(recovered) == 1
    with recovered[0].open() as handle:
        descriptors = PackReader.from_bytes(handle.read()).descriptors(
            store_id="spool", object_key=recovered[0].object_key
        )
    assert [d.metadata.capture_id for d in descriptors] == \
        ["adapter-0000", "adapter-0001"]


@pytest.mark.parametrize("dtype", sorted(ATEN))
def test_all_dtypes_map(tmp_path, dtype):
    sink, lease = _make_sink(tmp_path)
    width = WIDTH[dtype]
    payload = (torch.arange(16, dtype=torch.int64) % 2).to(TORCH_DTYPE[dtype])
    if dtype == "bool":
        payload = payload.to(torch.bool)
    sink.submit_envelope(
        LAYOUT, [_row(_meta(0, dtype=dtype), 0, 16 * width, dtype)], payload,
    )
    assert sink.flush_and_wait(30.0)
    assert sink.snapshot()["persisted_records"] == 1


def test_layout_mismatch_is_refused(tmp_path):
    sink, lease = _make_sink(tmp_path)
    with pytest.raises(RuntimeError, match="layout"):
        sink.submit_envelope(
            "some_other_layout",
            [_row(_meta(0), 0, None, "float32")],
            torch.zeros(16, dtype=torch.float32),
        )


def test_cell_mistypes_are_refused(tmp_path):
    sink, lease = _make_sink(tmp_path)
    payload = torch.zeros(16, dtype=torch.float32)
    # Not exactly two cells is a binding-level shape; here the row dict
    # misses the slice half, which the adapter must refuse.
    with pytest.raises(Exception):
        sink.submit_envelope(
            LAYOUT,
            [{"metadata_json": json.dumps(_meta(0))}],
            payload,
        )


def test_dtype_mismatch_is_refused(tmp_path):
    sink, lease = _make_sink(tmp_path)
    payload = torch.zeros(16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="[Dd]type"):
        sink.submit_envelope(
            LAYOUT, [_row(_meta(0, dtype="int32"), 0, 64, "float32")],
            payload,
        )


def test_shape_mismatch_is_refused(tmp_path):
    sink, lease = _make_sink(tmp_path)
    payload = torch.zeros(16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="[Ss]hape"):
        sink.submit_envelope(
            LAYOUT, [_row(_meta(0), 0, 64, "float32", shape=(8,))],
            payload,
        )


def test_slice_overrun_is_refused(tmp_path):
    sink, lease = _make_sink(tmp_path)
    payload = torch.zeros(16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="[Ss]lice|exceed"):
        sink.submit_envelope(
            LAYOUT, [_row(_meta(0), 0, 64 + 4, "float32")],
            payload,
        )


def test_unaligned_offset_is_refused(tmp_path):
    sink, lease = _make_sink(tmp_path)
    payload = torch.zeros(17, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="[Aa]lign"):
        sink.submit_envelope(
            LAYOUT, [_row(_meta(0), 1, 64, "float32")],
            payload,
        )


def test_non_cpu_tensor_is_refused(tmp_path):
    sink, lease = _make_sink(tmp_path)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device for the non-CPU case")
    payload = torch.zeros(16, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="CPU"):
        sink.submit_envelope(
            LAYOUT, [_row(_meta(0), 0, None, "float32")],
            payload,
        )


def test_submit_without_lease_is_refused(tmp_path):
    sink = native_sink.NativePackSink(spool_root=str(tmp_path), layout=LAYOUT)
    with pytest.raises(RuntimeError, match="[Aa]ttach"):
        sink.submit_envelope(
            LAYOUT, [_row(_meta(0), 0, None, "float32")],
            torch.zeros(16, dtype=torch.float32),
        )
    with pytest.raises(RuntimeError, match="[Aa]ttach"):
        sink.flush_and_wait(1.0)


# --- the torch sink boundary: scalars, numeric metadata, and the ring base ---


def _read_staged(tmp_path):
    spool = DurablePackSpool(tmp_path, max_bytes=1 << 40)
    recovered = spool.recover()
    assert len(recovered) == 1, recovered
    with recovered[0].open() as handle:
        reader = PackReader.from_bytes(handle.read())
        descriptors = reader.descriptors(
            store_id="spool", object_key=recovered[0].object_key
        )
        return [(d.metadata, reader.read_payload(d)) for d in descriptors]


def test_a_scalar_envelope_is_admitted(tmp_path):
    """shape=[] is a valid rank-0 capture: one element of dtype width.

    It was refused at admission -- "capture shape must be an integer list" --
    because the shape parser split the empty array into one empty item.
    """
    meta = _meta(0)
    meta["shape"] = []
    CaptureMetadata.from_mapping(meta)  # valid scalar on the Python side
    sink, lease = _make_sink(tmp_path)
    sink.submit_envelope(
        LAYOUT, [_row(meta, 0, 4, "float32", shape=())],
        torch.tensor(3.5, dtype=torch.float32),
    )
    assert sink.flush_and_wait(30.0)
    sink.rethrow_if_failed()
    assert sink.snapshot()["persisted_records"] == 1
    ((staged, payload),) = _read_staged(tmp_path)
    assert staged.shape == ()
    assert payload == torch.tensor(3.5, dtype=torch.float32).numpy().tobytes()


@pytest.mark.parametrize("field, value", [
    ("layer_number", 0.5),
    ("captured_at_ns", 123.5),
    ("step_number", -1),
])
def test_invalid_numeric_metadata_is_refused_not_converted(tmp_path, field,
                                                            value):
    """What CaptureMetadata refuses, the sink refuses -- nothing is persisted.

    These used to be persisted as 0, 123 and 18446744073709551615.
    """
    from dmi.storage.capture.model import CaptureStorageError

    meta = _meta(0)
    meta[field] = value
    with pytest.raises((CaptureStorageError, ValueError, TypeError)):
        CaptureMetadata.from_mapping(meta)
    sink, lease = _make_sink(tmp_path)
    with pytest.raises(RuntimeError, match="metadata"):
        sink.submit_envelope(
            LAYOUT, [_row(meta, 0, 64, "float32")],
            torch.zeros(16, dtype=torch.float32),
        )
    assert sink.snapshot()["submitted_records"] == 0


def test_a_fractional_shape_dimension_is_refused_not_truncated(tmp_path):
    from dmi.storage.capture.model import CaptureStorageError

    meta = _meta(0)
    meta["shape"] = [16.5]
    with pytest.raises((CaptureStorageError, ValueError, TypeError)):
        CaptureMetadata.from_mapping(meta)
    sink, lease = _make_sink(tmp_path)
    with pytest.raises(RuntimeError, match="metadata"):
        sink.submit_envelope(
            LAYOUT, [_row(meta, 0, 64, "float32", shape=(16,))],
            torch.zeros(16, dtype=torch.float32),
        )
    assert sink.snapshot()["submitted_records"] == 0


def test_a_non_bmp_hook_name_round_trips(tmp_path):
    """json.dumps writes U+1F600 as a surrogate pair; the sink must combine it."""
    meta = _meta(0)
    meta["hook_name"] = "block\U0001F600"
    row = _row(meta, 0, 64, "float32")
    assert "\\ud83d\\ude00" in row["metadata_json"]
    sink, lease = _make_sink(tmp_path)
    sink.submit_envelope(LAYOUT, [row], torch.zeros(16, dtype=torch.float32))
    assert sink.flush_and_wait(30.0)
    sink.rethrow_if_failed()
    ((staged, _),) = _read_staged(tmp_path)
    assert staged.hook_name == "block\U0001F600"


def test_the_sink_derives_from_the_engines_record_sink(tmp_path):
    """NativePackSink must be a `_native_backend.RecordSink`, not a look-alike.

    create_record_runtime checks isinstance against the MAIN backend's
    RecordSink and calls the inherited `_acquire_engine`. This module
    registers no RecordSink of its own when the main backend is present
    (RING_TYPES_ARE_STANDINS is False) and derives from the main
    registration instead. On a host without the main backend the stand-ins
    are in use and there is no engine to attach to; the GPU suite
    (test_native_sink_ring_e2e.py) covers the real attachment.
    """
    sink = native_sink.NativePackSink(spool_root=str(tmp_path), layout=LAYOUT)
    assert hasattr(sink, "_acquire_engine")
    if native_sink.RING_TYPES_ARE_STANDINS:
        pytest.skip("the full native backend is not built on this host")
    from dmi.transport import native

    assert isinstance(sink, native.RecordSink)
    assert native_sink.NativePackSink.__mro__[1] is native.RecordSink
