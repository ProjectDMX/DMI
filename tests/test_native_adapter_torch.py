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
