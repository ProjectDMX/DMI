"""A1 conformance: the native pack builder must produce byte-identical packs.

The oracle is the Python reference (PackWriter). The golden manifest pins the
byte contract for the canonical corpus; this test goes further and compares
native output to reference output directly, corpus by corpus, so a divergence
names the case that moved.

The native side is driven through ``native/build/conformance_main``, a
dependency-free stdin/stdout JSON protocol, so the test needs no pybind and
runs in the CPU gate. Skip (with reason) when the binary has not been built.

Build: make -C native build/conformance_main
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi.storage.capture import CaptureMetadata, CaptureRecord, PackWriter  # noqa: E402

DRIVER = REPO_ROOT / "native" / "build" / "conformance_main"

pytestmark = pytest.mark.cpu

if not DRIVER.exists():
    pytest.skip(
        "native/build/conformance_main is not built; run "
        "`make -C native build/conformance_main`",
        allow_module_level=True,
    )


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()


def _unb64(text: str) -> bytes:
    import base64

    return base64.b64decode(text)


def _native_build(
    pack_id: str,
    created_at_ns: int,
    max_pack_bytes: int,
    records: list[dict],
    max_records: int | None = None,
) -> dict:
    request = {
        "op": "build",
        "pack_id": pack_id,
        "created_at_ns": created_at_ns,
        "max_pack_bytes": max_pack_bytes,
        "records": records,
    }
    if max_records is not None:
        request["max_records"] = max_records
    proc = subprocess.run(
        [str(DRIVER)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"native driver produced no output: {proc.stderr}")
    return json.loads(lines[0])


def _meta_mapping(meta: CaptureMetadata) -> dict:
    mapping = meta.to_mapping()
    mapping["payload_b64"] = None
    return {k: v for k, v in mapping.items()}


def _assert_packs_match(
    pack_id, created_at_ns, records, max_pack_bytes, max_records=None
):
    """Build with both writers; require identical bytes and checksum."""
    reference = PackWriter(
        pack_id=pack_id,
        created_at_ns=created_at_ns,
        max_pack_bytes=max_pack_bytes,
        max_records=max_records or 1_000_000,
    )
    native_records = []
    for record in records:
        reference.append(record)
        row = _meta_mapping(record.metadata)
        row["payload_b64"] = _b64(record.payload)
        native_records.append(row)

    response = _native_build(
        str(pack_id), created_at_ns, max_pack_bytes, native_records, max_records
    )
    assert response["ok"], response

    sealed = reference.seal()
    native_bytes = _unb64(response["data_b64"])
    assert response["data_sha256"] == sealed.checksum
    assert native_bytes == sealed.data, (
        "native pack bytes diverge from the reference: "
        f"native {len(native_bytes)}B vs reference {len(sealed.data)}B"
    )
    assert response["checksum"] == sealed.checksum
    assert response["footer_offset"] == sealed.footer_offset
    assert response["record_count"] == sealed.record_count


# --- corpora -------------------------------------------------------------------

from tests.tools.golden_workload import PACK_ID, _corpus  # noqa: E402


def test_native_matches_reference_on_the_golden_corpus():
    _assert_packs_match(
        PACK_ID, 1_700_000_000_000_000_000, _corpus(), 8 * 1024 * 1024
    )


def test_native_matches_reference_with_null_and_escaped_text():
    meta = dict(
        capture_id='quote"back\\slash',
        tenant_id="tenant-é日本",
        experiment_id="ctrl\x01\x1f\x7f",
        run_id="tab\tnewline\nreturn\rbackspace\bformfeed\f",
        session_id="session",
        request_id="r",
        sequence_id="s",
        model_id="m",
        model_revision="mr",
        adapter_revision=None,
        capture_policy_version="v",
        hook_name="h",
        layer_number=-1,
        producer_rank=2**32 - 1,
        step_number=2**64 - 1,
        token_start=0,
        token_end=2**64 - 1,
        batch_position=2**32 - 1,
        dtype="uint8",
        shape=(1, 2, 3),
        captured_at_ns=2**64 - 1,
    )
    records = [CaptureRecord(metadata=CaptureMetadata(**meta), payload=b"\x00\x01\x02\x03\x04\x05")]
    _assert_packs_match(
        "018f0000-0000-7000-8000-00000000dead", 42, records, 8 * 1024 * 1024
    )


def test_native_matches_reference_multi_record_and_alignment():
    records = [
        CaptureRecord(
            metadata=CaptureMetadata(
                capture_id=f"c-{i:03d}",
                tenant_id="t",
                experiment_id="e",
                run_id="r",
                session_id="s",
                request_id=f"q{i}",
                sequence_id=f"n{i}",
                model_id="m",
                model_revision="mr",
                adapter_revision=f"a{i}" if i % 3 else None,
                capture_policy_version="v",
                hook_name="h",
                layer_number=i % 8,
                producer_rank=i,
                step_number=100 + i,
                token_start=i,
                token_end=i + 7,
                batch_position=i,
                dtype=("uint8" if i % 2 else "int32"),
                shape=(3,) if i % 2 else (2,),
                captured_at_ns=1_000 + i,
            ),
            payload=bytes(range(i % 5, (i % 5) + (3 if i % 2 else 8))),
        )
        for i in range(64)
    ]
    _assert_packs_match(
        "018f0000-0000-7000-8000-00000000beef", 7, records, 64 * 1024
    )


def test_native_and_reference_agree_on_empty_shape():
    meta = dict(
        capture_id="empty-shape",
        tenant_id="t",
        experiment_id="e",
        run_id="r",
        session_id="s",
        request_id="q",
        sequence_id="n",
        model_id="m",
        model_revision="mr",
        adapter_revision="a",
        capture_policy_version="v",
        hook_name="h",
        layer_number=0,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=0,
        batch_position=0,
        dtype="uint8",
        shape=(),
        captured_at_ns=1,
    )
    # An empty shape still renders "shape":[] but math.prod(()) == 1, so the
    # payload is one dtype byte, not zero.
    records = [CaptureRecord(metadata=CaptureMetadata(**meta), payload=b"\x2a")]
    _assert_packs_match(
        "018f0000-0000-7000-8000-000000000001", 1, records, 8 * 1024 * 1024
    )


def test_native_and_reference_agree_on_multi_dim_shape():
    meta = dict(
        capture_id="multi-dim",
        tenant_id="t",
        experiment_id="e",
        run_id="r",
        session_id="s",
        request_id="q",
        sequence_id="n",
        model_id="m",
        model_revision="mr",
        adapter_revision="a",
        capture_policy_version="v",
        hook_name="h",
        layer_number=0,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=0,
        batch_position=0,
        dtype="uint8",
        shape=(2, 3),
        captured_at_ns=1,
    )
    records = [CaptureRecord(metadata=CaptureMetadata(**meta), payload=bytes(6))]


def test_native_rejects_what_the_reference_rejects():
    base = dict(
        capture_id="c",
        tenant_id="t",
        experiment_id="e",
        run_id="r",
        session_id="s",
        request_id="q",
        sequence_id="n",
        model_id="m",
        model_revision="mr",
        adapter_revision=None,
        capture_policy_version="v",
        hook_name="h",
        layer_number=0,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="uint8",
        shape=(1,),
        captured_at_ns=1,
    )
    payload = _b64(b"\x00")

    # duplicate capture id
    rows = [{**_meta_mapping(CaptureMetadata(**base)), "payload_b64": payload}] * 2
    response = _native_build(
        "018f0000-0000-7000-8000-000000000002", 1, 8 * 1024 * 1024, rows
    )
    assert not response["ok"] and "duplicate" in response["status"].lower()

    # record limit
    rows = [{**_meta_mapping(CaptureMetadata(**{**base, "capture_id": f"c{i}"})),
             "payload_b64": payload} for i in range(3)]
    response = _native_build(
        "018f0000-0000-7000-8000-000000000003", 1, 8 * 1024 * 1024, rows,
        max_records=2,
    )
    assert not response["ok"] and "record limit" in response["status"].lower()

    # capacity: two records that fit alone, not together
    big = _b64(b"\x00" * 4096)
    rows = [
        {**_meta_mapping(CaptureMetadata(**{**base, "capture_id": "big1"})),
         "payload_b64": big},
        {**_meta_mapping(CaptureMetadata(**{**base, "capture_id": "big2"})),
         "payload_b64": big},
    ]
    response = _native_build(
        "018f0000-0000-7000-8000-000000000004", 1, 8 * 1024, rows
    )
    assert not response["ok"] and "exceed" in response["status"].lower()

    # the reference agrees on all three
    import contextlib

    @contextlib.contextmanager
    def _raises():
        raised = None
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            raised = exc
        assert raised is not None, "expected the reference to reject"

    with _raises():
        w = PackWriter(pack_id="018f0000-0000-7000-8000-000000000005",
                       created_at_ns=1, max_pack_bytes=8 * 1024 * 1024)
        w.append(CaptureRecord(metadata=CaptureMetadata(**base), payload=b"\x00"))
        w.append(CaptureRecord(metadata=CaptureMetadata(**base), payload=b"\x00"))


def test_native_crc32_matches_zlib():
    import random as _random
    import zlib

    generator = _random.Random(7)
    for size in (0, 1, 7, 64, 4096, 65536):
        data = generator.randbytes(size)
        proc = subprocess.run(
            [str(DRIVER)],
            input=json.dumps({"op": "crc32", "data_b64": _b64(data)}) + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        response = json.loads(proc.stdout.strip())
        expected = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
        assert response["ok"] and response["crc32"] == expected, (size, response)


def test_native_pack_matches_recorded_golden_manifest():
    """Close the loop to the recorded artifact, not just to live Python.

    tests/data/capture_golden_manifest.json pins the whole-object sha256
    (53a087...) produced by the reference writer. The native writer must
    produce the identical digest for the identical corpus — transitively
    proven by the byte-equality tests, asserted directly here so a drift
    in either direction names itself.
    """
    import hashlib
    import json as _json

    with open(
        REPO_ROOT / "tests" / "data" / "capture_golden_manifest.json"
    ) as handle:
        manifest = _json.load(handle)
    expected = manifest["pack"]["sha256"]
    assert expected == "53a0873af5b5932ceb3e44223492aec11eadfb1d9298cb4cef81d8ca5337fd4e"

    records = []
    for record in _corpus():
        row = _meta_mapping(record.metadata)
        row["payload_b64"] = _b64(record.payload)
        records.append(row)
    response = _native_build(
        str(PACK_ID), 1_700_000_000_000_000_000, 8 * 1024 * 1024, records
    )
    assert response["ok"], response
    assert response["data_sha256"] == expected
    assert hashlib.sha256(_unb64(response["data_b64"])).hexdigest() == expected
