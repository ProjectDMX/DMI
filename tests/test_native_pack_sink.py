"""A3b: the native sink admits, packs, and stages exactly like the reference.

The killer test is end-to-end: submit the golden corpus through the C++
sink, then read the staged packs back with the Python PackReader — capture
ids, tenant binding, and checksums must all resolve. Scope sealing, linger,
admission policies, duplicates, oversized records, and object-key parity
are pinned against the Python pipeline's observable behavior.

Build: make -C native build/conformance_sink
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi.storage.capture import (  # noqa: E402
    CaptureMetadata,
    CaptureRecord,
    DurablePackSpool,
    PackReader,
)
from dmi.storage.capture.pipeline import object_key_for  # noqa: E402
from tests.tools.golden_workload import _corpus  # noqa: E402

DRIVER = REPO_ROOT / "native" / "build" / "conformance_sink"

pytestmark = pytest.mark.cpu

if not DRIVER.exists():
    pytest.skip(
        "native/build/conformance_sink is not built; run "
        "`make -C native build/conformance_sink`",
        allow_module_level=True,
    )


class SinkSession:
    """One driver process; ops are newline-JSON in, one JSON object out."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [str(DRIVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def call(self, **fields) -> dict:
        self.proc.stdin.write(json.dumps(fields) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=30)


@pytest.fixture()
def sink():
    session = SinkSession()
    yield session
    session.close()


def _open(session, root, **overrides):
    config = {
        "op": "open",
        "root": str(root),
        "max_bytes": 1 << 40,
        "max_queue_records": 256,
        "max_queue_bytes": 16 * 1024 * 1024,
        "max_pack_bytes": 128 * 1024 * 1024,
        "max_pack_records": 10_000,
        "max_linger_ns": 1_000_000_000,
        "overload": "drop_newest",
        "admission_timeout": -1,
    }
    config.update(overrides)
    response = session.call(**config)
    assert response["ok"], response


def _submit(session, record: CaptureRecord) -> str:
    mapping = record.metadata.to_mapping()
    response = session.call(
        op="submit", metadata=mapping,
        payload_b64=base64.b64encode(record.payload).decode(),
    )
    assert response["ok"], response
    return response["admission"]


def _meta(index: int, session_id: str = "session-0",
          tenant_id: str = "tenant", dtype: str = "uint8",
          shape=(64,)) -> CaptureMetadata:
    width = {"bool": 1, "uint8": 1, "int32": 4, "float32": 4, "int64": 8}[dtype]
    return CaptureMetadata(
        capture_id=f"capture-{index:06d}",
        tenant_id=tenant_id,
        experiment_id="experiment",
        run_id="run",
        session_id=session_id,
        request_id=f"request-{index}",
        sequence_id=f"sequence-{index}",
        model_id="model",
        model_revision="mr",
        adapter_revision=None,
        capture_policy_version="v",
        hook_name="h",
        layer_number=index % 4,
        producer_rank=0,
        step_number=index,
        token_start=index,
        token_end=index + 1,
        batch_position=0,
        dtype=dtype,
        shape=shape,
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def _record(index: int, **kwargs) -> CaptureRecord:
    meta = _meta(index, **kwargs)
    width = {"bool": 1, "uint8": 1, "int32": 4, "float32": 4, "int64": 8}[meta.dtype]
    size = 1
    for dim in meta.shape:
        size *= dim
    return CaptureRecord(metadata=meta, payload=bytes(size * width))


def test_golden_corpus_end_to_end(sink, tmp_path):
    _open(sink, tmp_path / "spool")
    for record in _corpus():
        assert _submit(sink, record) == "accepted"
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["submitted_records"] == 10
    assert snapshot["admitted_records"] == 10
    assert snapshot["persisted_records"] == 10
    # Every golden record carries a distinct producer_rank, so each is its
    # own scope: 10 packs via 9 session seals; the explicit flush sealed the
    # last pack as MANUAL, so shutdown seals nothing (same as the reference).
    assert snapshot["packs_persisted"] == 10
    assert snapshot["flush_session"] == 9
    assert snapshot["flush_manual"] == 1
    assert snapshot["flush_shutdown"] == 0
    assert snapshot["failures"] == 0
    assert snapshot["dropped_records"] == 0

    # Python reads the native-staged packs: ids, checksums, tenant binding.
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    recovered = spool.recover()
    assert len(recovered) == 10
    seen = []
    for entry in recovered:
        with entry.open() as handle:
            descriptors = PackReader.from_bytes(handle.read()).descriptors(
                store_id="spool", object_key=entry.object_key
            )
        seen.extend(d.metadata.capture_id for d in descriptors)
        assert entry.object_key.startswith("v1/tenant=tenant-golden/")
    assert sorted(seen) == [f"golden-{i:02d}" for i in range(10)]


def test_session_change_seals_a_pack(sink, tmp_path):
    _open(sink, tmp_path / "spool")
    assert _submit(sink, _record(0, session_id="s0")) == "accepted"
    assert _submit(sink, _record(1, session_id="s1")) == "accepted"
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["packs_persisted"] == 2
    assert snapshot["flush_session"] == 1
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    assert len(spool.recover()) == 2


def test_linger_seals_without_close(sink, tmp_path):
    _open(sink, tmp_path / "spool", max_linger_ns=50_000_000)
    assert _submit(sink, _record(0)) == "accepted"
    deadline = time.time() + 10
    while time.time() < deadline:
        snapshot = sink.call(op="snapshot")["snapshot"]
        if snapshot["packs_persisted"] == 1:
            break
        time.sleep(0.05)
    assert snapshot["packs_persisted"] == 1
    assert snapshot["flush_linger"] == 1
    sink.call(op="close", timeout=30)


def test_drop_newest_counts_drops(sink, tmp_path):
    _open(sink, tmp_path / "spool", max_queue_records=4,
          max_queue_bytes=1 << 20, overload="drop_newest")
    outcomes = [_submit(sink, _record(i)) for i in range(64)]
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["submitted_records"] == 64
    assert (snapshot["admitted_records"] + snapshot["dropped_records"] +
            snapshot["timed_out_records"] +
            snapshot["rejected_closed_records"]) == 64
    assert snapshot["persisted_records"] == snapshot["admitted_records"]
    assert outcomes.count("accepted") == snapshot["admitted_records"]


def test_block_with_timeout_counts_timeouts(sink, tmp_path):
    # A queue that can never fit a 64-byte payload (63-byte cap): BLOCK
    # waits, then times out. Deterministic — no thread timing involved.
    _open(sink, tmp_path / "spool", max_queue_records=256,
          max_queue_bytes=63, overload="block", admission_timeout=0.05)
    assert _submit(sink, _record(0)) == "timed_out"
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["timed_out_records"] == 1
    assert snapshot["persisted_records"] == 0


def test_duplicate_capture_is_dropped_not_failed(sink, tmp_path):
    _open(sink, tmp_path / "spool")
    assert _submit(sink, _record(0)) == "accepted"
    assert _submit(sink, _record(0)) == "accepted"  # same capture_id
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["duplicate_records"] == 1
    assert snapshot["persisted_records"] == 1
    assert snapshot["failures"] == 0


def test_oversized_record_is_rejected_up_front(sink, tmp_path):
    _open(sink, tmp_path / "spool", max_pack_bytes=1024)
    big = CaptureRecord(
        metadata=_meta(0, dtype="uint8", shape=(2048,)), payload=bytes(2048)
    )
    assert _submit(sink, big) == "too_large"
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["oversized_records"] == 1
    assert snapshot["packs_persisted"] == 0


def test_small_packs_split_by_size(sink, tmp_path):
    _open(sink, tmp_path / "spool", max_pack_bytes=4096)
    for i in range(8):
        assert _submit(sink, _record(i)) == "accepted"
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["packs_persisted"] > 1
    assert snapshot["persisted_records"] == 8
    assert snapshot["flush_size"] >= 1


def test_object_key_parity_with_python():
    session = SinkSession()
    try:
        cases = [
            ("tenant", "session", 0, 1_700_000_000_000_000_000),
            ("tenant with space", "s", 3, 1_700_000_000_000_000_000),
            ("t~tilde", "séssion-日本", 7, 0),
            ("sha256-" + "ab" * 32, "s", 0, 2**63),
            ("x" * 200, "s", 0, 1_700_000_000_000_000_000),
            ("tenant=equals", "a/b?c", 2**32 - 1, 1_000_000_000),
        ]
        pack_id = "018f0000-0000-7000-8000-000000000f01"
        from dmi.storage.capture.pack import PackWriter
        from dmi.storage.capture.pipeline import ReadyPack
        from dmi.storage.capture.pack import SealedPack
        for tenant, sess, rank, captured in cases:
            response = session.call(
                op="object_key", tenant_id=tenant, session_id=sess,
                producer_rank=rank, captured_at_ns=captured, pack_id=pack_id,
            )
            assert response["ok"], response
            meta = _meta(0, session_id=sess, tenant_id=tenant)
            object.__setattr__(meta, "producer_rank", rank)
            object.__setattr__(meta, "captured_at_ns", captured)
            ready = ReadyPack(
                pack=SealedPack(
                    pack_id=pack_id, created_at_ns=captured, data=b"",
                    record_count=0, footer_offset=0, checksum="0" * 64,
                ),
                first_metadata=meta,
                reason=None,
            )
            assert response["object_key"] == object_key_for(ready), tenant
    finally:
        session.close()


def test_scopes_stay_separate_at_n4(sink, tmp_path):
    # Four sessions share one worker pool: every staged pack must still
    # hold a single scope, and every record must be accounted exactly once.
    _open(sink, tmp_path / "spool", num_workers=4)
    total = 0
    for session in range(4):
        for i in range(32):
            assert _submit(sink, _record(session * 32 + i,
                                         session_id=f"s{session}")) == "accepted"
            total += 1
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == total
    assert snapshot["failures"] == 0
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    recovered = spool.recover()
    seen = []
    for entry in recovered:
        with entry.open() as handle:
            descriptors = PackReader.from_bytes(handle.read()).descriptors(
                store_id="spool", object_key=entry.object_key
            )
        scopes = {(d.metadata.session_id, d.metadata.producer_rank)
                  for d in descriptors}
        assert len(scopes) == 1, scopes
        seen.extend(d.metadata.capture_id for d in descriptors)
    assert sorted(seen) == sorted(f"capture-{i:06d}" for i in range(total))


def test_flush_covers_all_workers(sink, tmp_path):
    # A flush with packs open on several workers must persist all of them:
    # persisted == admitted afterwards, with nothing left unstaged.
    _open(sink, tmp_path / "spool", num_workers=4)
    for i in range(64):
        assert _submit(sink, _record(i, session_id=f"s{i % 8}")) == "accepted"
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="snapshot")["snapshot"]
    assert snapshot["persisted_records"] == 64
    assert snapshot["stage_packs"] == 0
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 64
    assert snapshot["failures"] == 0


def _submit_row(session, metadata_json: str, payload: bytes, dtype: str,
                shape: list) -> dict:
    return session.call(
        op="submit_row", metadata_json=metadata_json,
        payload_b64=base64.b64encode(payload).decode(), dtype=dtype,
        shape=shape,
    )


def _row_meta(index: int = 0, **overrides) -> dict:
    import json as _json

    meta = _meta(index)
    mapping = meta.to_mapping()
    mapping.update(overrides)
    return _json.dumps(mapping)


def test_submit_row_end_to_end(sink, tmp_path):
    import json as _json

    _open(sink, tmp_path / "spool")
    for index, (dtype, width) in enumerate(
        [("bool", 1), ("float32", 4), ("int64", 8)]
    ):
        payload = bytes((i % 251 for i in range(16 * width)))
        mapping = _meta(index, dtype=dtype,
                        shape=(16,)).to_mapping()
        response = _submit_row(
            sink, _json.dumps(mapping), payload, dtype, [16]
        )
        assert response["ok"], response
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 3
    assert snapshot["failures"] == 0
    # One scope, one pack; dtypes survive the row path.
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    recovered = spool.recover()
    assert len(recovered) == 1
    with recovered[0].open() as handle:
        descriptors = PackReader.from_bytes(handle.read()).descriptors(
            store_id="spool", object_key=recovered[0].object_key
        )
    assert [d.metadata.dtype for d in descriptors] == \
        ["bool", "float32", "int64"]


def test_submit_row_rejects_mismatches(sink, tmp_path):
    import json as _json

    _open(sink, tmp_path / "spool")
    payload = bytes(64)
    good = _row_meta(0, dtype="uint8", shape=[64])

    response = _submit_row(sink, good, payload, "int32", [64])
    assert not response["ok"] and "dtype" in response["status"].lower()

    response = _submit_row(sink, good, payload, "uint8", [32, 2])
    assert not response["ok"] and "shape" in response["status"].lower()

    response = _submit_row(sink, good, payload, "uint8", [32])
    assert not response["ok"] and "shape" in response["status"].lower()

    response = _submit_row(sink, good, bytes(63), "uint8", [64])
    assert not response["ok"] and "payload" in response["status"].lower()

    response = _submit_row(sink, '{"capture_id":1,', payload, "uint8", [64])
    assert not response["ok"]

    response = _submit_row(sink, good, payload, "float999", [64])
    assert not response["ok"] and "dtype" in response["status"].lower()

    # layer_number == -1 is legal (logits-style captures), not "missing".
    logits = _row_meta(1, layer_number=-1)
    response = _submit_row(sink, logits, payload, "uint8", [64])
    assert response["ok"], response
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 1


def test_pipeline_head_to_head_with_python_reference(tmp_path):
    """Same corpus through both pipelines; descriptor-level equality.

    Pack ids and object keys legitimately differ (random UUIDs), so the
    comparison is semantic: the union of (capture_id -> metadata mapping +
    payload bytes) staged by each pipeline must match exactly, plus equal
    submitted/admitted/persisted/pack counts.
    """
    from dmi.storage.capture import (
        DurablePackSink,
        DurablePackSpool,
        HostCapturePipeline,
        OverloadPolicy,
        PipelineConfig,
    )
    from dmi.storage.capture.pipeline import AdmissionResult

    payload = bytes(4096)
    records = []
    for session in range(4):
        for i in range(50):
            index = session * 50 + i
            records.append(_record(index, session_id=f"h2h-{session}"))

    def _pack_contents(spool_root: Path) -> dict:
        spool = DurablePackSpool(spool_root, max_bytes=1 << 40)
        union = {}
        for entry in spool.recover():
            with entry.open() as handle:
                for descriptor in PackReader.from_bytes(
                    handle.read()
                ).descriptors(store_id="x", object_key=entry.object_key):
                    meta = descriptor.metadata
                    union[meta.capture_id] = (
                        json.dumps(meta.to_mapping(), sort_keys=True),
                        descriptor.locator.checksum,
                        descriptor.locator.stored_length,
                    )
        return union

    # Reference pipeline, spool mode, same bounds.
    reference_root = tmp_path / "reference"
    python_spool = DurablePackSpool(reference_root / "spool",
                                    max_bytes=1 << 40)
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=256,
            max_queue_bytes=16 * 1024 * 1024,
            max_pack_bytes=256 * 1024,
            max_pack_records=10_000,
            max_linger_ns=1_000_000_000,
            overload_policy=OverloadPolicy.DROP_NEWEST,
        ),
        DurablePackSink(python_spool),
    )
    pipeline.start()
    for record in records:
        assert pipeline.submit(record) is AdmissionResult.ACCEPTED
    python_snapshot = pipeline.close(timeout=30)

    # Native pipeline, same corpus and bounds.
    session = SinkSession()
    try:
        _open(session, tmp_path / "native",
              max_pack_bytes=256 * 1024)
        for record in records:
            assert _submit(session, record) == "accepted"
        assert session.call(op="flush", timeout=30)["ok"]
        native_snapshot = session.call(op="close", timeout=30)["snapshot"]
    finally:
        session.close()

    assert native_snapshot["submitted_records"] == \
        python_snapshot.submitted_records == len(records)
    assert native_snapshot["admitted_records"] == \
        python_snapshot.admitted_records == len(records)
    assert native_snapshot["persisted_records"] == \
        python_snapshot.persisted_records == len(records)
    assert native_snapshot["packs_persisted"] == \
        python_snapshot.packs_persisted == 4
    assert native_snapshot["failures"] == python_snapshot.failures == 0

    python_union = _pack_contents(reference_root / "spool")
    native_union = _pack_contents(tmp_path / "native")
    assert set(native_union) == set(python_union) == \
        {f"capture-{i:06d}" for i in range(len(records))}
    assert native_union == python_union
