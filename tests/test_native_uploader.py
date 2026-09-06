"""A4: the native uploader drains a spool into the object store.

Staged packs come from the native sink driver (A3b); uploads go to the fake
S3 (same server the client tests use, with server-side botocore signature
verification). Pinned: preflight idempotence, hash-gated upload, post-upload
visibility, parallel batch outcomes by position, retry taxonomy, byte gate,
and corrupt-staged refusal.

Build: make -C native build/conformance_store build/conformance_sink
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
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

STORE_DRIVER = REPO_ROOT / "native" / "build" / "conformance_store"
SINK_DRIVER = REPO_ROOT / "native" / "build" / "conformance_sink"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not STORE_DRIVER.exists() or not SINK_DRIVER.exists(),
        reason="native store/sink drivers are not built; run "
        "`make -C native build/conformance_store build/conformance_sink`",
    ),
]

# Reuse the fake S3 server (signature-verifying) from the client tests.
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS,
    BUCKET,
    REGION,
    SECRET,
    STATE,
    FakeS3Handler,
    _base as _client_base,
    _call as _store_call,
    fake_s3,
)


class DriverSession:
    def __init__(self, binary: Path):
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
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
        self.proc.wait(timeout=60)


def _stage(session: DriverSession, root: Path, index: int,
           payload: bytes = b"0123456789abcdef" * 64) -> dict:
    """Stage one pack through the native sink; return its StagedPack JSON.

    The sink mints pack ids, so the new entry is identified by diffing the
    ready set before and after.
    """
    before = set(root.rglob("*.dmi-pack.ready")) if root.exists() else set()
    meta = CaptureMetadata(
        capture_id=f"upload-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v", hook_name="h",
        layer_number=0, producer_rank=0, step_number=index, token_start=index,
        token_end=index + 1, batch_position=0, dtype="uint8",
        shape=(len(payload),), captured_at_ns=1_700_000_000_000_000_000 + index,
    )
    assert session.call(
        op="open", root=str(root), max_bytes=1 << 40,
        max_queue_records=256, max_queue_bytes=1 << 20,
        max_pack_bytes=8 * 1024 * 1024, max_pack_records=10_000,
        max_linger_ns=1_000_000_000, overload="drop_newest",
        admission_timeout=-1,
    )["ok"]
    response = session.call(
        op="submit", metadata=meta.to_mapping(),
        payload_b64=base64.b64encode(payload).decode(),
    )
    assert response["admission"] == "accepted", response
    assert session.call(op="flush", timeout=30)["ok"]
    snapshot = session.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 1, snapshot
    new = set(root.rglob("*.dmi-pack.ready")) - before
    assert len(new) == 1, new
    # Recover through the spool driver for the entry's JSON form.
    recover = subprocess.run(
        [str(STORE_DRIVER.parent / "conformance_spool")],
        input=json.dumps({"op": "recover", "root": str(root),
                          "max_bytes": 1 << 40}) + "\n",
        capture_output=True, text=True, timeout=30,
    )
    entries = json.loads(recover.stdout.strip())["staged"]
    match = [e for e in entries if Path(e["path"]) in new]
    assert len(match) == 1, entries
    return match[0]


def _store_base(endpoint: str, **overrides) -> dict:
    request = {
        "endpoint": endpoint,
        "bucket": BUCKET,
        "region": REGION,
        "access": ACCESS,
        "secret": SECRET,
        "token": None,
        "insecure": True,
        "connect_timeout": 5,
        "read_timeout": 15,
        "max_attempts": 4,
        "store_id": "native-test",
    }
    request.update(overrides)
    return request


def _upload_pending(store: DriverSession, endpoint: str, root: Path,
                    limit: int = -1, **overrides) -> dict:
    fields = _store_base(endpoint, **overrides)
    # Positional args always win; the rest are defaults the caller may
    # override (a plain update would silently clobber max_in_flight_bytes
    # and friends — which is exactly how the byte-gate test went green
    # locally and red here).
    fields["op"] = "upload_pending"
    fields["root"] = str(root)
    fields["spool_max_bytes"] = 1 << 40
    fields["limit"] = limit
    fields.setdefault("max_workers", 4)
    fields.setdefault("max_in_flight_bytes", 1 << 30)
    return store.call(**fields)


def test_sink_to_store_end_to_end(fake_s3, tmp_path):
    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        staged = _stage(sink, tmp_path / "spool", 1)
        result = _upload_pending(store, fake_s3, tmp_path / "spool")
        assert result["ok"], result
        snap = result["snapshot"]
        assert snap["attempted_packs"] == 1
        assert snap["uploaded_packs"] == 1
        assert snap["failed_packs"] == 0
        assert len(result["refs"]) == 1
        ref = result["refs"][0]
        assert ref["pack_id"] == staged["pack_id"]
        assert ref["checksum"] == staged["checksum"]
        assert ref["object_bytes"] == staged["object_bytes"]
        assert ref["store_id"] == "native-test"
        # The staged file is gone (remove-after-commit).
        assert not list((tmp_path / "spool").rglob("*.dmi-pack.ready"))
        # The object is really there with the DMI metadata the Python store
        # requires — read it back through the pack path.
        fetched = _store_call(
            "get", **_client_base(fake_s3), key=staged["object_key"],
            offset=0, length=staged["object_bytes"],
        )
        assert fetched["ok"], fetched
        descriptors = PackReader.from_bytes(
            base64.b64decode(fetched["data_b64"])
        ).descriptors(store_id="native-test",
                      object_key=staged["object_key"])
        assert [d.metadata.capture_id for d in descriptors] == ["upload-0001"]
    finally:
        sink.close()
        store.close()


def test_second_upload_is_a_preflight_hit(fake_s3, tmp_path):
    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        staged = _stage(sink, tmp_path / "spool", 2)
        # Keep the bytes aside: the first upload removes the staged file.
        staged_bytes = Path(staged["path"]).read_bytes()
        first = _upload_pending(store, fake_s3, tmp_path / "spool")
        assert first["ok"] and first["snapshot"]["uploaded_packs"] == 1
        # Restore the identical staged entry and upload again: the preflight
        # HEAD must bless it without a second PUT.
        Path(staged["path"]).write_bytes(staged_bytes)
        puts_before = sum(
            1 for c in STATE.calls
            if c["method"] == "PUT" and staged["object_key"] in c["path"]
        )
        fields = _store_base(fake_s3)
        fields.update(
            op="upload_one", root=str(tmp_path / "spool"),
            spool_max_bytes=1 << 40, max_workers=1,
            max_in_flight_bytes=1 << 30, staged=staged,
        )
        second = store.call(**fields)
        assert second["ok"], second
        puts_after = sum(
            1 for c in STATE.calls
            if c["method"] == "PUT" and staged["object_key"] in c["path"]
        )
        assert puts_after == puts_before
    finally:
        sink.close()
        store.close()


def test_parallel_batch_reports_by_position(fake_s3, tmp_path):
    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        for i in range(6):
            _stage(sink, tmp_path / "spool", 10 + i)
        result = _upload_pending(store, fake_s3, tmp_path / "spool")
        assert result["ok"], result
        snap = result["snapshot"]
        assert snap["attempted_packs"] == 6
        assert snap["uploaded_packs"] == 6
        assert snap["failed_packs"] == 0
        assert snap["peak_active_uploads"] >= 1
        assert len(result["refs"]) == 6
        assert all(r["pack_id"] for r in result["refs"])
        assert all(f["pack_id"] == "" for f in result["failures"])
    finally:
        sink.close()
        store.close()


def test_retry_then_success_counts_retries(fake_s3, tmp_path):
    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        staged = _stage(sink, tmp_path / "spool", 5)
        # Move the staged ready file under the fault prefix: same pack id,
        # so the spool contract still holds.
        readies = list((tmp_path / "spool").rglob("*.dmi-pack.ready"))
        assert len(readies) == 1
        fault_dir = tmp_path / "spool" / "fault" / "once-500"
        fault_dir.mkdir(parents=True, exist_ok=True)
        readies[0].rename(fault_dir / readies[0].name)
        # Transport retries disabled: the single 500 must surface as an
        # uploader-level retry, not be absorbed inside one attempt.
        result = _upload_pending(store, fake_s3, tmp_path / "spool",
                                 max_attempts=1)
        assert result["ok"], result
        assert result["snapshot"]["uploaded_packs"] == 1
        assert result["snapshot"]["retries"] >= 1
    finally:
        sink.close()
        store.close()


def test_corrupt_staged_bytes_are_refused(fake_s3, tmp_path):
    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        staged = _stage(sink, tmp_path / "spool", 3)
        with open(staged["path"], "r+b") as handle:
            handle.seek(100)
            handle.write(b"CORRUPT!")
        # upload_one, not upload_pending: recover() would quarantine the
        # corrupt file before the uploader ever sees it (also pinned, in
        # the spool suite). Here the staged entry reaches UploadOne with a
        # valid checksum claim over corrupt bytes.
        fields = _store_base(fake_s3)
        fields.update(
            op="upload_one", root=str(tmp_path / "spool"),
            spool_max_bytes=1 << 40, max_workers=1,
            max_in_flight_bytes=1 << 30, staged=staged,
        )
        result = store.call(**fields)
        assert not result["ok"], result
        assert "checksum" in result["what"].lower()
        # Nothing reached the store.
        puts = [c for c in STATE.calls if c["method"] == "PUT"]
        assert not puts
    finally:
        sink.close()
        store.close()


def test_pack_over_the_byte_gate_fails_fast(fake_s3, tmp_path):
    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        staged = _stage(sink, tmp_path / "spool", 4)
        result = _upload_pending(
            store, fake_s3, tmp_path / "spool", max_in_flight_bytes=16
        )
        assert result["ok"], result
        assert result["snapshot"]["failed_packs"] == 1
        assert result["failures"][0]["attempts"] == 0
        assert "in-flight" in result["failures"][0]["error"]
    finally:
        sink.close()
        store.close()


def test_mixed_batch_reports_oversized_pack_at_its_position(fake_s3, tmp_path):
    """refs[i] pairs with failures[i] positionally, in recover() order.

    One oversized pack mid-batch must be refused at ITS position: the
    failure named there, every position holding exactly one of
    ref/failure. (The single-oversized-pack case passes by construction;
    mixing oversized and normal packs is what orders the outcome writes.)
    """
    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        spool_root = tmp_path / "spool"
        for i in range(3):
            _stage(sink, spool_root, 20 + i)
        _stage(sink, spool_root, 30, payload=b"\0" * (1 << 18))
        recover = subprocess.run(
            [str(STORE_DRIVER.parent / "conformance_spool")],
            input=json.dumps({"op": "recover", "root": str(spool_root),
                              "max_bytes": 1 << 40}) + "\n",
            capture_output=True, text=True, timeout=30,
        )
        staged = json.loads(recover.stdout.strip())["staged"]
        big = [i for i, e in enumerate(staged) if e["object_bytes"] > 64 << 10]
        assert len(big) == 1
        big_index, big_pack_id = big[0], staged[big[0]]["pack_id"]

        result = _upload_pending(store, fake_s3, spool_root,
                                 max_in_flight_bytes=64 << 10)
        assert result["ok"], result
        snap = result["snapshot"]
        assert snap["attempted_packs"] == 4
        assert snap["uploaded_packs"] == 3
        assert snap["failed_packs"] == 1
        assert len(result["refs"]) == 4, result
        assert len(result["failures"]) == 4, result
        for i in range(4):
            has_ref = bool(result["refs"][i]["pack_id"])
            has_failure = bool(result["failures"][i]["pack_id"])
            assert has_ref != has_failure, (i, result["failures"])
        failure = result["failures"][big_index]
        assert failure["pack_id"] == big_pack_id, result["failures"]
        assert failure["attempts"] == 0
        assert "in-flight" in failure["error"]
    finally:
        sink.close()
        store.close()


def test_uploader_head_to_head_with_python_reference(fake_s3, tmp_path):
    """Same staged bytes through both uploaders; identical objects land.

    One pack is staged into two spool dirs (same pack id, checksum, bytes).
    The Python SpoolUploader (boto3) takes one, the native uploader the
    other. The objects must match byte-for-byte with equal DMI metadata,
    and both refs must describe the same pack.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    from dmi.storage.capture import (
        CaptureMetadata,
        CaptureRecord,
        PackWriter,
        S3PackStore,
        S3StoreConfig,
        SpoolUploader,
    )
    from tests.tools.golden_workload import PACK_ID, _corpus

    writer = PackWriter(
        pack_id=PACK_ID, created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=8 * 1024 * 1024,
    )
    for record in _corpus():
        writer.append(record)
    sealed = writer.seal()

    roots = [tmp_path / "py-spool", tmp_path / "native-spool"]
    for root in roots:
        spool = DurablePackSpool(root, max_bytes=1 << 40)
        spool.stage(
            sealed,
            f"v1/tenant=tenant-golden/date=2026-09-05/session=session-golden"
            f"/rank=0/{PACK_ID}.dmi-pack",
        )

    py_config = S3StoreConfig(
        endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
        access_key_id=ACCESS, secret_access_key=SECRET,
        store_id="py-ref", allow_insecure_http=True,
    )
    py_store = S3PackStore.from_config(py_config)
    py_refs = SpoolUploader(
        DurablePackSpool(roots[0], max_bytes=1 << 40), py_store
    ).upload_pending()
    assert len(py_refs) == 1

    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    try:
        result = _upload_pending(store, fake_s3, roots[1])
        assert result["ok"], result
        assert result["snapshot"]["uploaded_packs"] == 1
        native_ref = result["refs"][0]
    finally:
        sink.close()
        store.close()

    assert native_ref["pack_id"] == py_refs[0].pack_id == str(PACK_ID)
    assert native_ref["checksum"] == py_refs[0].checksum == sealed.checksum
    assert native_ref["object_bytes"] == py_refs[0].object_bytes
    assert native_ref["object_key"] == py_refs[0].object_key

    client = boto3.client(
        "s3", endpoint_url=fake_s3, region_name=REGION,
        aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
        config=BotoConfig(signature_version="s3v4",
                          s3={"addressing_style": "path"}),
    )
    bodies = {}
    metas = {}
    for ref in (py_refs[0], native_ref):
        key = ref["object_key"] if isinstance(ref, dict) else ref.object_key
        response = client.get_object(Bucket=BUCKET, Key=key)
        bodies[key] = response["Body"].read()
        metas[key] = {k.lower(): v for k, v in response["Metadata"].items()}
    assert bodies[py_refs[0].object_key] == bodies[native_ref["object_key"]]
    assert bodies[native_ref["object_key"]] == sealed.data
    assert metas[py_refs[0].object_key] == metas[native_ref["object_key"]]
    assert metas[native_ref["object_key"]]["dmi-sha256"] == sealed.checksum
    assert metas[native_ref["object_key"]]["dmi-format"] == "dmi-pack-v1"
