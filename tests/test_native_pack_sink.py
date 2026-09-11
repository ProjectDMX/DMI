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
    PackFormatError,
    PackReader,
)
from dmi.storage.capture.pipeline import object_key_for  # noqa: E402
from tests.tools.golden_workload import _corpus  # noqa: E402

DRIVER = REPO_ROOT / "native" / "build" / "conformance_sink"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_sink is not built; run "
        "`make -C native build/conformance_sink`",
    ),
]


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

    def call_line(self, line: str) -> dict:
        """Send one request verbatim, for literals json.dumps would rewrite."""
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def call(self, **fields) -> dict:
        return self.call_line(json.dumps(fields))

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
    # The golden corpus is one capture per dtype the format accepts — the
    # count tracks the dtype table, not a fixed number.
    from tests.tools.golden_workload import _DTYPES
    expected = len(_DTYPES)
    assert snapshot["submitted_records"] == expected
    assert snapshot["admitted_records"] == expected
    assert snapshot["persisted_records"] == expected
    # Every golden record carries a distinct producer_rank, so each is its
    # own scope: one pack per record via session seals; the explicit flush
    # sealed the last pack as MANUAL, so shutdown seals nothing (same as
    # the reference).
    assert snapshot["packs_persisted"] == expected
    assert snapshot["flush_session"] == expected - 1
    assert snapshot["flush_manual"] == 1
    assert snapshot["flush_shutdown"] == 0
    assert snapshot["failures"] == 0
    assert snapshot["dropped_records"] == 0

    # Python reads the native-staged packs: ids, checksums, tenant binding.
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    recovered = spool.recover()
    from tests.tools.golden_workload import _DTYPES
    assert len(recovered) == len(_DTYPES)
    seen = []
    for entry in recovered:
        with entry.open() as handle:
            descriptors = PackReader.from_bytes(handle.read()).descriptors(
                store_id="spool", object_key=entry.object_key
            )
        seen.extend(d.metadata.capture_id for d in descriptors)
        assert entry.object_key.startswith("v1/tenant=tenant-golden/")
    assert sorted(seen) == [
        f"golden-{i:02d}" for i in range(len(_DTYPES))]


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


def test_block_with_timeout_admits_a_record_that_exactly_fits(sink, tmp_path):
    """A 64-byte payload under a 64-byte cap is admitted, not refused.

    This test used to submit the same 64-byte payload under a 63-byte cap
    and assert "timed_out": with no screening of max_queue_bytes at
    admission, a record the queue could never hold entered the wait loop
    and aged out of it. The oracle (_BoundedQueue.put) answers TOO_LARGE
    for that record before it ever waits, so the refusal moved to
    test_a_record_the_queue_can_never_hold_is_too_large and what is left
    here is the boundary the new screen must not overshoot: the bound is
    `n <= max_queue_bytes`, so a record of exactly the cap still fits.

    The timeout path itself needs a queue that is full of records a
    consumer has not drained yet. The oracle pins it by stalling its sink
    (tests/_faults.BlockingPackSink, test_capture_pipeline.py); this
    driver has no equivalent stall, and every alternative here would be a
    race against the packer thread.
    """
    _open(sink, tmp_path / "spool", max_queue_records=256,
          max_queue_bytes=64, overload="block", admission_timeout=0.05)
    assert _submit(sink, _record(0)) == "accepted"
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["admitted_records"] == 1
    assert snapshot["oversized_records"] == 0
    assert snapshot["timed_out_records"] == 0
    assert snapshot["persisted_records"] == 1


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


@pytest.mark.parametrize("overload, timeout", [
    ("drop_newest", -1),
    # A timeout, so a regression cannot hang the suite: under BLOCK with
    # admission_timeout=-1 this record waits for room that can never exist.
    ("block", 0.05),
])
def test_a_record_the_queue_can_never_hold_is_too_large(sink, tmp_path,
                                                        overload, timeout):
    """max_queue_bytes is an admission bound, not just a wait condition.

    The oracle (_BoundedQueue.put, pipeline.py) answers TOO_LARGE for a
    record bigger than the queue's byte cap BEFORE it enters the wait loop,
    and HostCapturePipeline.submit counts it as oversized -- exactly as it
    does for a record past max_pack_bytes, which it checks first. The native
    sink screened only max_pack_bytes, so a record between the two bounds
    passed admission and reached a loop whose condition
    (queue_bytes_ + n <= max_queue_bytes) is unsatisfiable even on an empty
    queue: DROP_NEWEST called it dropped, BLOCK with a timeout called it
    timed out, and BLOCK without one waited forever. With the shipped
    defaults (16 MiB queue, 128 MiB pack) that is every record over 16 MiB.
    """
    _open(sink, tmp_path / "spool", max_queue_records=256,
          max_queue_bytes=1024, max_pack_bytes=1 << 20,
          overload=overload, admission_timeout=timeout)
    big = CaptureRecord(
        metadata=_meta(0, dtype="uint8", shape=(2048,)), payload=bytes(2048)
    )
    assert _submit(sink, big) == "too_large"
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["oversized_records"] == 1
    assert snapshot["dropped_records"] == 0
    assert snapshot["timed_out_records"] == 0
    assert snapshot["admitted_records"] == 0
    assert snapshot["persisted_records"] == 0
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


def _python_object_key(tenant, sess, rank, captured, pack_id) -> str:
    """The oracle's key for these arguments, via `object_key_for`."""
    from dmi.storage.capture.pack import SealedPack
    from dmi.storage.capture.pipeline import ReadyPack

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
    return object_key_for(ready)


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
            # A double dot inside a component is legal for both key builders
            # (quote() spares '.'), and so is a component that is only dots.
            ("a..b", "s..t", 0, 1_700_000_000_000_000_000),
            ("..", "s", 0, 1_700_000_000_000_000_000),
        ]
        pack_id = "018f0000-0000-7000-8000-000000000f01"
        for tenant, sess, rank, captured in cases:
            response = session.call(
                op="object_key", tenant_id=tenant, session_id=sess,
                producer_rank=rank, captured_at_ns=captured, pack_id=pack_id,
            )
            assert response["ok"], response
            assert response["object_key"] == _python_object_key(
                tenant, sess, rank, captured, pack_id), tenant
    finally:
        session.close()


def test_dotted_tenant_stages_and_does_not_latch_the_sink(sink, tmp_path):
    # A tenant whose name contains ".." yields the legal key segment
    # "tenant=a..b" — Python's spool accepts it (only a whole component of
    # "", "." or ".." is an escape), so the native spool must too. Getting
    # this wrong is not merely a lost pack: the spool refusal fails the
    # worker, which closes the sink for every later submit.
    _open(sink, tmp_path / "spool")
    assert _submit(sink, _record(0, tenant_id="a..b")) == "accepted"
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="snapshot")["snapshot"]
    assert snapshot["persisted_records"] == 1
    assert snapshot["failures"] == 0

    # The sink is still open: a later submit is admitted, not "closed".
    assert _submit(sink, _record(1, tenant_id="tenant-ok")) == "accepted"
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 2
    assert snapshot["failures"] == 0

    # Python reads back the dotted-tenant pack from the native spool.
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    keys = sorted(entry.object_key for entry in spool.recover())
    assert len(keys) == 2
    assert any(key.startswith("v1/tenant=a..b/") for key in keys), keys


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


# --- integer bounds ------------------------------------------------------------
#
# CaptureMetadata refuses these values outright, so the row path must refuse
# them too rather than accumulate them modulo 2**64 and admit whatever falls
# out. The mapping is patched after construction to get past the reference
# validator and put the literal on the wire.


@pytest.mark.parametrize(
    "field, value",
    [
        # 2**64 + 1 wraps to 1: a wrapped timestamp is a plausible-looking
        # answer, which is worse than a refusal.
        ("captured_at_ns", 2**64 + 1),
        ("step_number", 2**64),
        ("token_end", 2**64 + 7),
        # 2**64 + 5 wraps to 5, which then passes the UInt32 bound.
        ("producer_rank", 2**64 + 5),
        ("batch_position", 2**64 + 5),
        # 2**64 + 3 wraps to 3, a legal layer index.
        ("layer_number", 2**64 + 3),
        # 2**64 - 1 is INSIDE the 64-bit union the parse accepts, so it is not
        # caught by any of the above: its two's-complement bit pattern is -1,
        # which is layer_number's legal "no layer" sentinel, and the bound
        # ValidateMetadata applies (>= -1) admits it. It is the single input
        # that aliases -- 2**64 - 2 lands on -2 and 2**63 on INT64_MIN, both
        # refused -- which is why the wider literals above all miss it. The
        # union stays legal for the UInt64 counters; only a SIGNED field must
        # refuse the unsigned half.
        ("layer_number", 2**64 - 1),
        # Below INT64_MIN by one; the negative branch has the wider limit.
        ("token_start", -(2**63) - 1),
        ("layer_number", -(2**63) - 1),
        # A digit run far longer than any 64-bit value.
        ("captured_at_ns", int("9" * 40)),
        ("layer_number", -int("9" * 40)),
    ],
)
def test_submit_row_refuses_out_of_range_integers(sink, tmp_path, field, value):
    _open(sink, tmp_path / "spool")
    payload = bytes(64)
    overrides = {"dtype": "uint8", "shape": [64], "token_start": 0}
    overrides[field] = value
    metadata = _row_meta(0, **overrides)
    response = _submit_row(sink, metadata, payload, "uint8", [64])
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 0
    # The oracle is what makes this a parity bug rather than a taste call.
    mapping = _meta(0).to_mapping()
    mapping.update(overrides)
    mapping["shape"] = tuple(mapping["shape"])
    with pytest.raises(ValueError):
        CaptureMetadata(**mapping)


@pytest.mark.parametrize(
    "dim",
    [
        # The seven scalars above go through the checked parse; the shape
        # dimensions three lines below them in the same function accumulate
        # into a bare uint32_t with no bound, so a dimension over 2**32 wraps
        # into range and is PACKED instead of being refused. 2**31 itself is
        # refused by ValidateMetadata -- only the WRAPPED values slip through,
        # which is why no dimension bound test caught this.
        2**32 + 1,
        2**32 + 3,
        2**64 + 5,
        # A digit run far longer than any 64-bit value still wraps to 7.
        2**32 * 10**25 + 7,
    ],
)
def test_submit_row_refuses_a_shape_dimension_that_does_not_fit(sink, tmp_path,
                                                                dim):
    _open(sink, tmp_path / "spool")
    # The envelope carries the value the metadata WRAPS to, which is what let
    # the row past the envelope/metadata shape agreement check and into a pack.
    envelope = [dim % 2**32]
    assert 0 < envelope[0] <= 2**31 - 1, envelope
    metadata = _row_meta(0, dtype="uint8", shape=[dim], token_start=0)
    response = _submit_row(sink, metadata, bytes(envelope[0]), "uint8",
                           envelope)
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 0
    mapping = _meta(0).to_mapping()
    mapping.update(dtype="uint8", shape=(dim,), token_start=0)
    with pytest.raises(ValueError):
        CaptureMetadata(**mapping)


def test_submit_row_keeps_the_shape_dimension_boundary_exact(sink, tmp_path):
    """2**31 - 1 parses and is refused by validation, not by the bound.

    The dimension bound is the accumulator's, exactly as it is for the
    scalars: it refuses a literal with no uint32 to hold it, and leaves the
    field's own narrower range (0 .. 2**31 - 1) to ValidateMetadata. If the
    two were merged, the reason reported for 2**31 would be wrong.
    """
    _open(sink, tmp_path / "spool")
    # 2**31 - 1 is the largest dimension CaptureMetadata admits: the parse
    # must carry it through to the payload-size check, not refuse it.
    metadata = _row_meta(0, dtype="uint8", shape=[2**31 - 1], token_start=0)
    response = _submit_row(sink, metadata, b"", "uint8", [2**31 - 1])
    assert not response["ok"], response
    assert response["what"].startswith("payload length does not match"), response
    assert str(2**31 - 1) in response["what"], response

    # One past it, and the whole way to the accumulator's own ceiling, the
    # reason is validation -- the field's range -- and not the bound.
    for dim in (2**31, 2**32 - 1):
        metadata = _row_meta(0, dtype="uint8", shape=[dim], token_start=0)
        response = _submit_row(sink, metadata, b"", "uint8", [dim])
        assert not response["ok"], (dim, response)
        assert response["what"] == "capture metadata failed validation", (
            dim, response)

    # A legal dimension still packs.
    metadata = _row_meta(0, dtype="uint8", shape=[64], token_start=0)
    assert _submit_row(sink, metadata, bytes(64), "uint8", [64])["ok"]


INTEGER_FIELDS = ("layer_number", "producer_rank", "step_number",
                  "token_start", "token_end", "batch_position",
                  "captured_at_ns")


@pytest.mark.parametrize("literal", ("null", '"abc"', "true", '""', "[]"))
@pytest.mark.parametrize("field", INTEGER_FIELDS)
def test_submit_row_refuses_a_present_field_that_is_not_an_integer(
        sink, tmp_path, field, literal):
    """A key that is present with a non-integer value is not a -1.

    The presence loop above these fields runs HasKey, so by the time the
    scan reports kAbsent the key IS there and simply is not an integer. The
    parse handed every such value the historical -1, which for layer_number
    is that field's legal "no layer" sentinel: `"layer_number": null` and
    `"layer_number": "abc"` were both admitted and packed as -1. The oracle
    raises PackFormatError "invalid capture metadata: layer_number must be
    an integer in [-1, 2^31 - 1]".

    Deleting the FindInt wrapper made this -1 convention the only sentinel
    left in the parse, which is why it is fixed here rather than deferred
    again.
    """
    _open(sink, tmp_path / "spool")
    mapping = _meta(0).to_mapping()
    mapping.update(dtype="uint8", shape=[64], token_start=0)
    metadata = json.dumps(mapping)
    needle = f'"{field}": {mapping[field]}'
    assert metadata.count(needle) == 1, (needle, metadata)
    metadata = metadata.replace(needle, f'"{field}": {literal}')

    response = _submit_row(sink, metadata, bytes(64), "uint8", [64])
    assert not response["ok"], (field, literal, response)
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 0, snapshot

    mapping[field] = json.loads(literal)
    with pytest.raises(PackFormatError):
        CaptureMetadata.from_mapping(mapping)


def test_submit_row_keeps_the_64_bit_boundaries_exact(sink, tmp_path):
    """Everything a 64-bit field can legally hold must still parse exactly."""
    _open(sink, tmp_path / "spool")
    payload = bytes(64)
    # The counters are UInt64 in the catalog, so the whole unsigned range is
    # legal input and must survive the parse bit for bit -- not just the half
    # of it that fits in an int64. captured_at_ns stays under 2**63 because
    # the spool's ready-file name is what bounds it, not the parse.
    metadata = _row_meta(0, dtype="uint8", shape=[64], layer_number=-1,
                         step_number=2**64 - 1, token_start=2**63,
                         token_end=2**64 - 1, captured_at_ns=2**63 - 1,
                         producer_rank=2**32 - 1, batch_position=0)
    response = _submit_row(sink, metadata, payload, "uint8", [64])
    assert response["ok"], response
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 1
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    recovered = spool.recover()
    with recovered[0].open() as handle:
        descriptors = PackReader.from_bytes(handle.read()).descriptors(
            store_id="spool", object_key=recovered[0].object_key
        )
    stored = descriptors[0].metadata
    assert stored.layer_number == -1
    assert stored.step_number == 2**64 - 1
    assert stored.token_start == 2**63
    assert stored.token_end == 2**64 - 1
    assert stored.captured_at_ns == 2**63 - 1
    assert stored.producer_rank == 2**32 - 1
    assert stored.batch_position == 0


@pytest.mark.parametrize(
    "field", ("step_number", "token_start", "token_end", "captured_at_ns")
)
def test_submit_row_admits_the_whole_unsigned_range(sink, tmp_path, field):
    """2**64 - 1 is what CaptureMetadata allows, so the row path must too."""
    _open(sink, tmp_path / "spool")
    overrides = {"dtype": "uint8", "shape": [64], "token_start": 0,
                 "token_end": 2**64 - 1}
    overrides[field] = 2**64 - 1
    response = _submit_row(sink, _row_meta(0, **overrides), bytes(64),
                           "uint8", [64])
    assert response["ok"], response


def test_submit_row_parses_the_int64_limits_before_validating_them(sink,
                                                                   tmp_path):
    """INT64_MAX/INT64_MIN/0/-0 parse exactly; only validation refuses them.

    layer_number is the only signed field, so it is the one place where the
    difference between "parsed, then out of the field's range" and "does not
    fit in 64 bits at all" is observable. If the parser started refusing the
    int64 limits themselves, these would report the wrong reason.
    """
    _open(sink, tmp_path / "spool")
    payload = bytes(64)

    for limit in (2**63 - 1, -(2**63)):
        metadata = _row_meta(0, dtype="uint8", shape=[64], token_start=0,
                             layer_number=limit)
        response = _submit_row(sink, metadata, payload, "uint8", [64])
        assert not response["ok"], response
        assert "out of range" not in response["what"], (limit, response)
        assert "validation" in response["what"], (limit, response)

    # The negative limit is asymmetric: |INT64_MIN| is one larger than
    # INT64_MAX, so the pair below is what pins it. One step past INT64_MIN
    # flips the reason from validation to out-of-range, and nothing else does.
    metadata = _row_meta(0, dtype="uint8", shape=[64], token_start=0,
                         layer_number=-(2**63) - 1)
    response = _submit_row(sink, metadata, payload, "uint8", [64])
    assert not response["ok"] and "out of range" in response["what"], response

    # 0, -0 and a plain small value are all accepted, -0 included: the raw
    # literal never reaches Python's int, so it stays on the wire.
    for index, literal in enumerate(("0", "-0", "3")):
        metadata = _row_meta(index, dtype="uint8", shape=[64], token_start=0,
                             token_end=1, layer_number=0)
        metadata = metadata.replace('"layer_number": 0',
                                    '"layer_number": ' + literal)
        assert '"layer_number": ' + literal in metadata
        response = _submit_row(sink, metadata, payload, "uint8", [64])
        assert response["ok"], (literal, response)

    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["submitted_records"] == 3


# --- the driver's own integer fields -------------------------------------------
#
# `submit_row` refuses these already, because SubmitRow's metadata parse is
# checked. The driver's OTHER integer decodes -- the `open` config, the
# `object_key` arguments and `submit`'s metadata mapping -- were not, and each
# read FindInt's -1 as something legal: the unsigned config limits became
# 18446744073709551615, `num_workers` fell back to 1, and a metadata counter
# wrapped modulo 2**64 and was admitted.


@pytest.mark.parametrize(
    "field",
    ("max_bytes", "max_queue_records", "max_queue_bytes", "max_pack_bytes",
     "max_pack_records", "max_linger_ns", "num_workers"),
)
def test_open_refuses_an_out_of_range_limit(sink, tmp_path, field):
    """A limit that cannot be represented must not become UINT64_MAX."""
    config = {
        "op": "open", "root": str(tmp_path / "spool"), "max_bytes": 1 << 40,
        "max_queue_records": 256, "max_queue_bytes": 16 * 1024 * 1024,
        "max_pack_bytes": 128 * 1024 * 1024, "max_pack_records": 10_000,
        "max_linger_ns": 1_000_000_000, "overload": "drop_newest",
        "admission_timeout": -1, "num_workers": 1,
    }
    config[field] = 2**64 + 1
    response = sink.call(**config)
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert field in response["what"], response
    # The sink was never constructed, so the next op sees no open sink.
    assert sink.call(op="snapshot")["what"] == "sink is not open"


@pytest.mark.parametrize(
    "field, value",
    [
        ("producer_rank", 2**64 + 1),
        ("captured_at_ns", 2**64 + 1),
        ("producer_rank", int("9" * 40)),
        ("captured_at_ns", -(2**63) - 1),
    ],
)
def test_object_key_refuses_an_out_of_range_integer(sink, field, value):
    """-1 rendered rank=18446744073709551615 into a key Python never mints."""
    request = {
        "op": "object_key", "tenant_id": "tenant", "session_id": "session",
        "producer_rank": 0, "captured_at_ns": 1_700_000_000_000_000_000,
        "pack_id": "018f0000-0000-7000-8000-000000000f01",
    }
    request[field] = value
    response = sink.call(**request)
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert field in response["what"], response


@pytest.mark.parametrize(
    "field, value",
    [
        ("captured_at_ns", 2**64 + 1),
        ("step_number", 2**64),
        ("token_end", 2**64 + 7),
        ("producer_rank", 2**64 + 5),
        ("batch_position", 2**64 + 5),
        ("layer_number", 2**64 + 3),
        # Inside the union, aliasing onto the legal "no layer" sentinel --
        # the same defect the row path carries, at the driver's own decode.
        ("layer_number", 2**64 - 1),
        ("token_start", -(2**63) - 1),
        ("layer_number", -(2**63) - 1),
        ("captured_at_ns", int("9" * 40)),
    ],
)
def test_submit_refuses_out_of_range_metadata_integers(sink, tmp_path, field,
                                                       value):
    """The mapping path must refuse what the row path already refuses."""
    _open(sink, tmp_path / "spool")
    mapping = _meta(0).to_mapping()
    mapping[field] = value
    response = sink.call(
        op="submit", metadata=mapping,
        payload_b64=base64.b64encode(bytes(64)).decode(),
    )
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert field in response["what"], response
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["submitted_records"] == 0, snapshot


@pytest.mark.parametrize(
    "dim",
    [
        # ParseMetadata puts the seven scalars above through the checked
        # parse and then accumulates the shape dimensions into a bare
        # uint32_t seven lines below them, with no bound -- the same defect
        # the row path carried at record_row.cpp and the pack driver carries
        # at meta_shape. A dimension over 2**32 wraps INTO range: [2**32 + 1]
        # arrived as (1,) and was admitted and persisted, where
        # CaptureMetadata raises "shape dimensions must be integers in
        # [0, 2^31 - 1]".
        2**32 + 1,
        2**32 + 3,
        2**64 + 5,
        # A digit run far longer than any 64-bit value still wraps to 7.
        2**32 * 10**25 + 7,
    ],
)
def test_submit_refuses_a_shape_dimension_that_does_not_fit(sink, tmp_path,
                                                            dim):
    """The mapping path must refuse the dimension the row path refuses."""
    _open(sink, tmp_path / "spool")
    wrapped = dim % 2**32
    assert 0 < wrapped <= 2**31 - 1, wrapped
    mapping = _meta(0, dtype="uint8", shape=(64,)).to_mapping()
    mapping["shape"] = [dim]
    response = sink.call(
        op="submit", metadata=mapping,
        payload_b64=base64.b64encode(bytes(wrapped)).decode(),
    )
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert "shape" in response["what"], response
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["submitted_records"] == 0, snapshot
    assert snapshot["persisted_records"] == 0, snapshot
    oracle = _meta(0).to_mapping()
    oracle.update(dtype="uint8", shape=(dim,))
    with pytest.raises(ValueError):
        CaptureMetadata(**oracle)


def test_submit_keeps_the_shape_dimension_boundary_exact(sink, tmp_path):
    """A legal wide dimension still reaches the sink.

    The bound is the accumulator's, exactly as it is for the scalars: it
    refuses a literal with no uint32 to hold it and nothing narrower, so
    this op cannot pass the test above by refusing every wide dimension.
    """
    _open(sink, tmp_path / "spool")
    # A wholly ordinary row still packs, first: the wide dimensions below
    # describe more bytes than they carry, which latches the sink on the
    # packing thread -- a separate concern from this parse.
    assert _submit(sink, CaptureRecord(_meta(1), bytes(64))) == "accepted"
    for dim in (0, 1, 2**31 - 1, 2**32 - 1):
        mapping = _meta(0, dtype="uint8", shape=(64,)).to_mapping()
        mapping["shape"] = [dim]
        response = sink.call(
            op="submit", metadata=mapping,
            payload_b64=base64.b64encode(b"").decode(),
        )
        assert response["ok"], (dim, response)
        assert "admission" in response, (dim, response)


@pytest.mark.parametrize(
    "dim",
    [
        # The FOURTH accumulation of the same class: submit_row's ENVELOPE
        # shape, an int64_t. It wraps modulo 2**64, not 2**32, so the 32-bit
        # witnesses above do NOT wrap here -- 2**32 + 1 fits an int64 and
        # correctly mismatches the metadata. Only a literal past the 64-bit
        # accumulator aliases, and it aliases ONTO the metadata dimension,
        # which is what carried it past the envelope/metadata agreement check
        # and into a pack: 2**64 + 1 against metadata shape [1] returned
        # {"ok": true} and persisted 1 record.
        2**64 + 1,
        2**64 + 3,
        2**64 * 10**6 + 7,
    ],
)
def test_submit_row_refuses_an_envelope_dimension_that_does_not_fit(
        sink, tmp_path, dim):
    _open(sink, tmp_path / "spool")
    aliased = dim % 2**64
    assert 0 < aliased <= 2**31 - 1, aliased
    metadata = _row_meta(0, dtype="uint8", shape=[aliased], token_start=0)
    response = _submit_row(sink, metadata, bytes(aliased), "uint8", [dim])
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert "shape" in response["what"], response
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 0, snapshot
    oracle = _meta(0).to_mapping()
    oracle.update(dtype="uint8", shape=(dim,), token_start=0)
    with pytest.raises(ValueError):
        CaptureMetadata(**oracle)


def test_submit_row_keeps_the_envelope_dimension_boundary_exact(sink,
                                                                tmp_path):
    """INT64_MAX is what the envelope accumulator holds, so it must parse.

    Past the metadata's own range the answer must stay "envelope shape !=
    metadata shape" -- the mismatch -- and not become an unrepresentable
    literal, so this op cannot pass the test above by refusing every wide
    envelope dimension.
    """
    _open(sink, tmp_path / "spool")
    for dim in (2**31, 2**32 + 1, 2**63 - 1):
        metadata = _row_meta(0, dtype="uint8", shape=[1], token_start=0)
        response = _submit_row(sink, metadata, b"\x00", "uint8", [dim])
        assert not response["ok"], (dim, response)
        assert response["what"] == "envelope shape != metadata shape", (
            dim, response)
    # 2**63 wraps to INT64_MIN, which the negative check already catches --
    # it must stay a mismatch too, not become an accepted dimension.
    metadata = _row_meta(0, dtype="uint8", shape=[1], token_start=0)
    response = _submit_row(sink, metadata, b"\x00", "uint8", [2**63])
    assert not response["ok"], response
    # A legal row still packs.
    metadata = _row_meta(1, dtype="uint8", shape=[64], token_start=1)
    assert _submit_row(sink, metadata, bytes(64), "uint8", [64])["ok"]


@pytest.mark.parametrize(
    "field", ("step_number", "token_start", "token_end", "captured_at_ns")
)
def test_submit_admits_the_whole_unsigned_range(sink, tmp_path, field):
    """2**64 - 1 is what CaptureMetadata allows, so `submit` must too.

    captured_at_ns is the one exception in practice, not in the parse: the
    spool's ready-file name bounds it, so the record is admitted here and
    the staging refusal (if any) is a separate concern.
    """
    _open(sink, tmp_path / "spool")
    mapping = _meta(0).to_mapping()
    mapping["token_start"] = 0
    mapping[field] = 2**64 - 1
    response = sink.call(
        op="submit", metadata=mapping,
        payload_b64=base64.b64encode(bytes(64)).decode(),
    )
    assert response["ok"], response
    assert response["admission"] == "accepted", response


def test_object_key_keeps_the_64_bit_boundaries_exact():
    """INT64_MAX, 2**64 - 1, 0 and -0 all still reach the key builder.

    The value being parametrized is `captured_at_ns`, and it IS observable:
    `object_key_for` derives the `date=` segment from it. So the whole key
    is compared against the oracle, the way `test_object_key_parity_with_
    python` does it -- pinning only "rank=4294967295" observed nothing but
    `producer_rank`, which is fixed across the loop, so a clamp or a wrong
    wide-value computation of `captured_at_ns` passed.

    `-0` is sent as a raw literal: `json.dumps(-0)` emits `0`, so going
    through `call` would not have exercised the JSON scan's negative arm at
    all. This is the same trick `test_minus_zero_is_still_zero` and
    `test_submit_row_parses_the_int64_limits_before_validating_them` use.
    """
    pack_id = "018f0000-0000-7000-8000-000000000f01"
    session = SinkSession()
    try:
        for captured in (0, 2**63 - 1, 2**63, 2**64 - 1):
            response = session.call(
                op="object_key", tenant_id="tenant", session_id="session",
                producer_rank=2**32 - 1, captured_at_ns=captured,
                pack_id=pack_id,
            )
            assert response["ok"], (captured, response)
            assert response["object_key"] == _python_object_key(
                "tenant", "session", 2**32 - 1, captured, pack_id), captured

        # -0 on the wire, which json.dumps would have flattened to 0.
        line = json.dumps({
            "op": "object_key", "tenant_id": "tenant",
            "session_id": "session", "producer_rank": 2**32 - 1,
            "captured_at_ns": 0, "pack_id": pack_id,
        })
        line = line.replace('"captured_at_ns": 0', '"captured_at_ns": -0')
        assert '"captured_at_ns": -0' in line
        response = session.call_line(line)
        assert response["ok"], response
        assert response["object_key"] == _python_object_key(
            "tenant", "session", 2**32 - 1, 0, pack_id), response
    finally:
        session.close()


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


# --- metadata decoding at the sink boundary -----------------------------------
#
# The sink's metadata parser is the native side of CaptureMetadata.from_mapping:
# everything the Python constructor refuses must be refused here, and
# everything it admits must come out of the staged pack byte-identical.
# Three gaps were reachable through submit_row with a valid envelope:
#   * a JSON surrogate pair (json.dumps writes U+1F600 as 😀) was
#     decoded as two three-byte sequences -- six bytes that are not UTF-8;
#   * a digit prefix stood in for the whole number (0.5 -> 0, 123.5 -> 123,
#     [16.5] -> [16]) and a negative counter wrapped to 2**64-1;
#   * the scalar shape [] was refused as "not an integer list".


def _read_staged_metadata(root):
    spool = DurablePackSpool(root, max_bytes=1 << 40)
    recovered = spool.recover()
    assert len(recovered) == 1, recovered
    with recovered[0].open() as handle:
        descriptors = PackReader.from_bytes(handle.read()).descriptors(
            store_id="spool", object_key=recovered[0].object_key
        )
    return [d.metadata for d in descriptors]


@pytest.mark.parametrize("hook_name", ["block\U0001F600", "block中文"])
def test_a_non_bmp_identifier_survives_the_metadata_decoder(sink, tmp_path,
                                                             hook_name):
    """A surrogate pair is ONE code point; the BMP name is the control."""
    _open(sink, tmp_path / "spool")
    metadata = _row_meta(0, dtype="uint8", shape=[64], hook_name=hook_name)
    assert "\\ud83d\\ude00" in metadata or "\\u4e2d" in metadata, metadata
    response = _submit_row(sink, metadata, bytes(64), "uint8", [64])
    assert response["ok"], response
    assert sink.call(op="flush", timeout=30)["ok"]
    assert sink.call(op="close", timeout=30)["snapshot"]["persisted_records"] == 1
    (staged,) = _read_staged_metadata(tmp_path / "spool")
    assert staged.hook_name == hook_name
    assert staged.hook_name.encode("utf-8") == hook_name.encode("utf-8")


@pytest.mark.parametrize("field, value, reason", [
    ("layer_number", 0.5, "not an integer"),
    ("captured_at_ns", 123.5, "not an integer"),
    ("step_number", -1, "out of range"),
    ("producer_rank", -1, "out of range"),
    ("token_start", -1, "out of range"),
    ("batch_position", -1, "out of range"),
])
def test_invalid_numeric_metadata_is_refused_not_converted(sink, tmp_path,
                                                            field, value,
                                                            reason):
    """The whole token must be an integer, and unsigned counters non-negative.

    Python's CaptureMetadata refuses each of these (the oracle assertion
    below); the sink used to persist 0, 123 and 18446744073709551615.
    """
    from dmi.storage.capture.model import CaptureStorageError

    _open(sink, tmp_path / "spool")
    metadata = _row_meta(0, dtype="uint8", shape=[64], **{field: value})
    response = _submit_row(sink, metadata, bytes(64), "uint8", [64])
    assert not response["ok"], (field, value, response)
    assert reason in response["what"], (field, value, response)
    with pytest.raises((CaptureStorageError, ValueError, TypeError)):
        CaptureMetadata.from_mapping(json.loads(metadata))
    assert sink.call(op="close", timeout=30)["snapshot"]["submitted_records"] == 0


def test_a_fractional_shape_dimension_is_refused_not_truncated(sink, tmp_path):
    from dmi.storage.capture.model import CaptureStorageError

    _open(sink, tmp_path / "spool")
    metadata = _row_meta(0, dtype="uint8", shape=[64])
    metadata = metadata.replace('"shape": [64]', '"shape": [64.5]')
    assert '"shape": [64.5]' in metadata
    response = _submit_row(sink, metadata, bytes(64), "uint8", [64])
    assert not response["ok"], response
    assert "integer list" in response["what"], response
    with pytest.raises((CaptureStorageError, ValueError, TypeError)):
        CaptureMetadata.from_mapping(json.loads(metadata))


def test_a_scalar_shape_is_admitted_as_one_element(sink, tmp_path):
    """shape=[] is rank 0: one element, dtype-width bytes."""
    _open(sink, tmp_path / "spool")
    metadata = _row_meta(0, dtype="float32", shape=[])
    assert CaptureMetadata.from_mapping(json.loads(metadata)).shape == ()
    payload = b"\x00\x00\x60\x40"  # 3.5f
    response = _submit_row(sink, metadata, payload, "float32", [])
    assert response["ok"], response
    assert sink.call(op="flush", timeout=30)["ok"]
    assert sink.call(op="close", timeout=30)["snapshot"]["persisted_records"] == 1
    (staged,) = _read_staged_metadata(tmp_path / "spool")
    assert staged.shape == ()
    assert staged.dtype == "float32"


def test_a_missing_shape_is_still_refused(sink, tmp_path):
    """The scalar fix must not turn an ABSENT shape into a scalar."""
    _open(sink, tmp_path / "spool")
    mapping = json.loads(_row_meta(0, dtype="uint8", shape=[64]))
    del mapping["shape"]
    response = _submit_row(sink, json.dumps(mapping), bytes(64), "uint8", [64])
    assert not response["ok"], response
    assert "integer list" in response["what"], response
