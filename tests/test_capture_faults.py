"""How the capture storage path behaves when its dependencies misbehave.

Phase 6 cannot switch the default sink until this is characterised, and the
Python implementation is now the conformance reference -- so these tests double
as the specification a native writer has to satisfy. Each one names the fault,
then asserts the *observable* consequence: what survives, what is reported, and
what is never silently lost.
"""

from __future__ import annotations

from pathlib import Path
import time
from uuid import UUID

import pytest

from tests._faults import (
    FaultInjected,
    FaultyClickHouseClient,
    FaultyPackSink,
    FaultyPackStore,
    duplicate_on,
    fail_on,
    fail_then_succeed,
    truncate_on,
)

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    CatalogIndexer,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    DirectPackSink,
    FilesystemPackStore,
    HostCapturePipeline,
    PackIndex,
    PackFormatError,
    PackWriter,
    PipelineConfig,
    PipelineFailedError,
)


pytestmark = pytest.mark.cpu


PACK_ID = UUID("018f0000-0000-7000-8000-00000000fa01")


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


def _record(capture_id: str, *, step: int = 0) -> CaptureRecord:
    return CaptureRecord(metadata=_metadata(capture_id, step=step), payload=b"\x01" * 8)


def _sealed(*records: CaptureRecord, pack_id: UUID = PACK_ID):
    writer = PackWriter(
        pack_id=pack_id,
        created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=1024 * 1024,
    )
    for record in records:
        writer.append(record)
    return writer.seal()


def _ids():
    from itertools import count

    counter = count(1)
    return lambda: UUID(int=next(counter))


# --- object store faults -----------------------------------------------------


def test_a_short_read_is_detected_rather_than_silently_truncating(tmp_path: Path):
    inner = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed(_record("capture-a"))
    ref = inner.put(sealed, "packs/a.dmi-pack")
    store = FaultyPackStore(inner, read_range=truncate_on(1, by=1))

    # A store returning fewer bytes than asked for must never be mistaken for a
    # valid short object -- that would corrupt a payload silently. The trailer
    # is fixed-width, so a short read cannot even be unpacked.
    with pytest.raises(PackFormatError, match="trailer is truncated"):
        PackIndex.from_store(store, ref).descriptors()


def test_a_read_failure_propagates_instead_of_producing_a_partial_pack(
    tmp_path: Path,
):
    inner = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed(_record("capture-a"), _record("capture-b", step=1))
    ref = inner.put(sealed, "packs/a.dmi-pack")
    store = FaultyPackStore(inner, read_range=fail_on(2))

    with pytest.raises(FaultInjected):
        PackIndex.from_store(store, ref).descriptors()

    # The trailer read happened; the footer read is what failed.
    assert store.call_counts["read_range"] == 2


def test_an_immutable_key_written_twice_is_not_a_conflict(tmp_path: Path):
    inner = FilesystemPackStore(tmp_path, store_id="local")
    store = FaultyPackStore(inner, put=duplicate_on(1))
    sealed = _sealed(_record("capture-a"))

    # An ambiguous upload that actually landed twice must converge, because the
    # writer cannot tell "never arrived" from "arrived, ack lost".
    ref = store.put(sealed, "packs/a.dmi-pack")

    assert ref.checksum == sealed.checksum
    assert store.stat(ref).size == len(sealed.data)


# --- pipeline / sink faults --------------------------------------------------


def test_a_sink_failure_is_surfaced_and_closes_admission(tmp_path: Path):
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    sink = FaultyPackSink(DirectPackSink(store), persist=fail_on(1))
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=1 << 20,
            max_pack_bytes=1024 * 1024,
            max_pack_records=1,
            max_linger_ns=1_000_000_000,
        ),
        sink,
        pack_id_factory=_ids(),
    )
    pipeline.start()
    pipeline.submit(_record("capture-a"))

    # Losing durable storage is not recoverable at this layer, so it must fail
    # loudly rather than continue accepting captures it cannot persist.
    with pytest.raises(PipelineFailedError):
        pipeline.close(timeout=2)
    assert pipeline.snapshot().failures == 1


def test_admission_is_refused_once_the_pipeline_has_failed(tmp_path: Path):
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    sink = FaultyPackSink(DirectPackSink(store), persist=fail_on(1))
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=1 << 20,
            max_pack_bytes=1024 * 1024,
            max_pack_records=1,
            max_linger_ns=1_000_000_000,
        ),
        sink,
        pack_id_factory=_ids(),
    )
    pipeline.start()
    pipeline.submit(_record("capture-a"))

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and pipeline.snapshot().failures == 0:
        time.sleep(0.01)
    assert pipeline.snapshot().failures == 1

    # Once persistence is broken, further submissions must be refused rather
    # than accepted into a queue nothing will ever drain.
    with pytest.raises(PipelineFailedError):
        pipeline.submit(_record("capture-c", step=2))
    with pytest.raises(PipelineFailedError):
        pipeline.close(timeout=2)


# --- catalog / ClickHouse faults ---------------------------------------------


class _Client:
    """Minimal in-memory ClickHouse stand-in that records inserted rows.

    Holds just enough state to answer the version allocator's three queries:
    the max-claim select, the claim insert, and the per-version owner select.
    """

    def __init__(self):
        self.inserted: list[tuple[str, list]] = []
        self.claims: list[tuple[int, str]] = []

    def execute(self, query, params=None, **kwargs):
        if query.lstrip().upper().startswith("INSERT"):
            self.inserted.append((query, list(params or [])))
            if "version_claims" in query:
                self.claims.extend((row[0], str(row[1])) for row in params)
            return []
        if "version_claims" in query:
            if "max(version)" in query:
                return [(max((v for v, _ in self.claims), default=None),)]
            wanted = params["version"]
            return [(cid,) for v, cid in self.claims if v == wanted]
        if "index_watermark" in query:
            published = [
                p[0][0] for q, p in self.inserted if "index_watermark" in q
            ]
            return [(max(published, default=None),)]
        return []


def _indexer(store, client, **config):
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig(**config))
    return CatalogIndexer(store, writer, clock_ns=lambda: 7)


def test_an_insert_failure_does_not_commit_the_pack(tmp_path: Path):
    inner = FilesystemPackStore(tmp_path, store_id="local")
    ref = inner.put(_sealed(_record("capture-a")), "packs/a.dmi-pack")
    # Insert 1 is the version claim; insert 2 is the descriptor batch.
    client = FaultyClickHouseClient(_Client(), insert=fail_on(2))

    with pytest.raises(FaultInjected):
        _indexer(inner, client).index([ref])

    # Descriptors are written before the pack commit marker precisely so a
    # failure here leaves the pack uncommitted and the batch replayable.
    assert client.call_counts.get("insert") == 2


def test_a_duplicated_insert_is_absorbed_by_replay_semantics(tmp_path: Path):
    inner = FilesystemPackStore(tmp_path, store_id="local")
    ref = inner.put(_sealed(_record("capture-a")), "packs/a.dmi-pack")
    backing = _Client()
    # Insert 1 is the version claim; duplicate the descriptor insert (2).
    client = FaultyClickHouseClient(backing, insert=duplicate_on(2))

    result = _indexer(inner, client).index([ref])

    # The physical row lands twice; that is expected and is what the
    # ReplacingMergeTree views collapse. What must not happen is a failure.
    assert result.failed_packs == 0
    assert result.indexed_rows == 1
    descriptor_inserts = [q for q, _ in backing.inserted if "capture_raw" in q]
    assert len(descriptor_inserts) == 2


def test_a_corrupt_pack_fails_only_its_own_pack(tmp_path: Path):
    inner = FilesystemPackStore(tmp_path, store_id="local")
    good = inner.put(_sealed(_record("capture-a")), "packs/good.dmi-pack")
    bad = inner.put(
        _sealed(_record("capture-b", step=1), pack_id=UUID(int=0xBAD)),
        "packs/bad.dmi-pack",
    )
    # Fail the footer read of the second pack only.
    store = FaultyPackStore(inner, read_range=fail_on(3, 4))

    result = _indexer(store, _Client()).index([good, bad])

    # One bad pack must not poison the batch: the healthy pack still indexes,
    # and the failure is attributed to the pack that caused it.
    assert result.indexed_packs == 1
    assert result.failed_packs == 1
    assert result.failures[0].object_key == "packs/bad.dmi-pack"


def test_a_transient_outage_is_survivable_by_retrying_the_batch(tmp_path: Path):
    inner = FilesystemPackStore(tmp_path, store_id="local")
    ref = inner.put(_sealed(_record("capture-a")), "packs/a.dmi-pack")
    backing = _Client()
    client = FaultyClickHouseClient(backing, insert=fail_then_succeed(1))

    with pytest.raises(FaultInjected):
        _indexer(inner, client).index([ref])

    # Nothing was committed, so the identical call now succeeds -- indexing is
    # replayable rather than requiring manual repair.
    result = _indexer(inner, client).index([ref])

    assert result.indexed_packs == 1
    assert result.failed_packs == 0
