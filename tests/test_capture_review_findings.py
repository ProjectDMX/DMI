"""Regressions found by review that the existing suites could not reach.

Each test names the gap in coverage that let the bug through, because the
pattern matters more than the individual defect: every one of these sits just
outside a boundary the tests already exercised.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import pytest

from tests._faults import FaultInjected, FaultyClickHouseClient, fail_on

from dmi.storage.capture import (
    AdmissionResult,
    CaptureMetadata,
    CaptureRecord,
    CatalogIndexer,
    CatalogIndexerConfig,
    CatalogReconciler,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    DirectPackSink,
    FilesystemPackStore,
    HostCapturePipeline,
    PackRef,
    PackWriter,
    PipelineConfig,
    object_key_for,
)
from dmi.storage.capture.filesystem import validate_object_key


pytestmark = pytest.mark.cpu


PACK_ID = UUID("018f0000-0000-7000-8000-00000000ab01")


def _metadata(**overrides) -> CaptureMetadata:
    base = dict(
        capture_id="capture-a",
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
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    base.update(overrides)
    return CaptureMetadata(**base)


def _record(metadata: CaptureMetadata) -> CaptureRecord:
    return CaptureRecord(metadata=metadata, payload=b"\x01" * 8)


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


# --- object keys -------------------------------------------------------------
#
# Gap: key generation and key validation were tested separately, never against
# each other, so a character one produced and the other refused went unnoticed.


@pytest.mark.parametrize(
    "character",
    [c for c in map(chr, range(32, 127)) if quote(c, safe="-_.=") == c],
    ids=lambda c: f"U+{ord(c):04X}",
)
def test_every_character_key_encoding_passes_through_is_accepted(character: str):
    """Whatever the encoder leaves alone, the validator must accept.

    `quote` treats `~` as always-safe per RFC 3986 regardless of its `safe`
    argument, so it survives encoding -- but the key pattern rejects it.
    """
    metadata = _metadata(tenant_id=f"acme{character}lab")

    from dmi.storage.capture.pipeline import _key_component

    encoded = _key_component(metadata.tenant_id)
    validate_object_key(f"tenant={encoded}/x.dmi-pack")


def test_an_identifier_needing_no_escape_still_yields_a_usable_key(tmp_path: Path):
    """The end-to-end consequence: an unencodable id kills the pipeline.

    The key is rejected by the store, the sink raises, and the persistence
    thread treats that as fatal -- so one tenant name takes down capture for
    every tenant.
    """
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=1 << 20,
            max_pack_bytes=1024 * 1024,
            max_pack_records=1,
            max_linger_ns=1_000_000_000,
        ),
        DirectPackSink(store),
        pack_id_factory=_ids(),
    )
    pipeline.start()

    assert (
        pipeline.submit(_record(_metadata(tenant_id="acme~lab")))
        is AdmissionResult.ACCEPTED
    )
    snapshot = pipeline.close(timeout=2)

    assert snapshot.failures == 0, "an unencodable identifier failed the pipeline"
    assert snapshot.persisted_records == 1


# --- catalog commit ordering -------------------------------------------------
#
# Gap: the fault tests failed whole operations, never the gap *between* two
# writes that must agree.


class _Client:
    def __init__(self):
        self.statements: list[str] = []
        self.published: list[tuple[str, list]] = []
        self.claims: list[tuple[int, str]] = []

    def execute(self, query, params=None, **kwargs):
        self.statements.append(query)
        if query.lstrip().upper().startswith("INSERT"):
            self.published.append((query, list(params or [])))
            if "version_claims" in query:
                self.claims.extend((row[0], str(row[1])) for row in params)
            return []
        # The version allocator's three queries need real state to answer.
        if "version_claims" in query:
            if "max(version)" in query:
                return [(max((v for v, _ in self.claims), default=None),)]
            wanted = params["version"]
            return [(cid,) for v, cid in self.claims if v == wanted]
        if "index_watermark" in query:
            versions = [
                p[0][0] for q, p in self.published if "index_watermark" in q
            ]
            return [(max(versions, default=None),)]
        return []


def test_a_pack_is_never_both_skipped_on_replay_and_invisible_to_readers():
    """The durability window between the two commit writes.

    `committed_pack_ids` reads the inventory to skip replays; readers bound the
    snapshot by the commit log. If the inventory is written first and the
    process dies before the log, the pack is skipped forever *and* never
    visible -- silent, permanent data loss. Writing the log first makes the
    same crash merely redundant work.
    """
    backing = _Client()
    # Fail the second of the two INSERTs that commit_packs performs.
    client = FaultyClickHouseClient(backing, insert=fail_on(2))
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())
    ref = PackRef(
        pack_id=str(PACK_ID),
        store_id="local",
        object_key="packs/a.dmi-pack",
        object_bytes=1024,
        checksum="0" * 64,
        record_count=1,
    )

    with pytest.raises(FaultInjected):
        writer.commit_packs([ref], index_version=1)

    written = [s for s in backing.statements if s.startswith("INSERT")]
    assert len(written) == 1, "expected exactly one of the two writes to land"
    assert "pack_commit_log" in written[0], (
        "the surviving write was the inventory, so this pack is now skipped on "
        "replay and invisible to readers; the commit log must be written first"
    )


# --- indexer robustness ------------------------------------------------------
#
# Gap: reconciliation was tested over buckets containing only valid packs.


class _Inventory:
    """A store whose bucket contains one object that is not a pack."""

    store_id = "local"

    def __init__(self, store, refs, bad_key: str):
        self._store = store
        self._refs = {ref.object_key: ref for ref in refs}
        self._bad_key = bad_key

    def inspect(self, object_key: str) -> PackRef:
        if object_key == self._bad_key:
            raise ValueError(f"not a dmi pack: {object_key}")
        return self._refs[object_key]

    def read_range(self, ref, offset, length):
        return self._store.read_range(ref, offset, length)

    def put(self, pack, object_key):
        return self._store.put(pack, object_key)

    def stat(self, ref):
        return self._store.stat(ref)


def test_one_foreign_object_in_the_bucket_does_not_abort_reconciliation(
    tmp_path: Path,
):
    """A bucket holds whatever anyone put there.

    `index_object_keys` inspects every key before handing the batch to the
    indexer, outside its per-pack failure handling, so a single unreadable
    object aborts the whole rebuild instead of being recorded as one failure.
    """
    store = FilesystemPackStore(tmp_path, store_id="local")
    good = store.put(_sealed(_record(_metadata())), "packs/good.dmi-pack")
    inventory = _Inventory(store, [good], bad_key="packs/README.txt")
    indexer = CatalogIndexer(inventory, ClickHouseCatalogWriter(_Client(),
                                                                ClickHouseCatalogConfig()),
                             config=CatalogIndexerConfig(max_packs=8),
                             clock_ns=lambda: 7)
    reconciler = CatalogReconciler(inventory, indexer)

    result = reconciler.index_object_keys(["packs/good.dmi-pack", "packs/README.txt"])

    assert result.indexed_packs == 1, "the valid pack should still be indexed"
    assert result.failed_packs == 1
    assert "README" in result.failures[0].object_key


# --- version monotonicity ----------------------------------------------------
#
# Gap: every indexer test used a fixed or increasing clock.


def test_a_clock_that_steps_backwards_cannot_publish_under_a_pinned_watermark(
    tmp_path: Path,
):
    """Wall clocks move backwards; versions must not.

    An NTP correction (or a second indexer with a skewed clock) once let a
    newer batch take a version below one a reader already pinned, so rows
    appeared inside a snapshot that was taken before they existed. Versions
    now come from the catalog's own allocator, so no wall-clock value
    participates in the ordering at all -- the clock stamps published_at_ns
    only, and a rollback must leave publications strictly increasing.
    """
    store = FilesystemPackStore(tmp_path, store_id="local")
    first = store.put(_sealed(_record(_metadata())), "packs/first.dmi-pack")
    second = store.put(
        _sealed(_record(_metadata(capture_id="capture-b")), pack_id=UUID(int=2)),
        "packs/second.dmi-pack",
    )

    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())
    clock = iter([2_000, 1_000])  # second batch stamped *earlier*
    indexer = CatalogIndexer(store, writer, clock_ns=lambda: next(clock))

    first_result = indexer.index([first])
    second_result = indexer.index([second])

    # Both batches must still be indexed -- refusing would stop capture over a
    # clock correction -- but the second must not reuse or undercut a version a
    # reader may already have pinned.
    assert first_result.indexed_packs == 1 and second_result.indexed_packs == 1
    published = [
        params[0][0]
        for query, params in client.published
        if "index_watermark" in query
    ]
    assert published == sorted(published), f"versions went backwards: {published}"
    assert len(set(published)) == len(published), "a version was reused"
