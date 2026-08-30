"""The catalog's DB-owned version allocator.

Catalog versions used to be derived from the indexer's wall clock, so two live
indexers with skewed clocks could publish a version below a watermark a reader
had already pinned. Versions are now handed out by the catalog itself through
an append-only, sole-claimant claims table: monotonic and unique, with no
clock anywhere in the ordering. These tests drive that protocol over a shared
in-memory "server" with real state, so two writers contend for real.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    CatalogIndexer,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    FilesystemPackStore,
    PackWriter,
)


pytestmark = pytest.mark.cpu


_SMALL_CLAIM = "00000000-0000-0000-0000-000000000000"
_LARGE_CLAIM = "ffffffff-ffff-ffff-ffff-ffffffffffff"


class _CatalogServer:
    """An in-memory stand-in for the claims and watermark tables.

    SELECT answers are computed from real state rather than canned, so two
    writers sharing one server observe each other's claims exactly the way two
    indexers sharing one ClickHouse would. ``on_owner_read`` runs between a
    claimant's INSERT and its owners SELECT -- the race window the sole-claimant
    protocol exists to close.
    """

    def __init__(self):
        self.claims: list[tuple[int, str]] = []
        self.watermarks: list[int] = []
        self.inserts: list[str] = []
        self.on_owner_read = None

    def execute(self, query, params=None, **kwargs):
        if query.lstrip().upper().startswith("INSERT"):
            self.inserts.append(query)
            if "version_claims" in query:
                self.claims.extend((int(row[0]), str(row[1])) for row in params)
            elif "index_watermark" in query:
                self.watermarks.extend(int(row[0]) for row in params)
            return []
        if "version_claims" in query:
            if "max(version)" in query:
                return [(max((v for v, _ in self.claims), default=None),)]
            version = params["version"]
            if self.on_owner_read is not None:
                self.on_owner_read(version)
            return [(cid,) for v, cid in self.claims if v == version]
        if "index_watermark" in query:
            return [(max(self.watermarks, default=None),)]
        return []


def _writer(server, **config) -> ClickHouseCatalogWriter:
    return ClickHouseCatalogWriter(server, ClickHouseCatalogConfig(**config))


# --- the allocator over a shared server ---------------------------------------


def test_two_writers_allocate_distinct_strictly_increasing_versions():
    server = _CatalogServer()
    first, second = _writer(server), _writer(server)

    versions = [writer.allocate_version() for writer in (first, second) * 3]

    assert versions == sorted(versions), f"versions went backwards: {versions}"
    assert len(set(versions)) == len(versions), "a version was reused"


def test_allocations_interleaved_with_publications_stay_monotonic():
    server = _CatalogServer()
    writer = _writer(server)

    first = writer.allocate_version()
    writer.publish_watermark(
        index_version=first, published_at_ns=1, indexed_rows=0, indexed_packs=0
    )
    second = _writer(server).allocate_version()

    assert second > first


@pytest.mark.parametrize("competitor", [_SMALL_CLAIM, _LARGE_CLAIM])
def test_a_contested_version_is_abandoned_for_a_higher_one(competitor: str):
    """Sole-claimant means ANY contest aborts, whichever claim id sorts first.

    A competing claim lands between the allocator's INSERT and its owners
    SELECT. The allocator must abandon the contested version entirely and
    return a higher one -- resolving the tie in either direction would let both
    contenders keep it when each sees only its own insert first.
    """
    server = _CatalogServer()
    writer = _writer(server)
    contested: list[int] = []

    def contest(version: int) -> None:
        contested.append(version)
        server.claims.append((version, competitor))
        server.on_owner_read = None  # contest the first attempt only

    server.on_owner_read = contest
    version = writer.allocate_version()

    assert contested == [1], "the first candidate should have been contested"
    assert version > contested[0], "a contested version must never be returned"


def test_allocation_raises_after_exhausting_its_attempts():
    server = _CatalogServer()
    writer = _writer(server, allocation_attempts=3)
    server.on_owner_read = lambda version: server.claims.append(
        (version, _LARGE_CLAIM)
    )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        writer.allocate_version()

    assert sum("version_claims" in q for q in server.inserts) == 3


def test_a_restarted_writer_allocates_above_previous_publications():
    server = _CatalogServer()
    first = _writer(server)
    version = first.allocate_version()
    first.publish_watermark(
        index_version=version, published_at_ns=1, indexed_rows=0, indexed_packs=0
    )

    # A fresh instance holds no in-memory state; the claims table alone must
    # keep it above everything already returned.
    assert _writer(server).allocate_version() > version


def test_an_allocated_version_must_fit_uint64():
    server = _CatalogServer()
    server.claims.append((2**64 - 1, _LARGE_CLAIM))

    with pytest.raises(ValueError, match="UInt64"):
        _writer(server).allocate_version()


def test_a_claims_table_returning_garbage_is_rejected():
    server = _CatalogServer()
    server.claims.append(("7", _LARGE_CLAIM))

    with pytest.raises(ValueError, match="claims table"):
        _writer(server).allocate_version()


def test_allocation_attempts_must_be_a_positive_integer():
    for value in (0, -1, 1.5):
        with pytest.raises(ValueError, match="allocation_attempts"):
            ClickHouseCatalogConfig(allocation_attempts=value)


# --- indexers over the shared server -------------------------------------------


def _metadata(capture_id: str, step: int) -> CaptureMetadata:
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


def _refs(tmp_path: Path, count: int):
    store = FilesystemPackStore(tmp_path, store_id="local")
    refs = []
    for index in range(count):
        writer = PackWriter(
            pack_id=UUID(int=index + 1),
            created_at_ns=1_700_000_000_000_000_000 + index,
            max_pack_bytes=1024 * 1024,
        )
        writer.append(CaptureRecord(_metadata(f"capture-{index}", index), b"abcdefgh"))
        sealed = writer.seal()
        refs.append(store.put(sealed, f"packs/{sealed.pack_id}.dmi-pack"))
    return store, refs


def test_clock_rollback_across_indexers_cannot_reorder_publications(tmp_path: Path):
    """Two indexers with skewed clocks, then the skew reversed.

    Publications must be strictly increasing regardless of what any wall clock
    says: the version is allocator-owned and the clock stamps published_at_ns
    only.
    """
    server = _CatalogServer()
    store, refs = _refs(tmp_path, 4)

    for clock_value, ref in ((1_000, refs[0]), (100, refs[1]),
                             (100, refs[2]), (1_000, refs[3])):
        indexer = CatalogIndexer(store, _writer(server), clock_ns=lambda v=clock_value: v)
        result = indexer.index([ref])
        assert result.indexed_packs == 1

    assert server.watermarks == sorted(server.watermarks)
    assert len(set(server.watermarks)) == len(server.watermarks)


# --- the indexer's guards against a broken allocator ---------------------------


class _HostileWriter:
    """A CatalogWriter whose allocator misbehaves in a scripted way."""

    def __init__(self, allocate, *, last_published: int = 0):
        self._allocate = allocate
        self._last_published = last_published

    def committed_pack_ids(self, identities):
        return set()

    def write_descriptors(self, descriptors, *, index_version):
        pass

    def commit_packs(self, refs, *, index_version):
        pass

    def publish_watermark(
        self, *, index_version, published_at_ns, indexed_rows, indexed_packs
    ):
        pass

    def last_published_version(self):
        return self._last_published

    def allocate_version(self):
        return self._allocate()


class _Store:
    store_id = "local"


@pytest.mark.parametrize("value", [-1, "7", True, 4.0])
def test_the_indexer_rejects_a_non_integer_or_negative_allocation(value):
    indexer = CatalogIndexer(_Store(), _HostileWriter(lambda: value))

    with pytest.raises(ValueError, match="allocate_version"):
        indexer.index([])


@pytest.mark.parametrize("allocated", [5, 4])
def test_the_indexer_refuses_a_non_monotonic_allocation(allocated: int):
    # The durable watermark says 5 has been published; an allocator handing
    # out 5 (or lower) again would let this batch land inside pinned snapshots.
    indexer = CatalogIndexer(
        _Store(), _HostileWriter(lambda: allocated, last_published=5)
    )

    with pytest.raises(RuntimeError, match="non-monotonic"):
        indexer.index([])
