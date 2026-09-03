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

from tests._catalog_fakes import FakeLeaseTable
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
    PackIndex,
    PackIntegrityError,
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

    # The encoder moved to `pack`, which is where the verifier that has to
    # agree with it lives: `_descriptors` compares a pack's footer against
    # the key it was found under.
    from dmi.storage.capture.pack import key_component

    encoded = key_component(metadata.tenant_id)
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
        self.published: list[tuple[str, object]] = []
        self.claims: list[tuple[int, str]] = []
        self.watermarks: list[int] = []
        self.publishes: list[tuple[int, str]] = []
        self.manifest: list[tuple[int, str, str, str]] = []
        self.lease = FakeLeaseTable()

    def execute(self, query, params=None, **kwargs):
        self.statements.append(query)
        leased = self.lease.execute(query, params)
        if leased is not None:
            return leased
        if query.lstrip().upper().startswith("INSERT"):
            self.published.append((query, params))
            if "version_claims" in query:
                self.claims.extend((row[0], str(row[1])) for row in params)
            elif "snapshot_manifest" in query:
                if self.lease.fence_admits(query, params):
                    self.manifest.extend(
                        (
                            params["index_version"],
                            params["publish_id"],
                            store_id,
                            pack_id,
                        )
                        for store_id, pack_id in params["members"]
                    )
            elif "index_watermark" in query:
                version = params["index_version"]
                # The fence check runs UNCONDITIONALLY: short-circuited behind
                # the barrier, a statement missing the fence would slip by
                # whenever the barrier already refused it.
                fenced = self.lease.fence_admits(query, params)
                if fenced and version > max(self.watermarks, default=0):
                    self.watermarks.append(version)
                    self.publishes.append((version, params["publish_id"]))
            return []
        if "publish_id" in query and "index_watermark" in query:
            return [
                (publish_id,)
                for version, publish_id in self.publishes
                if version == params["version"]
            ]
        if "snapshot_manifest" in query and "SELECT count()" in query:
            wanted = set(params["members"])
            found = {
                (store_id, pack_id)
                for version, publish_id, store_id, pack_id in self.manifest
                if version == params["index_version"]
                and publish_id == params["publish_id"]
                and (store_id, pack_id) in wanted
            }
            return [(len(found),)]
        # The version allocator's three queries need real state to answer.
        if "version_claims" in query:
            if "max(version)" in query:
                return [(max((v for v, _ in self.claims), default=None),)]
            wanted = params["version"]
            return [(cid,) for v, cid in self.claims if v == wanted]
        if "index_watermark" in query:
            return [(max(self.watermarks, default=None),)]
        return []


def test_a_pack_is_never_both_skipped_on_replay_and_invisible_to_readers(
    tmp_path: Path,
):
    """The durability window between publishing and recording the replay guard.

    `committed_pack_ids` reads the inventory to skip replays; readers bound the
    snapshot by the manifest a publish writes. If the inventory landed first
    and the process died before the publish, the pack would be skipped forever
    *and* never visible -- silent, permanent data loss. Publishing first makes
    the same crash merely redundant work, which this proves by crashing there
    and then running the next pass.
    """
    store = FilesystemPackStore(tmp_path, store_id="local")
    ref = store.put(_sealed(_record(_metadata())), "packs/a.dmi-pack")

    backing = _Client()
    # index() inserts: version claim, descriptors, lease renewal, manifest,
    # another renewal, watermark, then the inventory. Fail the last one -- the
    # crash this ordering exists for. The lease claim below is insert 1.
    client = FaultyClickHouseClient(backing, insert=fail_on(8))
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())
    writer.acquire_publisher_lease("indexer-a")

    with pytest.raises(FaultInjected):
        CatalogIndexer(store, writer, clock_ns=lambda: 7).index([ref])

    written = [s for s in backing.statements if s.startswith("INSERT")]
    assert any("snapshot_manifest" in s for s in written), (
        "the pack was never made a member of a snapshot"
    )
    assert any("index_watermark" in s for s in written), (
        "the version was never published, so the descriptors are invisible"
    )
    assert not any("pack_inventory_raw" in s for s in written), (
        "the inventory landed despite the injected failure"
    )

    # The next pass sees no inventory row, so it re-indexes rather than
    # skipping a pack it never made visible. It is a different publisher, so it
    # waits out the crashed one's lease -- reached here by moving the fake
    # server clock rather than by sleeping.
    healthy = ClickHouseCatalogWriter(backing, ClickHouseCatalogConfig())
    backing.lease.now_ns += ClickHouseCatalogConfig().lease_ttl_ns + 1
    healthy.acquire_publisher_lease("indexer-b")
    result = CatalogIndexer(store, healthy, clock_ns=lambda: 8).index([ref])

    assert result.indexed_packs == 1, "the crashed pack was skipped on replay"
    assert result.skipped_packs == 0


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
    writer = ClickHouseCatalogWriter(_Client(), ClickHouseCatalogConfig())
    writer.acquire_publisher_lease("indexer-a")
    indexer = CatalogIndexer(inventory, writer,
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
    writer.acquire_publisher_lease("indexer-a")
    clock = iter([2_000, 1_000])  # second batch stamped *earlier*
    indexer = CatalogIndexer(store, writer, clock_ns=lambda: next(clock))

    first_result = indexer.index([first])
    second_result = indexer.index([second])

    # Both batches must still be indexed -- refusing would stop capture over a
    # clock correction -- but the second must not reuse or undercut a version a
    # reader may already have pinned.
    assert first_result.indexed_packs == 1 and second_result.indexed_packs == 1
    published = client.watermarks
    assert published == sorted(published), f"versions went backwards: {published}"
    assert len(set(published)) == len(published), "a version was reused"


# --- a pack's footer is not evidence of whose it is -------------------------


def _pack_at(tmp_path: Path, object_key: str, *, tenant_id: str):
    """Put a pack for ``tenant_id`` at ``object_key``, whoever that key belongs to."""
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed(_record(_metadata(tenant_id=tenant_id)))
    return store, store.put(sealed, object_key)


def test_a_pack_whose_footer_names_another_tenant_is_refused(tmp_path: Path):
    """The forgery the object store cannot prevent, refused where it is visible.

    Anyone able to PUT into the bucket can write a well-formed pack whose
    footer carries another tenant's `tenant_id` and `capture_id`. Integrity
    proves nothing about it -- the pack is perfectly well formed -- and until
    the descriptors are built, nothing has ever compared what the pack CLAIMS
    to be against where it was found. Indexed, it would be admitted under the
    victim's tenant, and since the reader resolves a capture with `argMax` over
    `(index_version, store_id, pack_id)` it can become the pack that capture
    resolves to at every fresh watermark.
    """
    store, ref = _pack_at(
        tmp_path,
        "v1/tenant=victim/date=2026-09-01/session=s/rank=0/"
        f"{PACK_ID}.dmi-pack",
        tenant_id="attacker",
    )

    with pytest.raises(PackIntegrityError, match="not the tenant that key"):
        PackIndex.from_store(store, ref).descriptors()


def test_a_pack_under_its_own_tenants_prefix_is_accepted(tmp_path: Path):
    """The other side: the check must not refuse the ordinary case."""
    store, ref = _pack_at(
        tmp_path,
        "v1/tenant=tenant-a/date=2026-09-01/session=s/rank=0/"
        f"{PACK_ID}.dmi-pack",
        tenant_id="tenant-a",
    )

    descriptors = PackIndex.from_store(store, ref).descriptors()

    assert [item.metadata.tenant_id for item in descriptors] == ["tenant-a"]


def test_a_pack_mixing_two_tenants_is_refused_on_the_one_that_does_not_match(
    tmp_path: Path,
):
    """A pack is checked per tenant present, not per pack.

    Nothing about a pack read back OUT of a bucket guarantees it holds one
    tenant -- that is the writer's convention, and a forger is under no
    obligation to follow it. So the comparison cannot be lifted to a single
    tenant read off the first record: a pack whose first record matches the
    key and whose second carries the victim's `tenant_id` would then be
    admitted whole, which is the forgery this refuses with extra steps.
    """
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed(
        _record(_metadata(tenant_id="tenant-a")),
        _record(_metadata(tenant_id="attacker", capture_id="capture-b")),
    )
    ref = store.put(
        sealed,
        "v1/tenant=tenant-a/date=2026-09-01/session=s/rank=0/"
        f"{PACK_ID}.dmi-pack",
    )

    with pytest.raises(PackIntegrityError, match="tenant 'attacker'"):
        PackIndex.from_store(store, ref).descriptors()


def test_a_tenant_too_long_for_a_key_segment_still_binds(tmp_path: Path):
    """`key_component` digests an identifier too long for a segment.

    That encoding is one-way, so a comparison that only knew how to unquote
    would silently stop checking exactly the tenants whose names are least
    guessable. The verifier re-encodes instead.
    """
    from dmi.storage.capture.pack import key_component

    tenant = "t" * 300
    encoded = key_component(tenant)
    assert encoded.startswith("sha256-")

    store, ref = _pack_at(
        tmp_path,
        f"v1/tenant={encoded}/date=2026-09-01/session=s/rank=0/{PACK_ID}.dmi-pack",
        tenant_id=tenant,
    )
    assert PackIndex.from_store(store, ref).descriptors()

    other, forged = _pack_at(
        tmp_path / "other",
        f"v1/tenant={encoded}/date=2026-09-01/session=s/rank=0/{PACK_ID}.dmi-pack",
        tenant_id="somebody-else",
    )
    with pytest.raises(PackIntegrityError, match="not the tenant that key"):
        PackIndex.from_store(other, forged).descriptors()


def test_a_key_that_names_no_tenant_is_left_alone(tmp_path: Path):
    """A filesystem store, a fixture and a hand-placed object are all
    legitimately laid out some other way, and a check that guessed at those
    would refuse every one of them. What is enforced is that a key which DOES
    name a tenant names the pack's own."""
    store, ref = _pack_at(tmp_path, "packs/a.dmi-pack", tenant_id="tenant-a")

    assert PackIndex.from_store(store, ref).descriptors()
