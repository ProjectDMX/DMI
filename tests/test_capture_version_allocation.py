"""The catalog's DB-owned version allocator and its publisher lease.

Catalog versions used to be derived from the indexer's wall clock, so two live
indexers with skewed clocks could publish a version below a watermark a reader
had already pinned. Versions are now handed out by the catalog itself through
an append-only, sole-claimant claims table: monotonic and unique, with no
clock anywhere in the ordering. These tests drive that protocol over a shared
in-memory "server" with real state, so two writers contend for real.

The publisher lease is claimed by the same protocol over the same kind of
append-only table, and the tests for it live here for that reason. What they
CANNOT reach is the window the lease fence exists for: two publishers in one
interpreter are serialised by the interpreter, so their statements never
overlap on a server. That is driven against a real ClickHouse in
tests/test_clickhouse_snapshot_live.py.
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
    PackRef,
    PackWriter,
    PublisherLeaseError,
    PublisherLeaseHeldError,
    SnapshotPublishRaceError,
)
from tests._catalog_fakes import FakeLeaseTable


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
        self.publishes: list[tuple[int, str]] = []
        self.manifest: list[tuple[int, str, str, str]] = []
        self.inserts: list[str] = []
        self.on_owner_read = None
        self.lease = FakeLeaseTable()

    def execute(self, query, params=None, **kwargs):
        leased = self.lease.execute(query, params)
        if leased is not None:
            return leased
        if query.lstrip().upper().startswith("INSERT"):
            self.inserts.append(query)
            if "version_claims" in query:
                self.claims.extend((int(row[0]), str(row[1])) for row in params)
            elif "snapshot_manifest" in query:
                # Fenced too, so a fenced-out publisher records nothing here.
                if self.lease.fence_admits(query, params):
                    self.manifest.extend(
                        (int(params["index_version"]), str(params["publish_id"]),
                         str(store_id), str(pack_id))
                        for store_id, pack_id in params["members"]
                    )
            elif "index_watermark" in query:
                # The conditional publish, enforced: a version lands only when
                # it is strictly above the published head AND this publisher
                # still holds the lease. The fence check runs UNCONDITIONALLY
                # -- short-circuited behind the barrier, a statement missing
                # the fence would slip by whenever the barrier refused it.
                version = int(params["index_version"])
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


def _publisher(server, holder="indexer", **config) -> ClickHouseCatalogWriter:
    """A writer holding the publisher lease, which every publish requires.

    Only one of these can exist over a server at a time -- that is the whole
    invariant. A second publisher takes over only once the first lease has
    expired, which these tests reach by moving the fake server clock rather
    than by sleeping.
    """
    writer = _writer(server, **config)
    writer.acquire_publisher_lease(holder)
    return writer


def _expire_the_lease(server) -> None:
    """Move the server clock past the held lease, as a stalled indexer would."""
    server.lease.now_ns += ClickHouseCatalogConfig().lease_ttl_ns + 1


# --- the publisher lease ------------------------------------------------------
#
# The protocol, over a shared in-memory server. What is NOT here is the thing
# the fence exists for: two publishers in one interpreter are serialised by the
# interpreter, so their statements never overlap on a server and no fake state
# reproduces that window. tests/test_clickhouse_snapshot_live.py drives it
# against a real ClickHouse.


def test_only_one_publisher_can_hold_the_lease():
    server = _CatalogServer()
    first = _publisher(server, "indexer-a")

    with pytest.raises(PublisherLeaseHeldError) as raised:
        _publisher(server, "indexer-b")

    message = str(raised.value)
    assert "'indexer-a'" in message and first.publisher_lease.lease_id in message
    assert "wait for it to expire or stop it" in message


def test_publishing_without_a_lease_is_refused_before_anything_is_written():
    server = _CatalogServer()
    writer = _writer(server)

    with pytest.raises(PublisherLeaseError, match="no publisher lease is held"):
        writer.publish_snapshot(
            index_version=1, refs=(), published_at_ns=1, indexed_rows=0,
            indexed_packs=0,
        )

    assert server.inserts == [], "a publisher without a lease wrote something"
    assert server.watermarks == []


@pytest.mark.parametrize("holder", ["", 7, "x" * 257])
def test_a_lease_holder_must_be_a_short_non_empty_name(holder):
    """The name goes in the error another publisher reads, so it is bounded."""
    with pytest.raises(ValueError, match="holder"):
        _writer(_CatalogServer()).acquire_publisher_lease(holder)


def test_an_expired_lease_is_taken_over_and_the_old_holder_never_reaches_a_write():
    """Expiry is what makes a crashed indexer recoverable rather than terminal.

    The old holder is not told; it finds out at its next publish. The stop
    happens at the RENEWAL, client-side: `publish_snapshot` renews first, the
    renewal reads a head that is a live lease belonging to somebody else, and
    it refuses before any statement is issued. This test therefore says nothing
    about the fence -- it passed unchanged with `_lease_fence()` replaced by
    `1 = 1`, and its docstring used to claim the fence was what made "writes
    nothing" literal. The fence is exercised by
    `test_a_takeover_after_the_renewal_is_reported_as_a_lost_lease` below, and
    for real by tests/test_clickhouse_snapshot_live.py.
    """
    server = _CatalogServer()
    stalled = _publisher(server, "stalled")
    version = stalled.allocate_version()

    _expire_the_lease(server)
    successor = _publisher(server, "successor")
    assert successor.publisher_lease.term > stalled.publisher_lease.term
    issued = len(server.inserts)

    with pytest.raises(PublisherLeaseHeldError, match="'successor'"):
        stalled.publish_snapshot(
            index_version=version, refs=(), published_at_ns=1, indexed_rows=0,
            indexed_packs=0,
        )

    # The mechanism, pinned: not one statement was issued, so nothing was
    # written for the fence to have to reject.
    assert server.inserts[issued:] == [], (
        "the refusal happened after a statement was issued, not before"
    )
    assert server.watermarks == []
    assert server.manifest == []
    assert stalled.publisher_lease is None, (
        "a publisher that lost its lease must not go on believing it holds one"
    )


def test_a_takeover_after_the_renewal_is_reported_as_a_lost_lease():
    """The failure the fence catches, and why it is not a lost version race.

    Renewing before every fenced statement leaves one window: between renewal
    and the server executing the write. A takeover there passes the client-side
    check and fails the server-side fence, and the two failures need different
    recoveries -- a lost VERSION is repaired by allocating a higher one, which
    the indexer does automatically, while a lost LEASE would fail the same
    fence at every version. So it is deliberately not a
    ``SnapshotPublishRaceError``.
    """
    server = _CatalogServer()
    holder = _publisher(server, "indexer-a")
    version = holder.allocate_version()

    def take_over(_lease_id) -> None:
        server.lease.on_fence = None
        _expire_the_lease(server)
        _publisher(server, "successor")

    server.lease.on_fence = take_over
    with pytest.raises(PublisherLeaseError) as raised:
        holder.publish_snapshot(
            index_version=version, refs=(), published_at_ns=1, indexed_rows=0,
            indexed_packs=0,
        )

    assert "fenced out and made no snapshot visible" in str(raised.value)
    assert "'successor'" in str(raised.value)
    assert not isinstance(raised.value, SnapshotPublishRaceError)
    assert server.watermarks == []
    assert holder.publisher_lease is None


def test_a_publish_requires_a_full_timeout_of_lease_remaining():
    server = _CatalogServer()
    timeout = 1_000_000_000
    writer = _publisher(
        server,
        lease_ttl_ns=2_000_000_000,
        publish_timeout_ns=timeout,
    )

    def age_lease(_lease_id) -> None:
        server.lease.on_fence = None
        expires_at_ns = max(row[4] for row in server.lease.head())
        server.lease.now_ns = expires_at_ns - timeout

    server.lease.on_fence = age_lease

    with pytest.raises(SnapshotPublishRaceError):
        writer.publish_snapshot(
            index_version=1,
            refs=(),
            published_at_ns=1,
            indexed_rows=0,
            indexed_packs=0,
        )

    assert server.watermarks == []


def test_a_publish_renews_between_fenced_statements(tmp_path: Path):
    _, refs = _refs(tmp_path, 1)
    server = _CatalogServer()
    timeout = 1_000_000_000
    writer = _publisher(
        server,
        lease_ttl_ns=2_000_000_000,
        publish_timeout_ns=timeout,
    )
    execute = server.execute

    def advance_after_manifest(query, params=None, **kwargs):
        result = execute(query, params, **kwargs)
        if query.lstrip().startswith("INSERT") and "snapshot_manifest" in query:
            server.lease.now_ns += timeout
        return result

    server.execute = advance_after_manifest

    writer.publish_snapshot(
        index_version=1,
        refs=refs,
        published_at_ns=1,
        indexed_rows=1,
        indexed_packs=1,
    )

    assert server.watermarks == [1]


def test_a_contested_lease_term_is_abandoned_by_everyone_who_sees_it():
    """The sole-claimant rule, applied to the lease.

    A competing claim lands between the claimant's INSERT and its read-back.
    Resolving that tie in either direction would let both sides keep the term
    when each sees only its own insert first, so the only safe answer is for
    both to walk away and claim above it.
    """
    server = _CatalogServer()
    writer = _writer(server)
    contested: list[int] = []

    def contest(term: int) -> None:
        contested.append(term)
        server.lease.rows.append(
            (term, _LARGE_CLAIM, "competitor", 0, server.lease.now_ns + 10**12)
        )
        server.lease.on_claim_read = None  # contest the first attempt only

    server.lease.on_claim_read = contest
    lease = writer.acquire_publisher_lease("indexer-a")

    assert contested == [1], "the first term should have been contested"
    assert lease.term > contested[0], "a contested term must never be held"


def test_a_lease_claim_gives_up_after_exhausting_its_attempts():
    """Bounded, and loud: an unbounded retry would spin against a live rival."""
    server = _CatalogServer()
    writer = _writer(server, lease_attempts=3)
    server.lease.on_claim_read = lambda term: server.lease.rows.append(
        (term, _LARGE_CLAIM, "competitor", 0, 0)
    )

    with pytest.raises(PublisherLeaseError, match="after 3 attempts"):
        writer.acquire_publisher_lease("indexer-a")

    assert writer.publisher_lease is None
    assert len(server.lease.rows) == 6, "three claims, each contested once"


def test_a_holder_re_acquiring_its_own_live_lease_refreshes_it():
    """Acquiring twice is the documented path, not a contest.

    Both `capture-storage-design.md` and `renew_publisher_lease`'s own refusal
    tell an operator to acquire before publishing, so anything that restarts
    above this object calls acquire on a writer that already holds one.

    Minting a fresh `lease_id` there made the writer a stranger to its own row:
    the claim was refused as held by ITSELF, the local lease was dropped on the
    way out, and `release_publisher_lease()` then had nothing to release -- so
    the orphaned row stood for a whole `lease_ttl_ns` with no API left to clear
    it and every retry hit the same row.
    """
    server = _CatalogServer()
    writer = _publisher(server, "indexer-a")
    before = writer.publisher_lease

    after = writer.acquire_publisher_lease("indexer-a")

    assert after.lease_id == before.lease_id, "a refresh keeps the fencing token"
    assert after.term > before.term
    assert writer.publisher_lease == after
    # And the writer can still give it back, which is what the orphan denied.
    writer.release_publisher_lease()
    assert writer.publisher_lease is None
    assert _publisher(server, "indexer-b").publisher_lease is not None


def test_a_release_by_a_writer_that_no_longer_holds_the_lease_revokes_nothing():
    """A stale writer's orderly shutdown must not evict the current holder.

    The tombstone lands at `head.term + 1`, so it becomes the head whatever was
    there before. Written blind, a writer whose lease lapsed long ago takes the
    catalog away from whoever holds it now simply by shutting down cleanly: the
    successor is not told, its next publish is fenced out, and any third
    publisher can take the lease off it at once.
    """
    server = _CatalogServer()
    stale = _publisher(server, "stale")

    _expire_the_lease(server)
    healthy = _publisher(server, "healthy")
    rows_before = len(server.lease.rows)

    stale.release_publisher_lease()

    assert stale.publisher_lease is None, "the stale writer still gives up its own"
    assert len(server.lease.rows) == rows_before, "a non-holder wrote a tombstone"
    # The healthy holder is untouched: its lease still fences, and a third
    # publisher is still refused.
    assert server.lease.fence_passes(healthy.publisher_lease.lease_id)
    with pytest.raises(PublisherLeaseHeldError, match="'healthy'"):
        _publisher(server, "third")


def test_a_contested_lease_claim_retries_at_a_randomized_term(monkeypatch):
    """The retry must not send every contender back to the same next term.

    `head.term + 1` is what every contender computes, so a term abandoned by
    all of them is followed by a term all of them collide on again. Measured
    against a live 25.12: six publishers taking a cold lease 25 times each hard
    -failed 45% of the time with "every term was contested" -- a liveness
    failure invented entirely by the retry. `allocate_version` already adds
    `secrets.randbelow(8 * attempt + 1)` for exactly this reason; the lease
    claim now mirrors it, window for window.
    """
    import dmi.storage.capture.clickhouse_catalog as module

    draws: list[int] = []

    def top_of_the_window(bound: int) -> int:
        draws.append(bound)
        return bound - 1

    monkeypatch.setattr(module.secrets, "randbelow", top_of_the_window)

    server = _CatalogServer()
    writer = _writer(server, lease_attempts=3)
    server.lease.on_claim_read = lambda term: server.lease.rows.append(
        (term, _LARGE_CLAIM, "competitor", 0, 0)
    )

    with pytest.raises(PublisherLeaseError, match="after 3 attempts"):
        writer.acquire_publisher_lease("indexer-a")

    # The first attempt takes head + 1 unskipped; every later one draws from
    # the same widening window the allocator uses.
    assert draws == [8 * 1 + 1, 8 * 2 + 1]
    claimed = sorted({term for term, _, holder, _, _ in server.lease.rows
                      if holder == "indexer-a"})
    assert claimed == [1, 10, 27], claimed


def test_a_released_lease_is_available_at_once():
    """An orderly handover must not cost the next publisher a whole TTL."""
    server = _CatalogServer()
    first = _publisher(server, "indexer-a")

    first.release_publisher_lease()

    assert first.publisher_lease is None
    second = _publisher(server, "indexer-b")
    assert second.publisher_lease.term == 3, "the tombstone is term 2, the retake 3"
    # Idempotent: a second release with nothing held writes nothing more.
    rows = len(server.lease.rows)
    first.release_publisher_lease()
    assert len(server.lease.rows) == rows


def test_releasing_without_a_lease_writes_nothing():
    server = _CatalogServer()
    writer = _writer(server)

    writer.release_publisher_lease()

    assert server.lease.rows == []


def test_a_renewal_keeps_the_fencing_identity_and_extends_the_term():
    """Renewal is a fresh term under the same identity, not a new lease.

    The fence names the lease, so a renewal has to keep it; the term has to
    move, because that is what a takeover would have to beat; and the expiry
    has to move, because the whole point of renewing before each fenced write is
    that the fence runs with essentially a full TTL left.

    The clock is advanced deliberately. Asserted against a fake clock that
    never moves, `expires_at_ns >= expires_at_ns` is `x >= x` and the half of
    this test its name promises was not tested at all.
    """
    server = _CatalogServer()
    writer = _publisher(server, "indexer-a")
    before = writer.publisher_lease
    server.lease.now_ns += 1_000_000_000  # a second of the SERVER's clock

    after = writer.renew_publisher_lease()

    assert after.lease_id == before.lease_id
    assert after.term > before.term
    assert after.expires_at_ns > before.expires_at_ns
    assert after.expires_at_ns - after.acquired_at_ns == (
        ClickHouseCatalogConfig().lease_ttl_ns
    ), "a renewal has to buy a whole TTL, or the safety margin is not what it says"


@pytest.mark.parametrize(
    "config, message",
    [
        ({"lease_ttl_ns": 0}, "lease_ttl_ns"),
        ({"publish_timeout_ns": -1}, "publish_timeout_ns"),
        ({"lease_attempts": 1.5}, "lease_attempts"),
        ({"publish_timeout_ns": 30_000_000_000}, "must be below lease_ttl_ns"),
    ],
)
def test_the_lease_settings_are_validated(config, message):
    with pytest.raises(ValueError, match=message):
        ClickHouseCatalogConfig(**config)


def test_the_fake_matches_the_fence_the_writer_actually_emits():
    """The fakes are only worth anything while they enforce the REAL predicate.

    `FakeLeaseTable.fence_admits` matches the fence in the statement rather
    than trusting the `lease_id` parameter, which `publish_snapshot` passes
    unconditionally. That only works while the pattern and `_lease_fence()`
    agree, so the agreement is asserted here: a rewrite of the fence fails as
    one clear test instead of turning four suites vacuous.
    """
    from tests._catalog_fakes import _FENCE, MissingLeaseFence

    writer = _writer(_CatalogServer())
    assert _FENCE.fullmatch(writer._lease_fence()) is not None, (
        "tests/_catalog_fakes._FENCE no longer matches "
        "ClickHouseCatalogWriter._lease_fence(); update it deliberately"
    )
    # The fenced release is pinned the same way, for the same reason: the fake
    # models the head-resolve-and-write as one atomic statement, which is only
    # honest while the writer actually emits that statement.
    from tests._catalog_fakes import _RELEASE

    assert _RELEASE.fullmatch(writer._release_statement()) is not None, (
        "tests/_catalog_fakes._RELEASE no longer matches "
        "ClickHouseCatalogWriter._release_statement(); update it deliberately"
    )
    # And a statement without it is refused rather than quietly admitted.
    with pytest.raises(MissingLeaseFence, match="must be fenced"):
        FakeLeaseTable().fence_admits(
            "INSERT INTO `default`.`dmi_index_watermark` SELECT 1 FROM "
            "system.one WHERE 1 = 1",
            {"lease_id": "irrelevant"},
        )
    # So is a fence whose TEXT is intact but which no longer gates the write:
    # a weakening AROUND the fence must fail the same way as its absence.
    with pytest.raises(MissingLeaseFence, match="top-level AND conjunct"):
        FakeLeaseTable().fence_admits(
            "INSERT INTO `default`.`dmi_index_watermark` SELECT 1 FROM "
            "system.one WHERE " + writer._lease_fence() + " OR 1 = 1",
            {"lease_id": "irrelevant"},
        )


def test_the_indexer_does_not_absorb_a_lost_lease_as_a_lost_version(tmp_path: Path):
    """The two failures need different recoveries, so only one is retried.

    A lost VERSION is repaired by allocating a higher one, which `_publish`
    does. A lost LEASE is not: every retry would fail the same fence at every
    version, so absorbing it would spin `max_publish_attempts` times and then
    raise `SnapshotPublishExhaustedError("could not publish ... after N
    attempts")` -- burying the one message that names the holder and says to
    acquire again.
    """
    store, refs = _refs(tmp_path, 1)
    server = _CatalogServer()
    writer = _publisher(server, "stalled")
    indexer = CatalogIndexer(store, writer, clock_ns=lambda: 7)

    # A legitimate takeover lands between the renewal and the write, which is
    # the window the fence exists for.
    def take_over(_lease_id) -> None:
        server.lease.on_fence = None
        _expire_the_lease(server)
        _publisher(server, "successor")

    server.lease.on_fence = take_over

    with pytest.raises(PublisherLeaseError) as raised:
        indexer.index(refs)

    assert not isinstance(raised.value, SnapshotPublishRaceError)
    assert "'successor'" in str(raised.value)
    assert "Acquire a lease again" in str(raised.value)
    # It gave up at the first attempt rather than burning the retry budget.
    assert server.watermarks == []
    assert sum("version_claims" in q for q in server.inserts) == 1


def test_a_barrier_refusal_with_an_expired_own_lease_is_still_a_race():
    """An expiry between the fenced statement and the read-back is no takeover.

    The refusal classifier used to demand that the head be this writer's row
    AND still live at CHECK time, so an ordinary lost version race whose lease
    merely expired during the read-back window -- a stall, a short TTL -- was
    reported as a lost lease, with a message blaming this writer's own holder
    for fencing it out. That bypasses the indexer's retry, which renews the
    lease before every attempt and would have recovered. As long as the head
    is still this writer's row, nobody took the lease, and the race retry is
    the recovery.
    """
    server = _CatalogServer()
    writer = _publisher(server, "indexer-a")
    server.watermarks.append(10)  # somebody already published above this batch

    real_execute = server.execute

    def stalling_execute(query, params=None, **kwargs):
        result = real_execute(query, params, **kwargs)
        if query.lstrip().startswith("INSERT") and "index_watermark" in query:
            # The stall: the lease runs out AFTER the fenced statement was
            # evaluated and BEFORE the ownership read-back runs.
            _expire_the_lease(server)
        return result

    server.execute = stalling_execute

    with pytest.raises(SnapshotPublishRaceError):
        writer.publish_snapshot(
            index_version=5, refs=(), published_at_ns=7,
            indexed_rows=0, indexed_packs=0,
        )


# --- the allocator over a shared server ---------------------------------------


def test_two_writers_allocate_distinct_strictly_increasing_versions():
    server = _CatalogServer()
    first, second = _writer(server), _writer(server)

    versions = [writer.allocate_version() for writer in (first, second) * 3]

    assert versions == sorted(versions), f"versions went backwards: {versions}"
    assert len(set(versions)) == len(versions), "a version was reused"


def test_allocations_interleaved_with_publications_stay_monotonic():
    server = _CatalogServer()
    writer = _publisher(server)

    first = writer.allocate_version()
    writer.publish_snapshot(
        index_version=first, refs=(), published_at_ns=1, indexed_rows=0,
        indexed_packs=0,
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
    first = _publisher(server)
    version = first.allocate_version()
    first.publish_snapshot(
        index_version=version, refs=(), published_at_ns=1, indexed_rows=0,
        indexed_packs=0,
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


def test_a_losing_publish_leaves_its_membership_rows_inert():
    """Why a loser can leave its rows behind instead of cleaning up.

    The version barrier still has work to do under the lease, because a lease
    hands over. Indexer A allocates and then stalls long enough to lose its
    lease. B takes over, allocates higher and publishes. A comes back, takes
    the lease again -- B's has expired by now -- and publishes the version it
    allocated before any of this, which is below the published head.

    A holds the lease, so the fence lets its manifest rows through; the version
    barrier is what stops it. Those rows are durable and its version never
    reaches the watermark table, and the reader's membership clause requires
    BOTH, so they can never enter a snapshot. Cleaning them up is therefore
    unnecessary for correctness (though they do accumulate; see
    docs/catalog-descriptor-key.md on the GC obligation).
    """
    server = _CatalogServer()
    slow = _publisher(server, "slow")
    ref = PackRef(
        pack_id=str(UUID(int=7)), store_id="local",
        object_key="packs/a.dmi-pack", object_bytes=1024,
        checksum="0" * 64, record_count=1,
    )

    version_a = slow.allocate_version()
    _expire_the_lease(server)
    fast = _publisher(server, "fast")
    version_b = fast.allocate_version()
    assert version_b > version_a

    fast.publish_snapshot(
        index_version=version_b, refs=(), published_at_ns=1,
        indexed_rows=0, indexed_packs=0,
    )
    _expire_the_lease(server)
    slow.acquire_publisher_lease("slow")
    with pytest.raises(SnapshotPublishRaceError):
        slow.publish_snapshot(
            index_version=version_a, refs=[ref], published_at_ns=2,
            indexed_rows=1, indexed_packs=1,
        )

    assert any(row[0] == version_a for row in server.manifest), (
        "precondition: the loser really did write its membership rows"
    )
    assert version_a not in server.watermarks, (
        "the loser wrote a watermark row, which would admit its rows into a "
        "snapshot pinned at the winner's version"
    )
    assert server.watermarks == [version_b]


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
    """Successive indexers with skewed clocks, then the skew reversed.

    Publications must be strictly increasing regardless of what any wall clock
    says: the version is allocator-owned and the clock stamps published_at_ns
    only. Each indexer takes the lease over from the last, which is what one
    replacing another looks like.
    """
    server = _CatalogServer()
    store, refs = _refs(tmp_path, 4)

    for clock_value, ref in ((1_000, refs[0]), (100, refs[1]),
                             (100, refs[2]), (1_000, refs[3])):
        _expire_the_lease(server)
        indexer = CatalogIndexer(
            store, _publisher(server), clock_ns=lambda v=clock_value: v
        )
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

    def publish_snapshot(
        self, *, index_version, refs, published_at_ns, indexed_rows, indexed_packs
    ):
        pass

    def last_published_version(self):
        return self._last_published

    def allocate_version(self):
        return self._allocate()


class _Store:
    store_id = "local"


@pytest.mark.parametrize("value", [-1, "7", True, 4.0])
def test_the_indexer_rejects_a_non_integer_or_negative_allocation(
    tmp_path: Path, value
):
    # A real pack, because a batch with nothing to index no longer allocates
    # (or publishes) at all -- the guard runs on the way to real writes.
    store, refs = _refs(tmp_path, 1)
    indexer = CatalogIndexer(store, _HostileWriter(lambda: value))

    with pytest.raises(ValueError, match="allocate_version"):
        indexer.index(refs)


@pytest.mark.parametrize("allocated", [5, 4])
def test_the_indexer_refuses_a_non_monotonic_allocation(
    tmp_path: Path, allocated: int
):
    # The durable watermark says 5 has been published; an allocator handing
    # out 5 (or lower) again would let this batch land inside pinned snapshots.
    store, refs = _refs(tmp_path, 1)
    indexer = CatalogIndexer(
        store, _HostileWriter(lambda: allocated, last_published=5)
    )

    with pytest.raises(RuntimeError, match="non-monotonic"):
        indexer.index(refs)
