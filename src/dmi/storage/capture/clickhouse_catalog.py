from __future__ import annotations

import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from time import time_ns
from typing import Protocol
from uuid import uuid4

from .catalog import (
    CatalogVersionAllocationError,
    PackIdentity,
    SnapshotPublishConflictError,
    SnapshotPublishRaceError,
)
from .clickhouse_lease import (
    DECIDING_READ,
    ClickHouseLeaseCoordinator,
    LeaseHead,
    PublisherLease,
)
from .clickhouse_schema import (
    CAPTURE_COLUMNS,
    CAPTURE_TABLE_ORDER,
    FACET_COLUMNS,
    PACK_COLUMNS,
    SCHEMA_VERSION,
    ClickHouseCatalogSchema,
)
from .clickhouse_sql import (
    MAX_INLINE_PARAMETER_BYTES,
    inline_chunks,
    inline_tuple_bytes,
    membership_predicate,
)
from .model import CaptureDescriptor, PackRef

_CAPTURE_COLUMNS = CAPTURE_COLUMNS
_CAPTURE_TABLE_ORDER = CAPTURE_TABLE_ORDER
_DECIDING_READ = DECIDING_READ
_FACET_COLUMNS = FACET_COLUMNS
_MAX_INLINE_PARAMETER_BYTES = MAX_INLINE_PARAMETER_BYTES
_PACK_COLUMNS = PACK_COLUMNS
_SCHEMA_VERSION = SCHEMA_VERSION
_LeaseHead = LeaseHead
_inline_chunks_by_size = inline_chunks
_inline_tuple_bytes = inline_tuple_bytes
_membership_predicate = membership_predicate

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ClickHouseClient(Protocol):
    def execute(self, query: str, params=None, **kwargs): ...


def _identifier(value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid ClickHouse identifier: {value!r}")
    return value


def _quoted(value: str) -> str:
    return f"`{value}`"


@dataclass(frozen=True, slots=True)
class ClickHouseCatalogConfig:
    database: str = "default"
    table_prefix: str = "dmi"
    query_pack_limit: int = 10_000
    allocation_attempts: int = 16
    # How long a publisher lease stays live, and how long its publish statement
    # may run. The gap between them is the whole safety margin: a takeover
    # cannot happen until ``lease_ttl_ns`` after the holder last renewed, and
    # the holder's publish statement is capped ``publish_timeout_ns`` after it
    # started, so the two cannot overlap without the server overrunning its own
    # execution-time check. See publish_snapshot.
    lease_ttl_ns: int = 30_000_000_000
    publish_timeout_ns: int = 5_000_000_000
    # Compatibility-only: contested terms now wait for expiry instead of retrying.
    lease_attempts: int = 8
    # The WRITE half of the consistency story, and off by default because it is
    # a deployment decision rather than one this module can make.
    #
    # ``select_sequential_consistency`` is necessary but NOT sufficient on a
    # ReplicatedMergeTree: ClickHouse defines it against inserts made with
    # ``insert_quorum``, and it does not work with ``insert_quorum_parallel``
    # (on by default). Without this, two claimants on two replicas can still
    # each read themselves alone -- the sole-claimant protocols are sound on a
    # single node and on a quorum-writing cluster, and not in between.
    #
    # Set to the quorum size (2 or more) on a replicated deployment. Every
    # write the protocols DECIDE on -- version claims, lease claims, membership
    # and the watermark -- then waits for that many replicas, and the reads
    # above are answered against something durable. It is deliberately not
    # applied to the descriptor and inventory bulk writes: those decide
    # nothing, and paying quorum latency per batch is the cost that makes
    # operators turn the whole thing off.
    insert_quorum: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.database)
        _identifier(self.table_prefix)
        for name in (
            "query_pack_limit",
            "allocation_attempts",
            "lease_ttl_ns",
            "publish_timeout_ns",
            "lease_attempts",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.publish_timeout_ns >= self.lease_ttl_ns:
            raise ValueError(
                "publish_timeout_ns must be below lease_ttl_ns: the margin "
                "between them is what keeps a publish statement from still "
                "running when its lease becomes takeable"
            )
        if self.insert_quorum is not None and (
            type(self.insert_quorum) is not int or self.insert_quorum < 2
        ):
            raise ValueError(
                "insert_quorum must be None or an integer of at least 2: a "
                "quorum of one is what the default already gives, and setting "
                "it to 1 reads as protection that is not there"
            )
        if self.publish_timeout_ns % 1_000_000_000:
            # max_execution_time is sent as WHOLE seconds. A fractional value
            # is deployment roulette: modern servers parse the string, older
            # typed-settings serialization coerces through int() -- where 0.5
            # becomes 0, which ClickHouse reads as NO limit and the fence-vs-
            # TTL margin this config promises is silently gone.
            raise ValueError(
                "publish_timeout_ns must be a whole number of seconds: it is "
                "sent to the server as max_execution_time in seconds, and a "
                "fraction either fails to parse or truncates -- 0.5s can reach "
                "an older server as 0, which disables the cap entirely"
            )



def _inline_chunks(items: list[PackIdentity]):
    return _inline_chunks_by_size(items, item_bytes=_inline_tuple_bytes)


class ClickHouseCatalogWriter:
    def __init__(
        self, client: ClickHouseClient, config: ClickHouseCatalogConfig | None = None
    ) -> None:
        self._client = client
        self._config = config or ClickHouseCatalogConfig()
        prefix = self._config.table_prefix
        self._capture_raw = f"{prefix}_capture_raw"
        self._pack_raw = f"{prefix}_pack_inventory_raw"
        self._pack_view = f"{prefix}_pack_inventory"
        self._watermark = f"{prefix}_index_watermark"
        self._manifest = f"{prefix}_snapshot_manifest"
        self._version_claims = f"{prefix}_capture_version_claims"
        self._schema = ClickHouseCatalogSchema(
            client, self._config.database, self._config.table_prefix
        )
        self._objects = self._schema.objects
        self._legacy_objects = self._schema.legacy_objects
        self._leases = ClickHouseLeaseCoordinator(client, self._config)

    def ensure_schema(self) -> None:
        self._schema.ensure()

    def _rebuild_instruction(self) -> str:
        return self._schema.rebuild_instruction()

    def drop_schema(self) -> None:
        self._schema.drop()

    def committed_pack_ids(
        self, identities: Sequence[PackIdentity]
    ) -> set[PackIdentity]:
        if not identities:
            return set()
        if len(identities) > self._config.query_pack_limit:
            raise ValueError("pack identity query exceeds query_pack_limit")
        table = self._qualified(self._pack_view)
        committed: set[PackIdentity] = set()
        # Chunked: the identities land in the statement TEXT, and an unchunked
        # query_pack_limit's worth of tuples can breach max_query_size.
        for chunk in _inline_chunks(list(identities)):
            rows = self._client.execute(
                f"SELECT store_id, toString(pack_id) FROM {table} "
                "WHERE (store_id, pack_id) IN %(identities)s",
                {"identities": chunk},
                # A deciding read: this answer decides which packs the indexer
                # skips as already committed. Answered from a replica that has
                # not caught up, it under-reports and the pass re-indexes packs
                # that are already published -- redundant work, and a second
                # publish of a batch that needed none.
                settings=_DECIDING_READ,
            )
            committed.update(
                (self._text(row[0]), self._text(row[1])) for row in rows
            )
        return committed

    def write_descriptors(
        self, descriptors: Sequence[CaptureDescriptor], *, index_version: int
    ) -> None:
        if not descriptors:
            return
        self._validate_version(index_version)
        rows = [self._descriptor_row(item, index_version) for item in descriptors]
        self._client.execute(
            f"INSERT INTO {self._qualified(self._capture_raw)} "
            f"({', '.join(_CAPTURE_COLUMNS)}) VALUES",
            rows,
        )

    def publish_snapshot(
        self,
        *,
        index_version: int,
        refs: Sequence[PackRef],
        published_at_ns: int,
        indexed_rows: int,
        indexed_packs: int,
    ) -> None:
        """Make a version readable, after everything it covers is durable.

        Membership rows first, then the watermark row that admits them. A
        reader's membership clause requires both, so the order is what makes a
        half-finished publish invisible rather than partially visible.

        Both rows carry one ``publish_id``, minted here and used again to check
        the result, so that what this call verifies is "version V is MINE" and
        not merely "some row for V exists".

        **Both statements are fenced on the publisher lease, inside the
        server-side statement that does the writing.** A publisher whose lease
        has been taken over makes NO SNAPSHOT VISIBLE -- it does not make one
        visible and then discover it lost. That is the difference between this
        and every post-write check the design has rejected: a check that runs
        after the write cannot withdraw what the write already made durable,
        and no ``<= W`` predicate over an append-only table can be repaired
        that way.

        The guarantee is per STATEMENT rather than per publish, and the
        difference is worth stating because the documents used to claim the
        stronger one. A takeover before the manifest INSERT leaves nothing
        behind; a takeover in the GAP between the two statements -- a full
        client round trip, which ``max_execution_time`` does not bound because
        it caps each statement rather than the pair -- leaves the manifest rows
        of the first while the second is refused. Those rows are inert:
        membership pairs a manifest row with the watermark row of the SAME
        publish, and that row will never exist, so no snapshot admits them and
        no reader sees them. What they are not is absent. See
        docs/catalog-descriptor-key.md, "What this does not close".

        The lease is renewed before every fenced statement. Each fence requires
        at least ``publish_timeout_ns`` of lease life remaining, and the
        statement carries that same value as its ``max_execution_time``. Each
        manifest chunk is read back before renewal, so a zero-row conditional
        INSERT cannot be followed by a visible watermark. The cap is per
        statement, so it does not bound the gap between two statements. See
        docs/catalog-descriptor-key.md for what that does and does not close.

        A statement that TRIPS the cap raises the driver's own exception
        (``timeout_overflow_mode='throw'``, Code 159), not a
        ``CaptureStorageError`` -- the same species as the Code 125 path the
        design doc records -- and it raises BEFORE the ownership read-back, so
        whether the row landed is not known here. That is safe in the only
        direction that matters: the indexer aborts without ``commit_packs``,
        so if the row did land the pack is visible-but-not-yet-skippable and
        the next pass costs redundant work, never loss. Callers routing on
        this module's taxonomy should treat a driver exception from a publish
        as "outcome unknown: re-acquire and re-index".
        """
        self._validate_version(index_version)
        self._validate_version(published_at_ns)
        # Interpolated into the statement TEXT like the version is, so they get
        # the same guard. The driver renders an unknown type through str()
        # WITHOUT escaping, so "these are always len(...) at the only call
        # site" is a property of today's caller rather than of this method.
        self._validate_count(indexed_rows, "indexed_rows")
        self._validate_count(indexed_packs, "indexed_packs")
        lease = self.renew_publisher_lease()
        watermark = self._qualified(self._watermark)
        # One identity per ATTEMPT, not per allocated version: what has to be
        # verified is that this particular statement's row is the one standing
        # at V, and a second attempt at one version is a different write. Reusing
        # the allocator's claim_id would make a duplicate publish at V read back
        # as its own, which is the failure this identity exists to catch.
        publish_id = str(uuid4())
        settings = {
            **_DECIDING_READ,
            **self._publish_timeout(),
            **self._quorum_write(),
        }
        if refs:
            # Fenced too, so that a publisher fenced out BEFORE this statement
            # leaves the catalog byte for byte as it found it. These rows would
            # be inert either way -- membership pairs them with a watermark row
            # that will never exist -- but "wrote nothing" is a property a test
            # can assert directly, and "wrote something harmless" is one that
            # has to be argued every time the membership clause changes.
            #
            # It does not make the two statements atomic. A takeover landing in
            # the gap between them leaves these rows behind, inert; that is the
            # residual, and it is written up rather than glossed.
            #
            # Chunked, because the members ride in the statement TEXT (the
            # driver substitutes non-VALUES parameters client-side) and an
            # unbounded batch would breach the server's max_query_size. Every
            # chunk carries the same publish_id and the same fence; a chunk
            # refused mid-way leaves earlier chunks inert exactly as the
            # takeover gap does.
            members = [(ref.store_id, ref.pack_id) for ref in refs]
            for chunk in _inline_chunks(members):
                self._client.execute(
                    f"INSERT INTO {self._qualified(self._manifest)} "
                    "(index_version, publish_id, store_id, pack_id) "
                    "SELECT %(index_version)s, toUUID(%(publish_id)s), "
                    "tupleElement(member, 1), toUUID(tupleElement(member, 2)) "
                    "FROM (SELECT arrayJoin(%(members)s) AS member) "
                    f"WHERE {self._lease_fence()}",
                    {
                        "index_version": index_version,
                        "publish_id": publish_id,
                        "members": chunk,
                        "lease_id": lease.lease_id,
                        "publish_timeout_ns": self._config.publish_timeout_ns,
                    },
                    settings=settings,
                )
                if not self._manifest_chunk_published(
                    index_version=index_version,
                    publish_id=publish_id,
                    members=chunk,
                ):
                    self._reject_if_the_lease_is_gone(lease)
                    raise SnapshotPublishRaceError(
                        f"catalog version {index_version} did not publish its "
                        "complete manifest chunk, so no watermark was written "
                        "and no snapshot became visible. Allocate a higher "
                        "version and publish again."
                    )
                lease = self.renew_publisher_lease()
        # The barrier, the fence and the visibility write are ONE server-side
        # statement, so the gap between "am I the highest?", "do I still hold
        # the lease?" and "I am now visible" holds no client round trip -- no
        # network hop, no GC pause, no scheduler stall. A separate SELECT then
        # INSERT left that whole window open.
        #
        # The version barrier also subsumes the indexer's non-monotonic-version
        # guard: the server itself refuses a version that is not strictly above
        # the published head, so a broken allocator cannot publish underneath a
        # watermark a reader already pinned.
        self._client.execute(
            f"INSERT INTO {watermark} "
            "(index_version, publish_id, published_at_ns, indexed_rows, "
            "indexed_packs) "
            "SELECT %(index_version)s, toUUID(%(publish_id)s), "
            "%(published_at_ns)s, "
            "toUInt64(%(indexed_rows)s), toUInt32(%(indexed_packs)s) "
            "FROM system.one "
            # ifNull, not a bare scalar: max() over the empty watermark table
            # answers 0 on a default profile but NULL wherever
            # aggregate_functions_null_for_empty rewrites it, and NULL < v is
            # NULL -- which would refuse every FIRST publish into a fresh
            # catalog and misreport it as a lost race, forever. The client-side
            # _max_version already handles the None; the barrier has to match.
            f"WHERE ifNull((SELECT max(index_version) FROM {watermark}), 0) "
            "< %(index_version)s "
            f"AND {self._lease_fence()}",
            {
                "index_version": index_version,
                "publish_id": publish_id,
                "published_at_ns": published_at_ns,
                "indexed_rows": indexed_rows,
                "indexed_packs": indexed_packs,
                "lease_id": lease.lease_id,
                "publish_timeout_ns": self._config.publish_timeout_ns,
            },
            settings=settings,
        )
        # Ownership, not occupancy. ``count() > 0`` answers "does a row for V
        # exist?", and a row written by anything else -- a stray operator
        # INSERT, a second build, a publisher whose statement overlapped this
        # one -- reads as success. The sole-claimant allocator makes a foreign
        # row at V unlikely, not impossible, and reading the identity back costs
        # exactly what counting cost: the same one-row scan of the same key
        # range.
        #
        # Two questions, not one, because the answers need opposite recoveries.
        # "Is MY row there?" decides whether anything was published at all;
        # "is it the ONLY row there?" decides whether the version is solely
        # this publish's. Collapsing them into ``!= {publish_id}`` reported a
        # foreign row arriving AFTER this one as a lost race, which is the one
        # thing it is not: the watermark row is standing, its manifest rows are
        # paired with it, and its packs are visible.
        owners = {
            self._text(row[0])
            for row in self._client.execute(
                f"SELECT toString(publish_id) FROM {watermark} "
                "WHERE index_version = %(version)s",
                {"version": index_version},
                settings=_DECIDING_READ,
            )
        }
        if publish_id not in owners:
            # Two different failures wear the same shape here, and the caller's
            # recovery differs. A lost VERSION race is repaired by allocating a
            # higher one, which the indexer does. A lost LEASE is not: every
            # retry would fail the same fence, so it has to say so instead.
            self._reject_if_the_lease_is_gone(lease)
            raise SnapshotPublishRaceError(
                f"catalog version {index_version} lost the publish race: the "
                "conditional watermark INSERT was refused, so no row carrying "
                f"publish {publish_id} stands at that version and no snapshot "
                "can admit anything this attempt wrote. Allocate a higher "
                "version and publish again."
            )
        if owners != {publish_id}:
            # The opposite outcome, and it is NOT a lost race: this publish's
            # row IS standing at the version, its manifest rows are paired with
            # it, and its packs are visible. Reporting that as a loss would be
            # false -- and the indexer would retry it, publishing the same
            # batch a second time underneath a snapshot that already contains
            # it. So it is surfaced as what it is: a version whose contents
            # came from more than one publish.
            foreign = ", ".join(sorted(owners - {publish_id}))
            raise SnapshotPublishConflictError(
                f"catalog version {index_version} was published by this writer "
                f"(publish {publish_id}) AND by {foreign}. This publish is "
                "visible and must not be retried; the version's membership is "
                "now the union of both publishes. Something else is writing "
                f"`{self._config.database}`.`{self._config.table_prefix}_*` -- "
                "a second indexer sharing the prefix, or a hand-written INSERT."
            )

    # -- the publisher lease ------------------------------------------------

    @property
    def publisher_lease(self) -> PublisherLease | None:
        return self._leases.lease

    def acquire_publisher_lease(self, holder: str) -> PublisherLease:
        return self._leases.acquire(holder)

    def renew_publisher_lease(self) -> PublisherLease:
        return self._leases.renew()

    def release_publisher_lease(self) -> None:
        self._leases.release()

    def _release_statement(self) -> str:
        return self._leases.release_statement()

    def _claim_lease(self, holder: str, *, lease_id: str) -> PublisherLease:
        return self._leases.claim(holder, lease_id=lease_id)

    def _lease_head(self) -> _LeaseHead:
        return self._leases.head()

    def _lease_fence(self) -> str:
        return self._leases.fence()

    def _publish_timeout(self) -> dict[str, object]:
        return self._leases.publish_timeout()

    def _manifest_chunk_published(
        self,
        *,
        index_version: int,
        publish_id: str,
        members: list[PackIdentity],
    ) -> bool:
        rows = self._client.execute(
            "SELECT count() FROM (SELECT DISTINCT store_id, pack_id FROM "
            f"{self._qualified(self._manifest)} "
            "WHERE index_version = %(index_version)s "
            "AND publish_id = toUUID(%(publish_id)s) "
            "AND (store_id, pack_id) IN %(members)s)",
            {
                "index_version": index_version,
                "publish_id": publish_id,
                "members": members,
            },
            settings=_DECIDING_READ,
        )
        return rows[0][0] == len(set(members))

    def _reject_if_the_lease_is_gone(self, lease: PublisherLease) -> None:
        self._leases.reject_if_gone(lease)

    def collect_garbage(self) -> dict[str, int]:
        """Delete the rows the protocols append and never need again.

        Three tables grow with every pass and nothing collected them. A
        publisher publishing once a second appends ~86k lease rows a day; each
        publish that loses its version race leaves a full set of manifest rows
        behind; every allocation leaves a claim. The only removal path in the
        module was ``drop_schema()``, which destroys the catalog.

        Explicit rather than automatic, and never called from the write path:
        a mutation is a background rewrite of the parts it touches, so an
        indexer that ran one inline would pay for it in the middle of a
        publish. Run it from a maintenance job.

        Returns the number of rows removed per table. Every bound below is
        chosen so that the rows it deletes can never be resolved again by the
        fence, the allocator, or a membership read -- not merely that they look
        old.
        """
        removed: dict[str, int] = {}
        published = self.last_published_version()
        head = self._lease_head()
        settings = {"mutations_sync": 1}

        # Manifest rows of a publish that never reached the watermark, at a
        # version BELOW the published head. Below the head, its watermark
        # INSERT can no longer land -- the barrier requires strictly above --
        # so the pair the membership predicate needs will never exist. An
        # in-flight publish always sits ABOVE the head, which is what keeps
        # this from deleting membership out from under it.
        removed[self._manifest] = self._delete_rows(
            self._manifest,
            "index_version < %(published)s AND (index_version, publish_id) "
            f"NOT IN (SELECT index_version, publish_id FROM "
            f"{self._qualified(self._watermark)})",
            {"published": published},
            settings,
        )

        # Lease rows below the head TERM. The fence resolves exactly one row --
        # the highest ``(term, lease_id)`` -- and terms only ever increase, so
        # a row below the head can never become the head again. The head itself
        # is kept whether or not it has expired: it is what a takeover has to
        # sort above, and deleting it would let a stale claimant's next term
        # collide with a live one's.
        if head.term:
            removed[self._schema.lease_table] = self._delete_rows(
                self._schema.lease_table,
                "term < %(term)s",
                {"term": head.term},
                settings,
            )

        # Version claims at or below the published head. The allocator picks
        # above ``max(claims.version)`` AND above ``last_published_version()``,
        # so the watermark keeps the floor once these are gone -- which is why
        # the watermark table itself is never collected here, and why claims
        # ABOVE the head stay: one of them may be a version allocated by a pass
        # that has not published yet.
        removed[self._version_claims] = self._delete_rows(
            self._version_claims,
            "version <= %(published)s",
            {"published": published},
            settings,
        )
        return removed

    def _delete_rows(
        self, table: str, predicate: str, params: dict, settings: dict
    ) -> int:
        """Count the rows a predicate matches, then delete exactly those.

        Counted first because ``ALTER TABLE ... DELETE`` reports nothing about
        what it removed, and a maintenance job that cannot say what it deleted
        is one nobody runs twice. The count is a separate statement, so a row
        written between the two is simply collected on the next run.
        """
        qualified = self._qualified(table)
        matched = self._client.execute(
            f"SELECT count() FROM {qualified} WHERE {predicate}",
            params,
            settings=_DECIDING_READ,
        )[0][0]
        if matched:
            self._client.execute(
                f"ALTER TABLE {qualified} DELETE WHERE {predicate}",
                params,
                settings=settings,
            )
        return int(matched)

    def last_published_version(self) -> int:
        """Highest published index_version, or 0 when nothing is published."""
        return self._max_version(
            self._watermark,
            "index_version",
            "watermark table returned an invalid version",
        )

    def allocate_version(self) -> int:
        """Allocate the next catalog version: strictly monotonic and unique.

        Sole-claimant protocol over the append-only claims table. Each attempt
        picks a candidate above everything already claimed or published,
        inserts a claim for it, then reads the claims for that version back: a
        claimant proceeds ONLY when its post-insert read shows it is the sole
        claimant. On a server with monotonic read-your-writes visibility -- a
        single node gives it, and the read-back below carries
        ``select_sequential_consistency`` so that a replica does too --
        for two claimants of the same version the later insert always observes
        the earlier one, so at most one of them sees a singleton; contested
        versions are abandoned by everyone who sees the contest. Every
        RETURNED version is durably in the claims table, so later allocations
        always start above it: monotonic + unique, with no clock anywhere in
        the ordering. ``claimed_at_ns`` is diagnostic only.
        """
        attempts = self._config.allocation_attempts
        for attempt in range(attempts):
            claimed = self._max_version(
                self._version_claims,
                "version",
                "claims table returned an invalid version",
            )
            floor = max(claimed, self.last_published_version())
            # A randomized skip only after a collision, so two contenders that
            # keep colliding spread out instead of racing for floor + 1 again.
            candidate = floor + 1 + (secrets.randbelow(8 * attempt + 1) if attempt else 0)
            self._validate_version(candidate)
            claim_id = str(uuid4())
            self._client.execute(
                f"INSERT INTO {self._qualified(self._version_claims)} "
                "(version, claim_id, claimed_at_ns) VALUES",
                [(candidate, claim_id, time_ns())],
                settings=self._quorum_write() or None,
            )
            owners = self._client.execute(
                f"SELECT toString(claim_id) FROM "
                f"{self._qualified(self._version_claims)} "
                "WHERE version = %(version)s",
                {"version": candidate},
                settings=_DECIDING_READ,
            )
            if {self._text(row[0]) for row in owners} == {claim_id}:
                return candidate
            # Contested: someone else claimed the same version -- abandon it
            # entirely and retry above it.
        raise CatalogVersionAllocationError(
            f"could not allocate a catalog version after {attempts} attempts"
        )

    def _max_version(self, table: str, column: str, message: str) -> int:
        # A deciding read: the answer becomes the floor a claim is picked above,
        # and the head a publish must beat.
        rows = self._client.execute(
            f"SELECT max({column}) FROM {self._qualified(table)}",
            settings=_DECIDING_READ,
        )
        if not rows or not rows[0] or rows[0][0] is None:
            return 0
        value = rows[0][0]
        if type(value) is not int or value < 0:
            raise ValueError(message)
        return value

    def commit_packs(
        self, refs: Sequence[PackRef], *, index_version: int
    ) -> None:
        if not refs:
            return
        self._validate_version(index_version)
        rows = [
            (
                ref.pack_id, ref.store_id, ref.object_key, ref.object_bytes,
                ref.checksum, ref.record_count, index_version,
            )
            for ref in refs
        ]
        # The replay guard, and nothing else: committed_pack_ids reads this
        # inventory to skip packs it has already indexed. Snapshot membership
        # used to be written here too; it now belongs to publish_snapshot, so
        # that a pack becomes visible and becomes skippable at two clearly
        # ordered moments rather than one. CatalogIndexer calls this AFTER a
        # successful publish, so a crash in between leaves a pack that is
        # visible but not yet skippable -- redundant work next pass, never the
        # reverse.
        self._client.execute(
            f"INSERT INTO {self._qualified(self._pack_raw)} "
            f"({', '.join(_PACK_COLUMNS)}) VALUES",
            rows,
        )

    def _qualified(self, table: str) -> str:
        return f"{_quoted(self._config.database)}.{_quoted(table)}"

    def _quorum_write(self) -> dict[str, object]:
        """The settings that make a DECIDING write durable before it is read.

        Empty unless an operator has opted in, so a single-node deployment
        pays nothing and nothing changes for anyone who has not asked for it.
        ``insert_quorum_parallel`` is turned OFF alongside, because
        ``select_sequential_consistency`` does not work with it -- setting the
        quorum without clearing the parallel flag would buy latency and no
        guarantee at all.
        """
        quorum = self._config.insert_quorum
        if quorum is None:
            return {}
        return {"insert_quorum": quorum, "insert_quorum_parallel": 0}

    @staticmethod
    def _validate_count(value: object, name: str) -> None:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _validate_version(index_version: int) -> None:
        if type(index_version) is not int or not 0 <= index_version < 2**64:
            raise ValueError("index_version must fit UInt64")

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if not isinstance(value, str):
            raise ValueError(  # noqa: TRY004 - preserve the catalog API
                "ClickHouse returned a non-text identifier"
            )
        return value

    @staticmethod
    def _descriptor_row(item: CaptureDescriptor, index_version: int) -> tuple:
        metadata = item.metadata
        locator = item.locator
        return (
            metadata.capture_id, metadata.tenant_id, metadata.experiment_id,
            metadata.run_id, metadata.session_id, metadata.request_id,
            metadata.sequence_id, metadata.model_id, metadata.model_revision,
            metadata.adapter_revision, metadata.capture_policy_version,
            metadata.hook_name, metadata.layer_number, metadata.producer_rank,
            metadata.step_number, metadata.token_start, metadata.token_end,
            metadata.batch_position, metadata.dtype, list(metadata.shape),
            metadata.captured_at_ns, locator.pack_id, locator.store_id,
            locator.object_key, locator.object_bytes, locator.pack_checksum,
            locator.pack_record_count, locator.offset, locator.stored_length,
            locator.decoded_length, locator.codec, locator.checksum, index_version,
        )
