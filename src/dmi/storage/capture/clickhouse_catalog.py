from __future__ import annotations

import os
import secrets
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import sleep as _sleep, time_ns
from uuid import uuid4

from .catalog import (
    CatalogVersionAllocationError,
    PackIdentity,
    PublisherLeaseError,
    SnapshotPublishConflictError,
    SnapshotPublishRaceError,
)
from .clickhouse_lease import (
    ClickHouseLeaseCoordinator,
    PublisherLease,
)
from .clickhouse_schema import (
    CAPTURE_COLUMNS,
    PACK_COLUMNS,
    ClickHouseCatalogSchema,
)
from .clickhouse_sql import (
    DECIDING_READ,
    ClickHouseClient,
    identifier,
    inline_chunks,
    inline_tuple_bytes,
    inline_version_identity_bytes,
    quoted,
    quorum_write,
    text,
)
from .model import CaptureDescriptor, PackRef

# The least lease life a publish may be configured to reach the fence with.
# The fence is safe at any positive margin, but a margin under a client round
# trip is refused rather than accepted: the renewal that precedes every fenced
# statement would land with less than `publish_timeout_ns + clock_skew_ns` left
# on every attempt, so nothing could ever publish. 100 ms is a loopback round
# trip with room to spare and the smallest margin the live suites use.
MINIMUM_FENCE_MARGIN_NS = 100_000_000


def _inline_chunks(items: list[PackIdentity]):
    return inline_chunks(items, item_bytes=inline_tuple_bytes)


def _inline_publish_chunks(items: list[tuple[int, str]]):
    return inline_chunks(items, item_bytes=inline_version_identity_bytes)


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
    # The bound on how far apart two ClickHouse HOSTS' clocks may be, and the
    # third term of the margin above. Every lease timestamp is stamped by the
    # server rather than by a publisher, which removes the publishers' clocks
    # from the decision -- but on a replicated deployment "the server" is
    # several hosts, and ``expires_at_ns`` stamped by one host's ``now64()`` is
    # compared with another's inside the fence. Without this term a successor
    # on a fast host could consider the holder expired while the holder's
    # statement, admitted on a slow host, was still running:
    # ``clock_B - clock_A > lease_ttl_ns - publish_timeout_ns`` was enough.
    # The fence requires ``publish_timeout_ns + clock_skew_ns`` of lease life
    # remaining, which makes an overlap require a real skew ABOVE this bound;
    # see ClickHouseLeaseCoordinator.fence for the derivation.
    #
    # Zero is exact for a single host, which is what the defaults describe. A
    # replicated deployment -- declared here by ``insert_quorum`` -- must set
    # its measured bound (an NTP-disciplined cluster is typically well under
    # a second; measure it, and leave headroom), and construction refuses a
    # quorum without one.
    clock_skew_ns: int = 0
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
    # Set to the quorum size (2 or more) on a replicated deployment. Version
    # claims, leases, snapshot publication, and descriptors then wait for that
    # many replicas. Descriptor reads use sequential consistency and therefore
    # need the same write quorum. The replay-only pack inventory remains a cheap
    # bulk write because a stale read there causes redundant work, not data loss.
    #
    # **Set it once, for the life of the catalog.** Measured on 25.12 against
    # two replicas: once a replicated table has taken a quorum insert, a later
    # NON-quorum insert into it is invisible to a
    # ``select_sequential_consistency`` read -- plain read 2 rows, sequential
    # read 1. Every read-back in this module is a sequential read of the
    # writer's own insert, so turning this off on a catalog that had it on
    # leaves the protocols unable to confirm themselves. They fail loudly
    # (``CatalogVersionAllocationError``, "every candidate version was claimed
    # by another writer") rather than proceeding blind, which is the safe
    # direction and still an outage. Turning it ON mid-life is safe; turning it
    # off requires a rebuild.
    insert_quorum: int | None = None

    def __post_init__(self) -> None:
        identifier(self.database)
        identifier(self.table_prefix)
        for name in (
            "query_pack_limit",
            "allocation_attempts",
            "lease_ttl_ns",
            "publish_timeout_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.clock_skew_ns) is not int or self.clock_skew_ns < 0:
            raise ValueError("clock_skew_ns must be a non-negative integer")
        margin = self.lease_ttl_ns - (self.publish_timeout_ns + self.clock_skew_ns)
        if margin < MINIMUM_FENCE_MARGIN_NS:
            raise ValueError(
                "lease_ttl_ns must exceed publish_timeout_ns plus "
                f"clock_skew_ns by at least {MINIMUM_FENCE_MARGIN_NS} ns; this "
                f"leaves {margin} ns. That margin is what keeps a publish "
                "statement from still running when its lease becomes takeable, "
                "and the skew bound is the part of it two hosts' clocks can "
                "eat -- but it is also the whole time a renewed lease has to "
                "reach the fence. Thinner than a round trip, every publish is "
                "refused by its own fence and retried at a higher version "
                "until the attempts run out"
            )
        if self.insert_quorum is not None and self.clock_skew_ns == 0:
            raise ValueError(
                "clock_skew_ns must be set alongside insert_quorum: a quorum "
                "declares a replicated deployment, where lease expiries are "
                "stamped by one host's clock and fenced against another's, and "
                "a zero bound claims those clocks never disagree. Measure the "
                "cluster's host skew and set the bound above it"
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
        # One publish in flight per writer, and so per lease. The server-side
        # fence identifies a HOLDER, not an operation: two publishes issued
        # under one ``lease_id`` both renew it, both pass the fence, and the
        # version barrier inside each INSERT ... SELECT is evaluated when that
        # statement is admitted rather than serialised against the other, so
        # both watermark rows land -- in either order. A reader pinned at the
        # higher one then watches the lower version's membership arrive
        # underneath it. Reproduced against ClickHouse 25.12 with two threads
        # on one writer: 300 of 300 rounds landed both adjacent watermarks,
        # 57 exposed the higher before the lower. ClickHouse offers nothing to
        # serialise the two statements, so the exclusion has to sit where the
        # shared identity is minted: here, in the process that owns the lease.
        # Re-entrant, because ``publish_snapshot`` renews the lease and
        # ``ensure_schema`` acquires it, both under the same lock.
        self._serial = threading.RLock()
        # The lease belongs to THIS process. A forked child inherits
        # ``self._leases.lease`` byte for byte, and its publishes would share
        # the parent's ``lease_id`` across two address spaces that no lock can
        # reach -- the same race, with no way to observe it from either side.
        self._owner_pid = os.getpid()

    def _owned_by_this_process(self) -> None:
        if os.getpid() == self._owner_pid:
            return
        raise PublisherLeaseError(
            f"this ClickHouseCatalogWriter was created in process "
            f"{self._owner_pid} and is being used from process {os.getpid()}. "
            "A writer and the publisher lease it holds belong to one process: "
            "a forked copy would publish under the same lease_id as its "
            "parent, and two publishes under one lease are not excluded by "
            "the fence. Create a writer, and acquire a lease, in the process "
            "that publishes."
        )

    def ensure_schema(self, *, sleep: Callable[[float], None] = _sleep) -> None:
        # The writer's own coordinator, so an install is serialised on the
        # lease this writer may already hold rather than refused by it.
        # ``sleep`` is the wait between install-lease attempts, a parameter
        # only so a test can drive it.
        with self._serial:
            self._owned_by_this_process()
            self._schema.ensure(self._leases, sleep=sleep)

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
                settings=DECIDING_READ,
            )
            committed.update(
                (text(row[0]), text(row[1])) for row in rows
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
            f"({', '.join(CAPTURE_COLUMNS)}) VALUES",
            rows,
            settings=self._quorum_write() or None,
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

        Serialised per writer: one publish is in flight under this writer's
        lease at a time, and a second caller waits for it. See ``_serial`` in
        the constructor for why the fence alone does not give this.

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
        stronger one. A takeover before the first manifest INSERT
        leaves nothing behind; a takeover in a GAP between statements -- one
        per manifest chunk, then the watermark, each gap a full client round
        trip which ``max_execution_time`` does not bound because it caps each
        statement rather than the sequence -- leaves the manifest rows already
        written while the rest are refused. Those rows are inert:
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
        with self._serial:
            self._owned_by_this_process()
            self._publish_snapshot_serialised(
                index_version=index_version,
                refs=refs,
                published_at_ns=published_at_ns,
                indexed_rows=indexed_rows,
                indexed_packs=indexed_packs,
            )

    def _publish_snapshot_serialised(
        self,
        *,
        index_version: int,
        refs: Sequence[PackRef],
        published_at_ns: int,
        indexed_rows: int,
        indexed_packs: int,
    ) -> None:
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
            **DECIDING_READ,
            **self._leases.publish_timeout(),
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
                    f"WHERE {self._leases.fence()}",
                    {
                        "index_version": index_version,
                        "publish_id": publish_id,
                        "members": chunk,
                        **self._leases.fence_parameters(lease),
                    },
                    settings=settings,
                )
                if not self._manifest_chunk_published(
                    index_version=index_version,
                    publish_id=publish_id,
                    members=chunk,
                ):
                    self._leases.reject_if_gone(lease)
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
            f"AND {self._leases.fence()}",
            {
                "index_version": index_version,
                "publish_id": publish_id,
                "published_at_ns": published_at_ns,
                "indexed_rows": indexed_rows,
                "indexed_packs": indexed_packs,
                **self._leases.fence_parameters(lease),
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
            text(row[0])
            for row in self._client.execute(
                f"SELECT toString(publish_id) FROM {watermark} "
                "WHERE index_version = %(version)s",
                {"version": index_version},
                settings=DECIDING_READ,
            )
        }
        if publish_id not in owners:
            # Two different failures wear the same shape here, and the caller's
            # recovery differs. A lost VERSION race is repaired by allocating a
            # higher one, which the indexer does. A lost LEASE is not: every
            # retry would fail the same fence, so it has to say so instead.
            self._leases.reject_if_gone(lease)
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
        if refs and not self._manifest_is_whole(
            index_version=index_version,
            publish_id=publish_id,
            expected=len({(ref.store_id, ref.pack_id) for ref in refs}),
        ):
            # The watermark row stands, and admits nothing. The barrier and
            # the fence are evaluated when the statement is ADMITTED, and the
            # row lands later; a statement that stalled in between can land
            # BELOW a head a faster publisher established meanwhile. That is
            # the one ordering in which collect_garbage() -- which deletes
            # manifest rows below the head with no watermark row -- can have
            # removed this publish's membership before its watermark existed
            # to protect it. Confirmed only now, because until the watermark
            # row stood the manifest was collectable, and once it stands the
            # pair is what the retention bound keeps.
            #
            # Reported as a lost race because the recovery is the same: a
            # higher version, republished. It must NOT be committed: the packs
            # are not visible at this version, and an inventory row would make
            # every later pass skip them for good -- the exact
            # inventory-without-membership state ensure_schema refuses. The
            # standing watermark row is harmless: membership requires the
            # pair, and no manifest row carries this publish_id.
            raise SnapshotPublishRaceError(
                f"catalog version {index_version} published its watermark but "
                f"its membership (publish {publish_id}) had been collected "
                "before the watermark row landed: a retention pass ran while "
                "this statement was in flight below a newer head. The "
                "watermark row stands and admits nothing, so no snapshot "
                "contains these packs. Do not commit them; allocate a higher "
                "version and publish again."
            )

    # -- the publisher lease ------------------------------------------------

    @property
    def publisher_lease(self) -> PublisherLease | None:
        return self._leases.lease

    def acquire_publisher_lease(self, holder: str) -> PublisherLease:
        with self._serial:
            self._owned_by_this_process()
            return self._leases.acquire(holder)

    def renew_publisher_lease(self) -> PublisherLease:
        with self._serial:
            self._owned_by_this_process()
            return self._leases.renew()

    def release_publisher_lease(self) -> None:
        with self._serial:
            self._owned_by_this_process()
            self._leases.release()

    def _manifest_chunk_published(
        self,
        *,
        index_version: int,
        publish_id: str,
        members: list[PackIdentity],
    ) -> bool:
        return self._manifest_member_count(
            index_version=index_version, publish_id=publish_id, members=members
        ) == len(set(members))

    def _manifest_is_whole(
        self, *, index_version: int, publish_id: str, expected: int
    ) -> bool:
        """Does the publish still hold membership for every pack it admitted?

        Named no members, so it is one bounded statement however many packs the
        publish covered -- unlike the chunk read-back, which has to ask about
        exactly the members its statement carried.
        """
        return self._manifest_member_count(
            index_version=index_version, publish_id=publish_id
        ) == expected

    def _manifest_member_count(
        self,
        *,
        index_version: int,
        publish_id: str,
        members: list[PackIdentity] | None = None,
    ) -> int:
        """Distinct packs this publish holds membership for, optionally bounded.

        ``members`` narrows the count to the identities a particular statement
        carried; omitted, it counts everything the publish admitted. The
        parameter is left out of the query entirely when it is not used, so the
        statement text says which question was asked.
        """
        params: dict[str, object] = {
            "index_version": index_version,
            "publish_id": publish_id,
        }
        bound = ""
        if members is not None:
            params["members"] = members
            bound = " AND (store_id, pack_id) IN %(members)s"
        rows = self._client.execute(
            "SELECT count() FROM (SELECT DISTINCT store_id, pack_id FROM "
            f"{self._qualified(self._manifest)} "
            "WHERE index_version = %(index_version)s "
            f"AND publish_id = toUUID(%(publish_id)s){bound})",
            params,
            settings=DECIDING_READ,
        )
        return rows[0][0]

    def collect_garbage(
        self, *, sleep: Callable[[float], None] = _sleep
    ) -> dict[str, int]:
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

        ``sleep`` is the settling wait the manifest bound needs; it is a
        parameter only so a test can drive it.
        """
        removed: dict[str, int] = {}
        published = self.last_published_version()
        head = self._leases.head()
        settings = {"mutations_sync": 1}

        # Manifest rows of a publish that never reached the watermark, at a
        # version BELOW the published head. Below the head, its watermark
        # INSERT can no longer be ADMITTED -- the barrier requires strictly
        # above -- so the pair the membership predicate needs will never come
        # into existence through a new statement. A watermark statement that
        # was admitted above the head and then stalled can still LAND below it
        # after this runs; publish_snapshot confirms its membership after the
        # row stands and refuses the publish when that has happened, so those
        # packs are republished rather than committed without membership.
        #
        # Resolved with a plain SELECT on the initiator and deleted as literal
        # pairs, in two steps on purpose. The one-step form -- `ALTER TABLE
        # ... DELETE WHERE ... NOT IN (SELECT ... FROM {watermark})` -- reads a
        # second, independently replicated table inside the mutation, which
        # ClickHouse refuses on ReplicatedMergeTree under the default settings
        # (allow_nondeterministic_mutations = 0 and
        # mutations_execute_subqueries_on_initiator = 0 on 25.12). A literal
        # IN list is deterministic on every replica.
        removed[self._manifest] = self._delete_manifest_publishes(
            self._settled_orphan_publishes(published, sleep), settings
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

    def _orphaned_manifest_publishes(self, published: int) -> list[tuple[int, str]]:
        """Every ``(index_version, publish_id)`` below the head with no watermark row."""
        rows = self._client.execute(
            "SELECT DISTINCT index_version, toString(publish_id) FROM "
            f"{self._qualified(self._manifest)} "
            "WHERE index_version < %(published)s "
            "AND (index_version, publish_id) NOT IN "
            f"(SELECT index_version, publish_id FROM {self._qualified(self._watermark)}) "
            "ORDER BY index_version, publish_id",
            {"published": published},
            # Deciding: a replica behind on the watermark table would report a
            # published pair as orphaned and delete live membership.
            settings=DECIDING_READ,
        )
        return [(int(row[0]), text(row[1])) for row in rows]

    def _settled_orphan_publishes(
        self, published: int, sleep: Callable[[float], None]
    ) -> list[tuple[int, str]]:
        """Orphaned publishes that were still orphaned a publish timeout later.

        One read is not enough, and this is the finding that says so. "Below
        the head" means no watermark statement can be ADMITTED for that version
        any more; a statement admitted above the head and then stalled can
        still LAND below it. `publish_snapshot` confirms its membership after
        its watermark row stands, but that confirmation happens on the
        publisher's clock: it can read rows this pass is about to delete, and
        the publish then commits packs whose membership is gone.

        So the set is intersected across two reads a `publish_timeout_ns` apart.
        Any statement admitted before the first read is capped at that timeout
        by `max_execution_time`, so by the second read it has either landed --
        putting a watermark row beside the pair, which drops it from the second
        read -- or been aborted. Anything admitted afterwards is above the head
        and outside the bound entirely.

        The cap is checked between processing blocks rather than pre-empted, so
        a statement blocked in a lock can overrun it (see "The takeover
        instant" in docs/catalog-descriptor-key.md). This narrows the window to
        that residual instead of leaving it a full round trip wide; it does not
        claim to close it.
        """
        first = self._orphaned_manifest_publishes(published)
        if not first:
            return []
        sleep(self._config.publish_timeout_ns / 1e9)
        settled = set(first) & set(self._orphaned_manifest_publishes(published))
        return sorted(settled)

    def _delete_manifest_publishes(
        self, pairs: list[tuple[int, str]], settings: dict
    ) -> int:
        """Delete the manifest rows of these publishes, bounded per statement.

        The pairs ride in the statement TEXT, so an unbounded orphan set could
        breach the server's max_query_size; each chunk is counted and deleted
        on its own, and a pair straddles no chunk.
        """
        return sum(
            self._delete_rows(
                self._manifest,
                "(index_version, publish_id) IN %(pairs)s",
                {"pairs": chunk},
                settings,
            )
            for chunk in _inline_publish_chunks(pairs)
        )

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
            settings=DECIDING_READ,
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

        Serialised on the writer's lock, so two threads sharing a writer do
        not interleave the floor read, the claim and the read-back of one
        attempt with another's. This does not order allocation against
        publication -- a thread may allocate the lower version and publish
        second -- and does not need to: the barrier inside the watermark
        statement refuses a version that is not above the published head, and
        the indexer re-allocates. What the lock removes is the case the fence
        cannot see, two publishes in flight at once under one lease.
        """
        with self._serial:
            self._owned_by_this_process()
            return self._allocate_version_serialised()

    def _allocate_version_serialised(self) -> int:
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
                settings=DECIDING_READ,
            )
            if {text(row[0]) for row in owners} == {claim_id}:
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
            settings=DECIDING_READ,
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
            f"({', '.join(PACK_COLUMNS)}) VALUES",
            rows,
        )

    def _qualified(self, table: str) -> str:
        return f"{quoted(self._config.database)}.{quoted(table)}"

    def _quorum_write(self) -> dict[str, object]:
        return quorum_write(self._config)

    @staticmethod
    def _validate_count(value: object, name: str) -> None:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _validate_version(index_version: int) -> None:
        if type(index_version) is not int or not 0 <= index_version < 2**64:
            raise ValueError("index_version must fit UInt64")

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
