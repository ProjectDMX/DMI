from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .catalog import PublisherLeaseError, PublisherLeaseHeldError
from .clickhouse_sql import DECIDING_READ



class ClickHouseClient(Protocol):
    def execute(self, query: str, params=None, **kwargs): ...


class LeaseConfig(Protocol):
    database: str
    table_prefix: str
    lease_ttl_ns: int
    publish_timeout_ns: int
    clock_skew_ns: int
    insert_quorum: int | None


@dataclass(frozen=True, slots=True)
class PublisherLease:
    """A publisher lease whose timestamps come from the server clock."""

    term: int
    lease_id: str
    holder: str
    acquired_at_ns: int
    expires_at_ns: int


@dataclass(frozen=True, slots=True)
class LeaseHead:
    term: int
    claimants: int
    lease_id: str
    holder: str
    expires_at_ns: int
    live_until_ns: int
    now_ns: int


class ClickHouseLeaseCoordinator:
    def __init__(self, client: ClickHouseClient, config: LeaseConfig) -> None:
        self._client = client
        self._config = config
        self._table = f"{config.table_prefix}_publisher_lease"
        self._lease: PublisherLease | None = None

    @property
    def lease(self) -> PublisherLease | None:
        return self._lease

    def acquire(self, holder: str) -> PublisherLease:
        # Bounded in BYTES, which is what the column stores and what the
        # message promises. ``len()`` counts code points, so it admitted a
        # 256-character CJK or emoji holder at up to 1024 bytes.
        if not isinstance(holder, str) or not 0 < len(holder.encode()) <= 256:
            raise ValueError("holder must be a non-empty string of at most 256 bytes")
        held = self._lease
        return self.claim(
            holder, lease_id=held.lease_id if held is not None else str(uuid4())
        )

    def renew(self) -> PublisherLease:
        lease = self._lease
        if lease is None:
            raise PublisherLeaseError(
                "no publisher lease is held; call acquire_publisher_lease() "
                "before publishing. Only the lease holder can make a snapshot "
                "visible, and the check rides inside the publish statement, so "
                "publishing without one writes nothing."
            )
        return self.claim(lease.holder, lease_id=lease.lease_id)

    def release(self) -> None:
        """Give the lease back at once, without racing a successor for its term.

        The tombstone is an already-expired row at THIS writer's own term --
        the term it was granted -- and a lease's expiry is the minimum
        ``expires_at_ns`` written under its ``(term, lease_id)``, so the row
        ends the lease wherever it lands in the table. It reads nothing.

        The earlier form, ``INSERT ... SELECT term + 1 ... WHERE lease_id =
        :me`` over the head, looked like a compare-and-set and was not one: its
        WHERE was evaluated against the head as of its own SELECT, not against
        a concurrent insert. A stale release could read expired lease A at term
        T, claimant B could insert T + 1 and pass its singleton read-back, and
        A's already-running release could land its expired row at that same
        T + 1 -- a contested head until B's TTL, and B fenced out at once
        whenever A's UUID sorted higher (reproduced on 25.12 with a ``sleep``
        in the release). No serialisation of the two statements is available
        in ClickHouse, so the repair is a representation that cannot share or
        supersede a successor's term: a writer's own term is below every term
        granted after it.
        """
        lease = self._lease
        if lease is None:
            return
        self._client.execute(
            self.release_statement(),
            {"term": lease.term, "lease_id": lease.lease_id, "holder": lease.holder},
            # A deciding WRITE like the claim: the successor's head read is
            # what this row is written for.
            settings=self._quorum_write() or None,
        )
        self._lease = None

    def release_statement(self) -> str:
        return (
            f"INSERT INTO {self._qualified_table} "
            "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
            "SELECT toUInt64(%(term)s), toUUID(%(lease_id)s), %(holder)s, "
            "now_ns, now_ns "
            "FROM (SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)"
        )

    def claim(self, holder: str, *, lease_id: str) -> PublisherLease:
        head = self.head()
        self._reject_live(head, lease_id)
        term = head.term + 1
        self._validate_term(term)
        self._insert(term=term, lease_id=lease_id, holder=holder)
        rows = self._client.execute(
            "SELECT toString(lease_id), acquired_at_ns, expires_at_ns "
            f"FROM {self._qualified_table} WHERE term = %(term)s",
            {"term": term},
            settings=DECIDING_READ,
        )
        if {self._text(row[0]) for row in rows} == {lease_id}:
            self._lease = PublisherLease(
                term=term,
                lease_id=lease_id,
                holder=holder,
                acquired_at_ns=rows[0][1],
                expires_at_ns=rows[0][2],
            )
            return self._lease
        self._lease = None
        self._reject_live(self.head(), lease_id)
        raise PublisherLeaseError("publisher lease claim was not recorded")

    def head(self) -> LeaseHead:
        """Every lease at the top term, each with its effective expiry.

        One row per ``(term, lease_id)`` at the highest term, in the order the
        fence resolves them (ClickHouse's UUID collation, descending). A
        lease's expiry is the MINIMUM ``expires_at_ns`` written under its key,
        so a release tombstone at the holder's own term ends it. One read,
        because the head and the count of claimants at it have to come from
        the same snapshot of the table.
        """
        table = self._qualified_table
        rows = self._client.execute(
            "SELECT term, toString(lease_id), any(holder), min(expires_at_ns), "
            f"toUnixTimestamp64Nano(now64(9)) FROM {table} "
            f"WHERE term = (SELECT max(term) FROM {table}) "
            "GROUP BY term, lease_id ORDER BY lease_id DESC",
            settings=DECIDING_READ,
        )
        if not rows:
            return LeaseHead(0, 0, "", "", 0, 0, 0)
        term, lease_id, holder, expires_at_ns, now_ns = rows[0]
        return LeaseHead(
            term=term,
            claimants=len(rows),
            lease_id=self._text(lease_id),
            holder=self._text(holder),
            expires_at_ns=expires_at_ns,
            live_until_ns=max(row[3] for row in rows),
            now_ns=now_ns,
        )

    def fence(self) -> str:
        """The predicate every visibility write carries, evaluated server-side.

        Resolves ONE lease from the head term -- the highest ``lease_id`` in
        ClickHouse's UUID order, at its effective (minimum) expiry -- and asks
        two things of that one row: is it mine, and does it have more than the
        publish cap PLUS the configured host clock skew bound left.

        The skew bound is part of the margin because the two timestamps come
        from two clocks on a replicated deployment: ``expires_at_ns`` was
        stamped by ``now64()`` on the host that handled the claim, and this
        ``now64()`` runs on whichever host handles the publish. Let the holder's
        host be ahead of true time by ``a`` and a successor's by ``b``, with
        ``|b - a| <= S``. The holder's statement is admitted only while more
        than ``timeout + S`` remains on its host's clock and is capped at
        ``timeout`` on that same clock, so it finishes before true time
        ``e - S - a``; the successor may claim once its clock passes ``e``,
        i.e. after true time ``e - b``. The gap between them is
        ``S - (b - a) >= 0``: the two cannot overlap unless the real skew
        exceeds the configured bound.
        """
        table = self._qualified_table
        return (
            "(SELECT count() FROM ("
            "SELECT lease_id, min(expires_at_ns) AS expires_at_ns "
            f"FROM {table} "
            f"WHERE term = (SELECT max(term) FROM {table}) "
            "GROUP BY lease_id ORDER BY lease_id DESC LIMIT 1"
            ") WHERE lease_id = toUUID(%(lease_id)s) "
            "AND expires_at_ns > "
            "toUInt64(toUnixTimestamp64Nano(now64(9))) "
            "+ toUInt64(%(publish_timeout_ns)s) "
            "+ toUInt64(%(clock_skew_ns)s)) = 1"
        )

    def fence_parameters(self, lease: PublisherLease) -> dict[str, object]:
        """The parameters `fence()` reads, so callers cannot drop one."""
        return {
            "lease_id": lease.lease_id,
            "publish_timeout_ns": self._config.publish_timeout_ns,
            "clock_skew_ns": self._config.clock_skew_ns,
        }

    def publish_timeout(self) -> dict[str, object]:
        return {
            "max_execution_time": self._config.publish_timeout_ns // 1_000_000_000,
            "timeout_overflow_mode": "throw",
        }

    def reject_if_gone(self, lease: PublisherLease) -> None:
        head = self.head()
        if head.lease_id == lease.lease_id:
            return
        self._lease = None
        raise PublisherLeaseError(
            f"publisher lease {lease.lease_id} (term {lease.term}) no longer "
            f"stands at the head of `{self._config.database}`."
            f"`{self._config.table_prefix}_publisher_lease`, which is now term "
            f"{head.term} held by {head.holder!r}: the publish was fenced out "
            "and made no snapshot visible. Acquire a lease again and re-index; "
            "retrying with a higher version would fail the same fence. If the "
            "takeover landed after a manifest chunk, its rows are still there "
            "-- inert, since no watermark row will ever pair with them, and "
            "collectable by a retention job."
        )

    def _reject_live(self, head: LeaseHead, lease_id: str) -> None:
        if head.live_until_ns <= head.now_ns or (
            head.claimants == 1 and head.lease_id == lease_id
        ):
            return
        self._lease = None
        if head.claimants > 1:
            raise PublisherLeaseHeldError(
                f"publisher lease term {head.term} on "
                f"`{self._config.database}`."
                f"`{self._config.table_prefix}_*` is contested and remains "
                f"unavailable for another {head.live_until_ns - head.now_ns} "
                "ns. A claimant may have completed its ownership read-back "
                "before the competing row arrived, so no higher term is safe "
                "until every claim at this term expires."
            )
        raise PublisherLeaseHeldError(
            f"publisher lease on `{self._config.database}`."
            f"`{self._config.table_prefix}_*` is held by {head.holder!r} "
            f"(lease {head.lease_id}, term {head.term}) for another "
            f"{head.expires_at_ns - head.now_ns} ns. Only one publisher may "
            "make snapshots visible; wait for it to expire or stop it."
        )

    def _insert(self, *, term: int, lease_id: str, holder: str) -> None:
        self._client.execute(
            f"INSERT INTO {self._qualified_table} "
            "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
            "SELECT toUInt64(%(term)s), toUUID(%(lease_id)s), %(holder)s, "
            "now_ns, now_ns + toUInt64(%(ttl_ns)s) "
            "FROM (SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)",
            {
                "term": term,
                "lease_id": lease_id,
                "holder": holder,
                "ttl_ns": self._config.lease_ttl_ns,
            },
            # A DECIDING write: the head read that follows decides whether this
            # claimant holds the lease, and a read is only as sound as the
            # durability of the write it is asked about. Empty unless an
            # operator has opted in -- see ClickHouseCatalogConfig.
            settings=self._quorum_write() or None,
        )

    def _quorum_write(self) -> dict[str, object]:
        quorum = self._config.insert_quorum
        if quorum is None:
            return {}
        # insert_quorum_parallel is cleared alongside: ClickHouse's
        # select_sequential_consistency does not work with it, so the quorum
        # without this buys latency and no guarantee. The timeout is bounded
        # for the reason the catalog's copy explains: the default is 600s, and
        # nothing else caps a lease claim.
        return {
            "insert_quorum": quorum,
            "insert_quorum_parallel": 0,
            "insert_quorum_timeout": self._config.publish_timeout_ns // 1_000_000,
        }

    @property
    def _qualified_table(self) -> str:
        return f"`{self._config.database}`.`{self._table}`"

    @staticmethod
    def _validate_term(term: int) -> None:
        if type(term) is not int or not 0 <= term <= 2**64 - 1:
            raise ValueError("lease term must be an integer in [0, 2^64 - 1]")
    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if not isinstance(value, str):
            raise TypeError(
                f"expected text from ClickHouse, got {type(value).__name__}"
            )
        return value
