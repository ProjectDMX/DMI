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
        lease = self._lease
        if lease is None:
            return
        self._client.execute(
            self.release_statement(),
            {"lease_id": lease.lease_id, "holder": lease.holder},
            settings=DECIDING_READ,
        )
        self._lease = None

    def release_statement(self) -> str:
        table = self._qualified_table
        return (
            f"INSERT INTO {table} "
            "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
            "SELECT term + 1, toUUID(%(lease_id)s), %(holder)s, "
            "toUnixTimestamp64Nano(now64(9)), toUnixTimestamp64Nano(now64(9)) "
            f"FROM (SELECT term, lease_id FROM {table} "
            "ORDER BY term DESC, lease_id DESC LIMIT 1) "
            "WHERE lease_id = toUUID(%(lease_id)s)"
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
        rows = self._client.execute(
            "SELECT term, toString(lease_id), holder, expires_at_ns, "
            "toUnixTimestamp64Nano(now64(9)) FROM "
            f"{self._qualified_table} "
            "ORDER BY term DESC, lease_id DESC LIMIT 2",
            settings=DECIDING_READ,
        )
        if not rows:
            return LeaseHead(0, 0, "", "", 0, 0, 0)
        term, lease_id, holder, expires_at_ns, now_ns = rows[0]
        live_until_ns = expires_at_ns
        if len(rows) > 1 and rows[1][0] == term:
            live_until_ns, now_ns = self._client.execute(
                "SELECT max(expires_at_ns), "
                "toUnixTimestamp64Nano(now64(9)) FROM "
                f"{self._qualified_table} WHERE term = %(term)s",
                {"term": term},
                settings=DECIDING_READ,
            )[0]
        return LeaseHead(
            term=term,
            claimants=1 + sum(1 for row in rows[1:] if row[0] == term),
            lease_id=self._text(lease_id),
            holder=self._text(holder),
            expires_at_ns=expires_at_ns,
            live_until_ns=live_until_ns,
            now_ns=now_ns,
        )

    def fence(self) -> str:
        return (
            "(SELECT count() FROM ("
            "SELECT lease_id, expires_at_ns FROM "
            f"{self._qualified_table} "
            "ORDER BY term DESC, lease_id DESC LIMIT 1"
            ") WHERE lease_id = toUUID(%(lease_id)s) "
            "AND expires_at_ns > "
            "toUInt64(toUnixTimestamp64Nano(now64(9))) "
            "+ toUInt64(%(publish_timeout_ns)s)) = 1"
        )

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
        quorum = getattr(self._config, "insert_quorum", None)
        if quorum is None:
            return {}
        # insert_quorum_parallel is cleared alongside: ClickHouse's
        # select_sequential_consistency does not work with it, so the quorum
        # without this buys latency and no guarantee.
        return {"insert_quorum": quorum, "insert_quorum_parallel": 0}

    @property
    def _qualified_table(self) -> str:
        return f"`{self._config.database}`.`{self._table}`"

    @staticmethod
    def _validate_term(term: int) -> None:
        if type(term) is not int or not 0 <= term <= 2**64 - 1:
            raise ValueError("index_version must be an integer in [0, 2^64 - 1]")

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if not isinstance(value, str):
            raise TypeError(
                f"expected text from ClickHouse, got {type(value).__name__}"
            )
        return value
