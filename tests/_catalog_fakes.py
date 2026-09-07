"""An in-memory model of the publisher lease, shared by the catalog fakes.

Four suites drive `ClickHouseCatalogWriter` against a hand-written client, each
modelling a different part of the catalog. All four now have to answer the lease
statements, because `publish_snapshot` renews before it writes anything, and all
four have to APPLY the fence rather than ignore it -- a fake that lets a fenced
statement through makes every publish test pass vacuously, which is the mistake
the version barrier's comment already warns about.

Applying it means reading it off the STATEMENT. Keyed on the `lease_id`
parameter instead -- which `publish_snapshot` passes unconditionally -- the
fence could be deleted from the source outright and every CPU test still
passed, because the fakes were enforcing it on the writer's behalf. So
`fence_admits` matches the predicate the writer emits and raises
`MissingLeaseFence` when a statement that must carry it does not.

What this cannot model is the thing the fence exists for. Two publishers here
are serialised by the Python interpreter, so their statements never overlap on a
server and no amount of fake state reproduces the window. The load-bearing tests
for that live in `tests/test_clickhouse_snapshot_live.py` and run against a real
ClickHouse.
"""

from __future__ import annotations

import re


# A fixed starting point rather than a real clock: expiry is something the tests
# drive by moving `now_ns`, so nothing here depends on how long a test takes.
_EPOCH_NS = 1_700_000_000_000_000_000


class MissingLeaseFence(AssertionError):
    """A statement that has to carry the lease fence was issued without it."""


class UnexpectedLeaseStatement(AssertionError):
    """A lease-table write that is neither the claim nor the release."""


class ClientComputedClock(AssertionError):
    """A lease row shipped timestamps the SERVER was supposed to compute.

    Both lease timestamps come from the server's clock inside the writing
    statement, so no publisher's wall clock -- or the skew between two of them
    -- decides when a lease expires. A statement that carries them as client
    parameters has lost that property, however correct the values look.
    """


# The predicate `the emitted fence` emits, as a pattern.
# Everything the fence DOES is matched exactly; only the table name is loose,
# because each suite runs under its own prefix. Exactness is the point: the
# fence is the whole safety property, so a rewrite of it has to be restated
# here on purpose rather than silently accepted.
# `test_the_fake_matches_the_fence_the_writer_actually_emits` pins this pattern
# against the writer's own output, so drift fails as one clear test rather than
# as four suites going quietly vacuous.
#
# The head is the highest term; within it, a lease's expiry is the MINIMUM
# `expires_at_ns` written under its `(term, lease_id)`, so that a release --
# an expired row at the holder's own term -- ends the lease without needing a
# term of its own. The margin is the publish cap PLUS the configured host clock
# skew bound, because `expires_at_ns` was stamped by one host's `now64()` and
# is compared against another's.
_FENCE = re.compile(
    r"\(SELECT count\(\) FROM \("
    r"SELECT lease_id, min\(expires_at_ns\) AS expires_at_ns "
    r"FROM `[^`]+`\.`[^`]+_publisher_lease` "
    r"WHERE term = \(SELECT max\(term\) FROM `[^`]+`\.`[^`]+_publisher_lease`\) "
    r"GROUP BY lease_id ORDER BY lease_id DESC LIMIT 1"
    r"\) WHERE lease_id = toUUID\(%\(lease_id\)s\) "
    r"AND expires_at_ns > toUInt64\(toUnixTimestamp64Nano\(now64\(9\)\)\) "
    r"\+ toUInt64\(%\(publish_timeout_ns\)s\) "
    r"\+ toUInt64\(%\(clock_skew_ns\)s\)\) = 1"
)

# The release tombstone `the emitted release statement` emits: an already-
# expired row at the releasing writer's OWN term, stamped by the server. It
# reads nothing, because the earlier `INSERT ... SELECT term + 1` form resolved
# the head as of its own SELECT and so was not a compare-and-set against a
# concurrent claim -- a stale release could land at a successor's freshly
# granted term. A row at the writer's own term can never be a successor's
# term, so there is no read left to race. Matched exactly for the same reason
# `_FENCE` is, and pinned against the writer's own output by the same test.
_RELEASE = re.compile(
    r"INSERT INTO `[^`]+`\.`[^`]+_publisher_lease` "
    r"\(term, lease_id, holder, acquired_at_ns, expires_at_ns\) "
    r"SELECT toUInt64\(%\(term\)s\), toUUID\(%\(lease_id\)s\), %\(holder\)s, "
    r"now_ns, now_ns "
    r"FROM \(SELECT toUnixTimestamp64Nano\(now64\(9\)\) AS now_ns\)"
)


def uuid_order(text: str) -> tuple[int, int]:
    """ClickHouse's UUID collation, which is NOT the text order.

    A `UUID` is two `UInt64`s and ClickHouse compares the LOW half first, so
    `ORDER BY lease_id DESC` ranks `00000000-0000-0000-ffff-ffffffffffff`
    ABOVE `ffffffff-ffff-ffff-0000-000000000000` while Python's string order
    ranks it below. Verified on 25.12 against `ORDER BY u DESC LIMIT 1`.

    It matters because the fence and the head read both resolve a contested
    head term with `ORDER BY term DESC, lease_id DESC`, and a fake that
    resolved it by text order would model a different row than the server
    picks -- so any test reaching a contested head would be asserting the
    wrong outcome.
    """
    raw = text.replace("-", "")
    return int(raw[16:], 16), int(raw[:16], 16)


class FakeLeaseTable:
    """`{prefix}_publisher_lease`, with the server clock the tests control.

    ``rows`` is the append-only table, ``now_ns`` is the server's clock, and
    ``fence_admits`` is the predicate the real statements evaluate server-side.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[int, str, str, int, int]] = []
        self.now_ns = _EPOCH_NS
        # Runs between a claimant's INSERT and its read-back -- the window the
        # sole-claimant protocol exists to close.
        self.on_claim_read = None
        # Runs as a fenced statement evaluates its predicate: the window
        # between a publisher renewing its lease and the server executing the
        # write, which is the only place a takeover can land unnoticed.
        self.on_fence = None

    # -- the model ----------------------------------------------------------

    def head(self) -> list[tuple[int, str, str, int, int]]:
        """Every row at the highest term, release tombstones included.

        More than one LEASE at it means it was contested; more than one row for
        one lease means it was released.
        """
        if not self.rows:
            return []
        top = max(term for term, *_ in self.rows)
        return [row for row in self.rows if row[0] == top]

    def claimants(self) -> list[tuple[int, str, str, int]]:
        """The head as the statements see it: one entry per lease at the top
        term, each with its EFFECTIVE expiry (the minimum written under that
        `(term, lease_id)`, so a tombstone ends it), in the order the server
        resolves -- ClickHouse's UUID collation, descending."""
        grouped: dict[str, tuple[int, str, str, int]] = {}
        for term, lease_id, holder, _, expires_at_ns in self.head():
            seen = grouped.get(lease_id)
            grouped[lease_id] = (
                term,
                lease_id,
                holder,
                expires_at_ns if seen is None else min(seen[3], expires_at_ns),
            )
        return sorted(
            grouped.values(), key=lambda row: uuid_order(row[1]), reverse=True
        )

    def fence_passes(
        self, lease_id: object, publish_timeout_ns: int = 0, clock_skew_ns: int = 0
    ) -> bool:
        """The fencing predicate over the model, ignoring any statement."""
        if self.on_fence is not None:
            self.on_fence(lease_id)
        claimants = self.claimants()
        if not claimants:
            return False
        _, resolved, _, expires_at_ns = claimants[0]
        return (
            resolved == lease_id
            and expires_at_ns > self.now_ns + publish_timeout_ns + clock_skew_ns
        )

    def fence_admits(self, statement: str, params) -> bool:
        """Evaluate the fence THIS STATEMENT carries.

        Keyed on the SQL, never on the presence of the `lease_id` parameter.
        `publish_snapshot` passes that parameter whether or not the statement
        uses it, so a fake that keyed on it enforced the fence on the writer's
        behalf: deleting the fence from the source left every CPU test
        passing, which is the one thing these fakes must never do.

        The fence has to GATE the write, not merely appear in it, so beyond
        the text being present this requires it to stand as the final
        top-level AND conjunct of the WHERE clause: preceded by WHERE or AND,
        followed by nothing, with no OR/NOT in the clause ahead of it. That is
        a heuristic, not a SQL parser -- but it is exactly the shape both real
        statements have, and it refuses the cheap weakenings (`... OR 1`,
        `NOT (fence)`, a trailing `OR` clause) that would leave intact fence
        TEXT deciding nothing.
        """
        match = _FENCE.search(statement)
        if match is None:
            raise MissingLeaseFence(
                "a statement that must be fenced on the publisher lease was "
                "issued without the fence. Only the lease holder may make a "
                "snapshot visible, and the check has to ride inside the "
                "statement that writes -- a publisher fenced out after the "
                "write has already made it durable is the failure mode this "
                "design has rejected twice.\n"
                f"statement: {statement}"
            )
        prefix = statement[: match.start()].rstrip()
        suffix = statement[match.end() :].strip()
        # The outer WHERE clause ahead of the fence. The fence's own nested
        # WHERE sits inside the matched span, so it cannot be picked up here.
        where = prefix.rfind("WHERE")
        clause = prefix[where:] if where != -1 else prefix
        if (
            suffix
            or not prefix.endswith(("WHERE", "AND"))
            or " OR " in clause
            or " NOT " in clause
            or "NOT(" in clause
        ):
            raise MissingLeaseFence(
                "the lease fence must be the final top-level AND conjunct of "
                "the WHERE clause. Wrapped in OR/NOT or followed by further "
                "clauses, an intact fence no longer gates the write, and the "
                "statement would land rows for a publisher the fence refuses.\n"
                f"statement: {statement}"
            )
        # Both margins are REQUIRED parameters: a writer that stopped passing
        # the skew bound would otherwise be modelled as a zero-skew deployment.
        return self.fence_passes(
            params["lease_id"], params["publish_timeout_ns"], params["clock_skew_ns"]
        )

    # -- the client interface -----------------------------------------------

    def execute(self, query: str, params=None):
        """Answer a lease statement, or return None for anything else.

        Dispatch is on the statement's TARGET, not on whether the lease table
        is mentioned: the fenced manifest and watermark inserts name it too, in
        the subquery that is the whole point, and swallowing those here would
        hide the writes the caller has to check.
        """
        statement = query.lstrip()
        if statement.startswith("INSERT INTO"):
            if "publisher_lease" not in statement.split(maxsplit=3)[2]:
                return None
            if "ttl_ns" not in params:
                # The release tombstone: an already-expired row at the
                # releasing writer's OWN term. It resolves nothing, so there is
                # no head for it to be stale about -- a row at this writer's
                # term cannot be a successor's term. Matched off the SQL, like
                # the claim, so a release that grew a head read back could not
                # pass quietly.
                if _RELEASE.fullmatch(statement) is None:
                    raise UnexpectedLeaseStatement(
                        "a release tombstone must be an expired row at the "
                        "writer's own term, computed server-side and reading "
                        "nothing; this statement does not match the writer's "
                        "release.\n"
                        f"statement: {statement}"
                    )
                self.rows.append(
                    (
                        int(params["term"]),
                        str(params["lease_id"]),
                        str(params["holder"]),
                        self.now_ns,
                        self.now_ns,
                    )
                )
                return []
            # The claim INSERT. Its timestamps are modelled off the fake's
            # clock, which is only honest while the real statement computes
            # them SERVER-side -- so, as with the fence, that is read off the
            # statement rather than assumed.
            if "toUnixTimestamp64Nano(now64(9))" not in statement or not {
                "acquired_at_ns", "expires_at_ns", "now_ns"
            }.isdisjoint(params):
                raise ClientComputedClock(
                    "a lease claim must compute acquired_at_ns/expires_at_ns "
                    "on the SERVER, inside the INSERT ... SELECT "
                    "(toUnixTimestamp64Nano(now64(9))); shipping them as "
                    "client parameters lets a publisher's own clock decide "
                    "when its lease expires.\n"
                    f"statement: {statement}"
                )
            self.rows.append(
                (
                    int(params["term"]),
                    str(params["lease_id"]),
                    str(params["holder"]),
                    self.now_ns,
                    self.now_ns + int(params["ttl_ns"]),
                )
            )
            return []
        if not statement.startswith("SELECT") or "publisher_lease" not in query:
            return None
        if statement.startswith("SELECT count()"):
            # Not a lease statement: the retention pass counts rows in this
            # table before deleting them. Answering it here would hand a
            # caller lease columns for a question about a count.
            return None
        if "GROUP BY term, lease_id" in query:
            # The head read: every lease at the top term with its effective
            # expiry, in the order the fence resolves them.
            return [
                (term, lease_id, holder, expires_at_ns, self.now_ns)
                for term, lease_id, holder, expires_at_ns in self.claimants()
            ]
        term = int(params["term"])
        if self.on_claim_read is not None:
            self.on_claim_read(term)
        return [
            (lease_id, acquired_at_ns, expires_at_ns)
            for row_term, lease_id, _, acquired_at_ns, expires_at_ns in self.rows
            if row_term == term
        ]
