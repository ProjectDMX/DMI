"""An in-memory model of the publisher lease, shared by the catalog fakes.

Four suites drive `ClickHouseCatalogWriter` against a hand-written client, each
modelling a different part of the catalog. All four now have to answer the lease
statements, because `publish_snapshot` renews before it writes anything, and all
four have to APPLY the fence rather than ignore it -- a fake that lets a fenced
statement through makes every publish test pass vacuously, which is the mistake
the version barrier's comment already warns about.

What this cannot model is the thing the fence exists for. Two publishers here
are serialised by the Python interpreter, so their statements never overlap on a
server and no amount of fake state reproduces the window. The load-bearing tests
for that live in `tests/test_clickhouse_snapshot_live.py` and run against a real
ClickHouse.
"""

from __future__ import annotations


# A fixed starting point rather than a real clock: expiry is something the tests
# drive by moving `now_ns`, so nothing here depends on how long a test takes.
_EPOCH_NS = 1_700_000_000_000_000_000


class FakeLeaseTable:
    """`{prefix}_publisher_lease`, with the server clock the tests control.

    ``rows`` is the append-only table, ``now_ns`` is the server's clock, and
    ``fence_passes`` is the predicate the real statements evaluate server-side.
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
        """Every row at the highest term. More than one means it was contested."""
        if not self.rows:
            return []
        top = max(term for term, *_ in self.rows)
        return [row for row in self.rows if row[0] == top]

    def fence_passes(self, lease_id: object) -> bool:
        """The fencing predicate: the head row is this lease, and it is live."""
        if self.on_fence is not None:
            self.on_fence(lease_id)
        head = self.head()
        if not head:
            return False
        # The statement resolves the head with ORDER BY term DESC, lease_id DESC.
        term, resolved, _, _, expires_at_ns = max(head, key=lambda row: row[1])
        return resolved == lease_id and expires_at_ns > self.now_ns

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
        if "ORDER BY term DESC" in query:
            # The head, in the order the fence resolves it, and at most two
            # rows -- the second only says whether the head term is contested.
            ordered = sorted(self.rows, key=lambda row: row[:2], reverse=True)
            return [
                (term, lease_id, holder, expires_at_ns, self.now_ns)
                for term, lease_id, holder, _, expires_at_ns in ordered[:2]
            ]
        term = int(params["term"])
        if self.on_claim_read is not None:
            self.on_claim_read(term)
        return [
            (lease_id, acquired_at_ns, expires_at_ns)
            for row_term, lease_id, _, acquired_at_ns, expires_at_ns in self.rows
            if row_term == term
        ]
