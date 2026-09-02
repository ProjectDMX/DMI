"""ClickHouse-backed catalog reads for bounded analysis.

Reads are pinned to a watermark, so a selection resolves to the same captures
for as long as it lives. The snapshot boundary is **the set of packs committed
at or before that watermark**, read from an append-only snapshot manifest --
not a range of descriptor versions. A manifest row counts only once the publish
that wrote it -- ``(index_version, publish_id)``, not the version alone -- also
appears in the watermark log, which is what keeps a publish that lost its race
from leaking packs into an already-pinned snapshot.

That distinction is load-bearing. Descriptor rows live in a
``ReplacingMergeTree``, which is defined to keep only the highest version per
sorting key and delete the rest during a merge. Any snapshot phrased as
``index_version <= W`` over those rows therefore expires at a time nobody
controls: after a merge, the version it wanted is simply gone. Bounding on pack
commits instead depends only on an append-only log, which no merge rewrites.

A merge can only ever collapse rows describing one capture in ONE pack, because
pack identity is part of the table's sort key, and those rows are byte
identical -- re-indexing a pack reads the same immutable footer twice. So no
merge can destroy a row a pinned snapshot still needs.

Three live tests hold that down, and between them they cover both halves.
``test_a_merge_does_not_destroy_a_pinned_snapshot`` and
``test_two_packs_describing_one_capture_both_survive_a_merge`` force the merge
with ``OPTIMIZE ... FINAL`` and assert the pinned read still resolves and that
neither pack's row was collapsed into the other;
``test_re_indexing_one_pack_still_collapses_to_a_single_row`` asserts the
collapse that IS wanted still happens, so the sort key has not simply stopped
deduplicating. ``test_replay_is_invisible_because_it_rewrites_identical_
descriptors`` -- which this paragraph used to cite alone -- covers the premise
rather than the conclusion: it shows a replay writes the same descriptor, and
it neither forces a merge nor involves two packs.

Rows describing one capture in DIFFERENT packs survive side by side, and the
``argMax`` projection grouped on capture identity picks between them:
newest-wins. A reader pinned before the second pack was committed never sees
its rows at all, because the membership clause excludes that pack, so the pin
still resolves to the pack it was taken over. Two packs indexed in one batch
share an ``index_version``, so version alone does not order those rows; what
does is described at :meth:`ClickHouseCaptureCatalog._projection`.

The identity rule
-----------------

``(tenant_id, capture_id)`` identifies a capture. Every descriptor field except
the locator -- ``store_id``, ``pack_id``, ``object_key``, ``object_bytes``,
``pack_checksum``, ``pack_record_count``, ``payload_offset``,
``stored_length``, ``decoded_length``, ``codec``, ``payload_checksum`` -- is
immutable for that identity. Two rows describing one capture may differ ONLY in
where its bytes are, because everything else is read back from a pack footer
that was written once.

Two things rest on that rule, and neither is visible in the SQL.

The first is that the layers disagree about identity by inspection and are
nonetheless consistent. This module groups on the five-column ``_SORT_KEY``,
``CaptureSelection`` dedups on ``capture_id`` alone, and ``get_by_ids`` filters
on ``tenant_id`` plus ``capture_id``. Under the rule the three extra grouping
columns are functions of ``(tenant_id, capture_id)``, so grouping on five
columns partitions the rows exactly as grouping on two would: no capture can
split across two result rows, and no result row can mix two captures.

The second is that it is what makes the pre-aggregation ``WHERE`` filters in
``_filters`` safe, which is the load-bearing assumption behind the current
query shape. Those predicates run on raw rows, before the grouping that
resolves a capture to one pack. If a filtered column could differ between a
capture's rows, a filter could match only the row that loses the ``argMax`` --
admitting a capture on the strength of a description the reader will not return
-- or match none of them and drop a capture the winning row satisfies. Because
every filtered column is immutable, all of a capture's rows agree on it and the
filter's answer cannot depend on which row wins. Filtering after aggregation
would be the alternative, and it would forfeit the primary index.
``test_only_the_locator_may_differ_between_a_captures_rows`` pins the split
between the two kinds of column so that adding a mutable one has to be
deliberate.

The watermark itself comes from a published log rather than from
``max(index_version)`` over the descriptors, because one indexing call writes
descriptors across several INSERTs before the pack markers -- sampling the
descriptor table mid-call pins a batch that then keeps growing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from .clickhouse_catalog import ClickHouseCatalogConfig
from .clickhouse_schema import CAPTURE_COLUMNS
from .clickhouse_sql import (
    DECIDING_READ,
    ClickHouseClient,
    identifier,
    inline_chunks,
    inline_text_bytes,
    membership_predicate,
    quoted,
)
from .cursor import CursorKey, decode_cursor, encode_cursor
from .model import (
    CaptureDescriptor,
    CaptureMetadata,
    CapturePage,
    CaptureQuery,
    PackFormatError,
    PayloadLocator,
)

# Capture identity: what one result row is one of, and therefore the GROUP BY,
# the result ORDER BY and the keyset pagination order.
#
# Deliberately NOT the descriptor table's physical sort key
# (``clickhouse_catalog._CAPTURE_TABLE_ORDER``, which appends store_id and
# pack_id so a merge cannot collapse two packs' rows for one capture). Grouping
# on the physical key instead would emit one row per pack for a capture
# described in two of them: supersession would stop resolving newest-wins, and
# a page could carry the same capture twice with different locators.
#
# It is still a *prefix* of that physical key, which is what lets the tuple
# comparison that advances a page prune granules on the primary index instead
# of scanning; ``test_the_logical_sort_key_is_a_prefix_of_the_table_order``
# pins the two together.
_SORT_KEY = ("tenant_id", "experiment_id", "run_id", "captured_at_ns", "capture_id")
_SORT_KEY_SET = frozenset(_SORT_KEY)

# Every column the reader reads: the writer's column order minus
# ``index_version``, which orders the rows rather than describing a capture.
_PROJECTION = CAPTURE_COLUMNS[:-1]

# The columns that are not grouped on, and so have to be resolved out of a
# capture's rows. They travel as ONE tuple; see ``_projection``.
_RESOLVED = tuple(name for name in _PROJECTION if name not in _SORT_KEY_SET)

# A result row is the grouping columns in sort-key order followed by a single
# tuple holding every resolved column in ``_RESOLVED`` order.
_ROW_WIDTH = len(_SORT_KEY) + 1

_EQUALITY_FILTERS = ("tenant_id", "experiment_id", "run_id", "session_id", "model_id")

# The ordering argument the projection's argMax resolves on. It is a tuple, not
# ``index_version``, because it has to be a TOTAL order over the rows in one
# group; ``_projection`` explains why, and what breaks without it.
_RESOLUTION_ORDER = "(index_version, store_id, pack_id)"

@dataclass(frozen=True, slots=True)
class ClickHouseReaderConfig:
    """Bounds applied to every catalog read.

    The four ``max_*`` settings ride on the per-query settings map, so a breach
    surfaces as a server-side exception instead of a long-running scan.
    """

    database: str = "default"
    table_prefix: str = "dmi"
    max_capture_ids: int = 10_000
    max_rows_to_read: int = 50_000_000
    max_bytes_to_read: int = 4 * 1024**3
    max_execution_time: int = 15
    # Whether the statement that reads DESCRIPTORS waits for its replica to
    # catch up, as the watermark read before it already does.
    #
    # It is not redundant with that read, and this is the whole reason the
    # setting exists: a ReplicatedMergeTree keeps a replication log PER TABLE,
    # so a replica synced on `{prefix}_index_watermark` may still be behind on
    # `{prefix}_snapshot_manifest` and `{prefix}_capture_raw`. Left plain, a
    # read pinned to a genuinely published watermark returns a SHORT page --
    # and `search` issues a cursor at that watermark, so the next page resumes
    # PAST the rows that were missing. Captures are skipped in a walk that
    # reports success. Lag can only omit, never invent, so nothing unpublished
    # or cross-tenant is exposed either way.
    #
    # Costed rather than assumed: on a single node the server accepts and
    # ignores it (verified on 25.12), so this is free unless the tables are
    # replicated. On a replicated deployment it makes every page wait for the
    # replica, which is the price of a walk that does not skip. An operator who
    # would rather have the latency can turn it off and accept the gap.
    consistent_snapshot_reads: bool = True

    def __post_init__(self) -> None:
        identifier(self.database)
        identifier(self.table_prefix)
        for name in (
            "max_capture_ids",
            "max_rows_to_read",
            "max_bytes_to_read",
            "max_execution_time",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_catalog(cls, config: ClickHouseCatalogConfig) -> ClickHouseReaderConfig:
        """Read with the database and prefix a writer was configured with."""
        return cls(database=config.database, table_prefix=config.table_prefix)

    @property
    def settings(self) -> dict[str, object]:
        return {
            "max_rows_to_read": self.max_rows_to_read,
            "max_bytes_to_read": self.max_bytes_to_read,
            "max_execution_time": self.max_execution_time,
            "read_overflow_mode": "throw",
            "timeout_overflow_mode": "throw",
        }

    @property
    def bounded_read_settings(self) -> dict[str, object]:
        """Settings for a statement that resolves against a pinned watermark."""
        if not self.consistent_snapshot_reads:
            return self.settings
        return {**self.settings, **DECIDING_READ}


def _text(value: object, name: str) -> str:
    """Normalise a ClickHouse string column to ``str``."""
    if isinstance(value, bytes):
        # FixedString columns come back padded with NULs on some drivers.
        return value.rstrip(b"\x00").decode("utf-8")
    if isinstance(value, UUID):
        return str(value)
    if not isinstance(value, str):
        raise PackFormatError(f"catalog returned a non-text {name}: {type(value).__name__}")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise PackFormatError(f"catalog returned a non-integer {name}")
    return value


class ClickHouseCaptureCatalog:
    """A :class:`~.model.CaptureCatalog` backed by the derived ClickHouse catalog."""

    def __init__(
        self,
        client: ClickHouseClient,
        config: ClickHouseReaderConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or ClickHouseReaderConfig()
        self._capture_raw = f"{self._config.table_prefix}_capture_raw"
        self._watermark_table = f"{self._config.table_prefix}_index_watermark"
        self._manifest = f"{self._config.table_prefix}_snapshot_manifest"

    @property
    def config(self) -> ClickHouseReaderConfig:
        return self._config

    # -- public API ---------------------------------------------------------

    def current_watermark(self) -> str:
        """The newest version whose indexing batch is fully committed.

        Read from the published watermark log, never from the descriptor table.
        One indexing call writes descriptors across several INSERTs and then the
        pack markers; ``max(index_version)`` over the descriptors would expose
        that version between those writes, letting a reader pin a half-written
        batch that keeps growing under it.
        """
        return str(self._published_head())

    def _published_head(self, *, deciding: bool = False) -> int:
        """The published head version, 0 when nothing is published yet.

        ``deciding`` adds ``select_sequential_consistency`` for the one read
        whose answer refuses a call rather than merely pinning a snapshot.
        """
        settings = dict(self._config.settings)
        if deciding:
            settings.update(DECIDING_READ)
        rows = self._client.execute(
            f"SELECT max(index_version) FROM {self._qualified(self._watermark_table)}",
            settings=settings,
        )
        if not rows or rows[0][0] is None:
            return 0
        return _integer(rows[0][0], "watermark")

    def search(self, query: CaptureQuery) -> CapturePage:
        filter_hash = query.filter_hash

        if query.cursor is None:
            # A stale head here merely pins a slightly older snapshot, which a
            # fresh search cannot tell from having run a moment earlier, so the
            # read stays cheap rather than sequentially consistent.
            watermark = self._published_head()
            after: CursorKey | None = None
        else:
            # The head bounds decode_cursor, which REFUSES a cursor above it.
            # A cursor is caller data stamped with a watermark the catalog
            # itself issued; read from a lagging replica, that genuinely
            # published watermark reads as "ahead of the catalog" -- a hard,
            # false rejection of a valid cursor under ordinary replication lag
            # -- so this bounding read is deciding, exactly as get_by_ids's is.
            cursor = decode_cursor(
                query.cursor,
                filter_hash=filter_hash,
                max_watermark=self._published_head(deciding=True),
            )
            watermark = cursor.watermark
            after = cursor.key

        clauses, params = self._filters(query, watermark=watermark, after=after)
        # One row beyond the page tells us whether a cursor is owed, without a
        # second counting query.
        params["row_limit"] = query.limit + 1
        sql = (
            f"SELECT {self._projection()} FROM {self._qualified()} "
            f"WHERE {' AND '.join(clauses)} "
            f"GROUP BY {', '.join(quoted(name) for name in _SORT_KEY)} "
            f"ORDER BY {', '.join(quoted(name) for name in _SORT_KEY)} "
            "LIMIT %(row_limit)s"
        )
        rows = self._client.execute(
            sql, params, settings=self._config.bounded_read_settings
        )
        descriptors = tuple(self._descriptor(row) for row in rows)

        next_cursor = None
        if len(descriptors) > query.limit:
            descriptors = descriptors[: query.limit]
            next_cursor = encode_cursor(
                _key_of(descriptors[-1]), watermark=watermark, filter_hash=filter_hash
            )
        return CapturePage(
            items=descriptors, next_cursor=next_cursor, watermark=str(watermark)
        )

    def get_by_ids(
        self, capture_ids: Sequence[str], *, tenant_id: str, watermark: str
    ) -> tuple[CaptureDescriptor, ...]:
        if len(capture_ids) > self._config.max_capture_ids:
            raise ValueError(
                f"capture id lookup exceeds max_capture_ids: "
                f"{len(capture_ids)} > {self._config.max_capture_ids}"
            )
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if not capture_ids:
            return ()
        requested = _parse_watermark(watermark)
        # A selection is caller data. Its watermark must be one the indexer
        # actually published -- accepting an arbitrary value would read packs
        # whose manifest rows are written but whose watermark row has not
        # landed, defeating the publish ordering (the paginated path already
        # enforces this bound in decode_cursor).
        #
        # A DECIDING read, unlike search()'s: this answer refuses the call
        # outright, where a stale head in search() merely pins a slightly
        # older snapshot. Read from a lagging replica, a watermark the indexer
        # genuinely published reads as "exceeds the published watermark" -- a
        # hard, false rejection of valid caller data under ordinary
        # replication lag -- so this read carries the same
        # select_sequential_consistency the writer's deciding reads do
        # (accepted and ignored by a non-replicated server).
        if requested > self._published_head(deciding=True):
            raise ValueError(
                "selection watermark exceeds the published watermark"
            )
        # tenant_id leads the WHERE clause because it is the first ORDER BY
        # column: without it, a capture_id-only filter -- capture_id sits
        # behind captured_at_ns in the sort key, and a lookup supplies no time
        # bound -- prunes nothing, and every lookup scans the whole table
        # (then trips max_rows_to_read once the catalog is large). With it,
        # the primary index narrows the read to one tenant's range, and the
        # capture_id bloom-filter skip index prunes granules inside it.
        sql = (
            f"SELECT {self._projection()} FROM {self._qualified()} "
            "WHERE tenant_id = %(tenant_id)s AND "
            f"capture_id IN %(capture_ids)s AND {self._membership()} "
            f"GROUP BY {', '.join(quoted(name) for name in _SORT_KEY)}"
        )
        # Chunked by rendered bytes, because the ids land in the statement TEXT
        # and a full-size lookup can breach the server's max_query_size.
        # Each chunk reads the same pinned snapshot -- the membership bound is
        # the immutable set of packs published at or before the watermark -- so
        # the union of the chunk results is the unchunked result. Ids are sent
        # once each: within one statement the GROUP BY collapses a repeated id,
        # and two chunks naming the same id would return its row twice.
        unique_ids = list(dict.fromkeys(capture_ids))
        descriptors: list[CaptureDescriptor] = []
        for chunk in inline_chunks(unique_ids, item_bytes=inline_text_bytes):
            params = {
                "watermark": requested,
                "tenant_id": tenant_id,
                "capture_ids": chunk,
            }
            rows = self._client.execute(
                sql, params, settings=self._config.bounded_read_settings
            )
            descriptors.extend(self._descriptor(row) for row in rows)
        return tuple(descriptors)

    # -- SQL construction ---------------------------------------------------

    def _qualified(self, table: str | None = None) -> str:
        return (
            f"{quoted(self._config.database)}."
            f"{quoted(table or self._capture_raw)}"
        )

    def _membership(self) -> str:
        """The packs inside the snapshot, as a subquery on (store_id, pack_id).

        Two conditions, and the second is the whole point. A manifest row is
        written before its watermark row, so requiring the publish to appear in
        the watermark table is what stops a publish that lost the race -- which
        never wrote one -- from leaking its packs into a snapshot that was
        pinned before it ran.

        That second test pairs ``(index_version, publish_id)`` rather than
        matching the version alone, so a manifest row counts only when the SAME
        publish reached the watermark. On the version alone, the CONTENTS of
        snapshot V would be whatever anyone wrote at V -- a losing publisher's
        rows, an operator's stray INSERT -- while the winner of V unwittingly
        published them. Owning a version and owning its membership are separate
        claims and both are needed.

        Pack identity is the PAIR too: matching pack_id alone would let the same
        UUID published by a second store at a later version slip inside a
        pinned snapshot.

        The predicate itself has ONE definition,
        ``clickhouse_catalog.membership_predicate``, shared with the public
        view's DDL so the two cannot drift apart about what exists; this
        method only supplies the reader's snapshot bound.
        """
        return membership_predicate(
            self._qualified(self._manifest),
            self._qualified(self._watermark_table),
            bounded=True,
        )

    @staticmethod
    def _projection() -> str:
        """The SELECT list: identity columns direct, the rest as ONE aggregate.

        Capture-identity columns are grouped on, so they project directly.
        Everything else is resolved by a single ``argMax`` over a tuple of all
        of them. Both halves of that -- one aggregate, and the ordering argument
        it uses -- are load-bearing, each for a different reason.

        **One aggregate, so a row cannot be mixed.** A projection of
        twenty-seven separate ``argMax`` calls resolves each column
        independently, and ClickHouse does not define which row an ``argMax``
        picks out of a tie. Nothing in that shape forbids ``store_id`` coming
        from one row and ``object_key`` from another: a descriptor describing no
        pack that exists. Aggregating the whole tuple at once makes that
        impossible by construction rather than merely unobserved -- one
        aggregate keeps one row, and every column comes out of it.

        Stated plainly because it was checked: a mixed row could NOT be
        reproduced on 25.12. Forty-two combinations of ``max_threads``,
        two-level and external aggregation, JIT-compiled aggregates and block
        size produced none, and the reason looks structural (a group's aggregate
        states share one block and merge in lockstep, so every ``argMax`` in it
        keeps the same row). That is an observation about one build, not a
        promise the engine makes, and it is not why this shape is cheap -- so it
        is not a reason to unpick the tuple back into per-column aggregates.

        **A total ordering argument, so the row that wins cannot move.** This is
        the failure that reproduces. Ordering on ``index_version`` alone ties
        routinely: every pack indexed in one ``CatalogIndexer.index`` call is
        written at one version, so two packs describing the same capture in one
        batch produce rows whose ``index_version`` is equal. The engine breaks
        those ties consistently within a query but not across physical layouts:
        one pinned corpus resolved to a different pack at ``max_threads=1`` than
        it did above it, and to a different one again once a merge had put both
        rows in one part. A background merge does that at a time nobody
        controls, so a selection resolved before one and hydrated after it
        resolves to different bytes with nothing reporting a change.

        ``(index_version, store_id, pack_id)`` is a total order over the rows in
        a group. They differ by pack identity -- that is exactly why it is in
        the table's physical sort key -- so the tuple is unique per distinct row
        and the maximum is one row. Rows that still tie on the whole tuple are
        one pack re-indexed at one version, which rewrites byte-identical rows,
        so which of those wins cannot be observed.

        Across versions this is unchanged newest-wins: ``index_version`` leads
        the tuple, so a later pack still supersedes an earlier one. Within a
        version the winner is the highest ``(store_id, pack_id)`` -- there is no
        version ordering left to honour, and an arbitrary but FIXED choice is
        what a reader needs, so that a selection resolved twice resolves to the
        same bytes.

        The shape is also what keeps determinism affordable, which is why the
        two halves arrived together. ClickHouse compares a tuple ordering
        argument through a generic ``Field``, once per row per aggregate, and
        the ``GROUP BY`` completes before ``LIMIT`` applies, so every page pays
        for every row. On ``benchmarks.bench_capture_search`` at 50k rows, a
        100-row page costs 140.1 ms ordering twenty-seven aggregates on
        ``index_version``, 548.3 ms ordering twenty-seven of them on the tuple,
        and 171.7 ms ordering one -- +22.6% for determinism where the per-column
        form cost +291%. Across page sizes, pagination depth and the selectivity
        cases this shape runs +17% to +43%.
        """
        # Deliberately unaliased: naming an aggregate after a source column
        # shadows that column everywhere else in the statement, and ClickHouse
        # then rejects a filter on it with "Aggregate function ... is found in
        # WHERE in query". _descriptor maps the row positionally, so the
        # server-side column names are never read.
        resolved = ", ".join(quoted(name) for name in _RESOLVED)
        return ", ".join(
            [quoted(name) for name in _SORT_KEY]
            + [f"argMax(tuple({resolved}), {_RESOLUTION_ORDER})"]
        )

    def _filters(
        self, query: CaptureQuery, *, watermark: int, after: CursorKey | None
    ) -> tuple[list[str], dict[str, object]]:
        # The snapshot is the set of packs committed at or before the
        # watermark, not a range of descriptor versions: a version *range* over
        # descriptor rows is not durable, because ReplacingMergeTree deletes
        # rows sharing a sort key at a time nobody controls. Bounding on packs
        # is also what makes a pin resolve to the pack it was taken over when a
        # later pack re-describes the same capture.
        clauses = [self._membership()]
        params: dict[str, object] = {"watermark": watermark}

        # Equality and range filters apply to raw rows before grouping. That is
        # safe because a descriptor is derived from an immutable pack footer, so
        # re-indexing one capture rewrites identical values; it is also much
        # faster, since these predicates reach the primary index.
        for name in _EQUALITY_FILTERS:
            value = getattr(query, name)
            if value is not None:
                clauses.append(f"{quoted(name)} = %({name})s")
                params[name] = value
        if query.hook_names:
            clauses.append("hook_name IN %(hook_names)s")
            params["hook_names"] = list(query.hook_names)
        if query.layer_numbers:
            clauses.append("layer_number IN %(layer_numbers)s")
            params["layer_numbers"] = list(query.layer_numbers)
        if query.captured_after_ns is not None:
            clauses.append("captured_at_ns >= %(captured_after_ns)s")
            params["captured_after_ns"] = query.captured_after_ns
        if query.captured_before_ns is not None:
            clauses.append("captured_at_ns <= %(captured_before_ns)s")
            params["captured_before_ns"] = query.captured_before_ns

        if after is not None:
            columns = ", ".join(quoted(name) for name in _SORT_KEY)
            placeholders = ", ".join(f"%(after_{name})s" for name in _SORT_KEY)
            clauses.append(f"({columns}) > ({placeholders})")
            params.update(
                {
                    "after_tenant_id": after.tenant_id,
                    "after_experiment_id": after.experiment_id,
                    "after_run_id": after.run_id,
                    "after_captured_at_ns": after.captured_at_ns,
                    "after_capture_id": after.capture_id,
                }
            )
        return clauses, params

    # -- row mapping --------------------------------------------------------

    @staticmethod
    def _descriptor(row: Sequence[object]) -> CaptureDescriptor:
        # The grouping columns, then the one aggregate holding the rest. Flatten
        # the two back into a single mapping so the field reads below do not
        # have to know which side a column arrived on.
        if len(row) != _ROW_WIDTH:
            raise PackFormatError(
                f"catalog row has {len(row)} columns, expected {_ROW_WIDTH}"
            )
        resolved = row[-1]
        if not isinstance(resolved, (list, tuple)) or len(resolved) != len(_RESOLVED):
            raise PackFormatError(
                "catalog returned a malformed resolved-column tuple, expected "
                f"{len(_RESOLVED)} columns"
            )
        value: Mapping[str, object] = {
            **dict(zip(_SORT_KEY, row)),
            **dict(zip(_RESOLVED, resolved)),
        }
        shape = value["shape"]
        if not isinstance(shape, (list, tuple)):
            raise PackFormatError("catalog returned a non-array shape")

        adapter_revision = value["adapter_revision"]
        metadata = CaptureMetadata(
            capture_id=_text(value["capture_id"], "capture_id"),
            tenant_id=_text(value["tenant_id"], "tenant_id"),
            experiment_id=_text(value["experiment_id"], "experiment_id"),
            run_id=_text(value["run_id"], "run_id"),
            session_id=_text(value["session_id"], "session_id"),
            request_id=_text(value["request_id"], "request_id"),
            sequence_id=_text(value["sequence_id"], "sequence_id"),
            model_id=_text(value["model_id"], "model_id"),
            model_revision=_text(value["model_revision"], "model_revision"),
            adapter_revision=(
                None
                if adapter_revision is None
                else _text(adapter_revision, "adapter_revision")
            ),
            capture_policy_version=_text(
                value["capture_policy_version"], "capture_policy_version"
            ),
            hook_name=_text(value["hook_name"], "hook_name"),
            layer_number=_integer(value["layer_number"], "layer_number"),
            producer_rank=_integer(value["producer_rank"], "producer_rank"),
            step_number=_integer(value["step_number"], "step_number"),
            token_start=_integer(value["token_start"], "token_start"),
            token_end=_integer(value["token_end"], "token_end"),
            batch_position=_integer(value["batch_position"], "batch_position"),
            dtype=_text(value["dtype"], "dtype"),
            shape=tuple(_integer(dim, "shape") for dim in shape),
            captured_at_ns=_integer(value["captured_at_ns"], "captured_at_ns"),
        )
        locator = PayloadLocator(
            pack_id=_text(value["pack_id"], "pack_id"),
            store_id=_text(value["store_id"], "store_id"),
            object_key=_text(value["object_key"], "object_key"),
            object_bytes=_integer(value["object_bytes"], "object_bytes"),
            pack_checksum=_text(value["pack_checksum"], "pack_checksum"),
            pack_record_count=_integer(
                value["pack_record_count"], "pack_record_count"
            ),
            offset=_integer(value["payload_offset"], "payload_offset"),
            stored_length=_integer(value["stored_length"], "stored_length"),
            decoded_length=_integer(value["decoded_length"], "decoded_length"),
            codec=_text(value["codec"], "codec"),
            checksum=_text(value["payload_checksum"], "payload_checksum"),
        )
        try:
            return CaptureDescriptor(metadata=metadata, locator=locator)
        except ValueError as exc:  # pragma: no cover - defensive
            raise PackFormatError(f"invalid catalog descriptor: {exc}") from exc


def _key_of(descriptor: CaptureDescriptor) -> CursorKey:
    metadata = descriptor.metadata
    return CursorKey(
        tenant_id=metadata.tenant_id,
        experiment_id=metadata.experiment_id,
        run_id=metadata.run_id,
        captured_at_ns=metadata.captured_at_ns,
        capture_id=metadata.capture_id,
    )


def _parse_watermark(watermark: str) -> int:
    if not isinstance(watermark, str) or not watermark.isdigit():
        raise ValueError("watermark must be a decimal string")
    value = int(watermark)
    if value >= 2**64:
        raise ValueError("watermark must fit UInt64")
    return value
