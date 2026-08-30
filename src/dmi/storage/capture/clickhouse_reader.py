"""ClickHouse-backed catalog reads for bounded analysis.

Reads are pinned to a watermark, so a selection resolves to the same captures
for as long as it lives. The snapshot boundary is **the set of packs committed
at or before that watermark**, read from an append-only commit log -- not a
range of descriptor versions.

That distinction is load-bearing. Descriptor rows live in a
``ReplacingMergeTree``, which is defined to keep only the highest version per
sorting key and delete the rest during a merge. Any snapshot phrased as
``index_version <= W`` over those rows therefore expires at a time nobody
controls: after a merge, the version it wanted is simply gone. Bounding on pack
commits instead depends only on an append-only log, which no merge rewrites.

The reason this works is that a descriptor is derived from an immutable pack
footer, so re-indexing a pack rewrites byte-identical rows. There is no content
to choose between, which is why ``argMax`` here is deduplication rather than
version selection, and why it does not matter which duplicate a merge keeps.
``test_replay_is_invisible_because_it_rewrites_identical_descriptors`` guards
that invariant; if it ever breaks, this design has to be revisited.

The watermark itself comes from a published log rather than from
``max(index_version)`` over the descriptors, because one indexing call writes
descriptors across several INSERTs before the pack markers -- sampling the
descriptor table mid-call pins a batch that then keeps growing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
from uuid import UUID

from .clickhouse_catalog import (
    _CAPTURE_COLUMNS,
    ClickHouseCatalogConfig,
    ClickHouseClient,
    _identifier,
    _quoted,
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


# The catalog sort key, and therefore the keyset pagination order. These are the
# table's ORDER BY prefix, so the tuple comparison that advances a page prunes
# granules on the primary index instead of scanning.
_SORT_KEY = ("tenant_id", "experiment_id", "run_id", "captured_at_ns", "capture_id")
_SORT_KEY_SET = frozenset(_SORT_KEY)

# Projection order is the writer's column order minus ``index_version``, so a
# result row maps positionally onto the descriptor fields.
_PROJECTION = _CAPTURE_COLUMNS[:-1]

_EQUALITY_FILTERS = ("tenant_id", "experiment_id", "run_id", "session_id", "model_id")


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

    def __post_init__(self) -> None:
        _identifier(self.database)
        _identifier(self.table_prefix)
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
        self._commit_log = f"{self._config.table_prefix}_pack_commit_log"

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
        rows = self._client.execute(
            f"SELECT max(index_version) FROM {self._qualified(self._watermark_table)}",
            settings=self._config.settings,
        )
        if not rows or rows[0][0] is None:
            return "0"
        return str(_integer(rows[0][0], "watermark"))

    def search(self, query: CaptureQuery) -> CapturePage:
        max_watermark = int(self.current_watermark())
        filter_hash = query.filter_hash

        if query.cursor is None:
            watermark = max_watermark
            after: CursorKey | None = None
        else:
            cursor = decode_cursor(
                query.cursor, filter_hash=filter_hash, max_watermark=max_watermark
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
            f"GROUP BY {', '.join(_quoted(name) for name in _SORT_KEY)} "
            f"ORDER BY {', '.join(_quoted(name) for name in _SORT_KEY)} "
            "LIMIT %(row_limit)s"
        )
        rows = self._client.execute(sql, params, settings=self._config.settings)
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
        # whose commit-log rows are written but not yet covered by a
        # publish, defeating the publish-last ordering (the paginated path
        # already enforces this bound in decode_cursor).
        if requested > int(self.current_watermark()):
            raise ValueError(
                "selection watermark exceeds the published watermark"
            )
        params = {
            "watermark": requested,
            "tenant_id": tenant_id,
            "capture_ids": list(capture_ids),
        }
        # tenant_id leads the WHERE clause because it is the first ORDER BY
        # column: without it, a capture_id-only filter -- the LAST sort-key
        # column -- prunes nothing, and every lookup scans the whole table
        # (then trips max_rows_to_read once the catalog is large). With it,
        # the primary index narrows the read to one tenant's range, and the
        # capture_id bloom-filter skip index prunes granules inside it.
        sql = (
            f"SELECT {self._projection()} FROM {self._qualified()} "
            "WHERE tenant_id = %(tenant_id)s AND "
            "capture_id IN %(capture_ids)s AND (store_id, pack_id) IN "
            f"(SELECT store_id, pack_id FROM {self._qualified(self._commit_log)} "
            "WHERE index_version <= %(watermark)s) "
            f"GROUP BY {', '.join(_quoted(name) for name in _SORT_KEY)}"
        )
        rows = self._client.execute(sql, params, settings=self._config.settings)
        return tuple(self._descriptor(row) for row in rows)

    # -- SQL construction ---------------------------------------------------

    def _qualified(self, table: str | None = None) -> str:
        return (
            f"{_quoted(self._config.database)}."
            f"{_quoted(table or self._capture_raw)}"
        )

    @staticmethod
    def _projection() -> str:
        # Sort-key columns are grouped on, so they project directly; everything
        # else collapses whatever duplicate rows survive. Any version of a
        # descriptor is byte identical to any other, so argMax here is
        # deduplication rather than version selection.
        #
        # Deliberately unaliased: naming an aggregate after its own source
        # column shadows that column everywhere else in the statement, and
        # ClickHouse then rejects a filter on it with "Aggregate function ...
        # is found in WHERE in query". Rows map onto descriptors positionally,
        # so the server-side column names are never read.
        return ", ".join(
            _quoted(name)
            if name in _SORT_KEY_SET
            else f"argMax({_quoted(name)}, index_version)"
            for name in _PROJECTION
        )

    def _filters(
        self, query: CaptureQuery, *, watermark: int, after: CursorKey | None
    ) -> tuple[list[str], dict[str, object]]:
        # The snapshot is the set of packs committed at or before the
        # watermark, not a range of descriptor versions. Descriptor rows are
        # byte identical across re-indexing, so which version survives a merge
        # is irrelevant -- but a version *range* over them is not durable,
        # because ReplacingMergeTree deletes superseded rows.
        clauses = [
            # Pack identity is (store_id, pack_id): matching on pack_id alone
            # would let the same UUID committed by a second store at a later
            # version slip inside a pinned snapshot.
            "(store_id, pack_id) IN (SELECT store_id, pack_id FROM "
            f"{self._qualified(self._commit_log)} "
            "WHERE index_version <= %(watermark)s)"
        ]
        params: dict[str, object] = {"watermark": watermark}

        # Equality and range filters apply to raw rows before grouping. That is
        # safe because a descriptor is derived from an immutable pack footer, so
        # re-indexing one capture rewrites identical values; it is also much
        # faster, since these predicates reach the primary index.
        for name in _EQUALITY_FILTERS:
            value = getattr(query, name)
            if value is not None:
                clauses.append(f"{_quoted(name)} = %({name})s")
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
            columns = ", ".join(_quoted(name) for name in _SORT_KEY)
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
        if len(row) != len(_PROJECTION):
            raise PackFormatError(
                f"catalog row has {len(row)} columns, expected {len(_PROJECTION)}"
            )
        value: Mapping[str, object] = dict(zip(_PROJECTION, row))
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
