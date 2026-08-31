from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
from time import time_ns
from typing import Protocol, Sequence
from uuid import uuid4

from .catalog import (
    CatalogSchemaVersionError,
    PackIdentity,
    PublisherLeaseError,
    PublisherLeaseHeldError,
    SnapshotPublishConflictError,
    SnapshotPublishRaceError,
)
from .model import CaptureDescriptor, PackRef


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
    lease_attempts: int = 8

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


@dataclass(frozen=True, slots=True)
class PublisherLease:
    """A held publisher lease, as the catalog recorded it.

    ``term`` is the monotonic slot the sole-claimant protocol runs over;
    ``lease_id`` is the fencing token the publish statement checks. The two
    timestamps come from the SERVER's clock, so no publisher's wall clock
    participates in deciding whether a lease is live.
    """

    term: int
    lease_id: str
    holder: str
    acquired_at_ns: int
    expires_at_ns: int


@dataclass(frozen=True, slots=True)
class _LeaseHead:
    """The highest term in the lease table, read with the server's clock.

    ``claimants`` is what separates a lease from an abandoned claim: exactly
    one row means somebody holds that term, more than one means it was
    contested and everyone who saw the contest walked away from it.
    """

    term: int
    # Saturates at 2: the only question asked of it is whether the head term is
    # contested, and one claimant too many answers that as well as ten do.
    claimants: int
    lease_id: str
    holder: str
    expires_at_ns: int
    now_ns: int


_CAPTURE_COLUMNS = (
    "capture_id", "tenant_id", "experiment_id", "run_id", "session_id",
    "request_id", "sequence_id", "model_id", "model_revision",
    "adapter_revision", "capture_policy_version", "hook_name", "layer_number",
    "producer_rank", "step_number", "token_start", "token_end",
    "batch_position", "dtype", "shape", "captured_at_ns", "pack_id",
    "store_id", "object_key", "object_bytes", "pack_checksum",
    "pack_record_count", "payload_offset", "stored_length", "decoded_length",
    "codec", "payload_checksum", "index_version",
)

# Catalog facets: descriptor-derived columns that make the catalog filterable
# and sortable server-side. They are pure functions of columns the writer
# already stores, so MATERIALIZED computes them at insert with no indexer
# change and no extra object reads.
#
# The casts are not decoration. On ClickHouse 25.12/26.9 ``arrayProduct``
# returns Float64 and ``UInt64 - UInt64`` returns Int64, neither of which fits
# the declared column type.
#
# ``compression_ratio`` guards its own divisor with ``if(stored_length = 0, 0,
# ...)`` rather than ``nullIf(stored_length, 0)``: the justification here used
# to cite ``nullIf`` making the expression Nullable, which is true and is why
# it is not used, but naming a function the DDL does not contain sent a reader
# looking for it. A MATERIALIZED column of a non-Nullable type cannot take a
# Nullable expression at all, so the branch is the form that compiles.
_FACET_COLUMNS = (
    ("facet_version", "UInt16", "1"),
    ("element_count", "UInt64", "toUInt64(arrayProduct(shape))"),
    ("tensor_rank", "UInt8", "toUInt8(length(shape))"),
    ("token_span", "UInt64", "toUInt64(token_end - token_start)"),
    (
        "compression_ratio",
        "Float32",
        "toFloat32(if(stored_length = 0, 0, decoded_length / stored_length))",
    ),
)


def _facet_ddl() -> str:
    return ",\n".join(
        f"{name} {kind} MATERIALIZED {expression}"
        for name, kind, expression in _FACET_COLUMNS
    )


_PACK_COLUMNS = (
    "pack_id", "store_id", "object_key", "object_bytes", "pack_checksum",
    "record_count", "index_version",
)

# The descriptor table's PHYSICAL sort key, which is not the same thing as the
# logical capture identity the reader groups, orders and paginates on
# (``clickhouse_reader._SORT_KEY``, the five columns this starts with).
#
# ReplacingMergeTree deletes rows that share a sort key, keeping the highest
# index_version. On capture identity alone that is only safe while two rows for
# one capture are byte identical -- which nothing enforces. A pack copied to a
# second store and reconciled, or a producer retrying a capture_id after the
# first pack was sealed, both produce two rows for one capture with different
# locators; a merge would then silently delete one, and a snapshot pinned to the
# deleted row's pack fails with "selection no longer resolves". Appending pack
# identity gives those rows different keys, so no merge can collapse them.
#
# Rows for one capture in the SAME pack still share the full key and still
# collapse -- and those really are byte identical, which is the replay case the
# engine is here for.
_CAPTURE_TABLE_ORDER = (
    "tenant_id", "experiment_id", "run_id", "captured_at_ns", "capture_id",
    "store_id", "pack_id",
)

# The catalog schema this build creates and reads.
#
# Version 4 is the FIRST version that stamps itself: ``{prefix}_schema_version``
# arrived with it. So an unstamped catalog is one of versions 1, 2 or 3, and
# the stamp cannot say which -- which is why ``_verify_schema_compatibility``
# diagnoses an unstamped catalog from what the server reports it actually
# holds, and never from the absence of the stamp alone.
#
# Version 2 appends ``(store_id, pack_id)`` to the descriptor sort key and
# moves snapshot membership from ``{prefix}_pack_commit_log`` to
# ``{prefix}_snapshot_manifest``. NEITHER change can be applied to a catalog
# that already exists: ``CREATE TABLE IF NOT EXISTS`` is a no-op against a live
# table, so the old ORDER BY survives silently, and nothing backfills the
# manifest for the packs the inventory already lists -- ``committed_pack_ids``
# would skip every one of them as already indexed, so they would never become
# members of any snapshot and every pre-existing capture would be permanently
# invisible, with no error anywhere.
#
# Version 3 adds ``publish_id`` to the watermark and the manifest, so that a
# publish can verify it OWNS the version it wrote rather than merely that a row
# for that version exists, and so that membership pairs a manifest row with the
# watermark row of the same publish. ``CREATE TABLE IF NOT EXISTS`` adds no
# column to a live table, so a version 2 catalog would keep both tables in
# their old shape.
#
# Version 4 adds ``{prefix}_publisher_lease``. The table would be created by a
# rerun of the DDL, but the compatibility check runs first and refuses a
# catalog missing one of this build's TABLES -- correctly, because it cannot
# tell an unwritten table apart from a dropped one, and one of those is the
# state that hides every capture.
#
# So the schema is checked, not migrated: ``ensure_schema`` refuses anything it
# did not create, and the catalog is rebuilt from the packs instead. That is
# affordable precisely because the catalog is derived -- see
# ``_rebuild_instruction`` and docs/capture-storage-design.md.
_SCHEMA_VERSION = 4

# Settings for the statements whose answers DECIDE something: which claimant
# owns a version, whether a publish landed, what the published head is. Each of
# those is a read-back of the reader's own write, and the sole-claimant
# protocols are only sound while a later write always observes an earlier one.
#
# A single ClickHouse node gives that for free. A ReplicatedMergeTree does not:
# a replica serves reads from whatever log entries it has fetched, so a
# read-back can miss a row another publisher has already committed and two
# claimants can both see themselves alone. ``select_sequential_consistency``
# makes the read wait for the replica to reach the latest committed log entry,
# or throw -- either is an outcome the protocol can act on, where a stale answer
# is not.
#
# Set on the reads rather than validated at construction, deliberately. There is
# nothing to validate at construction: the tables need not exist yet, an
# operator can convert them to Replicated afterwards, and a warning nobody reads
# is not enforcement. Setting it is unconditional and costs nothing on a
# non-replicated table, where the server accepts and ignores it (verified on
# 25.12).
#
# The write-side half is NOT set here and is not claimed: ClickHouse pairs
# ``select_sequential_consistency`` with ``insert_quorum`` on the writes, and a
# replicated deployment also has to make the DESCRIPTOR inserts quorum-durable
# before their watermark row, which is a deployment decision about latency, not
# something this module can pick. See docs/catalog-descriptor-key.md.
_DECIDING_READ = {"select_sequential_consistency": 1}


@dataclass(frozen=True, slots=True)
class _CatalogObject:
    """One catalog object, as ``system.tables`` describes it.

    Read rather than assumed. Every claim a schema refusal makes about a
    catalog comes from here, because a diagnosis an operator can check and find
    FALSE is worse than a vague one: they conclude the refusal is spurious,
    work around it, and land on exactly the path the refusal exists to prevent.

    ``sorting_key`` is empty for a view, which is why the kind is carried
    beside it rather than inferred from it.
    """

    engine: str
    sorting_key: str

    @property
    def kind(self) -> str:
        # Every flavour of view ends in "View" (``View``, ``MaterializedView``,
        # ``LiveView``); every table engine does not.
        return "VIEW" if self.engine.endswith("View") else "TABLE"


def _key_columns(sorting_key: str) -> tuple[str, ...]:
    """A sort key as a comparable column tuple, whatever spacing it arrived in."""
    return tuple(part.strip() for part in sorting_key.split(",") if part.strip())


class ClickHouseCatalogWriter:
    def __init__(
        self, client: ClickHouseClient, config: ClickHouseCatalogConfig | None = None
    ) -> None:
        self._client = client
        self._config = config or ClickHouseCatalogConfig()
        prefix = self._config.table_prefix
        self._capture_raw = f"{prefix}_capture_raw"
        self._capture_view = f"{prefix}_capture"
        self._pack_raw = f"{prefix}_pack_inventory_raw"
        self._pack_view = f"{prefix}_pack_inventory"
        self._watermark = f"{prefix}_index_watermark"
        self._manifest = f"{prefix}_snapshot_manifest"
        self._version_claims = f"{prefix}_capture_version_claims"
        self._lease_table = f"{prefix}_publisher_lease"
        self._schema_table = f"{prefix}_schema_version"
        # Version 1's membership table. Nothing here reads or writes it; it is
        # named so that a version 1 catalog is recognised and so the rebuild
        # instruction lists every table an operator has to drop.
        self._commit_log = f"{prefix}_pack_commit_log"
        # Everything ensure_schema owns, in drop order: views before the tables
        # they read, so replaying this list top to bottom always succeeds.
        self._objects: tuple[tuple[str, str], ...] = (
            ("VIEW", self._capture_view),
            ("VIEW", self._pack_view),
            ("TABLE", self._capture_raw),
            ("TABLE", self._pack_raw),
            ("TABLE", self._version_claims),
            ("TABLE", self._lease_table),
            ("TABLE", self._watermark),
            ("TABLE", self._manifest),
            ("TABLE", self._schema_table),
        )
        # Objects an EARLIER schema created and this build never does. Probed
        # for two reasons: one left standing after a half-finished cleanup can
        # be the ONLY thing in the database, which is a state that has to be
        # named rather than guessed at, and the rebuild instruction has to list
        # every object an operator must drop.
        self._legacy_objects: tuple[tuple[str, str], ...] = (
            ("TABLE", self._commit_log),
        )
        # The lease this writer holds, or None. Per WRITER instance, not per
        # process: two writers over one server are two publishers as far as the
        # catalog is concerned, which is exactly what the fence has to police.
        self._lease: PublisherLease | None = None

    def ensure_schema(self) -> None:
        # Before any DDL: a statement issued against an incompatible catalog is
        # either a silent no-op (CREATE ... IF NOT EXISTS) or an edit to
        # something this build does not understand.
        self._verify_schema_compatibility()
        database = _quoted(self._config.database)
        capture_raw = f"{database}.{_quoted(self._capture_raw)}"
        capture_view = f"{database}.{_quoted(self._capture_view)}"
        pack_raw = f"{database}.{_quoted(self._pack_raw)}"
        pack_view = f"{database}.{_quoted(self._pack_view)}"
        watermark = f"{database}.{_quoted(self._watermark)}"
        manifest = f"{database}.{_quoted(self._manifest)}"
        self._client.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        # First, so an install interrupted partway leaves a catalog that says
        # "this build, unfinished" rather than one that is indistinguishable
        # from version 1 and refused forever. The row that stamps it is written
        # last, once every object below exists.
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {database}.{_quoted(self._schema_table)} (
version UInt32, applied_at_ns UInt64
) ENGINE = MergeTree ORDER BY version"""
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {capture_raw} (
capture_id String, tenant_id String, experiment_id String, run_id String,
session_id String, request_id String, sequence_id String, model_id String,
model_revision String, adapter_revision Nullable(String),
capture_policy_version String, hook_name LowCardinality(String), layer_number Int32,
producer_rank UInt32, step_number UInt64, token_start UInt64, token_end UInt64,
batch_position UInt32, dtype LowCardinality(String), shape Array(UInt32),
captured_at_ns UInt64, pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), pack_record_count UInt32,
payload_offset UInt64, stored_length UInt64, decoded_length UInt64,
codec LowCardinality(String), payload_checksum FixedString(8), index_version UInt64,
{_facet_ddl()}
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY ({', '.join(_CAPTURE_TABLE_ORDER)})"""
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {pack_raw} (
pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), record_count UInt32,
index_version UInt64
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY (store_id, pack_id)"""
        )
        # Tables created by an earlier build predate the facet columns, and
        # CREATE TABLE IF NOT EXISTS will not add them. ADD COLUMN IF NOT
        # EXISTS is idempotent, so this is safe on every start.
        for name, kind, expression in _FACET_COLUMNS:
            self._client.execute(
                f"ALTER TABLE {capture_raw} ADD COLUMN IF NOT EXISTS "
                f"{name} {kind} MATERIALIZED {expression}"
            )

        # Point lookups arrive with tenant + capture_id. The primary key
        # prunes to the tenant's range, but capture_id sits behind
        # captured_at_ns in the ORDER BY and a point lookup supplies no time
        # bound, so inside a large tenant the primary index cannot narrow
        # further; the bloom filter prunes granules within that range.
        self._client.execute(
            f"ALTER TABLE {capture_raw} ADD INDEX IF NOT EXISTS "
            "capture_id_bloom capture_id TYPE bloom_filter(0.01) GRANULARITY 4"
        )
        # Parts written after ADD INDEX are indexed at insert; MATERIALIZE
        # builds it for the parts that already exist, and is a no-op once
        # they are covered, so this is safe on every start.
        self._client.execute(
            f"ALTER TABLE {capture_raw} MATERIALIZE INDEX capture_id_bloom"
        )

        # A version becomes readable only once its whole batch is durable, so
        # the watermark cannot be derived from the descriptor table: a reader
        # sampling max(index_version) there sees a version mid-batch, between
        # the INSERTs that make it up. This table is written as the last step of
        # an indexing call instead. Plain MergeTree, because it is a log --
        # ReplacingMergeTree would eventually collapse the history a pinned
        # snapshot reads.
        #
        # ``publish_id`` names the publish that wrote the row. Verifying it is
        # what turns "a row for version V exists" into "version V is MINE"; see
        # publish_snapshot.
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {watermark} (
index_version UInt64, publish_id UUID, published_at_ns UInt64,
indexed_rows UInt64, indexed_packs UInt32
) ENGINE = MergeTree ORDER BY (index_version, publish_id)"""
        )

        # The version allocator's append-only claim ledger. A claimant owns a
        # version only when its post-insert read shows it is that version's
        # sole claimant; claimed_at_ns is diagnostic only -- no wall clock
        # participates in ordering.
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {database}.{_quoted(self._version_claims)} (
version UInt64, claim_id UUID, claimed_at_ns UInt64
) ENGINE = MergeTree ORDER BY (version, claim_id)"""
        )

        # The publisher lease: who is allowed to make a snapshot visible.
        #
        # Append-only, and claimed by the same sole-claimant protocol as a
        # version -- which is SAFE here for the same reason it is safe there. A
        # lease claim row is INERT: writing one confers nothing on its own,
        # because the only thing that reads it is the fencing predicate inside
        # the publish, and that predicate names a single row. A term claimed by
        # two publishers is abandoned by both, so it holds no lease and nobody
        # publishes under it until someone claims a higher one.
        #
        # ``term`` is the monotonic slot, not the clock: the head of the table
        # is the highest term, and ties on wall-clock time cannot make the head
        # ambiguous. ``acquired_at_ns`` and ``expires_at_ns`` are stamped by the
        # SERVER, and the fence compares them against the SERVER's clock, so no
        # publisher's own clock decides whether its lease is live.
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {database}.{_quoted(self._lease_table)} (
term UInt64, lease_id UUID, holder String, acquired_at_ns UInt64,
expires_at_ns UInt64
) ENGINE = MergeTree ORDER BY (term, lease_id)"""
        )

        # The snapshot boundary. "The catalog as of W" is "the packs whose
        # membership rows were published at or before W" -- a fact about packs,
        # not about descriptor rows. Keeping it here, append-only, is what lets
        # the descriptor table stay a ReplacingMergeTree: the only rows that
        # table can now collapse are rows for one capture in one pack, and
        # those are byte identical, so it does not matter which one a merge
        # keeps.
        #
        # These rows are written by publish_snapshot, not by commit_packs. The
        # predecessor table was written before the watermark, which let a
        # slower indexer's rows land underneath a watermark a reader had
        # already pinned. A row here counts only once its version reaches the
        # watermark table (see the reader's membership clause), and a publish
        # that loses the race never writes that watermark row -- so the rows it
        # left behind are inert.
        #
        # ``publish_id`` carries the same identity the watermark row carries, so
        # membership pairs the two: a manifest row counts only when the SAME
        # publish also reached the watermark table. Keying on index_version
        # alone would make the contents of snapshot V whatever anyone wrote at
        # V, which is a weaker claim than owning V.
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {manifest} (
index_version UInt64, publish_id UUID, store_id LowCardinality(String),
pack_id UUID
) ENGINE = MergeTree ORDER BY (index_version, publish_id, store_id, pack_id)"""
        )

        capture_public = ", ".join(_CAPTURE_COLUMNS[:-1])
        pack_public = ", ".join(_PACK_COLUMNS[:-1])
        # The public descriptor view: PUBLISHED descriptor rows,
        # engine-deduplicated. Both halves are needed and neither substitutes
        # for the other.
        #
        # FINAL hides the storage engine's transient duplicates, so a replayed
        # batch reads as one row without waiting for a merge. It applies no
        # membership at all, which is why this view used to show rows from
        # batches that were written and never published, and rows orphaned by a
        # crashed indexing pass -- data every reader correctly treats as
        # nonexistent. The bound is the same membership test the reader
        # applies, so the view and the reader now agree on what exists.
        #
        # Filtering under FINAL is sound here only because (store_id, pack_id)
        # is part of the table's sort key: rows a merge may collapse into one
        # all share those columns, so the predicate keeps or drops a whole
        # group and can never delete the representative FINAL would have kept.
        # A predicate on index_version has no such guarantee -- FINAL collapses
        # to the highest version and only then filters -- which is why
        # ``FINAL ... WHERE index_version <= W`` is not a snapshot.
        #
        # This deliberately does NOT group on capture identity. One row per
        # (capture, store, pack): a capture described by two packs legitimately
        # appears twice, and choosing between them is supersession, which
        # belongs to the reader -- ONE argMax over a tuple of every resolved
        # column, grouped on capture identity and ordered on
        # ``clickhouse_reader._RESOLUTION_ORDER``, which is
        # ``(index_version, store_id, pack_id)`` and not index_version alone.
        # An argMax here would be a second copy of those semantics living in
        # SQL, free to drift away from the reader's silently.
        #
        # CREATE OR REPLACE rather than IF NOT EXISTS: a catalog created by an
        # earlier build already holds the unbounded view, and IF NOT EXISTS
        # would leave it serving unpublished rows forever.
        #
        # The statement reads the manifest and the watermark, which is why both
        # tables are created above it. Reordered, ensure_schema breaks on a
        # fresh server only -- a rerun finds them already there.
        #
        # The bound is the publish PAIR, matching the reader's membership
        # clause: a manifest row counts once the publish that wrote it also
        # reached the watermark table. Every row in the watermark table is at or
        # below the published head by definition, so the pair test subsumes the
        # ``index_version <= max`` bound this used to carry.
        self._client.execute(
            f"CREATE OR REPLACE VIEW {capture_view} AS "
            f"SELECT {capture_public} FROM {capture_raw} FINAL "
            "WHERE (store_id, pack_id) IN ("
            f"SELECT store_id, pack_id FROM {manifest} "
            "WHERE (index_version, publish_id) IN "
            f"(SELECT index_version, publish_id FROM {watermark})"
            ")"
        )
        # The inventory needs no such bound. CatalogIndexer.index writes it
        # only after a successful publish -- descriptors, publish, inventory --
        # so a pack reaches this table only once it is already published.
        self._client.execute(
            f"CREATE VIEW IF NOT EXISTS {pack_view} AS "
            f"SELECT {pack_public} FROM {pack_raw} FINAL"
        )
        # Last, and conditional: the stamp means "every object above exists",
        # so an install that died partway is never mistaken for a finished one.
        # The condition is server-side, which is what makes a rerun idempotent
        # without a read-then-write window.
        self._client.execute(
            f"INSERT INTO {database}.{_quoted(self._schema_table)} "
            "(version, applied_at_ns) "
            "SELECT toUInt32(%(version)s), toUInt64(%(applied_at_ns)s) "
            "FROM system.one WHERE (SELECT count() FROM "
            f"{database}.{_quoted(self._schema_table)}) = 0",
            {"version": _SCHEMA_VERSION, "applied_at_ns": time_ns()},
        )

    def _verify_schema_compatibility(self) -> None:
        """Refuse a catalog this build cannot read, instead of half-upgrading it.

        Every failure here is a state where carrying on would be silent. The
        DDL below cannot repair any of them: ``CREATE TABLE IF NOT EXISTS`` is
        a no-op against a live table, so it leaves an older table exactly as it
        found it, and where a table is missing it creates an empty one beside
        rows that assume it is full. Either way the run succeeds and the next
        indexing pass skips every pack the inventory still lists, leaving those
        captures durable in object storage and invisible to every reader with
        no error anywhere. Refusing is the only outcome an operator can act on.

        A missing VIEW is the exception, and the only one: it holds no rows,
        the DDL below recreates it outright, and a recreated projection cannot
        disagree with data that survived. Everything else is refused, and the
        refusal describes the catalog the server actually reports rather than
        the one an absent table suggests -- see ``_unstamped_diagnosis``.

        An EMPTY stamp table is not an exception either. It narrows what the
        version means -- an install of this build that died before stamping,
        rather than a version this build cannot read -- and nothing more: the
        checks for a missing table and for an inventory without membership
        still have to run, because both describe states no rerun of the DDL
        repairs.
        """
        found = self._catalog_state()
        if not found:
            return  # A fresh install: nothing of ours is there to be wrong.
        self._reject_wrong_kinds(found)
        if not any(name in found for _, name in self._objects):
            raise CatalogSchemaVersionError(self._leftovers_only(found))
        if self._schema_table not in found:
            raise CatalogSchemaVersionError(self._unstamped_diagnosis(found))
        recorded = self._recorded_schema_version()
        if recorded is not None and recorded != _SCHEMA_VERSION:
            raise CatalogSchemaVersionError(
                f"catalog `{self._config.database}`.`{self._config.table_prefix}_*` "
                f"is at schema version {recorded} and this build reads version "
                f"{_SCHEMA_VERSION}. A higher version means a newer writer owns "
                "this catalog: upgrade this build rather than writing to it. A "
                "lower one is not upgraded in place. Between them the versions "
                "change the descriptor sort key, the membership table, the "
                "publish identity columns and the set of tables, and CREATE "
                "TABLE IF NOT EXISTS alters neither an existing table's columns "
                "nor its sort key. "
                + self._rebuild_instruction()
            )
        # The two checks below run whether or not the stamp table holds a row.
        # An EMPTY stamp table means an install of this build died between
        # creating the objects and recording the version, and re-running the
        # DDL is the right recovery -- but only over a catalog that is
        # otherwise whole, which is what these two decide.
        #
        # Returning early on the empty stamp skipped both of them, and clearing
        # the stamp is the obvious operator workaround for a refusal. Measured:
        # truncating it on a version 3 catalog made ensure_schema ACCEPT a
        # catalog with no `{prefix}_publisher_lease` and re-stamp it as 4 --
        # the in-place upgrade this design says is never performed. On a
        # version 2 catalog the DDL then died mid-way with `Code: 47, Unknown
        # expression or function identifier 'publish_id'`, leaving the catalog
        # half written: exactly the outcome ``_reject_wrong_kinds`` and this
        # whole method exist to prevent. And it made an inventory beside an
        # empty manifest -- the state that hides every capture -- accepted.
        stamp = (
            f"is stamped schema version {_SCHEMA_VERSION}"
            if recorded is not None
            else f"holds `{self._schema_table}` with no row in it (an install "
            "of this build that died before stamping)"
        )
        # TABLES only. A view holds no rows: it is a projection of the tables
        # below it, ``ensure_schema`` recreates it unconditionally
        # (CREATE OR REPLACE / CREATE IF NOT EXISTS), and recreating it cannot
        # disagree with anything that survived. Demanding a whole data rebuild
        # because somebody dropped a view is a cost with no risk behind it.
        # A missing TABLE is the opposite: recreating it EMPTY beside tables
        # that kept their rows is the state that hides every capture.
        missing = [
            name
            for kind, name in self._objects
            if kind == "TABLE" and name not in found
        ]
        if missing:
            raise CatalogSchemaVersionError(
                f"catalog `{self._config.database}`.`{self._config.table_prefix}_*` "
                f"{stamp} but is missing "
                + ", ".join(f"`{name}`" for name in missing)
                + ". Recreating those empty beside tables that kept their rows "
                "is not a repair, it is the dangerous state: a surviving pack "
                "inventory makes the next pass skip every pack it lists, so "
                "nothing refills what was dropped and those captures stay "
                "durable in object storage and invisible to every reader. "
                + self._rebuild_instruction()
            )
        if self._inventory_without_membership():
            raise CatalogSchemaVersionError(
                f"catalog `{self._config.database}`.`{self._config.table_prefix}_*` "
                f"lists packs in `{self._pack_raw}` while `{self._manifest}` is "
                "empty, so every pack is already marked indexed and none belongs "
                "to any snapshot. An indexing pass would skip all of them and "
                "leave a catalog that is empty but reports success. "
                + self._rebuild_instruction()
            )

    def _catalog_state(self) -> dict[str, _CatalogObject]:
        """What this catalog IS, as the server describes it.

        Engine and sort key come back with the names, in one statement, because
        every refusal below has to describe the catalog it actually met rather
        than the one the absence of a single table suggests.

        Objects of superseded schema versions are probed too: one left standing
        after a half-finished manual cleanup is often the only thing in the
        database, and a catalog with it still there is not one to create tables
        next to.
        """
        names = [name for _, name in self._objects + self._legacy_objects]
        rows = self._client.execute(
            "SELECT name, engine, sorting_key FROM system.tables "
            "WHERE database = %(database)s AND name IN %(names)s",
            {"database": self._config.database, "names": names},
        )
        wanted = set(names)
        found = {}
        for name, engine, sorting_key in rows:
            name = self._text(name)
            if name in wanted:
                found[name] = _CatalogObject(
                    engine=self._text(engine), sorting_key=self._text(sorting_key)
                )
        return found

    def _reject_wrong_kinds(self, found: dict[str, _CatalogObject]) -> None:
        """Refuse an object that is there but is not the kind it should be.

        ``CREATE OR REPLACE VIEW`` cannot replace a table, and a table this
        build would write to cannot be a view, so either mismatch fails
        somewhere further in with a ClickHouse error about the wrong object
        kind. Naming it here costs nothing -- the engine came back with the
        name -- and turns that into one sentence saying which object and what
        it is.
        """
        wrong = [
            f"`{name}` is a {found[name].kind} (engine {found[name].engine}) "
            f"where this build creates a {kind}"
            for kind, name in self._objects
            if name in found and found[name].kind != kind
        ]
        if not wrong:
            return
        raise CatalogSchemaVersionError(
            f"catalog `{self._config.database}`.`{self._config.table_prefix}_*` "
            "holds an object of the wrong kind: "
            + ", ".join(wrong)
            + ". Nothing this build issues converts one into the other, so the "
            "DDL below would fail partway and leave the catalog half written. "
            + self._rebuild_instruction()
        )

    def _leftovers_only(self, found: dict[str, _CatalogObject]) -> str:
        """Nothing of this build's is here, but the database is not empty.

        A drop that misses one superseded table wedges every start after it:
        the catalog is not empty, so it is not a fresh install; it has no
        version stamp, so it reads as an old schema; and the refusal then
        prescribes the rebuild the operator has just finished, over tables that
        no longer exist. Naming the one inert object and saying to drop it is
        the whole recovery.
        """
        leftovers = [name for _, name in self._legacy_objects if name in found]
        listed = ", ".join(
            f"`{self._config.database}`.`{name}`" for name in leftovers
        )
        subject = (
            f"the only object present is {listed}"
            if len(leftovers) == 1
            else f"the only objects present are {listed}"
        )
        it = "it" if len(leftovers) == 1 else "them"
        return (
            f"catalog `{self._config.database}`.`{self._config.table_prefix}_*` "
            f"holds none of the objects this build creates: {subject}, which no "
            "schema this build knows how to write ever creates or reads -- a "
            "superseded table a drop missed. It is why this start is not "
            "treated as a fresh install, and it is the only thing standing in "
            f"the way. Drop {it} and run ensure_schema() again. There is "
            "nothing to rebuild first: no descriptor, membership, inventory or "
            "watermark table of this catalog survives, so the next start "
            "creates the current schema from nothing."
        )

    def _unstamped_diagnosis(self, found: dict[str, _CatalogObject]) -> str:
        """Describe the differences THIS catalog actually has, and refuse.

        The stamp only says "version 4 or later"; every earlier version left no
        stamp at all, so its absence identifies a range, not a version. Reading
        the range off as "this is version 1" and then reciting version 1's
        differences is how a refusal comes to make claims an operator can check
        and find false -- and an operator who checks two named facts, finds both
        untrue and concludes the guard is broken goes around it, onto the path
        the guard exists to prevent.

        So every line below is read from ``system.tables`` /
        ``system.columns`` for this catalog. Where a difference is already
        absent it says so, because "this is NOT what is wrong" is information
        too.
        """
        prefix = self._config.table_prefix
        findings = []

        required_key = ", ".join(_CAPTURE_TABLE_ORDER)
        descriptor = found.get(self._capture_raw)
        if descriptor is None:
            findings.append(
                f"`{self._capture_raw}` is absent, so this catalog holds no "
                "descriptor rows at all"
            )
        elif _key_columns(descriptor.sorting_key) != _key_columns(required_key):
            findings.append(
                f"`{self._capture_raw}` is sorted on "
                f"({descriptor.sorting_key}) and this build requires "
                f"({required_key}). CREATE TABLE IF NOT EXISTS cannot alter an "
                "existing ORDER BY, so the old key would survive silently and "
                "a merge could still delete one of two rows describing one "
                "capture"
            )
        else:
            findings.append(
                f"`{self._capture_raw}` is already sorted on ({required_key}), "
                "the key this build requires -- so the descriptor sort key is "
                "NOT what is wrong with this catalog"
            )

        commit_log = self._commit_log in found
        manifest = self._manifest in found
        if commit_log and not manifest:
            findings.append(
                f"snapshot membership is in `{self._commit_log}`, which this "
                f"build never reads; it moved to `{self._manifest}`, which is "
                "absent, and nothing backfills it -- `committed_pack_ids` "
                f"reads `{self._pack_raw}`, so every pack already there would "
                "be skipped as indexed and would never become a member of any "
                "snapshot"
            )
        elif manifest and not commit_log:
            findings.append(
                f"snapshot membership is already in `{self._manifest}`, where "
                f"this build reads it, and `{self._commit_log}` is absent -- so "
                "there is NO commit-log migration owed here"
            )
        elif manifest and commit_log:
            findings.append(
                f"both membership tables are present: `{self._commit_log}`, "
                f"which this build never reads, and `{self._manifest}`, which "
                "it does -- so which packs are members depends on which table "
                "the rows were written to"
            )
        else:
            findings.append(
                f"neither `{self._commit_log}` nor `{self._manifest}` is "
                "present, so nothing in this catalog records which packs "
                "belong to a snapshot"
            )

        identity_tables = [
            name for name in (self._watermark, self._manifest) if name in found
        ]
        if identity_tables:
            without = sorted(
                set(identity_tables) - self._tables_with_publish_id(identity_tables)
            )
            if without:
                findings.append(
                    ", ".join(f"`{name}`" for name in without)
                    + f" {'has' if len(without) == 1 else 'have'} no "
                    "`publish_id` column, so a manifest row cannot be paired "
                    "with the watermark row of the SAME publish; CREATE TABLE "
                    "IF NOT EXISTS adds no column to a live table"
                )
            else:
                findings.append(
                    ", ".join(f"`{name}`" for name in identity_tables)
                    + " already carry `publish_id`"
                )

        absent = [name for _, name in self._objects if name not in found]
        if absent:
            # ``absent`` is what is NOT here, so both branches have to say so.
            # The singular one read "..., which is present" and fired in the
            # single-difference case, which is the one an operator checks.
            findings.append(
                "this build also creates "
                + ", ".join(f"`{name}`" for name in absent)
                + f", {'which is not' if len(absent) == 1 else 'none of which are'} "
                "present"
            )

        return (
            f"catalog `{self._config.database}`.`{prefix}_*` carries no schema "
            f"stamp and this build requires version {_SCHEMA_VERSION}. Version "
            f"{_SCHEMA_VERSION} is the first to write `{self._schema_table}`, "
            "so an unstamped catalog is one of the versions before it and the "
            "stamp cannot say which. What the server reports about THIS "
            "catalog:\n"
            + "".join(f"  - {finding}\n" for finding in findings)
            + "It is refused whichever version it is, and deliberately: the "
            "probes above compare object names, one sort key and one column. "
            "They do not compare column types, view definitions, codecs or "
            "skip indices, so this build cannot know that the differences "
            "listed are all of them. "
            + self._rebuild_instruction()
        )

    def _tables_with_publish_id(self, tables: list[str]) -> set[str]:
        """Which of `tables` carry the publish-identity column.

        Read from the server rather than inferred from the schema version,
        because the version is exactly what an unstamped catalog will not say.
        """
        rows = self._client.execute(
            "SELECT table FROM system.columns WHERE database = %(database)s "
            "AND table IN %(tables)s AND name = 'publish_id'",
            {"database": self._config.database, "tables": tables},
        )
        return {self._text(row[0]) for row in rows}

    def _recorded_schema_version(self) -> int | None:
        """The stamped version, or None when the table holds no row yet."""
        rows = self._client.execute(
            f"SELECT version FROM {self._qualified(self._schema_table)} "
            "ORDER BY version DESC LIMIT 1"
        )
        return rows[0][0] if rows else None

    def _inventory_without_membership(self) -> bool:
        """Is the replay guard populated while snapshot membership is empty?

        That pair is the signature of a catalog whose membership was dropped or
        never migrated: ``committed_pack_ids`` reads the inventory and reports
        every pack as done, while readers bound their snapshot by a manifest
        that names none of them.
        """
        packs, members = self._client.execute(
            f"SELECT (SELECT count() FROM {self._qualified(self._pack_raw)}), "
            f"(SELECT count() FROM {self._qualified(self._manifest)})"
        )[0]
        return bool(packs) and not members

    def _rebuild_instruction(self) -> str:
        """The one supported recovery, named in full so nobody has to guess.

        Generated from ``self._objects`` rather than written out, so a table
        added to this build cannot be left out of the drop list. This branch
        leaked tables twice from exactly that omission, and the docs procedure
        is held to the same list by a test.
        """
        database = self._config.database
        objects = ", ".join(
            f"`{database}`.`{name}`"
            for _, name in self._objects + self._legacy_objects
        )
        return (
            "The catalog is a derived projection over immutable packs, so "
            "rebuilding it loses nothing: stop every indexer, drop ALL of its "
            f"objects in this order (views first) -- {objects} -- then run "
            "ensure_schema() and CatalogReconciler.rebuild() over each pack "
            "store. Dropping the pack inventory is mandatory, not optional: "
            "committed_pack_ids reads it to skip replays, so an inventory left "
            "behind makes the rebuild skip every pack it exists to re-read and "
            "publish an empty catalog without failing. See "
            "docs/capture-storage-design.md, 'Catalog schema versions and "
            "rebuild'."
        )

    def committed_pack_ids(
        self, identities: Sequence[PackIdentity]
    ) -> set[PackIdentity]:
        if not identities:
            return set()
        if len(identities) > self._config.query_pack_limit:
            raise ValueError("pack identity query exceeds query_pack_limit")
        table = self._qualified(self._pack_view)
        rows = self._client.execute(
            f"SELECT store_id, toString(pack_id) FROM {table} "
            "WHERE (store_id, pack_id) IN %(identities)s",
            {"identities": list(identities)},
        )
        return {(self._text(row[0]), self._text(row[1])) for row in rows}

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

        The lease is renewed first, every time. That costs three round trips --
        head read, claim INSERT, read-back, 5.69 ms median on 25.12 -- and buys
        the whole safety margin: at the moment the fence is evaluated the lease
        has essentially a full ``lease_ttl_ns`` left, and a takeover cannot
        happen before it expires. Each publish statement is capped at
        ``publish_timeout_ns`` (converted to the SECONDS ``max_execution_time``
        takes), which is required to be below the TTL, so for both publishers
        to make a snapshot visible, the server would have to keep one INSERT in
        flight past its own execution-time check and on past the expiry of a
        lease renewed just before it started. The cap is per statement, so it
        does not bound the gap between the two. See
        docs/catalog-descriptor-key.md for what that does and does not close.
        """
        self._validate_version(index_version)
        self._validate_version(published_at_ns)
        lease = self.renew_publisher_lease()
        watermark = self._qualified(self._watermark)
        # One identity per ATTEMPT, not per allocated version: what has to be
        # verified is that this particular statement's row is the one standing
        # at V, and a second attempt at one version is a different write. Reusing
        # the allocator's claim_id would make a duplicate publish at V read back
        # as its own, which is the failure this identity exists to catch.
        publish_id = str(uuid4())
        settings = {**_DECIDING_READ, **self._publish_timeout()}
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
                    "members": [(ref.store_id, ref.pack_id) for ref in refs],
                    "lease_id": lease.lease_id,
                },
                settings=settings,
            )
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
            f"WHERE (SELECT max(index_version) FROM {watermark}) "
            "< %(index_version)s "
            f"AND {self._lease_fence()}",
            {
                "index_version": index_version,
                "publish_id": publish_id,
                "published_at_ns": published_at_ns,
                "indexed_rows": indexed_rows,
                "indexed_packs": indexed_packs,
                "lease_id": lease.lease_id,
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
        """The lease this writer holds, or None."""
        return self._lease

    def acquire_publisher_lease(self, holder: str) -> PublisherLease:
        """Take the publisher lease, or say who has it.

        Sole-claimant append-and-read-back over ``{prefix}_publisher_lease``,
        the same protocol the version allocator uses, and safe here for the
        same reason: a lease claim row is INERT. Writing one confers nothing --
        the only thing that reads the table is the fencing predicate inside a
        publish, and that predicate names a single row -- so a term claimed by
        two publishers is abandoned by both and holds no lease at all.

        A lease can be taken over once it has expired by the SERVER's clock, or
        immediately if the head term is contested, since a contested term is
        inert. A live lease held by SOMEBODY ELSE raises
        :class:`PublisherLeaseHeldError` rather than being stolen.

        What that does NOT promise is that only one writer object believes it
        holds a lease. The read-back proves a claimant is alone at ITS term, not
        that its term is the head: a claimant whose head read preceded a rival's
        row claims below that rival, reads back alone, and is handed a live
        lease while the rival sits above it. Verified on 25.12 with two writers
        holding terms 1 and 9. Only the head publishes -- the fence says so, and
        the loser is refused at its next renewal with a message naming the
        actual holder -- so this costs liveness on the loser, never safety.

        A writer that already holds a lease is re-acquiring its OWN, and that
        must succeed: the operating instructions say to acquire before
        publishing, so "acquire twice" is the documented path after a restart
        of anything above this object. Minting a fresh ``lease_id`` here would
        make this writer a stranger to the row it already owns -- the claim
        would be refused as held by itself, and there would be no API left to
        release the row it just orphaned. So a held lease keeps its fencing
        identity and the claim below refreshes it at a new term, exactly as a
        renewal does.
        """
        if not isinstance(holder, str) or not 0 < len(holder) <= 256:
            raise ValueError("holder must be a non-empty string of at most 256 bytes")
        held = self._lease
        return self._claim_lease(
            holder, lease_id=held.lease_id if held is not None else str(uuid4())
        )

    def renew_publisher_lease(self) -> PublisherLease:
        """Extend the held lease, keeping its fencing identity.

        A renewal is a fresh term carrying the same ``lease_id``, claimed by the
        same sole-claimant protocol, so a takeover racing the renewal contests
        that term and BOTH sides abandon it: the head is then inert, nobody can
        publish under it, and whoever wants the lease has to claim a higher
        term. Losing a renewal therefore raises rather than leaving this writer
        believing it still holds something.
        """
        lease = self._lease
        if lease is None:
            raise PublisherLeaseError(
                "no publisher lease is held; call acquire_publisher_lease() "
                "before publishing. Only the lease holder can make a snapshot "
                "visible, and the check rides inside the publish statement, so "
                "publishing without one writes nothing."
            )
        return self._claim_lease(lease.holder, lease_id=lease.lease_id)

    def release_publisher_lease(self) -> None:
        """Give back the lease THIS writer holds, and only that one.

        A tombstone: a fresh term with the same ``lease_id`` that is already
        expired when it lands. No read-back, because a release wins nothing --
        if a takeover contests the term, the head is inert and the lease is
        gone either way, which is the outcome a release wanted. Idempotent, and
        a no-op when no lease is held.

        **Fenced on the head being this writer's lease.** The tombstone lands
        at ``head.term + 1``, so it becomes the head whatever was there before:
        released blind, a writer whose lease lapsed long ago revokes whoever
        holds the catalog now, simply by shutting down in an orderly way. The
        successor is not told -- its next publish is fenced out, and any third
        publisher can take the lease off it at once, because the head the
        fence resolves to is now an expired row belonging to neither.

        The local state is dropped only after the tombstone is durable. The
        other order loses the lease locally while it stays live on the server:
        a failed INSERT would leave nothing able to release it and nothing able
        to publish under it until it expired.
        """
        lease = self._lease
        if lease is None:
            return
        head = self._lease_head()
        if head.lease_id != lease.lease_id:
            # Not ours any more. Somebody else's row is what the fence
            # resolves to, so there is nothing here to give back -- and
            # tombstoning above it would take THEIR lease away.
            self._lease = None
            return
        self._insert_lease_row(
            term=head.term + 1, lease_id=lease.lease_id, holder=lease.holder,
            ttl_ns=0,
        )
        self._lease = None

    def _claim_lease(self, holder: str, *, lease_id: str) -> PublisherLease:
        table = self._qualified(self._lease_table)
        attempts = self._config.lease_attempts
        for attempt in range(attempts):
            head = self._lease_head()
            if (
                head.claimants == 1
                and head.expires_at_ns > head.now_ns
                and head.lease_id != lease_id
            ):
                # A live lease that is not this writer's. Whatever it thought
                # it held is not the row the fence resolves to, so saying so is
                # the honest answer -- and note what justifies dropping the
                # local lease: the HEAD proves it is gone, not the mere fact
                # that a claim did not go through.
                self._lease = None
                raise PublisherLeaseHeldError(
                    f"publisher lease on `{self._config.database}`."
                    f"`{self._config.table_prefix}_*` is held by "
                    f"{head.holder!r} (lease {head.lease_id}, term {head.term}) "
                    f"for another {head.expires_at_ns - head.now_ns} ns. Only "
                    "one publisher may make snapshots visible; wait for it to "
                    "expire or stop it."
                )
            # A randomized skip after a collision, exactly as the version
            # allocator does it and for the same reason: every contender that
            # abandons a contested term recomputes ``head.term + 1`` and
            # collides on the next one too. Without the skip, six publishers
            # taking a cold lease failed ~48% of the time with "every term was
            # contested" -- a liveness failure invented entirely by the retry.
            term = head.term + 1 + (
                secrets.randbelow(8 * attempt + 1) if attempt else 0
            )
            self._validate_version(term)
            self._insert_lease_row(
                term=term, lease_id=lease_id, holder=holder,
                ttl_ns=self._config.lease_ttl_ns,
            )
            rows = self._client.execute(
                f"SELECT toString(lease_id), acquired_at_ns, expires_at_ns "
                f"FROM {table} WHERE term = %(term)s",
                {"term": term},
                settings=_DECIDING_READ,
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
            # Contested: abandon this term entirely and claim above it. Nobody
            # holds it, so nobody publishes under it.
        self._lease = None
        raise PublisherLeaseError(
            f"could not claim the publisher lease after {attempts} attempts; "
            "every term was contested"
        )

    def _insert_lease_row(
        self, *, term: int, lease_id: str, holder: str, ttl_ns: int
    ) -> None:
        # Both timestamps come from the SERVER, in one statement, so that no
        # publisher's wall clock -- or the skew between two of them -- decides
        # when a lease expires. The fence compares against the same clock.
        self._client.execute(
            f"INSERT INTO {self._qualified(self._lease_table)} "
            "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
            "SELECT toUInt64(%(term)s), toUUID(%(lease_id)s), %(holder)s, "
            "now_ns, now_ns + toUInt64(%(ttl_ns)s) "
            "FROM (SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)",
            {"term": term, "lease_id": lease_id, "holder": holder, "ttl_ns": ttl_ns},
        )

    def _lease_head(self) -> _LeaseHead:
        """The top of the lease table, in the order the fence resolves it.

        Two rows, one scan, in exactly the ``term DESC, lease_id DESC`` order
        the fencing predicate uses -- so the first row IS the row the fence will
        match, rather than something merely believed to be equivalent. The
        second row only has to answer "is the head term contested?", which is
        why two are enough however many claimants there are.

        The server clock rides on the same statement, so it and the expiry it
        is compared against cannot drift apart between two round trips.

        Phrased as five scalar subqueries over ``max(term)`` this was the single
        most expensive statement in a publish -- 7.8 ms of a 22 ms publish on
        25.12, because each subquery re-scanned for the head. One ordered read
        of two rows costs 1.4 ms.
        """
        rows = self._client.execute(
            "SELECT term, toString(lease_id), holder, expires_at_ns, "
            "toUnixTimestamp64Nano(now64(9)) FROM "
            f"{self._qualified(self._lease_table)} "
            "ORDER BY term DESC, lease_id DESC LIMIT 2",
            settings=_DECIDING_READ,
        )
        if not rows:
            return _LeaseHead(
                term=0, claimants=0, lease_id="", holder="", expires_at_ns=0,
                now_ns=0,
            )
        term, lease_id, holder, expires_at_ns, now_ns = rows[0]
        return _LeaseHead(
            term=term,
            claimants=1 + sum(1 for row in rows[1:] if row[0] == term),
            lease_id=self._text(lease_id),
            holder=self._text(holder),
            expires_at_ns=expires_at_ns,
            now_ns=now_ns,
        )

    def _lease_fence(self) -> str:
        """The fencing predicate, for use INSIDE a writing statement.

        Two conditions on ONE row: the head of the lease table is this
        publisher's lease, and it has not expired by the server's own clock.

        **One subquery reading one row**, not two subqueries returning a column
        each, and that is a correctness point before it is a cost one. Two
        scalar subqueries are two reads: a takeover landing between them could
        be answered with the OLD holder's ``lease_id`` and the NEW holder's
        ``expires_at_ns``, and the fence would pass for a publisher that had
        already been replaced. Here the ordered ``LIMIT 1`` resolves the head
        once and both conditions are asked of that one row. It is also cheaper
        -- measured on 25.12, the conditional publish costs 3.06 ms unfenced
        and 3.85 ms with a fence, against 4.84 ms for the two-subquery form.

        ``count() ... = 1`` rather than comparing a tuple, because an empty
        lease table has to make this FALSE and a tuple cannot. A scalar
        subquery selecting a tuple from no rows is not NULL on 25.12; it is
        ``Code: 125, Scalar subquery returned empty result of type
        Tuple(UUID, UInt8) which cannot be Nullable`` -- a raw ServerException,
        not a CaptureStorageError, thrown before the write is even considered.
        ``count()`` over the same one-row read always returns a row, so an
        empty table answers 0 and a publisher that never took a lease writes
        nothing, which is what this predicate is for.

        What it does NOT do is resolve a CONTESTED head term. The subquery
        names one row, so the higher ``lease_id`` of two claimants at the top
        term satisfies it. Safety there comes from ``_claim_lease``: both
        claimants see the contest in their read-back and neither returns a
        lease, so neither is holding the ``lease_id`` the fence would accept.
        """
        return (
            "(SELECT count() FROM ("
            "SELECT lease_id, expires_at_ns FROM "
            f"{self._qualified(self._lease_table)} "
            "ORDER BY term DESC, lease_id DESC LIMIT 1"
            ") WHERE lease_id = toUUID(%(lease_id)s) "
            "AND expires_at_ns > toUnixTimestamp64Nano(now64(9))) = 1"
        )

    def _publish_timeout(self) -> dict[str, object]:
        """Cap the publish statement below the lease TTL.

        The fence is evaluated when the statement starts and the row lands when
        it finishes, so the two are only separable by however long the server
        keeps the statement in flight. Capping that below ``lease_ttl_ns`` is
        what turns "unlikely" into "the server would have to overrun its own
        execution-time check": a takeover cannot happen until the lease renewed
        immediately before this statement expires.
        """
        return {
            "max_execution_time": self._config.publish_timeout_ns / 1_000_000_000,
            "timeout_overflow_mode": "throw",
        }

    def _reject_if_the_lease_is_gone(self, lease: PublisherLease) -> None:
        head = self._lease_head()
        if head.lease_id == lease.lease_id and head.expires_at_ns > head.now_ns:
            return
        self._lease = None
        raise PublisherLeaseError(
            f"publisher lease {lease.lease_id} (term {lease.term}) no longer "
            f"stands at the head of `{self._config.database}`."
            f"`{self._config.table_prefix}_publisher_lease`, which is now term "
            f"{head.term} held by {head.holder!r}: the publish was fenced out "
            "and made no snapshot visible. Acquire a lease again and re-index; "
            "retrying with a higher version would fail the same fence. If the "
            "takeover landed between the two publish statements, the manifest "
            "rows of the first are still there -- inert, since no watermark "
            "row will ever pair with them, and collectable by a retention job."
        )

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
        raise RuntimeError(
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

    @staticmethod
    def _validate_version(index_version: int) -> None:
        if type(index_version) is not int or not 0 <= index_version < 2**64:
            raise ValueError("index_version must fit UInt64")

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if not isinstance(value, str):
            raise ValueError("ClickHouse returned a non-text identifier")
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
