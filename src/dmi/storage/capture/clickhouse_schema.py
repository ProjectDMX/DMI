from __future__ import annotations

from dataclasses import dataclass
from time import time_ns

from .catalog import CatalogSchemaVersionError
from .clickhouse_sql import (
    DECIDING_READ,
    ClickHouseClient,
    membership_predicate,
    text,
)


CAPTURE_COLUMNS = (
    "capture_id",
    "tenant_id",
    "experiment_id",
    "run_id",
    "session_id",
    "request_id",
    "sequence_id",
    "model_id",
    "model_revision",
    "adapter_revision",
    "capture_policy_version",
    "hook_name",
    "layer_number",
    "producer_rank",
    "step_number",
    "token_start",
    "token_end",
    "batch_position",
    "dtype",
    "shape",
    "captured_at_ns",
    "pack_id",
    "store_id",
    "object_key",
    "object_bytes",
    "pack_checksum",
    "pack_record_count",
    "payload_offset",
    "stored_length",
    "decoded_length",
    "codec",
    "payload_checksum",
    "index_version",
)

FACET_COLUMNS = (
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

PACK_COLUMNS = (
    "pack_id",
    "store_id",
    "object_key",
    "object_bytes",
    "pack_checksum",
    "record_count",
    "index_version",
)

CAPTURE_TABLE_ORDER = (
    "tenant_id",
    "experiment_id",
    "run_id",
    "captured_at_ns",
    "capture_id",
    "store_id",
    "pack_id",
)

SCHEMA_VERSION = 4


def _quoted(value: str) -> str:
    return f"`{value}`"


def _facet_ddl() -> str:
    return ",\n".join(
        f"{name} {kind} MATERIALIZED {expression}"
        for name, kind, expression in FACET_COLUMNS
    )


def _key_columns(sorting_key: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in sorting_key.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class CatalogObject:
    engine: str
    sorting_key: str

    @property
    def kind(self) -> str:
        return "VIEW" if self.engine.endswith("View") else "TABLE"


class ClickHouseCatalogSchema:
    def __init__(self, client: ClickHouseClient, database: str, prefix: str) -> None:
        self._client = client
        self.database = database
        self.prefix = prefix
        self.capture_raw = f"{prefix}_capture_raw"
        self.capture_view = f"{prefix}_capture"
        self.pack_raw = f"{prefix}_pack_inventory_raw"
        self.pack_view = f"{prefix}_pack_inventory"
        self.watermark = f"{prefix}_index_watermark"
        self.manifest = f"{prefix}_snapshot_manifest"
        self.version_claims = f"{prefix}_capture_version_claims"
        self.lease_table = f"{prefix}_publisher_lease"
        self.schema_table = f"{prefix}_schema_version"
        self.commit_log = f"{prefix}_pack_commit_log"
        self.objects: tuple[tuple[str, str], ...] = (
            ("VIEW", self.capture_view),
            ("VIEW", self.pack_view),
            ("TABLE", self.capture_raw),
            ("TABLE", self.pack_raw),
            ("TABLE", self.version_claims),
            ("TABLE", self.lease_table),
            ("TABLE", self.watermark),
            ("TABLE", self.manifest),
            ("TABLE", self.schema_table),
        )
        self.legacy_objects: tuple[tuple[str, str], ...] = (("TABLE", self.commit_log),)

    def qualified(self, table: str) -> str:
        return f"{_quoted(self.database)}.{_quoted(table)}"

    def ensure(self) -> None:
        found = self._catalog_state()
        # Visibility is required before any VERDICT, not merely before the DDL.
        # `system.tables` is grant-filtered per role, so an object this role
        # holds no privilege on is absent from the state above -- there,
        # indistinguishable from one that was dropped -- and every refusal
        # below is read off that state. Asked afterwards, this check was
        # unreachable for exactly the catalog it exists for: a healthy stamped
        # catalog missing one GRANT was refused as "missing `X` ... drop ALL of
        # its objects", prescribing the teardown of a catalog whose only fault
        # was an ungranted object (measured on 25.12). Reading the state first
        # is safe because reading it refuses nothing, and `CHECK GRANT` names
        # an object rather than resolving one, so it needs neither the database
        # nor the objects to exist (also verified on 25.12).
        self._require_catalog_visibility()
        fresh = self._verify_compatibility(found)
        database = _quoted(self.database)
        capture_raw = self.qualified(self.capture_raw)
        capture_view = self.qualified(self.capture_view)
        pack_raw = self.qualified(self.pack_raw)
        pack_view = self.qualified(self.pack_view)
        watermark = self.qualified(self.watermark)
        manifest = self.qualified(self.manifest)
        self._client.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        if fresh:
            # Re-read what looked like nothing at all, now that the database
            # exists: a second installer that created objects between the two
            # reads is met with the same refusals as any other catalog rather
            # than with this build's DDL running over its work.
            self._verify_compatibility()
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.qualified(self.schema_table)} (
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
ORDER BY ({", ".join(CAPTURE_TABLE_ORDER)})"""
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {pack_raw} (
pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), record_count UInt32,
index_version UInt64
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY (store_id, pack_id)"""
        )
        for name, kind, expression in FACET_COLUMNS:
            self._client.execute(
                f"ALTER TABLE {capture_raw} ADD COLUMN IF NOT EXISTS "
                f"{name} {kind} MATERIALIZED {expression}"
            )
        self._client.execute(
            f"ALTER TABLE {capture_raw} ADD INDEX IF NOT EXISTS "
            "capture_id_bloom capture_id TYPE bloom_filter(0.01) GRANULARITY 4"
        )
        self._client.execute(
            f"ALTER TABLE {capture_raw} MATERIALIZE INDEX capture_id_bloom"
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {watermark} (
index_version UInt64, publish_id UUID, published_at_ns UInt64,
indexed_rows UInt64, indexed_packs UInt32
) ENGINE = MergeTree ORDER BY (index_version, publish_id)"""
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.qualified(self.version_claims)} (
version UInt64, claim_id UUID, claimed_at_ns UInt64
) ENGINE = MergeTree ORDER BY (version, claim_id)"""
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.qualified(self.lease_table)} (
term UInt64, lease_id UUID, holder String, acquired_at_ns UInt64,
expires_at_ns UInt64
) ENGINE = MergeTree ORDER BY (term, lease_id)"""
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {manifest} (
index_version UInt64, publish_id UUID, store_id LowCardinality(String),
pack_id UUID
) ENGINE = MergeTree ORDER BY (index_version, publish_id, store_id, pack_id)"""
        )
        self._client.execute(
            f"CREATE OR REPLACE VIEW {capture_view} AS "
            f"SELECT {', '.join(CAPTURE_COLUMNS[:-1])} FROM {capture_raw} FINAL "
            f"WHERE {membership_predicate(manifest, watermark, bounded=False)}"
        )
        self._client.execute(
            f"CREATE VIEW IF NOT EXISTS {pack_view} AS "
            f"SELECT {', '.join(PACK_COLUMNS[:-1])} FROM {pack_raw} FINAL"
        )
        self._confirm_catalog_is_complete()
        self._client.execute(
            f"INSERT INTO {self.qualified(self.schema_table)} "
            "(version, applied_at_ns) "
            "SELECT toUInt32(%(version)s), toUInt64(%(applied_at_ns)s) "
            "FROM system.one WHERE (SELECT count() FROM "
            f"{self.qualified(self.schema_table)}) = 0",
            {"version": SCHEMA_VERSION, "applied_at_ns": time_ns()},
        )
    def _require_catalog_visibility(self) -> None:
        for _, name in self.objects + self.legacy_objects:
            rows = self._client.execute(
                f"CHECK GRANT SHOW TABLES ON {self.qualified(name)}"
            )
            if rows and rows[0] and rows[0][0]:
                continue
            raise CatalogSchemaVersionError(
                f"catalog `{self.database}`.`{self.prefix}_*` cannot be checked "
                f"safely because this role lacks SHOW TABLES on "
                f"{self.qualified(name)}. Grant catalog visibility and run "
                "ensure_schema() again."
            )

    def _confirm_catalog_is_complete(self) -> None:
        found = self._catalog_state()
        expected = {name for _, name in self.objects}
        missing = sorted(expected - found.keys())
        legacy = sorted(name for _, name in self.legacy_objects if name in found)
        if missing or legacy:
            details = []
            if missing:
                details.append(
                    "not visible: " + ", ".join(f"`{x}`" for x in missing)
                )
            if legacy:
                details.append("superseded: " + ", ".join(f"`{x}`" for x in legacy))
            raise CatalogSchemaVersionError(
                f"catalog `{self.database}`.`{self.prefix}_*` is incomplete after "
                "schema creation (" + "; ".join(details) + "). It was not stamped."
            )
        self._reject_wrong_kinds(found)

    def _verify_compatibility(
        self, found: dict[str, CatalogObject] | None = None
    ) -> bool:
        """Returns True when this is a fresh install, i.e. nothing is there."""
        if found is None:
            found = self._catalog_state()
        if not found:
            return True
        self._reject_wrong_kinds(found)
        if not any(name in found for _, name in self.objects):
            raise CatalogSchemaVersionError(self._leftovers_only(found))
        if self.schema_table not in found:
            raise CatalogSchemaVersionError(self._unstamped_diagnosis(found))
        recorded = self._recorded_version()
        if recorded is not None and recorded != SCHEMA_VERSION:
            raise CatalogSchemaVersionError(
                f"catalog `{self.database}`.`{self.prefix}_*` is at schema "
                f"version {recorded} and this build reads version "
                f"{SCHEMA_VERSION}. A higher version means a newer writer owns "
                "this catalog: upgrade this build rather than writing to it. A "
                "lower one is not upgraded in place. " + self.rebuild_instruction()
            )
        stamp = (
            f"is stamped schema version {SCHEMA_VERSION}"
            if recorded is not None
            else f"holds `{self.schema_table}` with no row in it (an install "
            "of this build that died before stamping)"
        )
        self._reject_legacy_objects_beside_this_build(found)
        missing = [
            name for kind, name in self.objects if kind == "TABLE" and name not in found
        ]
        if missing and not self._holds_catalog_data(found):
            # Nothing survived that recreating them empty could hide, so this is
            # an unfinished install and the DDL completes it. That is the whole
            # reason the stamp table is created FIRST -- "an install interrupted
            # partway leaves a catalog that says this build, unfinished" -- and
            # refusing regardless made it false: an install that died between
            # two CREATEs was refused on every later start, reciting a surviving
            # inventory that was itself one of the tables never created, with a
            # manual drop of every object as the only offered recovery.
            #
            # Returned rather than fallen through: the membership check below
            # reads tables this catalog does not have yet.
            return False
        if missing:
            raise CatalogSchemaVersionError(
                f"catalog `{self.database}`.`{self.prefix}_*` {stamp} but is "
                "missing "
                + ", ".join(f"`{name}`" for name in missing)
                + ". Recreating those empty beside tables that kept their rows "
                "is not a repair: surviving inventory would skip their packs. "
                + self.rebuild_instruction()
            )
        self._reject_a_wrong_descriptor_sort_key(found, stamp)
        if self._inventory_without_membership():
            raise CatalogSchemaVersionError(
                f"catalog `{self.database}`.`{self.prefix}_*` lists one or more "
                f"packs in `{self.pack_raw}` without membership in "
                f"`{self.manifest}` admitted by a publish in `{self.watermark}`. "
                "Those packs are already marked indexed but belong to no "
                "snapshot, so an indexing pass would skip them and report "
                "success while their captures remain invisible. "
                + self.rebuild_instruction()
            )
        return False

    def _catalog_state(self) -> dict[str, CatalogObject]:
        names = [name for _, name in self.objects + self.legacy_objects]
        rows = self._client.execute(
            "SELECT name, engine, sorting_key FROM system.tables "
            "WHERE database = %(database)s AND name IN %(names)s",
            {"database": self.database, "names": names},
        )
        wanted = set(names)
        return {
            text(name): CatalogObject(text(engine), text(sorting_key))
            for name, engine, sorting_key in rows
            if text(name) in wanted
        }

    def _reject_a_wrong_descriptor_sort_key(
        self, found: dict[str, CatalogObject], stamp: str
    ) -> None:
        """The stamp is not evidence about the table standing next to it.

        Every other sort-key check lived on the UNSTAMPED path, so a catalog
        that says this version was taken at its word about the one table whose
        key this branch changed. A partial restore is enough to separate the
        two -- `{prefix}_capture_raw` recovered from a pre-`(store_id,
        pack_id)` backup beside a surviving stamp -- and from there nothing
        refuses it, `CREATE TABLE IF NOT EXISTS` cannot alter it, and
        ReplacingMergeTree goes back to collapsing two packs' rows for one
        capture on the next merge. `_catalog_state()` already reads
        `sorting_key`; this is the comparison it was read for.

        Only reached once the table is known present: a missing one is either
        the unfinished install handled above or already refused there, and in
        both cases the DDL creates it on this build's key.
        """
        descriptor = found.get(self.capture_raw)
        required = ", ".join(CAPTURE_TABLE_ORDER)
        if descriptor is None or _key_columns(descriptor.sorting_key) == _key_columns(
            required
        ):
            return
        raise CatalogSchemaVersionError(
            f"catalog `{self.database}`.`{self.prefix}_*` {stamp} but "
            f"`{self.capture_raw}` has ORDER BY ({descriptor.sorting_key}) "
            f"where this build requires ORDER BY ({required}). The stamp "
            "describes the build that wrote it, not the table beside it, and "
            "`CREATE TABLE IF NOT EXISTS` cannot alter a live sort key: left "
            "in place, a merge collapses two packs' rows for one capture. "
            + self.rebuild_instruction()
        )

    def _reject_wrong_kinds(self, found: dict[str, CatalogObject]) -> None:
        wrong = [
            f"`{name}` is a {found[name].kind} (engine {found[name].engine}) "
            f"where this build creates a {kind}"
            for kind, name in self.objects
            if name in found and found[name].kind != kind
        ]
        if wrong:
            raise CatalogSchemaVersionError(
                f"catalog `{self.database}`.`{self.prefix}_*` holds an object "
                "of the wrong kind: "
                + ", ".join(wrong)
                + ". "
                + self.rebuild_instruction()
            )

    def _leftovers_only(self, found: dict[str, CatalogObject]) -> str:
        leftovers = [name for _, name in self.legacy_objects if name in found]
        listed = ", ".join(f"`{self.database}`.`{name}`" for name in leftovers)
        subject = (
            f"the only object present is {listed}"
            if len(leftovers) == 1
            else f"the only objects present are {listed}"
        )
        it = "it" if len(leftovers) == 1 else "them"
        return (
            f"catalog `{self.database}`.`{self.prefix}_*` holds none of the "
            f"objects this build creates: {subject}. Drop {it} and run "
            "ensure_schema() again. There is nothing to rebuild first."
        )

    def _unstamped_diagnosis(self, found: dict[str, CatalogObject]) -> str:
        findings: list[str] = []
        required_key = ", ".join(CAPTURE_TABLE_ORDER)
        descriptor = found.get(self.capture_raw)
        if descriptor is None:
            findings.append(
                f"`{self.capture_raw}` is absent, so this catalog holds no "
                "descriptor rows at all"
            )
        elif _key_columns(descriptor.sorting_key) != _key_columns(required_key):
            findings.append(
                f"`{self.capture_raw}` has ORDER BY ({descriptor.sorting_key}) "
                f"and this build requires ORDER BY ({required_key})"
            )
        else:
            findings.append(
                f"`{self.capture_raw}` is already sorted on ({required_key}), "
                "the key this build requires -- so the descriptor sort key is "
                "NOT what is wrong with this catalog"
            )
        commit_log = self.commit_log in found
        manifest = self.manifest in found
        if commit_log and not manifest:
            findings.append(
                f"snapshot membership is in `{self.commit_log}`, which this "
                f"build never reads; it moved to `{self.manifest}`, which is absent"
            )
        elif manifest and not commit_log:
            findings.append(
                f"snapshot membership is already in `{self.manifest}`, where "
                f"this build reads it, and `{self.commit_log}` is absent -- so "
                "there is NO commit-log migration owed here"
            )
        elif manifest and commit_log:
            findings.append(
                f"both membership tables are present: `{self.commit_log}` and "
                f"`{self.manifest}`"
            )
        else:
            findings.append(
                f"neither `{self.commit_log}` nor `{self.manifest}` is present, "
                "so nothing records snapshot membership"
            )
        identity_tables = [
            name for name in (self.watermark, self.manifest) if name in found
        ]
        if identity_tables:
            without = sorted(
                set(identity_tables) - self._tables_with_publish_id(identity_tables)
            )
            if without:
                findings.append(
                    ", ".join(f"`{name}`" for name in without)
                    + " has no `publish_id` column"
                )
            else:
                findings.append(
                    ", ".join(f"`{name}`" for name in identity_tables)
                    + " already carry `publish_id`"
                )
        absent = [name for _, name in self.objects if name not in found]
        if absent:
            findings.append(
                "this build also creates "
                + ", ".join(f"`{name}`" for name in absent)
                + f", {'which is not' if len(absent) == 1 else 'none of which are'} present"
            )
        return (
            f"catalog `{self.database}`.`{self.prefix}_*` carries no schema "
            f"stamp and this build requires version {SCHEMA_VERSION}. Version "
            f"{SCHEMA_VERSION} is the first to write `{self.schema_table}`, so "
            "an unstamped catalog is one of the versions before it. What the "
            "server reports about THIS catalog:\n"
            + "".join(f"  - {finding}\n" for finding in findings)
            + "It is refused whichever version it is, and deliberately: this "
            "build cannot know that the differences listed are all of them. "
            + self.rebuild_instruction()
        )

    def _tables_with_publish_id(self, tables: list[str]) -> set[str]:
        rows = self._client.execute(
            "SELECT table FROM system.columns WHERE database = %(database)s "
            "AND table IN %(tables)s AND name = 'publish_id'",
            {"database": self.database, "tables": tables},
        )
        return {text(row[0]) for row in rows}

    def _recorded_version(self) -> int | None:
        rows = self._client.execute(
            f"SELECT version FROM {self.qualified(self.schema_table)} "
            "ORDER BY version DESC LIMIT 1"
        )
        return rows[0][0] if rows else None

    def _inventory_without_membership(self) -> bool:
        rows = self._client.execute(
            "SELECT count() FROM (SELECT store_id, pack_id FROM "
            f"{self.qualified(self.pack_raw)} FINAL "
            "WHERE (store_id, pack_id) NOT IN (SELECT store_id, pack_id FROM "
            f"{self.qualified(self.manifest)} WHERE (index_version, publish_id) "
            "IN (SELECT index_version, publish_id FROM "
            f"{self.qualified(self.watermark)})) LIMIT 1)",
            # A deciding read: this decides whether ensure_schema REFUSES.
            # Answered from a replica that has not fetched a young catalog's
            # membership yet, it refuses a healthy catalog and names a rebuild.
            settings=DECIDING_READ,
        )
        return bool(rows[0][0])

    def _reject_legacy_objects_beside_this_build(
        self, found: dict[str, CatalogObject]
    ) -> None:
        """Refuse a catalog an EARLIER build has also been writing.

        A superseded object standing beside this build's own is not a cleanup
        that was never finished -- ``_leftovers_only`` covers that, where
        nothing of this build is there. It is two builds sharing one prefix,
        and the older one is the dangerous half: its ``ensure_schema`` is all
        CREATE ... IF NOT EXISTS, so it no-ops over these tables and recreates
        its own; its publish writes the pack inventory and an UNCONDITIONAL
        watermark row carrying no publish identity, and never a manifest row.

        Nothing else here notices. The stamp reads this version, no table is
        missing, and ``_inventory_without_membership`` is a per-pack check, so
        it reports the older writer's packs only once they are already durable
        and invisible. Refusing costs a correct deployment nothing: the rebuild
        instruction lists these objects, so a catalog rebuilt by this build has
        none of them.
        """
        leftovers = [name for _, name in self.legacy_objects if name in found]
        if not leftovers:
            return
        raise CatalogSchemaVersionError(
            f"catalog `{self.database}`.`{self.prefix}_*` is at schema version "
            f"{SCHEMA_VERSION} and "
            + ", ".join(f"`{name}`" for name in leftovers)
            + " stands beside it: an object only an earlier build creates. "
            "Either that build is still writing this prefix -- in which case "
            "its packs are entering the pack inventory with no snapshot "
            "membership, so they are already invisible to every reader and "
            "already skipped by every indexing pass -- or a previous rebuild "
            "left it behind. Stop every writer that is not this build. "
            + self.rebuild_instruction()
        )

    def _holds_catalog_data(self, found: dict[str, CatalogObject]) -> bool:
        """Does any surviving table hold a row that a rerun could hide?

        Only the DATA tables are asked. A stamp row, a version claim or a lease
        row survives an interrupted install by design and hides nothing:
        recreating an empty table beside them loses no captures. Asked as one
        statement of existence probes, so a routine startup call does not count
        rows it does not need.
        """
        tables = [
            name
            for name in (
                self.capture_raw,
                self.pack_raw,
                self.watermark,
                self.manifest,
                self.commit_log,
            )
            if name in found
        ]
        if not tables:
            return False
        probes = " UNION ALL ".join(
            f"SELECT 1 FROM {self.qualified(name)} LIMIT 1" for name in tables
        )
        return bool(
            self._client.execute(
                f"SELECT count() FROM ({probes})", settings=DECIDING_READ
            )[0][0]
        )

    def rebuild_instruction(self) -> str:
        objects = ", ".join(
            f"`{self.database}`.`{name}`"
            for _, name in self.objects + self.legacy_objects
        )
        return (
            "The catalog is a derived projection over immutable packs, so "
            "rebuilding it loses nothing: stop every indexer, drop ALL of its "
            f"objects in this order (views first) -- {objects} -- then run "
            "ensure_schema(), acquire_publisher_lease() on the rebuilding "
            "writer, and CatalogReconciler.rebuild() over each pack store. "
            "Dropping the pack inventory is mandatory, not optional."
        )

    def drop(self) -> None:
        """Drop every object, whatever kind this build creates it as.

        ``DROP TABLE`` for all of them, because the catalog that most needs
        tearing down is the one whose kinds are WRONG: ``_reject_wrong_kinds``
        refuses a prefix where a view stands as a table and then prescribes
        exactly this method, and ``DROP VIEW`` against a table is refused even
        with IF EXISTS (Code 80 on 25.12) -- so a kind-specific drop aborted on
        its first statement against the only catalog that required it.
        ``DROP TABLE`` removes a view just as well (verified on 25.12). Views
        still go first: a table dropped out from under a surviving view leaves
        a broken projection for as long as the teardown takes.
        """
        for _, name in self.objects + self.legacy_objects:
            self._client.execute(f"DROP TABLE IF EXISTS {self.qualified(name)}")
