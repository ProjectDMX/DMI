"""What this build does when it meets a catalog an earlier build created.

Every other fixture in the suite starts from a schema this build just created,
so none of them could ever see an upgrade go wrong -- and one did. This branch
appended `(store_id, pack_id)` to the descriptor sort key and moved snapshot
membership from `{prefix}_pack_commit_log` to `{prefix}_snapshot_manifest`.
Against an existing catalog both changes are unreachable and silent:

* `CREATE TABLE IF NOT EXISTS` cannot alter a live table, so the old sort key
  survives and the merge deletion this branch exists to prevent stays possible;
* worse, `committed_pack_ids` reads the pack INVENTORY to skip replays while
  membership now reads the MANIFEST. On an upgraded catalog the inventory is
  populated and the manifest is empty, so the next pass skips every
  pre-existing pack as already committed, those packs never reach the manifest,
  and every capture in them is durable in object storage and invisible to every
  reader -- with the indexing pass reporting success.

Measured before the fix, over a version 1 catalog holding four captures in two
packs: `ensure_schema()` returned cleanly, `sorting_key` was still
`tenant_id, experiment_id, run_id, captured_at_ns, capture_id`, the following
rebuild reported `skipped=2 indexed=0`, and the reader returned 0 of 4
captures.

The version 1 fixture below is `main`'s DDL copied verbatim rather than
regenerated, because a fixture built from this branch's code would upgrade
itself into agreement with whatever the branch does and prove nothing.

Run against a reachable ClickHouse:

    DMI_CLICKHOUSE_HOST=127.0.0.1 python -m pytest \
        tests/test_clickhouse_schema_migration_live.py -m "manual and clickhouse" -q
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import importlib.util
from os import environ
from pathlib import Path
import sys
from uuid import UUID, uuid4

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureQuery,
    CaptureRecord,
    CatalogIndexer,
    CatalogReconciler,
    CatalogSchemaVersionError,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
    FilesystemPackStore,
    ObjectPage,
    PackIndex,
    PackWriter,
    StoredObject,
)
from dmi.storage.capture.clickhouse_schema import SCHEMA_VERSION as _SCHEMA_VERSION


pytestmark = [pytest.mark.manual, pytest.mark.clickhouse]


# `main`'s ensure_schema, statement for statement, with only the database and
# prefix parameterised. Rendered from `git show origin/main:` and pasted here:
# an approximation would let the fixture drift towards whatever this branch
# happens to create, which is the one thing it must not do. The five facet
# `ADD COLUMN IF NOT EXISTS` statements are omitted because the CREATE above
# them already declares those columns, so on a fresh table they change nothing.
_VERSION_ONE_SCHEMA = (
    "CREATE DATABASE IF NOT EXISTS `{database}`",
    """CREATE TABLE IF NOT EXISTS `{database}`.`{prefix}_capture_raw` (
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
facet_version UInt16 MATERIALIZED 1,
element_count UInt64 MATERIALIZED toUInt64(arrayProduct(shape)),
tensor_rank UInt8 MATERIALIZED toUInt8(length(shape)),
token_span UInt64 MATERIALIZED toUInt64(token_end - token_start),
compression_ratio Float32 MATERIALIZED toFloat32(if(stored_length = 0, 0, decoded_length / stored_length))
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY (tenant_id, experiment_id, run_id, captured_at_ns, capture_id)""",
    """CREATE TABLE IF NOT EXISTS `{database}`.`{prefix}_pack_inventory_raw` (
pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), record_count UInt32,
index_version UInt64
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY (store_id, pack_id)""",
    "ALTER TABLE `{database}`.`{prefix}_capture_raw` ADD INDEX IF NOT EXISTS "
    "capture_id_bloom capture_id TYPE bloom_filter(0.01) GRANULARITY 4",
    "ALTER TABLE `{database}`.`{prefix}_capture_raw` MATERIALIZE INDEX capture_id_bloom",
    """CREATE TABLE IF NOT EXISTS `{database}`.`{prefix}_index_watermark` (
index_version UInt64, published_at_ns UInt64, indexed_rows UInt64, indexed_packs UInt32
) ENGINE = MergeTree ORDER BY index_version""",
    """CREATE TABLE IF NOT EXISTS `{database}`.`{prefix}_capture_version_claims` (
version UInt64, claim_id UUID, claimed_at_ns UInt64
) ENGINE = MergeTree ORDER BY (version, claim_id)""",
    """CREATE TABLE IF NOT EXISTS `{database}`.`{prefix}_pack_commit_log` (
pack_id UUID, store_id LowCardinality(String), index_version UInt64
) ENGINE = MergeTree ORDER BY (index_version, store_id, pack_id)""",
    "CREATE VIEW IF NOT EXISTS `{database}`.`{prefix}_capture` AS SELECT "
    "capture_id, tenant_id, experiment_id, run_id, session_id, request_id, "
    "sequence_id, model_id, model_revision, adapter_revision, "
    "capture_policy_version, hook_name, layer_number, producer_rank, "
    "step_number, token_start, token_end, batch_position, dtype, shape, "
    "captured_at_ns, pack_id, store_id, object_key, object_bytes, "
    "pack_checksum, pack_record_count, payload_offset, stored_length, "
    "decoded_length, codec, payload_checksum "
    "FROM `{database}`.`{prefix}_capture_raw` FINAL",
    "CREATE VIEW IF NOT EXISTS `{database}`.`{prefix}_pack_inventory` AS SELECT "
    "pack_id, store_id, object_key, object_bytes, pack_checksum, record_count "
    "FROM `{database}`.`{prefix}_pack_inventory_raw` FINAL",
)

_VERSION_ONE_SORT_KEY = "tenant_id, experiment_id, run_id, captured_at_ns, capture_id"
_CURRENT_SORT_KEY = (
    "tenant_id, experiment_id, run_id, captured_at_ns, capture_id, "
    "store_id, pack_id"
)

# The documented rebuild: every object either build creates, views before the
# tables they read. Version 1's commit log is here because a drop that misses
# it leaves the next start refusing a catalog that looks like version 1.
_ALL_OBJECTS = (
    ("VIEW", "capture"),
    ("VIEW", "pack_inventory"),
    ("TABLE", "capture_raw"),
    ("TABLE", "pack_inventory_raw"),
    ("TABLE", "capture_version_claims"),
    ("TABLE", "publisher_lease"),
    ("TABLE", "index_watermark"),
    ("TABLE", "snapshot_manifest"),
    ("TABLE", "schema_version"),
    ("TABLE", "pack_commit_log"),
)


class _Inventory:
    """A `PackInventory` over a filesystem store, as the reconciler needs one."""

    def __init__(self, store: FilesystemPackStore, refs):
        self.store_id = store.store_id
        self._store = store
        self._refs = {ref.object_key: ref for ref in refs}

    def inspect(self, object_key):
        return self._refs[object_key]

    def list_objects(self, *, prefix="", cursor=None, limit=1000) -> ObjectPage:
        keys = sorted(key for key in self._refs if key.startswith(prefix))
        start = int(cursor or 0)
        chosen = keys[start : start + limit]
        following = start + len(chosen)
        return ObjectPage(
            items=tuple(
                StoredObject(key, self._refs[key].object_bytes) for key in chosen
            ),
            next_cursor=str(following) if following < len(keys) else None,
        )

    def read_range(self, ref, offset, length):
        return self._store.read_range(ref, offset, length)

    def stat(self, ref):
        return self._store.stat(ref)

    def put(self, pack, object_key):
        return self._store.put(pack, object_key)


def _metadata(index: int) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=f"capture-{index}",
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id="request-a",
        sequence_id="sequence-a",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=3,
        producer_rank=0,
        step_number=index,
        token_start=index,
        token_end=index + 1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def _packs(root: Path, packs: int = 2, per_pack: int = 2):
    """Real packs in a real store, because the rebuild re-reads their footers."""
    store = FilesystemPackStore(root, store_id="local")
    refs, descriptors, step = [], [], 0
    for number in range(1, packs + 1):
        pack = PackWriter(
            pack_id=UUID(f"018f0000-0000-7000-8000-{number:012d}"),
            created_at_ns=1_700_000_000_000_000_000 + number,
            max_pack_bytes=1024 * 1024,
        )
        for _ in range(per_pack):
            pack.append(CaptureRecord(_metadata(step), b"abcdefgh"))
            step += 1
        sealed = pack.seal()
        ref = store.put(sealed, f"packs/{sealed.pack_id}.dmi-pack")
        refs.append(ref)
        descriptors.extend(PackIndex.from_store(store, ref).descriptors())
    return store, _Inventory(store, refs), refs, descriptors


@contextmanager
def _server():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    client = clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=f"dmi_migration_test_{uuid4().hex}",
    )
    try:
        # Teardown is unconditional here: these tests create tables by hand
        # before any writer runs, so there is no point at which "nothing was
        # created yet" is safe to assume. Every drop is IF EXISTS.
        yield client, config
    finally:
        _drop_everything(client, config)


def _drop_everything(client, config) -> None:
    # The writer's own teardown, which replays its object list (current and
    # superseded) in drop order. _ALL_OBJECTS above stays as this suite's
    # independent statement of that list for the states it constructs by hand.
    ClickHouseCatalogWriter(client, config).drop_schema()


@contextmanager
def _role_that_cannot_see(client, config, hidden: str):
    """A real ClickHouse role granted every catalog object except `hidden`.

    Grant filtering is a server behaviour, not a client one: `system.tables`
    answers each role with only the objects it holds some privilege on, so a
    missing grant is indistinguishable from a missing object to anything that
    reads that table. Nothing but a second, restricted connection reproduces
    it, which is why this lives here and not beside the fakes.
    """
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    user = f"dmi_visibility_test_{uuid4().hex}"
    client.execute(f"CREATE USER `{user}` IDENTIFIED WITH no_password")
    try:
        for _, suffix in _ALL_OBJECTS:
            name = f"{config.table_prefix}_{suffix}"
            if name == f"{config.table_prefix}_{hidden}":
                continue
            client.execute(
                f"GRANT SHOW TABLES, SELECT, INSERT ON "
                f"`{config.database}`.`{name}` TO `{user}`"
            )
        yield clickhouse_driver.Client(
            host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
            port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
            user=user,
            database=config.database,
        )
    finally:
        client.execute(f"DROP USER IF EXISTS `{user}`")


def _install_version_one(client, config) -> None:
    for statement in _VERSION_ONE_SCHEMA:
        client.execute(
            statement.format(database=config.database, prefix=config.table_prefix)
        )


def _populate_version_one(client, config, writer, descriptors, refs) -> None:
    """A version 1 catalog with one published batch in it.

    The descriptor and inventory inserts go through the writer because the
    column lists did not change between the versions; the commit log and the
    watermark are written directly, because version 1's membership table is
    the one this build no longer has code for.
    """
    writer.write_descriptors(descriptors, index_version=1)
    client.execute(
        f"INSERT INTO `{config.database}`.`{config.table_prefix}_pack_commit_log` "
        "(pack_id, store_id, index_version) VALUES",
        [(ref.pack_id, ref.store_id, 1) for ref in refs],
    )
    client.execute(
        f"INSERT INTO `{config.database}`.`{config.table_prefix}_index_watermark` "
        "(index_version, published_at_ns, indexed_rows, indexed_packs) VALUES",
        [(1, 1, len(descriptors), len(refs))],
    )
    writer.commit_packs(refs, index_version=1)


def _count(client, config, suffix: str) -> int:
    return client.execute(
        f"SELECT count() FROM `{config.database}`.`{config.table_prefix}_{suffix}`"
    )[0][0]


def _sort_key(client, config) -> str:
    return client.execute(
        "SELECT sorting_key FROM system.tables "
        "WHERE database = %(database)s AND name = %(name)s",
        {
            "database": config.database,
            "name": f"{config.table_prefix}_capture_raw",
        },
    )[0][0]


def _recorded_version(client, config) -> int:
    return client.execute(
        f"SELECT version FROM `{config.database}`.`{config.table_prefix}_schema_version`"
    )[0][0]


def _object_names(client, config) -> set[str]:
    """Every object of this prefix the server actually holds."""
    return {
        row[0]
        for row in client.execute(
            "SELECT name FROM system.tables WHERE database = %(database)s "
            "AND name LIKE %(pattern)s",
            {"database": config.database, "pattern": f"{config.table_prefix}\\_%"},
        )
    }


def _columns(client, config, suffix: str) -> set[str]:
    return {
        row[0]
        for row in client.execute(
            "SELECT name FROM system.columns WHERE database = %(database)s "
            "AND table = %(table)s",
            {
                "database": config.database,
                "table": f"{config.table_prefix}_{suffix}",
            },
        )
    }


def _findings(message: str) -> list[str]:
    """The bulleted differences a refusal claims it found in this catalog."""
    return [
        line.strip()[2:] for line in message.splitlines() if line.startswith("  - ")
    ]


# --- the upgrade that cannot happen ------------------------------------------


def test_starting_against_a_version_one_catalog_is_refused(tmp_path: Path):
    """The whole defect, refused at the first statement that would hide it."""
    _, _, refs, descriptors = _packs(tmp_path)
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        _install_version_one(client, config)
        _populate_version_one(client, config, writer, descriptors, refs)

        # The fixture really is the old schema, and really does hold data.
        assert _sort_key(client, config) == _VERSION_ONE_SORT_KEY
        assert _count(client, config, "capture_raw") == len(descriptors)
        assert _count(client, config, "pack_inventory_raw") == len(refs)

        with pytest.raises(CatalogSchemaVersionError) as raised:
            writer.ensure_schema()

        message = str(raised.value)
        assert "carries no schema stamp" in message
        assert f"requires version {_SCHEMA_VERSION}" in message
        # Both incompatible changes, named: an operator has to know that no
        # ALTER gets them out of this. Each one is a finding read off THIS
        # catalog, and each is checked against the live schema first.
        findings = _findings(message)
        assert _sort_key(client, config) == _VERSION_ONE_SORT_KEY
        assert any(
            "ORDER BY" in item and _VERSION_ONE_SORT_KEY in item
            for item in findings
        ), findings
        assert f"{config.table_prefix}_pack_commit_log" in _object_names(
            client, config
        )
        assert f"{config.table_prefix}_snapshot_manifest" not in _object_names(
            client, config
        )
        assert any(
            f"membership is in `{config.table_prefix}_pack_commit_log`" in item
            and f"`{config.table_prefix}_snapshot_manifest`, which is absent" in item
            for item in findings
        ), findings
        # And the exact recovery, including the step it is fatal to skip.
        assert "CatalogReconciler.rebuild()" in message
        assert "Dropping the pack inventory is mandatory" in message

        # Refused before any DDL: the old catalog is exactly as it was, and in
        # particular this build has not created the manifest that would make
        # the next start look like a stamped, half-migrated catalog.
        assert _sort_key(client, config) == _VERSION_ONE_SORT_KEY
        assert _count(client, config, "capture_raw") == len(descriptors)
        assert (
            client.execute(
                "SELECT count() FROM system.tables WHERE database = %(database)s "
                "AND name IN %(names)s",
                {
                    "database": config.database,
                    "names": [
                        f"{config.table_prefix}_snapshot_manifest",
                        f"{config.table_prefix}_schema_version",
                    ],
                },
            )
            == [(0,)]
        )


# --- the upgrade this build will really perform first -------------------------
#
# `_VERSION_ONE_SCHEMA` above is the oldest catalog anyone might still be
# running. It is NOT the one this build meets first. That is this branch's own
# immediate predecessor, and it is a different schema: it already has the new
# descriptor sort key and the manifest, and it has no commit log at all. The
# first refusal read the absent version stamp as "this is version 1" and
# recited version 1's two differences at it -- both false of that catalog. An
# operator who checks two named facts, finds both untrue and concludes the
# guard is broken goes around it, which is the one path the guard exists to
# prevent. So the diagnosis is now read off the live schema, and this is the
# test whose absence let that ship.

# The predecessor's catalog writer, vendored under tests/data rather than read
# out of git history at run time.
#
# `git show bea3ed8:...` used to produce it, and could not keep producing it:
# this repository squash-merges, so that commit is reachable only from the
# feature branch and disappears when the branch is deleted. Every recent commit
# on `main` is single-parent -- PR #113 landed as `08ea071` that way -- so after
# a merge the extraction failed, the test skipped, and the workflow's
# "Fail if any live test was skipped" step turned post-merge CI red.
# `fetch-depth: 0` does not help: it retains history that exists, not an object
# that no longer does.
#
# The checksum is what `git show` was really providing. This file is a
# byte-exact snapshot of that commit's source, not a transcription, and the
# digest is verified on every load -- so the fixture still cannot drift towards
# whatever this branch happens to create, which is the one property a
# schema-upgrade fixture must never lose. Regenerate only to track a different
# predecessor, and then deliberately:
#
#     git show <commit>:src/dmi/storage/capture/clickhouse_catalog.py \
#         > tests/data/upstream_clickhouse_catalog_<short>.py
#     sha256sum tests/data/upstream_clickhouse_catalog_<short>.py
_UPSTREAM_COMMIT = "bea3ed828f69de0a56b2bf30023a8a49fc19745e"
_UPSTREAM_FIXTURE = "upstream_clickhouse_catalog_bea3ed8.py"
_UPSTREAM_SHA256 = "9223310c2ad2b0a5ef8937f8e59da63dec400f530e557081bbbf79e22f989577"


def _upstream_catalog_module():
    """The predecessor's own catalog writer, executed rather than described.

    A byte-exact snapshot of `{_UPSTREAM_COMMIT}`'s source, verified against
    `_UPSTREAM_SHA256` on every load: the predecessor's DDL is the
    predecessor's code, and an edit to the fixture fails loudly instead of
    quietly agreeing with this branch.

    Nothing here can skip. A missing or altered fixture is a repository fault,
    not an unavailable environment, and skipping on it is what let the old
    `git show` extraction pass as coverage it was no longer providing.

    Loaded under a name inside `dmi.storage.capture` so its `from .catalog
    import ...` resolves, and never written into the package directory.
    """
    source = Path(__file__).resolve().parent / "data" / _UPSTREAM_FIXTURE
    digest = sha256(source.read_bytes()).hexdigest()
    assert digest == _UPSTREAM_SHA256, (
        f"{source} is not {_UPSTREAM_COMMIT}'s source: expected sha256 "
        f"{_UPSTREAM_SHA256}, found {digest}. The fixture pins what this "
        "build is being compared AGAINST, so an edit to it silently weakens "
        "every assertion below."
    )
    name = "dmi.storage.capture._upstream_catalog_fixture"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None, source
    module = importlib.util.module_from_spec(spec)
    # `dataclass` resolves annotations through `sys.modules`, so the module has
    # to be registered before it is executed, and removed after so nothing else
    # in the session can import the predecessor by accident.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


def test_the_refusal_only_claims_differences_the_upstream_catalog_really_has(
    tmp_path: Path,
):
    """Every specific claim, checked against the catalog it was made about.

    The predecessor's catalog already has this build's descriptor sort key and
    writes membership to the manifest, so a refusal that says otherwise is
    simply wrong -- and being wrong here is worse than being vague, because the
    operator can check it. What it really lacks is the publish-identity column
    on the watermark and the manifest, the publisher lease, and the version
    stamp; those are the claims the message is allowed to make.
    """
    upstream = _upstream_catalog_module()
    store, _, refs, descriptors = _packs(tmp_path)
    with _server() as (client, config):
        # The predecessor builds and fills its own catalog, with its own code.
        older = upstream.ClickHouseCatalogWriter(
            client,
            upstream.ClickHouseCatalogConfig(
                database=config.database, table_prefix=config.table_prefix
            ),
        )
        older.ensure_schema()
        older.write_descriptors(descriptors, index_version=1)
        older.publish_snapshot(
            index_version=1,
            refs=refs,
            published_at_ns=1,
            indexed_rows=len(descriptors),
            indexed_packs=len(refs),
        )
        older.commit_packs(refs, index_version=1)
        assert _count(client, config, "capture_raw") == len(descriptors)
        assert _count(client, config, "snapshot_manifest") == len(refs)

        with pytest.raises(CatalogSchemaVersionError) as raised:
            ClickHouseCatalogWriter(client, config).ensure_schema()
        message = str(raised.value)
        findings = _findings(message)

        # What is actually true of this catalog, read from the server.
        present = _object_names(client, config)
        sort_key = _sort_key(client, config)
        assert sort_key == _CURRENT_SORT_KEY
        assert f"{config.table_prefix}_pack_commit_log" not in present
        assert f"{config.table_prefix}_snapshot_manifest" in present
        assert f"{config.table_prefix}_publisher_lease" not in present
        assert f"{config.table_prefix}_schema_version" not in present
        assert "publish_id" not in _columns(client, config, "index_watermark")
        assert "publish_id" not in _columns(client, config, "snapshot_manifest")

        # ...and the refusal says exactly that, difference by difference.
        assert "schema version 1" not in message
        assert any(
            sort_key in item and "NOT what is wrong" in item for item in findings
        ), findings
        assert any(
            f"membership is already in `{config.table_prefix}_snapshot_manifest`"
            in item
            and "NO commit-log migration owed" in item
            for item in findings
        ), findings
        assert not any(
            f"membership is in `{config.table_prefix}_pack_commit_log`" in item
            for item in findings
        ), findings
        assert any(
            "no `publish_id` column" in item
            and f"{config.table_prefix}_index_watermark" in item
            and f"{config.table_prefix}_snapshot_manifest" in item
            for item in findings
        ), findings
        assert any(
            f"`{config.table_prefix}_publisher_lease`" in item
            and f"`{config.table_prefix}_schema_version`" in item
            and "present" in item
            for item in findings
        ), findings

        # Refused all the same, with the reason for refusing a catalog whose
        # every probed difference is already accounted for.
        assert "refused whichever version it is" in message
        assert "cannot know that the differences listed are all of them" in message

        # And refused BEFORE any DDL: the predecessor's catalog is untouched,
        # in particular not now carrying a stamp or a lease table that would
        # make the next start read it as half-migrated.
        assert _object_names(client, config) == present
        assert _sort_key(client, config) == sort_key
        assert _count(client, config, "capture_raw") == len(descriptors)

        # The documented recovery works from here too, which is what makes the
        # refusal actionable rather than terminal.
        _drop_everything(client, config)
        writer = ClickHouseCatalogWriter(client, config)
        writer.ensure_schema()
        writer.acquire_publisher_lease("rebuild")
        result = CatalogReconciler(
            _Inventory(store, refs), CatalogIndexer(store, writer)
        ).rebuild(prefix="packs/", page_size=8)
        assert result.failures == ()
        assert result.indexed_packs == len(refs)
        assert _recorded_version(client, config) == _SCHEMA_VERSION


def test_a_lone_leftover_commit_log_names_itself_instead_of_wedging(tmp_path: Path):
    """One inert table nothing reads, left by a drop that missed it.

    It made the catalog non-empty, so `ensure_schema` did not treat the start
    as a fresh install; it carried no version stamp, so the start read as an
    old schema; and the refusal then prescribed the full rebuild the operator
    had just finished, over tables that no longer exist. Nothing could be
    created and no indexer could start until somebody noticed one table.
    """
    with _server() as (client, config):
        client.execute(
            f"CREATE TABLE `{config.database}`.`{config.table_prefix}"
            "_pack_commit_log` (pack_id UUID, store_id LowCardinality(String), "
            "index_version UInt64) ENGINE = MergeTree "
            "ORDER BY (index_version, store_id, pack_id)"
        )
        writer = ClickHouseCatalogWriter(client, config)

        with pytest.raises(CatalogSchemaVersionError) as raised:
            writer.ensure_schema()

        message = str(raised.value)
        assert (
            f"the only object present is `{config.database}`."
            f"`{config.table_prefix}_pack_commit_log`"
        ) in message
        assert "Drop it and run ensure_schema() again" in message
        # Not the rebuild: there is nothing left to rebuild from.
        assert "CatalogReconciler.rebuild()" not in message

        # And doing what it says gets the catalog started.
        client.execute(
            f"DROP TABLE `{config.database}`.`{config.table_prefix}_pack_commit_log`"
        )
        writer.ensure_schema()
        assert _recorded_version(client, config) == _SCHEMA_VERSION


def test_a_dropped_view_is_recreated_instead_of_forcing_a_rebuild(tmp_path: Path):
    """A view is derived from the tables, so losing one is not a data loss.

    The refusal used to treat every object alike, so a dropped view demanded
    the full rebuild -- which re-reads every pack footer in the store and
    leaves readers on an empty and then partial catalog while it runs. Missing
    TABLES stay refused; that is where rows really are at stake.
    """
    store, inventory, refs, descriptors = _packs(tmp_path)
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        reader = ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config)
        )
        writer.ensure_schema()
        writer.acquire_publisher_lease("rebuild")
        CatalogReconciler(inventory, CatalogIndexer(store, writer)).rebuild(
            prefix="packs/", page_size=8
        )
        assert _count(client, config, "capture") == len(descriptors)

        for view in ("capture", "pack_inventory"):
            client.execute(
                f"DROP VIEW `{config.database}`.`{config.table_prefix}_{view}`"
            )

        writer.ensure_schema()

        # Back, and serving exactly the published rows -- no rebuild, no
        # re-read of a single pack footer.
        assert _count(client, config, "capture") == len(descriptors)
        assert _count(client, config, "pack_inventory") == len(refs)
        assert len(reader.search(CaptureQuery(limit=100)).items) == len(descriptors)
        assert _recorded_version(client, config) == _SCHEMA_VERSION
        # A dropped TABLE is still refused: recreating one empty beside an
        # inventory that kept its rows is the state that hides every capture.
        client.execute(
            f"DROP TABLE `{config.database}`.`{config.table_prefix}_snapshot_manifest`"
        )
        with pytest.raises(CatalogSchemaVersionError, match="is missing"):
            writer.ensure_schema()


def test_the_documented_rebuild_restores_a_version_one_catalog(tmp_path: Path):
    """Drop everything, reconcile, and the captures are back -- with locators.

    This is what the refusal above sends an operator to do, so it is proven end
    to end rather than described: the packs are the only durable copy, and the
    catalog is reconstructed from their footers alone.
    """
    store, inventory, refs, descriptors = _packs(tmp_path)
    expected = {item.capture_id: item.locator for item in descriptors}
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        reader = ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config)
        )
        _install_version_one(client, config)
        _populate_version_one(client, config, writer, descriptors, refs)
        with pytest.raises(CatalogSchemaVersionError):
            writer.ensure_schema()

        # The procedure, exactly as documented.
        _drop_everything(client, config)
        writer.ensure_schema()
        writer.acquire_publisher_lease("rebuild")
        result = CatalogReconciler(
            inventory, CatalogIndexer(store, writer)
        ).rebuild(prefix="packs/", page_size=8)

        assert result.failures == ()
        assert result.skipped_packs == 0, "a surviving inventory row skipped a pack"
        assert result.indexed_packs == len(refs)
        assert result.indexed_rows == len(descriptors)

        # Version 2, with the sort key that was unreachable by any upgrade.
        assert _recorded_version(client, config) == _SCHEMA_VERSION
        assert _sort_key(client, config) == _CURRENT_SORT_KEY

        page = reader.search(CaptureQuery(limit=100, tenant_id="tenant-a"))
        assert {item.capture_id: item.locator for item in page.items} == expected
        # And every derived table is repopulated, not just the descriptors.
        assert _count(client, config, "snapshot_manifest") == len(refs)
        assert _count(client, config, "pack_inventory_raw") == len(refs)
        assert _count(client, config, "index_watermark") == 1
        assert _count(client, config, "capture") == len(descriptors)


def test_a_rebuild_that_keeps_the_inventory_is_refused_before_it_empties_the_catalog(
    tmp_path: Path,
):
    """Defect (b) on its own, without an old schema anywhere near it.

    A populated inventory beside an empty manifest is the state an upgrade
    produced, and it is also what a drop that spares `_pack_inventory_raw`
    leaves behind. The second half of this test is the damage, run deliberately:
    the reconciler has no guard of its own, so from this state it indexes
    nothing, publishes nothing, and returns a successful-looking result over an
    empty catalog. That is why `ensure_schema` refuses to start here at all.
    """
    store, inventory, refs, descriptors = _packs(tmp_path)
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        reader = ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config)
        )
        writer.ensure_schema()
        writer.acquire_publisher_lease("rebuild")
        indexer = CatalogIndexer(store, writer)
        CatalogReconciler(inventory, indexer).rebuild(prefix="packs/", page_size=8)
        assert len(reader.search(CaptureQuery(limit=100)).items) == len(descriptors)

        # Membership gone, replay guard intact: what "drop the catalog but keep
        # the inventory" leaves behind.
        client.execute(
            f"TRUNCATE TABLE `{config.database}`.`{config.table_prefix}_snapshot_manifest`"
        )

        # The descriptor rows are all still there, and not one of them is
        # readable -- membership is what decides, and it names nothing.
        assert _count(client, config, "capture_raw") == len(descriptors)
        assert reader.search(CaptureQuery(limit=100)).items == ()
        identities = [(ref.store_id, ref.pack_id) for ref in refs]
        assert writer.committed_pack_ids(identities) == set(identities), (
            "precondition: the inventory still reports every pack as indexed"
        )

        with pytest.raises(CatalogSchemaVersionError) as raised:
            writer.ensure_schema()
        assert "remain invisible" in str(raised.value)

        # The damage the refusal prevents, demonstrated rather than asserted
        # from the design: an indexing pass over this state succeeds and hides
        # every capture.
        damage = CatalogReconciler(inventory, indexer).rebuild(
            prefix="packs/", page_size=8
        )
        assert damage.failures == ()
        assert damage.skipped_packs == len(refs)
        assert damage.indexed_packs == 0
        assert damage.indexed_rows == 0
        assert reader.search(CaptureQuery(limit=100)).items == ()
        assert _count(client, config, "capture_raw") == len(descriptors)


# --- the version stamp -------------------------------------------------------


def test_a_fresh_install_records_this_builds_version_and_stays_idempotent(
    tmp_path: Path,
):
    _, _, refs, descriptors = _packs(tmp_path, packs=1)
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)

        writer.ensure_schema()
        writer.acquire_publisher_lease("fresh-install")
        assert _recorded_version(client, config) == _SCHEMA_VERSION

        writer.write_descriptors(descriptors, index_version=1)
        writer.publish_snapshot(
            index_version=1,
            refs=refs,
            published_at_ns=1,
            indexed_rows=len(descriptors),
            indexed_packs=len(refs),
        )
        writer.commit_packs(refs, index_version=1)

        # A stamped, populated catalog is accepted, and the stamp is written
        # once however many times the schema is ensured.
        writer.ensure_schema()
        writer.ensure_schema()

        assert _count(client, config, "schema_version") == 1
        assert _recorded_version(client, config) == _SCHEMA_VERSION
        assert _count(client, config, "capture_raw") == len(descriptors)
        assert _count(client, config, "capture") == len(descriptors)


def test_a_catalog_stamped_by_a_newer_build_is_refused():
    """A newer writer owns this catalog; this build must not touch it."""
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        writer.ensure_schema()
        client.execute(
            f"INSERT INTO `{config.database}`.`{config.table_prefix}_schema_version` "
            "(version, applied_at_ns) VALUES",
            [(_SCHEMA_VERSION + 1, 2)],
        )

        with pytest.raises(CatalogSchemaVersionError) as raised:
            writer.ensure_schema()

        assert f"schema version {_SCHEMA_VERSION + 1}" in str(raised.value)
        assert f"reads version {_SCHEMA_VERSION}" in str(raised.value)


def test_a_catalog_missing_one_of_its_tables_is_refused():
    """Half a drop is the dangerous state, so it is not quietly completed.

    "Dangerous" is what the surviving ROWS make it: recreating the manifest
    empty beside an inventory that kept its packs is the state where the next
    pass skips every pack it lists and nothing refills what was dropped. So the
    catalog here holds data before the table goes.
    """
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        writer.ensure_schema()
        client.execute(
            f"INSERT INTO `{config.database}`.`{config.table_prefix}"
            "_pack_inventory_raw` (pack_id, store_id, object_key, "
            "object_bytes, pack_checksum, record_count, index_version) VALUES",
            [(uuid4(), "local", "packs/a.dmi-pack", 128, "a" * 64, 1, 1)],
        )
        client.execute(
            f"DROP TABLE `{config.database}`.`{config.table_prefix}_snapshot_manifest`"
        )

        with pytest.raises(CatalogSchemaVersionError) as raised:
            writer.ensure_schema()

        message = str(raised.value)
        assert f"missing `{config.table_prefix}_snapshot_manifest`" in message
        assert "CatalogReconciler.rebuild()" in message


def test_an_install_interrupted_between_two_creates_is_completed():
    """The reason the stamp table is created first, over a real server.

    `ensure` writes `{prefix}_schema_version` before anything else so an
    interrupted install "leaves a catalog that says this build, unfinished
    rather than one refused forever". The missing-table refusal ran regardless
    of whether anything survived, so it refused that catalog too -- reciting a
    surviving pack inventory that was itself one of the tables that had never
    been created -- and the only recovery it offered was a manual drop of every
    object. With every data table empty, re-running the DDL is the completion,
    and the catalog works afterwards.
    """
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        writer.ensure_schema()
        # An install that died before the manifest existed, with no rows
        # anywhere -- exactly what a crash between two CREATEs leaves.
        client.execute(
            f"DROP TABLE `{config.database}`.`{config.table_prefix}_snapshot_manifest`"
        )

        writer.ensure_schema()

        assert client.execute(
            "SELECT count() FROM system.tables WHERE database = %(database)s "
            "AND name = %(name)s",
            {
                "database": config.database,
                "name": f"{config.table_prefix}_snapshot_manifest",
            },
        ) == [(1,)]
        # And it is a working catalog, not merely a complete-looking one.
        writer.acquire_publisher_lease("resumed")
        version = writer.allocate_version()
        writer.publish_snapshot(
            index_version=version, refs=(), published_at_ns=1,
            indexed_rows=0, indexed_packs=0,
        )
        writer.release_publisher_lease()


# --- visibility, which `system.tables` cannot tell apart from absence --------


def test_a_role_that_cannot_see_one_object_is_told_to_grant_it_not_to_rebuild():
    """A missing grant must not be diagnosed as a missing table.

    `system.tables` is grant-filtered per role, so an object this role holds
    no privilege on simply is not in the answer -- identical, to every reader
    of that table, to an object that was dropped. Every compatibility verdict
    is read off that answer, so a healthy stamped catalog was refused with
    "missing `{prefix}_snapshot_manifest` ... drop ALL of its objects", and an
    operator who follows that instruction destroys a catalog whose only fault
    was one absent GRANT. `_require_catalog_visibility` says the true thing,
    and it must therefore be reached FIRST -- before any verdict can be drawn
    from a filtered read.
    """
    with _server() as (client, config):
        writer = ClickHouseCatalogWriter(client, config)
        writer.ensure_schema()
        # A published row, so this is a live catalog holding data rather than
        # an unfinished install -- the state whose refusal names a rebuild.
        writer.acquire_publisher_lease("visibility")
        writer.publish_snapshot(
            index_version=1, refs=(), published_at_ns=1,
            indexed_rows=0, indexed_packs=0,
        )
        writer.release_publisher_lease()
        assert _recorded_version(client, config) == _SCHEMA_VERSION

        with _role_that_cannot_see(
            client, config, hidden="snapshot_manifest"
        ) as restricted:
            # The fixture really does hide exactly one object from this role,
            # and hides it in the one table every verdict is read from.
            visible = {
                row[0]
                for row in restricted.execute(
                    "SELECT name FROM system.tables WHERE database = "
                    "%(database)s AND name LIKE %(pattern)s",
                    {
                        "database": config.database,
                        "pattern": f"{config.table_prefix}\\_%",
                    },
                )
            }
            assert f"{config.table_prefix}_capture_raw" in visible
            assert f"{config.table_prefix}_snapshot_manifest" not in visible

            with pytest.raises(CatalogSchemaVersionError) as raised:
                ClickHouseCatalogWriter(restricted, config).ensure_schema()

        message = str(raised.value)
        assert "lacks SHOW TABLES" in message
        assert f"{config.table_prefix}_snapshot_manifest" in message
        assert "Grant catalog visibility" in message
        # And emphatically NOT the diagnosis that would have this operator
        # tear down a catalog that is fine.
        assert "missing" not in message
        assert "drop ALL of its objects" not in message
        assert "CatalogReconciler.rebuild()" not in message

        # The catalog the restricted role could not check is untouched.
        assert _object_names(client, config) == {
            f"{config.table_prefix}_{suffix}"
            for kind, suffix in _ALL_OBJECTS
            if suffix != "pack_commit_log"
        }
        assert _recorded_version(client, config) == _SCHEMA_VERSION


def test_the_capture_catalog_ignores_the_native_paths_offload_table():
    """The two storage paths share a server without seeing each other.

    DMI has two sinks and exactly one is ever active for a runtime: the native
    `ClickHouseRecordSink`, which writes whole tensors into its own table
    (`offload` by default), and the Python capture path, which writes packs to
    an object store and uses this catalog as the index. They are selected at
    `MonitoringEngine.create_record_runtime(..., record_sink=...)` and their
    ClickHouse footprints are disjoint by construction -- `{prefix}_*` here,
    one configured table there.

    "By construction" is worth an assertion anyway, because the schema guard
    refuses catalogs it does not recognise and `drop_schema` now issues
    `DROP TABLE` for every object it owns. Either could reach a neighbour's
    table by accident, and the failure would be somebody else's data.
    """
    with _server() as (client, config):
        native_table = f"`{config.database}`.`offload_{uuid4().hex[:8]}`"
        client.execute(
            f"CREATE TABLE {native_table} (model_id String, request_id String, "
            "payload String) ENGINE = MergeTree ORDER BY (model_id, request_id)"
        )
        try:
            client.execute(
                f"INSERT INTO {native_table} (model_id, request_id, payload) VALUES",
                [("m", "r", "bytes")],
            )
            writer = ClickHouseCatalogWriter(client, config)

            # The guard accepts a database that holds a stranger's table.
            writer.ensure_schema()
            writer.acquire_publisher_lease("coexistence")
            version = writer.allocate_version()
            writer.publish_snapshot(
                index_version=version, refs=(), published_at_ns=1,
                indexed_rows=0, indexed_packs=0,
            )

            # And the stranger is untouched by any of it.
            assert client.execute(f"SELECT count() FROM {native_table}") == [(1,)]

            # Including by the teardown, which drops every object it owns.
            writer.release_publisher_lease()
            writer.drop_schema()
            assert client.execute(f"SELECT count() FROM {native_table}") == [(1,)]
        finally:
            client.execute(f"DROP TABLE IF EXISTS {native_table}")
