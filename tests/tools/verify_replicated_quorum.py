"""Verify the publish protocol against genuinely replicated catalog tables.

Manual, and not collected by pytest: it needs a ClickHouse with Keeper
configured, which the CPU suite and the ordinary live suite both run without.
Nothing else in this repository exercises `ClickHouseCatalogConfig.insert_quorum`
or the module's claim that `select_sequential_consistency` is necessary but not
sufficient -- those are statements about ReplicatedMergeTree, and a
non-replicated server accepts and ignores every setting involved.

Two replicas of each protocol table live on ONE server under different table
names sharing a Keeper path. That is what a replica is, so quorum, sequential
consistency and replica loss all behave as they would across hosts, without
needing a second machine. What one server CANNOT model is two hosts' clocks:
`clock_skew_ns` is exercised as a parameter here, never as a measured skew.

Setup (ports chosen to avoid a default install). The configs live in
``tests/tools/quorum_harness/`` and write everything under
``/tmp/dmi-quorum-harness/``; copy ``users.xml`` next to the server's data
root once, then start both:

    mkdir -p /tmp/dmi-quorum-harness/{keeper/{logs,snapshots,state,storage,auxiliary},server/{data,tmp,user_files,format_schemas},tmp}
    cp tests/tools/quorum_harness/users.xml /tmp/dmi-quorum-harness/users.xml
    clickhouse-keeper  --config-file tests/tools/quorum_harness/keeper.xml --daemon   # tcp 9181
    clickhouse-server  --config-file tests/tools/quorum_harness/server.xml --daemon   # tcp 9010,
                                                    # <zookeeper> pointing at
                                                    # 9181 and
                                                    # <interserver_http_host>
                                                    # 127.0.0.1 so the replicas
                                                    # can fetch from each other

    PYTHONPATH=src:. python tests/tools/verify_replicated_quorum.py

Every check ASSERTS. An earlier version printed `FAIL` and exited 0, accepted
any exception where a specific one was expected, published an empty snapshot,
and never ran retention -- so it could not fail, and did not test the writes
the design document says it verifies. The exit status is the verdict now.

What it establishes, all measured on 25.12:

1. The whole publish protocol works with every deciding write quorum-durable:
   descriptors, a non-empty manifest, the watermark, the inventory, and a
   pinned read that resolves them.
2. `insert_quorum`, `insert_quorum_parallel = 0` and the bounded
   `insert_quorum_timeout` really rode on the statements (query log, required).
3. Retention works on ReplicatedMergeTree: the orphan sweep resolves its set on
   the initiator and deletes literal pairs, so the mutation is admitted under
   the default settings (a predicate reading another replicated table is not).
4. With quorum unset, fresh replicated tables behave normally.
5. Turning quorum OFF on a catalog that had it ON fails loudly rather than
   silently: once a replicated table has taken a quorum insert, a later
   non-quorum insert is invisible to a sequential-consistency read, so the
   read-backs cannot confirm the writer's own insert
   (`CatalogVersionAllocationError`).
6. An unsatisfiable quorum is refused immediately (Code 285,
   TOO_FEW_LIVE_REPLICAS) rather than parking for ClickHouse's 600-second
   default, because `insert_quorum_timeout` is bounded to `publish_timeout_ns`.
"""
from __future__ import annotations

from time import monotonic
from uuid import UUID, uuid4

from clickhouse_driver import Client
from clickhouse_driver.errors import ServerException

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import (
    CaptureQuery,
    CatalogVersionAllocationError,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
)
from dmi.storage.capture.clickhouse_schema import (
    CAPTURE_TABLE_ORDER,
    _facet_ddl,
    capture_view_definition,
    pack_view_definition,
)

TOO_FEW_LIVE_REPLICAS = 285

TABLES = {
    "capture_raw": (
        """capture_id String, tenant_id String, experiment_id String, run_id String,
session_id String, request_id String, sequence_id String, model_id String,
model_revision String, adapter_revision Nullable(String),
capture_policy_version String, hook_name LowCardinality(String), layer_number Int32,
producer_rank UInt32, step_number UInt64, token_start UInt64, token_end UInt64,
batch_position UInt32, dtype LowCardinality(String), shape Array(UInt32),
captured_at_ns UInt64, pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), pack_record_count UInt32,
payload_offset UInt64, stored_length UInt64, decoded_length UInt64,
codec LowCardinality(String), payload_checksum FixedString(8), index_version UInt64,
"""
        + _facet_ddl(),
        "ReplicatedReplacingMergeTree",
        "index_version",
        ", ".join(CAPTURE_TABLE_ORDER),
    ),
    "pack_inventory_raw": (
        """pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), record_count UInt32,
index_version UInt64""",
        "ReplicatedReplacingMergeTree",
        "index_version",
        "store_id, pack_id",
    ),
    "index_watermark": (
        """index_version UInt64, publish_id UUID, published_at_ns UInt64,
indexed_rows UInt64, indexed_packs UInt32""",
        "ReplicatedMergeTree",
        None,
        "index_version, publish_id",
    ),
    "snapshot_manifest": (
        """index_version UInt64, publish_id UUID, store_id LowCardinality(String),
pack_id UUID""",
        "ReplicatedMergeTree",
        None,
        "index_version, publish_id, store_id, pack_id",
    ),
    "capture_version_claims": (
        "version UInt64, claim_id UUID, claimed_at_ns UInt64",
        "ReplicatedMergeTree",
        None,
        "version, claim_id",
    ),
    "publisher_lease": (
        """term UInt64, lease_id UUID, holder String, acquired_at_ns UInt64,
expires_at_ns UInt64""",
        "ReplicatedMergeTree",
        None,
        "term, lease_id",
    ),
}


def create(client, prefix: str, replica: str, suffix: str) -> None:
    for name, (columns, engine, version, order) in TABLES.items():
        args = f"'/clickhouse/tables/{prefix}/{name}', '{replica}'" + (
            f", {version}" if version else ""
        )
        client.execute(
            f"CREATE TABLE IF NOT EXISTS default.`{prefix}_{name}{suffix}` "
            f"({columns}) ENGINE = {engine}({args}) ORDER BY ({order})"
        )
    # The public views, built from the production builder so they cannot
    # drift from what committed_pack_ids() and the reader actually read.
    # Main replica only: nothing reads `{prefix}_peer_*` views (the peer's
    # tables exist to satisfy quorum), and the verifier's peer naming
    # (`{prefix}_capture_raw_peer`) is not the layout the builders emit.
    if not suffix:
        client.execute(capture_view_definition('default', prefix))
        client.execute(pack_view_definition('default', prefix))


def drop(client, prefix: str) -> None:
    for name in TABLES:
        for suffix in ("_peer", ""):
            try:
                client.execute(f"ATTACH TABLE default.`{prefix}_{name}{suffix}`")
            except ServerException:
                pass
            client.execute(f"DROP TABLE IF EXISTS default.`{prefix}_{name}{suffix}` SYNC")
    # The views create() made; views first so they never dangle over their
    # dropped sources, and DROP TABLE removes a view just as well.
    client.execute(f"DROP TABLE IF EXISTS default.`{prefix}_capture` SYNC")
    client.execute(f"DROP TABLE IF EXISTS default.`{prefix}_pack_inventory` SYNC")


def _refs(descriptors):
    seen, refs = set(), []
    for item in descriptors:
        ref = item.locator.pack_ref
        if (ref.store_id, ref.pack_id) not in seen:
            seen.add((ref.store_id, ref.pack_id))
            refs.append(ref)
    return refs


def _expect(exception_type, call, *, code: int | None = None):
    """Run ``call`` and require exactly the failure the protocol promises."""
    try:
        result = call()
    except exception_type as exc:
        if code is not None and getattr(exc, "code", None) != code:
            raise AssertionError(
                f"expected {exception_type.__name__} with code {code}, got "
                f"code {getattr(exc, 'code', None)}: {exc}"
            ) from exc
        return exc
    raise AssertionError(
        f"expected {exception_type.__name__}, but the call returned {result!r}"
    )


def _query_log_settings(client, prefix: str, statement_fragment: str) -> list[tuple]:
    client.execute("SYSTEM FLUSH LOGS")
    return client.execute(
        "SELECT Settings['insert_quorum'], Settings['insert_quorum_parallel'], "
        "Settings['insert_quorum_timeout'] FROM system.query_log "
        "WHERE type = 'QueryFinish' AND query_kind = 'Insert' "
        "AND query LIKE %(like)s ORDER BY event_time DESC LIMIT 5",
        {"like": f"%{prefix}_{statement_fragment}%"},
    )


def main() -> None:
    client = Client(host="127.0.0.1", port=9010)
    prefix = f"qtest_{uuid4().hex[:8]}"
    control_prefix = f"{prefix}_ctl"
    try:
        _main(client, prefix, control_prefix)
    finally:
        for name in (prefix, control_prefix):
            drop(client, name)
        print("cleaned up")


def _main(client, prefix: str, control_prefix: str) -> None:
    create(client, prefix, "r1", "")        # the replica the writer is pointed at
    create(client, prefix, "r2", "_peer")   # its peer, which quorum has to reach
    print(f"created two replicas of {len(TABLES)} tables under {prefix}")

    config = ClickHouseCatalogConfig(
        database="default",
        table_prefix=prefix,
        insert_quorum=2,
        # One server, one clock: the bound is exercised as a fence parameter
        # here and cannot be MEASURED on this topology. A real deployment sets
        # its own.
        clock_skew_ns=100_000_000,
    )
    writer = ClickHouseCatalogWriter(client, config)
    reader = ClickHouseCaptureCatalog(client, ClickHouseReaderConfig.from_catalog(config))

    # 1. The whole protocol, with every deciding write quorum-durable, over a
    #    snapshot that actually has members and descriptors.
    corpus = synthetic_descriptors(6)
    refs = _refs(corpus)
    assert refs, "the corpus must span at least one pack"
    lease = writer.acquire_publisher_lease("quorum-check")
    version = writer.allocate_version()
    writer.write_descriptors(corpus, index_version=version)
    writer.publish_snapshot(
        index_version=version, refs=refs, published_at_ns=1,
        indexed_rows=len(corpus), indexed_packs=len(refs),
    )
    writer.commit_packs(refs, index_version=version)
    assert writer.last_published_version() == version
    assert writer.committed_pack_ids(
        [(ref.store_id, ref.pack_id) for ref in refs]
    ) == {(ref.store_id, ref.pack_id) for ref in refs}
    page = reader.search(CaptureQuery(limit=len(corpus) + 1))
    assert len(page.items) == len(corpus), (
        f"pinned read resolved {len(page.items)} of {len(corpus)} descriptors"
    )
    assert page.watermark == str(version)
    print(f"PASS  publish cycle with insert_quorum=2 (lease term {lease.term}, "
          f"version {version}, {len(corpus)} descriptors in {len(refs)} packs)")

    # 2. The settings really did ride on the statements. Required evidence:
    #    without it "quorum-durable" above is an assumption.
    assert client.execute("EXISTS TABLE system.query_log")[0][0], (
        "system.query_log is required to verify that the quorum settings were sent"
    )
    for fragment in ("index_watermark", "snapshot_manifest", "capture_raw",
                     "publisher_lease", "capture_version_claims"):
        rows = _query_log_settings(client, prefix, fragment)
        assert rows, f"no finished INSERT into {prefix}_{fragment} in the query log"
        quorum, parallel, timeout_ms = rows[0]
        assert quorum == "2", f"{fragment}: insert_quorum was {quorum!r}"
        assert parallel == "0", f"{fragment}: insert_quorum_parallel was {parallel!r}"
        assert timeout_ms == str(config.publish_timeout_ns // 1_000_000), (
            f"{fragment}: insert_quorum_timeout was {timeout_ms!r}"
        )
    print("PASS  server recorded insert_quorum=2, parallel=0 and the bounded "
          "timeout on every deciding INSERT")

    # 3. Retention on ReplicatedMergeTree. Orphan membership below the head,
    #    exactly what a lost version race leaves behind; the sweep must be
    #    ADMITTED (a mutation whose predicate read the watermark table would
    #    not be, under the default settings) and must delete only the orphan.
    second = writer.allocate_version()
    writer.write_descriptors(corpus, index_version=second)
    writer.publish_snapshot(
        index_version=second, refs=refs, published_at_ns=2,
        indexed_rows=len(corpus), indexed_packs=len(refs),
    )
    orphan = uuid4()
    client.execute(
        f"INSERT INTO default.`{prefix}_snapshot_manifest` "
        "(index_version, publish_id, store_id, pack_id) VALUES",
        [(version, orphan, ref.store_id, UUID(ref.pack_id)) for ref in refs],
        settings={"insert_quorum": 2, "insert_quorum_parallel": 0},
    )
    removed = writer.collect_garbage()
    assert removed[f"{prefix}_snapshot_manifest"] == len(refs), removed
    assert client.execute(
        f"SELECT count() FROM default.`{prefix}_snapshot_manifest` "
        "WHERE publish_id = %(id)s", {"id": orphan},
        settings={"select_sequential_consistency": 1},
    ) == [(0,)]
    assert client.execute(
        f"SELECT count() FROM default.`{prefix}_snapshot_manifest` "
        "WHERE index_version IN (%(a)s, %(b)s)", {"a": version, "b": second},
        settings={"select_sequential_consistency": 1},
    ) == [(2 * len(refs),)], "published membership must survive the sweep"
    assert len(reader.get_by_ids(
        [corpus[0].capture_id], tenant_id=corpus[0].metadata.tenant_id,
        watermark=str(version),
    )) == 1, "a pinned older snapshot still resolves after retention"
    assert not client.execute(
        "SELECT count() FROM system.mutations WHERE database = 'default' "
        "AND table LIKE %(like)s AND is_done = 0",
        {"like": f"{prefix}_%"},
    )[0][0], "a retention mutation is still pending or stuck"
    print(f"PASS  retention removed {removed} on replicated tables")

    # 4. The control, on FRESH tables: quorum off works normally.
    create(client, control_prefix, "r1", "")
    create(client, control_prefix, "r2", "_peer")
    control = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(database="default", table_prefix=control_prefix)
    )
    allocated = control.allocate_version()
    assert allocated == 1, allocated
    print(f"PASS  quorum unset on fresh replicated tables allocates {allocated}")

    # 5. Turning quorum OFF on a catalog that had it ON is refused, loudly.
    #    Once a replicated table has taken a quorum insert, a later non-quorum
    #    insert is INVISIBLE to a select_sequential_consistency read (measured
    #    on 25.12: plain read 2 rows, sequential read 1). Every read-back here
    #    is a sequential read of the writer's own insert, so the protocol
    #    cannot confirm itself and gives up rather than proceeding blind.
    mixed = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(database="default", table_prefix=prefix)
    )
    failure = _expect(CatalogVersionAllocationError, mixed.allocate_version)
    print(f"PASS  quorum turned off mid-life fails loudly: {failure}")

    # 6. The negative control: with the peer gone, quorum cannot be met and the
    #    write has to FAIL with the server's own code, promptly. Last, because
    #    a failed quorum insert leaves state behind.
    for name in TABLES:
        client.execute(f"DETACH TABLE default.`{prefix}_{name}_peer`")
    print("detached the peer replica; a quorum of 2 is now unsatisfiable")
    started = monotonic()
    failure = _expect(ServerException, writer.allocate_version, code=TOO_FEW_LIVE_REPLICAS)
    elapsed = monotonic() - started
    assert elapsed < config.publish_timeout_ns / 1e9 + 5, (
        f"insert_quorum_timeout is not bounded: refused after {elapsed:.1f}s"
    )
    print(f"PASS  quorum write refused after {elapsed:.1f}s with code "
          f"{TOO_FEW_LIVE_REPLICAS}: {str(failure).splitlines()[0][:60]}")


if __name__ == "__main__":
    main()
