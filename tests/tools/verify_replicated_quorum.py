"""Verify insert_quorum against genuinely replicated catalog tables.

Manual, and not collected by pytest: it needs a ClickHouse with Keeper
configured, which the CPU suite and the ordinary live suite both run without.
Nothing else in this repository exercises `ClickHouseCatalogConfig.insert_quorum`
or the module's claim that `select_sequential_consistency` is necessary but not
sufficient -- those are statements about ReplicatedMergeTree, and a
non-replicated server accepts and ignores every setting involved.

Two replicas of each protocol table live on ONE server under different table
names sharing a Keeper path. That is what a replica is, so quorum, sequential
consistency and replica loss all behave as they would across hosts, without
needing a second machine.

Setup (ports chosen to avoid a default install):

    clickhouse-keeper --config-file keeper.xml      # tcp_port 9181
    clickhouse-server  --config-file server.xml     # tcp_port 9010, with
                                                    # <zookeeper> pointing at
                                                    # 9181 and
                                                    # <interserver_http_host>
                                                    # 127.0.0.1 so the replicas
                                                    # can fetch from each other

    python tests/tools/verify_replicated_quorum.py

What it establishes, all measured on 25.12:

1. The whole publish protocol works with every deciding write quorum-durable.
2. With quorum unset, fresh replicated tables behave normally.
3. Turning quorum OFF on a catalog that had it ON fails loudly rather than
   silently: once a replicated table has taken a quorum insert, a later
   non-quorum insert is invisible to a sequential-consistency read, so the
   read-backs cannot confirm the writer's own insert.
4. An unsatisfiable quorum is refused immediately (Code 285) rather than
   parking for ClickHouse's 600-second default, because `insert_quorum_timeout`
   is bounded to `publish_timeout_ns`.
"""
from uuid import uuid4

from clickhouse_driver import Client

from dmi.storage.capture import ClickHouseCatalogConfig, ClickHouseCatalogWriter
from dmi.storage.capture.clickhouse_schema import CAPTURE_TABLE_ORDER, _facet_ddl

PREFIX = f"qtest_{uuid4().hex[:8]}"
ZK = f"/clickhouse/tables/{PREFIX}"

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


def create(client, replica: str, suffix: str) -> None:
    for name, (columns, engine, version, order) in TABLES.items():
        args = f"'{ZK}/{name}', '{replica}'" + (f", {version}" if version else "")
        client.execute(
            f"CREATE TABLE IF NOT EXISTS default.`{PREFIX}_{name}{suffix}` "
            f"({columns}) ENGINE = {engine}({args}) ORDER BY ({order})"
        )


def main() -> None:
    global PREFIX, ZK
    client = Client(host="127.0.0.1", port=9010)
    create(client, "r1", "")        # the replica the writer is pointed at
    create(client, "r2", "_peer")   # its peer, which quorum has to reach
    print(f"created two replicas of {len(TABLES)} tables under {PREFIX}")

    config = ClickHouseCatalogConfig(
        database="default", table_prefix=PREFIX, insert_quorum=2
    )
    writer = ClickHouseCatalogWriter(client, config)

    # 1. The whole protocol, with every deciding write quorum-durable.
    lease = writer.acquire_publisher_lease("quorum-check")
    version = writer.allocate_version()
    writer.publish_snapshot(
        index_version=version, refs=(), published_at_ns=1,
        indexed_rows=0, indexed_packs=0,
    )
    assert writer.last_published_version() == version
    print(f"PASS  publish cycle with insert_quorum=2 (lease term {lease.term}, "
          f"version {version})")

    # 2. The settings really did ride on the statements.
    rows = client.execute(
        "SELECT Settings['insert_quorum'], Settings['insert_quorum_parallel'], "
        "Settings['insert_quorum_timeout'] FROM system.query_log "
        "WHERE type = 'QueryFinish' AND query LIKE %(like)s "
        "AND Settings['insert_quorum'] != '' ORDER BY event_time DESC LIMIT 5",
        {"like": f"%{PREFIX}_index_watermark%"},
    ) if client.execute("EXISTS TABLE system.query_log")[0][0] else []
    if rows:
        print(f"PASS  server recorded insert_quorum settings: {rows[0]}")

    # 3. The control, on FRESH tables: quorum off works normally.
    fresh = f"{PREFIX}_ctl"
    kept_prefix, kept_zk = PREFIX, ZK
    PREFIX, ZK = fresh, f"/clickhouse/tables/{fresh}"
    create(client, "r1", "")
    create(client, "r2", "_peer")
    control = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(database="default", table_prefix=fresh)
    )
    print(f"PASS  quorum unset on fresh replicated tables allocates "
          f"{control.allocate_version()}")
    PREFIX, ZK = kept_prefix, kept_zk

    # 4. Turning quorum OFF on a catalog that had it ON is refused, loudly.
    #    Once a replicated table has taken a quorum insert, a later non-quorum
    #    insert is INVISIBLE to a select_sequential_consistency read (measured
    #    on 25.12: plain read 2 rows, sequential read 1). Every read-back here
    #    is a sequential read of the writer's own insert, so the protocol
    #    cannot confirm itself and gives up rather than proceeding blind.
    mixed = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(database="default", table_prefix=kept_prefix)
    )
    try:
        mixed.allocate_version()
        print("FAIL  quorum turned off mid-life went unnoticed")
    except Exception as exc:
        print(f"PASS  quorum turned off mid-life fails loudly: "
              f"{type(exc).__name__}")

    # 5. The negative control: with the peer gone, quorum cannot be met and the
    #    write has to FAIL rather than silently proceed. Last, because a failed
    #    quorum insert leaves state behind.
    for name in TABLES:
        client.execute(f"DETACH TABLE default.`{kept_prefix}_{name}_peer`")
    print("detached the peer replica; a quorum of 2 is now unsatisfiable")

    from time import monotonic
    started = monotonic()
    try:
        writer.allocate_version()
        print("FAIL  a quorum write succeeded with only one replica alive")
    except Exception as exc:
        elapsed = monotonic() - started
        print(f"PASS  quorum write refused after {elapsed:.1f}s: "
              f"{str(exc).splitlines()[0][:60]}")
        assert elapsed < 30, "insert_quorum_timeout is not bounded"

    for prefix in (kept_prefix, fresh):
        for name in TABLES:
            for suffix in ("_peer", ""):
                try:
                    client.execute(
                        f"ATTACH TABLE default.`{prefix}_{name}{suffix}`"
                    )
                except Exception:
                    pass
                client.execute(
                    f"DROP TABLE IF EXISTS default.`{prefix}_{name}{suffix}` SYNC"
                )
    print("cleaned up")


if __name__ == "__main__":
    main()
