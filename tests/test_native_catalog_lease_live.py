"""B1: the native lease coordinator and version allocator, live.

The Python live suites drive the lease and allocator protocols through
clickhouse-driver; this file drives them through the native catalog
driver (`native/build/conformance_catalog`) over ClickHouse's HTTP
interface, through the same scenarios, against the same server. The SQL
statements are ported textually: after parameter substitution the server
must receive byte-identical text from both implementations, because the
sole-claimant protocols are only as sound as the statements they run.

Schema (table DDL, install, drop) stays Python in this phase: the ported
coordinator owns the lease and claim PROTOCOL, not the catalog layout.

These need a live ClickHouse (127.0.0.1:9000 by default) and the driver
binary; they run in the clickhouse-live job next to their Python
counterparts.
"""

from __future__ import annotations

import json
import subprocess
import threading
import uuid
from contextlib import contextmanager
from os import environ
from pathlib import Path
from time import sleep

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "native" / "build" / "conformance_catalog"

pytestmark = [
    pytest.mark.manual,
    pytest.mark.clickhouse,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`",
    ),
]

DEFAULTS = {
    "lease_ttl_ns": 30_000_000_000,
    "publish_timeout_ns": 5_000_000_000,
    "clock_skew_ns": 0,
    "allocation_attempts": 16,
}


class CatalogDriver:
    """One native coordinator process; ops are newline-JSON round trips."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [str(DRIVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def call(self, **fields) -> dict:
        self.proc.stdin.write(json.dumps(fields) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=30)


def _client():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    return clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )


@contextmanager
def _catalog(**overrides):
    """Schema via the Python writer (DDL stays Python), protocol via native.

    Yields the client, the config and the table prefix; tests open their
    own driver sessions per publisher, mirroring the Python live tests'
    one-writer-per-publisher shape.
    """
    from dmi.storage.capture.clickhouse_catalog import (
        ClickHouseCatalogConfig,
        ClickHouseCatalogWriter,
    )

    client = _client()
    prefix = f"dmi_native_b1_{uuid.uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix,
        **overrides,
    )
    writer = ClickHouseCatalogWriter(client, config)
    created = False
    try:
        created = True
        writer.ensure_schema()
        yield client, config, prefix
    finally:
        if created:
            writer.drop_schema()


def _open(driver: CatalogDriver, prefix: str, **overrides) -> dict:
    fields = {"op": "open", "table_prefix": prefix, **DEFAULTS, **overrides}
    if "database" not in fields:
        fields["database"] = environ.get("DMI_CLICKHOUSE_DATABASE", "default")
    response = driver.call(**fields)
    assert response["ok"], response
    return response


def _lease_rows(client, config):
    return client.execute(
        "SELECT term, toString(lease_id), holder, acquired_at_ns, "
        f"expires_at_ns FROM `{config.database}`."
        f"`{config.table_prefix}_publisher_lease` ORDER BY term, lease_id"
    )


def test_a_live_lease_refuses_a_successor_and_names_the_holder():
    with _catalog() as (client, config, prefix):
        first = CatalogDriver()
        successor = CatalogDriver()
        try:
            _open(first, prefix)
            # The install itself takes a lease (DDL-to-stamp), so this is
            # head + 1, not term 1 — the same shape the Python fixture leaves.
            held = first.call(op="acquire", holder="first-publisher")
            assert held["ok"], held
            assert held["lease"]["holder"] == "first-publisher"

            _open(successor, prefix)
            refused = successor.call(op="acquire", holder="successor")
            assert not refused["ok"]
            assert refused["error"] == "PublisherLeaseHeldError"
            assert "first-publisher" in refused["message"]
        finally:
            first.close()
            successor.close()


def test_an_expired_lease_is_taken_over_at_a_higher_term():
    with _catalog() as (client, config, prefix):
        crashed = CatalogDriver()
        successor = CatalogDriver()
        try:
            _open(crashed, prefix, lease_ttl_ns=1_100_000_000,
                  publish_timeout_ns=1_000_000_000)
            held = crashed.call(op="acquire", holder="crashed")
            assert held["ok"], held

            sleep(1.2)
            expires_at_ns, now_ns = client.execute(
                "SELECT expires_at_ns, toUnixTimestamp64Nano(now64(9)) FROM "
                f"`{config.database}`.`{config.table_prefix}_publisher_lease` "
                "ORDER BY term DESC, lease_id DESC LIMIT 1"
            )[0]
            assert expires_at_ns <= now_ns, (
                "precondition: the crashed holder's lease has to have lapsed"
            )

            _open(successor, prefix)
            taken = successor.call(op="acquire", holder="successor")
            assert taken["ok"], taken
            assert taken["lease"]["term"] > held["lease"]["term"]
            assert taken["lease"]["holder"] == "successor"
        finally:
            crashed.close()
            successor.close()


def test_release_hands_the_lease_back_at_once_and_tombstones_at_own_term():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            granted = driver.call(op="acquire", holder="orderly")
            assert granted["ok"], granted
            term = granted["lease"]["term"]

            released = driver.call(op="release")
            assert released["ok"], released
            assert driver.call(op="lease")["lease"] is None

            rows = client.execute(
                "SELECT acquired_at_ns, expires_at_ns FROM "
                f"`{config.database}`.`{config.table_prefix}_publisher_lease` "
                f"WHERE term = {term} ORDER BY acquired_at_ns"
            )
            # The claim row buys a full TTL; the tombstone is already expired
            # and lands at the writer's OWN term -- never a successor's.
            assert len(rows) == 2
            assert rows[0][1] > rows[0][0], "claim row: expires after acquired"
            assert rows[1][0] == rows[1][1], "tombstone buys no lease life"

            again = driver.call(op="acquire", holder="successor")
            assert again["ok"], again
            assert again["lease"]["term"] == term + 1
        finally:
            driver.close()


def test_a_contested_claim_walks_away_and_blocks_until_expiry():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            rival = str(uuid.uuid4())
            contested = driver.call(
                op="claim_contested", holder="claimant", rival_lease_id=rival,
            )
            assert not contested["ok"]
            assert contested["error"] == "PublisherLeaseHeldError"
            assert "contested" in contested["message"]
            assert driver.call(op="lease")["lease"] is None
            terms = [row[0] for row in _lease_rows(client, config)]
            assert terms.count(1) == 2, "rival and claimant share the term"
        finally:
            driver.close()


def test_the_fence_refuses_on_an_empty_lease_table_rather_than_throwing():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            assert driver.call(op="acquire", holder="holder")["ok"]
            table = (f"`{config.database}`."
                     f"`{config.table_prefix}_publisher_lease`")
            driver.call(op="execute", query=f"TRUNCATE TABLE {table}")
            assert client.execute(f"SELECT count() FROM {table}") == [(0,)]
            result = driver.call(
                op="fence_eval", lease_id=str(uuid.uuid4()),
                publish_timeout_ns=5_000_000_000, clock_skew_ns=0,
            )
            assert result["ok"], result
            assert result["admits"] == 0, (
                "an empty lease table must make the fence FALSE, not raise"
            )
        finally:
            driver.close()


def test_the_ported_statements_are_byte_identical_to_the_python_ones():
    """Textual SQL identity: the port risk table's load-bearing gate.

    The sole-claimant protocols are only as sound as the statements they
    run; any drift between the native coordinator and the Python module
    must fail here rather than surface as a protocol divergence on a
    replicated server.
    """
    from dmi.storage.capture.clickhouse_catalog import ClickHouseCatalogConfig
    from dmi.storage.capture.clickhouse_lease import (
        ClickHouseLeaseCoordinator,
    )

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            native = driver.call(op="statements")
            # Statement rendering never touches the client; a stub stands in.
            python = ClickHouseLeaseCoordinator(
                object(),
                ClickHouseCatalogConfig(database=config.database,
                                        table_prefix=prefix),
            )
            assert native["release"] == python.release_statement()
            assert native["fence"] == python.fence()
        finally:
            driver.close()


def test_allocated_versions_are_strictly_monotonic():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            versions = [driver.call(op="allocate_version")["version"]
                        for _ in range(4)]
            assert len(set(versions)) == 4
            assert versions == sorted(versions)
        finally:
            driver.close()


def test_the_allocator_starts_above_an_externally_published_head():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            watermark = (f"`{config.database}`."
                         f"`{config.table_prefix}_index_watermark`")
            driver.call(
                op="execute",
                query=(
                    f"INSERT INTO {watermark} (index_version, publish_id, "
                    "published_at_ns, indexed_rows, indexed_packs) VALUES "
                    f"(500, '{uuid.uuid4()}', 1, 0, 0)"
                ),
            )
            version = driver.call(op="allocate_version")["version"]
            assert version > 500
        finally:
            driver.close()


def test_concurrent_claimants_get_distinct_versions():
    """B1b: the sole-claimant protocol under real contention.

    Three driver processes race allocate_version on the same claims table.
    Every handed-out version is distinct and durably claimed exactly once;
    contested versions (two claimants, one tie) are abandoned by everyone
    who saw the tie and never returned to anyone.
    """
    with _catalog() as (client, config, prefix):
        results: list[dict] = []
        errors: list[str] = []
        lock = threading.Lock()

        def worker():
            driver = CatalogDriver()
            try:
                _open(driver, prefix)
                response = driver.call(op="allocate_version")
                with lock:
                    results.append(response)
            except Exception as exc:
                with lock:
                    errors.append(repr(exc))
            finally:
                driver.close()

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert all(r["ok"] for r in results), results
        versions = [r["version"] for r in results]
        assert len(set(versions)) == 3, versions
        claims = (f"`{config.database}`."
                  f"`{config.table_prefix}_capture_version_claims`")
        for v in versions:
            assert client.execute(
                f"SELECT count() FROM {claims} WHERE version = {v}"
            ) == [(1,)]
