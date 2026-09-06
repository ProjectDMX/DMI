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

# Module-level so the fake-S3 fixture re-exports into this module's
# namespace (function-local imports do not register fixtures).
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS, BUCKET, REGION, SECRET, fake_s3,
)

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


# --- B2: descriptor batches, replay guard, fenced publish -------------------
#
# The Python live suites drive these through client wrappers (a takeover
# wedged into a client's execute, a renewal that lapses, a transport that
# dies mid-publish). The native driver reproduces the same wedges through
# publish-op fields: takeover_after_renow / takeover_after_chunks perform a
# successor's claim in-process at those exact points, and
# inject_transport_error kills the publish after its renewal. The
# statements and the control flow are what is under test; the wedge
# mechanism is scaffolding, not behavior.


def _descriptor_dicts(count):
    from benchmarks.bench_capture_catalog import synthetic_descriptors

    out = []
    for item in synthetic_descriptors(count):
        meta = item.metadata
        loc = item.locator
        out.append({
            "capture_id": meta.capture_id, "tenant_id": meta.tenant_id,
            "experiment_id": meta.experiment_id, "run_id": meta.run_id,
            "session_id": meta.session_id, "request_id": meta.request_id,
            "sequence_id": meta.sequence_id, "model_id": meta.model_id,
            "model_revision": meta.model_revision,
            "adapter_revision": meta.adapter_revision,
            "capture_policy_version": meta.capture_policy_version,
            "hook_name": meta.hook_name, "layer_number": meta.layer_number,
            "producer_rank": meta.producer_rank, "step_number": meta.step_number,
            "token_start": meta.token_start, "token_end": meta.token_end,
            "batch_position": meta.batch_position, "dtype": meta.dtype,
            "shape": list(meta.shape), "captured_at_ns": meta.captured_at_ns,
            "pack_id": loc.pack_id, "store_id": loc.store_id,
            "object_key": loc.object_key, "object_bytes": loc.object_bytes,
            "pack_checksum": loc.pack_checksum,
            "pack_record_count": loc.pack_record_count,
            "payload_offset": loc.offset, "stored_length": loc.stored_length,
            "decoded_length": loc.decoded_length, "codec": loc.codec,
            "payload_checksum": loc.checksum,
        })
    return out


def _refs_of(count):
    # synthetic_descriptors stages every row into the same pack, so the
    # publish's membership is that one (store_id, pack_id) identity.
    from benchmarks.bench_capture_catalog import synthetic_descriptors

    loc = synthetic_descriptors(count)[0].locator
    return [{"store_id": loc.store_id, "pack_id": loc.pack_id}]


def test_descriptor_batches_match_the_reference_row_for_row():
    """B2a head-to-head: same descriptors, byte-identical capture_raw rows.

    The Python writer and the native driver each write the synthetic
    corpus into their own prefix; every column of every row — the
    MATERIALIZED facets included — must agree.
    """
    from benchmarks.bench_capture_catalog import synthetic_descriptors
    from dmi.storage.capture.clickhouse_catalog import (
        ClickHouseCatalogConfig,
        ClickHouseCatalogWriter,
    )

    with _catalog() as (client, config, prefix):
        native_prefix = prefix + "_nat"
        native_writer = ClickHouseCatalogWriter(
            client,
            ClickHouseCatalogConfig(database=config.database,
                                    table_prefix=native_prefix))
        try:
            native_writer.ensure_schema()
            descriptors = _descriptor_dicts(3)
            reference = ClickHouseCatalogWriter(client, config)
            reference.write_descriptors(list(synthetic_descriptors(3)),
                                        index_version=7)
            driver = CatalogDriver()
            try:
                _open(driver, native_prefix)
                driver.call(op="write_descriptors", descriptors=descriptors,
                            index_version=7)
            finally:
                driver.close()

            def rows(table_prefix):
                return client.execute(
                    f"SELECT * FROM `{config.database}`."
                    f"`{table_prefix}_capture_raw` ORDER BY ALL")

            assert rows(prefix) == rows(native_prefix), (
                "native descriptor rows diverge from the reference")
        finally:
            native_writer.drop_schema()


def test_commit_packs_and_the_committed_readback():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            version = driver.call(op="allocate_version")["version"]
            driver.call(op="commit_packs",
                        refs=_refs_of(3), index_version=version)
            committed = driver.call(
                op="committed_pack_ids", identities=_refs_of(3))
            assert committed["committed"] == _refs_of(3), committed
            # A pack never committed is not reported as committed.
            other = driver.call(
                op="committed_pack_ids",
                identities=[{"store_id": "garage",
                             "pack_id": str(uuid.uuid4())}])
            assert other["committed"] == []
        finally:
            driver.close()


def test_a_publish_below_the_published_head_loses_the_race():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="acquire", holder="writer")
            first = driver.call(op="allocate_version")["version"]
            second = driver.call(op="allocate_version")["version"]
            assert second > first
            driver.call(op="write_descriptors", descriptors=_descriptor_dicts(2),
                        index_version=second)
            driver.call(op="publish_snapshot", index_version=second,
                        refs=_refs_of(2), published_at_ns=second,
                        indexed_rows=2, indexed_packs=1)
            assert driver.call(op="last_published_version")["version"] == second

            lost = driver.call(op="publish_snapshot", index_version=first,
                               refs=_refs_of(2), published_at_ns=first,
                               indexed_rows=2, indexed_packs=1)
            assert not lost["ok"]
            assert lost["error"] == "SnapshotPublishRaceError", lost
            assert driver.call(op="last_published_version")["version"] == second
        finally:
            driver.close()


def test_a_repeated_identity_in_one_chunk_still_publishes():
    """The chunk read-back counts DISTINCT packs, so it must expect DISTINCT.

    ``publish_snapshot`` takes whatever membership its caller hands it, and
    the same ``(store_id, pack_id)`` may appear twice in one batch --
    ``CatalogIndexer`` deduplicates upstream, but the writer contract does
    not require it to, and the Python oracle compares the manifest's
    DISTINCT count against ``len(set(members))`` for exactly this reason.
    Comparing it against the chunk's LENGTH instead makes a duplicate look
    like a half-written manifest: the publish aborts as a lost race, having
    written a complete manifest chunk and no watermark, and the caller burns
    a version re-publishing a batch that was never wrong.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="acquire", holder="writer")
            version = driver.call(op="allocate_version")["version"]
            # One identity, handed over twice in a single chunk.
            duplicated = _refs_of(1) * 2
            published = driver.call(
                op="publish_snapshot", index_version=version,
                refs=duplicated, published_at_ns=version,
                indexed_rows=2, indexed_packs=1)

            assert published["ok"], published
            assert driver.call(op="last_published_version")["version"] == version
            # The duplicate collapses in the manifest rather than multiplying.
            members = client.execute(
                "SELECT DISTINCT store_id, toString(pack_id) FROM "
                f"`{config.database}`.`{prefix}_snapshot_manifest` "
                "WHERE index_version = %(version)s", {"version": version})
            assert len(members) == 1, members
        finally:
            driver.close()


def test_a_taken_over_publisher_writes_nothing_at_all():
    """The load-bearing fence: the check rides inside the write.

    A's takeover lands (in-process, as the Python wrapper does it) between
    its renewal and its fenced statements: the manifest INSERT is refused,
    the read-back sees zero rows, and the writer is fenced out with
    NOTHING written — not one inert manifest row.
    """
    with _catalog(lease_ttl_ns=2_000_000_000,
                  publish_timeout_ns=1_000_000_000) as (client, config, prefix):
        driver = CatalogDriver()
        try:
            # The stalled publisher runs on the fixture's short TTL — the
            # wedge depends on ITS lease lapsing, not the default 30 s.
            _open(driver, prefix, lease_ttl_ns=2_000_000_000,
                  publish_timeout_ns=1_000_000_000)
            driver.call(op="acquire", holder="stalled")
            pinned = driver.call(op="allocate_version")["version"]
            driver.call(op="write_descriptors", descriptors=_descriptor_dicts(3),
                        index_version=pinned)
            driver.call(op="publish_snapshot", index_version=pinned,
                        refs=_refs_of(3), published_at_ns=pinned,
                        indexed_rows=3, indexed_packs=1)
            before = _catalog_state(client, config)

            lost = driver.call(op="allocate_version")["version"]
            driver.call(op="write_descriptors", descriptors=_descriptor_dicts(2),
                        index_version=lost)
            refused = driver.call(
                op="publish_snapshot", index_version=lost, refs=_refs_of(2),
                published_at_ns=lost, indexed_rows=2, indexed_packs=1,
                wedge_ns=2_500_000_000, takeover_after_renew="successor")
            assert not refused["ok"]
            assert refused["error"] == "PublisherLeaseError", refused
            assert "fenced out and made no snapshot visible" in refused["message"]
            assert "'successor'" in refused["message"]

            assert _catalog_state(client, config) == before
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_snapshot_manifest` "
                f"WHERE index_version = {lost}") == [(0,)]
            # The descriptors it wrote ARE durable: invisible because
            # nothing admitted them, not because they are gone.
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_capture_raw`"
            ) == [(5,)]
            assert driver.call(
                op="last_published_version")["version"] == pinned
        finally:
            driver.close()


def _catalog_state(client, config):
    return {
        table: client.execute(
            f"SELECT * FROM `{config.database}`."
            f"`{config.table_prefix}_{table}` ORDER BY ALL")
        for table in ("index_watermark", "snapshot_manifest")
    }


def test_a_takeover_between_the_two_publish_statements_leaves_orphan_rows():
    """The guarantee is per STATEMENT, not per publish.

    The takeover wedged after the manifest INSERT leaves those rows behind
    while the watermark is refused. They are inert — membership pairs them
    with a watermark row of the same publish that will never exist — but
    they are durable, and that is what "writes nothing" gets wrong.
    """
    with _catalog(lease_ttl_ns=2_000_000_000,
                  publish_timeout_ns=1_000_000_000) as (client, config, prefix):
        driver = CatalogDriver()
        try:
            # The stalled publisher runs on the fixture's short TTL — the
            # wedge depends on ITS lease lapsing, not the default 30 s.
            _open(driver, prefix, lease_ttl_ns=2_000_000_000,
                  publish_timeout_ns=1_000_000_000)
            driver.call(op="acquire", holder="stalled")
            pinned = driver.call(op="allocate_version")["version"]
            driver.call(op="write_descriptors", descriptors=_descriptor_dicts(3),
                        index_version=pinned)
            driver.call(op="publish_snapshot", index_version=pinned,
                        refs=_refs_of(3), published_at_ns=pinned,
                        indexed_rows=3, indexed_packs=1)

            lost = driver.call(op="allocate_version")["version"]
            driver.call(op="write_descriptors", descriptors=_descriptor_dicts(2),
                        index_version=lost)
            refused = driver.call(
                op="publish_snapshot", index_version=lost, refs=_refs_of(2),
                published_at_ns=lost, indexed_rows=2, indexed_packs=1,
                wedge_ns=2_500_000_000, takeover_after_chunks="successor")
            assert not refused["ok"], refused
            # WHICH refusal depends on where the takeover lands: the
            # post-chunk renewal meets the successor (Held), the fenced
            # statements' read-back is fenced out (Lease). Both are the
            # PublisherLeaseError family; what this test is about is the
            # orphan rows either way.
            assert refused["error"] in ("PublisherLeaseError",
                                        "PublisherLeaseHeldError"), refused
            assert "successor" in refused["message"]

            manifest = f"`{config.database}`.`{prefix}_snapshot_manifest`"
            assert client.execute(
                f"SELECT count() FROM {manifest} WHERE index_version = {lost}"
            ) == [(1,)], "the manifest INSERT must have landed before the takeover"
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_index_watermark` "
                f"WHERE index_version = {lost}") == [(0,)]
            # And the public view agrees: unpaired membership admits nothing.
            assert client.execute(
                f"SELECT count() FROM `{config.database}`.`{prefix}_capture`"
            ) == [(3,)]
        finally:
            driver.close()


def test_an_outcome_unknown_failure_quarantines_the_writer():
    with _catalog(lease_ttl_ns=1_100_000_000,
                  publish_timeout_ns=1_000_000_000) as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix, lease_ttl_ns=1_100_000_000,
                  publish_timeout_ns=1_000_000_000)
            driver.call(op="acquire", holder="writer")
            refused = driver.call(
                op="publish_snapshot", index_version=11, refs=(),
                published_at_ns=11, indexed_rows=0, indexed_packs=0,
                inject_transport_error=True)
            assert not refused["ok"]
            assert refused["error"] == "ClickHouseError", refused
            # Quarantined: every publish-path entry is refused while the
            # server statements may still be running past their fences.
            assert driver.call(op="quarantined")["quarantined"] is True
            still = driver.call(op="acquire", holder="again")
            assert not still["ok"]
            assert still["error"] == "WriterQuarantinedError", still
            assert driver.call(op="renew")["error"] == "WriterQuarantinedError"
            # The quarantine window is the lease TTL: past it, a fresh
            # lease may be taken (the discarded row has expired too).
            sleep(1.2)
            fresh = driver.call(op="acquire", holder="after-window")
            assert fresh["ok"], fresh
        finally:
            driver.close()


# --- B3: the native indexer, end-to-end with the Python reader oracle -------

def _stage_via_sink(sink, root, index):
    """Stage one pack through the native sink; return its staged JSON."""
    from tests.test_native_uploader import _stage as uploader_stage
    return uploader_stage(sink, root, index)


def test_indexed_packs_become_visible_through_the_python_reader(fake_s3):
    """B3 e2e: sink → uploader → native index → Python CaptureReader.

    Two packs staged by the native sink, uploaded by the native uploader
    to the fake S3, read back through the native pack-index path, and
    published by the native writer — and the PYTHON reader must resolve
    every capture with its metadata intact. The second pass must skip
    everything through the replay guard and write nothing.
    """
    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _stage, _store_base,
    )

    with _catalog() as (client, config, prefix):
        sink = DriverSession(SINK_DRIVER)
        store = DriverSession(STORE_DRIVER)
        driver = CatalogDriver()
        spool_root = None
        try:
            import tempfile
            spool_root = Path(tempfile.mkdtemp()) / "spool"
            staged = [_stage(sink, spool_root, 40 + i) for i in range(2)]
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1,
            )
            assert uploaded["ok"], uploaded
            refs = uploaded["refs"]
            assert len(refs) == 2, refs
            for ref in refs:
                assert ref["pack_id"], ref

            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            result = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
            )
            if not result["ok"]:
                print("INDEX FAILED:", result["message"])
                raise AssertionError(result)
            r = result["result"]
            assert r["indexed_packs"] == 2, r
            assert r["indexed_rows"] == 2, r
            assert r["failed_packs"] == 0, r
            first_watermark = driver.call(
                op="last_published_version")["version"]
            assert first_watermark > 0

            # THE ORACLE: the Python reader resolves both captures.
            from dmi.storage.capture import CaptureQuery
            from dmi.storage.capture.clickhouse_reader import (
                ClickHouseCaptureCatalog,
                ClickHouseReaderConfig,
            )
            reader = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            page = reader.search(CaptureQuery(limit=10))
            assert len(page.items) == 2, page
            capture_ids = {item.capture_id for item in page.items}
            staged_ids = {f"upload-{40 + i:04d}" for i in range(2)}
            assert capture_ids == staged_ids, capture_ids
            for item in page.items:
                assert item.metadata.tenant_id == "t"
                assert item.metadata.dtype == "uint8"

            # The replay guard: a second pass skips everything, allocates
            # nothing, and moves no watermark.
            again = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
            )
            assert again["ok"], again
            assert again["result"]["skipped_packs"] == 2, again
            assert again["result"]["indexed_packs"] == 0, again
            assert driver.call(
                op="last_published_version")["version"] == first_watermark
        finally:
            sink.close()
            store.close()
            driver.close()


def test_index_without_a_lease_is_refused_before_any_writes(fake_s3):
    """Refused before allocating a version or writing descriptor rows.

    A readable pack is required: an all-packs-failed batch returns
    per-pack failures without reaching the lease check, exactly as the
    Python indexer does.
    """
    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _stage, _store_base,
    )

    with _catalog() as (client, config, prefix):
        sink = DriverSession(SINK_DRIVER)
        store = DriverSession(STORE_DRIVER)
        driver = CatalogDriver()
        try:
            import tempfile
            spool_root = Path(tempfile.mkdtemp()) / "spool"
            staged = _stage(sink, spool_root, 50)
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1,
            )
            assert uploaded["ok"], uploaded
            refs = uploaded["refs"]
            assert len(refs) == 1, refs

            _open(driver, prefix)
            before = driver.call(op="last_published_version")["version"]
            refused = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
            )
            assert not refused["ok"]
            assert refused["error"] == "PublisherLeaseError", refused
            assert "holds no publisher lease" in refused["message"]
            # Nothing was written: no version burned, no watermark moved.
            assert driver.call(
                op="last_published_version")["version"] == before
        finally:
            sink.close()
            store.close()
            driver.close()
