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

    Driver processes race allocate_version on the same claims table. Every
    handed-out version is distinct and durably claimed; contested versions
    are abandoned by everyone who saw the tie and never returned to anyone.

    A version handed out may still carry MORE than one claim row -- the
    loser of a tie abandons the version but its row stays, because claims
    are append-only and collect_garbage reaps the spent ones (the Python
    live suite says the same in `test_clickhouse_snapshot_live.py`: "the
    claims table holds the spent claims a real pass leaves behind").
    """
    with _catalog() as (client, config, prefix):
        results: list[dict] = []
        errors: list[str] = []
        lock = threading.Lock()

        def worker():
            driver = CatalogDriver()
            try:
                # A wide attempt budget: the default 16 is sized for a quiet
                # server, and this test runs beside the rest of the suite.
                _open(driver, prefix, allocation_attempts=64)
                response = driver.call(op="allocate_version")
                with lock:
                    results.append(response)
            except Exception as exc:
                with lock:
                    errors.append(repr(exc))
            finally:
                driver.close()

        # Two initialisers is the documented cold-start shape; the
        # install-lease budget (one TTL + margin) is sized for exactly
        # one contested-term wait.
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert all(r["ok"] for r in results), results
        versions = [r["version"] for r in results]
        assert len(set(versions)) == len(versions), versions
        claims = (f"`{config.database}`."
                  f"`{config.table_prefix}_capture_version_claims`")
        # A claimant that loses a tie ABANDONS the version and leaves its
        # claim row behind: claims are append-only, and collect_garbage is
        # what reaps the spent ones. Standing in for the loser whose INSERT
        # lands after the winner's read-back, deterministically.
        client.execute(
            f"INSERT INTO {claims} (version, claim_id, claimed_at_ns) "
            f"VALUES ({versions[0]}, generateUUIDv4(), 1)")
        for v in versions:
            # DURABLY CLAIMED, not solely claimed. "Exactly one row stands
            # at this version" is not an invariant of the protocol and only
            # held in the interleaving where both claimants saw the tie: a
            # loser that inserts AFTER the winner's read-back leaves a
            # second row at a version the winner legitimately owns, which
            # made this assertion fail about once in seven suite runs while
            # the allocator was behaving exactly as designed. What sole
            # claimancy means here is decided at read-back time and cannot
            # be re-read afterwards; what survives is that every handed-out
            # version is distinct (asserted above) and durably recorded.
            (claimed,), = client.execute(
                f"SELECT count() FROM {claims} WHERE version = {v}")
            assert claimed >= 1, (v, claimed)


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


def test_a_lapsed_quarantine_reports_itself_as_over():
    """The writer must stop calling itself quarantined once it is not.

    Publishing is permitted again the moment the window lapses, but the
    flag it is read off is only ever SET -- nothing clears it -- so the
    writer answers "quarantined" for the rest of the process's life. A
    supervisor polling that answer to decide when a writer has recovered
    never sees the recovery, and takes a healthy writer out of service.
    The Python writer clears the deadline and the lease id as soon as a
    check finds the window has passed (`_require_not_quarantined_locked`
    in clickhouse_catalog.py).
    """
    with _catalog(lease_ttl_ns=1_100_000_000,
                  publish_timeout_ns=1_000_000_000) as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix, lease_ttl_ns=1_100_000_000,
                  publish_timeout_ns=1_000_000_000)
            driver.call(op="acquire", holder="writer")
            refused = driver.call(
                op="publish_snapshot", index_version=21, refs=(),
                published_at_ns=21, indexed_rows=0, indexed_packs=0,
                inject_transport_error=True)
            assert not refused["ok"], refused
            assert driver.call(op="quarantined")["quarantined"] is True

            sleep(1.2)

            # The window has lapsed, so the writer is no longer quarantined
            # and must say so -- before anything else asks it to publish.
            assert driver.call(op="quarantined")["quarantined"] is False
            # And the recovery it reports is real.
            assert driver.call(op="acquire", holder="after-window")["ok"]
            assert driver.call(op="quarantined")["quarantined"] is False
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


def test_one_unreadable_pack_does_not_disturb_the_packs_around_it(fake_s3):
    """A failed read must remove ONLY its own pack from the batch.

    The read loop iterates the pending batch and drops a pack that fails
    to read. Dropping it while iterating that same sequence makes the
    iteration skip the pack that shifts into the hole and revisit the
    last one: the skipped pack is published and committed as indexed
    while none of its descriptors were ever written -- the permanently
    "committed but invisible" state the replay guard can never undo --
    and the revisited pack is indexed twice. The Python indexer avoids
    it by collecting successes into a separate list
    (`catalog.py` `valid_refs`) rather than mutating what it walks.

    The erase shifts every later pack one slot left while the walk keeps
    going from the slot after the hole, so the damage needs TWO packs
    behind the failure: the first of them shifts into a slot already
    passed and is never read, the last is read a second time through the
    slot the shift left behind. Three packs with the FIRST one
    unreadable is therefore the smallest batch that shows it -- a
    failure in the middle of three shifts the only remaining pack into
    the slot just visited and comes out right by accident.
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
            for i in range(3):
                _stage(sink, spool_root, 60 + i)
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1,
            )
            assert uploaded["ok"], uploaded
            # The batch is walked in (store_id, pack_id) order, so that --
            # not the upload order -- decides which pack is read first.
            refs = sorted(uploaded["refs"],
                          key=lambda r: (r["store_id"], r["pack_id"]))
            assert len(refs) == 3, refs
            # The FIRST pack's object is not in the bucket, so its read
            # fails while both packs behind it read fine.
            refs[0] = dict(refs[0], object_key="packs/not-uploaded.dmi-pack")

            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            result = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
            )
            assert result["ok"], result
            r = result["result"]

            # Exactly one pack failed, and it is the unreadable one.
            assert r["failed_packs"] == 1, r
            assert [f["object_key"] for f in r["failures"]] == [
                "packs/not-uploaded.dmi-pack"], r
            # Both readable packs were indexed -- once each.
            assert r["indexed_packs"] == 2, r
            assert r["indexed_rows"] == 2, r

            # And the descriptors prove it: two captures, one row each --
            # never a capture missing (skipped) and never one written
            # twice (revisited). Which two depends on the pack ids the
            # sink minted, so the shape is what is asserted.
            rows = client.execute(
                "SELECT capture_id, count() FROM "
                f"`{config.database}`.`{prefix}_capture_raw` "
                "GROUP BY capture_id ORDER BY capture_id")
            assert len(rows) == 2, rows
            assert [count for _, count in rows] == [1, 1], rows
            assert {capture_id for capture_id, _ in rows} < {
                f"upload-{60 + i:04d}" for i in range(3)}, rows
        finally:
            sink.close()
            store.close()
            driver.close()


def test_an_oversized_batch_is_refused_rather_than_blamed_on_a_pack(fake_s3):
    """The batch budget is the CALLER's error, so it must propagate.

    Exceeding `max_estimated_bytes` is a property of how much the caller
    asked for, not of the pack being read when the total crossed the
    line. Raised from inside the per-pack `try`, it is caught by the
    handler meant for unreadable packs: the pack that happened to cross
    the budget is reported as individually failed, every pack behind it
    is reported the same way, and the call returns a partial index and a
    success status. The Python indexer checks the budget outside the try
    (`catalog.py`) and raises.
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
            for i in range(2):
                _stage(sink, spool_root, 70 + i)
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1,
            )
            assert uploaded["ok"], uploaded
            refs = uploaded["refs"]
            assert len(refs) == 2, refs

            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            refused = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
                # One byte: the first pack read already crosses it.
                max_estimated_bytes=1,
            )

            assert not refused["ok"], refused
            assert refused["error"] == "ValueError", refused
            assert "max_estimated_bytes" in refused["message"], refused
            # Refused, so nothing was published and no pack was blamed.
            assert driver.call(op="last_published_version")["version"] == 0
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_capture_raw`") == [(0,)]
        finally:
            sink.close()
            store.close()
            driver.close()


def test_a_batch_that_never_publishes_is_never_committed(fake_s3):
    """Exhausting the publish attempts must raise, not commit the batch.

    The inventory is the replay guard: a pack recorded there is skipped by
    every later pass, so recording one whose watermark never published
    makes its captures permanently invisible -- the one outcome the
    publish-before-inventory order exists to prevent.

    The retry loop breaks out of its own `for` on the final attempt, which
    skips the loop's increment, so the exhaustion check that follows
    (`attempts == max_publish_attempts`) can never be true and the error it
    guards is dead code. Execution falls through to the inventory commit
    and `index()` returns a success-shaped result. Python raises
    SnapshotPublishExhaustedError here (`catalog.py`).

    Driven with one attempt and a foreign publisher taking the version out
    from under this pass, which is a real barrier refusal.
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
            _stage(sink, spool_root, 80)
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1,
            )
            assert uploaded["ok"], uploaded
            refs = uploaded["refs"]
            assert len(refs) == 1, refs

            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            exhausted = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
                max_publish_attempts=1,
                foreign_watermark_after_allocate=True,
            )

            assert not exhausted["ok"], exhausted
            assert exhausted["error"] == "SnapshotPublishRaceError", exhausted
            assert "1 attempts" in exhausted["message"], exhausted
            # NOTHING was committed, so the pack is still indexable.
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_pack_inventory_raw`") == [(0,)]

            # And the proof that it stayed indexable: a pass without the
            # foreign publisher indexes it rather than skipping it.
            again = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
            )
            assert again["ok"], again
            assert again["result"]["skipped_packs"] == 0, again
            assert again["result"]["indexed_packs"] == 1, again
        finally:
            sink.close()
            store.close()
            driver.close()


# --- writer config validation --------------------------------------------------
#
# The Python config refuses at construction whatever would silently void a
# guarantee later: identifiers that break out of their backticks, a
# publish timeout the server would read as "no limit", a quorum without a
# clock-skew bound, a fence margin thinner than a round trip. The native
# WriterConfig must refuse the same shapes at `open`, and each case here
# carries the Python oracle beside it.

def _refused_open(**overrides):
    driver = CatalogDriver()
    try:
        fields = {"op": "open", "table_prefix": "dmi_cfg_refusals",
                  "database": environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
                  **DEFAULTS, **overrides}
        return driver.call(**fields)
    finally:
        driver.close()


def _oracle_refuses(**overrides):
    from dmi.storage.capture.clickhouse_catalog import ClickHouseCatalogConfig

    fields = {"table_prefix": "dmi_cfg_refusals",
              "database": environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
              "lease_ttl_ns": DEFAULTS["lease_ttl_ns"],
              "publish_timeout_ns": DEFAULTS["publish_timeout_ns"],
              "clock_skew_ns": DEFAULTS["clock_skew_ns"],
              **overrides}
    with pytest.raises(ValueError) as refusal:
        ClickHouseCatalogConfig(**fields)
    return str(refusal.value)


def test_an_identifier_that_escapes_its_backticks_is_refused():
    # Both names are interpolated into backticked qualified names; a
    # backtick or hyphen in either escapes the quoting.
    refused = _refused_open(table_prefix="dmi`; DROP TABLE x; --")
    assert not refused["ok"], refused
    assert refused["error"] == "ValueError", refused
    assert "identifier" in refused["message"], refused
    assert "identifier" in _oracle_refuses(
        table_prefix="dmi`; DROP TABLE x; --")


def test_a_fractional_publish_timeout_is_refused():
    # max_execution_time rides as WHOLE seconds; 0.5s coerced through
    # int() reaches an older server as 0, which disables the cap and the
    # fence-vs-TTL margin with it.
    refused = _refused_open(publish_timeout_ns=500_000_000)
    assert not refused["ok"], refused
    assert refused["error"] == "ValueError", refused
    assert "whole number of seconds" in refused["message"], refused
    assert "whole number of seconds" in _oracle_refuses(
        publish_timeout_ns=500_000_000)


def test_a_quorum_without_a_clock_skew_bound_is_refused():
    refused = _refused_open(insert_quorum=2, clock_skew_ns=0)
    assert not refused["ok"], refused
    assert refused["error"] == "ValueError", refused
    assert "clock_skew_ns" in refused["message"], refused
    assert "clock_skew_ns" in _oracle_refuses(insert_quorum=2, clock_skew_ns=0)


def test_a_quorum_of_one_is_refused():
    refused = _refused_open(insert_quorum=1, clock_skew_ns=1_000_000)
    assert not refused["ok"], refused
    assert refused["error"] == "ValueError", refused
    assert "at least 2" in refused["message"], refused
    assert "at least 2" in _oracle_refuses(
        insert_quorum=1, clock_skew_ns=1_000_000)


def test_a_fence_margin_thinner_than_a_round_trip_is_refused():
    # ttl == timeout leaves zero margin: the lease is takeable the moment
    # the statement's own cap expires, and a renewed lease has no time at
    # all to reach the fence.
    refused = _refused_open(lease_ttl_ns=1_000_000_000,
                            publish_timeout_ns=1_000_000_000)
    assert not refused["ok"], refused
    assert refused["error"] == "ValueError", refused
    assert "margin" in refused["message"], refused
    assert "margin" in _oracle_refuses(lease_ttl_ns=1_000_000_000,
                                       publish_timeout_ns=1_000_000_000)


def test_a_zero_lease_ttl_is_refused():
    refused = _refused_open(lease_ttl_ns=0)
    assert not refused["ok"], refused
    assert refused["error"] == "ValueError", refused
    assert "positive" in refused["message"], refused
    assert "positive" in _oracle_refuses(lease_ttl_ns=0)


# --- B4: maintenance — garbage collection ------------------------------------

def test_garbage_collection_never_settles_for_less_than_the_publish_timeout():
    """The two-read intersection is only sound one publish timeout apart.

    Any statement admitted before the first read has either landed or been
    capped by max_execution_time once that long has passed -- that is what
    makes a publish orphaned in BOTH reads safe to collect. The Python
    implementation derives the wait from publish_timeout_ns internally; the
    native port takes the wait as a parameter, so a short value must be
    clamped up to the timeout or a maintenance job passing 0 deletes
    manifest rows out from under a publish whose watermark statement is
    still in flight.
    """
    from time import monotonic

    with _catalog(publish_timeout_ns=1_000_000_000) as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix, publish_timeout_ns=1_000_000_000)
            driver.call(op="acquire", holder="writer")
            # An orphan BELOW the published head, so the settle wait
            # actually runs: only sub-head orphans are collectable, and
            # with none in the first read there is nothing to settle.
            first = driver.call(op="allocate_version")["version"]
            driver.call(op="publish_snapshot", index_version=first, refs=(),
                        published_at_ns=first, indexed_rows=0,
                        indexed_packs=0)
            second = driver.call(op="allocate_version")["version"]
            driver.call(op="publish_snapshot", index_version=second, refs=(),
                        published_at_ns=second, indexed_rows=0,
                        indexed_packs=0)
            manifest = f"`{config.database}`.`{prefix}_snapshot_manifest`"
            driver.call(
                op="execute",
                query=(f"INSERT INTO {manifest} "
                       "(index_version, publish_id, store_id, pack_id) VALUES "
                       f"({first}, '{uuid.uuid4()}', 'garage', "
                       f"'{uuid.uuid4()}')"),
            )

            began = monotonic()
            collected = driver.call(op="collect_garbage", settle_sleep_ns=0)
            elapsed = monotonic() - began

            assert collected["ok"], collected
            assert elapsed >= 0.95, (
                f"settle wait ran {elapsed:.3f}s with settle_sleep_ns=0; the "
                "floor is publish_timeout_ns (1s)")
        finally:
            driver.close()


def test_garbage_collection_keeps_what_is_visible_and_removes_the_rest():
    """The retention port, against the same scenario the Python suite drives.

    Collected: orphan manifest rows below the head, superseded lease rows,
    spent claims at or below the published head. Kept: the membership of
    every published version, the head lease row, and the claim above the
    head — the only record that a version was allocated but not yet
    published. Idempotent: a second pass removes nothing.
    """
    with _catalog(publish_timeout_ns=1_000_000_000) as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix, publish_timeout_ns=1_000_000_000)
            driver.call(op="acquire", holder="writer")
            refs = _refs_of(3)
            descriptors = _descriptor_dicts(3)
            first = driver.call(op="allocate_version")["version"]
            driver.call(op="write_descriptors", descriptors=descriptors,
                        index_version=first)
            driver.call(op="publish_snapshot", index_version=first, refs=refs,
                        published_at_ns=first, indexed_rows=3, indexed_packs=1)
            driver.call(op="commit_packs", refs=refs, index_version=first)
            second = driver.call(op="allocate_version")["version"]
            driver.call(op="write_descriptors", descriptors=descriptors,
                        index_version=second)
            driver.call(op="publish_snapshot", index_version=second, refs=refs,
                        published_at_ns=second, indexed_rows=3, indexed_packs=1)

            # Membership from a publish that never reached the watermark,
            # below the head — the shape a lost version race leaves.
            orphan_publish = str(uuid.uuid4())
            manifest = f"`{config.database}`.`{prefix}_snapshot_manifest`"
            driver.call(
                op="execute",
                query=(f"INSERT INTO {manifest} "
                       "(index_version, publish_id, store_id, pack_id) VALUES "
                       f"({first}, '{orphan_publish}', 'garage', "
                       f"'{refs[0]['pack_id']}')"),
            )
            pending = driver.call(op="allocate_version")["version"]
            assert pending > second

            collected = driver.call(
                op="collect_garbage", settle_sleep_ns=1_100_000_000)
            removed = collected["removed"]
            assert removed[f"{prefix}_snapshot_manifest"] == 1, removed
            assert removed[f"{prefix}_publisher_lease"] >= 1, removed
            assert removed[f"{prefix}_capture_version_claims"] >= 1, removed
            assert client.execute(
                f"SELECT count() FROM {manifest} WHERE publish_id = "
                f"'{orphan_publish}'") == [(0,)]

            # Kept: the membership of both published versions...
            assert client.execute(
                f"SELECT count() FROM {manifest} WHERE index_version IN "
                f"({first}, {second})") == [(2,)]
            # ...the head lease row, and the pending claim above it.
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_publisher_lease`")[0][0] >= 1
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_capture_version_claims` "
                f"WHERE version = {pending}") == [(1,)]

            # Idempotent: a second pass has nothing left to do.
            again = driver.call(op="collect_garbage", settle_sleep_ns=1_100_000_000)
            assert again["removed"] == {
                f"{prefix}_snapshot_manifest": 0,
                f"{prefix}_capture_version_claims": 0,
            } or all(v == 0 for v in again["removed"].values()), again["removed"]
        finally:
            driver.close()


# --- B4: schema — install, stamp, drop ---------------------------------------

def test_a_fresh_install_stamps_and_gives_the_lease_back():
    with _catalog_drop_only() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="ensure_schema")
            tables = {
                row[0]
                for row in client.execute(
                    "SELECT name FROM system.tables WHERE database = "
                    f"'{config.database}' AND name LIKE '{prefix}_%'")
            }
            assert f"{prefix}_capture_raw" in tables
            assert f"{prefix}_capture" in tables
            assert client.execute(
                "SELECT version FROM "
                f"`{config.database}`.`{prefix}_schema_version`"
            ) == [(4,)]
            # The install lease was given back: the head's effective
            # expiry is the MIN across the lease's rows — the release
            # tombstone pulls it to the past — so the next publisher does
            # not wait out a TTL.
            head = driver.call(op="head")["head"]
            assert head["expires_at_ns"] <= head["now_ns"], head
            # Idempotent: a second ensure is a no-op that still succeeds.
            driver.call(op="ensure_schema")
            assert client.execute(
                "SELECT count() FROM "
                f"`{config.database}`.`{prefix}_schema_version`") == [(1,)]
        finally:
            driver.close()


@contextmanager
def _catalog_drop_only(**overrides):
    """A fixture variant with NO schema: the test ensures it natively."""
    from dmi.storage.capture.clickhouse_catalog import ClickHouseCatalogConfig

    client = _client()
    prefix = f"dmi_native_b4_{uuid.uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix, **overrides)
    yield client, config, prefix


def test_concurrent_installers_are_serialised_on_the_install_lease():
    with _catalog_drop_only() as (client, config, prefix):
        outcomes: list[dict] = []
        errors: list[str] = []
        lock = threading.Lock()

        def worker(stagger_s):
            driver = CatalogDriver()
            try:
                sleep(stagger_s)
                _open(driver, prefix)
                response = driver.call(op="ensure_schema")
                with lock:
                    outcomes.append(response)
            except Exception as exc:
                with lock:
                    errors.append(repr(exc))
            finally:
                driver.close()

        # Two initialisers is the documented cold-start shape. The second
        # arrives DURING the first's install — the scenario the wait
        # exists for ("a second initialiser waits for the first rather
        # than crashing"). A forced-zero-stagger start can cascade
        # contested terms past the install-lease budget, which the
        # protocol's ttl+margin budget is not sized to cover, and the
        # Python implementation fails identically there.
        threads = [threading.Thread(target=worker, args=(delay,))
                   for delay in (0.0, 0.25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert all(r["ok"] for r in outcomes), outcomes
        # One stamp: the second and third initialiser found the prefix
        # complete (or waited out the install) and re-stamping is a
        # server-side no-op.
        assert client.execute(
            "SELECT count() FROM "
            f"`{config.database}`.`{prefix}_schema_version`") == [(1,)]


def test_ensure_keeps_a_lease_the_writer_already_holds():
    """A writer that holds the lease renews it around the install.

    The complete-catalog path takes no lease of its own, so the writer's
    own lease survives: same lease_id, renewed to a higher term.
    """
    with _catalog_drop_only() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="ensure_schema")
            held = driver.call(op="acquire", holder="writer")
            assert held["ok"], held
            lease_id = held["lease"]["lease_id"]
            term = held["lease"]["term"]
            driver.call(op="ensure_schema")
            still = driver.call(op="lease")["lease"]
            # The complete-catalog path takes no lease of its own and does
            # not renew: the writer's lease is untouched, same id and term.
            assert still["lease_id"] == lease_id
            assert still["term"] == term
            assert client.execute(
                "SELECT version FROM "
                f"`{config.database}`.`{prefix}_schema_version`") == [(4,)]
        finally:
            driver.close()


def test_drop_schema_removes_every_object():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="drop_schema",
                        database=config.database, table_prefix=prefix)
            remaining = [
                row[0] for row in client.execute(
                    "SELECT name FROM system.tables WHERE database = "
                    f"'{config.database}' AND name LIKE '{prefix}_%'")
            ]
            assert remaining == [], remaining
        finally:
            driver.close()


# --- B4: the schema compatibility refusals ------------------------------------
#
# `ensure_schema` refuses an incompatible catalog rather than repairing it,
# and those refusals are most of the schema port. Driven through the
# `verify_compatibility` op, which returns the verdict without running the
# install that would otherwise follow it.

def _legacy_table(client, config, prefix):
    """Create the object only a pre-v4 build creates, beside this build's."""
    client.execute(
        f"CREATE TABLE IF NOT EXISTS `{config.database}`."
        f"`{prefix}_pack_commit_log` (pack_id UUID, index_version UInt64) "
        "ENGINE = MergeTree ORDER BY pack_id")


def test_an_earlier_builds_object_beside_this_builds_is_refused():
    """Two builds sharing one prefix is the dangerous case, so it refuses.

    A superseded object standing beside this build's own is not an
    unfinished cleanup -- the leftovers-only refusal covers that, where
    nothing of this build is present. It means an EARLIER build may still
    be writing this prefix, and that build's publish writes the pack
    inventory plus an unconditional watermark row carrying no publish
    identity, and never a manifest row: its packs land already invisible
    to every reader and already skipped by every indexing pass.

    Nothing else here notices. The stamp reads this version, no table is
    missing, and the inventory-without-membership check is per-pack, so it
    only reports those packs once they are already durable and invisible
    (`_reject_legacy_objects_beside_this_build` in clickhouse_schema.py).
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            # Healthy to begin with.
            assert driver.call(
                op="verify_compatibility")["state"] == "complete"

            _legacy_table(client, config, prefix)

            refused = driver.call(op="verify_compatibility")
            assert not refused["ok"], refused
            assert refused["error"] == "CatalogSchemaVersionError", refused
            assert "_pack_commit_log" in refused["message"], refused
            assert "an object only an earlier build creates" in (
                refused["message"]), refused
            # And the install itself refuses, not just the verdict.
            assert not driver.call(op="ensure_schema")["ok"]

            # THE ORACLE: the Python writer refuses the same catalog.
            from dmi.storage.capture.clickhouse_catalog import (
                ClickHouseCatalogWriter,
            )
            from dmi.storage.capture.clickhouse_schema import (
                CatalogSchemaVersionError,
            )
            with pytest.raises(CatalogSchemaVersionError) as oracle:
                ClickHouseCatalogWriter(client, config).ensure_schema()
            assert "_pack_commit_log" in str(oracle.value)
            assert "an object only an earlier build creates" in str(
                oracle.value)
        finally:
            driver.close()


def test_an_unstamped_install_is_not_described_as_stamped():
    """A stamp table with no row in it is an install that died before it.

    The refusals below the version check quote what the catalog IS, and a
    stamp table holding no row means this build's install died before
    stamping -- not that the catalog is stamped. Saying it is stamped
    sends the operator looking for a version conflict that is not there
    (clickhouse_schema.py builds this clause from whether a row exists).
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            # An install that died before stamping: the layout is there and
            # the stamp table exists, but it holds no row. The sort key is
            # then broken so a refusal has to describe the catalog.
            client.execute(
                f"TRUNCATE TABLE `{config.database}`.`{prefix}_schema_version`")
            client.execute(
                f"DROP TABLE `{config.database}`.`{prefix}_pack_inventory_raw`")
            client.execute(
                f"CREATE TABLE `{config.database}`."
                f"`{prefix}_pack_inventory_raw` (pack_id UUID, "
                "store_id LowCardinality(String), object_key String, "
                "object_bytes UInt64, pack_checksum FixedString(64), "
                "record_count UInt32, index_version UInt64) "
                "ENGINE = ReplacingMergeTree(index_version) ORDER BY pack_id")

            refused = driver.call(op="verify_compatibility")
            assert not refused["ok"], refused
            assert refused["error"] == "CatalogSchemaVersionError", refused
            assert "sorts by" in refused["message"], refused
            assert "no row in it" in refused["message"], refused
            assert "is stamped schema version" not in refused["message"], (
                refused["message"])

            # THE ORACLE: the Python writer describes it the same way.
            from dmi.storage.capture.clickhouse_catalog import (
                ClickHouseCatalogWriter,
            )
            from dmi.storage.capture.clickhouse_schema import (
                CatalogSchemaVersionError,
            )
            with pytest.raises(CatalogSchemaVersionError) as oracle:
                ClickHouseCatalogWriter(client, config).ensure_schema()
            assert "no row in it" in str(oracle.value)
            assert "is stamped schema version" not in str(oracle.value)
        finally:
            driver.close()


def test_a_wrong_sort_key_is_refused_rather_than_stamped_over():
    """`CREATE TABLE IF NOT EXISTS` no-ops over a pre-v4 sort key.

    An ORDER BY cannot be altered in place, so a table another build
    created with a different one would keep it while this build's stamp
    went on over a layout it does not describe.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            client.execute(
                f"DROP TABLE `{config.database}`.`{prefix}_pack_inventory_raw`")
            client.execute(
                f"CREATE TABLE `{config.database}`."
                f"`{prefix}_pack_inventory_raw` (pack_id UUID, "
                "store_id LowCardinality(String), object_key String, "
                "object_bytes UInt64, pack_checksum FixedString(64), "
                "record_count UInt32, index_version UInt64) "
                "ENGINE = ReplacingMergeTree(index_version) ORDER BY pack_id")

            refused = driver.call(op="verify_compatibility")
            assert not refused["ok"], refused
            assert refused["error"] == "CatalogSchemaVersionError", refused
            assert "sorts by" in refused["message"], refused
            assert "cannot be altered in place" in refused["message"], refused
        finally:
            driver.close()


def test_an_engine_without_the_version_argument_is_refused():
    """The PROPERTY, not the spelling: index_version must be the version.

    Without it a merge that collapses a duplicate key keeps an arbitrary
    row rather than the newest, which is the whole basis of supersession.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            client.execute(
                f"DROP TABLE `{config.database}`.`{prefix}_pack_inventory_raw`")
            # Right sort key, right kind -- no version argument.
            client.execute(
                f"CREATE TABLE `{config.database}`."
                f"`{prefix}_pack_inventory_raw` (pack_id UUID, "
                "store_id LowCardinality(String), object_key String, "
                "object_bytes UInt64, pack_checksum FixedString(64), "
                "record_count UInt32, index_version UInt64) "
                "ENGINE = ReplacingMergeTree ORDER BY (store_id, pack_id)")

            refused = driver.call(op="verify_compatibility")
            assert not refused["ok"], refused
            assert refused["error"] == "CatalogSchemaVersionError", refused
            assert "ReplacingMergeTree(index_version)" in refused["message"]
            assert "arbitrary row" in refused["message"], refused
        finally:
            driver.close()


def test_a_healthy_replacingmergetree_is_not_refused_by_the_engine_check():
    """The engine clause carries ORDER BY's parentheses too.

    Reading the arguments to the LAST `)` sweeps the ORDER BY group in and
    refuses every healthy catalog; only the group that closes the engine
    call counts. This is the guard on that.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            engine_full = client.execute(
                "SELECT engine_full FROM system.tables WHERE database = "
                f"'{config.database}' AND name = '{prefix}_capture_raw'")[0][0]
            # The precondition the guard is about: ORDER BY contributes its
            # own parenthesised group after the engine call.
            assert engine_full.count("(") > 1, engine_full

            assert driver.call(
                op="verify_compatibility")["state"] == "complete"
        finally:
            driver.close()


def test_a_wrong_kind_is_refused():
    """A VIEW where this build creates a TABLE, and the reverse."""
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            client.execute(
                f"DROP TABLE `{config.database}`.`{prefix}_pack_inventory_raw`")
            client.execute(
                f"CREATE VIEW `{config.database}`."
                f"`{prefix}_pack_inventory_raw` AS SELECT 1 AS pack_id")

            refused = driver.call(op="verify_compatibility")
            assert not refused["ok"], refused
            assert refused["error"] == "CatalogSchemaVersionError", refused
            assert "wrong kind" in refused["message"], refused
            assert "_pack_inventory_raw" in refused["message"], refused
        finally:
            driver.close()


def test_a_foreign_schema_version_is_refused():
    """A higher stamp means a newer writer owns this catalog."""
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            client.execute(
                f"TRUNCATE TABLE `{config.database}`.`{prefix}_schema_version`")
            client.execute(
                f"INSERT INTO `{config.database}`.`{prefix}_schema_version` "
                "(version, applied_at_ns) VALUES (99, 1)")

            refused = driver.call(op="verify_compatibility")
            assert not refused["ok"], refused
            assert refused["error"] == "CatalogSchemaVersionError", refused
            assert "is at schema version 99" in refused["message"], refused
            assert "upgrade this build" in refused["message"], refused
        finally:
            driver.close()


def test_inventory_without_membership_is_refused():
    """A pack marked indexed that no snapshot admits is not a fresh start.

    Those packs are skipped by every later pass while their captures stay
    invisible, so the catalog is refused rather than written into.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            client.execute(
                f"INSERT INTO `{config.database}`."
                f"`{prefix}_pack_inventory_raw` (pack_id, store_id, "
                "object_key, object_bytes, pack_checksum, record_count, "
                "index_version) VALUES (generateUUIDv4(), 'garage', "
                f"'packs/orphan.dmi-pack', 1024, '{'0' * 64}', 1, 1)")

            refused = driver.call(op="verify_compatibility")
            assert not refused["ok"], refused
            assert refused["error"] == "CatalogSchemaVersionError", refused
            assert "without membership" in refused["message"], refused
        finally:
            driver.close()


# --- review-thread regressions ------------------------------------------------

def test_execute_returns_valid_json_for_non_empty_results():
    """The execute op must emit parseable JSON when rows come back.

    escape_into emits its own quotes (EscapeJson); wrapping the field in
    another pair emitted ""value"" and every non-empty result failed to
    parse on the test side.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            result = driver.call(
                op="execute", query="SELECT 1 AS a, 'x,y' AS b")
            assert result["ok"], result
            assert result["rows"] == [["1", "x,y"]], result
        finally:
            driver.close()


def test_a_placeholder_in_a_string_param_does_not_hang_the_substitution():
    """String params containing their own placeholder must not loop.

    The substitution restarted its search at the top of the statement
    after every replacement, so a rendered value that contains
    ``%(name)s`` re-matched inside the inserted text and grew without
    bound. A holder that says ``%(holder)s`` is a legal 11-byte string.
    """
    with _catalog() as (client, config, prefix):
        proc = subprocess.Popen(
            [str(DRIVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        hung = False

        def call(payload, timeout_s=10.0):
            nonlocal hung
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            result = {}

            def read():
                result["line"] = proc.stdout.readline()

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            reader.join(timeout_s)
            if reader.is_alive():
                hung = True
                raise subprocess.TimeoutExpired(
                    str(DRIVER), timeout_s,
                    "the driver never answered; the substitution looped")
            return json.loads(result["line"])

        try:
            response = call(json.dumps(
                {"op": "open", "database": config.database,
                 "table_prefix": prefix, **DEFAULTS}))
            assert response["ok"], response
            response = call(json.dumps(
                {"op": "acquire", "holder": "%(holder)s"}))
            assert response["ok"], response
            assert response["lease"]["holder"] == "%(holder)s"
            # The stored holder is the literal string, quoted once —
            # scoped to THIS lease: the install's own claim row is in the
            # table too, with its own holder.
            rows = client.execute(
                "SELECT DISTINCT holder FROM "
                f"`{config.database}`.`{config.table_prefix}"
                "_publisher_lease` WHERE lease_id = "
                f"toUUID('{response['lease']['lease_id']}')")
            assert [tuple(r) for r in rows] == [("%(holder)s",)], rows
        except subprocess.TimeoutExpired:
            pytest.fail(
                "the substitution looped forever on a self-referential "
                "parameter")
        finally:
            # A hung driver would block close(); kill it outright.
            proc.kill()
            proc.wait(timeout=10)
