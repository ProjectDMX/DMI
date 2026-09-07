"""Regression pins for the E2E verification loop's two fixes.

One: the replicated-quorum verifier's hand-rolled DDL lost the
``{prefix}_pack_inventory`` view ``committed_pack_ids()`` reads, so every run
died with Code 60 before any quorum behavior. The views are now built from
the production builders; these tests exercise the verifier's create() against
a fake client so the emitted SQL is checked, not the source text.

Two: ``E2E_RING_PAYLOAD_BYTES`` never reached the allocation that OOMs,
because the runner built ``MonitoringEngine`` without ring kwargs and the
engine auto-enables the transport during ``__init__``. The runner now passes
``enable_ring_transport=False`` at construction and enables once, with the
full tuned config -- one allocation, one source of truth for the ring shape.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dmi.storage.capture.clickhouse_schema import (
    capture_view_definition,
    pack_view_definition,
)
from dmi.storage.capture.clickhouse_sql import membership_predicate

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "tests" / "tools" / "verify_replicated_quorum.py"


class _RecordingClient:
    """Answers just enough of clickhouse_driver.Client for create()."""

    def __init__(self):
        self.statements: list[str] = []

    def execute(self, query, params=None, **kwargs):
        self.statements.append(query)


def _run_create(suffix: str):
    import sys

    sys.path.insert(0, str(REPO / "tests" / "tools"))
    import verify_replicated_quorum as vq

    client = _RecordingClient()
    vq.create(client, "qtest_pfx", "r1", suffix)
    return client.statements


class TestVerifierCreatesProductionViews:
    def test_create_emits_both_views_with_production_ddl(self):
        statements = _run_create("")

        views = [s for s in statements if s.startswith(("CREATE OR REPLACE VIEW", "CREATE VIEW"))]
        assert views == [
            capture_view_definition("default", "qtest_pfx"),
            pack_view_definition("default", "qtest_pfx"),
        ], "create() must emit the production builders' exact DDL, in order"

    def test_create_emits_no_view_for_the_peer_replica(self):
        statements = _run_create("_peer")

        assert not any(
            s.startswith(("CREATE OR REPLACE VIEW", "CREATE VIEW")) for s in statements
        ), "nothing reads the peer's views, and the peer naming is not what the builders emit"

    def test_the_emitted_capture_view_matches_the_member_predicate(self):
        capture = capture_view_definition("default", "pfx")
        expected = membership_predicate(
            "`default`.`pfx_snapshot_manifest`", "`default`.`pfx_index_watermark`", bounded=False
        )
        assert expected in capture
        assert "`default`.`pfx_capture_raw` FINAL" in capture
        assert "CREATE OR REPLACE VIEW `default`.`pfx_capture`" in capture

    def test_the_emitted_pack_view_projects_the_pack_columns(self):
        pack = pack_view_definition("default", "pfx")
        assert "CREATE VIEW IF NOT EXISTS `default`.`pfx_pack_inventory`" in pack
        assert "`default`.`pfx_pack_inventory_raw` FINAL" in pack
        # pack_id, store_id, object_key, object_bytes, pack_checksum, record_count
        assert len(re.findall(r",", pack.split("SELECT", 1)[1].split("FROM", 1)[0])) == 5

    def test_drop_removes_the_views_it_created(self):
        import sys

        sys.path.insert(0, str(REPO / "tests" / "tools"))
        import verify_replicated_quorum as vq

        client = _RecordingClient()
        vq.drop(client, "qtest_pfx")

        dropped = [s for s in client.statements if "DROP TABLE" in s or "ATTACH TABLE" in s]
        view_drops = [s for s in dropped if "_capture`" in s or "_pack_inventory`" in s]
        assert view_drops, "drop() must tear down the views create() made"


class TestRunnerEnablesTheRingOnce:
    def test_runner_does_not_double_enable(self):
        """The engine auto-enables at construction; a second
        enable_ring_transport() tears the first ring down and reallocates --
        2x the peak on exactly the busy-GPU this knob exists for."""
        source = (REPO / "tests" / "hf_monitored_runner.py").read_text()

        assert "enable_ring_transport=False" in source, (
            "the runner must disable the auto-enable so the ring is built "
            "once, from the fully tuned config"
        )
        assert source.count("enable_ring_transport(") - source.count(
            "enable_ring_transport=False"
        ) == 1, "exactly one real enable call"

    def test_the_single_enable_carries_the_tuned_config(self):
        source = (REPO / "tests" / "hf_monitored_runner.py").read_text()
        start = source.index("engine.enable_ring_transport(ring_cfg)")

        # ring_cfg itself must come from the env knobs, so the one enable
        # carries every tuned field.
        ring_cfg_block = source[source.index("ring_cfg = RingConfig()") : start]
        for env in (
            "E2E_RING_TASK_ENTRIES",
            "E2E_RING_PAYLOAD_BYTES",
            "E2E_RING_PINNED_BYTES",
        ):
            assert env in ring_cfg_block, f"{env} must feed ring_cfg"

    def test_no_ring_kwargs_leak_into_the_constructor(self):
        """The constructor's ring_* kwargs build a SECOND, lesser config that
        enable_ring_transport then replaces; the runner must not use them."""
        source = (REPO / "tests" / "hf_monitored_runner.py").read_text()
        start = source.index("engine = MonitoringEngine(")
        end = source.index(")", source.index("db_config", start))
        constructor_call = source[start:end]

        assert "ring_payload_mb" not in constructor_call
        assert "ring_pinned_mb" not in constructor_call
        assert "ring_task_entries" not in constructor_call
