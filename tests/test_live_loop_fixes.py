

# ---------------------------------------------------------------------------
# Review-loop finding: the replicated-quorum verifier hand-rolls its catalog
# DDL and lost the `{prefix}_pack_inventory` view when the schema gained
# views — committed_pack_ids() then fails with Code 60 before any quorum
# behavior runs. The verifier's view DDL must be built from production's
# builder, not re-typed.
# ---------------------------------------------------------------------------


class TestVerifierSharesProductionViewDDL:
    def test_verifier_create_emits_both_views_from_production_builder(self):
        import inspect

        from dmi.storage.capture.clickhouse_schema import (
            capture_view_definition,
            pack_view_definition,
        )

        source = inspect.getsource(
            __import__("tests.tools.verify_replicated_quorum", fromlist=["create"]).create
        )
        assert "capture_view_definition" in source, (
            "the verifier must build its views from the production builder, "
            "not hand-typed DDL that drifts"
        )
        assert "pack_view_definition" in source

    def test_production_view_definitions_are_the_real_ones(self):
        import re

        from dmi.storage.capture.clickhouse_schema import (
            capture_view_definition,
            pack_view_definition,
        )
        from dmi.storage.capture.clickhouse_sql import membership_predicate

        capture = capture_view_definition("default", "pfx")
        pack = pack_view_definition("default", "pfx")
        assert "CREATE OR REPLACE VIEW `default`.`pfx_capture`" in capture
        assert "`default`.`pfx_capture_raw` FINAL" in capture
        assert membership_predicate("`default`.`pfx_snapshot_manifest`", "`default`.`pfx_index_watermark`", bounded=False) in capture
        assert "CREATE VIEW IF NOT EXISTS `default`.`pfx_pack_inventory`" in pack
        assert "`default`.`pfx_pack_inventory_raw` FINAL" in pack
        # Both name every column but index_version, in production order.
        assert len(re.findall(r"\bstore_id\b", pack)) == 1


# ---------------------------------------------------------------------------
# Review-loop finding: E2E_RING_PAYLOAD_BYTES is documented as the ring-size
# control for the HF correctness suite, but the runner constructs
# MonitoringEngine without ring kwargs, and the engine auto-enables the ring
# transport at its 4 GiB default during __init__ — before the runner's later
# enable_ring_transport(ring_cfg) can size anything. On a GPU with <4 GiB
# free the suite OOMs at engine construction regardless of the env var.
# ---------------------------------------------------------------------------


class TestRunnerSizesTheRingBeforeAutoEnable:
    def test_runner_builds_the_engine_with_the_env_ring_size(self):
        import inspect
        from pathlib import Path

        runner = Path("tests/hf_monitored_runner.py").read_text()
        constructor = inspect.getsource(_runner_engine_construction)
        assert "ring_payload_mb" in constructor, (
            "the runner must pass ring_payload_mb/ring_pinned_mb to "
            "MonitoringEngine so the env-tuned size takes effect at "
            "construction, before the engine auto-enables the transport"
        )


class TestRunnerSizesTheRingBeforeAutoEnable:
    def test_runner_builds_the_engine_with_the_env_ring_size(self):
        import inspect
        from pathlib import Path

        source = Path("tests/hf_monitored_runner.py").read_text()
        # The engine construction must carry the env-tuned ring size, so the
        # auto-enable during __init__ allocates what the environment asked
        # for instead of the 4 GiB default.
        start = source.index("engine = MonitoringEngine(")
        end = source.index("engine.enable_ring_transport", start)
        call = source[start:end]
        assert "ring_payload_mb" in call, (
            "the runner must pass ring_payload_mb (and ring_pinned_mb) to "
            "MonitoringEngine so E2E_RING_PAYLOAD_BYTES takes effect at "
            "construction — the engine auto-enables the transport during "
            "__init__, before any later enable_ring_transport call"
        )
