from __future__ import annotations

import json

import pytest

from benchmarks.bench_capture_pipeline import (
    PipelineBenchmarkConfig,
    main,
    run_trial,
)


pytestmark = pytest.mark.cpu


def test_pipeline_benchmark_config_rejects_unbounded_inputs():
    with pytest.raises(ValueError, match="records"):
        PipelineBenchmarkConfig(records=0)
    with pytest.raises(ValueError, match="queue_records"):
        PipelineBenchmarkConfig(queue_records=0)
    with pytest.raises(ValueError, match="queue_bytes"):
        PipelineBenchmarkConfig(payload_bytes=64, queue_bytes=32)


@pytest.mark.parametrize("mode", ("direct", "spool"))
def test_pipeline_trial_persists_and_verifies_every_record(mode: str):
    trial = run_trial(
        PipelineBenchmarkConfig(
            mode=mode,
            records=12,
            payload_bytes=64,
            target_pack_bytes=4096,
            queue_records=4,
            queue_bytes=256,
            trials=1,
        )
    )

    assert trial.record_count == 12
    assert trial.persisted_records == 12
    assert trial.dropped_records == 0
    assert trial.packs_persisted > 0
    assert trial.queue_peak_records <= 4
    assert trial.queue_peak_bytes <= 256
    assert trial.seconds > 0
    assert trial.as_dict()["logical_gib_per_second"] > 0


def test_pipeline_benchmark_dry_run_emits_resolved_workload(capsys):
    assert main(
        [
            "--mode",
            "spool",
            "--records",
            "7",
            "--payload-bytes",
            "64",
            "--target-pack-bytes",
            "4096",
            "--queue-records",
            "2",
            "--queue-bytes",
            "128",
            "--trials",
            "1",
            "--dry-run",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["config"]["mode"] == "spool"
    assert result["config"]["records"] == 7
