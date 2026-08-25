from __future__ import annotations

import json

import pytest

from benchmarks.bench_capture_pack import (
    PackBenchmarkConfig,
    generate_payload_pool,
    main,
    run_trial,
)


pytestmark = pytest.mark.cpu


def test_pack_benchmark_config_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="records"):
        PackBenchmarkConfig(records=0)
    with pytest.raises(ValueError, match="payload_bytes"):
        PackBenchmarkConfig(payload_bytes=3, dtype="float32")
    with pytest.raises(ValueError, match="target_pack_bytes"):
        PackBenchmarkConfig(payload_bytes=64, target_pack_bytes=32)


def test_payload_pool_is_bounded_and_deterministic():
    config = PackBenchmarkConfig(
        records=4,
        payload_bytes=64,
        target_pack_bytes=256,
        pool_size=2,
        pattern="random",
        seed=11,
        trials=1,
    )

    first = generate_payload_pool(config)
    second = generate_payload_pool(config)

    assert first == second
    assert len(first) == 2
    assert all(len(payload) == 64 for payload in first)


def test_trial_reports_verified_pack_and_payload_counts():
    result = run_trial(
        PackBenchmarkConfig(
            records=9,
            payload_bytes=64,
            target_pack_bytes=2048,
            pool_size=2,
            pattern="zeros",
            trials=1,
        )
    )

    assert result.record_count == 9
    assert result.logical_bytes == 9 * 64
    assert result.pack_count > 1
    assert result.largest_pack_bytes <= 2048
    assert result.packed_bytes > result.logical_bytes
    assert result.seconds > 0
    assert result.as_dict()["logical_gib_per_second"] > 0


def test_dry_run_emits_resolved_workload(capsys):
    assert main(
        [
            "--records",
            "7",
            "--payload-bytes",
            "64",
            "--target-pack-bytes",
            "256",
            "--pool-size",
            "2",
            "--trials",
            "1",
            "--dry-run",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["config"]["records"] == 7
    assert result["config"]["payload_bytes"] == 64
