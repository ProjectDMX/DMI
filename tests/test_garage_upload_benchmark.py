from __future__ import annotations

import json

import pytest

from benchmarks.bench_garage_upload import GarageBenchmarkConfig, main


pytestmark = pytest.mark.cpu


def test_garage_benchmark_config_rejects_unbounded_inputs():
    with pytest.raises(ValueError, match="pack_payload_bytes"):
        GarageBenchmarkConfig(pack_payload_bytes=())
    with pytest.raises(ValueError, match="upload_workers"):
        GarageBenchmarkConfig(upload_workers=(0,))
    with pytest.raises(ValueError, match="multipart_chunk_bytes"):
        GarageBenchmarkConfig(multipart_chunk_bytes=1024)


def test_garage_benchmark_dry_run_needs_no_credentials(capsys):
    assert main(
        [
            "--pack-payload-bytes",
            "1MiB,2MiB",
            "--multipart-threshold-bytes",
            "1MiB",
            "--upload-workers",
            "1,2",
            "--packs-per-trial",
            "2",
            "--trials",
            "1",
            "--dry-run",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["config"]["pack_payload_bytes"] == [1024**2, 2 * 1024**2]
    assert result["config"]["upload_workers"] == [1, 2]
