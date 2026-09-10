from __future__ import annotations

import pytest

from benchmarks.profile_pipeline_stages import (
    _records,
    stage_spool_stage,
    stage_store_put,
)
from dmi.storage.capture import PackWriter
from dmi.storage.capture.pack import PackCapacityError


pytestmark = pytest.mark.cpu


def _small_corpus(monkeypatch):
    monkeypatch.setattr("benchmarks.profile_pipeline_stages.RECORDS", 16)
    monkeypatch.setattr("benchmarks.profile_pipeline_stages.PAYLOAD", 4096)
    return _records()


@pytest.mark.parametrize("stage", (stage_store_put, stage_spool_stage))
def test_stage_rate_covers_the_full_corpus(monkeypatch, stage):
    corpus = _small_corpus(monkeypatch)

    result = stage(corpus)

    assert result["seconds"] > 0
    assert result["gib_s"] > 0


@pytest.mark.parametrize("stage", (stage_store_put, stage_spool_stage))
def test_stage_does_not_swallow_pack_capacity_errors(monkeypatch, stage):
    # Reproduction for the harness bug that inflated the ledger's store
    # rows: the stages used to catch every exception from PackWriter.append
    # and drop the record, then divide the full corpus size by the measured
    # time — silently overstating throughput ~1.25x. An append failure must
    # propagate instead.
    corpus = _small_corpus(monkeypatch)
    real_append = PackWriter.append
    appends = 0

    def failing_after_first(self, record):
        nonlocal appends
        appends += 1
        if appends == 2:
            raise PackCapacityError("simulated mid-corpus overflow")
        return real_append(self, record)

    monkeypatch.setattr(PackWriter, "append", failing_after_first)

    with pytest.raises(PackCapacityError):
        stage(corpus)

    assert appends == 2, "stage stopped early instead of failing"
