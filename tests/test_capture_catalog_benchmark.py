from __future__ import annotations

import pytest

from benchmarks.bench_capture_catalog import measure_inserts, synthetic_descriptors


pytestmark = pytest.mark.cpu


class _Writer:
    def __init__(self):
        self.batches = []

    def write_descriptors(self, descriptors, *, index_version):
        self.batches.append(tuple(descriptors))


def test_catalog_benchmark_uses_bounded_insert_batches():
    writer = _Writer()
    result = measure_inserts(
        writer, synthetic_descriptors(5), batch_rows=2, trials=2
    )

    assert [len(batch) for batch in writer.batches] == [2, 2, 1, 2, 2, 1]
    assert result["inserts_per_trial"] == 3
    assert result["rows_per_second_median"] > 0
