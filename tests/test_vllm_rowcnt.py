"""vLLM row-count check — thin wrapper over the configurable matrix (plan §5).

Runs the monitored model (hooked + ring transport -> ClickHouse) and
validates schema + per-hook row counts (plus value comparison against the
reference when the model is supported by extract_hidden_states).  The
orchestration (vllm_monitored_runner -> vllm_rowcnt_comparator) now lives in
:mod:`tests.e2e_matrix`; this wrapper drives the matrix's vLLM ``row_count``
cell and asserts on its checks.

The test name is preserved because ``tests/tools/verify_vllm.sh`` invokes
this file and threads the model / ring-size / tolerance env vars the matrix
wrapper reads (E2E_MODEL, E2E_ENFORCE_EAGER, E2E_RING_PAYLOAD_MB,
E2E_RING_PINNED_MB, E2E_TOLERANCE, DMX_DB_HOST, DMX_DB_PORT).

Usage:
  python -m pytest tests/test_vllm_rowcnt.py -q -s
  E2E_MODEL=qwen3 python -m pytest tests/test_vllm_rowcnt.py -q -s
"""
from __future__ import annotations

import pytest

from tests._requirements import require_cuda, require_clickhouse, require_vllm
from tests.e2e_matrix import matrix_argv_from_env, run_single

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.vllm,
    pytest.mark.clickhouse,
    pytest.mark.e2e,
]


@require_cuda()
@require_vllm()
@require_clickhouse()
def test_vllm_rowcnt(subtests) -> None:
    """vLLM row-count validation: monitored run + schema / row-count checks."""
    cr = run_single(matrix_argv_from_env("vllm", "row_count"))
    if cr.error:
        pytest.fail(f"matrix cell errored: {cr.error}")
    assert cr.checks, "matrix produced no checks"
    for chk in cr.checks:
        with subtests.test(chk.name):
            assert chk.passed, chk.detail
