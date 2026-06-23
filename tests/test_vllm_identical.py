"""vLLM identical check — thin wrapper over the configurable matrix (plan §5).

Bitwise tensor comparison between the reference model (GPU buffer D2D
capture -> disk) and the monitored model (ring transport -> ClickHouse).
The orchestration (enable_ref_hooks -> vllm_ref_runner -> vllm_monitored_runner
-> vllm_identical_comparator) now lives in :mod:`tests.e2e_matrix`; this
wrapper drives the matrix's vLLM ``bitwise`` cell and asserts on its checks.

The test name is preserved because ``tests/tools/verify_vllm.sh`` and
``tests/tools/identical_vllm.sh`` invoke this file and thread the model /
ring-size / hook-selection env vars the matrix wrapper reads:

  E2E_MODEL, E2E_ENFORCE_EAGER, E2E_DTYPE, E2E_RING_PAYLOAD_MB,
  E2E_RING_PINNED_MB, E2E_HOOK_SELECTION (-> internal DMX_HOOK_SELECTION),
  E2E_REF_MAX_LEN, E2E_TP_SIZE, DMX_DB_HOST, DMX_DB_PORT.

Usage:
  python -m pytest tests/test_vllm_identical.py -q -s
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
def test_vllm_identical(subtests) -> None:
    """Bitwise: reference D2D buffers (disk) vs ring transport (ClickHouse)."""
    cr = run_single(matrix_argv_from_env("vllm", "bitwise"))
    if cr.error:
        pytest.fail(f"matrix cell errored: {cr.error}")
    assert cr.checks, "matrix produced no checks"
    for chk in cr.checks:
        with subtests.test(chk.name):
            assert chk.passed, chk.detail
