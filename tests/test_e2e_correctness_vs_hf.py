"""HF E2E correctness — thin wrappers over the configurable matrix (plan §5).

This file used to carry ~1.6k lines of in-process HF rollout + tensor
comparison logic, three tests (one permanently disabled via
``@skipif(True)``), 16 skip sites, and two ``_legacy`` bodies kept "for
reference".  All of that comparison logic now lives in :mod:`tests.lib` and
the dispatch in :mod:`tests.e2e_matrix`; these wrappers just drive the
matrix for the equivalent HF cell and assert on its checks.

The test *names* are preserved because ``tests/tools/verify_hf.sh`` invokes
them by node id (``::test_e2e_correctness_hf`` /
``::test_e2e_cuda_graphs_vs_eager_hf``) and threads the ring-size / model
env vars the matrix wrapper reads.

The removed ``test_e2e_correctness_hf_cuda_graphs`` was permanently disabled
(its compiled-rollout reference could not replicate generate()'s internal
StaticCache handling) and explicitly superseded by
``test_e2e_cuda_graphs_vs_eager_hf`` -- no still-passing assertion was lost.
"""
from __future__ import annotations

import pytest

from tests._requirements import require_cuda, require_clickhouse
from tests.e2e_matrix import matrix_argv_from_env, run_single

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.clickhouse,
    pytest.mark.hf,
]


def _assert_cell(subtests, cr) -> None:
    """Fail on a dispatch error; report each matrix check as a subtest."""
    if cr.error:
        pytest.fail(f"matrix cell errored: {cr.error}")
    assert cr.checks, "matrix produced no checks"
    for chk in cr.checks:
        with subtests.test(chk.name):
            assert chk.passed, chk.detail


@require_cuda()
@require_clickhouse()
def test_e2e_correctness_hf(subtests) -> None:
    """HF eager: hooked model (ring -> ClickHouse) vs original model.

    Equivalent matrix cell: ``--backend hf --mode eager --standard allclose``
    (HF dispatches to hf_comparator, which does the value comparison with
    ``E2E_TOLERANCE``; default 0.01 for eager).
    """
    cr = run_single(matrix_argv_from_env("hf", "allclose", mode="eager"))
    _assert_cell(subtests, cr)


@require_cuda()
@require_clickhouse()
def test_e2e_cuda_graphs_vs_eager_hf(subtests) -> None:
    """HF CUDA-graph monitored run vs eager reference, relaxed tolerance.

    The HF reference runner is always eager; the monitored runner honors
    ``E2E_CUDA_GRAPHS`` (set from the cuda_graph mode).  Tolerance defaults
    to 0.5 to absorb bf16 accumulation-order drift between compiled and
    uncompiled paths.
    """
    cr = run_single(matrix_argv_from_env(
        "hf", "allclose", mode="cuda_graph", default_tolerance="0.5"))
    _assert_cell(subtests, cr)
