"""Multi-GPU (TP=2) E2E smoke -- the ``multi_gpu`` suite's TP coverage.

The configurable matrix (:mod:`tests.e2e_matrix`) treats tensor-parallel size
as a first-class axis (``--tp`` / ``E2E_TP_SIZE``), but the single-GPU wrappers
(``test_vllm_identical`` / ``test_e2e_correctness_vs_hf``) all drive ``tp=1``.
This module drives vLLM matrix cells at ``tp=2`` so the documented
``-m multi_gpu`` suite (docs/testing.md) actually exercises TP sharding rather
than collecting nothing.

TP=2 is "where meaningful" for the sharded model: with two ranks the
attention/expert projections are split across GPUs, so the reference-vs-monitored
comparison validates that the ring transport reassembles per-rank shards
correctly.  ``qwen3`` is the default model (GQA + a non-trivial hidden size makes
the sharding observable); ``gpt2`` is too small for TP to be interesting.  The
model is still overridable via ``E2E_MODEL``.

HF TP=2 is not covered here: ``hf_reference_runner`` / ``hf_monitored_runner``
do not read ``E2E_TP_SIZE``, do not launch under ``torchrun``, and do not pass
``tp_plan="auto"``.  A test claiming HF TP=2 coverage would require 2 GPUs
without actually exercising tensor parallelism, which would be misleading.

Skip-guarded (``tests/_requirements``) so a runner with <2 GPUs, no vLLM, or no
ClickHouse skips with a reason instead of failing the job.
"""
from __future__ import annotations

import os

import pytest

from tests._requirements import (
    require_clickhouse,
    require_gpus,
    require_vllm,
)
from tests.e2e_matrix import matrix_argv_from_env, run_single

pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.clickhouse,
]


def _tp2_env(default_model: str = "qwen3") -> dict:
    """os.environ with TP forced to 2 (model still overridable via E2E_MODEL)."""
    env = dict(os.environ)
    env["E2E_TP_SIZE"] = "2"
    env.setdefault("E2E_MODEL", default_model)
    return env


def _assert_cell(subtests, cr) -> None:
    """Fail on a dispatch error; report each matrix check as a subtest."""
    if cr.error:
        pytest.fail(f"matrix cell errored: {cr.error}")
    assert cr.checks, "matrix produced no checks"
    for chk in cr.checks:
        with subtests.test(chk.name):
            assert chk.passed, chk.detail


@pytest.mark.vllm
@require_gpus(2)
@require_vllm()
@require_clickhouse()
def test_vllm_identical_tp2(subtests) -> None:
    """vLLM TP=2 transport-bitwise: reference D2D buffers vs ring -> ClickHouse.

    Equivalent matrix cell: ``--backend vllm --standard transport_bitwise --tp 2``.
    The bitwise standard stays exact under TP (sharding is a layout change, not a
    numeric one), so any per-rank reassembly bug surfaces as a non-zero max_abs.
    """
    argv = matrix_argv_from_env("vllm", "transport_bitwise", env=_tp2_env())
    _assert_cell(subtests, run_single(argv))


