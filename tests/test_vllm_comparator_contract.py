"""CPU regression tests for fail-closed vLLM comparators."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from tests.compare_disk_vs_ch import compare, read_clickhouse
from tests.vllm_identical_comparator import bitwise_check, compare_logprobs


pytestmark = pytest.mark.cpu


def _compare(tmp_path, baseline, candidate):
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(baseline, baseline_path)
    torch.save(candidate, candidate_path)
    checks = []

    def record(name, passed, detail=""):
        checks.append({"name": name, "passed": passed, "detail": detail})

    compare_logprobs(
        str(baseline_path),
        str(candidate_path),
        {},
        record,
        label="baseline vs candidate",
    )
    assert len(checks) == 1
    return checks[0]


def test_bitwise_comparison_fails_on_dtype_mismatch():
    result = bitwise_check(
        torch.tensor([1.0], dtype=torch.float16),
        torch.tensor([1.0], dtype=torch.bfloat16),
        "dtype",
    )

    assert not result["passed"]
    assert "dtype mismatch" in result["detail"]


def test_logprob_comparison_fails_on_token_length_only_mismatch(tmp_path):
    logprobs = torch.zeros((2, 4), dtype=torch.float32)
    result = _compare(
        tmp_path,
        {0: {"token_ids": [1, 2], "logprobs": logprobs}},
        {0: {"token_ids": [1], "logprobs": logprobs}},
    )

    assert not result["passed"]
    assert "token_ids differ" in result["detail"]


def test_logprob_comparison_fails_instead_of_truncating_shape(tmp_path):
    baseline = torch.zeros((2, 4), dtype=torch.float32)
    candidate = torch.zeros((2, 3), dtype=torch.float32)
    result = _compare(
        tmp_path,
        {0: {"token_ids": [1, 2], "logprobs": baseline}},
        {0: {"token_ids": [1, 2], "logprobs": candidate}},
    )

    assert not result["passed"]
    assert "shape mismatch" in result["detail"]


def test_logprob_comparison_requires_full_vocab_data(tmp_path):
    result = _compare(
        tmp_path,
        {0: {"token_ids": [1], "logprobs": None}},
        {0: {"token_ids": [1], "logprobs": None}},
    )

    assert not result["passed"]
    assert "unavailable" in result["detail"]


def test_clickhouse_reader_filters_to_one_capture(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, host, port):
            assert (host, port) == ("db", 9000)

        def execute(self, query, params, settings):
            calls.append((query, params, settings))
            return []

    monkeypatch.setitem(
        sys.modules,
        "clickhouse_driver",
        SimpleNamespace(Client=FakeClient),
    )

    data, count = read_clickhouse("db", 9000, model_id="capture-123")

    assert data == {}
    assert count == 0
    assert "WHERE model_id = %(model_id)s" in calls[0][0]
    assert calls[0][1] == {"model_id": "capture-123"}


def _storage_key(*, start: int = 0, end: int = 1, act: str = "blocks.hook_resid_pre"):
    return ("request", act, 0, 0, start, end)


def _write_storage_reference(tmp_path, tensor: torch.Tensor) -> None:
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    torch.save(tensor, request_dir / "resid_pre_L0_T0_1.pt")


def test_storage_comparison_rejects_same_value_at_wrong_token_range(tmp_path):
    reference = torch.tensor([[1.0]])
    _write_storage_reference(tmp_path, reference)

    passed, failed = compare(
        str(tmp_path),
        {_storage_key(start=1, end=2): reference},
        1,
    )

    assert passed == 0
    assert failed > 0


def test_storage_comparison_rejects_unexpected_hook_keys(tmp_path):
    reference = torch.tensor([[1.0]])
    _write_storage_reference(tmp_path, reference)
    storage = {
        _storage_key(): reference,
        _storage_key(act="blocks.attn.hook_q"): reference,
    }

    passed, failed = compare(str(tmp_path), storage, 2)

    assert passed == 1
    assert failed == 1


def test_storage_comparison_rejects_duplicate_row_keys(tmp_path):
    reference = torch.tensor([[1.0]])
    _write_storage_reference(tmp_path, reference)

    passed, failed = compare(
        str(tmp_path),
        {_storage_key(): reference},
        2,
    )

    assert passed == 1
    assert failed == 1
