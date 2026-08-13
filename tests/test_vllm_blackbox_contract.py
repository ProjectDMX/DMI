"""CPU tests for the public vLLM transparency-result contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.blackbox.contracts import transparency_mismatches


pytestmark = pytest.mark.cpu


def _payload() -> dict:
    return {
        "schema_version": 1,
        "mode": "baseline",
        "model": "model",
        "cudagraph": False,
        "tensor_parallel_size": 1,
        "prompts": ["hello"],
        "prompt_token_ids": [[1]],
        "token_ids": [[2]],
        "texts": ["world"],
        "finish_reasons": ["length"],
        "stop_reasons": [None],
    }


def test_transparency_ignores_only_the_execution_mode_label():
    baseline = _payload()
    monitored = deepcopy(baseline)
    monitored["mode"] = "monitored"

    assert transparency_mismatches(baseline, monitored) == []


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("prompt_token_ids", [[9]]),
        ("token_ids", [[9]]),
        ("texts", ["changed"]),
        ("finish_reasons", ["stop"]),
        ("stop_reasons", [42]),
    ],
)
def test_transparency_reports_changed_public_output(field, replacement):
    baseline = _payload()
    monitored = deepcopy(baseline)
    monitored[field] = replacement

    assert transparency_mismatches(baseline, monitored) == [field]


def test_transparency_reports_missing_fields_before_comparison():
    baseline = _payload()
    monitored = deepcopy(baseline)
    del monitored["token_ids"]

    assert transparency_mismatches(baseline, monitored) == [
        "missing required field: token_ids"
    ]
