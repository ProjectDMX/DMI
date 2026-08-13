"""CPU tests for the public vLLM transparency-result contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.blackbox.contracts import (
    baseline_envelope_mismatches,
    baseline_instabilities,
    metamorphic_mismatches,
    transparency_mismatches,
)
from tests.tools.smoke_vllm_model import _materialize_case, _serialize_request


pytestmark = pytest.mark.cpu


def _payload() -> dict:
    return {
        "schema_version": 2,
        "mode": "baseline",
        "model": "model",
        "cudagraph": False,
        "tensor_parallel_size": 1,
        "corpus": {
            "schema_version": 2,
            "name": "corpus",
            "cases": [
                {
                    "case_id": "case-a",
                    "input": {"form": "text", "text": "hello"},
                    "sampling": {"max_tokens": 1},
                },
                {
                    "case_id": "case-b",
                    "input": {"form": "text", "text": "hello"},
                    "sampling": {"max_tokens": 1},
                },
            ],
        },
        "executions": [
            {
                "execution_id": "batch",
                "case_order": ["case-a", "case-b"],
                "results": [
                    _result("case-a", "0", 2, "world"),
                    _result("case-b", "1", 3, "again"),
                ],
            },
            {
                "execution_id": "reversed",
                "case_order": ["case-b", "case-a"],
                "results": [
                    _result("case-b", "2", 3, "again"),
                    _result("case-a", "3", 2, "world"),
                ],
            },
        ],
    }


def _result(case_id: str, request_id: str, token_id: int, text: str) -> dict:
    return {
        "case_id": case_id,
        "request_id": request_id,
        "prompt": "hello",
        "prompt_token_ids": [1],
        "finished": True,
        "outputs": [
            {
                "index": 0,
                "text": text,
                "token_ids": [token_id],
                "cumulative_logprob": None,
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
    }


def test_transparency_ignores_only_the_execution_mode_label():
    baseline = _payload()
    monitored = deepcopy(baseline)
    monitored["mode"] = "monitored"

    assert transparency_mismatches(baseline, monitored) == []


@pytest.mark.parametrize(
    "path,replacement,expected",
    [
        (
            ("executions", 0, "results", 0, "prompt_token_ids"),
            [9],
            "executions[0].results[0].prompt_token_ids[0]",
        ),
        (
            ("executions", 0, "results", 0, "outputs", 0, "token_ids"),
            [9],
            "executions[0].results[0].outputs[0].token_ids[0]",
        ),
        (
            ("executions", 0, "results", 0, "outputs", 0, "text"),
            "changed",
            "executions[0].results[0].outputs[0].text",
        ),
        (
            ("executions", 0, "results", 0, "outputs", 0, "finish_reason"),
            "stop",
            "executions[0].results[0].outputs[0].finish_reason",
        ),
        (
            ("executions", 0, "results", 0, "outputs", 0, "stop_reason"),
            42,
            "executions[0].results[0].outputs[0].stop_reason: "
            "type NoneType != int",
        ),
    ],
)
def test_transparency_reports_changed_public_output(path, replacement, expected):
    baseline = _payload()
    monitored = deepcopy(baseline)
    monitored["mode"] = "monitored"
    target = monitored
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    assert transparency_mismatches(baseline, monitored) == [expected]


def test_transparency_reports_missing_fields_before_comparison():
    baseline = _payload()
    monitored = deepcopy(baseline)
    monitored["mode"] = "monitored"
    del monitored["executions"]

    assert transparency_mismatches(baseline, monitored) == [
        "missing required field: executions"
    ]


def test_baseline_envelope_keeps_stable_fields_exact_and_outputs_atomic():
    first = _payload()
    second = deepcopy(first)
    second["executions"][0]["results"][0]["outputs"][0].update(
        text="alternate",
        token_ids=[9],
    )
    monitored = deepcopy(first)
    monitored["mode"] = "monitored"
    monitored["executions"][0]["results"][0]["outputs"] = deepcopy(
        second["executions"][0]["results"][0]["outputs"]
    )

    assert baseline_instabilities([first, second]) == ["batch.case-a.outputs"]
    assert baseline_envelope_mismatches([first, second], monitored) == []

    monitored["executions"][0]["results"][0]["outputs"][0]["token_ids"] = [7]
    assert baseline_envelope_mismatches([first, second], monitored) == [
        "envelope.batch.case-a.outputs"
    ]


def test_baseline_envelope_never_relaxes_prompt_or_result_structure():
    first = _payload()
    second = deepcopy(first)
    monitored = deepcopy(first)
    monitored["mode"] = "monitored"
    monitored["executions"][0]["results"][0]["prompt_token_ids"] = [9]

    assert baseline_envelope_mismatches([first, second], monitored) == [
        "envelope[2].executions[0].results[0].prompt_token_ids[0]"
    ]


def test_baseline_envelope_requires_independent_replica():
    assert baseline_envelope_mismatches([_payload()], _payload()) == [
        "baseline envelope requires at least two baseline processes"
    ]


def test_metamorphic_contract_matches_results_by_case_identity():
    assert metamorphic_mismatches(_payload()) == []


def test_metamorphic_contract_rejects_declared_reverse_order_drift():
    payload = _payload()
    payload["executions"][1]["case_order"] = ["case-a", "case-b"]
    assert metamorphic_mismatches(payload) == ["executions.reversed.case_order"]


def test_metamorphic_contract_rejects_scheduler_result_order_drift():
    payload = _payload()
    payload["executions"][0]["results"].reverse()

    assert metamorphic_mismatches(payload) == ["executions.batch.result_order"]


def test_metamorphic_contract_rejects_prompt_attribution_drift():
    payload = _payload()
    payload["executions"][1]["results"][1]["prompt_token_ids"] = [9]
    assert metamorphic_mismatches(payload) == [
        "metamorphic.case-a.prompt_token_ids[0]"
    ]


def test_metamorphic_contract_allows_greedy_tokens_to_change_with_batch_shape():
    payload = _payload()
    payload["executions"][1]["results"][1]["outputs"][0].update(
        text="different greedy path",
        token_ids=[9],
    )

    assert metamorphic_mismatches(payload) == []


def test_metamorphic_contract_enforces_per_case_generation_bound():
    payload = _payload()
    payload["executions"][0]["results"][0]["outputs"][0]["token_ids"] = [1, 2]

    assert metamorphic_mismatches(payload) == [
        "executions.batch.results[0].outputs[0].token_ids.max_tokens",
        "executions.batch.results[0].outputs[0].token_ids.length_finish",
    ]


def test_metamorphic_contract_rejects_fields_missing_from_both_modes():
    payload = _payload()
    for execution in payload["executions"]:
        del execution["results"][0]["outputs"][0]["finish_reason"]

    assert metamorphic_mismatches(payload) == [
        "executions.batch.results[0].outputs[0].finish_reason: missing",
        "executions.batch.results[0].outputs[0].finish_reason",
        "executions.reversed.results[0].outputs[0].finish_reason: missing",
        "executions.reversed.results[0].outputs[0].finish_reason",
    ]


def test_metamorphic_contract_requires_unique_public_request_ids():
    payload = _payload()
    payload["executions"][0]["results"][1]["request_id"] = "0"

    assert metamorphic_mismatches(payload) == [
        "executions.batch.request_ids_not_unique"
    ]


def test_public_token_id_case_uses_the_public_tokenizer_contract():
    class Tokenizer:
        def encode(self, text):
            assert text == "hello"
            return (4, 5, 6)

    prompt, sampling = _materialize_case(
        {
            "case_id": "tokens",
            "input": {"form": "token_ids_from_text", "text": "hello"},
            "sampling": {"temperature": 0.0, "max_tokens": 3},
        },
        Tokenizer(),
        8,
    )

    assert prompt == [4, 5, 6]
    assert sampling == {"temperature": 0.0, "max_tokens": 3}


def test_request_serializer_keeps_identity_cardinality_and_completion_fields():
    from types import SimpleNamespace

    output = SimpleNamespace(
        request_id="request-1",
        prompt="hello",
        prompt_token_ids=(1, 2),
        finished=True,
        outputs=[
            SimpleNamespace(
                index=0,
                text="world",
                token_ids=(3, 4),
                cumulative_logprob=None,
                finish_reason="length",
                stop_reason=None,
            )
        ],
    )

    assert _serialize_request("case-a", output) == _result(
        "case-a", "request-1", 3, "world"
    ) | {
        "prompt_token_ids": [1, 2],
        "outputs": [
            {
                "index": 0,
                "text": "world",
                "token_ids": [3, 4],
                "cumulative_logprob": None,
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
    }
