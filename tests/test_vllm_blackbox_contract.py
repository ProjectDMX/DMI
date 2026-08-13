"""CPU tests for the public vLLM transparency-result contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.blackbox.contracts import (
    baseline_envelope_mismatches,
    baseline_instabilities,
    decision_branch_gap_limit,
    decision_logprob_drift_limit,
    metamorphic_mismatches,
    sampling_ambiguity_mismatches,
    transparency_mismatches,
)
from tests.tools.smoke_vllm_model import (
    _materialize_case,
    _serialize_logprob_trace,
    _serialize_request,
)


pytestmark = pytest.mark.cpu


def _payload() -> dict:
    return {
        "schema_version": 2,
        "mode": "baseline",
        "model": "model",
        "cudagraph": False,
        "tensor_parallel_size": 1,
        "decision_logprobs": 0,
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
                "decision_logprobs": None,
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


def _decision_row(token_id: int, logprob: float, rank: int) -> dict:
    return {
        "token_id": token_id,
        "logprob": logprob,
        "rank": rank,
        "decoded_token": str(token_id),
    }


def _with_decision_trace(
    payload: dict,
    *,
    execution: int,
    result: int,
    tokens: list[int],
    steps: list[list[dict]],
    mode: str,
) -> dict:
    traced = _with_complete_decision_traces(payload)
    traced["mode"] = mode
    target_result = traced["executions"][execution]["results"][result]
    case_id = target_result["case_id"]
    for case in traced["corpus"]["cases"]:
        if case["case_id"] == case_id:
            case["sampling"]["max_tokens"] = len(tokens)
    for candidate_execution in traced["executions"]:
        for candidate_result in candidate_execution["results"]:
            if candidate_result["case_id"] != case_id:
                continue
            candidate_output = candidate_result["outputs"][0]
            old_token = candidate_output["token_ids"][0]
            candidate_output["token_ids"] = [old_token] * len(tokens)
            candidate_output["text"] = " ".join(
                str(old_token) for _ in tokens
            )
            candidate_output["decision_logprobs"] = [
                [_decision_row(old_token, -0.1, 1)] for _ in tokens
            ]
            candidate_output["cumulative_logprob"] = -0.1 * len(tokens)
    output = target_result["outputs"][0]
    output["token_ids"] = tokens
    output["text"] = " ".join(str(token) for token in tokens)
    output["decision_logprobs"] = steps
    output["cumulative_logprob"] = sum(
        next(
            float(row["logprob"])
            for row in step
            if row["token_id"] == token_id
        )
        for token_id, step in zip(tokens, steps)
    )
    return traced


def _with_complete_decision_traces(payload: dict) -> dict:
    traced = deepcopy(payload)
    traced["decision_logprobs"] = 20
    for execution in traced["executions"]:
        for result in execution["results"]:
            for output in result["outputs"]:
                output["decision_logprobs"] = [
                    [_decision_row(token_id, -0.1, 1)]
                    for token_id in output["token_ids"]
                ]
                output["cumulative_logprob"] = -0.1 * len(
                    output["token_ids"]
                )
    return traced


def test_sampling_ambiguity_accepts_publicly_proven_near_tie():
    common = [_decision_row(2, -0.1, 1), _decision_row(3, -1.0, 2)]
    near_tie_left = [_decision_row(4, -2.0, 1), _decision_row(5, -2.1, 2)]
    near_tie_right = [_decision_row(4, -2.08, 2), _decision_row(5, -2.0, 1)]
    baseline = _with_decision_trace(
        _payload(),
        execution=0,
        result=0,
        tokens=[2, 4],
        steps=[common, near_tie_left],
        mode="baseline",
    )
    monitored = _with_decision_trace(
        baseline,
        execution=0,
        result=0,
        tokens=[2, 5],
        steps=[common, near_tie_right],
        mode="monitored",
    )

    assert metamorphic_mismatches(baseline) == []
    assert metamorphic_mismatches(monitored) == []
    assert sampling_ambiguity_mismatches(baseline, monitored) == []


def test_sampling_ambiguity_rejects_unproven_candidate():
    common = [_decision_row(2, -0.1, 1)]
    baseline = _with_decision_trace(
        _payload(),
        execution=0,
        result=0,
        tokens=[2, 4],
        steps=[common, [_decision_row(4, -0.1, 1)]],
        mode="baseline",
    )
    monitored = _with_decision_trace(
        baseline,
        execution=0,
        result=0,
        tokens=[2, 5],
        steps=[common, [_decision_row(5, -0.1, 1)]],
        mode="monitored",
    )

    assert sampling_ambiguity_mismatches(baseline, monitored) == [
        "ambiguity.batch.case-a.outputs[0].decision_candidates[1]"
    ]


def test_sampling_ambiguity_rejects_large_branch_gap():
    common = [_decision_row(2, -0.1, 1)]
    far_left = [_decision_row(4, -0.1, 1), _decision_row(5, -2.0, 2)]
    far_right = [_decision_row(4, -2.0, 2), _decision_row(5, -0.1, 1)]
    baseline = _with_decision_trace(
        _payload(),
        execution=0,
        result=0,
        tokens=[2, 4],
        steps=[common, far_left],
        mode="baseline",
    )
    monitored = _with_decision_trace(
        baseline,
        execution=0,
        result=0,
        tokens=[2, 5],
        steps=[common, far_right],
        mode="monitored",
    )

    mismatches = sampling_ambiguity_mismatches(baseline, monitored)

    assert "ambiguity.batch.case-a.outputs[0].decision_gap[1]" in mismatches
    assert "ambiguity.batch.case-a.outputs[0].decision_drift[1]" in mismatches


def test_sampling_ambiguity_allows_tied_candidate_with_third_rank():
    common = [_decision_row(2, -0.1, 1)]
    low_rank_left = [
        _decision_row(4, -2.0, 1),
        _decision_row(5, -2.1, 3),
    ]
    low_rank_right = [
        _decision_row(4, -2.08, 3),
        _decision_row(5, -2.0, 1),
    ]
    baseline = _with_decision_trace(
        _payload(),
        execution=0,
        result=0,
        tokens=[2, 4],
        steps=[common, low_rank_left],
        mode="baseline",
    )
    monitored = _with_decision_trace(
        baseline,
        execution=0,
        result=0,
        tokens=[2, 5],
        steps=[common, low_rank_right],
        mode="monitored",
    )

    assert sampling_ambiguity_mismatches(baseline, monitored) == []


def test_sampling_ambiguity_rejects_selected_token_below_public_maximum():
    common = [_decision_row(2, -0.1, 1)]
    near_tie_left = [
        _decision_row(4, -2.1, 2),
        _decision_row(5, -2.0, 1),
    ]
    near_tie_right = [
        _decision_row(4, -2.08, 2),
        _decision_row(5, -2.0, 1),
    ]
    baseline = _with_decision_trace(
        _payload(),
        execution=0,
        result=0,
        tokens=[2, 4],
        steps=[common, near_tie_left],
        mode="baseline",
    )
    monitored = _with_decision_trace(
        baseline,
        execution=0,
        result=0,
        tokens=[2, 5],
        steps=[common, near_tie_right],
        mode="monitored",
    )

    assert sampling_ambiguity_mismatches(baseline, monitored) == [
        "ambiguity.batch.case-a.outputs[0].selected_gap[1]"
    ]


def test_sampling_ambiguity_rejects_selected_logprob_drift_without_branch():
    baseline = _with_complete_decision_traces(_payload())
    monitored = deepcopy(baseline)
    monitored["mode"] = "monitored"
    row = monitored["executions"][0]["results"][0]["outputs"][0][
        "decision_logprobs"
    ][0][0]
    row["logprob"] = -0.5
    monitored["executions"][0]["results"][0]["outputs"][0][
        "cumulative_logprob"
    ] = -0.5

    assert transparency_mismatches(baseline, monitored) == []
    assert sampling_ambiguity_mismatches(baseline, monitored) == [
        "ambiguity.batch.case-a.outputs[0].decision_logprobs[0]"
    ]


def test_sampling_ambiguity_uses_calibrated_gpt2_drift_only() -> None:
    baseline = _with_complete_decision_traces(_payload())
    baseline["model"] = "gpt2"
    monitored = deepcopy(baseline)
    monitored["mode"] = "monitored"
    output = monitored["executions"][0]["results"][0]["outputs"][0]
    output["decision_logprobs"][0][0]["logprob"] = -0.45
    output["cumulative_logprob"] = -0.45

    assert decision_logprob_drift_limit(baseline) == 0.5
    assert sampling_ambiguity_mismatches(baseline, monitored) == []

    baseline["model"] = "qwen2"
    monitored["model"] = "qwen2"
    assert decision_logprob_drift_limit(baseline) == 0.25
    assert sampling_ambiguity_mismatches(baseline, monitored) == [
        "ambiguity.batch.case-a.outputs[0].decision_logprobs[0]"
    ]


def test_sampling_ambiguity_rejects_gpt2_drift_above_calibration() -> None:
    baseline = _with_complete_decision_traces(_payload())
    baseline["model"] = "gpt2"
    monitored = deepcopy(baseline)
    monitored["mode"] = "monitored"
    output = monitored["executions"][0]["results"][0]["outputs"][0]
    output["decision_logprobs"][0][0]["logprob"] = -0.7
    output["cumulative_logprob"] = -0.7

    assert sampling_ambiguity_mismatches(baseline, monitored) == [
        "ambiguity.batch.case-a.outputs[0].decision_logprobs[0]"
    ]


def test_sampling_ambiguity_uses_calibrated_gpt2_branch_gap_only() -> None:
    common = [_decision_row(2, -0.1, 1)]
    baseline_step = [
        _decision_row(4, -2.0, 1),
        _decision_row(5, -2.5, 2),
    ]
    monitored_step = [
        _decision_row(4, -2.25, 2),
        _decision_row(5, -2.25, 1),
    ]
    baseline = _with_decision_trace(
        _payload(),
        execution=0,
        result=0,
        tokens=[2, 4],
        steps=[common, baseline_step],
        mode="baseline",
    )
    baseline["model"] = "gpt2"
    monitored = _with_decision_trace(
        baseline,
        execution=0,
        result=0,
        tokens=[2, 5],
        steps=[common, monitored_step],
        mode="monitored",
    )

    assert decision_branch_gap_limit(baseline) == 0.5
    assert sampling_ambiguity_mismatches(baseline, monitored) == []

    baseline["model"] = "qwen2"
    monitored["model"] = "qwen2"
    assert decision_branch_gap_limit(baseline) == 0.25
    assert sampling_ambiguity_mismatches(baseline, monitored) == [
        "ambiguity.batch.case-a.outputs[0].decision_gap[1]"
    ]


def test_sampling_ambiguity_rejects_gpt2_branch_gap_above_calibration() -> None:
    common = [_decision_row(2, -0.1, 1)]
    baseline_step = [
        _decision_row(4, -2.0, 1),
        _decision_row(5, -2.51, 2),
    ]
    monitored_step = [
        _decision_row(4, -2.255, 2),
        _decision_row(5, -2.255, 1),
    ]
    baseline = _with_decision_trace(
        _payload(),
        execution=0,
        result=0,
        tokens=[2, 4],
        steps=[common, baseline_step],
        mode="baseline",
    )
    baseline["model"] = "gpt2"
    monitored = _with_decision_trace(
        baseline,
        execution=0,
        result=0,
        tokens=[2, 5],
        steps=[common, monitored_step],
        mode="monitored",
    )

    assert sampling_ambiguity_mismatches(baseline, monitored) == [
        "ambiguity.batch.case-a.outputs[0].decision_gap[1]"
    ]


def test_metamorphic_contract_validates_decision_trace_and_cumulative_sum():
    payload = _with_complete_decision_traces(_payload())

    assert metamorphic_mismatches(payload) == []

    payload["executions"][0]["results"][0]["outputs"][0][
        "cumulative_logprob"
    ] = 1.0
    assert metamorphic_mismatches(payload) == [
        "executions.batch.results[0].outputs[0].cumulative_logprob"
    ]


def test_metamorphic_contract_rejects_non_finite_decision_logprob():
    payload = _with_complete_decision_traces(_payload())
    payload["executions"][0]["results"][0]["outputs"][0][
        "decision_logprobs"
    ][0][0]["logprob"] = float("nan")

    assert metamorphic_mismatches(payload) == [
        "executions.batch.results[0].outputs[0].decision_logprobs[0]",
        "executions.batch.results[0].outputs[0].cumulative_logprob",
    ]


def test_metamorphic_contract_rejects_incomplete_decision_row():
    payload = _with_complete_decision_traces(_payload())
    del payload["executions"][0]["results"][0]["outputs"][0][
        "decision_logprobs"
    ][0][0]["decoded_token"]

    assert metamorphic_mismatches(payload) == [
        "executions.batch.results[0].outputs[0].decision_logprobs[0]",
        "executions.batch.results[0].outputs[0].cumulative_logprob",
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
                "decision_logprobs": None,
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
    }


def test_logprob_serializer_preserves_public_decision_evidence():
    from types import SimpleNamespace

    trace = [
        {
            5: SimpleNamespace(
                logprob=-0.25,
                rank=2,
                decoded_token="five",
            ),
            3: SimpleNamespace(
                logprob=-0.125,
                rank=1,
                decoded_token="three",
            ),
        }
    ]

    assert _serialize_logprob_trace(trace) == [
        [
            {
                "token_id": 3,
                "logprob": -0.125,
                "rank": 1,
                "decoded_token": "three",
            },
            {
                "token_id": 5,
                "logprob": -0.25,
                "rank": 2,
                "decoded_token": "five",
            },
        ]
    ]
