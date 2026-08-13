"""Contracts shared by public-API DMI black-box tests."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


_TRANSPARENCY_FIELDS = (
    "schema_version",
    "model",
    "cudagraph",
    "tensor_parallel_size",
    "prompts",
    "prompt_token_ids",
    "token_ids",
    "texts",
    "finish_reasons",
    "stop_reasons",
)

_TRANSPARENCY_V2_FIELDS = (
    "schema_version",
    "model",
    "cudagraph",
    "tensor_parallel_size",
    "corpus",
    "executions",
)


def _recursive_mismatches(
    baseline: Any,
    monitored: Any,
    *,
    path: str,
) -> list[str]:
    if type(baseline) is not type(monitored):
        return [
            f"{path}: type {type(baseline).__name__} "
            f"!= {type(monitored).__name__}"
        ]
    if isinstance(baseline, dict):
        mismatches: list[str] = []
        for key in sorted(set(baseline) | set(monitored)):
            child = f"{path}.{key}" if path else key
            if key not in baseline:
                mismatches.append(f"{child}: missing from baseline")
            elif key not in monitored:
                mismatches.append(f"{child}: missing from monitored")
            else:
                mismatches.extend(
                    _recursive_mismatches(baseline[key], monitored[key], path=child)
                )
        return mismatches
    if isinstance(baseline, list):
        if len(baseline) != len(monitored):
            return [f"{path}: length {len(baseline)} != {len(monitored)}"]
        mismatches = []
        for index, (left, right) in enumerate(zip(baseline, monitored)):
            mismatches.extend(
                _recursive_mismatches(left, right, path=f"{path}[{index}]")
            )
        return mismatches
    return [] if baseline == monitored else [path]


def transparency_mismatches(
    baseline: dict[str, Any],
    monitored: dict[str, Any],
) -> list[str]:
    """Return public output fields changed by enabling DMI monitoring."""

    schema_version = baseline.get("schema_version")
    if schema_version != monitored.get("schema_version"):
        return ["schema_version"]
    fields = (
        _TRANSPARENCY_V2_FIELDS
        if schema_version == 2
        else _TRANSPARENCY_FIELDS
    )
    missing = [
        field
        for field in fields
        if field not in baseline or field not in monitored
    ]
    if missing:
        return [f"missing required field: {field}" for field in missing]

    if schema_version != 2:
        return [field for field in fields if baseline[field] != monitored[field]]

    mode_errors = []
    if baseline.get("mode") != "baseline":
        mode_errors.append("baseline.mode")
    if monitored.get("mode") != "monitored":
        mode_errors.append("monitored.mode")
    if mode_errors:
        return mode_errors
    return _recursive_mismatches(
        {field: baseline[field] for field in fields},
        {field: monitored[field] for field in fields},
        path="",
    )


def _without_generated_outputs(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(payload)
    normalized["mode"] = "baseline"
    for execution in normalized.get("executions", []):
        if not isinstance(execution, dict):
            continue
        for result in execution.get("results", []):
            if isinstance(result, dict):
                result["outputs"] = "<baseline-envelope>"
    return normalized


def _output_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def baseline_instabilities(
    baselines: list[dict[str, Any]],
) -> list[str]:
    """Return identity-matched public completions varying across baselines."""

    if len(baselines) < 2:
        return []
    reference = baselines[0]
    instabilities: list[str] = []
    for execution_index, execution in enumerate(reference.get("executions", [])):
        for result_index, result in enumerate(execution.get("results", [])):
            observed = {
                _output_key(
                    baseline["executions"][execution_index]["results"][
                        result_index
                    ].get("outputs")
                )
                for baseline in baselines
            }
            if len(observed) > 1:
                instabilities.append(
                    f"{execution.get('execution_id')}.{result.get('case_id')}.outputs"
                )
    return instabilities


def baseline_envelope_mismatches(
    baselines: list[dict[str, Any]],
    monitored: dict[str, Any],
) -> list[str]:
    """Compare monitored output against a bounded observed baseline envelope.

    All public fields except generated candidates remain exact. Each monitored
    candidate list must equal one complete identity-matched candidate list seen
    in an independent baseline process; fields cannot be mixed across runs.
    """

    if len(baselines) < 2:
        return ["baseline envelope requires at least two baseline processes"]
    reference = baselines[0]
    mismatches: list[str] = []
    reference_shape = _without_generated_outputs(reference)
    for index, payload in enumerate([*baselines[1:], monitored], start=1):
        mismatches.extend(
            f"envelope[{index}].{path}"
            for path in _recursive_mismatches(
                reference_shape,
                _without_generated_outputs(payload),
                path="",
            )
        )
    if mismatches:
        return mismatches

    for execution_index, execution in enumerate(reference["executions"]):
        for result_index, result in enumerate(execution["results"]):
            observed = {
                _output_key(
                    baseline["executions"][execution_index]["results"][
                        result_index
                    ]["outputs"]
                )
                for baseline in baselines
            }
            candidate = monitored["executions"][execution_index]["results"][
                result_index
            ]["outputs"]
            if _output_key(candidate) not in observed:
                mismatches.append(
                    "envelope."
                    f"{execution.get('execution_id')}."
                    f"{result.get('case_id')}.outputs"
                )
    return mismatches


def metamorphic_mismatches(payload: dict[str, Any]) -> list[str]:
    """Check public attribution under reversal without assuming token invariance."""
    if payload.get("schema_version") != 2:
        return ["metamorphic oracle requires schema_version 2"]
    executions = payload.get("executions")
    if not isinstance(executions, list):
        return ["executions must be a list"]
    by_id = {
        execution.get("execution_id"): execution
        for execution in executions
        if isinstance(execution, dict)
    }
    if set(by_id) != {"batch", "reversed"}:
        return ["executions must contain exactly batch and reversed"]

    batch = by_id["batch"]
    reversed_run = by_id["reversed"]
    batch_order = batch.get("case_order")
    reversed_order = reversed_run.get("case_order")
    if not isinstance(batch_order, list) or reversed_order != list(
        reversed(batch_order)
    ):
        return ["executions.reversed.case_order"]

    corpus = payload.get("corpus")
    cases = corpus.get("cases") if isinstance(corpus, dict) else None
    if not isinstance(cases, list):
        return ["corpus.cases"]
    cases_by_id = {
        case.get("case_id"): case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }

    mismatches: list[str] = []
    if (
        len(cases_by_id) != len(cases)
        or len(set(batch_order)) != len(batch_order)
        or set(cases_by_id) != set(batch_order)
    ):
        mismatches.append("corpus.case_id_cardinality")
    for execution_id, execution in by_id.items():
        results = execution.get("results")
        if not isinstance(results, list):
            mismatches.append(f"executions.{execution_id}.results")
            continue
        result_order = [
            result.get("case_id") if isinstance(result, dict) else None
            for result in results
        ]
        if result_order != execution.get("case_order"):
            mismatches.append(f"executions.{execution_id}.result_order")
        request_ids = [
            result.get("request_id") if isinstance(result, dict) else None
            for result in results
        ]
        if any(not isinstance(value, str) or not value for value in request_ids):
            mismatches.append(f"executions.{execution_id}.request_ids_invalid")
        elif len(request_ids) != len(set(request_ids)):
            mismatches.append(f"executions.{execution_id}.request_ids_not_unique")
        for result_index, result in enumerate(results):
            if not isinstance(result, dict):
                mismatches.append(
                    f"executions.{execution_id}.results[{result_index}]"
                )
                continue
            case_id = result.get("case_id")
            case = cases_by_id.get(case_id)
            if case is None:
                continue
            base = f"executions.{execution_id}.results[{result_index}]"
            required_result_fields = {
                "case_id",
                "request_id",
                "prompt",
                "prompt_token_ids",
                "finished",
                "outputs",
            }
            for field in sorted(required_result_fields - set(result)):
                mismatches.append(f"{base}.{field}: missing")
            input_spec = case.get("input", {})
            expected_prompt = (
                input_spec.get("text")
                if input_spec.get("form") == "text"
                else None
            )
            if result.get("prompt") != expected_prompt:
                mismatches.append(f"{base}.prompt")
            prompt_token_ids = result.get("prompt_token_ids")
            if not isinstance(prompt_token_ids, list) or not all(
                isinstance(token_id, int) for token_id in prompt_token_ids
            ):
                mismatches.append(f"{base}.prompt_token_ids")
            if result.get("finished") is not True:
                mismatches.append(f"{base}.finished")
            outputs = result.get("outputs")
            if not isinstance(outputs, list) or len(outputs) != 1:
                mismatches.append(f"{base}.outputs")
                continue
            max_tokens = case.get("sampling", {}).get("max_tokens")
            min_tokens = case.get("sampling", {}).get("min_tokens", 0)
            for output_index, output in enumerate(outputs):
                output_base = f"{base}.outputs[{output_index}]"
                if not isinstance(output, dict):
                    mismatches.append(output_base)
                    continue
                required_output_fields = {
                    "index",
                    "text",
                    "token_ids",
                    "cumulative_logprob",
                    "finish_reason",
                    "stop_reason",
                }
                for field in sorted(required_output_fields - set(output)):
                    mismatches.append(f"{output_base}.{field}: missing")
                if output.get("index") != output_index:
                    mismatches.append(f"{output_base}.index")
                if not isinstance(output.get("text"), str):
                    mismatches.append(f"{output_base}.text")
                token_ids = output.get("token_ids")
                if not isinstance(token_ids, list) or not all(
                    isinstance(token_id, int) for token_id in token_ids
                ):
                    mismatches.append(f"{output_base}.token_ids")
                elif isinstance(max_tokens, int) and len(token_ids) > max_tokens:
                    mismatches.append(f"{output_base}.token_ids.max_tokens")
                elif isinstance(min_tokens, int) and len(token_ids) < min_tokens:
                    mismatches.append(f"{output_base}.token_ids.min_tokens")
                finish_reason = output.get("finish_reason")
                if finish_reason not in {"length", "stop"}:
                    mismatches.append(f"{output_base}.finish_reason")
                elif (
                    finish_reason == "length"
                    and isinstance(max_tokens, int)
                    and isinstance(token_ids, list)
                    and len(token_ids) != max_tokens
                ):
                    mismatches.append(f"{output_base}.token_ids.length_finish")
                if output.get("cumulative_logprob") is not None:
                    mismatches.append(f"{output_base}.cumulative_logprob")
                stop_reason = output.get("stop_reason")
                if stop_reason is not None and not isinstance(
                    stop_reason, (str, int)
                ):
                    mismatches.append(f"{output_base}.stop_reason")

    def keyed(execution: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            result["case_id"]: result
            for result in execution.get("results", [])
            if isinstance(result, dict) and isinstance(result.get("case_id"), str)
        }

    batch_results = keyed(batch)
    reversed_results = keyed(reversed_run)
    if (
        len(batch_results) != len(batch_order)
        or len(reversed_results) != len(batch_order)
        or set(batch_results) != set(reversed_results)
        or set(batch_results) != set(batch_order)
    ):
        mismatches.append("executions.case_id_cardinality")
        return mismatches
    for case_id in batch_order:
        # Changing batch order can change floating-point reduction shape. Even
        # greedy decoding is therefore not guaranteed to emit identical tokens.
        # Input attribution, however, is a public invariant and must not drift.
        left = batch_results[case_id]
        right = reversed_results[case_id]
        stable_fields = ("case_id", "prompt", "prompt_token_ids")
        mismatches.extend(
            _recursive_mismatches(
                {field: left.get(field) for field in stable_fields},
                {field: right.get(field) for field in stable_fields},
                path=f"metamorphic.{case_id}",
            )
        )
    return mismatches
