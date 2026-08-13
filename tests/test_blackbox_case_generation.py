"""CPU contracts for deterministic public-input testcase generation."""

from __future__ import annotations

import pytest

from tests.blackbox.case_generation import (
    GENERATOR_VERSION,
    deterministic_image_case,
    generate_cases,
    generate_prompts,
    validate_case_corpus,
)


pytestmark = pytest.mark.cpu


def test_generated_cases_are_reproducible_and_unique():
    first = generate_prompts(seed=17, count=20)
    second = generate_prompts(seed=17, count=20)

    assert first == second
    assert len(first) == len(set(first)) == 20


def test_generated_cases_change_with_seed():
    assert generate_prompts(seed=1, count=5) != generate_prompts(seed=2, count=5)


def test_generated_case_count_is_validated():
    with pytest.raises(ValueError, match="non-negative"):
        generate_prompts(seed=0, count=-1)


def test_generated_cases_are_auditable_and_reproducible():
    cases = generate_cases(seed=17, count=3)

    assert cases == generate_cases(seed=17, count=3)
    assert GENERATOR_VERSION == 2
    assert [case["case_id"] for case in cases] == [
        "generated-17-000",
        "generated-17-001",
        "generated-17-002",
    ]
    for case in cases:
        assert case["checklist_ids"]
        assert case["dimensions"]
        assert "differential" in case["oracles"]
        assert "reverse-batch-order" in case["oracles"]
        assert "scheduler-order-used-as-public-order" in case["kills"]


def test_repository_corpus_is_complete_and_auditable():
    import json
    from pathlib import Path

    corpus = json.loads(
        (Path(__file__).parent / "blackbox/cases/transparency.json").read_text()
    )

    validate_case_corpus(corpus)
    assert any(
        case["input"]["form"] == "token_ids_from_text"
        for case in corpus["cases"]
    )
    assert corpus["omitted_combinations"]


def test_corpus_validation_rejects_unmapped_oracle_free_cases():
    case = generate_cases(seed=1, count=1)[0]
    corpus = {
        "schema_version": 2,
        "name": "bad",
        "generator": {"name": "test", "version": 1},
        "executions": ["batch", "reversed"],
        "cases": [case],
        "omitted_combinations": [],
    }
    del case["kills"]

    with pytest.raises(ValueError, match="missing fields.*kills"):
        validate_case_corpus(corpus)


def test_deterministic_image_case_is_auditable() -> None:
    case = deterministic_image_case()
    corpus = {
        "schema_version": 2,
        "name": "image",
        "generator": {"name": "test", "version": 1},
        "executions": ["batch", "reversed"],
        "cases": [case],
        "omitted_combinations": [],
    }

    validate_case_corpus(corpus)
    assert case["input"]["form"] == "text_with_image"
    assert case["input"]["image"]["color"] == [17, 101, 203]
