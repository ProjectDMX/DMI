"""CPU contracts for deterministic public-input testcase generation."""

from __future__ import annotations

import pytest

from tests.blackbox.case_generation import generate_prompts


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
