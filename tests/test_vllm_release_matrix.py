"""CPU contracts for the version-pinned vLLM release-matrix runner."""

from __future__ import annotations

import pytest

from tests.tools.run_vllm_release_matrix import _selected_gpus, build_cases


pytestmark = pytest.mark.cpu


def test_release_matrix_requires_two_distinct_physical_gpus():
    assert _selected_gpus("2,0") == ("2", "0")
    with pytest.raises(ValueError, match="two distinct"):
        _selected_gpus("0")
    with pytest.raises(ValueError, match="two distinct"):
        _selected_gpus("0,0")
    with pytest.raises(ValueError, match="integer"):
        _selected_gpus("gpu0,1")


def test_all_matrix_covers_existing_architectures_and_storage_modes():
    cases = build_cases("all")
    case_ids = {case.case_id for case in cases}

    assert "focused-cpu-contracts" in case_ids
    for model in ("gpt2", "qwen2", "qwen3", "llama", "qwen2_moe"):
        assert any(case_id.startswith(f"public-{model}-") for case_id in case_ids)
    for model in ("gpt2", "qwen3", "llama", "qwen2_moe"):
        assert f"storage-{model}-eager-tp2" in case_ids
        assert f"storage-{model}-cudagraph-tp2" in case_ids


def test_static_only_models_use_bounded_two_gpu_storage_cells():
    cases = {case.case_id: case for case in build_cases("storage")}

    for model in ("llama", "qwen2_moe"):
        case = cases[f"storage-{model}-eager-tp2"]
        assert case.gpu_count == 2
        assert case.environment["DMX_HOOK_SELECTION"] == "resid_pre"
        assert case.environment["E2E_REF_MAX_LEN"] == "512"
