"""CPU contracts for the version-pinned vLLM release-matrix runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tests.tools.run_vllm_release_matrix import (
    MatrixCase,
    _case_environment,
    _fatal_runtime_log_markers,
    _pytest_summary,
    _resume_manifest,
    _selected_gpus,
    build_cases,
)


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


def test_case_environment_pins_shell_children_to_matrix_python(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    environment = _case_environment(
        MatrixCase("storage", ("bash", "runner.sh"), {}, gpu_count=2),
        ("2", "0"),
    )

    assert environment["DMI_MATRIX_PYTHON"] == sys.executable
    assert environment["PATH"].split(":", 1)[0] == str(
        Path(sys.executable).parent
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "2,0"
    assert environment["E2E_GPUS"] == "2,0"


def test_storage_wrapper_uses_the_pinned_matrix_python():
    wrapper = Path("tests/tools/run_tp_compare_vllm.sh").read_text()

    assert '"${DMI_MATRIX_PYTHON:-python}" -m tests.vllm_compare_runner' in wrapper


def test_release_matrix_rejects_worker_errors_despite_zero_exit_code():
    output = "generation completed\nWorkerProc hit an exception.\n"

    assert _fatal_runtime_log_markers(output) == (
        "WorkerProc hit an exception.",
    )
    assert _fatal_runtime_log_markers("generation completed\n") == ()


def test_pytest_summary_retains_skipped_prerequisites(tmp_path: Path):
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1">'
        '<testcase name="passed"/><testcase name="also-passed"/>'
        '<testcase name="missing"><skipped/></testcase>'
        "</testsuite></testsuites>"
    )

    assert _pytest_summary(report) == {
        "tests": 3,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
    }


def _resume_identity(tmp_path: Path) -> tuple[list[MatrixCase], dict]:
    log = tmp_path / "focused.log"
    log.write_text("passed\n")
    cases = [
        MatrixCase("focused", ("pytest",), {}),
        MatrixCase("gpu", ("pytest",), {}, gpu_count=1),
    ]
    manifest = {
        "schema_version": 1,
        "project_root": str(Path(__file__).resolve().parents[1]),
        "root_commit": "root-sha",
        "vllm_integration_commit": "vllm-sha",
        "runtime": {"vllm_version": "0.25.1"},
        "physical_gpus": ["1", "2"],
        "phase": "all",
        "results": [
            {
                "case_id": "focused",
                "status": "passed",
                "returncode": 0,
                "log": str(log),
                "pytest_summary": {
                    "tests": 1,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                },
            },
            {
                "case_id": "gpu",
                "status": "blocked-prerequisite",
            },
        ],
        "finished_at": "old",
        "passed": 1,
        "failed": 0,
        "blocked": 1,
    }
    return cases, manifest


def test_resume_keeps_only_a_verified_passed_prefix(tmp_path: Path):
    cases, manifest = _resume_identity(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    resumed = _resume_manifest(
        path,
        cases=cases,
        root_commit="root-sha",
        integration_commit="vllm-sha",
        runtime={"vllm_version": "0.25.1"},
        gpus=("1", "2"),
        phase="all",
    )

    assert [result["case_id"] for result in resumed["results"]] == ["focused"]
    assert len(resumed["resumed_at"]) == 1
    assert "finished_at" not in resumed
    assert "blocked" not in resumed


def test_resume_rejects_identity_or_evidence_mismatch(tmp_path: Path):
    cases, manifest = _resume_identity(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="root_commit mismatch"):
        _resume_manifest(
            path,
            cases=cases,
            root_commit="different",
            integration_commit="vllm-sha",
            runtime={"vllm_version": "0.25.1"},
            gpus=("1", "2"),
            phase="all",
        )

    manifest["results"][0]["log"] = str(tmp_path / "missing.log")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evidence log is missing"):
        _resume_manifest(
            path,
            cases=cases,
            root_commit="root-sha",
            integration_commit="vllm-sha",
            runtime={"vllm_version": "0.25.1"},
            gpus=("1", "2"),
            phase="all",
        )
