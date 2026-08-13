"""CPU contracts for the version-pinned vLLM release-matrix runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import tests.tools.run_vllm_release_matrix as release_matrix
from tests.tools.run_vllm_release_matrix import (
    MatrixCase,
    _case_environment,
    _fatal_runtime_log_markers,
    _known_upstream_runtime_log_markers,
    _pytest_summary,
    _resume_manifest,
    _selected_gpus,
    _wait_for_idle,
    build_cases,
)


pytestmark = pytest.mark.cpu


def test_release_matrix_accepts_one_or_more_distinct_physical_gpus():
    assert _selected_gpus("2") == ("2",)
    assert _selected_gpus("3,2,1,0") == ("3", "2", "1", "0")
    with pytest.raises(ValueError, match="one or more distinct"):
        _selected_gpus("")
    with pytest.raises(ValueError, match="one or more distinct"):
        _selected_gpus("0,0")
    with pytest.raises(ValueError, match="integer"):
        _selected_gpus("gpu0,1")


def test_idle_gate_retries_transient_post_case_utilization(monkeypatch):
    results = [
        release_matrix.subprocess.CompletedProcess([], 1, "busy", ""),
        release_matrix.subprocess.CompletedProcess([], 0, "idle", ""),
    ]
    sleeps = []
    monkeypatch.setattr(
        release_matrix.subprocess,
        "run",
        lambda *args, **kwargs: results.pop(0),
    )
    monkeypatch.setattr(release_matrix.time, "sleep", sleeps.append)
    args = type(
        "Args",
        (),
        {
            "idle_samples": 3,
            "idle_interval": 2.0,
            "idle_retries": 3,
            "idle_retry_interval": 5.0,
        },
    )()

    result, attempts = _wait_for_idle(("0", "1"), args)

    assert result.returncode == 0
    assert attempts == 2
    assert sleeps == [5.0]


def test_all_matrix_covers_existing_architectures_and_storage_modes():
    cases = build_cases("all")
    case_ids = {case.case_id for case in cases}

    assert "focused-cpu-contracts" in case_ids
    for model in (
        "apertus",
        "ernie45",
        "falcon_h1",
        "gemma3",
        "gpt2",
        "granite",
        "jamba",
        "lfm2",
        "minicpm4",
        "qwen2",
        "qwen3",
        "llama",
        "mistral",
        "olmo3",
        "phi3",
        "qwen2_moe",
    ):
        assert any(case_id.startswith(f"public-{model}-") for case_id in case_ids)
    for model in ("gpt2", "qwen3", "llama", "qwen2_moe"):
        assert f"storage-{model}-eager-tp2" in case_ids
        assert f"storage-{model}-cudagraph-tp2" in case_ids
    assert "storage-gemma3-eager-tp1" in case_ids
    assert "storage-gemma3-cudagraph-tp1" in case_ids
    assert "storage-phi3-eager-tp1" in case_ids
    assert "storage-phi3-cudagraph-tp1" in case_ids
    assert "storage-mistral-eager-tp1" in case_ids
    assert "storage-mistral-cudagraph-tp1" in case_ids
    assert "storage-falcon_h1-eager-tp1" in case_ids
    assert "storage-falcon_h1-cudagraph-tp1" in case_ids
    assert "storage-granite-eager-tp1" in case_ids
    assert "storage-granite-cudagraph-tp1" in case_ids
    assert "storage-jamba-eager-tp1" in case_ids
    assert "storage-jamba-cudagraph-tp1" in case_ids
    assert "storage-lfm2-eager-tp1" in case_ids
    assert "storage-lfm2-cudagraph-tp1" in case_ids
    assert "storage-olmo3-eager-tp1" in case_ids
    assert "storage-olmo3-cudagraph-tp1" in case_ids
    assert "storage-apertus-eager-tp1" in case_ids
    assert "storage-apertus-cudagraph-tp1" in case_ids
    assert "storage-ernie45-eager-tp1" in case_ids
    assert "storage-ernie45-cudagraph-tp1" in case_ids
    assert "storage-minicpm4-eager-tp1" in case_ids
    assert "storage-minicpm4-cudagraph-tp1" in case_ids


def test_sota_matrix_pins_lite_h100_cells() -> None:
    cases = {case.case_id: case for case in build_cases("sota")}
    gpt_oss_revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
    gemma4_e2b_revision = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    glm52_revision = "b4734de4facf877f85769a911abafc5283eab3d9"
    llama4_scout_revision = "c2b440bc2b8c784ad310291d035b8550a771f24f"
    minimax_m27_revision = "d494266a4affc0d2995ba1fa35c8481cbd84294b"
    qwen3_moe_revision = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    qwen36_revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

    assert set(cases) == {
        "focused-cpu-contracts",
        "public-gemma4_e2b-tp1-eager-graph",
        "public-gpt_oss-tp1-eager-graph",
        "public-glm52-tp32-eager-graph",
        "public-llama4_scout-tp4-eager-graph",
        "public-minimax_m27-tp4-eager-graph",
        "public-qwen3_moe-tp1-eager-graph",
        "public-qwen36-tp1-eager-graph",
        "storage-gpt_oss-eager-tp1",
        "storage-gpt_oss-cudagraph-tp1",
        "storage-gemma4_e2b-eager-tp1",
        "storage-gemma4_e2b-cudagraph-tp1",
        "storage-glm52-eager-tp32",
        "storage-glm52-cudagraph-tp32",
        "storage-llama4_scout-eager-tp4",
        "storage-llama4_scout-cudagraph-tp4",
        "storage-minimax_m27-eager-tp4",
        "storage-minimax_m27-cudagraph-tp4",
        "storage-qwen3_moe-eager-tp1",
        "storage-qwen3_moe-cudagraph-tp1",
        "storage-qwen36-eager-tp1",
        "storage-qwen36-cudagraph-tp1",
    }
    public = cases["public-gemma4_e2b-tp1-eager-graph"]
    assert public.model_id == "google/gemma-4-E2B-it"
    assert public.environment["DMI_BLACKBOX_MODEL_REVISION"] == gemma4_e2b_revision
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    assert public.environment["DMI_BLACKBOX_MULTIMODAL_IMAGE"] == "1"
    assert public.environment["DMI_BLACKBOX_IMAGE_PLACEHOLDER"] == "<|image|>"
    assert public.environment["DMI_BLACKBOX_GENERATED_CASES"] == "2"
    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-gemma4_e2b-{mode}-tp1"]
        assert storage.model_id == "google/gemma-4-E2B-it"
        assert storage.environment["E2E_MODEL_REVISION"] == gemma4_e2b_revision
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "2048"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"

    public = cases["public-gpt_oss-tp1-eager-graph"]
    assert public.model_id == "openai/gpt-oss-20b"
    assert public.environment["DMI_BLACKBOX_MODEL_REVISION"] == gpt_oss_revision
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-gpt_oss-{mode}-tp1"]
        assert storage.model_id == "openai/gpt-oss-20b"
        assert storage.environment["E2E_MODEL_REVISION"] == gpt_oss_revision
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "2048"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"

    public = cases["public-glm52-tp32-eager-graph"]
    assert public.model_id == "zai-org/GLM-5.2"
    assert public.gpu_count == 32
    assert public.environment["DMI_BLACKBOX_MODEL_REVISION"] == glm52_revision
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    assert public.environment["DMI_BLACKBOX_GENERATED_CASES"] == "2"
    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-glm52-{mode}-tp32"]
        assert storage.model_id == "zai-org/GLM-5.2"
        assert storage.gpu_count == 32
        assert storage.environment["E2E_MODEL_REVISION"] == glm52_revision
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "2048"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"

    public = cases["public-llama4_scout-tp4-eager-graph"]
    assert public.model_id == "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    assert public.gpu_count == 4
    assert public.environment["DMI_BLACKBOX_MODEL_REVISION"] == llama4_scout_revision
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    assert public.environment["DMI_BLACKBOX_MULTIMODAL_IMAGE"] == "1"
    assert public.environment["DMI_BLACKBOX_GENERATED_CASES"] == "2"
    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-llama4_scout-{mode}-tp4"]
        assert storage.model_id == "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        assert storage.gpu_count == 4
        assert storage.environment["E2E_MODEL_REVISION"] == llama4_scout_revision
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "2048"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"

    public = cases["public-minimax_m27-tp4-eager-graph"]
    assert public.model_id == "MiniMaxAI/MiniMax-M2.7"
    assert public.gpu_count == 4
    assert public.environment["DMI_BLACKBOX_MODEL_REVISION"] == (
        minimax_m27_revision
    )
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    assert public.environment["DMI_BLACKBOX_GENERATED_CASES"] == "2"
    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-minimax_m27-{mode}-tp4"]
        assert storage.model_id == "MiniMaxAI/MiniMax-M2.7"
        assert storage.gpu_count == 4
        assert storage.environment["E2E_MODEL_REVISION"] == minimax_m27_revision
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "2048"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"

    public = cases["public-qwen3_moe-tp1-eager-graph"]
    assert public.model_id == "Qwen/Qwen3-30B-A3B"
    assert public.environment["DMI_BLACKBOX_MODEL_REVISION"] == qwen3_moe_revision
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-qwen3_moe-{mode}-tp1"]
        assert storage.model_id == "Qwen/Qwen3-30B-A3B"
        assert storage.environment["E2E_MODEL_REVISION"] == qwen3_moe_revision
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "2048"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"

    public = cases["public-qwen36-tp1-eager-graph"]
    assert public.model_id == "Qwen/Qwen3.6-27B"
    assert public.environment["DMI_BLACKBOX_MODEL_REVISION"] == qwen36_revision
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    assert public.environment["DMI_BLACKBOX_MULTIMODAL_IMAGE"] == "1"
    assert public.environment["DMI_BLACKBOX_IMAGE_PLACEHOLDER"] == (
        "<|vision_start|><|image_pad|><|vision_end|>"
    )
    assert public.environment["DMI_BLACKBOX_GENERATED_CASES"] == "2"
    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-qwen36-{mode}-tp1"]
        assert storage.model_id == "Qwen/Qwen3.6-27B"
        assert storage.environment["E2E_MODEL_REVISION"] == qwen36_revision
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "2048"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"


def test_gemma3_matrix_resolves_the_fixture_subfolder() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-gemma3-tp1-eager-graph"]
    assert public.model_id == "shibatch/tinygemma3-2m"
    assert public.environment["DMI_BLACKBOX_MODEL_SUBFOLDER"] == "hf"
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"

    storage = cases["storage-gemma3-cudagraph-tp1"]
    assert storage.environment["E2E_MODEL_SUBFOLDER"] == "hf"
    assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_phi3_public_matrix_uses_the_production_checkpoint() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-phi3-tp1-eager-graph"]
    assert public.model_id == "microsoft/Phi-3.5-mini-instruct"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "microsoft/Phi-3.5-mini-instruct"
    )
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "512"

    storage = cases["storage-phi3-cudagraph-tp1"]
    assert storage.model_id == (
        "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"
    )
    assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_mistral_matrix_separates_production_and_graph_safe_fixtures() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-mistral-tp1-eager-graph"]
    assert public.model_id == "mistralai/Mistral-7B-Instruct-v0.2"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "512"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-mistral-{mode}-tp1"]
        assert storage.model_id == "openaccess-ai-collective/tiny-mistral"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.2"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_falcon_h1_matrix_separates_production_and_hybrid_fixture() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-falcon_h1-tp1-eager-graph"]
    assert public.model_id == "tiiuae/Falcon-H1-0.5B-Instruct"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "tiiuae/Falcon-H1-0.5B-Instruct"
    )
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "512"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-falcon_h1-{mode}-tp1"]
        assert storage.model_id == "tiiuae/Falcon-H1-Tiny-90M-Instruct"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.2"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_lfm2_matrix_separates_production_and_hybrid_fixture() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-lfm2-tp1-eager-graph"]
    assert public.model_id == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "LiquidAI/LFM2.5-1.2B-Instruct"
    )
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "512"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-lfm2-{mode}-tp1"]
        assert storage.model_id == "tiny-random/lfm2"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.2"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_jamba_matrix_uses_the_qualified_dense_production_checkpoint() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-jamba-tp1-eager-graph"]
    assert public.model_id == "ai21labs/AI21-Jamba2-3B"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "ai21labs/AI21-Jamba2-3B"
    )
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-jamba-{mode}-tp1"]
        assert storage.model_id == "ai21labs/AI21-Jamba2-3B"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.5"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_granite_matrix_uses_the_qualified_41_production_checkpoint() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-granite-tp1-eager-graph"]
    assert public.model_id == "ibm-granite/granite-4.1-3b"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "ibm-granite/granite-4.1-3b"
    )
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-granite-{mode}-tp1"]
        assert storage.model_id == "ibm-granite/granite-4.1-3b"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.6"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_olmo3_matrix_uses_the_branch_complete_production_checkpoint() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-olmo3-tp1-eager-graph"]
    assert public.model_id == "allenai/Olmo-3-7B-Instruct"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "allenai/Olmo-3-7B-Instruct"
    )
    assert public.environment["DMI_BLACKBOX_GPU_MEMORY_UTILIZATION"] == "0.85"
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-olmo3-{mode}-tp1"]
        assert storage.model_id == "allenai/Olmo-3-7B-Instruct"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.9"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "1024"
        assert storage.environment["E2E_RING_PINNED_MB"] == "1024"


def test_apertus_matrix_uses_the_qualified_8b_production_checkpoint() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-apertus-tp1-eager-graph"]
    assert public.model_id == "swiss-ai/Apertus-8B-Instruct-2509"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "swiss-ai/Apertus-8B-Instruct-2509"
    )
    assert public.environment["DMI_BLACKBOX_GPU_MEMORY_UTILIZATION"] == "0.9"
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-apertus-{mode}-tp1"]
        assert storage.model_id == "swiss-ai/Apertus-8B-Instruct-2509"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.93"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_ernie45_matrix_uses_the_qualified_03b_checkpoint() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-ernie45-tp1-eager-graph"]
    assert public.model_id == "baidu/ERNIE-4.5-0.3B-PT"
    assert public.environment["DMI_BLACKBOX_MODEL"] == (
        "baidu/ERNIE-4.5-0.3B-PT"
    )
    assert public.environment["DMI_BLACKBOX_GPU_MEMORY_UTILIZATION"] == "0.3"
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-ernie45-{mode}-tp1"]
        assert storage.model_id == "baidu/ERNIE-4.5-0.3B-PT"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.3"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"


def test_minicpm4_matrix_pins_remote_code_and_the_official_checkpoint() -> None:
    cases = {case.case_id: case for case in build_cases("all")}

    public = cases["public-minicpm4-tp1-eager-graph"]
    assert public.model_id == "openbmb/MiniCPM4.1-8B"
    assert public.environment["DMI_BLACKBOX_MODEL"] == "openbmb/MiniCPM4.1-8B"
    assert public.environment["DMI_BLACKBOX_GPU_MEMORY_UTILIZATION"] == "0.93"
    assert public.environment["DMI_BLACKBOX_MAX_MODEL_LEN"] == "128"
    assert public.environment["DMI_BLACKBOX_TRUST_REMOTE_CODE"] == "1"
    assert (
        public.environment["DMI_BLACKBOX_MODEL_REVISION"]
        == "3a8dfed9c79a45e07dbff95bcd49d792343fa1a3"
    )

    for mode in ("eager", "cudagraph"):
        storage = cases[f"storage-minicpm4-{mode}-tp1"]
        assert storage.model_id == "openbmb/MiniCPM4.1-8B"
        assert storage.environment["E2E_GPU_MEM_UTIL"] == "0.93"
        assert storage.environment["E2E_REF_MAX_LEN"] == "128"
        assert storage.environment["E2E_MAX_MODEL_LEN"] == "128"
        assert storage.environment["E2E_RING_PAYLOAD_MB"] == "1024"
        assert storage.environment["E2E_RING_PINNED_MB"] == "1024"
        assert storage.environment["E2E_TRUST_REMOTE_CODE"] == "1"
        assert (
            storage.environment["E2E_MODEL_REVISION"]
            == "3a8dfed9c79a45e07dbff95bcd49d792343fa1a3"
        )


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
    output = (
        "generation completed\n"
        "WorkerProc hit an exception.\n"
        "Process manager: force killing remaining processes count=1\n"
    )

    assert _fatal_runtime_log_markers(output) == (
        "WorkerProc hit an exception.",
        "Process manager: force killing remaining process",
    )
    assert _fatal_runtime_log_markers("generation completed\n") == ()


def test_release_matrix_rejects_distributed_teardown_warnings() -> None:
    output = (
        '[rank1] Failed to check the "should dump" flag on TCPStore, '
        "server shut down too early\n"
    )

    assert _fatal_runtime_log_markers(output) == (
        'Failed to check the "should dump" flag on TCPStore',
    )


def test_release_matrix_retains_reproduced_upstream_teardown_warnings() -> None:
    output = (
        "Executor: workers still running after grace period; "
        "sending SIGTERM count=2\n"
    )

    assert _fatal_runtime_log_markers(output) == ()
    assert _known_upstream_runtime_log_markers(output) == (
        "Executor: workers still running after grace period; sending SIGTERM",
    )


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
