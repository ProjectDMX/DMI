"""Black-box DMI transparency test using only vLLM's public offline API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests._requirements import require_cuda, require_model_cache, require_vllm
from tests.blackbox.case_generation import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    generate_cases,
)
from tests.blackbox.contracts import (
    MAX_DECISION_LOGPROB_DRIFT,
    MAX_GREEDY_BRANCH_GAP,
    MAX_GREEDY_SELECTED_GAP,
    baseline_envelope_mismatches,
    baseline_instabilities,
    metamorphic_mismatches,
    sampling_ambiguity_mismatches,
    transparency_mismatches,
)
from tests.process_group import run_process_group


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tests/tools/smoke_vllm_model.py"
CASES = PROJECT_ROOT / "tests/blackbox/cases/transparency.json"
MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
MODEL_ARG = os.environ.get("DMI_BLACKBOX_MODEL", "qwen2")
MODEL_ID = MODEL_ALIASES.get(MODEL_ARG, MODEL_ARG)
TP_SIZE = int(os.environ.get("DMI_BLACKBOX_TP_SIZE", "1"))
GPU_MEMORY_UTILIZATION = os.environ.get(
    "DMI_BLACKBOX_GPU_MEMORY_UTILIZATION", "0.4"
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.vllm,
    pytest.mark.e2e,
    require_cuda(),
    require_vllm(),
    require_model_cache(MODEL_ID),
]


def _run(
    mode: str,
    output: Path,
    *,
    cases: Path,
    cudagraph: bool,
) -> dict:
    command = [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--model",
        MODEL_ARG,
        "--cases",
        str(cases),
        "--output",
        str(output),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
    ]
    if cudagraph:
        command.append("--cudagraph")
    command.extend(
        [
            "--decision-logprobs",
            os.environ.get("DMI_BLACKBOX_DECISION_LOGPROBS", "20"),
        ]
    )

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    timeout = float(os.environ.get("DMI_BLACKBOX_PROCESS_TIMEOUT", "1800"))
    try:
        result = run_process_group(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"{mode} black-box runner exceeded {timeout}s and its process "
            f"group was terminated\nstdout:\n{(error.output or '')[-4000:]}\n"
            f"stderr:\n{(error.stderr or '')[-4000:]}"
        )
    assert result.returncode == 0, (
        f"{mode} black-box runner failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert output.is_file(), f"{mode} runner did not create {output}"
    return json.loads(output.read_text())


def _assert_public_payload(mode: str, payload: dict, seed: int) -> None:
    mismatches = metamorphic_mismatches(payload)
    assert not mismatches, (
        f"{mode} public schema/attribution relation failed: {mismatches}\n"
        f"case_seed={seed}\n"
        f"payload={json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


@pytest.mark.parametrize(
    "cudagraph",
    [False, True]
    if os.environ.get("DMI_BLACKBOX_CUDAGRAPH", "0") == "1"
    else [False],
    ids=lambda enabled: "cudagraph" if enabled else "eager",
)
def test_monitoring_is_transparent_at_the_public_vllm_api(tmp_path, cudagraph):
    artifact_root = os.environ.get("DMI_BLACKBOX_ARTIFACT_DIR")
    if artifact_root:
        run_dir = Path(artifact_root) / (
            "cudagraph" if cudagraph else "eager"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = tmp_path
    core = json.loads(CASES.read_text())
    seed = int(os.environ.get("DMI_BLACKBOX_SEED", "20260812"))
    generated_count = int(os.environ.get("DMI_BLACKBOX_GENERATED_CASES", "6"))
    core["name"] = f"{core['name']}+generated"
    core["generator"] = {
        "name": GENERATOR_NAME,
        "version": GENERATOR_VERSION,
        "seed": seed,
        "generated_count": generated_count,
    }
    core["cases"].extend(generate_cases(seed=seed, count=generated_count))
    cases = run_dir / "cases.json"
    cases.write_text(json.dumps(core, indent=2, ensure_ascii=False) + "\n")

    baseline = _run(
        "baseline",
        run_dir / "baseline.json",
        cases=cases,
        cudagraph=cudagraph,
    )
    monitored = _run(
        "monitored",
        run_dir / "monitored.json",
        cases=cases,
        cudagraph=cudagraph,
    )
    _assert_public_payload("baseline-1", baseline, seed)
    _assert_public_payload("monitored", monitored, seed)

    mismatches = transparency_mismatches(baseline, monitored)
    baselines = [baseline]
    envelope_mismatches: list[str] = []
    instabilities: list[str] = []
    ambiguity_mismatches = sampling_ambiguity_mismatches(
        baseline, monitored
    )
    decision_count = baseline.get("decision_logprobs")
    decision_evidence_available = (
        isinstance(decision_count, int)
        and not isinstance(decision_count, bool)
        and decision_count >= 2
    )
    if not mismatches:
        assert not ambiguity_mismatches, (
            "DMI changed public decision logprobs beyond the declared "
            f"tolerance: {ambiguity_mismatches}\ncase_seed={seed}"
        )
    if mismatches:
        if not ambiguity_mismatches:
            (run_dir / "stability.json").write_text(
                json.dumps(
                    {
                        "strict_mismatches": mismatches,
                        "accepted_oracle": "public-decision-logprob",
                        "sampling_ambiguity_mismatches": [],
                        "sampling_ambiguity_thresholds": {
                            "max_branch_gap": MAX_GREEDY_BRANCH_GAP,
                            "max_selected_gap": MAX_GREEDY_SELECTED_GAP,
                            "max_cross_run_drift": MAX_DECISION_LOGPROB_DRIFT,
                        },
                        "baseline_processes": 1,
                        "baseline_instabilities": [],
                        "envelope_mismatches": [],
                    },
                    indent=2,
                )
                + "\n"
            )
            return
        if decision_evidence_available:
            (run_dir / "stability.json").write_text(
                json.dumps(
                    {
                        "strict_mismatches": mismatches,
                        "accepted_oracle": None,
                        "sampling_ambiguity_mismatches": ambiguity_mismatches,
                        "sampling_ambiguity_thresholds": {
                            "max_branch_gap": MAX_GREEDY_BRANCH_GAP,
                            "max_selected_gap": MAX_GREEDY_SELECTED_GAP,
                            "max_cross_run_drift": MAX_DECISION_LOGPROB_DRIFT,
                        },
                        "baseline_processes": 1,
                        "baseline_instabilities": [],
                        "envelope_mismatches": [
                            "not attempted: public decision evidence was available"
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
            pytest.fail(
                "DMI output failed the public decision-logprob oracle: "
                f"strict={mismatches}, ambiguity={ambiguity_mismatches}\n"
                f"case_seed={seed}"
            )
        max_baselines = int(
            os.environ.get("DMI_BLACKBOX_MAX_BASELINES", "3")
        )
        if max_baselines < 2:
            raise ValueError("DMI_BLACKBOX_MAX_BASELINES must be at least 2")
        for replicate_index in range(2, max_baselines + 1):
            replicate = _run(
                "baseline",
                run_dir / f"baseline-replica-{replicate_index}.json",
                cases=cases,
                cudagraph=cudagraph,
            )
            baselines.append(replicate)
            _assert_public_payload(
                f"baseline-{replicate_index}", replicate, seed
            )
            envelope_mismatches = baseline_envelope_mismatches(
                baselines, monitored
            )
            instabilities = baseline_instabilities(baselines)
            if instabilities and not envelope_mismatches:
                break
        (run_dir / "stability.json").write_text(
            json.dumps(
                {
                    "strict_mismatches": mismatches,
                    "accepted_oracle": (
                        "baseline-envelope"
                        if instabilities and not envelope_mismatches
                        else None
                    ),
                    "sampling_ambiguity_mismatches": ambiguity_mismatches,
                    "sampling_ambiguity_thresholds": {
                        "max_branch_gap": MAX_GREEDY_BRANCH_GAP,
                        "max_selected_gap": MAX_GREEDY_SELECTED_GAP,
                        "max_cross_run_drift": MAX_DECISION_LOGPROB_DRIFT,
                    },
                    "baseline_processes": len(baselines),
                    "baseline_instabilities": instabilities,
                    "envelope_mismatches": envelope_mismatches,
                },
                indent=2,
            )
            + "\n"
        )
        assert instabilities and not envelope_mismatches, (
            "DMI output was outside the bounded public baseline envelope: "
            f"strict={mismatches}, instabilities={instabilities}, "
            f"envelope={envelope_mismatches}\ncase_seed={seed}\n"
            f"baseline={json.dumps(baseline, indent=2, ensure_ascii=False)}\n"
            f"monitored={json.dumps(monitored, indent=2, ensure_ascii=False)}"
        )
