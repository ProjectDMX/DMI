"""Black-box DMI transparency test using only vLLM's public offline API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests._requirements import require_cuda, require_model_cache, require_vllm
from tests.blackbox.case_generation import generate_prompts
from tests.blackbox.contracts import transparency_mismatches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tests/tools/smoke_vllm_model.py"
CASES = PROJECT_ROOT / "tests/blackbox/cases/transparency.json"
MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B",
}
MODEL_ARG = os.environ.get("DMI_BLACKBOX_MODEL", "qwen2")
MODEL_ID = MODEL_ALIASES.get(MODEL_ARG, MODEL_ARG)

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
    ]
    if cudagraph:
        command.append("--cudagraph")

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{mode} black-box runner failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert output.is_file(), f"{mode} runner did not create {output}"
    return json.loads(output.read_text())


@pytest.mark.parametrize(
    "cudagraph",
    [False, True]
    if os.environ.get("DMI_BLACKBOX_CUDAGRAPH", "0") == "1"
    else [False],
    ids=lambda enabled: "cudagraph" if enabled else "eager",
)
def test_monitoring_is_transparent_at_the_public_vllm_api(tmp_path, cudagraph):
    core = json.loads(CASES.read_text())
    seed = int(os.environ.get("DMI_BLACKBOX_SEED", "20260812"))
    generated_count = int(os.environ.get("DMI_BLACKBOX_GENERATED_CASES", "6"))
    core["name"] = f"{core['name']}+generated"
    core["seed"] = seed
    core["prompts"].extend(generate_prompts(seed=seed, count=generated_count))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(core, indent=2, ensure_ascii=False) + "\n")

    baseline = _run(
        "baseline",
        tmp_path / "baseline.json",
        cases=cases,
        cudagraph=cudagraph,
    )
    monitored = _run(
        "monitored",
        tmp_path / "monitored.json",
        cases=cases,
        cudagraph=cudagraph,
    )

    mismatches = transparency_mismatches(baseline, monitored)
    assert not mismatches, (
        f"DMI changed public vLLM fields: {mismatches}\n"
        f"case_seed={seed}\n"
        f"baseline={json.dumps(baseline, indent=2, ensure_ascii=False)}\n"
        f"monitored={json.dumps(monitored, indent=2, ensure_ascii=False)}"
    )
