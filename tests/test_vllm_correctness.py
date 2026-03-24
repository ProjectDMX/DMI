"""vLLM E2E correctness test — three subprocesses, parent never touches CUDA.

  1. Reference: original model + FullHiddenStatesConnector -> disk
     (skipped for models not supported by extract_hidden_states, e.g. GPT-2)
  2. Monitored: hooked model + DMXGPUWorker + ring transport -> ClickHouse
  3. Comparator: reads both, validates row counts + value comparison

Environment variables:
  E2E_MODEL             "gpt2" (default) or "qwen3"
  E2E_NUM_PROMPTS       Number of prompts (default 8)
  E2E_MAX_NEW_TOKENS    Tokens to generate per prompt (default 20)
  E2E_ENFORCE_EAGER     "1" to disable torch.compile + CUDA graphs (default "0")
  E2E_RING_PAYLOAD_MB   Ring payload size in MB (default 4096)
  E2E_RING_PINNED_MB    Pinned staging size in MB (default 4096)
  E2E_HOOK_SELECTION    Hook selection preset (default "vllm-full")
  E2E_COMPARE_LAYERS    "all" or comma-separated layer IDs for value comparison.
                        Requires model supported by extract_hidden_states.
                        GPT-2 not supported -- value comparison skipped with warning.
  E2E_TOLERANCE         Max abs diff tolerance (default "0.01")
  DMX_DB_HOST           ClickHouse host (default "localhost")
  DMX_DB_PORT           ClickHouse port (default 9000)

Requires:
  - ClickHouse running on DMX_DB_HOST:DMX_DB_PORT
  - VLLM_DISABLE_COMPILE_CACHE=1 (set automatically)
  - LD_PRELOAD for libstdc++ if needed (caller's responsibility)

Usage:
  python -m pytest tests/test_vllm_correctness.py -q -s
  E2E_MODEL=qwen3 E2E_COMPARE_LAYERS=all python -m pytest tests/test_vllm_correctness.py -q -s
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import pytest
import torch

_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
}

_EXTRACT_HS_SUPPORTED = {"llama", "qwen", "qwen2", "qwen3", "minicpm",
                         "gpt_oss", "hunyuan_vl", "hunyuan_v1_dense",
                         "afmoe", "nemotron_h"}


def _model_supports_extract_hs(model_id: str) -> bool:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return getattr(cfg, "model_type", "") in _EXTRACT_HS_SUPPORTED


@pytest.mark.skipif(
    not torch.backends.cuda.is_built(), reason="CUDA not built")
def test_vllm_correctness(subtests):
    """vLLM E2E correctness: three subprocesses, parent never touches CUDA."""

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    compare_layers_str = os.environ.get("E2E_COMPARE_LAYERS", "")

    run_dir = tempfile.mkdtemp(prefix="vllm_e2e_")
    ref_dir = os.path.join(run_dir, "ref")
    mon_dir = os.path.join(run_dir, "mon")
    result_file = os.path.join(run_dir, "result.json")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    do_value_compare = bool(compare_layers_str)
    if do_value_compare and not _model_supports_extract_hs(model_id):
        warnings.warn(
            f"Value comparison skipped: {model_id} not supported by "
            f"extract_hidden_states (supported: {sorted(_EXTRACT_HS_SUPPORTED)}). "
            f"Only row-count validation will be performed.",
            stacklevel=1,
        )
        do_value_compare = False

    print(f"\n{'=' * 60}")
    print(f"  vLLM correctness test (subprocess mode)")
    print(f"  model={model_id}")
    print(f"  value_compare={do_value_compare}")
    print(f"{'=' * 60}")

    try:
        # Step 1: Reference run (only if value comparison requested + supported)
        if do_value_compare:
            print("\n  [1/3] Reference run (original model + FullHiddenStatesConnector)...",
                  flush=True)
            r1 = subprocess.run(
                [sys.executable, "-m", "tests.vllm_reference_runner",
                 "--output-dir", ref_dir],
                env=os.environ, capture_output=True, text=True, cwd=project_root,
            )
            if r1.returncode != 0:
                pytest.fail(f"Reference runner failed:\n{r1.stderr[-2000:]}")
        else:
            print("\n  [1/3] Reference run skipped", flush=True)
            os.makedirs(ref_dir, exist_ok=True)
            with open(os.path.join(ref_dir, "meta.json"), "w") as f:
                json.dump({"skipped": True}, f)

        # Step 2: Monitored run (hooked model + ring transport)
        print("  [2/3] Monitored run (hooked model + DMXGPUWorker)...", flush=True)
        r2 = subprocess.run(
            [sys.executable, "-m", "tests.vllm_monitored_runner",
             "--output-dir", mon_dir],
            env=os.environ, capture_output=True, text=True, cwd=project_root,
        )
        if r2.returncode != 0:
            pytest.fail(f"Monitored runner failed:\n{r2.stderr[-2000:]}")

        # Step 3: Comparator (CPU only)
        print("  [3/3] Comparing...", flush=True)
        r3 = subprocess.run(
            [sys.executable, "-m", "tests.vllm_comparator",
             "--ref-dir", ref_dir,
             "--mon-dir", mon_dir,
             "--result-file", result_file],
            env=os.environ, capture_output=True, text=True, cwd=project_root,
        )
        if r3.returncode != 0:
            pytest.fail(f"Comparator failed:\n{r3.stderr[-2000:]}")

        # Read results
        with open(result_file) as f:
            results = json.load(f)

        # Report via subtests
        for test in results["tests"]:
            with subtests.test(test["name"]):
                assert test["passed"], test.get("detail", "")

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
