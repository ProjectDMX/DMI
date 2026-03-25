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
  python -m pytest tests/test_vllm_rowcnt.py -q -s
  E2E_MODEL=qwen3 E2E_COMPARE_LAYERS=all python -m pytest tests/test_vllm_rowcnt.py -q -s
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
}

@pytest.mark.skipif(
    not torch.backends.cuda.is_built(), reason="CUDA not built")
def test_vllm_rowcnt(subtests):
    """vLLM row-count validation: monitored run + row-count check."""

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)

    run_dir = tempfile.mkdtemp(prefix="vllm_rowcnt_")
    ref_dir = os.path.join(run_dir, "ref")
    mon_dir = os.path.join(run_dir, "mon")
    result_file = os.path.join(run_dir, "result.json")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ref_dir still needed by comparator (with skipped marker)
    os.makedirs(ref_dir, exist_ok=True)
    with open(os.path.join(ref_dir, "meta.json"), "w") as f:
        json.dump({"skipped": True}, f)

    print(f"\n{'=' * 60}")
    print(f"  vLLM row-count test")
    print(f"  model={model_id}")
    print(f"{'=' * 60}")

    try:
        # Step 1: Monitored run (hooked model + ring transport)
        print("\n  [1/2] Monitored run (hooked model + DMXGPUWorker)...", flush=True)
        r2 = subprocess.run(
            [sys.executable, "-m", "tests.vllm_monitored_runner",
             "--output-dir", mon_dir],
            env=os.environ, capture_output=True, text=True, cwd=project_root,
        )
        if r2.returncode != 0:
            pytest.fail(f"Monitored runner failed:\n{r2.stderr[-2000:]}")

        # Step 2: Comparator (CPU only, row-count validation)
        print("  [2/2] Checking row counts...", flush=True)
        r3 = subprocess.run(
            [sys.executable, "-m", "tests.vllm_rowcnt_comparator",
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
