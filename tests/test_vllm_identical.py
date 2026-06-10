"""vLLM identical check — bitwise tensor comparison between ref model
(GPU buffer D2D capture → disk) and monitored model (ring transport → ClickHouse).

Four steps, parent never touches CUDA:
  0. Sanity check: original vs ref model logprobs (informational, never fails)
  1. Reference run (RefDiskWorker, D2D capture → disk)
  2. Monitored run (DMXGPUWorker, ring transport → ClickHouse)
  3. Comparator (CPU only, logprob comparison + bitwise tensor check)

Environment variables:
  E2E_MODEL             "gpt2" (default) or "qwen3"
  E2E_NUM_PROMPTS       Number of prompts (default 8)
  E2E_MAX_NEW_TOKENS    Tokens to generate per prompt (default 20)
  E2E_ENFORCE_EAGER     "1" to disable CUDA graphs (default "1")
  E2E_DTYPE             Model dtype, e.g. "bfloat16", "float16", "auto" (default "bfloat16")
  E2E_REF_MAX_LEN       Max first-dim for buffers (default 8192)
  E2E_MAX_NUM_BATCHED_TOKENS  vLLM scheduler max_num_batched_tokens (default 512)
  E2E_RING_PAYLOAD_MB   Ring payload size (default 4096)
  E2E_RING_PINNED_MB    Pinned staging size (default 4096)
  E2E_HOOK_SELECTION    Public hook selection input (default "vllm-full");
                        translated to DMX_HOOK_SELECTION for subprocesses
  DMX_DB_HOST           ClickHouse host (default "localhost")
  DMX_DB_PORT           ClickHouse port (default 9000)

Requires:
  - ClickHouse running
  - VLLM_DISABLE_COMPILE_CACHE=1 (set automatically)

Usage:
  python -m pytest tests/test_vllm_identical.py -q -s
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.vllm,
    pytest.mark.clickhouse,
    pytest.mark.e2e,
]

_MODEL_REF_FILES = {
    "gpt2": "gpt2_ref.py",
    "qwen2_moe": "qwen2_moe_ref.py",
    "qwen3": "qwen3_ref.py",
    "llama": "llama_ref.py",
}


@pytest.mark.skipif(
    not torch.backends.cuda.is_built(), reason="CUDA not built")
def test_vllm_identical(subtests):
    """Bitwise comparison: ref model (disk) vs monitored model (ClickHouse)."""

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    hooks = os.environ.get("E2E_HOOK_SELECTION", "vllm-full")
    max_len = int(os.environ.get("E2E_REF_MAX_LEN", "8192"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "1")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(
        project_root, "integration", "vllm", "vllm",
        "model_executor", "models")

    ref_filename = _MODEL_REF_FILES.get(model_key)
    if ref_filename is None:
        pytest.skip(f"No ref model for {model_key}")
    model_file = os.path.join(models_dir, ref_filename)

    keep_artifacts = os.environ.get("E2E_KEEP_ARTIFACTS", "0") == "1"
    dump_compiled = os.environ.get("E2E_DUMP_COMPILED", "0") == "1"
    artifact_dir = os.environ.get("E2E_ARTIFACT_DIR")
    if artifact_dir:
        run_dir = os.path.abspath(artifact_dir)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = tempfile.mkdtemp(prefix="vllm_identical_")
    ref_dir = os.path.join(run_dir, "ref")
    mon_dir = os.path.join(run_dir, "mon")
    config_file = os.path.join(ref_dir, "ref_config.json")
    result_file = os.path.join(run_dir, "result.json")
    backup_file = os.path.join(run_dir, f"{ref_filename}.bak")
    orig_logprobs_file = os.path.join(run_dir, "logprobs_orig.pt")
    ref_logprobs_file = os.path.join(run_dir, "logprobs_ref.pt")

    print(f"\n{'=' * 60}")
    print(f"  vLLM identical check")
    print(f"  model={model_key}  hooks={hooks}  eager={enforce_eager}")
    print(f"  ref_max_len={max_len}")
    print(f"{'=' * 60}")

    # Build env for subprocesses (inherit + add our vars)
    sub_env = dict(os.environ)
    sub_env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    sub_env["E2E_ENFORCE_EAGER"] = enforce_eager
    sub_env["DMX_HOOK_SELECTION"] = hooks
    if dump_compiled:
        sub_env["TORCH_LOGS"] = "+output_code"

    try:
        # Backup ref model file
        shutil.copy2(model_file, backup_file)

        # Enable hooks via preprocessor
        print("\n  [0/3] Enabling ref hooks...", flush=True)
        os.makedirs(ref_dir, exist_ok=True)
        sys.path.insert(0, models_dir)
        from enable_ref_hooks import enable_ref_hooks
        enable_ref_hooks(
            model_file=model_file,
            hooks=hooks,
            max_len=max_len,
            output_dir=ref_dir,
            config_out=config_file,
        )

        # Step 0: Sanity check — original vs ref model logprobs
        # Runs AFTER enabling hooks to verify D2D copies don't affect output.
        print("\n  [0/4] Sanity check: original model logprobs...", flush=True)
        r0a = subprocess.run(
            [sys.executable, "-m", "tests.vllm_logprob_runner",
             "--output", orig_logprobs_file],
            env=sub_env, capture_output=True, text=True, cwd=project_root,
        )
        print(r0a.stdout[-1000:] if r0a.stdout else "", flush=True)
        if dump_compiled and r0a.stderr:
            with open(os.path.join(run_dir, "compile_orig.log"), "w") as f:
                f.write(r0a.stderr)
        if r0a.returncode != 0:
            print(r0a.stderr[-2000:] if r0a.stderr else "", flush=True)
            print("  WARNING: original logprob run failed, skipping sanity check")
            orig_logprobs_file = None

        print("  [0/4] Sanity check: ref model logprobs...", flush=True)
        ref_lp_env = dict(sub_env)
        ref_lp_env["REF_CONFIG"] = config_file
        r0b = subprocess.run(
            [sys.executable, "-m", "tests.vllm_logprob_runner",
             "--output", ref_logprobs_file, "--ref"],
            env=ref_lp_env, capture_output=True, text=True, cwd=project_root,
        )
        print(r0b.stdout[-1000:] if r0b.stdout else "", flush=True)
        if dump_compiled and r0b.stderr:
            with open(os.path.join(run_dir, "compile_ref_logprob.log"), "w") as f:
                f.write(r0b.stderr)
        if r0b.returncode != 0:
            print(r0b.stderr[-2000:] if r0b.stderr else "", flush=True)
            print("  WARNING: ref logprob run failed, skipping sanity check")
            ref_logprobs_file = None

        # Step 0c: Monitored model logprobs (baseline vs monitored comparison)
        mon_logprobs_file = os.path.join(run_dir, "logprobs_mon.pt")
        print("  [0/4] Sanity check: monitored model logprobs...", flush=True)
        r0c = subprocess.run(
            [sys.executable, "-m", "tests.vllm_logprob_runner",
             "--output", mon_logprobs_file, "--monitored"],
            env=sub_env, capture_output=True, text=True, cwd=project_root,
        )
        print(r0c.stdout[-1000:] if r0c.stdout else "", flush=True)
        if dump_compiled and r0c.stderr:
            with open(os.path.join(run_dir, "compile_mon_logprob.log"), "w") as f:
                f.write(r0c.stderr)
        if r0c.returncode != 0:
            print(r0c.stderr[-2000:] if r0c.stderr else "", flush=True)
            print("  WARNING: monitored logprob run failed, skipping")
            mon_logprobs_file = None

        # Step 1: Reference run
        print("\n  [1/4] Reference run (RefDiskWorker)...", flush=True)
        ref_env = dict(sub_env)
        ref_env["REF_CONFIG"] = config_file
        r1 = subprocess.run(
            [sys.executable, "-m", "tests.vllm_ref_runner",
             "--output-dir", ref_dir],
            env=ref_env, capture_output=True, text=True, cwd=project_root,
        )
        print(r1.stdout[-2000:] if r1.stdout else "", flush=True)
        if keep_artifacts and r1.stdout:
            with open(os.path.join(run_dir, "stdout_ref_runner.log"), "w") as f:
                f.write(r1.stdout)
        if dump_compiled and r1.stderr:
            with open(os.path.join(run_dir, "compile_ref_runner.log"), "w") as f:
                f.write(r1.stderr)
        if r1.returncode != 0:
            print(r1.stderr[-3000:] if r1.stderr else "", flush=True)
            pytest.fail(f"Ref runner failed (rc={r1.returncode})")

        # Restore ref model from backup (before monitored run)
        shutil.copy2(backup_file, model_file)

        # Step 2: Monitored run
        print("\n  [2/4] Monitored run (DMXGPUWorker)...", flush=True)
        r2 = subprocess.run(
            [sys.executable, "-m", "tests.vllm_monitored_runner",
             "--output-dir", mon_dir],
            env=sub_env, capture_output=True, text=True, cwd=project_root,
        )
        print(r2.stdout[-2000:] if r2.stdout else "", flush=True)
        if keep_artifacts and r2.stdout:
            with open(os.path.join(run_dir, "stdout_mon_runner.log"), "w") as f:
                f.write(r2.stdout)
        if dump_compiled and r2.stderr:
            with open(os.path.join(run_dir, "compile_mon_runner.log"), "w") as f:
                f.write(r2.stderr)
        if r2.returncode != 0:
            print(r2.stderr[-3000:] if r2.stderr else "", flush=True)
            pytest.fail(f"Monitored runner failed (rc={r2.returncode})")

        # Step 3: Comparator (includes logprob sanity check if available)
        print("\n  [3/4] Comparing (bitwise check)...", flush=True)
        cmp_cmd = [
            sys.executable, "-m", "tests.vllm_identical_comparator",
            "--ref-config", config_file,
            "--mon-dir", mon_dir,
            "--result-file", result_file,
        ]
        if orig_logprobs_file and os.path.exists(orig_logprobs_file):
            cmp_cmd += ["--orig-logprobs", orig_logprobs_file]
        if ref_logprobs_file and os.path.exists(ref_logprobs_file):
            cmp_cmd += ["--ref-logprobs", ref_logprobs_file]
        if mon_logprobs_file and os.path.exists(mon_logprobs_file):
            cmp_cmd += ["--mon-logprobs", mon_logprobs_file]
        r3 = subprocess.run(
            cmp_cmd,
            env=sub_env, capture_output=True, text=True, cwd=project_root,
        )
        if r3.stdout:
            # Always print LOGPROBS summary and PASS/FAIL lines first
            for line in r3.stdout.splitlines():
                if "[LOGPROBS" in line or "ALL PASSED" in line or "FAILED (" in line:
                    print(line, flush=True)
            # Then print tail for hidden state details
            print(r3.stdout[-2000:], flush=True)
        if r3.returncode != 0:
            print(r3.stderr[-3000:] if r3.stderr else "", flush=True)
            pytest.fail(f"Comparator failed (rc={r3.returncode})")

        # Report via subtests
        with open(result_file) as f:
            results = json.load(f)

        for test in results["tests"]:
            with subtests.test(test["name"]):
                assert test["passed"], test.get("detail", "")

    finally:
        # Always restore ref model file
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, model_file)
        if keep_artifacts:
            print(f"\n  [kept] run_dir = {run_dir}", flush=True)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
