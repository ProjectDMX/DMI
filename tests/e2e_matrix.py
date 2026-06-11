"""Configurable E2E matrix harness (plan §8).

One matrix-driven entry point replacing the hardcoded shell sweeps.  Each
axis is a comma-separated multi-value flag; the harness takes the Cartesian
product, runs every cell as subprocesses (reusing the existing runners and
comparators -- no inference logic is reimplemented), and writes one JSON
record per cell.

    python -m tests.e2e_matrix \
        --backend hf,vllm --model gpt2,qwen3 \
        --mode eager,cuda_graph --standard transport_bitwise \
        --hooks vllm-full --tp 1 --out results/e2e.jsonl

Cell dispatch
-------------
- ``vllm`` + ``bitwise`` / ``transport_bitwise``
      vllm_ref_runner (RefDiskWorker, D2D->disk) + vllm_monitored_runner
      (ring->ClickHouse) -> vllm_identical_comparator.
- ``vllm`` + ``row_count`` / ``allclose``
      vllm_monitored_runner -> vllm_rowcnt_comparator.
- ``hf``   (any standard)
      hf_reference_runner + hf_monitored_runner -> hf_comparator.

The public ``E2E_HOOK_SELECTION`` input is translated to the internal
``DMX_HOOK_SELECTION`` runtime contract in each subprocess env (plan §2).

``--dry-run`` prints the planned cells + dispatch commands without touching
CUDA / ClickHouse, so the expansion and env translation are unit-testable on
a CPU-only box.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

from tests.lib.report import CellResult, checks_from_legacy_result, write_jsonl, human_table

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model key -> vLLM ref model source filename (under model_executor/models).
_VLLM_REF_FILES = {
    "gpt2": "gpt2_ref.py",
    "qwen2_moe": "qwen2_moe_ref.py",
    "qwen3": "qwen3_ref.py",
    "llama": "llama_ref.py",
}

# Standards that compare reference D2D buffers against ring/ClickHouse output.
_VLLM_IDENTICAL_STANDARDS = {"bitwise", "transport_bitwise"}


@dataclass(frozen=True)
class Cell:
    """One point in the matrix."""

    backend: str
    model: str
    mode: str
    standard: str
    hooks: str
    tp: int = 1
    ring_mb: int = 4096
    dtype: str = "bfloat16"
    prompt_set: str = "smoke"

    @property
    def enforce_eager(self) -> str:
        return "1" if self.mode == "eager" else "0"


# ---------------------------------------------------------------------------
# Axis expansion
# ---------------------------------------------------------------------------


def _split(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_cells(args: argparse.Namespace) -> List[Cell]:
    """Cartesian product over every multi-value axis."""
    cells: List[Cell] = []
    for backend, model, mode, standard, hooks, tp, ring_mb, dtype, pset in itertools.product(
        _split(args.backend), _split(args.model), _split(args.mode),
        _split(args.standard), _split(args.hooks), _split(str(args.tp)),
        _split(str(args.ring_mb)), _split(args.dtype), _split(args.prompt_set),
    ):
        cells.append(Cell(
            backend=backend, model=model, mode=mode, standard=standard,
            hooks=hooks, tp=int(tp), ring_mb=int(ring_mb), dtype=dtype,
            prompt_set=pset,
        ))
    return cells


def cell_env(cell: Cell, args: argparse.Namespace, base: Optional[dict] = None) -> dict:
    """Build the subprocess env for a cell.

    Sets the public ``E2E_*`` knobs *and* the translated internal
    ``DMX_HOOK_SELECTION`` (plan §2) so the runners see one consistent
    configuration.
    """
    env = dict(base if base is not None else os.environ)
    env["E2E_MODEL"] = cell.model
    env["E2E_ENFORCE_EAGER"] = cell.enforce_eager
    env["E2E_CUDA_GRAPHS"] = "0" if cell.mode == "eager" else "1"
    env["E2E_DTYPE"] = cell.dtype
    env["E2E_TP_SIZE"] = str(cell.tp)
    env["E2E_RING_PAYLOAD_MB"] = str(cell.ring_mb)
    env["E2E_RING_PINNED_MB"] = str(cell.ring_mb)
    env["E2E_PROMPT_SET"] = cell.prompt_set
    # Public hook-selection input + internal runtime contract translation.
    env["E2E_HOOK_SELECTION"] = cell.hooks
    env["DMX_HOOK_SELECTION"] = cell.hooks
    env["E2E_NUM_PROMPTS"] = str(args.num_prompts)
    env["E2E_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    env["E2E_MAX_MODEL_LEN"] = str(args.max_model_len)
    env["E2E_MAX_NUM_BATCHED_TOKENS"] = str(args.max_batched_tokens)
    env["E2E_GPU_MEM_UTIL"] = str(args.gpu_mem_util)
    env["E2E_TOLERANCE"] = str(args.tolerance)
    env["DMX_DB_HOST"] = args.db_host
    env["DMX_DB_PORT"] = str(args.db_port)
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    return env


# ---------------------------------------------------------------------------
# Dispatch planning (no side effects -- the dry-run surface)
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One planned subprocess: a label + argv (env applied at run time)."""

    label: str
    argv: List[str]


def _runner(mod: str, *flags: str) -> List[str]:
    return [sys.executable, "-m", mod, *flags]


def plan_cell(cell: Cell, run_dir: str) -> tuple[List[Step], str, str]:
    """Return (steps, comparator_module, result_file) for a cell.

    Pure planning: builds the subprocess argv list without executing, so the
    same code path feeds both ``--dry-run`` and the real runner.
    """
    ref_dir = os.path.join(run_dir, "ref")
    mon_dir = os.path.join(run_dir, "mon")
    result_file = os.path.join(run_dir, "result.json")
    steps: List[Step] = []

    if cell.backend == "vllm":
        if cell.standard in _VLLM_IDENTICAL_STANDARDS:
            config_file = os.path.join(ref_dir, "ref_config.json")
            steps.append(Step("enable_ref_hooks", ["<in-process>", "enable_ref_hooks"]))
            steps.append(Step("vllm_ref", _runner("tests.vllm_ref_runner", "--output-dir", ref_dir)))
            steps.append(Step("vllm_monitored", _runner("tests.vllm_monitored_runner", "--output-dir", mon_dir)))
            steps.append(Step("compare", _runner(
                "tests.vllm_identical_comparator",
                "--ref-config", config_file, "--mon-dir", mon_dir,
                "--result-file", result_file)))
            return steps, "tests.vllm_identical_comparator", result_file
        # row_count / allclose -> monitored-only + rowcnt comparator
        steps.append(Step("vllm_monitored", _runner("tests.vllm_monitored_runner", "--output-dir", mon_dir)))
        steps.append(Step("compare", _runner(
            "tests.vllm_rowcnt_comparator",
            "--ref-dir", ref_dir, "--mon-dir", mon_dir,
            "--result-file", result_file)))
        return steps, "tests.vllm_rowcnt_comparator", result_file

    if cell.backend == "hf":
        steps.append(Step("hf_ref", _runner("tests.hf_reference_runner", "--output-dir", ref_dir)))
        steps.append(Step("hf_monitored", _runner("tests.hf_monitored_runner", "--output-dir", mon_dir)))
        steps.append(Step("compare", _runner(
            "tests.hf_comparator",
            "--ref-dir", ref_dir, "--mon-dir", mon_dir,
            "--result-file", result_file)))
        return steps, "tests.hf_comparator", result_file

    raise ValueError(f"unknown backend {cell.backend!r}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _enable_vllm_ref_hooks(cell: Cell, run_dir: str, env: dict) -> tuple[str, str, str]:
    """Run the in-process enable_ref_hooks preprocessor for a vLLM cell.

    Returns (model_file, backup_file, config_file).  Mirrors the flow in
    test_vllm_identical: back up the ref model source, generate the hooked
    ref + config, leaving restore to the caller's finally.
    """
    ref_dir = os.path.join(run_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)
    models_dir = os.path.join(
        PROJECT_ROOT, "integration", "vllm", "vllm",
        "model_executor", "models")
    ref_filename = _VLLM_REF_FILES.get(cell.model)
    if ref_filename is None:
        raise ValueError(f"no vLLM ref model registered for {cell.model!r}")
    model_file = os.path.join(models_dir, ref_filename)
    backup_file = os.path.join(run_dir, f"{ref_filename}.bak")
    config_file = os.path.join(ref_dir, "ref_config.json")
    max_len = int(os.environ.get("E2E_REF_MAX_LEN", "8192"))

    shutil.copy2(model_file, backup_file)
    sys.path.insert(0, models_dir)
    from enable_ref_hooks import enable_ref_hooks  # type: ignore
    enable_ref_hooks(
        model_file=model_file, hooks=cell.hooks, max_len=max_len,
        output_dir=ref_dir, config_out=config_file,
    )
    return model_file, backup_file, config_file


def _run_steps(steps: List[Step], env: dict, run_dir: str,
               *, restore=None, timeout: Optional[float] = None) -> Optional[str]:
    """Execute the runner/comparator steps in order.

    ``restore`` is a zero-arg callback run after the reference step (for the
    vLLM identical flow, which restores the ref model source before the
    monitored run).  ``timeout`` bounds each subprocess so a hung runner
    fails the cell instead of hanging the whole matrix.  Returns an error
    string on first failure, else None.
    """
    for step in steps:
        if step.argv and step.argv[0] == "<in-process>":
            continue  # enable_ref_hooks handled by the caller
        # vLLM identical: restore the patched ref source after the ref run,
        # before the monitored run starts.
        if restore is not None and step.label in ("vllm_monitored", "hf_monitored"):
            restore()
            restore = None
        env_step = dict(env)
        if step.label in ("vllm_ref",):
            env_step["REF_CONFIG"] = os.path.join(run_dir, "ref", "ref_config.json")
        try:
            proc = subprocess.run(
                step.argv, env=env_step, cwd=PROJECT_ROOT,
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"step {step.label} timed out after {timeout}s"
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:]
            return f"step {step.label} failed (rc={proc.returncode}): {tail}"
    return None


def run_cell(cell: Cell, args: argparse.Namespace) -> CellResult:
    """Run one cell end-to-end and return its :class:`CellResult`."""
    cr = CellResult(
        backend=cell.backend, model=cell.model, mode=cell.mode,
        standard=cell.standard, hook_selection=cell.hooks, tp=cell.tp,
        extra={"ring_mb": cell.ring_mb, "dtype": cell.dtype,
               "prompt_set": cell.prompt_set},
    )
    run_dir = tempfile.mkdtemp(prefix="e2e_matrix_")
    os.makedirs(os.path.join(run_dir, "ref"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "mon"), exist_ok=True)
    env = cell_env(cell, args)
    backup_file = model_file = None
    try:
        steps, _comparator, result_file = plan_cell(cell, run_dir)
        restore = None

        if cell.backend == "vllm" and cell.standard in _VLLM_IDENTICAL_STANDARDS:
            model_file, backup_file, _config = _enable_vllm_ref_hooks(cell, run_dir, env)

            def restore():  # noqa: E306  -- restore ref source pre-monitored run
                shutil.copy2(backup_file, model_file)

        elif cell.backend == "vllm":
            # rowcnt comparator still expects a ref meta.json (skipped marker).
            with open(os.path.join(run_dir, "ref", "meta.json"), "w") as f:
                json.dump({"skipped": True}, f)

        err = _run_steps(steps, env, run_dir, restore=restore,
                         timeout=args.cell_timeout)
        if err is not None:
            cr.error = err
            return cr.finalize()

        with open(result_file) as f:
            legacy = json.load(f)
        cr.checks = checks_from_legacy_result(legacy)
        return cr.finalize()

    except Exception as exc:  # noqa: BLE001 -- one bad cell must not abort the matrix
        cr.error = f"{type(exc).__name__}: {exc}"
        return cr.finalize()
    finally:
        if backup_file and model_file and os.path.exists(backup_file):
            shutil.copy2(backup_file, model_file)
        if not args.keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tests.e2e_matrix",
        description="Configurable E2E matrix over backend/model/mode/standard/hooks/tp.")
    p.add_argument("--backend", default="vllm", help="comma list: hf,vllm")
    p.add_argument("--model", default="gpt2", help="comma list: gpt2,qwen3,llama,qwen2_moe,<path>")
    p.add_argument("--mode", default="eager", help="comma list: eager,cuda_graph")
    p.add_argument("--standard", default="transport_bitwise",
                   help="comma list: bitwise,allclose,row_count,transport_bitwise")
    p.add_argument("--hooks", default="vllm-full",
                   help="comma list: preset (vllm-full,hidden-states) or single hook")
    p.add_argument("--tp", default="1", help="comma list of tensor-parallel sizes")
    p.add_argument("--ring-mb", dest="ring_mb", default="4096",
                   help="comma list of ring payload/pinned sizes in MB")
    p.add_argument("--dtype", default="bfloat16", help="comma list: bfloat16,float16,float32")
    p.add_argument("--prompt-set", dest="prompt_set", default="smoke",
                   help="comma list: smoke,math,chat,random")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-batched-tokens", type=int, default=512)
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="abs tolerance forwarded to comparators (E2E_TOLERANCE)")
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=9000)
    p.add_argument("--out", default=None, help="JSONL output path (one record per cell)")
    p.add_argument("--cell-timeout", type=float, default=1800.0,
                   help="per-subprocess timeout in seconds (hung runner fails the cell)")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="keep per-cell temp run dirs")
    p.add_argument("--dry-run", action="store_true",
                   help="print planned cells + dispatch commands; no CUDA/ClickHouse")
    return p


def _dry_run(cells: List[Cell], args: argparse.Namespace) -> int:
    print(f"# {len(cells)} cell(s) planned\n")
    for i, cell in enumerate(cells):
        steps, comparator, _result = plan_cell(cell, run_dir="<run_dir>")
        env = cell_env(cell, args, base={})
        print(f"[{i}] backend={cell.backend} model={cell.model} mode={cell.mode} "
              f"standard={cell.standard} hooks={cell.hooks} tp={cell.tp} "
              f"ring_mb={cell.ring_mb} dtype={cell.dtype} prompt_set={cell.prompt_set}")
        print(f"     env: E2E_HOOK_SELECTION={env['E2E_HOOK_SELECTION']} -> "
              f"DMX_HOOK_SELECTION={env['DMX_HOOK_SELECTION']}  "
              f"E2E_ENFORCE_EAGER={env['E2E_ENFORCE_EAGER']}")
        for step in steps:
            print(f"     - {step.label}: {' '.join(step.argv)}")
        print(f"     comparator: {comparator}\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cells = build_cells(args)
    if not cells:
        print("no cells to run (check axis values)", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(cells, args)

    results: List[CellResult] = []
    for i, cell in enumerate(cells):
        print(f"\n=== cell {i + 1}/{len(cells)}: {cell.backend}/{cell.model}/{cell.mode}/"
              f"{cell.standard}/{cell.hooks}/tp{cell.tp} ===", flush=True)
        cr = run_cell(cell, args)
        verdict = "ERROR" if cr.error else ("PASS" if cr.passed else "FAIL")
        print(f"    -> {verdict}" + (f": {cr.error}" if cr.error else ""), flush=True)
        results.append(cr)

    print("\n" + human_table(results))
    if args.out:
        write_jsonl(results, args.out)
        print(f"\nwrote {len(results)} record(s) to {args.out}")

    # Exit non-zero if any cell failed or errored, so CI can gate on it.
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
