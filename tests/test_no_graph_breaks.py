"""Compile-graph integrity gate (verification.md Sec.A, plan Phase 1.5 step 3).

Runs every compiled cell in the verification matrix with
``TORCH_LOGS=graph_breaks`` set, captures stderr, and asserts the only
graph-break events that fire are on the explicit allow-list.

Cell axes:
  - framework: HF (transformers), vLLM
  - model:     gpt2, Qwen/Qwen3-0.6B
  - mode:      eager, compiled

= 8 cells.  Eager cells should trivially have zero graph breaks (no
compile happening).  Compiled cells exercise the full ``HookPoint``
forward path inside ``torch.compile`` / CUDA graphs.

Each cell runs in a fresh Python subprocess so torch logging state and
CUDA-graph caches don't leak between cells, and so vLLM's worker
machinery doesn't drag its imports into pytest's process.

Skips:
  - cuda not available -> all cells skip.
  - vllm not importable / sqlite/libstdc++ env mismatch -> vLLM cells skip.
  - Qwen3 weights not cached locally -> Qwen3 cells skip with a hint.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.vllm, pytest.mark.hf]

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_GPT2_MODEL = "gpt2"
HF_QWEN3_MODEL = "Qwen/Qwen3-0.6B"

# Allow-list: fragments that, if present in a "Graph break" line, are
# tolerated.  Today there are no expected breaks in any cell -- every
# entry below is an escape hatch for known-benign breaks observed in
# the future.  Each entry must come with a comment explaining why.
ALLOWED_BREAK_FRAGMENTS: dict[tuple[str, str, str], list[str]] = {
    # ("framework", "model", "mode"): [fragment, ...]
    # e.g. ("hf", "qwen3", "compiled"): ["torch._dynamo.exc.Unsupported: data dependent"]
}

GRAPH_BREAK_RE = re.compile(r"Graph break", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Subprocess runners (one per framework).  Each prints "OK" to stdout on
# successful completion and emits any graph-break events to stderr (via
# TORCH_LOGS).  The pytest layer parses stderr.
# ---------------------------------------------------------------------------

_HF_RUNNER = dedent("""
    import os, sys
    os.environ.setdefault("TORCH_LOGS", "graph_breaks")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = sys.argv[1]
    mode = sys.argv[2]  # 'eager' or 'compiled'

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype).to('cuda').eval()

    inputs = tok(["Hello, world"], return_tensors='pt', padding=True).to('cuda')

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
    )

    if mode == 'compiled':
        from transformers import CompileConfig
        gen_kwargs['cache_implementation'] = 'static'
        gen_kwargs['compile_config'] = CompileConfig(mode='reduce-overhead', fullgraph=False)

    with torch.no_grad():
        _ = model.generate(**gen_kwargs)

    print("OK")
""")


_VLLM_RUNNER = dedent("""
    import os, sys
    os.environ.setdefault("TORCH_LOGS", "graph_breaks")

    model_name = sys.argv[1]
    mode = sys.argv[2]  # 'eager' or 'compiled'

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_name,
        enforce_eager=(mode == 'eager'),
        max_model_len=128,
        gpu_memory_utilization=0.5,
    )
    _ = llm.generate(["Hello, world"], SamplingParams(max_tokens=4, temperature=0.0))
    print("OK")
""")


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------


def _has_cuda() -> bool:
    """Probe via subprocess so pytest discovery doesn't import torch."""
    res = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, timeout=60,
    )
    return res.returncode == 0 and "True" in res.stdout


def _vllm_importable(env: dict) -> bool:
    res = subprocess.run(
        [sys.executable, "-c", "import vllm"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    return res.returncode == 0


def _hf_qwen3_available(env: dict) -> bool:
    """Check we can resolve Qwen3 weights without going to network."""
    code = (
        "from transformers import AutoConfig; "
        f"AutoConfig.from_pretrained({HF_QWEN3_MODEL!r}, local_files_only=True)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60, env=env,
    )
    return res.returncode == 0


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------

CELLS = [
    ("hf",   "gpt2",  "eager"),
    ("hf",   "gpt2",  "compiled"),
    ("hf",   "qwen3", "eager"),
    ("hf",   "qwen3", "compiled"),
    ("vllm", "gpt2",  "eager"),
    ("vllm", "gpt2",  "compiled"),
    ("vllm", "qwen3", "eager"),
    ("vllm", "qwen3", "compiled"),
]


def _build_env() -> dict:
    """Subprocess env:
      * prepend ``$CONDA_PREFIX/lib`` so vllm -> diskcache -> sqlite finds
        conda's libstdc++ (project_conda_libstdcpp memory).
      * pin ``CUDA_VISIBLE_DEVICES=0`` to dodge a torch._check_capability
        race we hit on multi-GPU boxes during cold init (the python-side
        ``torch.cuda.device_count()`` and the C++ ``num_gpus`` assertion
        can disagree mid-init when more than one GPU is enumerated).
        One GPU is enough for any cell here.
      * ``TORCH_LOGS=graph_breaks`` for the actual capture.
    """
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{ld}" if ld else f"{conda_prefix}/lib"
    env["TORCH_LOGS"] = "graph_breaks"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


@pytest.fixture(scope="module")
def env() -> dict:
    return _build_env()


@pytest.fixture(scope="module")
def cuda_available() -> bool:
    return _has_cuda()


@pytest.fixture(scope="module")
def vllm_available(env) -> bool:
    return _vllm_importable(env)


@pytest.fixture(scope="module")
def qwen3_available(env) -> bool:
    return _hf_qwen3_available(env)


@pytest.mark.parametrize(
    "framework,model_key,mode", CELLS,
    ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}" if isinstance(c, tuple) else str(c),
)
def test_no_graph_breaks(
    framework: str, model_key: str, mode: str,
    env: dict, cuda_available: bool,
    vllm_available: bool, qwen3_available: bool,
):
    if not cuda_available:
        pytest.skip("CUDA not available")
    if framework == "vllm" and not vllm_available:
        pytest.skip("vllm not importable in subprocess (likely sqlite/libstdc++ env)")
    if model_key == "qwen3" and not qwen3_available:
        pytest.skip(f"{HF_QWEN3_MODEL} weights not cached locally")

    if model_key == "gpt2":
        model_name = HF_GPT2_MODEL
    elif model_key == "qwen3":
        model_name = HF_QWEN3_MODEL
    else:
        pytest.fail(f"Unknown model_key: {model_key}")

    if framework == "hf":
        runner = _HF_RUNNER
    elif framework == "vllm":
        runner = _VLLM_RUNNER
    else:
        pytest.fail(f"Unknown framework: {framework}")

    proc = subprocess.run(
        [sys.executable, "-c", runner, model_name, mode],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=REPO_ROOT,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"runner crashed (returncode={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-4000:]}")

    # Parse stderr for "Graph break" events.  TORCH_LOGS=graph_breaks
    # routes Dynamo's break events through the standard logging system
    # (target stream: stderr).
    breaks = [
        line for line in proc.stderr.splitlines()
        if GRAPH_BREAK_RE.search(line)
    ]
    allowed = ALLOWED_BREAK_FRAGMENTS.get((framework, model_key, mode), [])
    unexpected = [
        b for b in breaks
        if not any(frag in b for frag in allowed)
    ]
    if unexpected:
        msg = [
            f"Unexpected graph break(s) in {framework}-{model_key}-{mode}:",
            *(f"  {b}" for b in unexpected[:30]),
        ]
        if len(unexpected) > 30:
            msg.append(f"  ... ({len(unexpected) - 30} more)")
        pytest.fail("\n".join(msg))
