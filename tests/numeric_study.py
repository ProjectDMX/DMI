"""Per-hook numeric-difference study (plan §9 / Phase 5).

Enables **one hook at a time** and reports the drift its monitoring path
introduces versus the **unhooked** baseline model.  The non-goal carried from
the issue holds: this does not *fix* numeric drift, it makes it *visible,
categorized, and reproducible*.

Algorithm (plan §9):

  1. Run the baseline **unhooked** model once; capture token ids + full logits.
  2. For each hook ``H`` in the selection, enable **only** ``H`` and run the
     monitored model:
       - ``--variant p`` (default): the production ``_p`` Hooked variant driven
         with ``hook_selection=H`` -- ``hook_selection`` already isolates a
         single hook, so no source patching is needed.
       - ``--variant compare``: the ``_compare`` variant under the hardened
         :func:`tests.isolate_hook.isolated_hook` context manager (plan §6),
         which patches the vendored source so only ``H``'s ``.copy_()`` line
         fires and asserts byte-identical restoration on exit.
  3. Compare against the baseline and record, per hook:
       - bitwise pass/fail (eager) or allclose-within-threshold (cuda graph),
         via the shared standards in :mod:`tests.lib.compare`,
       - max abs diff, mean abs diff, first differing token position,
       - top-k vocab diffs at the first differing position,
       - whether greedy (argmax) token ids diverged.
  4. Emit a machine-readable JSON artifact **and** a human-readable table.

Alert policy (plan §9), expressed through the §8 standards-by-mode choice:

  - **Eager** -> ``bitwise`` standard: *any* non-bitwise logits diff alerts.
  - **CUDA graph** -> ``allclose`` standard with a per-model/per-dtype
    threshold: a max abs diff over the threshold alerts.
  - **Always** alert on greedy token-id divergence (a hook flipped the
    argmax), on a runner error, on a shape mismatch (a possible hook identity
    swap), or on an empty capture.

The capture side reuses the proven subprocess-rollout pattern from
``tests/test_per_hook_isolation.py`` (a clean CUDA context / ring-transport
instance per cell), but saves the **full** ``[N, vocab]`` logits so the study
can report top-k vocab drift at the first divergence.

The comparison / report / alert core (``compute_drift``, ``format_table``, the
``*_to_dict`` helpers) is pure and CPU-testable; ``torch`` is imported lazily
inside the helpers that need it, so this module loads on a torch-less box.

CLI::

    python -m tests.numeric_study \\
        --framework hf --model qwen3 --mode eager \\
        --hooks q,k,resid_pre,final_logits \\
        --out results/numeric_qwen3_eager.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from tests.lib.compare import Check, allclose, bitwise

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-(model, dtype) max-abs-diff thresholds for the CUDA-graph allclose gate.
# Eager ignores the threshold (it gates strictly on bitwise equality); the
# CUDA-graph path tolerates inductor's per-class fusion noise up to this bound.
_DEFAULT_THRESHOLD = 0.15
_CUDA_GRAPH_THRESHOLDS: Dict[tuple, float] = {
    ("gpt2", "float16"): 0.15,
    ("qwen3", "float16"): 0.15,
    ("qwen2_moe", "float16"): 0.25,
}

# Default hooks when ``--hooks`` is not given: an attention projection, a
# residual-stream read, and the final-logits read.
_DEFAULT_HOOKS = ["q", "k", "resid_pre", "final_logits"]


def cuda_graph_threshold(model_key: str, dtype: str) -> float:
    """Resolve the CUDA-graph max-abs-diff alert threshold for a cell."""
    return _CUDA_GRAPH_THRESHOLDS.get((model_key, dtype), _DEFAULT_THRESHOLD)


def standard_for_mode(mode: str) -> str:
    """The §8 comparison standard for a mode: bitwise (eager) / allclose (cg)."""
    return "bitwise" if mode == "eager" else "allclose"


# ---------------------------------------------------------------------------
# Result dataclasses (JSON-serializable; no torch types stored)
# ---------------------------------------------------------------------------


@dataclass
class VocabDiff:
    """One vocab entry's logit drift at the first differing token position."""

    token_id: int
    baseline_logit: float
    monitored_logit: float
    abs_diff: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "baseline_logit": self.baseline_logit,
            "monitored_logit": self.monitored_logit,
            "abs_diff": self.abs_diff,
        }


@dataclass
class HookDrift:
    """Per-hook drift record vs the unhooked baseline.

    ``check`` is the shared-lib :class:`~tests.lib.compare.Check` for the
    logits comparison (carries passed / max_abs / mean_abs / first_diff_pos /
    detail).  The extra fields capture what the study adds on top.
    """

    hook: str
    check: Optional[Check] = None
    n_positions: int = 0
    vocab_size: int = 0
    first_diff_pos: int = -1  # token position; -1 == none
    token_ids_diverged: bool = False
    n_token_diff: int = 0
    topk_vocab_diffs: List[VocabDiff] = field(default_factory=list)
    shape_mismatch: bool = False
    error: Optional[str] = None
    alert: bool = False
    alert_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook": self.hook,
            "check": self.check.to_dict() if self.check is not None else None,
            "n_positions": self.n_positions,
            "vocab_size": self.vocab_size,
            "first_diff_pos": self.first_diff_pos,
            "token_ids_diverged": self.token_ids_diverged,
            "n_token_diff": self.n_token_diff,
            "topk_vocab_diffs": [v.to_dict() for v in self.topk_vocab_diffs],
            "shape_mismatch": self.shape_mismatch,
            "error": self.error,
            "alert": self.alert,
            "alert_reasons": self.alert_reasons,
        }


@dataclass
class StudyResult:
    framework: str
    model: str
    mode: str
    variant: str
    dtype: str
    standard: str
    threshold: float
    topk: int
    hooks: List[HookDrift] = field(default_factory=list)

    @property
    def any_alert(self) -> bool:
        return any(h.alert for h in self.hooks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "model": self.model,
            "mode": self.mode,
            "variant": self.variant,
            "dtype": self.dtype,
            "standard": self.standard,
            "threshold": self.threshold,
            "topk": self.topk,
            "any_alert": self.any_alert,
            "hooks": [h.to_dict() for h in self.hooks],
        }


# ---------------------------------------------------------------------------
# Pure comparison core (CPU-testable; torch imported lazily)
# ---------------------------------------------------------------------------


def _first_diff_row(abs_diff: "Any") -> int:
    """Index of the first token position holding any nonzero diff, or -1.

    ``abs_diff`` is a ``[N, vocab]`` tensor of absolute differences.
    """
    if abs_diff.numel() == 0:
        return -1
    row_has_diff = (abs_diff > 0).any(dim=-1)
    nz = row_has_diff.nonzero(as_tuple=False)
    return int(nz[0].item()) if nz.numel() > 0 else -1


def compute_drift(
    hook: str,
    baseline: Dict[str, "Any"],
    monitored: Dict[str, "Any"],
    *,
    mode: str,
    threshold: float,
    topk: int = 5,
) -> HookDrift:
    """Compare one monitored capture against the unhooked baseline.

    ``baseline`` / ``monitored`` are dicts with ``token_ids`` (int64 ``[N]``)
    and ``logits`` (float ``[N, vocab]``).  Uses the shared-lib standard for
    ``mode`` (``bitwise`` eager / ``allclose`` cuda graph) for the verdict and
    augments it with token-divergence + top-k vocab drift.  The §9 alert
    policy is applied here.  Never raises on bad input: records ``error`` /
    ``shape_mismatch`` so the caller can alert.
    """
    import torch

    drift = HookDrift(hook=hook)

    b_logits = baseline.get("logits")
    m_logits = monitored.get("logits")
    if b_logits is None or m_logits is None:
        drift.error = "missing logits in capture"
        return _finalize_alert(drift, mode=mode)

    b_logits = b_logits.float()
    m_logits = m_logits.float()
    drift.n_positions = int(b_logits.shape[0]) if b_logits.ndim >= 1 else 0
    drift.vocab_size = int(b_logits.shape[-1]) if b_logits.ndim >= 1 else 0

    if drift.n_positions == 0 or m_logits.shape[0] == 0:
        drift.error = "empty capture (no positions)"
        return _finalize_alert(drift, mode=mode)

    if tuple(b_logits.shape) != tuple(m_logits.shape):
        drift.shape_mismatch = True
        drift.error = (
            f"logits shape mismatch: baseline {tuple(b_logits.shape)} "
            f"vs monitored {tuple(m_logits.shape)}"
        )
        # Still run the shared standard so the Check records the mismatch.
        drift.check = _run_standard(b_logits, m_logits, mode=mode, threshold=threshold)
        return _finalize_alert(drift, mode=mode)

    # Shared-lib verdict (bitwise / allclose) -- carries max/mean/first stats.
    drift.check = _run_standard(b_logits, m_logits, mode=mode, threshold=threshold)

    abs_diff = (b_logits - m_logits).abs()
    drift.first_diff_pos = _first_diff_row(abs_diff)

    # Greedy (argmax) token divergence, recomputed from logits and
    # cross-checked against the stored ids.
    b_arg = b_logits.argmax(dim=-1)
    m_arg = m_logits.argmax(dim=-1)
    drift.n_token_diff = int((b_arg != m_arg).sum().item())
    drift.token_ids_diverged = drift.n_token_diff > 0
    b_ids = baseline.get("token_ids")
    m_ids = monitored.get("token_ids")
    if b_ids is not None and m_ids is not None and b_ids.shape == m_ids.shape:
        if not torch.equal(b_ids, m_ids):
            drift.token_ids_diverged = True
            drift.n_token_diff = max(drift.n_token_diff, int((b_ids != m_ids).sum().item()))

    # Top-k vocab diffs at the first differing position.
    pos = drift.first_diff_pos
    if pos >= 0 and topk > 0:
        row = abs_diff[pos]
        k = min(topk, int(row.numel()))
        top = torch.topk(row, k)
        for rank in range(k):
            tid = int(top.indices[rank].item())
            drift.topk_vocab_diffs.append(
                VocabDiff(
                    token_id=tid,
                    baseline_logit=float(b_logits[pos, tid].item()),
                    monitored_logit=float(m_logits[pos, tid].item()),
                    abs_diff=float(top.values[rank].item()),
                )
            )

    return _finalize_alert(drift, mode=mode)


def _run_standard(b_logits: "Any", m_logits: "Any", *, mode: str, threshold: float) -> Check:
    """Run the §8 standard for ``mode`` and return its :class:`Check`."""
    name = standard_for_mode(mode)
    if name == "bitwise":
        return bitwise(b_logits, m_logits, name="logits_bitwise")
    return allclose(b_logits, m_logits, name="logits_allclose", atol=threshold, rtol=0.0)


def _finalize_alert(drift: HookDrift, *, mode: str) -> HookDrift:
    """Apply the §9 alert policy to a populated :class:`HookDrift` in place."""
    reasons: List[str] = []
    if drift.error is not None:
        reasons.append(f"capture error: {drift.error}")
    if drift.shape_mismatch:
        reasons.append("logits shape mismatch (possible hook identity swap)")
    if drift.token_ids_diverged:
        reasons.append(f"greedy token ids diverged at {drift.n_token_diff} position(s)")
    if drift.check is not None and not drift.check.passed and not drift.shape_mismatch:
        if mode == "eager":
            reasons.append(f"non-bitwise drift in eager mode ({drift.check.detail})")
        else:
            reasons.append(f"exceeds cuda-graph threshold ({drift.check.detail})")
    drift.alert_reasons = reasons
    drift.alert = bool(reasons)
    return drift


# ---------------------------------------------------------------------------
# Human-readable table
# ---------------------------------------------------------------------------


def format_table(result: StudyResult) -> str:
    """Render a fixed-width human-readable table for the study result."""
    header = (
        f"numeric-difference study: {result.framework}/{result.model} "
        f"mode={result.mode} variant={result.variant} dtype={result.dtype} "
        f"standard={result.standard}"
        + (f" (threshold={result.threshold:g})" if result.standard == "allclose" else "")
    )
    cols = ("hook", "verdict", "max_abs", "mean_abs", "first_diff", "tok_div", "alert")
    widths = (16, 8, 12, 12, 10, 8, 6)
    sep = "  "

    def _row(values: tuple) -> str:
        return sep.join(str(v).ljust(w) for v, w in zip(values, widths))

    lines = [header, "-" * len(header), _row(cols)]
    for h in result.hooks:
        if h.error is not None and h.check is None:
            verdict, mx, mn, fd = "ERR", "-", "-", "-"
        else:
            chk = h.check
            verdict = "pass" if (chk and chk.passed) else "fail"
            mx = f"{chk.max_abs:.4g}" if chk and chk.max_abs is not None else "-"
            mn = f"{chk.mean_abs:.4g}" if chk and chk.mean_abs is not None else "-"
            fd = str(h.first_diff_pos) if h.first_diff_pos >= 0 else "-"
        lines.append(
            _row((
                h.hook, verdict, mx, mn, fd,
                str(h.n_token_diff) if h.token_ids_diverged else "0",
                "ALERT" if h.alert else "ok",
            ))
        )
        for vd in h.topk_vocab_diffs:
            lines.append(
                sep + f"    tok {vd.token_id}: base={vd.baseline_logit:.4g} "
                f"mon={vd.monitored_logit:.4g} |Δ|={vd.abs_diff:.4g}"
            )
        if h.alert:
            for reason in h.alert_reasons:
                lines.append(sep + f"    ! {reason}")
    n_alert = sum(1 for h in result.hooks if h.alert)
    lines.append("")
    lines.append(f"{len(result.hooks) - n_alert}/{len(result.hooks)} hooks clean"
                 + (f", {n_alert} alerting" if n_alert else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subprocess rollout runners (full-logits capture)
# ---------------------------------------------------------------------------

# Each rollout subprocess saves {token_ids: int64[N], logits: float32[N, vocab]}.
# Same on-disk shape for HF and vLLM so the comparison code is framework-agnostic.

_HF_RUNNER = dedent("""
    import argparse, os
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--hook', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--variant', required=True, choices=['p', 'compare'])
    ap.add_argument('--rollout', required=True, choices=['baseline', 'hooked'])
    ap.add_argument('--max-new-tokens', type=int, default=4)
    ap.add_argument('--prompt', default='Hello')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    MODEL_ALIASES = {
        'gpt2': 'gpt2',
        'qwen3': 'Qwen/Qwen3-0.6B',
        'qwen2_moe': 'Qwen/Qwen1.5-MoE-A2.7B',
        'llama': 'meta-llama/Llama-3.1-8B',
    }
    hf_id = MODEL_ALIASES[args.model_key]
    device = torch.device('cuda')
    dtype = torch.float16

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    if args.rollout == 'baseline':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, attn_implementation='eager'
        ).to(device).eval()
    elif args.variant == 'compare':
        # Patched _compare model (driver wraps this subprocess in isolated_hook).
        if args.model_key == 'qwen3':
            from transformers.models.qwen3_compare.modeling_qwen3 import CompareQwen3ForCausalLM as cls
        elif args.model_key == 'gpt2':
            from transformers.models.gpt2_compare.modeling_gpt2 import CompareGPT2LMHeadModel as cls
        elif args.model_key == 'qwen2_moe':
            from transformers.models.qwen2_moe_compare.modeling_qwen2_moe import CompareQwen2MoeForCausalLM as cls
        elif args.model_key == 'llama':
            from transformers.models.llama_compare.modeling_llama import CompareLlamaForCausalLM as cls
        else:
            raise ValueError(f'unsupported model_key={args.model_key!r}')
        model = cls.from_pretrained(
            hf_id, torch_dtype=dtype, attn_implementation='eager'
        ).to(device).eval()
        model.allocate_compare_buffers(1, 32, dtype=dtype, tp_size=1)
    else:  # variant p: production _p Hooked variant + hook_selection=H
        if args.model_key == 'qwen3':
            from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM as cls
        elif args.model_key == 'gpt2':
            from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel as cls
        elif args.model_key == 'qwen2_moe':
            from transformers.models.qwen2_moe_p.modeling_qwen2_moe import HookedQwen2MoeForCausalLM as cls
        else:
            raise ValueError(f'unsupported model_key={args.model_key!r}')
        model = cls.from_pretrained(
            hf_id, torch_dtype=dtype, attn_implementation='eager'
        ).to(device).eval()

    inputs = tok([args.prompt], return_tensors='pt', padding=True).to(device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if args.mode == 'cuda_graph':
        from transformers import CompileConfig
        gen_kwargs['cache_implementation'] = 'static'
        gen_kwargs['compile_config'] = CompileConfig(mode='reduce-overhead', fullgraph=False)

    if args.rollout == 'hooked' and args.variant == 'p':
        from monitoring import MonitoringEngine, MonitoringConfig
        from monitoring.config import CaptureSchedule
        from monitoring._native_engine import RingConfig
        from integration.hf_adapter import generate_with_monitoring
        cfg = MonitoringConfig(schedule=CaptureSchedule(capture_prefill=True, capture_decode=True))
        engine = MonitoringEngine(config=cfg, model_id='numeric_study')
        ring_cfg = RingConfig()
        ring_cfg.task_ring_entries = 1024
        ring_cfg.payload_ring_bytes = 64 * 1024 * 1024
        ring_cfg.pinned_staging_bytes = 64 * 1024 * 1024
        engine.enable_ring_transport(ring_cfg)
        model.monitoring_engine = engine
        try:
            out = generate_with_monitoring(model, hook_selection=args.hook, **gen_kwargs)
        finally:
            engine.close()
    else:
        with torch.no_grad():
            out = model.generate(**gen_kwargs)

    scores = torch.stack(out.scores, dim=0)            # [N, 1, vocab]
    logits = scores.squeeze(1).float().cpu()           # [N, vocab]
    token_ids = logits.argmax(dim=-1).to(torch.int64)  # [N]
    torch.save({'token_ids': token_ids, 'logits': logits}, args.out)
    print(f'OK {args.rollout}/{args.variant} N={logits.shape[0]} V={logits.shape[1]} -> {args.out}')
""")


_VLLM_RUNNER = dedent("""
    import argparse, os
    os.environ.setdefault('VLLM_DISABLE_COMPILE_CACHE', '1')
    import torch
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--hook', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--variant', required=True, choices=['p', 'compare'])
    ap.add_argument('--rollout', required=True, choices=['baseline', 'hooked'])
    ap.add_argument('--max-new-tokens', type=int, default=4)
    ap.add_argument('--prompt', default='Hello')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    MODEL_ALIASES = {
        'gpt2': 'gpt2',
        'qwen3': 'Qwen/Qwen3-0.6B',
        'qwen2_moe': 'Qwen/Qwen1.5-MoE-A2.7B',
    }
    model_name = MODEL_ALIASES[args.model_key]

    llm_kwargs = dict(
        model=model_name,
        max_model_len=128,
        gpu_memory_utilization=0.5,
        enforce_eager=(args.mode == 'eager'),
    )
    if args.rollout == 'hooked':
        additional_config = {'dmx_hook_selection': args.hook, 'dmx_db_host': ''}
        if args.variant == 'compare':
            llm_kwargs['worker_cls'] = 'tests.compare_worker.CompareWorker'
        else:
            llm_kwargs['worker_cls'] = 'integration.vllm_adapter.DMXGPUWorker'
        llm_kwargs['additional_config'] = additional_config

    llm = LLM(**llm_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, logprobs=20)
    outputs = llm.generate([args.prompt], params)

    completion = outputs[0].outputs[0]
    ids = list(completion.token_ids)
    step_logprobs = completion.logprobs or []
    # Dense [N, V] from the sparse top-k logprob dicts; unreported entries stay
    # at a large negative floor (sufficient for chosen-token + top-k drift).
    vocab = int(getattr(llm.llm_engine.model_config.hf_config, 'vocab_size', 0)) or 1
    N = len(ids)
    logits = torch.full((N, vocab), -1e30, dtype=torch.float32)
    for i in range(N):
        if i < len(step_logprobs) and step_logprobs[i] is not None:
            for tid, lp in step_logprobs[i].items():
                logits[i, int(tid)] = float(lp.logprob)
    token_ids = torch.tensor(ids, dtype=torch.int64)
    torch.save({'token_ids': token_ids, 'logits': logits}, args.out)
    print(f'OK {args.rollout}/{args.variant} N={N} V={vocab} -> {args.out}')

    try:
        llm.collective_rpc('stop_monitoring')
    except Exception:
        pass
""")


def _build_subprocess_env() -> dict:
    """Pin CUDA_VISIBLE_DEVICES=0 and put conda lib on LD_LIBRARY_PATH so vLLM
    imports resolve (mirrors test_per_hook_isolation / test_no_graph_breaks)."""
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{ld}" if ld else f"{conda_prefix}/lib"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _run_rollout(
    *,
    framework: str,
    model_key: str,
    hook: str,
    mode: str,
    variant: str,
    rollout: str,
    out_path: Path,
    max_new_tokens: int,
    prompt: str,
    env: dict,
    timeout: int = 600,
) -> None:
    """Spawn one rollout subprocess; raise with captured output on failure."""
    runner = _HF_RUNNER if framework == "hf" else _VLLM_RUNNER
    cmd = [
        sys.executable, "-c", runner,
        "--model-key", model_key,
        "--hook", hook,
        "--mode", mode,
        "--variant", variant,
        "--rollout", rollout,
        "--max-new-tokens", str(max_new_tokens),
        "--prompt", prompt,
        "--out", str(out_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rollout={rollout} variant={variant} hook={hook} failed "
            f"(rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_study(
    *,
    framework: str,
    model: str,
    mode: str,
    hooks: List[str],
    out_dir: Path,
    variant: str = "p",
    dtype: str = "float16",
    topk: int = 5,
    max_new_tokens: int = 4,
    prompt: str = "Hello",
    threshold: Optional[float] = None,
) -> StudyResult:
    """Run the full per-hook study and return a populated :class:`StudyResult`.

    Captures the unhooked baseline once, then one monitored rollout per hook,
    comparing each against the baseline and applying the §9 alert policy.  For
    ``variant='compare'`` each hooked rollout runs inside the hardened
    :func:`tests.isolate_hook.isolated_hook` context manager (plan §6).
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _build_subprocess_env()
    thr = threshold if threshold is not None else cuda_graph_threshold(model, dtype)

    result = StudyResult(
        framework=framework, model=model, mode=mode, variant=variant,
        dtype=dtype, standard=standard_for_mode(mode), threshold=thr, topk=topk,
    )

    # 1. Baseline (unhooked) captured once; hook arg is unused by the runner.
    baseline_path = out_dir / "baseline.pt"
    _run_rollout(
        framework=framework, model_key=model, hook=hooks[0] if hooks else "q",
        mode=mode, variant="p", rollout="baseline", out_path=baseline_path,
        max_new_tokens=max_new_tokens, prompt=prompt, env=env,
    )
    baseline = torch.load(baseline_path, map_location="cpu")

    # 2-3. One hook at a time; compare; apply alert policy.
    for hook in hooks:
        drift = HookDrift(hook=hook)
        try:
            hooked_path = out_dir / f"hooked_{hook}.pt"
            if variant == "compare":
                from tests.isolate_hook import isolated_hook

                with isolated_hook(framework, model, hook):
                    _run_rollout(
                        framework=framework, model_key=model, hook=hook, mode=mode,
                        variant="compare", rollout="hooked", out_path=hooked_path,
                        max_new_tokens=max_new_tokens, prompt=prompt, env=env,
                    )
            else:
                _run_rollout(
                    framework=framework, model_key=model, hook=hook, mode=mode,
                    variant="p", rollout="hooked", out_path=hooked_path,
                    max_new_tokens=max_new_tokens, prompt=prompt, env=env,
                )
            monitored = torch.load(hooked_path, map_location="cpu")
            drift = compute_drift(
                hook, baseline, monitored, mode=mode, threshold=thr, topk=topk,
            )
        except Exception as exc:  # subprocess crash / OOM / dirty restore -> alert
            drift.error = f"{type(exc).__name__}: {exc}"
            _finalize_alert(drift, mode=mode)
        result.hooks.append(drift)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-hook numeric-difference study vs the unhooked baseline."
    )
    ap.add_argument("--framework", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--mode", choices=["eager", "cuda_graph"], default="eager")
    ap.add_argument("--variant", choices=["p", "compare"], default="p",
                    help="p: production _p+hook_selection; compare: _compare under isolated_hook")
    ap.add_argument("--hooks", default=",".join(_DEFAULT_HOOKS),
                    help="Comma-separated hook short-names (e.g. q,k,resid_pre,final_logits).")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the CUDA-graph max-abs-diff alert threshold.")
    ap.add_argument("--work-dir", default=None,
                    help="Scratch dir for per-rollout .pt files (default: temp dir).")
    ap.add_argument("--out", default=None,
                    help="Write the JSON artifact here (default: stdout table only).")
    args = ap.parse_args(argv)

    hooks = [h.strip() for h in args.hooks.split(",") if h.strip()]
    if not hooks:
        ap.error("no hooks selected")

    import tempfile

    tmp_ctx = None
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="numeric_study_")
        work_dir = Path(tmp_ctx.name)

    try:
        result = run_study(
            framework=args.framework, model=args.model, mode=args.mode,
            hooks=hooks, out_dir=work_dir, variant=args.variant, dtype=args.dtype,
            topk=args.topk, max_new_tokens=args.max_new_tokens, prompt=args.prompt,
            threshold=args.threshold,
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    print(format_table(result))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote JSON artifact -> {out_path}")

    # Non-zero exit if any hook alerted, so CI / wrappers can gate on it.
    return 1 if result.any_alert else 0


if __name__ == "__main__":
    sys.exit(main())
