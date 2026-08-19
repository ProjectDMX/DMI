"""Per-hook isolation gate (verification.md Sec.C).

Phase 2b's smoke set ships in two layers:

1. **Unit tests** (no GPU, always run): exercise ``tests.isolate_hook``'s
   patcher -- correct line commenting, backup round-trip, error paths.
   These guard the patcher itself; if they fail the GPU smoke can't be
   trusted.

2. **GPU smoke** (marked ``@pytest.mark.slow``, opt-in): for each
   ``(framework, model, hook, mode)`` cell run a subprocess-driven
   3-way comparison:

     Original  -- vanilla AutoModelForCausalLM, no hooks.       L_orig
     Ours      -- _p variant + ``hook_selection=H``.             L_ours
     Ref       -- _compare variant patched to capture only H.    L_ref

   Logprob equality (``torch.equal(L_orig, L_ours)`` and
   ``torch.equal(L_orig, L_ref)``) catches per-hook perturbations of the
   forward path (the failure mode Sec.C exists to surface).  The tensor
   equality assertion (``T_ours_from_ch == T_ref_from_buffer``) is
   already covered for the all-hooks-on case by
   ``test_e2e_correctness_vs_hf.py`` and is run separately at Phase 4.

Smoke cells (per Phase 2 plan):
  - ``hf-qwen3-q-eager`` / ``hf-qwen3-q-compiled``
  - ``hf-qwen3-resid_pre-eager`` / ``hf-qwen3-resid_pre-compiled``
  - ``hf-qwen3-final_logits-eager`` / ``hf-qwen3-final_logits-compiled``

Eager cells assert all three logprob equalities strictly via
``torch.equal``.  Compiled cells use ``torch.allclose`` for both the
``L_orig`` vs ``L_ours`` and the ``L_orig`` vs ``L_ref`` checks: under
``torch.compile`` three distinct class hierarchies (``Qwen3ForCausalLM``,
``HookedQwen3ForCausalLM``, ``CompareQwen3ForCausalLM``) all produce
slightly different fp16 logprobs (~0.07-0.1 max diff) because inductor
makes different fusion / accumulation-order decisions per class.  The
existing E2E comparator (``tests/hf_comparator.py``) accepts the same
class-of-divergence with ``E2E_TOLERANCE=0.01`` (a raw activation
tolerance that translates to ~0.1 in logprob space after softmax).
The eager cells preserve the strict bitwise gate; the compiled cells
gate that the divergence stays bounded.

Every cell ALWAYS prints the actual max abs diff (regardless of
pass/fail) so a passing test still surfaces "barely passing" or
"unexpectedly drifting" trends.

Full sweep is the same parametrization expanded to every hook x model x
mode combination, gated behind ``-m slow``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from tests.isolate_hook import (
    _COPY_LINE_RE,
    _patched_source,
    compare_model_path,
    isolated_hook,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Unit tests for the patcher (no GPU)
# ---------------------------------------------------------------------------


SAMPLE_COMPARE_SOURCE = dedent("""
    class Foo:
        def forward(self, x):
            self._buf_q[:x.shape[0]].copy_(x)
            self._buf_k[:x.shape[0]].copy_(x)
            module._buf_attn_scores[:, :, :x.shape[2], :x.shape[3]].copy_(x)
            self._buf_resid_pre[:x.shape[0], :x.shape[1]].copy_(x)
            return x

        def allocate(self):
            # Allocation lines must NOT be commented (no .copy_() on RHS).
            self._buf_q = torch.empty(...)
            self._buf_resid_pre = torch.empty(...)
""").strip("\n")


@pytest.mark.cpu
class TestPatcherCorrectness:
    """Verify the regex + line-by-line patching logic."""

    def test_isolate_q_keeps_only_q_copy(self):
        patched, commented = _patched_source(SAMPLE_COMPARE_SOURCE, "q")
        assert sorted(set(commented)) == ["attn_scores", "k", "resid_pre"]
        # The q line is preserved verbatim
        assert "self._buf_q[:x.shape[0]].copy_(x)" in patched
        assert "# ISOLATE: self._buf_q" not in patched
        # The other capture lines are commented
        assert "# ISOLATE: self._buf_k" in patched
        assert "# ISOLATE: module._buf_attn_scores" in patched
        assert "# ISOLATE: self._buf_resid_pre" in patched

    def test_allocation_lines_are_not_commented(self):
        """`self._buf_q = torch.empty(...)` is allocation, not capture."""
        patched, _ = _patched_source(SAMPLE_COMPARE_SOURCE, "q")
        assert "self._buf_q = torch.empty(...)" in patched
        # No 'ISOLATE' marker on the allocation line for any buffer.
        for line in patched.splitlines():
            if "= torch.empty" in line:
                assert "ISOLATE" not in line

    def test_isolate_unknown_hook_comments_all(self):
        """If the hook name doesn't match any buffer, every .copy_() gets commented."""
        patched, commented = _patched_source(SAMPLE_COMPARE_SOURCE, "no_such_hook")
        assert sorted(set(commented)) == ["attn_scores", "k", "q", "resid_pre"]

    def test_indentation_preserved_on_commented_lines(self):
        patched, _ = _patched_source(SAMPLE_COMPARE_SOURCE, "q")
        # Original indent was 8 spaces (inside `def forward`)
        for line in patched.splitlines():
            if line.lstrip().startswith("# ISOLATE: "):
                indent = len(line) - len(line.lstrip())
                assert indent == 8, f"unexpected indent on: {line!r}"


@pytest.mark.cpu
class TestPatcherRoundTrip:
    """Verify the on-disk patch / unpatch context manager."""

    @pytest.mark.parametrize("framework,model_key", [
        ("hf", "gpt2"), ("hf", "qwen3"), ("hf", "llama"),
        ("vllm", "gpt2"), ("vllm", "qwen3"), ("vllm", "llama"),
    ])
    def test_round_trip_byte_identical(self, framework, model_key):
        """File contents before and after isolated_hook must match."""
        p = compare_model_path(framework, model_key)
        original = p.read_bytes()
        with isolated_hook(framework, model_key, "q") as (model_path, commented):
            assert p.read_bytes() != original, "patch did not modify the file"
            assert len(commented) > 0, "no _buf_* capture lines were commented"
            assert "q" not in commented, "q itself should not be in commented list"
        # Restored
        assert p.read_bytes() == original
        # Backup is gone
        backup = p.with_suffix(p.suffix + ".copy_isolate_backup")
        assert not backup.exists()

    def test_dirty_restore_raises_loudly(self, tmp_path, monkeypatch):
        """If the file can't be restored byte-identically, exit must raise.

        Simulates a body that clobbers the source *and* removes the backup
        (the failure mode §6 hardens against): the context manager's
        hash compare on exit must surface it as a loud RuntimeError rather
        than silently leaving the vendored submodule dirty.
        """
        from tests import isolate_hook
        target = tmp_path / "fake_compare.py"
        target.write_text(SAMPLE_COMPARE_SOURCE)
        fake_paths = {("test", "fake"): target}
        monkeypatch.setattr(isolate_hook, "_COMPARE_MODEL_PATHS", fake_paths)

        with pytest.raises(RuntimeError, match="byte-identically"):
            with isolate_hook.isolated_hook("test", "fake", "q"):
                # Corrupt the file and delete the backup so neither unpatch
                # nor the snapshot can restore the original bytes.
                target.write_text("corrupted contents that won't be restored")
                backup = target.with_suffix(target.suffix + ".copy_isolate_backup")
                backup.unlink()

    def test_stale_backup_raises(self, tmp_path, monkeypatch):
        """If a previous run crashed mid-patch leaving a backup, refuse."""
        from tests import isolate_hook
        # Create a fake compare-model entry pointing at a tmpdir file.
        target = tmp_path / "fake_compare.py"
        target.write_text(SAMPLE_COMPARE_SOURCE)
        # Pre-create the backup to simulate the crash.
        target.with_suffix(target.suffix + ".copy_isolate_backup").write_text("stale")

        fake_paths = {("test", "fake"): target}
        monkeypatch.setattr(isolate_hook, "_COMPARE_MODEL_PATHS", fake_paths)

        with pytest.raises(RuntimeError, match="Stale backup"):
            isolate_hook.patch("test", "fake", "q")


@pytest.mark.cpu
class TestRealCompareModelsContainAllExpectedHooks:
    """The smoke cells assume specific hooks have a `.copy_()` line in the
    real _compare files.  If a hook is missing the smoke fails opaquely,
    so verify upfront."""

    SMOKE_HOOKS = ["q", "resid_pre", "final_logits"]

    @pytest.mark.parametrize("framework,model_key", [
        ("hf", "qwen3"), ("hf", "gpt2"), ("hf", "llama"),
    ])
    def test_smoke_hooks_present_in_compare_source(self, framework, model_key):
        p = compare_model_path(framework, model_key)
        src = p.read_text()
        for hook in self.SMOKE_HOOKS:
            patched, commented = _patched_source(src, hook)
            isolated = []
            for line in patched.splitlines():
                m = _COPY_LINE_RE.match(line.rstrip())
                if m is not None:
                    isolated.append(m.group("buf"))
            assert hook in isolated, (
                f"{framework}/{model_key}: hook {hook!r} has no .copy_() "
                f"line in {p.name} (kept buffers: {isolated})"
            )


# ---------------------------------------------------------------------------
# GPU smoke cells (Phase 2b)
# ---------------------------------------------------------------------------

# Smoke set: 3 representative hooks x 2 modes per Phase 2 plan.  Full
# sweep (~80 HF cells) is the same parametrization with all hooks; it
# is behind ``@pytest.mark.slow`` to avoid being part of the default run.
SMOKE_CELLS = [
    ("hf",   "qwen3", "q",            "eager"),
    ("hf",   "qwen3", "q",            "compiled"),
    ("hf",   "qwen3", "resid_pre",    "eager"),
    ("hf",   "qwen3", "resid_pre",    "compiled"),
    ("hf",   "qwen3", "final_logits", "eager"),
    ("hf",   "qwen3", "final_logits", "compiled"),
    ("vllm", "qwen3", "q",            "eager"),
    ("vllm", "qwen3", "q",            "compiled"),
    ("vllm", "qwen3", "resid_pre",    "eager"),
    ("vllm", "qwen3", "resid_pre",    "compiled"),
    ("vllm", "qwen3", "final_logits", "eager"),
    ("vllm", "qwen3", "final_logits", "compiled"),
]

# Tolerance for compiled-mode logprob comparisons (both L_ours and L_ref
# diverge from L_orig by ~0.07-0.1 under torch.compile due to per-class
# inductor decisions; matches E2E_TOLERANCE=0.01 in tests/hf_comparator.py
# scaled into logprob space).
_COMPILED_ATOL = 0.15
_COMPILED_RTOL = 0.0


_HF_RUNNER = dedent("""
    '''HF rollout: vanilla / _p+hook_selection / _compare-isolated.

    Saves a dict {token_ids: int64[N], logprobs: float32[N]} -- the
    chosen-token IDs (greedy argmax) and the log-prob the model assigned
    to each chosen token.  Same format the vLLM runner emits, so the
    pytest assertion code is framework-agnostic.
    '''
    import argparse, os, sys
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument('--framework', required=True)
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--hook', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--rollout', required=True, choices=['orig', 'ours', 'ref'])
    args = ap.parse_args()
    assert args.framework == 'hf'

    MODEL_ALIASES = {
        'gpt2': 'gpt2',
        'qwen3': 'Qwen/Qwen3-0.6B',
        'qwen2_moe': 'Qwen/Qwen1.5-MoE-A2.7B',
        'llama': 'meta-llama/Llama-3.1-8B',
    }
    hf_id = MODEL_ALIASES[args.model_key]
    device = torch.device('cuda')
    dtype = torch.float16

    if args.rollout == 'orig':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_id)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, attn_implementation='eager'
        ).to(device).eval()
    elif args.rollout == 'ref':
        # Use the patched _compare model.  The driver patches the source
        # file before launching this subprocess and unpatches after.
        if args.model_key == 'qwen3':
            from transformers.models.qwen3_compare.modeling_qwen3 import CompareQwen3ForCausalLM as model_cls
        elif args.model_key == 'qwen2_moe':
            from transformers.models.qwen2_moe_compare.modeling_qwen2_moe import CompareQwen2MoeForCausalLM as model_cls
        elif args.model_key == 'gpt2':
            from transformers.models.gpt2_compare.modeling_gpt2 import CompareGPT2LMHeadModel as model_cls
        elif args.model_key == 'llama':
            from transformers.models.llama_compare.modeling_llama import CompareLlamaForCausalLM as model_cls
        else:
            raise ValueError(f'unsupported model_key={args.model_key!r} for HF ref rollout')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_id)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = model_cls.from_pretrained(
            hf_id, torch_dtype=dtype, attn_implementation='eager'
        ).to(device).eval()
        # Allocate buffers (only the isolated hook will get written).
        model.allocate_compare_buffers(1, 32, dtype=dtype, tp_size=1)
    elif args.rollout == 'ours':
        if args.model_key == 'qwen3':
            from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM as model_cls
        elif args.model_key == 'qwen2_moe':
            from transformers.models.qwen2_moe_p.modeling_qwen2_moe import HookedQwen2MoeForCausalLM as model_cls
        elif args.model_key == 'gpt2':
            from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel as model_cls
        else:
            raise ValueError(f'unsupported model_key={args.model_key!r} for HF ours rollout')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_id)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = model_cls.from_pretrained(
            hf_id, torch_dtype=dtype, attn_implementation='eager'
        ).to(device).eval()

    prompt = 'Hello'
    inputs = tok([prompt], return_tensors='pt', padding=True).to(device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    if args.mode == 'compiled':
        from transformers import CompileConfig
        gen_kwargs['cache_implementation'] = 'static'
        gen_kwargs['compile_config'] = CompileConfig(mode='reduce-overhead', fullgraph=False)

    if args.rollout == 'ours':
        # Drive monitoring through HFAdaptor with hook_selection=H.  The
        # ring transport runs without ClickHouse (no db_config).
        from monitoring import MonitoringEngine, MonitoringConfig
        from monitoring.config import CaptureSchedule
        from monitoring._native_engine import RingConfig
        from integration.hf_adapter import generate_with_monitoring
        cfg = MonitoringConfig(schedule=CaptureSchedule(capture_prefill=True, capture_decode=True))
        engine = MonitoringEngine(config=cfg, model_id='per_hook_isolation')
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

    # out.scores is a tuple of [batch=1, vocab] tensors, one per step.
    scores = torch.stack(out.scores, dim=0)            # [N, 1, vocab]
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    arg_step = scores.argmax(dim=-1)                   # [N, 1]  (greedy)
    token_ids = arg_step.squeeze(1).cpu().to(torch.int64)            # [N]
    chosen_lp = log_probs.gather(-1, arg_step.unsqueeze(-1)).squeeze(-1).squeeze(-1).cpu().float()  # [N]
    out_path = os.path.join(args.output_dir, f'{args.rollout}.pt')
    torch.save({'token_ids': token_ids, 'logprobs': chosen_lp}, out_path)
    print(f'OK {args.rollout} N={len(token_ids)} -> {out_path}')
""")


_VLLM_RUNNER = dedent("""
    '''vLLM rollout: vanilla / DMXGPUWorker+hook_selection / CompareWorker-isolated.

    Saves the same dict format as _HF_RUNNER:
      {token_ids: int64[N], logprobs: float32[N]}.

    Logprobs come from SamplingParams(logprobs=1).  The chosen token's
    logprob is what vLLM stores for the selected sample at each step.
    '''
    import argparse, os, sys
    os.environ.setdefault('VLLM_DISABLE_COMPILE_CACHE', '1')

    import torch
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument('--framework', required=True)
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--hook', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--rollout', required=True, choices=['orig', 'ours', 'ref'])
    args = ap.parse_args()
    assert args.framework == 'vllm'

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

    additional_config = {
        'dmx_hook_selection': args.hook,
        # Disable ClickHouse so concurrent rollouts in the smoke don't
        # see each other's rows; we only need logprobs here.
        'dmx_db_host': '',
    }

    if args.rollout == 'orig':
        # Vanilla vLLM, no worker_cls override -- standard model class.
        pass
    elif args.rollout == 'ours':
        llm_kwargs['worker_cls'] = 'integration.vllm_adapter.DMXGPUWorker'
        llm_kwargs['additional_config'] = additional_config
    elif args.rollout == 'ref':
        # CompareWorker subclasses DMXGPUWorker but remaps to the
        # _compare class (Qwen3CompareForCausalLM etc.).  The patcher
        # applied by the driver before this subprocess starts has
        # commented out every `.copy_()` line in the _compare source
        # except hook H's, so only that one buffer is written.
        llm_kwargs['worker_cls'] = 'tests.compare_worker.CompareWorker'
        llm_kwargs['additional_config'] = additional_config

    llm = LLM(**llm_kwargs)

    prompts = ['Hello']
    params = SamplingParams(temperature=0.0, max_tokens=4, logprobs=1)
    outputs = llm.generate(prompts, params)

    output = outputs[0]
    completion = output.outputs[0]
    ids = list(completion.token_ids)
    lps = []
    step_logprobs = completion.logprobs or []
    for i in range(len(ids)):
        chosen_id = ids[i]
        if i < len(step_logprobs) and step_logprobs[i] is not None:
            lp_obj = step_logprobs[i].get(chosen_id)
            lps.append(lp_obj.logprob if lp_obj is not None else float('-inf'))
        else:
            lps.append(float('-inf'))

    token_ids = torch.tensor(ids, dtype=torch.int64)
    logprobs = torch.tensor(lps, dtype=torch.float32)
    out_path = os.path.join(args.output_dir, f'{args.rollout}.pt')
    torch.save({'token_ids': token_ids, 'logprobs': logprobs}, out_path)
    print(f'OK {args.rollout} N={len(token_ids)} -> {out_path}')

    # Explicit per-worker flush+stop before process exit. Avoids the
    # implicit-shutdown race against vLLM's 8s deadline. No-op if the
    # worker doesn't carry stop_monitoring (e.g. baseline configs).
    try:
        llm.collective_rpc('stop_monitoring')
    except Exception:
        pass
""")


def _build_subprocess_env() -> dict:
    """Same env hardening as test_no_graph_breaks.py: pin CUDA_VISIBLE_DEVICES=0
    + put conda lib on LD_LIBRARY_PATH so vllm imports succeed."""
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{ld}" if ld else f"{conda_prefix}/lib"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _run_rollout(
    output_dir: Path, rollout: str, framework: str, model_key: str,
    hook: str, mode: str, env: dict,
) -> None:
    """Spawn one rollout subprocess.  Raises on non-zero exit."""
    if framework == "hf":
        runner = _HF_RUNNER
    elif framework == "vllm":
        runner = _VLLM_RUNNER
    else:
        raise ValueError(f"unsupported framework={framework!r}")
    cmd = [
        sys.executable, "-c", runner,
        "--framework", framework,
        "--model-key", model_key,
        "--hook", hook,
        "--mode", mode,
        "--output-dir", str(output_dir),
        "--rollout", rollout,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env, cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rollout={rollout} failed (returncode={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "framework,model_key,hook,mode", SMOKE_CELLS,
    ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}-{c[3]}" if isinstance(c, tuple) else str(c),
)
def test_per_hook_isolation_smoke(
    framework: str, model_key: str, hook: str, mode: str, tmp_path,
):
    """Smoke gate: enabling only hook H must not perturb logprobs vs
    the un-hooked rollout (both for our `_p` path and for the `_compare`
    path patched to capture H only).
    """
    import torch  # local import: this test is GPU-only

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    env = _build_subprocess_env()

    # Original: vanilla model, no hooks active anywhere.
    _run_rollout(tmp_path, "orig", framework, model_key, hook, mode, env)

    # Ours: _p variant + hook_selection=H.
    _run_rollout(tmp_path, "ours", framework, model_key, hook, mode, env)

    # Ref: _compare variant patched to isolate H.  Patch persists for
    # the lifetime of the with-block; ``isolated_hook`` invalidates the
    # cached bytecode so the subprocess imports the patched source rather
    # than a stale .pyc, and asserts byte-identical restoration on exit.
    with isolated_hook(framework, model_key, hook):
        _run_rollout(tmp_path, "ref", framework, model_key, hook, mode, env)

    L_orig = torch.load(tmp_path / "orig.pt", map_location="cpu")
    L_ours = torch.load(tmp_path / "ours.pt", map_location="cpu")
    L_ref = torch.load(tmp_path / "ref.pt", map_location="cpu")

    # Both runners save {"token_ids": int64[N], "logprobs": float32[N]}.
    for name, L in [("orig", L_orig), ("ours", L_ours), ("ref", L_ref)]:
        assert isinstance(L, dict) and "token_ids" in L and "logprobs" in L, (
            f"{name}.pt must be a dict with token_ids + logprobs; got {type(L)}"
        )

    assert L_orig["token_ids"].shape == L_ours["token_ids"].shape == L_ref["token_ids"].shape, (
        f"token_ids shape mismatch: orig={L_orig['token_ids'].shape} "
        f"ours={L_ours['token_ids'].shape} ref={L_ref['token_ids'].shape}"
    )

    def _diffs(a: dict, b: dict) -> tuple[int, float, float]:
        n_token_diff = int((a["token_ids"] != b["token_ids"]).sum().item())
        lp_diff = (a["logprobs"] - b["logprobs"]).abs()
        return n_token_diff, lp_diff.max().item(), lp_diff.mean().item()

    ours_tok_n, ours_lp_max, ours_lp_mean = _diffs(L_orig, L_ours)
    ref_tok_n, ref_lp_max, ref_lp_mean = _diffs(L_orig, L_ref)
    label = f"{framework}-{model_key}-{hook}-{mode}"
    print(
        f"\n[per_hook_isolation] {label}\n"
        f"  L_orig vs L_ours: token_diff={ours_tok_n}  "
        f"logprob_max={ours_lp_max:.6g}  logprob_mean={ours_lp_mean:.6g}\n"
        f"  L_orig vs L_ref:  token_diff={ref_tok_n}  "
        f"logprob_max={ref_lp_max:.6g}  logprob_mean={ref_lp_mean:.6g}",
        flush=True,
    )

    if mode == "eager":
        # Strict: tokens + logprobs both bitwise-equal in eager mode.
        assert torch.equal(L_orig["token_ids"], L_ours["token_ids"]), (
            f"token_ids differ for hook={hook!r} (eager, ours): "
            f"{ours_tok_n} positions differ"
        )
        assert torch.equal(L_orig["logprobs"], L_ours["logprobs"]), (
            f"logprobs differ for hook={hook!r} (eager, ours): max diff "
            f"{ours_lp_max:.4f}"
        )
        assert torch.equal(L_orig["token_ids"], L_ref["token_ids"]), (
            f"token_ids differ for hook={hook!r} (eager, ref): "
            f"{ref_tok_n} positions differ"
        )
        assert torch.equal(L_orig["logprobs"], L_ref["logprobs"]), (
            f"logprobs differ for hook={hook!r} (eager, ref): max diff "
            f"{ref_lp_max:.4f}"
        )
    else:
        # Compiled: chosen tokens still expected to match (greedy +
        # high-confidence top-1); chosen-token logprobs within tolerance
        # to absorb torch.compile's per-class fusion noise.
        assert torch.equal(L_orig["token_ids"], L_ours["token_ids"]), (
            f"token_ids differ for hook={hook!r} (compiled, ours): "
            f"{ours_tok_n} positions differ -- a hook installation "
            f"flipped the argmax under compile"
        )
        assert torch.allclose(
            L_orig["logprobs"], L_ours["logprobs"],
            atol=_COMPILED_ATOL, rtol=_COMPILED_RTOL,
        ), (
            f"logprobs for hook={hook!r} (compiled, ours): max diff "
            f"{ours_lp_max:.4f} exceeds atol={_COMPILED_ATOL}"
        )
        assert torch.equal(L_orig["token_ids"], L_ref["token_ids"]), (
            f"token_ids differ for hook={hook!r} (compiled, ref): "
            f"{ref_tok_n} positions differ"
        )
        assert torch.allclose(
            L_orig["logprobs"], L_ref["logprobs"],
            atol=_COMPILED_ATOL, rtol=_COMPILED_RTOL,
        ), (
            f"logprobs for hook={hook!r} (compiled, ref): max diff "
            f"{ref_lp_max:.4f} exceeds atol={_COMPILED_ATOL}"
        )
