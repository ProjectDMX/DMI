"""Run a vLLM model and save full-vocab logprobs to disk as tensors.

Two modes:
  --ref : load ref model (architecture remap + REF_CONFIG buffers)
  default: load original model (stock vLLM)

Usage:
    python -m tests.vllm_logprob_runner --output /tmp/logprobs_orig.pt
    REF_CONFIG=/tmp/ref_config.json python -m tests.vllm_logprob_runner --output /tmp/logprobs_ref.pt --ref
"""
import argparse
import os

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import torch
from vllm.v1.worker.gpu_worker import Worker


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
}

_ARCH_REMAP = {
    "GPT2LMHeadModel": "GPT2RefLMHeadModel",
    "Qwen3ForCausalLM": "Qwen3RefForCausalLM",
}


class RefLogprobWorker(Worker):
    """Minimal worker that remaps architecture to ref variant."""

    def load_model(self) -> None:
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs
        super().load_model()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--ref", action="store_true",
                   help="Use ref model (architecture remap + REF_CONFIG)")
    p.add_argument("--monitored", action="store_true",
                   help="Use DMXGPUWorker (ring transport hooks)")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed info about stored logprob tensors")
    p.add_argument("--random-prompts", action="store_true",
                   help="Use random integers in prompt template")
    p.add_argument("--math", action="store_true",
                   help="Use math sequence prompts")
    p.add_argument("--chat", action="store_true",
                   help="Use realistic conversation prompts")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for --random-prompts")
    args, _ = p.parse_known_args()

    from vllm import LLM, SamplingParams

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "1") == "1"
    model_dtype = os.environ.get("E2E_DTYPE", "auto")
    hook_selection = os.environ.get("DMX_HOOK_SELECTION", "vllm-full")
    ring_payload_mb = int(os.environ.get("E2E_RING_PAYLOAD_MB", "4096"))
    ring_pinned_mb = int(os.environ.get("E2E_RING_PINNED_MB", "4096"))
    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))

    # Pick numbers
    if args.random_prompts:
        import random
        seed = args.seed if args.seed is not None else int(os.environ.get("E2E_SEED", "42"))
        rng = random.Random(seed)
        numbers = [rng.randint(1, 100000) for _ in range(num_prompts)]
    else:
        numbers = list(range(1, num_prompts + 1))

    # Number -> prompt
    if args.chat:
        prompts = [f"The {n}th most spoken language in the world is" for n in numbers]
    elif args.math:
        prompts = [f"Start from {n}, generate a sequence of numbers with gap 1:" for n in numbers]
    else:
        prompts = [f"The answer to question {n} is" for n in numbers]
    if os.environ.get("E2E_PRINT_PROMPTS", "0") == "1":
        for i, p in enumerate(prompts):
            print(f"[vllm_logprob_runner] prompt[{i}]: {p!r}", flush=True)

    tp_size = int(os.environ.get("E2E_TP_SIZE", "1"))

    kwargs = dict(
        model=model_id,
        dtype=model_dtype,
        max_model_len=int(os.environ.get("E2E_MAX_MODEL_LEN", "512")),
        max_logprobs=-1,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.5")),
        tensor_parallel_size=tp_size,
    )
    if args.ref:
        kwargs["worker_cls"] = "tests.vllm_logprob_runner.RefLogprobWorker"
    elif args.monitored:
        kwargs["worker_cls"] = "monitoring.vllm_integration.DMXGPUWorker"
        kwargs["additional_config"] = {
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
        }

    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=-1)
    outputs = llm.generate(prompts, params)

    # Build dense logprob tensors: {prompt_idx: {token_ids, logprobs_tensor}}
    result = {}
    for i, o in enumerate(outputs):
        token_ids = list(o.outputs[0].token_ids)
        steps = o.outputs[0].logprobs  # list[dict[int, Logprob]]
        if not steps:
            result[i] = {"token_ids": token_ids, "logprobs": None}
            continue

        # Determine vocab size from first step (logprobs=-1 returns all via gpu_input_batch)
        vocab_size = max(max(step.keys()) for step in steps) + 1
        entries_per_step = [len(step) for step in steps]
        print(f"  prompt[{i}]: {len(steps)} tokens, vocab={vocab_size}, "
              f"entries_per_step={min(entries_per_step)}-{max(entries_per_step)}",
              flush=True)
        if min(entries_per_step) < vocab_size:
            print(f"  WARNING: NOT full vocab! Expected {vocab_size}, got {min(entries_per_step)}",
                  flush=True)
        num_tokens = len(steps)
        logprob_tensor = torch.full((num_tokens, vocab_size), float("-inf"),
                                    dtype=torch.float32)
        for t, step in enumerate(steps):
            for tid, lp in step.items():
                logprob_tensor[t, tid] = lp.logprob

        result[i] = {"token_ids": token_ids, "logprobs": logprob_tensor}
        print(f"  prompt[{i}]: {num_tokens} tokens, vocab={vocab_size}")

    torch.save(result, args.output)
    print(f"[vllm_logprob_runner] Saved {len(result)} prompts to {args.output}",
          flush=True)

    if args.verbose:
        for i in sorted(result.keys()):
            r = result[i]
            tids = r["token_ids"]
            lp = r["logprobs"]
            if lp is not None:
                finite = torch.isfinite(lp).sum().item()
                print(f"  stored[{i}]: tokens={len(tids)} "
                      f"logprobs shape={list(lp.shape)} dtype={lp.dtype} "
                      f"finite={finite}/{lp.numel()} "
                      f"first_step_entries={len(lp[0][lp[0] > float('-inf')])}",
                      flush=True)
            else:
                print(f"  stored[{i}]: tokens={len(tids)} logprobs=None",
                      flush=True)

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
