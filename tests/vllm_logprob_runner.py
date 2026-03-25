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
        hf_cfg.architectures = [_ARCH_REMAP.get(a, a) for a in archs]
        super().load_model()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--ref", action="store_true",
                   help="Use ref model (architecture remap + REF_CONFIG)")
    args, _ = p.parse_known_args()

    from vllm import LLM, SamplingParams

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "1") == "1"
    model_dtype = os.environ.get("E2E_DTYPE", "auto")

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    kwargs = dict(
        model=model_id,
        dtype=model_dtype,
        max_model_len=512,
        max_logprobs=-1,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.5,
    )
    if args.ref:
        kwargs["worker_cls"] = "tests.vllm_logprob_runner.RefLogprobWorker"

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

        # Determine vocab size from first step (logprobs=-1 returns all)
        vocab_size = max(max(step.keys()) for step in steps) + 1
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

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
