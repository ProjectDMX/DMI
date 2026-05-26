"""Standalone script: run vLLM with RefDiskWorker (ref model + D2D capture).

Tensors saved to disk.  Exits cleanly so GPU is released.

Usage:
    REF_CONFIG=/tmp/ref/ref_config.json \
    python -m tests.vllm_ref_runner --output-dir /tmp/ref
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import torch


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args, _ = p.parse_known_args()

    from vllm import LLM, SamplingParams

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "1") == "1"
    model_dtype = os.environ.get("E2E_DTYPE", "auto")
    tp_size = int(os.environ.get("E2E_TP_SIZE", "1"))

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    kwargs = dict(
        model=model_id,
        dtype=model_dtype,
        worker_cls="tests.ref_disk_worker.RefDiskWorker",
        max_model_len=int(os.environ.get("E2E_MAX_MODEL_LEN", "512")),
        max_num_batched_tokens=int(
            os.environ.get("E2E_MAX_NUM_BATCHED_TOKENS", "512")),
        enforce_eager=enforce_eager,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.5")),
        tensor_parallel_size=tp_size,
    )
    cg_mode = os.environ.get("E2E_CUDAGRAPH_MODE")
    if cg_mode:
        kwargs["compilation_config"] = {"cudagraph_mode": cg_mode}
        print(f"[vllm_ref_runner] cudagraph_mode={cg_mode}", flush=True)
    llm = LLM(**kwargs)

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)

    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    generated = {}
    for i, o in enumerate(outputs):
        generated[i] = len(o.outputs[0].token_ids)
        print(f"  prompt[{i}]: {generated[i]} tokens generated")

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "num_prompts": num_prompts,
            "max_new_tokens": max_new_tokens,
            "generated_tokens": generated,
        }, f)

    del llm
    torch.cuda.empty_cache()
    print("[vllm_ref_runner] Done", flush=True)


if __name__ == "__main__":
    main()
