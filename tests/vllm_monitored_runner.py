"""Standalone script: run vLLM with DMXGPUWorker (hooked model + ring transport).

Activations go to ClickHouse. Saves metadata to disk for the comparator.

Usage:
    python -m tests.vllm_monitored_runner --output-dir /tmp/vllm_mon
"""
import argparse
import json
import os
import uuid

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch


_MODEL_ALIASES = {
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "ernie45": "baidu/ERNIE-4.5-0.3B-PT",
    "falcon_h1": "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "gemma3": "shibatch/tinygemma3-2m",
    "gpt2": "gpt2",
    "gpt_oss": "openai/gpt-oss-20b",
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "granite": "ibm-granite/granite-4.1-3b",
    "jamba": "ai21labs/AI21-Jamba2-3B",
    "lfm2": "tiny-random/lfm2",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "openaccess-ai-collective/tiny-mistral",
    "minicpm4": "openbmb/MiniCPM4.1-8B",
    "olmo3": "allenai/Olmo-3-7B-Instruct",
    "phi3": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args, _ = p.parse_known_args()

    from vllm import LLM, SamplingParams

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    checkpoint_model_id = _MODEL_ALIASES.get(model_key, model_key)
    from tests.model_artifacts import resolve_model_artifact

    checkpoint_model_id = resolve_model_artifact(
        checkpoint_model_id,
        os.environ.get("E2E_MODEL_SUBFOLDER"),
    )
    capture_model_id = os.environ.get(
        "E2E_DMX_MODEL_ID",
        f"vllm-e2e::{model_key}::{uuid.uuid4().hex}",
    )
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "1") == "1"
    model_dtype = os.environ.get("E2E_DTYPE", "auto")
    ring_payload_mb = int(os.environ.get("E2E_RING_PAYLOAD_MB", "4096"))
    ring_pinned_mb = int(os.environ.get("E2E_RING_PINNED_MB", "4096"))
    hook_selection = os.environ.get("DMX_HOOK_SELECTION", "vllm-full")
    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))
    tp_size = int(os.environ.get("E2E_TP_SIZE", "1"))
    enable_ep = os.environ.get("E2E_ENABLE_EP", "0") == "1"
    all2all_backend = os.environ.get("E2E_ALL2ALL_BACKEND")

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    kwargs = dict(
        model=checkpoint_model_id,
        dtype=model_dtype,
        worker_cls="integration.vllm_adapter.DMXGPUWorker",
        additional_config={
            "dmx_model_id": capture_model_id,
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
        },
        max_model_len=int(os.environ.get("E2E_MAX_MODEL_LEN", "512")),
        max_num_batched_tokens=int(
            os.environ.get("E2E_MAX_NUM_BATCHED_TOKENS", "512")),
        enforce_eager=enforce_eager,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.5")),
        tensor_parallel_size=tp_size,
        trust_remote_code=os.environ.get("E2E_TRUST_REMOTE_CODE", "0") == "1",
        revision=os.environ.get("E2E_MODEL_REVISION"),
        tokenizer_revision=os.environ.get("E2E_MODEL_REVISION"),
    )
    if enable_ep:
        kwargs["enable_expert_parallel"] = True
    if all2all_backend:
        kwargs["all2all_backend"] = all2all_backend
    cg_mode = os.environ.get("E2E_CUDAGRAPH_MODE")
    if cg_mode:
        kwargs["compilation_config"] = {"cudagraph_mode": cg_mode}
        print(f"[vllm_monitored_runner] cudagraph_mode={cg_mode}", flush=True)
    llm = LLM(**kwargs)

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)

    # Explicit per-worker flush+stop before recording a successful run. The
    # implicit DMXGPUWorker.shutdown() path can race vLLM's shutdown deadline.
    # Do not swallow failures: incomplete ClickHouse data must fail the test.
    llm.collective_rpc("stop_monitoring")

    # Save metadata only after monitoring has flushed successfully.
    os.makedirs(args.output_dir, exist_ok=True)
    generated = {}
    for i, o in enumerate(outputs):
        generated[i] = len(o.outputs[0].token_ids)
        print(f"  prompt[{i}]: {generated[i]} tokens generated")

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({
            "model_id": capture_model_id,
            "checkpoint_model_id": checkpoint_model_id,
            "num_prompts": num_prompts,
            "max_new_tokens": max_new_tokens,
            "generated_tokens": generated,
            "db_host": db_host,
            "db_port": db_port,
        }, f)

    del llm
    torch.cuda.empty_cache()
    print(f"[vllm_monitored_runner] Done", flush=True)


if __name__ == "__main__":
    main()
