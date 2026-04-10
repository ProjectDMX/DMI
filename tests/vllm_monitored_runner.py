"""Standalone script: run vLLM with DMXGPUWorker (hooked model + ring transport).

Activations go to ClickHouse. Saves metadata to disk for the comparator.

Usage:
    python -m tests.vllm_monitored_runner --output-dir /tmp/vllm_mon
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import torch


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
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
    ring_payload_mb = int(os.environ.get("E2E_RING_PAYLOAD_MB", "4096"))
    ring_pinned_mb = int(os.environ.get("E2E_RING_PINNED_MB", "4096"))
    hook_selection = os.environ.get("DMX_HOOK_SELECTION", "vllm-full")
    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))
    tp_size = int(os.environ.get("E2E_TP_SIZE", "1"))

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    # Drop existing table
    try:
        import clickhouse_driver
        client = clickhouse_driver.Client(db_host, port=db_port)
        client.execute("DROP TABLE IF EXISTS default.offload")
    except Exception:
        pass

    kwargs = dict(
        model=model_id,
        dtype=model_dtype,
        worker_cls="monitoring.vllm_integration.DMXGPUWorker",
        additional_config={
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
        },
        max_model_len=int(os.environ.get("E2E_MAX_MODEL_LEN", "512")),
        enforce_eager=enforce_eager,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.5")),
        tensor_parallel_size=tp_size,
    )
    cg_mode = os.environ.get("E2E_CUDAGRAPH_MODE")
    if cg_mode:
        kwargs["compilation_config"] = {"cudagraph_mode": cg_mode}
        print(f"[vllm_monitored_runner] cudagraph_mode={cg_mode}", flush=True)
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
            "db_host": db_host,
            "db_port": db_port,
        }, f)

    del llm
    torch.cuda.empty_cache()
    print(f"[vllm_monitored_runner] Done", flush=True)


if __name__ == "__main__":
    main()
