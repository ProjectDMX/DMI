"""Transport correctness test: single vLLM run with CompareWorker.

The compare model has BOTH HookPoints (ring::producer → ClickHouse) and
.copy_() capture (→ disk) in the same compiled graph. After generate(),
compares disk vs ClickHouse for bitwise equality.

Usage:
    E2E_MODEL=qwen3 E2E_TP_SIZE=2 E2E_ENFORCE_EAGER=1 \
    python -m tests.vllm_compare_runner
"""
import os
import shutil
import sys
import tempfile
import uuid

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def main():
    from vllm import LLM, SamplingParams

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    checkpoint_model_id = _MODEL_ALIASES.get(model_key, model_key)
    capture_model_id = os.environ.get(
        "E2E_DMX_MODEL_ID",
        f"vllm-compare::{model_key}::{uuid.uuid4().hex}",
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

    compare_dir = tempfile.mkdtemp(prefix="vllm_compare_ref_")
    os.environ["COMPARE_OUTPUT_DIR"] = compare_dir

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    mode = "eager" if enforce_eager else "compiled"
    print(f"[compare] model={model_key} tp={tp_size} mode={mode} "
          f"hooks={hook_selection} prompts={num_prompts} tokens={max_new_tokens}",
          flush=True)
    print(f"[compare] ref_dir={compare_dir}", flush=True)

    kwargs = dict(
        model=checkpoint_model_id,
        dtype=model_dtype,
        worker_cls="tests.compare_worker.CompareWorker",
        additional_config={
            "dmx_model_id": capture_model_id,
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
        },
        max_model_len=int(os.environ.get("E2E_MAX_MODEL_LEN", "512")),
        max_num_batched_tokens=int(os.environ.get("E2E_MAX_NUM_BATCHED_TOKENS", "512")),
        enforce_eager=enforce_eager,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.5")),
        tensor_parallel_size=tp_size,
    )

    llm = None
    try:
        llm = LLM(**kwargs)
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = llm.generate(prompts, params)

        for i, o in enumerate(outputs):
            print(f"  prompt[{i}]: {len(o.outputs[0].token_ids)} tokens generated")

        # Explicit per-worker flush+stop before teardown. Without this, the
        # implicit DMXGPUWorker.shutdown() races vLLM's 8s deadline and may
        # drop tail rows -- exactly the data we're about to compare.
        llm.collective_rpc("stop_monitoring")
        del llm
        llm = None
        torch.cuda.empty_cache()

        # --- Compare disk (.copy_() buffers) vs ClickHouse ring transport. ---
        print("\n[compare] Comparing disk vs ClickHouse...", flush=True)

        from tests.compare_disk_vs_ch import read_clickhouse, compare

        ch_data, num_rows = read_clickhouse(
            db_host, db_port, model_id=capture_model_id
        )
        _, failed = compare(compare_dir, ch_data, num_rows)

        if failed > 0:
            sys.exit(1)
        print("[compare] ALL PASSED", flush=True)
    finally:
        if llm is not None:
            try:
                llm.collective_rpc("stop_monitoring")
            except Exception:
                pass
        try:
            from tests._clickhouse_test_utils import delete_capture

            delete_capture(db_host, db_port, capture_model_id)
        finally:
            shutil.rmtree(compare_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
