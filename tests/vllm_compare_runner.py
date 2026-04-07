"""Transport correctness test: single vLLM run with CompareWorker.

The compare model has BOTH HookPoints (ring::producer → ClickHouse) and
.copy_() capture (→ disk) in the same compiled graph. After generate(),
compares disk vs ClickHouse for bitwise equality.

Usage:
    E2E_MODEL=qwen3 E2E_TP_SIZE=2 E2E_ENFORCE_EAGER=1 \
    python -m tests.vllm_compare_runner
"""
import os
import sys
import tempfile

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import torch


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
}


def main():
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

    compare_dir = tempfile.mkdtemp(prefix="vllm_compare_ref_")
    os.environ["COMPARE_OUTPUT_DIR"] = compare_dir

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    # Drop existing table
    import clickhouse_driver
    ch_client = clickhouse_driver.Client(db_host, port=db_port)
    try:
        ch_client.execute("DROP TABLE IF EXISTS default.offload")
    except Exception:
        pass

    mode = "eager" if enforce_eager else "compiled"
    print(f"[compare] model={model_key} tp={tp_size} mode={mode} "
          f"hooks={hook_selection} prompts={num_prompts} tokens={max_new_tokens}",
          flush=True)
    print(f"[compare] ref_dir={compare_dir}", flush=True)

    kwargs = dict(
        model=model_id,
        dtype=model_dtype,
        worker_cls="tests.compare_worker.CompareWorker",
        additional_config={
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
        },
        max_model_len=512,
        max_num_batched_tokens=512,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=tp_size,
    )

    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)

    for i, o in enumerate(outputs):
        print(f"  prompt[{i}]: {len(o.outputs[0].token_ids)} tokens generated")

    # Shutdown flushes ring → ClickHouse (best-effort via death pipe path)
    del llm
    torch.cuda.empty_cache()

    # --- Compare disk (.copy_() buffers) vs ClickHouse (ring transport) ---
    print("\n[compare] Comparing disk vs ClickHouse...", flush=True)

    from tests.vllm_identical_comparator import (
        bitwise_check, _bytes_identical,
    )
    import re
    from pathlib import Path

    _DTYPE_MAP = {
        "torch.bfloat16": torch.bfloat16, "torch.float": torch.float32,
        "torch.half": torch.float16, "torch.float16": torch.float16,
        "torch.int": torch.int32, "torch.long": torch.int64,
        "torch.uint8": torch.uint8, "torch.int8": torch.int8,
        "torch.short": torch.int16, "torch.double": torch.float64,
        "torch.bool": torch.bool,
    }

    _BUF_TO_CH_ACT = {
        "resid_pre": "blocks.hook_resid_pre",
        "ln1": "blocks.hook_ln1",
        "q": "blocks.attn.hook_q",
        "k": "blocks.attn.hook_k",
        "v": "blocks.attn.hook_v",
        "z": "blocks.attn.hook_z",
        "attn_out": "blocks.hook_attn_out",
        "resid_mid": "blocks.hook_resid_mid",
        "ln2": "blocks.hook_ln2",
        "mlp_in": "blocks.hook_mlp_in",
        "mlp_out": "blocks.hook_mlp_out",
        "mlp_post": "blocks.hook_mlp_post",
        "embed": "hook_embed",
        "pos_embed": "hook_pos_embed",
        "resid_final": "hook_resid_final",
        "final_ln": "hook_final_ln",
        "final_logits": "final_logits",
        "token_ids": "token_ids",
    }

    _decode = lambda v: v.decode() if isinstance(v, bytes) else v

    # Read ClickHouse
    raw_rows = ch_client.execute(
        "SELECT model_id, request_id, act_name, layer_no, shard_rank, "
        "start_token_idx, end_token_idx, dtype, shape, bytes "
        "FROM default.offload",
        settings={"strings_as_bytes": True})

    ch_data: dict[tuple, torch.Tensor] = {}
    for row in raw_rows:
        _, req_id, act_name, layer_no, shard_rank, s, e, dtype_str, shape, payload = row
        dt = _DTYPE_MAP.get(_decode(dtype_str), torch.float32)
        t = torch.frombuffer(bytearray(payload), dtype=dt).reshape(list(shape))
        ch_data[(_decode(req_id), _decode(act_name), int(layer_no),
                 int(shard_rank), int(s), int(e))] = t

    print(f"[compare] {len(raw_rows)} rows in ClickHouse", flush=True)

    # Scan ref .pt files
    _PT_RE = re.compile(
        r"^(?P<hook>\w+?)(?:_L(?P<layer>\d+))?_T(?P<start>\d+)_(?P<end>\d+)"
        r"(?:_SR(?P<shard>\d+))?\.pt$"
    )

    ref_path = Path(compare_dir)
    passed = 0
    failed = 0
    not_found = 0

    for req_dir in sorted(ref_path.iterdir()):
        if not req_dir.is_dir():
            continue
        req_id = req_dir.name
        for pt_file in sorted(req_dir.iterdir()):
            m = _PT_RE.match(pt_file.name)
            if not m:
                continue
            hook = m.group("hook")
            layer = int(m.group("layer")) if m.group("layer") is not None else -1
            shard = int(m.group("shard")) if m.group("shard") is not None else 0
            start = int(m.group("start"))
            end = int(m.group("end"))

            ref_t = torch.load(str(pt_file), weights_only=True, map_location="cpu")

            ch_act = _BUF_TO_CH_ACT.get(hook, hook)
            ch_key = (req_id, ch_act, layer, shard, start, end)
            ch_t = ch_data.get(ch_key)

            if ch_t is None:
                # Try any token range for this (req, act, layer, shard)
                candidates = [
                    (k, v) for k, v in ch_data.items()
                    if k[0] == req_id and k[1] == ch_act and k[2] == layer
                    and k[3] == shard
                ]
                if len(candidates) == 1:
                    ch_t = candidates[0][1]

            label = f"{req_id}/{hook}"
            if layer >= 0:
                label += f"_L{layer}"
            label += f"_T{start}_{end}"

            if ch_t is None:
                print(f"  [FAIL] {label} -- not found in ClickHouse", flush=True)
                failed += 1
                not_found += 1
                continue

            ch_t = ch_t.cpu()
            if ref_t.shape != ch_t.shape:
                print(f"  [FAIL] {label} -- shape {list(ref_t.shape)} vs {list(ch_t.shape)}",
                      flush=True)
                failed += 1
                continue

            if ref_t.dtype != ch_t.dtype:
                print(f"  [FAIL] {label} -- dtype mismatch: ref={ref_t.dtype} ch={ch_t.dtype}",
                      flush=True)
                failed += 1
                continue

            if _bytes_identical(ref_t, ch_t):
                passed += 1
            else:
                diff = (ref_t.float() - ch_t.float()).abs().max().item()
                print(f"  [FAIL] {label} -- max_abs_diff={diff:.6e}", flush=True)
                failed += 1

    total = passed + failed
    print(f"\n[compare] Results: {passed}/{total} passed, {failed} failed", flush=True)
    if not_found > 0:
        print(f"[compare] ({not_found} not found in ClickHouse)", flush=True)

    if failed > 0:
        sys.exit(1)
    else:
        print("[compare] ALL PASSED", flush=True)


if __name__ == "__main__":
    main()
