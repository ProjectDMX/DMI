"""Standalone script: run vLLM with FullHiddenStatesConnector (original model).

Saves reference hidden states to disk for comparison.
Only works for models supported by extract_hidden_states (qwen, llama, etc.).
GPT-2 is NOT supported — the test should skip value comparison for GPT-2.

Usage:
    python -m tests.vllm_reference_runner --output-dir /tmp/vllm_ref
"""
import argparse
import json
import os
import shutil
import tempfile

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
    import safetensors.torch

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "0") == "1"
    compare_layers_str = os.environ.get("E2E_COMPARE_LAYERS", "")

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    num_layers = cfg.num_hidden_layers if hasattr(cfg, "num_hidden_layers") else 12

    # Parse layer IDs (skip layer 0 for "all" — extract_hidden_states
    # crashes with residual=None at layer 0)
    if compare_layers_str.strip().lower() == "all":
        layer_ids = list(range(1, num_layers))
    else:
        layer_ids = [int(x) for x in compare_layers_str.split(",") if x.strip()]

    if not layer_ids:
        print("[vllm_reference_runner] No layers to compare, skipping", flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
            json.dump({"skipped": True}, f)
        return

    tmpdir = tempfile.mkdtemp(prefix="vllm_ref_hs_")

    llm = LLM(
        model=model_id,
        max_model_len=512,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": layer_ids,
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "FullHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": tmpdir,
            },
        },
    )

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)

    # Collect reference hidden states per request.
    # vLLM internally appends -{uuid[:8]} to request IDs, so the
    # filenames are like "0-a51106ba.safetensors" not "0.safetensors".
    import glob
    os.makedirs(args.output_dir, exist_ok=True)
    ref_data = {}
    for o in outputs:
        req_id = o.request_id
        prefill_matches = glob.glob(os.path.join(tmpdir, f"{req_id}-*.safetensors"))
        prefill_matches = [m for m in prefill_matches if "_decode" not in m]
        decode_matches = glob.glob(os.path.join(tmpdir, f"{req_id}-*_decode.safetensors"))
        prefill_file = prefill_matches[0] if prefill_matches else None
        decode_file = decode_matches[0] if decode_matches else None

        parts = []
        if prefill_file and os.path.exists(prefill_file):
            data = safetensors.torch.load_file(prefill_file)
            parts.append(data["hidden_states"].cpu())
        if decode_file and os.path.exists(decode_file):
            data = safetensors.torch.load_file(decode_file)
            parts.append(data["hidden_states"].cpu())

        if parts:
            # [total_tokens, num_extracted_layers, hidden_size]
            ref_data[req_id] = torch.cat(parts, dim=0)

    # Save as torch files
    torch.save(ref_data, os.path.join(args.output_dir, "ref_hidden_states.pt"))
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "layer_ids": layer_ids,
            "num_prompts": num_prompts,
            "skipped": False,
        }, f)

    del llm
    torch.cuda.empty_cache()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[vllm_reference_runner] Done, {len(ref_data)} requests", flush=True)


if __name__ == "__main__":
    main()
