"""Standalone script: run ORIGINAL (non-hooked) model with output_hidden_states.

Saves raw generate() outputs to disk so the test process can reconstruct
_HFRef objects without loading a model.

Accepts the same env vars as test_e2e_correctness_vs_hf.py for parity.

Usage:
    python tests/hf_reference_runner.py --output-dir ./tmp/hf_ref
"""
import argparse
import os
import sys
import torch
from typing import Any, Dict, List


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args, _ = p.parse_known_args()

    # Read config from env (same vars as the test)
    batch_size = int(os.environ.get("E2E_BATCH_SIZE", "4"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "8"))
    model_aliases = {"qwen3": "Qwen/Qwen3-4B"}
    model_key = os.environ.get("E2E_MODEL", "gpt2")
    hf_model_id = model_aliases.get(model_key, model_key)

    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load ORIGINAL model (NOT hooked)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    prompts = [("Hello " * (i + 1)).strip() for i in range(batch_size)]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        return_dict_in_generate=True,
        output_logits=True,
        output_hidden_states=True,
        output_attentions=True,
        logits_to_keep=0,
    )

    with torch.inference_mode():
        gen_out = model.generate(**gen_kwargs)

    # Save raw components to disk
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(gen_out.sequences.detach().cpu(), os.path.join(args.output_dir, "sequences.pt"))
    torch.save(attention_mask.detach().cpu(), os.path.join(args.output_dir, "attention_mask.pt"))

    # hidden_states: tuple of step_tuples, each step_tuple has (n_layers+1) tensors
    # Save as list of lists of CPU tensors
    if gen_out.hidden_states:
        hs_cpu = []
        for step_tuple in gen_out.hidden_states:
            hs_cpu.append([t.detach().cpu() for t in step_tuple])
        torch.save(hs_cpu, os.path.join(args.output_dir, "hidden_states.pt"))

    # attentions: same structure
    if gen_out.attentions:
        attn_cpu = []
        for step_tuple in gen_out.attentions:
            attn_cpu.append([t.detach().cpu() for t in step_tuple])
        torch.save(attn_cpu, os.path.join(args.output_dir, "attentions.pt"))

    # logits
    if gen_out.logits:
        logits_cpu = [t.detach().cpu() for t in gen_out.logits]
        torch.save(logits_cpu, os.path.join(args.output_dir, "logits.pt"))

    # model dtype
    model_dtype = next(model.parameters()).dtype
    torch.save({
        "model_id": hf_model_id,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "model_dtype": str(model_dtype),
        "Pmax": int(input_ids.shape[1]),
    }, os.path.join(args.output_dir, "meta.pt"))

    print(f"[hf_reference_runner] Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
