#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import torch
from tqdm import tqdm

from common import (
    BatchMetrics,
    add_shared_args,
    batch_target_lengths,
    build_rendered_prompts,
    build_tokenizer,
    device_sync,
    iter_batches,
    load_jsonl_examples,
    make_output_path,
    maybe_sort_by_length,
    parse_pad_buckets,
    parsed_limit,
    resolve_model_id,
    summarize_run,
    tokenize_batch,
    warmup_batches,
    write_json,
)


DEFAULT_INTERNAL_HOOK_SET = "q,k,v,z,mlp_in,mlp_out,resid_mid"


def _default_hook_selection(capture_mode: str) -> str:
    if capture_mode == "hs_logits":
        return "hidden-states,final_ln,logits"
    return "hidden-states,final_ln"


def _collect_targets(model: Any, hook_selection: str) -> tuple[list[tuple[str, Any, str]], list[str]]:
    targets: list[tuple[str, Any, str]] = []
    names: list[str] = []
    parts = [chunk.strip() for chunk in str(hook_selection).split(",") if chunk.strip()]
    wants_logits = "logits" in parts
    non_logit_parts = [part for part in parts if part != "logits"]
    non_logit_set = set(non_logit_parts)
    hidden_state_set = {"hidden-states", "final_ln"}
    internal_hook_set = {part.strip() for part in DEFAULT_INTERNAL_HOOK_SET.split(",")}

    if not non_logit_parts or non_logit_set == hidden_state_set or non_logit_set == {"hidden-states"}:
        for layer_idx, layer in enumerate(model.model.layers):
            spec = (f"layers.{layer_idx}.hidden_state", layer, "output0")
            targets.append(spec)
            names.append(spec[0])
        targets.append(("final_ln", model.model.norm, "output"))
        names.append("final_ln")
    elif non_logit_set == internal_hook_set:
        for layer_idx, layer in enumerate(model.model.layers):
            specs = [
                (f"layers.{layer_idx}.q", layer.self_attn.q_norm, "output"),
                (f"layers.{layer_idx}.k", layer.self_attn.k_norm, "output"),
                (f"layers.{layer_idx}.v", layer.self_attn.v_proj, "output"),
                (f"layers.{layer_idx}.z", layer.self_attn.o_proj, "input0"),
                (f"layers.{layer_idx}.mlp_in", layer.post_attention_layernorm, "output"),
                (f"layers.{layer_idx}.mlp_out", layer.mlp, "output"),
                (f"layers.{layer_idx}.resid_mid", layer.post_attention_layernorm, "input0"),
            ]
            targets.extend(specs)
            names.extend(name for name, _, _ in specs)
    else:
        raise ValueError(f"unsupported NNsight hook selection: {hook_selection}")

    if wants_logits:
        targets.append(("logits", model.lm_head, "output"))
        names.append("logits")
    return targets, names


def _save_target(module: Any, kind: str):
    if kind == "output":
        return module.output.to("cpu").save()
    if kind == "output0":
        return module.output[0].to("cpu").save()
    if kind == "input0":
        return module.input[0].to("cpu").save()
    raise ValueError(f"unsupported target kind: {kind}")


def _run_batch(
    *,
    model: Any,
    targets: list[tuple[str, Any, str]],
    encoded: dict[str, torch.Tensor],
    batch_max_new_tokens: int,
) -> None:
    with model.generate(encoded, max_new_tokens=batch_max_new_tokens, do_sample=False) as tracer:
        step_tensors = list().save()
        for _ in tracer.iter[:]:
            step_tensors.append(tuple(_save_target(module, kind) for _name, module, kind in targets))

    for step in step_tensors:
        for tensor in step:
            _ = tensor.shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NNsight offline baseline."
    )
    add_shared_args(parser)
    parser.add_argument("--hook-selection", default="auto")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    try:
        from nnsight import LanguageModel
    except Exception as exc:
        raise RuntimeError("NNsight is required for this baseline") from exc

    model_id = resolve_model_id(args.model)
    device = torch.device("cuda")
    compile_requested = not args.disable_compile
    compile_enabled = False
    capture_logits = args.capture_mode == "hs_logits"
    hook_selection = (
        _default_hook_selection(str(args.capture_mode))
        if str(args.hook_selection) == "auto"
        else str(args.hook_selection)
    )

    examples = load_jsonl_examples(args.sample_file, limit=parsed_limit(args))
    tokenizer = build_tokenizer(model_id, local_files_only=args.local_files_only)
    rendered = maybe_sort_by_length(
        build_rendered_prompts(tokenizer, examples),
        enabled=not args.no_sort_by_length,
    )

    model = LanguageModel(
        model_id,
        tokenizer=tokenizer,
        device_map="cuda:0",
        dispatch=True,
        torch_dtype="float16",
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    targets, target_names = _collect_targets(model, hook_selection)

    batch_metrics = []
    pad_buckets = parse_pad_buckets(args.pad_buckets)
    with torch.no_grad():
        for warmup_batch in warmup_batches(rendered, args.batch_size, count=2):
            warmup_texts = [item["prompt_text"] for item in warmup_batch]
            warmup_encoded = tokenize_batch(
                tokenizer,
                warmup_texts,
                pad_buckets=pad_buckets,
                pad_to_multiple_of=int(args.pad_to_multiple_of),
                max_input_tokens=int(args.max_input_tokens),
            )
            warmup_encoded = {key: value.to(device) for key, value in warmup_encoded.items()}
            _run_batch(
                model=model,
                targets=targets,
                encoded=warmup_encoded,
                batch_max_new_tokens=max(batch_target_lengths(warmup_batch, int(args.max_new_tokens))),
            )
            device_sync(device)
    print("Warmup done (2 real batches).", flush=True)

    total_batches = math.ceil(len(rendered) / args.batch_size)
    with torch.no_grad():
        t0 = time.perf_counter()
        for batch_index, batch in tqdm(
            enumerate(iter_batches(rendered, args.batch_size)),
            total=total_batches,
            desc="nnsight",
        ):
            texts = [item["prompt_text"] for item in batch]
            encoded = tokenize_batch(
                tokenizer,
                texts,
                pad_buckets=pad_buckets,
                pad_to_multiple_of=int(args.pad_to_multiple_of),
                max_input_tokens=int(args.max_input_tokens),
            )
            target_lengths = batch_target_lengths(batch, int(args.max_new_tokens))
            batch_max_new_tokens = max(target_lengths)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_tokens = int(encoded["attention_mask"].sum().item())
            padded_tokens = int(encoded["input_ids"].numel())

            device_sync(device)
            batch_t0 = time.perf_counter()
            _run_batch(
                model=model,
                targets=targets,
                encoded=encoded,
                batch_max_new_tokens=batch_max_new_tokens,
            )
            device_sync(device)
            batch_t1 = time.perf_counter()

            batch_metrics.append(
                BatchMetrics(
                    batch_index=batch_index,
                    batch_size=len(batch),
                    input_tokens=input_tokens,
                    padded_tokens=padded_tokens,
                    target_generated_tokens=sum(target_lengths),
                    actual_generated_tokens=len(batch) * batch_max_new_tokens,
                    seconds=batch_t1 - batch_t0,
                )
            )
        total_seconds = time.perf_counter() - t0

    payload = summarize_run(
        baseline="nnsight",
        model=args.model,
        model_id=model_id,
        sample_file=args.sample_file,
        repeat_index=args.repeat_index,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        sort_by_length=not args.no_sort_by_length,
        compile_enabled=compile_enabled,
        dataset_size=len(rendered),
        total_seconds=total_seconds,
        batch_metrics=batch_metrics,
        extra={
            "local_files_only": bool(args.local_files_only),
            "hook_set": hook_selection,
            "nnsight_collect_scope": "generate_tracer_custom_hook_selection",
            "nnsight_module_count": len(target_names),
            "nnsight_module_names": target_names,
            "capture_mode": str(args.capture_mode),
            "pad_buckets": pad_buckets,
            "pad_to_multiple_of": int(args.pad_to_multiple_of),
            "max_input_tokens": int(args.max_input_tokens),
            "decode_length_mode": "batch_max_target",
            "max_new_tokens_cap": int(args.max_new_tokens),
            "compile_requested": bool(compile_requested),
            "compile_scope": "disabled",
            "compile_disabled_reason": "NNsight tracing/generation uses Python tracing hooks; compile disabled for stability",
        },
    )

    out_path = make_output_path(
        results_dir=args.results_dir,
        baseline="nnsight",
        model=args.model,
        sample_file=args.sample_file,
        batch_size=args.batch_size,
        repeat_index=args.repeat_index,
    )
    write_json(out_path, payload)
    print(f"Saved {out_path}")
    print(
        f"[nnsight] prompts/s={payload['prompts_per_s']:.3f} "
        f"target_tok/s={payload['target_generated_tokens_per_s']:.3f} "
        f"compute_tok/s={payload['actual_generated_tokens_per_s']:.3f}"
    )


if __name__ == "__main__":
    main()
