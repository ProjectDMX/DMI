#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time
from typing import Any

from nnsight import LanguageModel
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


def _collect_modules(model: Any) -> tuple[list[Any], list[str]]:
    modules = [layer for layer in model.model.layers]
    names = [f"layer_{idx}_output" for idx in range(len(modules))]
    return modules, names


def _run_batch(
    *,
    model: Any,
    modules: list[Any],
    encoded: dict[str, torch.Tensor],
    batch_max_new_tokens: int,
) -> None:
    with model.generate(encoded, max_new_tokens=batch_max_new_tokens, do_sample=False) as tracer:
        step_hidden_states = list().save()
        for _ in tracer.iter[:]:
            step_hidden_states.append(tuple(module.output.to("cpu").save() for module in modules))

    for step in step_hidden_states:
        for tensor in step:
            _ = tensor.shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NNsight offline baseline collecting per-layer hidden states only."
    )
    add_shared_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_id = resolve_model_id(args.model)
    device = torch.device("cuda")
    compile_requested = not args.disable_compile
    compile_enabled = False

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
    modules, module_names = _collect_modules(model)

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
                modules=modules,
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
                modules=modules,
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
            "nnsight_collect_scope": "generate_tracer_layer_outputs_only",
            "nnsight_collect_embeddings": False,
            "nnsight_collect_attention": False,
            "nnsight_collect_logits": False,
            "nnsight_collect_final_norm": False,
            "nnsight_module_count": len(module_names),
            "nnsight_module_names": module_names,
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
