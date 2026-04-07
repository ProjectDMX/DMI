#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time

import math

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from common import (
    BatchMetrics,
    add_shared_args,
    batch_target_lengths,
    build_rendered_prompts,
    build_tokenizer,
    compile_generate_kwargs,
    device_sync,
    iter_batches,
    load_jsonl_examples,
    make_output_path,
    warmup_decode_tokens,
    maybe_sort_by_length,
    parse_pad_buckets,
    parsed_limit,
    resolve_model_id,
    summarize_run,
    tokenize_batch,
    make_bucket_warmup_inputs,
    warmup_batches,
    write_json,
)


def _dist_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def _dist_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process() -> bool:
    return _dist_rank() == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="HF upper-bound offline generate baseline.")
    add_shared_args(parser)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_id = resolve_model_id(args.model)
    compile_enabled = not args.disable_compile
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    examples = load_jsonl_examples(args.sample_file, limit=parsed_limit(args))
    tokenizer = build_tokenizer(model_id, local_files_only=args.local_files_only)
    rendered = maybe_sort_by_length(
        build_rendered_prompts(tokenizer, examples),
        enabled=not args.no_sort_by_length,
    )

    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        attn_implementation="eager",
        dtype=torch.float16,
        local_files_only=args.local_files_only,
    )
    if int(args.tp_size) > 1:
        load_kwargs["tp_plan"] = "auto"
        load_kwargs["tp_size"] = int(args.tp_size)
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if int(args.tp_size) <= 1:
        model.to(device)
    model.eval()

    batch_metrics = []
    gen_kwargs = compile_generate_kwargs(compile_enabled)
    pad_buckets = parse_pad_buckets(args.pad_buckets)

    bucket_inputs = (
        make_bucket_warmup_inputs(
            tokenizer,
            pad_buckets,
            args.batch_size,
            device,
            active_tokens=(int(args.max_input_tokens) if int(args.max_input_tokens) > 0 else max(pad_buckets)),
        )
        if compile_enabled and pad_buckets
        else []
    )
    with torch.no_grad():
        for bi in bucket_inputs:
            _ = model.generate(
                input_ids=bi["input_ids"], attention_mask=bi["attention_mask"],
                max_new_tokens=16, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, **gen_kwargs,
            )
            device_sync(device)
        for warmup_batch in warmup_batches(rendered, args.batch_size, count=2):
            warmup_texts = [item["prompt_text"] for item in warmup_batch]
            warmup_encoded = tokenize_batch(tokenizer, warmup_texts, pad_buckets=pad_buckets,
                pad_to_multiple_of=int(args.pad_to_multiple_of), max_input_tokens=int(args.max_input_tokens))
            _ = model.generate(
                input_ids=warmup_encoded["input_ids"].to(device),
                attention_mask=warmup_encoded["attention_mask"].to(device),
                max_new_tokens=warmup_decode_tokens(warmup_batch, int(args.max_new_tokens)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id, **gen_kwargs,
            )
            device_sync(device)
    if _is_main_process():
        print(f"Warmup done ({len(bucket_inputs)} buckets + 2 real batches).", flush=True)

    total_batches = math.ceil(len(rendered) / args.batch_size)
    with torch.no_grad():
        t0 = time.perf_counter()
        for batch_index, batch in tqdm(
            enumerate(iter_batches(rendered, args.batch_size)),
            total=total_batches,
            desc="hf_upper_bound",
            disable=not _is_main_process(),
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

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            input_tokens = int(attention_mask.sum().item())
            padded_tokens = int(input_ids.numel())

            device_sync(device)
            batch_t0 = time.perf_counter()
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=batch_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
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
        baseline="hf_upper_bound",
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
            "pad_buckets": pad_buckets,
            "pad_to_multiple_of": int(args.pad_to_multiple_of),
            "max_input_tokens": int(args.max_input_tokens),
            "decode_length_mode": "per_sample_target",
            "max_new_tokens_cap": int(args.max_new_tokens),
            "requested_tp_size": int(args.tp_size),
            "model_tp_size": int(getattr(model, "_tp_size", 1) or 1),
            "dist_rank": _dist_rank(),
            "dist_world_size": _dist_world_size(),
            "local_rank": local_rank,
        },
    )

    out_path = make_output_path(
        results_dir=args.results_dir,
        baseline="hf_upper_bound",
        model=args.model,
        sample_file=args.sample_file,
        batch_size=args.batch_size,
        repeat_index=args.repeat_index,
    )
    launched_distributed = _dist_world_size() > 1
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if _is_main_process():
        write_json(out_path, payload)
        print(f"Saved {out_path}")
        print(
            f"[hf_upper_bound] prompts/s={payload['prompts_per_s']:.3f} "
            f"target_tok/s={payload['target_generated_tokens_per_s']:.3f} "
            f"compute_tok/s={payload['actual_generated_tokens_per_s']:.3f}"
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if launched_distributed:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
