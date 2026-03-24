#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import math
import time
from typing import Any, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, StaticCache

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


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def _materialize_prefill_outputs(outputs: Any) -> None:
    if outputs.hidden_states is not None:
        for hidden in outputs.hidden_states:
            hidden.detach().cpu()


def _materialize_decode_outputs(outputs: Any, active_mask: torch.Tensor) -> None:
    if bool(active_mask.any().item()):
        if outputs.hidden_states is not None:
            for hidden in outputs.hidden_states:
                hidden[active_mask].detach().cpu()


def _forward_accepts_position_ids(model: Any) -> bool:
    return 'position_ids' in inspect.signature(model.forward).parameters


def _run_batch_manual(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_lengths: Sequence[int],
    pad_token_id: int,
    compile_enabled: bool,
) -> None:
    device = input_ids.device
    batch_size, prompt_width = input_ids.shape
    batch_max_new_tokens = max(int(length) for length in target_lengths)
    wants_position_ids = _forward_accepts_position_ids(model)

    target_lengths_t = torch.tensor(target_lengths, device=device, dtype=torch.long)
    next_input = input_ids
    full_attention_mask = attention_mask

    static_cache = None
    past_key_values = None
    compiled_decode = None
    cache_position = None

    if compile_enabled:
        max_cache_len = prompt_width + batch_max_new_tokens + 4
        static_cache = StaticCache(
            config=model.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=model.dtype,
        )
        cache_position = torch.arange(prompt_width, device=device, dtype=torch.long)

        def _decode_step(
            decode_input_ids: torch.Tensor,
            decode_attention_mask: torch.Tensor,
            decode_cache: StaticCache,
            decode_cache_position: torch.Tensor,
        ) -> Any:
            kwargs = {
                'input_ids': decode_input_ids,
                'attention_mask': decode_attention_mask,
                'past_key_values': decode_cache,
                'cache_position': decode_cache_position,
                'use_cache': True,
                'output_hidden_states': True,
                'output_attentions': False,
                'return_dict': True,
            }
            return model(**kwargs)

        compiled_decode = torch.compile(_decode_step, mode='reduce-overhead', fullgraph=False)

    prefill_kwargs = {
        'input_ids': next_input,
        'attention_mask': full_attention_mask,
        'use_cache': True,
        'output_hidden_states': True,
        'output_attentions': False,
        'return_dict': True,
    }
    if wants_position_ids:
        prefill_kwargs['position_ids'] = _position_ids_from_attention_mask(full_attention_mask)
    if compile_enabled:
        prefill_kwargs['past_key_values'] = static_cache
        prefill_kwargs['cache_position'] = cache_position

    outputs = model(**prefill_kwargs)
    _materialize_prefill_outputs(outputs)
    if not compile_enabled:
        past_key_values = outputs.past_key_values

    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del outputs

    for decode_index in range(1, batch_max_new_tokens):
        prev_active_mask = target_lengths_t > decode_index
        if not bool(prev_active_mask.any().item()):
            break

        next_tokens = torch.where(
            prev_active_mask.unsqueeze(1),
            next_tokens,
            torch.full_like(next_tokens, int(pad_token_id)),
        )
        full_attention_mask = torch.cat(
            [full_attention_mask, torch.ones((batch_size, 1), device=device, dtype=full_attention_mask.dtype)],
            dim=1,
        )

        if compile_enabled:
            cache_position = cache_position[-1:] + 1
            torch.compiler.cudagraph_mark_step_begin()
            outputs = compiled_decode(next_tokens, full_attention_mask, static_cache, cache_position)
        else:
            step_kwargs = {
                'input_ids': next_tokens,
                'attention_mask': full_attention_mask,
                'past_key_values': past_key_values,
                'use_cache': True,
                'output_hidden_states': True,
                'output_attentions': False,
                'return_dict': True,
            }
            if wants_position_ids:
                step_kwargs['position_ids'] = _position_ids_from_attention_mask(full_attention_mask)[:, -1:].contiguous()
            outputs = model(**step_kwargs)
            past_key_values = outputs.past_key_values

        _materialize_decode_outputs(outputs, prev_active_mask)
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description='HF monitor baseline with a manual greedy decode loop and per-step CPU materialization.'
    )
    add_shared_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')

    model_id = resolve_model_id(args.model)
    compile_enabled = not args.disable_compile
    device = torch.device('cuda')

    examples = load_jsonl_examples(args.sample_file, limit=parsed_limit(args))
    tokenizer = build_tokenizer(model_id, local_files_only=args.local_files_only)
    rendered = maybe_sort_by_length(
        build_rendered_prompts(tokenizer, examples),
        enabled=not args.no_sort_by_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='eager',
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
    )
    model.to(device).eval()

    batch_metrics = []
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
    bucket_decode_tokens = warmup_decode_tokens(rendered, int(args.max_new_tokens))
    with torch.no_grad():
        for bi in bucket_inputs:
            _run_batch_manual(
                model=model,
                input_ids=bi['input_ids'], attention_mask=bi['attention_mask'],
                target_lengths=[bucket_decode_tokens] * args.batch_size,
                pad_token_id=int(tokenizer.pad_token_id),
                compile_enabled=compile_enabled,
            )
            device_sync(device)
        for warmup_batch in warmup_batches(rendered, args.batch_size, count=2):
            warmup_texts = [item['prompt_text'] for item in warmup_batch]
            warmup_encoded = tokenize_batch(tokenizer, warmup_texts, pad_buckets=pad_buckets,
                pad_to_multiple_of=int(args.pad_to_multiple_of), max_input_tokens=int(args.max_input_tokens))
            _run_batch_manual(
                model=model,
                input_ids=warmup_encoded['input_ids'].to(device),
                attention_mask=warmup_encoded['attention_mask'].to(device),
                target_lengths=batch_target_lengths(warmup_batch, int(args.max_new_tokens)),
                pad_token_id=int(tokenizer.pad_token_id),
                compile_enabled=compile_enabled,
            )
            device_sync(device)
    print(f'Warmup done ({len(bucket_inputs)} buckets + 2 real batches).', flush=True)

    total_batches = math.ceil(len(rendered) / args.batch_size)
    with torch.no_grad():
        t0 = time.perf_counter()
        for batch_index, batch in tqdm(
            enumerate(iter_batches(rendered, args.batch_size)),
            total=total_batches,
            desc='hf_monitor',
        ):
            texts = [item['prompt_text'] for item in batch]
            encoded = tokenize_batch(
                tokenizer,
                texts,
                pad_buckets=pad_buckets,
                pad_to_multiple_of=int(args.pad_to_multiple_of),
                max_input_tokens=int(args.max_input_tokens),
            )
            target_lengths = batch_target_lengths(batch, int(args.max_new_tokens))
            batch_max_new_tokens = max(target_lengths)

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            input_tokens = int(attention_mask.sum().item())
            padded_tokens = int(input_ids.numel())

            device_sync(device)
            batch_t0 = time.perf_counter()
            _run_batch_manual(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_lengths=target_lengths,
                pad_token_id=int(tokenizer.pad_token_id),
                compile_enabled=compile_enabled,
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
        baseline='hf_monitor',
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
            'local_files_only': bool(args.local_files_only),
            'hf_manual_decode_loop': True,
            'hf_output_scores': True,
            'hf_output_hidden_states': True,
            'hf_output_attentions': False,
            'materialize_to_cpu': True,
            'pad_buckets': pad_buckets,
            'pad_to_multiple_of': int(args.pad_to_multiple_of),
            'max_input_tokens': int(args.max_input_tokens),
            'decode_length_mode': 'per_sample_target',
            'max_new_tokens_cap': int(args.max_new_tokens),
            'compile_mode': 'manual_decode_step' if compile_enabled else 'eager_manual_loop',
            'compile_scope': 'decode_only' if compile_enabled else 'disabled',
            'prefill_mode': 'eager',
        },
    )

    out_path = make_output_path(
        results_dir=args.results_dir,
        baseline='hf_monitor',
        model=args.model,
        sample_file=args.sample_file,
        batch_size=args.batch_size,
        repeat_index=args.repeat_index,
    )
    write_json(out_path, payload)
    print(f'Saved {out_path}')
    print(
        f"[hf_monitor] prompts/s={payload['prompts_per_s']:.3f} "
        f"target_tok/s={payload['target_generated_tokens_per_s']:.3f} "
        f"compute_tok/s={payload['actual_generated_tokens_per_s']:.3f}"
    )


if __name__ == '__main__':
    main()
