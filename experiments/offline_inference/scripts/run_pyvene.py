#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

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


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    return value


def _register_qwen3_with_pyvene() -> None:
    import pyvene as pv
    import transformers.models as hf_models
    from pyvene.models.intervenable_modelcard import (
        type_to_dimension_mapping,
        type_to_module_mapping,
    )
    from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
        qwen2_lm_type_to_dimension_mapping,
        qwen2_lm_type_to_module_mapping,
    )

    qwen3_cls = hf_models.qwen3.modeling_qwen3.Qwen3ForCausalLM
    type_to_module_mapping[qwen3_cls] = qwen2_lm_type_to_module_mapping
    type_to_dimension_mapping[qwen3_cls] = qwen2_lm_type_to_dimension_mapping


def _build_pyvene_model(model: Any, max_prompt_units: int):
    import pyvene as pv

    num_layers = int(model.config.num_hidden_layers)
    representations = [
        pv.RepresentationConfig(
            layer=layer_idx,
            component="block_output",
            intervention_type=pv.CollectIntervention,
            max_number_of_units=max_prompt_units,
        )
        for layer_idx in range(num_layers)
    ]
    config = pv.IntervenableConfig(representations=representations)
    return pv.IntervenableModel(config, model=model), num_layers


def _collect_positions(
    seq_len: int,
    num_interventions: int,
) -> dict[str, list[list[list[int]]]]:
    positions = [[list(range(seq_len))]]
    return {"base": positions * int(num_interventions)}


def _materialize_collected(collected: Sequence[torch.Tensor]) -> None:
    for item in collected:
        item.detach().cpu()


def _materialize_decode_collected(
    collected: Sequence[torch.Tensor],
    active_mask: torch.Tensor,
) -> None:
    if not bool(active_mask.any().item()):
        return
    for item in collected:
        item[active_mask].detach().cpu()


def _run_batch_manual(
    *,
    pyvene_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_lengths: Sequence[int],
    pad_token_id: int,
) -> None:
    device = input_ids.device
    batch_size = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])
    target_lengths_t = torch.tensor(target_lengths, device=device, dtype=torch.long)
    batch_max_new_tokens = int(max(target_lengths))
    num_interventions = len(pyvene_model.interventions)

    full_attention_mask = attention_mask
    pyvene_out = pyvene_model(
        base={"input_ids": input_ids, "attention_mask": full_attention_mask},
        unit_locations=_collect_positions(seq_len, num_interventions),
        use_cache=True,
        return_dict=True,
    )
    outputs = pyvene_out.intervened_outputs
    collected = pyvene_out.collected_activations
    _materialize_collected(collected)
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    del outputs

    for decode_index in range(1, batch_max_new_tokens):
        active_mask = target_lengths_t > decode_index
        if not bool(active_mask.any().item()):
            break

        next_tokens = torch.where(
            active_mask.unsqueeze(1),
            next_tokens,
            torch.full_like(next_tokens, int(pad_token_id)),
        )
        full_attention_mask = torch.cat(
            [
                full_attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=full_attention_mask.dtype),
            ],
            dim=1,
        )
        pyvene_out = pyvene_model(
            base={
                "input_ids": next_tokens,
                "attention_mask": full_attention_mask,
                "past_key_values": past_key_values,
            },
            unit_locations=_collect_positions(1, num_interventions),
            use_cache=True,
            return_dict=True,
        )
        outputs = pyvene_out.intervened_outputs
        collected = pyvene_out.collected_activations
        _materialize_collected(collected)
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        del outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pyvene offline baseline collecting prompt hidden states only."
    )
    add_shared_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import pyvene as pv  # noqa: F401

    model_id = resolve_model_id(args.model)
    compile_requested = not args.disable_compile
    compile_enabled = False
    device = torch.device("cuda")

    if "qwen3" in model_id.lower():
        _register_qwen3_with_pyvene()

    examples = load_jsonl_examples(args.sample_file, limit=parsed_limit(args))
    tokenizer = build_tokenizer(model_id, local_files_only=args.local_files_only)
    rendered = maybe_sort_by_length(
        build_rendered_prompts(tokenizer, examples),
        enabled=not args.no_sort_by_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
    )
    model.to(device).eval()

    max_prompt_units = int(args.max_input_tokens) if int(args.max_input_tokens) > 0 else max(
        1, max(int(item["prompt_len"]) for item in rendered)
    )
    pyvene_model, num_layers = _build_pyvene_model(model, max_prompt_units=max_prompt_units)

    batch_metrics = []
    pad_buckets = parse_pad_buckets(args.pad_buckets)

    warmup_batch_list = warmup_batches(rendered, args.batch_size, count=2)
    with torch.no_grad():
        for warmup_batch in warmup_batch_list:
            for warmup_item in warmup_batch:
                warmup_encoded = tokenize_batch(
                    tokenizer,
                    [warmup_item["prompt_text"]],
                    pad_buckets=pad_buckets,
                    pad_to_multiple_of=int(args.pad_to_multiple_of),
                    max_input_tokens=int(args.max_input_tokens),
                )
                warmup_target_lengths = batch_target_lengths([warmup_item], int(args.max_new_tokens))
                warmup_input_ids = warmup_encoded["input_ids"].to(device)
                warmup_attention_mask = warmup_encoded["attention_mask"].to(device)
                _run_batch_manual(
                    pyvene_model=pyvene_model,
                    input_ids=warmup_input_ids,
                    attention_mask=warmup_attention_mask,
                    target_lengths=warmup_target_lengths,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
            device_sync(device)
    print("Warmup done.", flush=True)

    total_batches = math.ceil(len(rendered) / args.batch_size)
    with torch.no_grad():
        t0 = time.perf_counter()
        for batch_index, batch in tqdm(
            enumerate(iter_batches(rendered, args.batch_size)),
            total=total_batches,
            desc="pyvene",
        ):
            target_lengths = batch_target_lengths(batch, int(args.max_new_tokens))
            input_tokens = 0
            padded_tokens = 0

            device_sync(device)
            batch_t0 = time.perf_counter()
            for item, target_length in zip(batch, target_lengths):
                encoded = tokenize_batch(
                    tokenizer,
                    [item["prompt_text"]],
                    pad_buckets=pad_buckets,
                    pad_to_multiple_of=int(args.pad_to_multiple_of),
                    max_input_tokens=int(args.max_input_tokens),
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                input_tokens += int(attention_mask.sum().item())
                padded_tokens += int(input_ids.numel())
                _run_batch_manual(
                    pyvene_model=pyvene_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_lengths=[target_length],
                    pad_token_id=int(tokenizer.pad_token_id),
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
                    actual_generated_tokens=sum(target_lengths),
                    seconds=batch_t1 - batch_t0,
                )
            )
        total_seconds = time.perf_counter() - t0

    payload = summarize_run(
        baseline="pyvene",
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
            "pyvene_num_layers": int(num_layers),
            "pyvene_component": "block_output",
            "pyvene_collect_scope": "prefill_and_decode_manual_loop",
            "pyvene_unit": "pos",
            "pyvene_batch_mode": "per-sample-serial-within-batch",
            "pyvene_collect_attention": False,
            "pyvene_collect_logits": False,
            "pad_buckets": pad_buckets,
            "pad_to_multiple_of": int(args.pad_to_multiple_of),
            "max_input_tokens": int(args.max_input_tokens),
            "decode_length_mode": "per_sample_target",
            "max_new_tokens_cap": int(args.max_new_tokens),
            "compile_requested": bool(compile_requested),
            "compile_scope": "disabled",
            "compile_disabled_reason": "pyvene IntervenableModel manual loop uses Python hooks; compile disabled for stability",
        },
    )

    out_path = make_output_path(
        results_dir=args.results_dir,
        baseline="pyvene",
        model=args.model,
        sample_file=args.sample_file,
        batch_size=args.batch_size,
        repeat_index=args.repeat_index,
    )
    write_json(out_path, payload)
    print(f"Saved {out_path}")
    print(
        f"[pyvene] prompts/s={payload['prompts_per_s']:.3f} "
        f"target_tok/s={payload['target_generated_tokens_per_s']:.3f} "
        f"compute_tok/s={payload['actual_generated_tokens_per_s']:.3f}"
    )


if __name__ == "__main__":
    main()
