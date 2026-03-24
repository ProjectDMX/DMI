#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
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


DEFAULT_INTERNAL_HOOK_SET = "q,k,v,z,mlp_in,mlp_out,resid_mid"


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def _forward_accepts_position_ids(model: Any) -> bool:
    return "position_ids" in inspect.signature(model.forward).parameters


class TorchHookCollector:
    def __init__(self, model: Any, hook_selection: str):
        self.model = model
        self.hook_selection = hook_selection
        self.handles: list[Any] = []
        self.active = False
        self.hook_names: list[str] = []
        self._register()

    def _materialize(self, tensor: Any, *, reshape_v: bool = False) -> None:
        if not self.active or not isinstance(tensor, torch.Tensor):
            return
        if reshape_v and tensor.ndim == 3:
            num_kv_heads = int(self.model.config.num_key_value_heads)
            head_dim = int(self.model.model.layers[0].self_attn.head_dim)
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], num_kv_heads, head_dim)
        cpu_tensor = tensor.detach().cpu()
        _ = cpu_tensor.shape

    def _register_hidden_states(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.hook_names.append(f"layers.{layer_idx}.hidden_state")
            self.handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output: self._materialize(output[0] if isinstance(output, tuple) else output)
                )
            )

        self.hook_names.append("final_ln")
        self.handles.append(
            self.model.model.norm.register_forward_hook(
                lambda _module, _args, output: self._materialize(output)
            )
        )

    def _register_internal_hooks(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.hook_names.append(f"layers.{layer_idx}.q")
            self.handles.append(
                layer.self_attn.q_norm.register_forward_hook(
                    lambda _module, _args, output: self._materialize(output)
                )
            )

            self.hook_names.append(f"layers.{layer_idx}.k")
            self.handles.append(
                layer.self_attn.k_norm.register_forward_hook(
                    lambda _module, _args, output: self._materialize(output)
                )
            )

            self.hook_names.append(f"layers.{layer_idx}.v")
            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    lambda _module, _args, output: self._materialize(output, reshape_v=True)
                )
            )

            self.hook_names.append(f"layers.{layer_idx}.z")
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda _module, args: self._materialize(args[0])
                )
            )

            self.hook_names.append(f"layers.{layer_idx}.mlp_in")
            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda _module, _args, output: self._materialize(output)
                )
            )

            self.hook_names.append(f"layers.{layer_idx}.mlp_out")
            self.handles.append(
                layer.mlp.register_forward_hook(
                    lambda _module, _args, output: self._materialize(output)
                )
            )

            self.hook_names.append(f"layers.{layer_idx}.resid_mid")
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda _module, args: self._materialize(args[0])
                )
            )

    def _register_logits(self) -> None:
        self.hook_names.append("logits")
        self.handles.append(
            self.model.lm_head.register_forward_hook(
                lambda _module, _args, output: self._materialize(output)
            )
        )

    def _register(self) -> None:
        parts = [chunk.strip() for chunk in str(self.hook_selection).split(",") if chunk.strip()]
        wants_logits = "logits" in parts
        non_logit_parts = [part for part in parts if part != "logits"]
        non_logit_set = set(non_logit_parts)
        hidden_state_set = {"hidden-states", "final_ln"}
        internal_hook_set = {part.strip() for part in DEFAULT_INTERNAL_HOOK_SET.split(",")}

        if not non_logit_parts or non_logit_set == hidden_state_set or non_logit_set == {"hidden-states"}:
            self._register_hidden_states()
        elif non_logit_set == internal_hook_set:
            self._register_internal_hooks()
        else:
            raise ValueError(f"unsupported torch hook selection: {self.hook_selection}")

        if wants_logits:
            self._register_logits()

    def begin(self) -> None:
        self.active = True

    def end(self) -> None:
        self.active = False

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _default_hook_selection(capture_mode: str) -> str:
    if capture_mode == "hs_logits":
        return "hidden-states,final_ln,logits"
    return "hidden-states,final_ln"


def _run_batch_manual(
    *,
    model: Any,
    collector: TorchHookCollector,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_lengths: Sequence[int],
    pad_token_id: int,
) -> None:
    device = input_ids.device
    batch_size = int(input_ids.shape[0])
    batch_max_new_tokens = max(int(length) for length in target_lengths)
    wants_position_ids = _forward_accepts_position_ids(model)
    target_lengths_t = torch.tensor(target_lengths, device=device, dtype=torch.long)

    full_attention_mask = attention_mask
    prefill_kwargs = {
        "input_ids": input_ids,
        "attention_mask": full_attention_mask,
        "use_cache": True,
        "return_dict": True,
    }
    if wants_position_ids:
        prefill_kwargs["position_ids"] = _position_ids_from_attention_mask(full_attention_mask)

    collector.begin()
    outputs = model(**prefill_kwargs)
    collector.end()

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
            [
                full_attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=full_attention_mask.dtype),
            ],
            dim=1,
        )

        step_kwargs = {
            "input_ids": next_tokens,
            "attention_mask": full_attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        if wants_position_ids:
            step_kwargs["position_ids"] = _position_ids_from_attention_mask(full_attention_mask)[:, -1:].contiguous()

        collector.begin()
        outputs = model(**step_kwargs)
        collector.end()

        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch official forward/pre-hook offline baseline."
    )
    add_shared_args(parser)
    parser.add_argument("--hook-selection", default="auto")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_id = resolve_model_id(args.model)
    device = torch.device("cuda")
    compile_requested = not args.disable_compile
    compile_enabled = False
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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
    )
    model.to(device).eval()
    collector = TorchHookCollector(model, hook_selection=hook_selection)

    batch_metrics = []
    pad_buckets = parse_pad_buckets(args.pad_buckets)
    total_seconds = 0.0

    try:
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
                _run_batch_manual(
                    model=model,
                    collector=collector,
                    input_ids=warmup_encoded["input_ids"].to(device),
                    attention_mask=warmup_encoded["attention_mask"].to(device),
                    target_lengths=batch_target_lengths(warmup_batch, int(args.max_new_tokens)),
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                device_sync(device)
        print("Warmup done (2 real batches).", flush=True)

        total_batches = math.ceil(len(rendered) / args.batch_size)
        with torch.no_grad():
            t0 = time.perf_counter()
            for batch_index, batch in tqdm(
                enumerate(iter_batches(rendered, args.batch_size)),
                total=total_batches,
                desc="torch_hooks",
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
                _run_batch_manual(
                    model=model,
                    collector=collector,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_lengths=target_lengths,
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
                        actual_generated_tokens=len(batch) * batch_max_new_tokens,
                        seconds=batch_t1 - batch_t0,
                    )
                )
            total_seconds = time.perf_counter() - t0
    finally:
        collector.close()

    payload = summarize_run(
        baseline="torch_hooks",
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
            "hook_backend": "torch_official_forward_and_pre_hooks",
            "hook_count": len(collector.hook_names),
            "hook_names": collector.hook_names,
            "capture_mode": str(args.capture_mode),
            "pad_buckets": pad_buckets,
            "pad_to_multiple_of": int(args.pad_to_multiple_of),
            "max_input_tokens": int(args.max_input_tokens),
            "decode_length_mode": "per_sample_target",
            "max_new_tokens_cap": int(args.max_new_tokens),
            "compile_requested": bool(compile_requested),
            "compile_scope": "disabled",
            "compile_disabled_reason": "PyTorch forward/pre hooks use Python callbacks and are kept eager for stability",
        },
    )

    out_path = make_output_path(
        results_dir=args.results_dir,
        baseline="torch_hooks",
        model=args.model,
        sample_file=args.sample_file,
        batch_size=args.batch_size,
        repeat_index=args.repeat_index,
    )
    write_json(out_path, payload)
    print(f"Saved {out_path}")
    print(
        f"[torch_hooks] prompts/s={payload['prompts_per_s']:.3f} "
        f"target_tok/s={payload['target_generated_tokens_per_s']:.3f} "
        f"compute_tok/s={payload['actual_generated_tokens_per_s']:.3f}"
    )


if __name__ == "__main__":
    main()
