#!/usr/bin/env python3
"""Benchmark GPT-2 inference speed across different attention retrieval strategies.

This script compares three approaches:
  1. Baseline inference (no attention capture)
  2. PyTorch hooks that intercept each block's attention probabilities on-device
  3. Hugging Face forward pass with ``output_attentions=True`` (all layers)

A slice of WikiText is tokenized to a fixed sequence length so each method
processes the same number of tokens. Reported metrics include mean runtime,
standard deviation, per-run samples, and throughput in tokens/second. Optional
artifacts containing the captured attention tensors can be written to disk.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)
from torch.utils.hooks import RemovableHandle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPT-2 attention retrieval strategies")
    parser.add_argument(
        "--model_name",
        default="gpt2",
        help="Hugging Face model name to benchmark",
    )
    parser.add_argument(
        "--dataset_name",
        default="wikitext",
        help="Dataset repository name (passed to datasets.load_dataset)",
    )
    parser.add_argument(
        "--dataset_config",
        default="wikitext-2-raw-v1",
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--dataset_split",
        default="validation[:512]",
        help="Dataset split expression understood by datasets.load_dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional hard cap on the number of non-empty text samples to use",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples per batch")
    parser.add_argument(
        "--sequence_length",
        "--max_length",
        dest="sequence_length",
        type=int,
        default=128,
        help="Fixed token length per sample (truncate/pad inputs to this size)",
    )
    parser.add_argument(
        "--repeats",
        "--repeat",
        dest="repeats",
        type=int,
        default=3,
        help="How many timing runs to average for each method",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up runs per method before timing (set 0 to disable)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (defaults to cuda if available else cpu)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory to save captured attention tensors (.pt files)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run PyTorch profiler for each benchmark instead of timing",
    )
    parser.add_argument(
        "--collect_hidden_states",
        action="store_true",
        help="Capture hidden states from every transformer layer",
    )
    parser.add_argument(
        "--collect_attentions",
        action="store_true",
        help="Capture attention probabilities from every transformer layer",
    )
    return parser.parse_args()


def pick_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_samples(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    limit: int | None,
) -> List[str]:
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    texts: List[str] = []
    for record in dataset:
        text = record.get("text", "")
        if text and not text.isspace():
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def build_batches(
    tokenizer: AutoTokenizer,
    texts: Iterable[str],
    sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    texts = list(texts)
    batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
        )
        batches.append({key: tensor.to(device) for key, tensor in encoded.items()})
    return batches


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_execution(fn: Callable[[], None], repeats: int, device: torch.device) -> Dict[str, float]:
    durations: List[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        durations.append(time.perf_counter() - start)
    mean = statistics.mean(durations)
    stdev = statistics.stdev(durations) if len(durations) > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "runs": durations}


def profile_once(name: str, fn: Callable[[], None], device: torch.device) -> None:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        try:
            activities.append(ProfilerActivity.CUDA)
        except AttributeError:
            pass

    trace_root = Path("tb_traces") / name
    trace_root.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler(str(trace_root)),
    ) as prof:
        with record_function(name):
            fn()

    synchronize(device)
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    if args.sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    print("Loading dataset...")
    texts = load_text_samples(args.dataset_name, args.dataset_config, args.dataset_split, args.limit)
    if not texts:
        raise RuntimeError("No non-empty text samples were loaded; adjust your dataset parameters.")
    print(f"Prepared {len(texts)} text samples")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        # GPT-2 has no native pad token; reuse EOS to enable batching with padding.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(device)

    print("Tokenizing inputs...")
    batches = build_batches(tokenizer, texts, args.sequence_length, args.batch_size, device)
    total_sequences = sum(batch["input_ids"].shape[0] for batch in batches)
    total_tokens = sum(batch["input_ids"].numel() for batch in batches)
    print(
        "Built {} batches (batch size up to {})".format(len(batches), args.batch_size)
    )
    print(
        "Fixed sequence length: {} tokens | total sequences: {} | tokens per pass: {}".format(
            args.sequence_length, total_sequences, total_tokens
        )
    )

    timings: Dict[str, Dict[str, float]] = {}
    captured_results: Dict[str, Dict[str, Any]] = {}

    def to_cpu_recursive(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, list):
            return [to_cpu_recursive(x) for x in obj]
        if isinstance(obj, dict):
            return {key: to_cpu_recursive(val) for key, val in obj.items()}
        return obj

    def flush_label_to_cpu(label: str) -> None:
        if device.type != "cuda":
            return
        payload = captured_results.get(label)
        if payload is None:
            return
        captured_results[label] = to_cpu_recursive(payload)
        torch.cuda.empty_cache()

    def benchmark(label: str, fn: Callable[[], None]) -> None:
        for _ in range(args.warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.profile:
            profile_once(label, fn, device)
            stats = {"mean": float("nan"), "stdev": float("nan"), "runs": []}
            stats["tokens_per_second"] = float("nan")
        else:
            stats = time_execution(fn, args.repeats, device)
            if stats["mean"] > 0:
                stats["tokens_per_second"] = total_tokens / stats["mean"]
            else:
                stats["tokens_per_second"] = float("inf")
        timings[label] = stats
        flush_label_to_cpu(label)

    @torch.no_grad()
    def run_baseline() -> None:
        accumulator = torch.zeros((), device=device)
        for batch in batches:
            outputs = model(**batch, use_cache=False)
            accumulator = accumulator + outputs.logits[:, -1, :].mean()

        captured_results["baseline"] = {"summary": [[accumulator]]}

    benchmark("baseline", run_baseline)

    @torch.no_grad()
    def run_hook_variant(label: str, *, move_to_cpu: bool = False, dense: bool = False) -> None:
        transformer = getattr(model, "transformer", None)
        blocks: List[torch.nn.Module] | None = getattr(transformer, "h", None) if transformer is not None else None
        if not blocks:
            raise RuntimeError("Unexpected GPT-2 architecture; could not locate transformer blocks for hooking.")

        needs_attn = args.collect_attentions
        needs_hidden = args.collect_hidden_states

        def store_tensor(tensor: torch.Tensor) -> torch.Tensor:
            stored = tensor.detach()
            if move_to_cpu:
                stored = stored.cpu()
            return stored

        attentions_per_batch: List[List[torch.Tensor]] | None = [] if needs_attn else None
        hidden_states_per_batch: List[List[torch.Tensor]] | None = [] if needs_hidden else None

        dense_layer_cache: List[Dict[str, torch.Tensor | None]] | None = None
        if dense:
            dense_layer_cache = [
                {
                    "resid_pre": None,
                    "resid_post": None,
                    "ln1": None,
                    "ln2": None,
                    "mlp_in": None,
                    "mlp_out": None,
                    "attn_q": None,
                    "attn_k": None,
                    "attn_v": None,
                    "attn_output": None,
                    "attn_pattern": None,
                }
                for _ in blocks
            ]

        dense_attn_per_batch: List[List[Dict[str, torch.Tensor]]] | None = [] if dense and needs_attn else None
        dense_hidden_per_batch: List[List[Dict[str, torch.Tensor]]] | None = [] if dense and needs_hidden else None

        attn_layer_cache: List[torch.Tensor | None] | None = [None] * len(blocks) if needs_attn else None

        patched_attns: List[tuple[torch.nn.Module, Callable]] = []
        extra_hooks: List[RemovableHandle] = []

        for idx, block in enumerate(blocks):
            attn_module = block.attn
            original_forward = attn_module.forward

            def wrapped_forward(*f_args, _orig=original_forward, _idx=idx, **f_kwargs):
                want_attn = f_kwargs.get("output_attentions", False)
                use_cache = f_kwargs.get("use_cache", False)
                if needs_attn:
                    f_kwargs["output_attentions"] = True
                outputs = _orig(*f_args, **f_kwargs)

                if use_cache:
                    if len(outputs) == 3:
                        attn_output, present, attn_probs = outputs
                    elif len(outputs) == 2:
                        attn_output, present = outputs
                        attn_probs = None
                    else:
                        raise RuntimeError("Unexpected GPT-2 attention output shape when use_cache=True.")
                else:
                    present = None
                    attn_probs = None
                    if len(outputs) == 3:
                        attn_output, present, attn_probs = outputs
                    elif len(outputs) == 2:
                        attn_output, second = outputs
                        if want_attn:
                            attn_probs = second
                        else:
                            present = second
                    elif len(outputs) == 1:
                        attn_output = outputs[0]
                    else:
                        raise RuntimeError("Unexpected GPT-2 attention output shape when use_cache=False.")

                if dense_layer_cache is not None and (needs_attn or needs_hidden):
                    dense_layer_cache[_idx]["attn_output"] = store_tensor(attn_output)

                if needs_attn:
                    if attn_probs is None:
                        raise RuntimeError("Failed to capture attention probabilities via hook.")
                    attn_store = store_tensor(attn_probs)
                    if attn_layer_cache is None:
                        raise RuntimeError("Internal error: missing attention layer cache.")
                    attn_layer_cache[_idx] = attn_store
                    if dense_layer_cache is not None:
                        dense_layer_cache[_idx]["attn_pattern"] = attn_store

                if want_attn:
                    if attn_probs is None:
                        raise RuntimeError("Attention probabilities missing despite request.")
                    if use_cache:
                        return attn_output, present, attn_probs
                    return attn_output, attn_probs

                if use_cache:
                    return attn_output, present
                return attn_output, present

            attn_module.forward = wrapped_forward  # type: ignore[assignment]
            patched_attns.append((attn_module, original_forward))

            if dense_layer_cache is not None and needs_attn:
                def c_attn_hook(
                    module: torch.nn.Module,
                    module_input: tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                    _attn=attn_module,
                ) -> None:
                    q, k, v = module_output.split(_attn.split_size, dim=2)
                    num_heads = _attn.num_heads
                    head_dim = _attn.head_dim

                    def reshape(t: torch.Tensor) -> torch.Tensor:
                        bsz, seq_len, _ = t.size()
                        return store_tensor(
                            t.view(bsz, seq_len, num_heads, head_dim)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                        )

                    dense_layer_cache[_idx]["attn_q"] = reshape(q)
                    dense_layer_cache[_idx]["attn_k"] = reshape(k)
                    dense_layer_cache[_idx]["attn_v"] = reshape(v)

                hook = attn_module.c_attn.register_forward_hook(c_attn_hook)
                extra_hooks.append(hook)

            if dense_layer_cache is not None and needs_hidden:
                def block_pre_hook(module: torch.nn.Module, module_input: tuple[torch.Tensor, ...], _idx=idx) -> None:
                    dense_layer_cache[_idx]["resid_pre"] = store_tensor(module_input[0])

                def block_post_hook(
                    module: torch.nn.Module,
                    module_input: tuple[torch.Tensor, ...],
                    module_output: torch.Tensor | tuple[torch.Tensor, ...],
                    _idx=idx,
                ) -> None:
                    if isinstance(module_output, tuple):
                        hidden = module_output[0]
                    else:
                        hidden = module_output
                    dense_layer_cache[_idx]["resid_post"] = store_tensor(hidden)

                def ln1_hook(
                    module: torch.nn.Module,
                    module_input: tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                ) -> None:
                    dense_layer_cache[_idx]["ln1"] = store_tensor(module_output)

                def ln2_hook(
                    module: torch.nn.Module,
                    module_input: tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                ) -> None:
                    dense_layer_cache[_idx]["ln2"] = store_tensor(module_output)

                def mlp_pre_hook(
                    module: torch.nn.Module,
                    module_input: tuple[torch.Tensor, ...],
                    _idx=idx,
                ) -> None:
                    dense_layer_cache[_idx]["mlp_in"] = store_tensor(module_input[0])

                def mlp_post_hook(
                    module: torch.nn.Module,
                    module_input: tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                ) -> None:
                    dense_layer_cache[_idx]["mlp_out"] = store_tensor(module_output)

                extra_hooks.append(block.register_forward_pre_hook(block_pre_hook))
                extra_hooks.append(block.register_forward_hook(block_post_hook))
                extra_hooks.append(block.ln_1.register_forward_hook(ln1_hook))
                extra_hooks.append(block.ln_2.register_forward_hook(ln2_hook))
                extra_hooks.append(block.mlp.register_forward_pre_hook(mlp_pre_hook))
                extra_hooks.append(block.mlp.register_forward_hook(mlp_post_hook))

        try:
            for batch in batches:
                if attn_layer_cache is not None:
                    for i in range(len(attn_layer_cache)):
                        attn_layer_cache[i] = None
                if dense_layer_cache is not None:
                    for layer_dict in dense_layer_cache:
                        for key in list(layer_dict.keys()):
                            layer_dict[key] = None

                outputs = model(
                    **batch,
                    use_cache=False,
                    output_hidden_states=needs_hidden,
                )

                if needs_attn:
                    if attentions_per_batch is None or attn_layer_cache is None:
                        raise RuntimeError("Internal error: attention container not initialized.")
                    batch_attns: List[torch.Tensor] = []
                    for cached in attn_layer_cache:
                        if cached is None:
                            raise RuntimeError("Hook failed to capture all attention probabilities.")
                        batch_attns.append(cached)
                    attentions_per_batch.append(batch_attns)

                if needs_hidden:
                    if hidden_states_per_batch is None:
                        raise RuntimeError("Internal error: hidden state container not initialized.")
                    if outputs.hidden_states is None:
                        raise RuntimeError("Model did not return hidden states despite request.")
                    hidden_states_per_batch.append([store_tensor(hs) for hs in outputs.hidden_states])

                if dense_layer_cache is not None:
                    if dense and needs_attn and dense_attn_per_batch is not None:
                        attn_layer_details: List[Dict[str, torch.Tensor]] = []
                        for layer_values in dense_layer_cache:
                            detail: Dict[str, torch.Tensor] = {}
                            for key in ("attn_q", "attn_k", "attn_v", "attn_output", "attn_pattern"):
                                value = layer_values.get(key)
                                if value is not None:
                                    detail[key] = value
                            attn_layer_details.append(detail)
                        dense_attn_per_batch.append(attn_layer_details)

                    if dense and needs_hidden and dense_hidden_per_batch is not None:
                        hidden_layer_details: List[Dict[str, torch.Tensor]] = []
                        for layer_values in dense_layer_cache:
                            detail: Dict[str, torch.Tensor] = {}
                            for key in ("resid_pre", "ln1", "mlp_in", "mlp_out", "ln2", "resid_post"):
                                value = layer_values.get(key)
                                if value is not None:
                                    detail[key] = value
                            hidden_layer_details.append(detail)
                        dense_hidden_per_batch.append(hidden_layer_details)
        finally:
            for module, original in patched_attns:
                module.forward = original  # type: ignore[assignment]
            for handle in extra_hooks:
                handle.remove()

        result: Dict[str, Any] = {}
        if needs_attn and attentions_per_batch is not None:
            result["attentions"] = attentions_per_batch
        if needs_hidden and hidden_states_per_batch is not None:
            result["hidden_states"] = hidden_states_per_batch
        if dense and needs_attn and dense_attn_per_batch is not None:
            result["attentions_dense"] = dense_attn_per_batch
        if dense and needs_hidden and dense_hidden_per_batch is not None:
            result["hidden_states_dense"] = dense_hidden_per_batch

        captured_results[label] = result

    def run_hook_default() -> None:
        run_hook_variant("pytorch_hook")

    def run_hook_cpu() -> None:
        run_hook_variant("pytorch_hook_cpu", move_to_cpu=True)

    def run_hook_dense() -> None:
        run_hook_variant("pytorch_hook_dense", dense=True)

    benchmark("pytorch_hook", run_hook_default)
    benchmark("pytorch_hook_cpu", run_hook_cpu)
    benchmark("pytorch_hook_dense", run_hook_dense)

    @torch.no_grad()
    def run_with_output_features() -> None:
        attentions_per_batch: List[List[torch.Tensor]] | None = [] if args.collect_attentions else None
        hidden_states_per_batch: List[List[torch.Tensor]] | None = [] if args.collect_hidden_states else None
        for batch in batches:
            outputs = model(
                **batch,
                output_attentions=args.collect_attentions,
                output_hidden_states=args.collect_hidden_states,
                use_cache=False,
            )

            if args.collect_attentions:
                if outputs.attentions is None:
                    raise RuntimeError("Model did not return attention tensors.")
                if attentions_per_batch is None:
                    raise RuntimeError("Internal error: attention container not initialized.")
                layer_attns = [attn.detach() for attn in outputs.attentions]
                attentions_per_batch.append(layer_attns)

            if args.collect_hidden_states:
                if outputs.hidden_states is None:
                    raise RuntimeError("Model did not return hidden states.")
                if hidden_states_per_batch is None:
                    raise RuntimeError("Internal error: hidden state container not initialized.")
                hidden_states_per_batch.append([hs.detach() for hs in outputs.hidden_states])

        result: Dict[str, List[List[torch.Tensor]]] = {}
        if args.collect_attentions:
            if attentions_per_batch is None:
                raise RuntimeError("Internal error: missing reference attention results.")
            result["attentions"] = attentions_per_batch
        if args.collect_hidden_states:
            if hidden_states_per_batch is None:
                raise RuntimeError("Internal error: missing reference hidden state results.")
            result["hidden_states"] = hidden_states_per_batch
        captured_results["hf_reference"] = result

    benchmark("hf_reference", run_with_output_features)

    benchmark("baseline", run_baseline)

    def validate_captured_features() -> None:
        if args.collect_attentions:
            hook_attn_batches = captured_results.get("pytorch_hook", {}).get("attentions")
            hf_attn_batches = captured_results.get("hf_reference", {}).get("attentions")
            if hook_attn_batches is not None and hf_attn_batches is not None:
                print("\nValidating hook-captured attentions against HF output_attentions...")
                if len(hook_attn_batches) != len(hf_attn_batches):
                    raise RuntimeError(
                        "Mismatch in number of batches between hook-captured and HF attentions."
                    )

                for batch_idx, (hook_layers, hf_layers) in enumerate(zip(hook_attn_batches, hf_attn_batches)):
                    if len(hook_layers) != len(hf_layers):
                        raise RuntimeError(
                            f"Mismatch in number of layers for batch {batch_idx}: hook has {len(hook_layers)}, HF has {len(hf_layers)}."
                        )

                    for layer_idx, (hook_attn, hf_attn) in enumerate(zip(hook_layers, hf_layers)):
                        if hook_attn.shape != hf_attn.shape:
                            raise RuntimeError(
                                "Attention tensor shape mismatch at batch {} layer {}: hook {} vs HF {}.".format(
                                    batch_idx, layer_idx, tuple(hook_attn.shape), tuple(hf_attn.shape)
                                )
                            )

                        if batch_idx == 0:
                            print(
                                "  attention batch {} layer {} shape {} dtype {} device {}".format(
                                    batch_idx,
                                    layer_idx,
                                    tuple(hook_attn.shape),
                                    hook_attn.dtype,
                                    hook_attn.device,
                                )
                            )

                        if hook_attn.dtype != hf_attn.dtype:
                            hf_attn = hf_attn.to(dtype=hook_attn.dtype)
                        if hook_attn.device != hf_attn.device:
                            hf_attn = hf_attn.to(device=hook_attn.device)

                        if not torch.allclose(hook_attn, hf_attn, rtol=1e-4, atol=1e-6):
                            diff = (hook_attn - hf_attn).abs()
                            max_diff = diff.max().item()
                            mean_diff = diff.mean().item()
                            raise RuntimeError(
                                "Attention tensor values diverged at batch {} layer {} (max diff {:.3e}, mean diff {:.3e}).".format(
                                    batch_idx, layer_idx, max_diff, mean_diff
                                )
                            )

                print("Hook-captured attentions are consistent with HF outputs (shapes and values match).\n")

        if args.collect_hidden_states:
            hook_hs_batches = captured_results.get("pytorch_hook", {}).get("hidden_states")
            reference_hs_batches = captured_results.get("hf_reference", {}).get("hidden_states")
            if hook_hs_batches is not None and reference_hs_batches is not None:
                print("Validating hook-captured hidden states against HF outputs...")
                if len(hook_hs_batches) != len(reference_hs_batches):
                    raise RuntimeError(
                        "Mismatch in number of batches between hook-captured and HF hidden states."
                    )

                for batch_idx, (hook_layers, hf_layers) in enumerate(zip(hook_hs_batches, reference_hs_batches)):
                    if len(hook_layers) != len(hf_layers):
                        raise RuntimeError(
                            f"Mismatch in number of layers for batch {batch_idx}: hook has {len(hook_layers)}, HF has {len(hf_layers)}."
                        )

                    for layer_idx, (hook_hs, hf_hs) in enumerate(zip(hook_layers, hf_layers)):
                        if hook_hs.shape != hf_hs.shape:
                            raise RuntimeError(
                                "Hidden state tensor shape mismatch at batch {} layer {}: hook {} vs HF {}.".format(
                                    batch_idx, layer_idx, tuple(hook_hs.shape), tuple(hf_hs.shape)
                                )
                            )

                        if batch_idx == 0:
                            print(
                                "  hidden batch {} layer {} shape {} dtype {} device {}".format(
                                    batch_idx,
                                    layer_idx,
                                    tuple(hook_hs.shape),
                                    hook_hs.dtype,
                                    hook_hs.device,
                                )
                            )

                        if hook_hs.dtype != hf_hs.dtype:
                            hf_hs = hf_hs.to(dtype=hook_hs.dtype)
                        if hook_hs.device != hf_hs.device:
                            hf_hs = hf_hs.to(device=hook_hs.device)

                        if not torch.allclose(hook_hs, hf_hs, rtol=1e-4, atol=1e-6):
                            diff = (hook_hs - hf_hs).abs()
                            max_diff = diff.max().item()
                            mean_diff = diff.mean().item()
                            raise RuntimeError(
                                "Hidden state values diverged at batch {} layer {} (max diff {:.3e}, mean diff {:.3e}).".format(
                                    batch_idx, layer_idx, max_diff, mean_diff
                                )
                            )

                print("Hook-captured hidden states are consistent with HF outputs (shapes and values match).\n")
            elif reference_hs_batches is not None and hook_hs_batches is None:
                first_batch = reference_hs_batches[0]
                print("\nHidden state shapes (HF reference only):")
                for layer_idx, hs in enumerate(first_batch):
                    print(
                        "  hidden batch 0 layer {} shape {} dtype {} device {}".format(
                            layer_idx,
                            tuple(hs.shape),
                            hs.dtype,
                            hs.device,
                        )
                    )

    validate_captured_features()

    def move_results_to_cpu(labels: Iterable[str]) -> None:
        if device.type != "cuda":
            return

        for label in labels:
            payload = captured_results.get(label)
            if payload is None:
                continue
            captured_results[label] = to_cpu_recursive(payload)

        torch.cuda.empty_cache()

    move_results_to_cpu(["baseline", "pytorch_hook", "pytorch_hook_cpu", "pytorch_hook_dense", "hf_reference"])

    if args.profile:
        print(
            "\nProfiling complete. Trace files are written to ./tb_traces/<label> and key averages are shown above."
        )
    else:
        print("\nTiming results (seconds and tokens/sec):")
        for label, stats in timings.items():
            mean = stats["mean"]
            stdev = stats["stdev"]
            runs = ", ".join(f"{duration:.4f}" for duration in stats["runs"])
            tokens_per_second = stats.get("tokens_per_second", float("nan"))
            print(
                f"- {label:>18}: mean={mean:.4f}s stdev={stdev:.4f}s token/s={tokens_per_second:.2f}]"
            )

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for label, features in captured_results.items():
            if not isinstance(features, dict):
                torch.save(features, save_path / f"{label}.pt")
                continue
            for feature_name, tensors in features.items():
                torch.save(tensors, save_path / f"{label}_{feature_name}.pt")
        print(f"Saved captured tensors to {save_path.resolve()}")


if __name__ == "__main__":
    main()
