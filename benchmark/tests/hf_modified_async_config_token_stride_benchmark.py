"""Benchmark config token stride impact for HF Modified GPT-2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

from monitoring import MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection, MonitoringConfig


# EDIT ME: token stride values to test
TOKEN_STRIDES = [1, 10, 30, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HF Modified GPT-2 config token stride impact"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--prefill-tokens", type=int, default=1, help="Prompt tokens")
    parser.add_argument("--decode-steps", type=int, default=64, help="Decode steps per request")
    parser.add_argument("--steps", type=int, default=2, help="Number of requests")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--collect-hidden",
        action="store_true",
        help="Capture decoder hidden states",
    )
    parser.add_argument(
        "--collect-attention",
        action="store_true",
        help="Capture decoder attention tensors",
    )
    parser.add_argument(
        "--cache-dtype",
        default="none",
        choices=["none", "fp32", "fp16", "bf16"],
        help="Optional dtype to store cached activations in the monitoring engine",
    )
    parser.add_argument(
        "--engine-queue-size",
        type=int,
        default=0,
        help="Max queued async tasks to allow (0 = unbounded)",
    )
    parser.add_argument(
        "--engine-delay-steps",
        type=int,
        default=0,
        help="Defer processing by K steps (ring buffer); 0 = no delay",
    )
    parser.add_argument(
        "--output-dir",
        default="results/profile_hf_modified_token_stride",
        help="Output directory for timing JSON",
    )

    args = parser.parse_args()
    if not args.collect_hidden and not args.collect_attention:
        parser.error("At least one of --collect-hidden or --collect-attention must be provided.")
    if args.prefill_tokens < 1:
        parser.error("--prefill-tokens must be at least 1.")
    if args.decode_steps < 1:
        parser.error("--decode-steps must be at least 1.")
    if args.steps < 1:
        parser.error("--steps must be at least 1.")
    return args


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def map_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def build_inputs(
    batch_size: int,
    sequence_length: int,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    prompts = ["Transformers are powerful."] * batch_size
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )
    return encoded["input_ids"].to(device)


def greedy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def run_decode_loop(
    prefill_fn: Callable[[], Tuple[object, torch.Tensor]],
    decode_fn: Callable[[torch.Tensor, object], Tuple[torch.Tensor, object]],
    decode_steps: int,
) -> None:
    state, token = prefill_fn()
    for _ in range(decode_steps):
        logits, state = decode_fn(token, state)
        token = greedy_from_logits(logits)


def measure_async_model(
    fn: Callable[[], None],
    device: torch.device,
    engine: MonitoringEngine,
) -> Tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.current_stream().synchronize()
    main_elapsed = time.perf_counter() - start

    engine.resolve_all()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - start
    return main_elapsed, total_elapsed


def measure_model(fn: Callable[[], None], device: torch.device) -> Tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.current_stream().synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, elapsed


def run_hf_baseline(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    prompt_tokens: torch.Tensor,
) -> Dict[str, float]:
    print("\n== hf_baseline ==")

    hf_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    def prefill_fn(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        outputs = hf_model(
            prefill_tokens,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        next_token = greedy_from_logits(outputs.logits)
        past = outputs.past_key_values
        del outputs
        return past, next_token

    def decode_fn(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        outputs = hf_model(
            token,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        logits = outputs.logits
        next_past = outputs.past_key_values
        del outputs
        return logits, next_past

    def run_requests(request_count: int) -> None:
        with torch.no_grad():
            for _ in range(request_count):
                run_decode_loop(
                    lambda: prefill_fn(prompt_tokens),
                    decode_fn,
                    args.decode_steps,
                )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.warmup > 0:
        run_requests(args.warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()

    main_elapsed, total_elapsed = measure_model(lambda: run_requests(args.steps), device)

    total_decoded_tokens = args.decode_steps * args.steps * args.batch_size
    return {
        "main_duration": main_elapsed,
        "total_duration": total_elapsed,
        "tokens_per_second_main": total_decoded_tokens / main_elapsed if main_elapsed > 0 else float("inf"),
        "tokens_per_second_total": total_decoded_tokens / total_elapsed if total_elapsed > 0 else float("inf"),
    }


def run_config_case(
    label: str,
    config: MonitoringConfig,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    prompt_tokens: torch.Tensor,
) -> Dict[str, float]:
    print(f"\n== {label} ==")

    hf_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    hf_hooked_model = HookedGPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    hf_hooked_model.to(device)
    hf_hooked_model.eval()

    cache_dtype = None if args.cache_dtype == "none" else map_dtype(args.cache_dtype)
    monitoring_engine = MonitoringEngine(
        async_enabled=device.type == "cuda",
        cache_dtype=cache_dtype,
        queue_size=args.engine_queue_size,
        delay_steps=args.engine_delay_steps,
        config=config,
    )
    hf_hooked_model.monitoring_engine = monitoring_engine

    lm_head = hf_model.lm_head

    def prefill_fn(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        monitoring_engine.start_step(phase="prefill")
        try:
            outputs, cache_dict = hf_hooked_model.run_with_cache(
                prefill_tokens,
                use_cache=True,
                output_hidden_states=args.collect_hidden,
                output_attentions=args.collect_attention,
                return_dict=True,
            )
        finally:
            monitoring_engine.end_step()

        hidden_states = outputs.last_hidden_state
        logits = lm_head(hidden_states)
        next_token = greedy_from_logits(logits)
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        past = outputs.past_key_values
        cache_dict.clear()
        if monitoring_engine.async_enabled:
            monitoring_engine.clear_completed_results()
        del hidden_states, logits, outputs
        return past, next_token

    def decode_fn(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        monitoring_engine.start_step(phase="decode")
        try:
            outputs, cache_dict = hf_hooked_model.run_with_cache(
                token,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=args.collect_hidden,
                output_attentions=args.collect_attention,
                return_dict=True,
            )
        finally:
            monitoring_engine.end_step()

        hidden_states = outputs.last_hidden_state
        logits = lm_head(hidden_states)
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        next_past = outputs.past_key_values
        cache_dict.clear()
        if monitoring_engine.async_enabled:
            monitoring_engine.clear_completed_results()
        del hidden_states, outputs
        return logits, next_past

    def run_requests(request_count: int, request_id_start: int) -> None:
        with torch.no_grad():
            for i in range(request_count):
                monitoring_engine.begin_request(request_id_start + i)
                run_decode_loop(
                    lambda: prefill_fn(prompt_tokens),
                    decode_fn,
                    args.decode_steps,
                )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.warmup > 0:
        warmup_start = -max(1, args.warmup)
        run_requests(args.warmup, warmup_start)
        if monitoring_engine.async_enabled:
            monitoring_engine.resolve_all()
        if device.type == "cuda":
            torch.cuda.synchronize()

    main_elapsed, total_elapsed = measure_async_model(
        lambda: run_requests(args.steps, 0),
        device,
        monitoring_engine,
    )

    total_decoded_tokens = args.decode_steps * args.steps * args.batch_size
    metrics = {
        "main_duration": main_elapsed,
        "total_duration": total_elapsed,
        "tokens_per_second_main": total_decoded_tokens / main_elapsed if main_elapsed > 0 else float("inf"),
        "tokens_per_second_total": total_decoded_tokens / total_elapsed if total_elapsed > 0 else float("inf"),
    }

    monitoring_engine.close()
    return metrics


def main() -> None:
    args = parse_args()

    device = pick_device(args.device)
    dtype = map_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = build_inputs(args.batch_size, args.prefill_tokens, tokenizer, device)

    print(
        f"Using device: {device} | dtype: {args.dtype} | batch_size={args.batch_size}"
        f" | prefill_tokens={args.prefill_tokens} | decode_steps={args.decode_steps}"
        f" | requests={args.steps} | warmup={args.warmup}"
    )
    print(f"Token strides: {TOKEN_STRIDES}")

    base_hooks = HookSelection(mode="full")

    results: Dict[str, Dict[str, float]] = {}
    results["hf_baseline"] = run_hf_baseline(args, device, dtype, prompt_tokens)
    print(
        f"- hf_baseline: main_duration={results['hf_baseline']['main_duration']:.4f}s "
        f"total_duration={results['hf_baseline']['total_duration']:.4f}s "
        f"main_token/s={results['hf_baseline']['tokens_per_second_main']:.2f} "
        f"total_token/s={results['hf_baseline']['tokens_per_second_total']:.2f}"
    )

    for stride in TOKEN_STRIDES:
        cfg = MonitoringConfig(
            hooks=base_hooks,
            schedule=CaptureSchedule(
                step_stride=int(stride),
                step_offset=0,
                warmup_steps=0,
                capture_prefill=False,
                capture_decode=True,
                request_stride=1,
                request_offset=0,
                warmup_requests=0,
            ),
        )
        label = f"token_stride_{stride}"
        results[label] = run_config_case(label, cfg, args, device, dtype, prompt_tokens)
        print(
            f"- {label}: main_duration={results[label]['main_duration']:.4f}s "
            f"total_duration={results[label]['total_duration']:.4f}s "
            f"main_token/s={results[label]['tokens_per_second_main']:.2f} "
            f"total_token/s={results[label]['tokens_per_second_total']:.2f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "timing_results.json"
    results_data = {
        "config": {
            "batch_size": args.batch_size,
            "prefill_tokens": args.prefill_tokens,
            "decode_steps": args.decode_steps,
            "steps": args.steps,
            "warmup": args.warmup,
            "device": str(device),
            "dtype": args.dtype,
            "collect_hidden": args.collect_hidden,
            "collect_attention": args.collect_attention,
            "cache_dtype": args.cache_dtype,
            "engine_queue_size": args.engine_queue_size,
            "engine_delay_steps": args.engine_delay_steps,
            "token_strides": TOKEN_STRIDES,
        },
        "results": results,
    }
    with results_file.open("w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nTiming results saved to: {results_file.resolve()}")


if __name__ == "__main__":
    main()
