"""Benchmark for Hooked GPT-2 LMHead model (async with monitoring engine)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler

from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import MonitoringEngine

try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    class nvtx:
        @staticmethod
        def range_push(msg): pass
        @staticmethod
        def range_pop(): pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Hooked GPT-2 LMHead (async with monitoring engine)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--prefill-tokens", type=int, default=1, help="Prompt tokens")
    parser.add_argument("--decode-steps", type=int, default=64, help="Decode steps")
    parser.add_argument("--steps", type=int, default=3, help="Profiled iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--profile-dir",
        default="results/profile_hf_modified_lmhead_async",
        help="Profile output directory",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Enable NVTX annotations",
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
        "--no-profile",
        action="store_true",
        help="Skip profiling, measure wallclock only",
    )

    args = parser.parse_args()
    if not args.collect_hidden and not args.collect_attention:
        parser.error("At least one of --collect-hidden or --collect-attention must be provided.")
    if args.prefill_tokens < 1:
        parser.error("--prefill-tokens must be at least 1.")
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


def measure_async_model(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
    engine,
) -> Tuple[float, float]:
    """Measure wallclock time for async model without profiling overhead."""
    if device.type == "cuda":
        torch.cuda.synchronize()

    nvtx.range_push(f"measure_async_{label}")
    start = time.perf_counter()

    nvtx.range_push(f"async_compute_{label}")
    fn()
    nvtx.range_pop()  # async_compute

    if device.type == "cuda":
        torch.cuda.current_stream().synchronize()

    main_elapsed = time.perf_counter() - start

    nvtx.range_push(f"async_resolve_all_{label}")
    engine.resolve_all()
    nvtx.range_pop()  # async_resolve_all

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_elapsed = time.perf_counter() - start
    nvtx.range_pop()  # measure_async
    return main_elapsed, total_elapsed


def profile_async_model(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
    trace_dir: Path,
    engine,
) -> Tuple[float, float]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        try:
            activities.append(ProfilerActivity.CUDA)
        except AttributeError:
            pass

    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = tensorboard_trace_handler(str(trace_dir / label))

    nvtx.range_push(f"profile_async_{label}")
    wall_time_start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=handler,
    ) as prof:
        with record_function(label):
            nvtx.range_push(f"async_compute_{label}")
            fn()
            nvtx.range_pop()  # async_compute
        if device.type == "cuda":
            torch.cuda.current_stream().synchronize()
    main_elapsed = time.perf_counter() - wall_time_start

    nvtx.range_push(f"async_resolve_all_{label}")
    engine.resolve_all()
    nvtx.range_pop()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - wall_time_start
    nvtx.range_pop()
    return main_elapsed, total_elapsed


def run_decode_loop(
    prefill_fn: Callable[[], Tuple[object, torch.Tensor]],
    decode_fn: Callable[[torch.Tensor, object], Tuple[torch.Tensor, object]],
    decode_steps: int,
) -> None:
    state, token = prefill_fn()
    for _ in range(decode_steps):
        logits, state = decode_fn(token, state)
        token = greedy_from_logits(logits)


def main() -> None:
    args = parse_args()

    if args.nvtx:
        import os
        os.environ.setdefault("TL_ENABLE_NVTX", "1")

    device = pick_device(args.device)
    dtype = map_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = build_inputs(args.batch_size, args.prefill_tokens, tokenizer, device)

    print(
        f"Using device: {device} | dtype: {args.dtype} | cache_dtype: {args.cache_dtype}"
        f" | prefill_tokens={args.prefill_tokens} | decode_steps={args.decode_steps}"
        f" | collect_hidden={args.collect_hidden} | collect_attention={args.collect_attention}"
    )

    hf_hooked_model = HookedGPT2LMHeadModel.from_pretrained(
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
    )
    hf_hooked_model.monitoring_engine = monitoring_engine
    monitoring_engine.prepare_for_model(hf_hooked_model)

    def lmhead_prefill(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        nvtx.range_push("lmhead_prefill")
        monitoring_engine.start_step()
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
        logits = outputs.logits
        next_token = greedy_from_logits(logits)
        past = outputs.past_key_values
        cache_dict.clear()
        try:
            if monitoring_engine.async_enabled:
                monitoring_engine.clear_completed_results()
        except Exception:
            pass
        del outputs, logits
        nvtx.range_pop()
        return past, next_token

    def lmhead_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        nvtx.range_push("lmhead_decode")
        monitoring_engine.start_step()
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
        logits = outputs.logits
        next_past = outputs.past_key_values
        cache_dict.clear()
        try:
            if monitoring_engine.async_enabled:
                monitoring_engine.clear_completed_results()
        except Exception:
            pass
        del outputs
        nvtx.range_pop()
        return logits, next_past

    def run_one() -> None:
        run_decode_loop(lmhead_prefill, lmhead_decode, args.decode_steps)

    for _ in range(args.warmup):
        run_one()

    label = "hf_modified_lmhead_hook_async"
    if args.no_profile:
        main_elapsed, total_elapsed = measure_async_model(label, run_one, device, monitoring_engine)
    else:
        trace_dir = Path(args.profile_dir)
        main_elapsed, total_elapsed = profile_async_model(label, run_one, device, trace_dir, monitoring_engine)

    tokens_generated = args.decode_steps * args.batch_size
    main_tps = tokens_generated / main_elapsed if main_elapsed > 0 else 0.0
    total_tps = tokens_generated / total_elapsed if total_elapsed > 0 else 0.0

    results = {
        label: {
            "main_duration": main_elapsed,
            "total_duration": total_elapsed,
            "main_token/s": main_tps,
            "total_token/s": total_tps,
        }
    }

    output_dir = Path(args.profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "timing_results_lmhead.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Timing results:")
    print(f"- {label}: main_duration={main_elapsed:.4f}s total_duration={total_elapsed:.4f}s"
          f" main_token/s={main_tps:.2f} total_token/s={total_tps:.2f}")


if __name__ == "__main__":
    main()
