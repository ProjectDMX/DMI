"""Simplified benchmark for HF Modified model (no cache, baseline version)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

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
        description="Benchmark HF Modified GPT-2 (no cache baseline)"
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
        default="results/profile_hf_modified_no_cache",
        help="Profile output directory",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Enable NVTX annotations",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip profiling, measure wallclock only",
    )

    args = parser.parse_args()
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


def measure_model(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
) -> float:
    """Measure wallclock time without profiling overhead."""
    if device.type == "cuda":
        torch.cuda.synchronize()

    nvtx.range_push(f"measure_{label}")
    start = time.perf_counter()

    fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    nvtx.range_pop()  # measure
    return elapsed


def profile_model(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
    trace_dir: Path,
) -> float:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        try:
            activities.append(ProfilerActivity.CUDA)
        except AttributeError:
            pass

    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = tensorboard_trace_handler(str(trace_dir / label))

    nvtx.range_push(f"profile_{label}")
    wall_time_start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=handler,
    ) as prof:
        with record_function(label):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - wall_time_start
    nvtx.range_pop()  # profile
    return elapsed


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

    device = pick_device(args.device)
    dtype = map_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = build_inputs(args.batch_size, args.prefill_tokens, tokenizer, device)

    print(
        f"Using device: {device} | dtype: {args.dtype}"
        f" | prefill_tokens={args.prefill_tokens} | decode_steps={args.decode_steps}"
    )

    # Load base HF model for lm_head
    hf_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    # Load HF Modified (HookedGPT2Model) - no monitoring engine
    hf_hooked_model = HookedGPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    hf_hooked_model.to(device)
    hf_hooked_model.eval()

    # Get lm_head for logits projection from base model
    lm_head = hf_model.lm_head

    def project_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        return lm_head(hidden_states)

    # === hf_modified_no_cache functions (direct forward, no cache) ===
    def hf_modified_no_cache_prefill(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        nvtx.range_push("modified_prefill")
        nvtx.range_push("modified_prefill_forward")
        outputs = hf_hooked_model(
            prefill_tokens,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        nvtx.range_pop()  # modified_prefill_forward
        nvtx.range_push("modified_prefill_post")
        hidden_states = outputs.last_hidden_state
        nvtx.range_push("modified_prefill_project")
        logits = project_logits(hidden_states)
        nvtx.range_pop()  # modified_prefill_project
        next_token = greedy_from_logits(logits)
        nvtx.range_pop()  # modified_prefill_post
        past = outputs.past_key_values
        del hidden_states, logits, outputs
        nvtx.range_pop()  # modified_prefill
        return past, next_token

    def hf_modified_no_cache_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        nvtx.range_push("modified_decode")
        nvtx.range_push("modified_decode_forward")
        outputs = hf_hooked_model(
            token,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        nvtx.range_pop()  # modified_decode_forward
        nvtx.range_push("modified_decode_post")
        hidden_states = outputs.last_hidden_state
        nvtx.range_push("modified_decode_project")
        logits = project_logits(hidden_states)
        nvtx.range_pop()  # modified_decode_project
        next_past = outputs.past_key_values
        del hidden_states, outputs
        nvtx.range_pop()  # modified_decode_post
        nvtx.range_pop()  # modified_decode
        return logits, next_past

    def run_decode(prefill_fn, decode_fn, prefill_tokens=prompt_tokens):
        with torch.no_grad():
            for i in range(args.steps):
                nvtx.range_push(f"benchmark_iter_{i}")
                run_decode_loop(lambda: prefill_fn(prefill_tokens), decode_fn, args.decode_steps)
                nvtx.range_pop()  # benchmark_iter_i

    def warmup(prefill_fn, decode_fn, prefill_tokens=prompt_tokens):
        if args.warmup <= 0:
            return
        nvtx.range_push("warmup")
        with torch.no_grad():
            for i in range(args.warmup):
                nvtx.range_push(f"warmup_iter_{i}")
                run_decode_loop(lambda: prefill_fn(prefill_tokens), decode_fn, args.decode_steps)
                nvtx.range_pop()  # warmup_iter_i
        nvtx.range_pop()  # warmup

    traces_path = Path(args.profile_dir)
    total_decoded_tokens = args.decode_steps * args.steps * args.batch_size

    # Choose measurement function
    if args.no_profile:
        run_benchmark = lambda label, fn: measure_model(label, fn, device)
        print("Running benchmark WITHOUT profiling (pure wallclock time)")
    else:
        run_benchmark = lambda label, fn: profile_model(label, fn, device, traces_path)
        print("Running benchmark WITH profiling (trace files will be generated)")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run warmup
    warmup(hf_modified_no_cache_prefill, hf_modified_no_cache_decode)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Run benchmark
    decode_elapsed = run_benchmark(
        "hf_modified_no_cache",
        lambda: run_decode(hf_modified_no_cache_prefill, hf_modified_no_cache_decode),
    )

    timings = {
        "hf_modified_no_cache": {
            "duration": decode_elapsed,
            "tokens_per_second": total_decoded_tokens / decode_elapsed
            if decode_elapsed > 0
            else float("inf"),
        }
    }

    # Save results
    results_file = traces_path / "timing_results.json"
    traces_path.mkdir(parents=True, exist_ok=True)

    results_data = {
        "config": {
            "batch_size": args.batch_size,
            "prefill_tokens": args.prefill_tokens,
            "decode_steps": args.decode_steps,
            "steps": args.steps,
            "warmup": args.warmup,
            "device": str(device),
            "dtype": args.dtype,
            "profiling_enabled": not args.no_profile,
        },
        "timings": timings,
        "total_decoded_tokens": total_decoded_tokens,
    }

    with results_file.open("w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nTiming results saved to: {results_file.resolve()}")
    print("\nTiming results:")
    print(
        f"- hf_modified_no_cache: duration={decode_elapsed:.4f}s "
        f"token/s={timings['hf_modified_no_cache']['tokens_per_second']:.2f}"
    )

    if not args.no_profile:
        print(f"\nProfiler traces written under: {traces_path.resolve()}")
    if args.nvtx and device.type == "cuda":
        print("NVTX annotations enabled (MonitoringConfig.debug=True).")


if __name__ == "__main__":
    main()
