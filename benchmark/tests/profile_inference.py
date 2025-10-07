"""Profiler benchmark comparing TransformerLens vs Hugging Face GPT-2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile GPT-2 inference between TransformerLens and Hugging Face"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for profiling run")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Token length per sample (inputs are padded/truncated to this size)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of forward passes to record inside the profiler context",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warm-up iterations run before profiling",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (defaults to cuda if available else cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Computation dtype for both models",
    )
    parser.add_argument(
        "--profile-dir",
        default="results/profile_traces",
        help="Directory to write TensorBoard profiler traces",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Enable NVTX annotations inside TransformerLens hooks (sets TL_ENABLE_NVTX=1)",
    )
    parser.add_argument(
        "--collect-hidden",
        action="store_true",
        help="Capture hidden states for baselines that support it",
    )
    parser.add_argument(
        "--collect-attention",
        action="store_true",
        help="Capture attention tensors for baselines that support it",
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
        help="Skip profiling and only measure wallclock time (faster, no trace files)",
    )

    args = parser.parse_args()
    if not args.collect_hidden and not args.collect_attention:
        parser.error("At least one of --collect-hidden or --collect-attention must be provided.")
    return args


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def map_hf_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[name]


def map_tl_dtype(name: str) -> str:
    return {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
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


def measure_model(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
) -> float:
    """Measure wallclock time without profiling overhead."""
    import time

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return time.perf_counter() - start


def measure_async_model(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
    engine,
) -> Tuple[float, float]:
    """Measure wallclock time for async model without profiling overhead."""
    import time

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    fn()

    # Only sync the main compute stream, NOT the background cache stream
    if device.type == "cuda":
        torch.cuda.current_stream().synchronize()

    main_elapsed = time.perf_counter() - start
    engine.resolve_all()

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_elapsed = time.perf_counter() - start
    return main_elapsed, total_elapsed


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
        except AttributeError:  # pragma: no cover - older PyTorch versions
            pass

    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = tensorboard_trace_handler(str(trace_dir / label))

    import time

    wall_time_start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=handler,
    ) as prof:
        with record_function(label):
            if device.type == "cuda":
                torch.cuda.synchronize()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()

    return time.perf_counter() - wall_time_start


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
        except AttributeError:  # pragma: no cover
            pass

    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = tensorboard_trace_handler(str(trace_dir / label))

    import time

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
        # Sync only main stream before exiting profiler context
        if device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    main_elapsed = time.perf_counter() - wall_time_start
    engine.resolve_all()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - wall_time_start
    return main_elapsed, total_elapsed


def main() -> None:
    args = parse_args()

    if args.nvtx:
        os.environ.setdefault("TL_ENABLE_NVTX", "1")

    device = pick_device(args.device)
    hf_dtype = map_hf_dtype(args.dtype)
    tl_dtype = map_tl_dtype(args.dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model
    from transformer_lens import HookedTransformer
    from monitoring import MonitoringEngine

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = build_inputs(args.batch_size, args.sequence_length, tokenizer, device)

    print(
        f"Using device: {device} | dtype: {args.dtype}"
        f" | collect_hidden={args.collect_hidden}"
        f" | collect_attention={args.collect_attention}"
    )

    # Hugging Face model with eager attention for parity
    hf_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=hf_dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    # TransformerLens model
    tl_model = HookedTransformer.from_pretrained(
        "gpt2",
        device=device,
        dtype=tl_dtype,
    )
    tl_model.eval()

    # Modified Hugging Face GPT-2 with TransformerLens-style hooks
    hf_hooked_model = HookedGPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=hf_dtype,
    )
    hf_hooked_model.to(device)
    hf_hooked_model.eval()

    cache_dtype = None if args.cache_dtype == "none" else map_hf_dtype(args.cache_dtype)
    monitoring_engine = MonitoringEngine(
        async_enabled=device.type == "cuda",
        cache_dtype=cache_dtype,
        queue_size=args.engine_queue_size,
        delay_steps=args.engine_delay_steps,
    )
    hf_hooked_model.monitoring_engine = None

    if device.type == "cuda":
        torch.cuda.empty_cache()

    def setup_hf_hook(
        collect_hidden: bool,
        collect_attention: bool,
        move_to_cpu: bool = False,
    ) -> Tuple[Callable[[], None], Callable[[], None]]:
        transformer = getattr(hf_model, "transformer", None)
        blocks: List[torch.nn.Module] | None = getattr(transformer, "h", None) if transformer else None
        if not blocks:
            raise RuntimeError("Unexpected GPT-2 architecture; transformer blocks not found.")

        num_layers = len(blocks)

        attn_cache: List[torch.Tensor | None] = [None] * num_layers
        q_cache: List[torch.Tensor | None] = [None] * num_layers
        k_cache: List[torch.Tensor | None] = [None] * num_layers
        v_cache: List[torch.Tensor | None] = [None] * num_layers
        attn_output_cache: List[torch.Tensor | None] = [None] * num_layers
        resid_pre_cache: List[torch.Tensor | None] = [None] * num_layers
        resid_post_cache: List[torch.Tensor | None] = [None] * num_layers
        ln1_cache: List[torch.Tensor | None] = [None] * num_layers
        ln2_cache: List[torch.Tensor | None] = [None] * num_layers
        mlp_in_cache: List[torch.Tensor | None] = [None] * num_layers
        mlp_out_cache: List[torch.Tensor | None] = [None] * num_layers

        hidden_states_cache: List[torch.Tensor] = []

        def store_tensor(tensor: torch.Tensor) -> torch.Tensor:
            stored = tensor.detach()
            if move_to_cpu:
                stored = stored.cpu()
            return stored

        patched_attn: List[Tuple[torch.nn.Module, Callable[..., Tuple]]] = []
        extra_hooks: List[torch.utils.hooks.RemovableHandle] = []

        for idx, block in enumerate(blocks):
            attn_module = block.attn

            if collect_attention:
                original_forward = attn_module.forward

                def wrapped_forward(*f_args, _orig=original_forward, _idx=idx, **f_kwargs):
                    f_kwargs["output_attentions"] = True
                    outputs = _orig(*f_args, **f_kwargs)

                    if not isinstance(outputs, tuple) or len(outputs) != 2:
                        raise RuntimeError(
                            "Unexpected GPT-2 attention output structure during hook capture."
                        )

                    attn_output, attn_probs = outputs

                    attn_output_cache[_idx] = store_tensor(attn_output)
                    attn_cache[_idx] = store_tensor(attn_probs)

                    return attn_output, attn_probs

                attn_module.forward = wrapped_forward  # type: ignore[assignment]
                patched_attn.append((attn_module, original_forward))

                def c_attn_hook(
                    module: torch.nn.Module,
                    module_input: Tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                    _attn=attn_module,
                ) -> None:
                    q, k, v = module_output.split(_attn.split_size, dim=2)
                    num_heads = _attn.num_heads
                    head_dim = _attn.head_dim

                    def reshape(t: torch.Tensor) -> torch.Tensor:
                        batch, seq_len, _ = t.size()
                        return store_tensor(
                            t.view(batch, seq_len, num_heads, head_dim)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                        )

                    q_cache[_idx] = reshape(q)
                    k_cache[_idx] = reshape(k)
                    v_cache[_idx] = reshape(v)

                extra_hooks.append(attn_module.c_attn.register_forward_hook(c_attn_hook))

            if collect_hidden:
                def block_pre_hook(
                    module: torch.nn.Module,
                    module_inputs: Tuple[torch.Tensor, ...],
                    _idx=idx,
                ):
                    resid_pre_cache[_idx] = store_tensor(module_inputs[0])

                def block_post_hook(
                    module: torch.nn.Module,
                    module_inputs: Tuple[torch.Tensor, ...],
                    module_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                    _idx=idx,
                ) -> None:
                    hidden = module_output[0] if isinstance(module_output, tuple) else module_output
                    resid_post_cache[_idx] = store_tensor(hidden)

                def ln1_hook(
                    module: torch.nn.Module,
                    module_inputs: Tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                ) -> None:
                    ln1_cache[_idx] = store_tensor(module_output)

                def ln2_hook(
                    module: torch.nn.Module,
                    module_inputs: Tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                ) -> None:
                    ln2_cache[_idx] = store_tensor(module_output)

                def mlp_pre_hook(
                    module: torch.nn.Module,
                    module_inputs: Tuple[torch.Tensor, ...],
                    _idx=idx,
                ) -> None:
                    mlp_in_cache[_idx] = store_tensor(module_inputs[0])

                def mlp_post_hook(
                    module: torch.nn.Module,
                    module_inputs: Tuple[torch.Tensor, ...],
                    module_output: torch.Tensor,
                    _idx=idx,
                ) -> None:
                    mlp_out_cache[_idx] = store_tensor(module_output)

                extra_hooks.append(block.register_forward_pre_hook(block_pre_hook))
                extra_hooks.append(block.register_forward_hook(block_post_hook))
                extra_hooks.append(block.ln_1.register_forward_hook(ln1_hook))
                extra_hooks.append(block.ln_2.register_forward_hook(ln2_hook))
                extra_hooks.append(block.mlp.register_forward_pre_hook(mlp_pre_hook))
                extra_hooks.append(block.mlp.register_forward_hook(mlp_post_hook))

        def reset_attention_cache() -> None:
            if not collect_attention:
                return
            for idx in range(num_layers):
                attn_cache[idx] = None
                q_cache[idx] = None
                k_cache[idx] = None
                v_cache[idx] = None
                attn_output_cache[idx] = None

        def reset_hidden_cache() -> None:
            if not collect_hidden:
                return
            hidden_states_cache.clear()
            for idx in range(num_layers):
                resid_pre_cache[idx] = None
                resid_post_cache[idx] = None
                ln1_cache[idx] = None
                ln2_cache[idx] = None
                mlp_in_cache[idx] = None
                mlp_out_cache[idx] = None

        def step() -> None:
            reset_attention_cache()
            reset_hidden_cache()

            outputs = hf_model(
                input_ids,
                output_hidden_states=collect_hidden,
                output_attentions=collect_attention,
                use_cache=False,
            )

            if collect_hidden and outputs.hidden_states is not None:
                for hs in outputs.hidden_states:
                    hidden_states_cache.append(store_tensor(hs))

            # Drop references to avoid lingering allocations
            if collect_attention and outputs.attentions is not None:
                for attn in outputs.attentions:
                    _ = attn

            reset_hidden_cache()
            reset_attention_cache()

        def cleanup() -> None:
            for module, original in patched_attn:
                module.forward = original  # type: ignore[assignment]
            for handle in extra_hooks:
                handle.remove()

        return step, cleanup

    def run_model(step_fn: Callable[[], None]) -> None:
        with torch.no_grad():
            for _ in range(args.steps):
                step_fn()
                if device.type == "cuda":
                    torch.cuda.synchronize()

    def tl_step() -> None:
        tl_model(input_ids, return_type="logits")

    def tl_cache_step() -> None:
        def names_filter(name: str) -> bool:
            lname = name.lower()
            if args.collect_hidden and args.collect_attention:
                return True
            if args.collect_attention:
                return "attn" in lname
            return "attn" not in lname

        _, cache_dict = tl_model.run_with_cache(
            input_ids,
            return_cache_object=False,
            names_filter=names_filter,
            remove_batch_dim=False,
        )
        # ensure tensors stay on device but release python references promptly
        cache_dict.clear()

    def hf_step() -> None:
        hf_model(input_ids, use_cache=False)

    def hf_api_step() -> None:
        outputs = hf_model(
            input_ids,
            output_attentions=args.collect_attention,
            output_hidden_states=args.collect_hidden,
            use_cache=False,
            return_dict=True,
        )

        # Drop references promptly to avoid lingering GPU allocations between steps
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        del outputs

    def hf_hooked_step() -> None:
        hf_hooked_model(input_ids, use_cache=False)

    def hf_hooked_cache_step() -> None:
        def names_filter(name: str) -> bool:
            lname = name.lower()
            if args.collect_hidden and args.collect_attention:
                return True
            if args.collect_attention:
                return "attn" in lname
            return "attn" not in lname

        _, cache_dict = hf_hooked_model.run_with_cache(
            input_ids,
            names_filter=names_filter,
            return_cache_object=False,
            remove_batch_dim=False,
        )
        cache_dict.clear()

    def hf_hooked_async_cache_step() -> None:
        monitoring_engine.start_step()
        try:
            def names_filter(name: str) -> bool:
                lname = name.lower()
                if args.collect_hidden and args.collect_attention:
                    return True
                if args.collect_attention:
                    return "attn" in lname
                return "attn" not in lname

            _, cache_dict = hf_hooked_model.run_with_cache(
                input_ids,
                names_filter=names_filter,
                return_cache_object=False,
                remove_batch_dim=False,
            )
        finally:
            monitoring_engine.end_step()
        cache_dict.clear()

    # Warmup
    print("Running warmup iterations...")
    with torch.no_grad():
        for _ in range(args.warmup):
            tl_step()
            tl_cache_step()
            hf_step()
            hf_api_step()
            hf_hooked_step()
            hf_hooked_cache_step()
            hf_hooked_model.monitoring_engine = monitoring_engine
            hf_hooked_async_cache_step()
            hf_hooked_model.monitoring_engine = None
            if device.type == "cuda":
                torch.cuda.synchronize()

    if monitoring_engine.async_enabled:
        monitoring_engine.resolve_all()
        if device.type == "cuda":
            torch.cuda.synchronize()
    hf_hooked_model.monitoring_engine = None

    traces_path = Path(args.profile_dir)

    tokens_processed = args.batch_size * args.sequence_length * args.steps
    timings: Dict[str, Dict[str, float]] = {}

    if args.no_profile:
        print("Running benchmarks WITHOUT profiling (pure wallclock time measurement)")

        def run_benchmark(label: str, fn: Callable[[], None]) -> float:
            return measure_model(label, fn, device)

        def run_async_benchmark(label: str, fn: Callable[[], None], engine) -> Tuple[float, float]:
            return measure_async_model(label, fn, device, engine)

    else:
        print("Running benchmarks WITH profiling (trace files will be generated)")

        def run_benchmark(label: str, fn: Callable[[], None]) -> float:
            return profile_model(label, fn, device, traces_path)

        def run_async_benchmark(label: str, fn: Callable[[], None], engine) -> Tuple[float, float]:
            return profile_async_model(label, fn, device, traces_path, engine)

    tl_elapsed = run_benchmark("transformer_lens", lambda: run_model(tl_step))
    timings["transformer_lens"] = {
        "duration": tl_elapsed,
        "tokens_per_second": tokens_processed / tl_elapsed if tl_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    tl_cache_elapsed = profile_model(
        "transformer_lens_cache", lambda: run_model(tl_cache_step), device, traces_path
    )
    timings["transformer_lens_cache"] = {
        "duration": tl_cache_elapsed,
        "tokens_per_second": tokens_processed / tl_cache_elapsed
        if tl_cache_elapsed > 0
        else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_elapsed = profile_model("huggingface", lambda: run_model(hf_step), device, traces_path)
    timings["huggingface"] = {
        "duration": hf_elapsed,
        "tokens_per_second": tokens_processed / hf_elapsed if hf_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_api_elapsed = profile_model(
        "huggingface_api",
        lambda: run_model(hf_api_step),
        device,
        traces_path,
    )
    timings["huggingface_api"] = {
        "duration": hf_api_elapsed,
        "tokens_per_second": tokens_processed / hf_api_elapsed
        if hf_api_elapsed > 0
        else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hooked_elapsed = run_benchmark("huggingface_hooked", lambda: run_model(hf_hooked_step))
    timings["huggingface_hooked"] = {
        "duration": hf_hooked_elapsed,
        "tokens_per_second": tokens_processed / hf_hooked_elapsed
        if hf_hooked_elapsed > 0
        else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hooked_cache_elapsed = run_benchmark(
        "huggingface_hooked_cache", lambda: run_model(hf_hooked_cache_step)
    )
    timings["huggingface_hooked_cache"] = {
        "duration": hf_hooked_cache_elapsed,
        "tokens_per_second": tokens_processed / hf_hooked_cache_elapsed
        if hf_hooked_cache_elapsed > 0
        else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hooked_model.monitoring_engine = monitoring_engine
    if monitoring_engine.async_enabled:
        monitoring_engine.resolve_all()
        if device.type == "cuda":
            torch.cuda.synchronize()
    main_async_elapsed, total_async_elapsed = run_async_benchmark(
        "huggingface_hooked_async_cache",
        lambda: run_model(hf_hooked_async_cache_step),
        monitoring_engine,
    )
    timings["huggingface_hooked_async_cache"] = {
        "main_duration": main_async_elapsed,
        "total_duration": total_async_elapsed,
        "tokens_per_second_main": tokens_processed / main_async_elapsed
        if main_async_elapsed > 0
        else float("inf"),
        "tokens_per_second_total": tokens_processed / total_async_elapsed
        if total_async_elapsed > 0
        else float("inf"),
    }
    hf_hooked_model.monitoring_engine = None

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hook_step, hf_hook_cleanup = setup_hf_hook(
        collect_hidden=args.collect_hidden,
        collect_attention=args.collect_attention,
        move_to_cpu=False,
    )
    try:
        run_model(hf_hook_step)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        hf_hook_elapsed = run_benchmark("huggingface_hook", lambda: run_model(hf_hook_step))
        timings["huggingface_hook"] = {
            "duration": hf_hook_elapsed,
            "tokens_per_second": tokens_processed / hf_hook_elapsed
            if hf_hook_elapsed > 0
            else float("inf"),
        }
    finally:
        hf_hook_cleanup()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hook_cpu_step, hf_hook_cpu_cleanup = setup_hf_hook(
        collect_hidden=args.collect_hidden,
        collect_attention=args.collect_attention,
        move_to_cpu=True,
    )
    try:
        run_model(hf_hook_cpu_step)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        hf_hook_cpu_elapsed = run_benchmark(
            "huggingface_hook_cpu", lambda: run_model(hf_hook_cpu_step)
        )
        timings["huggingface_hook_cpu"] = {
            "duration": hf_hook_cpu_elapsed,
            "tokens_per_second": tokens_processed / hf_hook_cpu_elapsed
            if hf_hook_cpu_elapsed > 0
            else float("inf"),
        }
    finally:
        hf_hook_cpu_cleanup()

    monitoring_engine.close()

    traces_path.mkdir(parents=True, exist_ok=True)
    results_file = traces_path / "timing_results.json"

    results_data = {
        "config": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "steps": args.steps,
            "warmup": args.warmup,
            "device": str(device),
            "dtype": args.dtype,
            "collect_hidden": args.collect_hidden,
            "collect_attention": args.collect_attention,
            "cache_dtype": args.cache_dtype,
            "engine_queue_size": args.engine_queue_size,
            "engine_delay_steps": args.engine_delay_steps,
            "profiling_enabled": not args.no_profile,
        },
        "timings": timings,
        "total_processed_tokens": tokens_processed,
    }

    with results_file.open("w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nTiming results saved to: {results_file.resolve()}")

    print("\nTiming results (seconds and tokens/sec):")
    for label, stats in timings.items():
        if "duration" in stats:
            print(
                f"- {label:>30}: duration={stats['duration']:.4f}s token/s={stats['tokens_per_second']:.2f}"
            )
        else:
            print(
                f"- {label:>30}: main_duration={stats['main_duration']:.4f}s "
                f"total_duration={stats['total_duration']:.4f}s "
                f"main_token/s={stats['tokens_per_second_main']:.2f} "
                f"total_token/s={stats['tokens_per_second_total']:.2f}"
            )

    if not args.no_profile:
        print("\nProfiler traces written under:")
        print(f"  {traces_path.resolve()}")
    if args.nvtx and device.type == "cuda":
        print("NVTX annotations enabled for TransformerLens hooks (set TL_ENABLE_NVTX=1).")


if __name__ == "__main__":
    main()
