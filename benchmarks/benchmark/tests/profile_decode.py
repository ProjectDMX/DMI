"""Prefill + decode profiler benchmark comparing TransformerLens vs Hugging Face GPT-2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler
from torch.utils.hooks import RemovableHandle

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile GPT-2 prefill + decode across TransformerLens and Hugging Face baselines"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the run")
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=1,
        help="Number of prompt tokens used during the prefill phase",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=64,
        help="Number of greedy decode steps to execute",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of profiled decode iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up decode iterations")
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
        help="Capture decoder hidden states where supported",
    )
    parser.add_argument(
        "--collect-attention",
        action="store_true",
        help="Capture decoder attention tensors where supported",
    )

    args = parser.parse_args()
    if not args.collect_hidden and not args.collect_attention:
        parser.error("At least one of --collect-hidden or --collect-attention must be provided.")
    if args.prefill_tokens < 1:
        parser.error("--prefill-tokens must be >= 1")
    if args.decode_steps < 1:
        parser.error("--decode-steps must be >= 1")
    return args


def pick_device(device_arg: Optional[str]) -> torch.device:
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
    ):
        with record_function(label):
            if device.type == "cuda":
                torch.cuda.synchronize()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()

    return time.perf_counter() - wall_time_start


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


def setup_hf_decode_hook(
    hf_model,
    collect_hidden: bool,
    collect_attention: bool,
    move_to_cpu: bool = False,
):
    transformer = getattr(hf_model, "transformer", None)
    blocks: Optional[Iterable[torch.nn.Module]] = getattr(transformer, "h", None) if transformer else None
    if not blocks:
        raise RuntimeError("Unexpected GPT-2 architecture; transformer blocks not found.")

    num_layers = len(blocks)
    attn_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    q_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    k_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    v_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    attn_output_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    resid_pre_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    resid_post_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    ln1_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    ln2_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    mlp_in_cache: List[Optional[torch.Tensor]] = [None] * num_layers
    mlp_out_cache: List[Optional[torch.Tensor]] = [None] * num_layers

    collector_enabled = False

    def store_tensor(tensor: torch.Tensor) -> torch.Tensor:
        stored = tensor.detach()
        if move_to_cpu:
            stored = stored.cpu()
        return stored

    patched_attn: List[Tuple[torch.nn.Module, Callable[..., Tuple]]] = []
    extra_hooks: List[RemovableHandle] = []

    for idx, block in enumerate(blocks):
        attn_module = block.attn

        if collect_attention:
            original_forward = attn_module.forward

            def wrapped_forward(*f_args, _orig=original_forward, _idx=idx, **f_kwargs):
                if collector_enabled:
                    f_kwargs["output_attentions"] = True
                outputs = _orig(*f_args, **f_kwargs)
                if not collector_enabled:
                    return outputs

                if not isinstance(outputs, tuple) or len(outputs) != 2:
                    raise RuntimeError("Unexpected GPT-2 attention output structure during hook capture.")

                attn_output, attn_probs = outputs
                attn_output_cache[_idx] = store_tensor(attn_output)
                attn_cache[_idx] = store_tensor(attn_probs)
                return outputs

            attn_module.forward = wrapped_forward  # type: ignore[assignment]
            patched_attn.append((attn_module, original_forward))

            def c_attn_hook(
                module: torch.nn.Module,
                module_input: Tuple[torch.Tensor, ...],
                module_output: torch.Tensor,
                _idx=idx,
                _attn=attn_module,
            ) -> None:
                if not collector_enabled:
                    return
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
                if collector_enabled:
                    resid_pre_cache[_idx] = store_tensor(module_inputs[0])

            def block_post_hook(
                module: torch.nn.Module,
                module_inputs: Tuple[torch.Tensor, ...],
                module_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                _idx=idx,
            ) -> None:
                if not collector_enabled:
                    return
                hidden = module_output[0] if isinstance(module_output, tuple) else module_output
                resid_post_cache[_idx] = store_tensor(hidden)

            def ln1_hook(
                module: torch.nn.Module,
                module_inputs: Tuple[torch.Tensor, ...],
                module_output: torch.Tensor,
                _idx=idx,
            ) -> None:
                if collector_enabled:
                    ln1_cache[_idx] = store_tensor(module_output)

            def ln2_hook(
                module: torch.nn.Module,
                module_inputs: Tuple[torch.Tensor, ...],
                module_output: torch.Tensor,
                _idx=idx,
            ) -> None:
                if collector_enabled:
                    ln2_cache[_idx] = store_tensor(module_output)

            def mlp_pre_hook(
                module: torch.nn.Module,
                module_inputs: Tuple[torch.Tensor, ...],
                _idx=idx,
            ) -> None:
                if collector_enabled:
                    mlp_in_cache[_idx] = store_tensor(module_inputs[0])

            def mlp_post_hook(
                module: torch.nn.Module,
                module_inputs: Tuple[torch.Tensor, ...],
                module_output: torch.Tensor,
                _idx=idx,
            ) -> None:
                if collector_enabled:
                    mlp_out_cache[_idx] = store_tensor(module_output)

            extra_hooks.append(block.register_forward_pre_hook(block_pre_hook))
            extra_hooks.append(block.register_forward_hook(block_post_hook))
            extra_hooks.append(block.ln_1.register_forward_hook(ln1_hook))
            extra_hooks.append(block.ln_2.register_forward_hook(ln2_hook))
            extra_hooks.append(block.mlp.register_forward_pre_hook(mlp_pre_hook))
            extra_hooks.append(block.mlp.register_forward_hook(mlp_post_hook))

    def reset_attention_cache() -> None:
        for idx in range(num_layers):
            attn_cache[idx] = None
            q_cache[idx] = None
            k_cache[idx] = None
            v_cache[idx] = None
            attn_output_cache[idx] = None

    def reset_hidden_cache() -> None:
        for idx in range(num_layers):
            resid_pre_cache[idx] = None
            resid_post_cache[idx] = None
            ln1_cache[idx] = None
            ln2_cache[idx] = None
            mlp_in_cache[idx] = None
            mlp_out_cache[idx] = None

    def cleanup() -> None:
        for module, original in patched_attn:
            module.forward = original  # type: ignore[assignment]
        for handle in extra_hooks:
            handle.remove()

    def prefill(prefill_tokens: torch.Tensor) -> Tuple[Tuple, torch.Tensor]:
        nonlocal collector_enabled
        reset_attention_cache()
        reset_hidden_cache()
        collector_enabled = False
        outputs = hf_model(
            prefill_tokens,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        next_token = greedy_from_logits(outputs.logits)
        past_key_values = outputs.past_key_values
        del outputs
        reset_attention_cache()
        reset_hidden_cache()
        return past_key_values, next_token

    def decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        nonlocal collector_enabled
        collector_enabled = True
        reset_attention_cache()
        reset_hidden_cache()
        outputs = hf_model(
            token,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=collect_hidden,
            output_attentions=collect_attention,
            return_dict=True,
        )
        collector_enabled = False

        if collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs

        logits = outputs.logits
        next_past = outputs.past_key_values
        del outputs
        reset_attention_cache()
        reset_hidden_cache()
        return logits, next_past

    return prefill, decode, cleanup


def main() -> None:
    args = parse_args()

    if args.nvtx:
        os.environ.setdefault("TL_ENABLE_NVTX", "1")

    device = pick_device(args.device)
    hf_dtype = map_hf_dtype(args.dtype)
    tl_dtype = map_tl_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = build_inputs(args.batch_size, args.prefill_tokens, tokenizer, device)

    print(
        f"Using device: {device} | dtype: {args.dtype}"
        f" | prefill_tokens={args.prefill_tokens} | decode_steps={args.decode_steps}"
        f" | collect_hidden={args.collect_hidden} | collect_attention={args.collect_attention}"
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=hf_dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    tl_model = HookedTransformer.from_pretrained(
        "gpt2",
        device=device,
        dtype=tl_dtype,
    )
    tl_model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    def tl_prefill(prefill_tensor: torch.Tensor = prompt_tokens) -> Tuple[HookedTransformerKeyValueCache, torch.Tensor]:
        cache = HookedTransformerKeyValueCache.init_cache(
            tl_model.cfg, device=device, batch_size=prefill_tensor.size(0)
        )
        logits = tl_model(prefill_tensor, return_type="logits", past_kv_cache=cache)
        token = greedy_from_logits(logits)
        del logits
        return cache, token

    def tl_decode(token: torch.Tensor, cache: HookedTransformerKeyValueCache) -> Tuple[torch.Tensor, HookedTransformerKeyValueCache]:
        logits = tl_model(token, return_type="logits", past_kv_cache=cache)
        return logits, cache

    def tl_cache_prefill(prefill_tensor: torch.Tensor = prompt_tokens) -> Tuple[HookedTransformerKeyValueCache, torch.Tensor]:
        cache = HookedTransformerKeyValueCache.init_cache(
            tl_model.cfg, device=device, batch_size=prefill_tensor.size(0)
        )
        logits = tl_model(prefill_tensor, return_type="logits", past_kv_cache=cache)
        token = greedy_from_logits(logits)
        del logits
        return cache, token

    def tl_cache_decode(token: torch.Tensor, cache: HookedTransformerKeyValueCache) -> Tuple[torch.Tensor, HookedTransformerKeyValueCache]:
        def names_filter(name: str) -> bool:
            lname = name.lower()
            if args.collect_hidden and args.collect_attention:
                return True
            if args.collect_attention:
                return "attn" in lname
            return "attn" not in lname

        logits, cache_dict = tl_model.run_with_cache(
            token,
            return_cache_object=False,
            remove_batch_dim=False,
            past_kv_cache=cache,
            names_filter=names_filter,
        )
        cache_dict.clear()
        return logits, cache

    def hf_prefill() -> Tuple[Tuple, torch.Tensor]:
        outputs = hf_model(
            prompt_tokens,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        token = greedy_from_logits(outputs.logits)
        past = outputs.past_key_values
        del outputs
        return past, token

    def hf_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
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

    hf_prefill_fn = hf_prefill
    hf_decode_fn = hf_decode

    def hf_prefill_wrapper(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        return hf_prefill()

    def hf_decode_wrapper(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        return hf_decode(token, past_key_values)

    hf_prefill_fn = hf_prefill_wrapper
    hf_decode_fn = hf_decode_wrapper

    def hf_api_prefill(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        return hf_prefill()

    def hf_api_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        outputs = hf_model(
            token,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=args.collect_hidden,
            output_attentions=args.collect_attention,
            return_dict=True,
        )
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        logits = outputs.logits
        next_past = outputs.past_key_values
        del outputs
        return logits, next_past

    def run_decode(prefill_fn, decode_fn, prefill_tokens=prompt_tokens):
        with torch.no_grad():
            for _ in range(args.steps):
                run_decode_loop(lambda: prefill_fn(prefill_tokens), decode_fn, args.decode_steps)

    def warmup(prefill_fn, decode_fn, prefill_tokens=prompt_tokens):
        if args.warmup <= 0:
            return
        with torch.no_grad():
            for _ in range(args.warmup):
                run_decode_loop(lambda: prefill_fn(prefill_tokens), decode_fn, args.decode_steps)

    traces_path = Path(args.profile_dir)

    timings: Dict[str, Dict[str, float]] = {}
    total_decoded_tokens = args.decode_steps * args.steps * args.batch_size

    warmup(tl_prefill, tl_decode)
    tl_elapsed = profile_model("transformer_lens", lambda: run_decode(tl_prefill, tl_decode), device, traces_path)
    timings["transformer_lens"] = {
        "duration": tl_elapsed,
        "tokens_per_second": total_decoded_tokens / tl_elapsed if tl_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(tl_cache_prefill, tl_cache_decode)
    tl_cache_elapsed = profile_model(
        "transformer_lens_cache", lambda: run_decode(tl_cache_prefill, tl_cache_decode), device, traces_path
    )
    timings["transformer_lens_cache"] = {
        "duration": tl_cache_elapsed,
        "tokens_per_second": total_decoded_tokens / tl_cache_elapsed if tl_cache_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(hf_prefill_fn, hf_decode_fn)
    hf_elapsed = profile_model("huggingface", lambda: run_decode(hf_prefill_fn, hf_decode_fn), device, traces_path)
    timings["huggingface"] = {
        "duration": hf_elapsed,
        "tokens_per_second": total_decoded_tokens / hf_elapsed if hf_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(hf_api_prefill, hf_api_decode)
    hf_api_elapsed = profile_model(
        "huggingface_api", lambda: run_decode(hf_api_prefill, hf_api_decode), device, traces_path
    )
    timings["huggingface_api"] = {
        "duration": hf_api_elapsed,
        "tokens_per_second": total_decoded_tokens / hf_api_elapsed if hf_api_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hook_prefill, hf_hook_decode, hf_hook_cleanup = setup_hf_decode_hook(
        hf_model,
        collect_hidden=args.collect_hidden,
        collect_attention=args.collect_attention,
        move_to_cpu=False,
    )
    warmup(hf_hook_prefill, hf_hook_decode)
    try:
        hf_hook_elapsed = profile_model(
            "huggingface_hook", lambda: run_decode(hf_hook_prefill, hf_hook_decode), device, traces_path
        )
        timings["huggingface_hook"] = {
            "duration": hf_hook_elapsed,
            "tokens_per_second": total_decoded_tokens / hf_hook_elapsed if hf_hook_elapsed > 0 else float("inf"),
        }
    finally:
        hf_hook_cleanup()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hook_cpu_prefill, hf_hook_cpu_decode, hf_hook_cpu_cleanup = setup_hf_decode_hook(
        hf_model,
        collect_hidden=args.collect_hidden,
        collect_attention=args.collect_attention,
        move_to_cpu=True,
    )
    warmup(hf_hook_cpu_prefill, hf_hook_cpu_decode)
    try:
        hf_hook_cpu_elapsed = profile_model(
            "huggingface_hook_cpu",
            lambda: run_decode(hf_hook_cpu_prefill, hf_hook_cpu_decode),
            device,
            traces_path,
        )
        timings["huggingface_hook_cpu"] = {
            "duration": hf_hook_cpu_elapsed,
            "tokens_per_second": total_decoded_tokens / hf_hook_cpu_elapsed if hf_hook_cpu_elapsed > 0 else float("inf"),
        }
    finally:
        hf_hook_cpu_cleanup()

    print("\nTiming results (decode duration per run):")
    for label, stats in timings.items():
        print(
            f"- {label:>18}: duration={stats['duration']:.4f}s "
            f"tokens/s={stats['tokens_per_second']:.2f}"
        )

    print("\nProfiler traces written under:")
    print(f"  {traces_path.resolve()}")
    if args.nvtx and device.type == "cuda":
        print("NVTX annotations enabled for TransformerLens decode hooks (set TL_ENABLE_NVTX=1).")


if __name__ == "__main__":
    main()
