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

from transformer_lens import HookedTransformer
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

from monitoring import MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection, MonitoringConfig

MINIMAL_MONITORING_HOOKS = [
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_mlp_out",
]


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
    parser.add_argument(
        "--monitoring-bypass",
        action="store_true",
        help="Keep MonitoringEngine hooks active but disable all capture via schedule gating",
    )
    parser.add_argument(
        "--hook-selection",
        choices=["full", "attention", "mlp", "minimal"],
        default="full",
        help="Select which hooks MonitoringEngine enables (minimal keeps just a handful for overhead profiling)",
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


def measure_model(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
) -> float:
    """Measure wallclock time without profiling overhead."""
    import time

    if device.type == "cuda":
        torch.cuda.synchronize()

    nvtx.range_push(f"measure_{label}")
    start = time.perf_counter()
    fn()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    nvtx.range_pop()
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
            fn()

    return time.perf_counter() - wall_time_start


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

    nvtx.range_push(f"measure_async_{label}")
    start = time.perf_counter()

    nvtx.range_push(f"async_compute_{label}")
    fn()
    nvtx.range_pop()  # async_compute

    # Only sync the main compute stream, NOT the background cache stream
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
        except AttributeError:  # pragma: no cover
            pass

    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = tensorboard_trace_handler(str(trace_dir / label))

    import time

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
        # Sync only main stream before exiting profiler context
        if device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    main_elapsed = time.perf_counter() - wall_time_start
    nvtx.range_push(f"async_resolve_all_{label}")
    engine.resolve_all()
    nvtx.range_pop()  # async_resolve_all
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - wall_time_start
    nvtx.range_pop()  # profile_async
    return main_elapsed, total_elapsed


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

    hf_hooked_model = HookedGPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=hf_dtype,
    )
    hf_hooked_model.to(device)
    hf_hooked_model.eval()

    cache_dtype = None if args.cache_dtype == "none" else map_hf_dtype(args.cache_dtype)
    if args.hook_selection == "minimal":
        hook_selection = HookSelection(mode="custom", include=MINIMAL_MONITORING_HOOKS)
    else:
        hook_selection = HookSelection(mode=args.hook_selection)

    if args.monitoring_bypass:
        schedule = CaptureSchedule(
            capture_prefill=False,
            capture_decode=False,
            request_stride=10**12,
        )
    else:
        schedule = CaptureSchedule()

    monitoring_config = MonitoringConfig(
        hooks=hook_selection,
        schedule=schedule,
    )

    monitoring_engine = MonitoringEngine(
        async_enabled=device.type == "cuda",
        cache_dtype=cache_dtype,
        queue_size=args.engine_queue_size,
        delay_steps=args.engine_delay_steps,
        config=monitoring_config,
    )
    engine_attached_permanently = bool(args.monitoring_bypass)
    original_monitoring_engine = hf_hooked_model.monitoring_engine
    hf_hooked_model.monitoring_engine = monitoring_engine
    engine_init_ms = 0.0
    try:
        engine_init_ms = monitoring_engine.prepare_for_model(hf_hooked_model)
    except Exception:
        engine_init_ms = 0.0
    finally:
        if not engine_attached_permanently:
            hf_hooked_model.monitoring_engine = original_monitoring_engine

    if args.monitoring_bypass:
        native_backend = getattr(monitoring_engine, "_native_backend", None)
        native_using = bool(getattr(monitoring_engine, "_using_native_backend", False))
        native_builder_enabled = bool(getattr(monitoring_engine, "_native_builder_enabled", False))
        native_callback_enabled = bool(getattr(monitoring_engine, "_native_callback_enabled", False))
        print(
            "[Inline Debug] engine native_backend="
            f"{'yes' if native_backend is not None else 'no'} "
            f"using={native_using} builder={native_builder_enabled} callback={native_callback_enabled}"
        )
        inline_enabled = bool(getattr(hf_hooked_model, "_inline_monitoring_enabled", False))
        sample_cfg = None
        sample_name = None
        for hp_name, hp in hf_hooked_model.hook_dict.items():
            sample_cfg = getattr(hp, "_monitor_handle", None)
            if sample_cfg is not None:
                sample_name = hp_name
                break
        print(
            "[Inline Debug] enabled="
            f"{inline_enabled} sample_hook={sample_name if sample_cfg is not None else 'none'} "
            f"ticket={'yes' if sample_cfg is not None else 'no'}"
        )
        if not inline_enabled or sample_cfg is None:
            raise RuntimeError("Inline monitoring not enabled; cannot run bypass-inline profile.")

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
        try:
            if monitoring_engine.async_enabled:
                monitoring_engine.clear_completed_results()
        except Exception:
            pass
        return logits, cache

    lm_head = hf_model.lm_head

    def project_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        return lm_head(hidden_states)

    def hf_prefill_fn(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
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

    def hf_decode_fn(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
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

    def hf_api_prefill(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        return hf_prefill_fn(prefill_tokens)

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

    def hf_modified_prefill(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
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

    def hf_modified_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
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

    def hook_names_filter(name: str) -> bool:
        lname = name.lower()
        if args.collect_hidden and args.collect_attention:
            return True
        if args.collect_attention:
            return "attn" in lname
        return "attn" not in lname

    def hf_modified_hook_prefill(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        previous_engine = hf_hooked_model.monitoring_engine
        if not engine_attached_permanently:
            hf_hooked_model.monitoring_engine = None
        outputs, cache_dict = hf_hooked_model.run_with_cache(
            prefill_tokens,
            use_cache=True,
            output_hidden_states=args.collect_hidden,
            output_attentions=args.collect_attention,
            return_dict=True,
            names_filter=hook_names_filter,
            return_cache_object=False,
            remove_batch_dim=False,
        )
        hidden_states = outputs.last_hidden_state
        logits = project_logits(hidden_states)
        next_token = greedy_from_logits(logits)
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        past = outputs.past_key_values
        cache_dict.clear()
        try:
            if monitoring_engine.async_enabled:
                monitoring_engine.clear_completed_results()
        except Exception:
            pass
        if not engine_attached_permanently:
            hf_hooked_model.monitoring_engine = previous_engine
        del hidden_states, logits, outputs
        return past, next_token

    def hf_modified_hook_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        previous_engine = hf_hooked_model.monitoring_engine
        if not engine_attached_permanently:
            hf_hooked_model.monitoring_engine = None
        outputs, cache_dict = hf_hooked_model.run_with_cache(
            token,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=args.collect_hidden,
            output_attentions=args.collect_attention,
            return_dict=True,
            names_filter=hook_names_filter,
            return_cache_object=False,
            remove_batch_dim=False,
        )
        hidden_states = outputs.last_hidden_state
        logits = project_logits(hidden_states)
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        next_past = outputs.past_key_values
        cache_dict.clear()
        try:
            if monitoring_engine.async_enabled:
                monitoring_engine.clear_completed_results()
        except Exception:
            pass
        if not engine_attached_permanently:
            hf_hooked_model.monitoring_engine = previous_engine
        del hidden_states, outputs
        return logits, next_past

    def hf_modified_hook_async_prefill(prefill_tokens: torch.Tensor = prompt_tokens) -> Tuple[Tuple, torch.Tensor]:
        nvtx.range_push("async_prefill")
        monitoring_engine.start_step()
        try:
            nvtx.range_push("async_prefill_forward")
            outputs, cache_dict = hf_hooked_model.run_with_cache(
                prefill_tokens,
                use_cache=True,
                output_hidden_states=args.collect_hidden,
                output_attentions=args.collect_attention,
                return_dict=True,
            )
            nvtx.range_pop()  # async_prefill_forward
        finally:
            nvtx.range_push("async_prefill_end_step")
            monitoring_engine.end_step()
            nvtx.range_pop()  # async_prefill_end_step
        nvtx.range_push("async_prefill_post")
        hidden_states = outputs.last_hidden_state
        logits = project_logits(hidden_states)
        next_token = greedy_from_logits(logits)
        nvtx.range_pop()  # async_prefill_post
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        past = outputs.past_key_values
        cache_dict.clear()
        try:
            if monitoring_engine.async_enabled:
                monitoring_engine.clear_completed_results()
        except Exception:
            pass
        del hidden_states, logits, outputs
        nvtx.range_pop()  # async_prefill
        return past, next_token

    def hf_modified_hook_async_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        nvtx.range_push("async_decode")
        nvtx.range_push("async_decode_start_step")
        monitoring_engine.start_step()
        nvtx.range_pop()  # async_decode_start_step
        try:
            nvtx.range_push("async_decode_forward")
            outputs, cache_dict = hf_hooked_model.run_with_cache(
                token,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=args.collect_hidden,
                output_attentions=args.collect_attention,
                return_dict=True,
            )
            nvtx.range_pop()  # async_decode_forward
        finally:
            nvtx.range_push("async_decode_end_step")
            monitoring_engine.end_step()
            nvtx.range_pop()  # async_decode_end_step
        nvtx.range_push("async_decode_post")
        hidden_states = outputs.last_hidden_state
        logits = project_logits(hidden_states)
        if args.collect_attention and outputs.attentions is not None:
            for attn in outputs.attentions:
                _ = attn
        if args.collect_hidden and outputs.hidden_states is not None:
            for hs in outputs.hidden_states:
                _ = hs
        next_past = outputs.past_key_values
        cache_dict.clear()
        try:
            if monitoring_engine.async_enabled:
                monitoring_engine.clear_completed_results()
        except Exception:
            pass
        del hidden_states, outputs
        nvtx.range_pop()  # async_decode_post
        nvtx.range_pop()  # async_decode
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

    timings: Dict[str, Dict[str, float]] = {}
    total_decoded_tokens = args.decode_steps * args.steps * args.batch_size

    # Choose measurement function based on --no-profile flag
    if args.no_profile:
        def run_benchmark(label: str, fn: Callable[[], None]) -> float:
            return measure_model(label, fn, device)

        def run_async_benchmark(label: str, fn: Callable[[], None], engine) -> Tuple[float, float]:
            return measure_async_model(label, fn, device, engine)

        print("Running benchmarks WITHOUT profiling (pure wallclock time measurement)")
    else:
        def run_benchmark(label: str, fn: Callable[[], None]) -> float:
            return profile_model(label, fn, device, traces_path)

        def run_async_benchmark(label: str, fn: Callable[[], None], engine) -> Tuple[float, float]:
            return profile_async_model(label, fn, device, traces_path, engine)

        print("Running benchmarks WITH profiling (trace files will be generated)")

    warmup(tl_prefill, tl_decode)
    tl_elapsed = run_benchmark("transformer_lens", lambda: run_decode(tl_prefill, tl_decode))
    timings["transformer_lens"] = {
        "duration": tl_elapsed,
        "tokens_per_second": total_decoded_tokens / tl_elapsed if tl_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(tl_cache_prefill, tl_cache_decode)
    tl_cache_elapsed = run_benchmark(
        "transformer_lens_cache", lambda: run_decode(tl_cache_prefill, tl_cache_decode)
    )
    timings["transformer_lens_cache"] = {
        "duration": tl_cache_elapsed,
        "tokens_per_second": total_decoded_tokens / tl_cache_elapsed if tl_cache_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(hf_prefill_fn, hf_decode_fn)
    hf_elapsed = run_benchmark("huggingface", lambda: run_decode(hf_prefill_fn, hf_decode_fn))
    timings["huggingface"] = {
        "duration": hf_elapsed,
        "tokens_per_second": total_decoded_tokens / hf_elapsed if hf_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(hf_api_prefill, hf_api_decode)
    hf_api_elapsed = run_benchmark(
        "huggingface_api", lambda: run_decode(hf_api_prefill, hf_api_decode)
    )
    timings["huggingface_api"] = {
        "duration": hf_api_elapsed,
        "tokens_per_second": total_decoded_tokens / hf_api_elapsed if hf_api_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not args.monitoring_bypass:
        warmup(hf_modified_prefill, hf_modified_decode)
        hf_modified_elapsed = run_benchmark(
            "hf_modified",
            lambda: run_decode(hf_modified_prefill, hf_modified_decode),
        )
        timings["hf_modified"] = {
            "duration": hf_modified_elapsed,
            "tokens_per_second": total_decoded_tokens / hf_modified_elapsed
            if hf_modified_elapsed > 0
            else float("inf"),
        }

        if device.type == "cuda":
            torch.cuda.empty_cache()

        warmup(hf_modified_hook_prefill, hf_modified_hook_decode)
        hf_modified_hook_elapsed = run_benchmark(
            "hf_modified_hook",
            lambda: run_decode(hf_modified_hook_prefill, hf_modified_hook_decode),
        )
        timings["hf_modified_hook"] = {
            "duration": hf_modified_hook_elapsed,
            "tokens_per_second": total_decoded_tokens / hf_modified_hook_elapsed
            if hf_modified_hook_elapsed > 0
            else float("inf"),
        }

        if device.type == "cuda":
            torch.cuda.empty_cache()

    hf_hooked_model.monitoring_engine = monitoring_engine
    warmup(hf_modified_hook_async_prefill, hf_modified_hook_async_decode)
    if monitoring_engine.async_enabled:
        monitoring_engine.resolve_all()
        if device.type == "cuda":
            torch.cuda.synchronize()
    main_async_decode_elapsed, total_async_decode_elapsed = run_async_benchmark(
        "hf_modified_hook_async",
        lambda: run_decode(hf_modified_hook_async_prefill, hf_modified_hook_async_decode),
        monitoring_engine,
    )
    timings["hf_modified_hook_async"] = {
        "main_duration": main_async_decode_elapsed,
        "total_duration": total_async_decode_elapsed,
        "init_ms": engine_init_ms,
        "tokens_per_second_main": total_decoded_tokens / main_async_decode_elapsed
        if main_async_decode_elapsed > 0
        else float("inf"),
        "tokens_per_second_total": total_decoded_tokens / total_async_decode_elapsed
        if total_async_decode_elapsed > 0
        else float("inf"),
    }
    if not engine_attached_permanently:
        hf_hooked_model.monitoring_engine = None

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
        hf_hook_elapsed = run_benchmark(
            "huggingface_hook", lambda: run_decode(hf_hook_prefill, hf_hook_decode)
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
        hf_hook_cpu_elapsed = run_benchmark(
            "huggingface_hook_cpu",
            lambda: run_decode(hf_hook_cpu_prefill, hf_hook_cpu_decode),
        )
        timings["huggingface_hook_cpu"] = {
            "duration": hf_hook_cpu_elapsed,
            "tokens_per_second": total_decoded_tokens / hf_hook_cpu_elapsed if hf_hook_cpu_elapsed > 0 else float("inf"),
        }
    finally:
        hf_hook_cpu_cleanup()

    monitoring_engine.close()

    # Save timing results to JSON file
    import json
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
            "collect_hidden": args.collect_hidden,
            "collect_attention": args.collect_attention,
            "cache_dtype": args.cache_dtype,
            "engine_queue_size": args.engine_queue_size,
            "engine_delay_steps": args.engine_delay_steps,
            "profiling_enabled": not args.no_profile,
        },
        "timings": timings,
        "total_decoded_tokens": total_decoded_tokens,
    }

    with results_file.open("w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nTiming results saved to: {results_file.resolve()}")

    print("\nTiming results (decode duration per run):")
    for label, stats in timings.items():
        if "duration" in stats:
            print(
                f"- {label:>18}: duration={stats['duration']:.4f}s "
                f"tokens/s={stats['tokens_per_second']:.2f}"
            )
        else:
            print(
                f"- {label:>18}: main_duration={stats['main_duration']:.4f}s "
                f"total_duration={stats['total_duration']:.4f}s"
                f"{' init_ms=' + str(round(stats['init_ms'], 2)) if 'init_ms' in stats else ''}"
                f" main_token/s={stats['tokens_per_second_main']:.2f}"
                f" total_token/s={stats['tokens_per_second_total']:.2f}"
            )

    if not args.no_profile:
        print("\nProfiler traces written under:")
        print(f"  {traces_path.resolve()}")
    if args.nvtx and device.type == "cuda":
        print("NVTX annotations enabled for TransformerLens decode hooks (set TL_ENABLE_NVTX=1).")

    # If engine stats are enabled, also print hook-side stats even for sync baselines.
    try:
        import os as _os
        if bool(int(_os.environ.get("MON_ENGINE_STATS", "0"))):
            from transformers.models.gpt2_p.hook_points import get_monitoring_hook_stats  # type: ignore
            _hook_stats = get_monitoring_hook_stats()
            if _hook_stats:
                print("[Hook/Stats]", _hook_stats)
    except Exception:
        pass


if __name__ == "__main__":
    main()
