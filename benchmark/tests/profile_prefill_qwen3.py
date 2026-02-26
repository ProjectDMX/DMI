"""Prefill profiler benchmark for Hugging Face Qwen3 baselines."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler
from torch.utils.hooks import RemovableHandle

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3Model

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

from monitoring import MonitoringEngine
from monitoring.config import MonitoringConfig

MODEL_NAME = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Qwen3 prefill across Hugging Face baselines"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the run")
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=1,
        help="Number of prompt tokens used during the prefill phase",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of profiled prefill iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up prefill iterations")
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
        help="Enable NVTX annotations inside hook points (enables MonitoringConfig.debug)",
    )
    parser.add_argument(
        "--collect-hidden",
        action="store_true",
        help="Capture hidden states where supported",
    )
    parser.add_argument(
        "--collect-attention",
        action="store_true",
        help="Capture attention tensors where supported",
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
    if args.prefill_tokens < 1:
        parser.error("--prefill-tokens must be >= 1")
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


def run_prefill_loop(
    prefill_fn: Callable[[], Tuple[object, torch.Tensor]],
) -> None:
    prefill_fn()


def setup_hf_decode_hook(
    hf_model,
    collect_hidden: bool,
    collect_attention: bool,
    move_to_cpu: bool = False,
):
    model = getattr(hf_model, "model", None)
    blocks: Optional[Iterable[torch.nn.Module]] = getattr(model, "layers", None) if model else None
    if not blocks:
        raise RuntimeError("Unexpected Qwen3 architecture; transformer blocks not found.")

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
        attn_module = block.self_attn

        if collect_attention:
            original_forward = attn_module.forward

            def wrapped_forward(*f_args, _orig=original_forward, _idx=idx, **f_kwargs):
                outputs = _orig(*f_args, **f_kwargs)
                if not collector_enabled:
                    return outputs

                if not isinstance(outputs, tuple) or len(outputs) != 2:
                    raise RuntimeError("Unexpected Qwen3 attention output structure during hook capture.")

                attn_output, attn_probs = outputs
                attn_output_cache[_idx] = store_tensor(attn_output)
                attn_cache[_idx] = store_tensor(attn_probs)
                return outputs

            attn_module.forward = wrapped_forward  # type: ignore[assignment]
            patched_attn.append((attn_module, original_forward))

            def q_norm_hook(
                module: torch.nn.Module,
                module_input: Tuple[torch.Tensor, ...],
                module_output: torch.Tensor,
                _idx=idx,
            ) -> None:
                if collector_enabled:
                    q_cache[_idx] = store_tensor(module_output.permute(0, 2, 1, 3).contiguous())

            def k_norm_hook(
                module: torch.nn.Module,
                module_input: Tuple[torch.Tensor, ...],
                module_output: torch.Tensor,
                _idx=idx,
            ) -> None:
                if collector_enabled:
                    k_cache[_idx] = store_tensor(module_output.permute(0, 2, 1, 3).contiguous())

            def v_proj_hook(
                module: torch.nn.Module,
                module_input: Tuple[torch.Tensor, ...],
                module_output: torch.Tensor,
                _idx=idx,
                _attn=attn_module,
            ) -> None:
                if not collector_enabled:
                    return
                batch, seq_len, _ = module_output.size()
                head_dim = _attn.head_dim
                num_kv_heads = _attn.config.num_key_value_heads
                v_cache[_idx] = store_tensor(
                    module_output.view(batch, seq_len, num_kv_heads, head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

            extra_hooks.append(attn_module.q_norm.register_forward_hook(q_norm_hook))
            extra_hooks.append(attn_module.k_norm.register_forward_hook(k_norm_hook))
            extra_hooks.append(attn_module.v_proj.register_forward_hook(v_proj_hook))

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
            extra_hooks.append(block.input_layernorm.register_forward_hook(ln1_hook))
            extra_hooks.append(block.post_attention_layernorm.register_forward_hook(ln2_hook))
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

    device = pick_device(args.device)
    hf_dtype = map_hf_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = build_inputs(args.batch_size, args.prefill_tokens, tokenizer, device)

    print(
        f"Using device: {device} | dtype: {args.dtype}"
        f" | prefill_tokens={args.prefill_tokens}"
        f" | collect_hidden={args.collect_hidden} | collect_attention={args.collect_attention}"
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        torch_dtype=hf_dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    lm_head = hf_model.lm_head
    lm_head_state = {
        name: param.detach().cpu()
        for name, param in lm_head.state_dict().items()
    }
    lm_head_dtype = next(lm_head.parameters()).dtype
    lm_head_in_features = lm_head.in_features
    lm_head_out_features = lm_head.out_features
    lm_head_has_bias = lm_head.bias is not None

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
        hf_hooked_model.monitoring_engine = previous_engine
        del hidden_states, logits, outputs
        return past, next_token

    def hf_modified_hook_decode(token: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        previous_engine = hf_hooked_model.monitoring_engine
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

    def run_prefill(prefill_fn, prefill_tokens=prompt_tokens):
        with torch.no_grad():
            for i in range(args.steps):
                nvtx.range_push(f"benchmark_iter_{i}")
                run_prefill_loop(lambda: prefill_fn(prefill_tokens))
                nvtx.range_pop()  # benchmark_iter_i

    def warmup(prefill_fn, prefill_tokens=prompt_tokens):
        if args.warmup <= 0:
            return
        nvtx.range_push("warmup")
        with torch.no_grad():
            for i in range(args.warmup):
                nvtx.range_push(f"warmup_iter_{i}")
                run_prefill_loop(lambda: prefill_fn(prefill_tokens))
                nvtx.range_pop()  # warmup_iter_i
        nvtx.range_pop()  # warmup

    traces_path = Path(args.profile_dir)

    timings: Dict[str, Dict[str, float]] = {}
    total_prefill_tokens = args.prefill_tokens * args.steps * args.batch_size

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

    warmup(hf_prefill_fn)
    hf_elapsed = run_benchmark("huggingface", lambda: run_prefill(hf_prefill_fn))
    timings["huggingface"] = {
        "duration": hf_elapsed,
        "tokens_per_second": total_prefill_tokens / hf_elapsed if hf_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(hf_api_prefill)
    hf_api_elapsed = run_benchmark(
        "huggingface_api", lambda: run_prefill(hf_api_prefill)
    )
    timings["huggingface_api"] = {
        "duration": hf_api_elapsed,
        "tokens_per_second": total_prefill_tokens / hf_api_elapsed if hf_api_elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hook_prefill, hf_hook_decode, hf_hook_cleanup = setup_hf_decode_hook(
        hf_model,
        collect_hidden=args.collect_hidden,
        collect_attention=args.collect_attention,
        move_to_cpu=False,
    )
    warmup(hf_hook_prefill)
    try:
        hf_hook_elapsed = run_benchmark(
            "huggingface_hook", lambda: run_prefill(hf_hook_prefill)
        )
        timings["huggingface_hook"] = {
            "duration": hf_hook_elapsed,
            "tokens_per_second": total_prefill_tokens / hf_hook_elapsed if hf_hook_elapsed > 0 else float("inf"),
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
    warmup(hf_hook_cpu_prefill)
    try:
        hf_hook_cpu_elapsed = run_benchmark(
            "huggingface_hook_cpu",
            lambda: run_prefill(hf_hook_cpu_prefill),
        )
        timings["huggingface_hook_cpu"] = {
            "duration": hf_hook_cpu_elapsed,
            "tokens_per_second": total_prefill_tokens / hf_hook_cpu_elapsed if hf_hook_cpu_elapsed > 0 else float("inf"),
        }
    finally:
        hf_hook_cpu_cleanup()

    lm_head = None
    hf_model.to("cpu")
    del hf_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hooked_model = HookedQwen3Model.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        torch_dtype=hf_dtype,
    )
    hf_hooked_model.to(device)
    hf_hooked_model.eval()
    lm_head = torch.nn.Linear(
        lm_head_in_features,
        lm_head_out_features,
        bias=lm_head_has_bias,
        dtype=lm_head_dtype,
    )
    lm_head.load_state_dict(lm_head_state)
    lm_head.to(device=device, dtype=lm_head_dtype)
    lm_head.eval()

    cache_dtype = None if args.cache_dtype == "none" else map_hf_dtype(args.cache_dtype)
    monitoring_engine = MonitoringEngine(
        async_enabled=device.type == "cuda",
        cache_dtype=cache_dtype,
        queue_size=args.engine_queue_size,
        delay_steps=args.engine_delay_steps,
        config=MonitoringConfig(debug=args.nvtx),
    )
    hf_hooked_model.monitoring_engine = None
    engine_init_ms = 0.0
    try:
        engine_init_ms = monitoring_engine.prepare_for_model(hf_hooked_model)
    except Exception:
        engine_init_ms = 0.0

    warmup(hf_modified_prefill)
    hf_modified_elapsed = run_benchmark(
        "hf_modified",
        lambda: run_prefill(hf_modified_prefill),
    )
    timings["hf_modified"] = {
        "duration": hf_modified_elapsed,
        "tokens_per_second": total_prefill_tokens / hf_modified_elapsed
        if hf_modified_elapsed > 0
        else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    warmup(hf_modified_hook_prefill)
    hf_modified_hook_elapsed = run_benchmark(
        "hf_modified_hook",
        lambda: run_prefill(hf_modified_hook_prefill),
    )
    timings["hf_modified_hook"] = {
        "duration": hf_modified_hook_elapsed,
        "tokens_per_second": total_prefill_tokens / hf_modified_hook_elapsed
        if hf_modified_hook_elapsed > 0
        else float("inf"),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_hooked_model.monitoring_engine = monitoring_engine
    warmup(hf_modified_hook_async_prefill)
    if monitoring_engine.async_enabled:
        monitoring_engine.resolve_all()
        if device.type == "cuda":
            torch.cuda.synchronize()
    main_async_decode_elapsed, total_async_decode_elapsed = run_async_benchmark(
        "hf_modified_hook_async",
        lambda: run_prefill(hf_modified_hook_async_prefill),
        monitoring_engine,
    )
    timings["hf_modified_hook_async"] = {
        "main_duration": main_async_decode_elapsed,
        "total_duration": total_async_decode_elapsed,
        "init_ms": engine_init_ms,
        "tokens_per_second_main": total_prefill_tokens / main_async_decode_elapsed
        if main_async_decode_elapsed > 0
        else float("inf"),
        "tokens_per_second_total": total_prefill_tokens / total_async_decode_elapsed
        if total_async_decode_elapsed > 0
        else float("inf"),
    }
    hf_hooked_model.monitoring_engine = None

    if device.type == "cuda":
        torch.cuda.empty_cache()

    monitoring_engine.close()

    # Save timing results to JSON file
    import json
    results_file = traces_path / "timing_results.json"
    traces_path.mkdir(parents=True, exist_ok=True)

    results_data = {
        "config": {
            "batch_size": args.batch_size,
            "prefill_tokens": args.prefill_tokens,
            "decode_steps": 0,
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
        "total_prefill_tokens": total_prefill_tokens,
    }

    with results_file.open("w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nTiming results saved to: {results_file.resolve()}")

    print("\nTiming results (prefill duration per run):")
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
        print("NVTX annotations enabled for hook point prefill paths (MonitoringConfig.debug=True).")

    try:
        if args.nvtx:
            from monitoring.hook_points import get_monitoring_hook_stats
            _hook_stats = get_monitoring_hook_stats()
            if _hook_stats:
                print("[Hook/Stats]", _hook_stats)
    except Exception:
        pass


if __name__ == "__main__":
    main()
