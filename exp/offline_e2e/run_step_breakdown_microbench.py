#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import StaticCache

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
for _path in (SCRIPT_DIR, PROJECT_ROOT, PROJECT_ROOT / "transformers" / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from common import (
    DEFAULT_RESULTS_DIR,
    build_tokenizer,
    device_sync,
    ensure_dir,
    make_output_path,
    resolve_model_id,
    write_json,
)
from run_proj_dmi import _NullHostEngine, _build_db_host_cfg, _build_ring_cfg, _load_hooked_model
from run_torch_hooks import TorchHookCollector, _load_model_for_hook_selection


HS_LOGITS_HOOK_SELECTION = "hidden-states,final_ln,logits"


def _make_static_cache(model: Any, *, batch_size: int, max_cache_len: int, device: torch.device) -> StaticCache:
    return StaticCache(
        config=model.config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
    )


def _compile_prefill_step(model: Any, *, output_hidden_states: bool):
    wants_position_ids = _forward_accepts_position_ids(model)

    def _prefill_step(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: StaticCache,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "cache_position": cache_position,
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": output_hidden_states,
            "output_attentions": False,
            "logits_to_keep": 1,
        }
        if wants_position_ids:
            kwargs["position_ids"] = position_ids
        return model(**kwargs)

    return torch.compile(_prefill_step, mode="reduce-overhead", fullgraph=True)


def _compile_decode_step(model: Any, *, output_hidden_states: bool):
    wants_position_ids = _forward_accepts_position_ids(model)

    def _decode_step(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: StaticCache,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "cache_position": cache_position,
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": output_hidden_states,
            "output_attentions": False,
            "logits_to_keep": 1,
        }
        if wants_position_ids:
            kwargs["position_ids"] = position_ids
        return model(**kwargs)

    return torch.compile(_decode_step, mode="reduce-overhead", fullgraph=True)


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def _forward_accepts_position_ids(model: Any) -> bool:
    return "position_ids" in inspect.signature(model.forward).parameters


def _make_synthetic_batch(
    tokenizer: Any,
    *,
    batch_size: int,
    prefill_tokens: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    fill_token_id = int(
        tokenizer.eos_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None
        else tokenizer.pad_token_id
    )
    input_ids = torch.full((batch_size, prefill_tokens), fill_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, prefill_tokens), dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _copy_hf_outputs_to_cpu(outputs: Any) -> None:
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        _ = logits.detach().cpu().shape
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        for hidden in hidden_states:
            if isinstance(hidden, torch.Tensor):
                _ = hidden.detach().cpu().shape


def _hf_forward_kwargs(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any = None,
    use_cache: bool = True,
    output_hidden_states: bool = False,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "return_dict": True,
        "output_hidden_states": output_hidden_states,
        "output_attentions": False,
        "logits_to_keep": 1,
    }
    if _forward_accepts_position_ids(model):
        kwargs["position_ids"] = _position_ids_from_attention_mask(attention_mask)
        if input_ids.shape[1] == 1:
            kwargs["position_ids"] = kwargs["position_ids"][:, -1:].contiguous()
    if past_key_values is None:
        kwargs.pop("past_key_values")
    return kwargs


def _measure_hf_ideal_prefill(
    model: Any,
    encoded: Dict[str, torch.Tensor],
    *,
    compiled_prefill: Any = None,
) -> Tuple[float, float]:
    device_sync(encoded["input_ids"].device)
    t0 = time.perf_counter()
    if compiled_prefill is None:
        outputs = model(**_hf_forward_kwargs(model=model, **encoded, use_cache=True, output_hidden_states=False))
    else:
        cache = _make_static_cache(
            model,
            batch_size=int(encoded["input_ids"].shape[0]),
            max_cache_len=int(encoded["input_ids"].shape[1]) + 4,
            device=encoded["input_ids"].device,
        )
        cache_position = torch.arange(encoded["input_ids"].shape[1], device=encoded["input_ids"].device, dtype=torch.long)
        position_ids = _position_ids_from_attention_mask(encoded["attention_mask"]) if _forward_accepts_position_ids(model) else None
        torch.compiler.cudagraph_mark_step_begin()
        outputs = compiled_prefill(encoded["input_ids"], encoded["attention_mask"], cache, cache_position, position_ids)
    _ = outputs.logits.shape
    device_sync(encoded["input_ids"].device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, (t1 - t0) * 1000.0


def _build_decode_inputs_from_prefill(model: Any, encoded: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    device = encoded["input_ids"].device
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prefill_outputs = model(**_hf_forward_kwargs(model=model, input_ids=input_ids, attention_mask=attention_mask, use_cache=True))
    next_tokens = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_mask = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
        dim=1,
    )
    decode_kwargs = _hf_forward_kwargs(
        model=model,
        input_ids=next_tokens,
        attention_mask=decode_mask,
        past_key_values=prefill_outputs.past_key_values,
        use_cache=True,
        output_hidden_states=False,
    )
    if "cache_position" in inspect.signature(model.forward).parameters:
        decode_kwargs["cache_position"] = torch.tensor([input_ids.shape[1]], device=device, dtype=torch.long)
    return decode_kwargs


def _measure_hf_ideal_decode(
    model: Any,
    encoded: Dict[str, torch.Tensor],
    *,
    compiled_prefill: Any = None,
    compiled_decode: Any = None,
) -> Tuple[float, float]:
    device = encoded["input_ids"].device
    if compiled_prefill is None or compiled_decode is None:
        decode_kwargs = _build_decode_inputs_from_prefill(model, encoded)
        device_sync(device)
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        decode_outputs = model(**decode_kwargs)
    else:
        cache = _make_static_cache(
            model,
            batch_size=int(encoded["input_ids"].shape[0]),
            max_cache_len=int(encoded["input_ids"].shape[1]) + 4,
            device=device,
        )
        prefill_kwargs: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "past_key_values": cache,
            "cache_position": torch.arange(encoded["input_ids"].shape[1], device=device, dtype=torch.long),
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "logits_to_keep": 1,
        }
        if _forward_accepts_position_ids(model):
            prefill_kwargs["position_ids"] = _position_ids_from_attention_mask(encoded["attention_mask"])
        with torch.no_grad():
            prefill_outputs = model(**prefill_kwargs)
        next_tokens = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).clone()
        del prefill_outputs
        decode_mask = torch.cat(
            [encoded["attention_mask"], torch.ones((encoded["attention_mask"].shape[0], 1), device=device, dtype=encoded["attention_mask"].dtype)],
            dim=1,
        )
        decode_cache_position = torch.tensor([encoded["input_ids"].shape[1]], device=device, dtype=torch.long)
        decode_position_ids = _position_ids_from_attention_mask(decode_mask)[:, -1:].contiguous() if _forward_accepts_position_ids(model) else None
        device_sync(device)
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        decode_outputs = compiled_decode(next_tokens, decode_mask, cache, decode_cache_position, decode_position_ids)
    _ = decode_outputs.logits.shape
    device_sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, (t1 - t0) * 1000.0


def _measure_hf_api_prefill(
    model: Any,
    encoded: Dict[str, torch.Tensor],
    *,
    compiled_prefill: Any = None,
) -> Tuple[float, float]:
    device = encoded["input_ids"].device
    device_sync(device)
    t0 = time.perf_counter()
    if compiled_prefill is None:
        outputs = model(**_hf_forward_kwargs(model=model, **encoded, use_cache=True, output_hidden_states=True))
    else:
        cache = _make_static_cache(
            model,
            batch_size=int(encoded["input_ids"].shape[0]),
            max_cache_len=int(encoded["input_ids"].shape[1]) + 4,
            device=device,
        )
        cache_position = torch.arange(encoded["input_ids"].shape[1], device=device, dtype=torch.long)
        position_ids = _position_ids_from_attention_mask(encoded["attention_mask"]) if _forward_accepts_position_ids(model) else None
        torch.compiler.cudagraph_mark_step_begin()
        outputs = compiled_prefill(encoded["input_ids"], encoded["attention_mask"], cache, cache_position, position_ids)
    device_sync(device)
    t_compute = time.perf_counter()
    _copy_hf_outputs_to_cpu(outputs)
    device_sync(device)
    t_end = time.perf_counter()
    return (t_compute - t0) * 1000.0, (t_end - t0) * 1000.0


def _measure_hf_api_decode(
    model: Any,
    encoded: Dict[str, torch.Tensor],
    *,
    compiled_prefill: Any = None,
    compiled_decode: Any = None,
) -> Tuple[float, float]:
    device = encoded["input_ids"].device
    if compiled_prefill is None or compiled_decode is None:
        decode_kwargs = _build_decode_inputs_from_prefill(model, encoded)
        decode_kwargs["output_hidden_states"] = True
        device_sync(device)
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        decode_outputs = model(**decode_kwargs)
    else:
        cache = _make_static_cache(
            model,
            batch_size=int(encoded["input_ids"].shape[0]),
            max_cache_len=int(encoded["input_ids"].shape[1]) + 4,
            device=device,
        )
        prefill_kwargs: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "past_key_values": cache,
            "cache_position": torch.arange(encoded["input_ids"].shape[1], device=device, dtype=torch.long),
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "logits_to_keep": 1,
        }
        if _forward_accepts_position_ids(model):
            prefill_kwargs["position_ids"] = _position_ids_from_attention_mask(encoded["attention_mask"])
        with torch.no_grad():
            prefill_outputs = model(**prefill_kwargs)
        next_tokens = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).clone()
        del prefill_outputs
        decode_mask = torch.cat(
            [encoded["attention_mask"], torch.ones((encoded["attention_mask"].shape[0], 1), device=device, dtype=encoded["attention_mask"].dtype)],
            dim=1,
        )
        decode_cache_position = torch.tensor([encoded["input_ids"].shape[1]], device=device, dtype=torch.long)
        decode_position_ids = _position_ids_from_attention_mask(decode_mask)[:, -1:].contiguous() if _forward_accepts_position_ids(model) else None
        device_sync(device)
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        decode_outputs = compiled_decode(next_tokens, decode_mask, cache, decode_cache_position, decode_position_ids)
    device_sync(device)
    t_compute = time.perf_counter()
    _copy_hf_outputs_to_cpu(decode_outputs)
    device_sync(device)
    t_end = time.perf_counter()
    return (t_compute - t0) * 1000.0, (t_end - t0) * 1000.0


def _measure_torch_hooks_prefill(model: Any, collector: TorchHookCollector, encoded: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    device = encoded["input_ids"].device
    device_sync(device)
    t0 = time.perf_counter()
    collector.begin()
    outputs = model(**_hf_forward_kwargs(model=model, **encoded, use_cache=True, output_hidden_states=False))
    collector.end()
    _ = outputs.logits.shape
    device_sync(device)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    return total_ms, total_ms


def _measure_torch_hooks_decode(model: Any, collector: TorchHookCollector, encoded: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    device = encoded["input_ids"].device
    decode_kwargs = _build_decode_inputs_from_prefill(model, encoded)
    device_sync(device)
    t0 = time.perf_counter()
    collector.begin()
    decode_outputs = model(**decode_kwargs)
    collector.end()
    _ = decode_outputs.logits.shape
    device_sync(device)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    return total_ms, total_ms


def _create_proj_dmi_engine_for_model(
    *,
    model_id: str,
    model: Any,
    proj_dmi_mode: str,
    hook_selection: str,
    args: argparse.Namespace,
) -> Any:
    from monitoring import AdvanceConfig, MonitoringConfig, MonitoringEngine, NativePartialSealConfig
    from monitoring.config import CaptureSchedule, HookSelection
    from monitoring.generate import _install_monitoring_forward, _uninstall_monitoring_forward
    from monitoring import ring_transport

    mon_cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
        native_partial_seal=NativePartialSealConfig(
            enabled=True,
            chunk_bytes=256 * 1024,
            cap_enabled=True,
            cap_ratio=0.8,
            driver_guard_mb=1024,
        ),
        advance=AdvanceConfig(),
    )

    if proj_dmi_mode == "ring_db":
        engine = MonitoringEngine(
            async_enabled=True,
            config=mon_cfg,
            model_id=model_id,
            db_config=_build_db_host_cfg(args),
        )
    else:
        engine = MonitoringEngine(
            async_enabled=True,
            config=mon_cfg,
            model_id=model_id,
            host_engine=_NullHostEngine(),
        )

    engine.enable_ring_transport(_build_ring_cfg(args))
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    transport = ring_transport.get_active()
    if transport is None:
        raise RuntimeError("ring transport was not activated")
    transport._hook_selection = hook_selection
    _install_monitoring_forward(model)
    model._microbench_uninstall_monitoring_forward = _uninstall_monitoring_forward
    return engine


def _close_proj_dmi_microbench(model: Any, engine: Any) -> None:
    uninstall = getattr(model, "_microbench_uninstall_monitoring_forward", None)
    if uninstall is not None:
        try:
            uninstall(model)
        except Exception:
            pass
    try:
        engine.close()
    except Exception:
        pass
    try:
        model.monitoring_engine = None
    except Exception:
        pass


def _flush_proj_dmi_ring(engine: Any) -> Dict[str, int]:
    ring_engine = getattr(engine, "_ring_engine", None)
    if ring_engine is None:
        return {
            "flush_wait_us": 0,
            "pending_entries": 0,
            "pending_bytes": 0,
            "cpu_payload_head": 0,
            "cpu_payload_tail_committed": 0,
            "total_flushes": 0,
            "last_flush_entries": 0,
            "last_flush_bytes": 0,
            "last_flush_complete_monotonic_us": 0,
            "last_force_flush_wait_us": 0,
        }
    flush_wait_us = int(ring_engine.flush_and_wait())
    stats = ring_engine.get_stats()
    return {
        "flush_wait_us": flush_wait_us,
        "pending_entries": int(stats.pending_entries),
        "pending_bytes": int(stats.pending_bytes),
        "cpu_payload_head": int(stats.cpu_payload_head),
        "cpu_payload_tail_committed": int(stats.cpu_payload_tail_committed),
        "total_flushes": int(stats.total_flushes),
        "last_flush_entries": int(stats.last_flush_entries),
        "last_flush_bytes": int(stats.last_flush_bytes),
        "last_flush_complete_monotonic_us": int(stats.last_flush_complete_monotonic_us),
        "last_force_flush_wait_us": int(stats.last_force_flush_wait_us),
    }


def _set_proj_dmi_hook_enabled(model: Any, enabled: bool) -> None:
    from monitoring import ring_transport

    transport = ring_transport.get_active()
    if transport is None:
        return
    for spec in getattr(transport, "_active_specs", []):
        try:
            spec.module.enabled = enabled
        except Exception:
            pass


def _prepare_ring_forward(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any = None,
    cache_position: Any = None,
    logits_to_keep: int = 1,
) -> None:
    from monitoring import ring_transport

    engine = getattr(model, "monitoring_engine", None)
    transport = ring_transport.get_active()
    if engine is None or transport is None or not getattr(engine, "_using_ring_transport", False):
        return
    if not transport._using_forward_hooks:
        return

    engine._prepare_ring_step(input_ids, attention_mask, past_key_values, cache_position=cache_position)

    batch = int(input_ids.shape[0])
    q_len = int(input_ids.shape[1])
    is_static = past_key_values is not None and hasattr(past_key_values, "max_cache_len")
    kv_dim = ring_transport._get_kv_dim(past_key_values, q_len, is_static=is_static)

    hook_byte_sizes: List[int] = []
    model_cfg = transport._model_cfg
    if model_cfg is not None:
        for spec in transport._active_specs:
            shape = ring_transport._compute_hook_shape(
                spec.hook_type,
                model_cfg,
                batch,
                q_len,
                kv_dim,
                logits_to_keep=logits_to_keep,
            )
            if shape:
                dtype = spec.dtype if spec.dtype is not None else model_cfg.dtype
                elem_size = torch._utils._element_size(dtype)
                nbytes = elem_size
                for dim in shape:
                    nbytes *= dim
                hook_byte_sizes.append(nbytes)
            else:
                hook_byte_sizes.append(0)

    if not transport._force_cpu_direct:
        step_total_bytes = sum(ring_transport.align_up_py(size, 16) for size in hook_byte_sizes)
        n_hooks = len(hook_byte_sizes)
        result = transport._ring_engine.prepare_step(step_total_bytes, n_hooks)
        transport.cpu_direct = (result == 2)

    transport.pre_push_all_metas(batch, q_len, kv_dim, logits_to_keep=logits_to_keep)


def _dmi_prefill_kwargs(model: Any, encoded: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "use_cache": True,
        "return_dict": True,
        "logits_to_keep": 1,
    }
    if _forward_accepts_position_ids(model):
        kwargs["position_ids"] = _position_ids_from_attention_mask(encoded["attention_mask"])
    return kwargs


def _measure_dmi_prefill(
    model: Any,
    model_id: str,
    args: argparse.Namespace,
    encoded: Dict[str, torch.Tensor],
    *,
    engine: Any = None,
    compiled_prefill: Any = None,
) -> Tuple[float, float, Dict[str, int]]:
    device = encoded["input_ids"].device
    kwargs = _dmi_prefill_kwargs(model, encoded)
    owns_engine = engine is None
    if engine is None:
        engine = _create_proj_dmi_engine_for_model(
            model_id=model_id,
            model=model,
            proj_dmi_mode=str(args.proj_dmi_mode),
            hook_selection=str(args.hook_selection),
            args=args,
        )
    device_sync(device)
    t0 = time.perf_counter()
    if compiled_prefill is None:
        _prepare_ring_forward(model=model, input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"], logits_to_keep=1)
        outputs = model(**kwargs)
    else:
        cache = _make_static_cache(
            model,
            batch_size=int(encoded["input_ids"].shape[0]),
            max_cache_len=int(encoded["input_ids"].shape[1]) + 4,
            device=device,
        )
        cache_position = torch.arange(encoded["input_ids"].shape[1], device=device, dtype=torch.long)
        position_ids = kwargs.get("position_ids")
        _prepare_ring_forward(model=model, input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"], past_key_values=cache, cache_position=cache_position, logits_to_keep=1)
        torch.compiler.cudagraph_mark_step_begin()
        outputs = compiled_prefill(kwargs["input_ids"], kwargs["attention_mask"], cache, cache_position, position_ids)
    _ = outputs.logits.shape
    torch.cuda.current_stream().synchronize()
    t_compute = time.perf_counter()
    ring_stats = _flush_proj_dmi_ring(engine)
    if owns_engine:
        _close_proj_dmi_microbench(model, engine)
    compute_ms = (t_compute - t0) * 1000.0
    total_ms = compute_ms + (float(ring_stats["flush_wait_us"]) / 1000.0)
    return compute_ms, total_ms, ring_stats


def _measure_dmi_decode(
    model: Any,
    model_id: str,
    args: argparse.Namespace,
    encoded: Dict[str, torch.Tensor],
    *,
    engine: Any = None,
    compiled_prefill: Any = None,
    compiled_decode: Any = None,
) -> Tuple[float, float, Dict[str, int]]:
    device = encoded["input_ids"].device
    owns_engine = engine is None
    if compiled_prefill is None or compiled_decode is None:
        decode_kwargs = _build_decode_inputs_from_prefill(model, encoded)
        next_tokens = decode_kwargs["input_ids"]
        decode_mask = decode_kwargs["attention_mask"]
        past_key_values = decode_kwargs.get("past_key_values")
        cache_position = decode_kwargs.get("cache_position")
        if engine is None:
            engine = _create_proj_dmi_engine_for_model(
                model_id=model_id,
                model=model,
                proj_dmi_mode=str(args.proj_dmi_mode),
                hook_selection=str(args.hook_selection),
                args=args,
            )
        device_sync(device)
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        _prepare_ring_forward(
            model=model,
            input_ids=next_tokens,
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        decode_outputs = model(**decode_kwargs)
    else:
        cache = _make_static_cache(
            model,
            batch_size=int(encoded["input_ids"].shape[0]),
            max_cache_len=int(encoded["input_ids"].shape[1]) + 4,
            device=device,
        )
        prefill_kwargs = _dmi_prefill_kwargs(model, encoded)
        prefill_kwargs["past_key_values"] = cache
        prefill_kwargs["cache_position"] = torch.arange(encoded["input_ids"].shape[1], device=device, dtype=torch.long)
        if engine is not None:
            _set_proj_dmi_hook_enabled(model, False)
        try:
            with torch.no_grad():
                prefill_outputs = model(**prefill_kwargs)
        finally:
            if engine is not None:
                _set_proj_dmi_hook_enabled(model, True)
        next_tokens = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).clone()
        del prefill_outputs
        decode_mask = torch.cat(
            [encoded["attention_mask"], torch.ones((encoded["attention_mask"].shape[0], 1), device=device, dtype=encoded["attention_mask"].dtype)],
            dim=1,
        )
        decode_cache_position = torch.tensor([encoded["input_ids"].shape[1]], device=device, dtype=torch.long)
        decode_position_ids = _position_ids_from_attention_mask(decode_mask)[:, -1:].contiguous() if _forward_accepts_position_ids(model) else None
        if engine is None:
            engine = _create_proj_dmi_engine_for_model(
                model_id=model_id,
                model=model,
                proj_dmi_mode=str(args.proj_dmi_mode),
                hook_selection=str(args.hook_selection),
                args=args,
            )
        device_sync(device)
        t0 = time.perf_counter()
        _prepare_ring_forward(
            model=model,
            input_ids=next_tokens,
            attention_mask=decode_mask,
            past_key_values=cache,
            cache_position=decode_cache_position,
            logits_to_keep=1,
        )
        torch.compiler.cudagraph_mark_step_begin()
        decode_outputs = compiled_decode(next_tokens, decode_mask, cache, decode_cache_position, decode_position_ids)
    _ = decode_outputs.logits.shape
    torch.cuda.current_stream().synchronize()
    t_compute = time.perf_counter()
    ring_stats = _flush_proj_dmi_ring(engine)
    if owns_engine:
        _close_proj_dmi_microbench(model, engine)
    compute_ms = (t_compute - t0) * 1000.0
    total_ms = compute_ms + (float(ring_stats["flush_wait_us"]) / 1000.0)
    return compute_ms, total_ms, ring_stats


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill/decode-step breakdown microbenchmark.")
    parser.add_argument("--baseline", choices=["hf_ideal", "hf_api", "torch_hooks", "proj_dmi"], required=True)
    parser.add_argument("--baseline-label", default="")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hook-selection", default=HS_LOGITS_HOOK_SELECTION)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--proj-dmi-mode", choices=["ring_null", "ring_db"], default="ring_db")
    parser.add_argument("--ring-task-entries", type=int, default=65536)
    parser.add_argument("--ring-payload-mb", type=int, default=1024)
    parser.add_argument("--ring-pinned-mb", type=int, default=1024)
    parser.add_argument("--drain-poll-timeout-us", type=int, default=100)
    parser.add_argument("--drain-flush-task-ratio", type=float, default=0.0)
    parser.add_argument("--drain-flush-payload-ratio", type=float, default=0.0)
    parser.add_argument("--drain-flush-entry-threshold", type=int, default=0)
    parser.add_argument("--drain-flush-byte-threshold", type=int, default=0)
    parser.add_argument("--drain-flush-timeout-us", type=int, default=0)
    parser.add_argument("--clone-slices", action="store_true")
    parser.add_argument("--ch-parallelism", type=int, default=10)
    parser.add_argument("--ch-queue-max-items", type=int, default=1024)
    parser.add_argument("--ch-queue-max-size-mb", type=int, default=2048)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=9000)
    parser.add_argument("--db-user", default="default")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-database", default="default")
    parser.add_argument("--db-table", default="offload")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_id = resolve_model_id(args.model)
    tokenizer = build_tokenizer(model_id, local_files_only=bool(args.local_files_only))
    device = torch.device("cuda")
    encoded = _make_synthetic_batch(
        tokenizer,
        batch_size=int(args.batch_size),
        prefill_tokens=int(args.prefill_tokens),
        device=device,
    )

    baseline_label = args.baseline_label or args.baseline
    compile_enabled = args.baseline != "torch_hooks" and not bool(args.disable_compile)
    model = None
    collector = None
    compiled_prefill = None
    compiled_decode = None
    if args.baseline == "hf_ideal":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            local_files_only=bool(args.local_files_only),
        ).to(device).eval()
        if compile_enabled:
            compiled_prefill = _compile_prefill_step(model, output_hidden_states=False)
            compiled_decode = _compile_decode_step(model, output_hidden_states=False)
    elif args.baseline == "hf_api":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            local_files_only=bool(args.local_files_only),
        ).to(device).eval()
        if compile_enabled:
            compiled_prefill = _compile_prefill_step(model, output_hidden_states=True)
            compiled_decode = _compile_decode_step(model, output_hidden_states=True)
    elif args.baseline == "torch_hooks":
        model = _load_model_for_hook_selection(
            model_id,
            local_files_only=bool(args.local_files_only),
            hook_selection=str(args.hook_selection),
        ).to(device).eval()
        collector = TorchHookCollector(model, hook_selection=str(args.hook_selection))
    else:
        model = _load_hooked_model(model_id, local_files_only=bool(args.local_files_only))
        model.to(device).eval()

    try:
        prefill_compute_runs: List[float] = []
        prefill_total_runs: List[float] = []
        decode_compute_runs: List[float] = []
        decode_total_runs: List[float] = []
        prefill_ring_stats_runs: List[Dict[str, int]] = []
        decode_ring_stats_runs: List[Dict[str, int]] = []

        if args.baseline != "proj_dmi":
            with torch.no_grad():
                for _ in range(int(args.warmup)):
                    if args.baseline == "hf_ideal":
                        _measure_hf_ideal_prefill(model, encoded, compiled_prefill=compiled_prefill)
                        _measure_hf_ideal_decode(model, encoded, compiled_prefill=compiled_prefill, compiled_decode=compiled_decode)
                    elif args.baseline == "hf_api":
                        _measure_hf_api_prefill(model, encoded, compiled_prefill=compiled_prefill)
                        _measure_hf_api_decode(model, encoded, compiled_prefill=compiled_prefill, compiled_decode=compiled_decode)
                    else:
                        _measure_torch_hooks_prefill(model, collector, encoded)
                        _measure_torch_hooks_decode(model, collector, encoded)
                    device_sync(device)
            print(f"Warmup done ({int(args.warmup)} iterations).", flush=True)

            with torch.no_grad():
                for _ in range(int(args.iters)):
                    if args.baseline == "hf_ideal":
                        prefill_compute_ms, prefill_total_ms = _measure_hf_ideal_prefill(model, encoded, compiled_prefill=compiled_prefill)
                        decode_compute_ms, decode_total_ms = _measure_hf_ideal_decode(model, encoded, compiled_prefill=compiled_prefill, compiled_decode=compiled_decode)
                    elif args.baseline == "hf_api":
                        prefill_compute_ms, prefill_total_ms = _measure_hf_api_prefill(model, encoded, compiled_prefill=compiled_prefill)
                        decode_compute_ms, decode_total_ms = _measure_hf_api_decode(model, encoded, compiled_prefill=compiled_prefill, compiled_decode=compiled_decode)
                    else:
                        prefill_compute_ms, prefill_total_ms = _measure_torch_hooks_prefill(model, collector, encoded)
                        decode_compute_ms, decode_total_ms = _measure_torch_hooks_decode(model, collector, encoded)

                    prefill_compute_runs.append(prefill_compute_ms)
                    prefill_total_runs.append(prefill_total_ms)
                    decode_compute_runs.append(decode_compute_ms)
                    decode_total_runs.append(decode_total_ms)
        else:
            prefill_engine = _create_proj_dmi_engine_for_model(
                model_id=model_id,
                model=model,
                proj_dmi_mode=str(args.proj_dmi_mode),
                hook_selection=str(args.hook_selection),
                args=args,
            )
            try:
                dmi_compiled_prefill = (
                    _compile_prefill_step(model, output_hidden_states=False) if compile_enabled else None
                )
                with torch.no_grad():
                    for _ in range(int(args.warmup)):
                        _measure_dmi_prefill(model, model_id, args, encoded, engine=prefill_engine, compiled_prefill=dmi_compiled_prefill)
                        device_sync(device)
                print(f"Warmup done ({int(args.warmup)} iterations).", flush=True)
                with torch.no_grad():
                    for _ in range(int(args.iters)):
                        prefill_compute_ms, prefill_total_ms, prefill_ring_stats = _measure_dmi_prefill(
                            model, model_id, args, encoded, engine=prefill_engine, compiled_prefill=dmi_compiled_prefill
                        )
                        prefill_compute_runs.append(prefill_compute_ms)
                        prefill_total_runs.append(prefill_total_ms)
                        prefill_ring_stats_runs.append(prefill_ring_stats)
            finally:
                _close_proj_dmi_microbench(model, prefill_engine)

            decode_engine = _create_proj_dmi_engine_for_model(
                model_id=model_id,
                model=model,
                proj_dmi_mode=str(args.proj_dmi_mode),
                hook_selection=str(args.hook_selection),
                args=args,
            )
            try:
                dmi_compiled_decode = (
                    _compile_decode_step(model, output_hidden_states=False) if compile_enabled else None
                )
                with torch.no_grad():
                    for _ in range(int(args.warmup)):
                        _measure_dmi_decode(
                            model,
                            model_id,
                            args,
                            encoded,
                            engine=decode_engine,
                            compiled_prefill=dmi_compiled_prefill if compile_enabled else None,
                            compiled_decode=dmi_compiled_decode,
                        )
                        device_sync(device)
                with torch.no_grad():
                    for _ in range(int(args.iters)):
                        decode_compute_ms, decode_total_ms, decode_ring_stats = _measure_dmi_decode(
                            model,
                            model_id,
                            args,
                            encoded,
                            engine=decode_engine,
                            compiled_prefill=dmi_compiled_prefill if compile_enabled else None,
                            compiled_decode=dmi_compiled_decode,
                        )
                        decode_compute_runs.append(decode_compute_ms)
                        decode_total_runs.append(decode_total_ms)
                        decode_ring_stats_runs.append(decode_ring_stats)
            finally:
                _close_proj_dmi_microbench(model, decode_engine)

        prefill_summary = {
            "compute_ms": _mean(prefill_compute_runs),
            "total_ms": _mean(prefill_total_runs),
        }
        prefill_summary["transfer_tail_ms"] = max(0.0, prefill_summary["total_ms"] - prefill_summary["compute_ms"])

        decode_summary = {
            "compute_ms": _mean(decode_compute_runs),
            "total_ms": _mean(decode_total_runs),
        }
        decode_summary["transfer_tail_ms"] = max(0.0, decode_summary["total_ms"] - decode_summary["compute_ms"])

        payload = {
            "baseline": str(args.baseline),
            "baseline_label": baseline_label,
            "model": str(args.model),
            "model_id": model_id,
            "batch_size": int(args.batch_size),
            "prefill_tokens": int(args.prefill_tokens),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "hook_selection": str(args.hook_selection),
            "repeat_index": int(args.repeat_index),
            "compile_enabled": bool(compile_enabled),
            "measurement_kind": "prefill_decode_step_breakdown",
            "timing_semantics": {
                "compute_ms": "Wall time until the GPU main compute stream is synchronized.",
                "total_ms": (
                    "For proj_dmi, compute_ms plus the C++-measured ring flush wait until pending payloads are "
                    "drained to CPU staging/committed tail; for other baselines, wall time until extraction/copy completes."
                ),
            },
            "prefill": {
                **prefill_summary,
                "compute_runs_ms": [float(v) for v in prefill_compute_runs],
                "total_runs_ms": [float(v) for v in prefill_total_runs],
                "ring_stats_runs": prefill_ring_stats_runs if args.baseline == "proj_dmi" else None,
            },
            "decode_1": decode_summary,
            "decode_1_runs_ms": {
                "compute_runs_ms": [float(v) for v in decode_compute_runs],
                "total_runs_ms": [float(v) for v in decode_total_runs],
                "ring_stats_runs": decode_ring_stats_runs if args.baseline == "proj_dmi" else None,
            },
            "hook_count": (len(collector.hook_names) if collector is not None else None),
            "ring_payload_mb": int(args.ring_payload_mb) if args.baseline == "proj_dmi" else None,
            "ring_pinned_mb": int(args.ring_pinned_mb) if args.baseline == "proj_dmi" else None,
        }

        results_dir = Path(args.results_dir)
        ensure_dir(results_dir)
        output_path = make_output_path(
            results_dir=results_dir,
            baseline=baseline_label,
            model=args.model,
            sample_file=f"synthetic_prefill_{int(args.prefill_tokens)}.jsonl",
            batch_size=int(args.batch_size),
            repeat_index=int(args.repeat_index),
        )
        write_json(output_path, payload)
        print(f"Saved {output_path}")
        print(
            f"[{baseline_label}] bs={int(args.batch_size)} "
            f"prefill compute={prefill_summary['compute_ms']:.3f} total={prefill_summary['total_ms']:.3f} "
            f"decode compute={decode_summary['compute_ms']:.3f} total={decode_summary['total_ms']:.3f}",
            flush=True,
        )
    finally:
        if collector is not None:
            collector.close()


if __name__ == "__main__":
    main()
