#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

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
from run_nnsight import _collect_targets as _nnsight_collect_targets
from run_nnsight import _save_target as _nnsight_save_target
from run_proj_dmi import _NullHostEngine, _build_db_host_cfg, _build_ring_cfg, _load_hooked_model


DEFAULT_HEAVY_HOOKS = "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits"
DEFAULT_LIGHT_HOOKS = "logits"


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def _forward_accepts_position_ids(model: Any) -> bool:
    return "position_ids" in inspect.signature(model.forward).parameters


def _make_synthetic_prefill_batch(
    tokenizer: Any,
    *,
    batch_size: int,
    prefill_tokens: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    pad_token_id = int(tokenizer.pad_token_id)
    fill_token_id = int(
        tokenizer.eos_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None
        else tokenizer.pad_token_id
    )
    input_ids = torch.full((batch_size, prefill_tokens), fill_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, prefill_tokens), dtype=torch.long, device=device)
    if pad_token_id != fill_token_id:
        input_ids[:, 0] = pad_token_id
        attention_mask[:, 0] = 0
        input_ids[:, 0] = fill_token_id
        attention_mask[:, 0] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _base_forward_kwargs(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
        "output_hidden_states": False,
        "output_attentions": False,
        "logits_to_keep": 1,
    }
    if _forward_accepts_position_ids(model):
        kwargs["position_ids"] = _position_ids_from_attention_mask(attention_mask)
    return kwargs


def _run_hf_forward(model: Any, encoded: Dict[str, torch.Tensor]) -> None:
    kwargs = _base_forward_kwargs(
        model=model,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    outputs = model(**kwargs)
    _ = outputs.logits.shape


def _run_torch_hooks_forward(model: Any, collector: Any, encoded: Dict[str, torch.Tensor]) -> None:
    kwargs = _base_forward_kwargs(
        model=model,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    collector.begin()
    outputs = model(**kwargs)
    collector.end()
    _ = outputs.logits.shape


def _run_nnsight_forward(model: Any, targets: list[tuple[str, Any, str]], encoded: Dict[str, torch.Tensor]) -> None:
    captured = None
    with model.trace(encoded, use_cache=False, return_dict=True, logits_to_keep=1):
        captured = tuple(_nnsight_save_target(module, kind) for _name, module, kind in targets)
    for tensor in captured or ():
        _ = tensor.shape


def _prepare_ring_prefill(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> None:
    from monitoring import ring_transport

    engine = getattr(model, "monitoring_engine", None)
    transport = ring_transport.get_active()
    if engine is None or transport is None or not getattr(engine, "_using_ring_transport", False):
        return
    if not transport._using_forward_hooks:
        return

    engine._prepare_ring_step(input_ids, attention_mask, None, cache_position=None)

    batch = int(input_ids.shape[0])
    q_len = int(input_ids.shape[1])
    kv_dim = q_len
    transport.cpu_direct = False

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


def _run_proj_dmi_forward(model: Any, encoded: Dict[str, torch.Tensor]) -> None:
    kwargs = _base_forward_kwargs(
        model=model,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    _prepare_ring_prefill(
        model=model,
        input_ids=kwargs["input_ids"],
        attention_mask=kwargs["attention_mask"],
        logits_to_keep=int(kwargs["logits_to_keep"]),
    )
    outputs = model(**kwargs)
    _ = outputs.logits.shape


def _estimate_dmi_payload_mb(model: Any, hook_selection: str, prefix_len: int, batch_size: int) -> float:
    from monitoring import ring_transport

    transport = ring_transport.get_active()
    if transport is None or transport._model_cfg is None or not transport._active_specs:
        return 0.0
    total_bytes = 0
    kv_dim = int(prefix_len)
    for spec in transport._active_specs:
        shape = ring_transport._compute_hook_shape(
            spec.hook_type,
            transport._model_cfg,
            int(batch_size),
            int(prefix_len),
            kv_dim,
            logits_to_keep=1,
        )
        if not shape:
            continue
        dtype = spec.dtype if spec.dtype is not None else transport._model_cfg.dtype
        nbytes = torch._utils._element_size(dtype)
        for dim in shape:
            nbytes *= dim
        total_bytes += ring_transport.align_up_py(nbytes, 16)
    return float(total_bytes) / (1024.0 * 1024.0)


def _setup_proj_dmi(
    *,
    model_id: str,
    local_files_only: bool,
    proj_dmi_mode: str,
    hook_selection: str,
    args: argparse.Namespace,
) -> tuple[Any, Any]:
    from monitoring import AdvanceConfig, MonitoringConfig, MonitoringEngine, NativePartialSealConfig
    from monitoring.config import CaptureSchedule, HookSelection
    from monitoring.generate import _install_monitoring_forward, _uninstall_monitoring_forward
    from monitoring import ring_transport

    mon_cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=False),
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
    model = _load_hooked_model(model_id, local_files_only=local_files_only)
    model.to(torch.device("cuda")).eval()
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    transport = ring_transport.get_active()
    if transport is None:
        raise RuntimeError("ring transport was not activated")
    transport._hook_selection = hook_selection
    _install_monitoring_forward(model)
    model._prefill_uninstall_monitoring_forward = _uninstall_monitoring_forward
    return model, engine


def _close_proj_dmi(model: Any, engine: Any) -> None:
    uninstall = getattr(model, "_prefill_uninstall_monitoring_forward", None)
    if uninstall is not None:
        try:
            uninstall(model)
        except Exception:
            pass
    try:
        engine.close()
    except Exception:
        pass


def _load_torch_hook_model(model_id: str, *, local_files_only: bool, hook_selection: str):
    from run_torch_hooks import _load_model_for_hook_selection

    model = _load_model_for_hook_selection(
        model_id,
        local_files_only=local_files_only,
        hook_selection=hook_selection,
    )
    model.to(torch.device("cuda")).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill-only backpressure experiment runner.")
    parser.add_argument("--baseline", choices=["hf_native", "torch_hooks", "nnsight", "proj_dmi"], required=True)
    parser.add_argument("--baseline-label", default="")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--num-microbatches", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hook-selection", default=DEFAULT_HEAVY_HOOKS)
    parser.add_argument("--proj-dmi-mode", choices=["ring_null", "ring_db"], default="ring_db")
    parser.add_argument("--ring-task-entries", type=int, default=65536)
    parser.add_argument("--ring-payload-mb", type=int, default=16384)
    parser.add_argument("--ring-pinned-mb", type=int, default=16384)
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
    if int(args.prefill_tokens) <= 0:
        raise ValueError("prefill-tokens must be > 0")
    if int(args.num_microbatches) <= 0:
        raise ValueError("num-microbatches must be > 0")
    tokenizer = build_tokenizer(model_id, local_files_only=bool(args.local_files_only))

    baseline_label = args.baseline_label or args.baseline
    device = torch.device("cuda")
    model = None
    engine = None
    collector = None
    nnsight_targets = None
    nnsight_target_names = None
    extra: Dict[str, Any] = {}

    if args.baseline == "hf_native":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            local_files_only=bool(args.local_files_only),
        )
        model.to(device).eval()
        extra["hook_set"] = "none"
    elif args.baseline == "torch_hooks":
        model = _load_torch_hook_model(
            model_id,
            local_files_only=bool(args.local_files_only),
            hook_selection=str(args.hook_selection),
        )
        from run_torch_hooks import TorchHookCollector

        collector = TorchHookCollector(model, hook_selection=str(args.hook_selection))
        extra["hook_set"] = str(args.hook_selection)
        extra["hook_count"] = len(collector.hook_names)
        extra["hook_names"] = collector.hook_names
    elif args.baseline == "nnsight":
        from nnsight import LanguageModel

        model = LanguageModel(
            model_id,
            tokenizer=tokenizer,
            device_map="cuda:0",
            dispatch=True,
            torch_dtype="float16",
            attn_implementation="eager",
            local_files_only=bool(args.local_files_only),
        )
        nnsight_targets, nnsight_target_names = _nnsight_collect_targets(model, str(args.hook_selection))
        extra["hook_set"] = str(args.hook_selection)
        extra["hook_count"] = len(nnsight_target_names)
        extra["hook_names"] = nnsight_target_names
    elif args.baseline == "proj_dmi":
        model, engine = _setup_proj_dmi(
            model_id=model_id,
            local_files_only=bool(args.local_files_only),
            proj_dmi_mode=str(args.proj_dmi_mode),
            hook_selection=str(args.hook_selection),
            args=args,
        )
        extra["hook_set"] = str(args.hook_selection)
        extra["proj_dmi_mode"] = str(args.proj_dmi_mode)
        extra["ring_payload_mb"] = int(args.ring_payload_mb)
        extra["ring_pinned_mb"] = int(args.ring_pinned_mb)
        extra["ring_task_entries"] = int(args.ring_task_entries)
    else:
        raise ValueError(f"unsupported baseline: {args.baseline}")

    per_prefix: List[Dict[str, Any]] = []
    batch_records: List[Dict[str, Any]] = []

    try:
        warmup_encoded = _make_synthetic_prefill_batch(
            tokenizer,
            batch_size=int(args.batch_size),
            prefill_tokens=int(args.prefill_tokens),
            device=device,
        )
        with torch.no_grad():
            for _ in range(2):
                if args.baseline == "hf_native":
                    _run_hf_forward(model, warmup_encoded)
                elif args.baseline == "torch_hooks":
                    _run_torch_hooks_forward(model, collector, warmup_encoded)
                elif args.baseline == "nnsight":
                    _run_nnsight_forward(model, nnsight_targets, warmup_encoded)
                else:
                    _run_proj_dmi_forward(model, warmup_encoded)
                device_sync(device)
        print("Warmup done (2 fixed microbatches).", flush=True)

        with torch.no_grad():
            encoded = _make_synthetic_prefill_batch(
                tokenizer,
                batch_size=int(args.batch_size),
                prefill_tokens=int(args.prefill_tokens),
                device=device,
            )
            input_tokens = int(encoded["attention_mask"].sum().item())
            padded_tokens = int(encoded["input_ids"].numel())
            batch_times_ms: List[float] = []
            for microbatch_index in range(int(args.num_microbatches)):
                device_sync(device)
                t0 = time.perf_counter()
                if args.baseline == "hf_native":
                    _run_hf_forward(model, encoded)
                elif args.baseline == "torch_hooks":
                    _run_torch_hooks_forward(model, collector, encoded)
                elif args.baseline == "nnsight":
                    _run_nnsight_forward(model, nnsight_targets, encoded)
                else:
                    _run_proj_dmi_forward(model, encoded)
                device_sync(device)
                t1 = time.perf_counter()

                wall_ms = (t1 - t0) * 1000.0
                batch_times_ms.append(wall_ms)
                batch_records.append(
                    {
                        "microbatch_index": int(microbatch_index),
                        "batch_size": int(args.batch_size),
                        "input_tokens": int(input_tokens),
                        "padded_tokens": int(padded_tokens),
                        "wall_time_ms": float(wall_ms),
                    }
                )

            estimate_mb = None
            if args.baseline == "proj_dmi":
                estimate_mb = _estimate_dmi_payload_mb(
                    model, str(args.hook_selection), int(args.prefill_tokens), int(args.batch_size)
                )

            per_prefix.append(
                {
                    "prefill_tokens": int(args.prefill_tokens),
                    "num_microbatches": int(args.num_microbatches),
                    "mean_wall_time_ms": float(sum(batch_times_ms) / len(batch_times_ms)),
                    "batch_wall_time_ms": [float(value) for value in batch_times_ms],
                    "mean_input_tokens": float(input_tokens),
                    "mean_padded_tokens": float(padded_tokens),
                    "estimated_dmi_step_mb": estimate_mb,
                }
            )
            print(
                f"[{baseline_label}] prefill={int(args.prefill_tokens):>3d} "
                f"microbatches={int(args.num_microbatches):>2d} "
                f"mean_ms={per_prefix[-1]['mean_wall_time_ms']:.3f}",
                flush=True,
            )

        payload = {
            "baseline": str(args.baseline),
            "baseline_label": baseline_label,
            "model": str(args.model),
            "model_id": model_id,
            "repeat_index": int(args.repeat_index),
            "batch_size": int(args.batch_size),
            "dataset_size": int(args.batch_size * int(args.num_microbatches)),
            "measurement_kind": "fixed_prefill_microbatch_latency",
            "decode_tokens": 0,
            "prefill_tokens": int(args.prefill_tokens),
            "num_microbatches": int(args.num_microbatches),
            "local_files_only": bool(args.local_files_only),
            "per_prefix": per_prefix,
            "batch_records": batch_records,
            **extra,
        }

        out_path = make_output_path(
            results_dir=args.results_dir,
            baseline=baseline_label,
            model=args.model,
            sample_file=f"synthetic_prefill_{int(args.prefill_tokens)}",
            batch_size=args.batch_size,
            repeat_index=args.repeat_index,
        )
        write_json(out_path, payload)
        print(f"Saved {out_path}")
    finally:
        if collector is not None:
            collector.close()
        if engine is not None and model is not None:
            _close_proj_dmi(model, engine)


if __name__ == "__main__":
    main()
