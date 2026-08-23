"""Hugging Face adapter implementation.

The adapter translates Hugging Face batch state into DMI's framework-neutral
step protocol. Monitored generation entry points live in :mod:`.generation`.
"""
from __future__ import annotations

import functools
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..base import BackendAdapter
from ..types import StepContext
from ...hooks.specs import (
    HookSpec,
    ModelShapeConfig,
    align_up_py,
    compute_hook_shape,
)
from ...transport.ring import _get_kv_dim
from .model_shape import _make_model_shape_from_hf_config


# ---------------------------------------------------------------------------
# Prepare-step profiling helpers
# ---------------------------------------------------------------------------

# Module-level profiling list for the prepare-step wrapper.  Enabled by
# RING_PROFILE_PREPARE=1.  Lives here next to the wrapper that fills it.
_prepare_profile_times: List[dict] = []


def print_prepare_profile() -> None:
    """Print summary of prepare-step profiling data."""
    if not _prepare_profile_times:
        print("[prepare_profile] No data collected. Set RING_PROFILE_PREPARE=1.")
        return
    keys = ["orig_prepare", "ring_step", "shape_compute", "prepare_step", "push_metas", "total"]
    n = len(_prepare_profile_times)
    print(f"[prepare_profile] {n} steps:")
    for k in keys:
        vals = [d.get(k, 0.0) for d in _prepare_profile_times if k in d]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        mx = max(vals)
        print(f"  {k:20s}: avg={avg:.3f} ms  max={mx:.3f} ms  ({len(vals)} samples)")
    results = [d.get("result") for d in _prepare_profile_times if "result" in d]
    if results:
        from collections import Counter
        dist = Counter(results)
        labels = {0: "RING_OK", 1: "RING_FLUSHED", 2: "CPU_DIRECT", -1: "FORCE_CPU_DIRECT"}
        parts = [f"{labels.get(k, str(k))}={v}" for k, v in sorted(dist.items())]
        print(f"  {'prepare_step results':20s}: {', '.join(parts)}")


# Model-shape helpers live in ``dmi.adapters.huggingface.model_shape``. The
# former ``_make_model_shape(model)`` wrapper was deleted after its only caller
# moved to the adapter-level helper.


# ---------------------------------------------------------------------------
# HuggingFaceAdapter
# ---------------------------------------------------------------------------

class HuggingFaceAdapter(BackendAdapter):
    """``BackendAdapter`` for HuggingFace ``transformers`` models.

    Owns per-call batch state that used to live on ``MonitoringEngine``:
      * ``_batch_request_ids``: list[str] auto-generated as
        ``f"{group_id}:{i}"`` per active request.
      * ``_batch_starts``: per-request next token offset.
      * ``_batch_finished``: per-request post-EOS latch (mirrors HF's
        ``unfinished_sequences``).  Once True, decode emits zero-length
        token ranges for that request unless ``no_strip_right_pad=True``.
      * ``_prefill_kv_offsets``: per-request left-pad in the kv dim,
        computed once per prefill from the 2D/4D attention mask.
      * ``_orig_prepare``: original ``prepare_inputs_for_generation``,
        saved during ``attach_model`` so ``detach_model`` can restore it.

    The group-id counter is engine-scoped (``engine.next_auto_group_id``)
    and is bumped on every batch reset so back-to-back generate() calls
    or mid-call batch shrinks each get a unique group prefix.
    """

    def __init__(
        self,
        engine: Any,
        model_id: str,
        *,
        no_strip_left_pad: bool = False,
        no_strip_right_pad: bool = False,
        eos_token_id: Any = None,
    ) -> None:
        super().__init__(engine, model_id)
        self._debug_step: bool = bool(os.environ.get("RING_DEBUG_STEP"))
        self._batch_request_ids: Optional[List[str]] = None
        self._batch_starts: Optional[List[int]] = None
        self._batch_finished: Optional[List[bool]] = None
        self._prefill_kv_offsets: Optional[List[int]] = None
        self._orig_prepare: Any = None
        self.request_ids_in_this_generate: List[str] = []
        self.token_ranges_in_this_generate: Dict[str, List[Tuple[int, int]]] = {}
        # Per-instance defaults.  Per-call ``attach_model(...)`` overrides
        # only when the kwarg is explicitly passed (None means inherit).
        self._no_strip_left_pad: bool = bool(no_strip_left_pad)
        self._no_strip_right_pad: bool = bool(no_strip_right_pad)
        # User-supplied eos_token_id (or None for auto-detect at attach
        # time).  Stored verbatim; resolution into ``_eos_token_ids``
        # happens in ``attach_model`` so we can chain to the model's
        # generation_config / config when neither constructor nor
        # attach_model call passes an explicit value.
        self._eos_token_id_arg: Any = eos_token_id
        self._eos_token_ids: frozenset = frozenset()

    # --- abstract overrides ---------------------------------------------
    def detect_model_shape(self, model: Any) -> ModelShapeConfig:
        dtype = getattr(model, "dtype", None)
        cfg = _make_model_shape_from_hf_config(model.config, dtype=dtype)
        if cfg is None:
            raise RuntimeError(
                "HuggingFaceAdapter.detect_model_shape: model.config is missing "
                "hidden_size/num_attention_heads (or n_embd/n_head)."
            )
        # HF DMI is TP-only for now: the whole torch.distributed world
        # is the TP group. DP/PP not handled on the HF path yet.
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            cfg.tp_size = max(1, dist.get_world_size())
        return cfg

    def detect_parallel_ranks(self) -> Tuple[int, int, int, int]:
        """torch.distributed-based TP detection.  Works with both
        tp_plan='auto' and torchrun-launched manual TP."""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            return (dist.get_rank(), 0, 0, 0)
        return (0, 0, 0, 0)

    def is_pp_first(self) -> bool:
        return True  # HF generate() runs single-rank w.r.t. pipeline parallelism

    def is_pp_last(self) -> bool:
        return True

    def on_capacity_exceeded(self, ctx: StepContext) -> None:
        # No-op.  transport.force_eager is owned by adaptor_base
        # before_forward.  Kept as a framework hook for HF-specific
        # reactions to overflow (none today).
        return

    # --- override of base no-op stub ------------------------------------
    def _warn_once_capacity(
        self, ctx: StepContext, total_bytes: int, n_hooks: int
    ) -> None:
        shape_key = (ctx.batch, ctx.q_len)
        if shape_key in self._warned_shapes:
            return
        self._warned_shapes.add(shape_key)
        re = self.ring_engine
        if re is None:
            return
        pcap = re.payload_cap()
        scap = re.staging_cap()
        if total_bytes > pcap and total_bytes > scap:
            reason = (
                f"exceeds both GPU ring ({pcap / 1e6:.0f} MB) "
                f"and pinned staging ({scap / 1e6:.0f} MB)"
            )
        elif total_bytes > pcap:
            reason = f"exceeds GPU ring ({pcap / 1e6:.0f} MB)"
        else:
            reason = f"exceeds pinned staging ({scap / 1e6:.0f} MB)"
        warnings.warn(
            f"[ring_transport] Step data ({total_bytes / 1e6:.1f} MB) {reason}. "
            f"Falling back to synced eager CPU offload for all {n_hooks} hooks.",
            stacklevel=2,
        )

    # --- eos resolution -------------------------------------------------
    @staticmethod
    def _normalize_eos(value: Any) -> "frozenset[int]":
        """Coerce ``int`` / ``list[int]`` / ``torch.Tensor`` / ``None`` into
        ``frozenset[int]``.  ``None`` -> empty frozenset."""
        if value is None:
            return frozenset()
        if isinstance(value, torch.Tensor):
            return frozenset(int(t) for t in value.flatten().tolist())
        if isinstance(value, int):
            return frozenset({value})
        # Assume iterable (list, tuple, set, ...)
        return frozenset(int(t) for t in value)

    def _resolve_eos_token_ids(
        self, model: Any, attach_arg: Any
    ) -> "frozenset[int]":
        """Resolution chain for the post-EOS strip's eos token set.  Runs
        each time ``attach_model`` is called.

        Priority order:
          1. ``attach_arg`` (per-call kwarg) if non-None.
          2. ``self._eos_token_id_arg`` (constructor kwarg) if non-None.
          3. ``model.generation_config.eos_token_id`` if present and non-None.
          4. ``model.config.eos_token_id`` if present and non-None.
          5. Empty frozenset (silent fallback; strip never latches).
        """
        if attach_arg is not None:
            return self._normalize_eos(attach_arg)
        if self._eos_token_id_arg is not None:
            return self._normalize_eos(self._eos_token_id_arg)
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            v = getattr(gen_cfg, "eos_token_id", None)
            if v is not None:
                return self._normalize_eos(v)
        cfg = getattr(model, "config", None)
        if cfg is not None:
            v = getattr(cfg, "eos_token_id", None)
            if v is not None:
                return self._normalize_eos(v)
        return frozenset()

    # --- attach / detach ------------------------------------------------
    def attach_model(
        self, model: Any, hook_selection: str = "full",
        install_prepare_wrapper: bool = True,
        no_strip_left_pad: Optional[bool] = None,
        no_strip_right_pad: Optional[bool] = None,
        eos_token_id: Any = None,
    ) -> None:
        """Resolve shape, install ring hooks, and (optionally) wrap
        ``prepare_inputs_for_generation`` so each forward pass triggers
        ``before_forward(model_inputs)``.

        ``install_prepare_wrapper=False`` is used by
        ``generate_greedy_with_monitoring`` which calls
        ``before_forward_manual`` directly per step.

        ``no_strip_left_pad`` / ``no_strip_right_pad``: when not ``None``,
        override the per-instance default set in ``__init__``.  ``None``
        (default) inherits the constructor value.  Per-call wins only
        when explicitly passed.

        ``eos_token_id``: per-call override (highest priority).  When
        ``None``, falls back to the constructor's ``eos_token_id``; if
        that is also ``None``, auto-detects from
        ``model.generation_config.eos_token_id`` then
        ``model.config.eos_token_id``; if neither is set, the resolved
        set is empty and the post-EOS strip never latches.  Accepts
        ``int``, ``list[int]``, or ``torch.Tensor``; normalised to
        ``frozenset[int]``.
        """
        if no_strip_left_pad is not None:
            self._no_strip_left_pad = bool(no_strip_left_pad)
        if no_strip_right_pad is not None:
            self._no_strip_right_pad = bool(no_strip_right_pad)
        self._eos_token_ids = self._resolve_eos_token_ids(model, eos_token_id)
        super().attach_model(model, hook_selection)

        # Startup validation: warn if pinned staging < GPU ring.
        try:
            re = self.transport._ring_engine
            pcap = re.payload_cap()
            scap = re.staging_cap()
            if scap < pcap:
                warnings.warn(
                    f"[ring_transport] Pinned staging ({scap / 1e6:.0f} MB) "
                    f"< GPU ring ({pcap / 1e6:.0f} MB). "
                    f"Effective capacity is staging-limited. "
                    f"Recommend --ring-pinned-mb >= --ring-payload-mb.",
                    stacklevel=2,
                )
        except Exception:
            pass

        if os.environ.get("RING_DEBUG_SPECS"):
            print(
                f"[ring] HuggingFaceAdapter.attach_model: active={len(self.active_specs)} "
                f"model_cfg={self.model_cfg is not None}"
            )

        if install_prepare_wrapper:
            orig_prepare = getattr(model, "prepare_inputs_for_generation", None)
            if orig_prepare is None:
                return
            if getattr(model, "_monitoring_orig_prepare", None) is not None:
                # already wrapped; leave as is
                return
            self._orig_prepare = orig_prepare
            adaptor_self = self
            _profile = os.environ.get("RING_PROFILE_PREPARE", "") == "1"

            @functools.wraps(orig_prepare)
            def _prepare_wrapper(*args: Any, **kwargs: Any) -> Any:
                if adaptor_self.transport is None or adaptor_self.transport.null_offload:
                    return orig_prepare(*args, **kwargs)
                if _profile:
                    _t0 = time.perf_counter()
                model_inputs = orig_prepare(*args, **kwargs)
                if _profile:
                    _t_orig = time.perf_counter()
                # Drive the per-step protocol through the adapter.  The
                # base driver picks up RING_PROFILE_PREPARE and emits
                # detailed timing entries when enabled (best-effort: if
                # the driver throws we still return the inputs).
                try:
                    adaptor_self.before_forward(model_inputs)
                except Exception:
                    pass
                if _profile:
                    _t_end = time.perf_counter()
                    _prepare_profile_times.append({
                        "orig_prepare": (_t_orig - _t0) * 1000,
                        "total": (_t_end - _t0) * 1000,
                    })
                return model_inputs

            model._monitoring_orig_prepare = orig_prepare
            model.prepare_inputs_for_generation = _prepare_wrapper

    def detach_model(self, model: Any) -> None:
        orig = getattr(model, "_monitoring_orig_prepare", None)
        if orig is not None:
            model.prepare_inputs_for_generation = orig
            model._monitoring_orig_prepare = None
        if self._orig_prepare is not None:
            self._orig_prepare = None
        if self.transport is not None:
            self.transport._using_forward_hooks = False
            self.transport._active_specs = []

    # --- step context ----------------------------------------------------
    def build_step_context(
        self, model_inputs: Any
    ) -> Optional[StepContext]:
        """Port of the pre-refactor ``MonitoringEngine._prepare_ring_step``
        plus the kv_offsets / q_len / kv_dim derivation that used to live
        in ``_install_prepare_wrapper``.

        Returns ``None`` to skip the step (degenerate batch, missing mask,
        etc.) -- the driver short-circuits in that case.
        """
        if not isinstance(model_inputs, dict):
            return None
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        past_key_values = model_inputs.get("past_key_values")
        cache_position = model_inputs.get("cache_position")
        try:
            logits_to_keep = int(model_inputs.get("logits_to_keep", 0))
        except Exception:
            logits_to_keep = 0

        if input_ids is None or not hasattr(input_ids, "shape"):
            return None
        try:
            input_shape = tuple(input_ids.shape)
        except Exception:
            return None
        if not input_shape:
            return None
        try:
            batch_size = int(input_shape[0])
        except Exception:
            return None
        if batch_size <= 0:
            return None

        # Detect prefill vs decode -- prefer cache_position (set explicitly
        # by HF when StaticCache is active), fall back to past_key_values
        # / input_ids shape heuristics for the dynamic-cache path.
        if cache_position is not None:
            try:
                is_prefill = int(cache_position[0]) == 0
            except Exception:
                is_prefill = past_key_values is None
        else:
            is_prefill = past_key_values is None
            try:
                if hasattr(input_ids, "dim") and int(input_ids.dim()) >= 2:
                    if int(input_ids.shape[1]) > 1:
                        is_prefill = True
            except Exception:
                pass

        # Reset batch state on prefill or batch-size change.  Each reset
        # bumps the engine-scoped group counter so within one generate()
        # call mid-stream batch shrinks still get fresh request IDs.
        current_ids = self._batch_request_ids
        need_reset = (
            is_prefill or current_ids is None or len(current_ids) != batch_size
        )
        if need_reset:
            gid = self.engine.next_auto_group_id()
            self._batch_request_ids = [f"{gid}:{i}" for i in range(batch_size)]
            for rid in self._batch_request_ids:
                if rid not in self.request_ids_in_this_generate:
                    self.request_ids_in_this_generate.append(rid)
            self._batch_starts = [0] * batch_size
            self._batch_finished = [False] * batch_size
            # On (re)prefill, recompute kv_offsets from the attention mask
            # (it can change per-prefill if the user sends a fresh mask).
            self._prefill_kv_offsets = None

        req_ids = self._batch_request_ids
        starts = self._batch_starts
        finished = self._batch_finished
        if req_ids is None or starts is None or finished is None:
            return None

        # Qwen3 + StaticCache passes attention_mask as a dict
        # {"full_attention": <4D tensor>}; unwrap before scanning.
        if isinstance(attention_mask, dict):
            if "full_attention" in attention_mask:
                attention_mask = attention_mask["full_attention"]
            else:
                return None

        # Compute kv_offsets once per prefill from the 2D/4D mask.
        # Left-padded HF batches: dynamic and static caches both have the
        # same left padding in the kv dimension (cache_position=arange).
        # kv_offset = pad_len = seq_len - real_len.
        if (
            self._prefill_kv_offsets is None
            and attention_mask is not None
            and hasattr(attention_mask, "dim")
        ):
            try:
                ndim = attention_mask.dim()
                if ndim == 2:
                    seq_len = int(attention_mask.shape[1])
                    real_lens = attention_mask.sum(dim=1).tolist()
                    self._prefill_kv_offsets = [
                        seq_len - int(rl) for rl in real_lens
                    ]
                elif ndim == 4:
                    am_b = int(attention_mask.shape[0])
                    kvo: List[int] = []
                    for b in range(am_b):
                        row = attention_mask[b, 0, -1, :]
                        pad = int((row < 0).long().argmin().item())
                        if pad == 0 and row[0] < 0:
                            pad = int(row.shape[0])
                        kvo.append(pad)
                    self._prefill_kv_offsets = kvo
            except Exception:
                pass

        # Build per-request token_ranges.
        token_ranges: List[Tuple[int, int]] = []
        no_strip_left_pad = self._no_strip_left_pad
        if is_prefill:
            if attention_mask is None or not hasattr(attention_mask, "dim"):
                return None
            try:
                ndim = int(attention_mask.dim())
                if ndim == 2:
                    lengths = (
                        attention_mask.sum(dim=1).tolist()
                        if not no_strip_left_pad
                        else [attention_mask.shape[1]] * attention_mask.shape[0]
                    )
                elif ndim == 4 and len(input_shape) >= 2 and int(input_shape[1]) > 0:
                    # 4D causal mask [batch, 1, q_len, kv_dim] -- used by
                    # static-cache generate.  Values: 0.0 = attend, large
                    # negative = masked (NOT 0/1).  Count non-masked
                    # positions among the first q_len key slots using the
                    # last query row (most permissive for left-padded
                    # causal sequences).
                    q_len_mask = int(input_shape[1])
                    lengths = (
                        (attention_mask[:, 0, -1, :q_len_mask] >= 0.0)
                        .sum(dim=-1).long().tolist()
                        if not no_strip_left_pad
                        else [q_len_mask] * int(attention_mask.shape[0])
                    )
                else:
                    return None
                lengths = [int(v) for v in lengths]
            except Exception:
                return None
            if len(lengths) != batch_size:
                return None
            for i in range(batch_size):
                start_i = int(starts[i])
                delta_i = int(lengths[i])
                if delta_i < 0:
                    delta_i = 0
                end_i = start_i + delta_i
                token_ranges.append((start_i, end_i))
                starts[i] = end_i
        else:
            # Decode: optional post-EOS strip.  Detection runs one step
            # late by construction -- ``input_ids[:, -1]`` is the token
            # appended by the previous step's argmax.  At step N
            # (EOS-producing): last_id is T_{N-1}, no latch, capture
            # normally.  At step N+1 (EOS-feeding): last_id = T_N = EOS,
            # latch and strip from here.  This keeps the activation that
            # produced the first EOS while dropping every step thereafter.
            #
            # One ``.tolist()`` per step (B bytes) -- single GPU->CPU sync,
            # mirroring HF's own per-step ``unfinished_sequences.max() == 0``
            # sync.  Skipped entirely when the eos set is empty (auto-detect
            # found nothing) or when the user opted out via
            # ``no_strip_right_pad=True``.
            no_strip_right_pad = self._no_strip_right_pad
            if (
                self._eos_token_ids
                and not no_strip_right_pad
                and hasattr(input_ids, "shape")
                and len(input_ids.shape) >= 2
                and int(input_ids.shape[1]) >= 1
            ):
                try:
                    last_ids_list = input_ids[:, -1].tolist()
                except Exception:
                    last_ids_list = None
                if last_ids_list is not None and len(last_ids_list) == batch_size:
                    eos_set = self._eos_token_ids
                    for i in range(batch_size):
                        if not finished[i] and last_ids_list[i] in eos_set:
                            finished[i] = True

            for i in range(batch_size):
                start_i = int(starts[i])
                if finished[i] and not no_strip_right_pad:
                    token_ranges.append((start_i, start_i))
                else:
                    end_i = start_i + 1
                    token_ranges.append((start_i, end_i))
                    starts[i] = end_i

        if self._debug_step:
            print(
                f"[ring_step] prefill={is_prefill} "
                f"token_ranges={token_ranges} finished={list(finished)}"
            )

        for rid, token_range in zip(req_ids, token_ranges):
            self.token_ranges_in_this_generate.setdefault(rid, []).append(
                (int(token_range[0]), int(token_range[1]))
            )

        # Derive q_len, kv_dim, dim0_offsets.
        q_len = int(input_shape[1]) if len(input_shape) >= 2 else 1
        is_static = (
            past_key_values is not None
            and hasattr(past_key_values, "max_cache_len")
        )
        kv_dim = _get_kv_dim(past_key_values, q_len, is_static=is_static)

        tp_rank = (
            getattr(self.model_cfg, "tp_rank", 0)
            if self.model_cfg is not None
            else 0
        )
        kv_offsets = (
            list(self._prefill_kv_offsets)
            if self._prefill_kv_offsets is not None
            else [0] * batch_size
        )
        token_ids_dtype = (
            input_ids.dtype if hasattr(input_ids, "dtype") else None
        )

        return StepContext(
            model_id=str(self.model_id),
            flattened=False,
            req_ids=list(req_ids),
            token_ranges=token_ranges,
            dim0_offsets=list(range(batch_size)),
            kv_offsets=kv_offsets,
            tp_rank=tp_rank,
            batch=batch_size,
            q_len=q_len,
            kv_dim=kv_dim,
            logits_to_keep=logits_to_keep,
            token_ids_dtype=token_ids_dtype,
        )

    # --- manual entry for generate_greedy --------------------------------
    def before_forward_manual(
        self,
        input_ids: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        cache_position: Any = None,
        logits_to_keep: int = 0,
    ) -> None:
        """Manual entry for ``generate_greedy`` (no
        ``prepare_inputs_for_generation`` to wrap).  Synthesizes the dict
        shape and runs the canonical driver."""
        self.before_forward({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "logits_to_keep": logits_to_keep,
        })

    # --- Phase-3 upfront capacity check helper --------------------------
    def decode_step_bytes(self, batch: int, kv_dim_estimate: int,
                          logits_to_keep: int = 1) -> int:
        """Estimate aligned bytes for one decode step (q_len=1) at the
        given kv_dim.  Used by ``generate_with_monitoring``'s upfront
        capacity check to decide whether to disable compilation entirely
        (see Phase 3 in ``generate_with_monitoring``).
        """
        if self.model_cfg is None:
            return 0
        total = 0
        for spec in self.active_specs:
            shape = compute_hook_shape(
                spec.hook_type, self.model_cfg,
                batch, q_len=1, kv_dim=kv_dim_estimate,
                logits_to_keep=logits_to_keep,
            )
            if not shape:
                continue
            dtype = spec.dtype if spec.dtype is not None else self.model_cfg.dtype
            elem_size = torch._utils._element_size(dtype)
            nbytes = elem_size
            for d in shape:
                nbytes *= d
            total += align_up_py(nbytes, 16)
        return total


# Compatibility spelling for integrations written against DMI API v1.
HFAdaptor = HuggingFaceAdapter


__all__ = [
    "HuggingFaceAdapter",
    "HFAdaptor",
    "_prepare_profile_times",
    "print_prepare_profile",
]
