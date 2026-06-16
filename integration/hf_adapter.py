"""HF integration: HFAdaptor + monitored generate() entry points.

Phase 2a of the unified-adaptor refactor consolidates the HF-specific
orchestration that used to be sprinkled across ``monitoring/engine.py``
and ``monitoring/generate.py`` into one file.

Key pieces:
  * ``HFAdaptor`` -- concrete ``BackendAdaptor`` for HF models.  Owns
    per-call batch-tracking state (request IDs, per-request token starts /
    finished flags, prefill kv_offsets) that used to live on
    ``MonitoringEngine``.  ``build_step_context`` ports the body of the
    pre-refactor ``MonitoringEngine._prepare_ring_step`` plus the
    kv_offsets / q_len / kv_dim derivation from
    ``_install_prepare_wrapper``.
  * ``generate_with_monitoring`` and ``generate_greedy_with_monitoring``
    -- HF entry points; both run through ``HFAdaptor`` rather than
    touching transport state directly.
  * ``_make_model_shape_from_hf_config`` -- shared helper, lives in
    ``integration/model_shape.py`` and is also imported by VLLMAdaptor.

``monitoring/generate.py`` is now a thin re-export shim; Phase 5 deletes
it once external callers migrate to ``integration.hf_adapter``.
"""
from __future__ import annotations

import functools
import inspect
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from monitoring import ring_transport
from monitoring.adaptor_base import BackendAdaptor
from monitoring.ring_transport import (
    HookSpec,
    ModelShapeConfig,
    _compute_hook_shape,
    _get_kv_dim,
    align_up_py,
    install_ring_hooks,
)
from monitoring.selection import (
    apply_hook_selection,
    filter_by_pp_rank,
    filter_by_tp_rank,
)
from monitoring.step_context import StepContext
from integration.model_shape import _make_model_shape_from_hf_config


# ---------------------------------------------------------------------------
# Profiling helpers (moved from monitoring/generate.py)
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


# Model-shape helpers were extracted to integration/model_shape.py in
# Phase 3a so VLLMAdaptor can share them.  The legacy
# `_make_model_shape(model)` wrapper that fed the old
# `monitoring/vllm_integration.py` substantial-implementation was
# deleted alongside Phase 3a's shim shrink (its only caller).


# ---------------------------------------------------------------------------
# GreedyGenerateTimings (moved from monitoring/generate.py)
# ---------------------------------------------------------------------------

@dataclass
class GreedyGenerateTimings:
    """Optional per-step timing data from generate_greedy_with_monitoring."""
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    decode_steps: int = 0
    batch_size: int = 0
    prefill_tokens: int = 0
    step_ms: List[float] = field(default_factory=list)

    @property
    def prefill_tok_per_s(self) -> float:
        if self.prefill_ms <= 0:
            return 0.0
        return self.batch_size * self.prefill_tokens / self.prefill_ms * 1000.0

    @property
    def decode_tok_per_s(self) -> float:
        if self.decode_ms <= 0 or self.decode_steps <= 0:
            return 0.0
        return self.batch_size * self.decode_steps / self.decode_ms * 1000.0

    @property
    def e2e_tok_per_s(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        total_tokens = self.batch_size * (self.prefill_tokens + self.decode_steps)
        return total_tokens / self.total_ms * 1000.0

    @property
    def tpot_ms(self) -> float:
        if self.decode_steps <= 0:
            return 0.0
        return self.decode_ms / self.decode_steps


# ---------------------------------------------------------------------------
# HFAdaptor
# ---------------------------------------------------------------------------

class HFAdaptor(BackendAdaptor):
    """``BackendAdaptor`` for HuggingFace ``transformers`` models.

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
                "HFAdaptor.detect_model_shape: model.config is missing "
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
                f"[ring] HFAdaptor.attach_model: active={len(self.active_specs)} "
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
            shape = _compute_hook_shape(
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


# ---------------------------------------------------------------------------
# generate_with_monitoring (rewritten to use HFAdaptor)
# ---------------------------------------------------------------------------

def _generate_with_monitoring_impl(
    model: Any, *args: Any,
    hook_selection: Optional[str] = None,
    no_strip_left_pad: bool = False,
    no_strip_right_pad: bool = False,
    eos_token_id: Any = None,
    _return_model_id: bool = False,
    _return_internal_metadata: bool = False,
    **kwargs: Any,
):
    """Run HF ``generate()`` with ring-transport monitoring hooks active.

    Hooks are installed before ``generate()`` and removed on return.

    Args:
        hook_selection: preset name controlling which hooks are enabled.
            "full" (default) -- all hooks
            "hf-only"        -- hidden states + attention weights + logits
            "hidden-states"  -- residual stream + embeddings + final LN
            "logits"         -- final logits only
        no_strip_left_pad: if True, keep the full mask width when computing
            prefill ``token_ranges`` (i.e. emit a row for every model-input
            position including left-padding).  Default False (strip left-pad).
        no_strip_right_pad: if True, keep decode rows even after a request
            hits EOS (HF inserts pad in lockstep batches; those rows are
            captured normally).  Default False (strip post-EOS noise).
        eos_token_id: per-call override for EOS detection.  Accepts
            ``int``, ``list[int]``, or ``torch.Tensor``.  When ``None``,
            HFAdaptor auto-detects from ``model.generation_config.eos_token_id``
            then ``model.config.eos_token_id``; if neither is set, the
            post-EOS strip never latches.  Has no effect when
            ``no_strip_right_pad=True``.

    For CUDA graph capture, use HF's built-in ``CompileConfig``::

        from transformers import CompileConfig
        generate_with_monitoring(
            model, ...,
            cache_implementation="static",
            compile_config=CompileConfig(mode="reduce-overhead", fullgraph=False),
        )

    If the model is externally compiled (``torch.compile`` on the model
    or its forward) AND static cache is used, this function strips the
    external compilation and injects an equivalent ``CompileConfig`` so
    HF compiles only the decode path (prefill stays uncompiled).
    """
    import types

    # ------------------------------------------------------------------
    # Phase 1: Strip external compilation when static cache is active.
    #
    # WHY: HF generate() has a prefill/decode split:
    #   * Prefill runs via uncompiled self(...) (different shapes each call).
    #   * Decode runs via compiled model_forward from get_compiled_call()
    #     (stable shapes, CUDA graphs via mode="reduce-overhead").
    #
    # External torch.compile (model.forward = torch.compile(...) or
    # model = torch.compile(model)) wraps BOTH prefill and decode in
    # one compiled function.  When prefill triggers the safety net's
    # branch 3 (single tensor > ring capacity), the .cpu() call causes
    # graph breaks.  With mode="reduce-overhead", this fragments the
    # forward into many tiny CUDA graph segments.  When decode then
    # runs with a different topology (no graph breaks, full ring path),
    # the CUDA graph tree's shared buffer pool is corrupted -> segfault.
    #
    # FIX: Strip external compilation and inject HF's CompileConfig.
    # HF's get_compiled_call() only compiles decode, leaving prefill
    # uncompiled.  Prefill can safely use the safety net (eager, no CUDA
    # graphs).  Decode uses ring path (no graph breaks, full CUDA graph).
    #
    # WHEN: Only when static cache is active (cache_implementation="static"
    # or external StaticCache in past_key_values).  Without static cache,
    # HF doesn't compile at all, so external compilation is harmless.
    # ------------------------------------------------------------------
    _saved_compiled_forward = None
    _saved_cache_impl = None

    _cache_impl_static = (
        kwargs.get("cache_implementation") == "static"
        or getattr(getattr(model, "generation_config", None),
                   "cache_implementation", None) == "static"
    )
    _pkv = kwargs.get("past_key_values")
    _external_compileable_cache = (
        _pkv is not None and hasattr(_pkv, "is_compileable") and _pkv.is_compileable
    )
    has_static_cache = _cache_impl_static or _external_compileable_cache
    is_compiled_model = hasattr(model, "_orig_mod")
    check_target = getattr(model, "_orig_mod", model)
    is_compiled_forward = "forward" in check_target.__dict__

    if has_static_cache and (is_compiled_model or is_compiled_forward):
        if is_compiled_model:
            model = model._orig_mod
        elif is_compiled_forward:
            _saved_compiled_forward = check_target.__dict__["forward"]
            cls_forward = type(check_target).forward
            check_target.forward = types.MethodType(cls_forward, check_target)

        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "cache_implementation", None) is not None:
            if "cache_implementation" not in kwargs:
                kwargs["cache_implementation"] = gen_cfg.cache_implementation
            _saved_cache_impl = gen_cfg.cache_implementation
            gen_cfg.cache_implementation = None

        if "compile_config" not in kwargs:
            try:
                from transformers import CompileConfig
                kwargs["compile_config"] = CompileConfig(
                    mode="reduce-overhead", fullgraph=False)
            except ImportError:
                pass

        warnings.warn(
            "[ring_transport] External torch.compile detected with static cache. "
            "Stripping external compilation; HF will compile decode only via "
            "CompileConfig (prefill stays uncompiled). Recommend: remove "
            'torch.compile() and pass compile_config=CompileConfig('
            'mode="reduce-overhead", fullgraph=False) to generate() directly.',
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Phase 2: install monitoring via HFAdaptor
    # ------------------------------------------------------------------
    target = getattr(model, "_orig_mod", model)
    _restore_engine: Any = None
    if target is not model:
        outer_engine = getattr(model, "monitoring_engine", None)
        _restore_engine = getattr(target, "monitoring_engine", None)
        if outer_engine is not None:
            target.monitoring_engine = outer_engine

    engine = getattr(target, "monitoring_engine", None)
    if engine is None:
        engine = getattr(model, "monitoring_engine", None)
    adaptor: Optional[HFAdaptor] = None
    if engine is None:
        raise RuntimeError(
            "generate_with_monitoring() requires model.monitoring_engine to "
            "be set to a MonitoringEngine instance."
        )
    if engine._ring_transport is None:
        raise RuntimeError(
            "generate_with_monitoring() found model.monitoring_engine, but "
            "ring transport is disabled. Construct MonitoringEngine with "
            "enable_ring_transport=True or call engine.enable_ring_transport(...)."
        )
    adaptor = HFAdaptor(
        engine, engine._model_id,
        no_strip_left_pad=no_strip_left_pad,
        no_strip_right_pad=no_strip_right_pad,
        eos_token_id=eos_token_id,
    )
    # Expose the adaptor on the engine so external callers (e.g. the HF
    # compare-runner test harness) can read per-step batch tracking
    # (_batch_request_ids, _batch_starts).
    engine._hf_adaptor = adaptor
    adaptor.request_ids_in_this_generate = []
    adaptor.token_ranges_in_this_generate = {}
    adaptor.attach_model(
        target,
        hook_selection=hook_selection or "full",
        install_prepare_wrapper=True,
    )

    # ------------------------------------------------------------------
    # Phase 3: Check if a single decode step exceeds ring capacity.
    #
    # WHY disable_compile is essential when overflow is expected:
    # HookPoint.forward's safety-net check on `transport.force_eager`
    # is Python.  Under CUDA graphs the branch is baked at warmup
    # (force_eager False) and the captured forward replays the fast
    # path regardless of runtime force_eager.  So the safety net is
    # only reachable in eager forwards.  Detecting overflow upfront +
    # disable_compile keeps the whole generate() eager so per-batch
    # before_forward + safety net can run.
    #
    # Out of scope: runtime overflow with CUDA graphs active for a
    # shape we didn't predict (e.g. dynamic batching).  Doesn't arise
    # in current HF usage (StaticCache fixes shapes at warmup).
    # ------------------------------------------------------------------
    if adaptor is not None and adaptor.model_cfg is not None and adaptor.active_specs:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
            batch = int(input_ids.shape[0])
            re = adaptor.ring_engine
            effective_cap = min(re.payload_cap(), re.staging_cap())

            input_len = int(input_ids.shape[1])
            try:
                gen_cfg_copy, _ = model._prepare_generation_config(
                    kwargs.get("generation_config"), **kwargs)
                gen_cfg_copy = model._prepare_generated_length(
                    generation_config=gen_cfg_copy,
                    has_default_max_length=(
                        kwargs.get("max_length") is None
                        and gen_cfg_copy.max_length is not None),
                    has_default_min_length=(
                        kwargs.get("min_length") is None
                        and gen_cfg_copy.min_length is not None),
                    model_input_name="input_ids",
                    inputs_tensor=input_ids,
                    input_ids_length=input_len,
                )
                kv_dim_estimate = int(gen_cfg_copy.max_length) - 1
            except Exception:
                max_new = int(kwargs.get("max_new_tokens", 20))
                kv_dim_estimate = input_len + max_new - 1
                warnings.warn(
                    "[ring_transport] Could not call model._prepare_generated_length "
                    "to estimate kv_dim for capacity check. Falling back to "
                    f"kv_dim={kv_dim_estimate} (input_len={input_len} + "
                    f"max_new_tokens={max_new} - 1). This may under-estimate "
                    "attn_scores/pattern hook sizes if max_position_embeddings "
                    "caps max_length.",
                    stacklevel=2,
                )

            decode_bytes = adaptor.decode_step_bytes(batch, kv_dim_estimate)

            if decode_bytes > effective_cap:
                # Decode steps will overflow.  Force eager dispatch so
                # before_forward's per-batch capacity check + HookPoint's
                # safety net (ring D2D where it fits, submit_cpu_direct
                # where it doesn't) can run.
                had_compile = bool(
                    kwargs.pop("compile_config", None) is not None
                    or kwargs.pop("cache_implementation", None) is not None
                )
                kwargs["disable_compile"] = True
                msg = (
                    f"[ring_transport] Decode step ({decode_bytes / 1e6:.1f} MB) "
                    f"exceeds ring capacity ({effective_cap / 1e6:.0f} MB). "
                    f"Using eager dispatch + per-hook safety net."
                )
                if had_compile:
                    msg += " Disabled CUDA graph compilation for this generate() call."
                warnings.warn(msg, stacklevel=2)

    try:
        gen = model.generate(*args, **kwargs)
        if _return_internal_metadata:
            token_ranges = {
                rid: tuple(ranges)
                for rid, ranges in adaptor.token_ranges_in_this_generate.items()
            }
            return (
                gen,
                engine._model_id,
                tuple(adaptor.request_ids_in_this_generate),
                token_ranges,
            )
        if _return_model_id:
            return gen, engine._model_id
        return gen
    finally:
        if adaptor is not None:
            adaptor.detach_model(target)
        if target is not model:
            target.monitoring_engine = _restore_engine
        if _saved_compiled_forward is not None:
            check_target.forward = _saved_compiled_forward
        if _saved_cache_impl is not None:
            gen_cfg = getattr(model, "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.cache_implementation = _saved_cache_impl
        # force_eager is owned by before_forward (per-batch reassignment);
        # no cleanup needed -- the next generate()'s first batch will set it.


def generate_with_monitoring(
    model: Any, *args: Any,
    hook_selection: Optional[str] = None,
    no_strip_left_pad: bool = False,
    no_strip_right_pad: bool = False,
    eos_token_id: Any = None,
    **kwargs: Any,
):
    """Run HF ``generate()`` with monitoring hooks and return HF output unchanged."""
    return _generate_with_monitoring_impl(
        model, *args,
        hook_selection=hook_selection,
        no_strip_left_pad=no_strip_left_pad,
        no_strip_right_pad=no_strip_right_pad,
        eos_token_id=eos_token_id,
        **kwargs,
    )


def generate_with_monitoring_dict(
    model: Any, *args: Any,
    hook_selection: Optional[str] = None,
    no_strip_left_pad: bool = False,
    no_strip_right_pad: bool = False,
    eos_token_id: Any = None,
    reader: Any = None,
    internal_requirements: Any = None,
    **kwargs: Any,
):
    """Run monitored HF ``generate()`` and return dict-style output with DMI internals.

    This API always forces ``return_dict_in_generate=True`` and attaches one DMI
    extension, ``dmi_internal``, which lazily reads captured internals.
    """
    gen_kwargs = dict(kwargs)
    if gen_kwargs.get("return_dict_in_generate") is False:
        warnings.warn(
            "generate_with_monitoring_dict() requires "
            "return_dict_in_generate=True; overriding the supplied False value.",
            UserWarning,
            stacklevel=2,
        )
    gen_kwargs["return_dict_in_generate"] = True

    output, model_id, request_ids, token_ranges = _generate_with_monitoring_impl(
        model, *args,
        hook_selection=hook_selection,
        no_strip_left_pad=no_strip_left_pad,
        no_strip_right_pad=no_strip_right_pad,
        eos_token_id=eos_token_id,
        _return_internal_metadata=True,
        **gen_kwargs,
    )

    from monitoring.internal_mapper import make_lazy_internal
    dmi_internal = make_lazy_internal(
        model_id,
        reader=reader,
        requirements=internal_requirements,
        request_ids=request_ids,
        token_ranges=token_ranges,
    )
    try:
        output.dmi_internal = dmi_internal
    except Exception:
        object.__setattr__(output, "dmi_internal", dmi_internal)
    return output


# ---------------------------------------------------------------------------
# generate_greedy_with_monitoring (manual prefill + decode loop)
# ---------------------------------------------------------------------------

def generate_greedy_with_monitoring(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    eos_token_id: Any = None,
    pad_token_id: Optional[int] = None,
    logits_to_keep: int = 0,
    cuda_graphs: bool = False,
    monitoring: bool = False,
    hook_selection: Optional[str] = None,
    no_strip_left_pad: bool = False,
    no_strip_right_pad: bool = False,
    timings: Optional[GreedyGenerateTimings] = None,
) -> List[Any]:
    """Greedy-argmax generate loop, no HF generate() overhead.

    Follows the same pattern as the hf_offload manual loop in
    benchmark/bench_hf_transport.py:
      * Optional torch.compile + StaticCache + CUDA graphs for decode.
      * Without cuda_graphs: HF default (DynamicCache) for KV cache.
      * Per-step CPU sync via token.cpu() (matches HF generate()'s
        implicit GPU->CPU sync from ``unfinished_sequences.max() == 0``).

    Supports EOS stopping (after min_new_tokens), min_new_tokens,
    max_new_tokens.  Greedy argmax only -- no beam search, no sampling.

    Args:
        model: HF model (AutoModelForCausalLM or similar).
        input_ids: [B, seq_len] input token IDs on CUDA.
        attention_mask: [B, seq_len] attention mask on CUDA.
        max_new_tokens: maximum tokens to generate.
        min_new_tokens: minimum tokens before EOS can stop generation.
        eos_token_id: EOS token ID.  None = never stop early.
        pad_token_id: pad token ID (unused, kept for compat).
        logits_to_keep: 0 = all rows, 1 = last position only.
        cuda_graphs: if True, compile decode step with reduce-overhead +
            StaticCache.  If False, use HF default DynamicCache, no compile.
        monitoring: if True, install ring transport hooks via HFAdaptor and
            call before_forward_manual before each forward pass.
        hook_selection: hook selection preset (e.g. "hidden-states", "full").
            Only used when monitoring=True.
        no_strip_left_pad: forwarded to ``HFAdaptor`` when monitoring=True.  If
            True, keep the full mask width in prefill ``token_ranges``
            (i.e. emit a row for every model-input position including
            left-padding).  Default False (strip left-pad).
        no_strip_right_pad: forwarded to ``HFAdaptor`` when monitoring=True.
            If True, keep decode rows even after a request hits EOS.
            Default False (strip post-EOS).
        timings: if provided, filled with timing data.

    Returns:
        List of generated token ID tensors on CPU, one per batch element.
        Each tensor has shape [num_generated_tokens].
    """
    device = input_ids.device
    B, Pmax = input_ids.shape

    _wants_position_ids = (
        "position_ids" in inspect.signature(model.forward).parameters
    )

    def _position_ids_from_mask(mask: Any) -> Any:
        pos = mask.long().cumsum(dim=-1) - 1
        pos.masked_fill_(mask == 0, 0)
        return pos

    adaptor: Optional[HFAdaptor] = None
    if monitoring:
        engine = getattr(model, "monitoring_engine", None)
        if engine is not None and engine._ring_transport is not None:
            adaptor = HFAdaptor(
                engine, engine._model_id,
                no_strip_left_pad=no_strip_left_pad,
                no_strip_right_pad=no_strip_right_pad,
                eos_token_id=eos_token_id,
            )
            engine._hf_adaptor = adaptor
            adaptor.attach_model(
                model,
                hook_selection=hook_selection or "full",
                install_prepare_wrapper=False,
            )

    try:
        if cuda_graphs:
            from transformers import StaticCache
            max_cache_len = Pmax + max_new_tokens + 4
            cache = StaticCache(
                config=model.config, batch_size=B,
                max_cache_len=max_cache_len, device=device,
                dtype=model.dtype,
            )
        else:
            cache = None

        def _decode_step_static(token, cache, cache_position):
            kwargs: Dict[str, Any] = {
                "input_ids": token,
                "use_cache": True,
                "past_key_values": cache,
                "cache_position": cache_position,
                "output_hidden_states": False,
                "output_attentions": False,
                "return_dict": True,
                "logits_to_keep": logits_to_keep,
            }
            if _wants_position_ids:
                kwargs["position_ids"] = cache_position.unsqueeze(0).expand(
                    token.shape[0], -1)
            return model(**kwargs)

        if cuda_graphs:
            compiled_decode = torch.compile(
                _decode_step_static, mode="reduce-overhead", fullgraph=False)
        else:
            compiled_decode = None

        do_timing = timings is not None
        torch.cuda.synchronize()
        t0 = time.perf_counter() if do_timing else 0.0

        with torch.no_grad():
            prefill_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "output_hidden_states": False,
                "output_attentions": False,
                "return_dict": True,
                "logits_to_keep": logits_to_keep,
            }
            if cache is not None:
                prefill_kwargs["past_key_values"] = cache
                prefill_kwargs["cache_position"] = torch.arange(
                    Pmax, device=device, dtype=torch.long)
            if _wants_position_ids:
                prefill_kwargs["position_ids"] = _position_ids_from_mask(
                    attention_mask)

            if adaptor is not None:
                adaptor.before_forward_manual(
                    input_ids, attention_mask,
                    past_key_values=prefill_kwargs.get("past_key_values"),
                    cache_position=prefill_kwargs.get("cache_position"),
                    logits_to_keep=logits_to_keep,
                )

            out = model(**prefill_kwargs)

        token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_cpu = token.squeeze(-1).cpu()  # forces GPU->CPU sync

        if cuda_graphs:
            cache_pos = torch.tensor([Pmax], device=device, dtype=torch.long)
        else:
            cache = out.past_key_values

        t_decode_start = time.perf_counter() if do_timing else 0.0

        unfinished_sequences = torch.ones(B, dtype=torch.long, device=device)

        _prev_step_t = t_decode_start if do_timing else 0.0
        all_generated: List[Any] = [token.squeeze(-1)]
        with torch.no_grad():
            for step in range(max_new_tokens - 1):
                if compiled_decode is not None:
                    if adaptor is not None:
                        adaptor.before_forward_manual(
                            token, attention_mask,
                            past_key_values=cache, cache_position=cache_pos,
                            logits_to_keep=logits_to_keep,
                        )
                    torch.compiler.cudagraph_mark_step_begin()
                    out = compiled_decode(token, cache, cache_pos)
                else:
                    decode_kwargs: Dict[str, Any] = {
                        "input_ids": token,
                        "past_key_values": cache,
                        "use_cache": True,
                        "output_hidden_states": False,
                        "output_attentions": False,
                        "return_dict": True,
                        "logits_to_keep": logits_to_keep,
                    }
                    if _wants_position_ids:
                        seq_pos = Pmax + step + 1
                        decode_kwargs["position_ids"] = torch.full(
                            (B, 1), seq_pos, device=device, dtype=torch.long)
                    if adaptor is not None:
                        adaptor.before_forward_manual(
                            token, attention_mask,
                            past_key_values=cache,
                            logits_to_keep=logits_to_keep,
                        )
                    out = model(**decode_kwargs)
                    cache = out.past_key_values

                token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                if cuda_graphs:
                    cache_pos = cache_pos + 1

                all_generated.append(token.squeeze(-1))

                tokens_generated = step + 2
                if (eos_token_id is not None
                        and tokens_generated > min_new_tokens):
                    unfinished_sequences = unfinished_sequences & (
                        token.squeeze(-1) != eos_token_id).long()
                this_peer_finished = unfinished_sequences.max() == 0  # GPU->CPU sync

                if do_timing:
                    step_t = time.perf_counter()
                    timings.step_ms.append((step_t - _prev_step_t) * 1000.0)
                    _prev_step_t = step_t

                if this_peer_finished:
                    break

        torch.cuda.synchronize()

        if do_timing:
            t_end = time.perf_counter()
            timings.total_ms = (t_end - t0) * 1000.0
            timings.prefill_ms = (t_decode_start - t0) * 1000.0
            timings.decode_ms = (t_end - t_decode_start) * 1000.0
            timings.decode_steps = len(timings.step_ms)
            timings.batch_size = B
            timings.prefill_tokens = Pmax

        generated_ids = torch.stack(all_generated, dim=1).cpu()
        results: List[Any] = []
        for b in range(B):
            seq = generated_ids[b]
            if eos_token_id is not None:
                eos_positions = (seq == eos_token_id).nonzero(as_tuple=False)
                if len(eos_positions) > 0:
                    seq = seq[:int(eos_positions[0].item()) + 1]
            results.append(seq)

        return results

    finally:
        if adaptor is not None:
            adaptor.detach_model(model)


__all__ = [
    "HFAdaptor",
    "GreedyGenerateTimings",
    "generate_with_monitoring",
    "generate_with_monitoring_dict",
    "generate_greedy_with_monitoring",
    "_prepare_profile_times",
    "print_prepare_profile",
]
