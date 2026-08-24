"""Monitored generation entry points for Hugging Face models."""
from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .adapter import (
    HuggingFaceAdapter,
    _prepare_profile_times,
    print_prepare_profile,
)


# ---------------------------------------------------------------------------
# Greedy generation timing model
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
# generate_with_monitoring (rewritten to use HuggingFaceAdapter)
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
            HuggingFaceAdapter auto-detects from ``model.generation_config.eos_token_id``
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
    # Phase 2: install monitoring via HuggingFaceAdapter
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
    adaptor: Optional[HuggingFaceAdapter] = None
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
    adaptor = HuggingFaceAdapter(
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

    from ...storage.internals import make_lazy_internal
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
    benchmarks/bench_hf_transport.py:
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
        monitoring: if True, install ring transport hooks via HuggingFaceAdapter and
            call before_forward_manual before each forward pass.
        hook_selection: hook selection preset (e.g. "hidden-states", "full").
            Only used when monitoring=True.
        no_strip_left_pad: forwarded to ``HuggingFaceAdapter`` when monitoring=True.  If
            True, keep the full mask width in prefill ``token_ranges``
            (i.e. emit a row for every model-input position including
            left-padding).  Default False (strip left-pad).
        no_strip_right_pad: forwarded to ``HuggingFaceAdapter`` when monitoring=True.
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

    adaptor: Optional[HuggingFaceAdapter] = None
    if monitoring:
        engine = getattr(model, "monitoring_engine", None)
        if engine is not None and engine._ring_transport is not None:
            adaptor = HuggingFaceAdapter(
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
    "GreedyGenerateTimings",
    "generate_with_monitoring",
    "generate_with_monitoring_dict",
    "generate_greedy_with_monitoring",
    "_prepare_profile_times",
    "print_prepare_profile",
]
