# tests/correctness/hf_reference.py
"""HuggingFace reference implementations for correctness comparison.

Two reference modes:
  - ROL (manual KV-cache rollout): full [T, vocab] logits, hidden states, attn patterns.
  - GEN (hf generate()): token_ids + decode-only output_scores.

NOTE:
  - tests/correctness/tensor_utils.py no longer exists.
  - Attention-matrix stitching now uses monitoring.segment_merger.merge_segments().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from monitoring.segment_merger import merge_segments


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _HFRef:
    token_ids: torch.Tensor  # [T] cpu long
    final_logits: torch.Tensor  # [T, vocab] cpu
    hidden_states: List[torch.Tensor]  # [n_layer+1] each [T, d] cpu
    attn_pattern: List[torch.Tensor]  # [n_layer] each [H, T, T] cpu


@dataclass
class _HFGenRef:
    token_ids: torch.Tensor  # [T] cpu long (prompt + generated)
    # generate() exposes per-step scores for generated tokens only.
    # scores[s] is the distribution used to pick generated token at step s (0-indexed),
    # shape [vocab] on CPU.
    scores: List[torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_left_pad(ids_row: torch.Tensor, attn_row: torch.Tensor) -> torch.Tensor:
    true_len = int(attn_row.sum().item())
    if true_len <= 0:
        return ids_row[:0]
    return ids_row[-true_len:]


def _position_ids_from_attention_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """HF-generate style position_ids that are stable under left-padding."""
    pos = attn_mask.to(torch.long).cumsum(dim=-1) - 1
    pos.masked_fill_(attn_mask == 0, 0)
    return pos


def _positions_for_unpadded(true_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(true_len, device=device, dtype=torch.long)


def _hf_forward_with_optional_position_ids(
    hf_model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    **kwargs: Any,
) -> Any:
    if position_ids is None:
        return hf_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    try:
        return hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
    except TypeError as e:
        msg = str(e)
        if "position_ids" in msg and (
            "unexpected keyword argument" in msg or "got an unexpected keyword argument" in msg
        ):
            return hf_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        raise


def _parse_request_id(req_id: str) -> Tuple[int, int]:
    """Parse '<group_id>:<local_index>'."""
    m = re.match(r"^(\d+):(\d+)$", req_id)
    if not m:
        raise AssertionError(f"unexpected request_id format: {req_id!r}")
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# GEN reference: HF generate() with output_scores
# ---------------------------------------------------------------------------


@torch.no_grad()
def _hf_generate_collect_scores_batched(
    *,
    hf_model: Any,
    input_ids_batch: torch.Tensor,
    attention_mask_batch: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device,
    cache_implementation: Optional[str] = None,
) -> List[_HFGenRef]:
    """Run HF generate() ONCE on the full padded batch; return per-row (unpadded, EOS-trimmed) refs."""
    hf_model.eval()
    input_ids = input_ids_batch.to(device=device, dtype=torch.long)
    attn = attention_mask_batch.to(device=device, dtype=torch.long)

    B, Pmax = input_ids.shape

    gen_kwargs: Dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        logits_to_keep=0,
    )
    if cache_implementation is not None:
        gen_kwargs["cache_implementation"] = cache_implementation

    gen_out = hf_model.generate(**gen_kwargs)

    seqs = gen_out.sequences.detach().cpu().to(torch.long)  # [B, Pmax+G]
    scores_steps: List[torch.Tensor] = []
    if getattr(gen_out, "scores", None) is not None:
        for s in gen_out.scores:
            scores_steps.append(s.detach().cpu())

    prompt_lens_t = attn.sum(dim=1)  # [B]
    pad_lens_t = Pmax - prompt_lens_t  # [B]
    prompt_lens = prompt_lens_t.detach().cpu().tolist()
    pad_lens = pad_lens_t.detach().cpu().tolist()

    out_refs: List[_HFGenRef] = []
    for b in range(B):
        pad_len = int(pad_lens[b])
        _ = int(prompt_lens[b])  # used implicitly by slicing

        prompt_tok = seqs[b, pad_len:Pmax]  # [prompt_len]
        gen_tok_full = seqs[b, Pmax:]  # [G]
        gen_len = int(gen_tok_full.numel())
        if gen_len > 0:
            eos_hits = (gen_tok_full == int(eos_token_id)).nonzero(as_tuple=False)
            if eos_hits.numel() > 0:
                gen_len = int(eos_hits[0].item()) + 1  # keep EOS
        gen_tok = gen_tok_full[:gen_len]

        # Drop the last generated token: it was predicted but never processed
        # by a forward pass, so monitoring DB won't have it.
        keep_gen = max(0, gen_len - 1)
        tok_ids = torch.cat([prompt_tok, gen_tok[:keep_gen]], dim=0).detach().cpu().to(torch.long)

        row_scores: List[torch.Tensor] = []
        for s in range(min(keep_gen, len(scores_steps))):
            row_scores.append(scores_steps[s][b].detach().cpu())

        out_refs.append(_HFGenRef(token_ids=tok_ids, scores=row_scores))

    return out_refs


# ---------------------------------------------------------------------------
# GEN-HS reference: HF generate() with hidden_states + scores
# ---------------------------------------------------------------------------


@torch.no_grad()
def _hf_generate_collect_hidden_states_batched(
    *,
    hf_model: Any,
    input_ids_batch: torch.Tensor,
    attention_mask_batch: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device,
    cache_implementation: Optional[str] = None,
    logits_to_keep: int = 0,
    want_attentions: bool = True,
) -> List[_HFRef]:
    """Run HF generate() with output_hidden_states + output_attentions; return _HFRef per row.

    Unlike _hf_greedy_rollout_collect_all_batched (manual forward loop), this
    uses generate() end-to-end and is safe under torch.compile + reduce-overhead.

    Provides:
      - token_ids:     [T] (prompt + generated, unpadded, EOS-trimmed)
      - final_logits:  [G, vocab] (decode-only scores; G = number of generated tokens)
      - hidden_states: [n_layers+1] each [T, d] (prefill + decode concatenated)
      - attn_pattern:  [n_layers] each [H, T, T] (stitched from per-step attentions)
    """
    hf_model.eval()
    input_ids = input_ids_batch.to(device=device, dtype=torch.long)
    attn = attention_mask_batch.to(device=device, dtype=torch.long)

    B, Pmax = input_ids.shape

    gen_kwargs: Dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_logits=True,  # raw logits in model dtype (not float32 scores)
        output_hidden_states=True,
        output_attentions=bool(want_attentions),
        logits_to_keep=logits_to_keep,
    )
    if cache_implementation is not None:
        gen_kwargs["cache_implementation"] = cache_implementation

    gen_out = hf_model.generate(**gen_kwargs)

    seqs = gen_out.sequences.detach().cpu().to(torch.long)  # [B, Pmax+G]
    logits_list = list(gen_out.logits) if getattr(gen_out, "logits", None) else []

    # hidden_states: tuple of step_tuples
    # step_tuples[step] = tuple of (n_layers+1) tensors
    # Prefill (step 0): each tensor is [B, Pmax, d]
    # Decode (step s>0): each tensor is [B, 1, d]
    hs_steps = gen_out.hidden_states if getattr(gen_out, "hidden_states", None) else []

    # attentions: tuple of step_tuples
    # step_tuples[step] = tuple of n_layers tensors
    # Prefill (step 0): each tensor is [B, H, Pmax, Pmax]
    # Decode (step s>0): each tensor is [B, H, 1, seq_len_so_far]
    attn_steps = gen_out.attentions if getattr(gen_out, "attentions", None) else []

    # Detect model dtype (generate() upcasts logits to float32 internally)
    model_dtype = next(hf_model.parameters()).dtype

    prompt_lens_t = attn.sum(dim=1)
    pad_lens_t = Pmax - prompt_lens_t
    prompt_lens = prompt_lens_t.detach().cpu().tolist()
    pad_lens = pad_lens_t.detach().cpu().tolist()

    out_refs: List[_HFRef] = []
    for b in range(B):
        pad_len = int(pad_lens[b])

        prompt_tok = seqs[b, pad_len:Pmax]
        gen_tok_full = seqs[b, Pmax:]
        gen_len = int(gen_tok_full.numel())
        if gen_len > 0:
            eos_hits = (gen_tok_full == int(eos_token_id)).nonzero(as_tuple=False)
            if eos_hits.numel() > 0:
                gen_len = int(eos_hits[0].item()) + 1
        gen_tok = gen_tok_full[:gen_len]

        # Drop the last generated token: it was predicted but never processed by a
        # forward pass, so there are no hidden states / attentions for it.  This
        # aligns token_ids with the monitoring DB which only records forward passes.
        n_decode_steps = len(hs_steps) - 1 if hs_steps else 0
        keep_gen = min(gen_len, n_decode_steps)
        tok_ids = torch.cat([prompt_tok, gen_tok[:keep_gen]], dim=0).detach().cpu().to(torch.long)

        # Hidden states per layer: concat prefill (stripped of pad) + decode steps
        n_layers_plus_1 = len(hs_steps[0]) if hs_steps else 0
        hidden_states_per_layer: List[torch.Tensor] = []
        for l_idx in range(n_layers_plus_1):
            chunks: List[torch.Tensor] = []
            # Prefill: strip left-pad
            prefill_hs = hs_steps[0][l_idx][b, pad_len:, :].detach().cpu()  # [plen, d]
            chunks.append(prefill_hs)
            # Decode steps
            for s in range(keep_gen):
                decode_hs = hs_steps[s + 1][l_idx][b].detach().cpu()  # [1, d] or [d]
                if decode_hs.ndim == 1:
                    decode_hs = decode_hs.unsqueeze(0)
                chunks.append(decode_hs)
            hidden_states_per_layer.append(torch.cat(chunks, dim=0))  # [T, d]

        # Attention patterns per layer: stitch prefill + decode steps -> [H, T, T]
        attn_per_layer: List[torch.Tensor] = []
        if attn_steps:
            n_attn_layers = len(attn_steps[0])
            for l_idx in range(n_attn_layers):
                attn_chunks: List[torch.Tensor] = []
                # Prefill: [B, H, Pmax, Pmax] -> [H, plen, plen] (strip left-pad both dims)
                a0 = attn_steps[0][l_idx][b]  # [H, Pmax, Pmax]
                if a0.ndim == 4 and a0.shape[0] == 1:
                    a0 = a0.squeeze(0)
                a0 = a0[:, pad_len:, pad_len:].detach().cpu()  # [H, plen, plen]
                attn_chunks.append(a0)
                # Decode steps: [B, H, 1, seq_so_far] -> [H, 1, unpadded_seq_so_far]
                for s in range(keep_gen):
                    ab = attn_steps[s + 1][l_idx][b]  # [H, 1, seq_so_far]
                    if ab.ndim == 2:
                        ab = ab.unsqueeze(1)  # [H, K] -> [H, 1, K]
                    # strip left-pad on key dim
                    ab = ab[..., pad_len:].detach().cpu()
                    attn_chunks.append(ab)
                # Stitch using merge_segments (same logic as rollout)
                merged = merge_segments(attn_chunks, "blocks.attn.hook_pattern")
                if merged is None:
                    merged = torch.empty((0,), dtype=torch.float32)
                attn_per_layer.append(merged)

        # Logits (decode-only: positions plen-1 .. plen+keep_gen-2).
        # HF generate() stores logits in float32 internally; cast back to model dtype.
        logit_rows: List[torch.Tensor] = []
        for s in range(min(keep_gen, len(logits_list))):
            logit_rows.append(logits_list[s][b].detach().cpu())
        if logit_rows:
            final_logits = torch.stack(logit_rows, dim=0).to(model_dtype)  # [G, vocab]
        else:
            final_logits = torch.empty((0,), dtype=model_dtype)

        out_refs.append(
            _HFRef(
                token_ids=tok_ids,
                final_logits=final_logits,
                hidden_states=hidden_states_per_layer,
                attn_pattern=attn_per_layer,
            )
        )

    return out_refs


# ---------------------------------------------------------------------------
# ROL reference: manual greedy KV-cache rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def _hf_greedy_rollout_collect_all_batched(
    *,
    hf_model: Any,
    input_ids_batch: torch.Tensor,
    attention_mask_batch: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device,
    want_hidden_states: bool = True,
    want_attentions: bool = True,
    compiled: bool = False,
) -> List[_HFRef]:
    """Manual greedy KV-cache rollout, batched with same left-padded inputs as monitoring.

    When compiled=True, uses StaticCache + torch.compile(mode="reduce-overhead",
    fullgraph=False) on the decode forward step, with cudagraph_mark_step_begin()
    before each step.  Hidden states / attentions are cloned via .detach().cpu()
    immediately after each step, before CUDA graph buffers are overwritten.

    Returns per-row refs with left-pad stripped and per-row EOS-trim:
      - token_ids:    [T]
      - final_logits: [T, vocab]
      - hidden_states per layer: [T, d]
      - attn_pattern per layer:  [H, T, T]
    """
    hf_model.eval()
    input_ids = input_ids_batch.to(device=device, dtype=torch.long).clone()
    attn = attention_mask_batch.to(device=device, dtype=torch.long).clone()

    B, Pmax = input_ids.shape
    prompt_lens_t = attn.sum(dim=1)
    pad_lens_t = Pmax - prompt_lens_t
    prompt_lens = prompt_lens_t.detach().cpu().tolist()
    pad_lens = pad_lens_t.detach().cpu().tolist()

    # --- Setup for compiled mode: StaticCache + compiled forward ---
    static_cache = None
    compiled_forward = None
    cache_position: Optional[torch.Tensor] = None
    if compiled:
        from transformers import StaticCache
        max_cache_len = Pmax + max_new_tokens + 4
        static_cache = StaticCache(
            config=hf_model.config, batch_size=B,
            max_cache_len=max_cache_len, device=device,
            dtype=torch.float16,
        )
        cache_position = torch.arange(Pmax, device=device)

        def _decode_step(inp, cache, cpos, attn_mask):
            return hf_model(
                input_ids=inp,
                attention_mask=attn_mask,
                cache_position=cpos,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=bool(want_hidden_states),
                output_attentions=bool(want_attentions),
                return_dict=True,
            )

        compiled_forward = torch.compile(
            _decode_step, mode="reduce-overhead", fullgraph=False,
        )

    # --- Prefill ---
    if compiled:
        # position_ids from attention_mask (same as generate() does internally).
        # cache_position is for cache indexing; position_ids is for embeddings.
        pos0 = _position_ids_from_attention_mask(attn)
        torch.compiler.cudagraph_mark_step_begin()
        out = hf_model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=pos0,
            cache_position=cache_position,
            past_key_values=static_cache,
            use_cache=True,
            output_hidden_states=bool(want_hidden_states),
            output_attentions=bool(want_attentions),
            return_dict=True,
            logits_to_keep=0,
        )
    else:
        pos0 = _position_ids_from_attention_mask(attn)
        out = _hf_forward_with_optional_position_ids(
            hf_model,
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=pos0,
            use_cache=True,
            output_hidden_states=bool(want_hidden_states),
            output_attentions=bool(want_attentions),
            return_dict=True,
            logits_to_keep=0,
        )
    past = out.past_key_values

    seq_ids: List[List[int]] = []
    logits_chunks_by_row: List[List[torch.Tensor]] = []
    hidden_chunks_by_row_by_layer: List[List[List[torch.Tensor]]] = []
    attn_chunks_by_row_by_layer: List[List[List[torch.Tensor]]] = []

    for b in range(B):
        pad_len = int(pad_lens[b])
        seq_ids.append(input_ids[b, pad_len:].detach().cpu().tolist())
        lp = out.logits[b, pad_len:, :].detach().cpu()
        logits_chunks_by_row.append([lp])

    if want_hidden_states and out.hidden_states is not None:
        n_hs = len(out.hidden_states)
        for b in range(B):
            pad_len = int(pad_lens[b])
            per_layer: List[List[torch.Tensor]] = []
            for l in range(n_hs):
                hs = out.hidden_states[l][b, pad_len:, :].detach().cpu()
                per_layer.append([hs])
            hidden_chunks_by_row_by_layer.append(per_layer)
    else:
        hidden_chunks_by_row_by_layer = [[] for _ in range(B)]

    if want_attentions and out.attentions is not None:
        n_attn = len(out.attentions)
        for b in range(B):
            pad_len = int(pad_lens[b])
            per_layer_a: List[List[torch.Tensor]] = []
            for l in range(n_attn):
                a0 = out.attentions[l][b]
                # normalize possible shapes: [1,H,T,T] -> [H,T,T]
                if a0.ndim == 4 and a0.shape[0] == 1:
                    a0 = a0.squeeze(0)
                # strip left-pad in both query and key dims
                a0 = a0[:, pad_len:, pad_len:].detach().cpu()
                per_layer_a.append([a0])
            attn_chunks_by_row_by_layer.append(per_layer_a)
    else:
        attn_chunks_by_row_by_layer = [[] for _ in range(B)]

    # --- Greedy decode loop ---
    unfinished = torch.ones((B,), dtype=torch.long, device=device)
    cur_out = out

    for _step in range(max_new_tokens):
        next_tokens = cur_out.logits[:, -1, :].argmax(dim=-1)  # [B]
        prev_unfinished = unfinished

        next_tokens = next_tokens * prev_unfinished + int(pad_token_id) * (1 - prev_unfinished)

        for b in range(B):
            if int(prev_unfinished[b].item()) == 1:
                seq_ids[b].append(int(next_tokens[b].item()))

        step_inp = next_tokens.view(B, 1)
        attn = torch.cat([attn, prev_unfinished[:, None]], dim=1)

        if compiled:
            cache_position = cache_position[-1:] + 1
            torch.compiler.cudagraph_mark_step_begin()
            cur_out = compiled_forward(step_inp, static_cache, cache_position, attn)
        else:
            pos_full = _position_ids_from_attention_mask(attn)
            pos_step = pos_full[:, -1].unsqueeze(1)
            cur_out = _hf_forward_with_optional_position_ids(
                hf_model,
                input_ids=step_inp,
                attention_mask=attn,
                position_ids=pos_step,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=bool(want_hidden_states),
                output_attentions=bool(want_attentions),
                return_dict=True,
            )
            past = cur_out.past_key_values

        # Clone immediately — CUDA graph buffers are overwritten on next step
        for b in range(B):
            if int(prev_unfinished[b].item()) != 1:
                continue

            # logits
            sl = cur_out.logits[b]
            if sl.ndim == 1:
                sl = sl.unsqueeze(0)
            elif sl.ndim == 2 and sl.shape[0] == 1:
                pass
            else:
                sl = sl.view(1, -1)
            logits_chunks_by_row[b].append(sl.detach().cpu())

            # hidden states
            if want_hidden_states and cur_out.hidden_states is not None:
                for l, hs in enumerate(cur_out.hidden_states):
                    hsb = hs[b]
                    if hsb.ndim == 1:
                        hsb = hsb.unsqueeze(0)
                    hidden_chunks_by_row_by_layer[b][l].append(hsb.detach().cpu())

            # attentions
            if want_attentions and cur_out.attentions is not None:
                pad_len = int(pad_lens[b])
                for l, a in enumerate(cur_out.attentions):
                    ab = a[b]
                    # normalize [H, K] -> [H, 1, K]
                    if ab.ndim == 2:
                        ab = ab.unsqueeze(1)
                    # strip left-pad on key dim
                    ab = ab[..., pad_len:]
                    attn_chunks_by_row_by_layer[b][l].append(ab.detach().cpu())

        unfinished = prev_unfinished * (next_tokens != int(eos_token_id)).to(torch.long)
        if int(unfinished.max().item()) == 0:
            break

    # Stitch per-row outputs.
    # Drop last token: it was predicted but never processed by a forward pass,
    # so monitoring DB won't have hidden states / attentions for it.
    out_refs: List[_HFRef] = []
    for b in range(B):
        tok_full = torch.tensor(seq_ids[b], dtype=torch.long, device="cpu")
        tok = tok_full[:-1] if tok_full.numel() > 0 else tok_full

        # logits: concat time, then drop last row
        lchunks = logits_chunks_by_row[b]
        lnorm: List[torch.Tensor] = []
        for t in lchunks:
            lnorm.append(t if t.ndim == 2 else t.unsqueeze(0))
        flog = torch.cat(lnorm, dim=0)
        if flog.shape[0] > 0:
            flog = flog[:-1, :]

        # hidden states: per layer concat time, then drop last row
        hfull: List[torch.Tensor] = []
        if hidden_chunks_by_row_by_layer[b]:
            for per_layer in hidden_chunks_by_row_by_layer[b]:
                nn: List[torch.Tensor] = []
                for t in per_layer:
                    nn.append(t if t.ndim == 2 else t.unsqueeze(0))
                h = torch.cat(nn, dim=0)
                if h.shape[0] > 0:
                    h = h[:-1, :]
                hfull.append(h)

        # attentions: per layer stitch, then trim last query row and key col
        afull: List[torch.Tensor] = []
        if attn_chunks_by_row_by_layer[b]:
            for per_layer in attn_chunks_by_row_by_layer[b]:
                merged = merge_segments(per_layer, "blocks.attn.hook_pattern")
                if merged is None:
                    merged = torch.empty((0,), dtype=torch.float32)
                elif merged.ndim >= 2:
                    # [H, T, T] -> [H, T-1, T-1]
                    merged = merged[..., :-1, :-1]
                afull.append(merged)

        out_refs.append(
            _HFRef(token_ids=tok, final_logits=flog, hidden_states=hfull, attn_pattern=afull)
        )

    return out_refs