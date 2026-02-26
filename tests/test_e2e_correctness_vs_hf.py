# tests/test_e2e_correctness_vs_hf.py
# PYTHONPATH=./:./monitoring:$PYTHONPATH E2E_PRINT_TEXT=1 E2E_HF_DROP_LAST_TOKEN=1 E2E_PRINT_TOPK_LOGITS=1 pytest -q -s tests/test_e2e_correctness_vs_hf.py
"""E2E correctness test: monitoring DB vs HuggingFace Transformers (HF-driven ground truth).

This test runs the repo monitoring pipeline end-to-end (native backend + host engine + ClickHouse),
then uses HuggingFace Transformers as the reference implementation (no TransformerLens).

IMPORTANT: "prompt" in this test means FULL TOKEN SEQUENCE = prefill + decode
-----------------------------------------------------------------------
We treat DB `token_ids` as the ground-truth sequence of tokens for each request. That sequence
includes the initial prompt tokens (prefill) plus any decode tokens that were appended.

HF reference in THIS VERSION
-----------------------------------------------------------------------
We run HF references using the SAME batching + left-padding inputs as the monitored run:

  - ROL (manual rollout, batched):
      * Incremental greedy KV-cache rollout over the full padded batch, using generate-style
        position_ids derived from attention_mask (so left-padding doesn't shift positions).
      * We strip left-pad per row and stop per row at EOS (no trailing padded steps), so outputs
        align to DB token_ids.

  - GEN (HF generate(), batched):
      * Run hf_model.generate() ONCE on the full padded batch (same input_ids/attention_mask as monitoring),
        with output_scores=True.
      * Strip left-pad per row and trim at EOS so sequences align to DB token_ids.

For logits we effectively compare THREE sources:
  - DB final_logits (from ClickHouse)
  - ROL final_logits (manual rollout; [T, vocab])
  - GEN scores (from generate(); available only for positions t in [prompt_len-1, T-2])

Request-id convention (from MonitoringEngine._register_db_step in engine.py)
-----------------------------------------------------------------------
When a new batch is (re)initialized:

    gid = self._auto_batch_group_id
    self._auto_batch_group_id += 1
    self._active_batch_request_ids = [f"{gid}:{i}" for i in range(batch_size)]

So request_id == "<group_id>:<local_index>" where local_index is the batch row index.

We map DB requests to HF batch rows using local_index, and add safety checks.

DB tensor shapes (IMPORTANT)
-----------------------------------------------------------------------
DB offloaded tensors DO NOT have batch dim now:
  - token_ids:                 [T]
  - hook_embed/pos/final_ln:   [T, d_model]
  - resid_pre/post:            [T, d_model]
  - attn pattern/scores:       [n_heads, Tq, Tk]
  - final_logits:              [R, vocab] or [vocab]  (R is often full sequence length)

What we compare (HF-available / reconstructible)
-----------------------------------------------------------------------
Given DB token_ids per request, we run an HF incremental rollout and compare:

  - token_ids:
      * HF rollout token_ids must equal DB token_ids for that request.
      * HF generate() token_ids must equal DB token_ids for that request.
  - final_logits:
      * Compare DB logits rows for the final sequence (length T) to HF rollout logits [T, vocab].
      * Also print GEN top-k logits where available (decode-step scores).
  - GPT2-like only (transformer.wte/wpe/ln_f exist):
      * hook_embed:    wte(token_ids)
      * hook_pos_embed wpe(position_ids) with positions 0..T-1
      * hook_final_ln  ln_f(last_hidden_state)
  - attentions:
      * blocks.attn.hook_pattern compare to HF attentions (probabilities) stitched to [H, T, T]
  - hidden states:
      * blocks.hook_resid_pre  compare to HF hidden_states[layer] stitched to [T, d]
      * blocks.hook_resid_post compare to HF hidden_states[layer+1] stitched to [T, d]

Env vars
-----------------------------------------------------------------------
  - E2E_BATCH_SIZE (default 4)
  - E2E_MAX_NEW_TOKENS (default 8)
  - E2E_MODEL (default "gpt2"; "qwen3" alias supported)
  - E2E_CHUNK_BYTES (default 262144)
  - E2E_NO_DB (default 0)

  - E2E_PRINT_TEXT (default 0): if 1, print decoded text from DB token_ids and from HF rollout + HF generate().
  - E2E_HF_DROP_LAST_TOKEN (default 0): if 1, drop the last token (and aligned tensors) from HF refs before compares.
  - E2E_PRINT_TOPK_LOGITS (default 0): if 1, print top-k logits at every position for DB vs ROL vs GEN.
  - E2E_PRINT_TOPK_LOGITS_K (default 5): top-k to print per position.

ClickHouse
-----------------------------------------------------------------------
  - DMX_DB_HOST, DMX_DB_PORT, DMX_DB_USER, DMX_DB_PASSWORD, DMX_DB_DATABASE, DMX_DB_TABLE
"""

from __future__ import annotations

import ast
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytest
import torch


# -----------------------------
# Model aliases (match benchmark)
# -----------------------------

_MODEL_ALIASES = {"qwen3": "Qwen/Qwen3-4B"}


def _resolve_model_id(model: str) -> str:
    return _MODEL_ALIASES.get(model.lower(), model)


def _is_attn_scores(act_name: str) -> bool:
    return act_name.endswith("attn.hook_attn_scores")


def _is_attn_pattern(act_name: str) -> bool:
    return act_name.endswith("attn.hook_pattern")


# -----------------------------
# DB decoding (v1/v2)
# -----------------------------

BytesLike = Union[bytes, bytearray, memoryview]

TORCH_DTYPES_NAME2TYPE: Dict[str, torch.dtype] = {
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float64": torch.float64,
    "torch.double": torch.float64,
    "torch.float16": torch.float16,
    "torch.half": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.short": torch.int16,
    "torch.int32": torch.int32,
    "torch.int": torch.int32,
    "torch.int64": torch.int64,
    "torch.long": torch.int64,
    "torch.bool": torch.bool,
}

def get_num_layers_from_config(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model has no .config")

    # common names across HF models
    for attr in ("num_hidden_layers", "n_layer", "n_layers", "num_layers", "num_attention_layers"):
        if hasattr(cfg, attr):
            v = int(getattr(cfg, attr))
            if v > 0:
                return v

    raise ValueError(f"could not find num layers in config attrs={dir(cfg)}")

def _to_bytes(v: Any) -> bytes:
    if v is None:
        raise TypeError("cannot convert None to bytes")
    if isinstance(v, bytes):
        return v
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, bytearray):
        return bytes(v)
    # ClickHouse may return Array(UInt8) as list[int]
    if isinstance(v, (list, tuple)) and (not v or isinstance(v[0], int)):
        return bytes(int(x) & 0xFF for x in v)
    if isinstance(v, str):
        return v.encode("utf-8", errors="surrogateescape")
    raise TypeError(f"expected bytes-like, got {type(v)!r}")


def torch_decode_v1(meta: BytesLike, payload: Any) -> torch.Tensor:
    meta_b = _to_bytes(meta)
    meta_json = json.loads(meta_b.decode("utf-8", errors="surrogateescape"))
    dtype_str = meta_json["dtype"]
    shape = tuple(int(x) for x in meta_json["shape"])
    dtype = TORCH_DTYPES_NAME2TYPE[dtype_str]
    data_b = _to_bytes(payload)
    return torch.frombuffer(data_b, dtype=dtype).clone().view(*shape)


def torch_decode_v2(dtype_cell: Any, shape_cell: Any, payload: Any) -> torch.Tensor:
    if isinstance(dtype_cell, (bytes, bytearray, memoryview)):
        dtype_str = dtype_cell.decode("utf-8", errors="surrogateescape")
    else:
        dtype_str = str(dtype_cell)

    if isinstance(shape_cell, (bytes, bytearray, memoryview)):
        shape_raw = shape_cell.decode("utf-8", errors="surrogateescape")
        try:
            shape = tuple(int(x) for x in ast.literal_eval(shape_raw))
        except Exception:
            shape = tuple(int(x) for x in re.findall(r"\d+", shape_raw))
    elif isinstance(shape_cell, (list, tuple)):
        shape = tuple(int(x) for x in shape_cell)
    else:
        shape_raw = str(shape_cell)
        try:
            shape = tuple(int(x) for x in ast.literal_eval(shape_raw))
        except Exception:
            shape = tuple(int(x) for x in re.findall(r"\d+", shape_raw))

    dtype = TORCH_DTYPES_NAME2TYPE[dtype_str]
    data_b = _to_bytes(payload)
    return torch.frombuffer(data_b, dtype=dtype).clone().view(*shape)


@dataclass
class _DBConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    table: str
    secure: bool = False


class _ClickHouseReader:
    def __init__(self, cfg: _DBConfig):
        from clickhouse_driver import Client as CHClient

        settings = {"strings_as_bytes": 1}
        self._client = CHClient(
            host=cfg.host,
            port=int(cfg.port),
            user=cfg.username,
            password=cfg.password,
            database=cfg.database,
            secure=bool(cfg.secure),
            settings=settings,
        )
        self._db = cfg.database
        self._table = cfg.table
        self._columns = self._describe_columns()
        self._value_cols = self._infer_value_columns(self._columns)
        self._has_shard_rank = "shard_rank" in self._columns

    def _describe_columns(self) -> List[str]:
        rows = self._client.execute(f"DESCRIBE TABLE {self._db}.{self._table}")
        cols: List[str] = []
        for r in rows:
            if not r:
                continue
            c = r[0]
            if isinstance(c, (bytes, bytearray, memoryview)):
                c = c.decode("utf-8", errors="surrogateescape")
            else:
                c = str(c)
            if c.startswith("b'") and c.endswith("'"):
                c = c[2:-1]
            cols.append(c)
        return cols

    @staticmethod
    def _infer_value_columns(columns: Sequence[str]) -> Tuple[str, ...]:
        if "json" in columns and "bytes" in columns:
            return ("json", "bytes")
        if "dtype" in columns and "shape" in columns and "bytes" in columns:
            return ("dtype", "shape", "bytes")
        raise RuntimeError(f"Could not infer value columns. Columns present: {list(columns)}")

    def fetch_all_rows_for_model(self, *, model_id: str):
        shard_expr = "shard_rank" if self._has_shard_rank else "0"
        value_select = ", ".join(self._value_cols)
        q = (
            "SELECT model_id, request_id, act_name, layer_no, start_token_idx, end_token_idx, "
            f"{shard_expr} as shard_rank, {value_select} "
            f"FROM {self._db}.{self._table} "
            "WHERE model_id = %(model_id)s "
            "ORDER BY request_id, act_name, layer_no, start_token_idx, end_token_idx"
        )
        rows = self._client.execute(q, {"model_id": model_id})
        out = []
        for r in rows:
            _model_raw, req_raw, act_raw, layer_no, s, e, shard_rank, *vals = r
            req_id = (
                req_raw.decode("utf-8", errors="surrogateescape")
                if isinstance(req_raw, (bytes, bytearray, memoryview))
                else str(req_raw)
            )
            act_name = (
                act_raw.decode("utf-8", errors="surrogateescape")
                if isinstance(act_raw, (bytes, bytearray, memoryview))
                else str(act_raw)
            )
            out.append((req_id, act_name, int(layer_no), int(s), int(e), int(shard_rank), vals))
        return out

    def decode_tensor(self, vals: Sequence[Any]) -> torch.Tensor:
        if len(vals) == 2:
            meta, payload = vals
            return torch_decode_v1(meta, payload)
        if len(vals) == 3:
            dtype_cell, shape_cell, payload = vals
            return torch_decode_v2(dtype_cell, shape_cell, payload)
        raise RuntimeError(f"Unexpected value columns length: {len(vals)}")


# -----------------------------
# Merge DB chunks (NO batch dim)
# -----------------------------


class _OnDimSegments:
    """Non-attn: concat along token dimension 0 (DB tensors are [T, ...])."""

    def __init__(self, token_dim: int = 0):
        self._token_dim = token_dim
        self._chunks: List[torch.Tensor] = []

    def extend(self, chunks: Sequence[torch.Tensor]) -> None:
        self._chunks.extend(list(chunks))

    def read_and_merge(self) -> torch.Tensor:
        return torch.cat(self._chunks, dim=self._token_dim)


class _AttnMatrixSegments:
    """Attn matrices: expected chunk shape [H, q_chunk, k_up_to_now] (no batch)."""

    def __init__(self, fill_value: float):
        self._fill_value = float(fill_value)
        self._chunks: List[torch.Tensor] = []

    def extend(self, chunks: Sequence[torch.Tensor]) -> None:
        for t in chunks:
            # defensive: if some path still writes [1,H,Q,K]
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            self._chunks.append(t)

    def read_and_merge(self) -> torch.Tensor:
        td_inc, td_sum = 1, 2  # [H, Q, K]
        total_k = int(self._chunks[-1].shape[td_sum])
        padded: List[torch.Tensor] = []
        for t in self._chunks:
            if int(t.shape[td_sum]) > total_k:
                t = t.narrow(td_sum, 0, total_k)
            pad_len = total_k - int(t.shape[td_sum])
            if pad_len > 0:
                pad_shape = list(t.shape)
                pad_shape[td_sum] = pad_len
                pad_t = torch.full(pad_shape, self._fill_value, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad_t], dim=td_sum)
            padded.append(t)

        merged = torch.cat(padded, dim=td_inc)
        if int(merged.shape[td_inc]) > total_k:
            merged = merged.narrow(td_inc, 0, total_k)
        return merged


def merge_segments(chunks: Sequence[torch.Tensor], act_name: str) -> torch.Tensor:
    if _is_attn_scores(act_name):
        mgr = _AttnMatrixSegments(fill_value=float("-inf"))
        mgr.extend(chunks)
        return mgr.read_and_merge()
    if _is_attn_pattern(act_name):
        mgr = _AttnMatrixSegments(fill_value=0.0)
        mgr.extend(chunks)
        return mgr.read_and_merge()
    mgr2 = _OnDimSegments(token_dim=0)
    mgr2.extend(chunks)
    return mgr2.read_and_merge()


# -----------------------------
# HF helpers
# -----------------------------


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


def _hf_forward_with_optional_position_ids(
    hf_model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    **kwargs: Any,
):
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
        if "position_ids" in msg and ("unexpected keyword argument" in msg or "got an unexpected keyword argument" in msg):
            return hf_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        raise


def _positions_for_unpadded(true_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(true_len, device=device, dtype=torch.long)


def _parse_request_id(req_id: str) -> Tuple[int, int]:
    """Parse '<group_id>:<local_index>'."""
    m = re.match(r"^(\d+):(\d+)$", req_id)
    if not m:
        raise AssertionError(f"unexpected request_id format: {req_id!r}")
    return int(m.group(1)), int(m.group(2))


@dataclass
class _HFRef:
    token_ids: torch.Tensor  # [T] cpu long
    final_logits: torch.Tensor  # [T, vocab] cpu
    hidden_states: List[torch.Tensor]  # [n_layer+1] each [T, d] cpu
    attn_pattern: List[torch.Tensor]  # [n_layer] each [H, T, T] cpu


@dataclass
class _HFGenRef:
    token_ids: torch.Tensor  # [T] cpu long (prompt+gen)
    # generate() exposes per-step scores for generated tokens only.
    # scores[s] is the distribution used to pick generated token at step s (0-indexed),
    # shape [vocab] on CPU.
    scores: List[torch.Tensor]


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
) -> List[_HFGenRef]:
    """Run HF generate() ONCE on the full padded batch; return per-row (unpadded, EOS-trimmed) refs."""
    hf_model.eval()
    input_ids = input_ids_batch.to(device=device, dtype=torch.long)
    attn = attention_mask_batch.to(device=device, dtype=torch.long)

    B, Pmax = input_ids.shape

    gen_out = hf_model.generate(
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

    seqs = gen_out.sequences.detach().cpu().to(torch.long)  # [B, Pmax+G]
    scores_steps: List[torch.Tensor] = []
    if getattr(gen_out, "scores", None) is not None:
        # tuple/list of length gen_len_global, each [B, vocab]
        for s in gen_out.scores:
            scores_steps.append(s.detach().cpu())

    prompt_lens_t = attn.sum(dim=1)  # [B]
    pad_lens_t = (Pmax - prompt_lens_t)  # [B]
    prompt_lens = prompt_lens_t.detach().cpu().tolist()
    pad_lens = pad_lens_t.detach().cpu().tolist()

    out_refs: List[_HFGenRef] = []
    for b in range(B):
        pad_len = int(pad_lens[b])
        prompt_len = int(prompt_lens[b])
        _ = prompt_len  # (used implicitly by slicing)

        # prompt part (strip left pad using known pad_len)
        prompt_tok = seqs[b, pad_len:Pmax]  # [prompt_len]

        # generated part
        gen_tok_full = seqs[b, Pmax:]  # [G]
        gen_len = int(gen_tok_full.numel())
        if gen_len > 0:
            eos_hits = (gen_tok_full == int(eos_token_id)).nonzero(as_tuple=False)
            if eos_hits.numel() > 0:
                gen_len = int(eos_hits[0].item()) + 1  # keep EOS
        gen_tok = gen_tok_full[:gen_len]

        tok_ids = torch.cat([prompt_tok, gen_tok], dim=0).detach().cpu().to(torch.long)

        row_scores: List[torch.Tensor] = []
        # scores are step-aligned to generated tokens (decode only)
        for s in range(min(gen_len, len(scores_steps))):
            row_scores.append(scores_steps[s][b].detach().cpu())

        out_refs.append(_HFGenRef(token_ids=tok_ids, scores=row_scores))

    return out_refs


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
) -> List[_HFRef]:
    """
    Manual greedy KV-cache rollout, but run batched with the same (left-padded) inputs as monitoring.

    Returns per-row refs with left-pad stripped and per-row EOS-trim (no trailing pad steps):
      - token_ids:    [T]
      - final_logits: [T, vocab]
      - hidden_states per layer: [T, d]
      - attn_pattern per layer:  [H, T, T]
    """
    hf_model.eval()
    input_ids = input_ids_batch.to(device=device, dtype=torch.long).clone()
    attn = attention_mask_batch.to(device=device, dtype=torch.long).clone()

    B, Pmax = input_ids.shape
    prompt_lens_t = attn.sum(dim=1)  # [B]
    pad_lens_t = (Pmax - prompt_lens_t)  # [B]
    prompt_lens = prompt_lens_t.detach().cpu().tolist()
    pad_lens = pad_lens_t.detach().cpu().tolist()

    # Prefill (use generate-style position_ids for left padding if supported)
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

    # Per-row collectors (variable length)
    seq_ids: List[List[int]] = []
    logits_chunks_by_row: List[List[torch.Tensor]] = []

    hidden_chunks_by_row_by_layer: List[List[List[torch.Tensor]]] = []
    attn_chunks_by_row_by_layer: List[List[List[torch.Tensor]]] = []

    # init prompt pieces (strip left pad via pad_len slice)
    for b in range(B):
        pad_len = int(pad_lens[b])
        _prompt_len = int(prompt_lens[b])
        _ = _prompt_len  # (used implicitly by slicing)

        seq_ids.append(input_ids[b, pad_len:].detach().cpu().tolist())

        # logits prompt slice: [prompt_len, vocab]
        lp = out.logits[b, pad_len:, :].detach().cpu()
        logits_chunks_by_row.append([lp])

    if want_hidden_states and out.hidden_states is not None:
        n_hs = len(out.hidden_states)
        for b in range(B):
            pad_len = int(pad_lens[b])
            per_layer: List[List[torch.Tensor]] = []
            for l in range(n_hs):
                hs = out.hidden_states[l][b, pad_len:, :].detach().cpu()  # [prompt_len,d]
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
                a0 = out.attentions[l][b]  # [H,Pmax,Pmax] (or similar)
                if a0.ndim == 4 and a0.shape[0] == 1:
                    a0 = a0.squeeze(0)
                a0 = a0[:, pad_len:, pad_len:].detach().cpu()  # [H,prompt_len,prompt_len]
                per_layer_a.append([a0])
            attn_chunks_by_row_by_layer.append(per_layer_a)
    else:
        attn_chunks_by_row_by_layer = [[] for _ in range(B)]

    # Greedy decode loop (HF-generate-like: finished rows get pad + mask=0 thereafter)
    unfinished = torch.ones((B,), dtype=torch.long, device=device)
    cur_out = out

    for _step in range(max_new_tokens):
        next_tokens = cur_out.logits[:, -1, :].argmax(dim=-1)  # [B]
        prev_unfinished = unfinished

        # force pad for already-finished rows
        next_tokens = next_tokens * prev_unfinished + int(pad_token_id) * (1 - prev_unfinished)

        # append token ids ONLY for rows that were unfinished entering this step
        for b in range(B):
            if int(prev_unfinished[b].item()) == 1:
                seq_ids[b].append(int(next_tokens[b].item()))

        # attention_mask: eos token itself should be visible (mask=1), so append prev_unfinished
        attn = torch.cat([attn, prev_unfinished[:, None]], dim=1)

        # position_ids for the step token (shape [B,1])
        pos_full = _position_ids_from_attention_mask(attn)
        pos_step = pos_full[:, -1].unsqueeze(1)

        step_inp = next_tokens.view(B, 1)
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

        # collect per-row tensors only while row is "real" this step
        for b in range(B):
            if int(prev_unfinished[b].item()) != 1:
                continue

            # logits: ensure [1,vocab]
            sl = cur_out.logits[b]
            if sl.ndim == 2 and sl.shape[0] == 1:
                sl = sl
            elif sl.ndim == 1:
                sl = sl.unsqueeze(0)
            else:
                sl = sl.view(1, -1)
            logits_chunks_by_row[b].append(sl.detach().cpu())

            if want_hidden_states and cur_out.hidden_states is not None:
                for l, hs in enumerate(cur_out.hidden_states):
                    hsb = hs[b]
                    if hsb.ndim == 1:
                        hsb = hsb.unsqueeze(0)
                    hidden_chunks_by_row_by_layer[b][l].append(hsb.detach().cpu())

            if want_attentions and cur_out.attentions is not None:
                pad_len = int(pad_lens[b])
                for l, a in enumerate(cur_out.attentions):
                    ab = a[b]  # [H,1,cur_len] typically
                    if ab.ndim == 2:
                        ab = ab.unsqueeze(1)
                    # drop initial left-pad keys => [H,1,prompt_len+step]
                    ab = ab[..., pad_len:]
                    attn_chunks_by_row_by_layer[b][l].append(ab.detach().cpu())

        # update unfinished AFTER including EOS token
        unfinished = prev_unfinished * (next_tokens != int(eos_token_id)).to(torch.long)
        if int(unfinished.max().item()) == 0:
            break

    # stitch per-row outputs
    out_refs: List[_HFRef] = []
    for b in range(B):
        tok = torch.tensor(seq_ids[b], dtype=torch.long, device="cpu")

        # logits => [T,vocab]
        lchunks = logits_chunks_by_row[b]
        lnorm: List[torch.Tensor] = []
        for t in lchunks:
            lnorm.append(t if t.ndim == 2 else t.unsqueeze(0))
        flog = torch.cat(lnorm, dim=0)

        # hidden => [n_layer+1][T,d]
        hfull: List[torch.Tensor] = []
        if hidden_chunks_by_row_by_layer[b]:
            for per_layer in hidden_chunks_by_row_by_layer[b]:
                nn: List[torch.Tensor] = []
                for t in per_layer:
                    nn.append(t if t.ndim == 2 else t.unsqueeze(0))
                hfull.append(torch.cat(nn, dim=0))

        # attn => [n_layer][H,T,T]
        afull: List[torch.Tensor] = []
        if attn_chunks_by_row_by_layer[b]:
            for per_layer in attn_chunks_by_row_by_layer[b]:
                mgr = _AttnMatrixSegments(fill_value=0.0)
                mgr.extend(per_layer)
                afull.append(mgr.read_and_merge())

        out_refs.append(_HFRef(token_ids=tok, final_logits=flog, hidden_states=hfull, attn_pattern=afull))

    return out_refs


# -----------------------------
# Test
# -----------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA + native backend required")
def test_e2e_correctness_vs_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import clickhouse_driver  # noqa: F401
    except Exception:
        pytest.skip("clickhouse-driver is required")

    if int(os.environ.get("E2E_NO_DB", "0")) == 1:
        pytest.skip("E2E_NO_DB=1; DB is required")

    # native monitoring extension
    try:
        from monitoring import (  # type: ignore
            ClickHouseClientConfig,
            HostEngineConfig,
            MonitoringConfig,
            MonitoringEngine,
            NativePartialSealConfig,
            StageConfig,
        )
        from monitoring.config import CaptureSchedule, HookSelection  # type: ignore
        from monitoring.generate import generate_with_monitoring  # type: ignore
    except Exception as exc:
        pytest.skip(f"monitoring native extension not available: {exc}")

    # repo-hooked model classes for the monitored run
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel  # type: ignore
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
    except Exception as exc:
        pytest.skip(f"transformers or repo Hooked* classes not available: {exc}")

    batch_size = int(os.environ.get("E2E_BATCH_SIZE", "4"))
    if batch_size < 1:
        raise ValueError("E2E_BATCH_SIZE must be >= 1")
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "8"))
    model_id = _resolve_model_id(os.environ.get("E2E_MODEL", "gpt2"))
    chunk_bytes = int(os.environ.get("E2E_CHUNK_BYTES", str(256 * 1024)))

    print_text = int(os.environ.get("E2E_PRINT_TEXT", "0")) == 1
    hf_drop_last_token = int(os.environ.get("E2E_HF_DROP_LAST_TOKEN", "0")) == 1

    print_topk_logits = int(os.environ.get("E2E_PRINT_TOPK_LOGITS", "0")) == 1
    topk_k = int(os.environ.get("E2E_PRINT_TOPK_LOGITS_K", "5"))
    if topk_k < 1:
        raise ValueError("E2E_PRINT_TOPK_LOGITS_K must be >= 1")

    # native env toggles (bench-like)
    monkeypatch.setenv("MON_NATIVE_TO_CPU", os.environ.get("MON_NATIVE_TO_CPU", "1"))
    monkeypatch.setenv("MON_NATIVE_CALLBACK", os.environ.get("MON_NATIVE_CALLBACK", "1"))
    monkeypatch.setenv("MON_NATIVE_BUILDER", os.environ.get("MON_NATIVE_BUILDER", "1"))
    monkeypatch.setenv("MON_NATIVE_BATCH", os.environ.get("MON_NATIVE_BATCH", "0"))
    monkeypatch.setenv("MON_NATIVE_AUTOCLEAR", os.environ.get("MON_NATIVE_AUTOCLEAR", "0"))

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    # Nested prompts to exercise left padding in the monitored run.
    prompts = [("Hello " * (i + 1)).strip() for i in range(batch_size)]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Initial prompt tokens for safety prefix checks (prompt-only)
    hf_initial_prompt_tokens: List[torch.Tensor] = []
    for j in range(batch_size):
        hf_initial_prompt_tokens.append(
            _strip_left_pad(input_ids[j].detach().cpu(), attention_mask[j].detach().cpu()).to(torch.long)
        )

    mon_cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
        native_partial_seal=NativePartialSealConfig(
            enabled=True,
            chunk_bytes=int(chunk_bytes),
            cap_enabled=True,
            cap_ratio=0.8,
            driver_guard_mb=1024,
        ),
    )
    if hasattr(mon_cfg, "eos_token_id"):
        mon_cfg.eos_token_id = eos_id
    if hasattr(mon_cfg, "pad_token_id"):
        mon_cfg.pad_token_id = pad_id

    # ClickHouse config (native)
    db_cfg_native = ClickHouseClientConfig()
    db_cfg_native.host = os.environ.get("DMX_DB_HOST", "localhost")
    db_cfg_native.port = int(os.environ.get("DMX_DB_PORT", "9000"))
    db_cfg_native.username = os.environ.get("DMX_DB_USER", "default")
    db_cfg_native.password = os.environ.get("DMX_DB_PASSWORD", "")
    db_cfg_native.database = os.environ.get("DMX_DB_DATABASE", "default")
    db_cfg_native.table = os.environ.get("DMX_DB_TABLE", "offload")
    db_cfg_native.secure = False
    db_cfg_native.client_side_compress = "none"
    db_cfg_native.client_settings = None
    db_cfg_native.create_database_if_missing = True
    db_cfg_native.drop_existing_database = True
    db_cfg_native.index_granularity = 8192

    stage_one = StageConfig.process_future(parallelism=1, name="process_future")
    stage_two = StageConfig.clickhouse_insert(db_cfg_native, parallelism=10, name="clickhouse_insert")
    host_cfg = HostEngineConfig(stages=[stage_one, stage_two])

    unique_model_id = f"e2e_correctness_vs_hf::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(async_enabled=True, config=mon_cfg, model_id=unique_model_id, db_config=host_cfg)

    # Monitored run (fp16 + eager)
    model_cls = HookedQwen3ForCausalLM if "qwen3" in model_id.lower() else HookedGPT2LMHeadModel
    mon_model = model_cls.from_pretrained(model_id, attn_implementation="eager", torch_dtype=torch.float16)
    mon_model.to(device).eval()
    mon_model.monitoring_engine = engine
    engine.prepare_for_model(mon_model)

    try:
        with torch.no_grad():
            _ = generate_with_monitoring(
                mon_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                logits_to_keep=0,
            )
    finally:
        engine.close()

    # Read DB
    reader = _ClickHouseReader(
        _DBConfig(
            host=str(db_cfg_native.host),
            port=int(db_cfg_native.port),
            username=str(db_cfg_native.username),
            password=str(db_cfg_native.password),
            database=str(db_cfg_native.database),
            table=str(db_cfg_native.table),
            secure=bool(getattr(db_cfg_native, "secure", False)),
        )
    )
    rows = reader.fetch_all_rows_for_model(model_id=unique_model_id)
    if not rows:
        pytest.fail(f"No rows found in ClickHouse for model_id={unique_model_id}")

    grouped: Dict[str, Dict[Tuple[int, str], List[Tuple[int, int, torch.Tensor]]]] = {}
    for req_id, act_name, layer_no, s, e, _shard_rank, vals in rows:
        t = reader.decode_tensor(vals).cpu()
        grouped.setdefault(req_id, {}).setdefault((int(layer_no), str(act_name)), []).append((int(s), int(e), t))

    request_ids = sorted(grouped.keys(), key=_parse_request_id)

    def _sort_chunks(chunks: List[Tuple[int, int, torch.Tensor]]) -> List[Tuple[int, int, torch.Tensor]]:
        return sorted(chunks, key=lambda x: (x[0], x[1]))

    def _validate_contiguous(chunks_sorted: List[Tuple[int, int, torch.Tensor]], expected_end: int, ctx: str) -> None:
        if not chunks_sorted:
            raise AssertionError(f"{ctx}: no chunks")
        if chunks_sorted[0][0] != 0:
            raise AssertionError(f"{ctx}: first chunk start={chunks_sorted[0][0]} expected 0")
        prev_end = chunks_sorted[0][1]
        for s2, e2, _t in chunks_sorted[1:]:
            if s2 != prev_end:
                raise AssertionError(f"{ctx}: non-contiguous chunks: start={s2} prev_end={prev_end}")
            prev_end = e2
        if prev_end != expected_end:
            raise AssertionError(f"{ctx}: coverage end={prev_end} expected_end={expected_end}")

    # Safety: this run should have exactly one group_id (single batch reset)
    seen_group_ids: set[int] = set()

    # Merge DB token_ids per request; map req_id -> local index
    db_token_ids_by_req: Dict[str, torch.Tensor] = {}
    local_index_by_req: Dict[str, int] = {}
    prompt_len_by_req: Dict[str, int] = {}

    for req_id in request_ids:
        group_id, local_i = _parse_request_id(req_id)
        seen_group_ids.add(group_id)
        if not (0 <= local_i < batch_size):
            raise AssertionError(f"{req_id}: local_index={local_i} out of range batch_size={batch_size}")
        local_index_by_req[req_id] = local_i

        hooks_map = grouped[req_id]
        if (-1, "token_ids") not in hooks_map:
            raise AssertionError(f"{req_id}: DB missing token_ids")

        tok_chunks = _sort_chunks(hooks_map[(-1, "token_ids")])
        db_tok_end = int(tok_chunks[-1][1])
        _validate_contiguous(tok_chunks, expected_end=db_tok_end, ctx=f"{req_id} token_ids")

        db_tok = merge_segments([t for _, _, t in tok_chunks], "token_ids").to(torch.long)
        if db_tok.ndim != 1:
            db_tok = db_tok.view(-1)

        # Safety check: initial prompt tokens must be a prefix of DB token_ids for this local index.
        prompt0 = hf_initial_prompt_tokens[local_i]
        plen0 = int(prompt0.numel())
        if db_tok.numel() < plen0 or not torch.equal(db_tok[:plen0], prompt0):
            raise AssertionError(
                f"{req_id}: request_id->row mapping safety check failed (initial prompt prefix). "
                f"initial_prompt_len={plen0} db_tok_len={db_tok.numel()}"
            )

        db_token_ids_by_req[req_id] = db_tok.cpu()
        prompt_len_by_req[req_id] = plen0

    if len(seen_group_ids) != 1:
        raise AssertionError(f"expected exactly one group_id for this test run, got {sorted(seen_group_ids)}")

    # Build HF references:
    #   - ROL: manual rollout (full [T, vocab], [T, d], [H, T, T])
    #   - GEN: HF generate() token_ids + output_scores (decode-only logits)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    ).to(device).eval()

    # Optional: modules for GPT2-like reconstructions
    wte = getattr(getattr(hf_model, "transformer", None), "wte", None)
    wpe = getattr(getattr(hf_model, "transformer", None), "wpe", None)
    ln_f = getattr(getattr(hf_model, "transformer", None), "ln_f", None)

    req_order = sorted(request_ids, key=_parse_request_id)

    hf_ref_by_req: Dict[str, _HFRef] = {}
    hf_gen_by_req: Dict[str, _HFGenRef] = {}

    def _decode(ids: torch.Tensor) -> str:
        return tokenizer.decode(ids.tolist(), skip_special_tokens=False)

    # ===== Batched HF refs (same padding/batching as monitoring run) =====
    hf_refs_batch = _hf_greedy_rollout_collect_all_batched(
        hf_model=hf_model,
        input_ids_batch=input_ids,
        attention_mask_batch=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        device=device,
        want_hidden_states=True,
        want_attentions=True,
    )
    hf_gens_batch = _hf_generate_collect_scores_batched(
        hf_model=hf_model,
        input_ids_batch=input_ids,
        attention_mask_batch=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        device=device,
    )
    if len(hf_refs_batch) != batch_size or len(hf_gens_batch) != batch_size:
        raise AssertionError(
            f"HF batched refs unexpected batch: rollout={len(hf_refs_batch)} gen={len(hf_gens_batch)} batch_size={batch_size}"
        )

    for req_id in req_order:
        i = local_index_by_req[req_id]
        plen = int(prompt_len_by_req[req_id])

        ref = hf_refs_batch[i]
        gen_ref = hf_gens_batch[i]

        if hf_drop_last_token:
            # Drop last token from ROL (and aligned tensors)
            if int(ref.token_ids.numel()) > 0:
                T = int(ref.token_ids.numel())
                ref = _HFRef(
                    token_ids=ref.token_ids[:-1],
                    final_logits=ref.final_logits[:-1, :],
                    hidden_states=[h[:-1, :] for h in ref.hidden_states],
                    attn_pattern=[a[:, : T - 1, : T - 1] for a in ref.attn_pattern],
                )

            # Drop last token from GEN; also trim scores to new generated length
            if int(gen_ref.token_ids.numel()) > 0:
                P = int(plen)
                new_seq = gen_ref.token_ids[:-1]
                new_gen_len = max(0, int(new_seq.numel()) - P)
                gen_ref = _HFGenRef(
                    token_ids=new_seq,
                    scores=gen_ref.scores[:new_gen_len],
                )

        hf_ref_by_req[req_id] = ref
        hf_gen_by_req[req_id] = gen_ref

        # Token equality checks against DB
        db_seq = db_token_ids_by_req[req_id].to(torch.long)
        if not torch.equal(ref.token_ids, db_seq):
            raise AssertionError(
                f"{req_id}: HF rollout tokens != DB token_ids "
                f"(hf_len={int(ref.token_ids.numel())} db_len={int(db_seq.numel())})"
            )
        if not torch.equal(gen_ref.token_ids, db_seq):
            raise AssertionError(
                f"{req_id}: HF generate tokens != DB token_ids "
                f"(hf_len={int(gen_ref.token_ids.numel())} db_len={int(db_seq.numel())})"
            )

        if print_text:
            db_prompt = db_seq[:plen]
            db_gen = db_seq[plen:]

            rol_prompt = ref.token_ids[:plen]
            rol_gen = ref.token_ids[plen:]

            gen_prompt = gen_ref.token_ids[:plen]
            gen_gen = gen_ref.token_ids[plen:]

            print(f"\n=== {req_id} (local_index={i}) ===")
            print(f"DB:  prompt_tokens={plen} generated_tokens={int(db_gen.numel())} total_tokens={int(db_seq.numel())}")
            print(f"DB PROMPT:    {_decode(db_prompt)!r}")
            print(f"DB GENERATED: {_decode(db_gen)!r}")
            print(f"DB FULL:      {_decode(db_seq)!r}")

            print(
                f"ROL: prompt_tokens={plen} generated_tokens={int(rol_gen.numel())} total_tokens={int(ref.token_ids.numel())}"
            )
            print(f"ROL PROMPT:   {_decode(rol_prompt)!r}")
            print(f"ROL GENERATED:{_decode(rol_gen)!r}")
            print(f"ROL FULL:     {_decode(ref.token_ids)!r}")

            print(
                f"GEN: prompt_tokens={plen} generated_tokens={int(gen_gen.numel())} total_tokens={int(gen_ref.token_ids.numel())}"
            )
            print(f"GEN PROMPT:   {_decode(gen_prompt)!r}")
            print(f"GEN GENERATED:{_decode(gen_gen)!r}")
            print(f"GEN FULL:     {_decode(gen_ref.token_ids)!r}")

            print("TOKENS MATCH?: YES (DB==ROL==GEN)")

    # Helpers for top-k printing
    def _tok_piece(tok_id: int) -> str:
        try:
            return tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            return f"<tok_id={tok_id}>"

    def _fmt_topk(ids_row: torch.Tensor, vals_row: torch.Tensor) -> str:
        parts: List[str] = []
        for tid, v in zip(ids_row.tolist(), vals_row.tolist()):
            parts.append(f"{int(tid)}:{_tok_piece(int(tid))!r}:{float(v):.6g}")
        return " | ".join(parts)

    # Compare per request
    for req_id in req_order:
        i = local_index_by_req[req_id]
        hooks_map = grouped[req_id]

        seq = db_token_ids_by_req[req_id].to(torch.long)  # [T]
        
        seq_len = int(seq.numel())
        ref = hf_ref_by_req[req_id]
        gen_ref = hf_gen_by_req[req_id]
        prompt_len = int(prompt_len_by_req[req_id])
        gen_base_pos = prompt_len - 1  # scores[0] corresponds to logits at pos (prompt_len-1)

        if seq_len != int(ref.token_ids.numel()):
            raise AssertionError(f"{req_id}: seq_len mismatch DB={seq_len} ROL={int(ref.token_ids.numel())}")
        if seq_len != int(gen_ref.token_ids.numel()):
            raise AssertionError(f"{req_id}: seq_len mismatch DB={seq_len} GEN={int(gen_ref.token_ids.numel())}")

        # --- final_logits (DB vs ROL) ---
        if (-1, "final_logits") not in hooks_map:
            raise AssertionError(f"{req_id}: DB missing final_logits")

        log_chunks = _sort_chunks(hooks_map[(-1, "final_logits")])
        db_logits = merge_segments([t for _, _, t in log_chunks], "final_logits")
        if db_logits.ndim == 1:
            db_logits = db_logits.unsqueeze(0)
        if db_logits.ndim != 2:
            raise AssertionError(f"{req_id}: final_logits unexpected shape {tuple(db_logits.shape)}")

        rows_db = int(db_logits.shape[0])
        vocab_db = int(db_logits.shape[1])
        if rows_db < seq_len:
            raise AssertionError(f"{req_id}: db_logits rows < seq_len: rows={rows_db} seq_len={seq_len}")

        db_slice = db_logits[rows_db - seq_len : rows_db, :]  # [T, vocab]
        rol_slice = ref.final_logits  # [T, vocab]

        if tuple(db_slice.shape) != tuple(rol_slice.shape):
            raise AssertionError(
                f"{req_id}: final_logits shape mismatch db={tuple(db_slice.shape)} rol={tuple(rol_slice.shape)}"
            )

        if print_topk_logits:
            print(f"\n=== TOP{topk_k} LOGITS {req_id} (local_index={i}) ===")
            print(f"seq_len={seq_len} vocab={int(db_slice.shape[1])}")

            db_topv, db_topi = torch.topk(db_slice.float(), k=topk_k, dim=-1)  # [T,k]
            rol_topv, rol_topi = torch.topk(rol_slice.float(), k=topk_k, dim=-1)  # [T,k]

            for tpos in range(seq_len):
                cur_id = int(seq[tpos].item())
                cur_piece = _tok_piece(cur_id)

                if tpos + 1 < seq_len:
                    nxt_id = int(seq[tpos + 1].item())
                    nxt_piece = _tok_piece(nxt_id)
                    label_str = f" next={nxt_id}:{nxt_piece!r}"
                else:
                    label_str = " next=<none>"

                print(f"\npos={tpos} tok={cur_id}:{cur_piece!r}{label_str}")
                print(f"  DB:  {_fmt_topk(db_topi[tpos], db_topv[tpos])}")
                print(f"  ROL: {_fmt_topk(rol_topi[tpos], rol_topv[tpos])}")

                # GEN: available only for positions gen_base_pos .. gen_base_pos+len(scores)-1 (which ends at T-2)
                if gen_base_pos >= 0 and gen_base_pos <= tpos <= (gen_base_pos + len(gen_ref.scores) - 1):
                    sidx = tpos - gen_base_pos
                    gs = gen_ref.scores[sidx]
                    g_topv, g_topi = torch.topk(gs.float(), k=topk_k, dim=-1)
                    print(f"  GEN: {_fmt_topk(g_topi, g_topv)}")
                else:
                    print("  GEN: <n/a>")

        # Keep strict compare for DB vs ROL logits (currently optional)
        if not torch.equal(db_slice, rol_slice):
            diff = (db_slice.float() - rol_slice.float()).abs()
            max_abs = float(diff.max().item())
            flat_idx = int(diff.view(-1).argmax().item())
            r = flat_idx // vocab_db
            c = flat_idx % vocab_db
            raise AssertionError(f"{req_id}: final_logits mismatch (max_abs={max_abs}) at row={r} vocab_idx={c}")


        # --- hook_embed / hook_pos_embed (GPT2-like) ---
        if (
            wte is not None
            and wpe is not None
            and (-1, "hook_embed") in hooks_map
            and (-1, "hook_pos_embed") in hooks_map
        ):
            ids = seq.to(device)
            pos = _positions_for_unpadded(seq_len, device=device)
            emb = wte(ids).detach().cpu()  # [T,d]
            pos_emb = wpe(pos).detach().cpu()  # [T,d]

            chunks = _sort_chunks(hooks_map[(-1, "hook_embed")])
            _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} hook_embed")
            db_t = merge_segments([t for _, _, t in chunks], "hook_embed")
            if tuple(db_t.shape) != tuple(emb.shape):
                raise AssertionError(f"{req_id}: hook_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(emb.shape)}")
            if not torch.equal(db_t, emb):
                max_abs = float((db_t.float() - emb.float()).abs().max().item())
                raise AssertionError(f"{req_id}: hook_embed mismatch (max_abs={max_abs})")

            chunks = _sort_chunks(hooks_map[(-1, "hook_pos_embed")])
            _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} hook_pos_embed")
            db_t = merge_segments([t for _, _, t in chunks], "hook_pos_embed")
            if tuple(db_t.shape) != tuple(pos_emb.shape):
                raise AssertionError(
                    f"{req_id}: hook_pos_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(pos_emb.shape)}"
                )
            if not torch.equal(db_t, pos_emb):
                max_abs = float((db_t.float() - pos_emb.float()).abs().max().item())
                raise AssertionError(f"{req_id}: hook_pos_embed mismatch (max_abs={max_abs})")

        """
        # --- hook_final_ln (GPT2-like) ---
        if ln_f is not None and (-1, "hook_final_ln") in hooks_map and ref.hidden_states:
            last_h = ref.hidden_states[-1].to(device)  # [T,d]
            fin = ln_f(last_h).detach().cpu()  # [T,d]
            chunks = _sort_chunks(hooks_map[(-1, "hook_final_ln")])
            _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} hook_final_ln")
            db_t = merge_segments([t for _, _, t in chunks], "hook_final_ln")
            if tuple(db_t.shape) != tuple(fin.shape):
                raise AssertionError(f"{req_id}: hook_final_ln shape mismatch db={tuple(db_t.shape)} hf={tuple(fin.shape)}")
            if not torch.equal(db_t, fin):
                max_abs = float((db_t.float() - fin.float()).abs().max().item())
                raise AssertionError(f"{req_id}: hook_final_ln mismatch (max_abs={max_abs})")
        """

        # --- per-layer: attention pattern + resid_pre/post ---
        n_layers = len(ref.attn_pattern) if ref.attn_pattern else 0

        for layer_no in range(n_layers):
            key = (layer_no, "blocks.attn.hook_pattern")
            if key in hooks_map:
                pat = ref.attn_pattern[layer_no]  # [H,T,T]
                chunks = _sort_chunks(hooks_map[key])
                _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} layer{layer_no} pattern")
                db_t = merge_segments([t for _, _, t in chunks], "blocks.attn.hook_pattern")
                if db_t.ndim == 4 and db_t.shape[0] == 1:
                    db_t = db_t.squeeze(0)
                if tuple(db_t.shape) != tuple(pat.shape):
                    raise AssertionError(
                        f"{req_id}: pattern shape mismatch layer={layer_no} db={tuple(db_t.shape)} hf={tuple(pat.shape)}"
                    )
                if not torch.equal(db_t, pat):
                    max_abs = float((db_t.float() - pat.float()).abs().max().item())
                    raise AssertionError(f"{req_id}: pattern mismatch layer={layer_no} (max_abs={max_abs})")

            key = (layer_no, "blocks.hook_resid_pre")
            if key in hooks_map and ref.hidden_states and layer_no < len(ref.hidden_states):
                hs = ref.hidden_states[layer_no]  # [T,d]
                chunks = _sort_chunks(hooks_map[key])
                _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} layer{layer_no} resid_pre")
                db_t = merge_segments([t for _, _, t in chunks], "blocks.hook_resid_pre")
                if tuple(db_t.shape) != tuple(hs.shape):
                    raise AssertionError(
                        f"{req_id}: resid_pre shape mismatch layer={layer_no} db={tuple(db_t.shape)} hf={tuple(hs.shape)}"
                    )
                if not torch.equal(db_t, hs):
                    max_abs = float((db_t.float() - hs.float()).abs().max().item())
                    raise AssertionError(f"{req_id}: resid_pre mismatch layer={layer_no} (max_abs={max_abs})")

            key = (layer_no, "blocks.hook_resid_post")
            # print(f"len of hidden_states: {len(ref.hidden_states)}")
            num_layers = get_num_layers_from_config(hf_model)
            if key in hooks_map and ref.hidden_states and (layer_no + 1) < num_layers:
                hs = ref.hidden_states[layer_no + 1]  # [T,d]
                chunks = _sort_chunks(hooks_map[key])
                _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} layer{layer_no} resid_post")
                db_t = merge_segments([t for _, _, t in chunks], "blocks.hook_resid_post")
                if tuple(db_t.shape) != tuple(hs.shape):
                    raise AssertionError(
                        f"{req_id}: resid_post shape mismatch layer={layer_no} db={tuple(db_t.shape)} hf={tuple(hs.shape)}"
                    )
                if not torch.equal(db_t, hs):
                    max_abs = float((db_t.float() - hs.float()).abs().max().item())
                    raise AssertionError(f"{req_id}: resid_post mismatch layer={layer_no} (max_abs={max_abs})")