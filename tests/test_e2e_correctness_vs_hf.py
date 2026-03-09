# tests/test_e2e_correctness_hf.py
# PYTHONPATH=./:./monitoring:$PYTHONPATH E2E_PRINT_TEXT=1 E2E_HF_DROP_LAST_TOKEN=1 E2E_PRINT_TOPK_LOGITS=1 pytest -q -s tests/test_e2e_correctness_vs_hf.py
"""E2E correctness test: monitoring DB vs HuggingFace Transformers (HF-driven ground truth).

This test runs the repo monitoring pipeline end-to-end (native backend + host engine + ClickHouse),
then uses HuggingFace Transformers as the reference implementation (no TransformerLens).

IMPORTANT: "prompt" in this test means FULL TOKEN SEQUENCE = prefill + decode
-----------------------------------------------------------------------
We treat DB `token_ids` as the ground-truth sequence of tokens for each request. That sequence
includes the initial prompt tokens (prefill) plus any decode tokens that were appended.

HF reference modes
-----------------------------------------------------------------------
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

Env vars
-----------------------------------------------------------------------
  - E2E_BATCH_SIZE (default 4)
  - E2E_MAX_NEW_TOKENS (default 8)
  - E2E_MODEL (default "gpt2"; "qwen3" alias supported)
  - E2E_CHUNK_BYTES (default 262144)

  - E2E_PRINT_TEXT (default 0): if 1, print decoded text from DB token_ids and from HF rollout + HF generate().
  - E2E_HF_DROP_LAST_TOKEN (default 0): if 1, drop the last token (and aligned tensors) from HF refs before compares.
  - E2E_PRINT_TOPK_LOGITS (default 0): if 1, print top-k logits at every position for DB vs ROL vs GEN.
  - E2E_PRINT_TOPK_LOGITS_K (default 5): top-k to print per position.

ClickHouse
-----------------------------------------------------------------------
  - DMX_DB_HOST, DMX_DB_PORT, DMX_DB_USER, DMX_DB_PASSWORD, DMX_DB_DATABASE, DMX_DB_TABLE
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, List, Tuple

import pytest
import torch

from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
from monitoring.segment_merger import merge_segments, parse_internal_id

from .hf_reference import (
    _HFGenRef,
    _HFRef,
    _hf_generate_collect_scores_batched,
    _hf_greedy_rollout_collect_all_batched,
    _parse_request_id,
    _positions_for_unpadded,
    _strip_left_pad,
)

# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

_MODEL_ALIASES = {"qwen3": "Qwen/Qwen3-4B"}


def _resolve_model_id(model: str) -> str:
    return _MODEL_ALIASES.get(model.lower(), model)


# ---------------------------------------------------------------------------
# Small utils (inlined to avoid test-only deps)
# ---------------------------------------------------------------------------


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact bitwise equality (treats NaNs as equal if their payload bits match)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    a8 = a.cpu().contiguous().view(torch.uint8)
    b8 = b.cpu().contiguous().view(torch.uint8)
    return torch.equal(a8, b8)


def get_num_layers_from_config(hf_model) -> int:
    cfg = getattr(hf_model, "config", None)
    if cfg is None:
        raise ValueError("HF model has no .config; cannot determine num_layers")

    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))

    # Some models nest text config (e.g., multi-modal wrappers)
    for sub in ("text_config", "model_config", "llm_config"):
        subcfg = getattr(cfg, sub, None)
        if subcfg is None:
            continue
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            if hasattr(subcfg, attr):
                return int(getattr(subcfg, attr))

    raise ValueError(f"Could not infer num_layers from config type={type(cfg)!r}")


def _canon_layer_and_act(act_name_raw: str, layer_no_raw: int) -> Tuple[int, str]:
    """Handle both schemas:
    - (layer_no, act_name) stored separately, act_name like 'blocks.attn.hook_pattern'
    - act_name stored as internal_id 'blocks.<L>.attn.hook_pattern' with layer_no possibly -1
    """
    try:
        layer_from_act, act = parse_internal_id(act_name_raw)
        if layer_from_act != -1:
            return int(layer_from_act), str(act)
    except Exception:
        # parse_internal_id is intentionally strict (expects 'blocks.<int>...'); fall back.
        pass
    return int(layer_no_raw), str(act_name_raw)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA + native backend required")
def test_e2e_correctness_hf(subtests) -> None:
    try:
        import clickhouse_driver  # noqa: F401
    except Exception:
        pytest.skip("clickhouse-driver is required")

    try:
        from monitoring import (  # type: ignore
            AdvanceConfig,
            HostEngineConfig,
            MonitoringConfig,
            MonitoringEngine,
            NativePartialSealConfig,
        )
        from monitoring._native_engine import (  # type: ignore
            ClickHouseClientConfig,
            RingConfig,
            StageConfig,
        )
        from monitoring.config import CaptureSchedule, HookSelection  # type: ignore
        from monitoring.generate import generate_with_monitoring  # type: ignore
    except Exception as exc:
        pytest.skip(f"monitoring native extension not available: {exc}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel  # type: ignore
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
    except Exception as exc:
        pytest.skip(f"transformers or repo Hooked* classes not available: {exc}")

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    batch_size = int(os.environ.get("E2E_BATCH_SIZE", "4"))
    if batch_size < 1:
        raise ValueError("E2E_BATCH_SIZE must be >= 1")

    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "8"))
    hf_model_id = _resolve_model_id(os.environ.get("E2E_MODEL", "gpt2"))
    chunk_bytes = int(os.environ.get("E2E_CHUNK_BYTES", str(256 * 1024)))

    print_text = int(os.environ.get("E2E_PRINT_TEXT", "0")) == 1
    hf_drop_last_token = int(os.environ.get("E2E_HF_DROP_LAST_TOKEN", "0")) == 1
    print_topk_logits = int(os.environ.get("E2E_PRINT_TOPK_LOGITS", "0")) == 1
    topk_k = int(os.environ.get("E2E_PRINT_TOPK_LOGITS_K", "5"))
    if topk_k < 1:
        raise ValueError("E2E_PRINT_TOPK_LOGITS_K must be >= 1")

    device = torch.device("cuda")

    # -----------------------------------------------------------------------
    # Tokenizer + prompts
    # -----------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    prompts = [("Hello " * (i + 1)).strip() for i in range(batch_size)]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Initial prompt tokens for safety prefix checks
    hf_initial_prompt_tokens: List[torch.Tensor] = []
    for j in range(batch_size):
        hf_initial_prompt_tokens.append(
            _strip_left_pad(
                input_ids[j].detach().cpu(),
                attention_mask[j].detach().cpu(),
            ).to(torch.long)
        )

    # -----------------------------------------------------------------------
    # Monitoring config (config-driven; no env var toggles)
    # -----------------------------------------------------------------------

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
        advance=AdvanceConfig(),
    )
    if hasattr(mon_cfg, "eos_token_id"):
        mon_cfg.eos_token_id = eos_id
    if hasattr(mon_cfg, "pad_token_id"):
        mon_cfg.pad_token_id = pad_id

    # -----------------------------------------------------------------------
    # ClickHouse config (for the monitored run)
    # -----------------------------------------------------------------------

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

    insert_stage = StageConfig.clickhouse_insert(db_cfg_native, parallelism=10, name="clickhouse_insert")
    host_cfg = HostEngineConfig(stages=[insert_stage])

    # Ring engine config (ring transport replaces NativeMonitoringEngine D2H)
    ring_cfg = RingConfig()
    ring_cfg.task_ring_entries = 1024
    ring_cfg.payload_ring_bytes = 4 * 1024 * 1024 * 1024  # 4 GB
    ring_cfg.chunk_bytes = 4 * 1024 * 1024                # 4 MB chunks
    ring_cfg.pinned_pool_bytes = 4 * 1024 * 1024 * 1024  # 4 GB pinned ring

    # -----------------------------------------------------------------------
    # Monitored run
    # -----------------------------------------------------------------------

    unique_run_model_id = f"e2e_correctness_hf::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(
        async_enabled=True, config=mon_cfg, model_id=unique_run_model_id, db_config=host_cfg
    )
    engine.enable_ring_transport(ring_cfg)

    model_cls = HookedQwen3ForCausalLM if "qwen3" in hf_model_id.lower() else HookedGPT2LMHeadModel
    mon_model = model_cls.from_pretrained(hf_model_id, attn_implementation="eager", torch_dtype=torch.float16)
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

    # -----------------------------------------------------------------------
    # Read DB (monitoring.clickhouse_reader + monitoring.segment_merger)
    # -----------------------------------------------------------------------

    ch = CHClickhouseDriverReadOnly(
        host=str(db_cfg_native.host),
        port=int(db_cfg_native.port),
        username=str(db_cfg_native.username),
        password=str(db_cfg_native.password),
        database=str(db_cfg_native.database),
        table=str(db_cfg_native.table),
        secure=bool(getattr(db_cfg_native, "secure", False)),
        client_settings=getattr(db_cfg_native, "client_settings", None),
        decode_strings=True,
    )
    try:
        rows = ch.prefix_get((unique_run_model_id, ), return_full_key_tuple=True)
    finally:
        ch.close()

    if not rows:
        pytest.fail(f"No rows found in ClickHouse for model_id={unique_run_model_id}")

    # If multiple shard_ranks are present, pick rank 0 if available, else the minimum.
    shard_ranks = sorted({int(key[4]) for key, _t in rows})
    chosen_shard_rank = 0 if 0 in shard_ranks else (shard_ranks[0] if shard_ranks else 0)
    rows = [(k, t) for (k, t) in rows if int(k[4]) == chosen_shard_rank]

    grouped: Dict[str, Dict[Tuple[int, str], List[Tuple[int, int, torch.Tensor]]]] = {}
    for full_key, t_raw in rows:
        # full_key = (model_id, request_id, act_name, layer_no, shard_rank, start_token_idx, end_token_idx)
        _model_id, req_id, act_name_raw, layer_no_raw, _shard_rank, s, e = full_key

        layer_no, act_name = _canon_layer_and_act(str(act_name_raw), int(layer_no_raw))

        t = t_raw.detach().cpu()
        grouped.setdefault(str(req_id), {}).setdefault((layer_no, act_name), []).append(
            (int(s), int(e), t)
        )

    request_ids = sorted(grouped.keys(), key=_parse_request_id)

    def _sort_chunks(chunks: List[Tuple[int, int, torch.Tensor]]) -> List[Tuple[int, int, torch.Tensor]]:
        return sorted(chunks, key=lambda x: (x[0], x[1]))

    def _validate_contiguous(
        chunks_sorted: List[Tuple[int, int, torch.Tensor]], expected_end: int, ctx: str
    ) -> None:
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

    # -----------------------------------------------------------------------
    # Build HF references (no assertions here)
    # -----------------------------------------------------------------------

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    ).to(device).eval()

    # Optional modules for GPT2-like reconstructions
    wte = getattr(getattr(hf_model, "transformer", None), "wte", None)
    wpe = getattr(getattr(hf_model, "transformer", None), "wpe", None)
    # Qwen3-like: embed_tokens on model sub-module (no separate pos embed)
    embed_tokens = getattr(getattr(hf_model, "model", None), "embed_tokens", None)

    # Compute num_layers once (used in per-layer comparison loop below)
    num_layers = get_num_layers_from_config(hf_model)

    req_order = sorted(request_ids, key=_parse_request_id)

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
            f"HF batched refs unexpected batch: rollout={len(hf_refs_batch)} "
            f"gen={len(hf_gens_batch)} batch_size={batch_size}"
        )

    hf_ref_by_req: Dict[str, _HFRef] = {}
    hf_gen_by_req: Dict[str, _HFGenRef] = {}

    def _decode(ids: torch.Tensor) -> str:
        return tokenizer.decode(ids.tolist(), skip_special_tokens=False)

    for req_id in req_order:
        i = local_index_by_req[req_id]
        plen = int(prompt_len_by_req[req_id])

        ref = hf_refs_batch[i]
        gen_ref = hf_gens_batch[i]

        if hf_drop_last_token:
            if int(ref.token_ids.numel()) > 0:
                T = int(ref.token_ids.numel())
                ref = _HFRef(
                    token_ids=ref.token_ids[:-1],
                    final_logits=ref.final_logits[:-1, :],
                    hidden_states=[h[:-1, :] for h in ref.hidden_states],
                    attn_pattern=[a[:, : T - 1, : T - 1] for a in ref.attn_pattern],
                )
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

        if print_text:
            db_seq = db_token_ids_by_req[req_id]
            db_prompt = db_seq[:plen]
            db_gen = db_seq[plen:]
            rol_prompt = ref.token_ids[:plen]
            rol_gen = ref.token_ids[plen:]
            gen_prompt = gen_ref.token_ids[:plen]
            gen_gen = gen_ref.token_ids[plen:]

            print(f"\n=== {req_id} (local_index={i}, shard_rank={chosen_shard_rank}) ===")
            print(f"DB:  prompt_tokens={plen} generated_tokens={int(db_gen.numel())} total_tokens={int(db_seq.numel())}")
            print(f"DB PROMPT:    {_decode(db_prompt)!r}")
            print(f"DB GENERATED: {_decode(db_gen)!r}")
            print(f"DB FULL:      {_decode(db_seq)!r}")
            print(f"ROL: prompt_tokens={plen} generated_tokens={int(rol_gen.numel())} total_tokens={int(ref.token_ids.numel())}")
            print(f"ROL PROMPT:   {_decode(rol_prompt)!r}")
            print(f"ROL GENERATED:{_decode(rol_gen)!r}")
            print(f"ROL FULL:     {_decode(ref.token_ids)!r}")
            print(f"GEN: prompt_tokens={plen} generated_tokens={int(gen_gen.numel())} total_tokens={int(gen_ref.token_ids.numel())}")
            print(f"GEN PROMPT:   {_decode(gen_prompt)!r}")
            print(f"GEN GENERATED:{_decode(gen_gen)!r}")
            print(f"GEN FULL:     {_decode(gen_ref.token_ids)!r}")
            print("TOKENS MATCH?: YES (DB==ROL==GEN)")

    # Helpers for top-k logit printing
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

    # -----------------------------------------------------------------------
    # Subtests: one per assertion
    # -----------------------------------------------------------------------

    for req_id in req_order:
        i = local_index_by_req[req_id]
        hooks_map = grouped[req_id]

        seq = db_token_ids_by_req[req_id].to(torch.long)  # [T]
        seq_len = int(seq.numel())
        ref = hf_ref_by_req[req_id]
        gen_ref = hf_gen_by_req[req_id]
        prompt_len = int(prompt_len_by_req[req_id])
        gen_base_pos = prompt_len - 1  # scores[0] corresponds to logits at pos (prompt_len-1)

        # --- token_ids ---
        with subtests.test(msg=f"{req_id}/token_ids_rol"):
            assert bitwise_equal(ref.token_ids, seq), (
                f"HF rollout tokens != DB token_ids "
                f"(hf_len={int(ref.token_ids.numel())} db_len={seq_len})"
            )

        with subtests.test(msg=f"{req_id}/token_ids_gen"):
            assert bitwise_equal(gen_ref.token_ids, seq), (
                f"HF generate tokens != DB token_ids "
                f"(hf_len={int(gen_ref.token_ids.numel())} db_len={seq_len})"
            )

        # --- final_logits ---
        logits_chunks_raw = hooks_map.get((-1, "final_logits"), [])
        if logits_chunks_raw:
            lchunks = sorted(logits_chunks_raw, key=lambda x: (x[0], x[1]))
            db_logits_full = merge_segments([t for _, _, t in lchunks], "final_logits")
            if db_logits_full.ndim == 1:
                db_logits_full = db_logits_full.unsqueeze(0)
            db_slice = db_logits_full[-seq_len:, :]
            rol_slice = ref.final_logits[:seq_len, :]
            vocab_db = int(db_slice.shape[1])

            if print_topk_logits:
                print(f"\n=== TOP{topk_k} LOGITS {req_id} (local_index={i}, shard_rank={chosen_shard_rank}) ===")
                print(f"seq_len={seq_len} vocab={vocab_db}")
                db_topv, db_topi = torch.topk(db_slice.float(), k=topk_k, dim=-1)
                rol_topv, rol_topi = torch.topk(rol_slice.float(), k=topk_k, dim=-1)
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
                    if gen_base_pos >= 0 and gen_base_pos <= tpos <= (gen_base_pos + len(gen_ref.scores) - 1):
                        sidx = tpos - gen_base_pos
                        gs = gen_ref.scores[sidx]
                        g_topv, g_topi = torch.topk(gs.float(), k=topk_k, dim=-1)
                        print(f"  GEN: {_fmt_topk(g_topi, g_topv)}")
                    else:
                        print("  GEN: <n/a>")

            with subtests.test(msg=f"{req_id}/final_logits"):
                if not bitwise_equal(db_slice, rol_slice):
                    diff = (db_slice.float() - rol_slice.float()).abs()
                    max_abs = float(diff.max().item())
                    flat_idx = int(diff.view(-1).argmax().item())
                    r = flat_idx // vocab_db
                    c = flat_idx % vocab_db
                    pytest.fail(f"final_logits mismatch (max_abs={max_abs}) at row={r} vocab_idx={c}")

        # --- hook_embed / hook_pos_embed (GPT2-like: both wte and wpe) ---
        if (
            wte is not None
            and wpe is not None
            and (-1, "hook_embed") in hooks_map
            and (-1, "hook_pos_embed") in hooks_map
        ):
            ids = seq.to(device)
            pos = _positions_for_unpadded(seq_len, device=device)
            emb = wte(ids).detach().cpu()       # [T, d]
            pos_emb = wpe(pos).detach().cpu()   # [T, d]

            chunks = sorted(hooks_map[(-1, "hook_embed")], key=lambda x: (x[0], x[1]))
            _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} hook_embed")
            db_t = merge_segments([t for _, _, t in chunks], "hook_embed")
            with subtests.test(msg=f"{req_id}/hook_embed"):
                assert tuple(db_t.shape) == tuple(emb.shape), (
                    f"hook_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(emb.shape)}"
                )
                assert bitwise_equal(db_t, emb), (
                    f"hook_embed mismatch (max_abs={float((db_t.float() - emb.float()).abs().max().item())})"
                )

            chunks = sorted(hooks_map[(-1, "hook_pos_embed")], key=lambda x: (x[0], x[1]))
            _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} hook_pos_embed")
            db_t = merge_segments([t for _, _, t in chunks], "hook_pos_embed")
            with subtests.test(msg=f"{req_id}/hook_pos_embed"):
                assert tuple(db_t.shape) == tuple(pos_emb.shape), (
                    f"hook_pos_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(pos_emb.shape)}"
                )
                assert bitwise_equal(db_t, pos_emb), (
                    f"hook_pos_embed mismatch (max_abs={float((db_t.float() - pos_emb.float()).abs().max().item())})"
                )

        # --- hook_embed only (Qwen3-like: RoPE, no separate pos embed) ---
        elif (
            embed_tokens is not None
            and wpe is None
            and (-1, "hook_embed") in hooks_map
        ):
            emb = embed_tokens(seq.to(device)).detach().cpu()  # [T, d]
            chunks = sorted(hooks_map[(-1, "hook_embed")], key=lambda x: (x[0], x[1]))
            _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} hook_embed")
            db_t = merge_segments([t for _, _, t in chunks], "hook_embed")
            with subtests.test(msg=f"{req_id}/hook_embed"):
                assert tuple(db_t.shape) == tuple(emb.shape), (
                    f"hook_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(emb.shape)}"
                )
                assert bitwise_equal(db_t, emb), (
                    f"hook_embed mismatch (max_abs={float((db_t.float() - emb.float()).abs().max().item())})"
                )

        # --- hook_final_ln ---
        # TODO: hook_final_ln comparison skipped.

        # --- per-layer: attention pattern + resid_pre/post ---
        # Support both GPT-2 naming (blocks.attn.hook_pattern, blocks.hook_resid_*)
        # and Qwen3 naming (layers.self_attn.hook_pattern, layers.hook_resid_*)
        _ATTN_PATTERN_KEYS = ("blocks.attn.hook_pattern", "layers.self_attn.hook_pattern")
        _RESID_PRE_KEYS = ("blocks.hook_resid_pre", "layers.hook_resid_pre")
        _RESID_POST_KEYS = ("blocks.hook_resid_post", "layers.hook_resid_post")

        n_layers = len(ref.attn_pattern) if ref.attn_pattern else 0
        assert n_layers == num_layers, (
            f"{req_id}: attn_pattern layer count mismatch: rollout={n_layers} config={num_layers}"
        )

        for layer_no in range(n_layers):
            key = next(((layer_no, k) for k in _ATTN_PATTERN_KEYS if (layer_no, k) in hooks_map), None)
            if key is not None:
                pat = ref.attn_pattern[layer_no]  # [H, T, T]
                chunks = sorted(hooks_map[key], key=lambda x: (x[0], x[1]))
                _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} layer{layer_no} pattern")
                db_t = merge_segments([t for _, _, t in chunks], key[1])
                if db_t.ndim == 4 and db_t.shape[0] == 1:
                    db_t = db_t.squeeze(0)
                with subtests.test(msg=f"{req_id}/layer{layer_no}/attn_pattern"):
                    assert tuple(db_t.shape) == tuple(pat.shape), (
                        f"pattern shape mismatch layer={layer_no} db={tuple(db_t.shape)} hf={tuple(pat.shape)}"
                    )
                    assert bitwise_equal(db_t, pat), (
                        f"pattern mismatch layer={layer_no} "
                        f"(max_abs={float((db_t.float() - pat.float()).abs().max().item())})"
                    )

            key = next(((layer_no, k) for k in _RESID_PRE_KEYS if (layer_no, k) in hooks_map), None)
            if key is not None and ref.hidden_states and layer_no < len(ref.hidden_states):
                hs = ref.hidden_states[layer_no]  # [T, d]
                chunks = sorted(hooks_map[key], key=lambda x: (x[0], x[1]))
                _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} layer{layer_no} resid_pre")
                db_t = merge_segments([t for _, _, t in chunks], key[1])
                with subtests.test(msg=f"{req_id}/layer{layer_no}/resid_pre"):
                    assert tuple(db_t.shape) == tuple(hs.shape), (
                        f"resid_pre shape mismatch layer={layer_no} db={tuple(db_t.shape)} hf={tuple(hs.shape)}"
                    )
                    assert bitwise_equal(db_t, hs), (
                        f"resid_pre mismatch layer={layer_no} "
                        f"(max_abs={float((db_t.float() - hs.float()).abs().max().item())})"
                    )

            key = next(((layer_no, k) for k in _RESID_POST_KEYS if (layer_no, k) in hooks_map), None)
            # len(ref.hidden_states) == n_layers + 1
            # here we skip the last one, because that is not a full layer, just after
            # the final_ln
            if key is not None and ref.hidden_states and (layer_no + 1) < n_layers:
                hs = ref.hidden_states[layer_no + 1]  # [T, d]
                chunks = sorted(hooks_map[key], key=lambda x: (x[0], x[1]))
                _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} layer{layer_no} resid_post")
                db_t = merge_segments([t for _, _, t in chunks], key[1])
                with subtests.test(msg=f"{req_id}/layer{layer_no}/resid_post"):
                    assert tuple(db_t.shape) == tuple(hs.shape), (
                        f"resid_post shape mismatch layer={layer_no} db={tuple(db_t.shape)} hf={tuple(hs.shape)}"
                    )
                    assert bitwise_equal(db_t, hs), (
                        f"resid_post mismatch layer={layer_no} "
                        f"(max_abs={float((db_t.float() - hs.float()).abs().max().item())})"
                    )


# ---------------------------------------------------------------------------
# CUDA-graph correctness test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA + native backend required")
def test_e2e_correctness_hf_cuda_graphs(subtests) -> None:
    """Same as test_e2e_correctness_hf but with torch.compile + static KV cache (CUDA graphs).

    This test specifically validates that monitoring captures ALL decode steps, not just the
    CUDA graph capture step.  With the current ring_producer_op Python-dispatch bug the test
    will fail because only 1 of N decode steps is monitored.

    Run with:
        E2E_HF_DROP_LAST_TOKEN=1 pytest -q -s tests/test_e2e_correctness_vs_hf.py::test_e2e_correctness_hf_cuda_graphs
    """
    try:
        import clickhouse_driver  # noqa: F401
    except Exception:
        pytest.skip("clickhouse-driver is required")

    try:
        from monitoring import (  # type: ignore
            AdvanceConfig,
            HostEngineConfig,
            MonitoringConfig,
            MonitoringEngine,
            NativePartialSealConfig,
        )
        from monitoring._native_engine import (  # type: ignore
            ClickHouseClientConfig,
            RingConfig,
            StageConfig,
        )
        from monitoring.config import CaptureSchedule, HookSelection  # type: ignore
        from monitoring.generate import generate_with_monitoring  # type: ignore
    except Exception as exc:
        pytest.skip(f"monitoring native extension not available: {exc}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel  # type: ignore
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
    except Exception as exc:
        pytest.skip(f"transformers or Hooked* classes not available: {exc}")

    # -----------------------------------------------------------------------
    # Config — fixed small values to keep the test fast
    # -----------------------------------------------------------------------
    batch_size       = int(os.environ.get("E2E_BATCH_SIZE",         "4"))
    max_new_tokens   = int(os.environ.get("E2E_MAX_NEW_TOKENS",     "8"))
    hf_model_id      = os.environ.get("E2E_MODEL", "gpt2")
    hf_model_id      = _MODEL_ALIASES.get(hf_model_id.lower(), hf_model_id)
    chunk_bytes      = int(os.environ.get("E2E_CHUNK_BYTES", str(256 * 1024)))
    hf_drop_last_token = bool(int(os.environ.get("E2E_HF_DROP_LAST_TOKEN", "0")))

    device = torch.device("cuda")

    # -----------------------------------------------------------------------
    # Tokenizer + prompts
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    prompts = [("Hello " * (i + 1)).strip() for i in range(batch_size)]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    hf_initial_prompt_tokens: List[torch.Tensor] = []
    for j in range(batch_size):
        hf_initial_prompt_tokens.append(
            _strip_left_pad(
                input_ids[j].detach().cpu(),
                attention_mask[j].detach().cpu(),
            ).to(torch.long)
        )

    # -----------------------------------------------------------------------
    # Monitoring + DB config
    # -----------------------------------------------------------------------
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
        advance=AdvanceConfig(),
    )

    db_cfg_native = ClickHouseClientConfig()
    db_cfg_native.host     = os.environ.get("DMX_DB_HOST",     "localhost")
    db_cfg_native.port     = int(os.environ.get("DMX_DB_PORT", "9000"))
    db_cfg_native.username = os.environ.get("DMX_DB_USER",     "default")
    db_cfg_native.password = os.environ.get("DMX_DB_PASSWORD", "")
    db_cfg_native.database = os.environ.get("DMX_DB_DATABASE", "default")
    db_cfg_native.table    = os.environ.get("DMX_DB_TABLE",    "offload")
    db_cfg_native.secure                   = False
    db_cfg_native.client_side_compress     = "none"
    db_cfg_native.client_settings          = None
    db_cfg_native.create_database_if_missing = True
    db_cfg_native.drop_existing_database   = True
    db_cfg_native.index_granularity        = 8192

    insert_stage = StageConfig.clickhouse_insert(db_cfg_native, parallelism=10, name="clickhouse_insert")
    host_cfg     = HostEngineConfig(stages=[insert_stage])

    ring_cfg = RingConfig()
    ring_cfg.task_ring_entries  = 1024
    ring_cfg.payload_ring_bytes = 4 * 1024 * 1024 * 1024
    ring_cfg.chunk_bytes        = 4 * 1024 * 1024
    ring_cfg.pinned_pool_bytes  = 4 * 1024 * 1024 * 1024

    # -----------------------------------------------------------------------
    # Monitored model — compiled with torch.compile + static cache (CUDA graphs)
    # -----------------------------------------------------------------------
    unique_run_model_id = f"e2e_cuda_graphs::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(
        async_enabled=True, config=mon_cfg,
        model_id=unique_run_model_id, db_config=host_cfg,
    )
    engine.enable_ring_transport(ring_cfg)

    model_cls = HookedQwen3ForCausalLM if "qwen3" in hf_model_id.lower() else HookedGPT2LMHeadModel
    mon_model = model_cls.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16,
    )
    mon_model.to(device).eval()

    # Compile BEFORE attaching the monitoring engine so torch.compile sees the
    # original forward.  Monitoring hooks are then installed per generate call.
    mon_model = torch.compile(mon_model, mode="reduce-overhead")

    mon_model.monitoring_engine = engine
    engine.prepare_for_model(mon_model)

    try:
        with torch.no_grad():
            gen_out = generate_with_monitoring(
                mon_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                cache_implementation="static",  # enables CUDA graph capture in HF
            )
    finally:
        engine.close()

    # Build per-request reference sequences from the generate() output.
    # We use the compiled model's own output as the reference — this avoids
    # any comparison against a different model that may compute different
    # values under static cache or torch.compile.
    gen_out_cpu = gen_out.detach().cpu().long()  # [batch, total_len]
    ref_seqs: List[torch.Tensor] = []
    for j in range(batch_size):
        seq = _strip_left_pad(gen_out_cpu[j], (gen_out_cpu[j] != pad_id).long())
        ref_seqs.append(seq)

    # -----------------------------------------------------------------------
    # Read DB
    # -----------------------------------------------------------------------
    from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
    from monitoring.segment_merger import merge_segments, parse_internal_id

    ch = CHClickhouseDriverReadOnly(
        host=str(db_cfg_native.host),
        port=int(db_cfg_native.port),
        username=str(db_cfg_native.username),
        password=str(db_cfg_native.password),
        database=str(db_cfg_native.database),
        table=str(db_cfg_native.table),
        secure=bool(getattr(db_cfg_native, "secure", False)),
        client_settings=getattr(db_cfg_native, "client_settings", None),
        decode_strings=True,
    )
    try:
        rows = ch.prefix_get((unique_run_model_id,), return_full_key_tuple=True)
    finally:
        ch.close()

    if not rows:
        pytest.fail(f"No rows found in ClickHouse for model_id={unique_run_model_id!r}. "
                    "This means monitoring produced no output at all under CUDA graphs.")

    shard_ranks  = sorted({int(key[4]) for key, _t in rows})
    chosen_shard = 0 if 0 in shard_ranks else shard_ranks[0]
    rows = [(k, t) for (k, t) in rows if int(k[4]) == chosen_shard]

    grouped: Dict[str, Dict[Tuple[int, str], List[Tuple[int, int, torch.Tensor]]]] = {}
    for full_key, t_raw in rows:
        _model_id, req_id, act_name_raw, layer_no_raw, _shard, s, e = full_key
        layer_no, act_name = _canon_layer_and_act(str(act_name_raw), int(layer_no_raw))
        grouped.setdefault(str(req_id), {}).setdefault(
            (layer_no, act_name), []
        ).append((int(s), int(e), t_raw.detach().cpu()))

    request_ids = sorted(grouped.keys(), key=_parse_request_id)

    # -----------------------------------------------------------------------
    # HF reference (no compile — plain eager generate)
    # -----------------------------------------------------------------------
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16,
    ).to(device).eval()

    hf_gens_batch = _hf_generate_collect_scores_batched(
        hf_model=hf_model,
        input_ids_batch=input_ids,
        attention_mask_batch=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Per-request assertions
    # -----------------------------------------------------------------------
    local_index_by_req: Dict[str, int] = {}
    for req_id in request_ids:
        group_id, local_i = _parse_request_id(req_id)
        local_index_by_req[req_id] = local_i

    for req_id in sorted(request_ids, key=_parse_request_id):
        local_i   = local_index_by_req[req_id]
        hooks_map = grouped[req_id]
        gen_ref   = hf_gens_batch[local_i]
        prompt0   = hf_initial_prompt_tokens[local_i]
        plen      = int(prompt0.numel())

        if hf_drop_last_token and gen_ref.token_ids.numel() > 0:
            new_seq     = gen_ref.token_ids[:-1]
            new_gen_len = max(0, int(new_seq.numel()) - plen)
            gen_ref = _HFGenRef(token_ids=new_seq, scores=gen_ref.scores[:new_gen_len])

        # --- token_ids present and complete ---
        with subtests.test(msg=f"{req_id}/cuda_graph/token_ids_present"):
            assert (-1, "token_ids") in hooks_map, (
                f"{req_id}: DB missing token_ids under CUDA graphs"
            )

        if (-1, "token_ids") not in hooks_map:
            continue

        tok_chunks = sorted(hooks_map[(-1, "token_ids")], key=lambda x: (x[0], x[1]))
        db_tok = merge_segments([t for _, _, t in tok_chunks], "token_ids").to(torch.long)
        if db_tok.ndim != 1:
            db_tok = db_tok.view(-1)

        # Prompt prefix safety check
        with subtests.test(msg=f"{req_id}/cuda_graph/prompt_prefix"):
            assert db_tok.numel() >= plen and torch.equal(db_tok[:plen], prompt0), (
                f"{req_id}: DB token_ids prompt prefix mismatch under CUDA graphs"
            )

        # The key correctness check: DB must cover the full generated sequence.
        # With the CUDA-graph bug only the capture-step tokens arrive; the DB
        # sequence is far shorter than what HF generate produced.
        hf_seq = gen_ref.token_ids  # cpu long
        with subtests.test(msg=f"{req_id}/cuda_graph/token_ids_match_hf"):
            assert bitwise_equal(db_tok, hf_seq), (
                f"{req_id}: DB token_ids do not match HF generate() under CUDA graphs. "
                f"db_len={db_tok.numel()} hf_len={hf_seq.numel()} "
                f"(if db_len << hf_len the CUDA-graph monitoring bug is present)"
            )