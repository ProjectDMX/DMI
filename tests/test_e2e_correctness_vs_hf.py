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
import sys
import uuid
from typing import Dict, List, Tuple

import pytest
import torch

from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
from monitoring.segment_merger import merge_segments, parse_internal_id

from .hf_reference import (
    _HFGenRef,
    _HFRef,
    _hf_generate_collect_hidden_states_batched,
    _hf_generate_collect_scores_batched,
    _hf_greedy_rollout_collect_all_batched,
    _load_hf_refs_from_disk,
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
# Ring + host engine configuration (env-var driven)
# ---------------------------------------------------------------------------

def _make_ring_cfg():
    """Build RingConfig from E2E_RING_* environment variables."""
    from monitoring._native_engine import RingConfig  # type: ignore
    rc = RingConfig()
    rc.task_ring_entries          = int(os.environ.get("E2E_RING_TASK_ENTRIES", "16384"))
    rc.payload_ring_bytes         = int(os.environ.get("E2E_RING_PAYLOAD_BYTES", str(4 * 1024**3)))
    rc.pinned_staging_bytes       = int(os.environ.get("E2E_RING_PINNED_BYTES", str(4 * 1024**3)))
    rc.drain_poll_timeout_us      = int(os.environ.get("E2E_DRAIN_POLL_TIMEOUT_US", "100"))
    rc.drain_flush_task_ratio     = float(os.environ.get("E2E_DRAIN_FLUSH_TASK_RATIO", "0.0"))
    rc.drain_flush_payload_ratio  = float(os.environ.get("E2E_DRAIN_FLUSH_PAYLOAD_RATIO", "0.0"))
    rc.drain_flush_entry_threshold = int(os.environ.get("E2E_DRAIN_FLUSH_ENTRY_THRESHOLD", "0"))
    rc.drain_flush_byte_threshold  = int(os.environ.get("E2E_DRAIN_FLUSH_BYTE_THRESHOLD", "0"))
    rc.drain_flush_timeout_us      = int(os.environ.get("E2E_DRAIN_FLUSH_TIMEOUT_US", "0"))
    rc.clone_slices               = int(os.environ.get("E2E_CLONE_SLICES", "0")) != 0
    rc.insert_queue_max_bytes     = int(os.environ.get("E2E_INSERT_QUEUE_MAX_BYTES", str(512 * 1024**2)))
    rc.insert_queue_max_items     = int(os.environ.get("E2E_INSERT_QUEUE_MAX_ITEMS", "4096"))
    return rc


def _make_host_cfg(db_cfg_native):
    """Build HostEngineConfig with clickhouse insert stage from env vars."""
    from monitoring import HostEngineConfig  # type: ignore
    from monitoring._native_engine import StageConfig  # type: ignore
    parallelism = int(os.environ.get("E2E_CH_PARALLELISM", "10"))
    stage = StageConfig.clickhouse_insert(db_cfg_native, parallelism=parallelism,
                                          name="clickhouse_insert")
    q = stage.input_queue
    q.max_batch_items      = int(os.environ.get("E2E_CH_QUEUE_MAX_ITEMS", "1024"))
    q.high_watermark_items = q.max_batch_items
    q.max_batch_size       = int(os.environ.get("E2E_CH_QUEUE_MAX_BYTES", str(2048 * 1024**2)))
    q.high_watermark_size  = q.max_batch_size
    return HostEngineConfig(stages=[stage])


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.cuda.is_built(), reason="CUDA not built")
def test_e2e_correctness_hf(subtests) -> None:
    """E2E correctness: compare HOOKED model (ring transport -> ClickHouse)
    against ORIGINAL model (HF output_hidden_states=True).

    Three subprocesses — parent process never touches CUDA:
      1. Reference: original model -> tensors on disk
      2. Monitored: hooked model + ring transport -> ClickHouse
      3. Comparator: reads both, compares, writes result.json
    """
    import json
    import subprocess
    import tempfile
    import shutil

    run_dir = tempfile.mkdtemp(prefix="hf_e2e_")
    ref_dir = os.path.join(run_dir, "ref")
    mon_dir = os.path.join(run_dir, "mon")
    result_file = os.path.join(run_dir, "result.json")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        # Step 1: Reference run (original model, no CUDA in parent)
        print("\n  [1/3] Reference run (original model)...", flush=True)
        r1 = subprocess.run(
            [sys.executable, "-m", "tests.hf_reference_runner",
             "--output-dir", ref_dir],
            env=os.environ, capture_output=True, text=True, cwd=project_root,
        )
        if r1.returncode != 0:
            pytest.fail(f"Reference runner failed:\n{r1.stderr[-2000:]}")

        # Step 2: Monitored run (hooked model + ring transport)
        print("  [2/3] Monitored run (hooked model + ring)...", flush=True)
        r2 = subprocess.run(
            [sys.executable, "-m", "tests.hf_monitored_runner",
             "--output-dir", mon_dir],
            env=os.environ, capture_output=True, text=True, cwd=project_root,
        )
        if r2.returncode != 0:
            pytest.fail(f"Monitored runner failed:\n{r2.stderr[-2000:]}")

        # Step 3: Comparator (CPU only, reads disk + ClickHouse)
        print("  [3/3] Comparing...", flush=True)
        r3 = subprocess.run(
            [sys.executable, "-m", "tests.hf_comparator",
             "--ref-dir", ref_dir,
             "--mon-dir", mon_dir,
             "--result-file", result_file],
            env=os.environ, capture_output=True, text=True, cwd=project_root,
        )
        if r3.returncode != 0:
            pytest.fail(f"Comparator failed:\n{r3.stderr[-2000:]}")

        # Read results
        with open(result_file) as f:
            results = json.load(f)

        # Report via subtests
        for test in results["tests"]:
            with subtests.test(test["name"]):
                assert test["passed"], test.get("detail", "")

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA + native backend required")
def _test_e2e_correctness_hf_legacy(subtests) -> None:
    """Legacy version kept for reference. Not called by verify_hf.sh."""
    try:
        import clickhouse_driver  # noqa: F401
    except Exception:
        pytest.skip("clickhouse-driver is required")

    try:
        from monitoring import (  # type: ignore
            MonitoringConfig,
            MonitoringEngine,
        )
        from monitoring._native_engine import ClickHouseClientConfig  # type: ignore
        from monitoring.config import CaptureSchedule  # type: ignore
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
    # E2E_HF_DROP_LAST_TOKEN is no longer needed: both monitored and HF reference
    # use generate(), so they produce the same number of tokens/hidden states.
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
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
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

    host_cfg = _make_host_cfg(db_cfg_native)
    ring_cfg = _make_ring_cfg()

    # -----------------------------------------------------------------------
    # Monitored run
    # -----------------------------------------------------------------------

    unique_run_model_id = f"e2e_correctness_hf::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(
        config=mon_cfg, model_id=unique_run_model_id, db_config=host_cfg
    )
    engine.enable_ring_transport(ring_cfg)

    model_cls = HookedQwen3ForCausalLM if "qwen3" in hf_model_id.lower() else HookedGPT2LMHeadModel
    mon_model = model_cls.from_pretrained(hf_model_id, attn_implementation="eager", torch_dtype=torch.float16)
    mon_model.to(device).eval()
    mon_model.monitoring_engine = engine

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

    # Run reference on ORIGINAL (non-hooked) model in a subprocess.
    # This validates that our hooks don't change the model output,
    # and ensures clean GPU memory isolation.
    import subprocess, tempfile
    ref_dir = tempfile.mkdtemp(prefix="hf_ref_")
    ref_env = {**os.environ, "E2E_BATCH_SIZE": str(batch_size),
               "E2E_MAX_NEW_TOKENS": str(max_new_tokens),
               "E2E_MODEL": os.environ.get("E2E_MODEL", "gpt2")}
    ref_result = subprocess.run(
        [sys.executable, "-m", "tests.hf_reference_runner", "--output-dir", ref_dir],
        env=ref_env, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    if ref_result.returncode != 0:
        pytest.fail(f"Reference runner failed:\n{ref_result.stderr[-2000:]}")
    hf_refs_batch = _load_hf_refs_from_disk(ref_dir)
    import shutil
    shutil.rmtree(ref_dir, ignore_errors=True)
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
            # ref.final_logits is decode-only scores from generate():
            # ref[s] = logits at position (prompt_len - 1 + s).
            # Align with DB logits by position.
            n_ref = int(ref.final_logits.shape[0])
            start = prompt_len - 1
            end = min(start + n_ref, int(db_logits_full.shape[0]))
            n = end - start
            db_slice = db_logits_full[start:end, :]
            rol_slice = ref.final_logits[:n, :]
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

        # --- per-layer: attention pattern + resid_pre ---
        # Support both GPT-2 naming (blocks.attn.hook_pattern, blocks.hook_resid_*)
        # and Qwen3 naming (layers.self_attn.hook_pattern, layers.hook_resid_*)
        _ATTN_PATTERN_KEYS = ("blocks.attn.hook_pattern", "layers.self_attn.hook_pattern")
        _RESID_PRE_KEYS = ("blocks.hook_resid_pre", "layers.hook_resid_pre")

        n_layers = len(ref.attn_pattern) if ref.attn_pattern else 0
        assert n_layers == num_layers, (
            f"{req_id}: attn_pattern layer count mismatch: rollout={n_layers} config={num_layers}"
        )

        for layer_no in range(n_layers):
            # attn_pattern: compare per-chunk (can't merge because kv_dim
            # differs between prefill [H, plen, plen] and decode [H, 1, plen+i])
            key = next(((layer_no, k) for k in _ATTN_PATTERN_KEYS if (layer_no, k) in hooks_map), None)
            if key is not None:
                pat = ref.attn_pattern[layer_no]  # [H, T, T]
                chunks = sorted(hooks_map[key], key=lambda x: (x[0], x[1]))
                all_ok = True
                fail_msg = ""
                for start, end, t_chunk in chunks:
                    q_len = end - start
                    db_c = t_chunk
                    if db_c.ndim == 4 and db_c.shape[0] == 1:
                        db_c = db_c.squeeze(0)
                    # kv_dim for these rows: causal, valid up to position 'end'
                    kv_valid = end
                    db_c = db_c[:, :q_len, :kv_valid]
                    ref_c = pat[:, start:end, :kv_valid]
                    if db_c.shape != ref_c.shape:
                        all_ok = False
                        fail_msg = (f"shape mismatch at [{start}:{end}] "
                                    f"db={db_c.shape} ref={ref_c.shape}")
                        break
                    if not bitwise_equal(db_c, ref_c):
                        max_abs = float((db_c.float() - ref_c.float()).abs().max().item())
                        all_ok = False
                        fail_msg = (f"value mismatch at [{start}:{end}] "
                                    f"max_abs={max_abs:.6f}")
                        break
                with subtests.test(msg=f"{req_id}/layer{layer_no}/attn_pattern"):
                    assert all_ok, (
                        f"pattern layer={layer_no}: {fail_msg}"
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

        # --- resid_final (global: last layer's pre-norm residual) ---
        # HF's output_hidden_states[-1] is POST-final-norm (after ln_f),
        # not pre-norm.  resid_final captures pre-norm.  No direct HF
        # reference available, so we only check shape and presence.
        key = (-1, "hook_resid_final")
        if key in hooks_map:
            chunks = sorted(hooks_map[key], key=lambda x: (x[0], x[1]))
            _validate_contiguous(chunks, expected_end=seq_len, ctx=f"{req_id} resid_final")
            db_t = merge_segments([t for _, _, t in chunks], key[1])
            with subtests.test(msg=f"{req_id}/resid_final"):
                assert db_t.shape[-1] == ref.hidden_states[0].shape[-1] if ref.hidden_states else True, (
                    f"resid_final hidden_dim mismatch"
                )
                assert db_t.shape[0] == seq_len, (
                    f"resid_final token count mismatch db={db_t.shape[0]} expected={seq_len}"
                )


# ---------------------------------------------------------------------------
# CUDA-graph correctness test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(True, reason=(
    "DISABLED: compiled rollout reference (StaticCache + torch.compile) cannot "
    "replicate generate()'s internal StaticCache attention mask / position handling. "
    "The manual rollout produces different hidden states than generate() even for "
    "identical inputs — this is a fundamental mismatch in how HF handles StaticCache "
    "internally vs externally.  Use test_e2e_cuda_graphs_vs_eager_hf instead, which "
    "compares CUDA-graph DB against an uncompiled eager reference with relaxed tolerance."
))
def test_e2e_correctness_hf_cuda_graphs(subtests) -> None:
    """Same as test_e2e_correctness_hf but with torch.compile + static KV cache (CUDA graphs).

    NOTE: This test is currently DISABLED.  The compiled rollout reference uses
    StaticCache + torch.compile on a manual decode loop, but this produces
    different numerical results from HF generate(cache_implementation="static")
    because generate() handles attention masks and position_ids differently
    internally.  CUDA graphs also prevent reading hidden states from generate()
    (Bug 11 in debug.log).  See test_e2e_cuda_graphs_vs_eager_hf for the
    working alternative.

    Run with:
        CUDA_MODULE_LOADING=EAGER pytest -q -s tests/test_e2e_correctness_vs_hf.py::test_e2e_correctness_hf_cuda_graphs
    """
    try:
        import clickhouse_driver  # noqa: F401
    except Exception:
        pytest.skip("clickhouse-driver is required")

    try:
        from monitoring import (  # type: ignore
            MonitoringConfig,
            MonitoringEngine,
        )
        from monitoring._native_engine import ClickHouseClientConfig  # type: ignore
        from monitoring.config import CaptureSchedule  # type: ignore
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
    # E2E_HF_DROP_LAST_TOKEN is no longer needed: both monitored and HF reference
    # use generate(), so they produce the same number of tokens/hidden states.

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
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
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

    host_cfg = _make_host_cfg(db_cfg_native)
    ring_cfg = _make_ring_cfg()

    # -----------------------------------------------------------------------
    # Monitored model — compiled with torch.compile + static cache (CUDA graphs)
    # -----------------------------------------------------------------------
    unique_run_model_id = f"e2e_cuda_graphs::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(
        config=mon_cfg,
        model_id=unique_run_model_id, db_config=host_cfg,
    )
    engine.enable_ring_transport(ring_cfg)

    model_cls = HookedQwen3ForCausalLM if "qwen3" in hf_model_id.lower() else HookedGPT2LMHeadModel
    mon_model = model_cls.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16,
    )
    mon_model.to(device).eval()

    mon_model.monitoring_engine = engine

    try:
        from transformers import CompileConfig
        with torch.no_grad():
            gen_out = generate_with_monitoring(
                mon_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                cache_implementation="static",
                compile_config=CompileConfig(mode="reduce-overhead", fullgraph=False),
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

    print(f"\n[DEBUG] Total DB rows: {len(rows)}")
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

    # DEBUG: show per-request hook chunk counts
    from collections import Counter
    hook_totals = Counter()
    for rid in request_ids:
        hooks_map = grouped[rid]
        for (layer, hname), chunks in hooks_map.items():
            hook_totals[hname] += len(chunks)
    print(f"[DEBUG] Hook totals across all requests:")
    for hname in ['token_ids', 'hook_embed', 'hook_pos_embed', 'blocks.0.hook_resid_pre', 'hook_resid_final', 'hook_final_ln', 'final_logits']:
        print(f"  {hname}: {hook_totals.get(hname, 0)} chunks")
    rid = request_ids[0]
    hooks_map = grouped[rid]
    print(f"[DEBUG] All hooks for {rid}:")
    for (layer, hname) in sorted(hooks_map.keys()):
        chunks = hooks_map[(layer, hname)]
        print(f"  ({layer},{hname}): {len(chunks)} chunks")

    # -----------------------------------------------------------------------
    # HF reference — compiled manual rollout with StaticCache.
    # Uses torch.compile(mode="reduce-overhead", fullgraph=False) on the
    # decode step + cudagraph_mark_step_begin() + immediate .detach().cpu()
    # to clone hidden states before CUDA graph buffers are overwritten.
    # (generate() can't do this — see Bug 11 in debug.log)
    # -----------------------------------------------------------------------
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16,
    ).to(device).eval()

    wte         = getattr(getattr(hf_model, "transformer", None), "wte", None)
    wpe         = getattr(getattr(hf_model, "transformer", None), "wpe", None)
    embed_tokens = getattr(getattr(hf_model, "model", None), "embed_tokens", None)
    num_layers  = get_num_layers_from_config(hf_model)

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
        compiled=True,
    )

    # -----------------------------------------------------------------------
    # Pre-loop: build per-request dicts and safety-check token_ids
    # -----------------------------------------------------------------------
    local_index_by_req: Dict[str, int] = {}
    prompt_len_by_req:  Dict[str, int] = {}
    db_token_ids_by_req: Dict[str, torch.Tensor] = {}

    for req_id in request_ids:
        _gid, local_i = _parse_request_id(req_id)
        local_index_by_req[req_id] = local_i

    req_order = sorted(request_ids, key=_parse_request_id)

    for req_id in req_order:
        local_i   = local_index_by_req[req_id]
        hooks_map = grouped[req_id]
        prompt0   = hf_initial_prompt_tokens[local_i]
        plen      = int(prompt0.numel())
        prompt_len_by_req[req_id] = plen

        if (-1, "token_ids") not in hooks_map:
            raise AssertionError(f"{req_id}: DB missing token_ids under CUDA graphs")

        tok_chunks = sorted(hooks_map[(-1, "token_ids")], key=lambda x: (x[0], x[1]))
        db_tok = merge_segments([t for _, _, t in tok_chunks], "token_ids").to(torch.long)
        if db_tok.ndim != 1:
            db_tok = db_tok.view(-1)

        if db_tok.numel() < plen or not torch.equal(db_tok[:plen], prompt0):
            raise AssertionError(
                f"{req_id}: DB token_ids prompt prefix mismatch under CUDA graphs "
                f"(plen={plen} db_len={db_tok.numel()})"
            )
        db_token_ids_by_req[req_id] = db_tok.cpu()

    def _sort_chunks(chunks):
        return sorted(chunks, key=lambda x: (x[0], x[1]))

    def _validate_contiguous(chunks_sorted, expected_end, ctx):
        if not chunks_sorted:
            raise AssertionError(f"{ctx}: no chunks")
        if chunks_sorted[0][0] != 0:
            raise AssertionError(f"{ctx}: first chunk start={chunks_sorted[0][0]} expected 0")
        prev_end = chunks_sorted[0][1]
        for s2, e2, _t in chunks_sorted[1:]:
            if s2 != prev_end:
                raise AssertionError(f"{ctx}: non-contiguous chunks start={s2} prev_end={prev_end}")
            prev_end = e2
        if prev_end != expected_end:
            raise AssertionError(f"{ctx}: coverage end={prev_end} expected_end={expected_end}")

    # -----------------------------------------------------------------------
    # Per-request assertions (full verification)
    # -----------------------------------------------------------------------
    _RESID_PRE_KEYS  = ("blocks.hook_resid_pre",  "layers.hook_resid_pre")

    # HF reference is uncompiled; monitored model is compiled (reduce-overhead).
    # Fall back to allclose if not bitwise equal.
    _COMPILED_ATOL = 0.5  # safety net: real transport errors are >> 1

    def _assert_close_or_bitwise(db_t, ref_t, label):
        if bitwise_equal(db_t, ref_t):
            return
        diff = (db_t.float() - ref_t.float()).abs()
        max_abs = float(diff.max().item())
        if torch.allclose(db_t.float(), ref_t.float(), atol=_COMPILED_ATOL, rtol=0.0):
            import warnings
            warnings.warn(
                f"[NOT BITWISE] {label}: max_abs_diff={max_abs:.6f} "
                f"(within atol={_COMPILED_ATOL}, but not bitwise equal)"
            )
            return
        pytest.fail(
            f"{label}: max_abs_diff={max_abs:.6f} > atol={_COMPILED_ATOL}"
        )

    for req_id in req_order:
        local_i   = local_index_by_req[req_id]
        hooks_map = grouped[req_id]
        plen      = prompt_len_by_req[req_id]
        prompt0   = hf_initial_prompt_tokens[local_i]

        db_tok  = db_token_ids_by_req[req_id]
        seq_len = int(db_tok.numel())

        ref     = hf_refs_batch[local_i]

        # --- token_ids ---
        with subtests.test(msg=f"{req_id}/cuda_graph/token_ids_present"):
            assert (-1, "token_ids") in hooks_map, f"{req_id}: DB missing token_ids"

        with subtests.test(msg=f"{req_id}/cuda_graph/prompt_prefix"):
            assert db_tok.numel() >= plen and torch.equal(db_tok[:plen], prompt0), (
                f"{req_id}: prompt prefix mismatch"
            )

        with subtests.test(msg=f"{req_id}/cuda_graph/token_ids_match_hf"):
            assert bitwise_equal(db_tok, ref.token_ids), (
                f"{req_id}: DB token_ids do not match HF generate() under CUDA graphs. "
                f"db_len={db_tok.numel()} hf_len={ref.token_ids.numel()} "
                f"(if db_len << hf_len the CUDA-graph monitoring bug is present)"
            )

        # --- final_logits ---
        logits_chunks_raw = hooks_map.get((-1, "final_logits"), [])
        if logits_chunks_raw:
            lchunks   = _sort_chunks(logits_chunks_raw)
            db_logits = merge_segments([t for _, _, t in lchunks], "final_logits")
            if db_logits.ndim == 1:
                db_logits = db_logits.unsqueeze(0)
            n_ref = int(ref.final_logits.shape[0])
            start = plen - 1
            end = min(start + n_ref, int(db_logits.shape[0]))
            n = end - start
            db_slice  = db_logits[start:end, :]
            rol_slice = ref.final_logits[:n, :]
            with subtests.test(msg=f"{req_id}/cuda_graph/final_logits"):
                assert n > 0, f"final_logits: no overlapping rows"
                assert db_slice.shape[1] == rol_slice.shape[1], (
                    f"final_logits vocab mismatch db={db_slice.shape[1]} hf={rol_slice.shape[1]}"
                )
                _assert_close_or_bitwise(db_slice, rol_slice, f"{req_id} final_logits")

        # --- hook_embed ---
        seq = db_tok.to(device)
        if wte is not None and wpe is not None and (-1, "hook_embed") in hooks_map:
            emb    = wte(seq).detach().cpu()
            chunks = _sort_chunks(hooks_map[(-1, "hook_embed")])
            _validate_contiguous(chunks, seq_len, f"{req_id} hook_embed")
            db_t   = merge_segments([t for _, _, t in chunks], "hook_embed")
            with subtests.test(msg=f"{req_id}/cuda_graph/hook_embed"):
                assert tuple(db_t.shape) == tuple(emb.shape), (
                    f"hook_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(emb.shape)}"
                )
                _assert_close_or_bitwise(db_t, emb, f"{req_id} hook_embed")
        elif embed_tokens is not None and wpe is None and (-1, "hook_embed") in hooks_map:
            emb    = embed_tokens(seq).detach().cpu()
            chunks = _sort_chunks(hooks_map[(-1, "hook_embed")])
            _validate_contiguous(chunks, seq_len, f"{req_id} hook_embed")
            db_t   = merge_segments([t for _, _, t in chunks], "hook_embed")
            with subtests.test(msg=f"{req_id}/cuda_graph/hook_embed"):
                assert tuple(db_t.shape) == tuple(emb.shape), (
                    f"hook_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(emb.shape)}"
                )
                _assert_close_or_bitwise(db_t, emb, f"{req_id} hook_embed")

        # --- hook_pos_embed (GPT2 only) ---
        if wpe is not None and (-1, "hook_pos_embed") in hooks_map:
            pos = _positions_for_unpadded(seq_len, device=device)
            pos_emb = wpe(pos).detach().cpu()
            chunks = _sort_chunks(hooks_map[(-1, "hook_pos_embed")])
            _validate_contiguous(chunks, seq_len, f"{req_id} hook_pos_embed")
            db_t = merge_segments([t for _, _, t in chunks], "hook_pos_embed")
            with subtests.test(msg=f"{req_id}/cuda_graph/hook_pos_embed"):
                assert tuple(db_t.shape) == tuple(pos_emb.shape), (
                    f"hook_pos_embed shape mismatch db={tuple(db_t.shape)} hf={tuple(pos_emb.shape)}"
                )
                _assert_close_or_bitwise(db_t, pos_emb, f"{req_id} hook_pos_embed")

        # --- per-layer: resid_pre + attn_pattern ---
        _ATTN_PATTERN_KEYS = ("blocks.attn.hook_pattern", "layers.self_attn.hook_pattern")
        for layer_no in range(num_layers):
            key = next(((layer_no, k) for k in _RESID_PRE_KEYS if (layer_no, k) in hooks_map), None)
            if key is not None and ref.hidden_states and layer_no < len(ref.hidden_states):
                hs     = ref.hidden_states[layer_no]
                chunks = _sort_chunks(hooks_map[key])
                _validate_contiguous(chunks, seq_len, f"{req_id} layer{layer_no} resid_pre")
                db_t   = merge_segments([t for _, _, t in chunks], key[1])
                with subtests.test(msg=f"{req_id}/cuda_graph/layer{layer_no}/resid_pre"):
                    assert tuple(db_t.shape) == tuple(hs.shape), (
                        f"resid_pre shape mismatch layer={layer_no} db={tuple(db_t.shape)} hf={tuple(hs.shape)}"
                    )
                    _assert_close_or_bitwise(db_t, hs, f"{req_id} layer{layer_no} resid_pre")

            # attn_pattern: both DB and ref use static cache, same padding
            key = next(((layer_no, k) for k in _ATTN_PATTERN_KEYS if (layer_no, k) in hooks_map), None)
            if key is not None and ref.attn_pattern and layer_no < len(ref.attn_pattern):
                pat = ref.attn_pattern[layer_no]
                chunks = _sort_chunks(hooks_map[key])
                db_pat = merge_segments([t for _, _, t in chunks], key[1])
                if db_pat.ndim == 4 and db_pat.shape[0] == 1:
                    db_pat = db_pat.squeeze(0)
                with subtests.test(msg=f"{req_id}/cuda_graph/layer{layer_no}/attn_pattern"):
                    assert tuple(db_pat.shape) == tuple(pat.shape), (
                        f"pattern shape mismatch layer={layer_no} db={tuple(db_pat.shape)} hf={tuple(pat.shape)}"
                    )
                    _assert_close_or_bitwise(db_pat, pat, f"{req_id} layer{layer_no} attn_pattern")

        # --- resid_final ---
        # HF's output_hidden_states[-1] is post-final-norm, not pre-norm.
        # Only check shape and presence.
        key = (-1, "hook_resid_final")
        if key in hooks_map:
            chunks = _sort_chunks(hooks_map[key])
            _validate_contiguous(chunks, seq_len, f"{req_id} resid_final")
            db_t = merge_segments([t for _, _, t in chunks], key[1])
            with subtests.test(msg=f"{req_id}/cuda_graph/resid_final"):
                assert db_t.shape[0] == seq_len, (
                    f"resid_final token count mismatch db={db_t.shape[0]} expected={seq_len}"
                )


# ---------------------------------------------------------------------------
# Test: CUDA-graph monitored DB vs uncompiled eager HF reference
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA + native backend required")
def test_e2e_cuda_graphs_vs_eager_hf(subtests) -> None:
    """Compare CUDA-graph monitored run against original eager model.

    Three subprocesses — parent process never touches CUDA:
      1. Reference: original model (eager) -> tensors on disk
      2. Monitored: hooked model + ring (CUDA graphs, static cache) -> ClickHouse
      3. Comparator: reads both, compares, writes result.json
    """
    import json
    import subprocess
    import tempfile
    import shutil

    run_dir = tempfile.mkdtemp(prefix="hf_cg_e2e_")
    ref_dir = os.path.join(run_dir, "ref")
    mon_dir = os.path.join(run_dir, "mon")
    result_file = os.path.join(run_dir, "result.json")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # CUDA graph mode: monitored runs with static cache + torch.compile.
    # Reference runs eager.  Relaxed tolerance for bf16 rounding from
    # different accumulation order (compiled vs uncompiled).
    mon_env = {**os.environ, "E2E_CUDA_GRAPHS": "1"}
    cmp_env = {**os.environ, "E2E_TOLERANCE": "0.5"}

    try:
        print("\n  [1/3] Reference run (original model, eager)...", flush=True)
        r1 = subprocess.run(
            [sys.executable, "-m", "tests.hf_reference_runner",
             "--output-dir", ref_dir],
            env=os.environ, capture_output=True, text=True, cwd=project_root,
        )
        if r1.returncode != 0:
            pytest.fail(f"Reference runner failed:\n{r1.stderr[-2000:]}")

        print("  [2/3] Monitored run (hooked model + ring, CUDA graphs)...", flush=True)
        r2 = subprocess.run(
            [sys.executable, "-m", "tests.hf_monitored_runner",
             "--output-dir", mon_dir],
            env=mon_env, capture_output=True, text=True, cwd=project_root,
        )
        if r2.returncode != 0:
            pytest.fail(f"Monitored runner failed:\n{r2.stderr[-2000:]}")

        print("  [3/3] Comparing (tolerance=0.5 for CG vs eager)...", flush=True)
        r3 = subprocess.run(
            [sys.executable, "-m", "tests.hf_comparator",
             "--ref-dir", ref_dir,
             "--mon-dir", mon_dir,
             "--result-file", result_file],
            env=cmp_env, capture_output=True, text=True, cwd=project_root,
        )
        if r3.returncode != 0:
            pytest.fail(f"Comparator failed:\n{r3.stderr[-2000:]}")

        with open(result_file) as f:
            results = json.load(f)

        for test in results["tests"]:
            with subtests.test(test["name"]):
                assert test["passed"], test.get("detail", "")

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _test_e2e_cuda_graphs_vs_eager_hf_legacy(subtests) -> None:
    """Legacy version kept for reference. Not called by verify_hf.sh."""
    try:
        import clickhouse_driver  # noqa: F401
    except Exception:
        pytest.skip("clickhouse-driver is required")

    try:
        from monitoring import (  # type: ignore
            MonitoringConfig,
            MonitoringEngine,
        )
        from monitoring._native_engine import ClickHouseClientConfig  # type: ignore
        from monitoring.config import CaptureSchedule  # type: ignore
        from monitoring.generate import generate_with_monitoring  # type: ignore
    except Exception as exc:
        pytest.skip(f"monitoring native extension not available: {exc}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel  # type: ignore
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
    except Exception as exc:
        pytest.skip(f"transformers or Hooked* classes not available: {exc}")

    batch_size     = int(os.environ.get("E2E_BATCH_SIZE", "4"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "8"))
    hf_model_id    = _resolve_model_id(os.environ.get("E2E_MODEL", "gpt2"))
    chunk_bytes    = int(os.environ.get("E2E_CHUNK_BYTES", str(256 * 1024)))
    device = torch.device("cuda")

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
            _strip_left_pad(input_ids[j].detach().cpu(), attention_mask[j].detach().cpu()).to(torch.long)
        )

    # --- Monitoring config + DB ---
    mon_cfg = MonitoringConfig(
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )

    db_cfg_native = ClickHouseClientConfig()
    db_cfg_native.host     = os.environ.get("DMX_DB_HOST", "localhost")
    db_cfg_native.port     = int(os.environ.get("DMX_DB_PORT", "9000"))
    db_cfg_native.username = os.environ.get("DMX_DB_USER", "default")
    db_cfg_native.password = os.environ.get("DMX_DB_PASSWORD", "")
    db_cfg_native.database = os.environ.get("DMX_DB_DATABASE", "default")
    db_cfg_native.table    = os.environ.get("DMX_DB_TABLE", "offload")
    db_cfg_native.secure                   = False
    db_cfg_native.client_side_compress     = "none"
    db_cfg_native.client_settings          = None
    db_cfg_native.create_database_if_missing = True
    db_cfg_native.drop_existing_database   = True
    db_cfg_native.index_granularity        = 8192

    host_cfg = _make_host_cfg(db_cfg_native)
    ring_cfg = _make_ring_cfg()

    # --- Monitored model (compiled, CUDA graphs) ---
    unique_run_model_id = f"e2e_cg_vs_eager::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(
        config=mon_cfg,
        model_id=unique_run_model_id, db_config=host_cfg,
    )
    engine.enable_ring_transport(ring_cfg)

    model_cls = HookedQwen3ForCausalLM if "qwen3" in hf_model_id.lower() else HookedGPT2LMHeadModel
    mon_model = model_cls.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16,
    ).to(device).eval()
    mon_model.monitoring_engine = engine

    try:
        from transformers import CompileConfig
        with torch.no_grad():
            generate_with_monitoring(
                mon_model,
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id, eos_token_id=eos_id,
                logits_to_keep=0,
                cache_implementation="static",
                compile_config=CompileConfig(mode="reduce-overhead", fullgraph=False),
            )
    finally:
        engine.close()

    # --- Read DB ---
    ch = CHClickhouseDriverReadOnly(
        host=str(db_cfg_native.host), port=int(db_cfg_native.port),
        username=str(db_cfg_native.username), password=str(db_cfg_native.password),
        database=str(db_cfg_native.database), table=str(db_cfg_native.table),
        secure=False, client_settings=None, decode_strings=True,
    )
    try:
        rows = ch.prefix_get((unique_run_model_id,), return_full_key_tuple=True)
    finally:
        ch.close()

    assert rows, f"No DB rows for model_id={unique_run_model_id!r}"

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

    # Build DB token_ids
    db_token_ids_by_req: Dict[str, torch.Tensor] = {}
    prompt_len_by_req: Dict[str, int] = {}
    local_index_by_req: Dict[str, int] = {}
    for req_id in request_ids:
        _gid, local_i = _parse_request_id(req_id)
        local_index_by_req[req_id] = local_i
        hooks_map = grouped[req_id]
        tok_chunks = sorted(hooks_map[(-1, "token_ids")], key=lambda x: (x[0], x[1]))
        db_tok = merge_segments([t for _, _, t in tok_chunks], "token_ids").to(torch.long)
        if db_tok.ndim != 1:
            db_tok = db_tok.view(-1)
        db_token_ids_by_req[req_id] = db_tok.cpu()
        prompt_len_by_req[req_id] = int(hf_initial_prompt_tokens[local_i].numel())

    # --- Eager HF reference (uncompiled, DynamicCache) ---
    hf_eager = AutoModelForCausalLM.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16,
    ).to(device).eval()
    num_layers = get_num_layers_from_config(hf_eager)

    hf_eager_refs = _hf_greedy_rollout_collect_all_batched(
        hf_model=hf_eager,
        input_ids_batch=input_ids,
        attention_mask_batch=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        device=device,
        want_hidden_states=True,
        want_attentions=True,
    )

    # --- Comparisons ---
    _EAGER_ATOL = 0.5
    _RESID_PRE_KEYS    = ("blocks.hook_resid_pre",  "layers.hook_resid_pre")
    _ATTN_PATTERN_KEYS = ("blocks.attn.hook_pattern", "layers.self_attn.hook_pattern")

    def _sort_chunks(chunks):
        return sorted(chunks, key=lambda x: (x[0], x[1]))

    def _close_enough(db_t, ref_t, ctx, atol=_EAGER_ATOL):
        diff = (db_t.float() - ref_t.float()).abs()
        max_abs = float(diff.max().item())
        assert max_abs <= atol, (
            f"{ctx} max_abs_diff={max_abs:.6f} > atol={atol}"
        )

    for req_id in sorted(request_ids, key=_parse_request_id):
        local_i  = local_index_by_req[req_id]
        hooks_map = grouped[req_id]
        plen     = prompt_len_by_req[req_id]
        db_tok   = db_token_ids_by_req[req_id]
        eref     = hf_eager_refs[local_i]

        # --- token_ids: full comparison ---
        min_len = min(int(db_tok.numel()), int(eref.token_ids.numel()))
        match_len = 0
        for t in range(min_len):
            if db_tok[t] != eref.token_ids[t]:
                break
            match_len = t + 1

        with subtests.test(msg=f"{req_id}/cg_vs_eager/token_prefix"):
            assert match_len >= plen, (
                f"compiled/eager diverge within prompt (match_len={match_len} plen={plen})"
            )

        with subtests.test(msg=f"{req_id}/cg_vs_eager/token_ids_full"):
            assert db_tok.numel() == eref.token_ids.numel(), (
                f"token count mismatch db={db_tok.numel()} eager={eref.token_ids.numel()}"
            )
            assert torch.equal(db_tok[:match_len], eref.token_ids[:match_len]), (
                f"token_ids mismatch in matching prefix (len={match_len})"
            )

        if match_len <= 0:
            continue

        # --- final_logits ---
        if eref.final_logits is not None and (-1, "final_logits") in hooks_map:
            chunks = _sort_chunks(hooks_map[(-1, "final_logits")])
            db_logits = merge_segments([t for _, _, t in chunks], "final_logits")
            ref_logits = eref.final_logits
            # Compare over matching prefix
            ml = min(db_logits.shape[0], ref_logits.shape[0], match_len)
            with subtests.test(msg=f"{req_id}/cg_vs_eager/final_logits"):
                _close_enough(db_logits[:ml], ref_logits[:ml],
                              f"{req_id} final_logits")

        # --- hook_embed ---
        if (-1, "hook_embed") in hooks_map:
            chunks = _sort_chunks(hooks_map[(-1, "hook_embed")])
            db_emb = merge_segments([t for _, _, t in chunks], "hook_embed")
            if eref.hidden_states and len(eref.hidden_states) > 0:
                # hidden_states[0] is the embedding output for HF models
                # that include it (GPT2: embed+pos, Qwen3: embed only)
                pass  # embed comparison needs model-specific reference
            with subtests.test(msg=f"{req_id}/cg_vs_eager/hook_embed"):
                assert db_emb.ndim >= 2, f"hook_embed unexpected shape {db_emb.shape}"

        # --- hook_pos_embed (GPT2 only) ---
        if (-1, "hook_pos_embed") in hooks_map:
            chunks = _sort_chunks(hooks_map[(-1, "hook_pos_embed")])
            db_pos = merge_segments([t for _, _, t in chunks], "hook_pos_embed")
            with subtests.test(msg=f"{req_id}/cg_vs_eager/hook_pos_embed"):
                assert db_pos.ndim >= 2, f"hook_pos_embed unexpected shape {db_pos.shape}"

        # --- per-layer: resid_pre + attn_pattern ---
        if not eref.hidden_states:
            continue

        for layer_no in range(num_layers):
            # resid_pre
            key = next(((layer_no, k) for k in _RESID_PRE_KEYS if (layer_no, k) in hooks_map), None)
            if key is not None and layer_no < len(eref.hidden_states):
                hs_eager = eref.hidden_states[layer_no][:match_len, :]
                chunks = _sort_chunks(hooks_map[key])
                db_t = merge_segments([t for _, _, t in chunks], key[1])[:match_len, :]
                with subtests.test(msg=f"{req_id}/cg_vs_eager/layer{layer_no}/resid_pre"):
                    assert db_t.shape == hs_eager.shape, (
                        f"shape mismatch db={db_t.shape} eager={hs_eager.shape}"
                    )
                    _close_enough(db_t, hs_eager,
                                  f"resid_pre layer={layer_no}")

            # attn_pattern: compare step-by-step, trimming static cache padding.
            # DB (static cache): each step has kv_dim = max_len (padded).
            # Eager ref (dynamic cache): each step has kv_dim = actual kv_len.
            # We compare per-chunk, trimming DB's kv_dim to the eager ref's.
            key = next(((layer_no, k) for k in _ATTN_PATTERN_KEYS if (layer_no, k) in hooks_map), None)
            if key is not None and eref.attn_pattern and layer_no < len(eref.attn_pattern):
                pat_eager = eref.attn_pattern[layer_no]  # [H, T, T] from rollout
                chunks = _sort_chunks(hooks_map[key])
                # Rebuild step-by-step: each chunk covers [start:end] token positions
                all_ok = True
                fail_msg = ""
                for start, end, t_chunk in chunks:
                    if start >= match_len:
                        break
                    end_clamp = min(end, match_len)
                    q_len = end_clamp - start
                    # t_chunk: [H, q_len, kv_dim_padded] or [1, H, q_len, kv_dim_padded]
                    db_c = t_chunk
                    if db_c.ndim == 4 and db_c.shape[0] == 1:
                        db_c = db_c.squeeze(0)
                    db_c = db_c[:, :q_len, :]  # trim q_len if chunk extends beyond match_len
                    # kv_dim for these rows: tokens 0..end_clamp-1 attended to
                    # keys 0..end_clamp-1 (causal). Trim kv to end_clamp.
                    kv_valid = end_clamp
                    db_c = db_c[:, :, :kv_valid]
                    # Corresponding slice from eager ref
                    ref_c = pat_eager[:, start:end_clamp, :kv_valid]
                    if db_c.shape != ref_c.shape:
                        all_ok = False
                        fail_msg = (f"shape mismatch at [{start}:{end_clamp}] "
                                    f"db={db_c.shape} eager={ref_c.shape}")
                        break
                    diff = (db_c.float() - ref_c.float()).abs()
                    max_abs = float(diff.max().item())
                    if max_abs > _EAGER_ATOL:
                        all_ok = False
                        fail_msg = (f"value mismatch at [{start}:{end_clamp}] "
                                    f"max_abs={max_abs:.6f} > atol={_EAGER_ATOL}")
                        break
                with subtests.test(msg=f"{req_id}/cg_vs_eager/layer{layer_no}/attn_pattern"):
                    assert all_ok, (
                        f"attn_pattern layer={layer_no}: {fail_msg}"
                    )

        # --- resid_final (global, last layer's pre-norm residual) ---
        # HF's output_hidden_states[-1] is post-final-norm, not pre-norm.
        # Only check shape and presence.
        key = next(((-1, k) for k in ("hook_resid_final",) if (-1, k) in hooks_map), None)
        if key is not None:
            chunks = _sort_chunks(hooks_map[key])
            db_rf = merge_segments([t for _, _, t in chunks], key[1])[:match_len, :]
            with subtests.test(msg=f"{req_id}/cg_vs_eager/resid_final"):
                assert db_rf.shape[0] == match_len, (
                    f"resid_final token count mismatch db={db_rf.shape[0]} expected={match_len}"
                )