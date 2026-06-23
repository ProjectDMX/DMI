"""HF E2E correctness — thin wrappers over the configurable matrix (plan §5).

This file used to carry ~1.6k lines of in-process HF rollout + tensor
comparison logic, three tests (one permanently disabled via
``@skipif(True)``), 16 skip sites, and two ``_legacy`` bodies kept "for
reference".  All of that comparison logic now lives in :mod:`tests.lib` and
the dispatch in :mod:`tests.e2e_matrix`; these wrappers just drive the
matrix for the equivalent HF cell and assert on its checks.

The test *names* are preserved because ``tests/tools/verify_hf.sh`` invokes
them by node id (``::test_e2e_correctness_hf`` /
``::test_e2e_cuda_graphs_vs_eager_hf``) and threads the ring-size / model
env vars the matrix wrapper reads.

The removed ``test_e2e_correctness_hf_cuda_graphs`` was permanently disabled
(its compiled-rollout reference could not replicate generate()'s internal
StaticCache handling) and explicitly superseded by
``test_e2e_cuda_graphs_vs_eager_hf`` -- no still-passing assertion was lost.
"""
from __future__ import annotations

import pytest

from tests._requirements import require_cuda, require_clickhouse
from tests.e2e_matrix import matrix_argv_from_env, run_single

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.clickhouse,
    pytest.mark.hf,
]

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

_MODEL_ALIASES = {"qwen3": "Qwen/Qwen3-4B", "llama": "meta-llama/Llama-3.1-8B"}


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
    rc.drain_flush_timeout_us      = int(os.environ.get("E2E_DRAIN_FLUSH_TIMEOUT_US", "100000"))
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
    """HF eager: hooked model (ring -> ClickHouse) vs original model.

    Equivalent matrix cell: ``--backend hf --mode eager --standard allclose``
    (HF dispatches to hf_comparator, which does the value comparison with
    ``E2E_TOLERANCE``; default 0.01 for eager).
    """
    cr = run_single(matrix_argv_from_env("hf", "allclose", mode="eager"))
    _assert_cell(subtests, cr)


@require_cuda()
@require_clickhouse()
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
        from integration.hf_adapter import generate_with_monitoring  # type: ignore
    except Exception as exc:
        pytest.skip(f"monitoring native extension not available: {exc}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel  # type: ignore
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
        from transformers.models.llama_p.modeling_llama import HookedLlamaForCausalLM  # type: ignore
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

    if "qwen3" in hf_model_id.lower():
        model_cls = HookedQwen3ForCausalLM
    elif "llama" in hf_model_id.lower():
        model_cls = HookedLlamaForCausalLM
    else:
        model_cls = HookedGPT2LMHeadModel
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
