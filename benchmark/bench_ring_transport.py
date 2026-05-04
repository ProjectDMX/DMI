"""Ring-transport benchmark.

Compares four modes:
  baseline         -- plain HF generate, no monitoring
  ring_kernels_only-- ring transport active, producer kernels fire but do zero work (null_mode on)
  ring_null        -- ring transport active (GPU->CPU transfer), null sink (no DB write)
  ring_db          -- ring transport + ClickHouse ingestion

Per-step (prefill vs decode) timing is intentionally NOT reported here.
Inserting a GPU sync barrier between every step breaks CUDA-graph pipelining
and makes per-step numbers unreliable.  Instead, run dedicated prefill /
decode benchmarks using the appropriate --prefill-len / --decode-len flags:

  Prefill:  --prefill-len N --decode-len 1
  Decode:   --prefill-len 1 --decode-len N

Each run reports total wall time (ms) and throughput (tok/s).
For prefill, tok/s counts prompt tokens processed (batch * prefill_len).
For decode,  tok/s counts new tokens generated  (batch * decode_len).

Usage:
  python -m benchmark.bench_ring_transport --model qwen3 --modes baseline,ring_null
"""

from __future__ import annotations

import argparse
import functools
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Timed close: ring drain vs DB drain
# ---------------------------------------------------------------------------

def _timed_close(engine) -> Dict[str, float]:
    result = {"ring_ms": 0.0, "db_ms": 0.0, "cleanup_ms": 0.0}

    ring_engine = getattr(engine, "_ring_engine", None)
    if ring_engine is not None:
        t0 = time.perf_counter()
        try:
            ring_engine.stop()
        except Exception:
            pass
        result["ring_ms"] = (time.perf_counter() - t0) * 1000.0

    host_engine = getattr(engine, "_host_engine", None)
    if host_engine is not None:
        t0 = time.perf_counter()
        try:
            host_engine.stop()
        except Exception:
            pass
        result["db_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    try:
        engine.close()
    except Exception:
        pass
    result["cleanup_ms"] = (time.perf_counter() - t0) * 1000.0

    return result


# ---------------------------------------------------------------------------
# Null host-engine sink
# ---------------------------------------------------------------------------

class _NullHostEngine:
    def start(self)                   -> None:  pass
    def stop(self, *a, **kw)          -> None:  pass
    def join(self, *a, **kw)          -> bool:  return True
    def close_input(self)             -> None:  pass
    def request_abort(self)           -> None:  pass
    def failures(self)                -> list:  return []
    def raise_if_failed(self)         -> None:  pass
    def submit_direct(self, *a, **kw) -> None:  pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    model: str = "gpt2"
    batch_size: int = 4
    prefill_len: int = 32
    decode_len: int = 16
    warmup: int = 1
    iters: int = 3
    modes: List[str] = field(default_factory=lambda: ["baseline", "ring_null"])
    cuda_graphs: bool = False
    logits_to_keep: int = 0  # 0 = keep all, 1 = last position only (HF default)
    hook_selection: str = "full"  # full, hf-only, hidden-states, logits, attention
    hf_offload_hidden_states: bool = False
    hf_offload_attentions: bool = False
    hf_offload_logits: bool = False

    csv_path: Optional[str] = None

    ring_task_entries: int = 65536
    ring_payload_mb: int = 4096
    ring_pinned_mb: int = 4096
    drain_poll_timeout_us: int = 100
    drain_flush_task_ratio: float = 0.0
    drain_flush_payload_ratio: float = 0.0
    drain_flush_entry_threshold: int = 0
    drain_flush_byte_threshold: int = 0
    drain_flush_timeout_us: int = 0
    clone_slices: bool = False

    ch_parallelism: int = 10
    ch_queue_max_items: int = 1024
    ch_queue_max_size_mb: int = 2048

    db_host: str = "localhost"
    db_port: int = 9000
    db_user: str = "default"
    db_password: str = ""
    db_database: str = "default"
    db_table: str = "offload_bench"


# ---------------------------------------------------------------------------
# Engine builders
# ---------------------------------------------------------------------------

def _make_ring_cfg(cfg: BenchConfig):
    from monitoring._native_engine import RingConfig  # type: ignore
    rc = RingConfig()
    rc.task_ring_entries  = cfg.ring_task_entries
    rc.payload_ring_bytes = cfg.ring_payload_mb * 1024 * 1024
    rc.pinned_staging_bytes  = cfg.ring_pinned_mb  * 1024 * 1024
    rc.drain_poll_timeout_us       = cfg.drain_poll_timeout_us
    rc.drain_flush_task_ratio      = cfg.drain_flush_task_ratio
    rc.drain_flush_payload_ratio   = cfg.drain_flush_payload_ratio
    rc.drain_flush_entry_threshold = cfg.drain_flush_entry_threshold
    rc.drain_flush_byte_threshold  = cfg.drain_flush_byte_threshold
    rc.drain_flush_timeout_us      = cfg.drain_flush_timeout_us
    rc.clone_slices                = cfg.clone_slices
    rc.insert_queue_max_bytes      = cfg.ch_queue_max_size_mb * 1024 * 1024
    rc.insert_queue_max_items      = cfg.ch_queue_max_items
    return rc


def _make_monitoring_cfg(cfg: BenchConfig):
    from monitoring import MonitoringConfig  # type: ignore
    from monitoring.config import CaptureSchedule  # type: ignore
    return MonitoringConfig(
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )


def _make_null_engine(cfg: BenchConfig, model_id: str):
    from monitoring import MonitoringEngine  # type: ignore
    engine = MonitoringEngine(
        config=_make_monitoring_cfg(cfg),
        model_id=model_id, host_engine=_NullHostEngine(),
    )
    engine.enable_ring_transport(_make_ring_cfg(cfg))
    return engine


def _make_db_engine(cfg: BenchConfig, model_id: str):
    from monitoring import HostEngineConfig, MonitoringEngine  # type: ignore
    from monitoring._native_engine import ClickHouseClientConfig, StageConfig  # type: ignore

    ch = ClickHouseClientConfig()
    ch.host = cfg.db_host;  ch.port = cfg.db_port
    ch.username = cfg.db_user;  ch.password = cfg.db_password
    ch.database = cfg.db_database;  ch.table = cfg.db_table
    ch.secure = False;  ch.client_side_compress = "none"
    ch.client_settings = None
    ch.create_database_if_missing = True
    ch.drop_existing_database = True
    ch.index_granularity = 8192

    stage = StageConfig.clickhouse_insert(ch, parallelism=cfg.ch_parallelism, name="ch_insert")
    q = stage.input_queue
    q.max_batch_items      = cfg.ch_queue_max_items
    q.high_watermark_items = cfg.ch_queue_max_items
    q.max_batch_size       = cfg.ch_queue_max_size_mb * 1024 * 1024
    q.high_watermark_size  = cfg.ch_queue_max_size_mb * 1024 * 1024

    engine = MonitoringEngine(
        config=_make_monitoring_cfg(cfg),
        model_id=model_id, db_config=HostEngineConfig(stages=[stage]),
    )
    engine.enable_ring_transport(_make_ring_cfg(cfg))
    return engine


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _make_inputs(tokenizer, cfg: BenchConfig, device: torch.device):
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 20)
    ids = ids[:cfg.prefill_len]
    rows = [torch.tensor(ids, dtype=torch.long) for _ in range(cfg.batch_size)]
    input_ids = torch.stack(rows).to(device)
    return input_ids, torch.ones_like(input_ids)


# ---------------------------------------------------------------------------
# Per-iteration runner -- returns total wall time only
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    total_ms: float
    prefill_ms: float
    decode_ms: float
    decode_steps: int


def _run_one(model, input_ids, attention_mask,
             cfg: BenchConfig, eos_id: int, pad_id: int,
             use_monitoring: bool) -> RunResult:
    """Run one generate() and return timing breakdown.

    When cuda_graphs is enabled, uses generate_greedy_with_monitoring() for both baseline
    and monitored modes — same lean loop, only difference is monitoring=True/False.
    This eliminates HF generate()'s per-step Python overhead from the comparison.

    When cuda_graphs is disabled, falls back to HF generate() / generate_with_monitoring()
    for the full HF experience.
    """
    if cfg.cuda_graphs:
        return _run_one_greedy(model, input_ids, attention_mask,
                               cfg, eos_id, pad_id, use_monitoring)
    return _run_one_hf_generate(model, input_ids, attention_mask,
                                cfg, eos_id, pad_id, use_monitoring)


def _run_one_greedy(model, input_ids, attention_mask,
                    cfg: BenchConfig, eos_id: int, pad_id: int,
                    use_monitoring: bool) -> RunResult:
    """Run one iteration using generate_greedy_with_monitoring (lean manual loop)."""
    from integration.hf_adapter import generate_greedy_with_monitoring, GreedyGenerateTimings

    timings = GreedyGenerateTimings()
    with torch.no_grad():
        generate_greedy_with_monitoring(
            model, input_ids, attention_mask,
            max_new_tokens=cfg.decode_len,
            min_new_tokens=cfg.decode_len,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            logits_to_keep=cfg.logits_to_keep,
            cuda_graphs=True,
            monitoring=use_monitoring,
            hook_selection=cfg.hook_selection if use_monitoring else None,
            timings=timings,
        )

    return RunResult(
        total_ms=timings.total_ms,
        prefill_ms=timings.prefill_ms,
        decode_ms=timings.decode_ms,
        decode_steps=timings.decode_steps,
    )


def _run_one_hf_generate(model, input_ids, attention_mask,
                         cfg: BenchConfig, eos_id: int, pad_id: int,
                         use_monitoring: bool) -> RunResult:
    """Run one iteration using HF generate() (original path).

    Per-step timing relies on Python-side ``time.perf_counter()`` recorded at
    the start of each generate step (inside ``prepare_inputs_for_generation``).
    This is accurate because HF's generate loop unconditionally executes::

        this_peer_finished = unfinished_sequences.max() == 0

    at the end of every step (transformers/generation/utils.py), which transfers
    a GPU scalar to CPU and forces an implicit cudaStreamSynchronize.  By the
    time the next ``prepare_inputs_for_generation`` call begins, all GPU work
    from the previous step has completed.
    """
    from integration.hf_adapter import generate_with_monitoring  # type: ignore

    extra = {}
    extra["logits_to_keep"] = cfg.logits_to_keep

    # Wrap prepare_inputs_for_generation to record per-step timestamps.
    step_timestamps: List[float] = []
    orig_prepare = model.prepare_inputs_for_generation

    @functools.wraps(orig_prepare)
    def _timed_prepare(*args, **kwargs):
        step_timestamps.append(time.perf_counter())
        return orig_prepare(*args, **kwargs)

    model.prepare_inputs_for_generation = _timed_prepare

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if use_monitoring:
                generate_with_monitoring(
                    model, input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=cfg.decode_len,
                    min_new_tokens=cfg.decode_len,
                    do_sample=False,
                    pad_token_id=pad_id, eos_token_id=eos_id,
                    hook_selection=cfg.hook_selection, **extra,
                )
            else:
                model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=cfg.decode_len,
                    min_new_tokens=cfg.decode_len,
                    do_sample=False,
                    pad_token_id=pad_id, eos_token_id=eos_id, **extra,
                )

        torch.cuda.synchronize()
        t_end = time.perf_counter()
    finally:
        model.prepare_inputs_for_generation = orig_prepare

    total_ms = (t_end - t0) * 1000.0
    if len(step_timestamps) >= 2:
        prefill_ms = (step_timestamps[1] - step_timestamps[0]) * 1000.0
        decode_ms = (t_end - step_timestamps[1]) * 1000.0
        decode_steps = len(step_timestamps) - 1
    else:
        prefill_ms = total_ms
        decode_ms = 0.0
        decode_steps = 0

    return RunResult(total_ms=total_ms, prefill_ms=prefill_ms,
                     decode_ms=decode_ms, decode_steps=decode_steps)



# ---------------------------------------------------------------------------
# Per-mode benchmark
# ---------------------------------------------------------------------------

def _run_mode(mode: str, model, input_ids, attention_mask,
              cfg: BenchConfig, eos_id: int, pad_id: int) -> dict:

    use_monitoring = mode not in ("baseline", "hf_offload")

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"{'='*60}")

    # --- hf_offload: completely independent code path. ---
    # Loads its own uncompiled model, compiles only the decode step function
    # (NOT the model), uses StaticCache + cudagraph_mark_step_begin().
    # Follows exactly the pattern from _compiled_hf_decode_refs in the
    # correctness test.
    if mode == "hf_offload":
        from transformers import AutoModelForCausalLM, StaticCache

        model_id_str = _MODEL_ALIASES.get(cfg.model.lower(), cfg.model)
        print(f"  Loading fresh uncompiled AutoModelForCausalLM...", flush=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id_str, attn_implementation="eager", torch_dtype=torch.float16,
        ).to(input_ids.device).eval()

        B, Pmax = input_ids.shape
        max_cache_len = Pmax + cfg.decode_len + 4

        want_hs   = cfg.hf_offload_hidden_states
        want_attn = cfg.hf_offload_attentions
        want_log  = cfg.hf_offload_logits
        print(f"  offload: hidden_states={want_hs} attentions={want_attn} logits={want_log}",
              flush=True)

        def _hf_prefill(cache_obj):
            cache_obj.reset()
            cache_pos = torch.arange(Pmax, device=input_ids.device)
            out = hf_model(
                input_ids=input_ids, attention_mask=attention_mask,
                cache_position=cache_pos, past_key_values=cache_obj,
                use_cache=True,
                output_hidden_states=want_hs,
                output_attentions=want_attn,
                return_dict=True,
                logits_to_keep=cfg.logits_to_keep,
            )
            return out

        def _hf_decode_step(token, cache, cache_position):
            out = hf_model(
                token, use_cache=True, past_key_values=cache,
                cache_position=cache_position,
                output_hidden_states=want_hs,
                output_attentions=want_attn,
                return_dict=True,
                logits_to_keep=cfg.logits_to_keep,
            )
            hs = tuple(out.hidden_states) if want_hs and out.hidden_states else ()
            at = tuple(out.attentions) if want_attn and out.attentions else ()
            return out.logits, hs, at

        if cfg.cuda_graphs:
            compiled_decode = torch.compile(
                _hf_decode_step, mode="reduce-overhead", fullgraph=False)
        else:
            compiled_decode = _hf_decode_step

        def _run_one_offload() -> RunResult:
            """Manual prefill + decode loop with per-step timing.

            HF's generate() forces an implicit CPU sync every step via
            ``unfinished_sequences.max() == 0`` (GPU scalar -> CPU).  This
            manual loop does not have that, so we add an explicit
            ``token.max().item()`` after each step to match the same sync
            behavior and ensure Python-side timestamps are accurate.
            """
            cache = StaticCache(
                config=hf_model.config, batch_size=B,
                max_cache_len=max_cache_len, device=input_ids.device,
                dtype=torch.float16,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                # Prefill (uncompiled)
                out = _hf_prefill(cache)
                if want_hs and out.hidden_states:
                    for hs in out.hidden_states:
                        hs.detach().cpu()
                if want_attn and out.attentions:
                    for a in out.attentions:
                        a.detach().cpu()
                if want_log:
                    out.logits.detach().cpu()

                token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                # Force CPU sync to match HF generate()'s implicit sync
                token.max().item()
                cache_pos = torch.tensor([Pmax], device=input_ids.device, dtype=torch.long)

                t_decode_start = time.perf_counter()

                # Decode
                for step in range(cfg.decode_len - 1):
                    torch.compiler.cudagraph_mark_step_begin()
                    logits, hidden_states, attentions = compiled_decode(
                        token, cache, cache_pos)
                    # Clone immediately
                    if want_hs:
                        for hs in hidden_states:
                            hs.detach().cpu()
                    if want_attn:
                        for a in attentions:
                            a.detach().cpu()
                    if want_log:
                        logits.detach().cpu()

                    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    # Force CPU sync to match HF generate()'s implicit
                    # ``this_peer_finished = unfinished_sequences.max() == 0``
                    token.max().item()
                    cache_pos = cache_pos + 1

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            total_ms = (t_end - t0) * 1000.0
            prefill_ms = (t_decode_start - t0) * 1000.0
            decode_ms = (t_end - t_decode_start) * 1000.0
            decode_steps = cfg.decode_len - 1
            return RunResult(total_ms=total_ms, prefill_ms=prefill_ms,
                             decode_ms=decode_ms, decode_steps=decode_steps)

        print(f"  Warming up ({cfg.warmup} iter)...", flush=True)
        for _ in range(cfg.warmup):
            _run_one_offload()

        torch.cuda.synchronize()
        time.sleep(1)
        print("  -- warmup done, starting measured iters --", flush=True)

        all_runs: List[RunResult] = []
        for i in range(cfg.iters):
            r = _run_one_offload()
            all_runs.append(r)
            tpot = r.decode_ms / r.decode_steps if r.decode_steps > 0 else 0.0
            print(f"  iter {i+1:2d}:  total={r.total_ms:7.1f} ms  "
                  f"prefill={r.prefill_ms:.1f} ms  "
                  f"decode={r.decode_ms:.1f} ms  "
                  f"TPOT={tpot:.2f} ms", flush=True)

        mean_t = statistics.mean([r.total_ms for r in all_runs])
        std_t  = statistics.stdev([r.total_ms for r in all_runs]) if len(all_runs) > 1 else 0.0
        mean_prefill = statistics.mean([r.prefill_ms for r in all_runs])
        mean_decode  = statistics.mean([r.decode_ms for r in all_runs])
        mean_steps   = statistics.mean([r.decode_steps for r in all_runs])
        mean_tpot    = mean_decode / mean_steps if mean_steps > 0 else 0.0
        decode_throughput = (cfg.batch_size * mean_steps / mean_decode * 1000) if mean_decode > 0 else 0.0
        prefill_throughput = (cfg.batch_size * cfg.prefill_len / mean_prefill * 1000) if mean_prefill > 0 else 0.0
        e2e_throughput = (cfg.batch_size * (cfg.prefill_len + mean_steps) / mean_t * 1000) if mean_t > 0 else 0.0
        print(f"\n  Summary:  total={mean_t:.1f} ms  std={std_t:.1f} ms  "
              f"prefill={mean_prefill:.1f} ms  "
              f"decode={mean_decode:.1f} ms  "
              f"TPOT={mean_tpot:.2f} ms  "
              f"decode_throughput={decode_throughput:.1f} tok/s")

        del hf_model, compiled_decode
        torch.cuda.empty_cache()

        return {
            "mode": mode, "mean_ms": mean_t, "std_ms": std_t,
            "min_ms": min(r.total_ms for r in all_runs),
            "max_ms": max(r.total_ms for r in all_runs),
            "prefill_throughput": prefill_throughput,
            "decode_throughput": decode_throughput,
            "e2e_throughput": e2e_throughput,
            "prefill_ms": mean_prefill, "decode_ms": mean_decode,
            "tpot_ms": mean_tpot,
            "close_ring_ms": 0.0, "close_db_ms": 0.0,
        }

    engine = None
    if mode in ("ring_null", "ring_kernels_only"):
        model_id = f"bench::{mode}::{uuid.uuid4().hex[:8]}"
        engine = _make_null_engine(cfg, model_id)
        model.monitoring_engine = engine
    elif mode == "ring_db":
        model_id = f"bench::{mode}::{uuid.uuid4().hex[:8]}"
        engine = _make_db_engine(cfg, model_id)
        model.monitoring_engine = engine
    else:
        model.monitoring_engine = None

    try:
        # Warmup -- null mode so producer kernels fire (same CUDA graph topology)
        # but the kernel body is a no-op; ring buffer and drain pipeline are idle.
        print(f"  Warming up ({cfg.warmup} iter)...", flush=True)
        ring_engine    = getattr(engine, "_ring_engine",    None)
        ring_transport = getattr(engine, "_ring_transport", None)
        if ring_engine is not None:
            ring_engine.set_null_mode(True)
        if ring_transport is not None:
            ring_transport.null_offload = True
        for _ in range(cfg.warmup):
            _run_one(model, input_ids, attention_mask,
                     cfg, eos_id, pad_id, use_monitoring)
        if mode != "ring_kernels_only":
            if ring_engine is not None:
                ring_engine.set_null_mode(False)
            if ring_transport is not None:
                ring_transport.null_offload = False

        torch.cuda.synchronize()
        time.sleep(1)
        print("  -- warmup done, starting measured iters --", flush=True)

        # Measured iterations
        all_runs: List[RunResult] = []
        close_t: Dict[str, float] = {"ring_ms": 0.0, "db_ms": 0.0, "cleanup_ms": 0.0}

        for i in range(cfg.iters):
            r = _run_one(model, input_ids, attention_mask,
                         cfg, eos_id, pad_id, use_monitoring)
            all_runs.append(r)
            tpot = r.decode_ms / r.decode_steps if r.decode_steps > 0 else 0.0
            print(f"  iter {i+1:2d}:  total={r.total_ms:7.1f} ms  "
                  f"prefill={r.prefill_ms:.1f} ms  "
                  f"decode={r.decode_ms:.1f} ms  "
                  f"TPOT={tpot:.2f} ms",
                  flush=True)

        # Close on last iter
        if engine is not None:
            close_t = _timed_close(engine)
            engine = None

        ring_ms    = close_t["ring_ms"]
        db_ms      = close_t["db_ms"]
        cleanup_ms = close_t["cleanup_ms"]
        print(f"\n  Close breakdown:")
        print(f"    ring drain  : {ring_ms:7.1f} ms")
        print(f"    db drain    : {db_ms:7.1f} ms")
        print(f"    cleanup     : {cleanup_ms:7.1f} ms")

        # Print pre-forward profiling if enabled
        try:
            from integration.hf_adapter import _prepare_profile_times, print_prepare_profile
            if _prepare_profile_times:
                print()
                print_prepare_profile()
                _prepare_profile_times.clear()
        except Exception:
            pass

        mean_t = statistics.mean([r.total_ms for r in all_runs])
        std_t  = statistics.stdev([r.total_ms for r in all_runs]) if len(all_runs) > 1 else 0.0
        mean_prefill = statistics.mean([r.prefill_ms for r in all_runs])
        mean_decode  = statistics.mean([r.decode_ms for r in all_runs])
        mean_steps   = statistics.mean([r.decode_steps for r in all_runs])
        mean_tpot    = mean_decode / mean_steps if mean_steps > 0 else 0.0
        decode_throughput = (cfg.batch_size * mean_steps / mean_decode * 1000) if mean_decode > 0 else 0.0
        prefill_throughput = (cfg.batch_size * cfg.prefill_len / mean_prefill * 1000) if mean_prefill > 0 else 0.0
        e2e_throughput = (cfg.batch_size * (cfg.prefill_len + mean_steps) / mean_t * 1000) if mean_t > 0 else 0.0
        print(f"\n  Summary:  total={mean_t:.1f} ms  std={std_t:.1f} ms  "
              f"prefill={mean_prefill:.1f} ms  "
              f"decode={mean_decode:.1f} ms  "
              f"TPOT={mean_tpot:.2f} ms  "
              f"decode_throughput={decode_throughput:.1f} tok/s")

        # Print prepare_wrapper profiling if available
        try:
            from integration.hf_adapter import _prepare_profile_times
            times_list = _prepare_profile_times
        except ImportError:
            times_list = None
        if times_list:
            # Skip warmup entries (first decode_len+1 calls)
            skip = cfg.warmup * (cfg.decode_len + 1)
            measured = times_list[skip:]
            if measured:
                keys = list(measured[0].keys())
                print(f"\n  prepare_wrapper profiling ({len(measured)} calls, warmup={skip} skipped):")
                for k in keys:
                    vals = [d[k] for d in measured if k in d]
                    if vals:
                        avg_v = statistics.mean(vals)
                        min_v = min(vals)
                        max_v = max(vals)
                        total_v = sum(vals)
                        print(f"    {k:20s}: avg={avg_v:.3f} ms  min={min_v:.3f}  "
                              f"max={max_v:.3f}  total={total_v:.1f} ms")
            times_list.clear()

        return {
            "mode":               mode,
            "mean_ms":            mean_t,
            "std_ms":             std_t,
            "min_ms":             min(r.total_ms for r in all_runs),
            "max_ms":             max(r.total_ms for r in all_runs),
            "prefill_throughput": prefill_throughput,
            "decode_throughput":  decode_throughput,
            "e2e_throughput":     e2e_throughput,
            "prefill_ms":         mean_prefill,
            "decode_ms":          mean_decode,
            "tpot_ms":            mean_tpot,
            "close_ring_ms":      ring_ms,
            "close_db_ms":        db_ms,
        }

    finally:
        if engine is not None:
            try:
                _timed_close(engine)
            except Exception:
                pass
        model.monitoring_engine = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(
        description="Ring-transport benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Workload")
    g.add_argument("--model",           default="gpt2")
    g.add_argument("--batch-size",      type=int, default=4)
    g.add_argument("--prefill-len",     type=int, default=32)
    g.add_argument("--decode-len",     type=int, default=16)
    g.add_argument("--warmup",          type=int, default=1)
    g.add_argument("--iters",           type=int, default=3)
    g.add_argument("--modes",           default="baseline,ring_null")
    g.add_argument("--cuda-graphs",     action="store_true")
    g.add_argument("--logits-to-keep",  type=int, default=0,
                   help="0=keep all logits, 1=last position only (HF default)")
    g.add_argument("--hook-selection",  default="full",
                   help="Comma-separated hook selection (default: full). "
                        "Presets: full, vllm-full, hf-only. "
                        "Individual: resid_pre, q, k, v, z, final_logits, etc. "
                        "Aliases: hidden-states=resid_pre, logits=final_logits")
    g.add_argument("--hf-offload-hidden-states", action="store_true",
                   help="hf_offload: output + .cpu() hidden_states")
    g.add_argument("--hf-offload-attentions",    action="store_true",
                   help="hf_offload: output + .cpu() attentions")
    g.add_argument("--hf-offload-logits",        action="store_true",
                   help="hf_offload: .cpu() logits")
    g.add_argument("--hf-offload-all",           action="store_true",
                   help="hf_offload: shorthand for all three above")

    g = p.add_argument_group("Ring engine -- GPU buffers")
    g.add_argument("--ring-task-entries", type=int, default=65536,
                   help="Task ring slot count")
    g.add_argument("--ring-payload-mb",   type=int, default=4096,
                   help="GPU payload ring size (MiB)")
    g.add_argument("--ring-pinned-mb",    type=int, default=4096,
                   help="Pinned staging ring size (MiB, 0 = payload size)")

    g = p.add_argument_group("Ring engine -- drain thread")
    g.add_argument("--drain-poll-timeout-us", type=int, default=100,
                   help="Drain thread poll timeout in us (must be > 0)")
    g.add_argument("--drain-flush-task-ratio",    type=float, default=0.0,
                   help="Flush at N%% task ring usage (0 = disabled)")
    g.add_argument("--drain-flush-payload-ratio", type=float, default=0.0,
                   help="Flush at N%% payload ring usage (0 = disabled)")
    g.add_argument("--drain-flush-entry-threshold", type=int, default=0,
                   help="Flush after N entries ready (0 = disabled)")
    g.add_argument("--drain-flush-byte-threshold",  type=int, default=0,
                   help="Flush after N payload bytes ready (0 = disabled)")
    g.add_argument("--drain-flush-timeout-us",     type=int, default=0,
                   help="Flush after complete tensor pending N us (0 = disabled)")

    g = p.add_argument_group("Ring engine -- p2p")
    g.add_argument("--clone-slices", action="store_true",
                   help="Clone per-request slices before submit")

    g = p.add_argument_group("ClickHouse stage")
    g.add_argument("--ch-parallelism",       type=int, default=10,
                   help="Insert thread parallelism")
    g.add_argument("--ch-queue-max-items",   type=int, default=1024,
                   help="Insert queue item limit")
    g.add_argument("--ch-queue-max-size-mb", type=int, default=2048,
                   help="Insert queue byte limit (MiB)")

    g = p.add_argument_group("Output")
    g.add_argument("--csv", default=None, metavar="FILE",
                   help="Append results as CSV rows (creates file with header if missing)")

    g = p.add_argument_group("ClickHouse connection")
    g.add_argument("--db-host",     default="localhost")
    g.add_argument("--db-port",     type=int, default=9000)
    g.add_argument("--db-user",     default="default")
    g.add_argument("--db-password", default="")
    g.add_argument("--db-database", default="default")
    g.add_argument("--db-table",    default="offload_bench")

    ns = p.parse_args()
    return BenchConfig(
        model=ns.model, batch_size=ns.batch_size, prefill_len=ns.prefill_len,
        decode_len=ns.decode_len, warmup=ns.warmup, iters=ns.iters,
        modes=[m.strip() for m in ns.modes.split(",")],
        cuda_graphs=bool(ns.cuda_graphs),
        logits_to_keep=ns.logits_to_keep,
        hook_selection=ns.hook_selection,
        csv_path=ns.csv,
        hf_offload_hidden_states=bool(ns.hf_offload_hidden_states or ns.hf_offload_all),
        hf_offload_attentions=bool(ns.hf_offload_attentions or ns.hf_offload_all),
        hf_offload_logits=bool(ns.hf_offload_logits or ns.hf_offload_all),
        ring_task_entries=ns.ring_task_entries, ring_payload_mb=ns.ring_payload_mb,
        ring_pinned_mb=ns.ring_pinned_mb,
        drain_poll_timeout_us=ns.drain_poll_timeout_us,
        drain_flush_task_ratio=ns.drain_flush_task_ratio,
        drain_flush_payload_ratio=ns.drain_flush_payload_ratio,
        drain_flush_entry_threshold=ns.drain_flush_entry_threshold,
        drain_flush_byte_threshold=ns.drain_flush_byte_threshold,
        drain_flush_timeout_us=ns.drain_flush_timeout_us,
        clone_slices=bool(ns.clone_slices),
        ch_parallelism=ns.ch_parallelism, ch_queue_max_items=ns.ch_queue_max_items,
        ch_queue_max_size_mb=ns.ch_queue_max_size_mb,
        db_host=ns.db_host, db_port=ns.db_port, db_user=ns.db_user,
        db_password=ns.db_password, db_database=ns.db_database, db_table=ns.db_table,
    )


_MODEL_ALIASES = {"qwen3": "Qwen/Qwen3-4B"}


def main() -> None:
    cfg = _parse_args()
    model_id = _MODEL_ALIASES.get(cfg.model.lower(), cfg.model)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    valid = {"baseline", "ring_kernels_only", "ring_null", "ring_db", "hf_offload"}
    for m in cfg.modes:
        if m not in valid:
            raise SystemExit(f"Unknown mode {m!r}. Valid: {sorted(valid)}")

    workload = ("prefill-only" if cfg.decode_len == 1
                else "decode-only" if cfg.prefill_len == 1
                else "mixed")
    print(f"Model        : {model_id}")
    print(f"Workload     : {workload}  "
          f"(batch={cfg.batch_size}  prefill={cfg.prefill_len}  decode={cfg.decode_len})")
    print(f"Warmup/Iters : {cfg.warmup} / {cfg.iters}")
    print(f"Modes        : {cfg.modes}")
    print(f"CUDA graphs  : {'yes (CompileConfig + static cache)' if cfg.cuda_graphs else 'no'}")
    print(f"Hook select  : {cfg.hook_selection}")
    print(f"Ring buffers : payload={cfg.ring_payload_mb} MB  "
          f"pinned={cfg.ring_pinned_mb} MB  "
          f"tasks={cfg.ring_task_entries}")

    from transformers import AutoTokenizer  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    input_ids, attention_mask = _make_inputs(tokenizer, cfg, device)
    print(f"Actual prompt length: {input_ids.shape[1]} tokens\n")

    is_qwen = "qwen3" in model_id.lower()
    _RING_MODES = {"ring_null", "ring_kernels_only", "ring_db"}
    _VANILLA_MODES = {"baseline"}
    _SELF_MANAGED_MODES = {"hf_offload"}  # loads/frees its own model

    def _load_hooked():
        print(f"Loading {'HookedQwen3ForCausalLM' if is_qwen else 'HookedGPT2LMHeadModel'}...",
              flush=True)
        if is_qwen:
            from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
            m = HookedQwen3ForCausalLM.from_pretrained(
                model_id, attn_implementation="eager", torch_dtype=torch.float16)
        else:
            from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel  # type: ignore
            m = HookedGPT2LMHeadModel.from_pretrained(
                model_id, attn_implementation="eager", torch_dtype=torch.float16)
        m.to(device).eval()
        # No external torch.compile -- HF's generate() handles compilation
        # internally via CompileConfig passed in kwargs (see _run_one).
        return m

    def _load_vanilla():
        from transformers import AutoModelForCausalLM  # type: ignore
        print(f"Loading vanilla AutoModelForCausalLM...", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", torch_dtype=torch.float16)
        m.to(device).eval()
        return m

    # Group modes by model type so we load/free each model once per group.
    results = []
    cur_model = None
    cur_type = None  # "hooked" or "vanilla"

    import gc

    def _free_model():
        nonlocal cur_model, cur_type
        if cur_model is not None:
            del cur_model
            cur_model = None
            cur_type = None
            gc.collect()
            torch.cuda.empty_cache()

    for mode_idx, mode in enumerate(cfg.modes):
        if mode in _SELF_MANAGED_MODES:
            # hf_offload loads/frees its own model inside _run_mode
            _free_model()
            r = _run_mode(mode, None, input_ids, attention_mask,
                          cfg, eos_id, pad_id)
        else:
            needed = "hooked" if mode in _RING_MODES else "vanilla"
            if needed != cur_type:
                _free_model()
                cur_model = _load_hooked() if needed == "hooked" else _load_vanilla()
                cur_type = needed
            r = _run_mode(mode, cur_model, input_ids, attention_mask,
                          cfg, eos_id, pad_id)
        if r is not None:
            results.append(r)

    _free_model()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    W = 140
    print(f"\n{'='*W}")
    print(f"{'SUMMARY  --  ' + workload + ('  (CUDA graphs)' if cfg.cuda_graphs else '  (eager)'):^{W}}")
    print(f"{'='*W}")
    print(f"{'mode':<12}  {'total':>8}  {'prefill':>9}  {'decode':>9}  "
          f"{'TPOT':>8}  {'pfill tok/s':>11}  {'dec tok/s':>9}  {'e2e tok/s':>9}  "
          f"{'ring drain':>10}  {'db drain':>8}")
    print("-" * W)
    for r in results:
        print(f"{r['mode']:<12}  "
              f"{r['mean_ms']:>7.1f}ms  "
              f"{r['prefill_ms']:>8.1f}ms  "
              f"{r['decode_ms']:>8.1f}ms  "
              f"{r['tpot_ms']:>7.2f}ms  "
              f"{r.get('prefill_throughput', 0.0):>11.1f}  "
              f"{r['decode_throughput']:>9.1f}  "
              f"{r.get('e2e_throughput', 0.0):>9.1f}  "
              f"{r['close_ring_ms']:>9.1f}ms  "
              f"{r['close_db_ms']:>7.1f}ms")

    print()
    baseline  = next((r for r in results if r["mode"] == "baseline"),  None)
    ring_null = next((r for r in results if r["mode"] == "ring_null"), None)
    ring_db   = next((r for r in results if r["mode"] == "ring_db"),   None)

    if baseline and ring_null:
        ov = ring_null["mean_ms"] - baseline["mean_ms"]
        print(f"  ring_null vs baseline : {ov:+.1f} ms  "
              f"({ov / baseline['mean_ms'] * 100:+.1f}%)  [transport, no DB]")
    if ring_null and ring_db:
        ov = ring_db["mean_ms"] - ring_null["mean_ms"]
        print(f"  ring_db   vs ring_null: {ov:+.1f} ms  "
              f"({ov / ring_null['mean_ms'] * 100:+.1f}%)  [DB write overhead]")
    if baseline and ring_db:
        ov = ring_db["mean_ms"] - baseline["mean_ms"]
        print(f"  ring_db   vs baseline : {ov:+.1f} ms  "
              f"({ov / baseline['mean_ms'] * 100:+.1f}%)  [total monitoring overhead]")

    # -----------------------------------------------------------------------
    # CSV output
    # -----------------------------------------------------------------------
    if cfg.csv_path and results:
        import os
        write_header = not os.path.exists(cfg.csv_path)
        with open(cfg.csv_path, "a") as f:
            if write_header:
                f.write("batch-size,prefill-length,decode-length,mode,cuda-graph,"
                        "prefill(ms),decode(ms),TPOT(ms),decode-throughput(tok/s)\n")
            for r in results:
                f.write(f"{cfg.batch_size},{cfg.prefill_len},{cfg.decode_len},"
                        f"{r['mode']},{cfg.cuda_graphs},"
                        f"{r['prefill_ms']:.1f},{r['decode_ms']:.1f},"
                        f"{r['tpot_ms']:.2f},{r['decode_throughput']:.1f}\n")
        print(f"\nCSV {'created' if write_header else 'appended'}: {cfg.csv_path}")


if __name__ == "__main__":
    main()
