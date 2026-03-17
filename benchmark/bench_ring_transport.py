"""Ring-transport benchmark.

Compares four modes:
  baseline         — plain HF generate, no monitoring
  ring_kernels_only— ring transport active, producer kernels fire but do zero work (null_mode on)
  ring_null        — ring transport active (GPU→CPU transfer), null sink (no DB write)
  ring_db          — ring transport + ClickHouse ingestion

Per-step (prefill vs decode) timing is intentionally NOT reported here.
Inserting a GPU sync barrier between every step breaks CUDA-graph pipelining
and makes per-step numbers unreliable.  Instead, run dedicated prefill /
decode benchmarks using the appropriate --prompt-len / --max-new-tokens flags:

  Prefill:  --prompt-len N --max-new-tokens 1
  Decode:   --prompt-len 1 --max-new-tokens N

Each run reports total wall time (ms) and throughput (tok/s).
For prefill, tok/s counts prompt tokens processed (batch * prompt_len).
For decode,  tok/s counts new tokens generated  (batch * max_new_tokens).

Usage:
  python -m benchmark.bench_ring_transport --model qwen3 --modes baseline,ring_null
"""

from __future__ import annotations

import argparse
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
    prompt_len: int = 32
    max_new_tokens: int = 16
    warmup: int = 1
    iters: int = 3
    modes: List[str] = field(default_factory=lambda: ["baseline", "ring_null"])
    cuda_graphs: bool = False
    logits_to_keep: int = 0  # 0 = keep all, 1 = last position only (HF default)
    hf_offload_hidden_states: bool = False
    hf_offload_attentions: bool = False
    hf_offload_logits: bool = False

    ring_task_entries: int = 65536
    ring_payload_mb: int = 4096
    ring_pinned_mb: int = 4096
    drain_poll_timeout_us: int = 100
    drain_flush_task_ratio: float = 0.0
    drain_flush_payload_ratio: float = 0.0
    drain_flush_entry_threshold: int = 0
    drain_flush_byte_threshold: int = 0
    drain_flush_timeout_us: int = 0
    bypass_budget_mb: int = 256
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
    rc.bypass_budget_bytes         = cfg.bypass_budget_mb * 1024 * 1024
    rc.clone_slices                = cfg.clone_slices
    rc.insert_queue_max_bytes      = cfg.ch_queue_max_size_mb * 1024 * 1024
    rc.insert_queue_max_items      = cfg.ch_queue_max_items
    return rc


def _make_monitoring_cfg(cfg: BenchConfig):
    from monitoring import AdvanceConfig, MonitoringConfig, NativePartialSealConfig  # type: ignore
    from monitoring.config import CaptureSchedule, HookSelection  # type: ignore
    return MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
        native_partial_seal=NativePartialSealConfig(
            enabled=True, chunk_bytes=256 * 1024,  # native engine partial seal
            cap_enabled=True, cap_ratio=0.8, driver_guard_mb=1024,
        ),
        advance=AdvanceConfig(),
    )


def _make_null_engine(cfg: BenchConfig, model_id: str):
    from monitoring import MonitoringEngine  # type: ignore
    engine = MonitoringEngine(
        async_enabled=True, config=_make_monitoring_cfg(cfg),
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
        async_enabled=True, config=_make_monitoring_cfg(cfg),
        model_id=model_id, db_config=HostEngineConfig(stages=[stage]),
    )
    engine.enable_ring_transport(_make_ring_cfg(cfg))
    return engine


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _make_inputs(tokenizer, cfg: BenchConfig, device: torch.device):
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 20)
    ids = ids[:cfg.prompt_len]
    rows = [torch.tensor(ids, dtype=torch.long) for _ in range(cfg.batch_size)]
    input_ids = torch.stack(rows).to(device)
    return input_ids, torch.ones_like(input_ids)


# ---------------------------------------------------------------------------
# Per-iteration runner — returns total wall time only
# ---------------------------------------------------------------------------

def _run_one(model, input_ids, attention_mask,
             cfg: BenchConfig, eos_id: int, pad_id: int,
             use_monitoring: bool) -> float:
    """Run one generate() and return total wall time in ms."""
    from monitoring.generate import generate_with_monitoring  # type: ignore

    extra = {}
    extra["logits_to_keep"] = cfg.logits_to_keep
    if cfg.cuda_graphs:
        extra["cache_implementation"] = "static"

    # cudaDeviceSynchronize — wait for any leftover GPU work before the timer
    # starts (including drain-thread D2H from a previous iteration).
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        if use_monitoring:
            generate_with_monitoring(
                model, input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens, do_sample=False,
                pad_token_id=pad_id, eos_token_id=eos_id, **extra,
            )
        else:
            model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens, do_sample=False,
                pad_token_id=pad_id, eos_token_id=eos_id, **extra,
            )

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0



# ---------------------------------------------------------------------------
# Per-mode benchmark
# ---------------------------------------------------------------------------

def _run_mode(mode: str, model, input_ids, attention_mask,
              cfg: BenchConfig, eos_id: int, pad_id: int) -> dict:

    use_monitoring = mode not in ("baseline", "hf_offload")
    # For prefill benchmark (max_new_tokens=1): count prompt tokens processed.
    # For decode benchmark (prompt_len=1):      count new tokens generated.
    # For mixed (prompt_len>1, max_new_tokens>1): count new tokens generated.
    if cfg.max_new_tokens == 1:
        tokens_out = cfg.batch_size * cfg.prompt_len   # prefill-mode: prompt throughput
    else:
        tokens_out = cfg.batch_size * cfg.max_new_tokens

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
        max_cache_len = Pmax + cfg.max_new_tokens + 4

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

        def _run_one_offload():
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
                cache_pos = torch.tensor([Pmax], device=input_ids.device, dtype=torch.long)

                # Decode
                for step in range(cfg.max_new_tokens - 1):
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
                    cache_pos = cache_pos + 1

            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        print(f"  Warming up ({cfg.warmup} iter)...", flush=True)
        for _ in range(cfg.warmup):
            _run_one_offload()

        all_total_ms: List[float] = []
        for i in range(cfg.iters):
            total_ms = _run_one_offload()
            all_total_ms.append(total_ms)
            print(f"  iter {i+1:2d}:  total={total_ms:7.1f} ms  "
                  f"({tokens_out / total_ms * 1000:.1f} tok/s)", flush=True)

        mean_t = statistics.mean(all_total_ms)
        std_t  = statistics.stdev(all_total_ms) if len(all_total_ms) > 1 else 0.0
        print(f"\n  Summary:  mean={mean_t:.1f} ms  std={std_t:.1f} ms  "
              f"min={min(all_total_ms):.1f}  max={max(all_total_ms):.1f}  "
              f"throughput={tokens_out / mean_t * 1000:.1f} tok/s")

        del hf_model, compiled_decode
        torch.cuda.empty_cache()

        return {
            "mode": mode, "mean_ms": mean_t, "std_ms": std_t,
            "min_ms": min(all_total_ms), "max_ms": max(all_total_ms),
            "throughput": tokens_out / mean_t * 1000,
            "close_ring_ms": 0.0, "close_db_ms": 0.0,
        }

    engine = None
    if mode in ("ring_null", "ring_kernels_only"):
        model_id = f"bench::{mode}::{uuid.uuid4().hex[:8]}"
        engine = _make_null_engine(cfg, model_id)
        model.monitoring_engine = engine
        engine.prepare_for_model(model)
    elif mode == "ring_db":
        model_id = f"bench::{mode}::{uuid.uuid4().hex[:8]}"
        engine = _make_db_engine(cfg, model_id)
        model.monitoring_engine = engine
        engine.prepare_for_model(model)
    else:
        model.monitoring_engine = None

    try:
        # Warmup — null mode so producer kernels fire (same CUDA graph topology)
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

        # Measured iterations
        all_total_ms: List[float] = []
        close_t: Dict[str, float] = {"ring_ms": 0.0, "db_ms": 0.0, "cleanup_ms": 0.0}

        for i in range(cfg.iters):
            total_ms = _run_one(model, input_ids, attention_mask,
                                cfg, eos_id, pad_id, use_monitoring)
            all_total_ms.append(total_ms)
            print(f"  iter {i+1:2d}:  total={total_ms:7.1f} ms  "
                  f"({tokens_out / total_ms * 1000:.1f} tok/s)",
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

        mean_t = statistics.mean(all_total_ms)
        std_t  = statistics.stdev(all_total_ms) if len(all_total_ms) > 1 else 0.0
        print(f"\n  Summary:  mean={mean_t:.1f} ms  std={std_t:.1f} ms  "
              f"min={min(all_total_ms):.1f}  max={max(all_total_ms):.1f}  "
              f"throughput={tokens_out / mean_t * 1000:.1f} tok/s")

        # Print prepare_wrapper profiling if available
        try:
            from monitoring.generate import _prepare_profile_times
            times_list = _prepare_profile_times
        except ImportError:
            times_list = None
        if times_list:
            # Skip warmup entries (first max_new_tokens+1 calls)
            skip = cfg.warmup * (cfg.max_new_tokens + 1)
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
            "mode":          mode,
            "mean_ms":       mean_t,
            "std_ms":        std_t,
            "min_ms":        min(all_total_ms),
            "max_ms":        max(all_total_ms),
            "throughput":    tokens_out / mean_t * 1000,
            "close_ring_ms": ring_ms,
            "close_db_ms":   db_ms,
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
    g.add_argument("--prompt-len",      type=int, default=32)
    g.add_argument("--max-new-tokens",  type=int, default=16)
    g.add_argument("--warmup",          type=int, default=1)
    g.add_argument("--iters",           type=int, default=3)
    g.add_argument("--modes",           default="baseline,ring_null")
    g.add_argument("--cuda-graphs",     action="store_true")
    g.add_argument("--logits-to-keep",  type=int, default=0,
                   help="0=keep all logits, 1=last position only (HF default)")
    g.add_argument("--hf-offload-hidden-states", action="store_true",
                   help="hf_offload: output + .cpu() hidden_states")
    g.add_argument("--hf-offload-attentions",    action="store_true",
                   help="hf_offload: output + .cpu() attentions")
    g.add_argument("--hf-offload-logits",        action="store_true",
                   help="hf_offload: .cpu() logits")
    g.add_argument("--hf-offload-all",           action="store_true",
                   help="hf_offload: shorthand for all three above")

    g = p.add_argument_group("Ring engine — GPU buffers")
    g.add_argument("--ring-task-entries", type=int, default=65536,
                   help="Task ring slot count")
    g.add_argument("--ring-payload-mb",   type=int, default=4096,
                   help="GPU payload ring size (MiB)")
    g.add_argument("--ring-pinned-mb",    type=int, default=4096,
                   help="Pinned staging ring size (MiB, 0 = payload size)")

    g = p.add_argument_group("Ring engine — drain thread")
    g.add_argument("--drain-poll-timeout-us", type=int, default=100,
                   help="Drain thread poll timeout in µs (must be > 0)")
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

    g = p.add_argument_group("Ring engine — bypass / p2p")
    g.add_argument("--bypass-budget-mb", type=int, default=256,
                   help="Large tensor bypass budget (MiB)")
    g.add_argument("--clone-slices", action="store_true",
                   help="Clone per-request slices before submit")

    g = p.add_argument_group("ClickHouse stage")
    g.add_argument("--ch-parallelism",       type=int, default=10,
                   help="Insert thread parallelism")
    g.add_argument("--ch-queue-max-items",   type=int, default=1024,
                   help="Insert queue item limit")
    g.add_argument("--ch-queue-max-size-mb", type=int, default=2048,
                   help="Insert queue byte limit (MiB)")

    g = p.add_argument_group("ClickHouse connection")
    g.add_argument("--db-host",     default="localhost")
    g.add_argument("--db-port",     type=int, default=9000)
    g.add_argument("--db-user",     default="default")
    g.add_argument("--db-password", default="")
    g.add_argument("--db-database", default="default")
    g.add_argument("--db-table",    default="offload_bench")

    ns = p.parse_args()
    return BenchConfig(
        model=ns.model, batch_size=ns.batch_size, prompt_len=ns.prompt_len,
        max_new_tokens=ns.max_new_tokens, warmup=ns.warmup, iters=ns.iters,
        modes=[m.strip() for m in ns.modes.split(",")],
        cuda_graphs=bool(ns.cuda_graphs),
        logits_to_keep=ns.logits_to_keep,
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
        bypass_budget_mb=ns.bypass_budget_mb,
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

    workload = ("prefill-only" if cfg.max_new_tokens == 1
                else "decode-only" if cfg.prompt_len == 1
                else "mixed")
    print(f"Model        : {model_id}")
    print(f"Workload     : {workload}  "
          f"(batch={cfg.batch_size}  prompt={cfg.prompt_len}  new={cfg.max_new_tokens})")
    print(f"Warmup/Iters : {cfg.warmup} / {cfg.iters}")
    print(f"Modes        : {cfg.modes}")
    print(f"CUDA graphs  : {'yes (torch.compile + static cache)' if cfg.cuda_graphs else 'no'}")
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
        if cfg.cuda_graphs:
            print("  Compiling with torch.compile(mode='reduce-overhead')...", flush=True)
            m = torch.compile(m, mode="reduce-overhead")
            print("  done.", flush=True)
        return m

    def _load_vanilla():
        from transformers import AutoModelForCausalLM  # type: ignore
        print(f"Loading vanilla AutoModelForCausalLM...", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", torch_dtype=torch.float16)
        m.to(device).eval()
        if cfg.cuda_graphs:
            print("  Compiling with torch.compile(mode='reduce-overhead')...", flush=True)
            m = torch.compile(m, mode="reduce-overhead")
            print("  done.", flush=True)
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

    for mode in cfg.modes:
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
    W = 76
    print(f"\n{'='*W}")
    print(f"{'SUMMARY  —  ' + workload + ('  (CUDA graphs)' if cfg.cuda_graphs else '  (eager)'):^{W}}")
    print(f"{'='*W}")
    print(f"{'mode':<12}  {'mean':>8}  {'±std':>6}  {'min':>7}  {'max':>7}  "
          f"{'tok/s':>7}  {'ring drain':>10}  {'db drain':>8}")
    print("-" * W)
    for r in results:
        print(f"{r['mode']:<12}  "
              f"{r['mean_ms']:>7.1f}ms  "
              f"{r['std_ms']:>5.1f}ms  "
              f"{r['min_ms']:>6.1f}ms  "
              f"{r['max_ms']:>6.1f}ms  "
              f"{r['throughput']:>7.1f}  "
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


if __name__ == "__main__":
    main()
