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

    ring_task_entries: int = 65536
    ring_payload_mb: int = 4096
    ring_chunk_kb: int = 4096
    ring_pinned_mb: int = 4096
    drain_poll_timeout_us: int = 0
    drain_notify_on_forward: bool = True
    drain_flush_task_ratio: float = 0.0
    drain_flush_payload_ratio: float = 0.0
    drain_flush_entry_threshold: int = 0
    drain_flush_byte_threshold: int = 0
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
    rc.chunk_bytes        = cfg.ring_chunk_kb   * 1024
    rc.pinned_staging_bytes  = cfg.ring_pinned_mb  * 1024 * 1024
    rc.drain_poll_timeout_us       = cfg.drain_poll_timeout_us
    rc.drain_notify_on_forward     = cfg.drain_notify_on_forward
    rc.drain_flush_task_ratio      = cfg.drain_flush_task_ratio
    rc.drain_flush_payload_ratio   = cfg.drain_flush_payload_ratio
    rc.drain_flush_entry_threshold = cfg.drain_flush_entry_threshold
    rc.drain_flush_byte_threshold  = cfg.drain_flush_byte_threshold
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
            enabled=True, chunk_bytes=cfg.ring_chunk_kb * 1024,
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
    ch.drop_existing_database = False
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
    if hasattr(model.config, "n_layer"):
        extra["logits_to_keep"] = 0
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

    use_monitoring = mode != "baseline"
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

    g = p.add_argument_group("Ring engine — GPU buffers")
    g.add_argument("--ring-task-entries", type=int, default=65536,
                   help="Task ring slot count")
    g.add_argument("--ring-payload-mb",   type=int, default=4096,
                   help="GPU payload ring size (MiB)")
    g.add_argument("--ring-chunk-kb",     type=int, default=4096,
                   help="Max chunk size (KiB)")
    g.add_argument("--ring-pinned-mb",    type=int, default=4096,
                   help="Pinned staging ring size (MiB, 0 = payload size)")

    g = p.add_argument_group("Ring engine — drain thread")
    g.add_argument("--drain-poll-timeout-us", type=int, default=0,
                   help="Drain thread poll timeout in µs (0 = no timeout)")
    g.add_argument("--no-drain-notify", action="store_true",
                   help="Disable notify_drain() before each forward pass")
    g.add_argument("--drain-flush-task-ratio",    type=float, default=0.0,
                   help="Flush at N%% task ring usage (0 = disabled)")
    g.add_argument("--drain-flush-payload-ratio", type=float, default=0.0,
                   help="Flush at N%% payload ring usage (0 = disabled)")
    g.add_argument("--drain-flush-entry-threshold", type=int, default=0,
                   help="Flush after N entries ready (0 = disabled)")
    g.add_argument("--drain-flush-byte-threshold",  type=int, default=0,
                   help="Flush after N payload bytes ready (0 = disabled)")

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
        ring_task_entries=ns.ring_task_entries, ring_payload_mb=ns.ring_payload_mb,
        ring_chunk_kb=ns.ring_chunk_kb, ring_pinned_mb=ns.ring_pinned_mb,
        drain_poll_timeout_us=ns.drain_poll_timeout_us,
        drain_notify_on_forward=not ns.no_drain_notify,
        drain_flush_task_ratio=ns.drain_flush_task_ratio,
        drain_flush_payload_ratio=ns.drain_flush_payload_ratio,
        drain_flush_entry_threshold=ns.drain_flush_entry_threshold,
        drain_flush_byte_threshold=ns.drain_flush_byte_threshold,
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
    valid = {"baseline", "ring_kernels_only", "ring_null", "ring_db"}
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
          f"tasks={cfg.ring_task_entries}  chunk={cfg.ring_chunk_kb} KB")

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
    print(f"Loading {'HookedQwen3ForCausalLM' if is_qwen else 'HookedGPT2LMHeadModel'}...",
          flush=True)
    try:
        if is_qwen:
            from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
            model = HookedQwen3ForCausalLM.from_pretrained(
                model_id, attn_implementation="eager", torch_dtype=torch.float16)
        else:
            from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel  # type: ignore
            model = HookedGPT2LMHeadModel.from_pretrained(
                model_id, attn_implementation="eager", torch_dtype=torch.float16)
    except Exception as exc:
        raise SystemExit(f"Failed to load hooked model: {exc}") from exc

    model.to(device).eval()

    if cfg.cuda_graphs:
        print("Compiling with torch.compile(mode='reduce-overhead')...", flush=True)
        model = torch.compile(model, mode="reduce-overhead")
        print("  done.", flush=True)

    results = []
    for mode in cfg.modes:
        r = _run_mode(mode, model, input_ids, attention_mask,
                      cfg, eos_id, pad_id)
        if r is not None:
            results.append(r)

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
