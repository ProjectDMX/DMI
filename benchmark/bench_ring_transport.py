"""Ring-transport benchmark (detailed).

Compares three modes:
  baseline  — plain HF generate, no monitoring
  ring_null — ring transport active (GPU→CPU transfer), null sink (no DB write)
  ring_db   — ring transport + ClickHouse ingestion

Reports:
  - Per-step forward-pass latency (prefill step 0, decode steps 1..N)
  - Close breakdown: ring drain vs DB drain
  - Data volume estimate (hooks * bytes/hook)
  - Side-by-side summary across modes

Usage:
  python -m benchmark.bench_ring_transport --model qwen3 --modes baseline,ring_null,ring_db
  python -m benchmark.bench_ring_transport --model gpt2  --modes baseline,ring_null
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
# Step-level timing wrapper
# ---------------------------------------------------------------------------

class _StepTimer:
    """Patches model.forward to record per-call GPU-synced wall times."""

    def __init__(self, model) -> None:
        self._model = model
        self._orig  = model.forward
        self.times_ms: List[float] = []
        timer = self

        def _timed(*args, **kwargs):
            torch.cuda.synchronize()
            t0  = time.perf_counter()
            out = timer._orig(*args, **kwargs)
            torch.cuda.synchronize()
            timer.times_ms.append((time.perf_counter() - t0) * 1000.0)
            return out

        model.forward = _timed

    def restore(self) -> None:
        self._model.forward = self._orig

    def reset(self) -> None:
        self.times_ms.clear()


# ---------------------------------------------------------------------------
# Timed close: ring drain vs DB drain
# ---------------------------------------------------------------------------

def _timed_close(engine) -> Dict[str, float]:
    """Stop ring engine, host engine, and remaining engine resources
    separately, returning wall-time ms for each phase."""
    result = {"ring_ms": 0.0, "db_ms": 0.0, "cleanup_ms": 0.0}

    # Phase 1: ring engine drain (GPU→CPU pipeline)
    ring_engine = getattr(engine, "_ring_engine", None)
    if ring_engine is not None:
        t0 = time.perf_counter()
        try:
            ring_engine.stop()
        except Exception:
            pass
        result["ring_ms"] = (time.perf_counter() - t0) * 1000.0

    # Phase 2: DB pipeline drain (ClickHouse insert queue)
    host_engine = getattr(engine, "_host_engine", None)
    if host_engine is not None:
        t0 = time.perf_counter()
        try:
            host_engine.stop()
        except Exception:
            pass
        result["db_ms"] = (time.perf_counter() - t0) * 1000.0

    # Phase 3: remaining cleanup (native backend, deactivate ring transport)
    t0 = time.perf_counter()
    try:
        engine.close()   # ring/host already stopped; this is fast cleanup
    except Exception:
        pass
    result["cleanup_ms"] = (time.perf_counter() - t0) * 1000.0

    return result


# ---------------------------------------------------------------------------
# Null host-engine sink
# ---------------------------------------------------------------------------

class _NullHostEngine:
    """Drop-all sink: satisfies the host-engine API but discards every tensor."""
    def start(self)                         -> None:  pass
    def stop(self, *a, **kw)                -> None:  pass
    def join(self, *a, **kw)                -> bool:  return True
    def close_input(self)                   -> None:  pass
    def request_abort(self)                 -> None:  pass
    def failures(self)                      -> list:  return []
    def raise_if_failed(self)               -> None:  pass
    def submit_direct(self, *a, **kw)       -> None:  pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    # Workload
    model: str = "gpt2"
    batch_size: int = 4
    prompt_len: int = 32
    max_new_tokens: int = 16
    warmup: int = 1
    iters: int = 3
    modes: List[str] = field(default_factory=lambda: ["baseline", "ring_null", "ring_db"])

    # Ring engine
    ring_task_entries: int = 65536
    ring_payload_mb: int = 4096
    ring_chunk_kb: int = 4096
    ring_pinned_mb: int = 4096

    # ClickHouse stage
    ch_parallelism: int = 10
    ch_queue_max_items: int = 1024
    ch_queue_max_size_mb: int = 2048

    # DB connection
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
    rc.pinned_pool_bytes  = cfg.ring_pinned_mb  * 1024 * 1024
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
    q.max_batch_items       = cfg.ch_queue_max_items
    q.high_watermark_items  = cfg.ch_queue_max_items
    q.max_batch_size        = cfg.ch_queue_max_size_mb * 1024 * 1024
    q.high_watermark_size   = cfg.ch_queue_max_size_mb * 1024 * 1024

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
# Per-iteration runner
# ---------------------------------------------------------------------------

def _run_one(model, timer: _StepTimer, input_ids, attention_mask,
             cfg: BenchConfig, eos_id: int, pad_id: int,
             use_monitoring: bool) -> Tuple[float, List[float]]:
    """Run one generate() and return (total_wall_ms, step_times_ms)."""
    from monitoring.generate import generate_with_monitoring  # type: ignore

    # logits_to_keep is only supported by GPT-2 variant in this repo
    extra = {}
    if hasattr(model.config, "n_layer"):
        extra["logits_to_keep"] = 0

    timer.reset()
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
                pad_token_id=pad_id, eos_token_id=eos_id,
            )

    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, list(timer.times_ms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ms(v: float) -> str:
    return f"{v:7.1f} ms"

def _step_stats(times: List[float]) -> str:
    if not times:
        return "n/a"
    if len(times) == 1:
        return f"{times[0]:.1f} ms"
    return f"mean={statistics.mean(times):.1f}  std={statistics.stdev(times):.1f}  min={min(times):.1f}  max={max(times):.1f} ms"


# ---------------------------------------------------------------------------
# Per-mode benchmark
# ---------------------------------------------------------------------------

def _run_mode(mode: str, model, input_ids, attention_mask,
              cfg: BenchConfig, eos_id: int, pad_id: int,
              timer: _StepTimer) -> dict:

    use_monitoring = mode != "baseline"
    tokens_out = cfg.batch_size * cfg.max_new_tokens

    print(f"\n{'='*64}")
    print(f"  Mode: {mode}")
    print(f"{'='*64}")

    engine = None
    if mode == "ring_null":
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
        # Warmup
        print(f"  Warming up ({cfg.warmup} iter)...", flush=True)
        for _ in range(cfg.warmup):
            _run_one(model, timer, input_ids, attention_mask,
                     cfg, eos_id, pad_id, use_monitoring)

        # Measured iterations
        all_total_ms: List[float]         = []
        all_prefill_ms: List[float]        = []
        all_decode_ms: List[List[float]]   = []  # per-iter list of decode step times
        all_close: List[Dict[str, float]]  = []

        for i in range(cfg.iters):
            total_ms, steps = _run_one(model, timer, input_ids, attention_mask,
                                       cfg, eos_id, pad_id, use_monitoring)

            prefill_ms = steps[0]  if steps else 0.0
            decode_ms  = steps[1:] if steps else []

            # Per-iter close (re-create engine for ring modes so close is fresh each iter)
            close_t: Dict[str, float] = {"ring_ms": 0.0, "db_ms": 0.0, "cleanup_ms": 0.0}
            if engine is not None and i == cfg.iters - 1:
                # Only close on the last iter; keep engine alive between iters
                close_t = _timed_close(engine)
                engine = None

            all_total_ms.append(total_ms)
            all_prefill_ms.append(prefill_ms)
            all_decode_ms.append(decode_ms)
            all_close.append(close_t)

            decode_mean = statistics.mean(decode_ms) if decode_ms else 0.0
            print(f"  iter {i+1:2d}:  total={total_ms:7.1f} ms  "
                  f"prefill={prefill_ms:6.1f} ms  "
                  f"decode/step={decode_mean:6.1f} ms  "
                  f"({tokens_out/total_ms*1000:.1f} tok/s)",
                  flush=True)

        # Close timing (from last iter)
        ct = all_close[-1]
        ring_ms    = ct["ring_ms"]
        db_ms      = ct["db_ms"]
        cleanup_ms = ct["cleanup_ms"]
        print(f"\n  Close breakdown:")
        print(f"    ring drain  : {_fmt_ms(ring_ms)}")
        print(f"    db drain    : {_fmt_ms(db_ms)}")
        print(f"    cleanup     : {_fmt_ms(cleanup_ms)}")
        print(f"    total       : {_fmt_ms(ring_ms + db_ms + cleanup_ms)}")

        # Step-level summary
        flat_decode = [t for ts in all_decode_ms for t in ts]
        print(f"\n  Step latency summary ({cfg.iters} iters):")
        print(f"    prefill (step 0) : {_step_stats(all_prefill_ms)}")
        print(f"    decode  (steps 1+): {_step_stats(flat_decode)}")

        # Generate summary
        mean_t = statistics.mean(all_total_ms)
        std_t  = statistics.stdev(all_total_ms) if len(all_total_ms) > 1 else 0.0
        print(f"\n  Generate summary:")
        print(f"    mean={mean_t:.1f} ms  std={std_t:.1f} ms  "
              f"min={min(all_total_ms):.1f} ms  max={max(all_total_ms):.1f} ms")
        print(f"    throughput: {tokens_out / mean_t * 1000:.1f} tok/s")

        return {
            "mode":           mode,
            "gen_mean_ms":    mean_t,
            "gen_std_ms":     std_t,
            "gen_min_ms":     min(all_total_ms),
            "gen_max_ms":     max(all_total_ms),
            "throughput":     tokens_out / mean_t * 1000,
            "prefill_mean_ms": statistics.mean(all_prefill_ms),
            "prefill_std_ms":  statistics.stdev(all_prefill_ms) if len(all_prefill_ms) > 1 else 0.0,
            "decode_mean_ms":  statistics.mean(flat_decode) if flat_decode else 0.0,
            "decode_std_ms":   statistics.stdev(flat_decode) if len(flat_decode) > 1 else 0.0,
            "close_ring_ms":  ring_ms,
            "close_db_ms":    db_ms,
            "close_total_ms": ring_ms + db_ms + cleanup_ms,
        }

    finally:
        if engine is not None:
            try:
                _timed_close(engine)
            except Exception:
                pass
        model.monitoring_engine = None


# ---------------------------------------------------------------------------
# Data volume estimate
# ---------------------------------------------------------------------------

def _estimate_data_volume(model, cfg: BenchConfig) -> str:
    """Rough estimate of bytes captured per generate call."""
    try:
        hf_cfg = model.config
        # Try common attribute names for layers / hidden dim
        n_layers  = next((getattr(hf_cfg, a) for a in
                          ("num_hidden_layers", "n_layer", "num_layers") if hasattr(hf_cfg, a)), None)
        d_model   = next((getattr(hf_cfg, a) for a in
                          ("hidden_size", "n_embd", "d_model") if hasattr(hf_cfg, a)), None)
        n_heads   = next((getattr(hf_cfg, a) for a in
                          ("num_attention_heads", "n_head") if hasattr(hf_cfg, a)), None)
        if n_layers is None or d_model is None:
            return "unknown"

        B  = cfg.batch_size
        T  = cfg.prompt_len + cfg.max_new_tokens   # rough total tokens
        N  = cfg.max_new_tokens + 1                # forward passes (1 prefill + N decode)
        bw = 2                                     # float16

        # Main residual stream hooks: ~6 per layer × [B, T_step, d_model]
        # Attention hooks: pattern [B, n_heads, T, T] + scores [B, n_heads, T, T]
        # Rough: 20 hooks/layer with average tensor size ~ B * avg_seq * d_model * bw
        avg_seq   = (cfg.prompt_len + cfg.prompt_len // 2)  # prefill heavy
        bytes_per_layer_per_step = 20 * B * avg_seq * d_model * bw
        total = n_layers * N * bytes_per_layer_per_step
        if total < 1024**3:
            return f"~{total / 1024**2:.0f} MB"
        return f"~{total / 1024**3:.2f} GB"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(
        description="Ring-transport detailed benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Workload")
    g.add_argument("--model",           default="gpt2")
    g.add_argument("--batch-size",      type=int, default=4)
    g.add_argument("--prompt-len",      type=int, default=32)
    g.add_argument("--max-new-tokens",  type=int, default=16)
    g.add_argument("--warmup",          type=int, default=1)
    g.add_argument("--iters",           type=int, default=3)
    g.add_argument("--modes",           default="baseline,ring_null,ring_db")

    g = p.add_argument_group("Ring engine")
    g.add_argument("--ring-task-entries", type=int, default=1024)
    g.add_argument("--ring-payload-mb",   type=int, default=4096)
    g.add_argument("--ring-chunk-kb",     type=int, default=4096)
    g.add_argument("--ring-pinned-mb",    type=int, default=4096)

    g = p.add_argument_group("ClickHouse stage")
    g.add_argument("--ch-parallelism",       type=int, default=10)
    g.add_argument("--ch-queue-max-items",   type=int, default=1024)
    g.add_argument("--ch-queue-max-size-mb", type=int, default=2048)

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
        ring_task_entries=ns.ring_task_entries, ring_payload_mb=ns.ring_payload_mb,
        ring_chunk_kb=ns.ring_chunk_kb, ring_pinned_mb=ns.ring_pinned_mb,
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
    valid = {"baseline", "ring_null", "ring_db"}
    for m in cfg.modes:
        if m not in valid:
            raise SystemExit(f"Unknown mode {m!r}. Valid: {sorted(valid)}")

    print(f"Model        : {model_id}")
    print(f"Batch / Toks : {cfg.batch_size} x {cfg.prompt_len} prompt + {cfg.max_new_tokens} new")
    print(f"Warmup/Iters : {cfg.warmup} / {cfg.iters}")
    print(f"Modes        : {cfg.modes}")
    print(f"Ring buffers : GPU payload={cfg.ring_payload_mb} MB  "
          f"CPU pinned={cfg.ring_pinned_mb} MB  chunk={cfg.ring_chunk_kb} KB")

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

    data_vol = _estimate_data_volume(model, cfg)
    print(f"Estimated data volume per generate call: {data_vol}")

    timer = _StepTimer(model)

    results = []
    for mode in cfg.modes:
        r = _run_mode(mode, model, input_ids, attention_mask,
                      cfg, eos_id, pad_id, timer)
        if r is not None:
            results.append(r)

    timer.restore()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    W = 80
    print(f"\n{'='*W}")
    print(f"{'SUMMARY':^{W}}")
    print(f"{'='*W}")

    tokens_out = cfg.batch_size * cfg.max_new_tokens
    col = 14
    h1 = f"{'mode':<12}  {'gen mean':>9}  {'± std':>7}  {'prefill':>8}  {'decode/step':>11}  {'tok/s':>7}  {'close ring':>10}  {'close db':>9}"
    print(h1)
    print("-" * W)
    for r in results:
        print(
            f"{r['mode']:<12}  "
            f"{r['gen_mean_ms']:>8.1f}ms  "
            f"{r['gen_std_ms']:>6.1f}ms  "
            f"{r['prefill_mean_ms']:>7.1f}ms  "
            f"{r['decode_mean_ms']:>10.1f}ms  "
            f"{r['throughput']:>7.1f}  "
            f"{r['close_ring_ms']:>9.1f}ms  "
            f"{r['close_db_ms']:>8.1f}ms"
        )

    # Overhead breakdown
    baseline  = next((r for r in results if r["mode"] == "baseline"),  None)
    ring_null = next((r for r in results if r["mode"] == "ring_null"), None)
    ring_db   = next((r for r in results if r["mode"] == "ring_db"),   None)

    print()
    if baseline and ring_null:
        ov = ring_null["gen_mean_ms"] - baseline["gen_mean_ms"]
        print(f"  ring_null overhead vs baseline  : {ov:+.1f} ms/iter  "
              f"({ov / baseline['gen_mean_ms'] * 100:+.1f}%)  "
              f"[transport cost, no DB]")
    if ring_null and ring_db:
        ov = ring_db["gen_mean_ms"] - ring_null["gen_mean_ms"]
        db_close = ring_db["close_db_ms"] - ring_null["close_db_ms"]
        print(f"  ring_db  overhead vs ring_null  : {ov:+.1f} ms/iter  "
              f"({ov / ring_null['gen_mean_ms'] * 100:+.1f}%)  "
              f"[DB write cost on critical path]")
        print(f"  ring_db  extra close vs ring_null: {db_close:+.1f} ms  "
              f"[DB queue drain after generate]")
    if baseline and ring_db:
        ov = ring_db["gen_mean_ms"] - baseline["gen_mean_ms"]
        print(f"  ring_db  overhead vs baseline   : {ov:+.1f} ms/iter  "
              f"({ov / baseline['gen_mean_ms'] * 100:+.1f}%)  "
              f"[total monitoring overhead]")


if __name__ == "__main__":
    main()
