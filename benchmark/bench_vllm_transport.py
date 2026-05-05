"""vLLM ring-transport benchmark.

Compares modes:
  baseline         -- plain vLLM generate, no monitoring
  ring_null        -- ring transport active (GPU->ring->CPU), null sink (no DB write)
  ring_db          -- ring transport + ClickHouse ingestion

Reports total wall time (ms) and throughput (tok/s) for the generate call.
TTFT (time to first token) is not measured — vLLM's offline LLM API
does not expose per-token timing.

Usage:
  python -m benchmark.bench_vllm_transport --model gpt2 --modes baseline,ring_null
  python -m benchmark.bench_vllm_transport --model qwen3 --num-prompts 8 --max-tokens 64
  python -m benchmark.bench_vllm_transport --model gpt2 --modes baseline,ring_null,ring_db

Requires:
  - VLLM_DISABLE_COMPILE_CACHE=1 (set automatically)
  - LD_PRELOAD for libstdc++ (caller's responsibility)
  - ClickHouse running for ring_db mode
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import List

# Force disable compile cache
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import torch


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B",
}


@dataclass
class BenchConfig:
    model: str = "gpt2"
    num_prompts: int = 8
    prompt_len: int = 0  # 0 = default multi-token, 1 = single token
    max_tokens: int = 20
    warmup: int = 2
    iters: int = 5
    modes: List[str] = field(default_factory=lambda: ["baseline", "ring_null"])
    enforce_eager: bool = False
    # Ring engine
    ring_payload_mb: int = 4096
    ring_pinned_mb: int = 4096
    hook_selection: str = "vllm-full"
    # ClickHouse (for ring_db)
    db_host: str = "localhost"
    db_port: int = 9000
    # vLLM
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.5


@dataclass
class RunResult:
    total_ms: float = 0.0
    num_tokens: int = 0

    @property
    def tok_per_sec(self) -> float:
        return self.num_tokens / (self.total_ms / 1000) if self.total_ms > 0 else 0


def _make_prompts(n: int, prompt_len: int = 0) -> list[str]:
    if prompt_len == 1:
        # Single-token prompts for decode-heavy benchmarks
        return [f"Hello" for _ in range(n)]
    return [f"The answer to question {i+1} is" for i in range(n)]


def _run_mode(mode: str, cfg: BenchConfig) -> None:
    from vllm import LLM, SamplingParams

    model_id = _MODEL_ALIASES.get(cfg.model, cfg.model)
    prompts = _make_prompts(cfg.num_prompts, getattr(cfg, 'prompt_len', 0))
    params = SamplingParams(temperature=0.0, max_tokens=cfg.max_tokens)

    # Build additional_config based on mode
    additional_config = {}
    worker_cls = None

    if mode == "baseline":
        pass  # no monitoring
    elif mode in ("ring_null", "ring_db"):
        worker_cls = "integration.vllm_adapter.DMXGPUWorker"
        additional_config = {
            "dmx_hook_selection": cfg.hook_selection,
            "dmx_ring_payload_mb": cfg.ring_payload_mb,
            "dmx_ring_pinned_mb": cfg.ring_pinned_mb,
        }
        if mode == "ring_null":
            additional_config["dmx_null_mode"] = True
        elif mode == "ring_db":
            additional_config["dmx_db_host"] = cfg.db_host
            additional_config["dmx_db_port"] = cfg.db_port
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create LLM
    kwargs = dict(
        model=model_id,
        max_model_len=cfg.max_model_len,
        enforce_eager=cfg.enforce_eager,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    if worker_cls:
        kwargs["worker_cls"] = worker_cls
        kwargs["additional_config"] = additional_config

    print(f"\n  Creating LLM for mode={mode} ...", flush=True)
    llm = LLM(**kwargs)

    # Warmup
    for i in range(cfg.warmup):
        _ = llm.generate(prompts, params)
    torch.cuda.synchronize()

    # Timed iterations
    results: list[RunResult] = []
    for i in range(cfg.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        r = RunResult(total_ms=(t1 - t0) * 1000, num_tokens=total_tokens)
        results.append(r)
        print(f"  iter {i+1:2d}:  total={r.total_ms:7.1f} ms  "
              f"tokens={r.num_tokens}  "
              f"throughput={r.tok_per_sec:.1f} tok/s", flush=True)

    # Summary
    times = [r.total_ms for r in results]
    tps = [r.tok_per_sec for r in results]
    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0
    mean_tps = statistics.mean(tps)

    print(f"\n  Summary:  total={mean_t:.1f} ms  std={std_t:.1f} ms  "
          f"throughput={mean_tps:.1f} tok/s")

    # Explicit per-worker flush+stop before LLM teardown. Avoids the
    # implicit DMXGPUWorker.shutdown() race against vLLM's 8s deadline
    # (see vllm_adapter.py:587 warning). No-op for baseline mode where
    # the worker isn't a DMXGPUWorker.
    try:
        llm.collective_rpc("stop_monitoring")
    except Exception:
        pass
    del llm
    torch.cuda.empty_cache()

    return mean_t, mean_tps


def main():
    cfg = _parse_args()
    model_id = _MODEL_ALIASES.get(cfg.model, cfg.model)

    W = 70
    print(f"\n{'=' * W}")
    print(f"  vLLM Ring Transport Benchmark")
    print(f"  model={model_id}  prompts={cfg.num_prompts}  max_tokens={cfg.max_tokens}")
    print(f"  enforce_eager={cfg.enforce_eager}  ring={cfg.ring_payload_mb}MB")
    print(f"  hooks={cfg.hook_selection}")
    print(f"  warmup={cfg.warmup}  iters={cfg.iters}")
    print(f"  modes={cfg.modes}")
    print(f"{'=' * W}")

    summaries = {}
    for mode in cfg.modes:
        print(f"\n{'─' * W}")
        print(f"  MODE: {mode}")
        print(f"{'─' * W}")

        # Clean caches between modes
        import shutil
        for d in [os.path.expanduser("~/.cache/vllm"),
                  f"/tmp/torchinductor_{os.environ.get('USER', 'user')}/"]:
            shutil.rmtree(d, ignore_errors=True)

        mean_t, mean_tps = _run_mode(mode, cfg)
        summaries[mode] = (mean_t, mean_tps)

    # Final comparison
    print(f"\n{'=' * W}")
    print(f"  COMPARISON")
    print(f"{'=' * W}")
    baseline_t = summaries.get("baseline", (None, None))[0]
    for mode, (mean_t, mean_tps) in summaries.items():
        overhead = ""
        if baseline_t is not None and mode != "baseline" and baseline_t > 0:
            pct = (mean_t - baseline_t) / baseline_t * 100
            overhead = f"  overhead={pct:+.1f}%"
        print(f"  {mode:20s}  {mean_t:7.1f} ms  {mean_tps:7.1f} tok/s{overhead}")
    print()


def _parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(description="vLLM ring-transport benchmark")

    g = p.add_argument_group("Workload")
    g.add_argument("--model", default="gpt2")
    g.add_argument("--num-prompts", type=int, default=8)
    g.add_argument("--prompt-len", type=int, default=0,
                   help="0=default multi-token, 1=single token per prompt")
    g.add_argument("--max-tokens", type=int, default=20)
    g.add_argument("--warmup", type=int, default=2)
    g.add_argument("--iters", type=int, default=5)
    g.add_argument("--modes", default="baseline,ring_null",
                   help="comma-separated: baseline,ring_null,ring_db")
    g.add_argument("--enforce-eager", action="store_true")
    g.add_argument("--hook-selection", default="vllm-full")

    g = p.add_argument_group("Ring engine")
    g.add_argument("--ring-payload-mb", type=int, default=4096)
    g.add_argument("--ring-pinned-mb", type=int, default=4096)

    g = p.add_argument_group("ClickHouse (ring_db mode)")
    g.add_argument("--db-host", default="localhost")
    g.add_argument("--db-port", type=int, default=9000)

    g = p.add_argument_group("vLLM")
    g.add_argument("--max-model-len", type=int, default=512)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.5)

    ns = p.parse_args()
    return BenchConfig(
        model=ns.model,
        num_prompts=ns.num_prompts,
        prompt_len=ns.prompt_len,
        max_tokens=ns.max_tokens,
        warmup=ns.warmup,
        iters=ns.iters,
        modes=[m.strip() for m in ns.modes.split(",")],
        enforce_eager=ns.enforce_eager,
        ring_payload_mb=ns.ring_payload_mb,
        ring_pinned_mb=ns.ring_pinned_mb,
        hook_selection=ns.hook_selection,
        db_host=ns.db_host,
        db_port=ns.db_port,
        max_model_len=ns.max_model_len,
        gpu_memory_utilization=ns.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
