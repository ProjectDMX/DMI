#!/usr/bin/env python3
"""Generate synthetic PCIe D2H/H2D pressure with CUDA copies."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path
from typing import Any

from common import append_jsonl, event_base, load_manifest, read_json


STOP = threading.Event()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--gpus", default=None, help="Comma-separated CUDA device indices. Empty means CUDA default.")
    p.add_argument("--direction", choices=("d2h", "h2d", "bidirectional"), default=None)
    p.add_argument("--block-mb", type=int, default=None)
    p.add_argument("--inflight", type=int, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--target-mb-s", type=float, default=None, help="Per-GPU soft cap; 0 means uncapped.")
    p.add_argument("--sample-interval-s", type=float, default=None)
    p.add_argument("--dry-run", default=None)
    return p.parse_args()


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = read_json(args.config) if args.config else {}
    mapping = {
        "gpus": args.gpus,
        "direction": args.direction,
        "block_mb": args.block_mb,
        "inflight": args.inflight,
        "duration_s": args.duration_s,
        "target_mb_s": args.target_mb_s,
        "sample_interval_s": args.sample_interval_s,
        "dry_run": args.dry_run,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    cfg.setdefault("gpus", "")
    cfg.setdefault("direction", "bidirectional")
    cfg.setdefault("block_mb", 64)
    cfg.setdefault("inflight", 2)
    cfg.setdefault("duration_s", 30.0)
    cfg.setdefault("target_mb_s", 0.0)
    cfg.setdefault("sample_interval_s", 1.0)
    cfg.setdefault("dry_run", True)
    cfg["dry_run"] = parse_bool(cfg["dry_run"])
    return cfg


def parse_gpus(gpus: str, torch_mod: Any) -> list[int]:
    if gpus.strip():
        return [int(x) for x in gpus.split(",") if x.strip()]
    if torch_mod.cuda.is_available():
        return [torch_mod.cuda.current_device()]
    return []


def dry_run_worker(manifest: dict[str, Any], out: Path, cfg: dict[str, Any]) -> None:
    deadline = time.monotonic() + float(cfg["duration_s"])
    interval = float(cfg["sample_interval_s"])
    seq = 0
    while time.monotonic() < deadline and not STOP.is_set():
        row = event_base(manifest, source="synthetic_pcie_hog", event_type="dry_run_sample")
        row.update(
            {
                "seq": seq,
                "direction": cfg["direction"],
                "dry_run": True,
                "h2d_bytes": 0,
                "d2h_bytes": 0,
                "h2d_mb_s": 0.0,
                "d2h_mb_s": 0.0,
                "note": "No CUDA copies issued. Set DRY_RUN=0 to enable pressure.",
            }
        )
        append_jsonl(out, row)
        seq += 1
        STOP.wait(interval)


def rate_limit(target_mb_s: float, bytes_total: int, start_mono: float) -> None:
    if target_mb_s <= 0:
        return
    target_s = bytes_total / (target_mb_s * 1024 * 1024)
    elapsed = time.monotonic() - start_mono
    if target_s > elapsed:
        STOP.wait(target_s - elapsed)


def gpu_worker(
    *,
    gpu: int,
    manifest: dict[str, Any],
    out: Path,
    cfg: dict[str, Any],
    torch_mod: Any,
) -> None:
    torch_mod.cuda.set_device(gpu)
    device = torch_mod.device(f"cuda:{gpu}")
    block_bytes = int(cfg["block_mb"]) * 1024 * 1024
    inflight = max(1, int(cfg["inflight"]))
    direction = str(cfg["direction"])
    interval = float(cfg["sample_interval_s"])
    target_mb_s = float(cfg["target_mb_s"])
    deadline = time.monotonic() + float(cfg["duration_s"])

    host = [torch_mod.empty(block_bytes, dtype=torch_mod.uint8, pin_memory=True) for _ in range(inflight)]
    device_buf = [torch_mod.empty(block_bytes, dtype=torch_mod.uint8, device=device) for _ in range(inflight)]
    streams = [torch_mod.cuda.Stream(device=device) for _ in range(inflight)]
    torch_mod.cuda.synchronize(device)

    sample_start = time.monotonic()
    rate_start = sample_start
    h2d_bytes = 0
    d2h_bytes = 0
    total_bytes = 0
    iterations = 0

    while time.monotonic() < deadline and not STOP.is_set():
        idx = iterations % inflight
        stream = streams[idx]
        # Keep at most one copy queued per stream.  Without this guard the CPU
        # can enqueue hundreds of GiB before the first sampling synchronize,
        # making both duration_s and the per-window measurements meaningless.
        stream.synchronize()
        with torch_mod.cuda.stream(stream):
            if direction in {"h2d", "bidirectional"}:
                device_buf[idx].copy_(host[idx], non_blocking=True)
                h2d_bytes += block_bytes
                total_bytes += block_bytes
            if direction in {"d2h", "bidirectional"}:
                host[idx].copy_(device_buf[idx], non_blocking=True)
                d2h_bytes += block_bytes
                total_bytes += block_bytes
        iterations += 1

        now = time.monotonic()
        if now - sample_start >= interval:
            torch_mod.cuda.synchronize(device)
            elapsed = max(time.monotonic() - sample_start, 1e-9)
            row = event_base(manifest, source="synthetic_pcie_hog", event_type="copy_sample")
            row.update(
                {
                    "gpu_index": gpu,
                    "direction": direction,
                    "block_mb": int(cfg["block_mb"]),
                    "inflight": inflight,
                    "target_mb_s": target_mb_s,
                    "iterations": iterations,
                    "h2d_bytes": h2d_bytes,
                    "d2h_bytes": d2h_bytes,
                    "h2d_mb_s": h2d_bytes / 1024 / 1024 / elapsed,
                    "d2h_mb_s": d2h_bytes / 1024 / 1024 / elapsed,
                    "dry_run": False,
                }
            )
            append_jsonl(out, row)
            sample_start = time.monotonic()
            h2d_bytes = 0
            d2h_bytes = 0

        rate_limit(target_mb_s, total_bytes, rate_start)
        if target_mb_s > 0 and time.monotonic() - rate_start >= 1.0:
            rate_start = time.monotonic()
            total_bytes = 0

    torch_mod.cuda.synchronize(device)


def main() -> None:
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())

    args = parse_args()
    cfg = merged_config(args)
    run_dir = Path(args.run_dir)
    out = run_dir / "synthetic_pcie_hog.jsonl"
    manifest = load_manifest(run_dir)

    if cfg["dry_run"]:
        dry_run_worker(manifest, out, cfg)
        return

    try:
        import torch
    except ImportError as exc:
        row = event_base(manifest, source="synthetic_pcie_hog", event_type="error")
        row.update({"error": f"torch import failed: {exc}", "dry_run": False})
        append_jsonl(out, row)
        raise SystemExit("PyTorch is required for non-dry synthetic pressure") from exc

    if not torch.cuda.is_available():
        row = event_base(manifest, source="synthetic_pcie_hog", event_type="error")
        row.update({"error": "torch.cuda.is_available() is false", "dry_run": False})
        append_jsonl(out, row)
        raise SystemExit("CUDA is required for non-dry synthetic pressure")

    gpus = parse_gpus(str(cfg["gpus"]), torch)
    if not gpus:
        raise SystemExit("No GPU selected; set --gpus or GPUS")

    threads = [
        threading.Thread(
            target=gpu_worker,
            kwargs={"gpu": gpu, "manifest": manifest, "out": out, "cfg": cfg, "torch_mod": torch},
            daemon=True,
        )
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
