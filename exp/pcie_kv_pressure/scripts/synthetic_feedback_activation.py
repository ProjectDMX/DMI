#!/usr/bin/env python3
"""Verify native governor feedback activation under sustained D2H pressure."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class D2HPressure:
    """Keep several CUDA streams occupied with large device-to-host copies."""

    def __init__(
        self,
        torch_mod: Any,
        *,
        device: int,
        block_bytes: int,
        inflight: int,
    ) -> None:
        self._torch = torch_mod
        self._device = device
        self._block_bytes = block_bytes
        self._inflight = max(1, inflight)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.bytes_copied = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="dmi-feedback-d2h-pressure",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise TimeoutError("D2H pressure worker did not initialize")
        self.raise_if_failed()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                raise TimeoutError("D2H pressure worker did not stop")
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("D2H pressure worker failed") from self._error

    def _run(self) -> None:
        try:
            torch_mod = self._torch
            torch_mod.cuda.set_device(self._device)
            device = torch_mod.device(f"cuda:{self._device}")
            hosts = [
                torch_mod.empty(
                    self._block_bytes,
                    dtype=torch_mod.uint8,
                    pin_memory=True,
                )
                for _ in range(self._inflight)
            ]
            devices = [
                torch_mod.empty(
                    self._block_bytes,
                    dtype=torch_mod.uint8,
                    device=device,
                )
                for _ in range(self._inflight)
            ]
            streams = [
                torch_mod.cuda.Stream(device=device) for _ in range(self._inflight)
            ]
            self._ready.set()

            while not self._stop.is_set():
                for host, device_buf, stream in zip(hosts, devices, streams):
                    stream.synchronize()
                    with torch_mod.cuda.stream(stream):
                        host.copy_(device_buf, non_blocking=True)
                    self.bytes_copied += self._block_bytes

            for stream in streams:
                stream.synchronize()
        except BaseException as exc:
            self._error = exc
            self._ready.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ring-mb", type=int, default=512)
    parser.add_argument("--dmi-batch-mb", type=int, default=64)
    parser.add_argument("--dmi-chunk-mb", type=int, default=16)
    parser.add_argument("--max-d2h-chunk-mb", type=int, default=32)
    parser.add_argument("--hog-block-mb", type=int, default=64)
    parser.add_argument("--hog-inflight", type=int, default=2)
    parser.add_argument("--hog-settle-ms", type=float, default=20.0)
    parser.add_argument("--baseline-s", type=float, default=0.6)
    parser.add_argument("--pressure-timeout-s", type=float, default=2.0)
    parser.add_argument("--batch-timeout-s", type=float, default=5.0)
    parser.add_argument("--feedback-window-ms", type=int, default=100)
    parser.add_argument("--hot-windows-to-yield", type=int, default=2)
    parser.add_argument("--p-hi", type=float, default=0.35)
    parser.add_argument("--p-lo", type=float, default=0.15)
    parser.add_argument("--pressure-ewma-alpha", type=float, default=0.20)
    parser.add_argument("--hard-watermark-ratio", type=float, default=0.80)
    parser.add_argument("--allow-no-activation", action="store_true")
    return parser.parse_args()


def probe_metrics(
    stats: dict[str, int], previous: dict[str, int]
) -> dict[str, float | int]:
    probe_bytes = int(stats["d2h_probe_bytes"]) - int(
        previous["d2h_probe_bytes"]
    )
    event_us = int(stats["d2h_probe_event_us"]) - int(
        previous["d2h_probe_event_us"]
    )
    host_us = int(stats["d2h_probe_host_us"]) - int(
        previous["d2h_probe_host_us"]
    )
    effective_us = max(event_us, host_us)
    return {
        "probe_bytes": probe_bytes,
        "probe_event_us": event_us,
        "probe_host_us": host_us,
        "probe_queue_us": max(0, host_us - event_us),
        "probe_event_gbps": (
            probe_bytes * 8.0 / (event_us * 1000.0) if event_us > 0 else 0.0
        ),
        "probe_effective_gbps": (
            probe_bytes * 8.0 / (effective_us * 1000.0)
            if effective_us > 0
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.dmi_batch_mb >= args.ring_mb * args.hard_watermark_ratio:
        raise ValueError("DMI batch must stay below the governor hard watermark")
    if args.dmi_batch_mb % args.dmi_chunk_mb:
        raise ValueError("dmi-batch-mb must be a multiple of dmi-chunk-mb")

    import torch

    from monitoring import _native_engine as native
    from monitoring.governor import PCIeGovernor, PCIeGovernorConfig
    from monitoring.ring_transport import RingTransport, activate, deactivate

    torch.cuda.set_device(args.device)
    ring_bytes = args.ring_mb * MIB
    batch_bytes = args.dmi_batch_mb * MIB
    chunk_bytes = args.dmi_chunk_mb * MIB
    num_chunks = batch_bytes // chunk_bytes

    cfg = native.RingConfig()
    cfg.payload_ring_bytes = ring_bytes
    cfg.pinned_staging_bytes = ring_bytes
    cfg.task_ring_entries = max(128, num_chunks * 8)
    cfg.drain_poll_timeout_us = 50
    cfg.drain_flush_byte_threshold = 1
    cfg.drain_flush_timeout_us = 0

    engine = native.RingEngine(cfg, None)
    engine.init()
    engine.start()
    transport = RingTransport(engine)
    activate(transport)
    governor = PCIeGovernor(
        engine,
        PCIeGovernorConfig(
            enabled=True,
            hard_watermark_ratio=args.hard_watermark_ratio,
            max_d2h_chunk_bytes=args.max_d2h_chunk_mb * MIB,
            feedback_window_ms=args.feedback_window_ms,
            hot_windows_to_yield=args.hot_windows_to_yield,
            p_hi=args.p_hi,
            p_lo=args.p_lo,
            pressure_ewma_alpha=args.pressure_ewma_alpha,
        ),
    )
    pressure = D2HPressure(
        torch,
        device=args.device,
        block_bytes=args.hog_block_mb * MIB,
        inflight=args.hog_inflight,
    )
    source = torch.empty(
        chunk_bytes,
        dtype=torch.uint8,
        device=f"cuda:{args.device}",
    )
    producer_stream = torch.cuda.current_stream(device=args.device)
    rows: list[dict[str, Any]] = []
    window_index = 0
    slow_path_flushes = 0
    phase_started = time.monotonic()
    last_stats = {str(k): int(v) for k, v in engine.link_stats().items()}

    def submit_dmi_batch() -> None:
        nonlocal slow_path_flushes
        d2h_target = int(engine.link_stats()["d2h_bytes"]) + batch_bytes
        result = int(engine.prepare_step(batch_bytes, num_chunks))
        if result not in (0, 1):
            raise RuntimeError(f"prepare_step failed: {result}")
        slow_path_flushes += int(result == 1)
        for hook_id in range(num_chunks):
            torch.ops.ring.producer(
                transport._ring_payload,
                source,
                0,
                hook_id,
            )
        producer_stream.synchronize()

        deadline = time.monotonic() + args.batch_timeout_s
        while True:
            stats = engine.link_stats()
            complete = (
                int(stats["d2h_bytes"]) >= d2h_target
                and int(stats["payload_reserved_bytes"]) == 0
                and int(stats["staging_used_bytes"]) == 0
            )
            if complete:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "DMI probe batch did not drain before batch-timeout-s"
                )
            time.sleep(0.0005)

    def sample_window(phase: str) -> dict[str, Any] | None:
        nonlocal last_stats, window_index
        previous_window = int(governor.snapshot()["window_started_ns"])
        governor.on_step()
        snapshot = governor.snapshot()
        if int(snapshot["window_started_ns"]) == previous_window:
            return None

        stats = {str(k): int(v) for k, v in engine.link_stats().items()}
        row = {
            "window": window_index,
            "phase": phase,
            "elapsed_s": time.monotonic() - phase_started,
            "baseline_gbps": float(snapshot["baseline_gbps"]),
            "ewma_pressure": float(snapshot["ewma_pressure"]),
            "hot_windows": int(snapshot["hot_windows"]),
            "feedback_active": bool(snapshot["feedback_active"]),
            "feedback_yields": int(snapshot["feedback_yields"]),
        }
        row.update(probe_metrics(stats, last_stats))
        rows.append(row)
        last_stats = stats
        window_index += 1
        return row

    activation_row: dict[str, Any] | None = None
    activation_latency_ms: float | None = None
    pressure_started = 0.0
    try:
        governor.on_step()
        last_stats = {str(k): int(v) for k, v in engine.link_stats().items()}

        baseline_deadline = time.monotonic() + args.baseline_s
        while time.monotonic() < baseline_deadline:
            submit_dmi_batch()
            sample_window("baseline")

        if governor.snapshot()["baseline_gbps"] <= 0:
            raise RuntimeError("baseline phase produced no usable DMI probe samples")

        pressure.start()
        time.sleep(max(0.0, args.hog_settle_ms) / 1000.0)
        pressure_started = time.monotonic()
        pressure_deadline = pressure_started + args.pressure_timeout_s
        while time.monotonic() < pressure_deadline:
            pressure.raise_if_failed()
            submit_dmi_batch()
            row = sample_window("pressure")
            if row is not None and row["feedback_active"]:
                activation_row = row
                activation_latency_ms = (
                    time.monotonic() - pressure_started
                ) * 1000.0
                break
    finally:
        try:
            pressure.stop()
        finally:
            try:
                governor.disable()
            finally:
                deactivate()
                engine.stop()

    summary = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "activated": activation_row is not None,
        "activation_latency_ms": activation_latency_ms,
        "activation_window": activation_row,
        "hog_d2h_bytes": pressure.bytes_copied,
        "slow_path_flushes": slow_path_flushes,
        "windows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "feedback_activation.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    if activation_row is None and not args.allow_no_activation:
        raise SystemExit(
            "feedback did not activate; inspect feedback_activation.json and "
            "increase sustained pressure or timeout"
        )
    if slow_path_flushes:
        raise SystemExit(
            "feedback benchmark entered prepare_step slow path; reduce dmi-batch-mb"
        )


if __name__ == "__main__":
    main()
