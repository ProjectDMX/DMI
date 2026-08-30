"""CPU-only benchmark for bounded capture packing and local persistence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Sequence

from dmi.storage.capture import (
    AdmissionResult,
    CaptureMetadata,
    CaptureRecord,
    DirectPackSink,
    DurablePackSink,
    DurablePackSpool,
    FilesystemPackStore,
    HostCapturePipeline,
    OverloadPolicy,
    PackReader,
    PipelineConfig,
)

from .bench_capture_pack import parse_byte_size


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkConfig:
    mode: str = "direct"
    records: int = 10_000
    payload_bytes: int = 64 * 1024
    target_pack_bytes: int = 128 * 1024**2
    queue_records: int = 256
    queue_bytes: int = 256 * 64 * 1024
    pool_size: int = 64
    seed: int = 17
    trials: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {"direct", "spool"}:
            raise ValueError("mode must be direct or spool")
        for name in (
            "records",
            "payload_bytes",
            "target_pack_bytes",
            "queue_records",
            "queue_bytes",
            "pool_size",
            "trials",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.payload_bytes % 4:
            raise ValueError("payload_bytes must be a multiple of four")
        if self.target_pack_bytes < self.payload_bytes + 1024:
            raise ValueError("target_pack_bytes is too small for one record")
        if self.queue_bytes < self.payload_bytes:
            raise ValueError("queue_bytes must hold at least one payload")


@dataclass(frozen=True, slots=True)
class PipelineTrial:
    mode: str
    record_count: int
    persisted_records: int
    logical_bytes: int
    packed_bytes: int
    packs_persisted: int
    dropped_records: int
    queue_peak_records: int
    queue_peak_bytes: int
    admission_max_ns: int
    persist_max_ns: int
    seconds: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            **asdict(self),
            "logical_gib_per_second": self.logical_bytes / self.seconds / 1024**3,
            "packed_gib_per_second": self.packed_bytes / self.seconds / 1024**3,
            "space_amplification": self.packed_bytes / self.logical_bytes,
        }


def _payloads(config: PipelineBenchmarkConfig) -> tuple[bytes, ...]:
    generator = random.Random(config.seed)
    return tuple(
        generator.randbytes(config.payload_bytes)
        for _ in range(min(config.records, config.pool_size))
    )


def _record(config: PipelineBenchmarkConfig, index: int, payload: bytes) -> CaptureRecord:
    return CaptureRecord(
        metadata=CaptureMetadata(
            capture_id=f"capture-{index:012d}",
            tenant_id="benchmark",
            experiment_id="capture-pipeline",
            run_id=f"seed-{config.seed}",
            session_id="session-0",
            request_id=f"request-{index // 128}",
            sequence_id=f"sequence-{index // 128}",
            model_id="synthetic",
            model_revision="benchmark-v1",
            adapter_revision=None,
            capture_policy_version="all-v1",
            hook_name="resid_pre",
            layer_number=index % 32,
            producer_rank=0,
            step_number=index,
            token_start=index,
            token_end=index + 1,
            batch_position=index % 128,
            dtype="float32",
            shape=(config.payload_bytes // 4,),
            captured_at_ns=1_700_000_000_000_000_000 + index,
        ),
        payload=payload,
    )


def _verify(root: Path, mode: str) -> int:
    paths = (
        root.rglob("*.dmi-pack")
        if mode == "direct"
        else root.rglob("*.dmi-pack.ready")
    )
    records = 0
    for path in paths:
        records += len(PackReader.from_bytes(path.read_bytes()).descriptors(
            store_id="verify", object_key=path.name
        ))
    return records


def run_trial(config: PipelineBenchmarkConfig) -> PipelineTrial:
    payloads = _payloads(config)
    with tempfile.TemporaryDirectory(prefix="dmi-capture-pipeline-") as directory:
        root = Path(directory)
        if config.mode == "direct":
            sink = DirectPackSink(
                FilesystemPackStore(root / "objects", store_id="local")
            )
            verify_root = root / "objects"
        else:
            spool_bytes = max(
                config.target_pack_bytes * 2,
                config.records * (config.payload_bytes + 4096),
            )
            sink = DurablePackSink(
                DurablePackSpool(root / "spool", max_bytes=spool_bytes)
            )
            verify_root = root / "spool"
        pipeline = HostCapturePipeline(
            PipelineConfig(
                max_queue_records=config.queue_records,
                max_queue_bytes=config.queue_bytes,
                max_pack_bytes=config.target_pack_bytes,
                max_pack_records=10_000,
                max_linger_ns=1_000_000_000,
                overload_policy=OverloadPolicy.BLOCK,
                admission_timeout=30,
            ),
            sink,
        )

        started = time.perf_counter()
        pipeline.start()
        for index in range(config.records):
            result = pipeline.submit(
                _record(config, index, payloads[index % len(payloads)])
            )
            if result is not AdmissionResult.ACCEPTED:
                raise RuntimeError(f"unexpected admission result: {result.value}")
        snapshot = pipeline.close(timeout=30)
        elapsed = max(time.perf_counter() - started, math.ulp(1.0))

        verified = _verify(verify_root, config.mode)
        if verified != config.records:
            raise RuntimeError(f"verified {verified} records, expected {config.records}")

    return PipelineTrial(
        mode=config.mode,
        record_count=config.records,
        persisted_records=snapshot.persisted_records,
        logical_bytes=config.records * config.payload_bytes,
        packed_bytes=snapshot.packed_bytes,
        packs_persisted=snapshot.packs_persisted,
        dropped_records=snapshot.dropped_records,
        queue_peak_records=snapshot.queue_peak_records,
        queue_peak_bytes=snapshot.queue_peak_bytes,
        admission_max_ns=snapshot.admission_duration.max_ns,
        persist_max_ns=snapshot.persist_duration.max_ns,
        seconds=elapsed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "spool"), default="direct")
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--payload-bytes", type=parse_byte_size, default=64 * 1024)
    parser.add_argument(
        "--target-pack-bytes", type=parse_byte_size, default=128 * 1024**2
    )
    parser.add_argument("--queue-records", type=int, default=256)
    parser.add_argument(
        "--queue-bytes", type=parse_byte_size, default=256 * 64 * 1024
    )
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PipelineBenchmarkConfig(
        mode=args.mode,
        records=args.records,
        payload_bytes=args.payload_bytes,
        target_pack_bytes=args.target_pack_bytes,
        queue_records=args.queue_records,
        queue_bytes=args.queue_bytes,
        pool_size=args.pool_size,
        seed=args.seed,
        trials=args.trials,
    )
    if args.dry_run:
        result = {"dry_run": True, "config": asdict(config)}
    else:
        trials = [run_trial(config) for _ in range(config.trials)]
        rates = [trial.as_dict()["logical_gib_per_second"] for trial in trials]
        result = {
            "dry_run": False,
            "config": asdict(config),
            "trials": [trial.as_dict() for trial in trials],
            "summary": {
                "median_logical_gib_per_second": statistics.median(rates),
                "min_logical_gib_per_second": min(rates),
                "max_logical_gib_per_second": max(rates),
            },
        }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
