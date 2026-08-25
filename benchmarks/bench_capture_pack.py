"""CPU-only benchmark for the immutable capture-pack writer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Sequence
from uuid import UUID

from dmi.storage.capture import CaptureMetadata, CaptureRecord, PackReader, PackWriter


_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}
_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


def parse_byte_size(value: str) -> int:
    normalized = value.strip().lower()
    split = len(normalized)
    while split and normalized[split - 1].isalpha():
        split -= 1
    number, unit = normalized[:split].strip(), normalized[split:]
    if not number.isdigit() or unit not in _BYTE_UNITS:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}")
    return int(number) * _BYTE_UNITS[unit]


@dataclass(frozen=True, slots=True)
class PackBenchmarkConfig:
    records: int = 10_000
    payload_bytes: int = 64 * 1024
    target_pack_bytes: int = 128 * 1024**2
    pool_size: int = 64
    pattern: str = "random"
    dtype: str = "float32"
    seed: int = 17
    trials: int = 5

    def __post_init__(self) -> None:
        for name in ("records", "payload_bytes", "target_pack_bytes", "pool_size", "trials"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.payload_bytes % _DTYPE_BYTES[self.dtype]:
            raise ValueError(
                f"payload_bytes must be a multiple of {_DTYPE_BYTES[self.dtype]}"
            )
        if self.target_pack_bytes < self.payload_bytes:
            raise ValueError("target_pack_bytes must be >= payload_bytes")
        if self.pattern not in {"zeros", "random"}:
            raise ValueError("pattern must be zeros or random")


@dataclass(frozen=True, slots=True)
class PackTrial:
    record_count: int
    logical_bytes: int
    packed_bytes: int
    largest_pack_bytes: int
    pack_count: int
    seconds: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "logical_gib_per_second": self.logical_bytes / self.seconds / 1024**3,
            "packed_gib_per_second": self.packed_bytes / self.seconds / 1024**3,
            "space_amplification": self.packed_bytes / self.logical_bytes,
        }


def generate_payload_pool(config: PackBenchmarkConfig) -> tuple[bytes, ...]:
    count = min(config.records, config.pool_size)
    if config.pattern == "zeros":
        return tuple(bytes(config.payload_bytes) for _ in range(count))
    generator = random.Random(config.seed)
    return tuple(generator.randbytes(config.payload_bytes) for _ in range(count))


def _metadata(config: PackBenchmarkConfig, index: int) -> CaptureMetadata:
    elements = config.payload_bytes // _DTYPE_BYTES[config.dtype]
    return CaptureMetadata(
        capture_id=f"capture-{index:012d}",
        tenant_id="benchmark",
        experiment_id="pack-writer",
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
        dtype=config.dtype,
        shape=(elements,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def run_trial(config: PackBenchmarkConfig) -> PackTrial:
    payloads = generate_payload_pool(config)
    packs = []
    writer: PackWriter | None = None
    pack_index = 0
    start = time.perf_counter()
    for index in range(config.records):
        record = CaptureRecord(
            metadata=_metadata(config, index),
            payload=payloads[index % len(payloads)],
        )
        if writer is None:
            writer = PackWriter(
                pack_id=UUID(int=pack_index + 1),
                created_at_ns=1_700_000_000_000_000_000 + pack_index,
                max_pack_bytes=config.target_pack_bytes,
                max_records=min(config.records, 10_000),
            )
        try:
            writer.append(record)
        except ValueError as exc:
            if writer.record_count == 0:
                raise ValueError(
                    "target_pack_bytes cannot hold one benchmark record"
                ) from exc
            packs.append(writer.seal())
            pack_index += 1
            writer = PackWriter(
                pack_id=UUID(int=pack_index + 1),
                created_at_ns=1_700_000_000_000_000_000 + pack_index,
                max_pack_bytes=config.target_pack_bytes,
                max_records=min(config.records, 10_000),
            )
            writer.append(record)
    if writer is not None:
        packs.append(writer.seal())
    elapsed = time.perf_counter() - start

    verified_records = 0
    for pack in packs:
        reader = PackReader.from_bytes(pack.data)
        descriptors = reader.descriptors(store_id="benchmark", object_key=pack.pack_id)
        for descriptor in descriptors:
            reader.read_payload(descriptor)
        verified_records += len(descriptors)
    if verified_records != config.records:
        raise RuntimeError(f"verified {verified_records} records, expected {config.records}")

    return PackTrial(
        record_count=config.records,
        logical_bytes=config.records * config.payload_bytes,
        packed_bytes=sum(len(pack.data) for pack in packs),
        largest_pack_bytes=max(len(pack.data) for pack in packs),
        pack_count=len(packs),
        seconds=max(elapsed, math.ulp(1.0)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--payload-bytes", type=parse_byte_size, default=64 * 1024)
    parser.add_argument("--target-pack-bytes", type=parse_byte_size, default=128 * 1024**2)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--pattern", choices=("zeros", "random"), default="random")
    parser.add_argument("--dtype", choices=tuple(_DTYPE_BYTES), default="float32")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PackBenchmarkConfig(
        records=args.records,
        payload_bytes=args.payload_bytes,
        target_pack_bytes=args.target_pack_bytes,
        pool_size=args.pool_size,
        pattern=args.pattern,
        dtype=args.dtype,
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
