"""CPU-only Garage upload scaling benchmark for staged DMI packs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Sequence
from uuid import uuid4

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    DurablePackSpool,
    PackWriter,
    ParallelSpoolUploader,
    ParallelUploadConfig,
    S3PackStore,
    S3StoreConfig,
)

from .bench_capture_pack import parse_byte_size


@dataclass(frozen=True, slots=True)
class GarageBenchmarkConfig:
    pack_payload_bytes: tuple[int, ...] = (64 * 1024**2, 128 * 1024**2)
    multipart_threshold_bytes: tuple[int, ...] = (32 * 1024**2, 64 * 1024**2)
    upload_workers: tuple[int, ...] = (1, 2, 4, 8)
    packs_per_trial: int = 8
    multipart_chunk_bytes: int = 16 * 1024**2
    multipart_concurrency: int = 4
    trials: int = 3
    seed: int = 17

    def __post_init__(self) -> None:
        for name in (
            "pack_payload_bytes",
            "multipart_threshold_bytes",
            "upload_workers",
        ):
            values = getattr(self, name)
            if not values or any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive integers")
        for name in (
            "packs_per_trial",
            "multipart_chunk_bytes",
            "multipart_concurrency",
            "trials",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.multipart_chunk_bytes < 5 * 1024**2:
            raise ValueError("multipart_chunk_bytes must be at least 5 MiB")


@dataclass(frozen=True, slots=True)
class GarageUploadTrial:
    pack_payload_bytes: int
    multipart_threshold_bytes: int
    upload_workers: int
    packs: int
    object_bytes: int
    uploaded_bytes: int
    retries: int
    peak_active_uploads: int
    peak_in_flight_bytes: int
    seconds: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "gib_per_second": self.uploaded_bytes / self.seconds / 1024**3,
        }


def _pack(payload: bytes, index: int):
    identity = uuid4()
    metadata = CaptureMetadata(
        capture_id=f"garage-benchmark-{identity}",
        tenant_id="benchmark",
        experiment_id="garage-upload",
        run_id=str(identity),
        session_id="session-0",
        request_id=f"request-{index}",
        sequence_id=f"sequence-{index}",
        model_id="synthetic",
        model_revision="benchmark-v1",
        adapter_revision=None,
        capture_policy_version="all-v1",
        hook_name="resid_pre",
        layer_number=index,
        producer_rank=0,
        step_number=index,
        token_start=index,
        token_end=index + 1,
        batch_position=0,
        dtype="uint8",
        shape=(len(payload),),
        captured_at_ns=time.time_ns(),
    )
    writer = PackWriter(
        pack_id=identity,
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=len(payload) + 1024 * 1024,
    )
    writer.append(CaptureRecord(metadata=metadata, payload=payload))
    return writer.seal()


def run_trial(
    config: GarageBenchmarkConfig,
    store_config: S3StoreConfig,
    *,
    pack_payload_bytes: int,
    multipart_threshold_bytes: int,
    upload_workers: int,
) -> GarageUploadTrial:
    generator = random.Random(config.seed)
    payload = generator.randbytes(pack_payload_bytes)
    run_id = uuid4()
    with tempfile.TemporaryDirectory(prefix="dmi-garage-benchmark-") as directory:
        root = Path(directory)
        spool = DurablePackSpool(
            root / "spool",
            max_bytes=config.packs_per_trial * (pack_payload_bytes + 1024**2),
        )
        staged = []
        for index in range(config.packs_per_trial):
            pack = _pack(payload, index)
            key = f"benchmarks/dmi/{run_id}/{pack.pack_id}.dmi-pack"
            staged.append(spool.stage(pack, key))

        resolved_store = replace(
            store_config,
            multipart_threshold_bytes=multipart_threshold_bytes,
            multipart_chunk_bytes=config.multipart_chunk_bytes,
            multipart_concurrency=config.multipart_concurrency,
        )
        store = S3PackStore.from_config(resolved_store)
        byte_budget = max(item.object_bytes for item in staged) * upload_workers
        uploader = ParallelSpoolUploader(
            spool,
            store,
            ParallelUploadConfig(
                max_workers=upload_workers,
                max_in_flight_bytes=byte_budget,
            ),
        )

        started = time.perf_counter()
        result = uploader.upload_pending()
        elapsed = max(time.perf_counter() - started, math.ulp(1.0))
        if result.failures:
            raise RuntimeError(f"Garage upload failures: {result.failures}")
        for ref in result.refs:
            store.stat(ref)
            store.read_range(
                ref, max(0, ref.object_bytes - 32), min(32, ref.object_bytes)
            )

    return GarageUploadTrial(
        pack_payload_bytes=pack_payload_bytes,
        multipart_threshold_bytes=multipart_threshold_bytes,
        upload_workers=upload_workers,
        packs=config.packs_per_trial,
        object_bytes=sum(item.object_bytes for item in staged),
        uploaded_bytes=result.snapshot.uploaded_bytes,
        retries=result.snapshot.retries,
        peak_active_uploads=result.snapshot.peak_active_uploads,
        peak_in_flight_bytes=result.snapshot.peak_in_flight_bytes,
        seconds=elapsed,
    )


def _byte_list(value: str) -> tuple[int, ...]:
    return tuple(parse_byte_size(item.strip()) for item in value.split(","))


def _int_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _store_config_from_env(
    config: GarageBenchmarkConfig, multipart_threshold_bytes: int
) -> S3StoreConfig:
    required = {
        "endpoint_url": os.environ.get("DMI_S3_ENDPOINT"),
        "bucket": os.environ.get("DMI_S3_BUCKET"),
        "access_key_id": os.environ.get("DMI_S3_ACCESS_KEY_ID"),
        "secret_access_key": os.environ.get("DMI_S3_SECRET_ACCESS_KEY"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("missing Garage environment: " + ", ".join(missing))
    return S3StoreConfig(
        endpoint_url=required["endpoint_url"],
        bucket=required["bucket"],
        region=os.environ.get("DMI_S3_REGION", "garage"),
        access_key_id=required["access_key_id"],
        secret_access_key=required["secret_access_key"],
        store_id="garage-benchmark",
        allow_insecure_http=os.environ.get("DMI_S3_ALLOW_HTTP") == "1",
        multipart_threshold_bytes=multipart_threshold_bytes,
        multipart_chunk_bytes=config.multipart_chunk_bytes,
        multipart_concurrency=config.multipart_concurrency,
        max_pool_connections=max(
            32, max(config.upload_workers) * config.multipart_concurrency
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-payload-bytes", type=_byte_list, default="64MiB,128MiB")
    parser.add_argument("--multipart-threshold-bytes", type=_byte_list, default="32MiB,64MiB")
    parser.add_argument("--upload-workers", type=_int_list, default="1,2,4,8")
    parser.add_argument("--packs-per-trial", type=int, default=8)
    parser.add_argument("--multipart-chunk-bytes", type=parse_byte_size, default=16 * 1024**2)
    parser.add_argument("--multipart-concurrency", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = GarageBenchmarkConfig(
        pack_payload_bytes=args.pack_payload_bytes,
        multipart_threshold_bytes=args.multipart_threshold_bytes,
        upload_workers=args.upload_workers,
        packs_per_trial=args.packs_per_trial,
        multipart_chunk_bytes=args.multipart_chunk_bytes,
        multipart_concurrency=args.multipart_concurrency,
        trials=args.trials,
        seed=args.seed,
    )
    if args.dry_run:
        result = {"dry_run": True, "config": asdict(config)}
    else:
        trials = []
        for pack_bytes in config.pack_payload_bytes:
            for threshold in config.multipart_threshold_bytes:
                store_config = _store_config_from_env(config, threshold)
                for workers in config.upload_workers:
                    trials.extend(
                        run_trial(
                            config,
                            store_config,
                            pack_payload_bytes=pack_bytes,
                            multipart_threshold_bytes=threshold,
                            upload_workers=workers,
                        )
                        for _ in range(config.trials)
                    )
        rates = [trial.as_dict()["gib_per_second"] for trial in trials]
        result = {
            "dry_run": False,
            "config": asdict(config),
            "trials": [trial.as_dict() for trial in trials],
            "summary": {
                "median_gib_per_second": statistics.median(rates),
                "min_gib_per_second": min(rates),
                "max_gib_per_second": max(rates),
            },
        }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
