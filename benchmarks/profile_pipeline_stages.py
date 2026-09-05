"""T0.2 attribution: where the capture pipeline spends its time.

Single-threaded stage decomposition over the bench corpus (10k records x 64 KiB,
seed 17) plus the full pipeline for reference. Median of 3 trials per stage.
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import random
import statistics
import tempfile
import time
from pathlib import Path

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    DirectPackSink,
    DurablePackSink,
    DurablePackSpool,
    FilesystemPackStore,
    HostCapturePipeline,
    OverloadPolicy,
    PackWriter,
    PackAssembler,
    PipelineConfig,
    object_key_for,
)
from dmi.storage.capture.pack import _encode_json, _record_mapping
from dmi.storage.capture.pipeline import ReadyPack, FlushReason
import zlib

RECORDS = 10_000
PAYLOAD = 64 * 1024
SEED = 17


def _meta(index: int) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=f"capture-{index:012d}",
        tenant_id="benchmark",
        experiment_id="capture-pipeline",
        run_id=f"seed-{SEED}",
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
        shape=(PAYLOAD // 4,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def _records() -> list[CaptureRecord]:
    generator = random.Random(SEED)
    payloads = tuple(
        generator.randbytes(PAYLOAD) for _ in range(64)
    )
    return [
        CaptureRecord(metadata=_meta(i), payload=payloads[i % len(payloads)])
        for i in range(RECORDS)
    ]


def _rate(logical_bytes: int, seconds: float) -> float:
    return logical_bytes / seconds / 1024**3


def _median(fn, trials: int = 3):
    values = []
    for _ in range(trials):
        values.append(fn())
    seconds = statistics.median(v[0] for v in values)
    return seconds, values[-1][1]


def stage_writer_only(records):
    logical = RECORDS * PAYLOAD

    def run():
        started = time.perf_counter()
        writer = PackWriter(
            pack_id=__import__("uuid").uuid4(),
            created_at_ns=records[0].metadata.captured_at_ns,
            max_pack_bytes=128 * 1024**2,
            max_records=RECORDS,
        )
        packs = []
        for record in records:
            try:
                writer.append(record)
            except Exception:
                packs.append(writer.seal())
                writer = PackWriter(
                    pack_id=__import__("uuid").uuid4(),
                    created_at_ns=record.metadata.captured_at_ns,
                    max_pack_bytes=128 * 1024**2,
                    max_records=RECORDS,
                )
                writer.append(record)
        packs.append(writer.seal())
        return time.perf_counter() - started, sum(len(p.data) for p in packs)

    seconds, packed = _median(run)
    return {"stage": "writer_only", "seconds": seconds, "gib_s": _rate(logical, seconds)}


def stage_crc32(records):
    def run():
        started = time.perf_counter()
        total = 0
        for record in records:
            total += zlib.crc32(record.payload)
        return time.perf_counter() - started, total

    seconds, _ = _median(run)
    return {
        "stage": "crc32_only",
        "seconds": seconds,
        "gib_s": _rate(RECORDS * PAYLOAD, seconds),
    }


def stage_json_only(records):
    from dmi.storage.capture.pack import _IndexedRecord, _crc32

    def run():
        started = time.perf_counter()
        for index, record in enumerate(records):
            indexed = _IndexedRecord(
                metadata=record.metadata,
                offset=index * PAYLOAD,
                stored_length=PAYLOAD,
                decoded_length=PAYLOAD,
                codec="none",
                checksum=_crc32(record.payload),
            )
            _encode_json(_record_mapping(indexed))
        return time.perf_counter() - started, index

    seconds, _ = _median(run)
    return {"stage": "json_encode_only", "seconds": seconds, "gib_s": None}


def stage_assembler(records):
    logical = RECORDS * PAYLOAD

    def run():
        started = time.perf_counter()
        assembler = PackAssembler(
            max_pack_bytes=128 * 1024**2,
            max_records=RECORDS,
            max_linger_ns=1_000_000_000_000,
        )
        packs: list = []
        for index, record in enumerate(records):
            packs.extend(assembler.append(record, now_ns=index * 1000))
        packs.extend(assembler.flush(FlushReason.SESSION))
        packed = sum(len(p.pack.data) for p in packs)
        return time.perf_counter() - started, packed

    seconds, packed = _median(run)
    return {"stage": "assembler_only", "seconds": seconds, "gib_s": _rate(logical, seconds)}


def stage_store_put(records):
    logical = RECORDS * PAYLOAD

    def run():
        with tempfile.TemporaryDirectory() as directory:
            store = FilesystemPackStore(Path(directory) / "objects", store_id="local")
            writer = PackWriter(
                pack_id=__import__("uuid").uuid4(),
                created_at_ns=records[0].metadata.captured_at_ns,
                max_pack_bytes=512 * 1024**2,
                max_records=RECORDS,
            )
            for record in records:
                try:
                    writer.append(record)
                except Exception:
                    pass
            pack = writer.seal()
            ready = ReadyPack(pack, records[0].metadata, FlushReason.SESSION)
            started = time.perf_counter()
            store.put(pack, object_key_for(ready))
            return time.perf_counter() - started, len(pack.data)

    seconds, packed = _median(run)
    return {"stage": "store_put_only", "seconds": seconds, "gib_s": _rate(logical, seconds)}


def stage_spool_stage(records):
    logical = RECORDS * PAYLOAD

    def run():
        with tempfile.TemporaryDirectory() as directory:
            sink = DurablePackSink(
                DurablePackSpool(
                    Path(directory) / "spool",
                    max_bytes=2 * RECORDS * (PAYLOAD + 4096),
                )
            )
            writer = PackWriter(
                pack_id=__import__("uuid").uuid4(),
                created_at_ns=records[0].metadata.captured_at_ns,
                max_pack_bytes=512 * 1024**2,
                max_records=RECORDS,
            )
            for record in records:
                try:
                    writer.append(record)
                except Exception:
                    pass
            pack = writer.seal()
            ready = ReadyPack(pack, records[0].metadata, FlushReason.SESSION)
            started = time.perf_counter()
            sink.persist(ready)
            return time.perf_counter() - started, len(pack.data)

    seconds, packed = _median(run)
    return {"stage": "spool_stage_only", "seconds": seconds, "gib_s": _rate(logical, seconds)}


def stage_pipeline(records, mode: str):
    logical = RECORDS * PAYLOAD

    def run():
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if mode == "direct":
                sink = DirectPackSink(FilesystemPackStore(root / "objects", store_id="local"))
            else:
                sink = DurablePackSink(
                    DurablePackSpool(root / "spool", max_bytes=2 * RECORDS * (PAYLOAD + 4096))
                )
            pipeline = HostCapturePipeline(
                PipelineConfig(
                    max_queue_records=256,
                    max_queue_bytes=256 * 64 * 1024,
                    max_pack_bytes=128 * 1024**2,
                    max_pack_records=10_000,
                    max_linger_ns=1_000_000_000,
                    overload_policy=OverloadPolicy.BLOCK,
                    admission_timeout=30,
                ),
                sink,
            )
            started = time.perf_counter()
            pipeline.start()
            for record in records:
                pipeline.submit(record)
            pipeline.close(timeout=30)
            return time.perf_counter() - started, logical

    seconds, _ = _median(run)
    return {"stage": f"pipeline_{mode}", "seconds": seconds, "gib_s": _rate(logical, seconds)}


def profile_writer(records):
    profiler = cProfile.Profile()
    profiler.enable()
    stage_writer_only(records)
    profiler.disable()
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(18)
    return buffer.getvalue()


def main() -> None:
    records = _records()
    results = [
        stage_crc32(records),
        stage_json_only(records),
        stage_writer_only(records),
        stage_assembler(records),
        stage_store_put(records),
        stage_spool_stage(records),
        stage_pipeline(records, "direct"),
        stage_pipeline(records, "spool"),
    ]
    print(json.dumps(results, indent=2))
    print("\n--- cProfile: writer_only (top 18, cumulative) ---")
    print(profile_writer(records))


if __name__ == "__main__":
    main()
