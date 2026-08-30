from __future__ import annotations

from math import prod, sqrt
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from dmi.storage.capture import (
    ArtifactProducer,
    ArtifactRef,
    CaptureCatalog,
    CaptureMetadata,
    CapturePage,
    CaptureQuery,
    CaptureReader,
    CaptureRecord,
    CaptureSelection,
    CORE_SUMMARY_VERSION,
    ExtensionError,
    ExtensionRegistry,
    FilesystemPackStore,
    HydrationLimitError,
    PackReader,
    PackWriter,
    ScalarMetric,
    decode_tensor,
    summarize_tensor,
)
from dmi.storage.capture.model import _DTYPE_BYTES
from dmi.storage.capture.summary import numpy_dtypes


pytestmark = pytest.mark.cpu


PACK_ID = UUID("018f0000-0000-7000-8000-000000000001")
WATERMARK = "catalog-1"

# NumPy dtype per capture dtype, for building the tensors that go *into* a pack.
_SOURCE_DTYPES = {
    "bool": np.bool_,
    "uint8": np.uint8,
    "int8": np.int8,
    "int16": np.int16,
    "float16": np.float16,
    "int32": np.int32,
    "float32": np.float32,
    "int64": np.int64,
    "float64": np.float64,
}


def _metadata(capture_id: str, *, dtype: str, shape: tuple[int, ...], step: int = 0):
    return CaptureMetadata(
        capture_id=capture_id,
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id="request-a",
        sequence_id="sequence-a",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=3,
        producer_rank=0,
        step_number=step,
        token_start=step,
        token_end=step + 1,
        batch_position=0,
        dtype=dtype,
        shape=shape,
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


class _Catalog(CaptureCatalog):
    def __init__(self, descriptors):
        self._descriptors = tuple(descriptors)

    def search(self, query: CaptureQuery) -> CapturePage:
        return CapturePage(
            items=self._descriptors[: query.limit],
            next_cursor=None,
            watermark=WATERMARK,
        )

    def get_by_ids(self, capture_ids, *, tenant_id, watermark):
        assert watermark == WATERMARK
        wanted = set(capture_ids)
        return tuple(
            i
            for i in self._descriptors
            if i.capture_id in wanted and i.metadata.tenant_id == tenant_id
        )


class _RecordingStore(FilesystemPackStore):
    """A store that remembers every byte range it was asked for."""

    def __init__(self, root: Path, *, store_id: str = "local"):
        super().__init__(root, store_id=store_id)
        self.ranges: list[tuple[int, int]] = []

    def read_range(self, ref, offset, length):
        self.ranges.append((offset, length))
        return super().read_range(ref, offset, length)


def _build(tmp_path: Path, records, *, gap_bytes: int = 4096):
    """Pack the records, store them, and return a reader over the result."""
    writer = PackWriter(
        pack_id=PACK_ID,
        created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=64 * 1024 * 1024,
    )
    for record in records:
        writer.append(record)
    sealed = writer.seal()

    store = _RecordingStore(tmp_path)
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptors = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )
    reader = CaptureReader(
        _Catalog(descriptors), {"local": store}, max_coalesce_gap_bytes=gap_bytes
    )
    return reader, store, descriptors


def _tensor_record(capture_id: str, array: np.ndarray, dtype: str, *, step: int = 0):
    return CaptureRecord(
        metadata=_metadata(
            capture_id, dtype=dtype, shape=tuple(array.shape), step=step
        ),
        payload=array.tobytes(),
    )


# --- gate A: identical decoded tensors ---------------------------------------


@pytest.mark.parametrize("dtype", sorted(_SOURCE_DTYPES))
def test_gate_selected_hydration_returns_identical_decoded_tensors(
    tmp_path: Path, dtype: str
):
    rng = np.random.default_rng(seed=17)
    source = (rng.random(64) * 100).astype(_SOURCE_DTYPES[dtype])
    reader, _, _ = _build(tmp_path, [_tensor_record("capture-a", source, dtype)])

    selection = reader.select(CaptureQuery(limit=10))
    hydrated = reader.hydrate(selection, byte_limit=1 << 20)
    decoded = decode_tensor(hydrated[0].descriptor, hydrated[0].payload)

    assert decoded.dtype == source.dtype
    assert decoded.shape == source.shape
    # Bit-exact, not approximate: the gate is identity.
    assert np.array_equal(decoded, source)
    assert decoded.tobytes() == source.tobytes()


def test_gate_bfloat16_round_trips_every_bit_pattern(tmp_path: Path):
    # Includes both NaN encodings, both infinities, signed zero and denormals.
    patterns = np.array(
        [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x0001, 0x7F7F],
        dtype="<u2",
    )
    reader, _, _ = _build(
        tmp_path,
        [
            CaptureRecord(
                metadata=_metadata(
                    "capture-a", dtype="bfloat16", shape=(len(patterns),)
                ),
                payload=patterns.tobytes(),
            )
        ],
    )

    selection = reader.select(CaptureQuery(limit=10))
    hydrated = reader.hydrate(selection, byte_limit=1 << 20)
    decoded = decode_tensor(hydrated[0].descriptor, hydrated[0].payload)

    # Widening to float32 is a pure 16-bit shift, so narrowing must return the
    # original bits for every pattern -- NaN payloads included.
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded.view(np.uint32) >> 16, patterns.astype(np.uint32))
    assert bool(np.isnan(decoded[6])) and bool(np.isnan(decoded[7]))
    assert decoded[4] == np.inf and decoded[5] == -np.inf


def test_every_supported_dtype_is_decodable():
    # A dtype accepted by CaptureMetadata but unknown to the decoder would only
    # fail at analysis time, long after the capture was written.
    assert set(numpy_dtypes()) | {"bfloat16"} == set(_DTYPE_BYTES)


# --- gate B: no unrelated payload bytes --------------------------------------


def _selected_extents(descriptors):
    return [
        (item.locator.offset, item.locator.offset + item.locator.stored_length)
        for item in descriptors
    ]


def test_gate_reads_no_unrelated_bytes_without_coalescing(tmp_path: Path):
    rng = np.random.default_rng(seed=3)
    records = [
        _tensor_record(
            f"capture-{index}",
            rng.random(256).astype(np.float32),
            "float32",
            step=index,
        )
        for index in range(6)
    ]
    reader, store, descriptors = _build(tmp_path, records, gap_bytes=0)

    # Select every other capture, so unselected payloads sit between them.
    wanted = descriptors[::2]
    selection = CaptureSelection.create(
        wanted, catalog_watermark=WATERMARK, filter_hash="f" * 64
    )
    # Warm the footer cache: the first hydration of a pack pays two extra
    # range reads (trailer + footer) to verify catalog descriptors against
    # the authoritative footer. The gate below measures payload discipline.
    reader.hydrate(selection, byte_limit=1 << 20)
    store.ranges.clear()
    reader.hydrate(selection, byte_limit=1 << 20)

    extents = _selected_extents(wanted)
    for offset, length in store.ranges:
        assert any(
            start <= offset and offset + length <= end for start, end in extents
        ), f"range {(offset, length)} falls outside every selected payload"
    assert sum(length for _, length in store.ranges) == sum(
        item.locator.stored_length for item in wanted
    )


def test_gate_amplification_stays_within_the_coalescing_bound(tmp_path: Path):
    rng = np.random.default_rng(seed=5)
    records = [
        _tensor_record(
            f"capture-{index}",
            rng.random(16).astype(np.float32),  # 64 B, far below the gap
            "float32",
            step=index,
        )
        for index in range(8)
    ]
    gap = 4096
    reader, store, descriptors = _build(tmp_path, records, gap_bytes=gap)

    wanted = descriptors[::2]
    selection = CaptureSelection.create(
        wanted, catalog_watermark=WATERMARK, filter_hash="f" * 64
    )
    estimate = reader.estimate(selection)
    # Warm the footer cache (see test_gate_reads_no_unrelated_bytes above).
    reader.hydrate(selection, byte_limit=1 << 20)
    store.ranges.clear()
    reader.hydrate(selection, byte_limit=1 << 20)

    read_bytes = sum(length for _, length in store.ranges)
    stored_bytes = sum(item.locator.stored_length for item in wanted)
    unrelated = read_bytes - stored_bytes

    # Coalescing does pull in unselected bytes -- that is the point -- but only
    # up to the configured gap per join, and the estimate must predict it.
    assert unrelated > 0, "expected coalescing to span the unselected payloads"
    assert unrelated <= gap * max(0, len(wanted) - 1)
    assert read_bytes == estimate.request_bytes
    assert estimate.read_amplification == pytest.approx(read_bytes / stored_bytes)


def test_estimate_predicts_reads_exactly_without_coalescing(tmp_path: Path):
    rng = np.random.default_rng(seed=11)
    records = [
        _tensor_record(
            f"capture-{index}", rng.random(32).astype(np.float32), "float32", step=index
        )
        for index in range(4)
    ]
    reader, store, _ = _build(tmp_path, records, gap_bytes=0)

    selection = reader.select(CaptureQuery(limit=10))
    estimate = reader.estimate(selection)
    # Warm the footer cache (see test_gate_reads_no_unrelated_bytes above):
    # estimate() prices payload requests only, never the footer binding.
    reader.hydrate(selection, byte_limit=1 << 20)
    store.ranges.clear()
    reader.hydrate(selection, byte_limit=1 << 20)

    assert sum(length for _, length in store.ranges) == estimate.request_bytes
    assert len(store.ranges) == estimate.request_count
    assert estimate.read_amplification == pytest.approx(1.0)


# --- core summary numerics ----------------------------------------------------


def _descriptor_for(array: np.ndarray, dtype: str, tmp_path: Path):
    reader, _, descriptors = _build(
        tmp_path, [_tensor_record("capture-a", array, dtype)]
    )
    return reader, descriptors[0]


@pytest.mark.parametrize("dtype", ("float32", "float64", "int32", "int64", "uint8"))
def test_core_summary_matches_a_numpy_reference(tmp_path: Path, dtype: str):
    rng = np.random.default_rng(seed=23)
    source = (rng.random(512) * 50).astype(_SOURCE_DTYPES[dtype])
    _, descriptor = _descriptor_for(source, dtype, tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    reference = source.astype(np.float64)
    assert summary.element_count == source.size
    assert summary.finite_count == source.size
    assert summary.nan_count == 0 and summary.inf_count == 0
    assert summary.mean == pytest.approx(reference.mean())
    assert summary.minimum == pytest.approx(reference.min())
    assert summary.maximum == pytest.approx(reference.max())
    assert summary.abs_max == pytest.approx(np.abs(reference).max())
    assert summary.l2_norm == pytest.approx(sqrt((reference**2).sum()))
    assert summary.zero_fraction == pytest.approx(
        np.count_nonzero(reference == 0) / source.size
    )
    assert summary.summary_version == CORE_SUMMARY_VERSION


def test_core_summary_separates_non_finite_values_from_statistics(tmp_path: Path):
    source = np.array(
        [1.0, 2.0, 3.0, np.nan, np.inf, -np.inf, 0.0], dtype=np.float32
    )
    _, descriptor = _descriptor_for(source, "float32", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    assert summary.element_count == 7
    assert summary.nan_count == 1
    assert summary.inf_count == 2
    assert summary.finite_count == 4
    # Statistics cover the finite values only, so one NaN cannot erase them all.
    assert summary.mean == pytest.approx((1.0 + 2.0 + 3.0 + 0.0) / 4)
    assert summary.maximum == pytest.approx(3.0)
    assert summary.minimum == pytest.approx(0.0)
    assert summary.zero_fraction == pytest.approx(1 / 7)


def test_core_summary_of_an_all_nan_tensor_is_distinguishable_from_zeros(
    tmp_path: Path,
):
    source = np.array([np.nan, np.nan], dtype=np.float32)
    _, descriptor = _descriptor_for(source, "float32", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    assert summary.finite_count == 0
    assert summary.nan_count == 2
    assert (summary.mean, summary.l2_norm) == (0.0, 0.0)


def test_core_summary_handles_an_empty_tensor(tmp_path: Path):
    source = np.zeros((0,), dtype=np.float32)
    _, descriptor = _descriptor_for(source, "float32", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    assert summary.element_count == 0
    assert summary.l2_norm == 0.0


def test_int64_extremes_do_not_overflow_the_summary(tmp_path: Path):
    source = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)
    _, descriptor = _descriptor_for(source, "int64", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    # abs() of int64 min overflows in int64; the summary works in float64.
    assert summary.abs_max == pytest.approx(float(abs(int(np.iinfo(np.int64).min))))
    assert summary.l2_norm > 0


def test_summary_element_count_agrees_with_the_catalog_facet(tmp_path: Path):
    from dmi.storage.capture.clickhouse_catalog import _FACET_COLUMNS

    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    _, descriptor = _descriptor_for(source, "float32", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    # The facet is the catalog's cheap copy of this number. A divergence means a
    # descriptor and its payload disagree.
    facet_expression = dict(
        (name, expression) for name, _, expression in _FACET_COLUMNS
    )["element_count"]
    assert facet_expression == "toUInt64(arrayProduct(shape))"
    assert summary.element_count == prod(descriptor.metadata.shape)


# --- reader.summarize and its limits ------------------------------------------


def test_summarize_returns_one_summary_per_selected_capture(tmp_path: Path):
    rng = np.random.default_rng(seed=31)
    records = [
        _tensor_record(
            f"capture-{index}", rng.random(16).astype(np.float32), "float32", step=index
        )
        for index in range(3)
    ]
    reader, _, _ = _build(tmp_path, records)

    selection = reader.select(CaptureQuery(limit=10))
    summaries = reader.summarize(selection, byte_limit=1 << 20)

    assert [item.capture_id for item in summaries] == [
        f"capture-{index}" for index in range(3)
    ]
    assert all(item.core.summary_version == 1 for item in summaries)


def test_summarize_reads_nothing_beyond_the_selected_ranges(tmp_path: Path):
    rng = np.random.default_rng(seed=37)
    records = [
        _tensor_record(
            f"capture-{index}", rng.random(64).astype(np.float32), "float32", step=index
        )
        for index in range(4)
    ]
    reader, store, descriptors = _build(tmp_path, records, gap_bytes=0)

    selection = reader.select(CaptureQuery(limit=10))
    # Warm the footer cache (see test_gate_reads_no_unrelated_bytes above).
    reader.hydrate(selection, byte_limit=1 << 20)
    store.ranges.clear()
    reader.summarize(selection, byte_limit=1 << 20)

    # Summarising must not add a single read on top of hydration.
    assert sum(length for _, length in store.ranges) == sum(
        item.locator.stored_length for item in descriptors
    )


def test_summarize_refuses_too_many_captures(tmp_path: Path):
    records = [
        _tensor_record(
            f"capture-{index}", np.zeros(4, dtype=np.float32), "float32", step=index
        )
        for index in range(3)
    ]
    reader, _, _ = _build(tmp_path, records)
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(HydrationLimitError, match="capture limit"):
        reader.summarize(selection, byte_limit=1 << 20, max_summary_captures=2)


def test_summarize_refuses_too_many_elements(tmp_path: Path):
    reader, _, _ = _build(
        tmp_path, [_tensor_record("capture-a", np.zeros(256, dtype=np.float32), "float32")]
    )
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(HydrationLimitError, match="element limit"):
        reader.summarize(selection, byte_limit=1 << 20, max_summary_elements=64)


def test_summarize_still_honours_the_hydration_byte_limit(tmp_path: Path):
    reader, _, _ = _build(
        tmp_path, [_tensor_record("capture-a", np.zeros(256, dtype=np.float32), "float32")]
    )
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(HydrationLimitError, match="byte limit"):
        reader.summarize(selection, byte_limit=8)


# --- extension points ---------------------------------------------------------


class _Sink:
    def __init__(self):
        self.written: list[tuple[str, str, bytes]] = []

    def put(self, *, capture_id, kind, version, data, content_type):
        self.written.append((capture_id, kind, data))
        return ArtifactRef(
            artifact_id=f"{capture_id}:{kind}",
            kind=kind,
            version=version,
            store_id="local",
            object_key=f"artifacts/{capture_id}/{kind}",
            object_bytes=len(data),
            checksum="0" * 8,
            content_type=content_type,
        )


def _one_capture(tmp_path: Path):
    reader, _, _ = _build(
        tmp_path,
        [_tensor_record("capture-a", np.array([1.0, -3.0], dtype=np.float32), "float32")],
    )
    return reader, reader.select(CaptureQuery(limit=10))


def test_a_registered_metric_contributes_a_scalar(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)
    registry = ExtensionRegistry()
    registry.register_metric(
        ScalarMetric(name="range", version=1, compute=lambda a: float(a.max() - a.min()))
    )

    summaries = reader.summarize(selection, byte_limit=1 << 20, registry=registry)

    assert summaries[0].scalars == {"range": pytest.approx(4.0)}
    assert summaries[0].failures == ()


def test_a_failing_metric_cannot_fail_the_summary(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)
    registry = ExtensionRegistry()

    def _explode(array):
        raise RuntimeError("metric is broken")

    registry.register_metric(ScalarMetric(name="broken", version=2, compute=_explode))
    registry.register_metric(
        ScalarMetric(name="mean", version=1, compute=lambda a: float(a.mean()))
    )

    summaries = reader.summarize(selection, byte_limit=1 << 20, registry=registry)

    # The core summary and the healthy metric both survive.
    assert summaries[0].core.element_count == 2
    assert summaries[0].scalars == {"mean": pytest.approx(-1.0)}
    failure = summaries[0].failures[0]
    assert (failure.name, failure.version, failure.error_type) == (
        "broken",
        2,
        "RuntimeError",
    )
    assert "metric is broken" in failure.message


def test_a_metric_returning_a_non_number_is_a_failure(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)
    registry = ExtensionRegistry()
    registry.register_metric(
        ScalarMetric(name="wrong", version=1, compute=lambda a: "not a float")
    )

    summaries = reader.summarize(selection, byte_limit=1 << 20, registry=registry)

    assert summaries[0].scalars == {}
    assert summaries[0].failures[0].error_type == "TypeError"


def test_a_metric_overrunning_its_budget_is_a_failure(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)
    clock = iter([0, 10_000, 20_000, 30_000, 40_000])
    registry = ExtensionRegistry(time_budget_ns=1, timer_ns=lambda: next(clock))
    registry.register_metric(
        ScalarMetric(name="slow", version=1, compute=lambda a: 1.0)
    )

    summaries = reader.summarize(selection, byte_limit=1 << 20, registry=registry)

    assert summaries[0].scalars == {}
    assert summaries[0].failures[0].error_type == "TimeoutError"


def test_an_artifact_producer_writes_through_the_sink(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)
    sink = _Sink()
    registry = ExtensionRegistry()
    registry.register_producer(
        ArtifactProducer(
            kind="raw", version=3, produce=lambda a: (a.tobytes(), "application/octet-stream")
        )
    )

    summaries = reader.summarize(
        selection, byte_limit=1 << 20, registry=registry, artifact_sink=sink
    )

    artifact = summaries[0].artifacts[0]
    assert (artifact.kind, artifact.version) == ("raw", 3)
    assert artifact.object_bytes == 8
    assert sink.written[0][0] == "capture-a"


def test_artifact_producers_without_a_sink_record_a_failure(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)
    registry = ExtensionRegistry()
    registry.register_producer(
        ArtifactProducer(kind="raw", version=1, produce=lambda a: (b"x", "text/plain"))
    )

    summaries = reader.summarize(selection, byte_limit=1 << 20, registry=registry)

    # A registered producer that never ran must be visible -- a silent skip
    # would be indistinguishable from "produced nothing".
    assert summaries[0].artifacts == ()
    assert [failure.error_type for failure in summaries[0].failures] == [
        "MissingArtifactSink"
    ]
    assert summaries[0].failures[0].name == "raw"


def test_a_producer_returning_the_wrong_shape_is_a_failure(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)
    registry = ExtensionRegistry()
    registry.register_producer(
        ArtifactProducer(kind="bad", version=1, produce=lambda a: b"just bytes")
    )

    summaries = reader.summarize(
        selection, byte_limit=1 << 20, registry=registry, artifact_sink=_Sink()
    )

    assert summaries[0].artifacts == ()
    assert summaries[0].failures[0].error_type == "TypeError"


def test_registry_refuses_more_than_max_extensions():
    registry = ExtensionRegistry(max_extensions=2)
    registry.register_metric(ScalarMetric(name="a", version=1, compute=lambda x: 1.0))
    registry.register_producer(
        ArtifactProducer(kind="b", version=1, produce=lambda x: (b"", "text/plain"))
    )

    with pytest.raises(ExtensionError, match="max_extensions"):
        registry.register_metric(ScalarMetric(name="c", version=1, compute=lambda x: 1.0))


def test_registry_refuses_duplicate_names():
    registry = ExtensionRegistry()
    registry.register_metric(ScalarMetric(name="a", version=1, compute=lambda x: 1.0))

    with pytest.raises(ExtensionError, match="already registered"):
        registry.register_metric(ScalarMetric(name="a", version=2, compute=lambda x: 2.0))


@pytest.mark.parametrize(
    "name,version", (("", 1), ("x" * 129, 1), ("ok", 0), ("ok", -1))
)
def test_registry_refuses_malformed_identities(name: str, version: int):
    with pytest.raises(ExtensionError):
        ScalarMetric(name=name, version=version, compute=lambda x: 1.0)


def test_l2_norm_survives_large_magnitude_float64(tmp_path: Path):
    # sqrt(sum(x**2)) overflows float64 well before the true norm does: squaring
    # 1e200 needs 1e400. The scaled form keeps every squared term <= 1.
    source = np.array([1e200, -1e200], dtype=np.float64)
    _, descriptor = _descriptor_for(source, "float64", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    assert np.isfinite(summary.l2_norm)
    assert summary.l2_norm == pytest.approx(np.sqrt(2.0) * 1e200, rel=1e-12)
    assert summary.abs_max == pytest.approx(1e200)


def test_l2_norm_still_matches_the_plain_formula_at_normal_scale(tmp_path: Path):
    rng = np.random.default_rng(seed=101)
    source = (rng.random(256) * 10 - 5).astype(np.float64)
    _, descriptor = _descriptor_for(source, "float64", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    # Scaling must not cost accuracy where the naive form was already fine.
    assert summary.l2_norm == pytest.approx(float(np.sqrt((source**2).sum())))


# --- registry construction and introspection -----------------------------------


def test_registry_rejects_non_positive_bounds():
    with pytest.raises(ValueError, match="max_extensions"):
        ExtensionRegistry(max_extensions=0)
    with pytest.raises(ValueError, match="time_budget_ns"):
        ExtensionRegistry(time_budget_ns=0)


def test_registry_exposes_its_bounds_and_registrations():
    registry = ExtensionRegistry(max_extensions=4, time_budget_ns=123)
    metric = registry.register_metric(
        ScalarMetric(name="mean", version=1, compute=lambda a: float(a.mean()))
    )
    producer = registry.register_producer(
        ArtifactProducer(kind="raw", version=1, produce=lambda a: (b"", "text/plain"))
    )

    assert registry.max_extensions == 4
    assert registry.time_budget_ns == 123
    assert registry.metrics == (metric,)
    assert registry.producers == (producer,)
    assert len(registry) == 2


def test_registry_refuses_duplicate_producer_kinds():
    registry = ExtensionRegistry()
    registry.register_producer(
        ArtifactProducer(kind="raw", version=1, produce=lambda a: (b"", "text/plain"))
    )

    with pytest.raises(ExtensionError, match="already registered"):
        registry.register_producer(
            ArtifactProducer(
                kind="raw", version=2, produce=lambda a: (b"", "text/plain")
            )
        )


# --- reader construction and limit validation -----------------------------------


def test_reader_rejects_a_negative_coalesce_gap(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")

    with pytest.raises(ValueError, match="max_coalesce_gap_bytes"):
        CaptureReader(_Catalog(()), {"local": store}, max_coalesce_gap_bytes=-1)


def test_reader_requires_at_least_one_store():
    with pytest.raises(ValueError, match="at least one pack store"):
        CaptureReader(_Catalog(()), {})


def test_reader_rejects_a_mismatched_store_mapping(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")

    with pytest.raises(ValueError, match="store mapping key"):
        CaptureReader(_Catalog(()), {"other": store})


def test_hydrate_validates_its_limits(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)

    with pytest.raises(ValueError, match="byte_limit"):
        reader.hydrate(selection, byte_limit=-1)
    with pytest.raises(ValueError, match="request_limit"):
        reader.hydrate(selection, byte_limit=1 << 20, request_limit=0)


def test_summarize_validates_its_limits(tmp_path: Path):
    reader, selection = _one_capture(tmp_path)

    with pytest.raises(ValueError, match="max_summary_captures"):
        reader.summarize(selection, byte_limit=1 << 20, max_summary_captures=0)
    with pytest.raises(ValueError, match="max_summary_elements"):
        reader.summarize(selection, byte_limit=1 << 20, max_summary_elements=0)


def test_hydrate_rejects_a_descriptor_naming_an_unknown_store(tmp_path: Path):
    from dmi.storage.capture.model import PackFormatError

    _, _, descriptors = _build(
        tmp_path / "packs",
        [_tensor_record("capture-a", np.zeros(2, dtype=np.float32), "float32")],
    )
    other = FilesystemPackStore(tmp_path / "other", store_id="other")
    reader = CaptureReader(_Catalog(descriptors), {"other": other})
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(PackFormatError, match="unknown pack store"):
        reader.hydrate(selection, byte_limit=1 << 20)


def test_hydrate_rejects_a_catalog_returning_duplicates(tmp_path: Path):
    from dmi.storage.capture.model import DuplicateCaptureError

    _, store, descriptors = _build(
        tmp_path,
        [_tensor_record("capture-a", np.zeros(2, dtype=np.float32), "float32")],
    )
    reader = CaptureReader(_Catalog(descriptors + descriptors), {"local": store})
    selection = CaptureSelection.create(
        descriptors, catalog_watermark=WATERMARK, filter_hash="f" * 64
    )

    with pytest.raises(DuplicateCaptureError, match="duplicate capture"):
        reader.hydrate(selection, byte_limit=1 << 20)


# --- decoding preconditions -----------------------------------------------------


def test_decode_tensor_rejects_a_compressed_locator(tmp_path: Path):
    from dataclasses import replace

    from dmi.storage.capture.model import PackFormatError

    source = np.zeros(2, dtype=np.float32)
    _, descriptor = _descriptor_for(source, "float32", tmp_path)
    compressed = replace(descriptor, locator=replace(descriptor.locator, codec="zstd"))

    with pytest.raises(PackFormatError, match="unsupported codec"):
        decode_tensor(compressed, source.tobytes())


def test_decode_tensor_rejects_a_payload_of_the_wrong_length(tmp_path: Path):
    from dmi.storage.capture.model import PackIntegrityError

    source = np.zeros(2, dtype=np.float32)
    _, descriptor = _descriptor_for(source, "float32", tmp_path)

    with pytest.raises(PackIntegrityError, match="payload length"):
        decode_tensor(descriptor, source.tobytes()[:-1])


def test_core_summary_of_an_all_zero_tensor_has_zero_norm(tmp_path: Path):
    source = np.zeros(4, dtype=np.float32)
    _, descriptor = _descriptor_for(source, "float32", tmp_path)

    summary = summarize_tensor(descriptor, source.tobytes())

    assert summary.l2_norm == 0.0
    assert summary.abs_max == 0.0
    assert summary.zero_fraction == 1.0
