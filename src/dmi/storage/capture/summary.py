"""Versioned tensor summaries over hydrated payloads.

Everything here runs on payloads the caller has *already* fetched under the
hydration byte and request limits, so summarising adds no object-store traffic
and the phase gate -- identical decoded tensors, no unrelated payload bytes --
holds by construction.

That placement is forced by the indexer: :meth:`CatalogIndexer.index` reads pack
footers only, so nothing tensor-derived can be computed at index time without
downloading every payload. Descriptor-derived facts live in the catalog facets
instead (see :mod:`.clickhouse_catalog`).

NumPy is imported lazily so that importing :mod:`dmi.storage.capture` keeps
working in environments that only write packs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from .model import (
    CaptureDescriptor,
    PackFormatError,
    PackIntegrityError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


CORE_SUMMARY_VERSION = 1

# Explicit byte orders: a summary must not change meaning with the host.
_NUMPY_DTYPES = {
    "bool": "|b1",
    "uint8": "|u1",
    "int8": "|i1",
    "int16": "<i2",
    "float16": "<f2",
    "int32": "<i4",
    "float32": "<f4",
    "int64": "<i8",
    "float64": "<f8",
}
_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def _numpy():
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PackFormatError(
            "tensor summaries require numpy; install it to use summarize()"
        ) from exc
    return numpy


@dataclass(frozen=True, slots=True)
class CoreTensorSummaryV1:
    """Statistics over one decoded tensor.

    ``mean``, ``minimum``, ``maximum``, ``abs_max`` and ``l2_norm`` are computed
    over the **finite** elements only; ``nan_count`` and ``inf_count`` account
    for the rest. A tensor with no finite elements reports zeros, so a caller
    reads the counts to tell "all zero" from "all NaN".
    """

    summary_version: int
    element_count: int
    finite_count: int
    nan_count: int
    inf_count: int
    zero_fraction: float
    mean: float
    minimum: float
    maximum: float
    abs_max: float
    l2_norm: float


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A derived object, referenced rather than inlined into a result."""

    artifact_id: str
    kind: str
    version: int
    store_id: str
    object_key: str
    object_bytes: int
    checksum: str
    content_type: str


def decode_tensor(descriptor: CaptureDescriptor, payload: bytes) -> "np.ndarray":
    """Decode a hydrated payload into its tensor.

    ``bfloat16`` has no native NumPy dtype. It is read as ``uint16`` and widened
    to ``float32`` by a 16-bit left shift, which is exact for every bit pattern
    including NaN and Inf encodings.
    """
    numpy = _numpy()
    metadata = descriptor.metadata
    locator = descriptor.locator

    if locator.codec != "none":
        raise PackFormatError(f"unsupported codec for decoding: {locator.codec}")
    if len(payload) != metadata.logical_bytes:
        raise PackIntegrityError(
            "payload length does not match dtype and shape: "
            f"{len(payload)} != {metadata.logical_bytes}"
        )

    if metadata.dtype == "bfloat16":
        raw = numpy.frombuffer(payload, dtype="<u2")
        array = (raw.astype(numpy.uint32) << 16).view(numpy.float32)
    else:
        try:
            array = numpy.frombuffer(payload, dtype=_NUMPY_DTYPES[metadata.dtype])
        except KeyError as exc:  # pragma: no cover - guarded by CaptureMetadata
            raise PackFormatError(f"unsupported dtype: {metadata.dtype}") from exc

    return array.reshape(metadata.shape)


def summarize_tensor(
    descriptor: CaptureDescriptor, payload: bytes
) -> CoreTensorSummaryV1:
    """Compute the versioned core summary for one hydrated capture."""
    numpy = _numpy()
    array = decode_tensor(descriptor, payload)
    element_count = int(array.size)

    if element_count == 0:
        return CoreTensorSummaryV1(
            summary_version=CORE_SUMMARY_VERSION,
            element_count=0,
            finite_count=0,
            nan_count=0,
            inf_count=0,
            zero_fraction=0.0,
            mean=0.0,
            minimum=0.0,
            maximum=0.0,
            abs_max=0.0,
            l2_norm=0.0,
        )

    # float64 throughout: int64 magnitudes and squared sums both overflow their
    # own dtype long before they trouble a double.
    values = array.reshape(-1).astype(numpy.float64)
    zero_fraction = float(numpy.count_nonzero(values == 0.0) / element_count)

    if descriptor.metadata.dtype in _FLOAT_DTYPES:
        nan_mask = numpy.isnan(values)
        inf_mask = numpy.isinf(values)
        nan_count = int(numpy.count_nonzero(nan_mask))
        inf_count = int(numpy.count_nonzero(inf_mask))
        finite = values[~(nan_mask | inf_mask)]
    else:
        nan_count = inf_count = 0
        finite = values

    finite_count = int(finite.size)
    if finite_count == 0:
        return CoreTensorSummaryV1(
            summary_version=CORE_SUMMARY_VERSION,
            element_count=element_count,
            finite_count=0,
            nan_count=nan_count,
            inf_count=inf_count,
            zero_fraction=zero_fraction,
            mean=0.0,
            minimum=0.0,
            maximum=0.0,
            abs_max=0.0,
            l2_norm=0.0,
        )

    absolute = numpy.abs(finite)
    return CoreTensorSummaryV1(
        summary_version=CORE_SUMMARY_VERSION,
        element_count=element_count,
        finite_count=finite_count,
        nan_count=nan_count,
        inf_count=inf_count,
        zero_fraction=zero_fraction,
        mean=float(finite.mean()),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        abs_max=float(absolute.max()),
        l2_norm=float(numpy.sqrt(numpy.square(finite).sum())),
    )


def numpy_dtypes() -> Mapping[str, str]:
    """The dtype table, exposed so tests can assert full coverage."""
    return dict(_NUMPY_DTYPES)
