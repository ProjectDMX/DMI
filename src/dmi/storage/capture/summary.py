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


# Version 2: the order statistics (minimum, maximum, abs_max) of a non-float
# tensor are exact Python ints instead of float64-rounded floats. The values a
# summary serializes changed for integer magnitudes above 2**53 -- and abs_max
# can now be 2**63, one past int64 -- so the version moves with them: two
# builds writing different bytes under one version number would make a
# cross-implementation manifest comparison report NON-CONFORMANT for corpora
# that are byte-identical.
CORE_SUMMARY_VERSION = 2

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

    Exactness differs by statistic, and the difference is visible in the types:

    * ``minimum``, ``maximum`` and ``abs_max`` are order statistics -- they pick
      an element rather than combining elements, so nothing can overflow. For a
      non-float dtype they are **exact**, and carry a Python ``int``; ``abs_max``
      of an all-``-2**63`` int64 tensor is therefore ``2**63``, a value no int64
      can hold but a Python int can. For a float dtype they carry the selected
      ``float`` itself, equally exact.
    * ``mean`` and ``l2_norm`` accumulate across elements, which overflows int64
      (and, squared, overflows float64 too), so both are computed in float64 and
      are **approximate** for integer inputs above 2**53. ``zero_fraction`` is a
      ratio and is float by nature.
    """

    summary_version: int
    element_count: int
    finite_count: int
    nan_count: int
    inf_count: int
    zero_fraction: float
    mean: float
    minimum: float | int
    maximum: float | int
    abs_max: float | int
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

    # float64 for the *accumulating* statistics: int64 magnitudes and squared
    # sums both overflow their own dtype long before they trouble a double.
    # The order statistics do not accumulate and so do not need the widening --
    # which is lossy above 2**53 -- and are taken from ``flat`` further down.
    flat = array.reshape(-1)
    values = flat.astype(numpy.float64)
    zero_fraction = float(numpy.count_nonzero(values == 0.0) / element_count)

    is_float = descriptor.metadata.dtype in _FLOAT_DTYPES
    if is_float:
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
    # Scale before squaring. float64 holds values up to ~1e308, so squaring a
    # large-magnitude tensor overflows and the naive sqrt(sum(x**2)) returns inf
    # where the true norm is perfectly finite. Factoring out the largest
    # magnitude keeps every squared term at or below 1.
    scale = float(absolute.max())
    if scale == 0.0:
        l2_norm = 0.0
    else:
        l2_norm = scale * float(numpy.sqrt(numpy.square(absolute / scale).sum()))

    if is_float:
        minimum: float | int = float(finite.min())
        maximum: float | int = float(finite.max())
        abs_max: float | int = scale
    else:
        # Order statistics off the raw integers. numpy's integer min/max cannot
        # overflow, and skipping the float64 widening keeps magnitudes above
        # 2**53 exact instead of silently rounded. A non-float dtype admits no
        # NaN or Inf, so ``finite`` is the whole tensor and ``flat`` is the same
        # elements in their own dtype.
        minimum = int(flat.min())
        maximum = int(flat.max())
        # abs() in Python int space, not numpy's: |-2**63| has no int64
        # representation and would wrap back to -2**63. Python ints are
        # unbounded, so the true magnitude survives. The largest absolute value
        # is always at one end or the other, so the extremes suffice.
        abs_max = max(abs(minimum), abs(maximum))

    return CoreTensorSummaryV1(
        summary_version=CORE_SUMMARY_VERSION,
        element_count=element_count,
        finite_count=finite_count,
        nan_count=nan_count,
        inf_count=inf_count,
        zero_fraction=zero_fraction,
        mean=float(finite.mean()),
        minimum=minimum,
        maximum=maximum,
        abs_max=abs_max,
        l2_norm=l2_norm,
    )


def numpy_dtypes() -> Mapping[str, str]:
    """The dtype table, exposed so tests can assert full coverage."""
    return dict(_NUMPY_DTYPES)
