"""Comparison standards for the E2E matrix and pytest wrappers (plan §7).

Four standards, one common interface.  Each returns a :class:`Check`
recording pass/fail plus the numeric drift (``max_abs`` / ``mean_abs`` /
``first_diff_pos``) **even on a pass**, so a green cell still surfaces a
"barely passing" trend:

- ``bitwise``           -- exact equality (raw bytes / ``torch.equal``).
- ``allclose``          -- ``torch.allclose(atol, rtol)`` with a named,
                           reported threshold.
- ``row_count``         -- schema + segment-count validation only.
- ``transport_bitwise`` -- ``.copy_()`` reference buffers vs ClickHouse ring
                           output; exact, same engine as ``bitwise`` but a
                           distinct name so the gating policy (§8) can treat
                           transport separately from model-output transparency.

This module is pure-CPU and only depends on ``torch``; it is unit-tested
without CUDA / ClickHouse / vLLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class Check:
    """One comparison result.

    ``max_abs`` / ``mean_abs`` / ``first_diff_pos`` are recorded whenever
    they can be computed (even on pass) so trends stay visible.  ``detail``
    is a short human string for tables / assertion messages.
    """

    name: str
    passed: bool
    max_abs: Optional[float] = None
    mean_abs: Optional[float] = None
    first_diff_pos: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "passed": self.passed}
        if self.max_abs is not None:
            d["max_abs"] = self.max_abs
        if self.mean_abs is not None:
            d["mean_abs"] = self.mean_abs
        if self.first_diff_pos is not None:
            d["first_diff_pos"] = self.first_diff_pos
        if self.detail:
            d["detail"] = self.detail
        return d


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bytes_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True iff ``a`` and ``b`` have identical raw bytes.

    Reinterprets as a same-width integer dtype so the compare is a true
    bitwise check (no NaN / signed-zero surprises).  Falls back to a
    storage-bytes compare for exotic element sizes.
    """
    a_c = a.contiguous()
    b_c = b.contiguous()
    if a_c.shape != b_c.shape or a_c.element_size() != b_c.element_size():
        return False
    dt = _INT_VIEW.get(a_c.element_size())
    if dt is None:
        return bytes(a_c.untyped_storage()) == bytes(b_c.untyped_storage())
    return torch.equal(a_c.view(dt), b_c.view(dt))


def _abs_diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, Optional[int]]:
    """Return (max_abs, mean_abs, first_diff_pos) over the flattened tensors.

    ``first_diff_pos`` is the index of the first element that differs in the
    flattened view, or ``None`` if the tensors are elementwise equal.
    """
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    n = min(af.numel(), bf.numel())
    af = af[:n]
    bf = bf[:n]
    diff = (af - bf).abs()
    max_abs = float(diff.max().item()) if n else 0.0
    mean_abs = float(diff.mean().item()) if n else 0.0
    ne = torch.nonzero(af != bf, as_tuple=False)
    first = int(ne[0].item()) if ne.numel() else None
    return max_abs, mean_abs, first


def _precheck(a: torch.Tensor, b: torch.Tensor, name: str) -> Optional[Check]:
    """Shape/dtype gate shared by the tensor standards.

    Returns a failing :class:`Check` on mismatch, else ``None``.
    """
    if a.shape != b.shape:
        return Check(name, False, detail=f"shape mismatch: {list(a.shape)} vs {list(b.shape)}")
    if a.dtype != b.dtype:
        return Check(name, False, detail=f"dtype mismatch: {a.dtype} vs {b.dtype}")
    return None


# ---------------------------------------------------------------------------
# The four standards
# ---------------------------------------------------------------------------


def bitwise(a: torch.Tensor, b: torch.Tensor, name: str = "bitwise") -> Check:
    """Exact equality.  Records drift stats when it fails."""
    pre = _precheck(a, b, name)
    if pre is not None:
        return pre
    if bytes_identical(a, b):
        return Check(name, True, max_abs=0.0, mean_abs=0.0, detail="bitwise equal")
    max_abs, mean_abs, first = _abs_diff_stats(a, b)
    return Check(name, False, max_abs=max_abs, mean_abs=mean_abs,
                 first_diff_pos=first, detail=f"max_abs={max_abs:.6e}")


def transport_bitwise(a: torch.Tensor, b: torch.Tensor,
                      name: str = "transport_bitwise") -> Check:
    """``.copy_()`` reference buffer vs ring/ClickHouse output -- exact.

    Identical engine to :func:`bitwise`; a distinct standard name so the
    §8 gating policy can keep transport bitwise even when model-output
    transparency is allowed to use ``allclose`` under CUDA graphs.
    """
    return bitwise(a, b, name)


def allclose(a: torch.Tensor, b: torch.Tensor, name: str = "allclose",
             *, atol: float = 1e-3, rtol: float = 0.0) -> Check:
    """``torch.allclose`` with a named, reported threshold.

    Always records max/mean abs diff -- even on a pass -- so the threshold
    headroom stays visible.
    """
    pre = _precheck(a, b, name)
    if pre is not None:
        return pre
    max_abs, mean_abs, first = _abs_diff_stats(a, b)
    passed = bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))
    return Check(
        name, passed, max_abs=max_abs, mean_abs=mean_abs,
        first_diff_pos=None if passed else first,
        detail=f"max_abs={max_abs:.6e} (atol={atol:g}, rtol={rtol:g})",
    )


def row_count(per_hook_counts: dict, name: str = "row_count", *,
              min_per_layer_types: int = 10,
              require_final_logits: bool = True) -> Check:
    """Schema + segment-count validation only (no value comparison).

    ``per_hook_counts`` maps a ClickHouse ``act_name`` to the number of rows
    captured for it.  Validates that:

    - there are enough per-layer hook types (``blocks.*``),
    - every per-layer hook captured the same number of rows, and
    - the global ``final_logits`` hook is present (when required).
    """
    if not per_hook_counts:
        return Check(name, False, detail="no rows")
    per_layer = {k: v for k, v in per_hook_counts.items() if k.startswith("blocks.")}
    problems: list[str] = []
    if len(per_layer) < min_per_layer_types:
        problems.append(f"only {len(per_layer)} per-layer types (<{min_per_layer_types})")
    counts = set(per_layer.values())
    if len(counts) > 1:
        problems.append(f"uneven per-layer counts: {sorted(counts)}")
    if require_final_logits and "final_logits" not in per_hook_counts:
        problems.append("final_logits missing")
    passed = not problems
    detail = "ok" if passed else "; ".join(problems)
    return Check(name, passed, detail=detail)


# Registry of the tensor-pair standards for matrix dispatch by name.
TENSOR_STANDARDS: dict[str, Callable[..., Check]] = {
    "bitwise": bitwise,
    "transport_bitwise": transport_bitwise,
    "allclose": allclose,
}

ALL_STANDARDS = tuple(TENSOR_STANDARDS) + ("row_count",)


def compare_tensors(a: torch.Tensor, b: torch.Tensor, standard: str,
                    name: Optional[str] = None, **kwargs) -> Check:
    """Dispatch a tensor-pair comparison by standard name.

    ``standard`` must be one of :data:`TENSOR_STANDARDS` (``row_count`` is
    not a tensor-pair standard -- call :func:`row_count` directly).
    """
    fn = TENSOR_STANDARDS.get(standard)
    if fn is None:
        raise ValueError(
            f"unknown tensor standard {standard!r}; "
            f"expected one of {sorted(TENSOR_STANDARDS)}"
        )
    return fn(a, b, name or standard, **kwargs)
