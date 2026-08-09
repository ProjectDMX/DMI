"""Shared E2E test library (plan §7).

Consolidates the read / merge / align / compare / report logic that the
configurable matrix (:mod:`tests.e2e_matrix`), the pytest wrappers, and the
numeric-difference study all build on, so each rule lives in exactly one
place.

Submodules:
  align          -- left-pad strip, EOS trim, request_id "<gid>:<row>" parsing
  compare        -- Check + bitwise / allclose / row_count / transport_bitwise
  report         -- Check / CellResult dataclasses -> JSON(L) + human table
  clickhouse_io  -- read offload rows by request_id, dtype/hook maps, row counts
  segments       -- merge chunked segments -> dense tensors (segment_merger)
  disk_ref       -- load .pt / structured reference tensors written by ref workers
  hf_reference   -- ROL + GEN HF rollouts (re-export of tests.hf_reference)

Only ``align``, ``compare``, and ``report`` are imported eagerly here; they
are pure-CPU (torch-only).  The IO-heavy submodules are imported on demand
to keep ``import tests.lib`` cheap and offline-friendly.
"""
from __future__ import annotations

from tests.lib import align, compare, report  # noqa: F401

__all__ = ["align", "compare", "report"]
