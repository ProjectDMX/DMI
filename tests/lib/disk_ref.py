"""Load reference tensors written to disk by the ref workers (plan §7).

Two on-disk reference shapes exist:

- The vLLM ``RefDiskWorker`` writes per-request ``.pt`` files named
  ``{hook}_L{layer}_T{start}_{end}[_SR{rank}].pt`` under
  ``<ref_dir>/<request_id>/``.  :func:`scan_pt_ref_files` parses those.
- The HF reference runner writes a structured dump consumed via
  ``tests.hf_reference._load_hf_refs_from_disk``; :func:`load_hf_refs`
  re-exports it (lazily, to avoid importing the heavy HF module on CPU).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# {hook}[_L{layer}]_T{start}_{end}[_SR{rank}].pt
PT_RE = re.compile(
    r"^(?P<hook>\w+?)(?:_L(?P<layer>\d+))?_T(?P<start>\d+)_(?P<end>\d+)"
    r"(?:_SR(?P<shard>\d+))?\.pt$"
)


@dataclass(frozen=True)
class RefFile:
    """One parsed reference ``.pt`` file."""

    req_id: str
    hook: str
    layer: int      # -1 for global hooks
    shard: int      # 0 when TP == 1
    start: int
    end: int
    path: str

    @property
    def label(self) -> str:
        lbl = f"{self.req_id}/{self.hook}"
        if self.layer >= 0:
            lbl += f"_L{self.layer}"
        return lbl + f"_T{self.start}_{self.end}"


def parse_pt_name(name: str) -> Optional[dict]:
    """Parse a ref ``.pt`` filename into its fields, or ``None`` if unmatched."""
    m = PT_RE.match(name)
    if not m:
        return None
    return {
        "hook": m.group("hook"),
        "layer": int(m.group("layer")) if m.group("layer") is not None else -1,
        "shard": int(m.group("shard")) if m.group("shard") is not None else 0,
        "start": int(m.group("start")),
        "end": int(m.group("end")),
    }


def scan_pt_ref_files(ref_dir: str) -> List[RefFile]:
    """Walk ``<ref_dir>/<req_id>/*.pt`` and return parsed :class:`RefFile`s."""
    root = Path(ref_dir)
    out: List[RefFile] = []
    for req_dir in sorted(root.iterdir()):
        if not req_dir.is_dir():
            continue
        for pt_file in sorted(req_dir.iterdir()):
            parsed = parse_pt_name(pt_file.name)
            if parsed is None:
                continue
            out.append(RefFile(req_id=req_dir.name, path=str(pt_file), **parsed))
    return out


def load_pt(path: str):
    """Load a single reference tensor (CPU, weights-only)."""
    import torch
    return torch.load(path, weights_only=True, map_location="cpu")


def load_hf_refs(ref_dir: str):
    """Load the HF structured reference dump (delegates to tests.hf_reference)."""
    from tests.hf_reference import _load_hf_refs_from_disk
    return _load_hf_refs_from_disk(ref_dir)
