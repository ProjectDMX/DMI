"""Machine- and human-readable matrix output (plan §7, §8).

A :class:`CellResult` is one cell of the E2E matrix -- one
``(backend, model, mode, standard, hook_selection, tp, ...)`` point.  It
serialises to a single JSON record (one per line in the JSONL artifact, §8)
and renders into a compact human table.

Pure-CPU; depends only on the stdlib and :mod:`tests.lib.compare`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional

from tests.lib.compare import Check


@dataclass
class CellResult:
    """Result of one matrix cell.

    ``checks`` holds the per-tensor / per-hook :class:`Check` objects.
    ``passed`` is the cell verdict (all checks passed and no error).
    ``error`` carries a setup/dispatch failure message when the cell could
    not run at all (distinct from a cell that ran and failed a check).
    ``extra`` stashes axis values that don't have a first-class field
    (ring sizes, dtype, prompt-set, ...).
    """

    backend: str
    model: str
    mode: str
    standard: str
    hook_selection: str
    tp: int = 1
    passed: bool = False
    checks: List[Check] = field(default_factory=list)
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def finalize(self) -> "CellResult":
        """Set ``passed`` from the checks (no error + every check passed)."""
        self.passed = self.error is None and bool(self.checks) and all(
            c.passed for c in self.checks)
        return self

    def to_record(self) -> dict:
        rec: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "mode": self.mode,
            "standard": self.standard,
            "hook_selection": self.hook_selection,
            "tp": self.tp,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }
        if self.error is not None:
            rec["error"] = self.error
        if self.extra:
            rec["extra"] = self.extra
        return rec


def checks_from_legacy_result(result: dict) -> List[Check]:
    """Adapt a legacy comparator ``result.json`` into :class:`Check` objects.

    The existing comparators emit ``{"tests": [{"name", "passed", "detail"}]}``;
    this lets the matrix dispatch to them unchanged and still produce the
    new record shape.
    """
    out: List[Check] = []
    for t in result.get("tests", []):
        out.append(Check(
            name=t.get("name", "?"),
            passed=bool(t.get("passed", False)),
            detail=t.get("detail", ""),
        ))
    return out


def write_jsonl(results: List[CellResult], path: str) -> None:
    """Write one JSON record per line (creates parent dirs)."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_record()) + "\n")


def read_jsonl(path: str) -> List[dict]:
    """Read a JSONL artifact back into a list of records."""
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def human_table(results: List[CellResult]) -> str:
    """Render the matrix as a compact fixed-width table."""
    header = ("backend", "model", "mode", "standard", "hooks", "tp", "result", "n_fail")
    rows: List[tuple] = []
    for r in results:
        n_fail = sum(1 for c in r.checks if not c.passed)
        verdict = "ERROR" if r.error is not None else ("PASS" if r.passed else "FAIL")
        rows.append((
            r.backend, r.model, r.mode, r.standard, r.hook_selection,
            str(r.tp), verdict, str(n_fail),
        ))
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*header), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*row) for row in rows]
    n_pass = sum(1 for r in results if r.passed)
    n_err = sum(1 for r in results if r.error is not None)
    lines.append("")
    lines.append(f"{n_pass}/{len(results)} cells passed"
                 + (f", {n_err} errored" if n_err else ""))
    return "\n".join(lines)
