"""Patch a _compare model file so it captures only ONE hook's tensor.

Phase 2b of the unified-adaptor refactor needs a per-hook isolation gate
(verification.md Sec.C).  The methodology runs three rollouts per cell:

  Original  -- vanilla AutoModelForCausalLM, no hooks.  Logprobs L_orig.
  Ours      -- _p variant + hook_selection=H.  Logprobs L_ours, tensor T_ours via ring/CH.
  Ref       -- _compare variant patched so only H's `.copy_()` line fires.
               Logprobs L_ref, tensor T_ref via the preallocated buffer.

For the Ref rollout, every other `.copy_()` in the _compare file must be
disabled; otherwise the captured tensor for hook H is correct but the
forward pass also writes to other buffers, which keeps the un-hooked
tensors live and could mask hook-perturbation regressions.

This module owns the source-level patching.  It is callable both as a
library (``patch_compare_model``) and as a CLI for ad-hoc isolation runs.

The patch is applied **in-place** to the source file, with the original
saved to a sibling ``.copy_isolate_backup`` so a Ctrl-C / crash doesn't
permanently mangle the tree.  ``unpatch`` restores from backup and
deletes it.

Buffer naming convention in _compare files:
  ``self._buf_<HOOK>[...].copy_(...)``
  ``module._buf_<HOOK>[...].copy_(...)``
where <HOOK> matches the short_name in HOOK_DEFS (e.g. ``q``, ``resid_pre``).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Map (framework, model_key) -> path to the _compare model source file.
_COMPARE_MODEL_PATHS = {
    ("hf", "gpt2"): (
        REPO_ROOT
        / "integration/transformers/src/transformers/models/gpt2_compare/modeling_gpt2.py"
    ),
    ("hf", "qwen3"): (
        REPO_ROOT
        / "integration/transformers/src/transformers/models/qwen3_compare/modeling_qwen3.py"
    ),
    ("vllm", "gpt2"): (
        REPO_ROOT
        / "integration/vllm/vllm/model_executor/models/gpt2_compare.py"
    ),
    ("vllm", "qwen3"): (
        REPO_ROOT
        / "integration/vllm/vllm/model_executor/models/qwen3_compare.py"
    ),
}

# Matches the canonical ``.copy_()`` capture pattern in _compare files:
#   self._buf_<NAME>[...].copy_(...)
#   module._buf_<NAME>[...].copy_(...)
# Captures group 2 = buffer suffix (the hook short_name).
_COPY_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<rest>(?:self|module)\._buf_(?P<buf>\w+)\b.*?\.copy_\(.*\)\s*)$"
)

_BACKUP_SUFFIX = ".copy_isolate_backup"

# Comment marker used in patched files; chosen so a `grep ISOLATE:` lights
# up the patched lines and a stray patched file is obvious to a reader.
_ISOLATE_COMMENT = "# ISOLATE: "


def compare_model_path(framework: str, model_key: str) -> Path:
    """Resolve the on-disk path of a _compare model source."""
    key = (framework, model_key)
    if key not in _COMPARE_MODEL_PATHS:
        raise KeyError(
            f"No _compare model registered for {key!r}; "
            f"known keys: {sorted(_COMPARE_MODEL_PATHS)}"
        )
    p = _COMPARE_MODEL_PATHS[key]
    if not p.is_file():
        raise FileNotFoundError(f"_compare model file missing: {p}")
    return p


def _patched_source(src: str, hook: str) -> tuple[str, list[str]]:
    """Return (patched_source, list of buffer names that were commented).

    Lines whose buffer suffix matches ``hook`` exactly are left intact;
    every other ``.copy_()`` capture line is replaced with a commented
    no-op (``# ISOLATE: <original>``).
    """
    out_lines: list[str] = []
    commented_bufs: list[str] = []
    for line in src.splitlines(keepends=True):
        m = _COPY_LINE_RE.match(line.rstrip("\n"))
        if m is None:
            out_lines.append(line)
            continue
        buf = m.group("buf")
        if buf == hook:
            out_lines.append(line)
            continue
        # Comment the line; keep the original indentation outside the comment
        # so blank-line surrounding indentation is preserved if a tool reads
        # it back.  Append the trailing newline if present.
        eol = "\n" if line.endswith("\n") else ""
        out_lines.append(f"{m.group('indent')}{_ISOLATE_COMMENT}{m.group('rest')}{eol}")
        commented_bufs.append(buf)
    return "".join(out_lines), commented_bufs


def patch(framework: str, model_key: str, hook: str) -> tuple[Path, list[str]]:
    """Apply isolation patch in-place; return (model_path, commented_bufs).

    Saves the original file to ``<file><BACKUP_SUFFIX>`` first.  If a
    backup already exists from a previous run that didn't unpatch, this
    raises -- avoid silently overwriting in case the backup is still the
    one source-of-truth.
    """
    p = compare_model_path(framework, model_key)
    backup = p.with_suffix(p.suffix + _BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(
            f"Stale backup already present at {backup}; "
            f"a previous patch run did not unpatch.  Manually inspect / "
            f"restore before running again."
        )
    src = p.read_text(encoding="utf-8")
    patched, commented = _patched_source(src, hook)
    if not commented:
        # No-op: nothing matched.  Still create the backup so the caller's
        # unpatch() is symmetric and the failure mode (typo in hook name)
        # surfaces cleanly when the comparator finds zero buffer writes.
        pass
    shutil.copy2(p, backup)
    p.write_text(patched, encoding="utf-8")
    return p, commented


def unpatch(framework: str, model_key: str) -> Path:
    """Restore from backup and delete the backup file."""
    p = compare_model_path(framework, model_key)
    backup = p.with_suffix(p.suffix + _BACKUP_SUFFIX)
    if not backup.exists():
        raise FileNotFoundError(
            f"No backup at {backup}; nothing to unpatch."
        )
    shutil.copy2(backup, p)
    backup.unlink()
    return p


@contextlib.contextmanager
def patch_compare_model(
    framework: str, model_key: str, hook: str
) -> Iterator[tuple[Path, list[str]]]:
    """Context-manager form of patch / unpatch.

    Yields ``(model_path, commented_bufs)``.  Always restores on exit,
    even on exception, even on Ctrl-C (best effort -- the backup file
    persists across crashes).
    """
    p, commented = patch(framework, model_key, hook)
    try:
        yield p, commented
    finally:
        try:
            unpatch(framework, model_key)
        except FileNotFoundError:
            # Already unpatched (e.g. if the body called unpatch directly).
            pass


def _bytecode_paths_for(p: Path) -> list[Path]:
    """Return the __pycache__ .pyc files that import would load for ``p``."""
    cache_dir = p.parent / "__pycache__"
    if not cache_dir.is_dir():
        return []
    stem = p.stem
    return [c for c in cache_dir.glob(f"{stem}.*.pyc") if c.is_file()]


def invalidate_bytecode(framework: str, model_key: str) -> int:
    """Delete cached .pyc files for the _compare module so the next import
    picks up the patched source.  Returns count of files removed."""
    p = compare_model_path(framework, model_key)
    n = 0
    for c in _bytecode_paths_for(p):
        c.unlink()
        n += 1
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    ap = argparse.ArgumentParser(
        description="Patch a _compare model file to capture one hook only."
    )
    ap.add_argument("--framework", required=True, choices=["hf", "vllm"])
    ap.add_argument(
        "--model", required=True, choices=["gpt2", "qwen3"], dest="model_key"
    )
    ap.add_argument("--hook", required=True,
                    help="Short hook name (e.g. q, resid_pre, final_logits)")
    ap.add_argument("--unpatch", action="store_true",
                    help="Restore from backup; ignores --hook")
    args = ap.parse_args()

    if args.unpatch:
        p = unpatch(args.framework, args.model_key)
        print(f"unpatched: {p}")
        return 0

    p, commented = patch(args.framework, args.model_key, args.hook)
    invalidate_bytecode(args.framework, args.model_key)
    print(f"patched: {p}")
    print(f"  isolated hook: {args.hook}")
    print(f"  commented {len(commented)} other .copy_() line(s): "
          f"{sorted(set(commented))}")
    print(f"  backup at: {p.with_suffix(p.suffix + _BACKUP_SUFFIX)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
