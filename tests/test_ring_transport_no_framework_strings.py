"""Lint-style test: monitoring/ring_transport.py is framework-neutral.

Phase 1.5 verification gate.  After the unified-adaptor refactor, the core
transport module has one explicit attribution block naming the two batch
conventions (HF = batched, vLLM = packed/flattened) and otherwise refers
to them only by their neutral names ("batched", "packed", "flattened").

This test asserts the framework-leaking patterns we don't want to see
*outside* that attribution block:

  - hyphenated preset names (``vllm-...``, ``hf-...``) -- these belong in
    framework-specific adapters via ``register_preset(...)``.
  - lowercase ``vllm`` / framework-package identifiers
    (``huggingface``, ``transformers`` as a name) -- these usually point
    to a leaked import.

The single attribution block at the top of the file is allowed to use
the proper names ``vLLM`` and ``HF`` so a reader can map the conventions
back to where they originated.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

RING_TRANSPORT = (
    Path(__file__).resolve().parent.parent / "monitoring" / "ring_transport.py"
)

# Patterns that are *always* a framework leak -- never allowed anywhere
# in ring_transport.py, including inside the attribution block.
FORBIDDEN_PATTERNS = [
    re.compile(r"\bvllm-[A-Za-z0-9_-]*"),    # "vllm-full", "vllm-foo", etc.
    re.compile(r"\bhf-[A-Za-z0-9_-]*"),      # "hf-only", "hf-foo", etc.
    re.compile(r"\bhuggingface\b", re.IGNORECASE),
    re.compile(r"\bvllm\b"),                  # lowercase 'vllm' -- usually points to a package or preset
    # 'transformers' as an identifier prefix.  We allow it as part of
    # words like "transformer" / "transformers" only inside import-style
    # references, but ring_transport.py shouldn't import from transformers.
    re.compile(r"\btransformers\.", re.IGNORECASE),
    re.compile(r"\bfrom transformers\b", re.IGNORECASE),
    re.compile(r"\bimport transformers\b", re.IGNORECASE),
]


# The attribution block.  ``vLLM`` and ``HF`` are allowed *only* on
# lines that fall within these byte ranges (matched against substrings
# in the file).  Today the block is bounded by these two anchor strings
# placed in the file as section delimiters.
ATTRIBUTION_BLOCK_START = "# Two batch conventions used throughout this file"
ATTRIBUTION_BLOCK_END = (
    '# Beyond this attribution block the rest of the file refers to the\n'
    '# conventions by their neutral names ("batched" / "packed").'
)


def _attribution_lines(source: str) -> set[int]:
    """Return the 1-indexed line numbers covered by the attribution block."""
    lines = source.splitlines()
    start = end = None
    for i, line in enumerate(lines, start=1):
        if start is None and ATTRIBUTION_BLOCK_START in line:
            start = i
        elif start is not None and 'neutral names' in line:
            end = i
            break
    if start is None or end is None:
        pytest.fail(
            "Attribution block markers not found in ring_transport.py; "
            "either restore the block or update this test's anchors.")
    return set(range(start, end + 1))


def test_no_forbidden_framework_patterns():
    """Verify no `vllm-...` / `hf-...` / lowercase-vllm / transformers-import."""
    src = RING_TRANSPORT.read_text()
    failures: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(src.splitlines(), start=1):
        for pat in FORBIDDEN_PATTERNS:
            m = pat.search(line)
            if m is not None:
                failures.append((line_no, pat.pattern, line.rstrip()))
    if failures:
        msg_lines = [
            f"Forbidden framework-leaking pattern(s) in ring_transport.py:"
        ]
        for line_no, pat, line in failures:
            msg_lines.append(f"  line {line_no} matches /{pat}/: {line}")
        pytest.fail("\n".join(msg_lines))


def test_framework_proper_names_only_in_attribution_block():
    """``vLLM`` / ``HF`` outside the attribution block point to a leak."""
    src = RING_TRANSPORT.read_text()
    allowed = _attribution_lines(src)
    capitalized_pat = re.compile(r"\bvLLM\b|\bHF\b")
    failures = []
    for line_no, line in enumerate(src.splitlines(), start=1):
        if line_no in allowed:
            continue
        m = capitalized_pat.search(line)
        if m is not None:
            failures.append((line_no, m.group(0), line.rstrip()))
    if failures:
        msg_lines = [
            "Framework name(s) outside the attribution block -- either "
            "rephrase in neutral terms ('batched' / 'packed') or extend "
            "the attribution block:"
        ]
        for line_no, name, line in failures:
            msg_lines.append(f"  line {line_no} ({name}): {line}")
        pytest.fail("\n".join(msg_lines))


def test_attribution_block_present():
    """Sanity check: the attribution block exists.

    Catches accidental deletion that would silently let any framework
    name pass `test_framework_proper_names_only_in_attribution_block`.
    """
    src = RING_TRANSPORT.read_text()
    assert ATTRIBUTION_BLOCK_START in src, (
        "Attribution block start marker missing from ring_transport.py")
    # Both names must appear at least once inside the block -- otherwise the
    # block isn't doing its job (a reader can't map conventions back to
    # their framework origins).
    block_lines = _attribution_lines(src)
    block_text = "\n".join(
        line for i, line in enumerate(src.splitlines(), start=1)
        if i in block_lines)
    assert "HF" in block_text
    assert "vLLM" in block_text
