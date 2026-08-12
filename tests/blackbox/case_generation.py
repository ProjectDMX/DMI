"""Deterministic prompt generation for public-API differential tests."""

from __future__ import annotations

import random


_SUBJECTS = (
    "a cache entry",
    "the final token",
    "request 0007",
    "a multilingual user",
    "an empty-looking value '   '",
    "a shared prefix",
)
_ACTIONS = (
    "is summarized as",
    "should remain distinct from",
    "continues with",
    "maps deterministically to",
    "contains punctuation such as []{}<>",
)
_SUFFIXES = (
    "one concise sentence.",
    "three comma-separated words.",
    "a JSON-like value without code fences.",
    "Unicode text: naïve, Ελληνικά, 한글.",
    "the number immediately after 999.",
)


def generate_prompts(*, seed: int, count: int) -> list[str]:
    """Generate reproducible, unique strings using no model internals."""
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    prompts = []
    for index in range(count):
        subject = rng.choice(_SUBJECTS)
        action = rng.choice(_ACTIONS)
        suffix = rng.choice(_SUFFIXES)
        nonce = rng.randrange(1_000_000)
        prompts.append(
            f"case={index:03d}; nonce={nonce:06d}\n{subject} {action} {suffix}"
        )
    return prompts
