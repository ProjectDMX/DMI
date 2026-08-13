"""Deterministic prompt generation for public-API differential tests."""

from __future__ import annotations

import random


GENERATOR_NAME = "dmi-vllm-public-cases"
GENERATOR_VERSION = 2


def validate_case_corpus(payload: dict) -> None:
    """Fail closed when an auditable case corpus is structurally incomplete."""
    if payload.get("schema_version") != 2:
        raise ValueError("black-box corpus must use schema_version 2")
    if not isinstance(payload.get("name"), str) or not payload["name"]:
        raise ValueError("black-box corpus must have a non-empty name")
    generator = payload.get("generator")
    if not isinstance(generator, dict) or not isinstance(
        generator.get("name"), str
    ) or not isinstance(generator.get("version"), int):
        raise ValueError("black-box corpus must identify its generator and version")
    if payload.get("executions") != ["batch", "reversed"]:
        raise ValueError("black-box corpus must request batch and reversed executions")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("black-box corpus must contain a non-empty cases list")
    seen: set[str] = set()
    required = {
        "case_id",
        "checklist_ids",
        "input",
        "sampling",
        "dimensions",
        "oracles",
        "kills",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"case {index} is missing fields {missing}")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"case {index} has an invalid case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if not isinstance(case["checklist_ids"], list) or not case["checklist_ids"]:
            raise ValueError(f"{case_id} must map to checklist IDs")
        input_spec = case["input"]
        if (
            not isinstance(input_spec, dict)
            or input_spec.get("form")
            not in {"text", "token_ids_from_text", "text_with_image"}
            or not isinstance(input_spec.get("text"), str)
        ):
            raise ValueError(f"{case_id} has an invalid public input spec")
        if input_spec.get("form") == "text_with_image":
            image = input_spec.get("image")
            if (
                not isinstance(image, dict)
                or image.get("mode") != "RGB"
                or image.get("size") != [32, 32]
                or image.get("color") != [17, 101, 203]
            ):
                raise ValueError(
                    f"{case_id} has an invalid deterministic image spec"
                )
        sampling = case["sampling"]
        if not isinstance(sampling, dict) or sampling.get("temperature") != 0.0:
            raise ValueError(f"{case_id} must declare deterministic sampling")
        if not isinstance(sampling.get("max_tokens"), int) or sampling["max_tokens"] < 1:
            raise ValueError(f"{case_id} must declare positive max_tokens")
        if not isinstance(case["dimensions"], dict) or not case["dimensions"]:
            raise ValueError(f"{case_id} must declare covered dimensions")
        if not isinstance(case["oracles"], list) or not {
            "differential",
            "reverse-batch-order",
        }.issubset(case["oracles"]):
            raise ValueError(f"{case_id} must declare both public oracles")
        if not isinstance(case["kills"], list) or not case["kills"]:
            raise ValueError(f"{case_id} must name at least one killed fault")

    omitted = payload.get("omitted_combinations")
    if not isinstance(omitted, list):
        raise ValueError("black-box corpus must list omitted combinations")
    for item in omitted:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("dimension"), str)
            or not isinstance(item.get("reason"), str)
        ):
            raise ValueError("each omitted combination needs dimension and reason")


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


def generate_cases(*, seed: int, count: int) -> list[dict]:
    """Generate auditable public-API cases from a reproducible seed.

    The generator deliberately knows nothing about vLLM/DMI implementation
    objects. Every case declares its observable oracle and the plausible
    integration fault it is intended to reject.
    """
    prompts = generate_prompts(seed=seed, count=count)
    return [
        {
            "case_id": f"generated-{seed}-{index:03d}",
            "checklist_ids": ["P02", "P03", "P04", "P05", "P07"],
            "seed": seed,
            "input": {"form": "text", "text": prompt},
            "sampling": {"temperature": 0.0, "max_tokens": 8},
            "dimensions": {
                "prompt_form": "text",
                "batch_shape": "generated-ragged",
                "generation_length": "multi-step",
            },
            "oracles": ["differential", "reverse-batch-order"],
            "kills": [
                "compare-only-decoded-text",
                "scheduler-order-used-as-public-order",
            ],
        }
        for index, prompt in enumerate(prompts)
    ]


def deterministic_image_case(*, placeholder: str = "<|image|>") -> dict:
    """Return one implementation-blind public image-input contract."""

    if not placeholder:
        raise ValueError("image placeholder must be non-empty")

    return {
        "case_id": "deterministic-rgb-image",
        "checklist_ids": ["P02", "P03", "P04", "P05", "P07"],
        "input": {
            "form": "text_with_image",
            "text": f"{placeholder}Describe the dominant color in one word.",
            "image": {
                "mode": "RGB",
                "size": [32, 32],
                "color": [17, 101, 203],
            },
        },
        "sampling": {"temperature": 0.0, "max_tokens": 8},
        "dimensions": {
            "prompt_form": "public-multimodal-dict",
            "image_fixture": "deterministic-solid-rgb",
            "generation_length": "multi-step",
        },
        "oracles": ["differential", "reverse-batch-order"],
        "kills": [
            "multimodal-wrapper-bypasses-monitored-decoder",
            "encoder-embedding-merge-changes-public-output",
        ],
    }
