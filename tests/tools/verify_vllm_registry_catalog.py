#!/usr/bin/env python3
"""Verify that an exact official vLLM wheel can resolve a model catalog.

This is import/registry evidence only.  It does not approve DMI compatibility,
instantiate checkpoints, or replace the agent-authored API/model checklist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _lazy_target(entry: Any) -> str:
    return f"{entry.module_name}:{entry.class_name}"


def _class_target(model_cls: type[Any]) -> str:
    return f"{model_cls.__module__}:{model_cls.__name__}"


def _catalog_string(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def verify_catalog(catalog_path: Path) -> dict[str, Any]:
    import torch
    import vllm
    from vllm.model_executor.models import ModelRegistry

    catalog = json.loads(catalog_path.read_text())
    expected_version = catalog["target"]["vllm_version"]
    if vllm.__version__ != expected_version:
        raise RuntimeError(
            f"catalog requires vLLM {expected_version}, found {vllm.__version__}"
        )

    results: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for expected in catalog["architectures"]:
        architecture = _catalog_string(expected["architecture"])
        result = {"architecture": architecture}
        try:
            entry = ModelRegistry.models[architecture]
            registry_target = _lazy_target(entry)
            resolved_class = _class_target(entry.load_model_cls())
            result.update(
                registry_target=registry_target,
                resolved_class=resolved_class,
                status="resolved",
            )
            for field in ("registry_target", "resolved_class"):
                expected_value = _catalog_string(expected[field])
                if result[field] != expected_value:
                    raise AssertionError(
                        f"{architecture} {field}: expected {expected_value}, "
                        f"found {result[field]}"
                    )
        except BaseException as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failures.append(result)
        results.append(result)

    return {
        "schema_version": 1,
        "catalog": str(catalog_path.resolve()),
        "vllm_version": vllm.__version__,
        "vllm_path": str(Path(vllm.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "resolved": len(results) - len(failures),
        "failed": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    try:
        summary = verify_catalog(args.catalog)
    except BaseException as exc:
        print(f"catalog verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
