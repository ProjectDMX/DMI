"""Canonical package and clean source-tree layout checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def test_canonical_core_types_are_available():
    from dmi import CaptureSchedule, MonitoringEngine
    from dmi.storage.internals import InternalRequirements

    assert CaptureSchedule.__module__ == "dmi.config"
    assert MonitoringEngine.__module__ == "dmi.engine"
    assert InternalRequirements.__module__ == "dmi.storage.internals"


def test_adapter_exports_preferred_and_existing_class_names():
    from dmi.adapters.huggingface import HFAdaptor, HuggingFaceAdapter

    assert HuggingFaceAdapter is HFAdaptor


def test_source_tree_has_no_legacy_top_level_directories():
    repo_root = Path(__file__).resolve().parents[1]

    legacy_names = (
        "monitoring",
        "integration",
        "libs",
        "benchmark",
        "example",
        "Figures",
    )
    canonical_names = ("dmi", "native", "third_party", "benchmarks", "examples")

    assert all(not (repo_root / name).exists() for name in legacy_names)
    assert all((repo_root / name).is_dir() for name in canonical_names)


def test_plain_dmi_import_does_not_load_native_transport():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dmi; "
                "assert 'dmi.transport.native' not in sys.modules; "
                "assert 'dmi.transport.ring' not in sys.modules"
            ),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_hook_catalog_is_available_without_native_extension():
    from dmi.hooks.catalog import HOOK_DEFS

    ids = {row[0] for row in HOOK_DEFS}
    short_names = {row[2] for row in HOOK_DEFS}
    assert len(ids) == len(HOOK_DEFS)
    assert {"resid_pre", "q", "final_logits", "topk_weights"} <= short_names
