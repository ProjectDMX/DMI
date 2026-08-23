"""Package-layout and legacy-import compatibility checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def test_canonical_and_legacy_core_types_are_identical():
    from dmi.config import CaptureSchedule as CanonicalSchedule
    from dmi.engine import MonitoringEngine as CanonicalEngine
    from dmi.storage.internals import InternalRequirements as CanonicalRequirements
    from monitoring.config import CaptureSchedule as LegacySchedule
    from monitoring.engine import MonitoringEngine as LegacyEngine
    from monitoring.internal_mapper import InternalRequirements as LegacyRequirements

    assert CanonicalSchedule is LegacySchedule
    assert CanonicalEngine is LegacyEngine
    assert CanonicalRequirements is LegacyRequirements


def test_canonical_and_legacy_adapter_modules_share_state():
    from dmi.adapters.huggingface import adapter as canonical
    from integration import hf_adapter as legacy

    assert canonical is legacy
    assert canonical.HuggingFaceAdapter is canonical.HFAdaptor
    assert canonical._prepare_profile_times is legacy._prepare_profile_times


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
