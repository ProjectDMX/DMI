"""Checks that the Python hook catalog matches the compiled native ABI."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.native_backend


def test_python_hook_catalog_matches_native_extension():
    from dmi.hooks.catalog import HOOK_DEFS
    from dmi.transport.native import _load_extension

    try:
        native = _load_extension()
    except ImportError as exc:
        pytest.skip(str(exc))

    assert tuple(tuple(row) for row in native.HOOK_DEFS) == HOOK_DEFS
