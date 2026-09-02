"""Structural guards on the configurator's no-build-step frontend.

These are deliberately weaker than unit tests, and the reason is a design
constraint rather than an oversight: ``docs/dmi-configurator-plan.md`` gates
the static UI on "no build process" and rules out a Node toolchain, so the
repository has no JavaScript test runner to execute ``app.js`` against.

What that leaves is a choice between no CI coverage at all and asserting on
the source text. These tests take the second option for invariants where a
silent regression would be expensive and invisible -- a dropped async guard
reintroduces a race that only shows up under load, and nothing else in CI
would notice.

They assert a mechanism is *present*, not that it *works*. The behaviour is
verified against a real browser during development; see the commit that
introduced each guard. If a JS runner is ever added, replace these with real
tests of the behaviour.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dmi.ui.app import STATIC_DIR

pytestmark = pytest.mark.cpu


def _app_js() -> str:
    return (Path(STATIC_DIR) / "app.js").read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """Return the text of one top-level ``async function`` in app.js.

    Brace-matched rather than regex-terminated, so a nested block inside the
    function cannot truncate the body and hide a missing guard.
    """
    start = source.index(f"async function {name}(")
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


# ---------------------------------------------------------------------------
# Stale-response guards on the two async render paths
# ---------------------------------------------------------------------------

RENDER_PATHS = ("refreshOutput", "refreshEstimate")


@pytest.mark.parametrize("function_name", RENDER_PATHS)
def test_the_render_path_stamps_its_requests(function_name):
    """Without a request id there is nothing to compare a late reply against."""
    body = _function_body(_app_js(), function_name)

    assert re.search(r"\bvar\s+requestId\s*=", body), (
        f"{function_name} does not capture a request id, so it cannot tell a "
        "stale response from the current one"
    )


@pytest.mark.parametrize("function_name", RENDER_PATHS)
def test_the_render_path_drops_a_superseded_response(function_name):
    """A later request must win; an earlier reply that lands after it is dead."""
    body = _function_body(_app_js(), function_name)

    assert re.search(r"requestId\s*!==\s*\w+RequestId", body), (
        f"{function_name} never compares its request id before rendering, so "
        "a slow earlier response can overwrite a newer one"
    )


@pytest.mark.parametrize("function_name", RENDER_PATHS)
def test_the_render_path_guards_its_failure_branch_too(function_name):
    """A stale *error* must not blank a panel the newer request just filled."""
    body = _function_body(_app_js(), function_name)
    catch_index = body.index("catch")

    assert re.search(r"requestId\s*!==\s*\w+RequestId", body[catch_index:]), (
        f"{function_name} guards its success path but not its catch block, so "
        "a late failure can still clobber current content"
    )


def test_every_repeatable_fetch_path_is_covered_by_these_guards():
    """Catch a new fetch-and-render path being added without a guard.

    The distinction is whether a path can overlap itself. ``refreshOutput``
    and ``refreshEstimate`` are fired from debounced ``setTimeout`` calls, so
    two can be in flight at once and a late reply can lose the race. ``boot``
    is registered once on ``DOMContentLoaded`` and never scheduled, so it
    cannot overlap and needs no guard.

    Rather than whitelisting names, this derives the exemption: a path is
    exempt only if the source never schedules or calls it outside an
    ``addEventListener`` registration.
    """
    source = _app_js()
    fetchers = {
        name
        for name in re.findall(r"async function (\w+)\(", source)
        # `api(` not `await api(`: refreshOutput awaits a Promise.all of two
        # api() calls, so the awaited token is Promise.all, not api.
        if "api(" in _function_body(source, name)
    }

    repeatable = set()
    for name in fetchers:
        scheduled = re.search(rf"setTimeout\(\s*{name}\b", source)
        called = re.search(rf"(?<!function )(?<!addEventListener\(\"DOMContentLoaded\", ){name}\(\)", source)
        if scheduled or called:
            repeatable.add(name)

    assert repeatable == set(RENDER_PATHS), (
        f"repeatable api() paths changed: {sorted(repeatable)} vs guarded "
        f"{sorted(RENDER_PATHS)}. Add a stale-response guard to any new one "
        "and list it in RENDER_PATHS."
    )
