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
    """Return the text of one top-level function in app.js.

    Brace-matched rather than regex-terminated, so a nested block inside the
    function cannot truncate the body and hide a missing guard.
    """
    start = re.search(rf"(?:async\s+)?function\s+{name}\s*\(", source)
    if start is None:
        raise AssertionError(f"no function {name} in app.js")
    open_brace = source.index("{", start.end())
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


# ---------------------------------------------------------------------------
# Blocking-adjacent findings on the client: policy-less YAML must stay
# policy-less through load/save (3919326729), and Copy must not export a
# stale preview (3919326715). Same text-level mechanism assertion as above:
# no JS runner exists in this repo, so presence is what CI can verify.
# ---------------------------------------------------------------------------


def test_apply_state_preserves_policy_absence():
    """`Object.assign(defaultState(), next)` keeps defaultState's balanced
    policy when next.policy is absent -- a policy-less file silently gained
    one, and saving it changed its meaning."""
    source = _app_js()

    assert not re.search(
        r"state\s*=\s*Object\.assign\(defaultState\(\),\s*next\)", source
    ), "applyState must not shallow-merge next over a default that carries a policy"


def test_apply_state_sets_policy_from_next_only():
    source = _app_js()
    body = _function_body(source, "applyState")

    assert re.search(r"state\.policy\s*=.*next\.policy", body), (
        "applyState must set policy from next explicitly (undefined stays "
        "undefined) rather than inheriting the default policy"
    )


def test_copy_serializes_the_current_state_not_the_preview():
    """Edits reach the preview only after debounce + request latency; a Copy
    that reads the preview can export the previous configuration."""
    source = _app_js()
    body = _function_body(source, "btnCopyHandler") if "btnCopyHandler" in source else None

    # The handler is inline in bindActions; find the copy listener.
    bind_actions = _function_body(source, "bindActions")
    copy_start = bind_actions.index('"btn-copy"')
    copy_slice = bind_actions[copy_start : copy_start + 900]

    assert re.search(r"serializeCurrentState\(\)", copy_slice), (
        "the Copy handler must serialize the current state server-side, not "
        "copy the possibly-stale preview text"
    )
    assert re.search(r"/api/config/serialize", source), (
        "the shared serialize helper must hit the serialize endpoint"
    )
    assert not re.search(
        r"writeText\(dom\[.yaml-preview.\]", copy_slice
    ), "the Copy handler must not read the preview element directly"


# ---------------------------------------------------------------------------
# Independent-review round: boot-race guard, Copy stale-response stamp,
# tab a11y state.
# ---------------------------------------------------------------------------


def test_input_handlers_guard_on_initialization():
    """Listeners bind before the model fetch resolves; an unguarded handler
    reads a null state and dies, silently eating the event (and a mid-
    applyState crash would discard a loaded file)."""
    source = _app_js()
    body = _function_body(source, "bindSchedule")

    assert "if (!uiReady) return;" in body
    assert re.search(r"var uiReady = false;", source)
    assert "uiReady = true;" in source


def test_copy_carries_a_stale_response_stamp():
    source = _app_js()
    bind_actions = _function_body(source, "bindActions")
    copy_slice = bind_actions[bind_actions.index('"btn-copy"'):]
    copy_slice = copy_slice[:copy_slice.index("btn-open")]

    assert "requestId" in copy_slice and "copyRequestId" in copy_slice, (
        "Copy must stamp its serialize call like refreshOutput does, or two "
        "rapid copies can land out of order and leave the older YAML on the "
        "clipboard"
    )


def test_tabs_expose_selection_to_assistive_tech():
    source = _app_js()
    bind_tabs = _function_body(source, "bindTabs")
    assert 'setAttribute("aria-selected"' in bind_tabs

    from dmi.ui.app import STATIC_DIR as _STATIC_DIR
    markup = (Path(STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert 'role="tabpanel"' in markup


# ---------------------------------------------------------------------------
# Frontend review round: save in-flight guard, tick keyboard path, estimate
# loading state, status live region, meter value semantics, curl-only mode
# banner. Same text-level mechanism convention as above: presence pins for
# invariants whose behavior is verified in a real browser.
# ---------------------------------------------------------------------------


def _bind_actions_body():
    return _function_body(_app_js(), "bindActions")


def test_save_disables_itself_while_the_request_is_in_flight():
    """Two rapid Saves fire two concurrent writes; the disk survives (atomic
    replace) but the toasts interleave. The button must go inert for the
    round trip."""
    body = _bind_actions_body()
    save_slice = body[body.index('"btn-save"'):]

    assert 'dom["btn-save"].disabled = true' in save_slice
    assert 'dom["btn-save"].disabled = false' in save_slice, (
        "the button must re-arm however the request settles, or one failed "
        "save bricks the control for the session"
    )


def test_layer_ticks_are_keyboard_operable():
    """The rail ticks are the only interactive elements without a keyboard
    path (architecture nodes already have role/tabindex/keydown). A keyboard
    user otherwise cannot select layers at all."""
    source = _app_js()

    assert 'setAttribute("role", "checkbox")' in source.replace(
        "'checkbox'", '"checkbox"'
    ), "ticks must expose checkbox semantics"
    assert "aria-checked" in source, "ticks must report selection state"
    assert re.search(r'tick\.setAttribute\("tabindex", "0"\)', source), (
        "ticks must be focusable"
    )
    assert re.search(
        r'layer-rail"\]\.addEventListener\("keydown"[\s\S]{0,600}?pickLayerFromTick',
        source,
    ), "ticks need a keydown path into the same layer-picking logic as click"
    assert re.search(
        r'layer-rail"\]\.addEventListener\("click"[\s\S]{0,600}?pickLayerFromTick',
        source,
    ), "click and keyboard must share one picking function, not two copies"


def test_estimate_shows_a_loading_state_while_in_flight():
    """refreshEstimate can take seconds (it walks every rank); the panel
    must not sit on stale figures with no indication."""
    body = _function_body(_app_js(), "refreshEstimate")

    assert re.search(r"[Ll]oading|estimating|aria-busy", body), (
        "refreshEstimate must mark the panel loading before the fetch and "
        "clear it on the guarded response"
    )


def test_status_chip_is_a_live_region():
    """'Valid / N issues / Error' updates are text-only; a screen reader
    never hears them without a live region."""
    markup = (Path(STATIC_DIR) / "index.html").read_text(encoding="utf-8")

    assert re.search(
        r'id="status"[^>]*aria-live="polite"', markup
    ) or re.search(r'aria-live="polite"[^>]*id="status"', markup), (
        "the status chip must announce its updates"
    )


def test_ring_meter_exposes_its_value():
    """role=img with a rewritten label has no value semantics; a meter with
    aria-valuenow does."""
    markup = (Path(STATIC_DIR) / "index.html").read_text(encoding="utf-8")

    assert 'role="meter"' in markup, "the occupancy meter needs meter semantics"
    body = _app_js()
    assert 'setAttribute("aria-valuenow"' in body, (
        "the meter value must track the rendered fill"
    )


def test_curl_only_mode_has_a_banner():
    """On a token-gated (network) bind every mutating request 401s by
    design, and the panels currently render the curl hint as config errors.
    A persistent banner must say the server is in curl-only mode."""
    markup = (Path(STATIC_DIR) / "index.html").read_text(encoding="utf-8")

    assert 'id="auth-banner"' in markup
    assert re.search(r'id="auth-banner"[^>]*hidden', markup), (
        "the banner must start hidden on loopback binds"
    )
    body = _app_js()
    assert re.search(r"401", body) and re.search(r"auth-banner", body), (
        "a 401 response must reveal the banner"
    )
