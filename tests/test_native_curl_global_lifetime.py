"""libcurl is initialised once per process and never torn down by an object.

``curl_global_init``/``curl_global_cleanup`` are PROCESS-global, and libcurl
documents both as unsafe to call while another thread is inside the library.
``ClickHouseClient`` used to call the init in its constructor and the cleanup
in its **destructor**, so destroying one client dropped the library's refcount
for everything else in the process -- including ``SpoolUploader``'s worker
threads, which run up to ``max_workers`` concurrent ``curl_easy_perform``
calls, and ``S3Client``, which called no init of its own and relied on the
implicit one inside ``curl_easy_init``.

There is deliberately NO red test for the failure itself: reproducing it
needs a data race between a destructor on one thread and ``curl_easy_perform``
on another, and a race is not something to pin in CI -- a test that reproduces
it sometimes is a flake, and one that reproduces it never is decorative. What
is pinned instead is the invariant the fix establishes, in the two places it
is observable without a race:

  * the SOURCE invariant -- no per-object teardown exists anywhere in the
    native tree, and the one init lives in the one shared helper. This half
    fails on the pre-fix tree, at clickhouse_client.cpp's destructor.
  * the LIFETIME behaviour -- clients constructed and destroyed in sequence
    leave libcurl usable for the next one. Single-threaded, libcurl's own
    refcounting made this hold before the fix too, so it is a guard rather
    than a reproduction; it is what would break first if the teardown came
    back in a form the source scan did not recognise.

Build: make -C native build/conformance_catalog
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSRC = REPO_ROOT / "native" / "csrc"
DRIVER = REPO_ROOT / "native" / "build" / "conformance_catalog"

# The single home for the process-lifetime init. Anywhere else is a client
# initialising (and, historically, tearing down) libcurl on its own schedule.
INIT_HOME = CSRC / "common" / "curl_init.cpp"

pytestmark = pytest.mark.cpu


# Comments are stripped before scanning: these files DOCUMENT why the global
# calls are absent, and a prose mention is not a call site. Crude but exact
# enough for this -- no string literal in the tree contains "//" or "/*".
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _native_sources() -> list[Path]:
    return sorted(
        path
        for suffix in ("*.cpp", "*.h", "*.cu")
        for path in CSRC.rglob(suffix)
    )


def _code_of(path: Path) -> str:
    return _COMMENT.sub("", path.read_text(encoding="utf-8"))


def test_no_native_source_tears_down_libcurl_globally():
    """``curl_global_cleanup`` has no correct caller in a threaded process."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _native_sources()
        if "curl_global_cleanup" in _code_of(path)
    ]
    assert not offenders, (
        "curl_global_cleanup() is process-global and unsafe while another "
        "thread is inside libcurl; the uploader's workers are. Found in: "
        + ", ".join(offenders)
    )


def test_the_global_init_lives_in_exactly_one_place():
    """One init, behind one ``std::once_flag``, shared by every client."""
    callers = [
        str(path.relative_to(REPO_ROOT))
        for path in _native_sources()
        if "curl_global_init" in _code_of(path)
    ]
    assert callers == [str(INIT_HOME.relative_to(REPO_ROOT))], (
        "curl_global_init() belongs to the shared once-only helper in "
        f"{INIT_HOME.relative_to(REPO_ROOT)}; clients call "
        "dmi_common::EnsureCurlGlobalInit(). Found in: " + ", ".join(callers)
    )

    assert "std::call_once" in _code_of(INIT_HOME), (
        "the shared init must be once-only: a second curl_global_init is "
        "the same re-initialisation race the per-object version had"
    )


@pytest.mark.skipif(
    not DRIVER.exists(),
    reason="native/build/conformance_catalog is not built; run "
    "`make -C native build/conformance_catalog`",
)
def test_libcurl_survives_clients_constructed_and_destroyed_in_sequence():
    """Three clients in one process: the third still reaches libcurl.

    Each ``open`` builds a fresh ``ClickHouseClient`` and destroys the one
    before it, which is where the teardown used to run. The final ``execute``
    dials a port nothing listens on, so no server is needed: what it proves is
    that libcurl answered at all -- a clean, named ``ClickHouseError`` from
    ``curl_easy_perform`` rather than a crash or a hang inside a library some
    earlier destructor had dismantled.
    """
    request = {
        "op": "open",
        "database": "curl_lifetime_db",
        "table_prefix": "curl_lifetime",
        "lease_ttl_ns": 30_000_000_000,
        "publish_timeout_ns": 1_000_000_000,
        "clock_skew_ns": 1_000_000,
        "allocation_attempts": 3,
    }
    lines = [json.dumps(request)] * 3
    lines.append(json.dumps({"op": "execute", "query": "SELECT 1"}))

    env = dict(os.environ)
    env["DMI_CLICKHOUSE_HOST"] = "127.0.0.1"
    # Port 1: privileged and unbound, so the connect is refused immediately
    # rather than waiting out a timeout, and an operator's real ClickHouse on
    # 8123 cannot turn this into a live test by accident.
    env["DMI_CLICKHOUSE_HTTP_PORT"] = "1"

    proc = subprocess.run(
        [str(DRIVER)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    answers = [
        json.loads(line) for line in proc.stdout.splitlines() if line.strip()
    ]
    assert proc.returncode == 0, proc.stderr
    assert len(answers) == 4, proc.stdout + proc.stderr
    for answer in answers[:3]:
        assert answer == {"op": "open", "ok": True}
    # libcurl was reached and reported for itself; the transport error is the
    # closed port, not a dismantled library.
    assert answers[3]["ok"] is False
    assert answers[3]["error"] == "ClickHouseError"
    assert answers[3]["message"].startswith("curl: ")
