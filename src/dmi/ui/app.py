"""FastAPI application for DMI-configurator.

Thin by design. Every endpoint delegates to :mod:`dmi.configuration`, so the
validity the browser reports is the validity DMI itself computes -- there is no
second implementation of the rules in JavaScript.

Localhost-only, no database, no authentication, no websockets. The server holds
one descriptor and one optional configuration path for its lifetime.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..configuration import (
    ConfigValidationError,
    ConfigurationError,
    DMIConfig,
    ModelDescriptor,
    Workload,
    catalog_payload,
    check_ring_fit,
    ensure_valid,
    resolve_descriptor,
    config_to_dict,
    dump_config,
    estimate_config,
    estimate_payload,
    load_config,
    parse_config,
    save_config,
    validate_config,
)
from ..configuration.architecture import model_payload

STATIC_DIR = Path(__file__).parent / "static"

_FASTAPI_MISSING = (
    "DMI-configurator needs the optional UI dependencies. Install them with:\n"
    '    pip install "DMI[ui]"'
)


@dataclass
class UIState:
    """Everything the server knows, fixed at startup.

    ``save_path`` is server-side on purpose: the browser never supplies a
    filesystem path, so a page cannot write outside the location the user named
    on the command line.
    """

    descriptor: ModelDescriptor
    source: str
    save_path: Path
    initial_config: Optional[DMIConfig] = None


def default_save_path(source: str | Path, descriptor: ModelDescriptor) -> Path:
    """Put the configuration beside its source, or in the CWD for a model id."""
    target = Path(source)
    if target.is_dir():
        base = target
    elif target.exists():
        base = target.parent
    else:
        base = Path.cwd()  # a bare Hugging Face model id has no local directory
    return base / f"{descriptor.model.id}.dmi.yaml"


def build_state(
    source: str | Path, config_path: Optional[str | Path] = None
) -> UIState:
    """Load the model description and any starting configuration.

    ``source`` is anything :func:`resolve_descriptor` accepts: a DMI descriptor
    YAML, a model directory, a ``config.json``, or a Hugging Face model id.
    """
    descriptor = resolve_descriptor(source)

    # The save path doubles as the startup config: relaunching the
    # configurator picks up the file the previous session saved, whether or
    # not --config named it explicitly, so Save never silently clobbers an
    # authored configuration with a blank default.
    save_path = default_save_path(source, descriptor)
    if config_path is not None:
        save_path = Path(config_path)
    initial = load_config(save_path) if save_path.exists() else None

    return UIState(
        descriptor=descriptor,
        source=str(source),
        save_path=save_path,
        initial_config=initial,
    )


# Host names a browser legitimately reaches a loopback bind under. Anything
# else in the Host header means the request came through a name we never
# served -- the DNS-rebinding pattern -- and is rejected before any endpoint
# runs.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")


def create_app(
    source: str | Path,
    config_path: Optional[str | Path] = None,
    bind_host: str = "127.0.0.1",
) -> Any:
    """Build the FastAPI application.

    ``bind_host`` is the address the server will listen on. For a loopback
    bind (the default), requests whose ``Host`` header is not a loopback name
    are rejected: a DNS-rebinding page resolves its own hostname to 127.0.0.1
    and would otherwise be same-origin for the file-writing endpoints. A
    non-loopback bind is an explicit choice to serve the network, and a Host
    allowlist is not authentication -- so the app then mints a per-launch
    token and demands it on every non-read endpoint. The token is returned
    alongside the app so the server can print it exactly once; without it,
    any reachable peer could overwrite the configured file as the server user.

    The token, if any, is on ``app.state.launch_token`` for the server to
    print exactly once at startup; ``None`` for a loopback bind.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except ImportError as exc:  # pragma: no cover - exercised by install state
        raise RuntimeError(_FASTAPI_MISSING) from exc

    state = build_state(source, config_path)

    app = FastAPI(title="DMI-configurator", docs_url=None, redoc_url=None)
    if bind_host in _LOOPBACK_HOSTS:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=list(_LOOPBACK_HOSTS)
        )
        app.state.launch_token = None

        # Cross-site write defense: TrustedHostMiddleware stops DNS
        # rebinding, but a page at https://evil.example POSTing directly to
        # 127.0.0.1 sends Host: 127.0.0.1 -- which IS allowlisted -- and
        # navigator.sendBeacon can even send application/json without a
        # preflight. A browser always attaches Origin on cross-site
        # requests; curl and same-origin fetches legitimately omit it. So:
        # a mutating request that NAMES a foreign origin is refused.
        @app.middleware("http")
        async def _reject_cross_site_writes(request: Request, call_next):
            if request.method in ("GET", "HEAD", "OPTIONS"):
                return await call_next(request)
            origin = request.headers.get("Origin")
            if origin is not None:
                hostname = origin.split("://", 1)[-1].split("/", 1)[0]
                host = hostname.rsplit(":", 1)[0].strip("[]")
                if host not in ("127.0.0.1", "localhost", "::1"):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Cross-site configuration writes "
                                 "are refused by this local server."},
                    )
            return await call_next(request)
    else:
        app.state.launch_token = secrets.token_urlsafe(24)

        @app.middleware("http")
        async def _require_launch_token(request: Request, call_next):
            # Read-only surface stays open (the landing page, the model and
            # catalog descriptions) EXCEPT /api/config, whose response names
            # a server filesystem path and carries the starting config --
            # metadata a LAN peer has no business reading. Everything else
            # that is not a static asset demands the token; the browser
            # itself does not send it -- a network deployment is driven with
            # curl or a reverse proxy that injects the header.
            open_paths = ("/", "/api/model", "/api/catalog")
            if (
                request.method in ("GET", "HEAD", "OPTIONS")
                and request.url.path in open_paths
            ):
                return await call_next(request)
            if request.url.path.startswith("/static"):
                return await call_next(request)
            supplied = request.headers.get("X-DMI-Token", "").encode("utf-8", "replace")
            if not secrets.compare_digest(
                supplied, str(app.state.launch_token).encode("utf-8")
            ):
                # Raised in the middleware, so returned as a response directly:
                # an HTTPException raised inside BaseHTTPMiddleware dispatch
                # escapes FastAPI's exception handlers.
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": (
                            "This server is bound beyond loopback, so mutating "
                            "requests need the launch token: "
                            "curl -H 'X-DMI-Token: <token from startup>' ..."
                        )
                    },
                )
            return await call_next(request)

    app.state.ui = state
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _parse(payload: Any) -> DMIConfig:
        if not isinstance(payload, dict) or "config" not in payload:
            raise HTTPException(400, "Request body must be {'config': {...}}.")
        try:
            return parse_config(payload["config"])
        except ConfigurationError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/model")
    def get_model():
        return model_payload(state.descriptor)

    @app.get("/api/catalog")
    def get_catalog():
        return catalog_payload(state.descriptor.topology)

    @app.get("/api/config")
    def get_config():
        """The starting configuration, if one was passed on the command line."""
        if state.initial_config is None:
            return {"config": None, "path": str(state.save_path)}
        return {
            "config": config_to_dict(state.initial_config),
            "path": str(state.save_path),
        }

    @app.post("/api/validate")
    def post_validate(payload: dict):
        config = _parse(payload)
        issues = validate_config(config, state.descriptor)
        return {
            "valid": not any(issue.is_error for issue in issues),
            "issues": [issue.to_dict() for issue in issues],
        }

    @app.post("/api/estimate")
    def post_estimate(payload: dict):
        """Byte estimates for a configuration under a stated workload.

        The workload is supplied per request rather than held in server state:
        it describes the traffic the capture rides along with, not the capture,
        so the same configuration legitimately has several answers.
        """
        config = _parse(payload)
        workload_raw = payload.get("workload")
        # Type-check before defaulting: `or {}` alone would let a falsy
        # non-mapping such as `workload: []` slip through as "absent" instead
        # of being reported as the malformed body it is.
        if workload_raw is None:
            workload_raw = {}
        if not isinstance(workload_raw, dict):
            raise HTTPException(
                400,
                f"'workload' must be a mapping, got {type(workload_raw).__name__}.",
            )
        try:
            workload = Workload(**workload_raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"Invalid workload: {exc}") from exc

        try:
            estimate = estimate_config(config, state.descriptor, workload)
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(400, str(exc)) from exc

        body = estimate_payload(estimate)

        # Type-check the raw value: `or {}` alone would let a truthy
        # non-mapping such as a list or string through to `.get`, which raises
        # AttributeError and surfaces as a 500 instead of a 400.
        ring_raw = payload.get("ring")
        if ring_raw is not None and not isinstance(ring_raw, dict):
            raise HTTPException(
                400,
                f"'ring' must be a mapping, got {type(ring_raw).__name__}.",
            )
        ring = ring_raw or {}
        # `is not None`, not truthiness: payload_bytes=0 is a client error that
        # check_ring_fit rejects with a clear message, and treating it as
        # "absent" would silently drop the ring-fit result the caller asked
        # for.
        payload_bytes = ring.get("payload_bytes")
        if payload_bytes is None and ring.get("pinned_bytes") is not None:
            raise HTTPException(
                400,
                "'ring.pinned_bytes' was given without 'ring.payload_bytes': "
                "a fit verdict needs the payload ring to compare against.",
            )
        if payload_bytes is not None:
            if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int):
                raise HTTPException(
                    400,
                    f"'ring.payload_bytes' must be an integer, got "
                    f"{type(payload_bytes).__name__}.",
                )
            pinned_bytes = ring.get("pinned_bytes") or 0
            if isinstance(pinned_bytes, bool) or not isinstance(pinned_bytes, int):
                raise HTTPException(
                    400,
                    f"'ring.pinned_bytes' must be an integer, got "
                    f"{type(pinned_bytes).__name__}.",
                )
            try:
                task_entries = ring.get("task_entries")
                if task_entries is not None and (
                    isinstance(task_entries, bool)
                    or not isinstance(task_entries, int)
                    or task_entries < 1
                ):
                    raise HTTPException(
                        400,
                        f"'ring.task_entries' must be a positive integer, got "
                        f"{task_entries!r}.",
                    )
                fit = check_ring_fit(
                    estimate,
                    payload_bytes,
                    pinned_bytes,
                    task_entries=task_entries,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(400, f"Invalid ring sizes: {exc}") from exc
            body["ring_fit"] = {
                "effective_bytes": fit.effective_bytes,
                "peak_step_bytes": fit.peak_step_bytes,
                "fits": fit.fits,
                "occupancy_percent": fit.occupancy_percent,
                "detail": fit.detail,
            }
        return body

    @app.post("/api/config/serialize")
    def post_serialize(payload: dict):
        config = _parse(payload)
        return {"yaml": dump_config(config)}

    @app.post("/api/config/parse")
    def post_parse(payload: dict):
        if not isinstance(payload, dict) or "yaml" not in payload:
            raise HTTPException(400, "Request body must be {'yaml': '...'}.")
        if not isinstance(payload["yaml"], str):
            raise HTTPException(
                400, f"'yaml' must be a string, got {type(payload['yaml']).__name__}."
            )
        import yaml as _yaml

        try:
            document = _yaml.safe_load(payload["yaml"])
            config = parse_config(document)
        except _yaml.YAMLError as exc:
            raise HTTPException(400, f"Not valid YAML: {exc}") from exc
        except ConfigurationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"config": config_to_dict(config)}

    @app.post("/api/config/save")
    def post_save(payload: dict):
        """Write to the server-side path. The client cannot choose it."""
        config = _parse(payload)
        # Model-aware validation before the write: the file this produces is
        # the tool's output contract, so an empty hook set or an out-of-range
        # layer must be refused here, not persisted and caught by the runtime.
        try:
            ensure_valid(config, state.descriptor)
        except ConfigValidationError as exc:
            raise HTTPException(
                400,
                "; ".join(issue.message for issue in exc.issues),
            ) from exc
        try:
            save_config(config, state.save_path)
        except ConfigurationError as exc:
            # save_config reports filesystem trouble (OSError included) as
            # ConfigurationError, so that is what actually arrives here.
            raise HTTPException(500, f"Could not write {state.save_path}: {exc}") from exc
        state.initial_config = config
        return {"path": str(state.save_path)}

    return app


__all__ = ["UIState", "build_state", "create_app", "default_save_path", "STATIC_DIR"]
