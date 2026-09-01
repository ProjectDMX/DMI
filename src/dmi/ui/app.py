"""FastAPI application for DMI-configurator.

Thin by design. Every endpoint delegates to :mod:`dmi.configuration`, so the
validity the browser reports is the validity DMI itself computes -- there is no
second implementation of the rules in JavaScript.

Localhost-only, no database, no authentication, no websockets. The server holds
one descriptor and one optional configuration path for its lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..configuration import (
    ConfigurationError,
    DMIConfig,
    ModelDescriptor,
    Workload,
    catalog_payload,
    check_ring_fit,
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

    initial = None
    save_path = default_save_path(source, descriptor)
    if config_path is not None:
        save_path = Path(config_path)
        if save_path.exists():
            initial = load_config(save_path)

    return UIState(
        descriptor=descriptor,
        source=str(source),
        save_path=save_path,
        initial_config=initial,
    )


def create_app(source: str | Path, config_path: Optional[str | Path] = None):
    """Build the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised by install state
        raise RuntimeError(_FASTAPI_MISSING) from exc

    state = build_state(source, config_path)

    app = FastAPI(title="DMI-configurator", docs_url=None, redoc_url=None)
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
        try:
            workload = Workload(**(payload.get("workload") or {}))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"Invalid workload: {exc}") from exc

        try:
            estimate = estimate_config(config, state.descriptor, workload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        body = estimate_payload(estimate)

        ring = payload.get("ring") or {}
        payload_bytes = ring.get("payload_bytes")
        if payload_bytes:
            try:
                fit = check_ring_fit(
                    estimate,
                    int(payload_bytes),
                    int(ring.get("pinned_bytes") or 0),
                )
            except (TypeError, ValueError) as exc:
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
        try:
            save_config(config, state.save_path)
        except OSError as exc:
            raise HTTPException(500, f"Could not write {state.save_path}: {exc}") from exc
        return {"path": str(state.save_path)}

    return app


__all__ = ["UIState", "build_state", "create_app", "default_save_path", "STATIC_DIR"]
