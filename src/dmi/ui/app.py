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
    catalog_payload,
    config_to_dict,
    dump_config,
    load_config,
    load_descriptor,
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
    descriptor_path: Path
    save_path: Path
    initial_config: Optional[DMIConfig] = None


def default_save_path(descriptor_path: Path, descriptor: ModelDescriptor) -> Path:
    return descriptor_path.parent / f"{descriptor.model.id}.dmi.yaml"


def build_state(
    descriptor_path: str | Path, config_path: Optional[str | Path] = None
) -> UIState:
    """Load the descriptor and any starting configuration."""
    descriptor_path = Path(descriptor_path)
    descriptor = load_descriptor(descriptor_path)

    initial = None
    save_path = default_save_path(descriptor_path, descriptor)
    if config_path is not None:
        save_path = Path(config_path)
        if save_path.exists():
            initial = load_config(save_path)

    return UIState(
        descriptor=descriptor,
        descriptor_path=descriptor_path,
        save_path=save_path,
        initial_config=initial,
    )


def create_app(descriptor_path: str | Path, config_path: Optional[str | Path] = None):
    """Build the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised by install state
        raise RuntimeError(_FASTAPI_MISSING) from exc

    state = build_state(descriptor_path, config_path)

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
