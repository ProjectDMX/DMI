"""Resolve reproducible local paths for model repositories with subfolders."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


def _offline_mode() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def resolve_model_artifact(
    model_id: str,
    subfolder: str | None = None,
) -> str:
    """Return a loadable model path, resolving an HF repository subfolder.

    vLLM's offline ``LLM`` entrypoint does not expose Hugging Face's subfolder
    parameter. Keeping this resolution outside vLLM lets release cases name the
    immutable repository ID and subfolder instead of embedding one machine's
    snapshot hash.
    """

    if not subfolder:
        return model_id
    relative = PurePosixPath(subfolder)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("model subfolder must be a safe relative path")

    model_path = Path(model_id)
    if model_path.exists():
        root = model_path
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=model_id,
                local_files_only=_offline_mode(),
            )
        )
    resolved = root.joinpath(*relative.parts)
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(
            f"model subfolder has no config.json: {model_id}:{subfolder}"
        )
    return str(resolved)
