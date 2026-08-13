"""CPU contracts for reproducible model-subfolder resolution."""

from pathlib import Path

import pytest

from tests.model_artifacts import resolve_model_artifact


pytestmark = pytest.mark.cpu


def test_model_artifact_keeps_plain_model_ids_unchanged() -> None:
    assert resolve_model_artifact("publisher/model") == "publisher/model"


def test_model_artifact_resolves_checked_local_subfolder(tmp_path: Path) -> None:
    subfolder = tmp_path / "weights"
    subfolder.mkdir()
    (subfolder / "config.json").write_text("{}\n")

    assert resolve_model_artifact(str(tmp_path), "weights") == str(subfolder)


@pytest.mark.parametrize("subfolder", ("../escape", "/absolute"))
def test_model_artifact_rejects_unsafe_subfolders(subfolder: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        resolve_model_artifact("publisher/model", subfolder)


def test_model_artifact_requires_a_model_config(tmp_path: Path) -> None:
    (tmp_path / "weights").mkdir()

    with pytest.raises(FileNotFoundError, match="config.json"):
        resolve_model_artifact(str(tmp_path), "weights")
