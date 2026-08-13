"""CPU contracts for the version-pinned vLLM storage runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.vllm_compare_runner import _shutdown_llm


pytestmark = pytest.mark.cpu


def test_storage_runner_explicitly_shuts_down_engine_core() -> None:
    calls: list[float | None] = []
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(
                shutdown=lambda timeout=None: calls.append(timeout)
            )
        )
    )

    _shutdown_llm(llm, timeout=30.0)

    assert calls == [30.0]


def test_storage_runner_fails_closed_without_shutdown_contract() -> None:
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=object()))

    with pytest.raises(RuntimeError, match="no callable shutdown"):
        _shutdown_llm(llm)
