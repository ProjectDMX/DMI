"""CPU contracts for the shared-machine GPU idle gate."""

from __future__ import annotations

import pytest

from tests.tools import check_gpu_idle


pytestmark = pytest.mark.cpu


def test_idle_sample_accepts_only_selected_idle_gpu(monkeypatch):
    def fake_query(*fields, compute=False):
        if compute:
            return [["uuid-1", "123", "python"]]
        return [
            ["0", "uuid-0", "24", "0"],
            ["1", "uuid-1", "2048", "99"],
        ]

    monkeypatch.setattr(check_gpu_idle, "_nvidia_smi", fake_query)

    assert check_gpu_idle._sample({0}, 1024, 10) == []


def test_idle_sample_reports_process_memory_and_utilization(monkeypatch):
    def fake_query(*fields, compute=False):
        if compute:
            return [["uuid-1", "123", "python worker.py"]]
        return [["1", "uuid-1", "1024", "10"]]

    monkeypatch.setattr(check_gpu_idle, "_nvidia_smi", fake_query)

    assert check_gpu_idle._sample({1}, 1024, 10) == [
        "GPU 1 has compute processes: pid=123 python worker.py",
        "GPU 1 uses 1024 MiB (limit < 1024 MiB)",
        "GPU 1 utilization is 10% (limit < 10%)",
    ]


def test_idle_sample_rejects_unknown_gpu(monkeypatch):
    monkeypatch.setattr(
        check_gpu_idle,
        "_nvidia_smi",
        lambda *fields, compute=False: [["0", "uuid-0", "24", "0"]],
    )

    assert check_gpu_idle._sample({2}, 1024, 10) == ["unknown GPU indices: [2]"]
