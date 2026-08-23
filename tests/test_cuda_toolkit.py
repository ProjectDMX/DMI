from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu

_MODULE_PATH = Path(__file__).parents[1] / "native" / "cuda_toolkit.py"
_SPEC = importlib.util.spec_from_file_location("dmi_cuda_toolkit", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cuda_toolkit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cuda_toolkit
_SPEC.loader.exec_module(cuda_toolkit)


def _fake_toolkit(
    tmp_path: Path,
    name: str,
    version: str,
    *,
    target_layout: bool = False,
):
    root = tmp_path / name
    nvcc = root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(
        "#!/bin/sh\n"
        f"echo 'Cuda compilation tools, release {version}, V{version}.0'\n"
    )
    nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)

    target = f"{cuda_toolkit.platform.machine()}-linux"
    prefix = root / "targets" / target if target_layout else root
    include_dir = prefix / "include"
    library_dir = prefix / ("lib" if target_layout else "lib64")
    include_dir.mkdir(parents=True)
    library_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime.h").write_text("// test header\n")
    (library_dir / "libcudart.so").write_text("")
    return cuda_toolkit.inspect_toolkit(root, source=name)


@pytest.mark.parametrize("target_layout", [False, True])
def test_inspect_toolkit_keeps_one_root(tmp_path: Path, target_layout: bool) -> None:
    toolkit = _fake_toolkit(tmp_path, "cuda-13.0", "13.0", target_layout=target_layout)

    assert toolkit.version == (13, 0)
    assert toolkit.nvcc.is_relative_to(toolkit.root)
    assert toolkit.include_dir.is_relative_to(toolkit.root)
    assert toolkit.library_dir.is_relative_to(toolkit.root)


def test_auto_selection_prefers_unique_exact_version(tmp_path: Path) -> None:
    toolkits = [
        _fake_toolkit(tmp_path, "cuda-12.8", "12.8"),
        _fake_toolkit(tmp_path, "cuda-13.0", "13.0"),
        _fake_toolkit(tmp_path, "cuda-13.3", "13.3"),
    ]

    selected, warning = cuda_toolkit.choose_toolkit(
        toolkits, torch_cuda_version="13.0", explicit=False
    )

    assert selected.version == (13, 0)
    assert warning is None


def test_auto_selection_allows_one_same_major_with_warning(tmp_path: Path) -> None:
    toolkit = _fake_toolkit(tmp_path, "cuda-13.3", "13.3")

    selected, warning = cuda_toolkit.choose_toolkit(
        [toolkit], torch_cuda_version="13.0", explicit=False
    )

    assert selected == toolkit
    assert warning is not None
    assert "auto-selected CUDA 13.3" in warning


def test_auto_selection_rejects_ambiguous_same_major(tmp_path: Path) -> None:
    toolkits = [
        _fake_toolkit(tmp_path, "cuda-13.1", "13.1"),
        _fake_toolkit(tmp_path, "cuda-13.3", "13.3"),
    ]

    with pytest.raises(cuda_toolkit.ResolutionError, match="multiple CUDA 13.x"):
        cuda_toolkit.choose_toolkit(toolkits, torch_cuda_version="13.0", explicit=False)


def test_explicit_selection_rejects_major_mismatch(tmp_path: Path) -> None:
    toolkit = _fake_toolkit(tmp_path, "cuda-12.8", "12.8")

    with pytest.raises(cuda_toolkit.ResolutionError, match="major versions must match"):
        cuda_toolkit.choose_toolkit([toolkit], torch_cuda_version="13.0", explicit=True)


def test_resolver_skips_incomplete_candidates(tmp_path: Path) -> None:
    incomplete = tmp_path / "cuda-13.0-incomplete"
    (incomplete / "bin").mkdir(parents=True)
    nvcc = incomplete / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.0'\n")
    nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)
    complete = _fake_toolkit(tmp_path, "cuda-13.0", "13.0")

    selected, warning = cuda_toolkit.resolve_toolkit(
        torch_cuda_version="13.0",
        torch_cuda_home=None,
        candidates=[
            (incomplete, None, "incomplete"),
            (complete.root, None, "complete"),
        ],
    )

    assert selected.root == complete.root
    assert warning is None


def test_make_record_contains_coherent_paths(tmp_path: Path) -> None:
    toolkit = _fake_toolkit(tmp_path, "cuda-13.0", "13.0")
    torch_info = cuda_toolkit.TorchBuildInfo(
        cuda_version="13.0",
        cuda_home=None,
        include_dirs=("/torch/include", "/torch/api/include"),
        library_dirs=("/torch/lib",),
        cxx11_abi=1,
    )

    fields = cuda_toolkit.make_record(toolkit, torch_info).split("|")

    assert fields == [
        str(toolkit.root),
        str(toolkit.nvcc),
        str(toolkit.include_dir),
        str(toolkit.library_dir),
        "13.0",
        "13.0",
        "/torch/include;/torch/api/include",
        "/torch/lib",
        "1",
    ]


def test_link_check_accepts_only_expected_libcudart(
    monkeypatch, tmp_path: Path
) -> None:
    binary = tmp_path / "extension.so"
    binary.touch()
    output = "libcudart.so.13 => /cuda/lib64/libcudart.so.13 (0x1234)\n"
    monkeypatch.setattr(
        cuda_toolkit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    assert "libcudart.so.13" in cuda_toolkit.check_linked_cuda(
        binary, expected_major=13
    )


def test_link_check_rejects_mixed_libcudart(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "extension.so"
    binary.touch()
    output = (
        "libcudart.so.13 => /cuda-13/lib64/libcudart.so.13 (0x1234)\n"
        "libcudart.so.12 => /cuda-12/lib64/libcudart.so.12 (0x5678)"
    )
    monkeypatch.setattr(
        cuda_toolkit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    with pytest.raises(cuda_toolkit.ResolutionError, match="major\\(s\\) 12, 13"):
        cuda_toolkit.check_linked_cuda(binary, expected_major=13)
