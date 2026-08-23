"""Resolve one coherent CUDA toolkit for the native DMI build.

The resolver deliberately chooses a toolkit root, never a collection of
unrelated include and library directories.  Explicit compiler/root overrides
win.  Automatic resolution is only accepted when the installed CUDA-enabled
PyTorch makes the choice unambiguous.
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import re
import shutil
import site
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_CUDA_VERSION_RE = re.compile(r"release\s+(\d+)\.(\d+)")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)")


class ResolutionError(RuntimeError):
    """Raised when a coherent CUDA toolkit cannot be selected."""


@dataclass(frozen=True)
class Toolkit:
    root: Path
    nvcc: Path
    include_dir: Path
    library_dir: Path
    version: tuple[int, int]
    source: str

    @property
    def version_text(self) -> str:
        return f"{self.version[0]}.{self.version[1]}"


@dataclass(frozen=True)
class TorchBuildInfo:
    cuda_version: str
    cuda_home: str | None
    include_dirs: tuple[str, ...]
    library_dirs: tuple[str, ...]
    cxx11_abi: int


def parse_version(value: str, *, label: str) -> tuple[int, int]:
    match = _VERSION_RE.search(value)
    if match is None:
        raise ResolutionError(f"cannot parse {label} CUDA version from {value!r}")
    return int(match.group(1)), int(match.group(2))


def _resolve_executable(value: str | Path) -> Path:
    raw = os.fspath(value)
    candidate = shutil.which(raw) if os.sep not in raw else raw
    if not candidate:
        raise ResolutionError(f"CUDA compiler not found: {raw}")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ResolutionError(f"CUDA compiler is not executable: {path}")
    return path


def _first_matching_dir(
    candidates: Iterable[Path], patterns: Sequence[str]
) -> Path | None:
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        if any(any(path.glob(pattern)) for pattern in patterns):
            return path
    return None


def inspect_toolkit(
    root: str | Path,
    *,
    source: str,
    nvcc: str | Path | None = None,
) -> Toolkit:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ResolutionError(f"{source}: toolkit root does not exist: {root_path}")

    nvcc_path = (
        _resolve_executable(nvcc)
        if nvcc is not None
        else _resolve_executable(root_path / "bin" / "nvcc")
    )
    try:
        output = subprocess.check_output(
            [str(nvcc_path), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResolutionError(f"{source}: cannot run {nvcc_path}: {exc}") from exc
    version_match = _CUDA_VERSION_RE.search(output)
    if version_match is None:
        raise ResolutionError(
            f"{source}: cannot parse CUDA version from {nvcc_path} --version"
        )
    version = int(version_match.group(1)), int(version_match.group(2))

    machine_target = f"{platform.machine()}-linux"
    target_roots = [root_path / "targets" / machine_target]
    target_roots.extend(
        Path(path) for path in sorted(glob.glob(str(root_path / "targets" / "*")))
    )
    include_dir = _first_matching_dir(
        [root_path / "include", *(path / "include" for path in target_roots)],
        ("cuda_runtime.h", "cuda_runtime_api.h"),
    )
    library_dir = _first_matching_dir(
        [
            root_path / "lib64",
            root_path / "lib",
            *(path / "lib" for path in target_roots),
            *(path / "lib64" for path in target_roots),
        ],
        ("libcudart.so",),
    )
    if include_dir is None or library_dir is None:
        missing = []
        if include_dir is None:
            missing.append("CUDA headers")
        if library_dir is None:
            missing.append("libcudart")
        raise ResolutionError(
            f"{source}: {root_path} is not a complete CUDA toolkit "
            f"(missing {', '.join(missing)})"
        )

    return Toolkit(
        root=root_path,
        nvcc=nvcc_path,
        include_dir=include_dir,
        library_dir=library_dir,
        version=version,
        source=source,
    )


def _compiler_root(value: str, *, label: str) -> tuple[Path, Path, str]:
    compiler = _resolve_executable(value)
    return compiler.parent.parent, compiler, label


def _explicit_selection(
    *,
    cudacxx: str | None,
    nvcc: str | None,
    toolkit_root: str | None,
    cmake_toolkit_root: str | None,
    cuda_home: str | None,
    cuda_path: str | None,
) -> tuple[Path, Path | None, str] | None:
    for value, label in ((cudacxx, "CUDACXX"), (nvcc, "NVCC")):
        if value:
            return _compiler_root(value, label=label)
    for value, label in (
        (toolkit_root, "CUDA_TOOLKIT_ROOT"),
        (cmake_toolkit_root, "CUDAToolkit_ROOT"),
        (cuda_home, "CUDA_HOME"),
        (cuda_path, "CUDA_PATH"),
    ):
        if value:
            return Path(value), None, label
    return None


def auto_candidate_roots(
    *,
    torch_cuda_home: str | None,
    python_prefix: str | Path = sys.prefix,
    nvcc_on_path: str | None = None,
    site_directories: Sequence[str] | None = None,
    system_roots: Sequence[str | Path] | None = None,
) -> list[tuple[Path, Path | None, str]]:
    candidates: list[tuple[Path, Path | None, str]] = []
    path_nvcc = nvcc_on_path if nvcc_on_path is not None else shutil.which("nvcc")
    if path_nvcc:
        compiler = _resolve_executable(path_nvcc)
        candidates.append((compiler.parent.parent, compiler, "PATH nvcc"))
    if torch_cuda_home:
        candidates.append((Path(torch_cuda_home), None, "PyTorch CUDA_HOME"))
    candidates.append((Path(python_prefix), None, "Python prefix"))

    roots = list(system_roots if system_roots is not None else ("/usr/local/cuda",))
    if system_roots is None:
        roots.extend(sorted(glob.glob("/usr/local/cuda-*")))
    candidates.extend((Path(root), None, "system toolkit") for root in roots)

    if site_directories is None:
        site_directories = [*site.getsitepackages()]
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            site_directories.append(user_site)
        else:
            site_directories.extend(user_site)
    for base in site_directories:
        for root in sorted(glob.glob(os.path.join(base, "nvidia", "cu*"))):
            candidates.append((Path(root), None, "Python NVIDIA toolkit"))

    deduplicated: list[tuple[Path, Path | None, str]] = []
    seen: set[Path] = set()
    for root, compiler, source in candidates:
        resolved = root.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduplicated.append((resolved, compiler, source))
    return deduplicated


def choose_toolkit(
    toolkits: Sequence[Toolkit],
    *,
    torch_cuda_version: str,
    explicit: bool,
) -> tuple[Toolkit, str | None]:
    torch_version = parse_version(torch_cuda_version, label="PyTorch")
    if not toolkits:
        raise ResolutionError("no complete CUDA toolkits were found")

    if explicit:
        selected = toolkits[0]
        if selected.version[0] != torch_version[0]:
            raise ResolutionError(
                f"{selected.source} selects CUDA {selected.version_text}, but "
                f"PyTorch was built for CUDA {torch_cuda_version}; CUDA major "
                "versions must match"
            )
        warning = None
        if selected.version != torch_version:
            warning = (
                f"warning: {selected.source} selects CUDA {selected.version_text} "
                f"while PyTorch was built for CUDA {torch_cuda_version}"
            )
        return selected, warning

    exact = [toolkit for toolkit in toolkits if toolkit.version == torch_version]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        roots = ", ".join(str(toolkit.root) for toolkit in exact)
        raise ResolutionError(
            f"multiple CUDA {torch_cuda_version} toolkits were found: {roots}; "
            "set CUDA_HOME or CUDACXX explicitly"
        )

    same_major = [
        toolkit for toolkit in toolkits if toolkit.version[0] == torch_version[0]
    ]
    if len(same_major) == 1:
        selected = same_major[0]
        return selected, (
            f"warning: auto-selected CUDA {selected.version_text} at "
            f"{selected.root}; PyTorch was built for CUDA {torch_cuda_version}"
        )
    if len(same_major) > 1:
        choices = ", ".join(
            f"{toolkit.root} ({toolkit.version_text})" for toolkit in same_major
        )
        raise ResolutionError(
            f"multiple CUDA {torch_version[0]}.x toolkits match PyTorch: "
            f"{choices}; set CUDA_HOME or CUDACXX explicitly"
        )

    choices = ", ".join(
        f"{toolkit.root} ({toolkit.version_text})" for toolkit in toolkits
    )
    raise ResolutionError(
        f"no CUDA toolkit matches PyTorch CUDA {torch_cuda_version}; found: "
        f"{choices}; set CUDA_HOME or CUDACXX to a matching toolkit"
    )


def resolve_toolkit(
    *,
    torch_cuda_version: str,
    torch_cuda_home: str | None,
    cudacxx: str | None = None,
    nvcc: str | None = None,
    toolkit_root: str | None = None,
    cmake_toolkit_root: str | None = None,
    cuda_home: str | None = None,
    cuda_path: str | None = None,
    candidates: Sequence[tuple[Path, Path | None, str]] | None = None,
) -> tuple[Toolkit, str | None]:
    explicit_selection = _explicit_selection(
        cudacxx=cudacxx,
        nvcc=nvcc,
        toolkit_root=toolkit_root,
        cmake_toolkit_root=cmake_toolkit_root,
        cuda_home=cuda_home,
        cuda_path=cuda_path,
    )
    if explicit_selection is not None:
        root, compiler, source = explicit_selection
        toolkit = inspect_toolkit(root, source=source, nvcc=compiler)
        return choose_toolkit(
            [toolkit], torch_cuda_version=torch_cuda_version, explicit=True
        )

    candidate_roots = list(
        candidates
        if candidates is not None
        else auto_candidate_roots(torch_cuda_home=torch_cuda_home)
    )
    toolkits: list[Toolkit] = []
    rejected: list[str] = []
    for root, compiler, source in candidate_roots:
        try:
            toolkits.append(inspect_toolkit(root, source=source, nvcc=compiler))
        except ResolutionError as exc:
            rejected.append(str(exc))
    try:
        return choose_toolkit(
            toolkits, torch_cuda_version=torch_cuda_version, explicit=False
        )
    except ResolutionError as exc:
        detail = "\n".join(f"  - {item}" for item in rejected)
        if detail:
            raise ResolutionError(f"{exc}\nRejected candidates:\n{detail}") from exc
        raise


def make_record(toolkit: Toolkit, torch_info: TorchBuildInfo) -> str:
    fields = (
        str(toolkit.root),
        str(toolkit.nvcc),
        str(toolkit.include_dir),
        str(toolkit.library_dir),
        toolkit.version_text,
        torch_info.cuda_version,
        ";".join(torch_info.include_dirs),
        ";".join(torch_info.library_dirs),
        str(torch_info.cxx11_abi),
    )
    scalar_fields = (*fields[:6], fields[8])
    if any(
        "|" in field or ";" in field or any(char.isspace() for char in field)
        for field in scalar_fields
    ):
        raise ResolutionError(
            "build paths containing whitespace, '|' or ';' are unsupported by "
            "the native Makefile"
        )
    if any(
        "|" in path or ";" in path or any(char.isspace() for char in path)
        for path in (*torch_info.include_dirs, *torch_info.library_dirs)
    ):
        raise ResolutionError(
            "build paths containing whitespace, '|' or ';' are unsupported by "
            "the native Makefile"
        )
    return "|".join(fields)


def check_linked_cuda(binary: str | Path, *, expected_major: int) -> str:
    path = Path(binary).resolve()
    try:
        result = subprocess.run(
            ["ldd", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        raise ResolutionError(
            f"cannot inspect linked libraries for {path}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise ResolutionError(f"ldd failed for {path}:\n{result.stdout}")
    missing = [
        line.strip() for line in result.stdout.splitlines() if "not found" in line
    ]
    if missing:
        raise ResolutionError(
            "native extension has unresolved libraries: " + "; ".join(missing)
        )
    majors = {
        int(value) for value in re.findall(r"libcudart\.so\.(\d+)", result.stdout)
    }
    if not majors:
        raise ResolutionError("native extension does not resolve libcudart")
    if majors != {expected_major}:
        rendered = ", ".join(str(value) for value in sorted(majors))
        raise ResolutionError(
            f"native extension resolves libcudart major(s) {rendered}; "
            f"expected only {expected_major}"
        )
    return next(
        line.strip()
        for line in result.stdout.splitlines()
        if re.search(rf"libcudart\.so\.{expected_major}\b", line)
    )


def write_if_changed(path: str | Path, content: str) -> None:
    destination = Path(path)
    payload = content.rstrip() + "\n"
    if destination.exists() and destination.read_text() == payload:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(destination)


def _torch_build_info() -> TorchBuildInfo:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths

    if torch.version.cuda is None:
        raise ResolutionError(
            "the active PyTorch build has no CUDA support; install a CUDA-enabled "
            "PyTorch build before compiling DMI"
        )
    return TorchBuildInfo(
        cuda_version=str(torch.version.cuda),
        cuda_home=CUDA_HOME,
        include_dirs=tuple(include_paths()),
        library_dirs=tuple(library_paths()),
        cxx11_abi=int(torch._C._GLIBCXX_USE_CXX11_ABI),
    )


def _add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cudacxx", default=os.environ.get("CUDACXX"))
    parser.add_argument("--nvcc", default=os.environ.get("NVCC"))
    parser.add_argument(
        "--cuda-toolkit-root", default=os.environ.get("CUDA_TOOLKIT_ROOT")
    )
    parser.add_argument(
        "--cmake-toolkit-root", default=os.environ.get("CUDAToolkit_ROOT")
    )
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME"))
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve")
    _add_selector_arguments(resolve_parser)
    resolve_parser.add_argument(
        "--format", choices=("record", "human"), default="human"
    )

    check_parser = subparsers.add_parser("check-link")
    check_parser.add_argument("--binary", required=True)
    check_parser.add_argument("--expected-major", required=True, type=int)

    stamp_parser = subparsers.add_parser("write-stamp")
    stamp_parser.add_argument("--output", required=True)
    stamp_parser.add_argument("--record", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "check-link":
            linked = check_linked_cuda(args.binary, expected_major=args.expected_major)
            print(f"CUDA link check passed: {linked}")
            return 0
        if args.command == "write-stamp":
            write_if_changed(args.output, args.record)
            return 0

        torch_info = _torch_build_info()
        toolkit, warning = resolve_toolkit(
            torch_cuda_version=torch_info.cuda_version,
            torch_cuda_home=torch_info.cuda_home,
            cudacxx=args.cudacxx,
            nvcc=args.nvcc,
            toolkit_root=args.cuda_toolkit_root,
            cmake_toolkit_root=args.cmake_toolkit_root,
            cuda_home=args.cuda_home,
            cuda_path=args.cuda_path,
        )
        if warning:
            print(warning, file=sys.stderr)
        if args.format == "record":
            print(make_record(toolkit, torch_info))
        else:
            print(f"CUDA toolkit root: {toolkit.root}")
            print(f"CUDA compiler: {toolkit.nvcc}")
            print(f"CUDA version: {toolkit.version_text}")
            print(f"PyTorch CUDA version: {torch_info.cuda_version}")
            print(f"CUDA include directory: {toolkit.include_dir}")
            print(f"CUDA library directory: {toolkit.library_dir}")
        return 0
    except ResolutionError as exc:
        print(f"CUDA toolkit resolution failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
