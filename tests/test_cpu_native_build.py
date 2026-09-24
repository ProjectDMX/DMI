from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import get_type_hints

import pytest


@pytest.mark.cpu
def test_host_build_plan_has_no_cuda_toolchain_or_libraries():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-C", "native", "-B", "-n", "host"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "_host_backend" in output
    if sys.platform == "darwin":
        assert "-undefined,dynamic_lookup" in output
    for forbidden in ("nvcc", "-lcuda", "-lcudart", "-lc10_cuda", "-ltorch_cuda"):
        assert forbidden not in output


def _torch_include_flags() -> set[str]:
    """The ``-I`` flags that put torch's headers on a compile's search path."""
    from torch.utils.cpp_extension import include_paths

    flags = set()
    for path in include_paths():
        flags.add(f"-I{path}")
        flags.add(f"-I{os.path.realpath(path)}")
    return flags


def _torch_compile_lines(makefile_dir: str, target: str) -> list[str]:
    """The compiler commands a dry run of ``target`` would execute that pull
    in torch's headers.

    A command is selected by the torch include path on its command line, not
    by a macro only some recipes define: nvcc's flags never set
    TORCH_EXTENSION_NAME, so keying on it left every .cu compile unchecked.
    The flag is matched as a whole token, so the CUDA resolver's stamp
    record, which carries the same directories inside one quoted string, is
    not mistaken for a compile.
    """
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        # PYTHON is the interpreter running this test: its torch is the one
        # the build would target, and the Makefile's default `python` need
        # not exist.
        ["make", "-C", makefile_dir, "-B", "-n", target, f"PYTHON={sys.executable}"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot plan `{target}` here: {result.stderr[-300:]}")
    torch_flags = _torch_include_flags()
    return [line for line in result.stdout.splitlines()
            if torch_flags.intersection(line.split())]


@pytest.mark.parametrize(("makefile_dir", "target"), [
    # host plans on any machine, so CI's cpu job checks it.
    pytest.param("native", "host", marks=pytest.mark.cpu, id="host"),
    # The full backend needs the CUDA toolchain even to be PLANNED, which a
    # cpu runner does not have. It is a gpu test, not a cpu test that skips:
    # the cpu gate rightly fails any skip that is not absent hardware.
    pytest.param("native", "all", marks=pytest.mark.gpu, id="all"),
    # The ring test binaries that link torch compile against the same
    # headers, from their own Makefile and its own CXX_STD default. They
    # need the CUDA resolver to be planned, so they are gpu as well.
    pytest.param("tests/native/ring", "all", marks=pytest.mark.gpu,
                 id="ring-tests"),
])
def test_every_torch_including_compile_requests_cxx20(makefile_dir, target):
    """PyTorch's headers refuse anything older than C++20.

    ATen.h opens with ``#error C++20 or later compatible compiler is required``,
    and the pinned range here (torch>=2.8,<3) resolves to a release that
    enforces it. The backend and host targets were still compiled with
    -std=c++17, so a fresh install could not build either one -- and CI never
    noticed, because it only dry-runs `host` and the torch-free drivers never
    reach ATen. This pins the flag on the plan actually executed rather than on
    the Makefile's text, so it holds however the flags are assembled, and it
    covers g++ and nvcc compiles alike.

    Only ``host`` is marked cpu. Planning ``all`` resolves the CUDA toolkit and
    fails without one, so it runs under the gpu marker instead.
    """
    lines = _torch_compile_lines(makefile_dir, target)
    assert lines, f"no torch-including compile found in the `{target}` plan"
    stale = [line for line in lines
             if not any(f"-std={std}" in line
                        for std in ("c++20", "c++23", "c++26", "gnu++20", "gnu++23"))]
    assert not stale, (
        f"{len(stale)} of {len(lines)} torch-including compile(s) below C++20 "
        f"in `{makefile_dir}` `{target}`:\n" + stale[0][:300])


@pytest.mark.cpu
def test_host_export_falls_back_to_cpu_backend(monkeypatch):
    from dmi.transport import native

    sentinel = object()
    calls = []

    def load_named(name):
        calls.append(name)
        if name == "_native_backend":
            raise ImportError("full backend absent")
        return SimpleNamespace(DMXHostEngine=sentinel)

    monkeypatch.setattr(native, "_load_named_extension", load_named)
    monkeypatch.setattr(native, "_EXTENSION_MODULES", {})

    assert native.DMXHostEngine is sentinel
    assert calls == ["_native_backend", "_host_backend"]


@pytest.mark.native_backend
def test_clickhouse_stage_has_bounded_batching_defaults():
    from dmi.transport.native import ClickHouseClientConfig, StageConfig

    stage = StageConfig.clickhouse_insert(ClickHouseClientConfig())
    queue = stage.input_queue

    assert queue.min_batch_items is None
    assert queue.min_batch_size == 16 * 1024**2
    assert queue.max_linger_s == pytest.approx(0.05)
    assert queue.max_batch_items == 10_000
    assert queue.max_batch_size is None
    assert queue.high_watermark_items == 20_000
    assert queue.high_watermark_size == 512 * 1024**2


@pytest.mark.native_backend
def test_schema_driven_stage_is_additive_and_uses_bounded_batching_defaults():
    from dmi.api.v1 import RecordCellType, RecordColumn, RecordLayout, RecordSchema
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    schema = RecordSchema(
        (
            RecordLayout(
                name="event",
                table="event_records",
                columns=(RecordColumn("event_id", RecordCellType.INT64),),
                primary_key=("event_id",),
                order_by=("event_id",),
            ),
        )
    )
    config = ClickHouseClientConfig()
    assert not hasattr(config, "record_schema")

    stage = StageConfig.clickhouse_records(config, schema)
    queue = stage.input_queue
    assert stage.name == "clickhouse_records"
    assert queue.min_batch_items is None
    assert queue.min_batch_size == 16 * 1024**2
    assert queue.max_linger_s == pytest.approx(0.05)
    assert queue.max_batch_items == 10_000
    assert queue.high_watermark_items == 20_000
    assert queue.high_watermark_size == 512 * 1024**2

    engine = DMXHostEngine(stage)
    assert callable(engine.submit_record)
    assert callable(engine.flush_and_wait)


@pytest.mark.native_backend
def test_record_host_schema_identity_is_exact_but_layout_order_is_irrelevant():
    from dmi.api.v1 import RecordCellType, RecordColumn, RecordLayout, RecordSchema
    from dmi.transport import native
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    event_layout = RecordLayout(
        name="event",
        table="event_records",
        columns=(
            RecordColumn("run", RecordCellType.STRING),
            RecordColumn("event_id", RecordCellType.INT64),
            RecordColumn("score", RecordCellType.FLOAT64),
        ),
        primary_key=("run", "event_id"),
        order_by=("run", "event_id"),
    )
    tensor_layout = RecordLayout(
        name="tensor",
        table="tensor_records",
        columns=(
            RecordColumn("run", RecordCellType.STRING),
            RecordColumn(
                "payload",
                RecordCellType.TENSOR,
                dtype_column="payload_dtype",
                shape_column="payload_shape",
                bytes_column="payload_bytes",
            ),
        ),
        primary_key=("run",),
        order_by=("run",),
    )
    schema = RecordSchema((event_layout, tensor_layout), index_granularity=1024)
    reordered_schema = RecordSchema(
        (tensor_layout, event_layout), index_granularity=1024
    )

    def event_variant(
        *,
        name=event_layout.name,
        table=event_layout.table,
        columns=event_layout.columns,
        primary_key=event_layout.primary_key,
        order_by=event_layout.order_by,
    ):
        return RecordLayout(
            name=name,
            table=table,
            columns=columns,
            primary_key=primary_key,
            order_by=order_by,
        )

    def tensor_variant(*, dtype="payload_dtype", shape="payload_shape", bytes_="payload_bytes"):
        return RecordLayout(
            name="tensor",
            table="tensor_records",
            columns=(
                RecordColumn("run", RecordCellType.STRING),
                RecordColumn(
                    "payload",
                    RecordCellType.TENSOR,
                    dtype_column=dtype,
                    shape_column=shape,
                    bytes_column=bytes_,
                ),
            ),
            primary_key=("run",),
            order_by=("run",),
        )

    identity_mismatches = {
        "layout set": RecordSchema((event_layout,), index_granularity=1024),
        "layout name": RecordSchema(
            (event_variant(name="other_event"), tensor_layout),
            index_granularity=1024,
        ),
        "target table": RecordSchema(
            (event_variant(table="other_event_records"), tensor_layout),
            index_granularity=1024,
        ),
        "logical column order": RecordSchema(
            (
                event_variant(
                    columns=(
                        event_layout.columns[0],
                        event_layout.columns[2],
                        event_layout.columns[1],
                    )
                ),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "logical column name": RecordSchema(
            (
                event_variant(
                    columns=(
                        event_layout.columns[0],
                        event_layout.columns[1],
                        RecordColumn("metric", RecordCellType.FLOAT64),
                    )
                ),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "logical column type": RecordSchema(
            (
                event_variant(
                    columns=(
                        event_layout.columns[0],
                        event_layout.columns[1],
                        RecordColumn("score", RecordCellType.INT64),
                    )
                ),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "tensor dtype column": RecordSchema(
            (event_layout, tensor_variant(dtype="other_dtype")),
            index_granularity=1024,
        ),
        "tensor shape column": RecordSchema(
            (event_layout, tensor_variant(shape="other_shape")),
            index_granularity=1024,
        ),
        "tensor bytes column": RecordSchema(
            (event_layout, tensor_variant(bytes_="other_bytes")),
            index_granularity=1024,
        ),
        "primary key order": RecordSchema(
            (
                event_variant(primary_key=("event_id", "run")),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "ordering key order": RecordSchema(
            (
                event_variant(order_by=("event_id", "run")),
                tensor_layout,
            ),
            index_granularity=1024,
        ),
        "index granularity": RecordSchema(
            (event_layout, tensor_layout), index_granularity=2048
        ),
    }

    host = DMXHostEngine(
        StageConfig.clickhouse_records(ClickHouseClientConfig(), schema)
    )
    backend = native._load_host_extension()
    backend._validate_record_host_schema(host, reordered_schema)
    for field, mismatched_schema in identity_mismatches.items():
        try:
            backend._validate_record_host_schema(host, mismatched_schema)
        except ValueError as error:
            assert "does not match" in str(error)
        else:
            pytest.fail(f"schema identity omitted {field}")

    legacy_host = DMXHostEngine(
        StageConfig.clickhouse_insert(ClickHouseClientConfig())
    )
    with pytest.raises(RuntimeError, match="schema-driven record stage"):
        backend._validate_record_host_schema(legacy_host, schema)


@pytest.mark.native_backend
def test_clickhouse_client_exposes_socket_timeouts_and_worker_metrics():
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    config = ClickHouseClientConfig()
    assert config.connect_timeout_ms == 5000
    assert config.receive_timeout_ms == 0
    assert config.send_timeout_ms == 0

    engine = DMXHostEngine(StageConfig.clickhouse_insert(config, parallelism=3))
    metrics = engine.clickhouse_metrics()
    assert metrics.expected_workers == 3
    assert metrics.ready_workers == 0
    assert metrics.peak_active_inserts == 0
    assert [worker.worker_index for worker in metrics.workers] == [0, 1, 2]


@pytest.mark.native_backend
def test_engine_metrics_follow_mutated_stage_parallelism():
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    stage = StageConfig.clickhouse_insert(ClickHouseClientConfig(), parallelism=1)
    stage.parallelism = 3

    metrics = DMXHostEngine(stage).clickhouse_metrics()

    assert metrics.expected_workers == 3
    assert [worker.worker_index for worker in metrics.workers] == [0, 1, 2]


@pytest.mark.cpu
def test_ring_export_requires_full_backend(monkeypatch):
    from dmi.transport import native

    calls = []

    def load_named(name):
        calls.append(name)
        raise ImportError("backend absent")

    monkeypatch.setattr(native, "_load_named_extension", load_named)
    monkeypatch.setattr(native, "_EXTENSION_MODULES", {})

    with pytest.raises(ImportError, match="full native backend"):
        native.RingEngine
    assert calls == ["_native_backend"]


@pytest.mark.cpu
def test_v1_host_export_does_not_load_ring_backend(monkeypatch):
    import dmi.api.v1 as api
    from dmi.transport import native

    sentinel = object()
    host_module = SimpleNamespace(DMXHostEngine=sentinel)
    ring_before = sys.modules.get("dmi.transport.ring")
    cached = api.__dict__.pop("DMXHostEngine", None)
    monkeypatch.setattr(native, "_load_host_extension", lambda: host_module)
    try:
        assert api.DMXHostEngine is sentinel
        assert sys.modules.get("dmi.transport.ring") is ring_before
    finally:
        api.__dict__.pop("DMXHostEngine", None)
        if cached is not None:
            api.__dict__["DMXHostEngine"] = cached


@pytest.mark.cpu
def test_v1_model_shape_contract_does_not_load_ring_backend():
    import dmi.api.v1 as api

    ring_before = sys.modules.get("dmi.transport.ring")
    hints = get_type_hints(api.make_model_shape_from_hf_config)
    shape = api.make_model_shape_from_hf_config(
        SimpleNamespace(hidden_size=64, num_attention_heads=8)
    )

    assert hints["return"] == api.ModelShapeConfig | None
    assert isinstance(shape, api.ModelShapeConfig)
    assert sys.modules.get("dmi.transport.ring") is ring_before


# --- the capture extensions are part of the documented build -----------------
#
# `make -C native clean && make -C native` is what install.md and the backend
# guides tell a user to run. `clean` removes the whole build directory, so a
# capture extension that `all` does not rebuild is deleted by the documented
# command and comes back only when someone remembers its separate target.
# These pin the default build to the capture extensions, install them where
# the loader looks, and keep the libcurl lookup honest. Dry runs (`-n`), so
# nothing is compiled; the default goal is read from make's rule database
# (`-p`) under a `clean` goal, because a dry run of `all` itself resolves the
# CUDA toolkit, which a CPU host does not have.

_NATIVE_DIR = Path(__file__).resolve().parents[1] / "native"
_EXT_SUFFIX = __import__("sysconfig").get_config_var("EXT_SUFFIX")


def _make(*args: str, environment: dict[str, str] | None = None
          ) -> subprocess.CompletedProcess[str]:
    # The caller's make state must not leak in: MAKEFLAGS carries command-line
    # overrides such as CURL_INCDIR from an enclosing `make test-cpu ...`.
    # `environment` is what a test sets there on purpose instead.
    env = {
        key: value
        for key, value in __import__("os").environ.items()
        if key not in {"MAKEFLAGS", "MFLAGS", "MAKELEVEL", "CAPTURE",
                       "CURL_INCDIR", "CURL_LIBDIR", "CURL_SYSROOT",
                       "PKG_CONFIG"}
    }
    env.update(environment or {})
    return subprocess.run(
        ["make", "-C", str(_NATIVE_DIR), f"PYTHON={sys.executable}", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _commands(stdout: str) -> list[list[str]]:
    """The dry run's commands, backslash continuations joined, as argv."""
    return [command.split() for command in
            stdout.replace("\\\n", " ").splitlines() if command.strip()]


def _prerequisites(database: str, target: str) -> list[str]:
    for line in database.splitlines():
        if line.startswith(f"{target}:") and not line.startswith(f"{target}::"):
            return line.split(":", 1)[1].split()
    raise AssertionError(f"make has no rule for {target!r}")


def _installed(name: str) -> Path:
    from dmi.transport import native

    return Path(native.__file__).resolve().parents[1] / f"{name}{_EXT_SUFFIX}"


@pytest.mark.cpu
def test_default_build_includes_the_capture_extensions():
    from dmi.transport import native

    result = _make("-p", "-n", "clean")
    assert result.returncode == 0, result.stdout + result.stderr

    assert "capture" in _prerequisites(result.stdout, "all")
    capture = _prerequisites(result.stdout, "capture")
    for name in ("_dmi_native_sink", "_dmi_native_store"):
        installed = _installed(name)
        assert str(installed) in capture
        # Installed beside _native_backend, which the loader searches first.
        assert installed.parent == native._search_dirs()[0]
        # The documented target names stay: CI and the loaders' error
        # messages ask for them.
        assert str(installed) in _prerequisites(result.stdout, f"build/{name}")


@pytest.mark.cpu
def test_capture_opt_out_leaves_the_default_build_to_the_backend():
    result = _make("-p", "-n", "clean", "CAPTURE=0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "capture" not in _prerequisites(result.stdout, "all")


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("overrides", "environment", "value"),
    [
        (("CAPTURE=yes",), {}, "yes"),
        (("CAPTURE=true",), {}, "true"),
        # Some unrelated tool's exported CAPTURE, which make reads as a
        # variable like any other.
        ((), {"CAPTURE": "/var/tmp/capture"}, "/var/tmp/capture"),
    ],
    ids=["yes", "true", "unrelated-environment-variable"],
)
def test_capture_accepts_only_zero_or_one(overrides, environment, value):
    # Anything but exactly 1 used to mean CAPTURE=0: CAPTURE=yes built the
    # backend without the capture extensions and said nothing.
    result = _make("-n", "clean", *overrides, environment=environment)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert f"CAPTURE must be 0 or 1, not '{value}'" in output
    assert "rm -rf" not in output


@pytest.mark.cpu
def test_capture_setting_ignores_surrounding_whitespace():
    for value, captured in ((" 1 ", True), (" 0 ", False)):
        result = _make("-p", "-n", "clean", f"CAPTURE={value}")
        assert result.returncode == 0, result.stdout + result.stderr
        assert ("capture" in _prerequisites(result.stdout, "all")) is captured


@pytest.mark.cpu
def test_clean_then_capture_build_recreates_what_clean_removed():
    result = _make("-B", "-n", "clean", "capture")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output

    commands = _commands(result.stdout)
    removal = next(i for i, argv in enumerate(commands) if argv[:2] == ["rm", "-rf"])
    rebuilt = commands[removal + 1:]
    for name in ("_dmi_native_sink", "_dmi_native_store"):
        built, installed = f"build/{name}{_EXT_SUFFIX}", str(_installed(name))
        assert installed in commands[removal]
        assert any(argv[argv.index("-o") + 1] == built
                   for argv in rebuilt if "-o" in argv)
        assert any(argv[-1] == installed for argv in rebuilt)


@pytest.mark.cpu
def test_capture_extensions_install_as_links_to_the_build(tmp_path):
    # The loader searches src/dmi before native/build. Installed as copies,
    # the src/dmi file went stale the moment anything rebuilt native/build
    # alone -- another branch's `make build/_dmi_native_sink`, which writes
    # only there -- and shadowed the fresh build. A relative link is one
    # file under two names. A real run, in a stand-in checkout; `-o` holds
    # the (fake) built files as they are, so nothing is compiled.
    import os

    names = ("_dmi_native_sink", "_dmi_native_store")
    checkout = tmp_path / "checkout"
    (checkout / "src" / "dmi").mkdir(parents=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    built = {name: build_dir / f"{name}{_EXT_SUFFIX}" for name in names}
    installed = {name: checkout / "src" / "dmi" / f"{name}{_EXT_SUFFIX}"
                 for name in names}
    for name in names:
        built[name].write_bytes(b"built " + name.encode())
        # What this branch installed before: a copy, newer than the build,
        # so the file-named target alone would call it up to date.
        installed[name].write_bytes(b"stale copy")
    goal = (f"PROJECT_ROOT={checkout}", f"BUILD_DIR={build_dir}",
            *(f"-o{path}" for path in built.values()), "capture")

    result = _make(*goal)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in names:
        link = installed[name]
        assert link.is_symlink(), f"{link} was installed as a copy"
        assert not os.path.isabs(os.readlink(link)), os.readlink(link)
        assert os.path.samefile(link, built[name])

        # Rebuilt in native/build alone, the way ld writes a new file.
        fresh = build_dir / "fresh"
        fresh.write_bytes(b"rebuilt " + name.encode())
        os.replace(fresh, built[name])
        assert link.read_bytes() == b"rebuilt " + name.encode()

    again = _make(*goal)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "ln " not in again.stdout, "an installed link was redone"


@pytest.mark.cpu
def test_capture_targets_are_named_after_the_files_they_produce():
    # A target whose recipe writes some other path is never up to date, so
    # every `make` relinked both extensions. Named after the file, a second
    # run has nothing to do.
    result = _make("-p", "-n", "clean")
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("_dmi_native_sink", "_dmi_native_store"):
        built = f"build/{name}{_EXT_SUFFIX}"
        assert built in _prerequisites(result.stdout, str(_installed(name)))
        assert _prerequisites(result.stdout, built)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("overrides", "environment", "include", "library"),
    [
        # CI's invocation: explicit directories win over pkg-config.
        (("CURL_INCDIR=/explicit/include", "CURL_LIBDIR=/explicit/lib"), {},
         "/explicit/include", "/explicit/lib"),
        # A sysroot the user names wins over pkg-config too: asking for it
        # is the only reason to pass it. On the command line...
        (("CURL_SYSROOT=/sysroot",), {},
         "/sysroot/usr/include/x86_64-linux-gnu",
         "/sysroot/usr/lib/x86_64-linux-gnu"),
        # ...or exported.
        ((), {"CURL_SYSROOT": "/sysroot"},
         "/sysroot/usr/include/x86_64-linux-gnu",
         "/sysroot/usr/lib/x86_64-linux-gnu"),
        # A host with the dev package: pkg-config names the directories.
        ((), {}, "/pkgconfig/include", "/pkgconfig/lib"),
        # No pkg-config entry: the extracted dev package under CURL_SYSROOT.
        (("PKG_CONFIG=false", "CURL_SYSROOT=/sysroot"), {},
         "/sysroot/usr/include/x86_64-linux-gnu",
         "/sysroot/usr/lib/x86_64-linux-gnu"),
    ],
    ids=["explicit", "sysroot-over-pkg-config",
         "sysroot-from-environment-over-pkg-config", "pkg-config",
         "sysroot-fallback"],
)
def test_libcurl_directories_resolve_in_documented_order(
    tmp_path, overrides, environment, include, library
):
    fake = tmp_path / "pkg-config"
    fake.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *--exists*) exit 0 ;;\n"
        "  *includedir*) echo /pkgconfig/include ;;\n"
        "  *libdir*) echo /pkgconfig/lib ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    result = _make("-B", "-n", "build/conformance_store",
                   f"PKG_CONFIG={fake}", *overrides, environment=environment)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    link = next(argv for argv in _commands(result.stdout) if "-o" in argv
                and argv[argv.index("-o") + 1] == "build/conformance_store")
    assert f"-I{include}" in link
    assert f"-L{library}" in link


@pytest.mark.cpu
def test_missing_libcurl_names_the_package(tmp_path):
    # A real run, not a dry run: the check has to fail before the store's
    # compile starts. Empty directories are not enough to make libcurl
    # missing where its dev package is installed system-wide (CI installs
    # it), since the compiler's default paths still find it; a compiler that
    # cannot find <curl/curl.h> is what a host without the package has.
    compiler = tmp_path / "c++"
    compiler.write_text(
        "#!/bin/sh\n"
        "echo 'fatal error: curl/curl.h: No such file or directory' >&2\n"
        "exit 1\n"
    )
    compiler.chmod(0o755)
    # BUILD_DIR keeps the run, and the torch stamp it writes, out of the
    # checkout's native/build.
    build_dir = tmp_path / "build"
    result = _make("build/_dmi_native_store", f"BUILD_DIR={build_dir}",
                   f"CXX={compiler}",
                   f"CURL_INCDIR={tmp_path}", f"CURL_LIBDIR={tmp_path}")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "libcurl4-openssl-dev" in output
    # The default build includes the capture extensions, so this failure is
    # also what a host without libcurl gets from a plain `make`, which then
    # builds no _native_backend either. The in-memory path does not need
    # them: the message has to name the opt-out.
    assert "CAPTURE=0" in output
    assert f"-o {build_dir}/_dmi_native_store" not in output


@pytest.mark.cpu
def test_libcurl_dev_package_without_its_runtime_names_the_runtime_package(
    tmp_path,
):
    # libcurl4-openssl-dev ships libcurl.so as a link to libcurl.so.4.x, and
    # the file it names is in the runtime package, libcurl4. A sysroot with
    # the dev package alone has a dangling link, which ld skips in favour of
    # libcurl.a -- a static libcurl that fails to link, or links and fails at
    # import. The generic "install libcurl4-openssl-dev" would send the user
    # back to the package they just extracted.
    (tmp_path / "libcurl.so").symlink_to("libcurl.so.4.6.0")
    result = _make("check-libcurl", f"CURL_INCDIR={tmp_path}",
                   f"CURL_LIBDIR={tmp_path}")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert f"{tmp_path}/libcurl.so" in output
    assert "runtime package libcurl4" in output


def _curl_flags(result: subprocess.CompletedProcess[str]) -> list[str]:
    link = next(argv for argv in _commands(result.stdout) if "-o" in argv
                and argv[argv.index("-o") + 1] == "build/conformance_store")
    return [arg for arg in link
            if arg.startswith("-L") or (arg.startswith("-I")
                                        and not arg.startswith("-Icsrc"))]


@pytest.mark.cpu
def test_sysroot_fallback_applies_only_when_the_sysroot_exists(tmp_path):
    # With neither pkg-config nor explicit directories, the build falls back
    # to the default sysroot only if it is there (and the user's own).
    # Otherwise it adds no -I/-L at all, so the compiler's default paths --
    # where a system-wide dev package lives -- apply, and a /tmp path that
    # means nothing on this host is never searched ahead of them.
    absent = _make("-B", "-n", "build/conformance_store", "PKG_CONFIG=false",
                   f"CURL_SYSROOT_DEFAULT={tmp_path / 'absent'}")
    assert absent.returncode == 0, absent.stdout + absent.stderr
    assert _curl_flags(absent) == []

    present = _make("-B", "-n", "build/conformance_store", "PKG_CONFIG=false",
                    f"CURL_SYSROOT_DEFAULT={tmp_path}")
    assert present.returncode == 0, present.stdout + present.stderr
    assert _curl_flags(present) == [
        f"-I{tmp_path}/usr/include/x86_64-linux-gnu",
        f"-L{tmp_path}/usr/lib/x86_64-linux-gnu",
    ]


@pytest.mark.cpu
def test_capture_extensions_relink_when_torch_changes(tmp_path):
    # Named after their files, the extensions are up to date whenever their
    # sources are -- including after a torch upgrade in the venv, which
    # changes the libtorch and pybind11 they were built against. A stamp
    # recording the torch and Python build they need is their prerequisite,
    # rewritten only when that record changes.
    database = _make("-p", "-n", "clean")
    assert database.returncode == 0, database.stdout + database.stderr
    stamp = "build/torch-extension.config"
    for name in ("_dmi_native_sink", "_dmi_native_store"):
        assert stamp in _prerequisites(database.stdout,
                                       f"build/{name}{_EXT_SUFFIX}")

    import torch

    build_dir = tmp_path / "build"
    written = build_dir / "torch-extension.config"
    goal = (f"BUILD_DIR={build_dir}", str(written))
    result = _make(*goal)
    assert result.returncode == 0, result.stdout + result.stderr
    record = written.read_text()
    assert torch.__version__ in record
    assert _EXT_SUFFIX in record

    before = written.stat().st_mtime_ns
    assert _make(*goal).returncode == 0
    assert written.stat().st_mtime_ns == before, "an unchanged record rewrote the stamp"

    written.write_text(record.replace(torch.__version__, "0.0.0"))
    assert _make(*goal).returncode == 0
    assert written.read_text() == record
