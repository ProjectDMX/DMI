# Core installation

Set up DMI from a fresh clone: fetch submodules, install DMI editable, build the
native backend, and prepare the ClickHouse sink.

Tested on Linux + Python >=3.10. A CUDA-capable GPU is required because Ring² is
a GPU-resident capture and transport pipeline.

DMI currently supports installation from source.

## 0. System prerequisites

DMI builds C++/CUDA artifacts; the conda env covers Python deps but not
system toolchains. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev pkg-config
```

`libcurl4-openssl-dev` is for the native capture storage path's extensions,
which the default native build includes (step 5); `pkg-config` lets the build
find it.

Plus a complete CUDA toolkit whose major version matches the CUDA version used
to build PyTorch (`torch.version.cuda`) and is supported by your driver. Install
it per the
[official NVIDIA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
Verify:

```bash
nvcc --version
nvidia-smi
```

## 1. Clone the repository

The repo uses five git submodules: the DMI HuggingFace integration, the
version-matched DMI-vLLM integration, the version-matched DMI-SGLang
integration, the version-matched DMI-Megatron integration, and the
`clickhouse-cpp` C++ client. The commands below fetch all five repositories;
they do not install any Python integration.

The command below creates one backend checkout. If you plan to use multiple
backends, repeat it with distinct target directories such as `DMI-hf`,
`DMI-vllm`, and `DMI-megatron`; do not share one checkout between their
environments.

```bash
git clone --recursive https://github.com/ProjectDMX/DMI.git
cd DMI

# Or initialize submodules after cloning:
git submodule update --init --recursive
```

`.gitmodules` pins HTTPS URLs deliberately: it is the read path every user and
every piece of automation inherits, and an HTTPS clone of these repositories
needs no key setup. To use SSH instead, rewrite the remotes on your own machine
rather than editing `.gitmodules` -- the rewrite applies to the superproject and
every submodule, including nested ones, and survives a re-clone:

```bash
git config --global url."git@github.com:".insteadOf https://github.com/
```

That config needs no follow-up in an existing checkout. Git applies an
`insteadOf` rewrite when it RESOLVES a stored URL, so it takes effect
immediately -- `git -c 'url.git@github.com:.insteadOf=https://github.com/'
remote get-url origin` prints the SSH form against an origin stored as HTTPS.

`git submodule sync --recursive` is for the other case only: when the URLs in
`.gitmodules` themselves change. There it is needed, because `.git/config` and
`.git/modules/*/config` keep whatever URL they were cloned with and a plain
`git pull` will not adopt the new one. Do not run it just to pick up an
`insteadOf` setting -- it would overwrite any submodule URL you had set
deliberately in your own checkout, to no purpose.

Expected submodule paths:

- `third_party/transformers/` — modified HF Transformers (`gpt2_p`, `qwen3_p`, `llama_p`)
- `third_party/vllm-integration/` — DMI integration for an unmodified official vLLM installation
- `third_party/sglang-integration/` — DMI integration for an unmodified official SGLang installation
- `third_party/DMI-Megatron-Integration/` — DMI integration with its pinned Megatron-LM fork at `third_party/DMI-Megatron-Integration/third_party/megatron-lm/`
- `third_party/clickhouse-cpp/` — ClickHouse C++ client linked into the native backend

## 2. Install ClickHouse server

DMI writes captured tensors into a ClickHouse table. **ClickHouse 24.11 or
later is required**: the catalog's schema check asks `CHECK GRANT SHOW TABLES`
about each object before it reaches a verdict, and that statement arrived in
24.11. `system.tables` is grant-filtered per role, so without the probe an
object this role cannot see is indistinguishable from one that was dropped,
and there is nothing safe to fall back to. The suites and the published
measurements run against 25.12. Follow the
[ClickHouse installation guide](https://clickhouse.com/docs/install) for your
platform.

Start the server and confirm it accepts queries:

```bash
sudo systemctl enable --now clickhouse-server
sudo systemctl status clickhouse-server
clickhouse-client --query "SELECT 1"
```

Default DMI connection settings:

```text
DMX_DB_HOST=localhost
DMX_DB_PORT=9000
DMX_DB_USER=default
DMX_DB_PASSWORD=
DMX_DB_DATABASE=default
DMX_DB_TABLE=offload
```

If captured tensors accumulate and the ClickHouse data directory grows too large
between runs, you may want to clear old content. Refer to the ClickHouse
documentation for the appropriate cleanup procedure.

## 3. Set up the Python environment

Pick one of the two options below.

### 3a. Conda

If conda is not already installed, follow the
[Miniconda installation guide](https://docs.anaconda.com/miniconda/install/)
first. Then:

```bash
DMI_BACKEND_ENV=dmi-hf  # Example; use dmi-vllm or dmi-megatron for those checkouts.
conda env create -f environment.yml --name "$DMI_BACKEND_ENV"
conda activate "$DMI_BACKEND_ENV"
```

### 3b. venv

Install the Python `venv` module (Ubuntu/Debian):

```bash
sudo apt install python3-venv
```

Then create the environment, activate it, and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=21.3"
pip install -r requirements.txt
```

The pip minimum is required for the PEP 660 editable install used below. Conda
environments must provide the same or a newer pip version.

With the environment active and the checkout available, verify the PyTorch CUDA
build and inspect the coherent toolkit DMI selected:

```bash
python -c "import torch; print(torch.version.cuda)"
make -C native cuda-info
```

If multiple matching toolkits are installed, select one explicitly and rerun
`cuda-info`:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
# Alternatively: export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
make -C native cuda-info
```

## 4. Install DMI core

Install the root checkout editable. The native build in the next step writes
its importable extension directly into this source tree.

```bash
pip install -e .
```

## 5. Build native dependencies

Build the ClickHouse C++ client:

```bash
cmake -S third_party/clickhouse-cpp -B third_party/clickhouse-cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build third_party/clickhouse-cpp/build -j
```

For host-side ClickHouse ingestion and benchmarking without CUDA, build the
CPU-only backend:

```bash
make -C native host -j
# or: make host
```

This emits `_host_backend.<EXT_SUFFIX>.so` inside `native/` and as the
importable `src/dmi/_host_backend.<EXT_SUFFIX>.so`. It contains the host
pipeline and ClickHouse client but no ring transport or CUDA symbols.

For GPU capture and ring transport, build the full backend:

```bash
make -C native -j
# or simply: make
# no libcurl dev package? add CAPTURE=0 to skip the capture extensions below
```

Artifacts are emitted as `_native_backend.<EXT_SUFFIX>.so` inside `native/`
and as the importable `src/dmi/_native_backend.<EXT_SUFFIX>.so`. Host exports
prefer the full backend and fall back to `_host_backend`; ring exports always
require the full backend.

The same build also produces the two extensions of the native capture storage
path (`storage_backend="capture"`): `_dmi_native_sink.<EXT_SUFFIX>.so` (the
pack writer) and `_dmi_native_store.<EXT_SUFFIX>.so` (the storage service and
reader). Each is linked in `native/build/` and made importable as a symlink of
the same name in `src/dmi/`, so the file the loader finds first is always the
latest build. Their make target is `capture`, which needs no CUDA and can be built on its
own; `build/_dmi_native_sink` and `build/_dmi_native_store` build one each.

```bash
make -C native capture -j          # the capture extensions only, no CUDA
make -C native -j CAPTURE=0        # the full backend without them
```

They link libcurl. Without root, extract the `libcurl4-openssl-dev` package
together with the runtime package `libcurl4` (the dev package's `libcurl.so`
links to a file that ships in `libcurl4`), and point the build at them:

```bash
apt-get download libcurl4-openssl-dev libcurl4
for deb in libcurl4*.deb; do dpkg-deb -x "$deb" "$HOME/curl-sysroot"; done
make -C native -j CURL_SYSROOT="$HOME/curl-sysroot"
```

The build picks libcurl's header directory (`CURL_INCDIR`) and library
directory (`CURL_LIBDIR`) in this order, each of the two on its own; the first
that applies wins:

1. `CURL_INCDIR` / `CURL_LIBDIR` set on the command line or in the environment,
   for example
   `CURL_INCDIR=/usr/include/x86_64-linux-gnu CURL_LIBDIR=/usr/lib/x86_64-linux-gnu`
   (what CI passes).
2. A non-empty `CURL_SYSROOT` set on the command line or in the environment:
   `$CURL_SYSROOT/usr/include/x86_64-linux-gnu` and
   `$CURL_SYSROOT/usr/lib/x86_64-linux-gnu`, the layout the extraction above
   produces. `pkg-config` is not consulted.
3. `pkg-config libcurl` (the binary named by `PKG_CONFIG`), which knows both
   wherever `libcurl4-openssl-dev` is installed.
4. The default sysroot `CURL_SYSROOT_DEFAULT`, laid out as in 2. It defaults to
   `/tmp/opencode/sysroot`, where one development host keeps its extracted
   packages, and is used only if that directory exists and is owned by the
   user running `make`; on any other host it is skipped. Pass
   `CURL_SYSROOT_DEFAULT=<dir>` to move it.
5. Neither: no `-I`/`-L` is added, and the compiler's default paths apply.

The extensions load the system's `libcurl.so.4` at runtime, so a sysroot is
needed only to build.

Smoke check the package and host backend:

```bash
python -c "import dmi; print(dmi.__file__)"
python -c "from dmi.api.v1 import DMXHostEngine; print(DMXHostEngine.__module__)"
```

After a full build, smoke check the ring backend and the capture extensions:

```bash
python -c "from dmi.transport.native import RingConfig; print(RingConfig())"
python -c "import torch; from dmi.transport import native; [print(native._load_named_extension(name).__file__) for name in ('_native_backend', '_dmi_native_sink', '_dmi_native_store')]"
```

Run the dependency-free CPU gate without CUDA, ClickHouse, native artifacts,
model weights, or initialized framework forks:

```bash
make test-cpu
```

Build and verify the CPU-only native host backend:

```bash
make test-host
```

Tests with optional runtime prerequisites carry separate markers and skip when
those resources are unavailable. `make test-host` does not download or start
ClickHouse; use the host benchmark separately with a running server.

## 6. Choose one backend

Continue with the [HuggingFace guide](huggingface.md), [vLLM guide](vllm.md),
[SGLang guide](sglang.md), or [Megatron-LM guide](megatron.md). Use a separate environment and checkout
for each backend. The HuggingFace path installs a modified Transformers
checkout, the vLLM path installs its own official dependency set, and the
Megatron-LM path installs its version-matched integration and pinned fork. Do
not mix their framework dependencies in one environment.

The native extension is also environment-specific: its Python suffix, Torch
ABI, CUDA selection, and runtime paths come from the active environment. Each
backend guide therefore finishes by rebuilding the extension in that backend's
checkout.

## Troubleshooting

- **`ImportError` on `_native_backend`** — rebuild with
  `make -C native clean && make -C native -j`, then confirm `pip install -e .`
  used the active conda env.
- **`ImportError` on `_host_backend`** — build with
  `make -C native host -j`; this target does not require CUDA or `nvcc`.
- **`native/Makefile: cannot build against libcurl`** — install
  `libcurl4-openssl-dev` (`libcurl-devel` on Fedora/RHEL), or pass
  `CURL_SYSROOT`, or `CURL_INCDIR` and `CURL_LIBDIR`, as in step 5. To build
  without the capture extensions, pass `CAPTURE=0`.
- **`native/Makefile: .../libcurl.so links to a missing file`** — the sysroot
  has `libcurl4-openssl-dev` without `libcurl4`; extract both, as in step 5.
- **`ImportError` on `_dmi_native_sink` or `_dmi_native_store`** — rebuild
  with `make -C native capture -j`; `make -C native clean` removes them along
  with the full backend.
- **Linker errors against `libclickhouse-cpp-lib`** — rerun step 5 and confirm
  `third_party/clickhouse-cpp/build/clickhouse/` exists.
- **`Connection refused` to ClickHouse** — check
  `sudo systemctl status clickhouse-server`; DMI uses TCP port `9000`, not HTTP
  port `8123`.
- **CUDA arch mismatch** — the Makefile uses `SM_ARCH=native`. Override with
  `make -C native SM_ARCH=sm_89` for a fixed target such as RTX 4090.
