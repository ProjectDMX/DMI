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
sudo apt-get install -y build-essential cmake git
```

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

The repo uses four git submodules: the DMI HuggingFace integration, the
version-matched DMI-vLLM integration, the version-matched DMI-Megatron
integration, and the `clickhouse-cpp` C++ client. The commands below fetch all
four repositories; they do not install any Python integration.

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

Expected submodule paths:

- `third_party/transformers/` — modified HF Transformers (`gpt2_p`, `qwen3_p`, `llama_p`)
- `third_party/vllm-integration/` — DMI integration for an unmodified official vLLM installation
- `third_party/DMI-Megatron-Integration/` — DMI integration with its pinned Megatron-LM fork at `third_party/DMI-Megatron-Integration/third_party/megatron-lm/`
- `third_party/clickhouse-cpp/` — ClickHouse C++ client linked into the native backend

## 2. Install ClickHouse server

DMI writes captured tensors into a ClickHouse table. Follow the
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
```

Artifacts are emitted as `_native_backend.<EXT_SUFFIX>.so` inside `native/`
and as the importable `src/dmi/_native_backend.<EXT_SUFFIX>.so`. Host exports
prefer the full backend and fall back to `_host_backend`; ring exports always
require the full backend.

Smoke check the package and host backend:

```bash
python -c "import dmi; print(dmi.__file__)"
python -c "from dmi.api.v1 import DMXHostEngine; print(DMXHostEngine.__module__)"
```

After a full build, smoke check the ring backend:

```bash
python -c "from dmi.transport.native import RingConfig; print(RingConfig())"
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
or [Megatron-LM guide](megatron.md). Use a separate environment and checkout
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
- **Linker errors against `libclickhouse-cpp-lib`** — rerun step 5 and confirm
  `third_party/clickhouse-cpp/build/clickhouse/` exists.
- **`Connection refused` to ClickHouse** — check
  `sudo systemctl status clickhouse-server`; DMI uses TCP port `9000`, not HTTP
  port `8123`.
- **CUDA arch mismatch** — the Makefile uses `SM_ARCH=native`. Override with
  `make -C native SM_ARCH=sm_89` for a fixed target such as RTX 4090.
