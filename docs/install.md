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

The repo uses three git submodules: the DMI HuggingFace integration, the
version-matched DMI-vLLM integration, and the `clickhouse-cpp` C++ client.
The commands below fetch all three repositories; they do not install either
Python integration.

The command below creates one backend checkout. If you plan to use both
backends, repeat it with distinct target directories such as `DMI-hf` and
`DMI-vllm`; do not share one checkout between their environments.

```bash
git clone --recursive https://github.com/ProjectDMX/DMI.git
cd DMI

# Or initialize submodules after cloning:
git submodule update --init --recursive
```

Expected submodule paths:

- `third_party/transformers/` — modified HF Transformers (`gpt2_p`, `qwen3_p`, `llama_p`)
- `third_party/vllm-integration/` — DMI integration for an unmodified official vLLM installation
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
DMI_BACKEND_ENV=dmi-hf  # Example; use dmi-vllm in the vLLM checkout.
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

Build the DMI native backend:

```bash
make -C native -j
# or simply: make
```

Artifacts are emitted as `_native_backend.<EXT_SUFFIX>.so` inside `native/`
and as the importable `src/dmi/_native_backend.<EXT_SUFFIX>.so`.

Smoke check (loads the built `.so`):

```bash
python -c "import dmi; print(dmi.__file__)"
python -c "from dmi.transport.native import RingConfig; print(RingConfig())"
```

## 6. Choose one backend

Continue with either the [HuggingFace guide](huggingface.md) or the
[vLLM guide](vllm.md). Use a separate environment and checkout for each
backend. The HuggingFace path installs a modified Transformers checkout,
whereas the vLLM path installs its own official dependency set; do not install
the HuggingFace integration in the vLLM environment.

The native extension is also environment-specific: its Python suffix, Torch
ABI, CUDA selection, and runtime paths come from the active environment. Each
backend guide therefore finishes by rebuilding the extension in that backend's
checkout.

## Troubleshooting

- **`ImportError` on `_native_backend`** — rebuild with
  `make -C native clean && make -C native -j`, then confirm `pip install -e .`
  used the active conda env.
- **Linker errors against `libclickhouse-cpp-lib`** — rerun step 5 and confirm
  `third_party/clickhouse-cpp/build/clickhouse/` exists.
- **`Connection refused` to ClickHouse** — check
  `sudo systemctl status clickhouse-server`; DMI uses TCP port `9000`, not HTTP
  port `8123`.
- **CUDA arch mismatch** — the Makefile uses `SM_ARCH=native`. Override with
  `make -C native SM_ARCH=sm_89` for a fixed target such as RTX 4090.
