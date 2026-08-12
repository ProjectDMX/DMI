# Installation

Set up DMI from a fresh clone: fetch submodules, install the Python packages,
and let `pip` compile the native backend in one step.

Tested on Linux + CUDA 12.x + Python >=3.10. A CUDA-capable GPU is required because
Ring² is a GPU-resident capture and transport pipeline.

## 0. System prerequisites

DMI builds C++/CUDA artifacts. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
```

Plus a working CUDA toolkit (NVCC) matching your driver. DMI is tested
against CUDA 12.x; install per the
[official NVIDIA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
Verify:

```bash
nvcc --version
nvidia-smi
```

If `nvcc` is not on `PATH`, point the build at it explicitly:

```bash
export NVCC=/usr/local/cuda/bin/nvcc
```

## 1. Clone the repository

The repo uses three git submodules: a fork of HuggingFace `transformers`, a fork
of `vllm`, and the `clickhouse-cpp` C++ client.

```bash
git clone https://github.com/ProjectDMX/DMI.git
cd DMI

# Initialise submodules (clickhouse-cpp is required for the native build;
# transformers and vllm are installed separately in step 4).
git submodule update --init --recursive libs/clickhouse-cpp
git submodule update --init --recursive integration/transformers
```

Expected submodule paths:

- `integration/transformers/` — modified HF Transformers (`gpt2_p`, `qwen3_p`, `llama_p`)
- `integration/vllm/` — modified vLLM with DMI integration hooks
- `libs/clickhouse-cpp/` — ClickHouse C++ client linked into the native backend

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

## 3. Set up the Python environment

Pick one of the two options below.

### 3a. Conda

If conda is not already installed, follow the
[Miniconda installation guide](https://docs.anaconda.com/miniconda/install/)
first. Then:

```bash
conda env create -f environment.yml
conda activate proj-dmx
```

### 3b. venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Install DMI (compiles the native backend)

> **Build-isolation requirement** — the native backend build queries torch's
> include/lib paths at compile time, so torch must be importable during the
> build.  Always pass `--no-build-isolation`:

```bash
# Step 4a: install a CUDA-capable torch first (skip if already installed)
pip install torch

# Step 4b: install the vendored transformers fork
pip install -e integration/transformers --no-deps

# Step 4c: build and install DMI (compiles clickhouse-cpp + native backend)
pip install -e . --no-build-isolation
```

`pip install -e . --no-build-isolation` runs `setup.py`'s `NativeBuildExt`,
which performs these steps automatically:

1. `git submodule update --init libs/clickhouse-cpp` (if not already done)
2. `cmake -S libs/clickhouse-cpp -B libs/clickhouse-cpp/build` (configure)
3. `cmake --build libs/clickhouse-cpp/build` (build static lib)
4. `make -C monitoring` (compile the `.so` via nvcc + g++)

Artifacts are emitted as `monitoring_native_backend.<EXT_SUFFIX>.so` at the
project root and inside `monitoring/`.

### 4a (optional). Install the vLLM fork

Only needed for the vLLM integration path:

```bash
git submodule update --init --recursive integration/vllm
pip install -e integration/vllm
```

This may take a while because it is a full vLLM build.

## 5. Smoke check

```bash
# Verify the native backend loaded
python -c "from monitoring._native_engine import RingConfig; print(RingConfig())"

# End-to-end: capture activations into ClickHouse and query row count
python example/visualization/run_offload_hf.py
clickhouse-client --query "SELECT count() FROM default.offload WHERE model_id='demo_hf'"
```

Expect generated text on stdout and a non-zero row count.

## Troubleshooting

- **`ImportError` on `monitoring_native_backend`** — rebuild with
  `make -C monitoring clean && pip install -e . --no-build-isolation`.
  Confirm `nvcc` is on PATH and the active Python is the one that has torch.
- **`--no-build-isolation` omitted** — if torch is not importable during the
  build, the Makefile cannot detect include/lib paths; always pass the flag.
- **Linker errors against `libclickhouse-cpp-lib`** — the cmake step may have
  been skipped; delete `libs/clickhouse-cpp/build` and re-run
  `pip install -e . --no-build-isolation`.
- **`Connection refused` to ClickHouse** — check
  `sudo systemctl status clickhouse-server`; DMI uses TCP port `9000`, not HTTP
  port `8123`.
- **CUDA arch mismatch** — the Makefile uses `SM_ARCH=native`. Override with
  `make -C monitoring SM_ARCH=sm_89` for a fixed target such as RTX 4090.
