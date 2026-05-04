# Installation

Set up DMI from a fresh clone: fetch submodules, install the Python packages,
build the native backend, and prepare the optional ClickHouse sink.

Tested on Linux + CUDA 12.x + Python 3.10. A CUDA-capable GPU is required because
Ring² is a GPU-resident capture and transport pipeline.

## 1. Clone the repository

The repo uses three git submodules: a fork of HuggingFace `transformers`, a fork
of `vllm`, and the `clickhouse-cpp` C++ client.

```bash
git clone --recursive <your-repo-url> DMI
cd DMI

# If you forgot --recursive:
git submodule update --init --recursive
```

Expected submodule paths:

- `integration/transformers/` — modified HF Transformers (`gpt2_p`, `qwen3_p`)
- `integration/vllm/` — modified vLLM with DMI integration hooks
- `libs/clickhouse-cpp/` — ClickHouse C++ client linked into the native backend

## 2. Install ClickHouse server

The DB sink writes captured tensors into a ClickHouse table. Install the server
if you want persistent capture rather than transport-only runs.

```bash
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] \
https://packages.clickhouse.com/deb stable main" \
    | sudo tee /etc/apt/sources.list.d/clickhouse.list
sudo apt-get update
sudo apt-get install -y clickhouse-server clickhouse-client
```

Start the server:

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

To wipe local DB state between runs:

```bash
bash benchmark/clean_clickhouse.sh
```

## 3. Create the Python environment

```bash
conda env create -f environment.yml
conda activate proj-dmx
```

## 4. Install Python packages

Install the modified Transformers submodule, then DMI itself:

```bash
pip install -e integration/transformers/
pip install -e .
```

Install vLLM only if you want the vLLM path:

```bash
pip install -e integration/vllm/
```

This may take a while because it is a full vLLM build.

## 5. Build native dependencies

Build the ClickHouse C++ client:

```bash
cmake -S libs/clickhouse-cpp -B libs/clickhouse-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build libs/clickhouse-cpp/build -j
```

Build the DMI native backend:

```bash
make -C monitoring -j
# or simply: make
```

Artifacts are emitted as `monitoring_native_backend.<EXT_SUFFIX>.so` at the
project root and inside `monitoring/`. If `nvcc` is not on `PATH`, set:

```bash
export NVCC=/usr/local/cuda/bin/nvcc
```

Smoke check:

```bash
python -c "import monitoring; print(monitoring.__file__)"
python -c "from monitoring._native_engine import RingConfig; print(RingConfig())"
```

## Troubleshooting

- **`ImportError` on `monitoring_native_backend`** — rebuild with
  `make -C monitoring clean && make -C monitoring -j`, then confirm `pip install -e .`
  used the active conda env.
- **Linker errors against `libclickhouse-cpp-lib`** — rerun step 5 and confirm
  `libs/clickhouse-cpp/build/clickhouse/` exists.
- **`Connection refused` to ClickHouse** — check
  `sudo systemctl status clickhouse-server`; DMI uses TCP port `9000`, not HTTP
  port `8123`.
- **CUDA arch mismatch** — the Makefile uses `SM_ARCH=native`. Override with
  `make -C monitoring SM_ARCH=sm_89` for a fixed target such as RTX 4090.
