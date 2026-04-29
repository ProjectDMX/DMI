# DMI Quickstart

This guide walks through getting **Project-DMI** (Deep Model Inspection) running end-to-end:
clone with submodules, install ClickHouse, build native dependencies, and launch a benchmark
against either the **HuggingFace** path or the **vLLM** path.

Tested on Linux + CUDA 12.x + Python 3.10. A CUDA-capable GPU is required (the ring transport
is a GPU-resident pipeline).

---

## 1. Clone the repository

The repo uses three git submodules: a fork of HuggingFace `transformers`, a fork of `vllm`,
and the `clickhouse-cpp` C++ client.

```bash
git clone --recursive <your-repo-url> DMI
cd DMI

# If you forgot --recursive:
git submodule update --init --recursive
```

After this, you should have:

- `integration/transformers/` — modified HF transformers (provides `gpt2_p`, `qwen3_p`)
- `integration/vllm/` — modified vLLM with DMI integration hooks
- `libs/clickhouse-cpp/` — ClickHouse C++ client (linked into the native backend)

---

## 2. Install ClickHouse server

The DB sink writes captured tensors into a ClickHouse table. Install the server (Ubuntu/Debian
example):

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

Start the server (it listens on TCP port `9000` by default — that's what DMI connects to):

```bash
sudo systemctl enable --now clickhouse-server
sudo systemctl status clickhouse-server     # should be "active (running)"
clickhouse-client --query "SELECT 1"        # smoke check; expect "1"
```

Optional connection overrides via env (defaults shown):

```
DMX_DB_HOST=localhost
DMX_DB_PORT=9000
DMX_DB_USER=default
DMX_DB_PASSWORD=
DMX_DB_DATABASE=default
DMX_DB_TABLE=offload
```

To wipe state between runs, use the helper at `benchmark/clean_clickhouse.sh`:

```bash
bash benchmark/clean_clickhouse.sh
```

It stops the server, deletes `/var/lib/clickhouse/*`, and restarts the server.

---

## 3. Create the Python environment

```bash
conda env create -f environment.yml
conda activate proj-dmx
```

This creates a Python 3.10 env and installs everything in `requirements.txt`
(torch ≥ 2.8, HF stack, pandas, etc.).

---

## 4. Install the Python packages

Install the modified `transformers` (editable, so the `gpt2_p` / `qwen3_p` hook-aware
modules are importable), then DMI itself:

```bash
pip install -e integration/transformers/
pip install -e .
```

---

## 5. Build the ClickHouse C++ client

The DMI native backend (`monitoring_native_backend.so`) links against `libclickhouse-cpp-lib`
and its bundled deps (lz4, cityhash, absl, zstd). Build them in-tree:

```bash
cmake -S libs/clickhouse-cpp -B libs/clickhouse-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build libs/clickhouse-cpp/build -j
```

After this you should see `libs/clickhouse-cpp/build/clickhouse/libclickhouse-cpp-lib.a`
(and the contrib static libs under `libs/clickhouse-cpp/build/contrib/`).

---

## 6. Build the DMI native backend

Compiles the C++/CUDA monitoring extension (ring buffers, drain thread, ClickHouse
client, torch op for the producer kernel). Uses `nvcc` for `.cu` files and `g++` for
`.cpp` — paths are auto-discovered from the active Python interpreter.

```bash
make -C monitoring -j
# or simply:  make
```

Artifact: `monitoring_native_backend.<EXT_SUFFIX>.so` at the project root (and inside
`monitoring/`). If `nvcc` isn't on your `PATH`, set `NVCC=/usr/local/cuda/bin/nvcc` in
the environment.

Sanity check the import:

```bash
python -c "import monitoring; print(monitoring.__file__)"
python -c "from monitoring._native_engine import RingConfig; print(RingConfig())"
```

---

## 7. Install vLLM (only if you want the vLLM path)

The `integration/vllm/` submodule is a fork with DMI hooks. Install editable from
the submodule:

```bash
pip install -e integration/vllm/
```

This may take a while — it's a full vLLM build.

---

## 8. Launch — HuggingFace path

### 8a. Quick HF generate (no monitoring, sanity check)

```bash
python benchmark/scripts/hf_generate.py \
    --model gpt2 --device cuda --batch-size 8 --max-new-tokens 16
```

### 8b. HF + DMI monitoring, no DB sink

Captures internal states via the ring transport but discards them (good for measuring
transport overhead alone):

```bash
python benchmark/scripts/hf_monitoring_generate.py \
    --model qwen3 --device cuda --batch-size 8 --max-new-tokens 16 --no-db
```

### 8c. HF + DMI monitoring → ClickHouse

```bash
python benchmark/scripts/hf_monitoring_generate.py \
    --model qwen3 --device cuda --batch-size 8 --max-new-tokens 16
```

After it finishes, inspect captured rows:

```bash
clickhouse-client --query "SELECT count() FROM default.offload"
```

### 8d. HF ring-transport benchmark

Compares `baseline` (no monitoring), `ring_null` (transport active, no DB),
`ring_db` (full pipeline), and `hf_offload` (HF's own `output_hidden_states=True`
path) for apples-to-apples overhead numbers:

```bash
# Decode-heavy (single-token prompt, 16 decode steps)
python -m benchmark.bench_ring_transport \
    --model qwen3 --batch-size 4 \
    --prefill-len 1 --decode-len 16 \
    --warmup 1 --iters 3 \
    --modes baseline,ring_null,ring_db \
    --cuda-graphs
```

Useful flags (see `bench_ring_transport.py` for the full list):

- `--model gpt2 | qwen3` (alias for `Qwen/Qwen3-4B`)
- `--cuda-graphs` — capture decode under CUDA graphs (uses the lean greedy loop)
- `--hook-selection full | hf-only | hidden-states | logits | attention`
- `--ring-payload-mb`, `--ring-pinned-mb` — ring buffer sizes (default 4096 MiB each)
- `--csv path.csv` — append a result row

---

## 9. Launch — vLLM path

DMI plugs into vLLM via a custom worker class:
`monitoring.vllm_integration.DMXGPUWorker`.
Pass it through `--worker-cls` (or `worker_cls=` in the offline `LLM(...)` API),
plus `additional_config` for hook selection and DB connection.

### 9a. Offline use via the `LLM(...)` API

Below is a self-contained snippet — paste into a `.py` file and run. It loads vLLM
with the DMI worker and pushes captured tensors into ClickHouse:

```python
import os
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=512,
    enforce_eager=False,
    gpu_memory_utilization=0.5,
    worker_cls="monitoring.vllm_integration.DMXGPUWorker",
    additional_config={
        "dmx_hook_selection":   "vllm-full",
        "dmx_ring_payload_mb":  4096,
        "dmx_ring_pinned_mb":   4096,
        "dmx_db_host":          "localhost",
        "dmx_db_port":          9000,
        # Set "dmx_null_mode": True to skip DB writes (transport-only).
    },
)

prompts = [f"The answer to question {i+1} is" for i in range(8)]
params = SamplingParams(temperature=0.0, max_tokens=32)
for o in llm.generate(prompts, params):
    print(o.outputs[0].text)
```

To get a baseline (vanilla vLLM, no monitoring), drop `worker_cls` and
`additional_config`. To get a transport-only run (no DB write), add
`"dmx_null_mode": True` to `additional_config`.

### 9b. vLLM serve with DMI

```bash
vllm serve Qwen/Qwen3-8B \
    --worker-cls monitoring.vllm_integration.DMXGPUWorker \
    --additional-config '{
        "dmx_hook_selection": "vllm-full",
        "dmx_ring_payload_mb": 4096,
        "dmx_ring_pinned_mb": 4096,
        "dmx_db_host": "localhost",
        "dmx_db_port": 9000
    }'
```

If you need to disable vLLM's compile cache (recommended for benchmarking):

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
```

---

## 10. Troubleshooting

- **`ImportError` on `monitoring_native_backend`** — rebuild: `make -C monitoring clean && make -C monitoring -j`.
  Make sure `pip install -e .` was run from the same conda env.
- **Linker errors against `libclickhouse-cpp-lib`** — step 5 didn't complete; check
  `libs/clickhouse-cpp/build/clickhouse/` exists.
- **`Connection refused` to ClickHouse** — `sudo systemctl status clickhouse-server`;
  default port is `9000` (TCP), not `8123` (HTTP).
- **`libstdc++` mismatch under vLLM** — preload the conda libstdc++:
  `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python -m benchmark.bench_vllm_transport ...`
- **Stale DB rows between runs** — `bash benchmark/clean_clickhouse.sh` (stops the
  server, wipes `/var/lib/clickhouse/*`, restarts).
- **CUDA arch mismatch** — the Makefile uses `SM_ARCH=native`. Override with
  `make -C monitoring SM_ARCH=sm_89` (e.g. RTX 4090) if needed.

---

## 11. Repo map (where to look)

| Path | What's there |
|------|--------------|
| `monitoring/` | Python engine + `csrc/` C++/CUDA backend |
| `monitoring/csrc/ring/` | Ring buffers, drain thread, producer kernel |
| `monitoring/vllm_integration.py` | `DMXGPUWorker` for vLLM |
| `benchmark/scripts/hf_generate.py` | Vanilla HF generate (baseline) |
| `benchmark/scripts/hf_monitoring_generate.py` | HF + monitoring, optional DB |
| `benchmark/bench_ring_transport.py` | HF ring-transport overhead benchmark |
| `benchmark/data/prompts.txt` | Default prompt corpus |
| `benchmark/clean_clickhouse.sh` | Wipe `/var/lib/clickhouse/*` and restart server |
| `libs/clickhouse-cpp/` | ClickHouse C++ client (submodule) |
| `integration/transformers/` | Modified HF transformers (submodule) |
| `integration/vllm/` | Modified vLLM (submodule) |