# Proj-dmx (Huggineface/transformers)

**Prototype of Proj-dmx on HF/transformers library.** A white-box observability system for LLM inference. Capture and analyze internal model states (activations, attention weights, KV cache) with minimal performance overhead.

## Features

- **Internal State Monitoring**: Capture activations, attention patterns, and KV cache statistics during inference
- **Configurable Sampling**: Control capture frequency with step-level and request-level scheduling
- **Async Pipeline**: Non-blocking GPU→CPU transfer with pinned memory pools
- **Native C++ Backend**: High-performance hook callbacks with Python/C++ hybrid architecture
- **TransformerLens-style API**: Familiar `run_with_cache` interface for activation collection

## Architecture

```
┌──────────────────────────────────────────────────┐
│  HookedGPT2Model (modified transformers)         │
│  └── HookPoints → trigger callbacks              │
└─────────────────────┬────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│  MonitoringEngine                                │
│  ├── CaptureSchedule (token/request sampling)    │
│  ├── HookSelection (hooks sampling)              │
│  └── Native Backend routing                      │
└─────────────────────┬────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│  C++ Native Backend                              │
│  ├── Async GPU→CPU transfer                      │
│  ├── Pinned memory management                    │
│  └── Lock-free task queue                        │
└──────────────────────────────────────────────────┘
```


## Project Structure

```
HF_Prometheus/
├── monitoring/
│   ├── __init__.py
│   ├── engine.py          # MonitoringEngine
│   ├── config.py          # Configuration classes
│   ├── task.py            # Task definitions
│   └── csrc/              # C++ native backend
│       ├── native_engine.cpp
│       ├── hooks.cpp
│       └── ...
├── transformers/          # Git submodule (forked)
│   └── src/transformers/models/gpt2_p/
│       ├── modeling_gpt2.py   # HookedGPT2Model
│       └── hook_points.py     # HookPoint implementation
├── benchmark/
│   └── tests/             # Performance benchmarks
└── tests_monitoring/      # Unit tests
```


## Installation

```bash
# Clone with submodules
git clone --recursive git@github.com:Samfisheryu/vLLM-Prometheus.git
cd vLLM-Prometheus

# If already cloned without --recursive
git submodule update --init --recursive
```

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate proj-dmx
pip install -e transformers/  # Install local modified transformers
```

### Option 2: Pip

```bash
pip install -r requirements.txt
pip install -e transformers/  # Install local modified transformers
```

### Build C++ Extension

```bash
cd monitoring && make
```




## Quick Start

**Refer to [Quick_Start.ipynb](./Quick_Start.ipynb).**



## Run Benchmark

## Environment Variables for Benchmarks

| Variable | Default | Description |
|----------|---------|-------------|
| `MON_NATIVE_CALLBACK` | `1` | Use C++ callbacks (faster) |
| `MON_NATIVE_BATCH` | `1` | Batch hook submissions |
| `MON_NATIVE_TO_CPU` | `1` | Enable async GPU→CPU transfer |

### Args
```bash
steps: requests
warmup: warmup requests
decode-steps: decode token length
```

### profile_decode.py - Comprehensive Comparison

Compares multiple inference approaches (TransformerLens, HuggingFace, HookedGPT2Model) with profiling support:

```bash


# Basic run
MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 MON_NATIVE_BATCH=1 python benchmark/tests/profile_decode_qwen3.py --batch-size 1 --steps 1 --warmup 1 --collect-hidden --collect-attention --no-profile --dtype fp8

# With nsight profiling
MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 TL_ENABLE_NVTX=1 nsys profile --output=your_results_path/xxx --force-overwrite=true --trace=cuda,nvtx,osrt --sample=cpu --sampling-period=1000000 --cpuctxsw=process-tree --cuda-memory-usage=false  python benchmark/tests/profile_decode.py. --profile-dir your_results_dir/xxx. --batch-size 64  --decode-steps 64  --collect-hidden  --collect-attention  --steps 1  --warmup 1  --no-profile
```

**Tested configurations:**
- `transformer_lens` / `transformer_lens_cache` - Original TransformerLens
- `huggingface` / `huggingface_api` - Pure HuggingFace
- `hf_modified` / `hf_modified_hook` / `hf_modified_hook_async` - HookedGPT2Model with MonitoringEngine

### hf_modified_async_config_benchmark.py - Config Validation

Tests different MonitoringConfig settings (full capture vs sampled):

```bash
MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 MON_NATIVE_BATCH=1 python benchmark/tests/hf_modified_async_config_benchmark.py --batch-size 64 --steps 1 --warmup 1 --decode-steps 64 --collect-hidden --collect-attention
```

### hf_modified_async_config_token_stride_benchmark.py - Token Stride Impact

Measures performance impact of different `step_stride` values:

```bash
MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 MON_NATIVE_BATCH=1 python benchmark/tests/hf_modified_async_config_token_stride_benchmark.py --batch-size 64 --steps 1 --warmup 1 --decode-steps 64 --collect-hidden --collect-attention
```

Tests strides: `[1, 10, 30, 400]` - higher stride = fewer captures = faster

### hf_modified_async_config_request_stride_benchmark.py - Request Stride Impact

Measures performance impact of different `request_stride` values:

```bash
MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 MON_NATIVE_BATCH=1 python benchmark/tests/hf_modified_async_config_request_stride_benchmark.py --batch-size 64 --steps 10 --warmup 1 --decode-steps 64 --collect-hidden --collect-attention —no-profile
```

Tests strides: `[1, 2, 5, 100]` - higher stride = skip more requests

