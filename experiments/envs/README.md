# Environment Requirements

Each baseline uses a different Python environment. The `.requirements.txt` files
contain `pip freeze` outputs for reproducibility.

## Environment Mapping

| Baseline | Python | Requirements File | Notes |
|----------|--------|-------------------|-------|
| vLLM Baseline | 3.10 | `dmi.requirements.txt` | Uses dmi-env packages via PYTHONPATH |
| DMI | 3.10 | `dmi.requirements.txt` | Same env as baseline + DMI code |
| vLLM-Hook | 3.12 | `vllm_hook.requirements.txt` | Separate env with vllm-hook plugin |
| TRT-LLM | 3.10 | `trtllm.requirements.txt` | TensorRT-LLM 1.2.0 |

## Key Packages

### vLLM Baseline & DMI (Python 3.10)
- `torch==2.10.0` (CUDA 12.x)
- `flashinfer-python==0.6.4`
- `vllm==0.17.2.dev0` (via PYTHONPATH, not pip-installed)
- `transformers==4.57.0.dev0` (DMI fork, via PYTHONPATH)

### vLLM-Hook (Python 3.12)
- `torch` + `flashinfer`
- `vllm_hook_plugins` (pip install -e from experiments/vLLM-Hook/)
- vLLM 0.17.0 (via PYTHONPATH)

### TRT-LLM (Python 3.10)
- `tensorrt_llm` (from NVIDIA pip index)
- `tensorrt==10.x`
- Requires MPI (OpenMPI 4.1.5)

## Installation Notes

1. **CUDA 12.x and NVIDIA H100** are required for all baselines.
2. `pip install -r <file>` may not work directly — many packages need
   CUDA-specific wheel URLs. Use the files as a reference for versions.
3. For TRT-LLM, follow NVIDIA's official installation guide and then
   apply patches from `experiments/TensorRT-LLM/`.
4. vLLM 0.17.0 is used via PYTHONPATH, not pip-installed. Download the
   source and point PYTHONPATH to it.
