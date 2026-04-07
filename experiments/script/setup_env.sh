#!/bin/bash
# Environment setup for DMI experiments.
#
# Usage:
#   bash setup_env.sh baseline    # Create env for vLLM baseline & DMI
#   bash setup_env.sh hook        # Create env for vLLM-Hook
#   bash setup_env.sh trtllm      # Create env for TRT-LLM
#   bash setup_env.sh models      # Download HuggingFace models
#   bash setup_env.sh datasets    # Generate sampled datasets
#   bash setup_env.sh all         # Do everything
#
# Prerequisites: conda, CUDA 12.x, NVIDIA GPU (H100 recommended)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR=${WORK_DIR:-$(dirname "$REPO_ROOT")}

echo "REPO_ROOT: $REPO_ROOT"
echo "WORK_DIR:  $WORK_DIR"

setup_baseline() {
    echo ""
    echo "===== Setting up vLLM Baseline & DMI environment ====="
    echo ""

    conda create -n vllm-exp python=3.10 -y
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate vllm-exp

    # PyTorch (CUDA 12)
    pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

    # FlashInfer
    pip install flashinfer-python==0.6.4

    # vLLM dependencies (not vLLM itself — we use source via PYTHONPATH)
    pip install transformers==4.57.0 tokenizers sentencepiece
    pip install openai aiohttp fastapi uvicorn ray
    pip install numpy pandas matplotlib

    # DMI native extension
    cd "$REPO_ROOT"
    pip install -e .

    echo ""
    echo "Done! To use:"
    echo "  conda activate vllm-exp"
    echo "  ./run_vllm_baseline.sh --model qwen4b --rates '1 2 4'"
    echo "  ./run_dmi.sh --model qwen4b --rates '1 2 4'"
    echo ""
    echo "NOTE: vLLM 0.17.0 source is used via PYTHONPATH (set automatically by scripts)."
    echo "If you don't have it, download from: https://github.com/vllm-project/vllm/archive/refs/tags/v0.17.0.tar.gz"
    echo "Extract to: $WORK_DIR/vllm-0.17.0/"
}

setup_hook() {
    echo ""
    echo "===== Setting up vLLM-Hook environment ====="
    echo ""

    conda create -n hook-exp python=3.12 -y
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate hook-exp

    # PyTorch (CUDA 12)
    pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

    # FlashInfer
    pip install flashinfer-python==0.6.4

    # vLLM dependencies
    pip install transformers tokenizers sentencepiece
    pip install openai aiohttp fastapi uvicorn ray

    # Install vLLM-Hook plugin
    cd "$REPO_ROOT/experiments/vLLM-Hook/vllm_hook_plugins"
    pip install -e .

    echo ""
    echo "Done! To use:"
    echo "  conda activate hook-exp"
    echo "  ./run_vllm_hook.sh --model qwen4b --rates '1 2 4'"
    echo ""
    echo "NOTE: --enforce-eager is required (CUDA graphs bypass Python hooks)."
}

setup_trtllm() {
    echo ""
    echo "===== Setting up TRT-LLM environment ====="
    echo ""

    conda create -n trtllm-exp python=3.10 -y
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate trtllm-exp

    # TensorRT-LLM (includes torch)
    pip install tensorrt_llm==1.2.0

    # Benchmark client dependencies
    pip install openai aiohttp transformers tokenizers

    # Apply patches
    echo ""
    echo "Applying TRT-LLM patches..."
    TRTLLM_SRC="$REPO_ROOT/experiments/TensorRT-LLM/tensorrt_llm"
    TRTLLM_DST=$(python -c "import tensorrt_llm; print(tensorrt_llm.__path__[0])")

    cp "$TRTLLM_SRC/models/modeling_utils.py" "$TRTLLM_DST/models/"
    cp "$TRTLLM_SRC/llmapi/llm.py" "$TRTLLM_DST/llmapi/"
    cp "$TRTLLM_SRC/sampling_params.py" "$TRTLLM_DST/"
    echo "Patches applied to: $TRTLLM_DST"

    echo ""
    echo "Done! To use:"
    echo "  conda activate trtllm-exp"
    echo "  # Build engines first:"
    echo "  ./build_trtllm_engines.sh --model qwen4b"
    echo "  # Then run benchmark:"
    echo "  ./run_trtllm_d2h.sh --model qwen4b --rates '1 2 4'"
    echo ""
    echo "NOTE: MPI is required. Install OpenMPI or set MPIRUN=/path/to/mpirun"
}

download_models() {
    echo ""
    echo "===== Downloading HuggingFace models ====="
    echo ""

    export HF_HOME=${WORK_DIR}/hf_cache
    python -c "
from huggingface_hub import snapshot_download
for model in ['Qwen/Qwen3-4B', 'meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen3-14B']:
    print(f'Downloading {model}...')
    snapshot_download(model)
    print(f'  Done: {model}')
"
    echo "Models saved to: $HF_HOME"
}

generate_datasets() {
    echo ""
    echo "===== Generating sampled datasets ====="
    echo ""

    python "$SCRIPT_DIR/sample_datasets.py" \
        --output-dir "$REPO_ROOT/experiments/sampled_datasets" \
        --num-samples 500 \
        --seeds 42 123 456

    echo "Datasets saved to: $REPO_ROOT/experiments/sampled_datasets/"
}

# ── Main ────────────────────────────────────────────────────────────
case "${1:-help}" in
    baseline) setup_baseline ;;
    hook)     setup_hook ;;
    trtllm)   setup_trtllm ;;
    models)   download_models ;;
    datasets) generate_datasets ;;
    all)
        setup_baseline
        setup_hook
        setup_trtllm
        download_models
        generate_datasets
        ;;
    *)
        echo "Usage: bash setup_env.sh {baseline|hook|trtllm|models|datasets|all}"
        echo ""
        echo "  baseline  - Create conda env for vLLM baseline & DMI (Python 3.10)"
        echo "  hook      - Create conda env for vLLM-Hook (Python 3.12)"
        echo "  trtllm    - Create conda env for TRT-LLM + apply patches (Python 3.10)"
        echo "  models    - Download HuggingFace models (Qwen3-4B, Llama-3.1-8B, Qwen3-14B)"
        echo "  datasets  - Generate sampled datasets (ShareGPT, WildChat)"
        echo "  all       - Do everything"
        echo ""
        echo "See experiments/envs/*.requirements.txt for exact package versions."
        ;;
esac
