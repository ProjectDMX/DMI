#!/bin/bash
#SBATCH --job-name=trtllm-qwen14b
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --output=trtllm-build-qwen14b-%j.out
#SBATCH --error=trtllm-build-qwen14b-%j.err

set -eo pipefail

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${WORK_ROOT}"

NVIDIA_BASE=envs/trtllm/lib/python3.10/site-packages/nvidia
NVIDIA_LIBS=$(find $NVIDIA_BASE -maxdepth 2 -name lib -type d ! -path "*/cu13/*" 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH=/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-x86_64/gcc-11.3.0/openmpi-4.1.5-h3d4fsbq2zpfqyhmle4c44k35mvpw2bp/lib:${NVIDIA_LIBS}:$(pwd)/envs/trtllm/lib/cublas13_only
export LD_PRELOAD=$(pwd)/envs/trtllm/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_mca_base_env_list="LD_PRELOAD,LD_LIBRARY_PATH"
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$(pwd)/.cache

PYTHON=envs/trtllm/bin/python3.10
ENGINE_ROOT=$(pwd)/trtllm_engines

mkdir -p "$ENGINE_ROOT"

MODEL_14B=$(ls -d hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
if [ -z "$MODEL_14B" ]; then
    echo "ERROR: Qwen3-14B model not found in hf_cache"
    exit 1
fi

echo "=== $(date) === Building Qwen3-14B engine ==="
echo "Model: $MODEL_14B"

$PYTHON -u -c "
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig
import glob, shutil, os

cfg = BuildConfig()
cfg.max_batch_size = 256
cfg.max_input_len = 4096
cfg.max_seq_len = 4096
cfg.enable_debug_output = True

dst = '$ENGINE_ROOT/qwen14b_v2'

llm = LLM(model='$MODEL_14B', build_config=cfg)
llm.save(dst)
llm.close()
print('Qwen3-14B engine saved')

print('Contents:', os.listdir(dst), flush=True)
"

echo "=== $(date) === Done ==="
ls -lh "$ENGINE_ROOT/qwen14b_debug/"
