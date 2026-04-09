#!/bin/bash
#SBATCH --job-name=phase1-fusion-barrier
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/ynn1999
PROJECT=${SCRATCH}/DMI/DMI

ENV_PYTHON=${SCRATCH}/envs/vllm-h100/bin/python3.10
SITE=${SCRATCH}/dmi-env/lib/python3.10/site-packages

export PYTHONPATH=${PROJECT}/integration/vllm:${PROJECT}:${PROJECT}/transformers/src:${SITE}
export LD_LIBRARY_PATH=${SCRATCH}/dmi-env/lib:${PROJECT}/libs/clickhouse-cpp/build/clickhouse${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_MODULE_LOADING=EAGER
export HF_HOME=${SCRATCH}/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=${SCRATCH}/.cache

cd "${PROJECT}"

MODEL=qwen3-4b \
BATCH_SIZE=64 \
PREFILL_TOKENS=128 \
DECODE_STEPS=10 \
WARMUP=5 \
REPEATS=5 \
PYTHON_BIN="${ENV_PYTHON}" \
RESULTS_ROOT=experiments/offline_inference/results/phase1_fusion_barrier \
  bash experiments/offline_inference/phase1_fusion_barrier_benchmark.sh
