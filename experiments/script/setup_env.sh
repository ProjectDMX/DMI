#!/bin/bash
# Environment setup guide for DMI experiments.
#
# This script prints instructions for setting up the required environments.
# Run: source setup_env.sh <baseline>
#
# Baselines: baseline, dmi, hook, trtllm

cat <<'GUIDE'
=====================================================================
DMI Experiment Environment Setup
=====================================================================

All experiments require NVIDIA H100 GPU(s) and CUDA 12.x.

1. WORK DIRECTORY
   All environments and data live under a single working directory:
     export WORK_DIR=~/scratch.zaoxing-prj
     cd $WORK_DIR

2. CONDA ENVIRONMENTS NEEDED

   a) vllm-h100: Python 3.10 + PyTorch 2.x + FlashInfer
      Used by: vLLM baseline, DMI
      Key packages: torch, flashinfer, triton

   b) dmi-env: DMI-specific dependencies
      Used by: DMI (via PYTHONPATH + site-packages)
      Key packages: torch, flashinfer (+ newer libstdc++)

   c) vllm-hook-env: vLLM-Hook plugin environment
      Used by: vLLM-Hook baseline
      Key packages: torch, vllm_hook_plugins (pip install -e)

   d) trtllm: TensorRT-LLM 1.2.0
      Used by: TRT-LLM baseline
      Key packages: tensorrt_llm, tensorrt
      NOTE: Apply patches from experiments/TensorRT-LLM submodule

3. VLLM 0.17.0
   Download and extract vLLM 0.17.0 source to $WORK_DIR/vllm-0.17.0/
   Used via PYTHONPATH (not pip installed).

4. HUGGINGFACE MODELS
   Download to $WORK_DIR/hf_cache/:
     - Qwen/Qwen3-4B
     - meta-llama/Llama-3.1-8B-Instruct
     - Qwen/Qwen3-14B

5. SAMPLED DATASETS
   Generate with: python experiments/script/sample_datasets.py
   Or use pre-generated files in experiments/sampled_datasets/

6. TRT-LLM ENGINES (for TRT-LLM baseline only)
   Build with: ./experiments/script/build_trtllm_engines.sh --model qwen4b
   Requires GPU and ~30 min per model.

7. APPLYING TRT-LLM PATCHES
   The TensorRT-LLM submodule (experiments/TensorRT-LLM) contains the
   patched files. Copy them to your pip site-packages:

     TRTLLM_SRC=experiments/TensorRT-LLM/tensorrt_llm
     TRTLLM_DST=$(python -c "import tensorrt_llm; print(tensorrt_llm.__path__[0])")

     # Build-time patch (needed before building engines):
     cp $TRTLLM_SRC/models/modeling_utils.py $TRTLLM_DST/models/

     # Runtime patches (needed for serve with D2H):
     cp $TRTLLM_SRC/llmapi/llm.py $TRTLLM_DST/llmapi/
     cp $TRTLLM_SRC/sampling_params.py $TRTLLM_DST/

=====================================================================
GUIDE
