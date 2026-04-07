#!/usr/bin/env bash
set -euo pipefail

MODELS_CSV="llama3.1-8b" bash experiments/offline_inference/run_full_sweep_internal_hooks.sh "$@"

