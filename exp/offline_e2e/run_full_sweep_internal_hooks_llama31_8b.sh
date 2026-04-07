#!/usr/bin/env bash
set -euo pipefail

MODELS_CSV="llama3.1-8b" bash exp/offline_e2e/run_full_sweep_internal_hooks.sh "$@"

