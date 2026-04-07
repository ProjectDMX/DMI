# HS+Logits Full Sweep Bundle

This folder contains the entry scripts and Python runners needed for the offline `hs_logits` full sweep experiments.

## Entry sbatch scripts

- `run_full_sweep_hs_logits.sbatch`
  - Full `hs_logits` sweep for Qwen models (`qwen3-4b`, `qwen3-14b`)
- `run_full_sweep_hs_logits_llama31_8b.sbatch`
  - Full `hs_logits` sweep for `llama3.1-8b`

## Python runners used by the sweep

- `run_hf_upper_bound.py`
- `run_hf_monitor.py`
- `run_hf_monitor_manual.py`
- `run_proj_dmi.py`
- `run_torch_hooks.py`
- `run_nnsight.py`
- `common.py`

## Notes

- These scripts are copied from `exp/offline_e2e/`.
- The sbatch files still expect the original project layout and data paths under the repo, e.g. `benchmark/data/offline_e2e/...`.
- If you hand this folder to someone else inside the same repo checkout, they can run the sbatch scripts directly after adjusting environment paths if needed.
