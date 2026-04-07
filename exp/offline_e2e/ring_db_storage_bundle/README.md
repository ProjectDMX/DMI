# Ring DB Storage Bundle

This folder contains the scripts for the DMI `ring_db` storage ablation experiment.

## Files

- `run_ring_db_storage_e2e.py`
  - Main Python runner
- `run_ring_db_storage_e2e_qwen3_4b_local.sh`
  - Local wrapper script

## Default experiment

- model: `qwen3-4b`
- batch size: `64`
- prefill: `128`
- decode: `500`
- hook selection: `hidden-states,final_ln,logits`
- compile: enabled
- number of batches: `10`

## Modes

- `PROJ_DMI_MODE=ring_db`
  - Full ring + ClickHouse insertion
- `PROJ_DMI_MODE=ring_null`
  - Ring transport only, null sink

## Storage labels

- `STORAGE_LABEL=disk`
- `STORAGE_LABEL=tmpfs`

The storage label is only a tag in the output. The actual ClickHouse storage location must be switched outside the script.

## Example commands

```bash
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_db bash exp/offline_e2e/ring_db_storage_bundle/run_ring_db_storage_e2e_qwen3_4b_local.sh
STORAGE_LABEL=tmpfs PROJ_DMI_MODE=ring_db bash exp/offline_e2e/ring_db_storage_bundle/run_ring_db_storage_e2e_qwen3_4b_local.sh
STORAGE_LABEL=disk PROJ_DMI_MODE=ring_null bash exp/offline_e2e/ring_db_storage_bundle/run_ring_db_storage_e2e_qwen3_4b_local.sh
```
