# Megatron-LM usage

DMI supports Megatron-LM training through the version-matched `DMI-Megatron-Integration` submodule. That integration pins the corresponding `Megatron-LM-DMI` fork, so a recursive DMI clone provides the complete source set required for this backend.

## Install the Megatron-LM backend

Use a dedicated environment and DMI checkout for Megatron-LM. Do not reuse a HuggingFace or vLLM environment.

The version-matched integration owns the framework-specific requirements, tested dependency configuration, Transformer Engine source selection, pinned Megatron fork installation, and import verification. Follow its [installation guide](../third_party/DMI-Megatron-Integration/docs/install.md).

In a recursive DMI checkout, the integration and its Megatron fork are already present, so skip that guide's clone step. Establish the final PyTorch, CUDA, Transformer Engine, and Megatron dependency versions first, then complete the DMI [core installation](install.md) from this checkout and install the integration from `third_party/DMI-Megatron-Integration/`.

The DMI native extension is built against the active PyTorch and CUDA ABI. Even if the DMI core installation was completed earlier, rebuild the extension in the final Megatron environment if a later package installation upgrades or replaces PyTorch or its NVIDIA CUDA runtime packages, or if the CUDA toolkit used for compilation changes. Installing Megatron or Transformer Engine does not by itself require a rebuild when that ABI remains unchanged.

```bash
make -C native clean
make -C native -j
```

## Enable DMI in a training run

Use the pinned fork's `pretrain_gpt.py` with the normal Megatron model, data, optimizer, and parallelism arguments. Enable DMI by appending its existing CLI options to that command:

```bash
torchrun --nproc_per_node="$NPROC_PER_NODE" \
  third_party/DMI-Megatron-Integration/third_party/megatron-lm/pretrain_gpt.py \
  <your normal Megatron training arguments> \
  --dmi-enable \
  --dmi-hook-selection router-summary \
  --dmi-model-id "$MODEL_ID" \
  --dmi-db-host "$DMX_DB_HOST" \
  --dmi-db-port "${DMX_DB_PORT:-9000}" \
  --dmi-db-database "$DMX_DB_DATABASE" \
  --dmi-clickhouse-table dmi_training_tensors
```

`--dmi-hook-selection` accepts a comma-separated integration hook selection. When it is omitted, the integration defaults to `router-summary`. Hook-specific topology requirements are validated during startup before training proceeds.

The DMI options can also be supplied through their corresponding `DMI_*` environment variables. CLI values take precedence over environment values. For example:

```bash
export DMI_ENABLE=1
export DMI_HOOK_SELECTION=router-summary
export DMI_MODEL_ID=my-training-run
export DMI_DB_HOST=127.0.0.1
export DMI_DB_PORT=9000
export DMI_DB_DATABASE=default
export DMI_CLICKHOUSE_TABLE=dmi_training_tensors
```

## Choose the output mode

Set `--dmi-db-host` or `DMI_DB_HOST` to write training records to ClickHouse. The integration creates and writes its schema-driven training tables through DMI's public storage API. Complete the ClickHouse setup in the [core installation guide](install.md) before starting the run.

Leave the database host empty for capture and transport without persistence:

```bash
export DMI_DB_HOST=
```

In either mode, use the same Megatron workload and parallelism arguments you would use without DMI. The integration is activated only when `--dmi-enable` or `DMI_ENABLE=1` is present.
