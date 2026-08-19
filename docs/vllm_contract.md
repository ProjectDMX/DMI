# vLLM assumptions used by DMI

This file lists only the behavior that DMI expects from its pinned vLLM
version. If an upgrade changes one of these assumptions, the vLLM adapter or
monitored model implementations must be reviewed.

## Model runner

- DMI uses the V1 GPU model runner. The V2 runner is outside this contract.
- The worker lifecycle exposes `init_device`, `load_model`,
  `compile_or_warm_up_model`, `execute_model`, and `shutdown` in that order.
- `load_model` accepts the keyword-only `load_dummy_weights` argument and
  forwards its meaning unchanged.
- The V1 model runner exposes `_prepare_inputs` and
  `_determine_batch_execution_and_padding` with the signatures pinned by the
  adapter.
- `speculative_config` is absent.
- DP and DBO/ubatching are not enabled together. Any enabled parallel path
  preserves the layout, dispatch, and rank-ownership assumptions below.
- One worker process executes at most one monitored model forward at a time.

## Request layout

For a forward-producing `execute_model` call, vLLM is expected to:

1. update request state and finalize `input_batch`;
2. call `_prepare_inputs`;
3. call `_determine_batch_execution_and_padding`;
4. execute the model forward using the returned layout.

After `_prepare_inputs` returns:

- `input_batch.req_ids[:input_batch.num_reqs]` is the final packed request
  order used by model tensors;
- the ordered scheduled-token array passed to `_prepare_inputs` and
  `input_batch.num_computed_tokens_cpu` align element-for-element with those
  request IDs;
- each request contributes one contiguous token-row interval in that order;
- `scheduler_output.num_scheduled_tokens` contains the same request membership
  and counts, but its dictionary order is not the packed tensor order;
- the sum of scheduled-token counts is the real packed token-row count; and
- the input-batch arrays may be reused or mutated by later steps, so their
  values are valid only when snapshotted at this boundary.

For a prefix-cache hit, `num_computed_tokens_cpu` identifies the first token
executed by the current forward. Activations for the cached prefix are not
present in that forward.

Final logits contain one row per active request in packed request order.

## Dispatch and execution rows

- `_determine_batch_execution_and_padding` returns the final
  `CUDAGraphMode` and `BatchDescriptor` used by the following model forward.
- `BatchDescriptor.num_tokens` is the execution-row count after any vLLM
  padding. It is greater than or equal to the real packed token-row count.
- vLLM does not change request order or execution-row count between this
  return boundary and model forward without returning a corresponding new
  descriptor.
- The caller's eager request is preserved by dispatch.
- Calling vLLM's `cudagraph_dispatcher.dispatch` does not mutate model-runner,
  scheduler, allocator, or distributed state.
- Dispatcher decisions continue to reflect uniform decode, LoRA, encoder
  output, cascade attention, graph mode, and caller-eager conditions.

The only expected call to
`_determine_batch_execution_and_padding` before `_prepare_inputs` is the
PP+SP early dispatch. A later post-layout call still returns the descriptor
used by model forward.

For conservative execution bounds, configured scheduler maxima remain upper
bounds on scheduled tokens and active sequences. CUDA-graph capture ceilings
remain upper bounds on graph-padded execution rows, and SP padding continues
to round rows to the required TP multiple.

## Parallel layout

- All ranks participating in one forward use a compatible execution mode and
  descriptor shape.
- `get_pp_indices` describes the layer interval owned by each PP stage.
- Input-token and embedding work belongs to the first PP stage. Final
  residual, normalization, and logits work belongs to the last PP stage.
- Per-layer tensors belong to the PP stage that owns that layer.
- TP head, KV-head, and intermediate-dimension partitioning determines the
  local tensor shapes exposed by monitored model code.

## Monitored model interface

- Each monitored model exposes `get_hook_specs(model_wide=True)` as an unbound,
  all-layer inventory.
- The inventory order, shapes, dtypes, and ownership describe tensors produced
  by the real model forward.
- Monitored model variants preserve the constructor, forward, return-value,
  loader, mapper, compile, and parallel-layout semantics of their corresponding
  upstream vLLM implementations.
- A hooked tensor is the value used by the model, not a separately recomputed
  approximation.

## Compilation and graph replay

- vLLM/PyTorch compilation retains DMI producer custom-op nodes and their
  shared mutable-alias ordering dependency.
- CUDA-graph replay preserves captured tensor addresses and structural shapes
  while re-reading current row-count and chunk-size values.
- Persistent AOT-cache loading reconstructs the producer nodes with the same
  schemas and ordering semantics as the cold compiled graph.
- Graph execution uses the request order and execution-row count described by
  the dispatch result for that forward.

## Serving lifecycle

- vLLM can invoke a named worker method on every worker before server
  termination.
- Once request intake is stopped, no new model forward begins while
  `stop_monitoring` is running.
- Worker shutdown allows `stop_monitoring` to finish before worker processes
  and their distributed/runtime state are destroyed.
- Normal server shutdown alone is not assumed to provide a complete DMI
  storage flush.

## Upgrade boundary

Changes to the methods, call order, fields, or semantics above require an
adapter review even when names and signatures are unchanged. Changes confined
to unrelated vLLM paths or models that DMI does not integrate do not require a
DMI code change.
