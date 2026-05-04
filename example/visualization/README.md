# DMI visualization demo

Four mech-interp views of one prompt through Qwen3-0.6B, served from
ClickHouse via a Jupyter notebook.

## Run

Pre-requisite: project installed per the repo root `README.md`,
ClickHouse running, and `pip install matplotlib circuitsvis jupyter`.

```bash
python example/visualization/run_offload_hf.py
# (or run_offload_vllm.py to populate the demo_vllm slot.)
jupyter notebook example/visualization/visualize.ipynb
```

The notebook defaults to `MODEL_ID = "demo_hf"`.  Switch to
`"demo_vllm"` in the setup cell to render the vLLM run.  The notebook
reads from ClickHouse only -- no model loaded, no GPU needed.

## Prompt

```
When John and Mary went to the store, John gave Mary a
```

The canonical Indirect Object Identification (IOI) prompt.  The model
has to track that `Mary` already received the action, so the next
token should be a noun (`gift`, `book`, ...), not another name.

## What each plot shows

1. **Attention patterns** -- per (layer, head) heatmap of attention
   weights.  Hover for token labels.  HF only -- vLLM excludes
   attention weights.
2. **Residual-stream norm by layer** -- L2 norm of the residual stream
   per (layer, token).  One line per token.
3. **Per-token confidence** -- each generated token shaded by the
   model's top-1 probability when it picked it.
4. **Top-k alternative tokens** -- each generated token colored by
   its log-probability, with the model's top-10 alternatives on hover.

## Files

- `prompt.txt` -- edit and re-run an offload to swap.
- `run_offload_hf.py` / `run_offload_vllm.py` -- write to fixed
  `model_id="demo_hf"` / `"demo_vllm"`.  Re-running wipes only the
  script's own slot.
- `visualize.ipynb` -- 5 plot cells, output-stripped.
