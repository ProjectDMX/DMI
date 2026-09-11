# Model descriptors

A descriptor says what a model *is* — identity plus topology. It is what
DMI-configurator renders and validates against.

**You should not write one by hand.** The framework already knows the model,
and DMI reads a Hugging-Face-shaped config directly:

```bash
dmi describe-model ./Qwen3-8B --output qwen3-8b.yaml
```

`describe-model` accepts a model directory, a `config.json`, or a Hugging Face
model id, and handles the naming variants across model families
(`hidden_size`/`n_embd`, `num_hidden_layers`/`n_layer`, and so on).

Most of the time you do not need a descriptor file at all, because the
configurator takes the same sources directly:

```bash
dmi ui ./Qwen3-8B
dmi ui ./Qwen3-8B/config.json
dmi ui Qwen/Qwen3-8B          # needs `transformers` installed
```

A descriptor file is worth keeping when you want to configure a capture
somewhere the model is not present — picking layers on a laptop for a run that
happens on a cluster — or to override topology the extractor cannot read.

The descriptor is design-time only. At runtime DMI takes its shape from the
adapter's live `detect_model_shape(model)`, never from this file.

## What is here

`llama3-8b.yaml` — one worked example, matching what `describe-model` produces
for Llama 3 8B. Used by the test suite and handy for trying the UI without a
model checkout.
