# Benchmarks

End-to-end results for DMI's capture + transport pipeline against
observation-enabled baselines.

Baselines:

- **HuggingFace Ideal** — vanilla HF `generate`, no observation (used as 1.0)
- **HF Built-in Extraction** — HF's `output_hidden_states=True` / `output_attentions=True`
- **HF Stepwise Extraction** — HF returning internals one step at a time
- **Torch Hooks** — Python `register_forward_hook` instrumentation
- **NNsight** — [NNsight](https://nnsight.net/) tracing layer
- **vLLM Hook**, **TRT-LLM (Debug API)** — synchronous observation baselines
- **vLLM w/o Monitor** — vanilla vLLM, no observation

## Offline throughput

Setup: 1 hidden-state hook per layer + final-LN + logits (38 / 34 / 42 hooks
total on Qwen3-4B / Llama-3.1-8B / Qwen3-14B). Normalized to HuggingFace Ideal.

<p align="center">
  <img src="../Figures/offline_hs_logits_real.png" alt="Offline throughput with limited hooks, normalized to HF ideal" width="100%" />
</p>

DMI stays close to the HF-ideal line across all configurations. Python-callback
baselines (NNsight, Torch Hooks) collapse as hook count or batch size grows;
several configurations go OOM (red ×) at large batch sizes because they retain
captured tensors in the inference memory pool. HF's built-in extraction path
is bottlenecked similarly — it materializes internals on the hot path.

## Online serving — TPOT

Setup: vLLM serve, varying request rate on ShareGPT and WildChat.

<p align="center">
  <img src="../Figures/tpot_comparison.png" alt="Online TPOT vs request rate" width="100%" />
</p>

DMI tracks the no-monitor baseline closely. The synchronous hook/debug baselines
(vLLM Hook, TRT-LLM Debug API) saturate at substantially lower request rates
because their capture paths block the hot stream.

## Reproducing the paper experiments

The paper evaluation scripts live under [`../experiments`](../experiments).
They are the source of the offline and online results shown above.

| Experiment group | Directory | What it contains |
|---|---|---|
| Offline inference and microbenchmarks | [`../experiments/offline_inference`](../experiments/offline_inference) | HuggingFace / DMI / HF extraction / Torch Hooks / NNsight runs, plus hook-count, TP, storage, max-batch, and step-breakdown microbenchmarks |
| Online serving | [`../experiments/online_serving`](../experiments/online_serving) | vLLM baseline, DMI, vLLM-Hook, TRT-LLM Debug API, sampled datasets, benchmark driver, and plotting scripts |

Start with the README in each directory:

- [`../experiments/offline_inference/README.md`](../experiments/offline_inference/README.md)
- [`../experiments/online_serving/README.md`](../experiments/online_serving/README.md)

For local setup of this repo's native backend and ClickHouse sink, see
[`install.md`](install.md). For simple API examples outside the full paper
reproduction path, see [`huggingface.md`](huggingface.md) and [`vllm.md`](vllm.md).
