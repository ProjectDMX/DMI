# proj-dmx

# Research Proposal

## Project Title

White-Box Observability for LLM Inference: An End-to-End Research System for Neuron-Level Runtime Capture, Storage, and Query

## 1. Abstract

LLM inference is now critical infrastructure, yet most monitoring systems focus on service-level and high-level metrics (latency, throughput, token counts, error rates) and cannot explain internal causes behind quality regressions, latency anomalies, or behavioral drift. This project proposes an end-to-end, scalable, research-oriented white-box observability system for real inference engines. With minimal runtime overhead, it captures internal runtime signals such as inter-layer activations, attention/MLP states, and KV cache behavior, stores them in a structured, queryable form, and enables retrieval, comparison, and forensic analysis. The goal is not only to "capture signals," but to answer: under industrial constraints, which signals are worth capturing, how to capture them at controlled cost, and how they can materially improve diagnosis and understanding.

## 2. Background and Motivation

### 2.1 Problem Context

- LLM inference systems are complex (batching, KV cache management, fused kernels, parallelism, compilation), which breaks the chain from observed symptom to internal cause.
- Research tools (interpretability and hook tooling) typically target offline analysis and do not meet online inference performance constraints.
- There is a clear gap: inference-time, model-internal observability that is research-grade in granularity but usable in real inference settings.

### 2.2 Motivation

We believe future LLM infrastructure needs a new white-box observability layer:

- Supports both lightweight continuous monitoring (baseline distributions) and triggered deep forensics.
- Turns internal model signals from lab-only artifacts into operational observability assets.

## 3. Related Work and Gaps

1. Service-level observability (tracing/metrics) is strong but lacks internal model evidence.
2. Interpretability and hook tooling can access activations but is far from production engines and constraints.
3. Inference engine debug features are often ad-hoc exports, not sustainable for continuous observability.
4. Gap summary: there is no research system that systematically answers:
   - Under inference constraints, how to control the cost of white-box signal capture?
   - Which internal signals provide the highest information value for diagnosis?
   - How to organize internal signals into queryable, comparable, reproducible research assets?

## 4. Research Questions and Hypotheses

### RQ1: Can neuron/tensor-level observability be achieved with acceptable overhead under real inference constraints?

- H1: A tiered strategy (summary statistics + triggered deep capture + low-rate forensics) can provide diagnostic value without major throughput/latency impact.

### RQ2: Which internal signals are most informative for quality regression, latency anomalies, and behavioral drift?

- H2: More is not always better. A small set of key locations (e.g., specific residual stream summaries, attention distribution features, KV cache behavior metrics) can achieve high value-to-cost.

### RQ3: How can internal runtime states become queryable, comparable, and reproducible research assets?

- H3: With unified signal semantics and indexing (request/token/layer/version alignment), query APIs can significantly reduce regression diagnosis time and improve reproducibility.

### RQ4: Where is the boundary between inference engine compatibility and observability granularity?

- H4: Under heavy optimization (fusion, graph execution), full per-layer tensors are unsustainable; explainable summaries and sampled observability remain feasible and useful.

## 5. Objectives

1. Design and implement an end-to-end white-box observability research prototype (not a final commercial product).
2. Define reusable signal semantics and capture strategies: continuous, triggered, and forensic tiers.
3. Provide empirical evaluation on overhead, diagnostic effectiveness, compatibility boundaries, and usability.
4. Produce research outputs: methodology, benchmarks, system prototype, and potentially open-source modules.

