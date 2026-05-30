# Node-toggle de-risk probes

Standalone CUDA programs that de-risk the Axis-A node-toggle feature
(`cudaGraphNodeSetEnabled` on producer kernel nodes). They do **not** touch the
DMI build, ClickHouse, Python, or vLLM — each is a single self-contained `.cu`
that captures its own CUDA graph and runs directly on a local GPU.

Findings and full write-up: [`../node_toggle_investigation_report.md`](../node_toggle_investigation_report.md).

## `probe_dualring_toggle.cu` (primary)
Links the **real** `monitoring/csrc/ring/producer.cu` + dual-ring (`AllocatedRing`).
Verifies (Q2) that disabling producer nodes post-capture leaves the dual-ring
consistent/aligned, and measures (Q1) per-replay overhead + reconfigure cost.

```bash
nvcc -std=c++17 -arch=native -O2 -I../../monitoring/csrc \
     probe_dualring_toggle.cu ../../monitoring/csrc/ring/producer.cu -o probe_dr
CUDA_MODULE_LOADING=EAGER ./probe_dr
# scale the node count (e.g. to the ~145-hook config):  nvcc ... -DNPROD=145 ...
```

## `probe_node_toggle.cu` (synthetic)
Isolates the `cudaGraphNodeSetEnabled` primitive with a synthetic kernel; also
times the `null_mode` `cudaMemcpyToSymbol` toggle path.

```bash
nvcc -std=c++17 -arch=native -O2 probe_node_toggle.cu -o probe
CUDA_MODULE_LOADING=EAGER ./probe
```

Both timed with host wall-clock (`std::chrono`) for host-side API calls and
CUDA events for GPU replay time. Compiled binaries are git-ignored — rebuild
with the lines above.
