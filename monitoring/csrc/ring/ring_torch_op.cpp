#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include "ring_torch_op.h"

// Active ring engine pointer. Set via ring_set_active_engine() from Python
// activate()/deactivate(). Accessed only during CUDA graph CAPTURE (when this
// C++ impl body runs). During graph REPLAY only the captured cudaLaunchKernel
// args are re-used -- this pointer is never read.
static ring_py::RingEnginePy* g_active_engine = nullptr;

// Host-side call counter per hook_type (diagnostic)
#include <cstdio>
#include <atomic>
#define HOST_HOOK_MAX 32
static std::atomic<uint64_t> g_host_calls[HOST_HOOK_MAX] = {};

void ring_diag_reset_host_counters() {
    for (int i = 0; i < HOST_HOOK_MAX; ++i) g_host_calls[i].store(0);
}

void ring_diag_print_host_counters() {
    uint64_t total = 0;
    fprintf(stderr, "[ring_torch_op] host calls:");
    for (int i = 0; i < HOST_HOOK_MAX; ++i) {
        uint64_t v = g_host_calls[i].load();
        if (v) { fprintf(stderr, " %d=%lu", i, v); total += v; }
    }
    fprintf(stderr, "  total=%lu\n", total);
}

// Node-toggle capture flag (process-global, like g_active_engine). Set true
// only during the warmup CUDA-graph capture window via ring_set_toggle_capture.
static bool g_toggle_capture = false;
void ring_set_toggle_capture(bool enabled) {
    g_toggle_capture = enabled;
}

void ring_set_active_engine(ring_py::RingEnginePy* e) {
    g_active_engine = e;
    // Deactivation must also disable capture: g_toggle_capture is process-global,
    // so a stale "true" after the engine is cleared would record into nothing (#3).
    if (e == nullptr) ring_set_toggle_capture(false);
}

// Three side-effect ops, one per use case:
//
//   ring::producer(x, hook_type, hook_id)
//     Static path; copies all of x.nbytes(); today's behavior.
//
//   ring::producer_prefix(x, row_count, row_bytes, hook_type, hook_id)
//     Reads row_count[0] from device at kernel start; copies
//     row_count[0] * row_bytes bytes from x.  Shared-scalar pattern:
//     multiple HookPoints may pass the SAME row_count tensor.
//
//   ring::producer_chunked(x, chunk_bytes, hook_type, hook_id)
//     K = chunk_bytes.numel(); source viewed as K equal chunks of
//     (x.nbytes() / K) bytes each; copies first chunk_bytes[i] bytes
//     of chunk i, packed contiguously.
//
// CUDA-graph capture contract: the chosen op, kernel launch args,
// and device-pointer args are all baked at trace time.  The *values*
// at the captured pointers are re-read each replay; the pointers,
// K, and row_bytes are not.  K and row_bytes being fixed is natural:
// they reflect structural properties tied to the captured shape
// signature (any change implies a different shape signature, which
// would trigger re-capture upstream).  Caller's responsibility to
// keep tensors at stable addresses with fixed numel.  Not enforced
// here.
//
// Void return + _register_effectful_op prevents DCE at FX level.
// HookPoint.forward() returns x_cont (not original x) so inductor
// cannot DCE the .contiguous() copy + producer call for
// non-contiguous tensors.
// `ring_payload` is declared `Tensor(a!)` (mutated) in the schema and
// is the same tensor (a view of the engine's GPU payload buffer) for
// every producer call from the same engine.  The annotation gives AOT
// autograd a real R/W dependency between successive producer calls,
// which inductor must preserve -- preventing the kernel-launch reorder
// observed under HF's `CompileConfig(mode="reduce-overhead",
// fullgraph=False)` decode compile.  The impl doesn't need to touch
// `ring_payload`; the kernel reaches the same memory via
// `g_active_engine`, so the annotation truthfully describes the
// effect.
void ring_producer_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    int64_t hook_type, int64_t hook_id)
{
    if (!g_active_engine) { return; }
    if (hook_type >= 0 && hook_type < HOST_HOOK_MAX)
        g_host_calls[hook_type].fetch_add(1);

    if (tensor.is_cuda() && tensor.is_contiguous()) {
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        g_active_engine->hook_no_notify(
            reinterpret_cast<uint64_t>(tensor.data_ptr()),
            static_cast<uint64_t>(tensor.nbytes()),
            static_cast<uint32_t>(hook_type),
            reinterpret_cast<uint64_t>(stream.stream()));

        // Node-toggle (option B: basic producer only -- toggle requires
        // gpu_padding_strip=False so every hook dispatches here, not to
        // producer_prefix/producer_chunked). If capturing, record THIS
        // producer's kernel node (the current tail dependency) so it can be
        // enabled/disabled post-capture via cudaGraphNodeSetEnabled.
        if (g_toggle_capture) {
            cudaStreamCaptureStatus  cap_st    = cudaStreamCaptureStatusNone;
            unsigned long long       cap_id    = 0;
            cudaGraph_t              cap_graph = nullptr;
            const cudaGraphNode_t*   cap_deps  = nullptr;
            const cudaGraphEdgeData* cap_edges = nullptr;  // CUDA 13 signature
            size_t                   cap_nd    = 0;
            if (cudaStreamGetCaptureInfo(stream.stream(), &cap_st, &cap_id, &cap_graph,
                                         &cap_deps, &cap_edges, &cap_nd) == cudaSuccess
                && cap_st == cudaStreamCaptureStatusActive && cap_nd >= 1) {
                // The producer is a single kernel launch, so the capture tail
                // dependency should be a kernel node. Validate it (fail-closed):
                // if the tail node is NOT a kernel (multi-op producer, capture
                // event-join, or unexpected topology), do NOT register a wrong
                // node -- flag an anomaly so set_active_hooks refuses to activate.
                cudaGraphNode_t node = cap_deps[cap_nd - 1];
                cudaGraphNodeType ntype;
                if (cudaGraphNodeGetType(node, &ntype) == cudaSuccess
                    && ntype == cudaGraphNodeTypeKernel) {
                    g_active_engine->register_capture_node(
                        reinterpret_cast<uint64_t>(cap_graph),
                        static_cast<int>(hook_type), static_cast<int>(hook_id),
                        reinterpret_cast<uint64_t>(node));
                } else {
                    g_active_engine->note_capture_anomaly();
                }
            }
        }
    }
}

void ring_producer_prefix_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& row_count,
    int64_t row_bytes,
    int64_t hook_type, int64_t hook_id)
{
    if (!g_active_engine) { return; }
    if (hook_type >= 0 && hook_type < HOST_HOOK_MAX)
        g_host_calls[hook_type].fetch_add(1);

    if (tensor.is_cuda() && tensor.is_contiguous()
        && row_count.defined() && row_count.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        g_active_engine->hook_no_notify_prefix(
            reinterpret_cast<uint64_t>(tensor.data_ptr()),
            static_cast<uint64_t>(tensor.nbytes()),
            reinterpret_cast<uint64_t>(row_count.data_ptr()),
            static_cast<uint64_t>(row_bytes),
            static_cast<uint32_t>(hook_type),
            reinterpret_cast<uint64_t>(stream.stream()));
    }
}

void ring_producer_chunked_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& chunk_bytes,
    int64_t hook_type, int64_t hook_id)
{
    if (!g_active_engine) { return; }
    if (hook_type >= 0 && hook_type < HOST_HOOK_MAX)
        g_host_calls[hook_type].fetch_add(1);

    if (tensor.is_cuda() && tensor.is_contiguous()
        && chunk_bytes.defined() && chunk_bytes.is_cuda()) {
        const uint32_t K = static_cast<uint32_t>(chunk_bytes.numel());
        if (K == 0) return;
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        g_active_engine->hook_no_notify_chunked(
            reinterpret_cast<uint64_t>(tensor.data_ptr()),
            static_cast<uint64_t>(tensor.nbytes()),
            reinterpret_cast<uint64_t>(chunk_bytes.data_ptr()),
            K,
            static_cast<uint32_t>(hook_type),
            reinterpret_cast<uint64_t>(stream.stream()));
    }
}

TORCH_LIBRARY(ring, m) {
    m.def("producer(Tensor(a!) ring_payload, Tensor x, "
          "int hook_type, int hook_id) -> ()");
    m.def("producer_prefix(Tensor(a!) ring_payload, Tensor x, "
          "Tensor row_count, int row_bytes, "
          "int hook_type, int hook_id) -> ()");
    m.def("producer_chunked(Tensor(a!) ring_payload, Tensor x, "
          "Tensor chunk_bytes, int hook_type, int hook_id) -> ()");
}

TORCH_LIBRARY_IMPL(ring, CUDA, m) {
    m.impl("producer",         ring_producer_impl);
    m.impl("producer_prefix",  ring_producer_prefix_impl);
    m.impl("producer_chunked", ring_producer_chunked_impl);
}
