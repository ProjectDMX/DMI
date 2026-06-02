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

void ring_set_active_engine(ring_py::RingEnginePy* e) {
    g_active_engine = e;
    // Deactivation must also disable capture: g_toggle_capture is process-global,
    // so a later engine/capture would otherwise keep recording nodes (#3).
    if (e == nullptr) ring_set_toggle_capture(false);
}

// CPU-direct flag.  When true, ring_producer_impl copies tensor to CPU
// and submits via submit_cpu_direct() instead of launching the producer
// kernel.  Set per-step from Python before the model forward.
// Lives in C++ so HookPoint.forward() needs no Python-level branching,
// keeping the compiled graph free of non-serializable objects and
// immune to torch.compile guard dropping.
static bool g_cpu_direct = false;

void ring_set_cpu_direct(bool enabled) {
    g_cpu_direct = enabled;
}

// Node-toggle capture flag.  When true, after launching the producer kernel the
// op records its just-added graph node (the capture tail dependency) into the
// active engine's toggle registry, so it can be toggled post-capture.  Default
// off -> hot path unchanged.
static bool g_toggle_capture = false;
void ring_set_toggle_capture(bool enabled) {
    g_toggle_capture = enabled;
}

// Side-effect op: either launches the producer kernel (normal ring path)
// or copies tensor to CPU and submits directly (cpu_direct path).
//
// The branch is invisible to torch.compile -- this C++ body only runs
// during CUDA graph CAPTURE or eager dispatch (never during graph REPLAY).
// During replay, only the captured kernel is replayed (cpu_direct is
// never True during capture because force_eager prevents graph replay
// for cpu_direct steps).
//
// Void return + _register_effectful_op prevents DCE at FX level.
// HookPoint.forward() returns x_cont (not original x) so inductor cannot
// DCE the .contiguous() copy + producer call for non-contiguous tensors.
void ring_producer_impl(
    const at::Tensor& tensor, int64_t hook_type, int64_t hook_id)
{
    if (!g_active_engine) { return; }
    if (hook_type >= 0 && hook_type < HOST_HOOK_MAX)
        g_host_calls[hook_type].fetch_add(1);

    if (g_cpu_direct) {
        // CPU-direct path: ring is full, sync D2H + submit to p2p pipeline.
        // force_eager is set so no CUDA graph is active.
        if (tensor.is_cuda()) {
            // NOTE: compute nbytes BEFORE std::move to avoid use-after-move.
            auto cpu_tensor = tensor.detach().cpu();
            uint64_t nbytes = static_cast<uint64_t>(cpu_tensor.nbytes());
            g_active_engine->submit_cpu_direct(
                std::move(cpu_tensor), nbytes);
        }
        return;
    }

    // Normal ring path: async D2D copy into ring buffer via producer kernel.
    if (tensor.is_cuda() && tensor.is_contiguous()) {
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        g_active_engine->hook_no_notify(
            reinterpret_cast<uint64_t>(tensor.data_ptr()),
            static_cast<uint64_t>(tensor.nbytes()),
            static_cast<uint32_t>(hook_type),
            reinterpret_cast<uint64_t>(stream.stream())
        );

        // Node-toggle: if capturing, record THIS producer's kernel node (the
        // current tail dependency) so it can be enabled/disabled post-capture.
        if (g_toggle_capture) {
            cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
            unsigned long long      cap_id = 0;
            cudaGraph_t             cap_graph = nullptr;
            const cudaGraphNode_t*  cap_deps  = nullptr;
            const cudaGraphEdgeData* cap_edges = nullptr;  // CUDA 13 signature
            size_t                  cap_nd    = 0;
            if (cudaStreamGetCaptureInfo(stream.stream(), &cap_st, &cap_id, &cap_graph,
                                         &cap_deps, &cap_edges, &cap_nd) == cudaSuccess
                && cap_st == cudaStreamCaptureStatusActive && cap_nd >= 1) {
                g_active_engine->register_capture_node(
                    reinterpret_cast<uint64_t>(cap_graph),
                    static_cast<int>(hook_type), static_cast<int>(hook_id),
                    reinterpret_cast<uint64_t>(cap_deps[cap_nd - 1]));
            }
        }
    }
}

TORCH_LIBRARY(ring, m) {
    m.def("producer(Tensor x, int hook_type, int hook_id) -> ()");
}

TORCH_LIBRARY_IMPL(ring, CUDA, m) {
    m.impl("producer", ring_producer_impl);
}
