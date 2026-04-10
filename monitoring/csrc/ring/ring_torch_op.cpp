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
    }
}

TORCH_LIBRARY(ring, m) {
    m.def("producer(Tensor x, int hook_type, int hook_id) -> ()");
}

TORCH_LIBRARY_IMPL(ring, CUDA, m) {
    m.impl("producer", ring_producer_impl);
}
