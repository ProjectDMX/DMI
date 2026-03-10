#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include "ring_torch_op.h"

// Active ring engine pointer. Set via ring_set_active_engine() from Python
// activate()/deactivate(). Accessed only during CUDA graph CAPTURE (when this
// C++ impl body runs). During graph REPLAY only the captured cudaLaunchKernel
// args are re-used — this pointer is never read.
static ring_py::RingEnginePy* g_active_engine = nullptr;

void ring_set_active_engine(ring_py::RingEnginePy* e) {
    g_active_engine = e;
}

// Side-effect op: launches producer kernel to copy tensor bytes into ring
// buffer.  Void return + _register_effectful_op prevents DCE.
//
// Uses hook_no_notify() — the producer kernel only, no cudaLaunchHostFunc.
// The hostfunc callback (used by the legacy non-graph path to wake the drain
// thread) would be captured as a host node in the CUDA graph, causing an
// ~18μs GPU→CPU→GPU round-trip per hook per decode step.  The drain thread
// is instead notified via flush() at the end of each forward pass.
void ring_producer_impl(
    const at::Tensor& tensor, int64_t hook_type, int64_t hook_id)
{
    if (g_active_engine && tensor.is_cuda() && tensor.is_contiguous()) {
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        g_active_engine->hook_no_notify(
            reinterpret_cast<uint64_t>(tensor.data_ptr()),
            static_cast<uint64_t>(tensor.nbytes()),
            0,
            static_cast<uint32_t>(hook_type),
            static_cast<uint32_t>(hook_id),
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
