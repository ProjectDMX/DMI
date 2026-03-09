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
// buffer.  Returns void so AOT autograd's functionalization pass never sees
// alias annotations on custom ops (which it rejects).  Marked effectful via
// _register_effectful_op(_EffectType.ORDERED) in Python so inductor/FX cannot
// DCE the node even when its output is unused.
//
// CUDA graph capture: the cudaLaunchKernel call is recorded once; replay
// re-issues it with the same stream every decode step — no Python overhead.
void ring_producer_impl(
    const at::Tensor& tensor, int64_t hook_type, int64_t hook_id)
{
    if (g_active_engine && tensor.is_cuda() && tensor.is_contiguous()) {
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        g_active_engine->hook(
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
