#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include "ring_torch_op.h"
#include "producer.cuh"

#include <limits>

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

namespace {

const int32_t* checked_record_gate(
    const c10::optional<at::Tensor>& emit_gate,
    const at::Tensor& tensor) {
    if (!emit_gate.has_value() || !emit_gate->defined()) return nullptr;
    TORCH_CHECK(emit_gate->is_cuda(),
                "record producer emit_gate must be a CUDA tensor");
    TORCH_CHECK(emit_gate->is_contiguous(),
                "record producer emit_gate must be contiguous");
    TORCH_CHECK(emit_gate->scalar_type() == at::kInt,
                "record producer emit_gate must have dtype int32");
    TORCH_CHECK(emit_gate->numel() == 1,
                "record producer emit_gate must contain one element");
    TORCH_CHECK(emit_gate->device() == tensor.device(),
                "record producer emit_gate must be on the payload device");
    return emit_gate->data_ptr<int32_t>();
}

int32_t checked_emit_value(int64_t emit_value) {
    TORCH_CHECK(emit_value >= std::numeric_limits<int32_t>::min() &&
                    emit_value <= std::numeric_limits<int32_t>::max(),
                "record producer emit_value must fit in int32");
    return static_cast<int32_t>(emit_value);
}

void checked_record_tensor(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(),
                "record producer payload must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(),
                "record producer payload must be contiguous");
}

void checked_record_index_tensor(const at::Tensor& value,
                                 const at::Tensor& payload,
                                 const char* name) {
    TORCH_CHECK(value.defined() && value.is_cuda(), name,
                " must be a CUDA tensor");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.scalar_type() == at::kLong, name,
                " must have dtype int64");
    TORCH_CHECK(value.device() == payload.device(), name,
                " must be on the payload device");
}

}  // namespace

void ring_record_producer_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const c10::optional<at::Tensor>& emit_gate,
    int64_t emit_value) {
    if (!g_active_engine) return;
    checked_record_tensor(tensor);
    const int32_t* gate = checked_record_gate(emit_gate, tensor);
    auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    g_active_engine->record_no_notify(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        static_cast<uint64_t>(tensor.nbytes()),
        reinterpret_cast<uint64_t>(gate),
        checked_emit_value(emit_value),
        reinterpret_cast<uint64_t>(stream.stream()));
}

void ring_record_producer_prefix_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& row_count,
    int64_t row_bytes,
    const c10::optional<at::Tensor>& emit_gate,
    int64_t emit_value) {
    if (!g_active_engine) return;
    checked_record_tensor(tensor);
    checked_record_index_tensor(row_count, tensor, "record row_count");
    TORCH_CHECK(row_count.numel() == 1,
                "record row_count must contain one element");
    TORCH_CHECK(row_bytes > 0,
                "record row_bytes must be positive");
    const int32_t* gate = checked_record_gate(emit_gate, tensor);
    auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    g_active_engine->record_no_notify_prefix(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        static_cast<uint64_t>(tensor.nbytes()),
        reinterpret_cast<uint64_t>(row_count.data_ptr()),
        static_cast<uint64_t>(row_bytes),
        reinterpret_cast<uint64_t>(gate),
        checked_emit_value(emit_value),
        reinterpret_cast<uint64_t>(stream.stream()));
}

void ring_record_producer_chunked_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& chunk_bytes,
    const c10::optional<at::Tensor>& emit_gate,
    int64_t emit_value) {
    if (!g_active_engine) return;
    checked_record_tensor(tensor);
    checked_record_index_tensor(chunk_bytes, tensor, "record chunk_bytes");
    TORCH_CHECK(chunk_bytes.dim() == 1,
                "record chunk_bytes must be one-dimensional");
    TORCH_CHECK(chunk_bytes.numel() > 0 &&
                    chunk_bytes.numel() <= ring::PRODUCER_MAX_K,
                "record chunk count must be in [1, PRODUCER_MAX_K]");
    TORCH_CHECK(tensor.nbytes() % chunk_bytes.numel() == 0,
                "record payload bytes must divide evenly into chunks");
    const int32_t* gate = checked_record_gate(emit_gate, tensor);
    auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    g_active_engine->record_no_notify_chunked(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        static_cast<uint64_t>(tensor.nbytes()),
        reinterpret_cast<uint64_t>(chunk_bytes.data_ptr()),
        static_cast<uint32_t>(chunk_bytes.numel()),
        reinterpret_cast<uint64_t>(gate),
        checked_emit_value(emit_value),
        reinterpret_cast<uint64_t>(stream.stream()));
}

void ring_record_producer_seq_prefix_pack_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& valid_count,
    const at::Tensor& valid_prefix_sum,
    int64_t feature_bytes,
    const c10::optional<at::Tensor>& emit_gate,
    int64_t emit_value) {
    if (!g_active_engine) return;
    checked_record_tensor(tensor);
    checked_record_index_tensor(valid_count, tensor, "record valid_count");
    checked_record_index_tensor(
        valid_prefix_sum, tensor, "record valid_prefix_sum");
    TORCH_CHECK(valid_count.dim() == 1 && valid_count.numel() > 0,
                "record valid_count must be a non-empty one-dimensional tensor");
    TORCH_CHECK(valid_prefix_sum.dim() == 1 &&
                    valid_prefix_sum.numel() == valid_count.numel() + 1,
                "record valid_prefix_sum length must equal batch + 1");
    TORCH_CHECK(tensor.dim() >= 2 && tensor.size(1) == valid_count.numel(),
                "record sequence-pack payload must be [S, B, ...]");
    TORCH_CHECK(feature_bytes > 0,
                "record sequence-pack feature_bytes must be positive");
    TORCH_CHECK(tensor.nbytes() %
                    (valid_count.numel() * feature_bytes) == 0,
                "record sequence-pack payload has incompatible feature bytes");
    const int32_t* gate = checked_record_gate(emit_gate, tensor);
    auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    g_active_engine->record_no_notify_seq_prefix_pack(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        static_cast<uint64_t>(tensor.nbytes()),
        reinterpret_cast<uint64_t>(valid_count.data_ptr()),
        reinterpret_cast<uint64_t>(valid_prefix_sum.data_ptr()),
        static_cast<uint32_t>(valid_count.numel()),
        static_cast<uint64_t>(feature_bytes),
        reinterpret_cast<uint64_t>(gate),
        checked_emit_value(emit_value),
        reinterpret_cast<uint64_t>(stream.stream()));
}

void ring_record_producer_segmented_pack_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& segment_start,
    const at::Tensor& segment_end,
    int64_t feature_bytes,
    const c10::optional<at::Tensor>& emit_gate,
    int64_t emit_value) {
    if (!g_active_engine) return;
    checked_record_tensor(tensor);
    checked_record_index_tensor(segment_start, tensor, "record segment_start");
    checked_record_index_tensor(segment_end, tensor, "record segment_end");
    TORCH_CHECK(segment_start.dim() == 1 && segment_start.numel() > 0,
                "record segment_start must be a non-empty one-dimensional tensor");
    TORCH_CHECK(segment_end.dim() == 1 &&
                    segment_end.numel() == segment_start.numel(),
                "record segment_start/end lengths must match");
    TORCH_CHECK(feature_bytes > 0,
                "record segmented-pack feature_bytes must be positive");
    TORCH_CHECK(tensor.nbytes() % feature_bytes == 0,
                "record segmented-pack payload has incompatible feature bytes");
    const int32_t* gate = checked_record_gate(emit_gate, tensor);
    auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    g_active_engine->record_no_notify_segmented_pack(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        static_cast<uint64_t>(tensor.nbytes()),
        reinterpret_cast<uint64_t>(segment_start.data_ptr()),
        reinterpret_cast<uint64_t>(segment_end.data_ptr()),
        static_cast<uint32_t>(segment_start.numel()),
        static_cast<uint64_t>(feature_bytes),
        reinterpret_cast<uint64_t>(gate),
        checked_emit_value(emit_value),
        reinterpret_cast<uint64_t>(stream.stream()));
}

TORCH_LIBRARY(ring, m) {
    m.def("producer(Tensor(a!) ring_payload, Tensor x, "
          "int hook_type, int hook_id) -> ()");
    m.def("producer_prefix(Tensor(a!) ring_payload, Tensor x, "
          "Tensor row_count, int row_bytes, "
          "int hook_type, int hook_id) -> ()");
    m.def("producer_chunked(Tensor(a!) ring_payload, Tensor x, "
          "Tensor chunk_bytes, int hook_type, int hook_id) -> ()");
    m.def("record_producer(Tensor(a!) ring_payload, Tensor x, "
          "Tensor? emit_gate=None, int emit_value=0) -> ()");
    m.def("record_producer_prefix(Tensor(a!) ring_payload, Tensor x, "
          "Tensor row_count, int row_bytes, Tensor? emit_gate=None, "
          "int emit_value=0) -> ()");
    m.def("record_producer_chunked(Tensor(a!) ring_payload, Tensor x, "
          "Tensor chunk_bytes, Tensor? emit_gate=None, int emit_value=0) -> ()");
    m.def("record_producer_seq_prefix_pack(Tensor(a!) ring_payload, Tensor x, "
          "Tensor valid_count, Tensor valid_prefix_sum, int feature_bytes, "
          "Tensor? emit_gate=None, int emit_value=0) -> ()");
    m.def("record_producer_segmented_pack(Tensor(a!) ring_payload, Tensor x, "
          "Tensor segment_start, Tensor segment_end, int feature_bytes, "
          "Tensor? emit_gate=None, int emit_value=0) -> ()");
}

TORCH_LIBRARY_IMPL(ring, CUDA, m) {
    m.impl("producer",         ring_producer_impl);
    m.impl("producer_prefix",  ring_producer_prefix_impl);
    m.impl("producer_chunked", ring_producer_chunked_impl);
    m.impl("record_producer", ring_record_producer_impl);
    m.impl("record_producer_prefix", ring_record_producer_prefix_impl);
    m.impl("record_producer_chunked", ring_record_producer_chunked_impl);
    m.impl("record_producer_seq_prefix_pack",
           ring_record_producer_seq_prefix_pack_impl);
    m.impl("record_producer_segmented_pack",
           ring_record_producer_segmented_pack_impl);
}
