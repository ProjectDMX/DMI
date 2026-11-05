#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include "ring_torch_op.h"
#include "producer.cuh"
#include "toggle_device.h"   // dmx_capture_info

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

// Producer kernels launched by the ring producer ops, eager or captured.  The
// CUDA-graph replay guard compares it across a capture to tell a graph that
// captured producers from one that did not (dmi.node_toggle).
// Every producer op (static, prefix, chunked, gather, head, fused) counts
// each kernel it launches through count_producer_launch(), once per launch: a
// fused rows+heads node is one launch.  An op that launches without counting
// would make a capture holding only that op look producer-free, so its graph
// would skip the device-toggle upload and the replay guard.
static std::atomic<uint64_t> g_producer_launches{0};

static inline void count_producer_launch() {
    g_producer_launches.fetch_add(1, std::memory_order_relaxed);
}

uint64_t ring_producer_launch_count() {
    return g_producer_launches.load(std::memory_order_relaxed);
}

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

// Node-toggle capture flag (see ring_torch_op.h).
static bool g_toggle_capture = false;

void ring_set_toggle_capture(bool enabled) {
    g_toggle_capture = enabled;
}

void ring_set_active_engine(ring_py::RingEnginePy* e) {
    g_active_engine = e;
    // A stale capture flag after the engine is cleared would record into
    // nothing.
    if (e == nullptr) g_toggle_capture = false;
}

// Node-toggle capture: record THIS producer's kernel node (the capture's
// current tail dependency) so it can be enabled or disabled after
// instantiation with cudaGraphNodeSetEnabled.  No-op outside the capture
// window.
//   supported = true   (static and prefix producers): each is a single kernel
//     launch, so the tail dependency is its kernel node.  Validated: a
//     non-kernel tail (a multi-op producer, a capture event join) must not be
//     registered as the wrong node.
//   supported = false  (chunked producer): the toggle does not manage it.
//     Recorded as an anomaly so activation fails loud instead of leaving a
//     producer that fires while the gate filters its metadata.
// Every state where a producer fired under capture but its node could not be
// recorded is an anomaly.  The one silent case is a stream that is not
// actively capturing: an eager warmup run, not a graph node.
static void dmx_record_capture_node(cudaStream_t stream, int hook_type,
                                    int hook_id, bool supported) {
    if (!g_toggle_capture || !g_active_engine) return;
    cudaStreamCaptureStatus  cap_st    = cudaStreamCaptureStatusNone;
    unsigned long long       cap_id    = 0;
    cudaGraph_t              cap_graph = nullptr;
    const cudaGraphNode_t*   cap_deps  = nullptr;
    size_t                   cap_nd    = 0;
    cudaError_t ce = dmx_capture_info(stream, &cap_st, &cap_id, &cap_graph,
                                      &cap_deps, &cap_nd);
    if (ce != cudaSuccess) {
        // A healthy non-capturing stream returns success with status None,
        // so this is a genuine error: the node cannot be confirmed.
        g_active_engine->note_capture_anomaly();
        return;
    }
    if (cap_st != cudaStreamCaptureStatusActive) return;  // eager run, not a node
    if (cap_nd < 1 || !supported) {
        g_active_engine->note_capture_anomaly();
        return;
    }
    cudaGraphNode_t node = cap_deps[cap_nd - 1];
    cudaGraphNodeType ntype;
    if (cudaGraphNodeGetType(node, &ntype) == cudaSuccess
        && ntype == cudaGraphNodeTypeKernel) {
        g_active_engine->register_capture_node(
            reinterpret_cast<uint64_t>(cap_graph), hook_type, hook_id,
            reinterpret_cast<uint64_t>(node));
    } else {
        g_active_engine->note_capture_anomaly();
    }
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
        count_producer_launch();
        dmx_record_capture_node(stream.stream(), static_cast<int>(hook_type),
                                static_cast<int>(hook_id), /*supported=*/true);
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
        count_producer_launch();
        dmx_record_capture_node(stream.stream(), static_cast<int>(hook_type),
                                static_cast<int>(hook_id), /*supported=*/true);
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
        count_producer_launch();
        // Not toggle-managed: fail closed if captured inside the window.
        dmx_record_capture_node(stream.stream(), static_cast<int>(hook_type),
                                static_cast<int>(hook_id), /*supported=*/false);
    }
}

// A producer row table ([n, src0, len0, ...] int64) as the kernels read it:
// on the HookPoint tensor's device, contiguous, and holding at least its
// segment count.  A wrong table is a caller bug: refuse it rather than skip
// the producer, which would leave its reserved task unpublished.  The count
// word itself is never read here (that would be a host sync per launch and
// would bake a capture-time value into graphs); the kernels receive
// table.numel() and bound the segments they read by the table's storage
// (row_table.h, row_table_segment_count).
static void check_row_table(const at::Tensor& table, const at::Tensor& tensor,
                            const char* op) {
    TORCH_CHECK(table.is_cuda() && table.device() == tensor.device(),
                op, ": row table must be on the HookPoint tensor's device");
    TORCH_CHECK(table.scalar_type() == at::kLong && table.is_contiguous(),
                op, ": row table must be contiguous int64");
    TORCH_CHECK(table.numel() >= 1, op, ": row table must hold a segment count");
}

// ring::producer_gather(ring_payload, x, seg_table, row_bytes, hook_type,
//                       hook_id)
//   Copies the row runs listed in seg_table (producer.cuh table contract),
//   packed.  A single kernel launch, so it records its node like prefix.
void ring_producer_gather_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& seg_table,
    int64_t row_bytes,
    int64_t hook_type, int64_t hook_id)
{
    if (!g_active_engine) { return; }
    if (hook_type >= 0 && hook_type < HOST_HOOK_MAX)
        g_host_calls[hook_type].fetch_add(1);

    if (tensor.is_cuda() && tensor.is_contiguous() && seg_table.defined()) {
        check_row_table(seg_table, tensor, "ring::producer_gather");
        TORCH_CHECK(row_bytes > 0,
                    "ring::producer_gather: row_bytes must be positive");
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        g_active_engine->hook_no_notify_gather(
            reinterpret_cast<uint64_t>(tensor.data_ptr()),
            static_cast<uint64_t>(tensor.nbytes()),
            reinterpret_cast<uint64_t>(seg_table.data_ptr()),
            static_cast<int64_t>(seg_table.numel()),
            static_cast<uint64_t>(row_bytes),
            static_cast<uint32_t>(hook_type),
            reinterpret_cast<uint64_t>(stream.stream()));
        count_producer_launch();
        // Without this, a gather producer captured under the toggle would
        // fire unregistered: disabling its key could not disable its node
        // while metadata and reservation exclude it.
        dmx_record_capture_node(stream.stream(), static_cast<int>(hook_type),
                                static_cast<int>(hook_id), /*supported=*/true);
    }
}

// ring::producer_head(ring_payload, x, seg_table, arena, K, hook_type, hook_id)
//   In-graph linear heads at one HookPoint: K fp32 scores w_k . x[row] + b_k
//   for each row seg_table selects, from the stable fp32 arena (K x hidden
//   weights, K biases, split scratch).  Recorded as its own toggle node under
//   (hook_type, hook_id), independent of the HookPoint's row producer.
void ring_producer_head_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const at::Tensor& seg_table,
    const at::Tensor& arena,
    int64_t K,
    int64_t hook_type, int64_t hook_id)
{
    if (!g_active_engine) { return; }
    if (hook_type >= 0 && hook_type < HOST_HOOK_MAX)
        g_host_calls[hook_type].fetch_add(1);
    if (!(tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() >= 1)) return;
    TORCH_CHECK(seg_table.defined(), "ring::producer_head: a row table is required");
    check_row_table(seg_table, tensor, "ring::producer_head");
    TORCH_CHECK(arena.defined() && arena.is_cuda() && arena.device() == tensor.device()
                && arena.is_contiguous() && arena.scalar_type() == at::kFloat,
                "ring::producer_head: arena must be contiguous fp32 on the "
                "HookPoint tensor's device");
    const auto st = tensor.scalar_type();
    TORCH_CHECK(st == at::kBFloat16 || st == at::kFloat,
                "ring::producer_head supports bf16 and fp32 HookPoints only");
    TORCH_CHECK(K >= 1 && K <= static_cast<int64_t>(ring::HEAD_MAX_SLOTS),
                "ring::producer_head: K must be in [1, HEAD_MAX_SLOTS]");
    const int64_t hidden = tensor.size(tensor.dim() - 1);
    TORCH_CHECK(arena.numel() == static_cast<int64_t>(ring::head_arena_floats(
                    static_cast<uint32_t>(K), static_cast<uint32_t>(hidden))),
                "ring::producer_head: arena must hold K*hidden weights, K "
                "biases and the split scratch (head_arena_floats)");
    auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    g_active_engine->hook_no_notify_head(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        static_cast<uint64_t>(tensor.nbytes()),
        reinterpret_cast<uint64_t>(seg_table.data_ptr()),
        static_cast<int64_t>(seg_table.numel()),
        static_cast<uint32_t>(hidden),
        static_cast<uint32_t>(tensor.element_size()),
        reinterpret_cast<uint64_t>(arena.data_ptr()),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(hook_type),
        reinterpret_cast<uint64_t>(stream.stream()));
    count_producer_launch();
    dmx_record_capture_node(stream.stream(), static_cast<int>(hook_type),
                            static_cast<int>(hook_id), /*supported=*/true);
}

// ring::producer_fused(ring_payload, x, gather_table?, row_bytes, head_table?,
//                      arena, acc, K, weight_bf16, hook_type, hook_id,
//                      head_hook_type)
//   The HookPoint's row producer AND its heads in ONE node.  Each present
//   part has its own table and its own toggle key: (hook_type, hook_id) for
//   the rows, (head_hook_type, hook_id) for the heads.  Both keys are
//   registered to the same captured node, which is enabled iff either key
//   is; each part early-exits on its own -1 row, so "metadata pushed <=>
//   task published" still holds per key.
//   arena: int32 [head_node_words(K, hidden, weight_bf16)];
//   acc:   fp32 [acc_rows, HEAD_MAX_ACC], shared by all nodes.
void ring_producer_fused_impl(
    const at::Tensor& /*ring_payload*/,
    const at::Tensor& tensor,
    const c10::optional<at::Tensor>& gather_table,
    int64_t row_bytes,
    const c10::optional<at::Tensor>& head_table,
    const at::Tensor& arena,
    const at::Tensor& acc,
    int64_t K, int64_t weight_bf16,
    int64_t hook_type, int64_t hook_id, int64_t head_hook_type)
{
    if (!g_active_engine) { return; }
    if (hook_type >= 0 && hook_type < HOST_HOOK_MAX)
        g_host_calls[hook_type].fetch_add(1);
    const bool g = gather_table.has_value() && gather_table->defined();
    const bool h = head_table.has_value() && head_table->defined();
    if (!(tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() >= 1)) return;
    if (!g && !h) return;
    const auto st = tensor.scalar_type();
    TORCH_CHECK(st == at::kBFloat16 || st == at::kFloat,
                "ring::producer_fused supports bf16 and fp32 HookPoints only");
    const int64_t hidden = tensor.size(tensor.dim() - 1);
    if (g) {
        check_row_table(*gather_table, tensor, "ring::producer_fused (rows)");
        TORCH_CHECK(row_bytes == hidden * static_cast<int64_t>(tensor.element_size()),
                    "ring::producer_fused: row_bytes must equal hidden * element size");
        TORCH_CHECK(row_bytes % 16 == 0,
                    "ring::producer_fused: rows must be 16-byte multiples");
    }
    if (h) {
        check_row_table(*head_table, tensor, "ring::producer_fused (heads)");
        TORCH_CHECK(K >= 1 && K <= static_cast<int64_t>(ring::HEAD_MAX_SLOTS),
                    "ring::producer_fused: K must be in [1, HEAD_MAX_SLOTS]");
        TORCH_CHECK(arena.defined() && arena.is_cuda()
                    && arena.device() == tensor.device() && arena.is_contiguous()
                    && arena.scalar_type() == at::kInt
                    && arena.numel() == static_cast<int64_t>(ring::head_node_words(
                           static_cast<uint32_t>(K), static_cast<uint32_t>(hidden),
                           weight_bf16 != 0)),
                    "ring::producer_fused: arena must be int32 "
                    "[head_node_words(K, hidden, weight_bf16)]");
        TORCH_CHECK(acc.defined() && acc.is_cuda() && acc.device() == tensor.device()
                    && acc.is_contiguous()
                    && acc.scalar_type() == at::kFloat && acc.dim() == 2
                    && acc.size(1) == static_cast<int64_t>(ring::HEAD_MAX_ACC),
                    "ring::producer_fused: acc must be fp32 [rows, HEAD_MAX_ACC]");
    }
    auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    g_active_engine->hook_no_notify_fused(
        reinterpret_cast<uint64_t>(tensor.data_ptr()),
        static_cast<uint64_t>(tensor.nbytes()),
        static_cast<uint32_t>(hidden),
        static_cast<uint32_t>(tensor.element_size()),
        g ? reinterpret_cast<uint64_t>(gather_table->data_ptr()) : 0,
        g ? static_cast<int64_t>(gather_table->numel()) : 0,
        h ? reinterpret_cast<uint64_t>(head_table->data_ptr()) : 0,
        h ? static_cast<int64_t>(head_table->numel()) : 0,
        h ? reinterpret_cast<uint64_t>(arena.data_ptr()) : 0,
        static_cast<uint32_t>(K), static_cast<uint32_t>(weight_bf16 != 0),
        h ? reinterpret_cast<uint64_t>(acc.data_ptr()) : 0,
        h ? static_cast<uint64_t>(acc.size(0)) : 0,
        static_cast<uint32_t>(hook_type),
        reinterpret_cast<uint64_t>(stream.stream()));
    count_producer_launch();   // one node for both parts
    if (g) dmx_record_capture_node(stream.stream(), static_cast<int>(hook_type),
                                   static_cast<int>(hook_id), /*supported=*/true);
    if (h) dmx_record_capture_node(stream.stream(), static_cast<int>(head_hook_type),
                                   static_cast<int>(hook_id), /*supported=*/true);
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
    m.def("producer_gather(Tensor(a!) ring_payload, Tensor x, "
          "Tensor seg_table, int row_bytes, "
          "int hook_type, int hook_id) -> ()");
    m.def("producer_head(Tensor(a!) ring_payload, Tensor x, "
          "Tensor seg_table, Tensor arena, int K, "
          "int hook_type, int hook_id) -> ()");
    m.def("producer_fused(Tensor(a!) ring_payload, Tensor x, "
          "Tensor? gather_table, int row_bytes, Tensor? head_table, "
          "Tensor arena, Tensor acc, int K, int weight_bf16, "
          "int hook_type, int hook_id, int head_hook_type) -> ()");
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
    m.impl("producer_gather",  ring_producer_gather_impl);
    m.impl("producer_head",    ring_producer_head_impl);
    m.impl("producer_fused",   ring_producer_fused_impl);
    m.impl("record_producer", ring_record_producer_impl);
    m.impl("record_producer_prefix", ring_record_producer_prefix_impl);
    m.impl("record_producer_chunked", ring_record_producer_chunked_impl);
    m.impl("record_producer_seq_prefix_pack",
           ring_record_producer_seq_prefix_pack_impl);
    m.impl("record_producer_segmented_pack",
           ring_record_producer_segmented_pack_impl);
}
