// Host-only harness for the ring producer ops (no CUDA toolkit or GPU).
//
// native/csrc/ring/ring_torch_op.cpp is compiled unchanged against CPU-only
// Torch and tests/native/ring/cpu_cuda_stub/ (CUDA runtime API and
// <ATen/cuda/CUDAContext.h> stand-ins), and linked with this file, which
// supplies a recording RingEnginePy double.  Tensors are host buffers tagged
// as CUDA device 0 (from_blob with a target device), so the ops' own argument
// validation runs; nothing is ever launched.  tests/test_ring_torch_op_host.py
// builds it as a shared library and drives it through the C entry points
// below, from the real dmi.node_toggle lifecycle.
#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "producer.cuh"
#include "ring_engine_py.h"
#include "ring_torch_op.h"

// The op implementations (registered by TORCH_LIBRARY_IMPL in
// ring_torch_op.cpp; not declared in a header).
void ring_producer_impl(const at::Tensor&, const at::Tensor&, int64_t, int64_t);
void ring_producer_prefix_impl(const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, int64_t, int64_t, int64_t);
void ring_producer_chunked_impl(const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, int64_t, int64_t);
void ring_producer_gather_impl(const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, int64_t, int64_t, int64_t);
void ring_producer_head_impl(const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, int64_t,
                             int64_t, int64_t);
void ring_producer_fused_impl(const at::Tensor&, const at::Tensor&,
                              const c10::optional<at::Tensor>&, int64_t,
                              const c10::optional<at::Tensor>&,
                              const at::Tensor&, const at::Tensor&, int64_t,
                              int64_t, int64_t, int64_t, int64_t);

// ---------------------------------------------------------------------------
// RingEnginePy double: records which launcher each op reached and the capture
// nodes it registered.

namespace {

struct EngineRecord {
    uint64_t launcher_calls = 0;
    uint64_t anomalies = 0;
    // graph -> (hook_type, layer_no) of each registered node
    std::map<uint64_t, std::vector<std::pair<int, int>>> nodes;
};

EngineRecord g_engine;

}  // namespace

namespace ring_py {

struct RingEnginePy::Impl {};

RingEnginePy::RingEnginePy(RingConfig, SubmitFn) : impl_(new Impl) {}
RingEnginePy::~RingEnginePy() = default;

void RingEnginePy::hook_no_notify(uint64_t, uint64_t, uint32_t, uint64_t) {
    ++g_engine.launcher_calls;
}
void RingEnginePy::hook_no_notify_prefix(uint64_t, uint64_t, uint64_t,
                                         uint64_t, uint32_t, uint64_t) {
    ++g_engine.launcher_calls;
}
void RingEnginePy::hook_no_notify_chunked(uint64_t, uint64_t, uint64_t,
                                          uint32_t, uint32_t, uint64_t) {
    ++g_engine.launcher_calls;
}
void RingEnginePy::hook_no_notify_gather(uint64_t, uint64_t, uint64_t,
                                         int64_t, uint64_t, uint32_t,
                                         uint64_t) {
    ++g_engine.launcher_calls;
}
void RingEnginePy::hook_no_notify_head(uint64_t, uint64_t, uint64_t, int64_t,
                                       uint32_t, uint32_t, uint64_t, uint32_t,
                                       uint32_t, uint64_t) {
    ++g_engine.launcher_calls;
}
void RingEnginePy::hook_no_notify_fused(uint64_t, uint64_t, uint32_t,
                                        uint32_t, uint64_t, int64_t,
                                        uint64_t, int64_t, uint64_t,
                                        uint32_t, uint32_t, uint64_t,
                                        uint64_t, uint32_t, uint64_t) {
    ++g_engine.launcher_calls;
}
void RingEnginePy::record_no_notify(uint64_t, uint64_t, uint64_t, int32_t,
                                    uint64_t) {}
void RingEnginePy::record_no_notify_prefix(uint64_t, uint64_t, uint64_t,
                                           uint64_t, uint64_t, int32_t,
                                           uint64_t) {}
void RingEnginePy::record_no_notify_chunked(uint64_t, uint64_t, uint64_t,
                                            uint32_t, uint64_t, int32_t,
                                            uint64_t) {}
void RingEnginePy::record_no_notify_seq_prefix_pack(uint64_t, uint64_t,
                                                    uint64_t, uint64_t,
                                                    uint32_t, uint64_t,
                                                    uint64_t, int32_t,
                                                    uint64_t) {}
void RingEnginePy::record_no_notify_segmented_pack(uint64_t, uint64_t,
                                                   uint64_t, uint64_t,
                                                   uint32_t, uint64_t,
                                                   uint64_t, int32_t,
                                                   uint64_t) {}
void RingEnginePy::register_capture_node(uint64_t graph, int hook_type,
                                         int layer_no, uint64_t) {
    g_engine.nodes[graph].emplace_back(hook_type, layer_no);
}
void RingEnginePy::note_capture_anomaly() { ++g_engine.anomalies; }

}  // namespace ring_py

// ---------------------------------------------------------------------------
// Fixture tensors: fp32 x[ROWS, HIDDEN] and the per-op side tensors, all host
// memory tagged as CUDA device 0.

namespace {

constexpr int64_t ROWS = 4;
constexpr int64_t HIDDEN = 8;
constexpr int64_t ROW_BYTES = HIDDEN * 4;   // fp32 rows, a 16-byte multiple
constexpr int64_t K = 2;
constexpr int64_t HOOK_TYPE = 3;
constexpr int64_t HEAD_HOOK_TYPE = 24;

struct Fixture {
    std::vector<std::vector<unsigned char>> blobs;
    ring_py::RingEnginePy engine{ring_py::RingConfig{}, ring_py::SubmitFn{}};

    at::Tensor tensor(at::IntArrayRef sizes, at::ScalarType dtype,
                      bool on_cuda = true) {
        int64_t n = 1;
        for (auto s : sizes) n *= s;
        blobs.emplace_back(static_cast<size_t>(n) * at::elementSize(dtype) + 16);
        const at::Device device = on_cuda ? at::Device(at::kCUDA, 0)
                                          : at::Device(at::kCPU);
        return at::for_blob(blobs.back().data(), sizes)
            .options(at::TensorOptions().dtype(dtype).device(device))
            .target_device(device)
            .make_tensor();
    }
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

std::string g_last_error;

// Runs one op.  Returns 0 on success, 1 when the op threw, 2 for an unknown
// op name.
int run(const std::string& op) {
    Fixture& f = fixture();
    at::Tensor payload = f.tensor({16}, at::kByte);
    at::Tensor x = f.tensor({ROWS, HIDDEN}, at::kFloat);
    at::Tensor table = f.tensor({3}, at::kLong);
    const auto arena_floats = static_cast<int64_t>(ring::head_arena_floats(
        static_cast<uint32_t>(K), static_cast<uint32_t>(HIDDEN)));
    const auto node_words = static_cast<int64_t>(ring::head_node_words(
        static_cast<uint32_t>(K), static_cast<uint32_t>(HIDDEN), false));
    at::Tensor acc = f.tensor({ROWS, static_cast<int64_t>(ring::HEAD_MAX_ACC)},
                              at::kFloat);
    const c10::optional<at::Tensor> none;
    if (op == "static") {
        ring_producer_impl(payload, x, HOOK_TYPE, 0);
    } else if (op == "prefix") {
        ring_producer_prefix_impl(payload, x, f.tensor({1}, at::kLong),
                                  ROW_BYTES, HOOK_TYPE, 0);
    } else if (op == "chunked") {
        ring_producer_chunked_impl(payload, x, f.tensor({2}, at::kLong),
                                   HOOK_TYPE, 0);
    } else if (op == "gather") {
        ring_producer_gather_impl(payload, x, table, ROW_BYTES, HOOK_TYPE, 0);
    } else if (op == "head") {
        ring_producer_head_impl(payload, x, table,
                                f.tensor({arena_floats}, at::kFloat), K,
                                HEAD_HOOK_TYPE, 0);
    } else if (op == "fused") {   // rows and heads in one node
        ring_producer_fused_impl(payload, x, table, ROW_BYTES, table,
                                 f.tensor({node_words}, at::kInt), acc, K, 0,
                                 HOOK_TYPE, 0, HEAD_HOOK_TYPE);
    } else if (op == "fused_rows") {
        ring_producer_fused_impl(payload, x, table, ROW_BYTES, none,
                                 f.tensor({1}, at::kInt), acc, K, 0,
                                 HOOK_TYPE, 0, HEAD_HOOK_TYPE);
    } else if (op == "fused_heads") {
        ring_producer_fused_impl(payload, x, none, ROW_BYTES, table,
                                 f.tensor({node_words}, at::kInt), acc, K, 0,
                                 HOOK_TYPE, 0, HEAD_HOOK_TYPE);
    } else if (op == "fused_empty") {   // neither part: nothing launches
        ring_producer_fused_impl(payload, x, none, ROW_BYTES, none,
                                 f.tensor({1}, at::kInt), acc, K, 0,
                                 HOOK_TYPE, 0, HEAD_HOOK_TYPE);
    } else if (op == "gather_host_tensor") {   // not a CUDA tensor: skipped
        ring_producer_gather_impl(payload, f.tensor({ROWS, HIDDEN}, at::kFloat,
                                                    /*on_cuda=*/false),
                                  table, ROW_BYTES, HOOK_TYPE, 0);
    } else if (op == "gather_bad_table") {    // refused before launching
        ring_producer_gather_impl(payload, x, f.tensor({3}, at::kInt),
                                  ROW_BYTES, HOOK_TYPE, 0);
    } else {
        return 2;
    }
    return 0;
}

}  // namespace

extern "C" {

// Activates the engine double (ring_set_active_engine) on first use.
int dmx_host_op(const char* op, uint64_t capturing_graph) {
    Fixture& f = fixture();
    ring_set_active_engine(&f.engine);
    fake_capture_graph = reinterpret_cast<cudaGraph_t>(capturing_graph);
    int rc;
    try {
        rc = run(op);
        g_last_error.clear();
    } catch (const std::exception& e) {
        g_last_error = e.what();
        rc = 1;
    }
    fake_capture_graph = nullptr;
    return rc;
}

const char* dmx_host_last_error() { return g_last_error.c_str(); }

void dmx_host_set_toggle_capture(int enabled) {
    ring_set_toggle_capture(enabled != 0);
}

uint64_t dmx_host_launch_count() { return ring_producer_launch_count(); }

// Launcher calls the engine double received (one per launched producer).
uint64_t dmx_host_launcher_calls() { return g_engine.launcher_calls; }

uint64_t dmx_host_anomalies() { return g_engine.anomalies; }

// Nodes registered for `graph`; graph 0 counts every graph's.
uint64_t dmx_host_node_count(uint64_t graph) {
    if (graph != 0) {
        auto it = g_engine.nodes.find(graph);
        return it == g_engine.nodes.end() ? 0 : it->second.size();
    }
    uint64_t n = 0;
    for (const auto& kv : g_engine.nodes) n += kv.second.size();
    return n;
}

// Toggle key of the i-th node registered for `graph`: hook_type << 32 | layer.
uint64_t dmx_host_node_key(uint64_t graph, uint64_t i) {
    const auto& v = g_engine.nodes.at(graph).at(i);
    return (static_cast<uint64_t>(v.first) << 32) | static_cast<uint32_t>(v.second);
}

void dmx_host_clear_nodes() {
    g_engine.nodes.clear();
    g_engine.anomalies = 0;
}

}  // extern "C"
