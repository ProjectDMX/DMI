// Native tests for ToggleRegistry: capture-time node recording, eager and
// lazy cudaGraphNodeSetEnabled application, registry guards, teardown, and
// the device-side planner (toggle_device.h).
//
// Each "producer" here is a one-block marker kernel that adds its value to
// its own output slot, so a replay's effect shows exactly which nodes ran.

#include "ring/toggle_registry.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t error_ = (expr);                                        \
        if (error_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(error_));    \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

#define EXPECT(condition)                                                   \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                      \
                         __FILE__, __LINE__, #condition);                    \
            ++g_fail;                                                       \
        } else {                                                            \
            ++g_pass;                                                       \
        }                                                                   \
    } while (0)

static void banner(const char* name) { std::printf("[ TEST ] %s\n", name); }

__global__ void marker_kernel(int* out, int slot, int value) {
    if (threadIdx.x == 0) out[slot] += value;
}

// A marker that honours the control-row contract of the producers: a -1 row
// returns without writing.
__global__ void gated_marker_kernel(int* out, int slot, int value,
                                    const int64_t* control_row) {
    if (control_row[0] == -1) return;
    if (threadIdx.x == 0) out[slot] += value;
}

static constexpr int kHookType = 3;

struct Graph {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t exec{nullptr};
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};

// Record the capture's tail dependency exactly as the producer ops do.
static uint64_t tail_node(cudaStream_t stream, cudaGraph_t* graph_out) {
    cudaStreamCaptureStatus status;
    unsigned long long id;
    cudaGraph_t graph;
    const cudaGraphNode_t* deps;
    size_t n;
    CUDA_CHECK(dmx_capture_info(stream, &status, &id, &graph, &deps, &n));
    if (status != cudaStreamCaptureStatusActive || n < 1) return 0;
    *graph_out = graph;
    return reinterpret_cast<uint64_t>(deps[n - 1]);
}

// Capture one marker node per layer in `layers`; marker for layer L adds
// base + L to out[L].
static void capture_markers(ring_py::ToggleRegistry& registry, Graph& g,
                            cudaStream_t stream, int* out,
                            const std::vector<int>& layers, int base) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int layer : layers) {
        marker_kernel<<<1, 32, 0, stream>>>(out, layer, base + layer);
        cudaGraph_t graph = nullptr;
        const uint64_t node = tail_node(stream, &graph);
        EXPECT(node != 0);
        registry.register_node(reinterpret_cast<uint64_t>(graph), kHookType,
                               layer, node);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &g.graph));
    CUDA_CHECK(cudaGraphInstantiate(&g.exec, g.graph, 0));
}

static std::vector<int> read_and_clear(int* out, int n) {
    std::vector<int> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(out, 0, n * sizeof(int)));
    return host;
}

static std::vector<std::pair<int,int>> keys(const std::vector<int>& layers) {
    std::vector<std::pair<int,int>> result;
    for (int layer : layers) result.push_back({kHookType, layer});
    return result;
}

// Expected marker output: enabled layers carry base + layer, others 0.
static std::vector<int> expected(int n, const std::vector<int>& enabled, int base) {
    std::vector<int> result(n, 0);
    for (int layer : enabled) result[layer] = base + layer;
    return result;
}

static void replay(const Graph& g, cudaStream_t stream) {
    CUDA_CHECK(cudaGraphLaunch(g.exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void test_eager_apply_and_diff() {
    banner("eager apply disables exactly the unselected nodes, diff-based");
    constexpr int n = 6;
    int* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(out, 0, n * sizeof(int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        ring_py::ToggleRegistry registry;
        registry.set_capture(true);
        Graph g;
        capture_markers(registry, g, stream, out, {0, 1, 2, 3, 4, 5}, 10);
        registry.bind_exec(reinterpret_cast<uint64_t>(g.graph),
                           reinterpret_cast<uint64_t>(g.exec));
        registry.set_capture(false);
        EXPECT(registry.node_count() == 6);
        EXPECT(registry.bound_count() == 1);
        EXPECT(registry.complete());
        EXPECT(registry.uniform());
        EXPECT(registry.anomaly_count() == 0);
        // Inactive gate: every hook counts as enabled.
        EXPECT(registry.is_enabled(kHookType, 4));

        registry.set_enabled(keys({1, 4}));
        EXPECT(registry.apply_all() == 0);
        EXPECT(registry.last_apply_count() == 4);   // four nodes disabled
        replay(g, stream);
        EXPECT(read_and_clear(out, n) == expected(n, {1, 4}, 10));
        EXPECT(registry.is_enabled(kHookType, 4));
        EXPECT(!registry.is_enabled(kHookType, 0));

        registry.set_enabled(keys({1, 2, 4}));
        EXPECT(registry.apply_all() == 0);
        EXPECT(registry.last_apply_count() == 1);   // only node 2 flips
        replay(g, stream);
        EXPECT(read_and_clear(out, n) == expected(n, {1, 2, 4}, 10));

        // Enabled but never captured -> not effective (no payload behind it).
        registry.set_enabled(keys({2}) );
        std::vector<std::pair<int,int>> query = {{kHookType, 2}, {kHookType, 9}, {kHookType + 1, 2}};
        EXPECT(registry.effective_mask(query) == std::vector<int>({1, 0, 0}));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(out));
}

static void test_lazy_per_graph_apply() {
    banner("lazy apply brings only the graph about to replay up to date");
    constexpr int n = 4;
    int* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(out, 0, n * sizeof(int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        ring_py::ToggleRegistry registry;
        registry.set_capture(true);
        Graph a, b;
        capture_markers(registry, a, stream, out, {0, 1, 2, 3}, 0);
        capture_markers(registry, b, stream, out, {0, 1, 2, 3}, 100);
        const uint64_t ga = reinterpret_cast<uint64_t>(a.graph);
        const uint64_t gb = reinterpret_cast<uint64_t>(b.graph);
        registry.bind_exec(ga, reinterpret_cast<uint64_t>(a.exec));
        registry.bind_exec(gb, reinterpret_cast<uint64_t>(b.exec));
        registry.set_capture(false);
        EXPECT(registry.uniform());
        EXPECT(registry.complete());

        // A sequence of reconfigurations; each graph is only brought current
        // when it is about to replay, possibly several versions late.
        const std::vector<std::vector<int>> sets = {{0, 2}, {1, 3}, {0, 1, 2, 3}, {2}, {}};
        const std::vector<std::vector<char>> order = {
            {'B', 'B', 'A'}, {'A', 'B'}, {}, {'A', 'B'}, {'A', 'B'}};
        for (size_t i = 0; i < sets.size(); ++i) {
            registry.set_enabled(keys(sets[i]));
            EXPECT(!registry.applied_current(ga));
            EXPECT(!registry.applied_current(gb));
            for (char which : order[i]) {
                const Graph& g = which == 'A' ? a : b;
                const uint64_t raw = which == 'A' ? ga : gb;
                EXPECT(registry.ensure_current(raw) == 0);
                EXPECT(registry.applied_current(raw));
                EXPECT(registry.record_replay_event(raw, stream) == 0);
                replay(g, stream);
                EXPECT(read_and_clear(out, n) ==
                       expected(n, sets[i], which == 'A' ? 0 : 100));
            }
        }
        // Unknown graphs are skipped, not failed.
        EXPECT(registry.ensure_current(0x1234) == 0);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(out));
}

static void test_registry_guards() {
    banner("uniform/complete/ready guards and anomalies");
    constexpr int n = 4;
    int* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        ring_py::ToggleRegistry registry;
        registry.set_capture(true);
        const uint64_t v0 = registry.version();
        Graph a, b;
        capture_markers(registry, a, stream, out, {0, 1, 2}, 0);
        capture_markers(registry, b, stream, out, {0, 1}, 0);
        EXPECT(registry.version() > v0);
        const uint64_t ga = reinterpret_cast<uint64_t>(a.graph);
        const uint64_t gb = reinterpret_cast<uint64_t>(b.graph);
        EXPECT(!registry.uniform());           // graph b lacks layer 2
        registry.bind_exec(ga, reinterpret_cast<uint64_t>(a.exec));
        EXPECT(!registry.complete());          // b captured, not bound
        EXPECT(registry.is_ready(ga));
        EXPECT(!registry.is_ready(gb));
        registry.bind_exec(gb, reinterpret_cast<uint64_t>(b.exec));
        EXPECT(registry.complete());
        EXPECT(registry.is_ready(gb));
        const uint64_t v1 = registry.version();
        registry.note_anomaly();
        EXPECT(registry.anomaly_count() == 1);
        EXPECT(registry.version() > v1);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(out));
}

static void test_failed_lazy_apply_leaves_graph_stale() {
    banner("a failed lazy apply mutates nothing and leaves the graph stale");
    constexpr int n = 3;
    int* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(out, 0, n * sizeof(int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        ring_py::ToggleRegistry registry;
        registry.set_capture(true);
        Graph g;
        capture_markers(registry, g, stream, out, {0, 1, 2}, 5);
        const uint64_t raw = reinterpret_cast<uint64_t>(g.graph);
        registry.bind_exec(raw, reinterpret_cast<uint64_t>(g.exec));
        registry.set_enabled(keys({1}));
        registry.force_apply_error(static_cast<int>(cudaErrorInvalidValue));
        EXPECT(registry.ensure_current(raw) == static_cast<int>(cudaErrorInvalidValue));
        EXPECT(!registry.applied_current(raw));
        replay(g, stream);   // nodes untouched: all still enabled
        EXPECT(read_and_clear(out, n) == expected(n, {0, 1, 2}, 5));
        registry.force_apply_error(0);
        EXPECT(registry.ensure_current(raw) == 0);
        EXPECT(registry.applied_current(raw));
        replay(g, stream);
        EXPECT(read_and_clear(out, n) == expected(n, {1}, 5));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(out));
}

static void test_upload_skips_graphs_without_device_nodes() {
    banner("graphs without device-updatable producers are never uploaded");
    int* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, sizeof(int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        ring_py::ToggleRegistry registry;
        Graph g;
        capture_markers(registry, g, stream, out, {0}, 0);
        // Not a device graph: nothing to upload (the upload would only cost a
        // stream create and a synchronize).
        EXPECT(registry.upload_device_graph(reinterpret_cast<uint64_t>(g.graph),
                                            reinterpret_cast<uint64_t>(g.exec)) == 0);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(out));
}

static void test_pre_apply_sync_and_clear() {
    banner("pre-apply wait records replay events; clear re-enables nodes");
    constexpr int n = 3;
    int* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(out, 0, n * sizeof(int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        ring_py::ToggleRegistry registry;
        registry.set_pre_apply_sync(true);
        EXPECT(registry.pre_apply_sync());
        registry.set_capture(true);
        Graph g;
        capture_markers(registry, g, stream, out, {0, 1, 2}, 1);
        const uint64_t raw = reinterpret_cast<uint64_t>(g.graph);
        registry.bind_exec(raw, reinterpret_cast<uint64_t>(g.exec));
        registry.set_enabled(keys({0}));
        EXPECT(registry.ensure_current(raw) == 0);
        CUDA_CHECK(cudaGraphLaunch(g.exec, stream));
        EXPECT(registry.record_replay_event(raw, stream) == 0);
        registry.set_enabled(keys({2}));
        EXPECT(registry.ensure_current(raw) == 0);   // waits on the event
        replay(g, stream);
        const std::vector<int> two_replays = read_and_clear(out, n);
        EXPECT(two_replays == std::vector<int>({1, 0, 3}));

        registry.clear();
        EXPECT(registry.node_count() == 0);
        EXPECT(registry.bound_count() == 0);
        EXPECT(!registry.capture_enabled());
        EXPECT(registry.is_enabled(kHookType, 1));   // gate inactive again
        replay(g, stream);                           // every node re-enabled
        EXPECT(read_and_clear(out, n) == expected(n, {0, 1, 2}, 1));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(out));
}

#if DMX_HAS_DEVICE_GRAPH_UPDATE
static cudaError_t launch_device_updatable(int* out, int slot, int value,
                                           const int64_t* row,
                                           cudaStream_t stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(32);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attr[0].val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    attr[0].val.deviceUpdatableKernelNode.devNode = nullptr;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, gated_marker_kernel, out, slot, value, row);
}
#endif

static void test_device_planner() {
    banner("device planner enables each node from its own control row");
#if DMX_HAS_DEVICE_GRAPH_UPDATE
    ring_py::ToggleRegistry registry;
    if (!registry.set_device_mode(true)) {
        std::printf("  skipped: device graph updates unavailable\n");
        return;
    }
    constexpr int n = 5;
    int* out = nullptr;
    int64_t* rows = nullptr;
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(out, 0, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rows, n * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(rows, 0, n * sizeof(int64_t)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    registry.set_capture(true);   // allocates the pool
    EXPECT(registry.device_armed());

    Graph g;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; ++i) {
        cudaStreamCaptureStatus st;
        unsigned long long id;
        cudaGraph_t graph;
        const cudaGraphNode_t* deps;
        size_t nd;
        CUDA_CHECK(dmx_capture_info(stream, &st, &id, &graph, &deps, &nd));
        bool fresh = false;
        const int slot = registry.device_slot(reinterpret_cast<uint64_t>(graph),
                                              rows + i, &fresh);
        EXPECT(slot == 0);
        EXPECT(fresh == (i == 0));
        if (fresh) ring::launch_toggle_planner(registry.device_pool(), slot, stream);
        CUDA_CHECK(launch_device_updatable(out, i, 20 + i, rows + i, stream));
        CUDA_CHECK(dmx_capture_info(stream, &st, &id, &graph, &deps, &nd));
        cudaGraphNode_t node = deps[nd - 1];
        cudaKernelNodeAttrValue value = {};
        CUDA_CHECK(cudaGraphKernelNodeGetAttribute(
            node, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &value));
        EXPECT(value.deviceUpdatableKernelNode.deviceUpdatable);
        registry.register_device_entry(slot, reinterpret_cast<uint64_t>(graph),
                                       reinterpret_cast<uint64_t>(node), rows + i,
                                       value.deviceUpdatableKernelNode.devNode);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &g.graph));
    CUDA_CHECK(cudaGraphInstantiate(&g.exec, g.graph, 0));
    EXPECT(registry.upload_device_graph(reinterpret_cast<uint64_t>(g.graph),
                                        reinterpret_cast<uint64_t>(g.exec)) == 0);
    // At most once: binding the same graph later does not upload it again
    // (and an upload failure would count as an anomaly).
    registry.bind_exec(reinterpret_cast<uint64_t>(g.graph),
                       reinterpret_cast<uint64_t>(g.exec));
    EXPECT(registry.upload_device_graph(reinterpret_cast<uint64_t>(g.graph),
                                        reinterpret_cast<uint64_t>(g.exec)) == 0);
    registry.set_capture(false);
    EXPECT(registry.device_node_count() == n);
    EXPECT(registry.anomaly_count() == 0);

    const std::vector<std::vector<int64_t>> row_sets = {
        {0, 0, 0, 0, 0}, {-1, 3, -1, 0, -1}, {5, -1, 7, -1, 2}, {-1, -1, -1, -1, -1},
        {0, 0, 0, 0, 0}};
    for (const auto& row_values : row_sets) {
        CUDA_CHECK(cudaMemcpyAsync(rows, row_values.data(), n * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, stream));
        replay(g, stream);
        std::vector<int> want(n, 0);
        for (int i = 0; i < n; ++i) if (row_values[i] != -1) want[i] = 20 + i;
        EXPECT(read_and_clear(out, n) == want);
    }
    EXPECT(registry.device_errors() == 0);

    // A repeated control row starts a new group (the next forward).
    cudaGraph_t dummy = nullptr;
    bool fresh = false;
    EXPECT(registry.device_slot(reinterpret_cast<uint64_t>(dummy), rows, &fresh) == 1);
    EXPECT(fresh);

    // The mode is per capture window: clear() turns it off (the pool stays
    // allocated), and nothing is armed until it is enabled again.
    registry.clear();
    EXPECT(!registry.device_mode());
    EXPECT(!registry.device_armed());
    EXPECT(registry.device_slot(0, rows, &fresh) == -1);
    EXPECT(registry.set_device_mode(true));
    EXPECT(registry.device_armed());
    EXPECT(registry.set_device_mode(false));
    EXPECT(!registry.device_mode());

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(rows));
    CUDA_CHECK(cudaFree(out));
#else
    std::printf("  skipped: built without device graph updates\n");
#endif
}

int main() {
    setbuf(stdout, nullptr);
    CUDA_CHECK(cudaFree(nullptr));
    std::printf("test_toggle_registry\n");
    test_eager_apply_and_diff();
    test_lazy_per_graph_apply();
    test_registry_guards();
    test_failed_lazy_apply_leaves_graph_stale();
    test_pre_apply_sync_and_clear();
    test_upload_skips_graphs_without_device_nodes();
    test_device_planner();
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
