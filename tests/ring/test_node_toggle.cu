// tests/ring/test_node_toggle.cu — Phase 0 node-toggle test (performance-led).
//
// Productionizes docs/node_toggle_probe/probe_dualring_toggle.cu into the
// tests/ring harness. Links the real ring::producer_kernel + AllocatedRing and
// captures N producer launches into a CUDA graph, then uses
// cudaGraphNodeSetEnabled to disable/enable producer nodes between replays.
//
// MEASURES (performance, the focus here):
//   - per-replay cost: all enabled / all true-disabled / null_mode soft / adaptive subset
//   - reconfigure cost: single node + full-set flip (host wall-clock)
// VERIFIES (correctness, cheap so included):
//   - dual-ring stays aligned under toggle (remaining producers publish a
//     contiguous, gap-free, correctly-sequenced run of task entries)
//   - node->producer mapping is built EXPLICITLY (design-notes §3): we never
//     assume cudaGraphGetNodes() order == capture order.
//
// SCOPE: single stream + synced, so this exercises ring/data SEMANTICS only.
// It does NOT cover the stream-ordering / exec-mutation-lifecycle invariants
// (design-notes §1, #1-#5) — those require concurrent non-blocking streams and
// are deferred to the Phase 1 HF end-to-end prototype. The per-replay µs are
// isolated producer cost (no model compute), i.e. the work toggle removes;
// serving-% needs Phase 1. See docs/node_toggle_design_notes.md.
//
// Build:  make -C tests/ring test_node_toggle        (override count: NPROD=145)
// Run:    ./tests/ring/build/test_node_toggle

#include "ring/ring_alloc.h"
#include "ring/producer.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <vector>

using namespace ring;

// ---------------------------------------------------------------------------
// Harness (matches test_producer.cu conventions)
// ---------------------------------------------------------------------------
static int g_total = 0;
static int g_fail  = 0;

#define ASSERT(cond) do {                                          \
    ++g_total;                                                     \
    if (!(cond)) { ++g_fail;                                       \
        fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

#define CUDA_CHECK(e) do {                                         \
    cudaError_t _e = (e);                                          \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));       \
        std::exit(1); }                                            \
} while (0)

static void banner(const char* name) { printf("[ TEST ] %s\n", name); fflush(stdout); }

// ---------------------------------------------------------------------------
// Config (override at compile time: -DNPROD=145 -DSRC_KB=512)
// ---------------------------------------------------------------------------
#ifndef NPROD
#define NPROD 16
#endif
#ifndef SRC_KB
#define SRC_KB 1024            // per-producer tensor size in KiB
#endif
static constexpr int      N         = NPROD;
static constexpr uint64_t SRC_BYTES = uint64_t(SRC_KB) * 1024;
static uint64_t ALLOC;          // align_up(SRC_BYTES, 16)

// First byte of a payload identifies its writer (src[j] is filled with byte j).
static int payload_writer_id(uint8_t* payload_buf, uint64_t off) {
    uint8_t b; CUDA_CHECK(cudaMemcpy(&b, payload_buf + off, 1, cudaMemcpyDeviceToHost));
    return (int)b;
}

static void reset_ring(AllocatedRing& ar) {
    RingState& s = ar.state();
    CUDA_CHECK(cudaMemset(s.task_entries, 0xFF, s.task_cap * sizeof(TaskEntry)));
    *s.task_head = 0; *s.payload_head = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Replay once; return producer-ids that published, asserting alignment.
static std::vector<int> run_and_collect(AllocatedRing& ar, cudaGraphExec_t exec, cudaStream_t s) {
    reset_ring(ar);
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    RingState& rs = ar.state();
    uint64_t pub = *rs.task_head;
    std::vector<TaskEntry> ent(pub);
    if (pub) CUDA_CHECK(cudaMemcpy(ent.data(), rs.task_entries, pub * sizeof(TaskEntry), cudaMemcpyDeviceToHost));
    std::vector<int> ids;
    uint64_t expect_off = 0;
    for (uint64_t i = 0; i < pub; i++) {
        ASSERT(ent[i].ready_seq == i);                       // sequence protocol
        ASSERT(ent[i].tensor_total_bytes == SRC_BYTES);      // well-formed
        ASSERT(ent[i].payload_off1 == expect_off);           // contiguous, no gap
        ids.push_back(payload_writer_id(rs.payload_buf, ent[i].payload_off1));
        expect_off += ALLOC;
    }
    ASSERT(*rs.payload_head == pub * ALLOC);
    return ids;
}

// Raw replay throughput via CUDA events (graph launches ARE stream work, so
// events bracket real GPU time). INTENTIONAL overwrite: no consumer, rings wrap
// many times over `iters`; fine for timing — every config overwrites equally.
static double time_replays_us(AllocatedRing& ar, cudaGraphExec_t exec, cudaStream_t s, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    reset_ring(ar);
    for (int i = 0; i < 10; i++) cudaGraphLaunch(exec, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; i++) cudaGraphLaunch(exec, s);
    cudaEventRecord(b, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return double(ms) / iters * 1000.0;
}

int main() {
    setbuf(stdout, nullptr);
    printf("=== test_node_toggle (N=%d, %llu KiB/producer) ===\n",
           N, (unsigned long long)(SRC_BYTES >> 10));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int rt = 0, drv = 0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    printf("GPU: %s sm_%d%d | CUDA rt %d drv %d\n", p.name, p.major, p.minor, rt, drv);
    ALLOC = (SRC_BYTES + 15) & ~uint64_t(15);

    RingConfig cfg;
    cfg.task_ring_entries  = 4096;
    cfg.payload_ring_bytes = 512ULL * 1024 * 1024;
    AllocatedRing ar(cfg);
    ar.init();

    std::vector<uint8_t*> src(N);
    for (int j = 0; j < N; j++) {
        CUDA_CHECK(cudaMalloc(&src[j], SRC_BYTES));
        CUDA_CHECK(cudaMemset(src[j], j, SRC_BYTES));
    }

    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

    // Capture N real producer launches on one stream (as vLLM captures hooks).
    cudaGraph_t graph; cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < N; j++)
        launch_producer(ar.state(), src[j], SRC_BYTES, (uint32_t)j, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    size_t nn = 0; CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nn));
    std::vector<cudaGraphNode_t> nodes(nn); CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &nn));
    std::vector<cudaGraphNode_t> knodes;
    for (auto& nd : nodes) {
        cudaGraphNodeType t;
        if (cudaGraphNodeGetType(nd, &t) == cudaSuccess && t == cudaGraphNodeTypeKernel)
            knodes.push_back(nd);
    }
    if (knodes.size() != (size_t)N) {
        fprintf(stderr, "FATAL: expected %d kernel nodes, got %zu\n", N, (size_t)knodes.size());
        return 1;
    }

    // --- Explicit node -> producer-id map (design-notes §3): enable one node at
    // a time; the single published writer-id identifies the producer it drives.
    banner("build explicit node->producer map");
    std::vector<int> pid_node(N, -1);
    for (size_t k = 0; k < knodes.size(); k++) {
        for (size_t m = 0; m < knodes.size(); m++)
            CUDA_CHECK(cudaGraphNodeSetEnabled(exec, knodes[m], m == k ? 1 : 0));
        reset_ring(ar);
        CUDA_CHECK(cudaGraphLaunch(exec, s)); CUDA_CHECK(cudaStreamSynchronize(s));
        ASSERT(*ar.state().task_head == 1);
        TaskEntry e0; CUDA_CHECK(cudaMemcpy(&e0, ar.state().task_entries, sizeof(TaskEntry), cudaMemcpyDeviceToHost));
        int pid = payload_writer_id(ar.state().payload_buf, e0.payload_off1);
        if (pid >= 0 && pid < N) pid_node[pid] = (int)k;
    }
    bool bij = true; for (int j = 0; j < N; j++) if (pid_node[j] < 0) bij = false;
    ASSERT(bij);  // every producer id reachable

    auto enable_all   = [&]{ for (auto& nd : knodes) CUDA_CHECK(cudaGraphNodeSetEnabled(exec, nd, 1)); };
    auto set_disabled = [&](const std::set<int>& drop){
        for (int j = 0; j < N; j++)
            CUDA_CHECK(cudaGraphNodeSetEnabled(exec, knodes[pid_node[j]], drop.count(j) ? 0 : 1));
    };

    // --- Correctness: toggle keeps the dual-ring aligned ---
    banner("dual-ring alignment under toggle");
    enable_all();
    auto ids_full = run_and_collect(ar, exec, s);
    ASSERT(ids_full.size() == (size_t)N);
    for (int j = 0; j < N; j++) ASSERT(ids_full[j] == j);
    std::set<int> drop;
    for (int j = 2; j < N; j += 3) drop.insert(j);          // a representative subset
    set_disabled(drop);
    auto ids_sub = run_and_collect(ar, exec, s);
    std::vector<int> expect; for (int j = 0; j < N; j++) if (!drop.count(j)) expect.push_back(j);
    ASSERT(ids_sub == expect);                               // remaining aligned, no desync
    enable_all();
    ASSERT(run_and_collect(ar, exec, s) == ids_full);        // re-enable restores

    // ---------------------------------------------------------------------
    // PERFORMANCE
    // ---------------------------------------------------------------------
    banner("performance");
    const int IT = 2000;
    enable_all(); set_ring_null_mode(false);
    double t_full = time_replays_us(ar, exec, s, IT);
    for (auto& nd : knodes) CUDA_CHECK(cudaGraphNodeSetEnabled(exec, nd, 0));
    double t_off  = time_replays_us(ar, exec, s, IT);
    enable_all(); set_ring_null_mode(true);
    double t_null = time_replays_us(ar, exec, s, IT);
    set_ring_null_mode(false);
    { std::set<int> half; for (int j = 1; j < N; j += 2) half.insert(j); set_disabled(half); }
    double t_half = time_replays_us(ar, exec, s, IT);
    enable_all();

    printf("  per-replay (%d nodes x %llu KiB):\n", N, (unsigned long long)(SRC_BYTES >> 10));
    printf("    (a) all enabled        : %8.2f us\n", t_full);
    printf("    (b) all node-disabled  : %8.2f us   <- true disable (overhead toggle removes)\n", t_off);
    printf("    (c) all null_mode soft : %8.2f us   <- kernel still launched\n", t_null);
    printf("    (d) half node-disabled : %8.2f us\n", t_half);
    printf("    true-disable saves vs full : %8.2f us (%.1f%%)\n", t_full - t_off, 100.0 * (t_full - t_off) / t_full);
    printf("    true-disable saves vs null : %8.2f us (%.1f%% of full)\n", t_null - t_off, 100.0 * (t_null - t_off) / t_full);

    // Reconfigure cost — HOST wall-clock (these are host API calls, not stream work).
    enable_all(); CUDA_CHECK(cudaDeviceSynchronize());
    const int RC = 10000;
    auto h0 = std::chrono::steady_clock::now();
    for (int i = 0; i < RC; i++) CUDA_CHECK(cudaGraphNodeSetEnabled(exec, knodes[0], i & 1));
    auto h1 = std::chrono::steady_clock::now();
    double us_single = std::chrono::duration<double, std::micro>(h1 - h0).count() / RC;

    CUDA_CHECK(cudaDeviceSynchronize());
    const int FC = 2000;
    auto g0 = std::chrono::steady_clock::now();
    for (int i = 0; i < FC; i++) { int on = i & 1; for (auto& nd : knodes) CUDA_CHECK(cudaGraphNodeSetEnabled(exec, nd, on)); }
    auto g1 = std::chrono::steady_clock::now();
    double us_full = std::chrono::duration<double, std::micro>(g1 - g0).count() / FC;
    enable_all();

    printf("  reconfigure (host wall-clock):\n");
    printf("    single node            : %8.3f us/call (no re-instantiate)\n", us_single);
    printf("    full-set flip (%d nodes): %8.3f us  (%.3f us/node)\n", N, us_full, us_full / N);

    // ---------------------------------------------------------------------
    printf("\n%d / %d assertions passed\n", g_total - g_fail, g_total);
    if (g_fail) { fprintf(stderr, "%d FAILURES\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
