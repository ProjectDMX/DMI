// tests/ring/test_node_toggle_e2e.cu — Phase 1 end-to-end node-toggle test.
//
// Unlike test_node_toggle.cu (device ring only), this drives the FULL consumer
// pipeline on a NON-BLOCKING stream:
//
//   captured producer graph  →  RingEngine (drain thread → p2p thread)
//                            →  TensorMetaFifo (host meta lane)  →  SubmitFn
//
// It verifies the core Phase 1 invariant (design-notes §1, #3): the node-enabled
// set must be kept in LOCKSTEP with the host meta-push set, or the positional
// meta↔payload matching in p2p desyncs.
//
// Ground-truth trick: producer j fills its payload with byte j, and we push its
// meta with layer_no = j.  So for EVERY submitted slice, under lockstep:
//     submitted layer_no  ==  first byte of the slice         (no desync)
// and the delivered sequence == concatenation of the per-step enabled sets.
//
// Two scenarios prove the invariant is BOTH sufficient and necessary:
//   (1) lockstep=true  : push metas only for enabled producers  → expect 0 desync
//   (2) lockstep=false : push metas for ALL producers regardless → expect desync
//
// Reconfigure is done at a step boundary with the prior replay synced first
// (invariants #1/#2/#4), mirroring how a real decode loop must drive it.
//
// Build:  make -C tests/ring test_node_toggle_e2e
// Run:    ./tests/ring/build/test_node_toggle_e2e

#include "ring/ring_engine.h"
#include "ring/producer.cuh"
#include "ring/tensor_meta.h"
#include "ring/node_toggle.h"   // Phase 2: NodeToggleController (toggle-list API)

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>   // std::exit
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace ring;

static int g_total = 0, g_fail = 0;
#define ASSERT(c) do { ++g_total; if(!(c)){ ++g_fail; \
  fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);} } while(0)
#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); std::exit(1);} } while(0)

static constexpr int      N         = 12;            // producer nodes
static constexpr uint64_t SRC_BYTES = 64 * 1024;     // 64 KiB/producer

// SubmitFn collector (called from the p2p thread -> thread-safe).
struct Collector {
    std::mutex mu;
    std::vector<std::pair<int,int>> got;  // (layer_no, first_byte_of_slice)
    void add(int layer, int fb) { std::lock_guard<std::mutex> lk(mu); got.emplace_back(layer, fb); }
    size_t size() { std::lock_guard<std::mutex> lk(mu); return got.size(); }
};

struct Result { size_t delivered, expected; int desync, mismatch; };

// The reconfigure schedule (each step = set of ENABLED producer-ids).
static const std::vector<std::set<int>> STEPS = {
    {0,1,2,3,4,5,6,7,8,9,10,11},   // all
    {0,3,6,9},                     // sparse subset
    {1,2,4,5,7,8,10,11},           // complement-ish
    {5},                           // single
    {0,1,2,3,4,5,6,7,8,9,10,11},   // all again
};

static Result run_scenario(bool lockstep, std::vector<uint8_t*>& src) {
    Collector col;
    SubmitFn submit = [&col](const std::string&, int32_t, const std::string&,
                             const std::string&, int32_t layer_no,
                             int32_t, int32_t, at::Tensor slice) {
        int fb = -1;
        if (slice.defined() && slice.numel() > 0) {
            at::Tensor c = slice.contiguous();
            fb = (int)c.data_ptr<uint8_t>()[0];
        }
        col.add((int)layer_no, fb);
    };

    RingConfig cfg;
    cfg.task_ring_entries  = 4096;
    cfg.payload_ring_bytes = 256ULL * 1024 * 1024;
    ring_py::TensorMetaFifo fifo;
    RingEngine engine(cfg, fifo, submit);

    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    engine.init(s);

    // Capture the producer graph against the engine's rings.
    cudaGraph_t graph; cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < N; j++)
        launch_producer(engine.ring_state(), src[j], SRC_BYTES, (uint32_t)j, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    // Collect the kernel nodes. We do NOT assume cudaGraphGetNodes() returns
    // them in capture order, and we do NOT read kernel args by index (that would
    // implicitly bind to producer_kernel's argument order -- a silent break if
    // the signature changes).
    size_t nn = 0; CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nn));
    std::vector<cudaGraphNode_t> nodes(nn); CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &nn));
    std::vector<cudaGraphNode_t> knodes;
    for (auto& nd : nodes) {
        cudaGraphNodeType t;
        if (cudaGraphNodeGetType(nd, &t) == cudaSuccess && t == cudaGraphNodeTypeKernel) knodes.push_back(nd);
    }
    if ((int)knodes.size() != N) {
        fprintf(stderr, "FATAL: expected %d kernel nodes, got %zu\n", N, knodes.size()); std::exit(1);
    }

    // Build the node->producer-id map by RUNTIME OBSERVATION (no ABI coupling):
    // enable exactly one node, replay, and the single published payload's first
    // byte is the producer id it drives.  Done BEFORE engine.start() so the
    // drain/p2p consumer isn't running yet and we can read the task ring directly.
    auto reset_rings = [&] {
        RingState& rs = engine.ring_state();
        CUDA_CHECK(cudaMemset(rs.task_entries, 0xFF, rs.task_cap * sizeof(TaskEntry)));
        *rs.task_head = 0; *rs.payload_head = 0;
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    std::vector<cudaGraphNode_t> node_of_pid(N, nullptr);
    for (int k = 0; k < N; k++) {
        for (int m = 0; m < N; m++)
            CUDA_CHECK(cudaGraphNodeSetEnabled(exec, knodes[m], m == k ? 1 : 0));
        reset_rings();
        CUDA_CHECK(cudaGraphLaunch(exec, s)); CUDA_CHECK(cudaStreamSynchronize(s));
        RingState& rs = engine.ring_state();
        if (*rs.task_head != 1) {  // fatal: otherwise the entry below is sentinel
            fprintf(stderr, "FATAL: single-node probe published %llu entries (want 1)\n",
                    (unsigned long long)*rs.task_head); std::exit(1);
        }
        TaskEntry e0; CUDA_CHECK(cudaMemcpy(&e0, rs.task_entries, sizeof(TaskEntry), cudaMemcpyDeviceToHost));
        uint8_t fb; CUDA_CHECK(cudaMemcpy(&fb, rs.payload_buf + e0.payload_off1, 1, cudaMemcpyDeviceToHost));
        if ((int)fb < N) node_of_pid[fb] = knodes[k];
    }
    for (int j = 0; j < N; j++)
        if (!node_of_pid[j]) { fprintf(stderr, "FATAL: producer %d has no mapped node\n", j); std::exit(1); }
    // Phase 2 API: register each producer's node under its hook identity
    // (RESID_PRE, layer = producer id), in capture order. The controller is now
    // the single source of truth for BOTH node-toggle and the meta-push list.
    NodeToggleController ctrl;
    for (int j = 0; j < N; j++)
        ctrl.register_node(HookId{ring_py::HOOK_TYPE_RESID_PRE, j}, node_of_pid[j]);
    { std::string why; if (!ctrl.validate(&why)) { fprintf(stderr, "FATAL: controller invalid: %s\n", why.c_str()); std::exit(1); } }
    reset_rings();   // clean ring state before the consumer starts

    engine.start();

    std::vector<int> expected;   // producer-ids actually fired, in order (= enabled sets)
    size_t n_payloads = 0;
    for (auto& step : STEPS) {
        CUDA_CHECK(cudaStreamSynchronize(s));   // #1/#2: prior replay done before reconfigure

        ctrl.set_enabled_if([&](HookId h) { return step.count(h.layer_no) > 0; });

        // lockstep -> apply AND get the meta-push list from ONE snapshot
        // (apply_and_get_enabled: no mutation can interleave the two lanes);
        // violation -> apply, then push for ALL hooks (bypassing the controller),
        // the bug the API exists to prevent.
        std::vector<HookId> push_hooks;
        if (lockstep) {
            CUDA_CHECK(ctrl.apply_and_get_enabled(exec, push_hooks));
        } else {
            CUDA_CHECK(ctrl.apply(exec));
            for (int j = 0; j < N; j++) push_hooks.push_back(HookId{ring_py::HOOK_TYPE_RESID_PRE, j});
        }

        auto* ctx = new ring_py::StepContext();
        ctx->model_id = "m";
        ctx->flattened = true;
        ctx->requests.push_back(ring_py::RequestMeta{"r", 0, 1, 0, 0});
        std::vector<ring_py::TensorMeta> metas;
        for (size_t i = 0; i < push_hooks.size(); i++) {
            ring_py::TensorMeta m;
            m.hook_type    = push_hooks[i].hook_type;
            m.layer_no     = push_hooks[i].layer_no;
            m.shape        = { 1, (int64_t)SRC_BYTES };
            m.dtype        = (int)at::kByte;
            m.last_in_step = (i + 1 == push_hooks.size());
            metas.push_back(std::move(m));
        }
        fifo.push_step(ctx, metas);

        for (int pid : step) expected.push_back(pid);   // ascending == producer order
        n_payloads += step.size();

        CUDA_CHECK(cudaGraphLaunch(exec, s));            // #4: mutate-then-launch
    }
    CUDA_CHECK(cudaStreamSynchronize(s));

    for (int spin = 0; spin < 500 && col.size() < n_payloads; spin++)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    engine.stop();

    std::lock_guard<std::mutex> lk(col.mu);
    Result r{ col.got.size(), expected.size(), 0, 0 };
    size_t n = std::min(col.got.size(), expected.size());
    for (size_t i = 0; i < n; i++) {
        if (col.got[i].first != col.got[i].second) r.desync++;
        if (col.got[i].first != expected[i])       r.mismatch++;
    }
    return r;
}

int main() {
    setbuf(stdout, nullptr);
    printf("=== test_node_toggle_e2e (N=%d, %llu KiB/producer) ===\n",
           N, (unsigned long long)(SRC_BYTES >> 10));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s sm_%d%d\n", p.name, p.major, p.minor);
    set_ring_null_mode(false);   // pure toggle, not null_mode

    std::vector<uint8_t*> src(N);
    for (int j = 0; j < N; j++) {
        CUDA_CHECK(cudaMalloc(&src[j], SRC_BYTES));
        CUDA_CHECK(cudaMemset(src[j], j, SRC_BYTES));
    }

    printf("\n[scenario 1] LOCKSTEP (push metas only for enabled producers)\n");
    Result a = run_scenario(true, src);
    printf("  delivered %zu / expected %zu | desync=%d mismatch=%d\n",
           a.delivered, a.expected, a.desync, a.mismatch);
    ASSERT(a.delivered == a.expected);
    ASSERT(a.desync == 0);
    ASSERT(a.mismatch == 0);

    printf("\n[scenario 2] LOCKSTEP VIOLATION (push metas for ALL producers)\n");
    Result b = run_scenario(false, src);
    printf("  delivered %zu | desync=%d mismatch=%d  (expect desync detected)\n",
           b.delivered, b.desync, b.mismatch);
    ASSERT(b.desync > 0 || b.mismatch > 0);   // the test has teeth: desync IS observed

    printf("\n%d / %d assertions passed\n", g_total - g_fail, g_total);
    if (g_fail) { fprintf(stderr, "%d FAILURES\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
