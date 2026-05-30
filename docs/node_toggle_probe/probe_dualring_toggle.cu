// probe_dualring_toggle.cu — node-toggle probe against the REAL DMI dual-ring.
//
// Unlike probe_node_toggle.cu (synthetic kernel), this links the actual
// ring::producer_kernel + AllocatedRing (payload ring + task/meta ring) and
// captures them into a CUDA graph exactly the way vLLM captures DMI's hooks.
//
// It answers the two questions that matter for DMI specifically:
//
//  Q1  Overhead: with the real producer doing real D2D copies + task publish,
//      what does a replay cost (a) all nodes enabled, (b) nodes toggled OFF via
//      cudaGraphNodeSetEnabled ("true disable"), (c) null_mode (kernel launches,
//      early-returns)?  Does disabling actually drop overhead?
//
//  Q2  Feasibility under dual-ring: when a producer node is disabled post-capture,
//      does the dual-ring stay CONSISTENT and ALIGNED?  i.e. do the remaining
//      enabled producers publish a contiguous, correctly-offset run of task
//      entries with uncorrupted payload — proving the device side "closes up"
//      with no gap?  (Heads are advanced by the kernel itself at runtime, not
//      pre-reserved on the host, so a skipped node should leave no hole.)
//
// Build + run from docs/node_toggle_probe/ (relative paths resolve from there):
//   nvcc -std=c++17 -arch=native -O2 -I../../monitoring/csrc \
//        probe_dualring_toggle.cu ../../monitoring/csrc/ring/producer.cu -o probe_dr
//   CUDA_MODULE_LOADING=EAGER ./probe_dr

#include "ring/ring_alloc.h"
#include "ring/producer.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>   // std::exit
#include <chrono>    // host wall-clock timing for host-side API calls
#include <vector>
#include <set>
#include <algorithm>

using namespace ring;

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

static int g_fail = 0;
#define CHECK(cond,msg) do { if(!(cond)){ printf("  FAIL: %s\n", msg); g_fail++; } } while(0)

#ifndef NPROD
#define NPROD 16
#endif
static constexpr int      N         = NPROD;         // producer nodes (~a hook set); -DNPROD=145 to scale
static constexpr uint64_t SRC_BYTES = 1024*1024;     // 1 MB each
static uint64_t ALLOC;                                 // align_up(SRC_BYTES,16)

// Read the producer-id that wrote a given payload region: src[j] is filled with
// byte value j, so the first byte of the payload identifies the writer.
static int payload_writer_id(uint8_t* payload_buf, uint64_t off) {
    uint8_t b; CK(cudaMemcpy(&b, payload_buf+off, 1, cudaMemcpyDeviceToHost));
    return (int)b;
}

// Reset ring heads + task entries to pristine (between replays).
static void reset_ring(AllocatedRing& ar) {
    RingState& s = ar.state();
    CK(cudaMemset(s.task_entries, 0xFF, s.task_cap*sizeof(TaskEntry))); // ready_seq=SENTINEL
    *s.task_head = 0; *s.payload_head = 0;
    CK(cudaDeviceSynchronize());
}

// Replay once; return the ordered list of producer-ids that actually published,
// and assert payload offsets are contiguous (no gap) and entries well-formed.
static std::vector<int> run_and_collect(AllocatedRing& ar, cudaGraphExec_t exec,
                                        cudaStream_t s, const char* tag) {
    reset_ring(ar);
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    RingState& rs = ar.state();
    uint64_t published = *rs.task_head;
    std::vector<int> ids;
    uint64_t expect_off = 0;
    bool contiguous = true, formed = true;
    std::vector<TaskEntry> ent(published);
    CK(cudaMemcpy(ent.data(), rs.task_entries, published*sizeof(TaskEntry), cudaMemcpyDeviceToHost));
    for (uint64_t i=0;i<published;i++) {
        // ready_seq is the slot's logical sequence number (task_publish writes
        // ready_seq = seq_no = task_head, which starts at 0 after reset). So the
        // i-th published entry MUST have ready_seq == i -- this validates the
        // sequence protocol, not merely that the slot was published.
        if (ent[i].ready_seq != i) formed=false;
        if (ent[i].tensor_total_bytes != SRC_BYTES) formed=false;
        if (ent[i].payload_off1 != expect_off) contiguous=false; // single-span (no wrap: <=16MB < ring)
        ids.push_back(payload_writer_id(rs.payload_buf, ent[i].payload_off1));
        expect_off += ALLOC;
    }
    CHECK(formed,     "all published entries well-formed (ready_seq == slot index, correct size)");
    CHECK(contiguous, "published payload offsets are contiguous (ring closed up, no gap)");
    // payload_head must equal published * ALLOC exactly (heads advanced only by runners)
    CHECK(*rs.payload_head == published*ALLOC, "payload_head == #published * ALLOC");
    printf("  [%s] published=%llu  ids=[", tag, (unsigned long long)published);
    for (size_t i=0;i<ids.size();i++) printf("%d%s", ids[i], i+1<ids.size()?",":"");
    printf("]\n");
    return ids;
}

// Raw replay throughput via CUDA events (legitimate: graph launches ARE GPU
// stream work, so the events bracket real device time).
//
// INTENTIONAL OVERWRITE: there is no consumer/drain here and we do not reset
// between iterations, so over `iters` replays the producers wrap the payload
// ring (iters * N * SRC_BYTES >> payload_cap) and the task ring (iters * N >>
// task_cap) many times. That is fine for *timing* -- the producer kernel never
// blocks (no capacity check on this path) and every config overwrites equally,
// so the relative comparison (enabled vs disabled vs null_mode) is unaffected.
// It would NOT be valid for reading data back; use run_and_collect() for that.
static float time_replays(AllocatedRing& ar, cudaGraphExec_t exec, cudaStream_t s, int iters) {
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    reset_ring(ar);
    for (int i=0;i<10;i++) cudaGraphLaunch(exec,s);      // warmup
    CK(cudaStreamSynchronize(s));
    cudaEventRecord(a,s);
    for (int i=0;i<iters;i++) cudaGraphLaunch(exec,s);   // intentional overwrite (see note above)
    cudaEventRecord(b,s);
    CK(cudaStreamSynchronize(s));
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms/iters*1000.0f;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    int rt=0,drv=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    printf("GPU: %s sm_%d%d | CUDA rt %d drv %d | REAL dual-ring producer\n",
           p.name,p.major,p.minor,rt,drv);
    ALLOC = (SRC_BYTES + 15) & ~uint64_t(15);

    // --- rings.  Sized so a SINGLE run_and_collect() replay (<=16 entries,
    // <=16MB payload) never wraps, which keeps the correctness check simple.
    // The timing loop (time_replays) DOES wrap many times -- intentional, see
    // the note on time_replays(). ---
    RingConfig cfg;
    cfg.task_ring_entries  = 4096;
    cfg.payload_ring_bytes = 256ULL*1024*1024;
    AllocatedRing ar(cfg);
    ar.init();

    // --- N source buffers, src[j] filled with byte value j ---
    std::vector<uint8_t*> src(N);
    for (int j=0;j<N;j++) {
        CK(cudaMalloc(&src[j], SRC_BYTES));
        CK(cudaMemset(src[j], j, SRC_BYTES));
    }

    cudaStream_t s; CK(cudaStreamCreate(&s));

    // --- capture: N real producer launches on one stream (as vLLM captures hooks) ---
    cudaGraph_t graph; cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int j=0;j<N;j++)
        launch_producer(ar.state(), src[j], SRC_BYTES, (uint32_t)j, s);
    CK(cudaStreamEndCapture(s,&graph));
    CK(cudaGraphInstantiate(&exec,graph,0));

    size_t nn=0; CK(cudaGraphGetNodes(graph,nullptr,&nn));
    std::vector<cudaGraphNode_t> nodes(nn); CK(cudaGraphGetNodes(graph,nodes.data(),&nn));
    std::vector<cudaGraphNode_t> knodes;
    for (auto& nd: nodes){ cudaGraphNodeType t; if(cudaGraphNodeGetType(nd,&t)==cudaSuccess && t==cudaGraphNodeTypeKernel) knodes.push_back(nd);}
    printf("captured: %zu graph nodes, %zu kernel nodes (expected %d)\n", nn, knodes.size(), N);
    if (knodes.size() != (size_t)N) {
        printf("FATAL: expected exactly %d kernel nodes, got %zu — aborting before any indexing.\n",
               N, knodes.size());
        return 1;
    }

    // --- Build an explicit knodes-index -> producer-id map. ---
    // cudaGraphGetNodes() does NOT guarantee the returned node order matches the
    // capture/add order, so we must NOT assume knodes[j] is the j-th producer.
    // Probe it directly: enable exactly ONE kernel node at a time, replay, and
    // the single published payload's writer-id tells us which producer that node
    // drives. Then toggle by producer-id, never by raw enumeration index.
    std::vector<int> node_pid(knodes.size(), -1);   // node_pid[k] = producer id driven by knodes[k]
    for (size_t k=0;k<knodes.size();k++) {
        for (size_t m=0;m<knodes.size();m++)
            CK(cudaGraphNodeSetEnabled(exec, knodes[m], m==k ? 1 : 0));
        reset_ring(ar); CK(cudaGraphLaunch(exec,s)); CK(cudaStreamSynchronize(s));
        uint64_t pub = *ar.state().task_head;
        CHECK(pub==1, "exactly one node enabled => exactly one entry published");
        if (pub>=1) {
            TaskEntry e0; CK(cudaMemcpy(&e0, ar.state().task_entries, sizeof(TaskEntry), cudaMemcpyDeviceToHost));
            node_pid[k] = payload_writer_id(ar.state().payload_buf, e0.payload_off1);
        }
    }
    // invert to producer-id -> knodes index, and verify it's a bijection over 0..N-1
    std::vector<int> pid_node(N, -1);
    for (size_t k=0;k<knodes.size();k++)
        if (node_pid[k]>=0 && node_pid[k]<N) pid_node[node_pid[k]] = (int)k;
    bool map_ok=true; for (int j=0;j<N;j++) if (pid_node[j]<0) map_ok=false;
    CHECK(map_ok, "node<->producer map is a bijection over 0..N-1 (every producer reachable)");

    // toggle helpers that work in PRODUCER-ID space (driver-order independent)
    auto enable_all   = [&](){ for (auto& nd: knodes) CK(cudaGraphNodeSetEnabled(exec, nd, 1)); };
    auto set_disabled = [&](const std::set<int>& drop_pids){
        for (int j=0;j<N;j++) CK(cudaGraphNodeSetEnabled(exec, knodes[pid_node[j]], drop_pids.count(j)?0:1));
    };

    printf("\n=== Q2: dual-ring consistency under node-toggle ===\n");
    // (1) all enabled — publish order should equal producer-id order (= capture/stream order)
    enable_all();
    auto ids_full = run_and_collect(ar, exec, s, "all-enabled");
    bool ordered = (ids_full.size()==(size_t)N);
    for (int j=0;j<N && ordered;j++) ordered = (ids_full[j]==j);
    CHECK(ordered, "all-enabled publishes producer-ids 0..N-1 in stream order");

    // (2) disable a subset of PRODUCERS: drop {2,5,7,11,13}
    std::set<int> drop = {2,5,7,11,13};
    set_disabled(drop);
    auto ids_sub = run_and_collect(ar, exec, s, "subset-off");
    std::vector<int> expect;
    for (int j=0;j<N;j++) if(!drop.count(j)) expect.push_back(j);
    CHECK(ids_sub==expect, "remaining producer-ids == enabled set, in order, contiguous (NO desync, NO corruption)");

    // (3) re-enable all
    enable_all();
    auto ids_re = run_and_collect(ar, exec, s, "re-enabled");
    CHECK(ids_re==ids_full, "re-enable restores full aligned set");

    // (4) hammer toggles to check for state leak/corruption across reconfigs
    bool stable=true;
    for (int r=0;r<50;r++) {
        std::set<int> d = { r%N, (r*7+3)%N, (r*13+1)%N };
        set_disabled(d);
        std::vector<int> ex; for(int j=0;j<N;j++) if(!d.count(j)) ex.push_back(j);
        reset_ring(ar); CK(cudaGraphLaunch(exec,s)); CK(cudaStreamSynchronize(s));
        uint64_t pub=*ar.state().task_head;
        std::vector<TaskEntry> ee(pub); CK(cudaMemcpy(ee.data(),ar.state().task_entries,pub*sizeof(TaskEntry),cudaMemcpyDeviceToHost));
        std::vector<int> got; for(uint64_t i=0;i<pub;i++) got.push_back(payload_writer_id(ar.state().payload_buf,ee[i].payload_off1));
        if (got!=ex) stable=false;
    }
    enable_all();
    CHECK(stable, "50 randomized toggle reconfigs all stay aligned (no cumulative corruption)");

    printf("\n=== Q1: per-replay overhead (real producer, %d nodes x %lluMB) ===\n",
           N, (unsigned long long)(SRC_BYTES>>20));
    const int IT=2000;
    enable_all();
    set_ring_null_mode(false);
    float t_full = time_replays(ar,exec,s,IT);
    for (auto&nd:knodes) CK(cudaGraphNodeSetEnabled(exec,nd,0));
    float t_off  = time_replays(ar,exec,s,IT);
    enable_all();
    set_ring_null_mode(true);
    float t_null = time_replays(ar,exec,s,IT);
    set_ring_null_mode(false);
    // half disabled (realistic adaptive case): drop odd producer-ids
    { std::set<int> odd; for(int j=1;j<N;j+=2) odd.insert(j); set_disabled(odd); }
    float t_half = time_replays(ar,exec,s,IT);
    enable_all();

    printf("  (a) all enabled            : %8.2f us\n", t_full);
    printf("  (b) all node-disabled      : %8.2f us   <- true disable\n", t_off);
    printf("  (c) all null_mode soft     : %8.2f us   <- launches+early-return\n", t_null);
    printf("  (d) half node-disabled     : %8.2f us\n", t_half);
    printf("  true-disable saves vs full : %8.2f us (%.1f%%)\n", t_full-t_off, 100.f*(t_full-t_off)/t_full);
    printf("  true-disable saves vs null : %8.2f us (%.1f%% of full)\n", t_null-t_off, 100.f*(t_null-t_off)/t_full);

    // Reconfigure latency: cudaGraphNodeSetEnabled is a HOST API call that
    // enqueues no GPU work, so it must be timed with a host wall-clock, NOT
    // CUDA events (events bracket the GPU stream timeline, which is empty here).
    enable_all();
    CK(cudaDeviceSynchronize());
    const int RC=10000;
    auto h0 = std::chrono::steady_clock::now();
    for (int i=0;i<RC;i++) CK(cudaGraphNodeSetEnabled(exec, knodes[0], i&1));
    auto h1 = std::chrono::steady_clock::now();
    double us_per = std::chrono::duration<double,std::micro>(h1-h0).count() / RC;
    enable_all();  // leave graph fully enabled
    printf("  reconfigure cost (single node)     : %.3f us/call (host wall-clock, %d iters)\n",
           us_per, RC);

    // FULL-SET reconfigure: the realistic "go from full capture to fully off"
    // between steps requires N host calls, issued serially on the host critical
    // path before the next graph launch (no re-instantiation, but not free).
    CK(cudaDeviceSynchronize());
    const int FC=2000;
    auto g0 = std::chrono::steady_clock::now();
    for (int i=0;i<FC;i++) {
        int on = i&1;
        for (auto& nd: knodes) CK(cudaGraphNodeSetEnabled(exec, nd, on));   // traverse + N calls
    }
    auto g1 = std::chrono::steady_clock::now();
    double full_us = std::chrono::duration<double,std::micro>(g1-g0).count()/FC;
    enable_all();
    printf("  full-set flip (ALL %d nodes)       : %.3f us per reconfigure (%.3f us/node), no re-instantiate\n",
           N, full_us, full_us/N);

    printf("\n%s (%d checks failed)\n", g_fail==0?"ALL CHECKS PASSED":"SOME CHECKS FAILED", g_fail);
    return g_fail?1:0;
}
