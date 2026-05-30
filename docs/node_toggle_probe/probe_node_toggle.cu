// probe_node_toggle.cu — standalone de-risk probe for DMI Axis-A #1/#2.
//
// Answers, empirically, on the local driver:
//   Q-API:  does cudaGraphNodeSetEnabled work post-capture (no re-instantiate)?
//   Q4:     latency of toggling a node off/on between replays
//             vs the null_mode mechanism (cudaMemcpyToSymbol of a __device__ flag).
//   Q5:     per-replay cost of  (a) node enabled  (b) node truly disabled
//             (c) node launched-but-early-returns (null_mode style).
//
// It deliberately does NOT use DMI's rings/ClickHouse — it isolates the
// CUDA-graph node-toggle primitive itself, which is the thing that does not
// yet exist in the repo.
//
// Build: nvcc -std=c++17 -arch=native -O2 probe_node_toggle.cu -o probe
// Run:   CUDA_MODULE_LOADING=EAGER ./probe

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>   // host wall-clock for host-side API latency
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

// global "null_mode"-style soft-disable flag, mirrors ring::g_ring_null_mode
__device__ int g_null_mode = 0;

// Each "producer" node: does a chunk of memory traffic (to be a non-trivial
// launch, like a real D2D copy) and bumps a per-node launch counter.
__global__ void producer(const float* src, float* dst, int n, int* launch_count, int node_id) {
    if (g_null_mode) return;                  // null_mode soft disable: still launched
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = t; i < n; i += stride) dst[i] = src[i] * 1.0001f + node_id;
    if (t == 0) atomicAdd(&launch_count[node_id], 1);
}

static float time_replays(cudaGraphExec_t exec, cudaStream_t s, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    // warm
    for (int i=0;i<10;i++) cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);
    cudaEventRecord(a, s);
    for (int i=0;i<iters;i++) cudaGraphLaunch(exec, s);
    cudaEventRecord(b, s);
    cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms/iters*1000.0f; // us per replay
}

int main() {
    int dev=0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int rt=0, drv=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    printf("GPU: %s  sm_%d%d  | CUDA runtime %d driver %d\n",
           prop.name, prop.major, prop.minor, rt, drv);

    const int N = 16;            // number of producer nodes (≈ hooks/layer set)
    const int ELEMS = 256*1024;  // ~1 MB per node payload
    const size_t BYTES = ELEMS*sizeof(float);

    float *src, *dst; int *lc;
    CK(cudaMalloc(&src, BYTES));
    CK(cudaMalloc(&dst, BYTES*N));
    CK(cudaMallocManaged(&lc, N*sizeof(int)));
    CK(cudaMemset(src, 1, BYTES));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    dim3 grid(64), block(256);

    // ---- capture a graph with N producer nodes (stream capture, as vLLM does) ----
    cudaGraph_t graph; cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i=0;i<N;i++)
        producer<<<grid, block, 0, s>>>(src, dst+i*ELEMS, ELEMS, lc, i);
    CK(cudaStreamEndCapture(s, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));

    // enumerate kernel nodes in the captured graph
    size_t numNodes=0; CK(cudaGraphGetNodes(graph, nullptr, &numNodes));
    std::vector<cudaGraphNode_t> nodes(numNodes);
    CK(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
    printf("captured graph: %zu nodes (expected %d producer launches)\n", numNodes, N);

    // map nodes -> kernel nodes in capture order
    std::vector<cudaGraphNode_t> knodes;
    for (auto& nd : nodes) {
        cudaGraphNodeType ty; if (cudaGraphNodeGetType(nd,&ty)==cudaSuccess && ty==cudaGraphNodeTypeKernel)
            knodes.push_back(nd);
    }
    printf("kernel nodes: %zu\n", knodes.size());

    auto reset_lc=[&]{ for(int i=0;i<N;i++) lc[i]=0; cudaDeviceSynchronize(); };

    // ---- baseline: all enabled ----
    reset_lc(); CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    int ran=0; for(int i=0;i<N;i++) ran+=(lc[i]>0);
    printf("\n[all enabled]      nodes that ran: %d/%d\n", ran, N);

    // ---- Q-API: disable a subset post-capture via SetEnabled, NO re-instantiate ----
    // disable odd-indexed kernel nodes
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    int toggled=0;
    for (size_t i=0;i<knodes.size();i++) if (i%2==1) {
        cudaError_t e = cudaGraphNodeSetEnabled(exec, knodes[i], 0);
        if (e!=cudaSuccess){ printf("SetEnabled FAILED: %s\n", cudaGetErrorString(e)); return 1; }
        toggled++;
    }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float toggle_ms=0; cudaEventElapsedTime(&toggle_ms,a,b);
    printf("[SetEnabled] disabled %d nodes, total %.1f us  (%.2f us/node)\n",
           toggled, toggle_ms*1000.f, toggle_ms*1000.f/toggled);

    reset_lc(); CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    ran=0; int oddran=0,evenran=0;
    for(int i=0;i<N;i++){ ran+=(lc[i]>0); if(i%2)oddran+=(lc[i]>0); else evenran+=(lc[i]>0);}
    printf("[subset disabled]  ran: %d/%d  (even=%d expected %d, odd=%d expected 0)\n",
           ran,N,evenran,(N+1)/2,oddran);

    // re-enable
    for (size_t i=0;i<knodes.size();i++) if (i%2==1)
        CK(cudaGraphNodeSetEnabled(exec, knodes[i], 1));
    reset_lc(); CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    ran=0; for(int i=0;i<N;i++) ran+=(lc[i]>0);
    printf("[re-enabled]       ran: %d/%d\n", ran, N);

    // ---- Q4 + Q5: per-replay cost comparisons ----
    const int IT=2000;
    // (a) all enabled
    for (auto&nd:knodes) cudaGraphNodeSetEnabled(exec,nd,1);
    int zero=0; cudaMemcpyToSymbol(g_null_mode,&zero,sizeof(int));
    float t_full = time_replays(exec,s,IT);
    // (b) all truly disabled (node-level)
    for (auto&nd:knodes) cudaGraphNodeSetEnabled(exec,nd,0);
    float t_disabled = time_replays(exec,s,IT);
    // (c) all enabled but null_mode soft-disable (kernel launches, early-returns)
    for (auto&nd:knodes) cudaGraphNodeSetEnabled(exec,nd,1);
    int one=1; cudaMemcpyToSymbol(g_null_mode,&one,sizeof(int));
    float t_null = time_replays(exec,s,IT);
    cudaMemcpyToSymbol(g_null_mode,&zero,sizeof(int));

    printf("\nper-replay (%d nodes, ~1MB each):\n", N);
    printf("  (a) all enabled            : %7.2f us\n", t_full);
    printf("  (b) all node-disabled      : %7.2f us   <- 'true disable'\n", t_disabled);
    printf("  (c) all null_mode soft     : %7.2f us   <- kernels launch, early-return\n", t_null);
    printf("  true-disable saves vs null : %7.2f us  (%.1f%% of full)\n",
           t_null - t_disabled, 100.f*(t_null-t_disabled)/t_full);

    // ---- toggle-mechanism latency: SetEnabled vs cudaMemcpyToSymbol ----
    // Both are HOST API calls (SetEnabled enqueues no GPU work; cudaMemcpyToSymbol
    // of 4 bytes is effectively synchronous host-side). Time with a host
    // wall-clock, NOT CUDA events. Sync first so the device is idle.
    const int RC=10000;
    CK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for(int i=0;i<RC;i++) cudaGraphNodeSetEnabled(exec,knodes[0], i&1);
    auto t1 = std::chrono::steady_clock::now();
    double se_us = std::chrono::duration<double,std::micro>(t1-t0).count()/RC;
    auto t2 = std::chrono::steady_clock::now();
    for(int i=0;i<RC;i++){ int v=i&1; cudaMemcpyToSymbol(g_null_mode,&v,sizeof(int)); }
    auto t3 = std::chrono::steady_clock::now();
    double mc_us = std::chrono::duration<double,std::micro>(t3-t2).count()/RC;
    printf("\ntoggle latency (host wall-clock, %d iters):\n", RC);
    printf("  cudaGraphNodeSetEnabled        : %.3f us/call\n", se_us);
    printf("  cudaMemcpyToSymbol (null_mode) : %.3f us/call\n", mc_us);

    printf("\nOK\n");
    return 0;
}
