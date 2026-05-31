// probe_phase3_capture_handles.cu — Phase 3 feasibility probe.
//
// Question: can DMI obtain handles to ITS producer nodes inside a CUDA graph
// that someone ELSE (vLLM / torch.compile) captured, and then toggle them on
// that foreign exec graph?
//
// Mechanism under test: during capture, right after launching a producer kernel,
// DMI calls cudaStreamGetCaptureInfo to read the in-progress graph and the
// current "tail" dependency -- which is exactly the node it just added. It
// records that cudaGraphNode_t. The capture/instantiate is done by the "owner"
// (simulated here by a separate begin/end/instantiate that DMI does not drive).
// Then DMI calls cudaGraphNodeSetEnabled(owner_exec, recorded_node, ...).
//
// This models the real DMI position: its producer kernels are already launched
// DURING the framework's warmup capture, so a producer-side hook can query
// capture info at that exact point.  Combined with the separately-confirmed fact
// that torch.cuda.CUDAGraph exposes raw_cuda_graph_exec, this is the whole
// feasibility chain.
//
// Build: nvcc -std=c++17 -arch=native -O2 probe_phase3_capture_handles.cu -o probe_p3
// Run:   CUDA_MODULE_LOADING=EAGER ./probe_p3

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

static int g_fail = 0;
#define CHECK(c,msg) do { if(!(c)){ printf("  FAIL: %s\n", msg); g_fail++; } } while(0)

static constexpr int N = 8;     // producer "hooks"
static constexpr int ELEMS = 4096;

// Stand-in for model compute (interleaved between producers, like a real graph).
__global__ void compute(float* d, int* compute_count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < ELEMS) d[t] = d[t] * 1.0001f + 1.0f;
    if (t == 0) atomicAdd(compute_count, 1);
}

// Stand-in for a DMI producer kernel; bumps a per-id launch counter.
__global__ void producer(int* launch_count, int id) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&launch_count[id], 1);
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int rt=0, drv=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    printf("GPU: %s sm_%d%d | CUDA rt %d drv %d\n", p.name, p.major, p.minor, rt, drv);

    int *lc, *cc; float* d;
    CK(cudaMallocManaged(&lc, N*sizeof(int)));
    CK(cudaMallocManaged(&cc, sizeof(int)));
    CK(cudaMalloc(&d, ELEMS*sizeof(float)));
    CK(cudaMemset(d, 0, ELEMS*sizeof(float)));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    dim3 g((ELEMS+255)/256), b(256);

    // ---- "Owner" (think: vLLM) drives the capture.  DMI does NOT call
    // begin/end/instantiate; it only launches its producers into the stream and
    // records its node handles via capture info. ----
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));

    std::vector<cudaGraphNode_t> prod_node(N, nullptr);
    for (int i = 0; i < N; i++) {
        compute<<<g, b, 0, s>>>(d, cc);              // model compute
        producer<<<1, 32, 0, s>>>(lc, i);            // DMI producer

        // DMI-side hook: query the in-progress capture to recover the node it
        // just added (the current tail dependency).
        cudaStreamCaptureStatus status;
        unsigned long long id = 0;
        cudaGraph_t cap_graph = nullptr;
        const cudaGraphNode_t* deps = nullptr;
        const cudaGraphEdgeData* edges = nullptr;   // CUDA 13 signature
        size_t ndeps = 0;
        CK(cudaStreamGetCaptureInfo(s, &status, &id, &cap_graph, &deps, &edges, &ndeps));
        CHECK(status == cudaStreamCaptureStatusActive, "capture is active during producer launch");
        CHECK(cap_graph != nullptr, "capture info returns the in-progress cudaGraph_t");
        CHECK(ndeps >= 1, "capture info returns >=1 tail dependency");
        if (ndeps >= 1) prod_node[i] = deps[ndeps - 1];   // tail = the just-launched producer
    }

    cudaGraph_t graph = nullptr;
    CK(cudaStreamEndCapture(s, &graph));               // owner finalizes
    cudaGraphExec_t exec = nullptr;
    CK(cudaGraphInstantiate(&exec, graph, 0));          // owner instantiates

    // sanity: each recorded handle is a kernel node, and is the producer kernel.
    int prod_ok = 0;
    for (int i = 0; i < N; i++) {
        if (!prod_node[i]) continue;
        cudaGraphNodeType ty;
        if (cudaGraphNodeGetType(prod_node[i], &ty) == cudaSuccess && ty == cudaGraphNodeTypeKernel) prod_ok++;
    }
    CHECK(prod_ok == N, "all N recorded handles are kernel nodes");

    auto reset = [&]{ for(int i=0;i<N;i++) lc[i]=0; *cc=0; CK(cudaDeviceSynchronize()); };

    // baseline: everything runs
    reset(); CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    int ran=0; for(int i=0;i<N;i++) ran+=(lc[i]>0);
    printf("\n[baseline]     producers ran: %d/%d, compute ran: %d (want %d)\n", ran, N, *cc, N);
    CHECK(ran==N && *cc==N, "baseline: all producers + all compute ran");

    // ---- DMI disables a SUBSET of its producers on the OWNER's exec graph,
    // using only the node handles it recorded during capture. ----
    int disable[] = {1, 4, 6};
    for (int k : disable) CK(cudaGraphNodeSetEnabled(exec, prod_node[k], 0));
    reset(); CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    bool subset_ok = true;
    for (int i=0;i<N;i++) {
        bool want = !(i==1||i==4||i==6);
        if ((lc[i]>0) != want) subset_ok = false;
    }
    printf("[subset off]   producers ran: ["); for(int i=0;i<N;i++) printf("%d", lc[i]>0); printf("]  compute ran: %d\n", *cc);
    CHECK(subset_ok, "exactly the disabled producers were skipped (toggle worked on foreign exec)");
    CHECK(*cc == N, "model compute still ran fully (only producer nodes toggled)");

    // re-enable
    for (int k : disable) CK(cudaGraphNodeSetEnabled(exec, prod_node[k], 1));
    reset(); CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    ran=0; for(int i=0;i<N;i++) ran+=(lc[i]>0);
    CHECK(ran==N, "re-enable restores all producers");

    printf("\n%s\n", g_fail==0
        ? "FEASIBLE: recovered DMI node handles via cudaStreamGetCaptureInfo during a foreign capture, and toggled them on the owner's exec."
        : "PROBLEM: see failures above.");
    printf("%d checks failed\n", g_fail);
    return g_fail ? 1 : 0;
}
