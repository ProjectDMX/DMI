// GPU equivalence test for the static, prefix, chunked and gather producers.
//
// Runs a fixed pseudo-random script of launches (eager, and graph replays
// whose device-side row counts and row tables change between replays)
// against a small wrapping payload ring, and checks the ENTIRE ring state --
// every payload byte, every publication word, task_head, payload_head and
// actual_bytes_counter -- against an independent host model of the kernels'
// contract: a serial reference of the row-table clamps.  With device graph
// updates it also replays planner-gated device-updatable gather nodes
// (toggle_device.h), within one graph and across two graphs.
//
// Usage: test_producer_equivalence [dump_file]
//   exit 0 = all checks pass; dump_file (optional) receives the final ring
//   state so two builds (e.g. before/after a kernel change) can be compared
//   byte for byte with cmp.
//   -DDMI_TEST_MUTATE_PLANNER gates each device node by its neighbour's row
//   and must FAIL.
#include "ring/producer.cuh"
#include "ring/publication_word.h"
#include "ring/ring_alloc.h"
#include "ring/toggle_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ring;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    exit(2); } } while (0)

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static int64_t rint_(int64_t lo, int64_t hi) { return lo + int64_t(rnd() % uint64_t(hi - lo + 1)); }

static const uint64_t SRC_BYTES = 1200ull * 8192;
static const uint64_t CAP       = 24ull * 1024 * 1024 + 4096 + 16;   // odd-ish, 16-aligned
static const uint64_t TCAP      = 64;
static const int      TBL_N     = 300;  // > GATHER_MAX_SEGS: exercises the clamp
static const int64_t  TBL_W     = 1 + 2 * TBL_N;  // table words (table_numel)

struct Model {
    std::vector<uint8_t>   payload;
    std::vector<uint64_t>  words;   // publication word per task slot
    uint64_t th = 0, ph = 0, actual = 0;
    const std::vector<uint8_t>* src;

    void put(const TwoSpan& sp, uint64_t dst_off, const uint8_t* s, uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
            const uint64_t o = dst_off + i;
            const uint64_t p = (o < sp.len1) ? sp.off1 + o : sp.off2 + (o - sp.len1);
            payload[p] = s[i];
        }
    }
    void publish(const TwoSpan&, uint64_t alloc, uint64_t total) {
        words[th % TCAP] = PUBLICATION_READY | total;
        th += 1; ph += alloc; actual += total;
    }
    void stat(uint64_t off, uint64_t n) {
        const uint64_t a = align_up(n, PAYLOAD_ALIGN);
        TwoSpan sp = payload_compute_spans(ph, CAP, a);
        put(sp, 0, src->data() + off, n);
        publish(sp, a, n);
    }
    void prefix(uint64_t off, uint64_t upper, int64_t rc, uint64_t rb) {
        if (rc < 0) rc = 0;
        uint64_t n = uint64_t(rc) * rb; if (n > upper) n = upper;
        const uint64_t a = align_up(n, PAYLOAD_ALIGN);
        TwoSpan sp = payload_compute_spans(ph, CAP, a);
        put(sp, 0, src->data() + off, n);
        publish(sp, a, n);
    }
    void chunked(uint64_t off, uint64_t upper, const std::vector<int64_t>& T) {
        const uint64_t K = T.size(), cin = upper / K;
        uint64_t total = 0;
        for (uint64_t k = 0; k < K; ++k) {
            int64_t v = T[k]; if (v < 0) v = 0; if (v > int64_t(cin)) v = int64_t(cin);
            total += uint64_t(v);
        }
        const uint64_t a = align_up(total, PAYLOAD_ALIGN);
        TwoSpan sp = payload_compute_spans(ph, CAP, a);
        uint64_t d = 0;
        for (uint64_t k = 0; k < K; ++k) {
            int64_t v = T[k]; if (v < 0) v = 0; if (v > int64_t(cin)) v = int64_t(cin);
            put(sp, d, src->data() + off + k * cin, uint64_t(v)); d += uint64_t(v);
        }
        publish(sp, a, d);
    }
    // Serial reference: a single thread walking the table with a running
    // `remaining` clamp (the kernel's parallel load + warp scan must match).
    void gather(uint64_t off, uint64_t upper, const std::vector<int64_t>& t, uint64_t rb) {
        if (t[0] == -1) return;
        const int64_t U = rb ? int64_t(upper / rb) : 0;
        int64_t n = t[0]; if (n < 0) n = 0; if (n > int64_t(GATHER_MAX_SEGS)) n = GATHER_MAX_SEGS;
        int64_t remaining = U; std::vector<int64_t> srcr, cnt;
        for (int64_t k = 0; k < n; ++k) {
            int64_t rs = t[1 + 2 * k], nr = t[2 + 2 * k];
            if (rs < 0) rs = 0; if (rs > U) rs = U;
            if (nr < 0) nr = 0; if (nr > U - rs) nr = U - rs; if (nr > remaining) nr = remaining;
            srcr.push_back(rs); cnt.push_back(nr); remaining -= nr;
        }
        uint64_t rows = 0; for (int64_t c : cnt) rows += uint64_t(c);
        const uint64_t total = rows * rb, a = align_up(total, PAYLOAD_ALIGN);
        TwoSpan sp = payload_compute_spans(ph, CAP, a);
        uint64_t d = 0;
        for (size_t k = 0; k < cnt.size(); ++k) {
            put(sp, d, src->data() + off + uint64_t(srcr[k]) * rb, uint64_t(cnt[k]) * rb);
            d += uint64_t(cnt[k]) * rb;
        }
        publish(sp, a, total);
    }
};

static std::vector<int64_t> random_table(int64_t U) {
    std::vector<int64_t> t(1 + 2 * TBL_N, 0);
    const int mode = int(rint_(0, 9));
    if (mode == 0) { t[0] = -1; return t; }
    if (mode == 1) { t[0] = 0; return t; }
    if (mode == 2) { t[0] = 1; t[1] = 0; t[2] = U; return t; }                    // full
    if (mode == 3) { t[0] = 1; t[1] = rint_(0, U); t[2] = rint_(0, U); return t; } // one window (may clamp)
    if (mode == 4) { t[0] = TBL_N; for (int k = 0; k < TBL_N; ++k) { t[1+2*k] = rint_(0, U); t[2+2*k] = rint_(0, 3); } return t; }
    if (mode == 5) {   // garbage: negatives, out of range, sum > U
        int64_t n = rint_(2, 40); t[0] = n;
        for (int k = 0; k < n; ++k) { t[1+2*k] = rint_(-5, U + 5); t[2+2*k] = rint_(-3, U); }
        return t;
    }
    // modes 6..9: realistic fragmented selection: disjoint ascending runs
    int64_t n = rint_(2, 200), r = 0, k = 0;
    while (k < n && r < U) {
        r += rint_(0, 3); if (r >= U) break;
        int64_t len = rint_(1, 4); if (len > U - r) len = U - r;
        t[1+2*k] = r; t[2+2*k] = len; r += len; ++k;
    }
    t[0] = k; return t;
}

int main(int argc, char** argv) {
    CK(cudaSetDevice(0));
    RingConfig cfg; cfg.task_ring_entries = TCAP; cfg.payload_ring_bytes = CAP;
    AllocatedRing ar(cfg); ar.init(); CK(cudaDeviceSynchronize());
    RingState rs = ar.state();
    CK(cudaMemset(rs.payload_buf, 0xCD, CAP));

    Model m; m.payload.assign(CAP, 0xCD); m.words.assign(TCAP, 0);
    std::vector<uint8_t> hsrc(SRC_BYTES);
    for (auto& b : hsrc) b = uint8_t(rnd());
    m.src = &hsrc;
    uint8_t* src; CK(cudaMalloc(&src, SRC_BYTES));
    CK(cudaMemcpy(src, hsrc.data(), SRC_BYTES, cudaMemcpyHostToDevice));
    int64_t* dvec; CK(cudaMalloc(&dvec, sizeof(int64_t) * (1 + 2 * TBL_N)));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    const uint64_t rbs[] = {8192, 2064, 4096};
    int n_ops = 0;
    // ---- eager phase ----
    for (int it = 0; it < 600; ++it) {
        const uint64_t rb = rbs[rnd() % 3];
        const int64_t  maxrows = int64_t(SRC_BYTES / rb) - 8;
        // mostly decode-sized, sometimes prefill-sized (grid at the block cap)
        const int64_t  U = (rnd() % 8 == 0) ? rint_(600, maxrows < 1150 ? maxrows : 1150) : rint_(1, 300);
        const uint64_t upper = uint64_t(U) * rb;
        const uint64_t off = uint64_t(rint_(0, 7)) * 16;
        const int kind = int(rnd() % 4);
        if (off + upper > SRC_BYTES) continue;
        if (kind == 0) {
            uint64_t n = upper - uint64_t(rint_(0, 15));            // tail bytes
            launch_producer_static(rs, src + off, n, 0, s); m.stat(off, n);
        } else if (kind == 1) {
            int64_t rc = rint_(-2, U + 3);
            CK(cudaMemcpyAsync(dvec, &rc, 8, cudaMemcpyHostToDevice, s));
            launch_producer_prefix(rs, src + off, upper, dvec, rb, 0, s); m.prefix(off, upper, rc, rb);
        } else if (kind == 2) {
            const uint64_t upper_rows_bytes = upper;
            const uint32_t K = uint32_t(rint_(2, 8));
            // chunk_in = upper/K must stay uint4-aligned (callers' contract)
            const uint64_t upper = (upper_rows_bytes / (16ull * K)) * 16ull * K;
            if (upper == 0) continue;
            std::vector<int64_t> T(K);
            for (auto& v : T) v = rint_(-16, int64_t(upper / K) + 64) & ~int64_t(15);
            CK(cudaMemcpyAsync(dvec, T.data(), 8 * K, cudaMemcpyHostToDevice, s));
            launch_producer_chunked(rs, src + off, upper, dvec, K, 0, s); m.chunked(off, upper, T);
        } else {
            std::vector<int64_t> t = random_table(U);
            CK(cudaMemcpyAsync(dvec, t.data(), 8 * t.size(), cudaMemcpyHostToDevice, s));
            launch_producer_gather(rs, src + off, upper, dvec, TBL_W, rb, 0, s); m.gather(off, upper, t, rb);
        }
        CK(cudaStreamSynchronize(s));   // dvec is rewritten next iteration
        ++n_ops;
    }
    // ---- graph phase: grid baked by the upper bound, values re-read ----
    {
        const uint64_t rb = 8192; const int64_t U = 160; const uint64_t upper = U * rb;
        int64_t* dt; CK(cudaMalloc(&dt, sizeof(int64_t) * (1 + 2 * TBL_N)));
        int64_t* drc; CK(cudaMalloc(&drc, sizeof(int64_t)));
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        launch_producer_gather(rs, src, upper, dt, TBL_W, rb, 0, s);
        launch_producer_static(rs, src + 4096, 150 * rb, 0, s);
        launch_producer_prefix(rs, src + 16, upper, drc, rb, 0, s);
        launch_producer_gather(rs, src + 32, upper, dt, TBL_W, rb, 0, s);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int rep = 0; rep < 120; ++rep) {
            std::vector<int64_t> t = random_table(U);
            int64_t rc = rint_(0, U);
            CK(cudaMemcpyAsync(dt, t.data(), 8 * t.size(), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(drc, &rc, 8, cudaMemcpyHostToDevice, s));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaStreamSynchronize(s));
            m.gather(0, upper, t, rb); m.stat(4096, 150 * rb);
            m.prefix(16, upper, rc, rb); m.gather(32, upper, t, rb);
            n_ops += 4;
        }
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
#if DMX_HAS_DEVICE_GRAPH_UPDATE
    // ---- device-toggle phase: planner-gated device-updatable gather nodes ----
    // Same contract as an early-exit launch: a node the planner disables
    // (row[0] == -1) must leave the ring exactly as the -1 early exit does.
    // Half of the replays force -1 rows; others use random_table (which
    // itself yields -1 sometimes). 3 gather nodes + one ordinary static node.
    {
        const uint64_t rb = 8192; const int64_t U = 160; const uint64_t upper = U * rb;
        const int NG = 3;
        int64_t* dt[NG];
        for (int i = 0; i < NG; ++i) CK(cudaMalloc(&dt[i], sizeof(int64_t) * (1 + 2 * TBL_N)));
        ring_py::ToggleDevPool* pool;
        CK(cudaMallocManaged(&pool, sizeof(ring_py::ToggleDevPool)));
        std::memset(pool, 0, sizeof(ring_py::ToggleDevPool));
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        launch_toggle_planner(pool, 0, s);
        for (int i = 0; i < NG; ++i) {
            CK(launch_producer_gather_device_updatable(rs, src + 48 * i, upper, dt[i], TBL_W, rb, 0, s));
            cudaStreamCaptureStatus st; const cudaGraphNode_t* deps; size_t nd;
            CK(dmx_capture_info(s, &st, nullptr, nullptr, &deps, &nd));
            cudaKernelNodeAttrValue v = {};
            CK(cudaGraphKernelNodeGetAttribute(deps[nd - 1], cudaKernelNodeAttributeDeviceUpdatableKernelNode, &v));
            if (!v.deviceUpdatableKernelNode.deviceUpdatable || !v.deviceUpdatableKernelNode.devNode) { printf("FAIL devNode\n"); return 1; }
#ifdef DMI_TEST_MUTATE_PLANNER   // gate each node by its NEIGHBOUR's row -> must FAIL
            pool->entry[i] = ring_py::ToggleDevEntry{dt[(i + 1) % NG], v.deviceUpdatableKernelNode.devNode};
#else
            pool->entry[i] = ring_py::ToggleDevEntry{dt[i], v.deviceUpdatableKernelNode.devNode};
#endif
            pool->slot[0].n = i + 1;
            if (i == 0) launch_producer_static(rs, src + 4096, 150 * rb, 0, s);
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphUpload(ge, s));
        int disabled = 0;
        for (int rep = 0; rep < 200; ++rep) {
            std::vector<int64_t> t[NG];
            for (int i = 0; i < NG; ++i) {
                t[i] = random_table(U);
                if (rnd() % 2) { t[i].assign(1 + 2 * TBL_N, 0); t[i][0] = -1; }
                disabled += t[i][0] == -1;
                CK(cudaMemcpyAsync(dt[i], t[i].data(), 8 * t[i].size(), cudaMemcpyHostToDevice, s));
            }
            CK(cudaGraphLaunch(ge, s));
            CK(cudaStreamSynchronize(s));
            m.gather(0, upper, t[0], rb); m.stat(4096, 150 * rb);
            m.gather(48, upper, t[1], rb); m.gather(96, upper, t[2], rb);
            n_ops += 4;
        }
        printf("device-toggle phase: 200 replays, %d of %d gather nodes planner-disabled\n", disabled, 200 * NG);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        CK(cudaFree(pool));
    }
    // ---- cross-graph phase: ONE planner (in graph G0) gates the producers
    // of a later graph G1 too (a forward captured as consecutive segment
    // graphs). ----
    {
        const uint64_t rb = 4096; const int64_t U = 96; const uint64_t upper = U * rb;
        int64_t* dt[3];
        for (int i = 0; i < 3; ++i) CK(cudaMalloc(&dt[i], sizeof(int64_t) * (1 + 2 * TBL_N)));
        ring_py::ToggleDevPool* pool;
        CK(cudaMallocManaged(&pool, sizeof(ring_py::ToggleDevPool)));
        std::memset(pool, 0, sizeof(ring_py::ToggleDevPool));
        cudaGraph_t g[2]; cudaGraphExec_t ge[2];
        int e = 0;
        for (int gi = 0; gi < 2; ++gi) {
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
            if (gi == 0) launch_toggle_planner(pool, 0, s);
            for (int i = (gi == 0 ? 0 : 1); i < (gi == 0 ? 1 : 3); ++i) {
                CK(launch_producer_gather_device_updatable(rs, src + 64 * i, upper, dt[i], TBL_W, rb, 0, s));
                cudaStreamCaptureStatus st; const cudaGraphNode_t* deps; size_t nd;
                CK(dmx_capture_info(s, &st, nullptr, nullptr, &deps, &nd));
                cudaKernelNodeAttrValue v = {};
                CK(cudaGraphKernelNodeGetAttribute(deps[nd - 1], cudaKernelNodeAttributeDeviceUpdatableKernelNode, &v));
                pool->entry[e++] = ring_py::ToggleDevEntry{dt[i], v.deviceUpdatableKernelNode.devNode};
                pool->slot[0].n = e;
            }
            CK(cudaStreamEndCapture(s, &g[gi]));
            CK(cudaGraphInstantiate(&ge[gi], g[gi], 0));
            CK(cudaGraphUpload(ge[gi], s));
        }
        int disabled = 0;
        for (int rep = 0; rep < 200; ++rep) {
            std::vector<int64_t> t[3];
            for (int i = 0; i < 3; ++i) {
                t[i] = random_table(U);
                if (rnd() % 2) { t[i].assign(1 + 2 * TBL_N, 0); t[i][0] = -1; }
                disabled += t[i][0] == -1;
                CK(cudaMemcpyAsync(dt[i], t[i].data(), 8 * t[i].size(), cudaMemcpyHostToDevice, s));
            }
            CK(cudaGraphLaunch(ge[0], s));
            CK(cudaGraphLaunch(ge[1], s));
            CK(cudaStreamSynchronize(s));
            m.gather(0, upper, t[0], rb); m.gather(64, upper, t[1], rb); m.gather(128, upper, t[2], rb);
            n_ops += 3;
        }
        printf("cross-graph phase: 200 replays x 2 graphs, %d of 600 nodes planner-disabled, planner errors %llu\n",
               disabled, pool->errors);
        if (pool->errors) { printf("FAIL planner errors\n"); return 1; }
        for (int gi = 0; gi < 2; ++gi) { CK(cudaGraphExecDestroy(ge[gi])); CK(cudaGraphDestroy(g[gi])); }
        CK(cudaFree(pool));
    }
#endif  // DMX_HAS_DEVICE_GRAPH_UPDATE
    CK(cudaDeviceSynchronize());

    std::vector<uint8_t> dev(CAP);
    CK(cudaMemcpy(dev.data(), rs.payload_buf, CAP, cudaMemcpyDeviceToHost));
    std::vector<uint64_t> de(TCAP);
    for (uint64_t i = 0; i < TCAP; ++i)
        de[i] = __atomic_load_n(&rs.publication_slots[i], __ATOMIC_ACQUIRE);
    const uint64_t th = *rs.task_head, ph = *rs.payload_head, ac = *rs.actual_bytes_counter;

    int fails = 0;
    uint64_t bad = 0, first = ~0ull;
    for (uint64_t i = 0; i < CAP; ++i) if (dev[i] != m.payload[i]) { if (!bad) first = i; ++bad; }
    if (bad) { printf("FAIL payload: %lu bytes differ (first at %lu)\n", (unsigned long)bad, (unsigned long)first); ++fails; }
    if (de != m.words) { printf("FAIL publication words\n"); ++fails; }
    if (th != m.th || ph != m.ph || ac != m.actual) {
        printf("FAIL heads: task %lu/%lu payload %lu/%lu actual %lu/%lu\n",
               (unsigned long)th, (unsigned long)m.th, (unsigned long)ph, (unsigned long)m.ph,
               (unsigned long)ac, (unsigned long)m.actual); ++fails;
    }
    printf("ops=%d tasks=%lu payload_head=%lu (ring wrapped %lu times) actual=%lu -> %s\n",
           n_ops, (unsigned long)th, (unsigned long)ph, (unsigned long)(ph / CAP), (unsigned long)ac,
           fails ? "FAIL" : "PASS");
    if (argc > 1) {
        FILE* f = fopen(argv[1], "wb");
        if (!f) { perror("dump"); return 2; }
        fwrite(dev.data(), 1, CAP, f);
        fwrite(de.data(), sizeof(uint64_t), TCAP, f);
        fwrite(&th, 8, 1, f); fwrite(&ph, 8, 1, f); fwrite(&ac, 8, 1, f);
        fclose(f);
    }
    return fails ? 1 : 0;
}
