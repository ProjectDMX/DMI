// GPU equivalence test for the in-graph linear head node (producer_head /
// launch_producer_head).
//
// Script: eager launches with random hidden sizes (bf16 vector path, fp32
// scalar path), slot counts K in [1,16], random row tables (the gather table
// contract incl. -1 = disabled, 0 = empty task, garbage, > GATHER_MAX_SEGS),
// interleaved with ordinary gather producers into one small wrapping ring;
// then a captured CUDA graph (head node + gather node) replayed 200 times
// while BOTH the row table and the weight-slot arena change between replays
// (register/retire as data, no recapture).
//
// Host model: the same packed row order as the gather contract; scores in
// double. Checks: every non-score payload byte, every task entry, task_head,
// payload_head and actual_bytes_counter EXACTLY; every score within
// 1e-5 * sum_i |w_i h_i| + 1e-6 (fp32 accumulation), NaN where the slot is
// retired (b = NaN). Reports max abs / relative error.
//
// Usage: test_inline_head            (exit 0 = PASS)
//   -DDMI_TEST_MUTATE_SLOT: score slot k with slot (k+1)%K's weights -> must FAIL.
#include "ring/producer.cuh"
#include "ring/publication_word.h"
#include "ring/ring_alloc.h"

#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

using namespace ring;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    exit(2); } } while (0)

static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static uint64_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static int64_t rint_(int64_t lo, int64_t hi) { return lo + int64_t(rnd() % uint64_t(hi - lo + 1)); }
static float rflt() { return float(int64_t(rnd() % 2000001) - 1000000) / 1000000.f; }

static const uint64_t CAP   = 3ull * 1024 * 1024 + 4096 + 16;
static const uint64_t TCAP  = 64;
static const int      TBL_N = 300;
static const int64_t  TBL_W = 1 + 2 * TBL_N;  // table words (table_numel)
static const int      MAXROWS = 700;
static const int      MAXH  = 4096;

struct Model {
    std::vector<uint8_t>   payload;
    std::vector<uint64_t>  words;   // publication word per task slot
    std::map<uint64_t, std::pair<double, double>> score;   // payload pos -> (ref, scale)
    uint64_t th = 0, ph = 0, actual = 0;

    uint64_t pos(const TwoSpan& sp, uint64_t o) { return (o < sp.len1) ? sp.off1 + o : sp.off2 + (o - sp.len1); }
    void publish(const TwoSpan&, uint64_t alloc, uint64_t total) {
        words[th % TCAP] = PUBLICATION_READY | total;
        th += 1; ph += alloc; actual += total;
    }
    // Row set and order exactly as the gather contract (serial clamp form).
    static std::vector<int64_t> rows_of(const std::vector<int64_t>& t, int64_t U) {
        std::vector<int64_t> out;
        int64_t n = t[0]; if (n < 0) n = 0; if (n > int64_t(GATHER_MAX_SEGS)) n = GATHER_MAX_SEGS;
        int64_t remaining = U;
        for (int64_t k = 0; k < n; ++k) {
            int64_t rs = t[1 + 2 * k], nr = t[2 + 2 * k];
            if (rs < 0) rs = 0; if (rs > U) rs = U;
            if (nr < 0) nr = 0; if (nr > U - rs) nr = U - rs; if (nr > remaining) nr = remaining;
            for (int64_t j = 0; j < nr; ++j) out.push_back(rs + j);
            remaining -= nr;
        }
        return out;
    }
    void gather(const std::vector<uint8_t>& src, uint64_t off, uint64_t upper,
                const std::vector<int64_t>& t, uint64_t rb) {
        if (t[0] == -1) return;
        const int64_t U = int64_t(upper / rb);
        std::vector<int64_t> rows = rows_of(t, U);
        const uint64_t total = rows.size() * rb, a = align_up(total, PAYLOAD_ALIGN);
        TwoSpan sp = payload_compute_spans(ph, CAP, a);
        for (size_t r = 0; r < rows.size(); ++r)
            for (uint64_t i = 0; i < rb; ++i) {
                const uint64_t p = pos(sp, r * rb + i);
                payload[p] = src[off + uint64_t(rows[r]) * rb + i];
                score.erase(p - (p % 4));
            }
        publish(sp, a, total);
    }
    // h(row, i) supplied as doubles (bf16/fp32 exact values).
    template <typename H>
    void head(H h, int64_t U, const std::vector<int64_t>& t, uint32_t hidden,
              const std::vector<float>& arena, uint32_t K) {
        if (t[0] == -1) return;
        std::vector<int64_t> rows = rows_of(t, U);
        const uint64_t total = rows.size() * K * 4, a = align_up(total, PAYLOAD_ALIGN);
        TwoSpan sp = payload_compute_spans(ph, CAP, a);
        for (size_t r = 0; r < rows.size(); ++r)
            for (uint32_t k = 0; k < K; ++k) {
#ifdef DMI_TEST_MUTATE_SLOT
                const uint32_t kw = (k + 1) % K;
#else
                const uint32_t kw = k;
#endif
                double acc = 0, scale = 0;
                for (uint32_t i = 0; i < hidden; ++i) {
                    const double v = double(arena[uint64_t(kw) * hidden + i]) * h(rows[r], i);
                    acc += v; scale += std::fabs(v);
                }
                const double b = arena[uint64_t(K) * hidden + k];
                acc += b; scale += std::isnan(b) ? 0 : std::fabs(b);
                const uint64_t p = pos(sp, (r * K + k) * 4);
                score[p] = {acc, scale};
                const float f = float(acc);
                std::memcpy(&payload[p], &f, 4);   // placeholder; compared by tolerance
            }
        publish(sp, a, total);
    }
};

static std::vector<int64_t> random_table(int64_t U) {
    std::vector<int64_t> t(1 + 2 * TBL_N, 0);
    const int mode = int(rint_(0, 9));
    if (mode == 0) { t[0] = -1; return t; }
    if (mode == 1) { t[0] = 0; return t; }
    if (mode == 2) { t[0] = 1; t[1] = 0; t[2] = U; return t; }
    if (mode == 3) { t[0] = 1; t[1] = rint_(0, U); t[2] = rint_(0, U); return t; }
    if (mode == 4) { t[0] = TBL_N; for (int k = 0; k < TBL_N; ++k) { t[1+2*k] = rint_(0, U); t[2+2*k] = rint_(0, 2); } return t; }
    if (mode == 5) {
        int64_t n = rint_(2, 40); t[0] = n;
        for (int k = 0; k < n; ++k) { t[1+2*k] = rint_(-5, U + 5); t[2+2*k] = rint_(-3, U); }
        return t;
    }
    int64_t n = rint_(2, 200), r = 0, k = 0;
    while (k < n && r < U) {
        r += rint_(0, 3); if (r >= U) break;
        int64_t len = rint_(1, 4); if (len > U - r) len = U - r;
        t[1+2*k] = r; t[2+2*k] = len; r += len; ++k;
    }
    t[0] = k; return t;
}

static void random_arena(std::vector<float>& a, uint32_t K, uint32_t hidden) {
    a.assign(size_t(K) * hidden + K, 0.f);
    for (uint32_t k = 0; k < K; ++k) {
        const int state = int(rnd() % 6);          // 0 = retired slot (w = 0, b = NaN)
        for (uint32_t i = 0; i < hidden; ++i) a[size_t(k) * hidden + i] = state ? rflt() / 64.f : 0.f;
        a[size_t(K) * hidden + k] = state ? rflt() : NAN;
    }
}

int main() {
    CK(cudaSetDevice(0));
    RingConfig cfg; cfg.task_ring_entries = TCAP; cfg.payload_ring_bytes = CAP;
    AllocatedRing ar(cfg); ar.init(); CK(cudaDeviceSynchronize());
    RingState rs = ar.state();
    CK(cudaMemset(rs.payload_buf, 0xCD, CAP));
    Model m; m.payload.assign(CAP, 0xCD); m.words.assign(TCAP, 0);

    // Source rows: bf16 view and fp32 view of the same random values.
    const uint64_t NSRC = uint64_t(MAXROWS) * MAXH;
    std::vector<__nv_bfloat16> hb(NSRC); std::vector<float> hf(NSRC);
    for (uint64_t i = 0; i < NSRC; ++i) { hf[i] = rflt() * 4.f; hb[i] = __float2bfloat16(hf[i]); }
    std::vector<uint8_t> hbytes(NSRC * 2);
    std::memcpy(hbytes.data(), hb.data(), NSRC * 2);
    __nv_bfloat16* db; CK(cudaMalloc(&db, NSRC * 2)); CK(cudaMemcpy(db, hb.data(), NSRC * 2, cudaMemcpyHostToDevice));
    float* df; CK(cudaMalloc(&df, NSRC * 4)); CK(cudaMemcpy(df, hf.data(), NSRC * 4, cudaMemcpyHostToDevice));
    int64_t* dt; CK(cudaMalloc(&dt, 8 * (1 + 2 * TBL_N)));
    int64_t* dt2; CK(cudaMalloc(&dt2, 8 * (1 + 2 * TBL_N)));
    float* darena; CK(cudaMalloc(&darena, 4 * (16 * MAXH + 16 + 2048)));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    int n_ops = 0, n_head = 0;
    std::vector<float> arena;
    // ---- eager phase ----
    for (int it = 0; it < 400; ++it) {
        const int kind = int(rnd() % 3);
        if (kind == 0) {   // ordinary gather producer between head nodes
            const uint64_t rb = 8192; const int64_t U = rint_(1, 128); const uint64_t upper = U * rb;
            std::vector<int64_t> t = random_table(U);
            CK(cudaMemcpyAsync(dt, t.data(), 8 * t.size(), cudaMemcpyHostToDevice, s));
            launch_producer_gather(rs, reinterpret_cast<const uint8_t*>(db), upper, dt, TBL_W, rb, 0, s);
            m.gather(hbytes, 0, upper, t, rb);
        } else {
            const bool fp32 = kind == 2;
            const uint32_t hidden = fp32 ? uint32_t(rint_(1, 1100)) : (rnd() % 2 ? 4096u : uint32_t(rint_(1, 512)) * 8u);
            const uint32_t K = (rnd() % 3 == 0) ? 16u : uint32_t(rint_(1, 16));
            const int64_t maxU = int64_t(NSRC / hidden) < MAXROWS ? int64_t(NSRC / hidden) : MAXROWS;
            const int64_t U = (rnd() % 8 == 0) ? rint_(256, maxU) : rint_(1, 48);
            const uint64_t upper = uint64_t(U) * hidden * (fp32 ? 4 : 2);
            std::vector<int64_t> t = random_table(U);
            random_arena(arena, K, hidden);
            CK(cudaMemcpyAsync(dt, t.data(), 8 * t.size(), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(darena, arena.data(), 4 * arena.size(), cudaMemcpyHostToDevice, s));
            launch_producer_head(rs, fp32 ? reinterpret_cast<const uint8_t*>(df) : reinterpret_cast<const uint8_t*>(db),
                                 upper, dt, TBL_W, hidden, fp32 ? 4 : 2, darena, K, 24, s);
            if (fp32) m.head([&](int64_t r, uint32_t i) { return double(hf[uint64_t(r) * hidden + i]); }, U, t, hidden, arena, K);
            else      m.head([&](int64_t r, uint32_t i) { return double(__bfloat162float(hb[uint64_t(r) * hidden + i])); }, U, t, hidden, arena, K);
            ++n_head;
        }
        CK(cudaStreamSynchronize(s));
        ++n_ops;
    }
    // ---- graph phase: head node + gather node captured once; tables AND the
    // weight arena are rewritten between replays (runtime register/retire). ----
    {
        const uint32_t hidden = 4096, K = 8; const int64_t U = 48;
        const uint64_t upper = uint64_t(U) * hidden * 2;
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        launch_producer_gather(rs, reinterpret_cast<const uint8_t*>(db), upper, dt2, TBL_W, hidden * 2, 0, s);
        launch_producer_head(rs, reinterpret_cast<const uint8_t*>(db), upper, dt, TBL_W, hidden, 2, darena, K, 24, s);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        random_arena(arena, K, hidden);
        for (int rep = 0; rep < 200; ++rep) {
            std::vector<int64_t> t = random_table(U), t2 = random_table(U);
            if (rep % 5 == 0) {   // re-register / retire some slots: data only
                std::vector<float> fresh; random_arena(fresh, K, hidden);
                const uint32_t k = uint32_t(rnd() % K);
                for (uint32_t i = 0; i < hidden; ++i) arena[size_t(k) * hidden + i] = fresh[size_t(k) * hidden + i];
                arena[size_t(K) * hidden + k] = fresh[size_t(K) * hidden + k];
            }
            CK(cudaMemcpyAsync(dt, t.data(), 8 * t.size(), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(dt2, t2.data(), 8 * t2.size(), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(darena, arena.data(), 4 * arena.size(), cudaMemcpyHostToDevice, s));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaStreamSynchronize(s));
            m.gather(hbytes, 0, upper, t2, hidden * 2);
            m.head([&](int64_t r, uint32_t i) { return double(__bfloat162float(hb[uint64_t(r) * hidden + i])); }, U, t, hidden, arena, K);
            n_ops += 2; ++n_head;
        }
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
    CK(cudaDeviceSynchronize());

    std::vector<uint8_t> dev(CAP);
    CK(cudaMemcpy(dev.data(), rs.payload_buf, CAP, cudaMemcpyDeviceToHost));
    std::vector<uint64_t> de(TCAP);
    for (uint64_t i = 0; i < TCAP; ++i)
        de[i] = __atomic_load_n(&rs.publication_slots[i], __ATOMIC_ACQUIRE);
    const uint64_t th = *rs.task_head, ph = *rs.payload_head, ac = *rs.actual_bytes_counter;

    int fails = 0;
    uint64_t bad = 0, first = ~0ull, scores = 0, bad_scores = 0, nan_ok = 0;
    double max_abs = 0, max_rel = 0;
    std::vector<uint8_t> is_score(CAP, 0);
    for (auto& kv : m.score) {
        for (int j = 0; j < 4; ++j) is_score[kv.first + j] = 1;
        float d; std::memcpy(&d, &dev[kv.first], 4);
        const double ref = kv.second.first, scale = kv.second.second;
        ++scores;
        if (std::isnan(ref)) { if (std::isnan(d)) ++nan_ok; else ++bad_scores; continue; }
        const double err = std::fabs(double(d) - ref);
        if (!(err <= 1e-5 * scale + 1e-6)) ++bad_scores;
        if (err > max_abs) max_abs = err;
        if (std::fabs(ref) > 1e-3 && err / std::fabs(ref) > max_rel) max_rel = err / std::fabs(ref);
    }
    for (uint64_t i = 0; i < CAP; ++i)
        if (!is_score[i] && dev[i] != m.payload[i]) { if (!bad) first = i; ++bad; }
    if (bad) { printf("FAIL payload: %lu non-score bytes differ (first at %lu)\n", (unsigned long)bad, (unsigned long)first); ++fails; }
    if (bad_scores) { printf("FAIL scores: %lu of %lu outside tolerance\n", (unsigned long)bad_scores, (unsigned long)scores); ++fails; }
    if (de != m.words) { printf("FAIL publication words\n"); ++fails; }
    if (th != m.th || ph != m.ph || ac != m.actual) {
        printf("FAIL heads: task %lu/%lu payload %lu/%lu actual %lu/%lu\n",
               (unsigned long)th, (unsigned long)m.th, (unsigned long)ph, (unsigned long)m.ph,
               (unsigned long)ac, (unsigned long)m.actual); ++fails;
    }
    printf("ops=%d head_launches=%d tasks=%lu ring_wraps=%lu live_scores_checked=%lu (retired NaN %lu) "
           "max_abs_err=%.3g max_rel_err=%.3g -> %s\n",
           n_ops, n_head, (unsigned long)th, (unsigned long)(ph / CAP), (unsigned long)scores,
           (unsigned long)nan_ok, max_abs, max_rel, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
