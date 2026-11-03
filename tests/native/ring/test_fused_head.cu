// GPU equivalence test for the fused producer (producer_fused_kernel /
// launch_producer_fused): the HookPoint's
// row producer and its in-graph linear heads in one node, incl. multi-layer
// heads as per-layer partial sums.
//
// Phases:
//  1. eager: random hidden sizes (vector + scalar paths), bf16/fp32 sources,
//     bf16/fp32 weights, K in [1,16], random slot modes, random gather and
//     head tables (-1 / 0 / null part / garbage / > GATHER_MAX_SEGS) -- MID
//     only with disjoint tables (planner contract) -- interleaved with plain
//     producer_gather launches into one small wrapping ring;
//  2. multi-layer forwards (eager): L fused nodes per forward, slots holding
//     FIRST/MID.../LAST chains across nodes, rows routed per node;
//  3. batch invariance: a row's scores are bit-identical when computed alone,
//     inside a 37-row tile mix, and with a different capture bound (grid);
//  4. CUDA graph: a 6-node forward captured once, replayed 200 times with the
//     tables, slot modes and weights rewritten between replays (register /
//     retire as data, no recapture).
// Host model: exact ring bytes for rows, publication words, task/payload heads and
// actual_bytes_counter; scores in double from the same (bf16-rounded) weights,
// within 1e-5 * sum|w h| + 1e-6 (fp32 accumulation); NaN for free/FIRST/MID
// columns. The model mirrors the accumulator cell by cell (stale cells too).
//
// Mutants (NVCC_EXTRA=...; each must FAIL): -DDMI_TEST_MUTATE_SLOT
// (neighbour slot's weights), -DDMI_TEST_MUTATE_ACC (MID overwrites instead
// of adding), -DDMI_TEST_MUTATE_ORDER (head task published before the row
// task).
//
// Environment: FUSED_TEST_PHASES (default "1234"), FUSED_TEST_N1 (phase-1
// launches, default 500), FUSED_TEST_KINDS (bit mask of launch kinds),
// FUSED_TEST_CHECK_EACH (check the ring after every phase-1 launch).
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

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static int64_t rint_(int64_t lo, int64_t hi) { return lo + int64_t(rnd() % uint64_t(hi - lo + 1)); }
static float rflt() { return float(int64_t(rnd() % 2000001) - 1000000) / 1000000.f; }

static const uint64_t CAP     = 3ull * 1024 * 1024 + 4096 + 16;
static const uint64_t TCAP    = 64;
static const int      TBL_N   = 300;
static const int64_t  TBL_W   = 1 + 2 * TBL_N;  // table words (table_numel)
static const int      MAXROWS = 700;
static const int      MAXH    = 4096;
static const uint64_t ACC_ROWS = 600;   // < MAXROWS: rows beyond it exercise the bound

struct NodeW {
    uint32_t K = 1, hidden = 8; bool wbf16 = true;
    int mode[16]; int acc[16]; float bias[16];
    std::vector<float> w;                 // K*hidden, exactly the device values
    std::vector<int32_t> words() const {
        std::vector<int32_t> v(head_node_words(K, hidden, wbf16), 0);
        for (int k = 0; k < 16; ++k) { v[k] = mode[k]; v[16 + k] = acc[k]; std::memcpy(&v[HEAD_BIAS_WORD + k], &bias[k], 4); }
        if (wbf16) {
            __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&v[HEAD_HDR_WORDS]);
            for (size_t i = 0; i < w.size(); ++i) b[i] = __float2bfloat16(w[i]);
        } else std::memcpy(&v[HEAD_HDR_WORDS], w.data(), 4 * w.size());
        return v;
    }
    void randomize_slot(uint32_t k, int m, int a) {
        mode[k] = m; acc[k] = a; bias[k] = (m == HEAD_FREE) ? NAN : rflt();
        for (uint32_t i = 0; i < hidden; ++i) {
            float x = (m == HEAD_FREE) ? 0.f : rflt() / 64.f;
            if (wbf16) x = __bfloat162float(__float2bfloat16(x));
            w[size_t(k) * hidden + i] = x;
        }
    }
    void init(uint32_t K_, uint32_t h_, bool bf) {
        K = K_; hidden = h_; wbf16 = bf; w.assign(size_t(K) * hidden, 0.f);
        for (int k = 0; k < 16; ++k) { mode[k] = HEAD_FREE; acc[k] = 0; bias[k] = NAN; }
    }
};

struct Model {
    std::vector<uint8_t>   payload;
    std::vector<uint64_t>  words;   // publication word per task slot
    std::map<uint64_t, std::pair<double, double>> score;   // payload pos -> (ref, scale)
    std::vector<double> acc, accs;                           // [ACC_ROWS * HEAD_MAX_ACC]
    uint64_t th = 0, ph = 0, actual = 0;

    uint64_t pos(const TwoSpan& sp, uint64_t o) { return (o < sp.len1) ? sp.off1 + o : sp.off2 + (o - sp.len1); }
    void publish_entry(uint64_t seq, const TwoSpan&, uint64_t total) {
        words[seq % TCAP] = PUBLICATION_READY | total;
    }
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
    // plain producer_gather between fused nodes
    void gather(const std::vector<uint8_t>& src, uint64_t upper, const std::vector<int64_t>& t, uint64_t rb) {
        if (t[0] == -1) return;
        std::vector<int64_t> rows = rows_of(t, int64_t(upper / rb));
        const uint64_t total = rows.size() * rb, a = align_up(total, PAYLOAD_ALIGN);
        TwoSpan sp = payload_compute_spans(ph, CAP, a);
        write_rows(src, rows, rb, sp);
        publish_entry(th, sp, total); th += 1; ph += a; actual += total;
    }
    void write_rows(const std::vector<uint8_t>& src, const std::vector<int64_t>& rows, uint64_t rb, const TwoSpan& sp) {
        for (size_t r = 0; r < rows.size(); ++r)
            for (uint64_t i = 0; i < rb; ++i) {
                const uint64_t p = pos(sp, r * rb + i);
                payload[p] = src[uint64_t(rows[r]) * rb + i];
                score.erase(p - (p % 4));
            }
    }
    // fused node: gtab/htab may be null (part absent)
    template <typename H>
    void fused(const std::vector<uint8_t>& src_bytes, H h, uint32_t elem, int64_t U,
               const std::vector<int64_t>* gt, const std::vector<int64_t>* ht, const NodeW& nd) {
        const bool g_on = gt && (*gt)[0] != -1, h_on = ht && (*ht)[0] != -1;
        if (!g_on && !h_on) return;
        const uint64_t rb = uint64_t(nd.hidden) * elem;
        std::vector<int64_t> grows = g_on ? rows_of(*gt, U) : std::vector<int64_t>();
        std::vector<int64_t> hrows = h_on ? rows_of(*ht, U) : std::vector<int64_t>();
        const uint64_t gtot = grows.size() * rb, ga = align_up(gtot, PAYLOAD_ALIGN);
        const uint64_t htot = hrows.size() * nd.K * 4, ha = align_up(htot, PAYLOAD_ALIGN);
        const TwoSpan gsp = payload_compute_spans(ph, CAP, ga);
        const TwoSpan hsp = payload_compute_spans(ph + (g_on ? ga : 0), CAP, ha);
        if (g_on) write_rows(src_bytes, grows, rb, gsp);
        for (size_t r = 0; r < hrows.size(); ++r) {
            const int64_t sr = hrows[r];
            for (uint32_t k = 0; k < nd.K; ++k) {
#ifdef DMI_TEST_MUTATE_SLOT
                const uint32_t kw = (k + 1) % nd.K;
#else
                const uint32_t kw = k;
#endif
                const int m = nd.mode[k];
                double dot = 0, sc = 0;
                if (m != HEAD_FREE)
                    for (uint32_t i = 0; i < nd.hidden; ++i) {
                        const double v = double(nd.w[size_t(kw) * nd.hidden + i]) * h(sr, i);
                        dot += v; sc += std::fabs(v);
                    }
                double ref = NAN, scale = 0;
                const bool in_acc = uint64_t(sr) < ACC_ROWS;
                const size_t c = size_t(sr) * HEAD_MAX_ACC + size_t(nd.acc[k]);
                if (m == HEAD_SINGLE) { ref = dot + nd.bias[k]; scale = sc + std::fabs(nd.bias[k]); }
                else if (m == HEAD_FIRST && in_acc) { acc[c] = dot; accs[c] = sc; }
#ifdef DMI_TEST_MUTATE_ACC
                else if (m == HEAD_MID && in_acc) { acc[c] = dot; accs[c] = sc; }
#else
                else if (m == HEAD_MID && in_acc) { acc[c] += dot; accs[c] += sc; }
#endif
                else if (m == HEAD_LAST && in_acc) { ref = acc[c] + dot + nd.bias[k]; scale = accs[c] + sc + std::fabs(nd.bias[k]); }
                const uint64_t p = pos(hsp, (r * nd.K + k) * 4);
                score[p] = {ref, scale};
                const float f = float(ref); std::memcpy(&payload[p], &f, 4);
            }
        }
        uint64_t t0 = th;
#ifdef DMI_TEST_MUTATE_ORDER
        if (h_on) { publish_entry(t0++, hsp, htot); }
        if (g_on) { publish_entry(t0++, gsp, gtot); }
#else
        if (g_on) { publish_entry(t0++, gsp, gtot); }
        if (h_on) { publish_entry(t0++, hsp, htot); }
#endif
        th = t0;
        ph += (g_on ? ga : 0) + (h_on ? ha : 0);
        actual += (g_on ? gtot : 0) + (h_on ? htot : 0);
    }
};

static std::vector<int64_t> random_table(int64_t U, bool disjoint) {
    std::vector<int64_t> t(1 + 2 * TBL_N, 0);
    const int mode = int(rint_(0, 9));
    if (mode == 0) { t[0] = -1; return t; }
    if (mode == 1) { t[0] = 0; return t; }
    if (mode == 2) { t[0] = 1; t[1] = 0; t[2] = U; return t; }
    if (!disjoint && mode == 3) { t[0] = 1; t[1] = rint_(0, U); t[2] = rint_(0, U); return t; }
    if (!disjoint && mode == 4) { t[0] = TBL_N; for (int k = 0; k < TBL_N; ++k) { t[1+2*k] = rint_(0, U); t[2+2*k] = rint_(0, 2); } return t; }
    if (!disjoint && mode == 5) {
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

// Disjoint table containing every row of `must` plus random extra rows.
static std::vector<int64_t> table_with(int64_t U, const std::vector<char>& must, int extra_pct) {
    std::vector<char> sel(U, 0);
    for (int64_t r = 0; r < U; ++r) sel[r] = must[r] || int(rnd() % 100) < extra_pct;
    std::vector<int64_t> t(1 + 2 * TBL_N, 0);
    int64_t k = 0;
    for (int64_t r = 0; r < U && k < int64_t(GATHER_MAX_SEGS); ) {
        if (!sel[r]) { ++r; continue; }
        int64_t s = r; while (r < U && sel[r]) ++r;
        t[1 + 2 * k] = s; t[2 + 2 * k] = r - s; ++k;
    }
    t[0] = k; return t;
}

int main() {
    CK(cudaSetDevice(0));
    RingConfig cfg; cfg.task_ring_entries = TCAP; cfg.payload_ring_bytes = CAP;
    AllocatedRing ar(cfg); ar.init(); CK(cudaDeviceSynchronize());
    RingState rs = ar.state();
    CK(cudaMemset(rs.payload_buf, 0xCD, CAP));
    Model m; m.payload.assign(CAP, 0xCD); m.words.assign(TCAP, 0);
    m.acc.assign(ACC_ROWS * HEAD_MAX_ACC, 0.0); m.accs.assign(ACC_ROWS * HEAD_MAX_ACC, 0.0);

    const uint64_t NSRC = uint64_t(MAXROWS) * MAXH;
    std::vector<__nv_bfloat16> hb(NSRC); std::vector<float> hf(NSRC);
    for (uint64_t i = 0; i < NSRC; ++i) { hf[i] = rflt() * 4.f; hb[i] = __float2bfloat16(hf[i]); }
    std::vector<uint8_t> bbytes(NSRC * 2), fbytes(NSRC * 4);
    std::memcpy(bbytes.data(), hb.data(), NSRC * 2); std::memcpy(fbytes.data(), hf.data(), NSRC * 4);
    __nv_bfloat16* db; CK(cudaMalloc(&db, NSRC * 2)); CK(cudaMemcpy(db, hb.data(), NSRC * 2, cudaMemcpyHostToDevice));
    float* df; CK(cudaMalloc(&df, NSRC * 4)); CK(cudaMemcpy(df, hf.data(), NSRC * 4, cudaMemcpyHostToDevice));
    float* dacc; CK(cudaMalloc(&dacc, 4 * ACC_ROWS * HEAD_MAX_ACC)); CK(cudaMemset(dacc, 0, 4 * ACC_ROWS * HEAD_MAX_ACC));
    const int NT = 16;
    int64_t* dt[NT]; for (int i = 0; i < NT; ++i) CK(cudaMalloc(&dt[i], 8 * (1 + 2 * TBL_N)));
    const int NN = 8;
    int32_t* dn[NN]; for (int i = 0; i < NN; ++i) CK(cudaMalloc(&dn[i], 4 * head_node_words(16, MAXH, 0)));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    auto hb_at = [&](uint32_t hidden) { return [&, hidden](int64_t r, uint32_t i) { return double(__bfloat162float(hb[uint64_t(r) * hidden + i])); }; };
    auto hf_at = [&](uint32_t hidden) { return [&, hidden](int64_t r, uint32_t i) { return double(hf[uint64_t(r) * hidden + i]); }; };
    auto up_table = [&](int i, const std::vector<int64_t>& t) { CK(cudaMemcpyAsync(dt[i], t.data(), 8 * t.size(), cudaMemcpyHostToDevice, s)); };
    auto up_node = [&](int i, const NodeW& nd) { std::vector<int32_t> w = nd.words(); CK(cudaMemcpyAsync(dn[i], w.data(), 4 * w.size(), cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s)); };

    int n_fused = 0, n_ops = 0;
    const char* phases = getenv("FUSED_TEST_PHASES") ? getenv("FUSED_TEST_PHASES") : "1234";
    auto on = [&](char c) { return std::strchr(phases, c) != nullptr; };
    const int n1 = getenv("FUSED_TEST_N1") ? atoi(getenv("FUSED_TEST_N1")) : 500;
    const int kinds = getenv("FUSED_TEST_KINDS") ? atoi(getenv("FUSED_TEST_KINDS")) : 15;   // bit0 gather, bit1 bf16 head, bit2/3 fp32 head
    // ---- 1. eager random single nodes ----
    for (int it = 0; it < (on('1') ? n1 : 0); ++it) {
        const int kind = int(rnd() % 4);
        if (!((kinds >> kind) & 1)) continue;
        if (kind == 0) {
            const uint64_t rb = 8192; const int64_t U = rint_(1, 128); const uint64_t upper = U * rb;
            std::vector<int64_t> t = random_table(U, false); up_table(0, t);
            launch_producer_gather(rs, reinterpret_cast<const uint8_t*>(db), upper, dt[0], TBL_W, rb, 0, s);
            m.gather(bbytes, upper, t, rb);
        } else {
            const bool fp32 = kind == 3, wbf = rnd() % 2;
            const uint32_t hidden = fp32 ? uint32_t(rint_(1, 1100)) : (rnd() % 2 ? 4096u : (rnd() % 4 ? uint32_t(rint_(1, 512)) * 8u : uint32_t(rint_(1, 1100))));
            const uint32_t K = (rnd() % 3 == 0) ? 16u : uint32_t(rint_(1, 16));
            const int64_t maxU = int64_t(NSRC / hidden) < MAXROWS ? int64_t(NSRC / hidden) : MAXROWS;
            int64_t U = (rnd() % 8 == 0) ? rint_(256, maxU) : rint_(1, 48);
            if (uint64_t(U) * hidden * (fp32 ? 4 : 2) > CAP / 2) U = int64_t(CAP / 2 / (uint64_t(hidden) * (fp32 ? 4 : 2)));
            const uint64_t upper = uint64_t(U) * hidden * (fp32 ? 4 : 2);
            const bool garbage = rnd() % 2;
            NodeW nd; nd.init(K, hidden, wbf);
            for (uint32_t k = 0; k < K; ++k) {
                int md = int(rnd() % 5);
                if (garbage && md == HEAD_MID) md = HEAD_SINGLE;   // MID needs disjoint rows
                nd.randomize_slot(k, md, int(k));   // production: accumulator column = slot
            }
            std::vector<int64_t> gt = random_table(U, false), ht = random_table(U, !garbage);
            const int parts = int(rnd() % 4);   // 0: both, 1: gather null, 2: head null, 3: both
            // The row copy keeps the gather contract: uint4 rows (row_bytes % 16 == 0).
            const bool gnull = parts == 1 || (uint64_t(hidden) * (fp32 ? 4 : 2)) % 16 != 0, hnull = parts == 2;
            up_table(0, gt); up_table(1, ht); up_node(0, nd);
            launch_producer_fused(rs, fp32 ? reinterpret_cast<const uint8_t*>(df) : reinterpret_cast<const uint8_t*>(db),
                                  upper, hidden, fp32 ? 4 : 2, gnull ? nullptr : dt[0], TBL_W, hnull ? nullptr : dt[1], TBL_W,
                                  dn[0], K, wbf, dacc, ACC_ROWS, 0, s);
            if (fp32) m.fused(fbytes, hf_at(hidden), 4, U, gnull ? nullptr : &gt, hnull ? nullptr : &ht, nd);
            else      m.fused(bbytes, hb_at(hidden), 2, U, gnull ? nullptr : &gt, hnull ? nullptr : &ht, nd);
            ++n_fused;
        }
        CK(cudaStreamSynchronize(s));
        ++n_ops;
    }
    // ---- 2. multi-layer forwards (eager): chains FIRST -> MID* -> LAST ----
    int chains = 0;
    for (int f = 0; f < (on('2') ? 60 : 0); ++f) {
        const uint32_t hidden = (f % 3 == 0) ? 4096u : uint32_t(rint_(1, 64)) * 8u;
        const uint32_t K = uint32_t(rint_(2, 16)); const bool wbf = f % 2;
        const int L = int(rint_(2, 6));
        // A launch never exceeds the ring (the host reservation guarantees it): big row
        // counts only with narrow rows.
        const int64_t U = (f % 5 == 0 && hidden <= 1024) ? int64_t(rint_(256, 650)) : rint_(1, 48);
        const uint64_t upper = uint64_t(U) * hidden * 2;
        std::vector<NodeW> nodes(L);
        for (int l = 0; l < L; ++l) { nodes[l].init(K, hidden, wbf); }
        // a few chains over random layer subsets, each with its own acc column
        std::vector<char> need(U, 0);
        const int nch = int(rint_(1, K < 4 ? K : 4));
        for (int c = 0; c < nch; ++c) {
            const uint32_t k = uint32_t(c); const int a = int(k);
            int first = int(rint_(0, L - 2)), last = int(rint_(first + 1, L - 1));
            for (int l = first; l <= last; ++l) {
                const int md = (l == first) ? HEAD_FIRST : (l == last) ? HEAD_LAST : (rnd() % 3 ? HEAD_MID : HEAD_FREE);
                nodes[l].randomize_slot(k, md, a);
            }
            ++chains;
        }
        for (int l = 0; l < L; ++l)
            for (uint32_t k = uint32_t(nch); k < K; ++k) nodes[l].randomize_slot(k, rnd() % 2 ? HEAD_SINGLE : HEAD_FREE, 0);
        for (int64_t r = 0; r < U; ++r) need[r] = int(rnd() % 100) < 30;
        for (int l = 0; l < L; ++l) {
            std::vector<int64_t> ht = table_with(U, need, 10), gt = random_table(U, false);
            up_table(2 + l, ht); up_table(8 + l, gt); up_node(1 + l, nodes[l]);
            launch_producer_fused(rs, reinterpret_cast<const uint8_t*>(db), upper, hidden, 2, dt[8 + l], TBL_W, dt[2 + l], TBL_W,
                                  dn[1 + l], K, wbf, dacc, ACC_ROWS, 0, s);
            m.fused(bbytes, hb_at(hidden), 2, U, &gt, &ht, nodes[l]);
            CK(cudaStreamSynchronize(s));
            ++n_fused; ++n_ops;
            if (getenv("FUSED_TEST_CHECK_EACH")) {
                std::vector<uint8_t> dv(CAP); CK(cudaMemcpy(dv.data(), rs.payload_buf, CAP, cudaMemcpyDeviceToHost));
                uint64_t badb = 0, badsc = 0;
                std::vector<uint8_t> isc(CAP, 0); for (auto& kv : m.score) for (int j = 0; j < 4; ++j) isc[kv.first + j] = 1;
                for (uint64_t i = 0; i < CAP; ++i) if (!isc[i] && dv[i] != m.payload[i]) ++badb;
                for (auto& kv : m.score) { float d; std::memcpy(&d, &dv[kv.first], 4); const double ref = kv.second.first;
                    if (std::isnan(ref) ? !std::isnan(d) : !(std::fabs(d - ref) <= 1e-5 * kv.second.second + 1e-6)) ++badsc; }
                if (badb || badsc) { printf("first bad: forward %d layer %d hidden %u K %u U %ld gt0 %ld ht0 %ld badbytes %lu badscores %lu th %lu/%lu\n",
                    f, l, hidden, K, (long)U, (long)gt[0], (long)ht[0], (unsigned long)badb, (unsigned long)badsc,
                    (unsigned long)*rs.task_head, (unsigned long)m.th); exit(3); }
            }
        }
    }
    // ---- 3. batch invariance (bit-identical scores for one row) ----
    int inv_fail = 0;
    if (on('3')) {
        const uint32_t hidden = 4096, K = 16; NodeW nd; nd.init(K, hidden, true);
        for (uint32_t k = 0; k < K; ++k) nd.randomize_slot(k, HEAD_SINGLE, 0);
        up_node(0, nd);
        const int64_t row = 77;
        std::vector<std::vector<float>> got;
        for (int cfgi = 0; cfgi < 3; ++cfgi) {
            const int64_t U = cfgi == 2 ? 600 : 128;
            std::vector<int64_t> t(1 + 2 * TBL_N, 0);
            uint64_t idx = 0;   // packed index of `row`
            if (cfgi == 0) { t[0] = 1; t[1] = row; t[2] = 1; }
            else { t[0] = 3; t[1] = 3; t[2] = 30; t[3] = row - 2; t[4] = 5; t[5] = 100; t[6] = 2; idx = 32; }
            up_table(1, t);
            const uint64_t ph0 = *rs.payload_head;
            launch_producer_fused(rs, reinterpret_cast<const uint8_t*>(db), uint64_t(U) * hidden * 2, hidden, 2, nullptr, 0, dt[1], TBL_W,
                                  dn[0], K, 1, dacc, ACC_ROWS, 0, s);
            CK(cudaStreamSynchronize(s));
            m.fused(bbytes, hb_at(hidden), 2, U, nullptr, &t, nd);
            std::vector<float> v(K);
            const uint64_t total = Model::rows_of(t, U).size() * K * 4;
            const TwoSpan sp = payload_compute_spans(ph0, CAP, align_up(total, PAYLOAD_ALIGN));
            for (uint32_t k = 0; k < K; ++k) {
                const uint64_t o = (idx * K + k) * 4, p = (o < sp.len1) ? sp.off1 + o : sp.off2 + (o - sp.len1);
                CK(cudaMemcpy(&v[k], rs.payload_buf + p, 4, cudaMemcpyDeviceToHost));
            }
            got.push_back(v);
            n_fused++; n_ops++;
        }
        for (int c = 1; c < 3; ++c) if (std::memcmp(got[0].data(), got[c].data(), 4 * K) != 0) ++inv_fail;
    }
    // ---- 4. graph: 6-node forward, tables/modes/weights rewritten per replay ----
    int replays = 0;
    if (on('4')) {
        const uint32_t hidden = 4096, K = 8; const int L = 6; const int64_t U = 48;
        const uint64_t upper = uint64_t(U) * hidden * 2;
        std::vector<NodeW> nodes(L);
        for (int l = 0; l < L; ++l) { nodes[l].init(K, hidden, true); for (uint32_t k = 2; k < K; ++k) nodes[l].randomize_slot(k, rnd() % 2 ? HEAD_SINGLE : HEAD_FREE, 0); }
        auto set_chain = [&](uint32_t k, int a) {
            int first = int(rint_(0, L - 2)), last = int(rint_(first + 1, L - 1));
            for (int l = 0; l < L; ++l)
                nodes[l].randomize_slot(k, l == first ? HEAD_FIRST : l == last ? HEAD_LAST : (l > first && l < last) ? HEAD_MID : HEAD_FREE, a);
        };
        set_chain(0, 0); set_chain(1, 1);
        for (int l = 0; l < L; ++l) up_node(1 + l, nodes[l]);
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int l = 0; l < L; ++l)
            launch_producer_fused(rs, reinterpret_cast<const uint8_t*>(db), upper, hidden, 2, dt[8 + l % 8], TBL_W, dt[2 + l], TBL_W,
                                  dn[1 + l], K, 1, dacc, ACC_ROWS, 0, s);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int rep = 0; rep < 200; ++rep) {
            if (rep % 5 == 0) {   // register / retire / move chains: data only
                const uint32_t k = uint32_t(rnd() % K);
                if (k < 2) set_chain(k, int(k));
                else for (int l = 0; l < L; ++l) nodes[l].randomize_slot(k, rnd() % 2 ? HEAD_SINGLE : HEAD_FREE, 0);
                for (int l = 0; l < L; ++l) up_node(1 + l, nodes[l]);
            }
            std::vector<char> need(U, 0); for (int64_t r = 0; r < U; ++r) need[r] = int(rnd() % 100) < 40;
            std::vector<std::vector<int64_t>> ht(L), gt(L);
            for (int l = 0; l < L; ++l) {
                ht[l] = (rnd() % 10 == 0) ? std::vector<int64_t>(1 + 2 * TBL_N, -1) : table_with(U, need, 15);
                gt[l] = random_table(U, false);
                up_table(2 + l, ht[l]); up_table(8 + l, gt[l]);
            }
            CK(cudaGraphLaunch(ge, s));
            CK(cudaStreamSynchronize(s));
            for (int l = 0; l < L; ++l) m.fused(bbytes, hb_at(hidden), 2, U, &gt[l], &ht[l], nodes[l]);
            n_fused += L; n_ops += L; ++replays;
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
    if (inv_fail) { printf("FAIL batch invariance: %d configurations differ bitwise\n", inv_fail); ++fails; }
    printf("ops=%d fused_launches=%d chains=%d graph_replays=%d tasks=%lu ring_wraps=%lu scores_checked=%lu (NaN %lu) "
           "max_abs_err=%.3g max_rel_err=%.3g batch_invariant=%s -> %s\n",
           n_ops, n_fused, chains, replays, (unsigned long)th, (unsigned long)(ph / CAP), (unsigned long)scores,
           (unsigned long)nan_ok, max_abs, max_rel, inv_fail ? "no" : "yes", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
