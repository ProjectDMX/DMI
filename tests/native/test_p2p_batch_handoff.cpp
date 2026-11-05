// P2P post-processing with a BatchQueue-style consumer, and pooled-buffer
// retention accounting.
//
// 1. With only a batch sink installed (BatchQueue), HF batched captures --
//    token [batch, q_len, hidden] and attention [batch, heads, q_len, kv] --
//    used to fall through the whole-batch branch (packed, non-attention only)
//    and return at the empty per-slice consumer: nothing delivered, nothing
//    counted.  They now arrive as one-request items, one per request slice.
// 2. A pooled 32-byte payload pins its whole 64 KiB size class.  BatchQueue
//    and SliceQueue charge the allocation a queued item keeps alive, once per
//    distinct allocation, not the logical length.
// 3. p2p_pool_cap_bytes == 0 allocates exact-size buffers (no pool, no
//    size-class rounding).
// 4. The P2P delivery wait honours an absolute deadline.
//
// Compiles the real p2p_thread.cpp against ATen (CPU only).  The DrainThread
// is never started: its constructor is replaced below so no CUDA API runs,
// and cudaFreeHost aborts if anything reaches it.  Built and run by
// tests/test_native_ring_consumer_fixes.py.

#include <ATen/ATen.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "ring/batch_queue.h"
#include "ring/buffer_pool.h"
#include "ring/drain_thread.h"
#include "ring/pooled_block.h"
#include "ring/slice_queue.h"
#include "ring/tensor_meta.h"
// The test drives the private post-processing entry points directly.
#define private public
#include "ring/p2p_thread.h"
#undef private

extern "C" cudaError_t cudaFreeHost(void*) { std::abort(); }

namespace ring {
DrainThread::DrainThread(RingState& rs, PinnedStaging& st, const RingConfig& cfg)
    : ring_(rs), staging_(st), cfg_(cfg) {}
DrainThread::~DrainThread() noexcept = default;
// Staging is never used here (every task has alloc_bytes == 0).
void DrainThread::notify_staging_freed_bytes(uint64_t) { std::abort(); }
}  // namespace ring

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

struct Delivered {
    std::string act_name;
    int32_t layer_no;
    uint64_t step_seq;
    bool last_in_step;
    at::Tensor tensor;
    std::vector<std::string> req_ids;
    at::Tensor segments;
    ring_py::CaptureLease lease;
    int64_t ts;
    bool is_attn;
};

struct Fixture {
    ring::RingState state{};
    ring::PinnedStaging staging;
    ring::RingConfig cfg;
    ring_py::TensorMetaFifo fifo;
    std::unique_ptr<ring::DrainThread> drain;
    std::unique_ptr<ring::P2PThread> p2p;
    std::vector<Delivered> got;
    bool throw_in_sink = false;

    explicit Fixture(bool batch_submit = true, uint64_t pool_cap = 256ULL << 20) {
        cfg.batch_submit = batch_submit;
        cfg.p2p_pool_cap_bytes = pool_cap;
        drain = std::make_unique<ring::DrainThread>(state, staging, cfg);
        ring::BatchSink sink = [this](const std::string&, int32_t,
                                      const std::string& act, int32_t layer,
                                      uint64_t seq, bool last, at::Tensor t,
                                      std::vector<std::string> ids, at::Tensor seg,
                                      ring_py::CaptureLease lease, int64_t ts,
                                      bool is_attn) {
            if (throw_in_sink) throw std::runtime_error("consumer refused");
            got.push_back(Delivered{act, layer, seq, last, std::move(t), std::move(ids),
                                    std::move(seg), std::move(lease), ts, is_attn});
        };
        p2p = std::make_unique<ring::P2PThread>(*drain, fifo, cfg, ring::SubmitFn{},
                                                std::move(sink));
    }

    // Publish one single-hook step and post-process `tensor` for it.
    void run(bool flattened, int hook, std::vector<int64_t> shape,
             std::vector<ring_py::RequestMeta> reqs, const at::Tensor& tensor,
             const ring_py::CaptureLease& lease) {
        auto* ctx = new ring_py::StepContext;
        ctx->model_id = "m";
        ctx->flattened = flattened;
        ctx->requests = std::move(reqs);
        ctx->capture_lease = lease;
        ring_py::TensorMeta meta;
        meta.hook_type = hook;
        meta.layer_no = 3;
        meta.shape = shape;
        meta.dtype = int(at::kFloat);
        meta.last_in_step = true;
        std::vector<ring_py::TensorMeta> metas{meta};
        fifo.push_step(ctx, metas);
        // The P2P thread hands post-processing a flat byte buffer.
        at::Tensor bytes = tensor.contiguous().view(at::kByte).reshape({-1}).clone();
        ring::DrainTask task;
        task.tensor_total_bytes = bytes.nbytes();
        task.capture_ts_us = 1234;
        p2p->do_post_processing(bytes, task);
    }
};

ring_py::RequestMeta req(const char* id, int32_t start, int32_t end, int64_t dim0,
                         int32_t kv_offset = 0, int32_t orig = -1) {
    ring_py::RequestMeta r;
    r.req_id = id;
    r.start_token = start;
    r.end_token = end;
    r.dim0_offset = dim0;
    r.kv_offset = kv_offset;
    r.orig_position = orig;
    return r;
}

std::vector<int32_t> seg_row(const at::Tensor& seg) {
    auto c = seg.contiguous();
    const int32_t* p = c.data_ptr<int32_t>();
    return {p[0], p[1], p[2], p[3]};
}

void test_hf_token_fallback() {
    Fixture f;
    ring_py::CaptureLease lease(9);
    // [batch=2, q_len=3, hidden=4], left-padded: request a has 2 real rows,
    // request b 3.
    auto src = at::arange(24, at::kFloat).reshape({2, 3, 4});
    f.run(false, ring_py::HOOK_TYPE_RESID_PRE, {2, 3, 4},
          {req("a", 0, 2, 0, 0, 0), req("b", 5, 8, 1, 0, 1)}, src, lease);
    check(f.p2p->submit_exceptions() == 0, "hf token: no consumer error");
    check(f.got.size() == 2, "hf token: one item per request slice");
    if (f.got.size() != 2) return;
    const auto& a = f.got[0];
    const auto& b = f.got[1];
    check(a.req_ids == std::vector<std::string>{"a"}, "hf token: item 0 is request a");
    check(b.req_ids == std::vector<std::string>{"b"}, "hf token: item 1 is request b");
    check(a.tensor.sizes().vec() == std::vector<int64_t>{2, 4}, "hf token: a is [2, 4]");
    check(at::equal(a.tensor, src[0].narrow(0, 1, 2)), "hf token: a holds its last 2 rows");
    check(at::equal(b.tensor, src[1]), "hf token: b holds all 3 rows");
    check(seg_row(a.segments) == std::vector<int32_t>{0, 0, 2, 0}, "hf token: a segments row");
    check(seg_row(b.segments) == std::vector<int32_t>{0, 5, 8, 1}, "hf token: b segments row");
    check(a.segments.sizes().vec() == std::vector<int64_t>{1, 4}, "hf token: [1, 4] segments");
    check(!a.is_attn && !b.is_attn, "hf token: not attention");
    check(a.act_name == "blocks.hook_resid_pre" && a.layer_no == 3, "hf token: act/layer");
    check(a.step_seq == b.step_seq && a.step_seq > 0, "hf token: one step_seq per step");
    check(!a.last_in_step && b.last_in_step, "hf token: last_in_step on the last item only");
    check(a.lease.same_as(lease) && b.lease.same_as(lease), "hf token: lease carried");
    check(a.ts == 1234, "hf token: capture_ts_us carried");
}

void test_hf_attention_fallback() {
    Fixture f;
    ring_py::CaptureLease lease(10);
    // [batch=2, heads=2, q=3, kv=4].  Request a: 2 query rows, keys [1, 3)
    // (left-padded dynamic cache, kv_offset 1).  Request b: 1 decode row.
    auto src = at::arange(48, at::kFloat).reshape({2, 2, 3, 4});
    f.run(false, ring_py::HOOK_TYPE_PATTERN, {2, 2, 3, 4},
          {req("a", 0, 2, 0, 1), req("b", 3, 4, 1, 0)}, src, lease);
    check(f.got.size() == 2, "hf attention: one item per request slice");
    if (f.got.size() != 2) return;
    const auto& a = f.got[0];
    const auto& b = f.got[1];
    check(a.is_attn && b.is_attn, "hf attention: is_attn set");
    check(a.act_name == "blocks.attn.hook_pattern", "hf attention: act_name");
    check(a.tensor.sizes().vec() == std::vector<int64_t>{2, 2, 2},
          "hf attention: a is [heads, rows, kv]");
    check(at::equal(a.tensor, src[0].narrow(1, 1, 2).narrow(2, 1, 2)),
          "hf attention: a holds its query rows and real keys");
    check(b.tensor.sizes().vec() == std::vector<int64_t>{2, 1, 4},
          "hf attention: b is [heads, 1, kv]");
    check(at::equal(b.tensor, src[1].narrow(1, 2, 1)), "hf attention: b holds the last row");
    check(seg_row(a.segments) == std::vector<int32_t>{0, 0, 2, -1}, "hf attention: a segments");
    check(seg_row(b.segments) == std::vector<int32_t>{0, 3, 4, -1}, "hf attention: b segments");
    check(b.last_in_step && !a.last_in_step, "hf attention: last_in_step");
}

void test_fallback_consumer_errors_are_counted() {
    Fixture f;
    f.throw_in_sink = true;
    auto src = at::zeros({2, 3, 4}, at::kFloat);
    f.run(false, ring_py::HOOK_TYPE_RESID_PRE, {2, 3, 4},
          {req("a", 0, 3, 0), req("b", 0, 3, 1)}, src, {});
    check(f.p2p->submit_exceptions() == 2, "fallback: every refused slice is counted");
    check(f.p2p->last_error() == "consumer refused", "fallback: last error kept");
}

void test_packed_paths_unchanged() {
    Fixture f;
    auto src = at::arange(20, at::kFloat).reshape({5, 4});
    f.run(true, ring_py::HOOK_TYPE_RESID_PRE, {5, 4},
          {req("a", 0, 2, 0), req("b", 0, 3, 2)}, src, {});
    check(f.got.size() == 1, "packed token: one whole-batch item");
    if (!f.got.empty()) {
        check(f.got[0].req_ids == std::vector<std::string>{"a", "b"}, "packed: req ids");
        check(f.got[0].tensor.sizes().vec() == std::vector<int64_t>{5, 4}, "packed: whole tensor");
        check(!f.got[0].is_attn && f.got[0].last_in_step, "packed: flags");
    }
    // Packed attention is not sliceable per request on any path.
    Fixture g;
    auto attn = at::zeros({1, 2, 2}, at::kFloat);
    g.run(true, ring_py::HOOK_TYPE_PATTERN, {1, 2, 2}, {req("a", 0, 2, 0)}, attn, {});
    check(g.got.empty() && g.p2p->submit_exceptions() == 0, "packed attention: skipped");
}

void test_pooled_retention_is_charged_by_allocation() {
    const uint64_t block = ring::BufferPool::capacity_for(32);
    check(block == 64 * 1024, "the minimum size class is 64 KiB");
    ring::BufferPool pool(256ULL << 20);

    // The reviewer's case: 32 independent 32-byte pooled items, logical sum
    // exactly max_bytes.  Each pins a whole block: only the first fits (an
    // empty queue always admits one item), the rest are shed and counted.
    {
        ring_py::BatchQueue queue(1024);
        for (int i = 0; i < 32; ++i) {
            ring_py::HookBatchItem item;
            item.tensor = pool.acquire(32);
            item.nbytes = item.tensor.nbytes();
            item.act_name = "blocks.hook_resid_pre";
            item.req_ids = {"r"};
            check(ring::pooled_block_capacity(item.tensor) == block,
                  "a pooled tensor reports its block capacity");
            queue.push(std::move(item));
        }
        check(queue.retained_bytes() == block, "batch queue: charged one block");
        check(queue.size() == 1 && queue.dropped() == 31, "batch queue: 31 shed");
        check(queue.dropped_bytes() == 31 * 32, "batch queue: dropped bytes are logical");
        queue.drain(64, 0);
        check(queue.retained_bytes() == 0, "batch queue: drained to zero");
    }

    // Views of ONE block (one-request items of one step) share one charge,
    // released when the last of them leaves the queue.
    {
        ring_py::BatchQueue queue(1ULL << 20);
        at::Tensor whole = pool.acquire(4096);
        for (int i = 0; i < 4; ++i) {
            ring_py::HookBatchItem item;
            item.tensor = whole.narrow(0, i * 1024, 1024);
            item.nbytes = item.tensor.nbytes();
            item.act_name = "blocks.hook_resid_pre";
            queue.push(std::move(item));
        }
        check(queue.retained_bytes() == block, "shared block: charged once");
        auto first = queue.drain(3, 0);
        check(queue.retained_bytes() == block, "shared block: held while one item remains");
        queue.drain(1, 0);
        check(queue.retained_bytes() == 0, "shared block: released with the last item");
    }

    // An unpooled tensor is charged its storage.
    {
        ring_py::BatchQueue queue(1ULL << 20);
        ring_py::HookBatchItem item;
        item.tensor = at::zeros({100}, at::kByte);
        item.nbytes = 100;
        queue.push(std::move(item));
        check(queue.retained_bytes() == 100, "unpooled: charged its storage bytes");
    }

    // SliceQueue: retained bytes by allocation, optional byte cap.
    {
        ring_py::SliceQueue queue(1000, 64, /*max_bytes=*/block + 1);
        at::Tensor a = pool.acquire(64);
        at::Tensor b = pool.acquire(64);
        queue.push("m", 0, "a", "x", 0, 0, 1, a.narrow(0, 0, 32));
        queue.push("m", 0, "b", "x", 0, 0, 1, a.narrow(0, 32, 32));  // same block
        check(queue.retained_bytes() == block, "slice queue: shared block charged once");
        queue.push("m", 0, "c", "x", 0, 0, 1, b);                    // a second block
        check(queue.size() == 2 && queue.dropped() == 1, "slice queue: byte cap sheds");
        queue.drain(10, 0);
        check(queue.retained_bytes() == 0, "slice queue: drained to zero");
        ring_py::SliceQueue unbounded;   // max_bytes 0: no byte cap
        unbounded.push("m", 0, "a", "x", 0, 0, 1, pool.acquire(32));
        unbounded.push("m", 0, "b", "x", 0, 0, 1, pool.acquire(32));
        check(unbounded.size() == 2 && unbounded.retained_bytes() == 2 * block,
              "slice queue: no byte cap by default, retention still reported");
    }
}

void test_pool_cap_zero_allocates_exact_size() {
    for (uint64_t cap : {uint64_t{0}, uint64_t{256} << 20}) {
        Fixture f(true, cap);
        check((f.p2p->pool_ != nullptr) == (cap != 0), "pool exists iff cap > 0");
        std::vector<uint8_t> staged(48, 7);
        auto* ctx = new ring_py::StepContext;
        ctx->model_id = "m";
        ctx->flattened = true;
        ctx->requests = {req("a", 0, 3, 0)};
        ring_py::TensorMeta meta;
        meta.hook_type = ring_py::HOOK_TYPE_RESID_PRE;
        meta.layer_no = 0;
        meta.shape = {3, 4};
        meta.dtype = int(at::kFloat);
        meta.last_in_step = true;
        std::vector<ring_py::TensorMeta> metas{meta};
        f.fifo.push_step(ctx, metas);
        std::vector<ring::DrainTask> tasks(1);
        tasks[0].tensor_total_bytes = staged.size();
        tasks[0].data_ptr1 = staged.data();
        tasks[0].data_len1 = staged.size();
        tasks[0].alloc_bytes = 0;
        f.p2p->process(tasks);
        check(f.got.size() == 1, "staged task delivered");
        if (f.got.empty()) continue;
        const at::Tensor& t = f.got[0].tensor;
        if (cap == 0) {
            check(ring::pooled_block_capacity(t) == 0, "cap 0: not a pool block");
            check(t.storage().nbytes() == staged.size(), "cap 0: exact-size buffer");
            check(ring::backing_of(t).bytes == staged.size(), "cap 0: charged exact size");
        } else {
            check(ring::pooled_block_capacity(t) == 64 * 1024, "pooled: 64 KiB block");
            check(ring::backing_of(t).bytes == 64 * 1024, "pooled: charged the block");
        }
    }
}

void test_delivery_wait_honours_deadline() {
    // Nothing is ever processed (the thread is not started): a finite wait
    // must return false at its deadline, not block.
    Fixture f;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = f.p2p->wait_until_processed_until(
        1, t0 + std::chrono::milliseconds(50));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    check(!ok, "deadline wait: reports not delivered");
    check(ms >= 45 && ms < 5000, "deadline wait: returns at the deadline");
    // A deadline already in the past checks once.
    check(!f.p2p->wait_until_processed_until(1, t0), "past deadline: immediate false");
    check(f.p2p->wait_until_processed_until(0, t0), "reached target: true");
}

}  // namespace

int main() {
    test_hf_token_fallback();
    test_hf_attention_fallback();
    test_fallback_consumer_errors_are_counted();
    test_packed_paths_unchanged();
    test_pooled_retention_is_charged_by_allocation();
    test_pool_cap_zero_allocates_exact_size();
    test_delivery_wait_honours_deadline();
    if (g_failures) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
