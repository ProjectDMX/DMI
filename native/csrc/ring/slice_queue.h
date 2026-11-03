// ring/slice_queue.h -- per-slice in-process consumer queue.
//
// A Python-callable consumer makes the P2P thread take the GIL for every
// slice, which serializes extraction against a GIL-bound serving thread under
// load (ring backpressure, then producer stalls).  With this queue the P2P
// thread pushes finished slices in pure C++ (no GIL), and a Python consumer
// pulls them in batches with drain(), releasing the GIL while it blocks.
//
// Bounds: max_items, and optionally max_bytes (0 = no byte cap).  Retained
// bytes are the payload allocations the queued slices keep alive, each
// distinct allocation charged once at max(logical bytes, allocated bytes)
// (pooled_block.h): per-request views of one step's pooled buffer share one
// charge, and a 32-byte pooled slice is charged its 64 KiB block.
//
// Header-only; included where ATen is already linked.

#pragma once
#include <ATen/core/Tensor.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ring/drop_notice.h"
#include "ring/pooled_block.h"

namespace ring_py {

struct SliceItem {
    std::string model_id;
    int32_t     shard_rank;
    std::string req_id;
    std::string act_name;
    int32_t     layer_no;
    int32_t     start_token;
    int32_t     end_token;
    at::Tensor  slice;
    CaptureLease capture_lease;
    int64_t capture_ts_us = 0;
    ring::BackingRef backing;   // the allocation `slice` keeps alive
};

class SliceQueue {
public:
    explicit SliceQueue(size_t max_items = 200000, size_t drop_notice_cap = 4096,
                        uint64_t max_bytes = 0)
        : max_items_(max_items), max_bytes_(max_bytes), drops_(drop_notice_cap) {}

    // Pushed by the P2P thread in pure C++ (never the GIL).  Bounded: when
    // full, the slice is dropped, counted AND attributed -- a DropNotice
    // naming this req_id is filed (drop_notice.h).
    void push(const std::string& model_id, int32_t shard_rank,
              const std::string& req_id, const std::string& act_name,
              int32_t layer_no, int32_t start_token, int32_t end_token,
              at::Tensor slice, CaptureLease capture_lease = {},
              int64_t capture_ts_us = 0) {
        bool shed = false;
        const ring::BackingRef backing = ring::backing_of(slice);
        {
            std::lock_guard<std::mutex> lk(mu_);
            // The byte cap, like BatchQueue's, always admits into an empty
            // queue so a tiny cap cannot wedge the pipeline.
            const bool over_bytes = max_bytes_ > 0 && !q_.empty() &&
                ledger_.total() + ledger_.cost(backing) > max_bytes_;
            if (q_.size() >= max_items_ || over_bytes) {
                dropped_++;
                shed = true;
            } else {
                ledger_.add(backing);
                q_.push_back(SliceItem{model_id, shard_rank, req_id, act_name,
                                       layer_no, start_token, end_token,
                                       std::move(slice), std::move(capture_lease), capture_ts_us,
                                       backing});
            }
        }
        if (shed) {
            // Outside mu_: the notice buffer has its own lock (no nesting).
            DropNotice n;
            n.source      = kDropSourceSlice;
            n.hook        = act_name;
            n.layer_no    = layer_no;
            n.start_token = start_token;
            n.end_token   = end_token;
            n.nbytes      = static_cast<uint64_t>(slice.defined() ? slice.nbytes() : 0);
            n.ts_us       = drop_notice_now_us();
            n.req_ids.push_back(req_id);
            n.rank = capture_lease.rank_for(act_name, layer_no, n.rank);
            n.capture_lease = std::move(capture_lease);
            drops_.record(std::move(n));
            return;
        }
        cv_.notify_one();
    }

    // Wake any waiting consumer so it can observe shutdown / final items.
    void stop() {
        { std::lock_guard<std::mutex> lk(mu_); stopped_ = true; }
        cv_.notify_all();
    }

    // Coalesced notification for host-side control work, not a queue shutdown.
    void wake() {
        { std::lock_guard<std::mutex> lk(mu_); wake_pending_ = true; }
        cv_.notify_one();
    }

    // Pop up to max_items, waiting up to timeout_ms for the first one. MUST be
    // called with the GIL RELEASED (the binding wraps it in gil_scoped_release)
    // so the blocking wait never holds the GIL.
    std::vector<SliceItem> drain(size_t max_items, int timeout_ms) {
        std::vector<SliceItem> out;
        std::unique_lock<std::mutex> lk(mu_);
        if (q_.empty() && !stopped_) {
            cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                         [this] { return !q_.empty() || stopped_ || wake_pending_; });
        }
        wake_pending_ = false;
        while (!q_.empty() && out.size() < max_items) {
            ledger_.remove(q_.front().backing);
            out.push_back(std::move(q_.front()));
            q_.pop_front();
        }
        return out;
    }

    // Drain the drop notices filed since the last poll (see drop_notice.h).
    // Non-blocking.  Absence of notices == nothing was lost.
    DropNoticeBuffer::Drained drain_drop_notices(size_t max_items = 4096) {
        return drops_.drain(max_items);
    }
    const DropNoticeBuffer& drop_notices() const { return drops_; }

    uint64_t dropped() const { std::lock_guard<std::mutex> lk(mu_); return dropped_; }
    size_t   size()    const { std::lock_guard<std::mutex> lk(mu_); return q_.size(); }
    // Allocated bytes the queued slices keep alive (see the header comment).
    uint64_t retained_bytes() const { std::lock_guard<std::mutex> lk(mu_); return ledger_.total(); }
    uint64_t max_bytes()      const { std::lock_guard<std::mutex> lk(mu_); return max_bytes_; }

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<SliceItem>   q_;
    size_t                  max_items_;
    uint64_t                max_bytes_;          // 0 = no byte cap
    ring::BackingLedger     ledger_;
    uint64_t                dropped_ = 0;
    bool                    stopped_ = false;
    bool                    wake_pending_ = false;
    // Own mutex; never taken while mu_ is held (see push()).
    DropNoticeBuffer        drops_;
};

}  // namespace ring_py
