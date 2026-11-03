// ring/batch_queue.h -- whole-HookPoint in-process consumer queue (the batch
// handoff).  Companion to slice_queue.h.
//
// The per-slice path hands one item per (request x hook x step) across the
// C++ -> Python boundary: with many concurrent requests and many hooks that is
// thousands of handoffs per step, each paying a fixed cost (a locked push with
// string copies, a pybind tuple build, the consumer's routing) that is
// independent of what the consumer computes.
//
// The batch handoff hands one item per (hook x step) instead: the WHOLE
// contiguous tensor the producer wrote (a view of the P2P thread's pageable
// buffer -- never sliced, never cloned here), an int32 segments table and the
// request-id list.  The consumer reconstructs the per-request views on its own
// thread, or consumes the batch as a whole.
//
// Lifetime: the tensor aliases a pooled pageable buffer (buffer_pool.h).
// Consumers must not retain a view of it past their handling of the item;
// anything kept must be cloned.  clone_slices does not apply on this path.
//
// Backpressure: capped by RETAINED BYTES (item counts are meaningless when one
// item can be tens of MB).  On overflow whole batches are shed, counted and
// attributed: every shed item files a DropNotice (drop_notice.h) naming the
// requests that were on it.
//
// Retained bytes are the payload ALLOCATIONS the queued items keep alive, not
// their logical lengths: a pooled 32-byte tensor pins its whole 64 KiB size
// class (pooled_block.h).  Each distinct allocation is charged once, at
// max(logical bytes, allocated bytes), while any queued item refers to it --
// one-request items cut from one step's buffer share one charge.  Item
// metadata (segments table, request ids, act_name) is not charged: it is
// 16 bytes plus one id per request, small next to the payload.  nbytes and
// dropped_bytes stay logical payload bytes.
//
// Rank-ordered eviction: each item carries the drop rank of its
// (act_name, layer) stream, resolved at enqueue from the installed map
// (set_layer_class_map, or the step's CaptureLease).  On overflow the queue
// first makes room by EVICTING queued items strictly less urgent than the
// arriving one -- worst rank first, oldest first within a rank -- and refuses
// the newcomer only when nothing less urgent is left.  Consequences:
//   * a rank-0 batch is evicted only when every remaining item is rank 0;
//   * with no map installed every item ranks best-effort, no candidate is
//     ever strictly less urgent, and the queue behaves exactly like a FIFO
//     that drops the newcomer;
//   * set_class_eviction(false) keeps the map (per-rank loss is still
//     attributed) but restores drop-the-newcomer.
// Every eviction files a DropNotice with source "batch_queue_evict".
//
// Header-only; included where ATen is already linked.

#pragma once
#include <ATen/core/Tensor.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ring/drop_notice.h"
#include "ring/pooled_block.h"

namespace ring_py {

// One step x one hook.
struct HookBatchItem {
    CaptureLease             capture_lease;
    int64_t                  capture_ts_us = 0;
    std::string              model_id;
    int32_t                  shard_rank   = 0;
    std::string              act_name;
    int32_t                  layer_no     = -1;
    int32_t                  node_id      = 0;   // 0 in-process; wire placeholder
    uint64_t                 step_seq     = 0;   // monotone step counter
    bool                     last_in_step = false;
    // The whole reshaped [total_tokens, ...] tensor -- a VIEW of the p2p pool
    // buffer.  Never sliced, never cloned here.  Keeps the pool block alive
    // until this item (and every view the consumer derives) is dropped.
    at::Tensor               tensor;
    // Request ids, in segments row order (the step's request order, minus
    // requests with no rows).
    std::vector<std::string> req_ids;
    // int32 CPU [N,4]: (dim0_offset, db_start, db_end, orig_position).  Row j
    // gives request j's view: tensor[dim0_offset : dim0_offset + (db_end -
    // db_start)] for token hooks; final_logits uses one row at dim0_offset
    // (the request's original position).
    at::Tensor               segments;
    // Logical payload bytes (tensor.nbytes()); reported in drop notices.
    uint64_t                 nbytes       = 0;
    // A one-request item of the per-slice fallback holding an attention
    // matrix slice [heads, rows, kv]: take the tensor whole.
    bool                     is_attn      = false;
    // The allocation the tensor keeps alive (set by push; charged once per
    // distinct allocation for the byte cap).
    ring::BackingRef         backing;
    // Drop rank of this (act_name, layer) stream, resolved at enqueue.
    // Best-effort by default: with no map every item ranks the same, which is
    // FIFO drop-the-newcomer.
    uint8_t                  rank         = kRankBestEffort;
};

class BatchQueue {
public:
    // max_bytes: retained-byte cap.  When a push would exceed it, the whole
    // item is dropped, counted AND attributed.  Default 512 MiB.
    // drop_notice_cap: notices retained between consumer polls.
    explicit BatchQueue(uint64_t max_bytes = 512ULL * 1024 * 1024,
                        size_t drop_notice_cap = 4096)
        : max_bytes_(max_bytes), drops_(drop_notice_cap) {}

    // Install the (act_name, layer) -> drop-rank map, normally once before the
    // producer runs (safe later: guarded by mu_, affects later items only).
    // `entries` is (act_name, layer_no, rank); `default_rank` covers every
    // stream not named.  Installing a map switches the queue from FIFO to
    // rank-ordered eviction.
    void set_layer_class_map(
        const std::vector<std::tuple<std::string, int32_t, int>>& entries,
        int default_rank = static_cast<int>(kRankBestEffort)) {
        std::lock_guard<std::mutex> lk(mu_);
        act_ids_.clear();
        rank_of_.clear();
        default_rank_ = clamp_rank(default_rank);
        for (const auto& e : entries) {
            const std::string& act = std::get<0>(e);
            auto it = act_ids_.find(act);
            if (it == act_ids_.end())
                it = act_ids_.emplace(act, static_cast<uint32_t>(act_ids_.size())).first;
            rank_of_[key_of(it->second, std::get<1>(e))] = clamp_rank(std::get<2>(e));
        }
        class_map_set_ = true;
    }

    // Keep the map installed (per-rank loss is still attributed and counted)
    // but stop using rank to choose the victim: off => drop-the-newcomer.
    void set_class_eviction(bool on) {
        std::lock_guard<std::mutex> lk(mu_);
        class_eviction_ = on;
    }

    // Pushed by the P2P thread in pure C++ (never the GIL).  Bounded by
    // retained bytes.  On overflow: first try to make room by evicting queued
    // items that are STRICTLY LESS URGENT than this one (worst rank first, FIFO
    // within a rank); if that cannot free enough, refuse the newcomer.  Either
    // way every shed item is counted AND attributed (a DropNotice naming its
    // req_ids).
    void push(HookBatchItem item) {
        bool shed = false;
        std::vector<HookBatchItem> evicted;   // notices filed outside mu_
        item.backing = ring::backing_of(item.tensor);
        {
            std::lock_guard<std::mutex> lk(mu_);
            item.rank = item.capture_lease.rank_for(item.act_name, item.layer_no,
                                                  rank_locked(item.act_name, item.layer_no));
            if (retained_bytes_ + ledger_.cost(item.backing) > max_bytes_ && !q_.empty()) {
                if (class_eviction_)
                    make_room_locked(item, evicted);
                if (retained_bytes_ + ledger_.cost(item.backing) > max_bytes_ && !q_.empty()) {
                    // Nothing less urgent left to give up: drop the whole
                    // batch.  An empty queue always admits one item, even one
                    // larger than the cap, so a tiny cap cannot wedge the
                    // pipeline (retained_bytes reports the overshoot).
                    dropped_++;
                    dropped_bytes_ += item.nbytes;
                    dropped_by_rank_[item.rank]++;
                    dropped_bytes_by_rank_[item.rank] += item.nbytes;
                    shed = true;
                }
            }
            if (!shed) {
                charge_locked(item);
                pushed_++;
                depth_by_rank_[item.rank]++;
                q_.push_back(std::move(item));
            }
        }
        // Attribution runs outside mu_ (the notice buffer has its own lock: no
        // nesting, no lock-order hazard).
        for (auto& e : evicted) {
            DropNotice n;
            n.source   = kDropSourceBatchEvict;
            n.hook     = std::move(e.act_name);
            n.layer_no = e.layer_no;
            n.step_seq = e.step_seq;
            n.nbytes   = e.nbytes;
            n.ts_us    = drop_notice_now_us();
            n.rank     = e.rank;
            n.capture_lease = std::move(e.capture_lease);
            n.req_ids  = std::move(e.req_ids);
            drops_.record(std::move(n));
        }
        if (shed) {
            // `item` was not moved from on this branch, so its req_ids can be
            // moved straight into the notice -- no copy on the shed path.
            DropNotice n;
            n.source   = kDropSourceBatch;
            n.hook     = std::move(item.act_name);
            n.layer_no = item.layer_no;
            n.step_seq = item.step_seq;
            n.nbytes   = item.nbytes;
            n.ts_us    = drop_notice_now_us();
            n.rank     = item.rank;
            n.capture_lease = std::move(item.capture_lease);
            n.req_ids  = std::move(item.req_ids);
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

    // Control work lives in the host queue. Wake its consumer without a fake
    // tensor or a permanent stop. Remember the wake if it arrives before wait.
    void wake() {
        { std::lock_guard<std::mutex> lk(mu_); wake_pending_ = true; }
        cv_.notify_one();
    }

    // Pop up to max_items, waiting up to timeout_ms for the first one.  MUST be
    // called with the GIL RELEASED (the binding wraps it in gil_scoped_release)
    // so the blocking wait never holds the GIL.
    std::vector<HookBatchItem> drain(size_t max_items, int timeout_ms) {
        std::vector<HookBatchItem> out;
        std::unique_lock<std::mutex> lk(mu_);
        if (q_.empty() && !stopped_) {
            cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                         [this] { return !q_.empty() || stopped_ || wake_pending_; });
        }
        wake_pending_ = false;
        while (!q_.empty() && out.size() < max_items) {
            const HookBatchItem& taken = out.emplace_back(std::move(q_.front()));
            release_locked(taken);
            if (depth_by_rank_[taken.rank]) depth_by_rank_[taken.rank]--;
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

    uint64_t dropped()        const { std::lock_guard<std::mutex> lk(mu_); return dropped_; }
    uint64_t dropped_bytes()  const { std::lock_guard<std::mutex> lk(mu_); return dropped_bytes_; }
    size_t   size()           const { std::lock_guard<std::mutex> lk(mu_); return q_.size(); }
    uint64_t retained_bytes() const { std::lock_guard<std::mutex> lk(mu_); return retained_bytes_; }
    // Cumulative count of batches ever enqueued (monotone; survives drain):
    // > 0 proves the batch path actually fired.
    uint64_t pushed()         const { std::lock_guard<std::mutex> lk(mu_); return pushed_; }
    uint64_t max_bytes()      const { std::lock_guard<std::mutex> lk(mu_); return max_bytes_; }

    // -- per-rank counters ------------------------------------------------ //
    // Who lost what, by rank.  dropped_by_rank sums to dropped() and
    // dropped_bytes_by_rank to dropped_bytes(), evictions included.
    std::vector<uint64_t> dropped_by_rank() const {
        std::lock_guard<std::mutex> lk(mu_);
        return {dropped_by_rank_, dropped_by_rank_ + kRankCount};
    }
    std::vector<uint64_t> dropped_bytes_by_rank() const {
        std::lock_guard<std::mutex> lk(mu_);
        return {dropped_bytes_by_rank_, dropped_bytes_by_rank_ + kRankCount};
    }
    // Current queue depth per rank.
    std::vector<uint64_t> qsize_by_rank() const {
        std::lock_guard<std::mutex> lk(mu_);
        return {depth_by_rank_, depth_by_rank_ + kRankCount};
    }
    // The subset of dropped() that was EVICTED from the queue to make room for a
    // more urgent arrival (the rest were refused at the door).  0 under FIFO.
    uint64_t evicted()          const { std::lock_guard<std::mutex> lk(mu_); return evicted_; }
    bool     class_eviction()   const { std::lock_guard<std::mutex> lk(mu_); return class_eviction_; }
    bool     class_map_set()    const { std::lock_guard<std::mutex> lk(mu_); return class_map_set_; }
    size_t   class_map_size()   const { std::lock_guard<std::mutex> lk(mu_); return rank_of_.size(); }
    uint8_t  class_default_rank() const { std::lock_guard<std::mutex> lk(mu_); return default_rank_; }
    // The rank an (act_name, layer) item would get under the installed map.
    uint8_t  rank_for(const std::string& act, int32_t layer) const {
        std::lock_guard<std::mutex> lk(mu_); return rank_locked(act, layer);
    }

private:
    static uint64_t key_of(uint32_t act_id, int32_t layer) {
        return (static_cast<uint64_t>(act_id) << 32) |
               static_cast<uint32_t>(layer);   // layer -1 keys fine, cast is total
    }

    // Call with mu_ held.  Charge / release the allocation an item keeps
    // alive; an allocation shared by several queued items is charged once and
    // released when the last of them leaves.  The charge belongs to no rank:
    // which items keep a shared allocation alive changes as they come and go,
    // so eviction asks the ledger what a set of victims would actually free
    // (make_room_locked) instead of keeping bytes per rank.
    void charge_locked(const HookBatchItem& it) {
        retained_bytes_ += ledger_.add(it.backing);
    }
    void release_locked(const HookBatchItem& it) {
        retained_bytes_ -= ledger_.remove(it.backing);
    }

    // Call with mu_ held.  No map installed => every item is best-effort =>
    // all ranks equal => FIFO.
    uint8_t rank_locked(const std::string& act, int32_t layer) const {
        if (!class_map_set_) return kRankBestEffort;
        auto a = act_ids_.find(act);
        if (a == act_ids_.end()) return default_rank_;
        auto r = rank_of_.find(key_of(a->second, layer));
        return r == rank_of_.end() ? default_rank_ : r->second;
    }

    // Call with mu_ held.  Evict queued items STRICTLY less urgent
    // than `in` -- worst (highest) rank first, oldest-first within that rank --
    // the fewest that admit `in`.  Moves the victims into `out` so the caller
    // can file their notices after releasing mu_.
    //
    // "Rank 0 is evicted only when nothing else remains" falls out of the
    // strict comparison: a rank-0 arrival can evict ranks 1 and 2, and a
    // rank-0 item is never a candidate.  Two rank-0 items contend by the FIFO
    // rule: the newcomer is refused, never the resident.
    //
    // All or nothing: the victims are chosen first, without touching the
    // queue, and evicted only if they admit `in`; otherwise nothing is
    // evicted and the caller refuses `in`.  What a victim frees is decided per
    // allocation, from the references that would REMAIN: evicting a view of a
    // shared allocation frees it only when no surviving item -- of any rank
    // -- and not `in` itself still refers to it.  So
    //   retained after = retained_bytes_ - freed + cost(in),
    // where cost(in) is 0 when `in` aliases an allocation already queued (its
    // charge is never counted as freed: `in` keeps it).  Evicting everything
    // queued empties the queue, which always admits.
    void make_room_locked(const HookBatchItem& in, std::vector<HookBatchItem>& out) {
        const uint64_t in_cost = ledger_.cost(in.backing);
        auto admits = [&](uint64_t freed) {
            return retained_bytes_ - freed + in_cost <= max_bytes_;
        };
        std::vector<size_t> victims;                       // indices into q_
        std::unordered_map<const void*, uint64_t> leaving; // refs the victims drop
        uint64_t freed = 0;
        bool fits = false;
        for (int r = static_cast<int>(kRankMax);
             r > static_cast<int>(in.rank) && !fits; --r) {
            if (!depth_by_rank_[r]) continue;
            for (size_t i = 0; i < q_.size() && !fits; ++i) {
                const HookBatchItem& it = q_[i];
                if (it.rank != static_cast<uint8_t>(r)) continue;
                victims.push_back(i);
                const ring::BackingRef& b = it.backing;
                if (!b.key) {
                    freed += b.bytes;               // charged per item
                } else if (b.key != in.backing.key) {
                    const auto held = ledger_.held(b.key);
                    if (++leaving[b.key] == held.first) freed += held.second;
                }
                fits = admits(freed);
            }
        }
        if (!fits && victims.size() < q_.size()) return;   // evict nothing
        if (victims.empty()) return;
        // Commit, in the order chosen (notices follow eviction order).
        for (size_t i : victims) {
            HookBatchItem& it = q_[i];
            release_locked(it);
            if (depth_by_rank_[it.rank]) depth_by_rank_[it.rank]--;
            dropped_++;
            evicted_++;
            dropped_bytes_ += it.nbytes;
            dropped_by_rank_[it.rank]++;
            dropped_bytes_by_rank_[it.rank] += it.nbytes;
            out.push_back(std::move(it));
        }
        std::sort(victims.begin(), victims.end());
        for (auto i = victims.rbegin(); i != victims.rend(); ++i)
            q_.erase(q_.begin() + static_cast<std::ptrdiff_t>(*i));
    }

    mutable std::mutex          mu_;
    std::condition_variable     cv_;
    std::deque<HookBatchItem>   q_;
    uint64_t                    max_bytes_;
    uint64_t                    retained_bytes_ = 0;   // == ledger_.total()
    ring::BackingLedger         ledger_;
    uint64_t                    dropped_        = 0;
    uint64_t                    dropped_bytes_  = 0;
    uint64_t                    pushed_         = 0;
    uint64_t                    evicted_        = 0;
    bool                        stopped_        = false;
    bool                        wake_pending_   = false;
    // -- rank map + per-rank ledger (all guarded by mu_) ------------------- //
    std::unordered_map<std::string, uint32_t> act_ids_;   // act_name -> dense id
    std::unordered_map<uint64_t, uint8_t>     rank_of_;   // (act_id,layer) -> rank
    uint8_t                     default_rank_   = kRankBestEffort;
    bool                        class_map_set_  = false;
    bool                        class_eviction_ = true;
    uint64_t                    dropped_by_rank_[kRankCount]       = {0, 0, 0};
    uint64_t                    dropped_bytes_by_rank_[kRankCount] = {0, 0, 0};
    uint64_t                    depth_by_rank_[kRankCount]         = {0, 0, 0};
    // Own mutex; never taken while mu_ is held (see push()).
    DropNoticeBuffer            drops_;
};

}  // namespace ring_py
