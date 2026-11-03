// ring/drop_notice.h -- attribution for the consumer queues' load shedding.
//
// Both in-process consumer queues shed load on overflow: BatchQueue drops
// whole batches when its retained-byte cap would be exceeded, SliceQueue drops
// single slices when its item cap is reached.  Counting a drop is not enough
// for a consumer that aggregates per request: it has to know WHICH requests
// lost data, or it reports results over data it silently lacks.  Every drop
// therefore files a DropNotice naming the request ids that were on the dropped
// item; the consumer drains the notices and marks those requests incomplete.
//
// The notice buffer is itself bounded.  When it saturates, overflowing
// notices are counted and a sticky flag is raised; the consumer contract for
// that flag is pessimistic -- treat every request in flight since the previous
// poll as incomplete -- because the identity of the losers is then unknown.
//
// Invariant: no notice means nothing was lost.  A quiet buffer must never mean
// "we forgot to look".
//
// Header-only, ATen-free, and lock-order safe: the buffer has its own mutex
// and never calls back into the owning queue, so a queue may record a notice
// while holding or after releasing its own lock.

#pragma once
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ring/capture_lease.h"

namespace ring_py {

// Which queue shed the load.  Stable strings: consumers key on them.
inline constexpr const char* kDropSourceBatch = "batch_queue";
inline constexpr const char* kDropSourceSlice = "slice_queue";
// The shed item was not the arriving one: a queued, lower-priority batch was
// evicted to make room for it (BatchQueue rank-ordered eviction).
inline constexpr const char* kDropSourceBatchEvict = "batch_queue_evict";

// ---------------------------------------------------------------------------
// Drop rank -- the shedding order.
//
// The transport knows nothing about the consumer's analyses; it only knows
// that an (act_name, layer) stream has a priority, and priorities are totally
// ordered.  Lower rank = more urgent = shed later.  The consumer installs the
// map (BatchQueue::set_layer_class_map, or per configuration through a
// CaptureLease); a stream the map does not name gets the default rank.
// Three ranks cover a deadline per token, a deadline per finished request,
// and no deadline.
// ---------------------------------------------------------------------------
inline constexpr uint8_t kRankStreaming       = 0;  // needed every step
inline constexpr uint8_t kRankRequestBoundary = 1;  // needed when the request ends
inline constexpr uint8_t kRankBestEffort      = 2;  // no deadline
inline constexpr uint8_t kRankMax             = 2;  // ranks are [0, kRankMax]
inline constexpr size_t  kRankCount           = 3;
// A path that does not rank (SliceQueue).  Never stored on a queued item,
// only reported on a notice.
inline constexpr uint8_t kRankUnset           = 255;

// Clamp a rank crossing from Python: an invented rank becomes the documented
// default for the unknown, never "more sheddable than everything".
inline uint8_t clamp_rank(int r) {
    if (r < 0 || r > static_cast<int>(kRankMax)) return kRankBestEffort;
    return static_cast<uint8_t>(r);
}

// steady_clock microseconds (the clock of DrainTask::capture_ts_us).
inline int64_t drop_notice_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One shed item, with the identity of everything that was on it.
struct DropNotice {
    CaptureLease capture_lease;
    const char* source      = kDropSourceBatch;  // static string, never freed
    std::string hook;                            // act_name of the lost capture
    int32_t     layer_no    = -1;
    uint64_t    step_seq    = 0;   // batch path: the step that lost data
    int32_t     start_token = -1;  // slice path only (token ranges are
    int32_t     end_token   = -1;  //  per request in a batch's segments)
    uint64_t    nbytes      = 0;   // bytes of capture lost
    int64_t     ts_us       = 0;   // steady-clock time of the drop
    // Rank of the lost stream; kRankUnset on the slice path.
    uint8_t     rank        = kRankUnset;
    // Every request that had rows on the dropped item: a batch's req_ids, or
    // exactly one on the slice path.
    std::vector<std::string> req_ids;
};

class DropNoticeBuffer {
public:
    explicit DropNoticeBuffer(size_t cap = 4096) : cap_(cap) {}

    // Called on the drop path (P2P thread): no GIL, no allocation beyond the
    // notice itself.
    void record(DropNotice n) {
        std::lock_guard<std::mutex> lk(mu_);
        recorded_++;
        if (q_.size() >= cap_) {
            overflow_++;
            overflow_total_++;
            sticky_ = true;
            return;
        }
        q_.push_back(std::move(n));
    }

    struct Drained {
        std::vector<DropNotice> notices;
        // Notices lost since the previous drain.
        uint64_t overflow = 0;
        // Any notice lost since the previous drain: the consumer must treat
        // every request in flight over that window as incomplete.
        bool     sticky   = false;
        // Notices still buffered after this drain (max_items truncation).
        size_t   pending  = 0;
    };

    // Drain up to max_items notices and the overflow/sticky state, clearing
    // both.  Non-blocking.
    Drained drain(size_t max_items) {
        Drained out;
        std::lock_guard<std::mutex> lk(mu_);
        while (!q_.empty() && out.notices.size() < max_items) {
            out.notices.push_back(std::move(q_.front()));
            q_.pop_front();
        }
        out.overflow = overflow_;
        out.sticky   = sticky_;
        out.pending  = q_.size();
        overflow_    = 0;
        sticky_      = false;   // sticky covers one inter-poll window
        return out;
    }

    size_t   pending()        const { std::lock_guard<std::mutex> lk(mu_); return q_.size(); }
    // Cumulative, never reset by drain.
    uint64_t recorded()       const { std::lock_guard<std::mutex> lk(mu_); return recorded_; }
    uint64_t overflow_total() const { std::lock_guard<std::mutex> lk(mu_); return overflow_total_; }
    size_t   cap()            const { return cap_; }

private:
    mutable std::mutex     mu_;
    std::deque<DropNotice> q_;
    size_t                 cap_;
    uint64_t               recorded_       = 0;  // notices offered (kept + lost)
    uint64_t               overflow_       = 0;  // lost since the last drain
    uint64_t               overflow_total_ = 0;  // lost, cumulative
    bool                   sticky_         = false;
};

}  // namespace ring_py
