// BatchQueue rank-ordered eviction with views of a shared allocation.
//
// The byte cap charges each distinct allocation once while any queued item
// refers to it.  Eviction must judge what a set of victims would actually
// free from the references that REMAIN -- surviving aliases of any rank and
// the arriving item's own -- and evict nothing unless the arrival is then
// admitted.
//
// Compiles batch_queue.h against ATen (CPU only).  Built and run by
// tests/test_native_ring_consumer_fixes.py.

#include <ATen/ATen.h>

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "ring/batch_queue.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

// Ranks: "urgent*" 0, "mid" 1, anything else best-effort (2).
void install_map(ring_py::BatchQueue& q) {
    q.set_layer_class_map(
        {{"urgent", 0, 0}, {"newurgent", 0, 0}, {"mid", 0, 1}}, 2);
}

void push(ring_py::BatchQueue& q, const char* act, const at::Tensor& t) {
    ring_py::HookBatchItem item;
    item.act_name = act;
    item.layer_no = 0;
    item.tensor = t;
    item.nbytes = t.nbytes();
    item.req_ids = {act};
    q.push(std::move(item));
}

struct Outcome {
    size_t size;
    uint64_t retained, dropped, evicted;
    std::string items;
    uint64_t retained_after_drain;
};

Outcome finish(ring_py::BatchQueue& q) {
    Outcome o{q.size(), q.retained_bytes(), q.dropped(), q.evicted(), "", 0};
    for (const auto& item : q.drain(100, 0)) o.items += item.act_name + ",";
    o.retained_after_drain = q.retained_bytes();
    return o;
}

void expect(const char* label, const Outcome& o, size_t size, uint64_t retained,
            uint64_t dropped, uint64_t evicted, const std::string& items) {
    std::cout << label << " size=" << o.size << " retained=" << o.retained
              << " dropped=" << o.dropped << " evicted=" << o.evicted
              << " items=" << o.items
              << " retained_after_drain=" << o.retained_after_drain << '\n';
    const std::string l = label;
    check(o.size == size, l + ": size");
    check(o.retained == retained, l + ": retained");
    check(o.dropped == dropped, l + ": dropped");
    check(o.evicted == evicted, l + ": evicted");
    check(o.items == items, l + ": items");
    check(o.retained_after_drain == 0, l + ": nothing retained after drain");
}

}  // namespace

int main() {
    // A drained urgent view leaves its 80-byte allocation pinned only by a
    // low view.  Evicting that low view frees all 80 bytes: the new urgent
    // item fits (10 + 80 <= 100).
    {
        ring_py::BatchQueue q(100);
        install_map(q);
        auto shared = at::zeros({80}, at::kByte);
        push(q, "urgent", shared.narrow(0, 0, 40));
        push(q, "low", shared.narrow(0, 40, 40));
        q.drain(1, 0);
        push(q, "urgent", at::zeros({10}, at::kByte));
        push(q, "newurgent", at::zeros({80}, at::kByte));
        expect("underestimated_evictable_capacity", finish(q),
               2, 90, 1, 1, "urgent,newurgent,");
    }
    // A low and an urgent view share 80 bytes.  Evicting low frees nothing
    // (urgent keeps the allocation), so the newcomer cannot fit: nothing is
    // evicted and only the newcomer is refused.
    {
        ring_py::BatchQueue q(100);
        install_map(q);
        auto shared = at::zeros({80}, at::kByte);
        push(q, "low", shared.narrow(0, 0, 40));
        push(q, "urgent", shared.narrow(0, 40, 40));
        push(q, "newurgent", at::zeros({80}, at::kByte));
        expect("overestimated_evictable_capacity", finish(q),
               2, 80, 1, 0, "low,urgent,");
        check(q.qsize_by_rank() == std::vector<uint64_t>({0, 0, 0}),
              "depth by rank after drain");
    }
    // A shared allocation is freed only when its LAST view goes: the low view
    // alone frees nothing, the mid view then frees all 80 bytes.  Victims are
    // taken worst rank first, oldest first within a rank.
    {
        ring_py::BatchQueue q(120);
        install_map(q);
        auto shared = at::zeros({80}, at::kByte);
        push(q, "low", shared.narrow(0, 0, 40));
        push(q, "mid", shared.narrow(0, 40, 40));
        push(q, "low2", at::zeros({10}, at::kByte));
        push(q, "newurgent", at::zeros({100}, at::kByte));
        // 90 + 100 > 120.  low frees 0, low2 frees 10 (180 > 120), mid frees
        // the shared 80: 100 retained.
        expect("last_view_frees_the_allocation", finish(q),
               1, 100, 3, 3, "newurgent,");
    }
    // The arriving item aliases a queued allocation: it adds no bytes, so it
    // is admitted at the cap with nothing evicted.
    {
        ring_py::BatchQueue q(100);
        install_map(q);
        auto shared = at::zeros({80}, at::kByte);
        push(q, "low", shared.narrow(0, 0, 40));
        push(q, "low2", at::zeros({20}, at::kByte));
        push(q, "newurgent", shared.narrow(0, 40, 40));
        expect("arrival_aliases_a_queued_allocation", finish(q),
               3, 100, 0, 0, "low,low2,newurgent,");
    }
    // A less urgent arrival never evicts a more urgent resident.
    {
        ring_py::BatchQueue q(100);
        install_map(q);
        auto big = at::zeros({100}, at::kByte);
        push(q, "urgent", at::zeros({10}, at::kByte));      // 10
        push(q, "low", big.narrow(0, 0, 50));               // 110: refused
        expect("oversized_low_is_refused", finish(q), 1, 10, 1, 0, "urgent,");
    }
    {
        // Queue over its cap through the empty-queue rule (one item larger
        // than the cap); an urgent alias of that same allocation arrives.
        // Evicting the low view frees nothing, but it empties the queue,
        // which always admits: the urgent alias replaces it and the
        // allocation stays charged once.
        ring_py::BatchQueue q(50);
        install_map(q);
        auto big = at::zeros({80}, at::kByte);
        push(q, "low", big.narrow(0, 0, 40));               // 80 > 50, admitted
        push(q, "newurgent", big.narrow(0, 40, 40));
        expect("alias_over_cap_empties_the_queue", finish(q),
               1, 80, 1, 1, "newurgent,");
    }
    // Reclassifying streams while items are queued: an item keeps the rank it
    // was enqueued with, and the per-rank depths stay consistent.
    {
        ring_py::BatchQueue q(100);
        install_map(q);
        push(q, "a", at::zeros({60}, at::kByte));            // best-effort
        q.set_layer_class_map({{"a", 0, 0}, {"b", 0, 0}}, 2);
        check(q.qsize_by_rank() == std::vector<uint64_t>({0, 0, 1}),
              "queued item keeps its enqueue rank");
        push(q, "b", at::zeros({60}, at::kByte));            // urgent now
        check(q.qsize_by_rank() == std::vector<uint64_t>({1, 0, 0}),
              "depth by rank after eviction");
        expect("reclassified_while_queued", finish(q), 1, 60, 1, 1, "b,");
    }
    if (g_failures) return 1;
    std::cout << "batch queue shared eviction ok\n";
    return 0;
}
