// ring/pooled_block.h -- identify the backing allocation a CPU tensor keeps
// alive, and charge it once per queue.
//
// The P2P thread's buffer pool (buffer_pool.h) hands out size-classed blocks
// (power of two, 64 KiB floor) wrapped as tensors of the REQUESTED length, so
// tensor.nbytes() and even storage().nbytes() understate what a queued item
// keeps alive.  A pooled block carries a PooledBlockContext as its DataPtr
// context; backing_of() reads the real capacity from it.  Any other tensor is
// charged its storage's byte size.
//
// Several queued items may alias one allocation (per-request views of one
// step's buffer).  BackingLedger charges each distinct allocation once, at
// max(storage bytes, block capacity), while at least one queued item refers
// to it, and releases the charge when the last such item leaves the queue.
//
// Header-only; included where ATen is already linked.

#pragma once
#include <ATen/core/Tensor.h>
#include <c10/core/Storage.h>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace ring {

// DataPtr context of a pooled block.  The deleter (delete_pooled_block)
// destroys it through the virtual destructor, which returns the block to its
// pool (or frees it).
struct PooledBlockContext {
    PooledBlockContext(void* p, uint64_t cap) : ptr(p), capacity(cap) {}
    virtual ~PooledBlockContext() = default;
    void*    ptr;
    uint64_t capacity;   // bytes actually allocated for the block
};

inline void delete_pooled_block(void* ctx) {
    delete static_cast<PooledBlockContext*>(ctx);
}

// Allocated bytes behind a pooled tensor (or any view of it); 0 when the
// tensor's storage is not a pool block.
inline uint64_t pooled_block_capacity(const at::Tensor& t) {
    if (!t.defined() || !t.has_storage()) return 0;
    const c10::DataPtr& dp = t.storage().data_ptr();
    if (dp.get_deleter() != &delete_pooled_block) return 0;
    const auto* ctx = static_cast<const PooledBlockContext*>(dp.get_context());
    return ctx ? ctx->capacity : 0;
}

// The allocation a tensor keeps alive: an identity shared by every view of
// it, and the bytes it pins.
struct BackingRef {
    const void* key   = nullptr;   // storage base address; nullptr = nothing
    uint64_t    bytes = 0;
};

inline BackingRef backing_of(const at::Tensor& t) {
    BackingRef r;
    if (!t.defined()) return r;
    const uint64_t logical = static_cast<uint64_t>(t.nbytes());
    if (!t.has_storage()) {
        r.bytes = logical;
        return r;
    }
    const c10::Storage& s = t.storage();
    r.key = s.data();
    r.bytes = std::max<uint64_t>({logical, static_cast<uint64_t>(s.nbytes()),
                                  pooled_block_capacity(t)});
    if (!r.key) r.bytes = logical;   // no allocation to share
    return r;
}

// Retained-byte ledger: each distinct allocation is charged once while any
// queued item refers to it.  Not thread-safe; the owning queue guards it.
class BackingLedger {
public:
    // Bytes the ledger would grow by if `r` were added.
    uint64_t cost(const BackingRef& r) const {
        if (!r.key) return r.bytes;
        return blocks_.count(r.key) ? 0 : r.bytes;
    }
    // Add one reference.  Returns the bytes newly charged.
    uint64_t add(const BackingRef& r) {
        if (!r.key) { total_ += r.bytes; return r.bytes; }
        auto it = blocks_.find(r.key);
        if (it != blocks_.end()) { it->second.refs++; return 0; }
        blocks_.emplace(r.key, Entry{r.bytes, 1});
        total_ += r.bytes;
        return r.bytes;
    }
    // Drop one reference.  Returns the bytes released: the allocation's
    // charge when its last queued reference leaves, else 0.
    uint64_t remove(const BackingRef& r) {
        if (!r.key) { total_ -= r.bytes; return r.bytes; }
        auto it = blocks_.find(r.key);
        if (it == blocks_.end()) return 0;
        if (--it->second.refs > 0) return 0;
        const uint64_t bytes = it->second.bytes;
        blocks_.erase(it);
        total_ -= bytes;
        return bytes;
    }
    // Queued references to `key` and the bytes charged for it ({0, 0} when
    // the allocation is not held).  Lets a queue work out what removing a
    // set of items would release before it removes any.
    std::pair<uint64_t, uint64_t> held(const void* key) const {
        auto it = blocks_.find(key);
        if (it == blocks_.end()) return {0, 0};
        return {it->second.refs, it->second.bytes};
    }
    uint64_t total() const { return total_; }

private:
    struct Entry { uint64_t bytes; uint64_t refs; };
    std::unordered_map<const void*, Entry> blocks_;
    uint64_t total_ = 0;
};

}  // namespace ring
