// ring/pinned_pool.h — Single shared pinned (page-locked) host ring buffer.
//
// One cudaHostAlloc region is used as a ring: alloc() carves from the head,
// free() advances the tail.  Allocations complete in FIFO order so tail
// advances monotonically without locking.
//
// No mutex is needed: only the single drain thread calls alloc() and free().
// Wrap-around is handled by skip-padding: if the requested bytes don't fit
// linearly at the current head offset, the remaining bytes at the end are
// skipped and the allocation starts from offset 0.  alloc_bytes (returned via
// out-parameter) encodes the skip + actual size so free() accounts for both.

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace ring {

class PinnedPool {
public:
    PinnedPool() = default;
    ~PinnedPool() noexcept {
        if (base_) cudaFreeHost(base_);
    }

    PinnedPool(const PinnedPool&)            = delete;
    PinnedPool& operator=(const PinnedPool&) = delete;

    // Allocate a single pinned region of `total_bytes`.
    // Call once before start().
    void init(uint64_t total_bytes) {
        capacity_ = total_bytes;
        void* p = nullptr;
        if (cudaHostAlloc(&p, capacity_, cudaHostAllocDefault) != cudaSuccess)
            throw std::runtime_error("PinnedPool: cudaHostAlloc failed");
        base_  = static_cast<uint8_t*>(p);
        head_  = 0;
        tail_  = 0;
    }

    // Allocate `nbytes` bytes from the ring.
    // Returns pointer to the allocation and sets `alloc_bytes_out` to the
    // number of bytes to pass to free() (>= nbytes, includes any skip padding).
    // Returns nullptr if the ring does not have enough space (back-pressure).
    uint8_t* alloc(uint64_t nbytes, uint64_t& alloc_bytes_out) {
        // Round up to 128-byte alignment.
        const uint64_t align   = 128;
        const uint64_t aligned = (nbytes + align - 1) & ~(align - 1);
        const uint64_t used    = head_ - tail_;
        const uint64_t off     = head_ % capacity_;

        if (off + aligned <= capacity_) {
            // Fits linearly — no wrap needed.
            if (used + aligned > capacity_) return nullptr;  // ring full
            alloc_bytes_out = aligned;
            head_ += aligned;
            return base_ + off;
        }

        // Wrap-around: skip remaining bytes at end, restart from offset 0.
        const uint64_t skip  = capacity_ - off;
        const uint64_t total = skip + aligned;
        if (used + total > capacity_) return nullptr;  // not enough space
        alloc_bytes_out = total;
        head_ += total;
        return base_;  // allocation starts at offset 0
    }

    // Return `alloc_bytes` (as given by alloc()) to the ring.
    void free(uint64_t alloc_bytes) {
        tail_ += alloc_bytes;
    }

    uint64_t capacity() const { return capacity_; }

private:
    uint8_t*  base_     = nullptr;
    uint64_t  capacity_ = 0;
    uint64_t  head_     = 0;  // monotonic; head_ % capacity_ = write offset
    uint64_t  tail_     = 0;  // monotonic; tail_ % capacity_ = free offset
};

}  // namespace ring
