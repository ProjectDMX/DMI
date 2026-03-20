// ring/pinned_staging.h -- Pre-allocated pinned host ring buffer for batch D2H.
//
// Used by the drain thread (advances head after batch D2H) and the p2p thread
// (advances tail after copying to pageable memory).
//
// head_ is written only by drain thread, tail_ only by p2p thread.
// Synchronisation between the two threads uses staging_mu_ + staging_cv_
// (owned externally by the ring engine).

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace ring {

class PinnedStaging {
public:
    PinnedStaging() = default;
    ~PinnedStaging() noexcept {
        if (base_) cudaFreeHost(base_);
    }

    PinnedStaging(const PinnedStaging&)            = delete;
    PinnedStaging& operator=(const PinnedStaging&) = delete;

    void init(uint64_t total_bytes) {
        capacity_ = total_bytes;
        void* p = nullptr;
        if (cudaHostAlloc(&p, capacity_, cudaHostAllocDefault) != cudaSuccess)
            throw std::runtime_error("PinnedStaging: cudaHostAlloc failed");
        base_ = static_cast<uint8_t*>(p);
        head_ = 0;
        tail_ = 0;
    }

    uint8_t*  base()     const { return base_; }
    uint64_t  capacity() const { return capacity_; }

    uint64_t  head() const { return head_; }
    uint64_t  tail() const { return tail_; }

    uint64_t free_bytes() const { return capacity_ - (head_ - tail_); }
    uint64_t phys(uint64_t logical) const { return logical % capacity_; }

    void advance_head(uint64_t nbytes) { head_ += nbytes; }
    void advance_tail(uint64_t nbytes) { tail_ += nbytes; }

private:
    uint8_t*  base_     = nullptr;
    uint64_t  capacity_ = 0;
    uint64_t  head_     = 0;  // monotonic; drain thread advances
    uint64_t  tail_     = 0;  // monotonic; p2p thread advances
};

}  // namespace ring
