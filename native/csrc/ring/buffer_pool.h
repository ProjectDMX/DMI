// ring/buffer_pool.h -- refcount-recycled pageable buffers for the P2P thread.
//
// The P2P thread copies each step's tensor from the pinned staging ring into a
// pageable CPU buffer, then slices it per request or hands it over whole.
// PyTorch's default CPU allocator returns large blocks to the OS (glibc mmap
// threshold), so a fresh at::empty per step is freshly mmap'd and the copy
// takes cold page faults on every touched page.  This pool keeps a small set
// of size-classed buffers resident so their pages stay warm across steps.
//
// Lifetime: a buffer is not consumed inside the P2P thread.  Its views go to
// a consumer that may hold them for an unbounded time -- past the end of the
// step, and past the ring engine's own shutdown.  A buffer is therefore
// wrapped with at::from_blob(..., deleter) and returned only when the LAST
// reference across all derived views is dropped, on whatever thread drops
// it.  The deleter holds a reference to the pool's shared state, never to
// the BufferPool object, so a tensor outliving the pool frees its block
// instead of touching a destroyed pool.
//
// Accounting: a block is a size class (power of two, 64 KiB floor) larger
// than the tensor it backs.  Its DataPtr context is a PooledBlockContext
// (pooled_block.h) carrying the real capacity, so a queue that retains the
// tensor -- or any view of it -- can charge the allocation it keeps alive
// rather than the logical length (backing_of()).
//
// Header-only; included only where ATen is already linked.

#pragma once
#include "ring/pooled_block.h"

#include <ATen/ATen.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ring {

class BufferPool {
public:
    // idle_cap_bytes: maximum retained (free-list) bytes.  A returned buffer
    // that would push the free lists past it is freed instead.  0 disables
    // retention (no pooling).
    explicit BufferPool(uint64_t idle_cap_bytes = 256ULL * 1024 * 1024)
        : state_(std::make_shared<State>(idle_cap_bytes)) {}

    ~BufferPool() {
        // Free the idle blocks now; blocks still referenced by live tensors
        // are freed by their deleters when the last reference drops.
        std::lock_guard<std::mutex> lk(state_->mu);
        state_->closed = true;
        for (auto& kv : state_->free_lists)
            for (void* p : kv.second) std::free(p);
        state_->free_lists.clear();
        state_->idle_bytes = 0;
    }

    BufferPool(const BufferPool&)            = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    // A pageable buffer of at least nbytes, wrapped as a uint8 CPU tensor of
    // exactly nbytes.  When the tensor and every view derived from it die,
    // the block returns to the pool.  Returns an undefined tensor when
    // allocation fails: the caller falls back to at::empty.
    // pooled_block_capacity(tensor) reports the block's allocated size.
    at::Tensor acquire(uint64_t nbytes) {
        // Above 2^62 the power-of-two size class would overflow (and the
        // byte count would not fit the tensor's int64 size): no buffer.
        if (nbytes == 0 || nbytes > (1ULL << 62)) return at::Tensor{};
        const uint64_t cap = size_class(nbytes);

        void* ptr = nullptr;
        {
            std::lock_guard<std::mutex> lk(state_->mu);
            auto it = state_->free_lists.find(cap);
            if (it != state_->free_lists.end() && !it->second.empty()) {
                ptr = it->second.back();
                it->second.pop_back();
                state_->idle_bytes -= cap;
            }
        }
        if (!ptr) {
            ptr = aligned_alloc_bytes(cap);
            if (!ptr) return at::Tensor{};
        }

        // Exactly one owner of the block at every point, so a throw anywhere
        // returns it exactly once:
        //   1. the raw block, until its context exists (a failed `new`
        //      returns it here);
        //   2. `ctx`, until TensorMaker::context() takes it.  context() is
        //      noexcept and only stores the pointer, so the handover cannot
        //      fail half way;
        //   3. the TensorMaker, whose destructor deletes the context if
        //      make_tensor() throws, then the tensor's DataPtr.
        // Nothing here deletes the context once TensorMaker owns it.
        std::unique_ptr<Block> ctx;
        try {
            ctx.reset(new Block(ptr, cap, state_));
        } catch (...) {
            state_->release(ptr, cap);
            throw;
        }
        const int64_t sizes[1] = {static_cast<int64_t>(nbytes)};
        at::TensorMaker maker = at::for_blob(ptr, sizes);
        maker.context(ctx.release(), &delete_pooled_block);
        return maker
            .options(at::TensorOptions().dtype(at::kByte).device(at::kCPU))
            .make_tensor();
    }

    // Size class acquire() allocates for nbytes (diagnostic and tests).
    static uint64_t capacity_for(uint64_t nbytes) { return size_class(nbytes); }

    // Idle bytes currently retained (diagnostic).
    uint64_t idle_bytes() const {
        std::lock_guard<std::mutex> lk(state_->mu);
        return state_->idle_bytes;
    }

private:
    struct State {
        explicit State(uint64_t cap) : idle_cap_bytes(cap) {}
        // Return a block to its size class, or free it when over the idle
        // cap or after the pool closed.
        void release(void* ptr, uint64_t cap) {
            std::lock_guard<std::mutex> lk(mu);
            if (closed || idle_bytes + cap > idle_cap_bytes) {
                std::free(ptr);
                return;
            }
            free_lists[cap].push_back(ptr);
            idle_bytes += cap;
        }
        mutable std::mutex mu;
        std::unordered_map<uint64_t, std::vector<void*>> free_lists;  // class -> blocks
        uint64_t idle_bytes = 0;
        const uint64_t idle_cap_bytes;
        bool closed = false;
    };

    // DataPtr context of one handed-out block: returns it on destruction.
    // Holds the pool's shared state, never the BufferPool object.
    struct Block final : PooledBlockContext {
        Block(void* p, uint64_t cap, std::shared_ptr<State> st)
            : PooledBlockContext(p, cap), state(std::move(st)) {}
        ~Block() override { state->release(ptr, capacity); }
        std::shared_ptr<State> state;
    };

    // Round up to the next power of two, with a floor so small steps do not
    // fragment the free lists into many singleton classes.
    static uint64_t size_class(uint64_t nbytes) {
        uint64_t c = kMinClass;
        while (c < nbytes) c <<= 1;
        return c;
    }

    static void* aligned_alloc_bytes(uint64_t cap) {
        // cap is a power of two >= kMinClass: a valid alignment multiple.
        void* p = nullptr;
        if (posix_memalign(&p, kAlign, cap) != 0) return nullptr;
        return p;
    }

    static constexpr uint64_t kMinClass = 64ULL * 1024;  // 64 KiB floor
    static constexpr size_t   kAlign    = 64;

    std::shared_ptr<State> state_;
};

}  // namespace ring
