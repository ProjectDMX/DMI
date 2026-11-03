// BufferPool::acquire under allocation failure: every allocation acquire()
// makes -- the block's context, then ATen's StorageImpl, TensorImpl, ... --
// is failed in turn with std::bad_alloc.  Each failure must propagate the
// exception and return the block to the pool exactly once; the first count
// that does not fail must hand out a working tensor whose death returns the
// block exactly once.
//
// The pool is given an idle cap, so a returned block sits on its free list
// and idle_bytes() counts returns: one return is exactly one size class, a
// double return (the context deleted twice) is two, a leak is zero.
//
// Compiles buffer_pool.h against ATen (CPU only).  Built and run by
// tests/test_native_ring_consumer_fixes.py.

#include <ATen/ATen.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>

#include "ring/buffer_pool.h"
#include "ring/pooled_block.h"

namespace {

// > 0: fail the n-th operator new from now on, once.
std::atomic<int> g_fail_after{0};
std::atomic<int> g_injected{0};

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

}  // namespace

void* operator new(std::size_t n) {
    if (g_fail_after.load() > 0 && g_fail_after.fetch_sub(1) == 1) {
        g_injected.fetch_add(1);
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main() {
    at::zeros({1}, at::kByte);   // initialize ATen before injecting
    constexpr uint64_t kBytes = 32;
    const uint64_t cls = ring::BufferPool::capacity_for(kBytes);
    bool succeeded = false;
    int failures_seen = 0;
    for (int k = 1; k <= 64 && !succeeded; ++k) {
        ring::BufferPool pool(1ULL << 30);
        const int injected0 = g_injected.load();
        bool threw = false;
        at::Tensor t;
        g_fail_after.store(k);
        try {
            t = pool.acquire(kBytes);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        g_fail_after.store(0);
        const std::string at_k = " (allocation " + std::to_string(k) + ")";
        if (threw) {
            ++failures_seen;
            check(g_injected.load() == injected0 + 1, "one injected failure" + at_k);
            check(pool.idle_bytes() == cls,
                  "a failed acquire returns its block exactly once" + at_k +
                  ": idle_bytes=" + std::to_string(pool.idle_bytes()));
            // The returned block is reused, then returned once more.
            at::Tensor again = pool.acquire(kBytes);
            check(again.defined() && again.nbytes() == kBytes,
                  "the pool still serves after a failure" + at_k);
            check(pool.idle_bytes() == 0, "the returned block is reused" + at_k);
            again.reset();
            check(pool.idle_bytes() == cls, "reuse returns it once" + at_k);
        } else {
            succeeded = true;
            check(t.defined() && t.nbytes() == kBytes, "acquire succeeds" + at_k);
            check(ring::pooled_block_capacity(t) == cls, "block capacity");
            check(pool.idle_bytes() == 0, "a live tensor holds its block");
            at::Tensor view = t.narrow(0, 8, 8);
            t.reset();
            check(pool.idle_bytes() == 0, "a live view holds the block");
            view.reset();
            check(pool.idle_bytes() == cls, "the last view returns it once");
        }
    }
    check(succeeded, "acquire eventually succeeds");
    // The context, StorageImpl and TensorImpl allocations at least.
    check(failures_seen >= 3, "every allocation of acquire was failed: " +
                              std::to_string(failures_seen));
    if (g_failures) return 1;
    std::cout << "buffer pool failures=" << failures_seen << " ok\n";
    return 0;
}
