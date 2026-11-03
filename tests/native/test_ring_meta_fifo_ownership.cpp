// TensorMetaFifo owns the StepContexts it holds.
//
// Metadata can be published for a forward that then fails before any
// producer publishes a payload; the step's context stays queued until
// teardown.  The context carries the step's CaptureLease, so a FIFO that
// dropped its raw pointers without deleting them kept the lease non-idle
// forever and an integration could never retire that configuration.
//
// Covered: destruction and clear() release every pending context exactly
// once -- including multi-hook steps, whose metas share ONE context -- and a
// context popped by the consumer is the consumer's to delete (the FIFO no
// longer frees it).
//
// Standard library only (no ATen, no CUDA).  Built and run by
// tests/test_native_ring_consumer_fixes.py.

#include <iostream>
#include <vector>

#include "ring/tensor_meta.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

ring_py::StepContext* make_ctx(const ring_py::CaptureLease& lease) {
    auto* ctx = new ring_py::StepContext;
    ctx->capture_lease = lease;
    return ctx;
}

std::vector<ring_py::TensorMeta> metas(size_t n) {
    std::vector<ring_py::TensorMeta> out(n);
    if (n) out.back().last_in_step = true;
    return out;
}

}  // namespace

int main() {
    // Destroying the FIFO with a pending single-hook step frees its context.
    {
        ring_py::CaptureLease lease(17);
        check(lease.idle(), "a fresh lease is idle");
        {
            ring_py::TensorMetaFifo fifo;
            auto m = metas(1);
            fifo.push_step(make_ctx(lease), m);
            check(!lease.idle(), "a queued context holds the lease");
        }
        check(lease.idle(), "destroying the FIFO releases the pending lease");
    }

    // A multi-hook step pushes ONE context for all of its metas; several
    // pending steps, each released once (no double free under ASan/glibc).
    {
        ring_py::CaptureLease a(1), b(2);
        ring_py::TensorMetaFifo fifo;
        auto ma = metas(5);
        auto mb = metas(3);
        fifo.push_step(make_ctx(a), ma);
        fifo.push_step(make_ctx(b), mb);
        check(fifo.pending_contexts() == 2, "one context per step, not per meta");
        check(!a.idle() && !b.idle(), "pending steps hold their leases");
        fifo.clear();
        check(fifo.pending_contexts() == 0, "clear() empties the context queue");
        ring_py::TensorMeta m;
        check(!fifo.pop(m), "clear() empties the meta queue");
        check(a.idle() && b.idle(), "clear() releases every pending lease");
    }

    // The pop path: a popped context belongs to the consumer.  The FIFO must
    // not free it again on destruction, and the consumer's delete releases
    // the lease (the P2P thread deletes it after the step's last hook).
    {
        ring_py::CaptureLease lease(3);
        ring_py::StepContext* popped = nullptr;
        {
            ring_py::TensorMetaFifo fifo;
            auto m = metas(2);
            fifo.push_step(make_ctx(lease), m);
            ring_py::TensorMeta out;
            check(fifo.pop(out) && fifo.pop(out) && out.last_in_step,
                  "both metas pop in order");
            popped = fifo.pop_context();
            check(popped != nullptr, "the context pops");
            check(fifo.pop_context() == nullptr, "only one context per step");
        }
        check(!lease.idle(), "the consumer still owns the popped context");
        delete popped;
        check(lease.idle(), "the consumer's delete releases the lease");
    }

    // A context with a payload-less step still pending behind a popped one:
    // only the pending one is freed by the FIFO.
    {
        ring_py::CaptureLease first(4), second(5);
        ring_py::StepContext* popped = nullptr;
        {
            ring_py::TensorMetaFifo fifo;
            auto m1 = metas(1);
            auto m2 = metas(1);
            fifo.push_step(make_ctx(first), m1);
            fifo.push_step(make_ctx(second), m2);
            popped = fifo.pop_context();
        }
        check(second.idle(), "the still-pending context is freed with the FIFO");
        check(!first.idle(), "the popped context is not freed by the FIFO");
        delete popped;
        check(first.idle(), "and is released by its owner");
    }

    // A null context (defensive) is accepted and not queued.
    {
        ring_py::TensorMetaFifo fifo;
        auto m = metas(1);
        fifo.push_step(nullptr, m);
        check(fifo.pending_contexts() == 0, "a null context is not queued");
    }

    if (g_failures) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
