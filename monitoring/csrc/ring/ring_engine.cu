// ring/ring_engine.cu — RingEngine implementation (compiled with nvcc because
// it owns AllocatedRing which calls task_ring_init / cudaMemsetAsync).

#include "ring_engine.h"

#include <stdexcept>

namespace ring {

RingEngine::RingEngine(const RingConfig& cfg, ring_py::TensorMetaFifo& fifo,
                       SubmitFn submit_fn)
    : cfg_(cfg), ring_(cfg)
{
    // --- Validate config constraints ---

    if (cfg_.chunk_bytes % PAYLOAD_ALIGN != 0) {
        throw std::runtime_error("RingConfig: chunk_bytes must be a multiple of "
                                 "PAYLOAD_ALIGN (uint4 source alignment)");
    }
    if (cfg_.payload_ring_bytes % PAYLOAD_ALIGN != 0) {
        throw std::runtime_error("RingConfig: payload_ring_bytes must be a multiple of "
                                 "PAYLOAD_ALIGN (wrap offset alignment)");
    }
    if (cfg_.chunk_bytes > cfg_.payload_ring_bytes) {
        throw std::runtime_error("RingConfig: chunk_bytes must be <= payload_ring_bytes "
                                 "(one chunk must always fit in GPU ring)");
    }

    // payload_buf is allocated by cudaMalloc which guarantees >= 256-byte
    // alignment (CUDA Best Practices Guide §10.2.1.2).  Assert this so that
    // all payload offsets (multiples of PAYLOAD_ALIGN) produce aligned addresses.
    auto* buf = ring_.state().payload_buf;
    if (reinterpret_cast<uintptr_t>(buf) % PAYLOAD_ALIGN != 0) {
        throw std::runtime_error("RingEngine: payload_buf is not PAYLOAD_ALIGN-aligned "
                                 "(unexpected: cudaMalloc guarantees >= 256-byte alignment)");
    }

    const uint64_t staging_bytes = cfg.effective_staging_bytes();
    staging_.init(staging_bytes);

    if (cfg_.drain_poll_timeout_us == 0 && !cfg_.drain_notify_on_forward) {
        fprintf(stderr, "[ring_engine] WARNING: drain_poll_timeout_us=0 and "
                "drain_notify_on_forward=false — drain thread will only flush "
                "at stop(). Ring must hold all data or producer will deadlock.\n");
    }

    if (cfg_.drain_flush.timeout_us > 0) {
        if (cfg_.drain_poll_timeout_us == 0) {
            fprintf(stderr, "[ring_engine] WARNING: drain_flush.timeout_us=%lu but "
                    "drain_poll_timeout_us=0 — drain thread does not poll, so "
                    "timeout check only runs on notify_on_forward wake-ups "
                    "(or never if that is also disabled).\n",
                    (unsigned long)cfg_.drain_flush.timeout_us);
        } else if (cfg_.drain_flush.timeout_us < cfg_.drain_poll_timeout_us) {
            fprintf(stderr, "[ring_engine] WARNING: drain_flush.timeout_us=%lu < "
                    "drain_poll_timeout_us=%lu — effective resolution limited to "
                    "drain_poll_timeout_us.\n",
                    (unsigned long)cfg_.drain_flush.timeout_us,
                    (unsigned long)cfg_.drain_poll_timeout_us);
        }
    }

    drain_ = std::make_unique<DrainThread>(ring_.state(), staging_, cfg_);
    p2p_   = std::make_unique<P2PThread>(*drain_, fifo, cfg_, std::move(submit_fn));
}

RingEngine::~RingEngine() noexcept {
    // Drain thread final-flushes all GPU entries into the task queue, then
    // joins.  Only after that do we signal p2p to finish remaining tasks.
    if (drain_) {
        drain_->stop();
        drain_->signal_p2p_stop();
    }
    if (p2p_) p2p_->stop();
}

void RingEngine::init(cudaStream_t stream) {
    ring_.init(stream);
}

void RingEngine::start() {
    drain_->start();
    p2p_->start();
}

void RingEngine::stop() {
    drain_->stop();            // final-flush all GPU entries → task queue, then join
    drain_->signal_p2p_stop(); // no more tasks will be pushed; tell p2p to finish
    p2p_->stop();              // p2p processes remaining tasks, then joins
}

}  // namespace ring
