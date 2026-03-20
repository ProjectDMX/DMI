// ring/ring_engine.cu -- RingEngine implementation (compiled with nvcc because
// it owns AllocatedRing which calls task_ring_init / cudaMemsetAsync).

#include "ring_engine.h"

#include <stdexcept>

namespace ring {

RingEngine::RingEngine(const RingConfig& cfg, ring_py::TensorMetaFifo& fifo,
                       SubmitFn submit_fn)
    : cfg_(cfg), ring_(cfg)
{
    if (cfg_.payload_ring_bytes % PAYLOAD_ALIGN != 0) {
        throw std::runtime_error("RingConfig: payload_ring_bytes must be a multiple of "
                                 "PAYLOAD_ALIGN (wrap offset alignment)");
    }

    if (cfg_.drain_poll_timeout_us == 0) {
        throw std::runtime_error("RingConfig: drain_poll_timeout_us must be > 0. "
            "The drain thread must poll periodically to process entries "
            "published mid-forward.");
    }

    auto* buf = ring_.state().payload_buf;
    if (reinterpret_cast<uintptr_t>(buf) % PAYLOAD_ALIGN != 0) {
        throw std::runtime_error("RingEngine: payload_buf is not PAYLOAD_ALIGN-aligned "
                                 "(unexpected: cudaMalloc guarantees >= 256-byte alignment)");
    }

    const uint64_t staging_bytes = cfg.effective_staging_bytes();
    staging_.init(staging_bytes);

    drain_ = std::make_unique<DrainThread>(ring_.state(), staging_, cfg_);
    p2p_   = std::make_unique<P2PThread>(*drain_, fifo, cfg_, std::move(submit_fn));
}

RingEngine::~RingEngine() noexcept {
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
    drain_->stop();
    drain_->signal_p2p_stop();
    p2p_->stop();
}

}  // namespace ring
