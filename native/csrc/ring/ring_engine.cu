// ring/ring_engine.cu -- RingEngine implementation (compiled with nvcc because
// it owns AllocatedRing which calls task_ring_init / cudaMemsetAsync).

#include "ring_engine.h"

#include <stdexcept>

namespace ring {

RingEngine::RingEngine(const RingConfig& cfg, ring_py::TensorMetaFifo& fifo,
                       SubmitFn submit_fn, BatchSink batch_sink)
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
    p2p_   = std::make_unique<P2PThread>(*drain_, fifo, cfg_, std::move(submit_fn),
                                         std::move(batch_sink));
}

bool RingEngine::drain_to_sink_and_wait(uint32_t timeout_ms) {
    if (timeout_ms == 0) return drain_to_sink_and_wait_until(std::nullopt);
    return drain_to_sink_and_wait_until(
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms));
}

bool RingEngine::drain_to_sink_and_wait_until(
    std::optional<std::chrono::steady_clock::time_point> deadline) {
    if (!p2p_) {
        throw std::logic_error(
            "drain_to_sink_and_wait is a legacy-ring barrier; record rings "
            "use flush_records_and_wait");
    }
    // Everything published is handed to the P2P queue by the flush, so the
    // enqueued count read afterwards covers every task produced so far.
    // One deadline bounds both phases: a flush stuck behind saturated
    // staging (slow consumer) times out like the final delivery wait.
    if (!deadline) {
        drain_->force_flush_and_wait();
        drain_->rethrow_drain_failure();
        return p2p_->wait_until_processed(drain_->tasks_enqueued(), 0);
    }
    if (!drain_->force_flush_and_wait_until(*deadline)) return false;
    drain_->rethrow_drain_failure();
    return p2p_->wait_until_processed_until(drain_->tasks_enqueued(), *deadline);
}

RingEngine::RingEngine(const RingConfig& cfg,
                       std::shared_ptr<RecordSinkLease> lease,
                       RecordFailurePolicy failure_policy)
    : cfg_(cfg), ring_(cfg), record_sink_lease_(std::move(lease))
{
    if (cfg_.payload_ring_bytes % PAYLOAD_ALIGN != 0) {
        throw std::runtime_error(
            "RingConfig: payload_ring_bytes must be a multiple of "
            "PAYLOAD_ALIGN (wrap offset alignment)");
    }
    if (cfg_.drain_poll_timeout_us == 0) {
        throw std::runtime_error(
            "RingConfig: drain_poll_timeout_us must be > 0. "
            "The drain thread must poll periodically to process entries "
            "published mid-forward.");
    }

    auto* buf = ring_.state().payload_buf;
    if (reinterpret_cast<uintptr_t>(buf) % PAYLOAD_ALIGN != 0) {
        throw std::runtime_error(
            "RingEngine: payload_buf is not PAYLOAD_ALIGN-aligned "
            "(unexpected: cudaMalloc guarantees >= 256-byte alignment)");
    }

    staging_.init(cfg.effective_staging_bytes());
    drain_ = std::make_unique<DrainThread>(ring_.state(), staging_, cfg_);
    record_sink_ = record_sink_lease_
        ? record_sink_lease_->claim() : nullptr;
    try {
        record_p2p_ = std::make_unique<RecordP2PThread>(
            *drain_, record_sink_, failure_policy);
    } catch (...) {
        release_record_sink();
        throw;
    }
}

RingEngine::~RingEngine() noexcept {
    if (drain_) {
        drain_->stop();
        drain_->signal_p2p_stop();
    }
    if (p2p_) p2p_->stop();
    if (record_p2p_) record_p2p_->stop();
    release_record_sink();
}

void RingEngine::init(cudaStream_t stream) {
    ring_.init(stream);
}

void RingEngine::start() {
    if (record_mode() && record_sink_released_) {
        throw std::logic_error(
            "record RingEngine cannot restart after stop");
    }
    drain_->start();
    if (p2p_) p2p_->start();
    if (record_p2p_) record_p2p_->start();
}

void RingEngine::stop() {
    // Guard against double-stop (benchmark _timed_close + engine.close).
    if (!drain_->is_running()) {
        release_record_sink();
        return;
    }

    cudaDeviceSynchronize();
    drain_->force_flush_and_wait();

    drain_->stop();
    drain_->signal_p2p_stop();
    if (p2p_) p2p_->stop();
    if (record_p2p_) record_p2p_->stop();
    release_record_sink();
}

void RingEngine::release_record_sink() noexcept {
    if (record_sink_released_) return;
    if (record_sink_lease_) record_sink_lease_->release();
    record_sink_released_ = true;
}

RecordConsumer& RingEngine::record_consumer() {
    if (!record_p2p_) {
        throw std::logic_error("record consumer requested from legacy ring");
    }
    return record_p2p_->consumer();
}

const RecordConsumer& RingEngine::record_consumer() const {
    if (!record_p2p_) {
        throw std::logic_error("record consumer requested from legacy ring");
    }
    return record_p2p_->consumer();
}

}  // namespace ring
