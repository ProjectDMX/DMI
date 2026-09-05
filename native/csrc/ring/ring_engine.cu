// ring/ring_engine.cu -- RingEngine implementation (compiled with nvcc because
// it owns AllocatedRing which calls task_ring_init / cudaMemsetAsync).

#include "ring_engine.h"

#include <stdexcept>

namespace ring {

namespace {

RingConfig validate_legacy_config(const RingConfig& config) {
    if (config.recurring_d2h_windows.enabled) {
        throw std::invalid_argument(
            "recurring D2H windows require the HookPointV1 record engine");
    }
    return config;
}

int current_device() {
    int device = -1;
    const cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("RingEngine: cudaGetDevice failed: ") +
            cudaGetErrorString(error));
    }
    return device;
}

}  // namespace

RingEngine::RingEngine(const RingConfig& cfg, ring_py::TensorMetaFifo& fifo,
                       SubmitFn submit_fn)
    : cfg_(validate_legacy_config(cfg)), ring_(cfg_)
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

RingEngine::RingEngine(const RingConfig& cfg,
                       std::shared_ptr<RecordSinkLease> lease)
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
    recurring_d2h_windows_ = make_recurring_d2h_subsystem(
        cfg_.recurring_d2h_windows, current_device());
    drain_ = std::make_unique<DrainThread>(
        ring_.state(),
        staging_,
        cfg_,
        recurring_d2h_windows_
            ? &recurring_d2h_windows_->grant_controller() : nullptr,
        recurring_d2h_windows_
            ? &recurring_d2h_windows_->mode_controller() : nullptr,
        recurring_d2h_windows_
            ? std::function<void()>([this] {
                  recurring_d2h_windows_->record_capacity_forced_flush();
              })
            : std::function<void()>());
    record_sink_ = record_sink_lease_
        ? record_sink_lease_->claim() : nullptr;
    try {
        record_p2p_ = std::make_unique<RecordP2PThread>(
            *drain_, record_sink_);
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

void RingEngine::define_d2h_window_pattern(
    uint64_t period,
    std::vector<D2HWindowOffset> windows,
    std::optional<uint64_t> initial_counter,
    cudaStream_t framework_stream) {
    if (!recurring_d2h_windows_) {
        throw std::logic_error("recurring D2H windows are not enabled");
    }
    recurring_d2h_windows_->define_pattern(
        period,
        std::move(windows),
        initial_counter,
        framework_stream,
        *drain_);
}

D2HWindowProgressState RingEngine::d2h_window_progress_state() const {
    if (!recurring_d2h_windows_) {
        throw std::logic_error("recurring D2H windows are not enabled");
    }
    return recurring_d2h_windows_->progress_state();
}

D2HWindowRuntimeSnapshot RingEngine::d2h_window_runtime_snapshot() const noexcept {
    if (!recurring_d2h_windows_) {
        return D2HWindowRuntimeSnapshot{};
    }
    return recurring_d2h_windows_->snapshot();
}

}  // namespace ring
