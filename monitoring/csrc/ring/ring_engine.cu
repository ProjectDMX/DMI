// ring/ring_engine.cu — RingEngine implementation (compiled with nvcc because
// it owns AllocatedRing which calls task_ring_init / cudaMemsetAsync).

#include "ring_engine.h"

namespace ring {

RingEngine::~RingEngine() noexcept {
    // Stop drain thread first: it calls assembler_ → cb_ → cb_queue_.
    // Must happen before cb_mu_/cb_queue_/cb_cv_ member destructors run
    // (drain_ is declared before them so its destructor runs after them,
    //  but we must stop the thread while they are still alive).
    if (drain_) drain_->stop();
    // Now stop callback thread: no more items will be pushed to cb_queue_.
    if (cb_thread_.joinable()) {
        cb_running_.store(false);
        cb_cv_.notify_all();
        cb_thread_.join();
    }
}

RingEngine::RingEngine(const RingConfig& cfg, TensorCallback cb)
    : cfg_(cfg), ring_(cfg), cb_(std::move(cb))
{
    pool_.init(cfg.pinned_pool_bytes);

    // Push assembled tensors to the callback queue instead of calling cb_ directly.
    // This decouples the Python on_tensor callback (which needs the GIL) from the
    // drain thread, preventing GIL stalls from blocking task-slot freeing.
    assembler_ = std::make_unique<ChunkAssembler>(pool_,
        [this](AssembledTensor&& t) {
            {
                std::lock_guard<std::mutex> lk(cb_mu_);
                cb_queue_.push(std::move(t));
            }
            cb_cv_.notify_one();
        });

    drain_ = std::make_unique<DrainThread>(
        ring_.state(), pool_,
        [this](DrainedChunk&& c) { assembler_->push(std::move(c)); });
}

void RingEngine::init(cudaStream_t stream) {
    ring_.init(stream);
}

void RingEngine::start() {
    cb_running_.store(true);
    cb_thread_ = std::thread([this] { cb_loop(); });
    drain_->start();
}

void RingEngine::stop() {
    drain_->stop();  // blocks until drain thread done + all assembler calls queued
    // Signal callback thread to stop after draining remaining items
    cb_running_.store(false);
    cb_cv_.notify_all();
    if (cb_thread_.joinable()) cb_thread_.join();
}

void RingEngine::cb_loop() {
    while (true) {
        AssembledTensor t;
        {
            std::unique_lock<std::mutex> lk(cb_mu_);
            cb_cv_.wait(lk, [this] {
                return !cb_queue_.empty() || !cb_running_.load();
            });
            if (cb_queue_.empty()) break;  // stopped and queue drained
            t = std::move(cb_queue_.front());
            cb_queue_.pop();
        }
        try {
            cb_(std::move(t));  // may block for GIL — drain loop is unaffected
        } catch (...) {
            // swallow — cannot propagate from background thread
        }
    }
}

}  // namespace ring
