// Backend-neutral ownership and completion boundary for generic records.

#pragma once

#include "record_descriptor.h"

#include <ATen/ATen.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace ring {

// One descriptor paired with its owned, contiguous host payload.  submit()
// transfers both objects to the sink; the ring never accesses them again.
struct RecordEnvelope {
    RecordDescriptor descriptor;
    at::Tensor payload;
};

// Native sinks may enqueue work asynchronously, but they must expose one
// checked, non-closing durability barrier.  The exact durability boundary is
// a property of the concrete sink (for example, ClickHouse acknowledgement,
// local spool commit, or remote object-store commit).
//
// The ring invokes submit() from its single record p2p worker.  It invokes
// flush_and_wait() only after every earlier submit() has returned, so a sink
// owns any additional internal concurrency and backpressure policy.
class RecordSink {
public:
    using Duration = std::chrono::milliseconds;

    virtual ~RecordSink() = default;

    virtual void submit(RecordEnvelope envelope) = 0;
    virtual bool flush_and_wait(Duration timeout) = 0;
    virtual void rethrow_if_failed() const = 0;

    bool engine_owned() const {
        std::lock_guard<std::mutex> lock(engine_mu_);
        return engine_state_ != EngineState::AVAILABLE;
    }

protected:
    virtual void on_engine_acquire() {}
    virtual void on_engine_release() noexcept {}

private:
    friend class RecordSinkLease;

    enum class EngineState {
        AVAILABLE,
        ACQUIRING,
        OWNED,
        RELEASING,
    };

    void acquire_engine() {
        {
            std::lock_guard<std::mutex> lock(engine_mu_);
            if (engine_state_ != EngineState::AVAILABLE) {
                throw std::runtime_error(
                    "record sink is already owned by a RingEngine");
            }
            engine_state_ = EngineState::ACQUIRING;
        }
        try {
            on_engine_acquire();
        } catch (...) {
            std::lock_guard<std::mutex> lock(engine_mu_);
            engine_state_ = EngineState::AVAILABLE;
            throw;
        }
        std::lock_guard<std::mutex> lock(engine_mu_);
        engine_state_ = EngineState::OWNED;
    }

    void release_engine() noexcept {
        {
            std::lock_guard<std::mutex> lock(engine_mu_);
            if (engine_state_ != EngineState::OWNED) return;
            engine_state_ = EngineState::RELEASING;
        }
        on_engine_release();
        std::lock_guard<std::mutex> lock(engine_mu_);
        engine_state_ = EngineState::AVAILABLE;
    }

    mutable std::mutex engine_mu_;
    EngineState engine_state_{EngineState::AVAILABLE};
};

// Small RAII token used to reserve one sink before replacing an active ring.
// Acquisition performs no ring allocation.  Once claimed by a RingEngine, the
// lease is released only after its record worker has stopped.
class RecordSinkLease final {
public:
    static std::shared_ptr<RecordSinkLease> acquire(
        std::shared_ptr<RecordSink> sink) {
        if (!sink) return nullptr;
        auto lease = std::shared_ptr<RecordSinkLease>(
            new RecordSinkLease(std::move(sink)));
        lease->sink_->acquire_engine();
        lease->active_.store(true);
        return lease;
    }

    ~RecordSinkLease() noexcept { release(); }

    RecordSinkLease(const RecordSinkLease&) = delete;
    RecordSinkLease& operator=(const RecordSinkLease&) = delete;

    std::shared_ptr<RecordSink> claim() {
        if (claimed_.exchange(true)) {
            throw std::runtime_error(
                "record sink lease has already been claimed");
        }
        if (sink_ && !active_.load()) {
            throw std::runtime_error("record sink lease is no longer active");
        }
        return sink_;
    }

    void release() noexcept {
        if (active_.exchange(false)) sink_->release_engine();
    }

private:
    explicit RecordSinkLease(std::shared_ptr<RecordSink> sink)
        : sink_(std::move(sink)) {}

    std::shared_ptr<RecordSink> sink_;
    std::atomic<bool> active_{false};
    std::atomic<bool> claimed_{false};
};

}  // namespace ring
