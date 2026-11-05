// ring/capture_lease.h -- an integration-defined identity carried with a
// captured step through the legacy ring to its consumer.
//
// A consumer that changes what it captures over time (for example, which
// HookPoints or requests a step selects) attaches a lease to each step when it
// publishes the step's metadata.  The lease rides with the StepContext through
// the drain and P2P threads and arrives with every slice or batch the step
// produces, so the consumer can tell which configuration produced a payload,
// and its reference count tells the integration when every payload of a
// configuration has been delivered or dropped (idle()).
//
// A lease may also carry an immutable (act_name, layer) -> drop-rank map,
// shared by every step of one configuration; the batch queue uses it to order
// load shedding (drop_notice.h).
//
// Host metadata only: no Python objects and no GIL on the data path.  Copies
// share the configuration until the FIFO, the queues and the consumer release
// it.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ring_py {

class CaptureLease {
    using Ranks = std::map<std::pair<std::string, int32_t>, uint8_t>;
    struct Configuration {
        uint64_t id;
        std::shared_ptr<const Ranks> ranks;
    };
public:
    CaptureLease() = default;
    explicit CaptureLease(uint64_t id)
        : config_(std::make_shared<const Configuration>(Configuration{id, {}})) {}
    // entries: (act_name, layer_no, rank), ranks clamped to [0, 2].
    CaptureLease(uint64_t id,
                 const std::vector<std::tuple<std::string, int32_t, int>>& entries) {
        Ranks ranks;
        for (const auto& [hook, layer, rank] : entries)
            ranks[{hook, layer}] =
                static_cast<uint8_t>(rank < 0 ? 0 : rank > 2 ? 2 : rank);
        config_ = std::make_shared<const Configuration>(Configuration{
            id, std::make_shared<const Ranks>(std::move(ranks))});
    }
    // A new lease (own lifetime accounting) sharing `policy`'s immutable rank
    // map: no map allocation or copy per captured step.
    CaptureLease(uint64_t id, const CaptureLease& policy)
        : config_(std::make_shared<const Configuration>(Configuration{
              id, policy.config_ ? policy.config_->ranks : nullptr})) {}

    uint64_t id() const { return config_ ? config_->id : 0; }
    bool present() const { return bool(config_); }
    // No copy other than this one is alive: every payload of the step that
    // carried this lease has been consumed or dropped.
    bool idle() const { return config_ && config_.use_count() == 1; }
    bool same_as(const CaptureLease& other) const { return config_ == other.config_; }
    // Rank of (hook, layer) under this lease's map; `fallback` without a map,
    // the lowest priority (2) for a stream the map does not name.
    uint8_t rank_for(const std::string& hook, int32_t layer, uint8_t fallback) const {
        if (!config_ || !config_->ranks) return fallback;
        auto it = config_->ranks->find({hook, layer});
        return it == config_->ranks->end() ? 2 : it->second;
    }
    CaptureLease fork() const { return *this; }

private:
    std::shared_ptr<const Configuration> config_;
};

}  // namespace ring_py
