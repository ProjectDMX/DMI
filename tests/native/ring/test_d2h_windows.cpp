#include "ring/d2h_window_grant_policy.h"
#include "ring/d2h_window_mode.h"
#include "ring/d2h_window_pattern.h"
#include "ring/d2h_window_progress_layout.h"
#include "ring/recurring_d2h_grant_controller.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

#define EXPECT(condition)                                                              \
    do {                                                                               \
        if (!(condition)) {                                                            \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);  \
            ++failed;                                                                  \
        } else {                                                                       \
            ++passed;                                                                  \
        }                                                                              \
    } while (0)

template <typename Fn> void expect_invalid(Fn&& fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    EXPECT(rejected);
}

ring::D2HWindowAvailability available(uint64_t full, std::optional<uint64_t> first) {
    return ring::D2HWindowAvailability{full, first};
}

void test_packed_layout() {
    using Layout = ring::D2HWindowPackedProgressLayout;
    const Layout::Version version = 431;
    const Layout::Counter counter = 123456789;
    const auto packed = Layout::pack(version, counter);
    EXPECT(Layout::version(packed) == version);
    EXPECT(Layout::counter(packed) == counter);
    EXPECT(Layout::version(0) == Layout::kNoPatternVersion);
    EXPECT(Layout::counter(0) == 0);
}

void test_pattern_matcher() {
    ring::D2HWindowPatternMatcher matcher(10, {{1, 3}, {6, 9}});
    EXPECT(!matcher.match(0).has_value());
    const auto first = matcher.match(1);
    EXPECT(first.has_value());
    EXPECT(first->window_index == 0);
    EXPECT(first->occurrence == 0);
    EXPECT(first->absolute_begin == 1);
    EXPECT(first->absolute_end == 3);
    EXPECT(matcher.match(2).has_value());
    EXPECT(!matcher.match(3).has_value());
    const auto repeated = matcher.match(17);
    EXPECT(repeated.has_value());
    EXPECT(repeated->window_index == 1);
    EXPECT(repeated->occurrence == 1);
    EXPECT(repeated->absolute_begin == 16);
    EXPECT(repeated->absolute_end == 19);
    EXPECT(!matcher.match(19).has_value());

    expect_invalid([] { ring::D2HWindowPatternMatcher(0, {{0, 1}}); });
    expect_invalid([] { ring::D2HWindowPatternMatcher(4, {}); });
    expect_invalid([] { ring::D2HWindowPatternMatcher(4, {{1, 1}}); });
    expect_invalid([] { ring::D2HWindowPatternMatcher(4, {{1, 5}}); });
    expect_invalid([] { ring::D2HWindowPatternMatcher(8, {{4, 6}, {2, 3}}); });
    expect_invalid([] { ring::D2HWindowPatternMatcher(8, {{1, 5}, {4, 7}}); });
}

void test_grant_targets() {
    ring::LastKAdaptiveGrantPolicy policy(3, 2);
    auto decision = policy.choose(0, available(100, 10));
    EXPECT(decision.has_value() && decision->byte_limit == 100);
    EXPECT(!decision->minimum_record_probe);

    policy.observe({0, 100, false, false});
    decision = policy.choose(1, available(160, 10));
    EXPECT(decision.has_value() && decision->byte_limit == 160);

    ring::LastKAdaptiveGrantPolicy all_failed(3, 2);
    all_failed.observe({0, 100, true, false});
    decision = all_failed.choose(1, available(100, 10));
    EXPECT(decision.has_value() && decision->byte_limit == 50);
    all_failed.observe({1, 50, true, false});
    decision = all_failed.choose(2, available(100, 10));
    EXPECT(decision.has_value() && decision->byte_limit == 25);

    ring::LastKAdaptiveGrantPolicy well_formed(3, 2);
    well_formed.observe({0, 100, true, false});
    well_formed.observe({1, 50, false, false});
    decision = well_formed.choose(2, available(100, 10));
    EXPECT(decision.has_value() && decision->byte_limit == 75);

    ring::LastKAdaptiveGrantPolicy not_well_formed(3, 2);
    not_well_formed.observe({0, 80, false, false});
    not_well_formed.observe({1, 60, true, false});
    decision = not_well_formed.choose(2, available(100, 10));
    EXPECT(decision.has_value() && decision->byte_limit == 30);

    ring::LastKAdaptiveGrantPolicy evicted(2, 2);
    evicted.observe({0, 100, true, false});
    evicted.observe({1, 50, false, false});
    evicted.observe({2, 75, false, false});
    EXPECT(evicted.history_size() == 2);
    decision = evicted.choose(3, available(140, 10));
    EXPECT(decision.has_value() && decision->byte_limit == 140);
}

void test_minimum_record_probes() {
    ring::LastKAdaptiveGrantPolicy policy(2, 3);
    EXPECT(!policy.choose(0, available(0, std::nullopt)).has_value());

    policy.observe({0, 100, true, false});
    auto decision = policy.choose(1, available(80, 80));
    EXPECT(decision.has_value());
    EXPECT(decision->minimum_record_probe);
    EXPECT(decision->byte_limit == 80);
    policy.observe({1, 80, true, true});
    decision = policy.choose(2, available(80, 80));
    EXPECT(decision.has_value() && decision->minimum_record_probe);
    policy.observe({2, 80, true, true});
    EXPECT(policy.next_probe_occurrence() == 5);
    EXPECT(!policy.choose(4, available(80, 80)).has_value());
    decision = policy.choose(5, available(80, 80));
    EXPECT(decision.has_value() && decision->minimum_record_probe);
    policy.observe({5, 80, true, true});
    EXPECT(policy.next_probe_occurrence() == 8);

    decision = policy.choose(6, available(30, 30));
    EXPECT(decision.has_value() && !decision->minimum_record_probe);
    policy.observe({6, 30, false, false});
    EXPECT(!policy.next_probe_occurrence().has_value());

    ring::LastKAdaptiveGrantPolicy successful_probe(2, 3);
    successful_probe.observe({0, 100, true, false});
    successful_probe.observe({1, 80, false, true});
    EXPECT(successful_probe.history_size() == 1);
    EXPECT(!successful_probe.next_probe_occurrence().has_value());
    decision = successful_probe.choose(2, available(120, 80));
    EXPECT(decision.has_value() && decision->byte_limit == 120);

    ring::LastKAdaptiveGrantPolicy saturating(2, 5);
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    saturating.observe({maximum - 1, 80, true, true});
    saturating.observe({maximum - 1, 80, true, true});
    EXPECT(saturating.next_probe_occurrence() == maximum);
}

void test_runtime_modes() {
    expect_invalid([] { ring::D2HWindowModeController mode(0); });
    ring::D2HWindowModeController mode(2);
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_NO_PATTERN);
    EXPECT(!mode.window_scheduling_in_effect());
    EXPECT(!mode.record_capacity_forced_flush());
    mode.record_pattern_version_activation();
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_ACTIVE);
    EXPECT(mode.window_scheduling_in_effect());
    EXPECT(!mode.record_capacity_forced_flush());
    EXPECT(mode.record_capacity_forced_flush());
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_FALLBACK);
    EXPECT(!mode.record_capacity_forced_flush());
    mode.record_pattern_version_activation();
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_FALLBACK);

    ring::D2HWindowModeController resettable(3);
    resettable.record_pattern_version_activation();
    EXPECT(!resettable.record_capacity_forced_flush());
    resettable.reset_for_version_reuse();
    const auto snapshot = resettable.snapshot();
    EXPECT(snapshot.mode == ring::D2HWindowMode::ENABLED_NO_PATTERN);
    EXPECT(snapshot.capacity_forced_flush_count == 0);
    EXPECT(snapshot.capacity_flush_fallback_threshold == 3);
}

class FakeProgress final : public ring::D2HWindowProgressSource {
  public:
    ring::D2HWindowProgressSnapshot load() const noexcept override { return snapshot; }

    ring::D2HWindowProgressState state() const override { return {}; }

    void enqueue_reset(ring::D2HWindowPackedProgressLayout::Version version,
                       ring::D2HWindowPackedProgressLayout::Counter counter,
                       cudaStream_t) override {
        snapshot = {version, counter};
    }

    ring::D2HWindowProgressSnapshot snapshot{};
};

struct PolicyLog {
    std::vector<ring::D2HWindowAttempt> attempts;
};

class TrackingPolicy final : public ring::D2HWindowGrantPolicy {
  public:
    explicit TrackingPolicy(std::shared_ptr<PolicyLog> log) : log_(std::move(log)) {}

    std::optional<ring::D2HWindowGrantDecision>
    choose(uint64_t, ring::D2HWindowAvailability availability) const override {
        if (!availability.first_record_bytes.has_value())
            return std::nullopt;
        return ring::D2HWindowGrantDecision{availability.full_grant_bytes, false};
    }

    void observe(ring::D2HWindowAttempt attempt) override {
        log_->attempts.push_back(attempt);
    }

  private:
    std::shared_ptr<PolicyLog> log_;
};

void test_grant_controller() {
    FakeProgress progress;
    ring::D2HWindowModeController mode(3);
    std::vector<std::shared_ptr<PolicyLog>> logs;
    auto factory = [&logs]() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        auto log = std::make_shared<PolicyLog>();
        logs.push_back(log);
        return std::make_unique<TrackingPolicy>(std::move(log));
    };
    ring::RecurringD2HGrantController controller(progress, mode, factory, nullptr);

    controller.install_pending(1, 10, {{1, 4}, {6, 9}});
    controller.reconcile_progress();
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_NO_PATTERN);
    EXPECT(!controller.consider(available(64, 32)).has_value());

    progress.snapshot = {1, 1};
    controller.reconcile_progress();
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_ACTIVE);
    EXPECT(logs.size() == 2);
    auto admission = controller.consider(available(64, 32));
    EXPECT(admission.has_value());
    EXPECT(admission->version == 1);
    EXPECT(admission->window.window_index == 0);
    EXPECT(controller.commit(*admission, 64));
    EXPECT(!controller.commit(*admission, 64));
    controller.complete(*admission, 64);
    EXPECT(logs[0]->attempts.size() == 1);
    EXPECT(!logs[0]->attempts[0].overran);
    EXPECT(!controller.consider(available(64, 32)).has_value());

    progress.snapshot = {1, 11};
    controller.reconcile_progress();
    auto later = controller.consider(available(64, 32));
    EXPECT(later.has_value());
    progress.snapshot = {1, 14};
    EXPECT(!controller.commit(*later, 64));

    progress.snapshot = {1, 16};
    controller.reconcile_progress();
    auto second_window = controller.consider(available(64, 32));
    EXPECT(second_window.has_value());
    EXPECT(controller.commit(*second_window, 64));
    controller.complete(*second_window, 64);
    EXPECT(logs[1]->attempts.size() == 1);

    controller.install_pending(2, 10, {{1, 4}});
    progress.snapshot = {2, 1};
    controller.reconcile_progress();
    EXPECT(logs.size() == 3);
    controller.complete(*second_window, 64);
    EXPECT(logs[1]->attempts.size() == 1);
    EXPECT(logs[2]->attempts.empty());
}

}  // namespace

int main() {
    test_packed_layout();
    test_pattern_matcher();
    test_grant_targets();
    test_minimum_record_probes();
    test_runtime_modes();
    test_grant_controller();
    std::printf("Results: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
