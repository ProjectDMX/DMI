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

template <typename Fn> void expect_logic(Fn&& fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    EXPECT(rejected);
}

ring::D2HWindowAvailability available(uint64_t full, std::optional<uint64_t> first) {
    return ring::D2HWindowAvailability{full, first};
}

ring::D2HWindowGrantDecision require_decision(
    std::optional<ring::D2HWindowGrantDecision> decision) {
    EXPECT(decision.has_value());
    return decision.value_or(ring::D2HWindowGrantDecision{});
}

void observe(ring::BinaryAdaptiveGrantPolicy& policy, uint64_t occurrence,
             uint64_t bytes, bool overran,
             const ring::D2HWindowGrantDecision& decision) {
    policy.observe({occurrence, bytes, overran, decision});
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

void test_binary_grant_search_and_contradictions() {
    ring::BinaryAdaptiveGrantPolicy policy(3, 4);
    auto decision = require_decision(policy.choose(0, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::FULL_AVAILABLE);
    EXPECT(decision.byte_limit == std::numeric_limits<uint64_t>::max());

    observe(policy, 0, 100, true, decision);
    decision = require_decision(policy.choose(1, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::HALF_UNSAFE);
    EXPECT(decision.byte_limit == 50);

    observe(policy, 1, 50, false, decision);
    decision = require_decision(policy.choose(2, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MIDPOINT);
    EXPECT(decision.byte_limit == 75);

    observe(policy, 2, 75, true, decision);
    decision = require_decision(policy.choose(3, available(100, 10)));
    EXPECT(decision.byte_limit == 62);
    observe(policy, 3, 60, false, decision);
    decision = require_decision(policy.choose(4, available(100, 10)));
    EXPECT(decision.byte_limit == 67);

    observe(policy, 4, 40, true, decision);
    decision = require_decision(policy.choose(5, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::HALF_UNSAFE);
    EXPECT(decision.byte_limit == 20);

    ring::BinaryAdaptiveGrantPolicy upward(3, 4);
    decision = require_decision(upward.choose(0, available(100, 10)));
    observe(upward, 0, 100, true, decision);
    decision = require_decision(upward.choose(1, available(100, 10)));
    observe(upward, 1, 50, false, decision);
    upward.observe_timing_estimate(50, 120);
    decision = require_decision(upward.choose(2, available(120, 10)));
    EXPECT(decision.timing_revalidation);
    observe(upward, 2, 110, false, decision);
    upward.observe_timing_estimate(60, 60);
    decision = require_decision(upward.choose(3, available(120, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MAX_SAFE);
    EXPECT(decision.byte_limit == 110);
}

void test_record_granularity_and_timing_targets() {
    ring::BinaryAdaptiveGrantPolicy policy(3, 4);
    auto decision = require_decision(policy.choose(0, available(100, 10)));
    observe(policy, 0, 100, true, decision);
    decision = require_decision(policy.choose(1, available(100, 10)));
    observe(policy, 1, 50, false, decision);

    decision = require_decision(policy.choose(2, available(100, 10)));
    EXPECT(decision.byte_limit == 75);
    auto result = policy.observe({2, 50, false, decision});
    EXPECT(result.entered_record_granularity_stall);
    decision = require_decision(policy.choose(3, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MAX_SAFE);
    EXPECT(decision.byte_limit == 50);

    policy.observe_timing_estimate(50, 70);
    decision = require_decision(policy.choose(4, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::TIMING_ESTIMATE);
    EXPECT(decision.byte_limit == 70);
    observe(policy, 4, 50, false, decision);
    policy.observe_timing_estimate(50, 70);
    decision = require_decision(policy.choose(5, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MAX_SAFE);

    policy.observe_timing_estimate(50, 65);
    decision = require_decision(policy.choose(6, available(100, 10)));
    observe(policy, 6, 60, false, decision);
    decision = require_decision(policy.choose(7, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MIDPOINT);
    EXPECT(decision.byte_limit == 80);

    ring::BinaryAdaptiveGrantPolicy availability_limited(3, 4);
    decision = require_decision(availability_limited.choose(0, available(100, 10)));
    observe(availability_limited, 0, 100, true, decision);
    decision = require_decision(availability_limited.choose(1, available(50, 10)));
    observe(availability_limited, 1, 50, false, decision);
    decision = require_decision(availability_limited.choose(2, available(50, 10)));
    result = availability_limited.observe({2, 50, false, decision});
    EXPECT(!result.entered_record_granularity_stall);
    decision = require_decision(availability_limited.choose(3, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MIDPOINT);
    EXPECT(decision.byte_limit == 75);
}

void test_timing_revalidation_backoff() {
    ring::BinaryAdaptiveGrantPolicy policy(3, 4);
    auto decision = require_decision(policy.choose(0, available(100, 10)));
    observe(policy, 0, 100, true, decision);
    decision = require_decision(policy.choose(1, available(100, 10)));
    observe(policy, 1, 50, false, decision);
    policy.observe_timing_estimate(50, 120);

    decision = require_decision(policy.choose(2, available(120, 10)));
    EXPECT(decision.timing_revalidation);
    observe(policy, 2, 120, true, decision);
    decision = require_decision(policy.choose(3, available(120, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MIDPOINT);
    EXPECT(decision.byte_limit == 75);
    decision = require_decision(policy.choose(6, available(120, 10)));
    EXPECT(decision.timing_revalidation);
    observe(policy, 6, 120, true, decision);
    decision = require_decision(policy.choose(10, available(120, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MIDPOINT);
    decision = require_decision(policy.choose(14, available(120, 10)));
    EXPECT(decision.timing_revalidation);

    ring::BinaryAdaptiveGrantPolicy changed_bound(3, 4);
    decision = require_decision(changed_bound.choose(0, available(100, 10)));
    observe(changed_bound, 0, 100, true, decision);
    decision = require_decision(changed_bound.choose(1, available(100, 10)));
    observe(changed_bound, 1, 50, false, decision);
    changed_bound.observe_timing_estimate(50, 120);
    decision = require_decision(changed_bound.choose(2, available(120, 10)));
    observe(changed_bound, 2, 120, true, decision);
    decision = require_decision(changed_bound.choose(3, available(120, 10)));
    observe(changed_bound, 3, 70, true, decision);
    decision = require_decision(changed_bound.choose(4, available(120, 10)));
    EXPECT(decision.timing_revalidation);

    ring::BinaryAdaptiveGrantPolicy equal_estimate(3, 4);
    decision = require_decision(equal_estimate.choose(0, available(100, 10)));
    observe(equal_estimate, 0, 100, true, decision);
    decision = require_decision(equal_estimate.choose(1, available(100, 10)));
    observe(equal_estimate, 1, 50, false, decision);
    equal_estimate.observe_timing_estimate(50, 100);
    decision = require_decision(equal_estimate.choose(2, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::MIDPOINT);
    EXPECT(decision.byte_limit == 75);
}

void test_minimum_record_probes() {
    ring::BinaryAdaptiveGrantPolicy policy(3, 4);
    EXPECT(!policy.choose(0, available(0, std::nullopt)).has_value());

    auto decision = require_decision(policy.choose(0, available(100, 10)));
    observe(policy, 0, 100, true, decision);
    decision = require_decision(policy.choose(1, available(80, 80)));
    EXPECT(decision.minimum_record_probe());
    EXPECT(decision.byte_limit == 80);
    observe(policy, 1, 80, true, decision);
    EXPECT(!policy.choose(2, available(80, 80)).has_value());
    EXPECT(!policy.choose(3, available(80, 80)).has_value());

    decision = require_decision(policy.choose(4, available(80, 80)));
    EXPECT(decision.minimum_record_probe());
    observe(policy, 4, 80, true, decision);
    EXPECT(!policy.choose(9, available(80, 80)).has_value());

    decision = require_decision(policy.choose(5, available(30, 30)));
    EXPECT(!decision.minimum_record_probe());
    observe(policy, 5, 30, false, decision);
    EXPECT(!policy.choose(9, available(80, 80)).has_value());

    decision = require_decision(policy.choose(10, available(80, 80)));
    EXPECT(decision.minimum_record_probe());
    observe(policy, 10, 80, false, decision);
    decision = require_decision(policy.choose(11, available(120, 100)));
    EXPECT(decision.minimum_record_probe());

    ring::BinaryAdaptiveGrantPolicy saturating(5, 4);
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    decision = require_decision(saturating.choose(0, available(100, 10)));
    observe(saturating, 0, 100, true, decision);
    decision = require_decision(
        saturating.choose(maximum - 1, available(80, 80)));
    observe(saturating, maximum - 1, 80, true, decision);
    EXPECT(!saturating.choose(maximum - 1, available(80, 80)).has_value());
    EXPECT(saturating.choose(maximum, available(80, 80)).has_value());
}

void test_zero_byte_attempt_is_not_evidence() {
    ring::BinaryAdaptiveGrantPolicy policy(3, 4);
    auto decision = require_decision(policy.choose(0, available(0, 0)));
    observe(policy, 0, 0, false, decision);
    decision = require_decision(policy.choose(1, available(100, 10)));
    EXPECT(decision.kind == ring::D2HWindowGrantKind::FULL_AVAILABLE);
}

void test_runtime_modes() {
    expect_invalid([] { ring::D2HWindowModeController mode(0); });
    ring::D2HWindowModeController mode(2);
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_NO_PATTERN);
    EXPECT(!mode.window_scheduling_in_effect());
    EXPECT(!mode.record_capacity_forced_flush(false));
    mode.record_pattern_version_activation();
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_ACTIVE);
    EXPECT(mode.window_scheduling_in_effect());
    EXPECT(!mode.record_capacity_forced_flush(false));
    EXPECT(mode.record_capacity_forced_flush(false));
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_FALLBACK);
    EXPECT(!mode.record_capacity_forced_flush(false));
    mode.record_pattern_version_activation();
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_FALLBACK);

    ring::D2HWindowModeController resettable(3);
    resettable.record_pattern_version_activation();
    EXPECT(!resettable.record_capacity_forced_flush(false));
    EXPECT(!resettable.record_capacity_forced_flush(true));
    EXPECT(resettable.snapshot().capacity_forced_flush_count == 1);
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
    std::vector<std::pair<uint64_t, uint64_t>> timing_estimates;
};

class TrackingPolicy final : public ring::D2HWindowGrantPolicy {
  public:
    explicit TrackingPolicy(std::shared_ptr<PolicyLog> log) : log_(std::move(log)) {}

    std::optional<ring::D2HWindowGrantDecision>
    choose(uint64_t, ring::D2HWindowAvailability availability) const override {
        if (!availability.first_record_bytes.has_value())
            return std::nullopt;
        ring::D2HWindowGrantDecision decision;
        decision.byte_limit = availability.full_grant_bytes;
        decision.full_grant_bytes = availability.full_grant_bytes;
        return decision;
    }

    ring::D2HWindowPolicyObservation
    observe(ring::D2HWindowAttempt attempt) override {
        log_->attempts.push_back(attempt);
        return {};
    }

    void observe_timing_estimate(uint64_t measured_bytes,
                                 uint64_t estimate_bytes) override {
        log_->timing_estimates.emplace_back(measured_bytes, estimate_bytes);
    }

  private:
    std::shared_ptr<PolicyLog> log_;
};

struct ManualClock {
    void set_nanoseconds(int64_t nanoseconds) {
        now = ring::D2HWindowClock::time_point(
            std::chrono::nanoseconds(nanoseconds));
    }

    ring::D2HWindowClock::time_point now{};
};

void test_controller_timing_estimate_is_deterministic() {
    FakeProgress progress;
    ring::D2HWindowModeController mode(3);
    std::vector<std::shared_ptr<PolicyLog>> logs;
    auto factory = [&logs]() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        auto log = std::make_shared<PolicyLog>();
        logs.push_back(log);
        return std::make_unique<TrackingPolicy>(std::move(log));
    };
    auto clock = std::make_shared<ManualClock>();
    ring::RecurringD2HGrantController controller(
        progress, mode, factory, nullptr, [clock] { return clock->now; });
    controller.install_pending(1, 10, {{1, 4}});

    progress.snapshot = {1, 1};
    clock->set_nanoseconds(0);
    EXPECT(!controller.poll(available(0, std::nullopt)).has_value());

    clock->set_nanoseconds(1);
    auto admission = controller.poll(available(3, 3));
    EXPECT(admission.has_value());
    EXPECT(controller.commit(*admission, 3));

    clock->set_nanoseconds(3);
    progress.snapshot.counter = 2;
    controller.complete(*admission, 3);
    EXPECT(logs[0]->timing_estimates.empty());

    clock->set_nanoseconds(5);
    progress.snapshot.counter = 4;
    EXPECT(!controller.poll(available(0, std::nullopt)).has_value());
    EXPECT(logs[0]->timing_estimates.size() == 1);
    EXPECT(logs[0]->timing_estimates[0].first == 3);
    EXPECT(logs[0]->timing_estimates[0].second == 7);

    FakeProgress overflow_progress;
    ring::D2HWindowModeController overflow_mode(3);
    std::vector<std::shared_ptr<PolicyLog>> overflow_logs;
    auto overflow_factory =
        [&overflow_logs]() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        auto log = std::make_shared<PolicyLog>();
        overflow_logs.push_back(log);
        return std::make_unique<TrackingPolicy>(std::move(log));
    };
    auto overflow_clock = std::make_shared<ManualClock>();
    ring::RecurringD2HGrantController overflow_controller(
        overflow_progress, overflow_mode, overflow_factory, nullptr,
        [overflow_clock] { return overflow_clock->now; });
    overflow_controller.install_pending(1, 10, {{1, 4}});
    overflow_progress.snapshot = {1, 1};
    overflow_clock->set_nanoseconds(0);
    admission = overflow_controller.poll(available(
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max()));
    EXPECT(admission.has_value());
    EXPECT(overflow_controller.commit(
        *admission, std::numeric_limits<uint64_t>::max()));
    overflow_clock->set_nanoseconds(1);
    overflow_progress.snapshot.counter = 2;
    overflow_controller.complete(
        *admission, std::numeric_limits<uint64_t>::max());
    overflow_clock->set_nanoseconds(2);
    overflow_progress.snapshot.counter = 4;
    overflow_controller.poll(available(0, std::nullopt));
    EXPECT(overflow_logs[0]->timing_estimates.size() == 1);
    EXPECT(overflow_logs[0]->timing_estimates[0].second ==
           std::numeric_limits<uint64_t>::max());
}

void test_cross_window_overrun_does_not_supply_timing_evidence() {
    FakeProgress progress;
    ring::D2HWindowModeController mode(3);
    std::vector<std::shared_ptr<PolicyLog>> logs;
    auto factory = [&logs]() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        auto log = std::make_shared<PolicyLog>();
        logs.push_back(log);
        return std::make_unique<TrackingPolicy>(std::move(log));
    };
    auto clock = std::make_shared<ManualClock>();
    ring::RecurringD2HGrantController controller(
        progress, mode, factory, nullptr, [clock] { return clock->now; });
    controller.install_pending(1, 10, {{1, 3}, {5, 7}});

    progress.snapshot = {1, 1};
    clock->set_nanoseconds(0);
    auto first = controller.poll(available(10, 10));
    EXPECT(first.has_value());
    clock->set_nanoseconds(1);
    EXPECT(controller.commit(*first, 10));

    progress.snapshot.counter = 5;
    clock->set_nanoseconds(5);
    controller.complete(*first, 10);
    EXPECT(logs[0]->attempts.size() == 1);
    EXPECT(logs[0]->attempts[0].overran);
    EXPECT(logs[1]->attempts.empty());

    clock->set_nanoseconds(6);
    auto second = controller.poll(available(10, 10));
    EXPECT(second.has_value());
    clock->set_nanoseconds(7);
    EXPECT(controller.commit(*second, 10));
    progress.snapshot.counter = 6;
    clock->set_nanoseconds(8);
    controller.complete(*second, 10);
    progress.snapshot.counter = 7;
    clock->set_nanoseconds(10);
    controller.poll(available(0, std::nullopt));
    EXPECT(logs[1]->attempts.size() == 1);
    EXPECT(!logs[1]->attempts[0].overran);
    EXPECT(logs[1]->timing_estimates.empty());
}

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
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_NO_PATTERN);
    EXPECT(!controller.poll(available(64, 32)).has_value());

    progress.snapshot = {1, 1};
    auto admission = controller.poll(available(64, 32));
    EXPECT(mode.mode() == ring::D2HWindowMode::ENABLED_ACTIVE);
    EXPECT(logs.size() == 2);
    EXPECT(admission.has_value());
    EXPECT(admission->version == 1);
    EXPECT(admission->window.window_index == 0);
    EXPECT(controller.commit(*admission, 64));
    EXPECT(!controller.commit(*admission, 64));
    controller.complete(*admission, 64);
    EXPECT(logs[0]->attempts.size() == 1);
    EXPECT(!logs[0]->attempts[0].overran);
    EXPECT(!controller.poll(available(64, 32)).has_value());

    progress.snapshot = {1, 11};
    auto later = controller.poll(available(64, 32));
    EXPECT(later.has_value());
    progress.snapshot = {1, 14};
    EXPECT(!controller.commit(*later, 64));

    progress.snapshot = {1, 16};
    auto second_window = controller.poll(available(64, 32));
    EXPECT(second_window.has_value());
    EXPECT(controller.commit(*second_window, 64));
    controller.complete(*second_window, 64);
    EXPECT(logs[1]->attempts.size() == 1);

    progress.snapshot = {1, 21};
    auto rejected = controller.poll(available(64, 32));
    EXPECT(rejected.has_value());
    EXPECT(!controller.commit(*rejected, 65));
    EXPECT(logs[0]->attempts.size() == 1);
    EXPECT(controller.commit(*rejected, 64));
    controller.complete(*rejected, 64);
    EXPECT(logs[0]->attempts.size() == 2);

    controller.install_pending(2, 10, {{1, 4}});
    progress.snapshot = {2, 1};
    controller.poll(available(0, std::nullopt));
    EXPECT(logs.size() == 3);
    controller.complete(*second_window, 64);
    EXPECT(logs[1]->attempts.size() == 1);
    EXPECT(logs[2]->attempts.empty());
}

void test_capacity_forced_flush_count_aging_uses_pattern_period() {
    auto factory = []() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        return std::make_unique<TrackingPolicy>(std::make_shared<PolicyLog>());
    };

    FakeProgress progress;
    ring::D2HWindowModeController mode(2);
    ring::RecurringD2HGrantController controller(progress, mode, factory, nullptr);
    controller.install_pending(1, 10, {{1, 4}});
    progress.snapshot = {1, 5};
    controller.poll(available(0, std::nullopt));
    EXPECT(!controller.record_capacity_forced_flush(32));
    progress.snapshot.counter = 5 + 32 * 10;
    EXPECT(!controller.record_capacity_forced_flush(32));
    EXPECT(mode.snapshot().capacity_forced_flush_count == 1);
    ++progress.snapshot.counter;
    EXPECT(controller.record_capacity_forced_flush(32));

    FakeProgress within_progress;
    ring::D2HWindowModeController within_mode(2);
    ring::RecurringD2HGrantController within_controller(
        within_progress, within_mode, factory, nullptr);
    within_controller.install_pending(1, 10, {{1, 4}});
    within_progress.snapshot = {1, 5};
    within_controller.poll(available(0, std::nullopt));
    EXPECT(!within_controller.record_capacity_forced_flush(32));
    within_progress.snapshot.counter = 5 + 32 * 10 - 1;
    EXPECT(within_controller.record_capacity_forced_flush(32));
}

void test_pending_pattern_queue_promotes_versions_in_order() {
    FakeProgress progress;
    ring::D2HWindowModeController mode(3);
    std::vector<std::shared_ptr<PolicyLog>> logs;
    auto factory = [&logs]() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        auto log = std::make_shared<PolicyLog>();
        logs.push_back(log);
        return std::make_unique<TrackingPolicy>(std::move(log));
    };
    ring::RecurringD2HGrantController controller(progress, mode, factory, nullptr);

    controller.install_pending(1, 4, {{1, 3}});
    controller.install_pending(2, 6, {{1, 4}});
    controller.install_pending(3, 8, {{1, 5}});
    EXPECT(logs.size() == 3);

    progress.snapshot = {1, 1};
    auto admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 1);

    progress.snapshot = {2, 1};
    admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 2);

    progress.snapshot = {3, 1};
    admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 3);
}

void test_pending_pattern_queue_skips_obsolete_versions() {
    FakeProgress progress;
    ring::D2HWindowModeController mode(3);
    std::vector<std::shared_ptr<PolicyLog>> logs;
    auto factory = [&logs]() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        auto log = std::make_shared<PolicyLog>();
        logs.push_back(log);
        return std::make_unique<TrackingPolicy>(std::move(log));
    };
    ring::RecurringD2HGrantController controller(progress, mode, factory, nullptr);

    controller.install_pending(1, 4, {{1, 3}});
    controller.install_pending(2, 6, {{1, 4}});
    controller.install_pending(3, 8, {{1, 5}});

    progress.snapshot = {2, 1};
    auto admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 2);

    progress.snapshot = {3, 1};
    admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 3);

    progress.snapshot = {1, 1};
    EXPECT(!controller.poll(available(64, 32)).has_value());
}

void test_pending_pattern_queue_enforces_epochs_and_clears() {
    FakeProgress progress;
    ring::D2HWindowModeController mode(3);
    auto factory = []() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        return std::make_unique<TrackingPolicy>(std::make_shared<PolicyLog>());
    };
    ring::RecurringD2HGrantController controller(progress, mode, factory, nullptr);

    controller.install_pending(2, 4, {{1, 3}});
    expect_logic([&] { controller.install_pending(2, 4, {{1, 3}}); });
    expect_logic([&] { controller.install_pending(1, 4, {{1, 3}}); });

    progress.snapshot = {2, 1};
    controller.poll(available(0, std::nullopt));
    controller.install_pending(3, 4, {{1, 3}});
    controller.install_pending(4, 4, {{1, 3}});
    controller.cancel_pending_for_fallback();
    progress.snapshot = {3, 1};
    EXPECT(!controller.poll(available(64, 32)).has_value());

    controller.reset_for_version_reuse();
    controller.install_pending(1, 4, {{1, 3}});
    progress.snapshot = {1, 1};
    auto admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 1);
}

void test_pending_pattern_queue_cancels_only_the_requested_version() {
    FakeProgress progress;
    ring::D2HWindowModeController mode(3);
    auto factory = []() -> std::unique_ptr<ring::D2HWindowGrantPolicy> {
        return std::make_unique<TrackingPolicy>(std::make_shared<PolicyLog>());
    };
    ring::RecurringD2HGrantController controller(progress, mode, factory, nullptr);

    controller.install_pending(1, 4, {{1, 3}});
    controller.install_pending(2, 4, {{1, 3}});
    controller.install_pending(3, 4, {{1, 3}});
    controller.cancel_pending(2);

    progress.snapshot = {1, 1};
    auto admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 1);

    progress.snapshot = {2, 1};
    EXPECT(!controller.poll(available(64, 32)).has_value());

    progress.snapshot = {3, 1};
    admission = controller.poll(available(64, 32));
    EXPECT(admission.has_value() && admission->version == 3);
}

}  // namespace

int main() {
    test_packed_layout();
    test_pattern_matcher();
    test_binary_grant_search_and_contradictions();
    test_record_granularity_and_timing_targets();
    test_timing_revalidation_backoff();
    test_minimum_record_probes();
    test_zero_byte_attempt_is_not_evidence();
    test_runtime_modes();
    test_controller_timing_estimate_is_deterministic();
    test_cross_window_overrun_does_not_supply_timing_evidence();
    test_grant_controller();
    test_capacity_forced_flush_count_aging_uses_pattern_period();
    test_pending_pattern_queue_promotes_versions_in_order();
    test_pending_pattern_queue_skips_obsolete_versions();
    test_pending_pattern_queue_enforces_epochs_and_clears();
    test_pending_pattern_queue_cancels_only_the_requested_version();
    std::printf("Results: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
