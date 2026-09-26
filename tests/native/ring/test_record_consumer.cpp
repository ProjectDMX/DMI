#include "ring/record_consumer.h"
#include "ring/record_stall_budget.h"

#include <ATen/ATen.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(condition)                                                   \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                    \
                         __FILE__, __LINE__, #condition);                    \
            ++g_fail;                                                       \
        } else {                                                            \
            ++g_pass;                                                       \
        }                                                                   \
    } while (0)

class CapturingSink final : public ring::RecordSink {
public:
    void submit(ring::RecordEnvelope envelope) override {
        submitted.push_back(std::move(envelope));
    }

    bool flush_and_wait(Duration timeout) override {
        last_timeout = timeout;
        ++flushes;
        return true;
    }

    void rethrow_if_failed() const override {}

    std::vector<ring::RecordEnvelope> submitted;
    Duration last_timeout{0};
    int flushes{0};
};

class FailingSink final : public ring::RecordSink {
public:
    void submit(ring::RecordEnvelope) override {
        throw std::runtime_error("injected sink failure");
    }

    bool flush_and_wait(Duration) override { return true; }
    void rethrow_if_failed() const override {}
};

static at::Tensor byte_payload(const std::vector<float>& values) {
    return at::tensor(values, at::TensorOptions().dtype(at::kFloat))
        .view(at::kByte)
        .clone();
}

static ring::RecordDescriptor descriptor(std::string layout,
                                         std::string literal) {
    ring::RecordDescriptor result;
    result.layout = std::move(layout);
    result.rows = {{std::vector<ring::EncodedRecordCell>{
        std::move(literal), ring::PayloadSlice{}}}};
    return result;
}

static void test_fifo_delivers_backend_neutral_envelopes() {
    std::printf("[ TEST ] FIFO delivers raw descriptors and owned payloads\n");
    auto sink = std::make_shared<CapturingSink>();
    ring::RecordConsumer consumer(sink);

    consumer.push_descriptors({
        descriptor("layout_a", "first"),
        descriptor("layout_b", "second"),
    });
    consumer.consume_payload(byte_payload({1, 2}));
    consumer.consume_payload(byte_payload({3, 4, 5}));
    consumer.finish();

    EXPECT(sink->submitted.size() == 2);
    EXPECT(sink->submitted[0].descriptor.layout == "layout_a");
    EXPECT(sink->submitted[1].descriptor.layout == "layout_b");
    EXPECT(std::get<std::string>(
               sink->submitted[0].descriptor.rows[0].cells[0]) == "first");
    EXPECT(at::equal(
        sink->submitted[0].payload.view(at::kFloat),
        at::tensor({1.f, 2.f})));
    EXPECT(at::equal(
        sink->submitted[1].payload.view(at::kFloat),
        at::tensor({3.f, 4.f, 5.f})));
}

static void test_zero_row_descriptor_consumes_without_sink_submission() {
    std::printf("[ TEST ] zero-row descriptor consumes its zero-byte task\n");
    auto sink = std::make_shared<CapturingSink>();
    ring::RecordConsumer consumer(sink);

    ring::RecordDescriptor empty;
    empty.layout = "filtered";
    consumer.push_descriptor(std::move(empty));
    consumer.consume_payload(at::empty(
        {0}, at::TensorOptions().dtype(at::kByte).device(at::kCPU)));
    consumer.finish();

    EXPECT(sink->submitted.empty());
}

static void test_exact_association_failures() {
    std::printf("[ TEST ] descriptor/payload association failures\n");
    auto sink = std::make_shared<CapturingSink>();
    ring::RecordConsumer missing(sink);
    bool missing_failed = false;
    try {
        missing.consume_payload(byte_payload({1}));
    } catch (const std::runtime_error&) {
        missing_failed = true;
    }
    EXPECT(missing_failed);

    ring::RecordConsumer leftover(sink);
    ring::RecordDescriptor pending;
    pending.layout = "leftover";
    leftover.push_descriptor(std::move(pending));
    bool leftover_failed = false;
    try {
        leftover.finish();
    } catch (const std::runtime_error&) {
        leftover_failed = true;
    }
    EXPECT(leftover_failed);

    ring::RecordConsumer unconfigured(nullptr);
    unconfigured.push_descriptor(descriptor("events", "record"));
    bool missing_sink_failed = false;
    try {
        unconfigured.consume_payload(byte_payload({1}));
    } catch (const std::runtime_error&) {
        missing_sink_failed = true;
    }
    EXPECT(missing_sink_failed);
}

static void test_submit_failure_precedes_durable_idle() {
    std::printf("[ TEST ] submit failure precedes durable idle\n");
    ring::RecordConsumer consumer(std::make_shared<FailingSink>());
    consumer.push_descriptor(descriptor("submit_failure", "record"));

    constexpr int kWaiters = 64;
    std::atomic<int> ready{0};
    std::atomic<int> observed_failures{0};
    std::atomic<int> false_successes{0};
    std::atomic<int> timeouts{0};
    std::vector<std::thread> waiters;
    waiters.reserve(kWaiters);
    for (int index = 0; index < kWaiters; ++index) {
        waiters.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_release);
            try {
                if (!consumer.wait_until_idle(std::chrono::seconds(5))) {
                    timeouts.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                consumer.finish();
                false_successes.fetch_add(1, std::memory_order_relaxed);
            } catch (const std::runtime_error&) {
                observed_failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kWaiters) {
        std::this_thread::yield();
    }

    bool consume_failed = false;
    try {
        consumer.consume_payload(byte_payload({1}));
    } catch (const std::runtime_error&) {
        consume_failed = true;
    }
    for (auto& waiter : waiters) waiter.join();

    EXPECT(consume_failed);
    EXPECT(observed_failures.load(std::memory_order_relaxed) == kWaiters);
    EXPECT(false_successes.load(std::memory_order_relaxed) == 0);
    EXPECT(timeouts.load(std::memory_order_relaxed) == 0);
}

template <typename Fn>
static bool throws_runtime_error(Fn&& call) {
    try {
        call();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static void test_raise_policy_is_the_default_and_raises_at_the_producer() {
    std::printf("[ TEST ] raise_at_producer is the default and raises on push\n");
    ring::RecordConsumer consumer(std::make_shared<FailingSink>());
    EXPECT(consumer.snapshot().policy ==
           ring::RecordFailurePolicy::kRaiseAtProducer);
    consumer.push_descriptor(descriptor("raise", "record"));
    EXPECT(throws_runtime_error(
        [&] { consumer.consume_payload(byte_payload({1})); }));

    EXPECT(throws_runtime_error(
        [&] { consumer.push_descriptor(descriptor("raise", "later")); }));
    const ring::RecordConsumerSnapshot snapshot = consumer.snapshot();
    EXPECT(snapshot.failed);
    EXPECT(snapshot.failure.find("injected sink failure") != std::string::npos);
    EXPECT(snapshot.discarded_descriptors == 0);
    EXPECT(snapshot.discarded_payloads == 0);
}

static void test_disable_capture_discards_after_a_latch_and_still_fails_flush() {
    std::printf("[ TEST ] disable_capture discards after a latch; flush still fails\n");
    ring::RecordConsumer consumer(
        std::make_shared<FailingSink>(),
        ring::RecordFailurePolicy::kDisableCapture);
    // Three descriptors queued ahead of their payloads, as the forward
    // publishes them before the drain delivers.
    consumer.push_descriptors({
        descriptor("disable", "first"),
        descriptor("disable", "second"),
        descriptor("disable", "third"),
    });

    // The sink refuses the first: the consumer latches without throwing at
    // the p2p worker, and the two descriptors still queued can never be
    // stored, so they are dropped with it.
    EXPECT(!throws_runtime_error(
        [&] { consumer.consume_payload(byte_payload({1})); }));
    // The forward keeps publishing and the drain keeps delivering: neither
    // raises, and every one is counted.
    EXPECT(!throws_runtime_error(
        [&] { consumer.push_descriptor(descriptor("disable", "after")); }));
    EXPECT(!throws_runtime_error([&] {
        consumer.push_descriptors({descriptor("disable", "after-1"),
                                   descriptor("disable", "after-2")});
    }));
    for (int index = 0; index < 4; ++index) {
        EXPECT(!throws_runtime_error(
            [&] { consumer.consume_payload(byte_payload({2})); }));
    }

    const ring::RecordConsumerSnapshot snapshot = consumer.snapshot();
    EXPECT(snapshot.policy == ring::RecordFailurePolicy::kDisableCapture);
    EXPECT(snapshot.failed);
    EXPECT(snapshot.failure.find("injected sink failure") != std::string::npos);
    EXPECT(snapshot.discarded_descriptors == 5);
    // The refused payload plus the four delivered after the latch.
    EXPECT(snapshot.discarded_payloads == 5);
    EXPECT(consumer.pending_descriptors() == 0);

    // The failure still surfaces at every checked completion.
    EXPECT(throws_runtime_error([&] { consumer.rethrow_if_failed(); }));
    EXPECT(throws_runtime_error(
        [&] { consumer.wait_until_idle(std::chrono::milliseconds(10)); }));
    EXPECT(throws_runtime_error([&] { consumer.finish(); }));
}

static void test_disable_capture_latches_association_and_worker_failures() {
    std::printf("[ TEST ] disable_capture latches association and worker failures\n");
    auto sink = std::make_shared<CapturingSink>();
    ring::RecordConsumer orphan(
        sink, ring::RecordFailurePolicy::kDisableCapture);
    EXPECT(!throws_runtime_error(
        [&] { orphan.consume_payload(byte_payload({1})); }));
    EXPECT(orphan.snapshot().failed);
    EXPECT(orphan.snapshot().failure.find("without an encoded descriptor") !=
           std::string::npos);
    EXPECT(orphan.snapshot().discarded_payloads == 1);
    EXPECT(throws_runtime_error([&] { orphan.finish(); }));

    ring::RecordConsumer worker(
        sink, ring::RecordFailurePolicy::kDisableCapture);
    worker.push_descriptor(descriptor("worker", "queued"));
    worker.record_failure(std::make_exception_ptr(
        std::runtime_error("injected worker failure")));
    worker.push_descriptor(descriptor("worker", "after"));
    EXPECT(!throws_runtime_error(
        [&] { worker.consume_payload(byte_payload({1})); }));
    const ring::RecordConsumerSnapshot snapshot = worker.snapshot();
    EXPECT(snapshot.failure == "injected worker failure");
    EXPECT(snapshot.discarded_descriptors == 2);
    EXPECT(snapshot.discarded_payloads == 1);
    EXPECT(sink->submitted.empty());
    EXPECT(throws_runtime_error([&] { worker.rethrow_if_failed(); }));
}

// A step whose stall budget ran out: the records still queued for the sink
// and the rest of the step's records are skipped, and the next step is
// stored again.  Payloads keep pairing with their own descriptors across
// the window, under either policy, and nothing latches.
static void test_a_discard_window_skips_one_steps_records(
        ring::RecordFailurePolicy policy) {
    std::printf("[ TEST ] a discard window skips one step's records (%s)\n",
                policy == ring::RecordFailurePolicy::kDisableCapture
                    ? "disable_capture" : "raise");
    auto sink = std::make_shared<CapturingSink>();
    ring::RecordConsumer consumer(sink, policy);

    consumer.push_descriptors({descriptor("step", "queued-1"),
                               descriptor("step", "queued-2")});
    consumer.begin_discard_window();
    EXPECT(!throws_runtime_error([&] {
        consumer.push_descriptors({descriptor("step", "rest-1"),
                                   descriptor("step", "rest-2")});
    }));
    consumer.end_discard_window();
    consumer.push_descriptor(descriptor("step", "next-step"));

    // Nothing is idle until every skipped payload has arrived.
    EXPECT(!consumer.wait_until_idle(std::chrono::milliseconds(5)));
    EXPECT(throws_runtime_error([&] { consumer.finish(); }));
    for (float value = 1; value <= 5; ++value) {
        EXPECT(!throws_runtime_error(
            [&] { consumer.consume_payload(byte_payload({value})); }));
    }

    EXPECT(sink->submitted.size() == 1);
    if (sink->submitted.size() == 1) {
        EXPECT(std::get<std::string>(
                   sink->submitted[0].descriptor.rows[0].cells[0]) ==
               "next-step");
        EXPECT(at::equal(sink->submitted[0].payload.view(at::kFloat),
                         at::tensor({5.f})));
    }
    const ring::RecordConsumerSnapshot snapshot = consumer.snapshot();
    EXPECT(!snapshot.failed);
    EXPECT(snapshot.discarded_descriptors == 4);
    EXPECT(snapshot.discarded_payloads == 4);
    EXPECT(consumer.wait_until_idle(std::chrono::milliseconds(5)));
    EXPECT(!throws_runtime_error([&] { consumer.finish(); }));

    // An empty window skips nothing.
    consumer.begin_discard_window();
    consumer.end_discard_window();
    consumer.push_descriptor(descriptor("step", "after-empty"));
    consumer.consume_payload(byte_payload({6}));
    EXPECT(sink->submitted.size() == 2);
}

// A window that dropped every queued descriptor leaves the queue empty while
// the drain still owes the dropped descriptors' payloads.  A flush must wait
// for them: idle too early, and the next descriptor pushed after the flush
// would be paired with a skipped step's payload.
static void test_owed_payloads_hold_the_flush(
        ring::RecordFailurePolicy policy) {
    std::printf("[ TEST ] payloads owed to a discard window hold the flush (%s)\n",
                policy == ring::RecordFailurePolicy::kDisableCapture
                    ? "disable_capture" : "raise");
    auto sink = std::make_shared<CapturingSink>();
    ring::RecordConsumer consumer(sink, policy);

    consumer.push_descriptors({descriptor("step", "queued-1"),
                               descriptor("step", "queued-2")});
    consumer.begin_discard_window();
    consumer.end_discard_window();
    EXPECT(consumer.pending_descriptors() == 0);

    // Only owed payloads remain: not idle, and finish() names them.
    EXPECT(!consumer.wait_until_idle(std::chrono::milliseconds(5)));
    std::string leftover;
    try {
        consumer.finish();
    } catch (const std::runtime_error& error) {
        leftover = error.what();
    }
    EXPECT(leftover.find("owed") != std::string::npos);

    consumer.consume_payload(byte_payload({1}));
    EXPECT(!consumer.wait_until_idle(std::chrono::milliseconds(5)));
    EXPECT(throws_runtime_error([&] { consumer.finish(); }));

    // A waiter already blocked wakes when the last owed payload arrives.
    std::atomic<bool> idle{false};
    std::thread waiter([&] {
        idle.store(consumer.wait_until_idle(std::chrono::seconds(5)),
                   std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    consumer.consume_payload(byte_payload({2}));
    waiter.join();
    EXPECT(idle.load(std::memory_order_acquire));
    EXPECT(!throws_runtime_error([&] { consumer.finish(); }));
    EXPECT(sink->submitted.empty());
    EXPECT(!consumer.snapshot().failed);
}

// A window drops every record still queued, including earlier steps' that
// the sink has not taken yet, so the steps that lost records can outnumber
// the windows.  steps_with_discards counts each such step once; a window
// that dropped nothing counts no step.
static void test_steps_with_discards_counts_every_step_a_window_reaches(
        ring::RecordFailurePolicy policy) {
    std::printf("[ TEST ] steps_with_discards counts each step a window reaches (%s)\n",
                policy == ring::RecordFailurePolicy::kDisableCapture
                    ? "disable_capture" : "raise");
    auto sink = std::make_shared<CapturingSink>();
    ring::RecordConsumer consumer(sink, policy);

    consumer.begin_step();  // 1
    consumer.push_descriptor(descriptor("step", "s1"));
    consumer.begin_step();  // 2
    consumer.push_descriptors({descriptor("step", "s2a"),
                               descriptor("step", "s2b")});
    consumer.begin_step();  // 3: its budget runs out
    consumer.push_descriptor(descriptor("step", "s3a"));
    consumer.begin_discard_window();
    EXPECT(consumer.snapshot().steps_with_discards == 3);
    consumer.push_descriptor(descriptor("step", "s3b"));
    consumer.push_descriptors({});
    EXPECT(consumer.snapshot().steps_with_discards == 3);
    consumer.end_discard_window();

    consumer.begin_step();  // 4
    consumer.push_descriptor(descriptor("step", "s4"));
    consumer.begin_step();  // 5: runs out too
    consumer.begin_discard_window();
    EXPECT(consumer.snapshot().steps_with_discards == 4);
    consumer.push_descriptors({descriptor("step", "s5")});
    EXPECT(consumer.snapshot().steps_with_discards == 5);
    consumer.end_discard_window();

    consumer.begin_step();  // 6: a window that finds nothing to drop
    consumer.begin_discard_window();
    consumer.end_discard_window();
    consumer.begin_step();  // 7
    consumer.push_descriptor(descriptor("step", "s7"));

    for (float value = 1; value <= 8; ++value) {
        consumer.consume_payload(byte_payload({value}));
    }
    const ring::RecordConsumerSnapshot snapshot = consumer.snapshot();
    EXPECT(!snapshot.failed);
    EXPECT(snapshot.steps_with_discards == 5);
    EXPECT(snapshot.discarded_descriptors == 7);
    EXPECT(snapshot.discarded_payloads == 7);
    EXPECT(sink->submitted.size() == 1);
    if (sink->submitted.size() == 1) {
        EXPECT(std::get<std::string>(
                   sink->submitted[0].descriptor.rows[0].cells[0]) == "s7");
        EXPECT(at::equal(sink->submitted[0].payload.view(at::kFloat),
                         at::tensor({8.f})));
    }
    EXPECT(consumer.wait_until_idle(std::chrono::milliseconds(5)));
    EXPECT(!throws_runtime_error([&] { consumer.finish(); }));
}

// The worker asks before copying a payload out of the pinned staging
// whether anyone will store it.  A yes accounts the discard exactly as
// consume_payload would, so pairing and counters stay the same.
static void test_discard_next_payload_if_unwanted_matches_consume_payload() {
    std::printf("[ TEST ] discard_next_payload_if_unwanted matches consume_payload\n");
    for (const auto policy : {ring::RecordFailurePolicy::kRaiseAtProducer,
                              ring::RecordFailurePolicy::kDisableCapture}) {
        auto sink = std::make_shared<CapturingSink>();
        ring::RecordConsumer consumer(sink, policy);
        // Nothing owed: the payload is wanted and nothing changes.
        EXPECT(!consumer.discard_next_payload_if_unwanted());
        consumer.push_descriptors({descriptor("step", "skipped-1"),
                                   descriptor("step", "skipped-2")});
        consumer.begin_discard_window();
        consumer.end_discard_window();
        consumer.push_descriptor(descriptor("step", "kept"));
        EXPECT(consumer.discard_next_payload_if_unwanted());
        consumer.consume_payload(byte_payload({2}));
        EXPECT(!consumer.discard_next_payload_if_unwanted());
        consumer.consume_payload(byte_payload({3}));
        EXPECT(sink->submitted.size() == 1);
        if (sink->submitted.size() == 1) {
            EXPECT(at::equal(sink->submitted[0].payload.view(at::kFloat),
                             at::tensor({3.f})));
        }
        EXPECT(consumer.snapshot().discarded_payloads == 2);
        EXPECT(consumer.wait_until_idle(std::chrono::milliseconds(5)));

        consumer.record_failure(std::make_exception_ptr(
            std::runtime_error("injected latch")));
        if (policy == ring::RecordFailurePolicy::kDisableCapture) {
            // Capture is off: every later payload is unwanted.
            EXPECT(consumer.discard_next_payload_if_unwanted());
            EXPECT(consumer.snapshot().discarded_payloads == 3);
        } else {
            // Left to consume_payload, which raises the latch.
            EXPECT(!consumer.discard_next_payload_if_unwanted());
            EXPECT(consumer.snapshot().discarded_payloads == 2);
            EXPECT(throws_runtime_error(
                [&] { consumer.consume_payload(byte_payload({4})); }));
        }
    }
}

// Past a spent budget the reservation may wait the sink's admission bound
// plus a grace that covers copying out the whole ring at 1 GB/s.
static void test_record_drain_grace_scales_with_the_ring() {
    std::printf("[ TEST ] the post-budget drain grace scales with the ring\n");
    using std::chrono::nanoseconds;
    using std::chrono::seconds;
    EXPECT(ring::record_drain_grace(0, 0) == seconds(2));
    EXPECT(ring::record_drain_grace(4096, 4096) ==
           seconds(2) + nanoseconds(8192));
    // The engine's default 4 GiB ring and 4 GiB staging: about 10.6 s.
    constexpr uint64_t kGiB = 1ull << 30;
    EXPECT(ring::record_drain_grace(4 * kGiB, 4 * kGiB) ==
           seconds(2) + nanoseconds(8 * kGiB));
    EXPECT(ring::record_drain_grace(4 * kGiB, 0) >
           ring::record_drain_grace(1 * kGiB, 0));
    // Saturates instead of wrapping, and a deadline built on it is sane.
    const nanoseconds huge = ring::record_drain_grace(
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max());
    EXPECT(huge > seconds(2));
    EXPECT(std::chrono::steady_clock::now() + huge >
           std::chrono::steady_clock::now());
}

int main() {
    setbuf(stdout, nullptr);
    std::printf("test_record_consumer\n");
    test_fifo_delivers_backend_neutral_envelopes();
    test_zero_row_descriptor_consumes_without_sink_submission();
    test_exact_association_failures();
    test_submit_failure_precedes_durable_idle();
    test_raise_policy_is_the_default_and_raises_at_the_producer();
    test_disable_capture_discards_after_a_latch_and_still_fails_flush();
    test_disable_capture_latches_association_and_worker_failures();
    test_a_discard_window_skips_one_steps_records(
        ring::RecordFailurePolicy::kRaiseAtProducer);
    test_a_discard_window_skips_one_steps_records(
        ring::RecordFailurePolicy::kDisableCapture);
    test_owed_payloads_hold_the_flush(
        ring::RecordFailurePolicy::kRaiseAtProducer);
    test_owed_payloads_hold_the_flush(
        ring::RecordFailurePolicy::kDisableCapture);
    test_steps_with_discards_counts_every_step_a_window_reaches(
        ring::RecordFailurePolicy::kRaiseAtProducer);
    test_steps_with_discards_counts_every_step_a_window_reaches(
        ring::RecordFailurePolicy::kDisableCapture);
    test_discard_next_payload_if_unwanted_matches_consume_payload();
    test_record_drain_grace_scales_with_the_ring();
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
