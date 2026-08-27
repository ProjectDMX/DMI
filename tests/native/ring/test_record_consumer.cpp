#include "ring/record_consumer.h"

#include <ATen/ATen.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

int main() {
    setbuf(stdout, nullptr);
    std::printf("test_record_consumer\n");
    test_fifo_delivers_backend_neutral_envelopes();
    test_zero_row_descriptor_consumes_without_sink_submission();
    test_exact_association_failures();
    test_submit_failure_precedes_durable_idle();
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
