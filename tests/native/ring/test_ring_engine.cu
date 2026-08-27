// CUDA integration tests for producer -> drain -> pinned-staging delivery.

#include "ring/drain_thread.h"
#include "ring/pinned_staging.h"
#include "ring/producer.cuh"
#include "ring/ring_alloc.h"
#include "ring/ring_engine_py.h"
#include "ring/record_sink.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

class TrackingRecordSink final : public ring::RecordSink {
public:
    void submit(ring::RecordEnvelope) override { ++submissions; }

    bool flush_and_wait(Duration timeout) override {
        observed_timeout = timeout;
        ++flushes;
        return true;
    }

    void rethrow_if_failed() const override { ++failure_checks; }

    std::atomic<int> submissions{0};
    std::atomic<int> flushes{0};
    mutable std::atomic<int> failure_checks{0};
    Duration observed_timeout{0};
};

// RingEnginePy's diagnostic hooks are irrelevant to this focused native
// executable; the extension supplies their real definitions.
void ring_diag_reset_host_counters() {}
void ring_diag_print_host_counters() {}

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t error_ = (expr);                                        \
        if (error_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(error_));    \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

#define EXPECT(condition)                                                   \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                      \
                         __FILE__, __LINE__, #condition);                    \
            ++g_fail;                                                       \
        } else {                                                            \
            ++g_pass;                                                       \
        }                                                                   \
    } while (0)

static void banner(const char* name) {
    std::printf("[ TEST ] %s\n", name);
}

static ring::RingConfig make_config(uint64_t capacity = 4096) {
    ring::RingConfig cfg;
    cfg.task_ring_entries = 16;
    cfg.payload_ring_bytes = capacity;
    cfg.pinned_staging_bytes = capacity;
    cfg.drain_poll_timeout_us = 100;
    cfg.drain_flush.task_ratio = 0.0f;
    cfg.drain_flush.payload_ratio = 0.0f;
    cfg.drain_flush.entry_threshold = 0;
    cfg.drain_flush.byte_threshold = 0;
    cfg.drain_flush.timeout_us = 0;
    return cfg;
}

static std::vector<uint8_t> pattern(uint64_t size, uint8_t seed) {
    std::vector<uint8_t> result(size);
    for (uint64_t i = 0; i < size; ++i) {
        result[i] = static_cast<uint8_t>(seed + i * 29);
    }
    return result;
}

static uint8_t* upload(const std::vector<uint8_t>& source,
                       cudaStream_t stream) {
    if (source.empty()) {
        return nullptr;
    }
    uint8_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, source.size()));
    CUDA_CHECK(cudaMemcpyAsync(device, source.data(), source.size(),
                               cudaMemcpyHostToDevice, stream));
    return device;
}

static std::vector<uint8_t> task_bytes(const ring::DrainTask& task) {
    std::vector<uint8_t> result(task.tensor_total_bytes);
    if (task.data_len1 > 0) {
        std::memcpy(result.data(), task.data_ptr1, task.data_len1);
    }
    if (task.data_len2 > 0) {
        std::memcpy(result.data() + task.data_len1,
                    task.data_ptr2, task.data_len2);
    }
    return result;
}

class DrainHarness {
public:
    explicit DrainHarness(const ring::RingConfig& config)
        : cfg(config), allocated(cfg) {
        allocated.init();
        staging.init(cfg.effective_staging_bytes());
        drain = std::make_unique<ring::DrainThread>(
            allocated.state(), staging, cfg);
        CUDA_CHECK(cudaStreamCreate(&stream));
        drain->start();
    }

    ~DrainHarness() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        drain->stop();
        drain->signal_p2p_stop();
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    DrainHarness(const DrainHarness&) = delete;
    DrainHarness& operator=(const DrainHarness&) = delete;

    ring::DrainTask flush_one() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        drain->force_flush_and_wait();
        const uint64_t count = drain->wait_for_tasks();
        EXPECT(count == 1);
        std::vector<ring::DrainTask> tasks;
        drain->pop_tasks(count, tasks);
        EXPECT(tasks.size() == 1);
        return std::move(tasks.front());
    }

    void release(const ring::DrainTask& task) {
        drain->notify_staging_freed_bytes(task.alloc_bytes);
    }

    ring::RingConfig cfg;
    ring::AllocatedRing allocated;
    ring::PinnedStaging staging;
    std::unique_ptr<ring::DrainThread> drain;
    cudaStream_t stream{};
};

static void test_static_force_flush() {
    banner("static producer drains exact data");
    DrainHarness harness(make_config());
    const std::vector<uint8_t> source = pattern(333, 7);
    uint8_t* device = upload(source, harness.stream);

    const uint64_t reserved = ring::align_up(source.size(), ring::PAYLOAD_ALIGN);
    harness.drain->reserve(reserved, 1);
    ring::launch_producer_static(harness.allocated.state(), device,
                                 source.size(), 0, harness.stream);
    ring::DrainTask task = harness.flush_one();

    EXPECT(task.tensor_total_bytes == source.size());
    EXPECT(task.alloc_bytes == reserved);
    EXPECT(task.data_len1 + task.data_len2 == source.size());
    EXPECT(task_bytes(task) == source);
    EXPECT(harness.drain->cpu_task_head() == 1);
    EXPECT(harness.drain->cpu_task_tail_committed() == 1);
    EXPECT(harness.drain->cpu_payload_head() == reserved);
    EXPECT(harness.drain->cpu_payload_tail_committed() == reserved);

    harness.release(task);
    CUDA_CHECK(cudaFree(device));
}

static void test_prefix_force_flush() {
    banner("prefix producer drains selected rows");
    DrainHarness harness(make_config());
    constexpr uint64_t row_bytes = 32;
    const std::vector<uint8_t> source = pattern(8 * row_bytes, 23);
    uint8_t* device = upload(source, harness.stream);
    int64_t row_count = 3;
    int64_t* device_count = nullptr;
    CUDA_CHECK(cudaMalloc(&device_count, sizeof(row_count)));
    CUDA_CHECK(cudaMemcpyAsync(device_count, &row_count, sizeof(row_count),
                               cudaMemcpyHostToDevice, harness.stream));

    constexpr uint64_t actual = 3 * row_bytes;
    harness.drain->reserve(actual, 1);
    ring::launch_producer_prefix(harness.allocated.state(), device,
                                 source.size(), device_count, row_bytes, 0,
                                 harness.stream);
    ring::DrainTask task = harness.flush_one();

    std::vector<uint8_t> expected(source.begin(), source.begin() + actual);
    EXPECT(task.tensor_total_bytes == actual);
    EXPECT(task.alloc_bytes == actual);
    EXPECT(task_bytes(task) == expected);
    EXPECT(harness.drain->cpu_payload_tail_committed() == actual);

    harness.release(task);
    CUDA_CHECK(cudaFree(device_count));
    CUDA_CHECK(cudaFree(device));
}

static void test_repeated_wrap_delivery() {
    banner("repeated flushes preserve data across GPU and staging wraps");
    constexpr uint64_t capacity = 256;
    constexpr uint64_t bytes_per_tensor = 80;
    DrainHarness harness(make_config(capacity));

    for (uint64_t sequence = 0; sequence < 4; ++sequence) {
        const std::vector<uint8_t> source =
            pattern(bytes_per_tensor, static_cast<uint8_t>(50 + sequence));
        uint8_t* device = upload(source, harness.stream);
        harness.drain->reserve(bytes_per_tensor, 1);
        ring::launch_producer_static(harness.allocated.state(), device,
                                     source.size(), 0, harness.stream);
        ring::DrainTask task = harness.flush_one();

        EXPECT(task.tensor_total_bytes == bytes_per_tensor);
        EXPECT(task_bytes(task) == source);
        if (sequence == 3) {
            EXPECT(task.data_len1 == 16);
            EXPECT(task.data_len2 == 64);
        }
        harness.release(task);
        CUDA_CHECK(cudaFree(device));
    }

    EXPECT(*harness.allocated.state().task_head == 4);
    EXPECT(*harness.allocated.state().payload_head == 320);
    EXPECT(harness.drain->cpu_task_tail_committed() == 4);
    EXPECT(harness.drain->cpu_payload_tail_committed() == 320);
    EXPECT(harness.staging.head() == 320);
    EXPECT(harness.staging.tail() == 320);
}

static void test_zero_byte_delivery() {
    banner("zero-byte tensor still publishes one task");
    DrainHarness harness(make_config());
    harness.drain->reserve(0, 1);
    ring::launch_producer_static(harness.allocated.state(), nullptr, 0, 0,
                                 harness.stream);
    ring::DrainTask task = harness.flush_one();

    EXPECT(task.tensor_total_bytes == 0);
    EXPECT(task.alloc_bytes == 0);
    EXPECT(task.data_ptr1 == nullptr);
    EXPECT(task.data_len1 == 0);
    EXPECT(task.data_ptr2 == nullptr);
    EXPECT(task.data_len2 == 0);
    EXPECT(harness.drain->cpu_task_tail_committed() == 1);
    EXPECT(harness.drain->cpu_payload_tail_committed() == 0);
    harness.release(task);
}

static void test_record_reservation_reclaims_per_entry() {
    banner("record reservation reclaims each possibly-short task independently");
    DrainHarness harness(make_config());

    constexpr uint64_t row_bytes = 32;
    const std::vector<uint8_t> dynamic_source = pattern(256, 111);
    const std::vector<uint8_t> fixed_source = pattern(64, 137);
    uint8_t* dynamic_device = upload(dynamic_source, harness.stream);
    uint8_t* fixed_device = upload(fixed_source, harness.stream);
    int64_t row_count = 1;
    int64_t* device_count = nullptr;
    CUDA_CHECK(cudaMalloc(&device_count, sizeof(row_count)));
    CUDA_CHECK(cudaMemcpyAsync(device_count, &row_count, sizeof(row_count),
                               cudaMemcpyHostToDevice, harness.stream));

    const uint64_t dynamic_upper =
        ring::align_up(dynamic_source.size(), ring::PAYLOAD_ALIGN);
    const uint64_t fixed_bytes =
        ring::align_up(fixed_source.size(), ring::PAYLOAD_ALIGN);
    const uint64_t upper = dynamic_upper + fixed_bytes;
    harness.drain->reserve_record({
        {dynamic_upper, true},
        {fixed_bytes, false},
    });

    ring::launch_record_producer_prefix(
        harness.allocated.state(), dynamic_device, dynamic_source.size(),
        device_count, row_bytes, nullptr, 0, harness.stream);
    ring::DrainTask first = harness.flush_one();
    EXPECT(first.tensor_total_bytes == row_bytes);
    EXPECT(harness.drain->cpu_payload_head() == upper);
    EXPECT(harness.drain->pending_record_reclaims() == 0);
    harness.drain->apply_pending_record_reclaims();
    const uint64_t actual_aligned =
        ring::align_up(row_bytes, ring::PAYLOAD_ALIGN) + fixed_bytes;
    EXPECT(harness.drain->cpu_payload_head() == actual_aligned);
    harness.release(first);

    ring::launch_record_producer_static(
        harness.allocated.state(), fixed_device, fixed_source.size(),
        nullptr, 0, harness.stream);
    ring::DrainTask second = harness.flush_one();
    EXPECT(second.tensor_total_bytes == fixed_source.size());
    EXPECT(harness.drain->pending_record_reclaims() == 0);
    EXPECT(harness.drain->cpu_payload_head() == actual_aligned);
    EXPECT(harness.drain->cpu_payload_tail_committed() == actual_aligned);
    harness.drain->rethrow_record_reclaim_failure();
    harness.release(second);

    CUDA_CHECK(cudaFree(device_count));
    CUDA_CHECK(cudaFree(fixed_device));
    CUDA_CHECK(cudaFree(dynamic_device));
}

static void test_false_gated_record_reclaims_its_full_reservation() {
    banner("false-gated record publishes and reclaims a zero-byte task");
    DrainHarness harness(make_config());

    const std::vector<uint8_t> source = pattern(64, 173);
    uint8_t* device = upload(source, harness.stream);
    int32_t* gate = nullptr;
    const int32_t disabled = 0;
    CUDA_CHECK(cudaMalloc(&gate, sizeof(disabled)));
    CUDA_CHECK(cudaMemcpyAsync(gate, &disabled, sizeof(disabled),
                               cudaMemcpyHostToDevice, harness.stream));

    const uint64_t reserved =
        ring::align_up(source.size(), ring::PAYLOAD_ALIGN);
    harness.drain->reserve_record({{reserved, true}});
    ring::launch_record_producer_static(
        harness.allocated.state(), device, source.size(), gate, 1,
        harness.stream);
    ring::DrainTask task = harness.flush_one();

    EXPECT(task.tensor_total_bytes == 0);
    EXPECT(task.alloc_bytes == 0);
    EXPECT(harness.drain->pending_record_reclaims() == 0);
    EXPECT(harness.drain->cpu_payload_head() == reserved);
    harness.drain->apply_pending_record_reclaims();
    EXPECT(harness.drain->cpu_payload_head() == 0);
    EXPECT(harness.drain->cpu_payload_tail_committed() == 0);
    harness.drain->rethrow_record_reclaim_failure();
    harness.release(task);

    CUDA_CHECK(cudaFree(gate));
    CUDA_CHECK(cudaFree(device));
}

static void test_timed_drain_flush_uses_request_generations() {
    banner("timed drain flush does not reuse another request generation");
    ring::RingConfig cfg = make_config(512);
    cfg.pinned_staging_bytes = 256;
    DrainHarness harness(cfg);

    const std::vector<uint8_t> first_source = pattern(256, 191);
    uint8_t* first_device = upload(first_source, harness.stream);
    harness.drain->reserve(first_source.size(), 1);
    ring::launch_producer_static(
        harness.allocated.state(), first_device, first_source.size(), 0,
        harness.stream);
    ring::DrainTask first = harness.flush_one();

    // Keep the first task's staging allocation outstanding so the next flush
    // generation has a deterministic wait point inside the drain worker.
    const std::vector<uint8_t> second_source = pattern(64, 211);
    uint8_t* second_device = upload(second_source, harness.stream);
    harness.drain->reserve(second_source.size(), 1);
    ring::launch_producer_static(
        harness.allocated.state(), second_device, second_source.size(), 0,
        harness.stream);
    CUDA_CHECK(cudaStreamSynchronize(harness.stream));

    const bool blocked_generation_completed =
        harness.drain->force_flush_and_wait_until(
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(25));
    EXPECT(!blocked_generation_completed);

    harness.release(first);
    bool next_generation_completed =
        harness.drain->force_flush_and_wait_until(
            std::chrono::steady_clock::now() + std::chrono::seconds(2));
    EXPECT(next_generation_completed);
    if (!next_generation_completed) {
        harness.drain->force_flush_and_wait();
    }

    const uint64_t count = harness.drain->wait_for_tasks();
    EXPECT(count == 1);
    std::vector<ring::DrainTask> tasks;
    harness.drain->pop_tasks(count, tasks);
    EXPECT(tasks.size() == 1);
    if (!tasks.empty()) {
        EXPECT(task_bytes(tasks.front()) == second_source);
        harness.release(tasks.front());
    }

    CUDA_CHECK(cudaFree(second_device));
    CUDA_CHECK(cudaFree(first_device));
}

struct BlockingCallbackState {
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
};

static void CUDART_CB blocking_host_callback(void* data) {
    auto* state = static_cast<BlockingCallbackState*>(data);
    state->entered.store(true, std::memory_order_release);
    while (!state->release.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static void test_drain_worker_binds_owner_device() {
    banner("drain worker binds the ring owner device");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::printf("[ SKIP ] requires two CUDA devices\n");
        return;
    }

    int original_device = 0;
    CUDA_CHECK(cudaGetDevice(&original_device));

    BlockingCallbackState callback;
    cudaStream_t blocked_stream{};
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&blocked_stream,
                                         cudaStreamNonBlocking));
    CUDA_CHECK(cudaLaunchHostFunc(blocked_stream, blocking_host_callback,
                                  &callback));

    const auto callback_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!callback.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < callback_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool callback_entered =
        callback.entered.load(std::memory_order_acquire);
    EXPECT(callback_entered);

    bool stopped_without_waiting_for_device_zero = false;
    if (callback_entered) {
        CUDA_CHECK(cudaSetDevice(1));
        ring::RingConfig cfg = make_config();
        ring::AllocatedRing allocated(cfg);
        allocated.init();
        ring::PinnedStaging staging;
        staging.init(cfg.effective_staging_bytes());
        auto drain = std::make_unique<ring::DrainThread>(
            allocated.state(), staging, cfg);

        CUDA_CHECK(cudaSetDevice(0));
        drain->start();
        auto stopped = std::async(std::launch::async, [&drain] {
            drain->stop();
        });
        stopped_without_waiting_for_device_zero =
            stopped.wait_for(std::chrono::seconds(2)) ==
            std::future_status::ready;

        callback.release.store(true, std::memory_order_release);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamSynchronize(blocked_stream));
        stopped.get();

        CUDA_CHECK(cudaSetDevice(1));
        drain.reset();
    } else {
        callback.release.store(true, std::memory_order_release);
        CUDA_CHECK(cudaStreamSynchronize(blocked_stream));
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamDestroy(blocked_stream));
    CUDA_CHECK(cudaSetDevice(original_device));
    EXPECT(stopped_without_waiting_for_device_zero);
}

static void test_record_flush_bounds_current_stream_prefix_wait() {
    banner("record flush bounds its exact current-stream prefix wait");

    const c10::cuda::CUDAStream test_stream =
        c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard stream_guard(test_stream);

    ring_py::RingConfig cfg;
    cfg.task_ring_entries = 16;
    cfg.payload_ring_bytes = 4096;
    cfg.pinned_staging_bytes = 4096;
    cfg.drain_poll_timeout_us = 100;
    ring_py::RingEnginePy engine(
        cfg, std::shared_ptr<ring::RecordSink>{});
    engine.init();
    engine.start();

    BlockingCallbackState callback;
    cudaStream_t stream = test_stream.stream();
    CUDA_CHECK(cudaLaunchHostFunc(stream, blocking_host_callback, &callback));

    const auto callback_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!callback.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < callback_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool callback_entered =
        callback.entered.load(std::memory_order_acquire);
    EXPECT(callback_entered);

    bool completed = true;
    bool threw = false;
    std::chrono::steady_clock::duration elapsed{};
    if (callback_entered) {
        std::atomic<bool> flush_returned{false};
        std::thread watchdog([&] {
            const auto watchdog_deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(500);
            while (!flush_returned.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < watchdog_deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            callback.release.store(true, std::memory_order_release);
        });

        const auto started = std::chrono::steady_clock::now();
        try {
            completed = engine.flush_records_and_wait(20);
        } catch (...) {
            threw = true;
        }
        elapsed = std::chrono::steady_clock::now() - started;
        flush_returned.store(true, std::memory_order_release);
        callback.release.store(true, std::memory_order_release);
        watchdog.join();
    } else {
        callback.release.store(true, std::memory_order_release);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT(!threw);
    if (!threw && callback_entered) {
        EXPECT(!completed);
        EXPECT(elapsed < std::chrono::milliseconds(250));
    }
    engine.stop();
}

static void test_record_flush_reaches_sink_durability_boundary() {
    banner("record flush reaches the configured sink boundary");

    ring_py::RingConfig cfg;
    cfg.task_ring_entries = 16;
    cfg.payload_ring_bytes = 4096;
    cfg.pinned_staging_bytes = 4096;
    cfg.drain_poll_timeout_us = 100;
    auto sink = std::make_shared<TrackingRecordSink>();
    ring_py::RingEnginePy engine(cfg, sink);
    engine.init();
    engine.start();

    const bool completed = engine.flush_records_and_wait(1000);

    EXPECT(completed);
    EXPECT(sink->submissions.load(std::memory_order_acquire) == 0);
    EXPECT(sink->flushes.load(std::memory_order_acquire) == 1);
    EXPECT(sink->failure_checks.load(std::memory_order_acquire) == 2);
    EXPECT(sink->observed_timeout > std::chrono::milliseconds(0));
    EXPECT(sink->observed_timeout <= std::chrono::milliseconds(1000));
    engine.stop();
}

int main() {
    setbuf(stdout, nullptr);
    ring::set_ring_null_mode(false);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("test_ring_engine (current drain pipeline)\n");
    test_static_force_flush();
    test_prefix_force_flush();
    test_repeated_wrap_delivery();
    test_zero_byte_delivery();
    test_record_reservation_reclaims_per_entry();
    test_false_gated_record_reclaims_its_full_reservation();
    test_timed_drain_flush_uses_request_generations();
    test_drain_worker_binds_owner_device();
    test_record_flush_bounds_current_stream_prefix_wait();
    test_record_flush_reaches_sink_durability_boundary();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
