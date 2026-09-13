// Evaluation sink: retain schema/identity and optional measurements, discard tensors.
#pragma once

#include "record_schema.h"
#include "ring/record_sink.h"

#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <thread>
#include <variant>

namespace dmx_host {

class DropRecordSink final : public ring::RecordSink {
public:
    using Clock = std::function<int64_t()>;
    DropRecordSink(RecordSchema schema, std::string base_folder, int64_t rank,
                   bool timing_enabled = false, bool ring_metrics_enabled = false,
                   Clock clock = {});
    ~DropRecordSink() noexcept override;

    void submit(ring::RecordEnvelope envelope) override;
    bool flush_and_wait(Duration timeout) override;
    void rethrow_if_failed() const override;
    void close();
    void iteration_start(int64_t iteration);
    void iteration_end(int64_t iteration);
    void ring_metrics(int64_t iteration, std::map<std::string, uint64_t> metrics);
    bool timing_enabled() const { return timing_enabled_; }
    bool ring_metrics_enabled() const { return ring_metrics_enabled_; }
    const RecordSchema& schema() const { return schema_; }
    const std::string& path() const { return path_; }

protected:
    void on_engine_release() noexcept override;

private:
    using Cell = std::variant<std::string, int64_t, uint64_t, double,
                              std::vector<int64_t>>;
    using Fields = std::vector<std::pair<std::string, Cell>>;
    struct Event {
        std::string type;
        Fields fields;
        std::vector<Fields> rows;
        uint64_t sequence{0};
        bool barrier{false};
    };
    const RecordSchema schema_;
    const int64_t rank_;
    const bool timing_enabled_;
    const bool ring_metrics_enabled_;
    Clock clock_;
    std::string path_;
    std::ofstream file_;
    int lock_fd_{-1};
    std::thread writer_;
    mutable std::mutex mutex_;
    std::mutex close_mutex_;
    std::condition_variable ready_, written_;
    std::deque<Event> queue_;  // Deliberately unbounded; contains no tensor payloads.
    std::exception_ptr failure_;
    bool closing_{false};
    uint64_t submitted_{0}, completed_{0};
    std::mutex timing_mutex_;
    std::optional<int64_t> origin_ns_, iteration_start_ns_, active_iteration_;

    int64_t now_ns() const;
    std::optional<int64_t> relative_now();
    uint64_t enqueue(Event event);
    void writer_loop() noexcept;
    void write_event(const Event& event);
    void write_schema();
    static void write_string(std::ostream& out, const std::string& value);
    static void write_fields(std::ostream& out, const Fields& fields, bool leading_comma = true);
};

}  // namespace dmx_host
