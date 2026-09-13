#include "drop_record_sink.h"

#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace dmx_host {
namespace {
[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("drop record sink: " + message);
}

const char* dtype_name(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return "torch.float32";
        case at::kDouble: return "torch.float64";
        case at::kHalf: return "torch.float16";
        case at::kBFloat16: return "torch.bfloat16";
        case at::kLong: return "torch.int64";
        case at::kInt: return "torch.int32";
        case at::kShort: return "torch.int16";
        case at::kChar: return "torch.int8";
        case at::kByte: return "torch.uint8";
        case at::kBool: return "torch.bool";
        case at::ScalarType::UInt32: return "torch.uint32";
        case at::ScalarType::UInt64: return "torch.uint64";
        case at::ScalarType::UInt16: return "torch.uint16";
        default: invalid("unsupported tensor dtype");
    }
}

uint64_t slice_length(const ring::PayloadSlice& slice, uint64_t payload_bytes) {
    if (slice.offset_bytes > payload_bytes) invalid("slice offset exceeds payload");
    const auto available = payload_bytes - slice.offset_bytes;
    const auto length = slice.length_bytes.value_or(available);
    if (length > available) invalid("slice length exceeds payload");
    return length;
}

std::vector<int64_t> resolved_shape(const ring::PayloadSlice& slice,
                                   uint64_t length, uint64_t element_size) {
    if (length % element_size) invalid("slice length is not dtype-aligned");
    auto shape = slice.logical_shape;
    const auto dynamic = slice.inferred_dynamic_dim;
    if (dynamic < -1 || dynamic >= static_cast<int32_t>(shape.size()))
        invalid("invalid dynamic dimension");
    uint64_t fixed = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int32_t>(i) == dynamic) continue;
        if (shape[i] < 0) invalid("negative dimension");
        const auto dim = static_cast<uint64_t>(shape[i]);
        if (dim && fixed > std::numeric_limits<uint64_t>::max() / dim)
            invalid("shape overflow");
        fixed *= dim;
    }
    const auto elements = length / element_size;
    if (dynamic >= 0) {
        if (!fixed || elements % fixed ||
            elements / fixed > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            invalid("cannot resolve dynamic shape");
        shape[dynamic] = static_cast<int64_t>(elements / fixed);
    } else if (fixed != elements) {
        invalid("shape does not match slice length");
    }
    return shape;
}
}  // namespace

DropRecordSink::DropRecordSink(RecordSchema schema, std::string base_folder,
                               int64_t rank, bool timing_enabled,
                               bool ring_metrics_enabled, Clock clock)
    : schema_(std::move(schema)), rank_(rank), timing_enabled_(timing_enabled),
      ring_metrics_enabled_(ring_metrics_enabled), clock_(std::move(clock)) {
    if (base_folder.empty() || rank < 0) invalid("base folder and nonnegative global rank required");
    std::ostringstream rank_name;
    rank_name << "rank_" << std::setfill('0') << std::setw(5) << rank;
    const auto directory = std::filesystem::path(base_folder) / rank_name.str();
    std::filesystem::create_directories(directory);
    path_ = (directory / "events.jsonl").string();
    // Prevent a second process/engine from concurrently writing this rank's file.
    lock_fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (lock_fd_ < 0) invalid("cannot open " + path_);
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        ::close(lock_fd_);
        lock_fd_ = -1;
        invalid("another writer owns " + path_);
    }
    try {
        file_.exceptions(std::ios::badbit | std::ios::failbit);
        file_.open(path_, std::ios::out | std::ios::app);
        file_.imbue(std::locale::classic());
        enqueue(Event{"schema"});
        writer_ = std::thread([this] { writer_loop(); });
    } catch (...) {
        ::close(lock_fd_);
        lock_fd_ = -1;
        throw;
    }
}

DropRecordSink::~DropRecordSink() noexcept {
    try { close(); } catch (...) {}
}

int64_t DropRecordSink::now_ns() const {
    if (clock_) return clock_();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::optional<int64_t> DropRecordSink::relative_now() {
    // Do not even initialize a time origin or read a clock when disabled.
    if (!timing_enabled_) return std::nullopt;
    std::lock_guard<std::mutex> lock(timing_mutex_);
    if (!origin_ns_) return std::nullopt;  // Setup records precede the first iteration.
    return now_ns() - *origin_ns_;
}

uint64_t DropRecordSink::enqueue(Event event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) std::rethrow_exception(failure_);
    if (closing_) invalid("cannot submit after close");
    event.sequence = ++submitted_;
    queue_.push_back(std::move(event));
    ready_.notify_one();
    return submitted_;
}

void DropRecordSink::submit(ring::RecordEnvelope envelope) {
    // This is the completed owned-CPU record boundary, before materialization/enqueue.
    const auto arrival = timing_enabled_ ? relative_now() : std::optional<int64_t>{};
    const auto& payload = envelope.payload;
    if (!payload.defined() || !payload.device().is_cpu() ||
        payload.scalar_type() != at::kByte || payload.dim() != 1 || !payload.is_contiguous())
        invalid("expected owned contiguous CPU byte payload");
    const auto layout = std::find_if(schema_.layouts.begin(), schema_.layouts.end(),
        [&](const auto& item) { return item.name == envelope.descriptor.layout; });
    if (layout == schema_.layouts.end()) invalid("unknown record layout");
    Event event{"record"};
    event.fields = {{"layout", layout->name}};
    if (arrival) event.fields.emplace_back("cpu_arrival_ns", *arrival);
    const uint64_t bytes = payload.numel();
    for (const auto& encoded : envelope.descriptor.rows) {
        if (encoded.cells.size() != layout->columns.size()) invalid("row width mismatch");
        Fields row;
        for (size_t i = 0; i < encoded.cells.size(); ++i) {
            const auto& column = layout->columns[i];
            const auto& cell = encoded.cells[i];
            const auto* slice = std::get_if<ring::PayloadSlice>(&cell);
            if (!slice) {
                std::visit([&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (!std::is_same_v<T, ring::PayloadSlice>) {
                        if constexpr (std::is_same_v<T, int32_t>)
                            row.emplace_back(column.name, static_cast<int64_t>(value));
                        else row.emplace_back(column.name, value);
                    }
                }, cell);
                continue;
            }
            if (slice->dtype < 0 || slice->dtype >= static_cast<int32_t>(at::ScalarType::NumOptions))
                invalid("invalid encoded dtype");
            const auto dtype = static_cast<at::ScalarType>(slice->dtype);
            const uint64_t element = at::elementSize(dtype);
            if (!element || slice->offset_bytes % element) invalid("unaligned slice");
            const auto length = slice_length(*slice, bytes);
            if (slice->materialization == ring::PayloadMaterialization::TENSOR) {
                row.emplace_back(column.dtype_column, std::string(dtype_name(dtype)));
                row.emplace_back(column.shape_column, resolved_shape(*slice, length, element));
                // Deliberately omit the schema's tensor bytes column, even if named differently.
            } else {
                if (length != element) invalid("scalar requires one element");
                const auto scalar = payload.narrow(0, slice->offset_bytes, length).view(dtype);
                if (slice->materialization == ring::PayloadMaterialization::FLOAT_SCALAR) {
                    if (!c10::isFloatingType(dtype)) invalid("floating scalar dtype mismatch");
                    row.emplace_back(column.name, scalar.item<double>());
                } else {
                    if (!c10::isIntegralType(dtype, false)) invalid("integer scalar dtype mismatch");
                    row.emplace_back(column.name, scalar.item<int64_t>());
                }
            }
        }
        event.rows.push_back(std::move(row));
    }
    // No Tensor or view is retained by Event. The writer queue owns metadata only.
    envelope.payload = at::Tensor{};
    enqueue(std::move(event));
}

void DropRecordSink::iteration_start(int64_t iteration) {
    if (!timing_enabled_) return;
    std::lock_guard<std::mutex> lock(timing_mutex_);
    if (active_iteration_) invalid("iteration already active");
    const auto now = now_ns();
    if (!origin_ns_) origin_ns_ = now;
    active_iteration_ = iteration;
    iteration_start_ns_ = now;
    enqueue(Event{"iteration_start", {{"iteration", iteration}, {"time_ns", now - *origin_ns_}}});
}

void DropRecordSink::iteration_end(int64_t iteration) {
    if (!timing_enabled_) return;
    std::lock_guard<std::mutex> lock(timing_mutex_);
    if (!active_iteration_ || *active_iteration_ != iteration) invalid("iteration end mismatch");
    const auto now = now_ns();
    enqueue(Event{"iteration_end", {{"iteration", iteration}, {"time_ns", now - *origin_ns_},
                                     {"duration_ns", now - *iteration_start_ns_}}});
    active_iteration_.reset();
    iteration_start_ns_.reset();
}

void DropRecordSink::ring_metrics(int64_t iteration, std::map<std::string, uint64_t> metrics) {
    if (!ring_metrics_enabled_) return;
    Event event{"ring_metrics", {{"iteration", iteration}}};
    const auto time = relative_now();
    if (time) event.fields.emplace_back("time_ns", *time);
    for (const auto& item : metrics) event.fields.emplace_back(item.first, item.second);
    enqueue(std::move(event));
}

bool DropRecordSink::flush_and_wait(Duration timeout) {
    Event barrier;
    barrier.barrier = true;
    const auto target = enqueue(std::move(barrier));
    std::unique_lock<std::mutex> lock(mutex_);
    const auto done = written_.wait_for(lock, timeout, [&] { return failure_ || completed_ >= target; });
    if (failure_) std::rethrow_exception(failure_);
    return done;
}

void DropRecordSink::rethrow_if_failed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) std::rethrow_exception(failure_);
}

void DropRecordSink::close() {
    std::lock_guard<std::mutex> close_lock(close_mutex_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }
    ready_.notify_one();
    if (writer_.joinable()) writer_.join();
    if (lock_fd_ >= 0) { ::close(lock_fd_); lock_fd_ = -1; }
    rethrow_if_failed();
}

void DropRecordSink::on_engine_release() noexcept {
    // The ring releases its lease after its final host handoff. Preserve failures
    // for the owning engine's checked close, since this callback is noexcept.
    try { close(); } catch (...) {}
}

void DropRecordSink::writer_loop() noexcept {
    try {
        for (;;) {
            Event event;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return closing_ || !queue_.empty(); });
                if (queue_.empty()) break;
                event = std::move(queue_.front());
                queue_.pop_front();
            }
            if (event.barrier) file_.flush();
            else write_event(event);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                completed_ = event.sequence;
            }
            written_.notify_all();
        }
        file_.flush();
        file_.close();
    } catch (...) {
        const auto failure = std::current_exception();
        file_.exceptions(std::ios::goodbit);
        file_.close();
        std::lock_guard<std::mutex> lock(mutex_);
        failure_ = failure;
        queue_.clear();
    }
    written_.notify_all();
}

void DropRecordSink::write_string(std::ostream& out, const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << static_cast<char>(c);
    }
    out << '"';
}

void DropRecordSink::write_fields(std::ostream& out, const Fields& fields, bool leading_comma) {
    for (const auto& [name, cell] : fields) {
        if (leading_comma) out << ',';
        leading_comma = true;
        write_string(out, name);
        out << ':';
        std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::string>) write_string(out, value);
            else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                out << '[';
                for (size_t i = 0; i < value.size(); ++i) { if (i) out << ','; out << value[i]; }
                out << ']';
            } else if constexpr (std::is_same_v<T, double>) {
                if (std::isfinite(value)) out << std::setprecision(17) << value;
                else write_string(out, std::isnan(value) ? "NaN" : (value < 0 ? "-Infinity" : "Infinity"));
            } else out << value;
        }, cell);
    }
}

void DropRecordSink::write_schema() {
    file_ << ",\"timing_enabled\":" << (timing_enabled_ ? "true" : "false")
          << ",\"ring_metrics_enabled\":" << (ring_metrics_enabled_ ? "true" : "false")
          << ",\"time_origin\":\"first_training_iteration_start_on_this_rank\""
          << ",\"clock\":\"steady_clock\",\"time_unit\":\"ns\",\"layouts\":[";
    for (size_t i = 0; i < schema_.layouts.size(); ++i) {
        if (i) file_ << ',';
        const auto& layout = schema_.layouts[i];
        file_ << "{\"name\":"; write_string(file_, layout.name);
        file_ << ",\"table\":"; write_string(file_, layout.table);
        file_ << ",\"columns\":[";
        for (size_t j = 0; j < layout.columns.size(); ++j) {
            if (j) file_ << ',';
            const auto& col = layout.columns[j];
            file_ << "{\"name\":"; write_string(file_, col.name);
            static const char* types[] = {"string", "int32", "int64", "float64", "int64_array", "tensor"};
            file_ << ",\"type\":"; write_string(file_, types[static_cast<size_t>(col.type)]);
            if (col.type == RecordCellType::TENSOR) {
                write_fields(file_, {{"dtype_column", col.dtype_column}, {"shape_column", col.shape_column},
                                      {"omitted_bytes_column", col.bytes_column}});
            }
            file_ << '}';
        }
        file_ << "]}";
    }
    file_ << ']';
}

void DropRecordSink::write_event(const Event& event) {
    file_ << "{\"type\":"; write_string(file_, event.type);
    file_ << ",\"rank\":" << rank_ << ",\"sequence\":" << event.sequence;
    write_fields(file_, event.fields);
    if (event.type == "schema") write_schema();
    if (event.type == "record") {
        file_ << ",\"rows\":[";
        for (size_t i = 0; i < event.rows.size(); ++i) {
            if (i) file_ << ',';
            file_ << '{';
            write_fields(file_, event.rows[i], false);
            file_ << '}';
        }
        file_ << ']';
    }
    file_ << "}\n";
}
}  // namespace dmx_host
