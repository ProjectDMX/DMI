#include "native_pack_sink.h"

#include <ATen/ATen.h>

#include <chrono>
#include <limits>
#include <stdexcept>

namespace dmi_sink {

namespace {

[[noreturn]] void invalid(const std::string& what) {
  throw std::runtime_error("NativePackSink: " + what);
}

// Logical shape of the slice, with the dynamic dim resolved from the byte
// count exactly like the reference resolve_shape does.
//
// `element_bytes` is the slice dtype's width (DtypeWidth of the mapped name,
// which the caller has already proved non-zero). A dynamic dim counts
// ELEMENTS, not bytes: the reference divides the slice length by the element
// size first and only then factors it over the fixed dims. Dividing the raw
// byte count resolved every dtype wider than one byte too large by exactly
// that width -- a float32 [-1, 4] slice of 32 bytes came out [8, 4] where the
// reference says [2, 4] -- and SubmitRow refused the row against its own
// metadata, so every capture with a dynamic dim failed hard.
bool ResolveShape(const ring::PayloadSlice& slice, uint64_t length_bytes,
                  uint64_t element_bytes, std::vector<int64_t>* shape_out,
                  std::string* error) {
  *shape_out = slice.logical_shape;
  const int dim = slice.inferred_dynamic_dim;
  if (dim >= static_cast<int>(shape_out->size())) {
    if (error) *error = "inferred dynamic dim exceeds shape rank";
    return false;
  }
  // Checked like the reference's checked_product: a negative fixed dim has no
  // uint64 meaning, and an unchecked multiply can wrap before the modulo
  // below -- a wrapped fixed can divide elements evenly, admitting a shape
  // whose fixed product never fit (a zero inferred dim made `elements % fixed`
  // pass on an overflowed fixed). With no dynamic dim the walk covers the
  // whole shape, so SubmitRow's own unchecked multiply never sees a product
  // that wrapped to match an empty payload.
  uint64_t fixed = 1;
  for (size_t i = 0; i < shape_out->size(); ++i) {
    if (static_cast<int>(i) == dim) continue;
    const int64_t dim_value = (*shape_out)[i];
    if (dim_value < 0) {
      if (error) *error = "negative logical tensor dimension";
      return false;
    }
    const uint64_t dimension = static_cast<uint64_t>(dim_value);
    if (dimension != 0 &&
        fixed > std::numeric_limits<uint64_t>::max() / dimension) {
      if (error) *error = "logical tensor shape overflows uint64";
      return false;
    }
    fixed *= dimension;
  }
  if (dim < 0) return true;
  if (element_bytes == 0 || length_bytes % element_bytes != 0) {
    if (error) *error = "payload-slice bytes are not divisible by dtype size";
    return false;
  }
  const uint64_t elements = length_bytes / element_bytes;
  if (fixed == 0 || elements % fixed != 0) {
    if (error) *error = "payload bytes do not factor over the fixed dims";
    return false;
  }
  const uint64_t inferred = elements / fixed;
  if (inferred >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    if (error) *error = "inferred tensor dimension exceeds int64";
    return false;
  }
  (*shape_out)[dim] = static_cast<int64_t>(inferred);
  return true;
}

}  // namespace

std::string AtenDtypeName(int32_t scalar_type) {
  switch (static_cast<at::ScalarType>(scalar_type)) {
    case at::ScalarType::Bool: return "bool";
    case at::ScalarType::Byte: return "uint8";
    case at::ScalarType::Char: return "int8";
    case at::ScalarType::Short: return "int16";
    case at::ScalarType::Half: return "float16";
    case at::ScalarType::BFloat16: return "bfloat16";
    case at::ScalarType::Int: return "int32";
    case at::ScalarType::Float: return "float32";
    case at::ScalarType::Long: return "int64";
    case at::ScalarType::Double: return "float64";
    // The dtypes the validation layer admits and the pack format carries:
    // the adapter must accept every one of them, or a capture fails at the
    // ring boundary that the rest of the pipeline already handles.
    // torch.uint16/uint32 arrived in torch 2.3; float8 in 2.1.
    case at::ScalarType::UInt16: return "uint16";
    case at::ScalarType::UInt32: return "uint32";
    case at::ScalarType::Float8_e4m3fn: return "float8_e4m3fn";
    case at::ScalarType::Float8_e5m2: return "float8_e5m2";
    default: return "";
  }
}

NativePackSink::NativePackSink(std::unique_ptr<PackSink> sink,
                               std::string layout)
    : sink_(std::move(sink)), layout_(std::move(layout)) {
  if (!sink_) invalid("PackSink is required");
  if (layout_.empty()) invalid("layout is required");
  std::string error;
  const std::string start_error = sink_->Start(&error);
  if (!start_error.empty()) invalid("sink start failed: " + start_error);
  baseline_ = sink_->Snapshot();
}

std::string NativePackSink::LossError() const {
  const SinkSnapshot now = sink_->Snapshot();
  std::string losses;
  const auto add = [&](const char* name, uint64_t current, uint64_t base) {
    if (current == base) return;
    if (!losses.empty()) losses += ", ";
    losses += std::string(name) + "=" + std::to_string(current - base);
  };
  // record_adapter.py _LOSS_COUNTERS, in its order.
  add("dropped_records", now.dropped_records, baseline_.dropped_records);
  add("timed_out_records", now.timed_out_records,
      baseline_.timed_out_records);
  add("oversized_records", now.oversized_records,
      baseline_.oversized_records);
  add("duplicate_records", now.duplicate_records,
      baseline_.duplicate_records);
  add("rejected_closed_records", now.rejected_closed_records,
      baseline_.rejected_closed_records);
  add("failures", now.failures, baseline_.failures);
  if (losses.empty()) return "";
  return "NativePackSink: pipeline reported lost records (" + losses + ")";
}

std::optional<ring::RecordSink::Duration>
NativePackSink::admission_bound() const {
  const SinkConfig& config = sink_->config();
  if (config.overload == Overload::kDropNewest) return Duration::zero();
  if (config.admission_timeout_s < 0) return std::nullopt;
  // Rounded up: a bound the ring waits on must not be shorter than the
  // sink's own deadline.
  return std::chrono::ceil<Duration>(
      std::chrono::duration<double>(config.admission_timeout_s));
}

NativePackSink::~NativePackSink() = default;

void NativePackSink::submit(ring::RecordEnvelope envelope) {
  if (!engine_owned()) invalid("sink is not attached to a RingEngine");
  // A record lost after admission (oversized framing, a duplicate id) is
  // counted on the pack worker, after its submit returned. Refusing the
  // next envelope is how the ring hears of it: its record runtime latches
  // and reports the loss instead of storing around a hole.
  rethrow_if_failed();
  const ring::RecordDescriptor& descriptor = envelope.descriptor;
  if (descriptor.layout != layout_) invalid("unexpected record layout");
  if (descriptor.rows.empty()) {
    invalid("descriptor must contain at least one row");
  }
  const at::Tensor& payload = envelope.payload;
  if (payload.device().type() != at::DeviceType::CPU) {
    invalid("capture payload must be a CPU tensor");
  }
  if (!payload.is_contiguous()) {
    invalid("capture payload must be contiguous");
  }
  const auto* bytes = static_cast<const uint8_t*>(payload.data_ptr());
  const auto total =
      static_cast<uint64_t>(payload.numel()) *
      static_cast<uint64_t>(payload.element_size());

  // One admission deadline for the whole envelope, started here.
  EnvelopeAdmission admission(*sink_);
  for (const ring::EncodedRecordRow& row : descriptor.rows) {
    // Cheap structural check first (mirrors the reference sink).
    if (row.cells.size() != 2) invalid("descriptor row must contain two cells");
    const auto* metadata_json = std::get_if<std::string>(&row.cells[0]);
    const auto* slice = std::get_if<ring::PayloadSlice>(&row.cells[1]);
    if (metadata_json == nullptr || slice == nullptr) {
      invalid("descriptor requires metadata JSON followed by one payload slice");
    }
    if (slice->materialization != ring::PayloadMaterialization::TENSOR) {
      invalid("capture packs require tensor materialization");
    }
    if (slice->offset_bytes > total) {
      invalid("payload-slice offset exceeds physical payload");
    }
    const uint64_t available = total - slice->offset_bytes;
    const uint64_t length =
        slice->length_bytes.has_value() ? *slice->length_bytes : available;
    if (length > available) {
      invalid("payload slice exceeds physical payload");
    }
    const std::string dtype_name = AtenDtypeName(slice->dtype);
    if (dtype_name.empty()) invalid("unsupported payload dtype");
    // The slice must be dtype-aligned; SubmitRow re-checks logical size.
    const int width = DtypeWidth(dtype_name);
    if (width <= 0 ||
        slice->offset_bytes % static_cast<uint64_t>(width) != 0) {
      invalid("payload-slice offset is not dtype-aligned");
    }
    std::vector<int64_t> shape;
    std::string shape_error;
    if (!ResolveShape(*slice, length, static_cast<uint64_t>(width), &shape,
                      &shape_error)) {
      invalid(shape_error);
    }
    RowInput input;
    input.metadata_json = *metadata_json;
    input.payload = bytes + slice->offset_bytes;
    input.payload_bytes = static_cast<size_t>(length);
    input.dtype_name = dtype_name;
    input.shape = std::move(shape);
    std::string detail;
    const RowStatus status = admission.SubmitRow(input, &detail);
    if (status != RowStatus::kOk) {
      invalid(std::string(RowStatusName(status)) +
              (detail.empty() ? "" : ": " + detail));
    }
  }
}

bool NativePackSink::flush_and_wait(Duration timeout) {
  if (!engine_owned()) invalid("sink is not attached to a RingEngine");
  const double timeout_s = std::chrono::duration<double>(timeout).count();
  // Everything admitted before the barrier must be persisted by it.
  const uint64_t target = sink_->Snapshot().admitted_records;
  std::string error;
  const bool ok = sink_->Flush(timeout_s < 0 ? -1.0 : timeout_s, &error);
  if (!error.empty()) {
    throw std::runtime_error("NativePackSink: flush failed: " + error);
  }
  if (!ok) return false;
  const std::string losses = LossError();
  if (!losses.empty()) throw std::runtime_error(losses);
  const uint64_t persisted = sink_->Snapshot().persisted_records;
  if (persisted - baseline_.persisted_records <
      target - baseline_.admitted_records) {
    throw std::runtime_error(
        "NativePackSink: durability mismatch: admitted=" +
        std::to_string(target - baseline_.admitted_records) +
        ", persisted=" +
        std::to_string(persisted - baseline_.persisted_records));
  }
  return true;
}

void NativePackSink::rethrow_if_failed() const {
  if (!engine_owned()) invalid("sink is not attached to a RingEngine");
  std::string error = sink_->LastError();
  if (!error.empty()) {
    throw std::runtime_error("NativePackSink: pipeline failed: " + error);
  }
  const std::string losses = LossError();
  if (!losses.empty()) throw std::runtime_error(losses);
}

}  // namespace dmi_sink
