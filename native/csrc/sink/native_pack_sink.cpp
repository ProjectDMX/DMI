#include "native_pack_sink.h"

#include <ATen/ATen.h>

#include <chrono>
#include <stdexcept>

namespace dmi_sink {

namespace {

[[noreturn]] void invalid(const std::string& what) {
  throw std::runtime_error("NativePackSink: " + what);
}

// Logical shape of the slice, with the dynamic dim resolved from the byte
// count exactly like the reference checked_payload_view does.
bool ResolveShape(const ring::PayloadSlice& slice, uint64_t length_bytes,
                  std::vector<int64_t>* shape_out, std::string* error) {
  *shape_out = slice.logical_shape;
  if (slice.inferred_dynamic_dim < 0) return true;
  const int dim = slice.inferred_dynamic_dim;
  if (dim >= static_cast<int>(shape_out->size())) {
    if (error) *error = "inferred dynamic dim exceeds shape rank";
    return false;
  }
  uint64_t fixed = 1;
  for (size_t i = 0; i < shape_out->size(); ++i) {
    if (static_cast<int>(i) == dim) continue;
    fixed *= static_cast<uint64_t>((*shape_out)[i]);
  }
  if (fixed == 0 || length_bytes % fixed != 0) {
    if (error) *error = "payload bytes do not factor over the fixed dims";
    return false;
  }
  (*shape_out)[dim] =
      static_cast<int64_t>(length_bytes / fixed);
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
}

NativePackSink::~NativePackSink() = default;

void NativePackSink::submit(ring::RecordEnvelope envelope) {
  if (!engine_owned()) invalid("sink is not attached to a RingEngine");
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
    if (!ResolveShape(*slice, length, &shape, &shape_error)) {
      invalid(shape_error);
    }
    RowInput input;
    input.metadata_json = *metadata_json;
    input.payload = bytes + slice->offset_bytes;
    input.payload_bytes = static_cast<size_t>(length);
    input.dtype_name = dtype_name;
    input.shape = std::move(shape);
    std::string detail;
    const RowStatus status = SubmitRow(*sink_, input, &detail);
    if (status != RowStatus::kOk) {
      invalid(std::string(RowStatusName(status)) +
              (detail.empty() ? "" : ": " + detail));
    }
  }
}

bool NativePackSink::flush_and_wait(Duration timeout) {
  if (!engine_owned()) invalid("sink is not attached to a RingEngine");
  const double timeout_s = std::chrono::duration<double>(timeout).count();
  std::string error;
  const bool ok = sink_->Flush(timeout_s < 0 ? -1.0 : timeout_s, &error);
  if (!error.empty()) {
    throw std::runtime_error("NativePackSink: flush failed: " + error);
  }
  return ok;
}

void NativePackSink::rethrow_if_failed() const {
  if (!engine_owned()) invalid("sink is not attached to a RingEngine");
  std::string error = sink_->LastError();
  if (!error.empty()) {
    throw std::runtime_error("NativePackSink: pipeline failed: " + error);
  }
}

}  // namespace dmi_sink
