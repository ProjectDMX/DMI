#include "record_row.h"

#include "../common/json.h"
#include "pack_sink.h"

namespace dmi_sink {

namespace jc = dmi_common;

int DtypeWidth(const std::string& dtype_name) {
  // The same set as model.py's _DTYPE_BYTES: the sink must accept every
  // dtype the validation layer admits, or a capture fails at admission
  // that the rest of the pipeline (catalog columns, hydration decoder)
  // already handles.
  if (dtype_name == "bool" || dtype_name == "uint8" ||
      dtype_name == "int8" || dtype_name == "float8_e4m3fn" ||
      dtype_name == "float8_e5m2") {
    return 1;
  }
  if (dtype_name == "uint16" || dtype_name == "int16" ||
      dtype_name == "float16" || dtype_name == "bfloat16") {
    return 2;
  }
  if (dtype_name == "uint32" || dtype_name == "int32" ||
      dtype_name == "float32") {
    return 4;
  }
  if (dtype_name == "int64" || dtype_name == "float64") return 8;
  return 0;
}

bool ParseMetadataJson(const std::string& text,
                       dmi_pack::RecordMetadata* out,
                       std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };
  // The object may carry surrounding whitespace; Find* scans for keys.
  out->capture_id = jc::FindString(text, "capture_id");
  out->tenant_id = jc::FindString(text, "tenant_id");
  out->experiment_id = jc::FindString(text, "experiment_id");
  out->run_id = jc::FindString(text, "run_id");
  out->session_id = jc::FindString(text, "session_id");
  out->request_id = jc::FindString(text, "request_id");
  out->sequence_id = jc::FindString(text, "sequence_id");
  out->model_id = jc::FindString(text, "model_id");
  out->model_revision = jc::FindString(text, "model_revision");
  if (!jc::FindNull(text, "adapter_revision")) {
    out->adapter_revision = jc::FindString(text, "adapter_revision");
  } else {
    out->adapter_revision.reset();
  }
  out->capture_policy_version = jc::FindString(text, "capture_policy_version");
  out->hook_name = jc::FindString(text, "hook_name");
  // Presence first: layer_number == -1 is legal (logits-style captures),
  // so FindInt's missing sentinel cannot stand in for absence.
  for (const char* name :
       {"layer_number", "producer_rank", "step_number", "token_start",
        "token_end", "batch_position", "captured_at_ns"}) {
    if (!jc::HasKey(text, name)) {
      return fail("capture metadata is missing an integer field");
    }
  }
  // A literal wider than 64 bits has no representation to validate: the old
  // scan wrapped it, so 2**64+1 arrived as captured_at_ns == 1 and was
  // packed. CaptureMetadata raises on these, so refuse the row instead.
  // kAbsent cannot mean "missing" here (HasKey already ran) -- it is a
  // present-but-not-an-integer value, which keeps its historical -1.
  bool out_of_range = false;
  auto integer = [&](const char* name) -> int64_t {
    int64_t value = 0;
    const jc::IntFind found = jc::FindIntChecked(text, name, &value);
    if (found == jc::IntFind::kOutOfRange) out_of_range = true;
    return found == jc::IntFind::kOk ? value : -1;
  };
  const int64_t layer = integer("layer_number");
  const int64_t producer = integer("producer_rank");
  const int64_t step = integer("step_number");
  const int64_t token_start = integer("token_start");
  const int64_t token_end = integer("token_end");
  const int64_t batch = integer("batch_position");
  const int64_t captured = integer("captured_at_ns");
  if (out_of_range) {
    return fail("capture metadata integer is out of range");
  }
  out->layer_number = layer;
  out->producer_rank = static_cast<uint64_t>(producer);
  out->step_number = static_cast<uint64_t>(step);
  out->token_start = static_cast<uint64_t>(token_start);
  out->token_end = static_cast<uint64_t>(token_end);
  out->batch_position = static_cast<uint64_t>(batch);
  out->dtype = jc::FindString(text, "dtype");
  out->shape.clear();
  for (const auto& item :
       jc::SplitElements(jc::Unwrap(jc::FindArray(text, "shape")))) {
    size_t q = 0;
    while (q < item.size() && item[q] == ' ') ++q;
    uint32_t v = 0;
    bool any = false;
    while (q < item.size() && item[q] >= '0' && item[q] <= '9') {
      v = v * 10 + static_cast<uint32_t>(item[q] - '0');
      ++q;
      any = true;
    }
    if (!any) return fail("capture shape must be an integer list");
    out->shape.push_back(v);
  }
  out->captured_at_ns = static_cast<uint64_t>(captured);
  if (ValidateMetadata(*out) != dmi_pack::Status::kOk) {
    return fail("capture metadata failed validation");
  }
  return true;
}

RowStatus SubmitRow(PackSink& sink, const RowInput& row, std::string* detail) {
  dmi_pack::RecordMetadata metadata;
  std::string error;
  if (!ParseMetadataJson(row.metadata_json, &metadata, &error)) {
    if (detail) *detail = error;
    return RowStatus::kBadMetadata;
  }
  if (DtypeWidth(row.dtype_name) == 0 || row.dtype_name != metadata.dtype) {
    if (detail) {
      *detail = "envelope dtype " + row.dtype_name + " != metadata dtype " +
                metadata.dtype;
    }
    return RowStatus::kDtypeMismatch;
  }
  if (row.shape.size() != metadata.shape.size()) {
    if (detail) *detail = "envelope rank != metadata rank";
    return RowStatus::kShapeMismatch;
  }
  uint64_t elements = 1;
  for (size_t i = 0; i < row.shape.size(); ++i) {
    if (row.shape[i] < 0 ||
        static_cast<uint64_t>(row.shape[i]) != metadata.shape[i]) {
      if (detail) *detail = "envelope shape != metadata shape";
      return RowStatus::kShapeMismatch;
    }
    elements *= static_cast<uint64_t>(row.shape[i]);
  }
  const uint64_t logical =
      elements * static_cast<uint64_t>(DtypeWidth(row.dtype_name));
  if (row.payload_bytes != logical) {
    if (detail) {
      *detail = "payload length does not match dtype and shape: " +
                std::to_string(row.payload_bytes) +
                " != " + std::to_string(logical);
    }
    return RowStatus::kSizeMismatch;
  }
  const Admission admission =
      sink.Submit(std::move(metadata), row.payload, row.payload_bytes);
  if (admission != Admission::kAccepted) {
    if (detail) *detail = AdmissionName(admission);
    return RowStatus::kNotAccepted;
  }
  return RowStatus::kOk;
}

}  // namespace dmi_sink
