// One capture row: metadata JSON + sliced payload bytes -> PackSink.
//
// Torch-free: the ATen-dependent adapter (native_pack_sink) unpacks the
// envelope into this plain form, so all validation and submission logic is
// unit-testable without torch. Mirrors record_adapter.py row handling plus
// the CaptureRecord constructor checks.

#ifndef DMI_SINK_RECORD_ROW_H_
#define DMI_SINK_RECORD_ROW_H_

#include <cstdint>
#include <string>
#include <vector>

#include "../pack/pack_builder.h"

namespace dmi_sink {

class PackSink;  // pack_sink.h (included by the .cpp, not here)

// One row's inputs, already sliced: payload holds exactly the capture bytes.
struct RowInput {
  std::string metadata_json;
  const uint8_t* payload = nullptr;
  size_t payload_bytes = 0;
  // Declared by the envelope (ATen side), cross-checked against metadata.
  std::string dtype_name;
  std::vector<int64_t> shape;
};

enum class RowStatus {
  kOk = 0,
  kBadMetadata,   // JSON does not parse to valid metadata
  kDtypeMismatch, // envelope dtype != metadata dtype (or unknown)
  kShapeMismatch, // envelope shape != metadata shape
  kSizeMismatch,  // payload bytes != dtype*shape logical bytes
  kNotAccepted,   // sink refused durable admission
};

inline const char* RowStatusName(RowStatus s) {
  switch (s) {
    case RowStatus::kOk: return "ok";
    case RowStatus::kBadMetadata: return "invalid capture metadata";
    case RowStatus::kDtypeMismatch: return "dtype does not match metadata";
    case RowStatus::kShapeMismatch: return "shape does not match metadata";
    case RowStatus::kSizeMismatch: return "payload bytes do not match metadata";
    case RowStatus::kNotAccepted: return "sink refused durable admission";
  }
  return "unknown";
}

// Parse a canonical metadata JSON object into RecordMetadata. Field names
// match CaptureMetadata.to_mapping(); shape is an integer array;
// adapter_revision may be null.
bool ParseMetadataJson(const std::string& text, dmi_pack::RecordMetadata* out,
                       std::string* error);

// Bytes per element for the ten supported dtype names; 0 when unknown.
int DtypeWidth(const std::string& dtype_name);

// Validate the row against its metadata and submit for durable admission.
// On kNotAccepted, `detail` carries the sink's admission name.
RowStatus SubmitRow(PackSink& sink, const RowInput& row, std::string* detail);

}  // namespace dmi_sink

#endif  // DMI_SINK_RECORD_ROW_H_
