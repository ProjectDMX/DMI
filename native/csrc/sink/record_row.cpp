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
  //
  // One latch across every identifier: an escape the decoder cannot turn into
  // a code point (non-hex \uXXXX digits, or a surrogate with no partner) is a
  // refusal, not a value. Both are things the oracle refuses -- json.loads
  // raises "Invalid \uXXXX escape" for the first and _validate_text's
  // .encode("utf-8") raises UnicodeEncodeError for the second -- while native
  // encoded them as U+25553 and as three-byte CESU-8, which ValidText admits
  // (it checks byte SHAPE, not the surrogate range) and the record was packed
  // and inserted.
  bool text_ok = true;
  const auto text_field = [&text, &text_ok](const char* name) {
    return jc::FindString(text, name, 0, &text_ok);
  };
  out->capture_id = text_field("capture_id");
  out->tenant_id = text_field("tenant_id");
  out->experiment_id = text_field("experiment_id");
  out->run_id = text_field("run_id");
  out->session_id = text_field("session_id");
  out->request_id = text_field("request_id");
  out->sequence_id = text_field("sequence_id");
  out->model_id = text_field("model_id");
  out->model_revision = text_field("model_revision");
  if (!jc::FindNull(text, "adapter_revision")) {
    out->adapter_revision = text_field("adapter_revision");
  } else {
    out->adapter_revision.reset();
  }
  out->capture_policy_version = text_field("capture_policy_version");
  out->hook_name = text_field("hook_name");
  if (!text_ok) {
    return fail("capture metadata text is not encodable UTF-8");
  }
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
  //
  // kAbsent cannot mean "missing" here (HasKey already ran) -- the key IS
  // there and simply is not an integer, so it is refused too rather than
  // taking the -1 it used to. -1 is layer_number's legal "no layer"
  // sentinel, so `"layer_number": null` and `"layer_number": "abc"` were
  // both admitted and packed as -1, where CaptureMetadata raises
  // "layer_number must be an integer in [-1, 2^31 - 1]". There is no
  // sentinel left in this parse: every one of the seven outcomes is now
  // either a value or a refusal.
  //
  // layer_number is the one field here the oracle types as SIGNED (Int32),
  // and it reads the int64 straight, so it must not be handed the unsigned
  // half of the union: 18446744073709551615 is the bit pattern -1, which is
  // this field's legal "no layer" sentinel, and ValidateMetadata -- which
  // only refuses < -1 or > 2**31 - 1 -- admits it. It is the SINGLE aliasing
  // input, since 2**64 - 2 lands on -2 and 2**63 on INT64_MIN, both refused.
  bool out_of_range = false;
  bool not_an_integer = false;
  auto integer = [&](const char* name,
                     jc::IntDomain domain = jc::IntDomain::kUnion) -> int64_t {
    int64_t value = 0;
    const jc::IntFind found = jc::FindIntChecked(text, name, &value, domain);
    if (found == jc::IntFind::kOutOfRange) out_of_range = true;
    if (found == jc::IntFind::kAbsent) not_an_integer = true;
    return found == jc::IntFind::kOk ? value : 0;
  };
  const int64_t layer = integer("layer_number", jc::IntDomain::kSigned);
  // The six counters are UNSIGNED in the oracle (UInt32/UInt64 columns, and
  // CaptureMetadata refuses anything below zero), so a negative literal is
  // refused while its sign is still in the text: cast afterwards,
  // step_number=-1 became 18446744073709551615 and was packed.
  const int64_t producer = integer("producer_rank", jc::IntDomain::kUnsigned);
  const int64_t step = integer("step_number", jc::IntDomain::kUnsigned);
  const int64_t token_start = integer("token_start", jc::IntDomain::kUnsigned);
  const int64_t token_end = integer("token_end", jc::IntDomain::kUnsigned);
  const int64_t batch = integer("batch_position", jc::IntDomain::kUnsigned);
  const int64_t captured = integer("captured_at_ns", jc::IntDomain::kUnsigned);
  // Out of range first: it is the more specific answer for a value that IS
  // an integer literal, and a row can only carry one refusal.
  if (out_of_range) {
    return fail("capture metadata integer is out of range");
  }
  if (not_an_integer) {
    return fail("capture metadata field is not an integer");
  }
  out->layer_number = layer;
  out->producer_rank = static_cast<uint64_t>(producer);
  out->step_number = static_cast<uint64_t>(step);
  out->token_start = static_cast<uint64_t>(token_start);
  out->token_end = static_cast<uint64_t>(token_end);
  out->batch_position = static_cast<uint64_t>(batch);
  out->dtype = jc::FindString(text, "dtype");
  out->shape.clear();
  // Presence and array-ness first: FindArray answers "" for a missing key
  // and for a non-array value alike, and Unwrap("") is also "" -- which is
  // exactly what a legal EMPTY array unwraps to. The two have to be told
  // apart before the split.
  const std::string shape_array = jc::FindArray(text, "shape");
  if (shape_array.empty()) {
    return fail("capture shape must be an integer list");
  }
  std::string shape_inside = jc::Unwrap(shape_array);
  {
    size_t start = 0, end = shape_inside.size();
    while (start < end && shape_inside[start] == ' ') ++start;
    while (end > start && shape_inside[end - 1] == ' ') --end;
    shape_inside = shape_inside.substr(start, end - start);
  }
  // `[]` is a legal shape: a rank-0 (scalar) capture of exactly one
  // element, which CaptureMetadata admits and the pack index already reads.
  // SplitElements yields ONE empty item for it, not zero dimensions, so the
  // empty case is taken before the split -- it used to be refused here as
  // "not an integer list" while the same record indexed fine.
  const std::vector<std::string> shape_items =
      shape_inside.empty() ? std::vector<std::string>()
                           : jc::SplitElements(shape_inside);
  for (const auto& item : shape_items) {
    size_t q = 0;
    while (q < item.size() && item[q] == ' ') ++q;
    // Bounded the way the scalars above are bounded, and for the same
    // reason: an unbounded accumulator wrapped, so shape [4294967297]
    // arrived as (1,) and was packed, where CaptureMetadata raises "shape
    // dimensions must be integers in [0, 2^31 - 1]". [2**31] was already
    // refused by ValidateMetadata, so only the WRAPPED dimensions slipped
    // through. The bound here is the accumulator's own, exactly as
    // FindIntChecked's is 64 bits: the field's narrower range stays with
    // ValidateMetadata, so one past 2**31 - 1 still reports as validation
    // and not as an unrepresentable literal.
    uint32_t v = 0;
    bool any = false;
    bool over = false;
    while (q < item.size() && item[q] >= '0' && item[q] <= '9') {
      const uint32_t digit = static_cast<uint32_t>(item[q] - '0');
      // "v * 10 + digit > UINT32_MAX", rearranged to not overflow itself.
      if (v > (~uint32_t{0} - digit) / 10) over = true;
      if (!over) v = v * 10 + digit;
      ++q;
      any = true;
    }
    // The whole item has to be the integer: `16.5` scanned as 16 and was
    // packed as (16,), where `type(dim) is not int` refuses it upstream.
    // Only trailing spaces may follow the digits.
    size_t tail = q;
    while (tail < item.size() && item[tail] == ' ') ++tail;
    if (!any || tail != item.size()) {
      return fail("capture shape must be an integer list");
    }
    if (over) return fail("capture shape dimension is out of range");
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
