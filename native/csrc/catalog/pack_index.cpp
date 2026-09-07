#include "pack_index.h"

#include <cstring>
#include <set>

#include "../common/json.h"
#include "../pack/pack_builder.h"
#include "catalog_writer.h"
#include "lease_coordinator.h"

namespace jc = dmi_common;

namespace dmi_catalog {

namespace {

constexpr size_t kHeaderSize = 64;   // <8sHHI16sQI20s
constexpr size_t kTrailerSize = 64;  // <8sHHQQI32s
constexpr uint64_t kMaxFooterBytes = 64ull * 1024 * 1024;
constexpr size_t kMaxRecords = 1'000'000;
constexpr uint64_t kUint64Max = ~0ull;
// The metadata model's own bounds (model.py `CaptureMetadata.__post_init__`):
// the catalog stores producer_rank/batch_position as UInt32, the counters and
// captured_at_ns as UInt64, layer_number as Int32 and shape as Array(UInt32)
// whose dimensions the reader re-checks against Int32's range.
constexpr size_t kMaxRank = 32;
constexpr size_t kTextLimit = 512;
constexpr uint64_t kMaxShapeDim = 0x7FFFFFFFull;    // 2^31 - 1
constexpr uint64_t kMaxUint32 = 0xFFFFFFFFull;      // 2^32 - 1
constexpr uint64_t kMaxLayerNumber = 0x7FFFFFFFull;  // 2^31 - 1

[[noreturn]] void format_error(const std::string& what) {
  throw CatalogError(CatalogError::Kind::kValue, what);
}

// `CaptureMetadata.from_mapping` wraps every ValueError `__post_init__`
// raises as `invalid capture metadata: {exc}`, so the refusals that come
// from the metadata model's own bounds carry that prefix and the ones that
// come from the record around it (`_parse_record`) do not.
[[noreturn]] void metadata_error(const std::string& what) {
  format_error("invalid capture metadata: " + what);
}

std::string field_text(const std::string& object, const char* key,
                       const char* what) {
  if (!jc::HasKey(object, key)) {
    format_error(std::string("pack record is missing ") + what);
  }
  return jc::FindString(object, key);
}

// A JSON integer literal with the range information Python's
// arbitrary-precision ints give the oracle for free. `ok` is false for
// anything that is not an integer literal — a string, a bool, a float —
// and `saturated` says the magnitude did not fit uint64, in which case
// `magnitude` is CLAMPED to its maximum so that a later bound comparison
// still refuses instead of wrapping back into range.
struct JsonInteger {
  bool ok = false;
  bool negative = false;
  bool saturated = false;
  uint64_t magnitude = 0;
};

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string trimmed(const std::string& text) {
  size_t start = 0, end = text.size();
  while (start < end && is_space(text[start])) ++start;
  while (end > start && is_space(text[end - 1])) --end;
  return text.substr(start, end - start);
}

JsonInteger parse_integer(const std::string& token) {
  JsonInteger out;
  const std::string text = trimmed(token);
  size_t i = 0;
  if (i < text.size() && text[i] == '-') {
    out.negative = true;
    ++i;
  }
  size_t digits = 0;
  for (; i < text.size(); ++i) {
    const char c = text[i];
    // '.', 'e', 'E' and '+' all land here: a JSON float is not an integer,
    // which is exactly what `type(value) is not int` says on the oracle's
    // side.
    if (c < '0' || c > '9') return JsonInteger();
    ++digits;
    if (out.saturated ||
        out.magnitude > (kUint64Max - static_cast<uint64_t>(c - '0')) / 10) {
      out.saturated = true;
      out.magnitude = kUint64Max;
    } else {
      out.magnitude = out.magnitude * 10 + static_cast<uint64_t>(c - '0');
    }
  }
  if (digits == 0) return JsonInteger();
  // JSON allows -0, and Python reads it as 0.
  if (out.magnitude == 0) out.negative = false;
  out.ok = true;
  return out;
}

// The raw value text for `key`, up to the next delimiter. Same
// first-occurrence-anywhere scan as `jc::FindInt`, but it hands back the
// literal so it can be range-checked rather than silently wrapped.
bool value_token(const std::string& object, const char* key,
                 std::string* out) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = std::string("\"") + key + "\"" + sep;
    const size_t at = object.find(needle);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    while (q < object.size() && is_space(object[q])) ++q;
    const size_t start = q;
    while (q < object.size() && object[q] != ',' && object[q] != '}' &&
           object[q] != ']') {
      ++q;
    }
    *out = object.substr(start, q - start);
    return true;
  }
  return false;
}

JsonInteger field_integer(const std::string& object, const char* key) {
  std::string token;
  if (!value_token(object, key, &token)) {
    format_error(std::string("pack record is missing ") + key);
  }
  return parse_integer(token);
}

// One metadata counter against the oracle's upper bound. `-1` and
// `2**32 + 9` are both refused here rather than wrapped into the column's
// range by ClickHouse, which accepts either silently.
uint64_t bounded_field(const std::string& metadata, const char* key,
                       uint64_t upper) {
  const JsonInteger value = field_integer(metadata, key);
  if (!value.ok || value.negative || value.saturated ||
      value.magnitude > upper) {
    metadata_error(std::string(key) + " must be a non-negative integer <= " +
                   std::to_string(upper));
  }
  return value.magnitude;
}

// One identifier: non-empty and within the text limit, measured in UTF-8
// bytes exactly as `_validate_text` measures it. `optional` in
// `_validate_text` covers a value of None and nothing else, so it has no
// counterpart here: adapter_revision's absent form is a JSON null, read
// before this is reached, and its empty-STRING form is refused like any
// other.
std::string checked_text(const std::string& metadata, const char* key) {
  const std::string value = field_text(metadata, key, key);
  if (value.empty() || value.size() > kTextLimit) {
    metadata_error(std::string(key) + " must be non-empty UTF-8 within " +
                   std::to_string(kTextLimit) + " bytes");
  }
  return value;
}

// Bytes per element for the v1 dtype set — the Python metadata model's
// `_DTYPE_BYTES` table, which is the same ten the pack builder pins in
// `kDtypes` and the adapter's ATEN mapping pins again. A dtype outside it
// is refused here, not sized by a wider table: the Python reader rebuilds
// `CaptureMetadata` from the row this renders and would refuse the dtype on
// read-back.
size_t dtype_bytes(const std::string& dtype) {
  if (dtype == "bool" || dtype == "uint8" || dtype == "int8") return 1;
  if (dtype == "int16" || dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "int32" || dtype == "float32") return 4;
  if (dtype == "int64" || dtype == "float64") return 8;
  metadata_error("unsupported dtype: '" + dtype + "'");
}

// The shape array as an integer list, which is all `from_mapping` checks
// before the model itself looks at rank and per-dimension bounds — and it
// checks it of the WHOLE list first, so the refusals have to come out in
// that order too.
std::vector<JsonInteger> parse_shape(const std::string& metadata) {
  if (!jc::HasKey(metadata, "shape")) {
    format_error("pack record is missing shape");
  }
  const std::string array = jc::FindArray(metadata, "shape");
  if (array.empty()) format_error("capture shape must be an integer list");
  std::vector<JsonInteger> dims;
  const std::string inside = trimmed(jc::Unwrap(array));
  // `[]` is a legal shape: a rank-0 capture whose logical size is one
  // element. SplitElements yields a single empty element for it, so the
  // empty case is taken before the split rather than parsed as a dimension.
  if (inside.empty()) return dims;
  for (const std::string& raw : jc::SplitElements(inside)) {
    const JsonInteger dim = parse_integer(raw);
    if (!dim.ok) format_error("capture shape must be an integer list");
    dims.push_back(dim);
  }
  return dims;
}

// prod(shape) * itemsize, overflow-CHECKED. Thirty-two dimensions of
// 2^31-1 is a shape the model admits and whose product does not fit
// uint64; a wrapping product can land back on decoded_length and admit a
// record whose metadata describes something else entirely.
bool logical_bytes(const std::vector<JsonInteger>& dims, size_t element_bytes,
                   uint64_t* out) {
  uint64_t product = element_bytes;
  for (const JsonInteger& dim : dims) {
    if (dim.magnitude == 0) {
      product = 0;
      continue;
    }
    if (product > kUint64Max / dim.magnitude) return false;
    product *= dim.magnitude;
  }
  *out = product;
  return true;
}

bool is_json_object(const std::string& text) {
  return text.size() >= 2 && text.front() == '{' && text.back() == '}';
}

// Whether `text` is a well-formed JSON value of some shape OTHER than an
// object — which is the difference between the oracle's "not valid JSON"
// and its "invalid format marker". A number is recognised in its integer
// form only; a float footer would be called malformed instead of
// wrong-shaped, and both sides refuse it either way.
bool looks_like_json_value(const std::string& text) {
  if (text.empty()) return false;
  if (text.front() == '[') return text.back() == ']';
  if (text.front() == '"') return text.size() >= 2 && text.back() == '"';
  if (text == "true" || text == "false" || text == "null") return true;
  return parse_integer(text).ok;
}

// Where one record's bytes sit, handed back so the caller can check the
// footer's range ORDERING after the record itself has been validated.
struct RecordRange {
  uint64_t offset = 0;
  uint64_t end = 0;
};

// The footer record's metadata object, validated by consumption: every
// column the catalog stores is read here, and a missing or ill-typed
// field is a format error at the boundary.
std::string render_record_row(const std::string& raw, const PackRefData& ref,
                              uint64_t footer_offset,
                              std::set<std::string>* seen_ids,
                              RecordRange* range) {
  const std::string metadata = jc::FindObject(raw, "metadata");
  if (metadata.empty()) format_error("pack record metadata must be an object");
  const JsonInteger offset = field_integer(raw, "offset");
  const JsonInteger stored = field_integer(raw, "stored_length");
  const JsonInteger decoded = field_integer(raw, "decoded_length");
  const std::string codec = field_text(raw, "codec", "codec");
  const std::string checksum = field_text(raw, "checksum", "checksum");
  if (!offset.ok || offset.negative || offset.magnitude < kHeaderSize ||
      !stored.ok || stored.negative || !decoded.ok || decoded.negative) {
    format_error("pack record range is invalid");
  }
  // Checked addition. `offset + stored` on int64 overflows for a footer
  // naming an offset near int64's maximum, and signed overflow is
  // undefined: at -O2 the wrapped sum compared BELOW footer_offset and the
  // record was admitted.
  if (stored.magnitude > kUint64Max - offset.magnitude ||
      offset.magnitude + stored.magnitude > footer_offset) {
    format_error("pack record extends into the footer");
  }
  if (codec != "none" || stored.magnitude != decoded.magnitude) {
    format_error("dmi-pack-v1 supports only uncompressed records");
  }
  if (checksum.size() != 8) format_error("pack record checksum is invalid");
  for (const char c : checksum) {
    if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) {
      format_error("pack record checksum is invalid");
    }
  }

  // The metadata bounds, in the order CaptureMetadata applies them. The
  // footer is EXTERNAL data re-read at query time, long after and by
  // another process than whatever wrote it, so these are checked here and
  // not merely trusted from the builder that sealed the pack: one
  // out-of-range value ClickHouse would wrap silently is one poison row
  // the Python reader refuses for the whole page it lands on.
  const std::vector<JsonInteger> dims = parse_shape(metadata);
  const std::string capture_id = checked_text(metadata, "capture_id");
  const std::string tenant_id = checked_text(metadata, "tenant_id");
  const std::string experiment_id = checked_text(metadata, "experiment_id");
  const std::string run_id = checked_text(metadata, "run_id");
  const std::string session_id = checked_text(metadata, "session_id");
  const std::string request_id = checked_text(metadata, "request_id");
  const std::string sequence_id = checked_text(metadata, "sequence_id");
  const std::string model_id = checked_text(metadata, "model_id");
  const std::string model_revision = checked_text(metadata, "model_revision");
  const std::string capture_policy_version =
      checked_text(metadata, "capture_policy_version");
  const std::string hook_name = checked_text(metadata, "hook_name");
  const bool no_adapter_revision = jc::FindNull(metadata, "adapter_revision");
  const std::string adapter_revision =
      no_adapter_revision ? std::string()
                          : checked_text(metadata, "adapter_revision");
  const std::string dtype = field_text(metadata, "dtype", "dtype");
  const size_t element_bytes = dtype_bytes(dtype);
  if (dims.size() > kMaxRank) {
    metadata_error("shape rank must not exceed " + std::to_string(kMaxRank));
  }
  for (const JsonInteger& dim : dims) {
    if (dim.negative || dim.saturated || dim.magnitude > kMaxShapeDim) {
      metadata_error("shape dimensions must be integers in [0, 2^31 - 1]");
    }
  }
  const uint64_t producer_rank =
      bounded_field(metadata, "producer_rank", kMaxUint32);
  const uint64_t batch_position =
      bounded_field(metadata, "batch_position", kMaxUint32);
  const uint64_t step_number =
      bounded_field(metadata, "step_number", kUint64Max);
  const uint64_t token_start =
      bounded_field(metadata, "token_start", kUint64Max);
  const uint64_t token_end = bounded_field(metadata, "token_end", kUint64Max);
  const uint64_t captured_at_ns =
      bounded_field(metadata, "captured_at_ns", kUint64Max);
  const JsonInteger layer = field_integer(metadata, "layer_number");
  if (!layer.ok || layer.saturated ||
      (layer.negative ? layer.magnitude > 1
                      : layer.magnitude > kMaxLayerNumber)) {
    metadata_error("layer_number must be an integer in [-1, 2^31 - 1]");
  }
  if (token_end < token_start) {
    metadata_error("token_end must be >= token_start");
  }

  uint64_t logical = 0;
  if (!logical_bytes(dims, element_bytes, &logical) ||
      logical != decoded.magnitude) {
    format_error("record length does not match metadata dtype and shape");
  }
  // AFTER the record is otherwise valid, and before the caller's range
  // ordering: _parse_records validates the record, then refuses a repeated
  // capture id, then refuses an out-of-order range. Checked first, a
  // duplicate came out as an overlap with the record it repeats.
  if (!seen_ids->insert(capture_id).second) {
    format_error("duplicate capture ID: " + capture_id);
  }

  std::vector<std::string> fields;
  fields.push_back(sql_quote(capture_id));
  fields.push_back(sql_quote(tenant_id));
  fields.push_back(sql_quote(experiment_id));
  fields.push_back(sql_quote(run_id));
  fields.push_back(sql_quote(session_id));
  fields.push_back(sql_quote(request_id));
  fields.push_back(sql_quote(sequence_id));
  fields.push_back(sql_quote(model_id));
  fields.push_back(sql_quote(model_revision));
  fields.push_back(no_adapter_revision ? "NULL"
                                       : sql_quote(adapter_revision));
  fields.push_back(sql_quote(capture_policy_version));
  fields.push_back(sql_quote(hook_name));
  fields.push_back(layer.negative
                       ? "-" + std::to_string(layer.magnitude)
                       : std::to_string(layer.magnitude));
  fields.push_back(std::to_string(producer_rank));
  fields.push_back(std::to_string(step_number));
  fields.push_back(std::to_string(token_start));
  fields.push_back(std::to_string(token_end));
  fields.push_back(std::to_string(batch_position));
  fields.push_back(sql_quote(dtype));
  fields.push_back(jc::FindArray(metadata, "shape"));
  fields.push_back(std::to_string(captured_at_ns));
  // Locator from the ref, record placement from the footer.
  fields.push_back(sql_uuid(ref.pack_id));
  fields.push_back(sql_quote(ref.store_id));
  fields.push_back(sql_quote(ref.object_key));
  fields.push_back(std::to_string(ref.object_bytes));
  fields.push_back(sql_quote(ref.checksum));
  fields.push_back(std::to_string(ref.record_count));
  fields.push_back(std::to_string(offset.magnitude));
  fields.push_back(std::to_string(stored.magnitude));
  fields.push_back(std::to_string(decoded.magnitude));
  fields.push_back(sql_quote(codec));
  fields.push_back(sql_quote(checksum));

  range->offset = offset.magnitude;
  range->end = offset.magnitude + stored.magnitude;

  std::string row;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) row += ",";
    row += fields[i];
  }
  return row;
}

}  // namespace

std::vector<std::string> read_pack_descriptor_rows(
    dmi_store::S3Client* s3, const PackRefData& ref) {
  if (ref.object_bytes < kHeaderSize + kTrailerSize + 2) {
    format_error("pack is truncated");
  }
  std::string error;
  std::vector<uint8_t> trailer;
  const uint64_t trailer_offset = ref.object_bytes - kTrailerSize;
  if (!s3->GetRange(ref.object_key, trailer_offset, kTrailerSize, &trailer,
                   &error)) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "pack trailer read failed: " + error);
  }
  if (trailer.size() != kTrailerSize) format_error("pack trailer is truncated");
  const auto read_u64 = [&](size_t at) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) {
      v |= static_cast<uint64_t>(trailer[at + i]) << (8 * i);
    }
    return v;
  };
  const auto read_u16 = [&](size_t at) {
    return static_cast<uint16_t>(trailer[at]) |
           (static_cast<uint16_t>(trailer[at + 1]) << 8);
  };
  const auto read_u32 = [&](size_t at) {
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) {
      v |= static_cast<uint32_t>(trailer[at + i]) << (8 * i);
    }
    return v;
  };
  // Compare with memcmp: the magic carries embedded NULs, and a
  // const-char* comparison would truncate at the first one.
  const char kTrailerMagic[8] = {'D', 'M', 'I', 'F', 'T', 'R', '\0', '\0'};
  if (std::memcmp(trailer.data(), kTrailerMagic, 8) != 0) {
    format_error("pack has an invalid trailer");
  }
  const uint16_t major = read_u16(8), minor = read_u16(10);
  const uint64_t footer_offset = read_u64(12);
  const uint64_t footer_length = read_u64(20);
  const uint32_t footer_crc = read_u32(28);
  if (major != 1 || minor > 0) {
    format_error("unsupported pack version");
  }
  if (footer_length > kMaxFooterBytes) {
    format_error("pack footer exceeds its size limit");
  }
  if (footer_offset < kHeaderSize ||
      footer_offset + footer_length != trailer_offset) {
    format_error("pack footer range is invalid");
  }
  std::vector<uint8_t> footer;
  if (!s3->GetRange(ref.object_key, footer_offset, footer_length, &footer,
                   &error)) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "pack footer read failed: " + error);
  }
  if (dmi_pack::Crc32(footer.data(), footer.size()) != footer_crc) {
    throw CatalogError(CatalogError::Kind::kValue, "footer checksum mismatch");
  }
  const std::string footer_text(reinterpret_cast<const char*>(footer.data()),
                                footer.size());
  // json.loads() refuses malformed text OUTRIGHT, and only text it decoded
  // reaches the shape check — so "is not valid JSON" and "has an invalid
  // format marker" are two different refusals on the oracle's side. This
  // port has no JSON parser and separates them structurally instead: a JSON
  // object opens with `{` and closes with `}`; a well-formed JSON value of
  // any other shape opens with one of the remaining value starters and
  // closes as that shape closes; anything else never decoded at all. The
  // one case the split cannot reach is a malformed object BODY, which opens
  // and closes like an object and falls through to the format-marker
  // refusal — both sides refuse it, only the sentence differs.
  const std::string footer_json = trimmed(footer_text);
  if (!is_json_object(footer_json)) {
    if (looks_like_json_value(footer_json)) {
      format_error("pack footer has an invalid format marker");
    }
    format_error("pack footer is not valid JSON");
  }
  if (jc::FindString(footer_text, "format") != "dmi-pack") {
    format_error("pack footer has an invalid format marker");
  }
  if (jc::FindInt(footer_text, "major_version") != major ||
      jc::FindInt(footer_text, "minor_version") != minor) {
    format_error("pack footer version does not match the trailer");
  }
  const std::string footer_pack_id = jc::FindString(footer_text, "pack_id");
  if (footer_pack_id != ref.pack_id) {
    format_error("pack footer identity does not match its object key");
  }
  // A `records` value that is not an array at all is an invalid record
  // list, which is what `isinstance(raw_records, list)` says. FindArray
  // returns nothing for a non-array, and splitting that yields one empty
  // "record" — blamed on the record's metadata, or on the ref's record
  // count, rather than on the list.
  const std::string records_array = jc::FindArray(footer_text, "records");
  if (records_array.empty()) {
    format_error("pack footer has an invalid record list");
  }
  const std::vector<std::string> records =
      jc::SplitElements(jc::Unwrap(records_array));
  if (records.size() > kMaxRecords) {
    format_error("pack footer has an invalid record list");
  }
  if (records.size() != ref.record_count) {
    format_error("pack record count does not match its object metadata");
  }
  std::vector<std::string> rows;
  std::set<std::string> seen_ids;
  uint64_t previous_end = kHeaderSize;
  for (const std::string& raw : records) {
    // Range ordering is checked AFTER the record itself, because that is
    // the order _parse_records refuses in: an offset of 0 in the first
    // record is an invalid range, not an overlap with the header.
    RecordRange range;
    rows.push_back(
        render_record_row(raw, ref, footer_offset, &seen_ids, &range));
    if (range.offset < previous_end) {
      format_error("pack record ranges overlap or are out of order");
    }
    previous_end = range.end;
  }
  return rows;
}

}  // namespace dmi_catalog
