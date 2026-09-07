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

[[noreturn]] void format_error(const std::string& what) {
  throw CatalogError(CatalogError::Kind::kValue, what);
}

std::string field_text(const std::string& object, const char* key,
                       const char* what) {
  if (!jc::HasKey(object, key)) {
    format_error(std::string("pack record is missing ") + what);
  }
  return jc::FindString(object, key);
}

int64_t field_int(const std::string& object, const char* key,
                  const char* what) {
  if (!jc::HasKey(object, key)) {
    format_error(std::string("pack record is missing ") + what);
  }
  return jc::FindInt(object, key);
}

// Bytes per element for the v1 dtype set (the Python metadata model's
// dtype table; the adapter's ATEN mapping pins the same ten).
size_t dtype_bytes(const std::string& dtype) {
  if (dtype == "bool" || dtype == "uint8" || dtype == "int8" ||
      dtype == "float8_e4m3fn" || dtype == "float8_e5m2") {
    return 1;
  }
  if (dtype == "uint16" || dtype == "int16" || dtype == "bfloat16" ||
      dtype == "float16") {
    return 2;
  }
  if (dtype == "uint32" || dtype == "int32" || dtype == "float32") {
    return 4;
  }
  if (dtype == "uint64" || dtype == "int64" || dtype == "float64") {
    return 8;
  }
  format_error("unknown dtype: " + dtype);
}

uint64_t shape_product(const std::string& shape_json) {
  uint64_t product = 1;
  // An empty shape array renders as [] → Unwrap → "" → SplitElements
  // yields ONE empty token, which parse would turn into 0 — turning every
  // scalar capture's logical_bytes to 0 and refusing the footer match.
  // A scalar is one element: no dimensions to multiply.
  if (jc::Unwrap(shape_json).empty()) return 1;
  for (const std::string& dim :
       jc::SplitElements(jc::Unwrap(shape_json))) {
    // Dims are JSON numbers; parse the raw text.
    uint64_t value = 0;
    bool negative = false, any = false;
    for (const char c : dim) {
      if (c == '-' && !any) {
        negative = true;
        continue;
      }
      if (c < '0' || c > '9') format_error("invalid shape dimension");
      any = true;
      value = value * 10 + static_cast<uint64_t>(c - '0');
    }
    if (!any) format_error("invalid shape dimension");
    if (negative) format_error("negative shape dimension");
    product *= value;
  }
  return product;
}

// The footer record's metadata object, validated by consumption: every
// column the catalog stores is read here, and a missing or ill-typed
// field is a format error at the boundary.
std::string render_record_row(const std::string& raw, const PackRefData& ref,
                              size_t footer_offset,
                              std::set<std::string>* seen_ids) {
  const std::string metadata = jc::FindObject(raw, "metadata");
  if (metadata.empty()) format_error("pack record metadata must be an object");
  const int64_t offset = field_int(raw, "offset", "offset");
  const int64_t stored = field_int(raw, "stored_length", "stored_length");
  const int64_t decoded = field_int(raw, "decoded_length", "decoded_length");
  const std::string codec = field_text(raw, "codec", "codec");
  const std::string checksum = field_text(raw, "checksum", "checksum");
  if (offset < static_cast<int64_t>(kHeaderSize) || stored < 0 || decoded < 0) {
    format_error("pack record range is invalid");
  }
  if (offset + stored > static_cast<int64_t>(footer_offset)) {
    format_error("pack record extends into the footer");
  }
  if (codec != "none" || stored != decoded) {
    format_error("dmi-pack-v1 supports only uncompressed records");
  }
  if (checksum.size() != 8) format_error("pack record checksum is invalid");
  for (const char c : checksum) {
    if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) {
      format_error("pack record checksum is invalid");
    }
  }

  const std::string capture_id = field_text(metadata, "capture_id", "capture_id");
  if (!seen_ids->insert(capture_id).second) {
    format_error("duplicate capture ID: " + capture_id);
  }
  const std::string dtype = field_text(metadata, "dtype", "dtype");
  const std::string shape = [&] {
    if (!jc::HasKey(metadata, "shape")) {
      format_error("pack record is missing shape");
    }
    return jc::FindArray(metadata, "shape");
  }();
  const uint64_t logical = shape_product(shape) * dtype_bytes(dtype);
  if (logical != static_cast<uint64_t>(decoded)) {
    format_error("record length does not match metadata dtype and shape");
  }

  std::vector<std::string> fields;
  const auto text_field = [&](const char* key) {
    fields.push_back(sql_quote(field_text(metadata, key, key)));
  };
  const auto int_field = [&](const char* key) {
    fields.push_back(std::to_string(field_int(metadata, key, key)));
  };
  text_field("capture_id");
  text_field("tenant_id");
  text_field("experiment_id");
  text_field("run_id");
  text_field("session_id");
  text_field("request_id");
  text_field("sequence_id");
  text_field("model_id");
  text_field("model_revision");
  fields.push_back(jc::FindNull(metadata, "adapter_revision")
                       ? "NULL"
                       : sql_quote(jc::FindString(metadata, "adapter_revision")));
  text_field("capture_policy_version");
  text_field("hook_name");
  fields.push_back(std::to_string(field_int(metadata, "layer_number", "layer_number")));
  int_field("producer_rank");
  int_field("step_number");
  int_field("token_start");
  int_field("token_end");
  int_field("batch_position");
  text_field("dtype");
  fields.push_back(shape);
  int_field("captured_at_ns");
  // Locator from the ref, record placement from the footer.
  fields.push_back(sql_uuid(ref.pack_id));
  fields.push_back(sql_quote(ref.store_id));
  fields.push_back(sql_quote(ref.object_key));
  fields.push_back(std::to_string(ref.object_bytes));
  fields.push_back(sql_quote(ref.checksum));
  fields.push_back(std::to_string(ref.record_count));
  fields.push_back(std::to_string(offset));
  fields.push_back(std::to_string(stored));
  fields.push_back(std::to_string(decoded));
  fields.push_back(sql_quote(codec));
  fields.push_back(sql_quote(checksum));

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
  const std::vector<std::string> records = jc::SplitElements(
      jc::Unwrap(jc::FindArray(footer_text, "records")));
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
    // The record's own range ordering is checked against the footer text;
    // duplicate capture IDs are caught by the renderer.
    const int64_t offset = jc::FindInt(raw, "offset");
    const int64_t stored = jc::FindInt(raw, "stored_length");
    if (static_cast<uint64_t>(offset) < previous_end) {
      format_error("pack record ranges overlap or are out of order");
    }
    previous_end = static_cast<uint64_t>(offset + stored);
    rows.push_back(render_record_row(raw, ref, footer_offset, &seen_ids));
  }
  return rows;
}

}  // namespace dmi_catalog
