#include "hydration.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstring>
#include <set>

#include "../common/json.h"
#include "../pack/pack_builder.h"
#include "catalog_writer.h"
#include "pack_index.h"
#include "lease_coordinator.h"

namespace jc = dmi_common;

namespace dmi_catalog {

namespace {

constexpr size_t kTrailerBytes = 64;


std::string json_escape(const std::string& value) {
  std::string out;
  jc::EscapeJson(value, &out);
  return out;
}

// Bytes per element for the v1 dtype set (dtype_bytes in pack_index).
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
  throw CatalogError(CatalogError::Kind::kValue,
                     "unsupported dtype: " + dtype);
}

bool is_float_dtype(const std::string& dtype) {
  // float8 is a float: NaN (e4m3fn and e5m2) and Inf (e5m2) ride the
  // NaN/Inf counting paths, and the order statistics stay float.
  return dtype == "float16" || dtype == "bfloat16" || dtype == "float32" ||
         dtype == "float64" || dtype == "float8_e4m3fn" ||
         dtype == "float8_e5m2";
}

// One element widened to double. bfloat16 is read as uint16 and widened to
// float32 by a 16-bit left shift — exact for every bit pattern. float16 is
// an IEEE half.
double decode_element(const std::string& dtype, const uint8_t* data,
                      size_t offset) {
  auto u16 = [&] {
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1]) << 8);
  };
  auto u32 = [&] {
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) {
      v |= static_cast<uint32_t>(data[offset + i]) << (8 * i);
    }
    return v;
  };
  auto u64 = [&] {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) {
      v |= static_cast<uint64_t>(data[offset + i]) << (8 * i);
    }
    return v;
  };
  float f32 = 0.0f;
  double f64 = 0.0;
  if (dtype == "uint8") return data[offset];
  if (dtype == "int8") return static_cast<int8_t>(data[offset]);
  if (dtype == "bool") return data[offset] != 0 ? 1 : 0;
  if (dtype == "uint16") return u16();
  if (dtype == "int16") return static_cast<int16_t>(u16());
  if (dtype == "uint32") return u32();
  if (dtype == "int32") return static_cast<int32_t>(u32());
  if (dtype == "uint64") return static_cast<double>(u64());
  if (dtype == "int64") return static_cast<double>(static_cast<int64_t>(u64()));
  if (dtype == "float32") {
    std::memcpy(&f32, data + offset, 4);
    return static_cast<double>(f32);
  }
  if (dtype == "float64") {
    std::memcpy(&f64, data + offset, 8);
    return f64;
  }
  if (dtype == "bfloat16") {
    // Read as uint16, widen to float32 by a 16-bit left shift.
    const uint32_t bits = static_cast<uint32_t>(u16()) << 16;
    std::memcpy(&f32, &bits, 4);
    return static_cast<double>(f32);
  }
  if (dtype == "float16") {
    // IEEE half to float, bit-exact.
    const uint16_t bits = u16();
    const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
    const uint32_t exponent = (bits >> 10) & 0x1F;
    const uint32_t mantissa = bits & 0x3FF;
    uint32_t out;
    if (exponent == 0) {
      // Zero or subnormal. A subnormal half is ±mantissa × 2⁻²⁴ -- a
      // perfectly NORMAL float32 -- so compute the value instead of
      // assembling bits. The previous assembly copied the low four bytes
      // of a DOUBLE into the bit field and dropped the sign: 0x0001
      // (2⁻²⁴) summarized as ~-1.2e19 and pulled the whole mean with it.
      const float value =
          static_cast<float>(mantissa) * 5.9604644775390625e-08f;  // 2^-24
      return (bits >> 15) != 0 ? -static_cast<double>(value)
                               : static_cast<double>(value);
    } else if (exponent == 0x1F) {
      out = sign | 0x7F800000u | (mantissa ? 0x400000u : 0);
    } else {
      out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    std::memcpy(&f32, &out, 4);
    return static_cast<double>(f32);
  }
  if (dtype == "float8_e4m3fn" || dtype == "float8_e5m2") {
    // OCP float8, widened to float32 by bit math — exact for every
    // pattern, matching summary.py's _decode_float8 bit for bit.
    // e4m3fn: 1/4/3, bias 7, no infinities (all-ones exponent is NaN).
    // e5m2: IEEE-shaped 1/5/2, bias 15, with infinities.
    const uint8_t byte = data[offset];
    double sign = byte >= 128 ? -1.0 : 1.0;
    const uint32_t body = byte & 0x7F;
    if (dtype == "float8_e4m3fn") {
      // e4m3fn: exponent 15 is NaN ONLY for mantissa 7; mantissas 0-6 are
      // the finite top-of-range values 256..448 (the "fn" — finite-only —
      // variant has no infinities). Measured against torch's
      // float8_e4m3fn view as the oracle.
      const uint32_t exponent = body >> 3;
      const double mantissa = static_cast<double>(body & 0x7);
      if (exponent == 15 && mantissa == 7) return std::nan("");
      if (exponent == 0) {
        return sign * std::ldexp(mantissa / 8.0, -6);
      }
      return sign * std::ldexp(1.0 + mantissa / 8.0,
                               static_cast<int>(exponent) - 7);
    }
    const uint32_t exponent = body >> 2;
    const double mantissa = static_cast<double>(body & 0x3);
    if (exponent == 31) {
      if (mantissa == 0) return sign * std::numeric_limits<double>::infinity();
      return std::nan("");
    }
    if (exponent == 0) return sign * std::ldexp(mantissa / 4.0, -14);
    return sign * std::ldexp(1.0 + mantissa / 4.0,
                             static_cast<int>(exponent) - 15);
    // e5m2 is IEEE-shaped and verified against torch in the parity gate.
  }
  throw CatalogError(CatalogError::Kind::kValue,
                     "unsupported dtype: " + dtype);
}

// One element as an EXACT int64, for the integer order statistics only.
// Routing them through decode_element's double is wrong twice over above
// 2**53: the value rounds, and static_cast<int64_t> of 2**63 is undefined
// (INT64_MIN on x86-64), so the strictly positive [2**63-2, 2**63-1]
// summarised maximally negative. summary.py takes int(flat.min()) off the
// raw dtype, so this reads the raw bytes and sign- or zero-extends them.
// The accumulating statistics (sum, scale, l2) keep decode_element: those
// legitimately want float64.
int64_t decode_element_int(const std::string& dtype, const uint8_t* data,
                           size_t offset) {
  auto u16 = [&] {
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1]) << 8);
  };
  auto u32 = [&] {
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) {
      v |= static_cast<uint32_t>(data[offset + i]) << (8 * i);
    }
    return v;
  };
  auto u64 = [&] {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) {
      v |= static_cast<uint64_t>(data[offset + i]) << (8 * i);
    }
    return v;
  };
  if (dtype == "uint8") return data[offset];
  if (dtype == "int8") return static_cast<int8_t>(data[offset]);
  if (dtype == "bool") return data[offset] != 0 ? 1 : 0;
  if (dtype == "uint16") return u16();
  if (dtype == "int16") return static_cast<int16_t>(u16());
  if (dtype == "uint32") return u32();
  if (dtype == "int32") return static_cast<int32_t>(u32());
  if (dtype == "int64") return static_cast<int64_t>(u64());
  // uint64 is refused rather than represented. It is not in the
  // authoritative dtype set (model.py's _DTYPE_BYTES, mirrored by
  // pack_builder.cpp's kDtypes), so Python's reader refuses such a
  // descriptor outright when it builds CaptureMetadata — there is no
  // oracle to match — and an int64 order statistic cannot name a uint64
  // above 2**63-1 anyway. Failing closed is the parity answer.
  throw CatalogError(
      CatalogError::Kind::kValue,
      "unsupported dtype for integer order statistics: " + dtype);
}

// Field positions in the 32-field layout the reader rows use:
// [0..4] sort key (capture_id sits at 4), then the 27 resolved columns.
constexpr size_t kCaptureId = 4, kAdapterRevision = 10, kDtype = 19,
                 kShape = 20, kPackId = 21, kStoreId = 22, kObjectKey = 23,
                 kObjectBytes = 24, kPackChecksum = 25, kPackRecordCount = 26,
                 kPayloadOffset = 27, kStoredLength = 28, kDecodedLength = 29,
                 kCodec = 30, kPayloadChecksum = 31;


// The two row layouts, by field name: the catalog rows are sort-key-first
// (the reader's search/get_by_ids projection); the footer rows are
// CAPTURE_COLUMNS order minus index_version. Built once, verified
// complete at first use — a field missing from either side is a build
// error in the making, so it refuses loudly instead.
const std::vector<std::pair<size_t, size_t>>& kCatalogToFooter() {
  static const std::vector<std::pair<size_t, size_t>> mapping = [] {
    // NOTE: capture_id appears once in the catalog layout at index 4
    // (the sort-key position); the resolved list in the reader skips
    // sort-key columns. The footer layout is CAPTURE_COLUMNS order with
    // capture_id at 0 and captured_at_ns at 20.
    const char* catalog_names_fixed[32] = {
        "tenant_id", "experiment_id", "run_id", "captured_at_ns",
        "capture_id",
        "session_id",   "request_id",   "sequence_id",
        "model_id",    "model_revision", "adapter_revision",
        "capture_policy_version", "hook_name",   "layer_number",
        "producer_rank", "step_number", "token_start", "token_end",
        "batch_position", "dtype",      "shape",
        "pack_id",     "store_id",     "object_key",   "object_bytes",
        "pack_checksum", "pack_record_count", "payload_offset",
        "stored_length", "decoded_length", "codec", "payload_checksum"};
    const char* footer_names[32] = {
        "capture_id",  "tenant_id",      "experiment_id", "run_id",
        "session_id",  "request_id",     "sequence_id",   "model_id",
        "model_revision", "adapter_revision",
        "capture_policy_version", "hook_name",   "layer_number",
        "producer_rank", "step_number", "token_start", "token_end",
        "batch_position", "dtype",      "shape",
        "captured_at_ns", "pack_id",    "store_id",
        "object_key",  "object_bytes",   "pack_checksum",
        "pack_record_count", "payload_offset", "stored_length",
        "decoded_length", "codec", "payload_checksum"};
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t c = 0; c < 32; ++c) {
      for (size_t f = 0; f < 32; ++f) {
        if (std::strcmp(catalog_names_fixed[c], footer_names[f]) == 0) {
          out.emplace_back(c, f);
          break;
        }
      }
      if (out.size() < c + 1) {
        // A field with no footer counterpart: the layouts have drifted.
        throw std::runtime_error(
            std::string("catalog field has no footer counterpart: ") +
            catalog_names_fixed[c]);
      }
    }
    return out;
  }();
  return mapping;
}

// The body of one SQL string literal, decoded. `at` is on the opening
// quote on entry and just past the closing quote on return. The escape
// set is sql_quote's map, entry for entry.
std::string unquote_sql(const std::string& row, size_t* at) {
  std::string out;
  size_t i = *at + 1;
  while (i < row.size()) {
    const char c = row[i];
    if (c == '\\' && i + 1 < row.size()) {
      const char next = row[i + 1];
      switch (next) {
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'r': out.push_back('\r'); break;
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case '0': out.push_back('\0'); break;
        case 'a': out.push_back('\a'); break;
        case 'v': out.push_back('\v'); break;
        default: out.push_back(next); break;  // \\ and \' decode to themselves
      }
      i += 2;
      continue;
    }
    if (c == '\'') {
      *at = i + 1;
      return out;
    }
    out.push_back(c);
    ++i;
  }
  throw CatalogError(CatalogError::Kind::kValue,
                     "pack footer row has an unterminated string");
}

// Split one rendered footer row into its typed fields on TOP-LEVEL commas.
// Top-level means depth-aware: a rank>=2 shape renders as [2,8] with an
// unquoted comma inside, and splitting on it shifted every later field
// (reproduced with float32 shaped (2,8) and (2,2,4)).
//
// Why typed, and why decoded: the catalog row this is compared against is
// already DECODED text (the reader's TSV layer undid ClickHouse's escapes),
// so the footer side has to reach the same representation. Comparing the
// footer's escaped text against it refused three valid hook names --
// block\resid, block'quoted and a tab-bearing one -- as "does not match
// the pack footer". And NULL has to stay a kind rather than a spelling: a
// normalisation that turned BOTH the unquoted NULL token and the quoted
// string 'NULL' into the same value accepted a catalog row whose
// adapter_revision was SQL NULL against a footer whose value was the
// four-letter string.
}  // namespace

std::vector<FooterField> split_footer_row(const std::string& rendered) {
  std::vector<FooterField> fields;
  size_t i = 0;
  for (;;) {
    FooterField field;
    while (i < rendered.size() && rendered[i] == ' ') ++i;
    if (i < rendered.size() && rendered[i] == '\'') {
      field.text = unquote_sql(rendered, &i);
    } else if (rendered.compare(i, 8, "toUUID('") == 0) {
      // UUID-typed columns render as toUUID('...') -- SQL the INSERT path
      // needs -- while the catalog carries the bare UUID string.
      i += 7;
      field.text = unquote_sql(rendered, &i);
      if (i >= rendered.size() || rendered[i] != ')') {
        throw CatalogError(CatalogError::Kind::kValue,
                           "pack footer row has a malformed UUID literal");
      }
      ++i;
    } else {
      // A bare token: NULL, a number, or a bracketed array. Spaces inside
      // it carry no meaning (an array renders with or without them
      // depending on the writer's JSON separators), so they are dropped.
      int depth = 0;
      for (; i < rendered.size() && !(rendered[i] == ',' && depth == 0);
           ++i) {
        const char c = rendered[i];
        if (c == '[') ++depth;
        if (c == ']') --depth;
        if (c != ' ') field.text.push_back(c);
      }
      if (field.text == "NULL") {
        field.is_null = true;
        field.text.clear();
      }
    }
    while (i < rendered.size() && rendered[i] == ' ') ++i;
    fields.push_back(std::move(field));
    if (i >= rendered.size()) break;
    if (rendered[i] != ',') {
      throw CatalogError(CatalogError::Kind::kValue,
                         "pack footer row has an unexpected token");
    }
    ++i;
  }
  return fields;
}

// Whether one catalog field agrees with its footer counterpart. The catalog
// rows carry a NULL as the empty string (the reader maps TSV's \N to
// absent), and an empty string is not a value either side admits for any
// text column, so nullness on the catalog side IS emptiness -- while on the
// footer side it is the unquoted token and nothing else. A quoted 'NULL'
// is the four-letter string, compared as such.
bool footer_field_matches(const std::string& catalog_value,
                          const FooterField& footer_value) {
  if (footer_value.is_null) return catalog_value.empty();
  if (catalog_value.empty()) return false;
  return catalog_value == footer_value.text;
}

namespace {

uint64_t shape_product(const std::string& shape_text) {
  uint64_t product = 1;
  // `[]` is the rank-0 shape: one element. SplitElements hands back ONE
  // empty item for it, which parse_u64_field then refuses -- so a scalar
  // that staged, uploaded, indexed and hydrated still failed to summarise
  // with "payload length does not match dtype and shape". The empty case
  // is the product's identity and is taken before the split.
  std::string inside = jc::Unwrap(shape_text);
  size_t start = 0, end = inside.size();
  while (start < end && inside[start] == ' ') ++start;
  while (end > start && inside[end - 1] == ' ') --end;
  inside = inside.substr(start, end - start);
  if (inside.empty()) return product;
  for (const std::string& dim : jc::SplitElements(inside)) {
    product *= parse_u64_field(dim, "shape");
  }
  return product;
}

}  // namespace

NativeCaptureReader::NativeCaptureReader(
    dmi_store::S3Client* s3, std::string bucket,
    std::shared_ptr<const ClickHouseClient> client, ReaderConfig catalog_config,
    int64_t max_coalesce_gap_bytes)
    : s3_(s3), bucket_(std::move(bucket)),
      catalog_(std::move(client), std::move(catalog_config)),
      max_coalesce_gap_bytes_(max_coalesce_gap_bytes) {}

Selection NativeCaptureReader::select(const SearchFilters& filters) const {
  const SearchPage page = catalog_.search(filters);
  if (page.next_cursor.has_value()) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        "selection exceeds one bounded page; narrow the query or paginate "
        "explicitly");
  }
  if (page.items.empty()) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "selection requires at least one capture");
  }
  std::set<std::string> tenants;
  std::vector<std::string> ids;
  for (const auto& item : page.items) {
    tenants.insert(item[0]);
    ids.push_back(item[kCaptureId]);
  }
  if (tenants.size() > 1) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "selection spans multiple tenants");
  }
  // The selection identity: the same compact sorted-keys JSON Python's
  // CaptureSelection.create hashes — version 3, the watermark, the filter
  // hash, the tenant, the ids in page order.
  // json.dumps(..., sort_keys=True, separators=(",", ":")): capture_ids
  // sorts before catalog_watermark, version last.
  std::string identity = std::string("{\"capture_ids\":[");
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) identity += ",";
    identity += json_escape(ids[i]);
  }
  identity += "],\"catalog_watermark\":" + json_escape(page.watermark) +
              ",\"filter_hash\":" + json_escape(filter_hash(filters)) +
              ",\"tenant_id\":" + json_escape(*tenants.begin()) +
              ",\"version\":3}";
  unsigned char digest[32];
  SHA256(reinterpret_cast<const unsigned char*>(identity.data()),
         identity.size(), digest);
  char hex[65];
  for (int i = 0; i < 32; ++i) {
    std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
  }
  return Selection{hex,
                   ids,
                   page.watermark,
                   filter_hash(filters),
                   *tenants.begin()};
}

std::vector<std::vector<std::string>> NativeCaptureReader::resolve(
    const Selection& selection) const {
  const auto resolved = catalog_.get_by_ids(
      selection.capture_ids, selection.tenant_id, selection.catalog_watermark);
  std::map<std::string, std::vector<std::string>> by_id;
  for (const auto& item : resolved) {
    if (!by_id.emplace(item[kCaptureId], item).second) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "catalog returned duplicate capture: " +
                             item[kCaptureId]);
    }
  }
  if (by_id.size() != selection.capture_ids.size()) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "selection no longer resolves at its catalog watermark");
  }
  std::vector<std::vector<std::string>> out;
  out.reserve(selection.capture_ids.size());
  for (const auto& id : selection.capture_ids) {
    out.push_back(by_id.at(id));
  }
  return out;
}

HydrationEstimateData NativeCaptureReader::estimate(
    const Selection& selection) const {
  const auto descriptors = resolve(selection);
  // Plan: group by (store_id, pack_id, object_key), sort by offset, coalesce
  // ranges up to the gap.
  std::map<std::string, std::vector<size_t>> grouped;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    const auto& d = descriptors[i];
    grouped[d[kStoreId] + "|" + d[kPackId] + "|" + d[kObjectKey]].push_back(i);
  }
  HydrationEstimateData estimate;
  estimate.capture_count = descriptors.size();
  estimate.object_count = grouped.size();
  uint64_t payload_requests = 0, payload_bytes = 0, footer_bytes = 0,
           logical = 0, stored = 0;
  for (const auto& [key, grouped_indexes] : grouped) {
    std::vector<size_t> indexes = grouped_indexes;
    std::sort(indexes.begin(), indexes.end(), [&](size_t a, size_t b) {
      return parse_u64_field(descriptors[a][kPayloadOffset], "offset") <
             parse_u64_field(descriptors[b][kPayloadOffset], "offset");
    });
    uint64_t start = 0, end = 0;
    bool open = false;
    for (const size_t index : indexes) {
      const uint64_t offset =
          parse_u64_field(descriptors[index][kPayloadOffset], "offset");
      const uint64_t length =
          parse_u64_field(descriptors[index][kStoredLength], "stored");
      if (!open) {
        open = true;
        start = offset;
        end = offset + length;
        continue;
      }
      if (static_cast<int64_t>(offset) <=
          static_cast<int64_t>(end) + max_coalesce_gap_bytes_) {
        end = std::max(end, offset + length);
        continue;
      }
      ++payload_requests;
      payload_bytes += end - start;
      start = offset;
      end = offset + length;
    }
    if (open) {
      ++payload_requests;
      payload_bytes += end - start;
    }
    // Footer reads: trailer + footer, bounded by the pack's own size.
    const auto& first = descriptors[indexes[0]];
    const uint64_t object_bytes = parse_u64_field(first[kObjectBytes], "bytes");
    footer_bytes += kTrailerBytes +
                    std::min<uint64_t>(64ull * 1024 * 1024,
                                       object_bytes - kTrailerBytes);
  }
  for (const auto& d : descriptors) {
    logical += parse_u64_field(d[kDecodedLength], "decoded");
    stored += parse_u64_field(d[kStoredLength], "stored");
  }
  estimate.request_count = payload_requests + 2 * grouped.size();
  estimate.request_bytes = payload_bytes + footer_bytes;
  estimate.logical_bytes = logical;
  estimate.stored_bytes = stored;
  return estimate;
}

std::vector<std::string> NativeCaptureReader::hydrate(
    const Selection& selection, int64_t byte_limit,
    int64_t request_limit) const {
  if (byte_limit < 0) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "byte_limit must be non-negative");
  }
  if (request_limit <= 0) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "request_limit must be positive");
  }
  const auto descriptors = resolve(selection);
  // Plan exactly as estimate does; the ranges are re-walked for the fetch.
  std::map<std::string, std::vector<size_t>> grouped;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    const auto& d = descriptors[i];
    grouped[d[kStoreId] + "|" + d[kPackId] + "|" + d[kObjectKey]].push_back(i);
  }

  struct Range {
    std::vector<size_t> indexes;
    uint64_t offset = 0, length = 0;
  };
  struct Plan {
    std::string store_id, pack_id, object_key;
    std::vector<size_t> descriptor_indexes;
    std::vector<Range> ranges;
  };
  std::vector<Plan> plans;
  uint64_t payload_requests = 0, payload_bytes = 0;
  for (const auto& [key, grouped_indexes] : grouped) {
    Plan plan;
    std::vector<size_t> indexes = grouped_indexes;
    plan.descriptor_indexes = indexes;
    std::sort(indexes.begin(), indexes.end(), [&](size_t a, size_t b) {
      return parse_u64_field(descriptors[a][kPayloadOffset], "offset") <
             parse_u64_field(descriptors[b][kPayloadOffset], "offset");
    });
    const std::string& first = descriptors[indexes[0]][kStoreId];
    plan.store_id = first;
    plan.pack_id = descriptors[indexes[0]][kPackId];
    plan.object_key = descriptors[indexes[0]][kObjectKey];
    uint64_t start = 0, end = 0;
    bool open = false;
    std::vector<size_t> current;
    for (const size_t index : indexes) {
      const uint64_t offset =
          parse_u64_field(descriptors[index][kPayloadOffset], "offset");
      const uint64_t length =
          parse_u64_field(descriptors[index][kStoredLength], "stored");
      if (!open) {
        open = true;
        current = {index};
        start = offset;
        end = offset + length;
        continue;
      }
      if (static_cast<int64_t>(offset) <=
          static_cast<int64_t>(end) + max_coalesce_gap_bytes_) {
        current.push_back(index);
        end = std::max(end, offset + length);
        continue;
      }
      plan.ranges.push_back({current, start, end - start});
      current = {index};
      start = offset;
      end = offset + length;
    }
    if (open) plan.ranges.push_back({current, start, end - start});
    payload_requests += plan.ranges.size();
    for (const auto& range : plan.ranges) payload_bytes += range.length;
    plans.push_back(std::move(plan));
  }

  if (payload_requests > static_cast<uint64_t>(request_limit)) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        "hydration request limit exceeded: " +
            std::to_string(payload_requests) + " > " +
            std::to_string(request_limit));
  }
  if (static_cast<int64_t>(payload_bytes) > byte_limit) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        "hydration byte limit exceeded: " + std::to_string(payload_bytes) +
            " > " + std::to_string(byte_limit));
  }

  // Phase one: the pack footer is the authority for what each payload IS.
  // Binding every catalog descriptor to the footer costs two small range
  // reads per uncached pack, before any payload range is fetched — so a
  // later pack cannot fail after earlier payloads have been fetched.
  // Those reads are REAL requests and bytes: charged against the same
  // budgets the payload ranges use, exactly as the Python reader's
  // _ReadBudget charges them -- each range at its ACTUAL length, checked
  // before that range is fetched. The footer's length is only known once
  // the trailer has been read, so the check cannot be hoisted ahead of the
  // pack as one estimate: an estimate that checked the running total
  // BEFORE charging the current pack let a one-pack selection fetch a
  // footer the byte budget did not cover (byte_limit=16, request_limit=3:
  // three GETs and the payload came back).
  struct ReadBudget {
    int64_t requests;
    int64_t bytes;
    const std::string* pack_id;
    void consume(uint64_t length) {
      if (requests < 1) {
        throw CatalogError(
            CatalogError::Kind::kValue,
            "hydration request limit exceeded: the footer verification for "
            "pack " + *pack_id + " needs a request the budget does not have "
            "after the payload ranges");
      }
      if (static_cast<int64_t>(length) > bytes) {
        throw CatalogError(
            CatalogError::Kind::kValue,
            "hydration byte limit exceeded: the footer verification for "
            "pack " + *pack_id + " needs " + std::to_string(length) +
            " more bytes than the " + std::to_string(bytes) +
            " remaining after the payload ranges");
      }
      --requests;
      bytes -= static_cast<int64_t>(length);
    }
  };
  ReadBudget budget{request_limit - static_cast<int64_t>(payload_requests),
                    byte_limit - static_cast<int64_t>(payload_bytes),
                    nullptr};
  for (const auto& plan : plans) {
    budget.pack_id = &plan.pack_id;
    const auto& first_descriptor = descriptors[plan.descriptor_indexes[0]];
    PackRefData ref{
        plan.pack_id, plan.store_id, plan.object_key,
        parse_u64_field(first_descriptor[kObjectBytes], "bytes"),
        first_descriptor[kPackChecksum],
        parse_u64_field(first_descriptor[kPackRecordCount], "records")};
    // The footer's own descriptor rows, one per record: the same 32-field
    // renderer the indexer reads, with the batch's version irrelevant here.
    // read_pack_descriptor_rows performs two GETs (trailer + footer) and
    // charges each against the budget before issuing it.
    const auto footer_rows = read_pack_descriptor_rows(
        s3_, ref, [&budget](uint64_t length) { budget.consume(length); });
    // The rows are rendered VALUES text (one string per record), split
    // into 32 typed fields each.
    std::map<std::string, std::vector<FooterField>> footer;
    for (const auto& rendered : footer_rows) {
      std::vector<FooterField> fields = split_footer_row(rendered);
      if (fields.size() != 32 || fields[0].is_null) {
        throw CatalogError(CatalogError::Kind::kValue,
                           "pack footer row does not have 32 fields");
      }
      // The footer rows are in CAPTURE_COLUMNS order — capture_id at 0,
      // dtype 18, shape 19 — not the reader's sort-key-first layout.
      const std::string capture_id = fields[0].text;
      footer.emplace(capture_id, std::move(fields));
    }
    for (const size_t index : plan.descriptor_indexes) {
      const auto& descriptor = descriptors[index];
      const auto it = footer.find(descriptor[kCaptureId]);
      if (it == footer.end()) {
        throw CatalogError(
            CatalogError::Kind::kValue,
            "catalog descriptor does not match the pack footer: " +
                descriptor[kCaptureId] + " is not in pack " + plan.pack_id);
      }
      // EVERY metadata and locator field must agree — hook name, tenant,
      // layer and the rest are part of the authority, not just the
      // placement. The two layouts name fields at different indexes; the
      // catalog->footer map is built once from the two name orders.
      for (const auto& [catalog_index, footer_index] : kCatalogToFooter()) {
        if (!footer_field_matches(descriptor[catalog_index],
                                  it->second[footer_index])) {
          throw CatalogError(
              CatalogError::Kind::kValue,
              "catalog descriptor does not match the pack footer: field " +
                  std::to_string(catalog_index) + " for " +
                  descriptor[kCaptureId]);
        }
      }
    }
  }

  // Phase two: fetch and verify. Resolution is tracked separately from
  // the payload bytes: a zero-dimension tensor's payload is genuinely
  // empty, which cannot be the sentinel for an unfilled slot.
  std::vector<std::string> payloads(descriptors.size());
  std::vector<char> resolved(descriptors.size(), 0);
  for (const auto& plan : plans) {
    const auto& first_descriptor = descriptors[plan.descriptor_indexes[0]];
    PackRefData ref{
        plan.pack_id, plan.store_id, plan.object_key,
        parse_u64_field(first_descriptor[kObjectBytes], "bytes"),
        first_descriptor[kPackChecksum],
        parse_u64_field(first_descriptor[kPackRecordCount], "records")};
    for (const auto& range : plan.ranges) {
      std::vector<uint8_t> block;
      std::string error;
      if (!s3_->GetRange(plan.object_key, range.offset, range.length, &block,
                         &error)) {
        throw CatalogError(CatalogError::Kind::kValue,
                           "payload range read failed: " + error);
      }
      if (block.size() != range.length) {
        throw CatalogError(CatalogError::Kind::kValue,
                           "object store returned a short range: " +
                               std::to_string(block.size()) + " != " +
                               std::to_string(range.length));
      }
      for (const size_t index : range.indexes) {
        const auto& descriptor = descriptors[index];
        const uint64_t start =
            parse_u64_field(descriptor[kPayloadOffset], "offset") -
            range.offset;
        const uint64_t length =
            parse_u64_field(descriptor[kStoredLength], "stored");
        const std::string payload(
            reinterpret_cast<const char*>(block.data()) + start, length);
        // verify_payload: length, the none codec, CRC32.
        if (payload.size() != length) {
          throw CatalogError(CatalogError::Kind::kValue,
                             "short record for " + descriptor[kCaptureId]);
        }
        if (descriptor[kCodec] != "none") {
          throw CatalogError(CatalogError::Kind::kValue,
                             "unsupported codec: " + descriptor[kCodec]);
        }
        const uint32_t crc = dmi_pack::Crc32(
            reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
        char hex[9];
        std::snprintf(hex, sizeof(hex), "%08x", crc);
        if (hex != descriptor[kPayloadChecksum]) {
          throw CatalogError(CatalogError::Kind::kValue,
                             "record checksum mismatch: " +
                                 descriptor[kCaptureId]);
        }
        payloads[index] = payload;
        resolved[index] = 1;
      }
    }
  }
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (!resolved[i]) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "hydration plan did not resolve every capture");
    }
  }
  return payloads;
}

std::vector<std::pair<std::string, CoreSummaryData>>
NativeCaptureReader::summarize_core(const Selection& selection,
                                    int64_t byte_limit, int64_t request_limit,
                                    uint64_t max_summary_captures,
                                    uint64_t max_summary_elements) const {
  if (max_summary_captures == 0 || max_summary_elements == 0) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "summary limits must be positive");
  }
  if (selection.capture_ids.size() > max_summary_captures) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        "summary capture limit exceeded: " +
            std::to_string(selection.capture_ids.size()) + " > " +
            std::to_string(max_summary_captures));
  }
  // Budget BEFORE fetching payloads: hydrate is the only reader of bytes,
  // and the element budget is knowable from the catalog rows alone.
  const auto descriptors = resolve(selection);
  uint64_t total_elements = 0;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    // prod(shape) once per capture, matching reader.py's
    // sum(math.prod(shape)). decoded_length / dtype_bytes is the SAME
    // number (the pack index binds decoded_length == prod(shape) *
    // dtype_bytes), so multiplying by it counted N**2 elements for an
    // N-element capture and refused everything from 8001 elements up.
    // The consistency between the two is validated separately below.
    total_elements += shape_product(descriptors[i][kShape]);
  }
  if (total_elements > max_summary_elements) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        "summary element limit exceeded: " + std::to_string(total_elements) +
            " > " + std::to_string(max_summary_elements));
  }
  const auto payloads = hydrate(selection, byte_limit, request_limit);

  std::vector<std::pair<std::string, CoreSummaryData>> summaries;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    const auto& descriptor = descriptors[i];
    const std::string& payload = payloads[i];
    const std::string dtype = descriptor[kDtype];
    const uint64_t elements =
        shape_product(descriptor[kShape]);
    if (payload.size() !=
        elements * dtype_bytes(dtype)) {
      throw CatalogError(
          CatalogError::Kind::kValue,
          "payload length does not match dtype and shape: " +
              std::to_string(payload.size()));
    }
    CoreSummaryData summary;
    summary.element_count = elements;
    summary.order_stats_are_integers = !is_float_dtype(dtype);
    const bool floating = is_float_dtype(dtype);
    uint64_t zeros = 0, nans = 0, infs = 0;
    double sum = 0.0, scale = 0.0;
    double min_f = 0, max_f = 0;
    int64_t min_i = 0, max_i = 0;
    bool first = true;
    // Two passes over a decode that is cheap per element.
    std::vector<double> finite;
    finite.reserve(payload.size() / dtype_bytes(dtype));
    for (size_t e = 0; e < elements; ++e) {
      const double value =
          decode_element(dtype, reinterpret_cast<const uint8_t*>(
                                    payload.data()),
                         e * dtype_bytes(dtype));
      if (floating) {
        if (std::isnan(value)) {
          ++nans;
          continue;
        }
        if (std::isinf(value)) {
          ++infs;
          continue;
        }
      }
      finite.push_back(value);
      sum += value;
      if (value == 0.0) ++zeros;
      if (first || value < min_f) min_f = value;
      if (first || value > max_f) max_f = value;
      scale = std::max(scale, std::fabs(value));
      if (summary.order_stats_are_integers) {
        // Off the raw bytes, not off `value`: the double round trip loses
        // every integer above 2**53 and turns 2**63 into INT64_MIN.
        const int64_t as_int = decode_element_int(
            dtype, reinterpret_cast<const uint8_t*>(payload.data()),
            e * dtype_bytes(dtype));
        if (first || as_int < min_i) min_i = as_int;
        if (first || as_int > max_i) max_i = as_int;
      }
      first = false;
    }
    summary.finite_count = finite.size();
    summary.nan_count = nans;
    summary.inf_count = infs;
    summary.zero_fraction =
        elements == 0
            ? 0.0
            : static_cast<double>(zeros) / static_cast<double>(elements);
    if (finite.empty()) {
      // Degenerate groups keep the dtype-determined order-statistic type.
      summary.mean = 0.0;
      summary.minimum = 0.0;
      summary.maximum = 0.0;
      summary.abs_max = 0.0;
      summary.l2_norm = 0.0;
      summary.order_stats_are_integers = !floating;
    } else {
      double l2 = 0.0;
      if (scale != 0.0) {
        for (const double value : finite) {
          l2 += (value / scale) * (value / scale);
        }
        l2 = scale * std::sqrt(l2);
      }
      summary.mean = sum / static_cast<double>(finite.size());
      summary.l2_norm = l2;
      if (floating) {
        summary.minimum = min_f;
        summary.maximum = max_f;
        summary.abs_max = scale;
      } else {
        // Order statistics off the raw integers: magnitudes above 2**53
        // stay exact. The magnitudes are compared in UNSIGNED space, where
        // |INT64_MIN| exists: comparing them as doubles (and negating in
        // int64) reported abs_max NEGATIVE for [INT64_MIN, INT64_MAX].
        // 0u - (uint64_t)min_i is well defined and yields exactly 2**63.
        summary.minimum = static_cast<double>(min_i);
        summary.maximum = static_cast<double>(max_i);
        const uint64_t mag_min = min_i < 0
                                     ? 0u - static_cast<uint64_t>(min_i)
                                     : static_cast<uint64_t>(min_i);
        const uint64_t mag_max = max_i < 0
                                     ? 0u - static_cast<uint64_t>(max_i)
                                     : static_cast<uint64_t>(max_i);
        const uint64_t magnitude = std::max(mag_min, mag_max);
        // abs_max (double) carries the exact magnitude — 2**63 is exactly
        // representable in a double. abs_max_int is Int64 on the wire and
        // genuinely cannot hold 2**63, the one magnitude an int64 tensor
        // can produce (|INT64_MIN|) and an Int64 field cannot name, so the
        // int field SATURATES at INT64_MAX for that single value. Python's
        // unbounded int is the oracle and reports the exact
        // 9223372036854775808; widening this wire field is a format change
        // and is deliberately not made here.
        summary.abs_max = static_cast<double>(magnitude);
        constexpr uint64_t kInt64Max =
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        summary.abs_max_int =
            magnitude > kInt64Max ? std::numeric_limits<int64_t>::max()
                                  : static_cast<int64_t>(magnitude);
        summary.minimum_int = min_i;
        summary.maximum_int = max_i;
      }
    }
    summaries.emplace_back(descriptor[kCaptureId], summary);
  }
  return summaries;
}

}  // namespace dmi_catalog
