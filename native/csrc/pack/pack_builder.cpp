#include "pack_builder.h"

#include <algorithm>

#ifdef __x86_64__
#include <nmmintrin.h>
#endif

namespace dmi_pack {

// --- CRC-32 (IEEE), bit-identical to zlib.crc32 -----------------------------

namespace {
// Slice-by-16: sixteen 256-entry tables so 16 bytes advance one CRC step
// each. Bit-identical to zlib.crc32 (polynomial 0xEDB88320, reflected);
// verified against zlib on every conformance corpus.
struct Crc32Tables {
  uint32_t v[16][256];
  constexpr Crc32Tables() : v{} {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      }
      v[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; ++i) {
      for (int t = 1; t < 16; ++t) {
        const uint32_t prev = v[t - 1][i];
        v[t][i] = (prev >> 8) ^ v[0][prev & 0xFF];
      }
    }
  }
  constexpr const uint32_t (&operator[](int t) const)[256] { return v[t]; }
};
constexpr Crc32Tables kCrc{};
}  // namespace

uint32_t Crc32(const uint8_t* data, size_t n, uint32_t crc) {
  crc ^= 0xFFFFFFFFu;
  // Process 16-byte blocks with the x16 slice tables. The block reads an
  // unaligned uint64 LE pair per step — fine on x86/ARM64.
  while (n >= 16) {
    uint32_t lo = crc ^ static_cast<uint32_t>(data[0] | data[1] << 8 |
                                              data[2] << 16 | data[3] << 24);
    const uint32_t hi = static_cast<uint32_t>(data[4] | data[5] << 8 |
                                              data[6] << 16 | data[7] << 24);
    const uint32_t w1 = static_cast<uint32_t>(data[8] | data[9] << 8 |
                                              data[10] << 16 | data[11] << 24);
    const uint32_t w2 = static_cast<uint32_t>(data[12] | data[13] << 8 |
                                              data[14] << 16 | data[15] << 24);
    crc = kCrc[15][lo & 0xFF] ^ kCrc[14][(lo >> 8) & 0xFF] ^
          kCrc[13][(lo >> 16) & 0xFF] ^ kCrc[12][(lo >> 24) & 0xFF] ^
          kCrc[11][hi & 0xFF] ^ kCrc[10][(hi >> 8) & 0xFF] ^
          kCrc[9][(hi >> 16) & 0xFF] ^ kCrc[8][(hi >> 24) & 0xFF] ^
          kCrc[7][w1 & 0xFF] ^ kCrc[6][(w1 >> 8) & 0xFF] ^
          kCrc[5][(w1 >> 16) & 0xFF] ^ kCrc[4][(w1 >> 24) & 0xFF] ^
          kCrc[3][w2 & 0xFF] ^ kCrc[2][(w2 >> 8) & 0xFF] ^
          kCrc[1][(w2 >> 16) & 0xFF] ^ kCrc[0][(w2 >> 24) & 0xFF];
    data += 16;
    n -= 16;
  }
  while (n--) {
    crc = kCrc[0][(crc ^ *data++) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// --- canonical JSON ----------------------------------------------------------

namespace {

// Python json.dumps with ensure_ascii=True escapes exactly: ", \, and the
// control range as \u00XX except the short forms \b \f \n \r \t; everything
// >= 0x7F as \uXXXX (surrogate pair above 0xFFFF).
void EncodeJsonString(const std::string& value, std::string* out) {
  out->push_back('"');
  size_t i = 0;
  const size_t n = value.size();
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(value[i]);
    if (c < 0x80) {
      switch (c) {
        case '"': out->append("\\\""); break;
        case '\\': out->append("\\\\"); break;
        case '\b': out->append("\\b"); break;
        case '\f': out->append("\\f"); break;
        case '\n': out->append("\\n"); break;
        case '\r': out->append("\\r"); break;
        case '\t': out->append("\\t"); break;
        default:
          if (c < 0x20 || c == 0x7F) {
            char buf[7];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out->append(buf);
          } else {
            out->push_back(static_cast<char>(c));
          }
      }
      ++i;
      continue;
    }
    // Decode one UTF-8 sequence; re-encode as \uXXXX (or a surrogate pair).
    uint32_t cp = 0;
    size_t len = 0;
    if ((c & 0xE0) == 0xC0 && i + 1 < n) {
      cp = (c & 0x1Fu) << 6 | (value[i + 1] & 0x3Fu);
      len = 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < n) {
      cp = (c & 0x0Fu) << 12 | (value[i + 1] & 0x3Fu) << 6 | (value[i + 2] & 0x3Fu);
      len = 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < n) {
      cp = (c & 0x07u) << 18 | (value[i + 1] & 0x3Fu) << 12 |
           (value[i + 2] & 0x3Fu) << 6 | (value[i + 3] & 0x3Fu);
      len = 4;
    } else {
      // Invalid UTF-8: Python would have raised at encode time; the reference
      // validates text earlier, so treat this as unreachable for valid input.
      // Emit U+FFFD to keep the writer total.
      out->append("\\ufffd");
      ++i;
      continue;
    }
    i += len;
    if (cp <= 0xFFFF) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
      out->append(buf);
    } else {
      const uint32_t v = cp - 0x10000;
      char buf[13];
      std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x",
                    0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
      out->append(buf);
    }
  }
  out->push_back('"');
}

}  // namespace

void EncodeRecordRow(const RecordMetadata& m, uint64_t offset,
                     uint64_t stored_length, uint64_t decoded_length,
                     const char* codec, const std::string& checksum,
                     std::string* out) {
  // Row shape from _record_mapping: checksum, codec, decoded_length,
  // metadata (nested object), offset, stored_length — metadata's own keys are
  // sorted inside it. json.dumps(sort_keys=True) flattens nothing: "metadata"
  // is one nested value.
  out->push_back('{');
  auto str_value = [&](const std::string& v) { EncodeJsonString(v, out); };
  auto uint_value = [&](uint64_t v) {
    char buf[24];
    int n = std::snprintf(buf, sizeof(buf), "%llu",
                          static_cast<unsigned long long>(v));
    out->append(buf, static_cast<size_t>(n));
  };
  auto int_value = [&](int64_t v) {
    char buf[24];
    int n = std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
    out->append(buf, static_cast<size_t>(n));
  };
  auto key2 = [&](const char* k) {
    EncodeJsonString(k, out);
    out->push_back(':');
  };

  key2("checksum"); str_value(checksum);
  out->push_back(',');
  key2("codec"); str_value(codec);
  out->push_back(',');
  key2("decoded_length"); uint_value(decoded_length);
  out->push_back(',');
  key2("metadata");
  out->push_back('{');
  bool first = true;
  auto mkey = [&](const char* k) {
    if (!first) out->push_back(',');
    EncodeJsonString(k, out);
    out->push_back(':');
    first = false;
  };
  mkey("adapter_revision");
  if (m.adapter_revision.has_value()) {
    str_value(*m.adapter_revision);
  } else {
    out->append("null");
  }
  mkey("batch_position"); uint_value(m.batch_position);
  mkey("capture_id"); str_value(m.capture_id);
  mkey("capture_policy_version"); str_value(m.capture_policy_version);
  mkey("captured_at_ns"); uint_value(m.captured_at_ns);
  mkey("dtype"); str_value(m.dtype);
  mkey("experiment_id"); str_value(m.experiment_id);
  mkey("hook_name"); str_value(m.hook_name);
  mkey("layer_number"); int_value(m.layer_number);
  mkey("model_id"); str_value(m.model_id);
  mkey("model_revision"); str_value(m.model_revision);
  mkey("producer_rank"); uint_value(m.producer_rank);
  mkey("request_id"); str_value(m.request_id);
  mkey("run_id"); str_value(m.run_id);
  mkey("sequence_id"); str_value(m.sequence_id);
  mkey("session_id"); str_value(m.session_id);
  mkey("shape");
  out->push_back('[');
  for (size_t i = 0; i < m.shape.size(); ++i) {
    if (i) out->push_back(',');
    uint_value(m.shape[i]);
  }
  out->push_back(']');
  mkey("step_number"); uint_value(m.step_number);
  mkey("tenant_id"); str_value(m.tenant_id);
  mkey("token_end"); uint_value(m.token_end);
  mkey("token_start"); uint_value(m.token_start);
  out->push_back('}');
  out->push_back(',');
  key2("offset"); uint_value(offset);
  out->push_back(',');
  key2("stored_length"); uint_value(stored_length);
  out->push_back('}');
}

// --- validation --------------------------------------------------------------

namespace {

const std::array<const char*, 10> kDtypes = {
    "bool", "uint8", "int8", "int16", "float16",
    "bfloat16", "int32", "float32", "int64", "float64"};

bool DtypeSupported(const std::string& dtype) {
  for (const char* name : kDtypes) {
    if (dtype == name) return true;
  }
  return false;
}

bool ValidText(const std::string& value, bool optional, size_t limit = 512) {
  if (value.empty()) return optional;
  if (value.size() > limit) return false;
  // Must be valid UTF-8 (the JSON encoder assumes it).
  size_t i = 0;
  while (i < value.size()) {
    const unsigned char c = static_cast<unsigned char>(value[i]);
    size_t len = 0;
    if (c < 0x80) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    else return false;
    if (i + len > value.size()) return false;
    for (size_t k = 1; k < len; ++k) {
      if ((static_cast<unsigned char>(value[i + k]) & 0xC0) != 0x80) return false;
    }
    i += len;
  }
  return true;
}

}  // namespace

Status ValidateMetadata(const RecordMetadata& m) {
  if (!ValidText(m.capture_id, false) || !ValidText(m.tenant_id, false) ||
      !ValidText(m.experiment_id, false) || !ValidText(m.run_id, false) ||
      !ValidText(m.session_id, false) || !ValidText(m.request_id, false) ||
      !ValidText(m.sequence_id, false) || !ValidText(m.model_id, false) ||
      !ValidText(m.model_revision, false) ||
      !ValidText(m.capture_policy_version, false) ||
      !ValidText(m.hook_name, false) ||
      !ValidText(m.adapter_revision.value_or(std::string()),
                 !m.adapter_revision.has_value())) {
    return Status::kBadArgument;
  }
  if (!DtypeSupported(m.dtype)) return Status::kBadArgument;
  if (m.shape.size() > kMaxRank) return Status::kBadArgument;
  for (uint32_t dim : m.shape) {
    if (dim > 0x7FFFFFFFu) return Status::kBadArgument;
  }
  if (m.producer_rank > 0xFFFFFFFFull || m.batch_position > 0xFFFFFFFFull) {
    return Status::kBadArgument;
  }
  if (m.layer_number < -1 || m.layer_number > 0x7FFFFFFFll) {
    return Status::kBadArgument;
  }
  if (m.token_end < m.token_start) return Status::kBadArgument;
  return Status::kOk;
}

// --- UUID ---------------------------------------------------------------------

bool ParseUuid(const std::string& text, std::array<uint8_t, 16>* out,
               std::string* canonical) {
  // 8-4-4-4-12 hex with optional surrounding braces; case-insensitive.
  std::string hex;
  hex.reserve(32);
  size_t start = 0, end = text.size();
  if (!text.empty() && text.front() == '{') ++start;
  if (end > start && text[end - 1] == '}') --end;
  for (size_t i = start; i < end; ++i) {
    const char c = text[i];
    if (c == '-') continue;
    const bool is_hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                        (c >= 'A' && c <= 'F');
    if (!is_hex) return false;
    hex.push_back(static_cast<char>(c >= 'A' && c <= 'F' ? c - 'A' + 'a' : c));
  }
  if (hex.size() != 32) return false;
  // Hyphens must be exactly at 8,13,18,23 of the canonical layout.
  size_t hyphens = 0;
  for (size_t i = start; i < end; ++i) {
    if (text[i] == '-') ++hyphens;
  }
  if (hyphens != 0 && hyphens != 4) return false;
  for (int i = 0; i < 16; ++i) {
    auto nib = [&](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      return c - 'a' + 10;
    };
    (*out)[static_cast<size_t>(i)] =
        static_cast<uint8_t>(nib(hex[2 * i]) << 4 | nib(hex[2 * i + 1]));
  }
  // Python's str(UUID) is lowercase with hyphens.
  if (canonical) {
    *canonical = hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" +
                 hex.substr(12, 4) + "-" + hex.substr(16, 4) + "-" +
                 hex.substr(20, 12);
  }
  return true;
}

// --- PackBuilder ----------------------------------------------------------------

namespace {
void PutU16(std::vector<uint8_t>* b, size_t at, uint16_t v) {
  (*b)[at] = static_cast<uint8_t>(v);
  (*b)[at + 1] = static_cast<uint8_t>(v >> 8);
}
void PutU32(std::vector<uint8_t>* b, size_t at, uint32_t v) {
  for (int i = 0; i < 4; ++i) (*b)[at + i] = static_cast<uint8_t>(v >> (8 * i));
}
void PutU64(std::vector<uint8_t>* b, size_t at, uint64_t v) {
  for (int i = 0; i < 8; ++i) (*b)[at + i] = static_cast<uint8_t>(v >> (8 * i));
}
size_t AlignUp(size_t n) {
  return (n + kPackAlignment - 1) / kPackAlignment * kPackAlignment;
}
const char kHeaderMagic[8] = {'D', 'M', 'I', 'P', 'A', 'C', 'K', '\0'};
const char kTrailerMagic[8] = {'D', 'M', 'I', 'F', 'T', 'R', '\0', '\0'};
}  // namespace

PackBuilder::PackBuilder(const std::string& pack_id, uint64_t created_at_ns,
                         uint64_t max_pack_bytes, uint64_t max_records)
    : max_pack_bytes_(max_pack_bytes), max_records_(max_records) {
  ParseUuid(pack_id, &uuid_, &pack_id_);
  created_at_ns_ = created_at_ns;
  buffer_.reserve(4096);
  buffer_.resize(kHeaderSize);
  std::memcpy(buffer_.data(), kHeaderMagic, 8);
  PutU16(&buffer_, 8, static_cast<uint16_t>(kPackMajorVersion));
  PutU16(&buffer_, 10, static_cast<uint16_t>(kPackMinorVersion));
  PutU32(&buffer_, 12, static_cast<uint32_t>(kHeaderSize));
  std::memcpy(buffer_.data() + 16, uuid_.data(), 16);
  PutU64(&buffer_, 32, created_at_ns);
  PutU32(&buffer_, 40, 0);  // reserved
  // 20 bytes of zeros at 44.
}

Status PackBuilder::Append(const PackRecord& record) {
  if (sealed_) return Status::kSealedState;
  if (record_count_ >= max_records_) return Status::kRecordLimit;
  const Status meta = ValidateMetadata(record.metadata);
  if (meta != Status::kOk) return meta;

  // Duplicate capture ids: linear scan over a sorted vector with binary
  // search; record counts per pack are bounded (max_records), and the
  // reference's per-record cost is a set lookup — a sorted insert is the same
  // order with zero hashing.
  auto it = std::lower_bound(
      capture_ids_.begin(), capture_ids_.end(), record.metadata.capture_id);
  if (it != capture_ids_.end() && *it == record.metadata.capture_id) {
    return Status::kDuplicateId;
  }

  const uint64_t payload = record.payload_bytes;
  const size_t offset = AlignUp(buffer_.size());
  const size_t padded = AlignUp(offset + payload);

  // Footer grows by this record's JSON plus a comma; project the seal size
  // exactly as the reference does before mutating anything.
  std::string row;
  row.reserve(512);
  std::string checksum(8, '0');
  {
    const uint32_t crc = Crc32(record.payload, payload);
    static const char* kHex = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
      checksum[7 - i] = kHex[(crc >> (4 * i)) & 0xF];
    }
  }
  EncodeRecordRow(record.metadata, offset, payload, payload, "none", checksum,
                  &row);
  const uint64_t footer_prefix_len = 96;  // computed below precisely instead
  (void)footer_prefix_len;
  // Footer = prefix + "[" + rows joined by "," + "]" + suffix. The prefix and
  // suffix are fixed for a given pack (they come from the empty-footer JSON),
  // so compute them once lazily. For projection we need their length: cache on
  // first use.
  if (footer_prefix_cache_.empty()) {
    // Canonical footer key order is json.dumps(sort_keys=True): created_at_ns,
    // format, major_version, minor_version, pack_id, records.
    std::string empty;
    empty.reserve(256);
    empty.append("{\"created_at_ns\":");
    empty.append(std::to_string(created_at_ns_));
    empty.append(",\"format\":\"dmi-pack\",\"major_version\":");
    empty.append(std::to_string(kPackMajorVersion));
    empty.append(",\"minor_version\":");
    empty.append(std::to_string(kPackMinorVersion));
    empty.append(",\"pack_id\":");
    EncodeJsonString(pack_id_, &empty);
    empty.append(",\"records\":[]}");
    const size_t bracket = empty.rfind("[]");
    footer_prefix_cache_ = empty.substr(0, bracket);
    footer_suffix_cache_ = empty.substr(bracket + 2);
  }
  const uint64_t footer_length = footer_prefix_cache_.size() + 2 +
                                 record_json_bytes_ + row.size() +
                                 record_json_.size() + footer_suffix_cache_.size();
  const uint64_t projected =
      static_cast<uint64_t>(padded) + footer_length + kTrailerSize;
  if (footer_length > kMaxFooterBytes || projected > max_pack_bytes_) {
    return Status::kCapacity;
  }

  buffer_.resize(offset);
  buffer_.insert(buffer_.end(), record.payload, record.payload + payload);
  buffer_.resize(padded, 0);
  capture_ids_.insert(it, record.metadata.capture_id);
  record_json_.push_back(std::move(row));
  record_json_bytes_ += record_json_.back().size();
  ++record_count_;
  return Status::kOk;
}

Status PackBuilder::Seal(SealedPack* out) {
  if (sealed_) return Status::kSealedState;
  if (record_count_ == 0) return Status::kEmpty;

  std::string footer;
  footer.reserve(footer_prefix_cache_.size() + 2 + record_json_bytes_ +
                 record_json_.size() + footer_suffix_cache_.size());
  footer.append(footer_prefix_cache_);
  footer.push_back('[');
  for (size_t i = 0; i < record_json_.size(); ++i) {
    if (i) footer.push_back(',');
    footer.append(record_json_[i]);
  }
  footer.push_back(']');
  footer.append(footer_suffix_cache_);

  const size_t footer_offset = AlignUp(buffer_.size());
  if (footer.size() > kMaxFooterBytes) return Status::kCapacity;
  if (footer_offset + footer.size() + kTrailerSize > max_pack_bytes_) {
    return Status::kCapacity;
  }

  buffer_.resize(footer_offset, 0);
  buffer_.insert(buffer_.end(), footer.begin(), footer.end());

  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  SHA256_Update(&ctx, buffer_.data(), buffer_.size());
  std::array<uint8_t, 32> body_digest{};
  // OpenSSL's SHA256_Final zeroizes the context on output, so a continued
  // update after it does NOT extend the hash. Keep a copy of the pre-final
  // context; the reference (hashlib) survives digest() without copying.
  SHA256_CTX body_ctx = ctx;
  SHA256_Final(body_digest.data(), &ctx);

  std::vector<uint8_t> trailer(kTrailerSize);
  std::memcpy(trailer.data(), kTrailerMagic, 8);
  // No-padding little-endian layout ("<8sHHQQI32s"), offsets 0/8/10/12/20/28/32.
  PutU16(&trailer, 8, static_cast<uint16_t>(kPackMajorVersion));
  PutU16(&trailer, 10, static_cast<uint16_t>(kPackMinorVersion));
  PutU64(&trailer, 12, footer_offset);
  PutU64(&trailer, 20, footer.size());
  PutU32(&trailer, 28, Crc32(reinterpret_cast<const uint8_t*>(footer.data()),
                             footer.size()));
  std::memcpy(trailer.data() + 32, body_digest.data(), 32);

  buffer_.insert(buffer_.end(), trailer.begin(), trailer.end());
  SHA256_Update(&body_ctx, trailer.data(), trailer.size());
  std::array<uint8_t, 32> object_digest{};
  // body_ctx carries body||trailer state: Final reads the object checksum out.
  SHA256_Final(object_digest.data(), &body_ctx);

  out->pack_id = pack_id_;
  out->created_at_ns = created_at_ns_;
  out->data = std::move(buffer_);
  out->record_count = record_count_;
  out->footer_offset = footer_offset;
  static const char* kHex = "0123456789abcdef";
  out->checksum.resize(64);
  for (int i = 0; i < 32; ++i) {
    out->checksum[2 * i] = kHex[object_digest[i] >> 4];
    out->checksum[2 * i + 1] = kHex[object_digest[i] & 0xF];
  }
  sealed_ = true;
  return Status::kOk;
}

}  // namespace dmi_pack
