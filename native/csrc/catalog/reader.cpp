#include "reader.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <set>

#include <openssl/sha.h>

#include "../common/json.h"
#include "catalog_writer.h"
#include "../pack/pack_builder.h"
#include "lease_coordinator.h"

namespace jc = dmi_common;

namespace dmi_catalog {

namespace {

constexpr const char* kSortKey[] = {"tenant_id", "experiment_id", "run_id",
                                    "captured_at_ns", "capture_id"};
// The writer's column order minus index_version, which orders rather than
// describes; the resolved columns are the projection minus the sort key.
constexpr const char* kProjection[] = {
    "capture_id",  "tenant_id",      "experiment_id", "run_id",
    "session_id",  "request_id",     "sequence_id",   "model_id",
    "model_revision", "adapter_revision", "capture_policy_version",
    "hook_name",   "layer_number",   "producer_rank", "step_number",
    "token_start", "token_end",      "batch_position", "dtype",
    "shape",       "captured_at_ns", "pack_id",       "store_id",
    "object_key",  "object_bytes",   "pack_checksum", "pack_record_count",
    "payload_offset", "stored_length", "decoded_length", "codec",
    "payload_checksum"};
constexpr const char* kResolutionOrder = "(index_version, store_id, pack_id)";

std::string quoted(const std::string& name) { return "`" + name + "`"; }

// sha256 over the same compact sorted-keys JSON Python's filter_hash hashes.
// The filter dict is Python's asdict minus cursor/limit, so each field is
// rendered in sorted field order with null for absent strings and [] for
// absent lists — identical bytes to json.dumps(..., sort_keys=True,
// separators=(",", ":")).
std::string filter_hash_impl(const SearchFilters& f) {
  auto quoted_string = [](const std::string& value) {
    std::string out;
    dmi_pack::EncodeJsonString(value, &out);
    return out;
  };
  auto opt = [&](const std::optional<std::string>& v) {
    return v.has_value() ? quoted_string(*v) : "null";
  };
  auto opt_int = [](const std::optional<uint64_t>& v) {
    return v.has_value() ? std::to_string(*v) : "null";
  };
  std::string hook_names = "[";
  for (size_t i = 0; i < f.hook_names.size(); ++i) {
    if (i) hook_names += ",";
    hook_names += quoted_string(f.hook_names[i]);
  }
  hook_names += "]";
  std::string layers = "[";
  for (size_t i = 0; i < f.layer_numbers.size(); ++i) {
    if (i) layers += ",";
    layers += std::to_string(f.layer_numbers[i]);
  }
  layers += "]";
  std::string json = std::string("{\"captured_after_ns\":") + opt_int(f.captured_after_ns) +
                     ",\"captured_before_ns\":" + opt_int(f.captured_before_ns) +
                     ",\"experiment_id\":" + opt(f.experiment_id) +
                     ",\"hook_names\":" + hook_names +
                     ",\"layer_numbers\":" + layers +
                     ",\"model_id\":" + opt(f.model_id) +
                     ",\"run_id\":" + opt(f.run_id) +
                     ",\"session_id\":" + opt(f.session_id) +
                     ",\"tenant_id\":" + opt(f.tenant_id) + "}";
  unsigned char digest[32];
  SHA256(reinterpret_cast<const unsigned char*>(json.data()), json.size(),
         digest);
  char hex[65];
  for (int i = 0; i < 32; ++i) {
    std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
  }
  return hex;
}

// Unpadded url-safe base64, byte-compatible with Python's
// urlsafe_b64encode(...).rstrip(b"=") and its canonical-alphabet decode.
std::string base64url_encode(const std::string& raw) {
  static const char* alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  std::string out;
  out.reserve((raw.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < raw.size(); i += 3) {
    const uint32_t v = (static_cast<unsigned char>(raw[i]) << 16) |
                       (static_cast<unsigned char>(raw[i + 1]) << 8) |
                       static_cast<unsigned char>(raw[i + 2]);
    out.push_back(alphabet[(v >> 18) & 63]);
    out.push_back(alphabet[(v >> 12) & 63]);
    out.push_back(alphabet[(v >> 6) & 63]);
    out.push_back(alphabet[v & 63]);
  }
  const size_t rest = raw.size() - i;
  if (rest == 1) {
    // One trailing byte: TWO characters (12 bits, 8 used, 4 zero) — a
    // third would decode to a phantom byte and the cursor would not be
    // byte-identical to Python's.
    const uint32_t v = static_cast<unsigned char>(raw[i]) << 16;
    out.push_back(alphabet[(v >> 18) & 63]);
    out.push_back(alphabet[(v >> 12) & 63]);
  } else if (rest == 2) {
    // Two trailing bytes: THREE characters (18 bits, 16 used, 2 zero) —
    // a fourth would decode to a phantom \x00 byte after the JSON.
    const uint32_t v = (static_cast<unsigned char>(raw[i]) << 16) |
                       (static_cast<unsigned char>(raw[i + 1]) << 8);
    out.push_back(alphabet[(v >> 18) & 63]);
    out.push_back(alphabet[(v >> 12) & 63]);
    out.push_back(alphabet[(v >> 6) & 63]);
  }
  return out;
}

bool urlsafe_alphabet(const std::string& cursor) {
  static const std::string ok =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  if (cursor.empty()) return false;
  for (const char c : cursor) {
    if (ok.find(c) == std::string::npos) return false;
  }
  return true;
}

int b64_value(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '-') return 62;
  if (c == '_') return 63;
  return -1;
}

std::string base64url_decode(const std::string& encoded) {
  std::string out;
  int buffer = 0, bits = 0;
  for (const char c : encoded) {
    const int v = b64_value(c);
    if (v < 0) throw CatalogError(CatalogError::Kind::kValue,
                                  "cursor is not canonical url-safe base64");
    buffer = (buffer << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((buffer >> bits) & 0xFF));
    }
  }
  // Reject a trailing group that cannot be part of any encoding: a final
  // lone character carries no whole byte. NOT a general non-canonical
  // check, and deliberately so -- CPython's b64decode(validate=True)
  // checks the alphabet rather than the unused trailing bits, so
  // `eyJ2IjoxfQ` and `eyJ2IjoxfR` both decode to `{"v":1}` on the Python
  // side. Rejecting them here would make the native reader STRICTER than
  // the oracle it is judged against, which is a parity break rather than
  // a hardening.
  if (bits >= 6 && (buffer & ((1 << bits) - 1)) != 0) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "cursor is not canonical url-safe base64");
  }
  return out;
}

// cursor._CURSOR_VERSION / model._CURSOR_LIMIT / model._TEXT_LIMIT.
constexpr uint64_t kCursorVersion = 1;
constexpr size_t kCursorLimitBytes = 2048;
constexpr size_t kTextLimitBytes = 512;

// model._validate_text, for a value the caller supplied non-null. The
// oracle's "UTF-8" is a property Python's str already carries, so the two
// refusals it can actually reach are the empty string and the byte bound --
// and the bound is on ENCODED bytes, which a std::string already holds.
void validate_text(const std::string& value, const char* name) {
  if (value.empty() || value.size() > kTextLimitBytes) {
    throw CatalogError(CatalogError::Kind::kValue,
                       std::string(name) + " must be non-empty UTF-8 within " +
                           std::to_string(kTextLimitBytes) + " bytes");
  }
}

// How many members a JSON object declares at the top level. Used to refuse
// a cursor carrying fields this version does not define, the way
// decode_cursor's unexpected-fields check does.
size_t top_level_member_count(const std::string& object) {
  const std::string inside = jc::Unwrap(object);
  if (inside.find_first_not_of(" \t\n") == std::string::npos) return 0;
  return jc::SplitElements(inside).size();
}

// The raw JSON token a key maps to, un-parsed. jc::FindInt has already
// destroyed the spelling by the time it returns a value, and the spelling
// is exactly what a bounded parse needs.
std::string find_token_in(const std::string& object, const char* key) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = std::string("\"") + key + "\"" + sep;
    const size_t at = object.find(needle);
    if (at == std::string::npos) continue;
    const size_t begin = at + needle.size();
    size_t q = begin;
    while (q < object.size() && object[q] != ',' && object[q] != '}' &&
           object[q] != ']') {
      ++q;
    }
    return object.substr(begin, q - begin);
  }
  return std::string();
}

// kNotJson is the JSON grammar itself, not the field's type or range: the
// oracle's json.loads refuses the token before decode_cursor ever looks at
// which field it belongs to, so all three fields report it the same way.
enum class NumberError { kNone, kNotJson, kNotInteger, kOutOfRange };

constexpr const char* kNotJsonMessage = "cursor does not contain valid JSON";

// The ONE parse every cursor integer goes through -- the envelope's `v` and
// `w` and the key's captured_at_ns. jc::FindInt accumulates digits into an
// int64_t with no bound check whatsoever, so a spelling above UInt64 did
// not refuse, it WRAPPED: `2**64 + 1` on `w` pinned the snapshot at
// watermark 1 (an `ok` answer with an empty page and watermark "1"), and on
// `v` it walked through the version gate below and served a full page.
// json.loads keeps the whole integer, so the oracle refuses both.
NumberError parse_cursor_uint(const std::string& token, uint64_t* out) {
  const size_t begin = token.find_first_not_of(" \t\n\r");
  const size_t end = token.find_last_not_of(" \t\n\r");
  const std::string text = begin == std::string::npos
                               ? std::string()
                               : token.substr(begin, end - begin + 1);
  const bool negative = !text.empty() && text[0] == '-';
  const std::string digits = negative ? text.substr(1) : text;
  if (digits.empty() ||
      digits.find_first_not_of("0123456789") != std::string::npos) {
    return NumberError::kNotInteger;
  }
  // A leading zero is not a JSON number at all -- `007` parses as `0`
  // followed by extra data, which json.loads reports as a decode error.
  // Scanning for digits alone accepted it, and `007` on the key's
  // captured_at_ns paged from position 7: a cursor that walks BACKWARDS
  // over rows the caller has already been served.
  if (digits.size() > 1 && digits[0] == '0') return NumberError::kNotJson;
  uint64_t value = 0;
  for (const char c : digits) {
    const uint64_t digit = static_cast<uint64_t>(c - '0');
    // Accumulate with the bound checked BEFORE the multiply: a wrapping
    // accumulator is the same defect one level down.
    if (value > (UINT64_MAX - digit) / 10) return NumberError::kOutOfRange;
    value = value * 10 + digit;
  }
  // A negative number is a legal JSON number and a legal Python int, so
  // the oracle refuses it on the RANGE check, not the type one.
  if (negative && value != 0) return NumberError::kOutOfRange;
  *out = value;
  return NumberError::kNone;
}

// `label` names the field the way cursor.py's _uint64 call site does, so a
// refusal reads with the oracle's wording.
uint64_t find_uint_in(const std::string& object, const char* key,
                      const char* label) {
  if (!jc::HasKey(object, key)) {
    throw CatalogError(CatalogError::Kind::kValue,
                       std::string("cursor is missing ") + key);
  }
  uint64_t value = 0;
  switch (parse_cursor_uint(find_token_in(object, key), &value)) {
    case NumberError::kNotJson:
      throw CatalogError(CatalogError::Kind::kValue, kNotJsonMessage);
    case NumberError::kNotInteger:
      throw CatalogError(CatalogError::Kind::kValue,
                         std::string(label) + " must be an integer");
    case NumberError::kOutOfRange:
      throw CatalogError(CatalogError::Kind::kValue,
                         std::string(label) + " must fit UInt64");
    case NumberError::kNone:
      break;
  }
  return value;
}

// Undo ClickHouse's TSV escaping: \\\\ → \\\, \\' → ', \\n, \\t, and \\N → the
// empty string (a NULL renders as an unquoted NUL marker; the reader maps
// it to absent, which the parity suite flattens both ways).
std::string unescape_tsv(const std::string& text) {
  if (text.find('\\') == std::string::npos) return text;
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '\\' || i + 1 >= text.size()) {
      out.push_back(text[i]);
      continue;
    }
    const char next = text[++i];
    switch (next) {
      case 'n': out.push_back('\n'); break;
      case 't': out.push_back('\t'); break;
      case 'r': out.push_back('\r'); break;
      case '\\': out.push_back('\\'); break;
      case '\'': out.push_back('\''); break;
      case '0': break;             // NUL byte: nothing survives text
      case 'N': break;             // NULL marker
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      default: out.push_back('\\'); out.push_back(next); break;
    }
  }
  return out;
}

// Parse one TSV-rendered tuple field — the argMax aggregate travels as
// ('a','b',123,...) with backslash escapes and \N for NULL.
std::vector<std::string> parse_tsv_tuple(const std::string& text) {
  std::vector<std::string> fields;
  std::string current;
  bool in_str = false;
  // Nesting inside a VALUE, which only a comma at depth 0 may split.
  // `shape` is Array(UInt32) and renders as [1,128,4096]: splitting on its
  // commas over-splits the row, and every rank-2-or-higher shape -- which
  // is to say every real activation -- failed the column-count check.
  int depth = 0;
  size_t begin = 0;
  size_t end = text.size();
  // The aggregate arrives wrapped in its own tuple parens; strip them once
  // rather than skipping every paren, which would also eat a value's.
  if (end >= 2 && text[0] == '(' && text[end - 1] == ')') {
    begin = 1;
    --end;
  }
  for (size_t i = begin; i < end; ++i) {
    const char c = text[i];
    if (in_str) {
      if (c == '\\' && i + 1 < end) {
        const char next = text[++i];
        if (next == 'N') {
          current.clear();
          in_str = false;  // NULL renders as \N inside a quoted string
          continue;
        }
        // ONE level of decoding, and exactly one, measured both ways. The
        // tuple's rendered text carries its own escapes (`a\b` arrives as
        // a\\b) and the TSV layer does not re-escape the rendered tuple,
        // so carrying escapes through doubled every backslash, while
        // adding a field-wide TSV unescape on top decoded twice and
        // turned `packs/a\b.dmi-pack` into a backspace. Grouped columns
        // are the opposite case -- plain TSV fields, unescaped where the
        // row is flattened, never here.
        switch (next) {
          case 'n': current.push_back('\n'); break;
          case 't': current.push_back('\t'); break;
          case 'r': current.push_back('\r'); break;
          case '0': current.push_back('\0'); break;
          default: current.push_back(next); break;  // \' and \\ included
        }
        continue;
      }
      if (c == '\'') {
        in_str = false;
        continue;
      }
      current.push_back(c);
      continue;
    }
    if (c == '\'') {
      in_str = true;
      continue;
    }
    if (c == '[' || c == '(') {
      ++depth;
      current.push_back(c);
      continue;
    }
    if (c == ']' || c == ')') {
      --depth;
      current.push_back(c);
      continue;
    }
    if (c == ',' && depth == 0) {
      fields.push_back(current);
      current.clear();
      continue;
    }
    current.push_back(c);
  }
  fields.push_back(current);
  return fields;
}

}  // namespace

std::string filter_hash(const SearchFilters& filters) {
  return filter_hash_impl(filters);
}

NativeCaptureCatalog::NativeCaptureCatalog(
    std::shared_ptr<const ClickHouseClient> client, ReaderConfig config)
    : client_(std::move(client)), config_(std::move(config)) {}

std::string NativeCaptureCatalog::qualified(
    const std::string& table) const {
  return "`" + config_.database + "`.`" + config_.table_prefix + "_" + table +
         "`";
}

std::map<std::string, std::string> NativeCaptureCatalog::settings() const {
  return {{"max_rows_to_read", std::to_string(config_.max_rows_to_read)},
          {"max_bytes_to_read", std::to_string(config_.max_bytes_to_read)},
          {"max_execution_time", std::to_string(config_.max_execution_time_s)},
          {"read_overflow_mode", "throw"},
          {"timeout_overflow_mode", "throw"}};
}

std::map<std::string, std::string>
NativeCaptureCatalog::bounded_read_settings() const {
  if (!config_.consistent_snapshot_reads) return settings();
  std::map<std::string, std::string> out = settings();
  out["select_sequential_consistency"] = "1";
  return out;
}

std::string NativeCaptureCatalog::membership() const {
  // clickhouse_sql.membership_predicate, bounded: the snapshot is the set
  // of packs whose publish reached the watermark at or before the bound.
  const std::string manifest = qualified("snapshot_manifest");
  const std::string watermark = qualified("index_watermark");
  return (
      "(store_id, pack_id) IN ("
      "SELECT store_id, pack_id FROM " + manifest + " "
      "WHERE index_version <= %(watermark)s AND (index_version, publish_id) IN "
      "(SELECT index_version, publish_id FROM " + watermark +
      " WHERE index_version <= %(watermark)s))");
}

std::string NativeCaptureCatalog::projection() const {
  // Identity columns grouped on; everything else as ONE aggregate over the
  // total ordering argument — a mixed row is impossible by construction,
  // and the winner cannot move across layouts or merges. See the Python
  // module for the derivations.
  // Identity columns are grouped on and project directly; the REST travel
  // as one aggregate — resolved is the projection minus the sort key.
  std::set<std::string> sort_key_set(std::begin(kSortKey),
                                     std::end(kSortKey));
  std::string resolved;
  std::string grouped;
  for (const char* column : kSortKey) {
    if (!grouped.empty()) grouped += ",";
    grouped += quoted(column);
  }
  for (const char* column : kProjection) {
    if (sort_key_set.count(column)) continue;
    if (!resolved.empty()) resolved += ",";
    resolved += quoted(column);
  }
  return grouped + ",argMax(tuple(" + resolved + ")," + kResolutionOrder + ")";
}

uint64_t NativeCaptureCatalog::published_head(bool deciding) const {
  // The configured bounds FIRST, then the deciding setting on top -- the
  // shape `_published_head` uses in clickhouse_reader.py. Sending the
  // deciding setting alone (or nothing) left the one read every search
  // takes as the only unbounded statement in the reader.
  std::map<std::string, std::string> read_settings = settings();
  if (deciding) {
    for (const auto& [key, value] : deciding_read()) {
      read_settings[key] = value;
    }
  }
  const std::vector<Row> rows = client_->execute(
      "SELECT max(index_version) FROM " + qualified("index_watermark"), {},
      read_settings);
  if (rows.empty() || rows[0].empty() || rows[0][0].empty()) return 0;
  return parse_u64_field(rows[0][0], "watermark");
}

std::string NativeCaptureCatalog::current_watermark() const {
  return std::to_string(published_head(false));
}

SearchPage NativeCaptureCatalog::search(const SearchFilters& filters) const {
  // The identity filters FIRST, as __post_init__ has them. An empty one is
  // not a wildcard: it rendered `AND `tenant_id` = ''`, matched nothing,
  // and came back as a successful empty page where the oracle refuses the
  // query -- and an unbounded one was shipped to the server rather than
  // refused here.
  for (const auto& [value, name] :
       std::vector<std::pair<const std::optional<std::string>*, const char*>>{
           {&filters.tenant_id, "tenant_id"},
           {&filters.experiment_id, "experiment_id"},
           {&filters.run_id, "run_id"},
           {&filters.session_id, "session_id"},
           {&filters.model_id, "model_id"}}) {
    if (value->has_value()) validate_text(**value, name);
  }
  if (filters.limit < 1 || filters.limit > 10'000) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "limit must be between 1 and 10000");
  }
  // The REST of CaptureQuery.__post_init__'s bounds, in its order. Only
  // `limit` was ported, so native ACCEPTED queries the oracle refuses
  // outright -- 1025 layer numbers came back as a successful page, which
  // is a caller error answered as data.
  if (filters.hook_names.size() > 128 || filters.layer_numbers.size() > 1024) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "query filters exceed their bounded cardinality");
  }
  // Each hook name is text too, and after the cardinality bound -- the
  // oracle's order, so a query breaking both reports the same one.
  for (const std::string& hook_name : filters.hook_names) {
    validate_text(hook_name, "hook_name");
  }
  for (const int64_t layer : filters.layer_numbers) {
    // -1 is the "no layer" sentinel; below it names nothing.
    if (layer < -1) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "layer numbers must be >= -1");
    }
  }
  // The captured bounds are unsigned here, so the oracle's non-negativity
  // check holds by construction; the ORDER of the pair does not, and a
  // window that runs backwards selects nothing rather than being refused.
  if (filters.captured_after_ns.has_value() &&
      filters.captured_before_ns.has_value() &&
      *filters.captured_before_ns < *filters.captured_after_ns) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "captured_before_ns must be >= captured_after_ns");
  }
  const std::string hash = filter_hash_impl(filters);
  uint64_t watermark;
  std::optional<std::array<Param, 5>> after;
  if (!filters.cursor.has_value()) {
    // A stale head here merely pins a slightly older snapshot: cheap read.
    watermark = published_head(false);
  } else {
    // The head bounds decode_cursor, which refuses a cursor above it — a
    // deciding read, because a lagging replica would hard-reject a valid
    // cursor under ordinary replication lag.
    const uint64_t max_watermark = published_head(true);
    const std::string canonical = *filters.cursor;
    if (!urlsafe_alphabet(canonical)) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "cursor is not canonical url-safe base64");
    }
    if (canonical.size() > kCursorLimitBytes) {
      throw CatalogError(
          CatalogError::Kind::kValue,
          "cursor exceeds the cursor limit of " +
              std::to_string(kCursorLimitBytes) + " bytes");
    }
    const std::string payload = base64url_decode(canonical);
    // THE ENVELOPE, before anything is read out of it. decode_cursor
    // requires version 1 and exactly {v,w,fh,k}; a reader that just picks
    // out `k` and `w` pages a v2 cursor with v1 semantics and accepts
    // fields it does not understand, which turns a format designed to
    // evolve into one that silently cannot.
    // The version gate takes ANY non-1 spelling, in range or out of it, on
    // the one message the oracle uses -- a `v` that does not fit UInt64 is
    // an unsupported version, not a version this reader may bound and then
    // compare.
    if (!jc::HasKey(payload, "v")) {
      throw CatalogError(CatalogError::Kind::kValue, "cursor is missing v");
    }
    const std::string version_token = find_token_in(payload, "v");
    uint64_t version = 0;
    const NumberError version_error = parse_cursor_uint(version_token,
                                                        &version);
    if (version_error == NumberError::kNotJson) {
      throw CatalogError(CatalogError::Kind::kValue, kNotJsonMessage);
    }
    if (version_error != NumberError::kNone || version != kCursorVersion) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "unsupported cursor version: " + version_token);
    }
    for (const char* field : {"v", "w", "fh", "k"}) {
      if (!jc::HasKey(payload, field)) {
        throw CatalogError(CatalogError::Kind::kValue,
                           std::string("cursor is missing ") + field);
      }
    }
    if (top_level_member_count(payload) != 4) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "cursor has unexpected fields");
    }
    if (jc::FindString(payload, "fh") != hash) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "cursor does not match the filters that issued it");
    }
    const uint64_t cursor_watermark =
        find_uint_in(payload, "w", "cursor watermark");
    if (cursor_watermark > max_watermark) {
      throw CatalogError(
          CatalogError::Kind::kValue,
          "cursor watermark is ahead of the catalog: " +
              std::to_string(cursor_watermark) + " > " +
              std::to_string(max_watermark));
    }
    const std::string key = jc::FindArray(payload, "k");
    const std::vector<std::string> parts = jc::SplitElements(jc::Unwrap(key));
    if (parts.size() != 5) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "cursor key must hold five fields");
    }
    // The key holds (tenant, experiment, run, captured_at_ns, capture_id);
    // strings are JSON literals, captured_at_ns is a JSON number. Each
    // string must decode NON-EMPTY, as cursor._text requires: falling back
    // to the raw token when the decode came up empty accepted a crafted
    // `""` component and paged after a two-character position no encoder
    // ever issues.
    auto literal = [&](size_t i) {
      const std::string text = jc::ParseLiteral(parts[i]);
      if (text.empty()) {
        throw CatalogError(CatalogError::Kind::kValue,
                           "cursor key components must be non-empty strings");
      }
      return text;
    };
    // captured_at_ns additionally travelled into the statement as a quoted
    // STRING parameter -- which the server coerces with String -> UInt64
    // and therefore WRAPS modulo 2**64. A crafted `2**64 + honest` named a
    // position it does not spell, and search answered `ok` with a page
    // silently short a row. Carrying the decoded value as a TYPED
    // parameter removes the coercion, which is the root of that one.
    // The bound itself is NOT special to this component: every cursor
    // integer -- `v` and `w` above included -- goes through the one
    // parse_cursor_uint above, because all three used to reach a digit
    // accumulator with no bound check at all.
    auto number = [&](size_t i) {
      uint64_t value = 0;
      switch (parse_cursor_uint(parts[i], &value)) {
        case NumberError::kNotJson:
          throw CatalogError(CatalogError::Kind::kValue, kNotJsonMessage);
        case NumberError::kNotInteger:
          throw CatalogError(CatalogError::Kind::kValue,
                             "cursor captured_at_ns must be an integer");
        case NumberError::kOutOfRange:
          throw CatalogError(CatalogError::Kind::kValue,
                             "cursor captured_at_ns must fit UInt64");
        case NumberError::kNone:
          break;
      }
      return value;
    };
    after = std::array<Param, 5>{literal(0), literal(1), literal(2),
                                 number(3), literal(4)};
    watermark = cursor_watermark;
  }

  Params params{{"watermark", watermark}};
  std::string clauses = membership();
  for (const auto& [value, name] :
       std::vector<std::pair<const std::optional<std::string>*, const char*>>{
           {&filters.tenant_id, "tenant_id"},
           {&filters.experiment_id, "experiment_id"},
           {&filters.run_id, "run_id"},
           {&filters.session_id, "session_id"},
           {&filters.model_id, "model_id"}}) {
    if (value->has_value()) {
      clauses += " AND " + quoted(name) + " = %(" + name + ")s";
      params.emplace(name, **value);
    }
  }
  if (!filters.hook_names.empty()) {
    std::string rendered = "(";
    for (size_t i = 0; i < filters.hook_names.size(); ++i) {
      if (i) rendered += ",";
      rendered += sql_quote(filters.hook_names[i]);
    }
    rendered += ")";
    clauses += " AND hook_name IN " + rendered;
  }
  if (!filters.layer_numbers.empty()) {
    std::string rendered = "(";
    for (size_t i = 0; i < filters.layer_numbers.size(); ++i) {
      if (i) rendered += ",";
      rendered += std::to_string(filters.layer_numbers[i]);
    }
    rendered += ")";
    clauses += " AND layer_number IN " + rendered;
  }
  if (filters.captured_after_ns.has_value()) {
    clauses += " AND captured_at_ns >= %(captured_after_ns)s";
    params.emplace("captured_after_ns", *filters.captured_after_ns);
  }
  if (filters.captured_before_ns.has_value()) {
    clauses += " AND captured_at_ns <= %(captured_before_ns)s";
    params.emplace("captured_before_ns", *filters.captured_before_ns);
  }
  if (after.has_value()) {
    const char* names[] = {"tenant_id", "experiment_id", "run_id",
                           "captured_at_ns", "capture_id"};
    std::string columns, placeholders;
    for (int i = 0; i < 5; ++i) {
      if (i) {
        columns += ",";
        placeholders += ",";
      }
      columns += quoted(names[i]);
      placeholders += "%(after_" + std::string(names[i]) + ")s";
      params.emplace("after_" + std::string(names[i]), (*after)[i]);
    }
    clauses += " AND (" + columns + ") > (" + placeholders + ")";
  }

  // One row beyond the page tells whether a cursor is owed, without a
  // second counting query.
  params.emplace("row_limit", static_cast<uint64_t>(filters.limit + 1));
  std::string grouped, order;
  for (const char* column : kSortKey) {
    if (!grouped.empty()) {
      grouped += ",";
      order += ",";
    }
    grouped += quoted(column);
    order += quoted(column);
  }
  const std::vector<Row> rows = client_->execute(
      "SELECT " + projection() + " FROM " + qualified("capture_raw") +
          " WHERE " + clauses + " GROUP BY " + grouped + " ORDER BY " + order +
          " LIMIT %(row_limit)s",
      params, bounded_read_settings());

  SearchPage page;
  page.watermark = std::to_string(watermark);
  std::vector<std::vector<std::string>> items;
  for (const Row& row : rows) {
    if (row.size() != 6) {  // 5 sort-key columns + the aggregate
      throw CatalogError(CatalogError::Kind::kValue,
                         "catalog row has " + std::to_string(row.size()) +
                             " columns, expected 6");
    }
    std::vector<std::string> resolved = parse_tsv_tuple(row[5]);
    if (resolved.size() != std::size(kProjection) - 5) {
      throw CatalogError(
          CatalogError::Kind::kValue,
          "catalog returned a malformed resolved-column tuple, expected " +
              std::to_string(std::size(kProjection) - 5) + " columns");
    }
    // Flatten: the sort-key columns, then the resolved ones.
    //
    // The two halves arrive with DIFFERENT escaping, which is the whole
    // subtlety here. A grouped column is its own TSV field, so ClickHouse
    // escapes its quotes and backslashes and this is the only place that
    // can undo them -- without it, `alan's run` came back `alan\'s run`.
    // The aggregate is one field holding an already-quoted tuple, whose
    // inner escapes parse_tsv_tuple handles; unescaping it here as well
    // ate a real backslash (`packs/a\b.dmi-pack` became a backspace).
    std::vector<std::string> item;
    for (int i = 0; i < 5; ++i) item.push_back(unescape_tsv(row[i]));
    for (auto& field : resolved) item.push_back(field);
    items.push_back(std::move(item));
  }
  if (rows.size() > static_cast<size_t>(filters.limit)) {
    items.resize(static_cast<size_t>(filters.limit));
    // encode_cursor(_key_of(last), watermark, filter_hash): the key is the
    // LAST KEPT ITEM's sort-key tuple — tenant, experiment, run,
    // captured_at_ns (a JSON number), capture_id — straight off the raw
    // row, not the flattened item, whose layout is not the projection
    // order. The page kept rows[0..limit-1]; the row beyond it only said
    // a cursor is owed.
    const Row& last = rows[static_cast<size_t>(filters.limit) - 1];
    std::string key = "[";
    for (int i = 0; i < 5; ++i) {
      if (i) key += ",";
      if (i == 3) {
        key += last[i];  // captured_at_ns: a JSON number, no TSV escapes
      } else {
        // The row value is raw TSV — unescape FIRST (the same values the
        // items carry), THEN render as JSON (EncodeJsonString), or the
        // envelope hashes and carries the TSV-escaped form, which is not
        // the value Python's cursor would decode to.
        std::string value = unescape_tsv(last[i]);
        dmi_pack::EncodeJsonString(value, &key);
      }
    }
    key += "]";
    const std::string payload =
        std::string("{\"fh\":\"") + hash + "\",\"k\":" + key +
        ",\"v\":1,\"w\":" + std::to_string(watermark) + "}";
    page.next_cursor = base64url_encode(payload);
  }
  page.items = std::move(items);
  return page;
}

std::vector<std::vector<std::string>> NativeCaptureCatalog::get_by_ids(
    const std::vector<std::string>& capture_ids,
    const std::string& tenant_id, const std::string& watermark) const {
  if (static_cast<int>(capture_ids.size()) > config_.max_capture_ids) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        "capture id lookup exceeds max_capture_ids: " +
            std::to_string(capture_ids.size()) + " > " +
            std::to_string(config_.max_capture_ids));
  }
  if (tenant_id.empty()) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "tenant_id must be a non-empty string");
  }
  if (capture_ids.empty()) return {};
  // NON-EMPTY and all digits. `all_of` over an empty range is true, so an
  // empty watermark passed the check and parse_u64_field then answered 0
  // ("empty renders as 0", by its own contract) -- the call resolved
  // against a snapshot admitting nothing and returned no rows, where the
  // caller's real fault was an unpopulated watermark. The predicate takes
  // an unsigned char because ::isdigit is undefined for a negative char,
  // which any byte above 0x7F is on this platform.
  if (watermark.empty() ||
      !std::all_of(watermark.begin(), watermark.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      })) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "watermark must be a decimal string");
  }
  // WIDTH, next, and here rather than in parse_u64_field: this is the one
  // caller-data call that function has, so the refusal belongs in the
  // reader's own ValueError taxonomy with _parse_watermark's wording, not
  // as a transport-layer ClickHouseError. The digit check above passes on
  // an arbitrarily long run, and the parse then wrapped modulo 2**64 --
  // 2**64 arrived as 0 and 2**64 + head as head, so both slid under the
  // published-head guard below and resolved against a snapshot the caller
  // never named. parse_u64_field is bounded too, so the wrap is gone
  // either way; this check is what makes the ANSWER match the oracle's.
  //
  // Leading zeros are insignificant to `int()`, so they are stripped before
  // the width is judged: "0" * 4 + str(2**64 - 1) is 24 digits and the
  // oracle accepts it.
  {
    const size_t first = watermark.find_first_not_of('0');
    const std::string digits = first == std::string::npos
                                   ? std::string("0")
                                   : watermark.substr(first);
    if (digits.size() > 20 ||
        (digits.size() == 20 && digits > "18446744073709551615")) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "watermark must fit UInt64");
    }
  }
  const uint64_t requested = parse_u64_field(watermark, "watermark");
  // A selection is caller data: its watermark must be one the indexer
  // actually published. A deciding read — this answer refuses the call.
  if (requested > published_head(true)) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "selection watermark exceeds the published watermark");
  }
  // tenant_id leads the WHERE: it is the first ORDER BY column, so the
  // primary index narrows the read to one tenant's range and the bloom
  // filter prunes granules inside it.
  const std::string head =
      "SELECT " + projection() + " FROM " + qualified("capture_raw") +
      " WHERE tenant_id = %(tenant_id)s AND capture_id IN ";
  // Chunked by rendered bytes: the ids land in the statement TEXT, and a
  // full-size lookup can breach max_query_size. Ids are sent once each.
  std::vector<std::vector<std::string>> out;
  std::set<std::string> unique(capture_ids.begin(), capture_ids.end());
  std::vector<std::string> chunk;
  size_t size = 2;
  auto flush = [&]() {
    if (chunk.empty()) return;
    std::string ids;
    for (size_t i = 0; i < chunk.size(); ++i) {
      if (i) ids += ",";
      ids += sql_quote(chunk[i]);
    }
    // The ids land in the statement TEXT (chunked inline, like the
    // writer's members); the membership + snapshot bound ride as params.
    const std::vector<Row> rows = client_->execute(
        head + "(" + ids + ") AND " + membership() +
            " GROUP BY `tenant_id`,`experiment_id`,`run_id`,"
            "`captured_at_ns`,`capture_id`",
        {{"tenant_id", tenant_id}, {"watermark", requested}},
        bounded_read_settings());
    for (const Row& row : rows) {
      if (row.size() != 6) {
        throw CatalogError(CatalogError::Kind::kValue,
                           "catalog row has " + std::to_string(row.size()) +
                               " columns, expected 6");
      }
      std::vector<std::string> resolved = parse_tsv_tuple(row[5]);
      // The same width check search makes: a short or long tuple is a
      // malformed row either way, and letting it through here produced a
      // silently truncated descriptor instead of the error the Python
      // reader raises for both paths.
      if (resolved.size() != std::size(kProjection) - 5) {
        throw CatalogError(
            CatalogError::Kind::kValue,
            "catalog returned a malformed resolved-column tuple, expected " +
                std::to_string(std::size(kProjection) - 5) + " columns");
      }
      std::vector<std::string> item;
      // Grouped columns carry their own TSV escaping — see search.
      for (int i = 0; i < 5; ++i) item.push_back(unescape_tsv(row[i]));
      for (auto& field : resolved) item.push_back(field);
      out.push_back(std::move(item));
    }
    chunk.clear();
    size = 2;
  };
  for (const auto& id : unique) {
    const size_t encoded = 2 * id.size() + 2;
    if (encoded + 2 > 192 * 1024) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "item exceeds inline query byte budget");
    }
    const size_t separator = chunk.empty() ? 0 : 2;
    if (!chunk.empty() && size + separator + encoded > 192 * 1024) {
      flush();
    }
    chunk.push_back(id);
    size += (chunk.size() == 1 ? 0 : 2) + encoded;
  }
  flush();
  return out;
}

}  // namespace dmi_catalog
