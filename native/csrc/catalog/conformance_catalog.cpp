// B1/B2 conformance driver: the native lease coordinator, version
// allocator, and catalog writer over a newline-JSON protocol, the same
// shape as the other conformance drivers. One process is one writer
// instance; `open` (re)initializes it from the request's config fields.
// The coordinator-level ops (head/claim/claim_contested) route through
// the writer's coordinator, so B1 tests exercise the bare protocol and
// B2 tests the writer's quarantine-wrapped surface.

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "../common/json.h"
#include "../store/s3_client.h"
#include "catalog_writer.h"
#include "clickhouse_client.h"
#include "indexer.h"
#include "hydration.h"
#include "reader.h"
#include "schema.h"
#include "lease_coordinator.h"
#include "version_allocator.h"

namespace jc = dmi_common;

namespace {

using dmi_catalog::CatalogError;
using dmi_catalog::CatalogWriter;
using dmi_catalog::ClickHouseError;
using dmi_catalog::PackIdentity;
using dmi_catalog::PublisherLease;

const char* error_kind(CatalogError::Kind kind) {
  switch (kind) {
    case CatalogError::Kind::kHeld: return "PublisherLeaseHeldError";
    case CatalogError::Kind::kLease: return "PublisherLeaseError";
    case CatalogError::Kind::kAllocation:
      return "CatalogVersionAllocationError";
    case CatalogError::Kind::kPublishRace: return "SnapshotPublishRaceError";
    case CatalogError::Kind::kPublishConflict:
      return "SnapshotPublishConflictError";
    case CatalogError::Kind::kQuarantined: return "WriterQuarantinedError";
    case CatalogError::Kind::kSchema: return "CatalogSchemaVersionError";
    case CatalogError::Kind::kValue: return "ValueError";
  }
  return "CatalogError";
}

void escape_into(const std::string& value, std::string* out) {
  jc::EscapeJson(value, out);
}

// An integer field, refusing a literal that does not fit in the 64-bit union
// instead of handing the caller `jc::FindInt`'s -1.
//
// -1 was legal-looking at every site in this driver: the unsigned config
// fields (lease_ttl_ns, the reader's read bounds, the index versions and row
// counts) cast it to 18446744073709551615, `HasKey`-guarded fields took it as
// a value the caller had actually supplied, and a `uint`/`int` query parameter
// rendered 18446744073709551615 or -1 straight into a statement -- valid SQL
// carrying a value nobody asked for.
//
// The union itself is legal input and stays so: step_number, token_start/end
// and captured_at_ns are UInt64 in the catalog, so anything in
// [-2**63, 2**64 - 1] must round-trip through the two's-complement bit
// pattern exactly. Only a wider literal has no 64-bit answer at all, and
// ValueError is the shape `captured_bound` below already refuses one with.
//
// A field the catalog types as SIGNED must pass IntDomain::kSigned. The
// union's bit pattern is a legal-looking negative to such a field:
// layer_number is Int32, and 18446744073709551615 rendered as -1, its own
// "no layer" sentinel, straight into the column.
int64_t field_int(const std::string& text, const char* key,
                  jc::IntDomain domain = jc::IntDomain::kUnion) {
  int64_t value = 0;
  const jc::IntFind found = jc::FindIntChecked(text, key, &value, domain);
  if (found == jc::IntFind::kOutOfRange) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        std::string(key) + " does not fit a " +
            (domain == jc::IntDomain::kSigned ? "signed " : "") +
            "64-bit integer");
  }
  // kAbsent keeps answering -1: `limit` and `layer_number` both use it.
  return found == jc::IntFind::kOk ? value : -1;
}

void emit_lease(const PublisherLease& lease, std::string* out) {
  *out += ",\"lease\":{\"term\":" + std::to_string(lease.term) +
          ",\"lease_id\":\"" + lease.lease_id + "\",\"holder\":";
  escape_into(lease.holder, out);
  *out += ",\"acquired_at_ns\":" + std::to_string(lease.acquired_at_ns) +
          ",\"expires_at_ns\":" + std::to_string(lease.expires_at_ns) + "}";
}

std::string sql_string_or_null(const std::string& line, const char* key) {
  if (jc::FindNull(line, key)) return "NULL";
  return dmi_catalog::sql_quote(jc::FindString(line, key));
}

// The reader config every read op builds, in one place. The read bounds
// are overridable so a test can set one low enough that the server
// refuses -- which is how "the bounds actually ride on this statement"
// becomes observable rather than asserted.
dmi_catalog::ReaderConfig reader_config(const std::string& database,
                                        const std::string& table_prefix,
                                        const std::string& line) {
  dmi_catalog::ReaderConfig rc;
  rc.database = database;
  rc.table_prefix = table_prefix;
  if (jc::HasKey(line, "max_rows_to_read")) {
    rc.max_rows_to_read =
        static_cast<uint64_t>(field_int(line, "max_rows_to_read"));
  }
  if (jc::HasKey(line, "max_bytes_to_read")) {
    rc.max_bytes_to_read =
        static_cast<uint64_t>(field_int(line, "max_bytes_to_read"));
  }
  if (jc::HasKey(line, "max_execution_time_s")) {
    rc.max_execution_time_s =
        static_cast<uint64_t>(field_int(line, "max_execution_time_s"));
  }
  return rc;
}

// The query `limit`, which is an `int` on SearchFilters and is bounded by
// the reader at [1, 10000].
//
// Two things go wrong if this is written as one unconditional
// `static_cast<int>(field_int(line, "limit"))`.
//
// The cast TRUNCATES: a value above 2**32 that is not a multiple of 2**32
// lands back inside the band, so 4294967297 became 1 and the statement was
// executed at row_limit 2 where CaptureQuery raises. 4294967296 and
// 2**64 - 1 truncate to 0 and -1, which the band already refuses -- which is
// what hid it. The width is therefore checked BEFORE the cast, and refused
// with the bound's own message, because that is the message the oracle
// raises for every value outside [1, 10000].
//
// And the assignment being unconditional destroys the DEFAULT: CaptureQuery
// defaults limit to 1000, SearchFilters is initialised to 1000, but an
// absent key was overwritten with field_int's kAbsent -1 and refused. The
// key is presence-guarded here for the same reason reader_config guards its
// three bounds.
void read_limit(const std::string& line,
                dmi_catalog::SearchFilters* filters) {
  if (!jc::HasKey(line, "limit")) return;
  const int64_t limit = field_int(line, "limit");
  if (limit < std::numeric_limits<int>::min() ||
      limit > std::numeric_limits<int>::max()) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "limit must be between 1 and 10000");
  }
  filters->limit = static_cast<int>(limit);
}

// The elements of a JSON array, with an EMPTY array yielding none.
// SplitElements("") answers one empty element -- correct for splitting,
// wrong for "was anything listed" -- so an empty filter list became a
// filter on the empty string (matching nothing) rather than no filter at
// all, where the Python query treats an empty tuple as absent.
std::vector<std::string> read_array(const std::string& line,
                                    const char* key) {
  const std::string inside = jc::Unwrap(jc::FindArray(line, key));
  if (inside.find_first_not_of(" \t\n") == std::string::npos) return {};
  return jc::SplitElements(inside);
}

// `[{"name": ..., "str"|"uint"|"int": ...}, ...]` -> a Params map. One
// key per element names the variant arm, because the three arms render
// differently (a str is quoted, the integers are not).
dmi_catalog::Params read_params(const std::string& line) {
  dmi_catalog::Params params;
  for (const std::string& element :
       jc::SplitElements(jc::Unwrap(jc::FindArray(line, "params")))) {
    const std::string name = jc::FindString(element, "name");
    if (name.empty()) continue;
    if (jc::HasKey(element, "uint")) {
      params[name] = static_cast<uint64_t>(field_int(element, "uint"));
    } else if (jc::HasKey(element, "int")) {
      params[name] = static_cast<int64_t>(field_int(element, "int"));
    } else {
      params[name] = jc::FindString(element, "str");
    }
  }
  return params;
}

std::vector<PackIdentity> read_identities(const std::string& line,
                                          const char* key) {
  std::vector<PackIdentity> out;
  for (const std::string& element :
       jc::SplitElements(jc::Unwrap(jc::FindArray(line, key)))) {
    const std::string store = jc::FindString(element, "store_id");
    const std::string pack = jc::FindString(element, "pack_id");
    if (!store.empty() || !pack.empty()) out.emplace_back(store, pack);
  }
  return out;
}

// The 33 capture_raw columns in CAPTURE_COLUMNS order, rendered as one
// VALUES row. Facet columns are MATERIALIZED — the server computes them.
std::string render_descriptor_row(const std::string& descriptor) {
  std::vector<std::string> fields;
  const auto text_field = [&](const char* key) {
    fields.push_back(dmi_catalog::sql_quote(jc::FindString(descriptor, key)));
  };
  const auto int_field = [&](const char* key) {
    fields.push_back(std::to_string(field_int(descriptor, key)));
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
  fields.push_back(sql_string_or_null(descriptor, "adapter_revision"));
  text_field("capture_policy_version");
  text_field("hook_name");
  // The one SIGNED capture column: Int32, with -1 as its legal sentinel.
  fields.push_back(std::to_string(
      field_int(descriptor, "layer_number", jc::IntDomain::kSigned)));
  int_field("producer_rank");
  int_field("step_number");
  int_field("token_start");
  int_field("token_end");
  int_field("batch_position");
  text_field("dtype");
  std::string shape = "[";
  bool first = true;
  for (const std::string& dim : jc::SplitElements(
           jc::Unwrap(jc::FindArray(descriptor, "shape")))) {
    if (!first) shape += ",";
    first = false;
    // Dims are JSON numbers; the raw element text is already valid SQL.
    shape += dim;
  }
  shape += "]";
  fields.push_back(shape);
  int_field("captured_at_ns");
  fields.push_back(
      dmi_catalog::sql_uuid(jc::FindString(descriptor, "pack_id")));
  text_field("store_id");
  text_field("object_key");
  int_field("object_bytes");
  text_field("pack_checksum");
  int_field("pack_record_count");
  int_field("payload_offset");
  int_field("stored_length");
  int_field("decoded_length");
  text_field("codec");
  text_field("payload_checksum");
  std::string row;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) row += ",";
    row += fields[i];
  }
  return row;
}

// The six version-independent pack columns; commit_packs appends the
// batch's index_version itself, the same convention as write_descriptors.
std::string render_pack_row(const std::string& ref) {
  std::vector<std::string> fields;
  fields.push_back(dmi_catalog::sql_uuid(jc::FindString(ref, "pack_id")));
  fields.push_back(dmi_catalog::sql_quote(jc::FindString(ref, "store_id")));
  fields.push_back(dmi_catalog::sql_quote(jc::FindString(ref, "object_key")));
  fields.push_back(std::to_string(field_int(ref, "object_bytes")));
  fields.push_back(dmi_catalog::sql_quote(jc::FindString(ref, "pack_checksum")));
  fields.push_back(std::to_string(field_int(ref, "record_count")));
  std::string row;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) row += ",";
    row += fields[i];
  }
  return row;
}

std::string rows_to_json(const std::vector<dmi_catalog::Row>& rows) {
  std::string out = ",\"rows\":[";
  bool first_row = true;
  for (const auto& row : rows) {
    if (!first_row) out += ",";
    first_row = false;
    out += "[";
    bool first_field = true;
    for (const auto& field : row) {
      if (!first_field) out += ",";
      first_field = false;
      escape_into(field, &out);
    }
    out += "]";
  }
  return out + "]";
}

struct Session {
  std::shared_ptr<const dmi_catalog::ClickHouseClient> client;
  std::unique_ptr<CatalogWriter> writer;
  std::string database;
  std::string table_prefix;
};

Session make_session(const std::string& line) {
  Session session;
  dmi_catalog::WriterConfig config;
  config.database = jc::FindString(line, "database");
  config.table_prefix = jc::FindString(line, "table_prefix");
  config.lease_ttl_ns = static_cast<uint64_t>(field_int(line, "lease_ttl_ns"));
  config.publish_timeout_ns =
      static_cast<uint64_t>(field_int(line, "publish_timeout_ns"));
  config.clock_skew_ns = static_cast<uint64_t>(field_int(line, "clock_skew_ns"));
  config.allocation_attempts =
      static_cast<int>(field_int(line, "allocation_attempts"));
  if (jc::HasKey(line, "query_pack_limit")) {
    config.query_pack_limit = static_cast<int>(field_int(line, "query_pack_limit"));
  }
  if (jc::HasKey(line, "insert_quorum") &&
      !jc::FindNull(line, "insert_quorum")) {
    config.insert_quorum =
        static_cast<uint64_t>(field_int(line, "insert_quorum"));
  }
  const char* host = getenv("DMI_CLICKHOUSE_HOST");
  const char* port = getenv("DMI_CLICKHOUSE_HTTP_PORT");
  // HTTP interface: the driver speaks HTTP (libcurl), not the native TCP
  // protocol clickhouse-driver uses, so the port differs from the
  // Python-side suites' DMI_CLICKHOUSE_PORT. 8123 is ClickHouse's default.
  session.client = std::make_shared<const dmi_catalog::ClickHouseClient>(
      host != nullptr ? host : "127.0.0.1",
      static_cast<uint16_t>(port != nullptr ? std::atoi(port) : 8123));
  session.writer = std::make_unique<CatalogWriter>(session.client, config);
  session.database = config.database;
  session.table_prefix = config.table_prefix;
  return session;
}

std::string respond(const std::string& line, Session* session) {
  const std::string op = jc::FindString(line, "op");
  const std::string prefix = "{\"op\":\"" + op + "\",\"ok\":";
  std::string out;
  try {
    if (op == "open") {
      *session = make_session(line);
      return prefix + "true}";
    }
    if (op == "escape") {
      // No session and no server: the SQL string escaper alone, so the
      // escape_chars_map parity gate can run on the CPU gate.
      out = ",\"escaped\":";
      escape_into(dmi_catalog::sql_quote(jc::FindString(line, "value")), &out);
      return prefix + "true" + out + "}";
    }
    if (op == "substitute") {
      // The escaper's sibling, and session-less for the same reason: the
      // `%(name)s` SCANNER against clickhouse-driver's `query % escaped`.
      // `escape` gated the rendering of one value and nothing gated the
      // walk over the statement, which is where a parameter reaching into
      // another parameter's already-rendered text hid.
      out = ",\"statement\":";
      escape_into(dmi_catalog::substitute(jc::FindString(line, "query"),
                                          read_params(line)), &out);
      return prefix + "true" + out + "}";
    }
    if (op == "footer_row_fields") {
      // Session-less like `escape`: the footer-row decoder alone, so the
      // footer binding's representation rules -- decoded strings, NULL as
      // a kind rather than a spelling, toUUID unwrapped -- are pinned on
      // the CPU gate. Optionally compares each field against a catalog
      // value, the way hydrate does.
      const std::vector<dmi_catalog::FooterField> fields =
          dmi_catalog::split_footer_row(jc::FindString(line, "row"));
      std::vector<std::string> catalog;
      const std::string catalog_array = jc::FindArray(line, "catalog");
      if (!catalog_array.empty()) {
        for (const std::string& item :
             jc::SplitElements(jc::Unwrap(catalog_array))) {
          catalog.push_back(jc::ParseLiteral(item));
        }
      }
      out = ",\"fields\":[";
      for (size_t i = 0; i < fields.size(); ++i) {
        if (i) out += ",";
        out += "{\"null\":";
        out += fields[i].is_null ? "true" : "false";
        out += ",\"text\":";
        escape_into(fields[i].text, &out);
        if (i < catalog.size()) {
          out += ",\"matches\":";
          out += dmi_catalog::footer_field_matches(catalog[i], fields[i])
                     ? "true"
                     : "false";
        }
        out += "}";
      }
      out += "]";
      return prefix + "true" + out + "}";
    }
    if (op == "tuple_fields") {
      // Session-less like `escape` and `footer_row_fields`: the resolved
      // tuple parser alone. It decides what a catalog descriptor field IS
      // -- in particular whether it was SQL NULL -- and the footer binding
      // compares against exactly this, so it belongs on the CPU gate and
      // not only behind a live catalog.
      out = ",\"fields\":[";
      const std::vector<std::string> fields =
          dmi_catalog::parse_tsv_tuple(jc::FindString(line, "tuple"));
      for (size_t i = 0; i < fields.size(); ++i) {
        if (i) out += ",";
        escape_into(fields[i], &out);
      }
      out += "]";
      return prefix + "true" + out + "}";
    }
    if (session->writer == nullptr) {
      return prefix + "false,\"what\":\"call open first\"}";
    }
    CatalogWriter& writer = *session->writer;

    if (op == "head") {
      const auto head = writer.leases().head();
      out = ",\"head\":{\"term\":" + std::to_string(head.term) +
            ",\"claimants\":" + std::to_string(head.claimants) +
            ",\"lease_id\":\"" + head.lease_id + "\",\"holder\":";
      escape_into(head.holder, &out);
      out += ",\"expires_at_ns\":" + std::to_string(head.expires_at_ns) +
             ",\"live_until_ns\":" + std::to_string(head.live_until_ns) +
             ",\"now_ns\":" + std::to_string(head.now_ns) + "}";
    } else if (op == "acquire") {
      emit_lease(writer.acquire_lease(jc::FindString(line, "holder")), &out);
    } else if (op == "renew") {
      emit_lease(writer.renew_lease(), &out);
    } else if (op == "release") {
      writer.release_lease();
    } else if (op == "lease") {
      const PublisherLease* held = writer.held_lease();
      if (held != nullptr) emit_lease(*held, &out);
      else out = ",\"lease\":null";
    } else if (op == "discard_local_lease") {
      writer.leases().discard_local_lease();
    } else if (op == "claim") {
      emit_lease(writer.leases().claim(jc::FindString(line, "holder"),
                                       jc::FindString(line, "lease_id")),
                 &out);
    } else if (op == "claim_contested") {
      emit_lease(writer.leases().claim_contested(
                     jc::FindString(line, "holder"),
                     jc::FindString(line, "rival_lease_id")),
                 &out);
    } else if (op == "reject_if_gone") {
      writer.leases().reject_if_gone();
    } else if (op == "statements") {
      out = ",\"release\":";
      escape_into(writer.leases().release_statement(), &out);
      out += ",\"fence\":";
      escape_into(writer.leases().fence(), &out);
    } else if (op == "allocate_version") {
      out = ",\"version\":" + std::to_string(writer.allocate_version());
    } else if (op == "select") {
      dmi_catalog::ReaderConfig rc =
          reader_config(session->database, session->table_prefix, line);
      dmi_store::S3Config s3_config;
      s3_config.endpoint = jc::FindString(line, "endpoint");
      s3_config.bucket = jc::FindString(line, "bucket");
      s3_config.access_key = jc::FindString(line, "access");
      s3_config.secret_key = jc::FindString(line, "secret");
      s3_config.allow_insecure_http = jc::FindBool(line, "insecure");
      dmi_store::S3Client s3(s3_config);
      dmi_catalog::NativeCaptureReader reader(
          &s3, s3_config.bucket, session->client, rc);
      dmi_catalog::SearchFilters filters;
      if (jc::HasKey(line, "tenant_id") && !jc::FindNull(line, "tenant_id")) {
        filters.tenant_id = jc::FindString(line, "tenant_id");
      }
      read_limit(line, &filters);
      const dmi_catalog::Selection selection = reader.select(filters);
      out = ",\"selection\":{\"selection_id\":";
      escape_into(selection.selection_id, &out);
      out += ",\"capture_ids\":[";
      for (size_t i = 0; i < selection.capture_ids.size(); ++i) {
        if (i) out += ",";
        escape_into(selection.capture_ids[i], &out);
      }
      out += "],\"catalog_watermark\":";
      escape_into(selection.catalog_watermark, &out);
      out += ",\"filter_hash\":";
      escape_into(selection.filter_hash, &out);
      out += ",\"tenant_id\":";
      escape_into(selection.tenant_id, &out);
      out += "}";
    } else if (op == "hydrate") {
      dmi_catalog::ReaderConfig rc =
          reader_config(session->database, session->table_prefix, line);
      dmi_store::S3Config s3_config;
      s3_config.endpoint = jc::FindString(line, "endpoint");
      s3_config.bucket = jc::FindString(line, "bucket");
      s3_config.access_key = jc::FindString(line, "access");
      s3_config.secret_key = jc::FindString(line, "secret");
      s3_config.allow_insecure_http = jc::FindBool(line, "insecure");
      dmi_store::S3Client s3(s3_config);
      dmi_catalog::NativeCaptureReader reader(
          &s3, s3_config.bucket, session->client, rc);
      dmi_catalog::Selection selection;
      selection.selection_id = jc::FindString(line, "selection_id");
      for (const std::string& id : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "capture_ids")))) {
        selection.capture_ids.push_back(jc::ParseLiteral(id));
      }
      selection.catalog_watermark = jc::FindString(line, "catalog_watermark");
      selection.filter_hash = jc::FindString(line, "filter_hash");
      selection.tenant_id = jc::FindString(line, "tenant_id");
      const auto payloads = reader.hydrate(
          selection, field_int(line, "byte_limit"),
          jc::HasKey(line, "request_limit")
              ? field_int(line, "request_limit")
              : 1024);
      out = ",\"payloads\":[";
      for (size_t i = 0; i < payloads.size(); ++i) {
        if (i) out += ",";
        std::string encoded;
        jc::EncodeBase64(
            reinterpret_cast<const uint8_t*>(payloads[i].data()),
            payloads[i].size(), &encoded);
        out += "\"" + encoded + "\"";
      }
      out += "]";
    } else if (op == "summarize_core") {
      dmi_catalog::ReaderConfig rc =
          reader_config(session->database, session->table_prefix, line);
      dmi_store::S3Config s3_config;
      s3_config.endpoint = jc::FindString(line, "endpoint");
      s3_config.bucket = jc::FindString(line, "bucket");
      s3_config.access_key = jc::FindString(line, "access");
      s3_config.secret_key = jc::FindString(line, "secret");
      s3_config.allow_insecure_http = jc::FindBool(line, "insecure");
      dmi_store::S3Client s3(s3_config);
      dmi_catalog::NativeCaptureReader reader(
          &s3, s3_config.bucket, session->client, rc);
      dmi_catalog::Selection selection;
      selection.selection_id = jc::FindString(line, "selection_id");
      for (const std::string& id : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "capture_ids")))) {
        selection.capture_ids.push_back(jc::ParseLiteral(id));
      }
      selection.catalog_watermark = jc::FindString(line, "catalog_watermark");
      selection.filter_hash = jc::FindString(line, "filter_hash");
      selection.tenant_id = jc::FindString(line, "tenant_id");
      const auto summaries = reader.summarize_core(
          selection, field_int(line, "byte_limit"),
          jc::HasKey(line, "request_limit")
              ? field_int(line, "request_limit")
              : 1024,
          1000, 64'000'000ull);
      out = ",\"summaries\":[";
      for (size_t i = 0; i < summaries.size(); ++i) {
        if (i) out += ",";
        const auto& [capture_id, s] = summaries[i];
        out += "{\"capture_id\":";
        escape_into(capture_id, &out);
        // %.17g round-trips a double exactly; std::to_string would truncate
        // to six places and the parity comparison would fail on precision.
        const auto real = [](double value) {
          char buffer[40];
          std::snprintf(buffer, sizeof(buffer), "%.17g", value);
          return std::string(buffer);
        };
        out += ",\"summary_version\":" + std::to_string(s.summary_version) +
               ",\"element_count\":" + std::to_string(s.element_count) +
               ",\"finite_count\":" + std::to_string(s.finite_count) +
               ",\"nan_count\":" + std::to_string(s.nan_count) +
               ",\"inf_count\":" + std::to_string(s.inf_count) +
               ",\"zero_fraction\":" + real(s.zero_fraction) +
               ",\"mean\":" + real(s.mean) +
               ",\"minimum\":" + real(s.minimum) +
               ",\"maximum\":" + real(s.maximum) +
               ",\"abs_max\":" + real(s.abs_max) +
               ",\"l2_norm\":" + real(s.l2_norm) +
               ",\"minimum_int\":" + std::to_string(s.minimum_int) +
               ",\"maximum_int\":" + std::to_string(s.maximum_int) +
               ",\"abs_max_int\":" + std::to_string(s.abs_max_int) + "}";
      }
      out += "]";
    } else if (op == "current_watermark") {
      dmi_catalog::NativeCaptureCatalog reader(
          session->client,
          reader_config(session->database, session->table_prefix, line));
      out = ",\"watermark\":";
      escape_into(reader.current_watermark(), &out);
    } else if (op == "search") {
      dmi_catalog::ReaderConfig rc =
          reader_config(session->database, session->table_prefix, line);
      dmi_catalog::NativeCaptureCatalog reader(session->client, rc);
      dmi_catalog::SearchFilters filters;
      for (const char* key : {"tenant_id", "experiment_id", "run_id",
                              "session_id", "model_id"}) {
        if (jc::HasKey(line, key) && !jc::FindNull(line, key)) {
          if (key == std::string("tenant_id")) {
            filters.tenant_id = jc::FindString(line, key);
          } else if (key == std::string("experiment_id")) {
            filters.experiment_id = jc::FindString(line, key);
          } else if (key == std::string("run_id")) {
            filters.run_id = jc::FindString(line, key);
          } else if (key == std::string("session_id")) {
            filters.session_id = jc::FindString(line, key);
          } else if (key == std::string("model_id")) {
            filters.model_id = jc::FindString(line, key);
          }
        }
      }
      if (jc::HasKey(line, "hook_names") && !jc::FindNull(line, "hook_names")) {
        for (const std::string& hook : read_array(line, "hook_names")) {
          filters.hook_names.push_back(jc::ParseLiteral(hook));
        }
      }
      if (jc::HasKey(line, "layer_numbers") &&
          !jc::FindNull(line, "layer_numbers")) {
        for (const std::string& layer : read_array(line, "layer_numbers")) {
          filters.layer_numbers.push_back(
              static_cast<int64_t>(std::atoll(layer.c_str())));
        }
      }
      // CaptureQuery requires each captured bound to be a NON-NEGATIVE
      // INTEGER, and SearchFilters carries them as uint64_t -- which is
      // what makes the reader itself safe, and what makes this decode the
      // boundary that has to refuse a bad spelling.
      // static_cast<uint64_t>(jc::FindInt(...)) turned -1 into
      // 18446744073709551615, and a non-numeric value into the same
      // UINT64_MAX by way of FindInt's -1 sentinel, so search answered an
      // empty page where the oracle raises.
      const auto captured_bound = [&line](const char* key) -> uint64_t {
        std::string token;
        for (const char* sep : {": ", ":"}) {
          const std::string needle = std::string("\"") + key + "\"" + sep;
          const size_t at = line.find(needle);
          if (at == std::string::npos) continue;
          const size_t begin = at + needle.size();
          size_t q = begin;
          while (q < line.size() && line[q] != ',' && line[q] != '}') ++q;
          token = line.substr(begin, q - begin);
          break;
        }
        const size_t first = token.find_first_not_of(" \t\n\r");
        const size_t last = token.find_last_not_of(" \t\n\r");
        const std::string digits =
            first == std::string::npos
                ? std::string()
                : token.substr(first, last - first + 1);
        if (digits.empty() ||
            digits.find_first_not_of("0123456789") != std::string::npos) {
          throw CatalogError(CatalogError::Kind::kValue,
                             std::string(key) +
                                 " must be a non-negative integer");
        }
        uint64_t value = 0;
        for (const char c : digits) {
          const uint64_t digit = static_cast<uint64_t>(c - '0');
          // The oracle accepts an integer beyond UInt64 here and fails
          // later, in its driver's parameter binding; this protocol's
          // field cannot hold one, so refuse it rather than wrap it.
          if (value > (UINT64_MAX - digit) / 10) {
            throw CatalogError(CatalogError::Kind::kValue,
                               std::string(key) + " does not fit UInt64");
          }
          value = value * 10 + digit;
        }
        return value;
      };
      if (jc::HasKey(line, "captured_after_ns") &&
          !jc::FindNull(line, "captured_after_ns")) {
        filters.captured_after_ns = captured_bound("captured_after_ns");
      }
      if (jc::HasKey(line, "captured_before_ns") &&
          !jc::FindNull(line, "captured_before_ns")) {
        filters.captured_before_ns = captured_bound("captured_before_ns");
      }
      if (jc::HasKey(line, "cursor") && !jc::FindNull(line, "cursor")) {
        filters.cursor = jc::FindString(line, "cursor");
      }
      read_limit(line, &filters);
      const dmi_catalog::SearchPage page = reader.search(filters);
      out = ",\"items\":[";
      bool first_item = true;
      for (const auto& item : page.items) {
        if (!first_item) out += ",";
        first_item = false;
        out += "[";
        bool first_field = true;
        for (const auto& field : item) {
          if (!first_field) out += ",";
          first_field = false;
          escape_into(field, &out);
        }
        out += "]";
      }
      out += "],\"next_cursor\":";
      if (page.next_cursor.has_value()) {
        escape_into(*page.next_cursor, &out);
      } else {
        out += "null";
      }
      out += ",\"watermark\":";
      escape_into(page.watermark, &out);
    } else if (op == "get_by_ids") {
      dmi_catalog::ReaderConfig rc =
          reader_config(session->database, session->table_prefix, line);
      dmi_catalog::NativeCaptureCatalog reader(session->client, rc);
      std::vector<std::string> capture_ids;
      for (const std::string& id : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "capture_ids")))) {
        capture_ids.push_back(jc::ParseLiteral(id));
      }
      const auto items = reader.get_by_ids(
          capture_ids, jc::FindString(line, "tenant_id"),
          jc::FindString(line, "watermark"));
      out = ",\"items\":[";
      bool first_item = true;
      for (const auto& item : items) {
        if (!first_item) out += ",";
        first_item = false;
        out += "[";
        bool first_field = true;
        for (const auto& field : item) {
          if (!first_field) out += ",";
          first_field = false;
          escape_into(field, &out);
        }
        out += "]";
      }
      out += "]";
    } else if (op == "verify_compatibility") {
      // The verdict on its own, without the DDL that `ensure_schema` runs
      // after it: the refusals are most of the schema port, and reaching
      // them through `ensure_schema` alone means a test cannot tell a
      // refusal from a failure of the install that follows one.
      dmi_catalog::CatalogSchema schema(session->client, session->database,
                                        session->table_prefix);
      out = ",\"state\":\"" + schema.verify_compatibility() + "\"";
    } else if (op == "ensure_schema") {
      uint64_t retry_sleep_ns = 500'000'000ull;
      if (jc::HasKey(line, "retry_sleep_ns")) {
        retry_sleep_ns = static_cast<uint64_t>(field_int(line, "retry_sleep_ns"));
      }
      dmi_catalog::CatalogSchema schema(session->client, session->database,
                                        session->table_prefix);
      schema.ensure(&writer.leases(), retry_sleep_ns);
    } else if (op == "drop_schema") {
      dmi_catalog::CatalogSchema schema(session->client, session->database,
                                        session->table_prefix);
      schema.drop();
    } else if (op == "collect_garbage") {
      const auto removed = writer.collect_garbage(
          static_cast<uint64_t>(field_int(line, "settle_sleep_ns")));
      out = ",\"removed\":{";
      bool first_table = true;
      for (const auto& [table, count] : removed) {
        if (!first_table) out += ",";
        first_table = false;
        escape_into(table, &out);
        out += ":" + std::to_string(count);
      }
      out += "}";
    } else if (op == "last_published_version") {
      out = ",\"version\":" + std::to_string(writer.last_published_version());
    } else if (op == "max_version") {
      out = ",\"version\":" + std::to_string(writer.max_version(
                jc::FindString(line, "table"),
                jc::FindString(line, "column")));
    } else if (op == "quarantined") {
      out = std::string(",\"quarantined\":") +
            (writer.quarantined() ? "true" : "false");
    } else if (op == "write_descriptors") {
      std::vector<std::string> rows;
      for (const std::string& descriptor : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "descriptors")))) {
        rows.push_back(render_descriptor_row(descriptor));
      }
      writer.write_descriptors(
          rows, static_cast<uint64_t>(field_int(line, "index_version")));
    } else if (op == "commit_packs") {
      std::vector<std::string> rows;
      for (const std::string& ref : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "refs")))) {
        rows.push_back(render_pack_row(ref));
      }
      writer.commit_packs(
          rows, static_cast<uint64_t>(field_int(line, "index_version")));
    } else if (op == "committed_pack_ids") {
      const auto committed = writer.committed_pack_ids(
          read_identities(line, "identities"));
      out = ",\"committed\":[";
      bool first = true;
      for (const auto& [store_id, pack_id] : committed) {
        if (!first) out += ",";
        first = false;
        out += "{\"store_id\":\"" + store_id + "\",\"pack_id\":\"" +
               pack_id + "\"}";
      }
      out += "]";
    } else if (op == "publish_snapshot") {
      const std::string* takeover_after_renew = nullptr;
      const std::string* takeover_after_chunks = nullptr;
      std::string renew_holder, chunks_holder;
      if (jc::HasKey(line, "takeover_after_renew")) {
        renew_holder = jc::FindString(line, "takeover_after_renew");
        takeover_after_renew = &renew_holder;
      }
      if (jc::HasKey(line, "takeover_after_chunks")) {
        chunks_holder = jc::FindString(line, "takeover_after_chunks");
        takeover_after_chunks = &chunks_holder;
      }
      writer.publish_snapshot(
          static_cast<uint64_t>(field_int(line, "index_version")),
          read_identities(line, "refs"),
          static_cast<uint64_t>(field_int(line, "published_at_ns")),
          static_cast<uint64_t>(field_int(line, "indexed_rows")),
          static_cast<uint64_t>(field_int(line, "indexed_packs")),
          static_cast<uint64_t>(field_int(line, "wedge_ns")),
          takeover_after_renew, takeover_after_chunks,
          jc::FindBool(line, "inject_transport_error"));
    } else if (op == "publish_concurrent") {
      // #125, in the only shape that can test it: the driver runs one op at
      // a time, so two publishes can only overlap if THIS process runs them
      // on two threads. Each thread reports when it entered and left
      // publish_snapshot, and the test reads the intervals -- serialised
      // means they do not overlap, whatever the wedge asks for.
      struct Attempt {
        uint64_t started_ns = 0;
        uint64_t finished_ns = 0;
        bool ok = false;
        std::string error;
        std::string message;
      };
      const auto now_ns = [] {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
      };
      const std::vector<PackIdentity> refs = read_identities(line, "refs");
      const uint64_t rows =
          static_cast<uint64_t>(field_int(line, "indexed_rows"));
      const uint64_t packs =
          static_cast<uint64_t>(field_int(line, "indexed_packs"));
      const uint64_t wedge =
          static_cast<uint64_t>(field_int(line, "wedge_ns"));
      Attempt first, second;
      const auto publish = [&](Attempt* attempt, uint64_t version,
                               uint64_t wedge_ns) {
        attempt->started_ns = now_ns();
        try {
          writer.publish_snapshot(version, refs, version, rows, packs,
                                  wedge_ns);
          attempt->ok = true;
        } catch (const CatalogError& e) {
          attempt->error = error_kind(e.kind());
          attempt->message = e.what();
        } catch (const std::exception& e) {
          attempt->error = "DriverError";
          attempt->message = e.what();
        }
        attempt->finished_ns = now_ns();
      };
      const uint64_t version_a =
          static_cast<uint64_t>(field_int(line, "index_version_a"));
      const uint64_t version_b =
          static_cast<uint64_t>(field_int(line, "index_version_b"));
      std::thread a(publish, &first, version_a, wedge);
      // A short stagger so the wedged publish is demonstrably first; the
      // point is whether the second WAITS, not who wins a start race.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      std::thread b(publish, &second, version_b, 0ull);
      a.join();
      b.join();
      const auto emit = [&](const char* name, const Attempt& attempt) {
        std::string part = std::string(",\"") + name + "\":{\"ok\":" +
                           (attempt.ok ? "true" : "false") +
                           ",\"started_ns\":" +
                           std::to_string(attempt.started_ns) +
                           ",\"finished_ns\":" +
                           std::to_string(attempt.finished_ns);
        if (!attempt.ok) {
          part += ",\"error\":\"" + attempt.error + "\",\"message\":";
          std::string escaped;
          escape_into(attempt.message, &escaped);
          part += escaped;
        }
        return part + "}";
      };
      out = emit("first", first) + emit("second", second);
    } else if (op == "publish_from_forked_child") {
      // The other half of #125: a writer belongs to the process that built
      // it. The child inherits the lease id and the socket, and the pid
      // check must refuse it BEFORE any lock is taken -- a fork during a
      // publish copies a held lock whose owner thread does not exist in the
      // child, so a check behind the lock would hang there forever.
      int fds[2];
      if (pipe(fds) != 0) {
        return prefix + "false,\"what\":\"pipe failed\"}";
      }
      const pid_t child = fork();
      if (child == 0) {
        close(fds[0]);
        std::string report = "ok";
        try {
          writer.publish_snapshot(
              static_cast<uint64_t>(field_int(line, "index_version")),
              read_identities(line, "refs"),
              static_cast<uint64_t>(field_int(line, "published_at_ns")),
              static_cast<uint64_t>(field_int(line, "indexed_rows")),
              static_cast<uint64_t>(field_int(line, "indexed_packs")));
        } catch (const CatalogError& e) {
          report = std::string(error_kind(e.kind())) + ":" + e.what();
        } catch (const std::exception& e) {
          report = std::string("DriverError:") + e.what();
        }
        const ssize_t written =
            write(fds[1], report.data(), report.size());
        (void)written;
        close(fds[1]);
        _exit(0);
      }
      close(fds[1]);
      std::string report;
      char buffer[4096];
      ssize_t got;
      while ((got = read(fds[0], buffer, sizeof(buffer))) > 0) {
        report.append(buffer, static_cast<size_t>(got));
      }
      close(fds[0]);
      int status = 0;
      waitpid(child, &status, 0);
      out = ",\"child\":";
      escape_into(report, &out);
    } else if (op == "index") {
      // B3: read the packs through the store and run the indexer loop.
      dmi_store::S3Config s3_config;
      s3_config.endpoint = jc::FindString(line, "endpoint");
      s3_config.bucket = jc::FindString(line, "bucket");
      s3_config.region = jc::FindString(line, "region");
      s3_config.access_key = jc::FindString(line, "access");
      s3_config.secret_key = jc::FindString(line, "secret");
      s3_config.allow_insecure_http = jc::FindBool(line, "insecure");
      dmi_store::S3Client s3(s3_config);
      std::vector<dmi_catalog::PackRefData> refs;
      for (const std::string& element : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "refs")))) {
        dmi_catalog::PackRefData ref;
        ref.pack_id = jc::FindString(element, "pack_id");
        ref.store_id = jc::FindString(element, "store_id");
        ref.object_key = jc::FindString(element, "object_key");
        ref.object_bytes = static_cast<uint64_t>(field_int(element, "object_bytes"));
        ref.checksum = jc::FindString(element, "checksum");
        ref.record_count = static_cast<uint64_t>(field_int(element, "record_count"));
        refs.push_back(ref);
      }
      dmi_catalog::IndexerConfig index_config;
      if (jc::HasKey(line, "max_packs")) {
        index_config.max_packs = static_cast<int>(field_int(line, "max_packs"));
      }
      if (jc::HasKey(line, "max_estimated_bytes")) {
        index_config.max_estimated_bytes =
            static_cast<uint64_t>(field_int(line, "max_estimated_bytes"));
      }
      if (jc::HasKey(line, "max_publish_attempts")) {
        index_config.max_publish_attempts =
            static_cast<int>(field_int(line, "max_publish_attempts"));
      }
      if (jc::FindBool(line, "foreign_watermark_after_allocate")) {
        // A foreign writer publishing a HIGHER version between this pass's
        // allocation and its publish — the "something else is writing this
        // prefix" case the conflict error names. The barrier then refuses
        // the pass's own watermark, which is a real version race rather
        // than a simulated one.
        const auto client = session->client;
        const std::string watermark = "`" + session->database + "`.`" +
                                      session->table_prefix +
                                      "_index_watermark`";
        index_config.after_allocate = [client, watermark](uint64_t version) {
          client->execute(
              "INSERT INTO " + watermark +
                  " (index_version, publish_id, published_at_ns, "
                  "indexed_rows, indexed_packs) VALUES (%(version)s, "
                  "generateUUIDv4(), 1, 0, 0)",
              {{"version", version + 1}});
        };
      }
      const dmi_catalog::IndexResultData result =
          dmi_catalog::NativeIndexer(&s3, &writer, index_config)
              .index(refs);
      out = ",\"result\":{\"requested_packs\":" +
            std::to_string(result.requested_packs) +
            ",\"skipped_packs\":" + std::to_string(result.skipped_packs) +
            ",\"indexed_packs\":" + std::to_string(result.indexed_packs) +
            ",\"indexed_rows\":" + std::to_string(result.indexed_rows) +
            ",\"failed_packs\":" + std::to_string(result.failed_packs) +
            ",\"descriptor_inserts\":" +
            std::to_string(result.descriptor_inserts) + ",\"failures\":[";
      bool first_failure = true;
      for (const auto& failure : result.failures) {
        if (!first_failure) out += ",";
        first_failure = false;
        out += "{\"pack_id\":";
        escape_into(failure.pack_id, &out);
        out += ",\"object_key\":";
        escape_into(failure.object_key, &out);
        out += ",\"error_type\":";
        escape_into(failure.error_type, &out);
        out += ",\"message\":";
        escape_into(failure.message, &out);
        out += "}";
      }
      out += "]}";
    } else if (op == "fence_eval") {
      const bool admits = writer.leases().fence_eval(
          jc::FindString(line, "lease_id"),
          static_cast<uint64_t>(field_int(line, "publish_timeout_ns")),
          static_cast<uint64_t>(field_int(line, "clock_skew_ns")));
      out = std::string(",\"admits\":") + (admits ? "1" : "0");
    } else if (op == "execute") {
      out = rows_to_json(session->client->execute(
          jc::FindString(line, "query")));
    } else {
      return prefix + "false,\"what\":\"unknown op\"}";
    }
    return prefix + "true" + out + "}";
  } catch (const CatalogError& e) {
    std::string message;
    escape_into(e.what(), &message);
    return prefix + "false,\"error\":\"" + error_kind(e.kind()) +
           "\",\"message\":" + message + "}";
  } catch (const ClickHouseError& e) {
    std::string message;
    escape_into(e.what(), &message);
    return prefix + "false,\"error\":\"ClickHouseError\",\"message\":" +
           message + "}";
  } catch (const std::exception& e) {
    std::string message;
    escape_into(e.what(), &message);
    return prefix + "false,\"error\":\"DriverError\",\"message\":" + message +
           "}";
  }
}

}  // namespace

int main() {
  Session session;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::cout << respond(line, &session) << "\n";
    std::cout.flush();
  }
  return 0;
}
