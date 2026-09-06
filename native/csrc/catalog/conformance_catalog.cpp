// B1/B2 conformance driver: the native lease coordinator, version
// allocator, and catalog writer over a newline-JSON protocol, the same
// shape as the other conformance drivers. One process is one writer
// instance; `open` (re)initializes it from the request's config fields.
// The coordinator-level ops (head/claim/claim_contested) route through
// the writer's coordinator, so B1 tests exercise the bare protocol and
// B2 tests the writer's quarantine-wrapped surface.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../common/json.h"
#include "../store/s3_client.h"
#include "catalog_writer.h"
#include "clickhouse_client.h"
#include "indexer.h"
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

void emit_lease(const PublisherLease& lease, std::string* out) {
  *out += ",\"lease\":{\"term\":" + std::to_string(lease.term) +
          ",\"lease_id\":\"" + lease.lease_id + "\",\"holder\":";
  escape_into(lease.holder, out);
  *out += ",\"acquired_at_ns\":" + std::to_string(lease.acquired_at_ns) +
          ",\"expires_at_ns\":" + std::to_string(lease.expires_at_ns) + "}";
}

std::string render_sql_string(const std::string& value) {
  std::string out = "'";
  for (const char c : value) {
    if (c == '\\' || c == '\'') out.push_back('\\');
    if (c == '\n') {
      out += "\\n";
      continue;
    }
    if (c == '\t') {
      out += "\\t";
      continue;
    }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

std::string sql_string_or_null(const std::string& line, const char* key) {
  if (jc::FindNull(line, key)) return "NULL";
  return render_sql_string(jc::FindString(line, key));
}

std::string sql_int_or_string(const std::string& line, const char* key) {
  // Strings ride as quoted literals; everything numeric renders as-is.
  const std::string raw = jc::FindString(line, key);
  bool numeric = !raw.empty();
  for (const char c : raw) {
    if ((c < '0' || c > '9') && c != '-') numeric = false;
  }
  return numeric ? raw : render_sql_string(raw);
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
    fields.push_back(render_sql_string(jc::FindString(descriptor, key)));
  };
  const auto int_field = [&](const char* key) {
    fields.push_back(std::to_string(jc::FindInt(descriptor, key)));
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
  fields.push_back(std::to_string(jc::FindInt(descriptor, "layer_number")));
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
  fields.push_back("toUUID('" + jc::FindString(descriptor, "pack_id") + "')");
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

std::string render_pack_row(const std::string& ref, uint64_t index_version) {
  std::vector<std::string> fields;
  fields.push_back("toUUID('" + jc::FindString(ref, "pack_id") + "')");
  fields.push_back(render_sql_string(jc::FindString(ref, "store_id")));
  fields.push_back(render_sql_string(jc::FindString(ref, "object_key")));
  fields.push_back(std::to_string(jc::FindInt(ref, "object_bytes")));
  fields.push_back(render_sql_string(jc::FindString(ref, "pack_checksum")));
  fields.push_back(std::to_string(jc::FindInt(ref, "record_count")));
  fields.push_back(std::to_string(index_version));
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
  config.lease_ttl_ns = static_cast<uint64_t>(jc::FindInt(line, "lease_ttl_ns"));
  config.publish_timeout_ns =
      static_cast<uint64_t>(jc::FindInt(line, "publish_timeout_ns"));
  config.clock_skew_ns = static_cast<uint64_t>(jc::FindInt(line, "clock_skew_ns"));
  config.allocation_attempts =
      static_cast<int>(jc::FindInt(line, "allocation_attempts"));
  if (jc::HasKey(line, "query_pack_limit")) {
    config.query_pack_limit = static_cast<int>(jc::FindInt(line, "query_pack_limit"));
  }
  if (jc::HasKey(line, "insert_quorum") &&
      !jc::FindNull(line, "insert_quorum")) {
    config.insert_quorum =
        static_cast<uint64_t>(jc::FindInt(line, "insert_quorum"));
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
    } else if (op == "verify_compatibility") {
      dmi_catalog::CatalogSchema schema(session->client, session->database,
                                        session->table_prefix);
      out = ",\"state\":\"" + schema.verify_compatibility() + "\"";
    } else if (op == "ensure_schema") {
      uint64_t retry_sleep_ns = 500'000'000ull;
      if (jc::HasKey(line, "retry_sleep_ns")) {
        retry_sleep_ns = static_cast<uint64_t>(jc::FindInt(line, "retry_sleep_ns"));
      }
      dmi_catalog::CatalogSchema schema(session->client, session->database,
                                        session->table_prefix);
      schema.ensure(&writer.leases(), retry_sleep_ns);
    } else if (op == "verify_compatibility") {
      // The verdict on its own, without the DDL that `ensure_schema` runs
      // after it: the refusals are most of the schema port, and reaching
      // them through `ensure_schema` alone means a test cannot tell a
      // refusal from a failure of the install that follows one.
      dmi_catalog::CatalogSchema schema(session->client, session->database,
                                        session->table_prefix);
      out = ",\"state\":\"" + schema.verify_compatibility() + "\"";
    } else if (op == "drop_schema") {
      dmi_catalog::CatalogSchema schema(session->client, session->database,
                                        session->table_prefix);
      schema.drop();
    } else if (op == "collect_garbage") {
      const auto removed = writer.collect_garbage(
          static_cast<uint64_t>(jc::FindInt(line, "settle_sleep_ns")));
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
          rows, static_cast<uint64_t>(jc::FindInt(line, "index_version")));
    } else if (op == "commit_packs") {
      std::vector<std::string> rows;
      for (const std::string& ref : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "refs")))) {
        rows.push_back(render_pack_row(
            ref, static_cast<uint64_t>(jc::FindInt(line, "index_version"))));
      }
      writer.commit_packs(
          rows, static_cast<uint64_t>(jc::FindInt(line, "index_version")));
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
          static_cast<uint64_t>(jc::FindInt(line, "index_version")),
          read_identities(line, "refs"),
          static_cast<uint64_t>(jc::FindInt(line, "published_at_ns")),
          static_cast<uint64_t>(jc::FindInt(line, "indexed_rows")),
          static_cast<uint64_t>(jc::FindInt(line, "indexed_packs")),
          static_cast<uint64_t>(jc::FindInt(line, "wedge_ns")),
          takeover_after_renew, takeover_after_chunks,
          jc::FindBool(line, "inject_transport_error"));
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
        ref.object_bytes = static_cast<uint64_t>(jc::FindInt(element, "object_bytes"));
        ref.checksum = jc::FindString(element, "checksum");
        ref.record_count = static_cast<uint64_t>(jc::FindInt(element, "record_count"));
        refs.push_back(ref);
      }
      dmi_catalog::IndexerConfig index_config;
      if (jc::HasKey(line, "max_packs")) {
        index_config.max_packs = static_cast<int>(jc::FindInt(line, "max_packs"));
      }
      if (jc::HasKey(line, "max_estimated_bytes")) {
        index_config.max_estimated_bytes =
            static_cast<uint64_t>(jc::FindInt(line, "max_estimated_bytes"));
      }
      if (jc::HasKey(line, "max_publish_attempts")) {
        index_config.max_publish_attempts =
            static_cast<int>(jc::FindInt(line, "max_publish_attempts"));
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
          dmi_catalog::NativeIndexer(&s3, s3_config.bucket, &writer,
                                     index_config)
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
          static_cast<uint64_t>(jc::FindInt(line, "publish_timeout_ns")),
          static_cast<uint64_t>(jc::FindInt(line, "clock_skew_ns")));
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
