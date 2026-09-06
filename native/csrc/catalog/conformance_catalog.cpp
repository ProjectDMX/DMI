// B1 conformance driver: the native lease coordinator and version
// allocator over a newline-JSON protocol, the same shape as the other
// conformance drivers. One process is one coordinator instance; `open`
// (re)initializes it from the request's config fields.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../common/json.h"
#include "clickhouse_client.h"
#include "lease_coordinator.h"
#include "version_allocator.h"

namespace jc = dmi_common;

namespace {

using dmi_catalog::CatalogError;
using dmi_catalog::ClickHouseError;
using dmi_catalog::LeaseCoordinator;
using dmi_catalog::PublisherLease;

const char* error_kind(CatalogError::Kind kind) {
  switch (kind) {
    case CatalogError::Kind::kHeld: return "PublisherLeaseHeldError";
    case CatalogError::Kind::kLease: return "PublisherLeaseError";
    case CatalogError::Kind::kAllocation:
      return "CatalogVersionAllocationError";
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

struct Session {
  std::shared_ptr<const dmi_catalog::ClickHouseClient> client;
  std::unique_ptr<LeaseCoordinator> leases;
  std::unique_ptr<dmi_catalog::VersionAllocator> allocator;
};

Session make_session(const std::string& line) {
  Session session;
  dmi_catalog::LeaseConfig leases;
  leases.database = jc::FindString(line, "database");
  leases.table_prefix = jc::FindString(line, "table_prefix");
  leases.lease_ttl_ns =
      static_cast<uint64_t>(jc::FindInt(line, "lease_ttl_ns"));
  leases.publish_timeout_ns =
      static_cast<uint64_t>(jc::FindInt(line, "publish_timeout_ns"));
  leases.clock_skew_ns =
      static_cast<uint64_t>(jc::FindInt(line, "clock_skew_ns"));
  dmi_catalog::AllocatorConfig alloc;
  alloc.database = leases.database;
  alloc.table_prefix = leases.table_prefix;
  alloc.allocation_attempts =
      static_cast<int>(jc::FindInt(line, "allocation_attempts"));
  alloc.publish_timeout_ns = leases.publish_timeout_ns;
  if (jc::HasKey(line, "insert_quorum") &&
      !jc::FindNull(line, "insert_quorum")) {
    const auto quorum =
        static_cast<uint64_t>(jc::FindInt(line, "insert_quorum"));
    leases.insert_quorum = quorum;
    alloc.insert_quorum = quorum;
  }
  // HTTP interface: the driver speaks HTTP (libcurl), not the native TCP
  // protocol clickhouse-driver uses, so the port differs from the
  // Python-side suites' DMI_CLICKHOUSE_PORT. 8123 is ClickHouse's default.
  const char* host = getenv("DMI_CLICKHOUSE_HOST");
  const char* port = getenv("DMI_CLICKHOUSE_HTTP_PORT");
  session.client = std::make_shared<const dmi_catalog::ClickHouseClient>(
      host != nullptr ? host : "127.0.0.1",
      static_cast<uint16_t>(port != nullptr ? std::atoi(port) : 8123));
  session.leases = std::make_unique<LeaseCoordinator>(session.client, leases);
  session.allocator = std::make_unique<dmi_catalog::VersionAllocator>(
      session.client, alloc);
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
    if (session->leases == nullptr) {
      return prefix + "false,\"what\":\"call open first\"}";
    }

    if (op == "head") {
      const auto head = session->leases->head();
      out = ",\"head\":{\"term\":" + std::to_string(head.term) +
            ",\"claimants\":" + std::to_string(head.claimants) +
            ",\"lease_id\":\"" + head.lease_id + "\",\"holder\":";
      escape_into(head.holder, &out);
      out += ",\"expires_at_ns\":" + std::to_string(head.expires_at_ns) +
             ",\"live_until_ns\":" + std::to_string(head.live_until_ns) +
             ",\"now_ns\":" + std::to_string(head.now_ns) + "}";
    } else if (op == "acquire") {
      emit_lease(session->leases->acquire(jc::FindString(line, "holder")),
                 &out);
    } else if (op == "renew") {
      emit_lease(session->leases->renew(), &out);
    } else if (op == "release") {
      session->leases->release();
    } else if (op == "lease") {
      const PublisherLease* held = session->leases->lease();
      if (held != nullptr) emit_lease(*held, &out);
      else out = ",\"lease\":null";
    } else if (op == "discard_local_lease") {
      session->leases->discard_local_lease();
    } else if (op == "claim") {
      emit_lease(session->leases->claim(jc::FindString(line, "holder"),
                                        jc::FindString(line, "lease_id")),
                 &out);
    } else if (op == "claim_contested") {
      emit_lease(session->leases->claim_contested(
                     jc::FindString(line, "holder"),
                     jc::FindString(line, "rival_lease_id")),
                 &out);
    } else if (op == "reject_if_gone") {
      session->leases->reject_if_gone();
    } else if (op == "statements") {
      // The audited statements, placeholders intact — a ported test
      // compares these byte-for-byte against the Python module's, so any
      // drift between the two implementations fails a gate instead of a
      // protocol assumption.
      out = ",\"release\":";
      escape_into(session->leases->release_statement(), &out);
      out += ",\"fence\":";
      escape_into(session->leases->fence(), &out);
    } else if (op == "fence_eval") {
      const bool admits = session->leases->fence_eval(
          jc::FindString(line, "lease_id"),
          static_cast<uint64_t>(jc::FindInt(line, "publish_timeout_ns")),
          static_cast<uint64_t>(jc::FindInt(line, "clock_skew_ns")));
      out = std::string(",\"admits\":") + (admits ? "1" : "0");
    } else if (op == "allocate_version") {
      out = ",\"version\":" +
            std::to_string(session->allocator->allocate_version());
    } else if (op == "max_version") {
      out = ",\"version\":" + std::to_string(session->allocator->max_version(
                jc::FindString(line, "table"),
                jc::FindString(line, "column")));
    } else if (op == "execute") {
      const auto rows = session->client->execute(
          jc::FindString(line, "query"));
      out = ",\"rows\":[";
      bool first_row = true;
      for (const auto& row : rows) {
        if (!first_row) out += ",";
        first_row = false;
        out += "[";
        bool first_field = true;
        for (const auto& field : row) {
          if (!first_field) out += ",";
          first_field = false;
          out += "\"";
          escape_into(field, &out);
          out += "\"";
        }
        out += "]";
      }
      out += "]";
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
