#include "catalog_writer.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace dmi_catalog {

namespace {

constexpr size_t kMaxInlineParameterBytes = 192 * 1024;

// clickhouse_sql.inline_text_bytes: the parameter renders as a quoted
// string with room for escaping.
size_t inline_text_bytes(const std::string& value) {
  return 2 * value.size() + 2;
}

std::string escape_sql_string(const std::string& value) {
  std::string out;
  out.push_back('\'');
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

// Python's clickhouse-driver renders a list of (str, str) tuples as
// [('a','b'), ...]; the manifest INSERT arrayJoins exactly that.
std::string render_members(const std::vector<PackIdentity>& members) {
  std::string out = "[";
  bool first = true;
  for (const auto& [store_id, pack_id] : members) {
    if (!first) out += ",";
    first = false;
    out += "(" + escape_sql_string(store_id) + "," +
           escape_sql_string(pack_id) + ")";
  }
  out += "]";
  return out;
}

// clickhouse_sql.inline_chunks over (store_id, pack_id) identities:
// bounded so the statement text cannot breach the server's max_query_size.
std::vector<std::vector<PackIdentity>> inline_chunks(
    const std::vector<PackIdentity>& items) {
  std::vector<std::vector<PackIdentity>> chunks;
  std::vector<PackIdentity> chunk;
  size_t size = 2;
  for (const auto& item : items) {
    const size_t encoded =
        inline_text_bytes(item.first) + inline_text_bytes(item.second) + 4;
    if (encoded + 2 > kMaxInlineParameterBytes) {
      throw CatalogError(CatalogError::Kind::kValue,
                         "item exceeds inline query byte budget");
    }
    const size_t separator = chunk.empty() ? 0 : 2;
    if (!chunk.empty() && size + separator + encoded > kMaxInlineParameterBytes) {
      chunks.push_back(chunk);
      chunk.clear();
      size = 2;
    }
    chunk.push_back(item);
    size += (chunk.size() == 1 ? 0 : 2) + encoded;
  }
  if (!chunk.empty()) chunks.push_back(chunk);
  return chunks;
}

uint64_t now_monotonic_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

std::string sql_quote(const std::string& value) {
  return escape_sql_string(value);
}

CatalogWriter::CatalogWriter(std::shared_ptr<const ClickHouseClient> client,
                             WriterConfig config)
    : client_(std::move(client)), config_(std::move(config)) {
  LeaseConfig leases;
  leases.database = config_.database;
  leases.table_prefix = config_.table_prefix;
  leases.lease_ttl_ns = config_.lease_ttl_ns;
  leases.publish_timeout_ns = config_.publish_timeout_ns;
  leases.clock_skew_ns = config_.clock_skew_ns;
  leases.insert_quorum = config_.insert_quorum;
  leases_ = std::make_unique<LeaseCoordinator>(client_, leases);
  AllocatorConfig alloc;
  alloc.database = config_.database;
  alloc.table_prefix = config_.table_prefix;
  alloc.allocation_attempts = config_.allocation_attempts;
  alloc.publish_timeout_ns = config_.publish_timeout_ns;
  alloc.insert_quorum = config_.insert_quorum;
  allocator_ = std::make_unique<VersionAllocator>(client_, alloc);
}

std::string CatalogWriter::qualified(const char* table) const {
  return "`" + config_.database + "`.`" + config_.table_prefix + "_" + table +
         "`";
}

std::map<std::string, std::string> CatalogWriter::quorum_write() const {
  if (!config_.insert_quorum.has_value()) return {};
  return {{"insert_quorum", std::to_string(*config_.insert_quorum)},
          {"insert_quorum_parallel", "0"},
          {"insert_quorum_timeout",
           std::to_string(config_.publish_timeout_ns / 1'000'000)}};
}

void CatalogWriter::require_not_quarantined() const {
  if (quarantined_ && now_monotonic_ns() < quarantine_until_ns_) {
    throw CatalogError(
        CatalogError::Kind::kQuarantined,
        "this writer is quarantined: a previous publish failed with an "
        "unknown outcome, so its statements may still be running on the "
        "server past their fence evaluation. The lease was discarded "
        "without a release tombstone, so no successor can publish a "
        "higher watermark inside this window. Wait out the lease TTL, "
        "then acquire a fresh lease and re-index.");
  }
}

void CatalogWriter::quarantine() {
  // Discard WITHOUT the release tombstone: the server row stays live until
  // its TTL, which is what keeps a successor out of the window.
  leases_->discard_local_lease();
  quarantined_ = true;
  quarantine_until_ns_ = now_monotonic_ns() + config_.lease_ttl_ns;
}

bool CatalogWriter::quarantined(uint64_t* until_ns) const {
  if (until_ns != nullptr) *until_ns = quarantine_until_ns_;
  return quarantined_;
}

PublisherLease CatalogWriter::renew_for_publish() {
  require_not_quarantined();
  try {
    return leases_->renew();
  } catch (const CatalogError&) {
    throw;  // a known lease outcome releases the writer normally
  } catch (const std::exception&) {
    quarantine();
    throw;
  }
}

PublisherLease CatalogWriter::acquire_lease(const std::string& holder) {
  require_not_quarantined();
  try {
    return leases_->acquire(holder);
  } catch (const CatalogError&) {
    throw;
  } catch (const std::exception&) {
    quarantine();
    throw;
  }
}

PublisherLease CatalogWriter::renew_lease() { return renew_for_publish(); }

LeaseConfig CatalogWriter::leases_config_for_takeover() const {
  // A successor runs on the DEFAULT knobs (the Python live suites'
  // construction); only the stalled publisher's own lease has to lapse.
  LeaseConfig config;
  config.database = config_.database;
  config.table_prefix = config_.table_prefix;
  config.lease_ttl_ns = 30'000'000'000ull;
  config.publish_timeout_ns = 5'000'000'000ull;
  config.clock_skew_ns = config_.clock_skew_ns;
  config.insert_quorum = config_.insert_quorum;
  return config;
}

void CatalogWriter::write_descriptors(
    const std::vector<std::string>& rendered_rows, uint64_t index_version) {
  if (rendered_rows.empty()) return;
  require_not_quarantined();
  // The batch's index_version is the final column; the rows carry the
  // other 32 and are version-independent until written.
  std::string values;
  for (const auto& row : rendered_rows) {
    if (!values.empty()) values += ",";
    values += "(" + row + "," + std::to_string(index_version) + ")";
  }
  try {
    client_->execute(
        "INSERT INTO " + qualified("capture_raw") +
            " (capture_id, tenant_id, experiment_id, run_id, session_id, "
            "request_id, sequence_id, model_id, model_revision, "
            "adapter_revision, capture_policy_version, hook_name, "
            "layer_number, producer_rank, step_number, token_start, "
            "token_end, batch_position, dtype, shape, captured_at_ns, "
            "pack_id, store_id, object_key, object_bytes, pack_checksum, "
            "pack_record_count, payload_offset, stored_length, "
            "decoded_length, codec, payload_checksum, index_version) "
            "VALUES " +
            values,
        {}, quorum_write());
  } catch (const CatalogError&) {
    throw;
  } catch (const std::exception&) {
    quarantine();
    throw;
  }
}

std::set<PackIdentity> CatalogWriter::committed_pack_ids(
    const std::vector<PackIdentity>& identities) const {
  if (identities.empty()) return {};
  if (static_cast<int>(identities.size()) > config_.query_pack_limit) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "pack identity query exceeds query_pack_limit");
  }
  std::set<PackIdentity> committed;
  // Chunked: the identities land in the statement TEXT, and an unchunked
  // query_pack_limit's worth of tuples can breach max_query_size.
  for (const auto& chunk : inline_chunks(identities)) {
    const std::vector<Row> rows = client_->execute(
        "SELECT store_id, toString(pack_id) FROM " +
            qualified("pack_inventory") +
            " WHERE (store_id, pack_id) IN (" + render_members(chunk) + ")",
        {},
        // A deciding read: this answer decides which packs the indexer
        // skips as already committed.
        deciding_read());
    for (const Row& row : rows) {
      committed.emplace(row[0], row[1]);
    }
  }
  return committed;
}

void CatalogWriter::commit_packs(
    const std::vector<std::string>& rendered_rows, uint64_t index_version) {
  if (rendered_rows.empty()) return;
  require_not_quarantined();
  std::string values;
  for (const auto& row : rendered_rows) {
    if (!values.empty()) values += ",";
    values += "(" + row + ")";
  }
  try {
    client_->execute(
        "INSERT INTO " + qualified("pack_inventory_raw") +
            " (pack_id, store_id, object_key, object_bytes, pack_checksum, "
            "record_count, index_version) VALUES " +
            values,
        {}, quorum_write());
  } catch (const CatalogError&) {
    throw;
  } catch (const std::exception&) {
    quarantine();
    throw;
  }
}

int CatalogWriter::manifest_member_count(
    uint64_t index_version, const std::string& publish_id,
    const std::vector<PackIdentity>* members) const {
  std::string bound;
  if (members != nullptr) {
    bound = " AND (store_id, pack_id) IN (" + render_members(*members) + ")";
  }
  const std::vector<Row> rows = client_->execute(
      "SELECT count() FROM (SELECT DISTINCT store_id, pack_id FROM " +
          qualified("snapshot_manifest") +
          " WHERE index_version = %(index_version)s "
          "AND publish_id = toUUID(%(publish_id)s)" + bound + ")",
      {{"index_version", index_version}, {"publish_id", publish_id}},
      deciding_read());
  return rows.empty() ? 0 : static_cast<int>(parse_u64_field(rows[0][0], "count"));
}

uint64_t CatalogWriter::last_published_version() const {
  return allocator_->max_version("index_watermark", "index_version");
}

uint64_t CatalogWriter::allocate_version() {
  require_not_quarantined();
  return allocator_->allocate_version();
}

uint64_t CatalogWriter::max_version(const std::string& table,
                                    const std::string& column) const {
  return allocator_->max_version(table, column);
}

void CatalogWriter::publish_snapshot(
    uint64_t index_version, const std::vector<PackIdentity>& refs,
    uint64_t published_at_ns, uint64_t indexed_rows, uint64_t indexed_packs,
    uint64_t wedge_ns, const std::string* takeover_after_renew,
    const std::string* takeover_after_chunks, bool inject_transport_error) {
  require_not_quarantined();
  try {
    // The lease is renewed before every fenced statement, and each fence
    // requires publish_timeout_ns of lease life remaining, which the
    // statement carries as its max_execution_time.
    PublisherLease lease = renew_for_publish();
    if (takeover_after_renew != nullptr) {
      // Wedged between the renewal and the fenced statements.
      if (wedge_ns > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(wedge_ns));
      }
      LeaseCoordinator successor(client_, leases_config_for_takeover());
      successor.acquire(*takeover_after_renew);
    }
    if (inject_transport_error) {
      // Outcome-unknown: the Python live suite reproduces this through a
      // client wrapper that dies after the renewal; the statements may
      // still be running server-side past their fence evaluation.
      throw ClickHouseError("simulated connection reset during publish");
    }

    const std::map<std::string, std::string> settings = [&] {
      std::map<std::string, std::string> out = deciding_read();
      out["max_execution_time"] =
          std::to_string(config_.publish_timeout_ns / 1'000'000'000);
      out["timeout_overflow_mode"] = "throw";
      for (const auto& [k, v] : quorum_write()) out[k] = v;
      return out;
    }();

    // Membership rows first, then the watermark row that admits them; both
    // carry one publish_id minted here. Both are fenced inside the
    // server-side statement, so a publisher whose lease was taken over
    // makes NO snapshot visible.
    const std::string publish_id = new_uuid_v4();
    if (!refs.empty()) {
      for (const auto& chunk : inline_chunks(refs)) {
        client_->execute(
            "INSERT INTO " + qualified("snapshot_manifest") +
                " (index_version, publish_id, store_id, pack_id) "
                "SELECT %(index_version)s, toUUID(%(publish_id)s), "
                "tupleElement(member, 1), toUUID(tupleElement(member, 2)) "
                "FROM (SELECT arrayJoin(" +
                render_members(chunk) +
                ") AS member) "
                "WHERE " +
                leases_->fence(),
            {{"index_version", index_version},
             {"publish_id", publish_id},
             {"lease_id", lease.lease_id},
             {"publish_timeout_ns", config_.publish_timeout_ns},
             {"clock_skew_ns", config_.clock_skew_ns}},
            settings);
        // A zero-row conditional INSERT cannot be followed by a visible
        // watermark: every chunk is read back before the renewal.
        //
        // Against the chunk's DISTINCT identities, not its length: the
        // read-back counts distinct manifest members, so a caller that
        // handed the same identity over twice would otherwise look like a
        // half-written chunk and abort a publish that wrote everything it
        // was asked to. `len(set(members))` in the Python oracle.
        const std::set<PackIdentity> distinct_chunk(chunk.begin(),
                                                    chunk.end());
        if (manifest_member_count(index_version, publish_id, &chunk) !=
            static_cast<int>(distinct_chunk.size())) {
          leases_->reject_if_gone();
          throw CatalogError(
              CatalogError::Kind::kPublishRace,
              "catalog version " + std::to_string(index_version) +
                  " did not publish its complete manifest chunk, so no "
                  "watermark was written and no snapshot became visible. "
                  "Allocate a higher version and publish again.");
        }
        if (takeover_after_chunks != nullptr) {
          // Wedged after the manifest INSERT, before the renewal — the
          // window the Python live suite drives through its client
          // wrapper. The wedge lets the stalled publisher's lease lapse,
          // then the successor claims; the renewal that follows meets it.
          if (wedge_ns > 0) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(wedge_ns));
          }
          LeaseCoordinator successor(client_, leases_config_for_takeover());
          successor.acquire(*takeover_after_chunks);
        }
        lease = renew_for_publish();
      }
    }

    // The barrier, the fence and the visibility write are ONE server-side
    // statement. ifNull over the empty watermark table keeps the FIRST
    // publish from being refused as a lost race on NULL-comparison profiles.
    client_->execute(
        "INSERT INTO " + qualified("index_watermark") +
            " (index_version, publish_id, published_at_ns, indexed_rows, "
            "indexed_packs) "
            "SELECT %(index_version)s, toUUID(%(publish_id)s), "
            "%(published_at_ns)s, "
            "toUInt64(%(indexed_rows)s), toUInt32(%(indexed_packs)s) "
            "FROM system.one "
            "WHERE ifNull((SELECT max(index_version) FROM " +
            qualified("index_watermark") + "), 0) < %(index_version)s "
            "AND " +
            leases_->fence(),
        {{"index_version", index_version},
         {"publish_id", publish_id},
         {"published_at_ns", published_at_ns},
         {"indexed_rows", indexed_rows},
         {"indexed_packs", indexed_packs},
         {"lease_id", lease.lease_id},
         {"publish_timeout_ns", config_.publish_timeout_ns},
         {"clock_skew_ns", config_.clock_skew_ns}},
        settings);

    // Ownership, not occupancy: "is MY row there" and "is it the ONLY row
    // there" need opposite recoveries.
    std::set<std::string> owners;
    for (const Row& row : client_->execute(
             "SELECT toString(publish_id) FROM " +
                 qualified("index_watermark") +
                 " WHERE index_version = %(version)s",
             {{"version", index_version}}, deciding_read())) {
      owners.insert(row[0]);
    }
    if (!owners.count(publish_id)) {
      leases_->reject_if_gone();
      throw CatalogError(
          CatalogError::Kind::kPublishRace,
          "catalog version " + std::to_string(index_version) +
              " lost the publish race: the conditional watermark INSERT "
              "was refused, so no row carrying publish " + publish_id +
              " stands at that version and no snapshot can admit anything "
              "this attempt wrote. Allocate a higher version and publish "
              "again.");
    }
    if (owners.size() > 1) {
      std::string foreign;
      for (const auto& owner : owners) {
        if (owner == publish_id) continue;
        if (!foreign.empty()) foreign += ", ";
        foreign += owner;
      }
      throw CatalogError(
          CatalogError::Kind::kPublishConflict,
          "catalog version " + std::to_string(index_version) +
              " was published by this writer (publish " + publish_id +
              ") AND by " + foreign + ". This publish is visible and must "
              "not be retried; the version's membership is now the union "
              "of both publishes. Something else is writing `" +
              config_.database + "`.`" + config_.table_prefix +
              "_*` -- a second indexer sharing the prefix, or a "
              "hand-written INSERT.");
    }
    if (!refs.empty()) {
      std::set<PackIdentity> distinct(refs.begin(), refs.end());
      if (manifest_member_count(index_version, publish_id, nullptr) !=
          static_cast<int>(distinct.size())) {
        throw CatalogError(
            CatalogError::Kind::kPublishRace,
            "catalog version " + std::to_string(index_version) +
                " published its watermark but its membership (publish " +
                publish_id + ") had been collected before the watermark row "
                "landed. The watermark row stands and admits nothing, so no "
                "snapshot contains these packs. Do not commit them; allocate "
                "a higher version and publish again.");
      }
    }
  } catch (const CatalogError&) {
    throw;  // known outcomes release the writer normally
  } catch (const std::exception&) {
    // Outcome-unknown: the statements may still be running past their
    // fence evaluation. Quarantine, without the release tombstone.
    quarantine();
    throw;
  }
}

}  // namespace dmi_catalog
