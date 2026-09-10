#include "version_allocator.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <set>

#include "lease_coordinator.h"

namespace dmi_catalog {

namespace {

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

}  // namespace

VersionAllocator::VersionAllocator(
    std::shared_ptr<const ClickHouseClient> client, AllocatorConfig config)
    : client_(std::move(client)), config_(std::move(config)) {}

std::string VersionAllocator::qualified(const std::string& table) const {
  return "`" + config_.database + "`.`" + config_.table_prefix + "_" +
         table + "`";
}

std::map<std::string, std::string> VersionAllocator::quorum_write() const {
  if (!config_.insert_quorum.has_value()) return {};
  return {{"insert_quorum", std::to_string(*config_.insert_quorum)},
          {"insert_quorum_parallel", "0"},
          {"insert_quorum_timeout",
           std::to_string(config_.publish_timeout_ns / 1'000'000)}};
}

uint64_t VersionAllocator::max_version(const std::string& table,
                                       const std::string& column) const {
  const std::vector<Row> rows = client_->execute(
      "SELECT max(" + column + ") FROM " + qualified(table), {},
      deciding_read());
  if (rows.empty() || rows[0].empty() || rows[0][0].empty()) return 0;
  return parse_u64_field(rows[0][0], "version");
}

uint64_t VersionAllocator::allocate_version() {
  std::mt19937_64 rng(std::random_device{}());
  for (int attempt = 0; attempt < config_.allocation_attempts; ++attempt) {
    const uint64_t claimed = max_version("capture_version_claims", "version");
    const uint64_t floor = std::max(
        claimed, max_version("index_watermark", "index_version"));
    // A randomized skip only after a collision, so two contenders that
    // keep colliding spread out instead of racing for floor + 1 again.
    const uint64_t spread = attempt == 0
                                ? 0
                                : static_cast<uint64_t>(rng()) %
                                      static_cast<uint64_t>(8 * attempt + 1);
    const uint64_t candidate = floor + 1 + spread;
    const std::string claim_id = new_uuid_v4();
    client_->execute(
        "INSERT INTO " + qualified("capture_version_claims") +
            " (version, claim_id, claimed_at_ns) VALUES "
            "(%(version)s, toUUID(%(claim_id)s), %(claimed_at_ns)s)",
        {{"version", candidate},
         {"claim_id", claim_id},
         {"claimed_at_ns", now_ns()}},
        // A deciding write: the read-back below decides ownership.
        quorum_write());
    const std::vector<Row> owners = client_->execute(
        "SELECT toString(claim_id) FROM " +
            qualified("capture_version_claims") + " "
            "WHERE version = %(version)s",
        {{"version", candidate}}, deciding_read());
    std::set<std::string> ids;
    for (const Row& row : owners) ids.insert(row[0]);
    if (ids == std::set<std::string>{claim_id}) {
      return candidate;
    }
    // Contested: someone else claimed the same version -- abandon it
    // entirely and retry above it.
  }
  throw CatalogError(
      CatalogError::Kind::kAllocation,
      "could not allocate a catalog version after " +
          std::to_string(config_.allocation_attempts) + " attempts");
}

}  // namespace dmi_catalog
