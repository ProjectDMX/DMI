// B2: the catalog writer's cold-write protocol, ported from
// src/dmi/storage/capture/clickhouse_catalog.py — descriptor batches,
// the pack replay guard, and the fenced publish. The publish's deep
// derivations (fence inside the server-side statement, barrier + fence +
// visibility as one statement, ownership-not-occupancy read-back, the
// outcome-unknown quarantine) are documented in the Python module and in
// docs/catalog-descriptor-key.md; this port keeps the statements and the
// control flow and does not re-derive them.

#ifndef DMI_CATALOG_CATALOG_WRITER_H
#define DMI_CATALOG_CATALOG_WRITER_H

#include <cstdint>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lease_coordinator.h"
#include "version_allocator.h"

namespace dmi_catalog {

struct WriterConfig {
  std::string database;
  std::string table_prefix;
  uint64_t lease_ttl_ns = 30'000'000'000ull;
  uint64_t publish_timeout_ns = 5'000'000'000ull;
  uint64_t clock_skew_ns = 0;
  std::optional<uint64_t> insert_quorum;
  int query_pack_limit = 10'000;
  int allocation_attempts = 16;
};

using PackIdentity = std::pair<std::string, std::string>;  // store_id, pack_id

// SQL string literal, ClickHouse escaping — shared by the row renderers.
std::string sql_quote(const std::string& value);

class CatalogWriter {
 public:
  CatalogWriter(std::shared_ptr<const ClickHouseClient> client,
                WriterConfig config);

  // One process, one writer: the Python writer serialises publishes and
  // claims behind a lock because threads share it; the driver executes
  // ops one at a time, which is the same serialisation.

  void write_descriptors(const std::vector<std::string>& rendered_rows,
                         uint64_t index_version);
  std::set<PackIdentity> committed_pack_ids(
      const std::vector<PackIdentity>& identities) const;
  void commit_packs(const std::vector<std::string>& rendered_rows,
                    uint64_t index_version);
  PublisherLease acquire_lease(const std::string& holder);
  PublisherLease renew_lease();
  void release_lease() { leases_->release(); }
  const PublisherLease* held_lease() const { return leases_->lease(); }
  uint64_t allocate_version();
  uint64_t max_version(const std::string& table,
                       const std::string& column) const;
  // The wedges exist to reproduce, from the outside, what the Python
  // live suite reproduces through client wrappers: a takeover landing
  // between the renewal and the fenced statements, and between the
  // manifest chunks and the watermark. wedge_ns delays inside publish;
  // takeover_* performs a successor's claim at that point, in-process.
  void publish_snapshot(uint64_t index_version,
                        const std::vector<PackIdentity>& refs,
                        uint64_t published_at_ns, uint64_t indexed_rows,
                        uint64_t indexed_packs, uint64_t wedge_ns = 0,
                        const std::string* takeover_after_renew = nullptr,
                        const std::string* takeover_after_chunks = nullptr,
                        bool inject_transport_error = false);
  uint64_t last_published_version() const;
  // B4: delete the rows the protocols append and never need again —
  // manifest rows of publishes that never reached the watermark (settled
  // across two reads a publish timeout apart), lease rows below the head
  // term (the head itself is kept), and version claims at or below the
  // published head. Explicit, never called from the write path. Returns
  // the rows removed per table.
  std::map<std::string, uint64_t> collect_garbage(uint64_t settle_sleep_ns);

 private:
  uint64_t delete_rows(const char* table, const std::string& predicate,
                       const Params& params,
                       const std::map<std::string, std::string>& settings);
  std::vector<std::pair<uint64_t, std::string>> orphaned_manifest_publishes(
      uint64_t published) const;

 public:
  bool quarantined(uint64_t* until_ns = nullptr) const;
  LeaseCoordinator& leases() { return *leases_; }

 private:
  void require_not_quarantined() const;
  void quarantine();
  PublisherLease renew_for_publish();
  LeaseConfig leases_config_for_takeover() const;
  int manifest_member_count(uint64_t index_version,
                            const std::string& publish_id,
                            const std::vector<PackIdentity>* members) const;
  std::map<std::string, std::string> quorum_write() const;
  std::string qualified(const char* table) const;

  std::shared_ptr<const ClickHouseClient> client_;
  WriterConfig config_;
  std::unique_ptr<LeaseCoordinator> leases_;
  std::unique_ptr<VersionAllocator> allocator_;
  // Outcome-unknown quarantine: the lease is discarded WITHOUT the release
  // tombstone and every publish-path entry is refused until this instant.
  bool quarantined_ = false;
  uint64_t quarantine_until_ns_ = 0;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_CATALOG_WRITER_H
// (B4 additions appended)
