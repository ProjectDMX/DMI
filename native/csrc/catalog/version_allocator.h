// B1b: the DB-owned version allocator, ported from
// src/dmi/storage/capture/clickhouse_catalog.py (allocate_version and
// _max_version). Sole-claimant protocol over the append-only claims
// table: pick a candidate above everything claimed or published, insert
// a claim, read the claims for that version back, and proceed only as
// the sole claimant. Contested versions are abandoned by everyone who
// sees the tie. Every returned version is durably in the claims table.

#ifndef DMI_CATALOG_VERSION_ALLOCATOR_H
#define DMI_CATALOG_VERSION_ALLOCATOR_H

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "clickhouse_client.h"

namespace dmi_catalog {

struct AllocatorConfig {
  std::string database;
  std::string table_prefix;
  int allocation_attempts = 16;
  uint64_t publish_timeout_ns = 5'000'000'000ull;
  std::optional<uint64_t> insert_quorum;
};

class VersionAllocator {
 public:
  VersionAllocator(std::shared_ptr<const ClickHouseClient> client,
                   AllocatorConfig config);

  uint64_t allocate_version();
  // Highest `column` in `table`, or 0 when the table is empty. A deciding
  // read: the answer is the floor a claim is picked above.
  uint64_t max_version(const std::string& table,
                       const std::string& column) const;

 private:
  std::string qualified(const std::string& table) const;
  std::map<std::string, std::string> quorum_write() const;

  std::shared_ptr<const ClickHouseClient> client_;
  AllocatorConfig config_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_VERSION_ALLOCATOR_H
