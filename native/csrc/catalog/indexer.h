// B3: the catalog indexer loop, ported from catalog.py's CatalogIndexer —
// dedupe with identity-conflict failures, the committed replay guard, the
// per-pack read with per-pack failures, the batch byte guard, the
// refuse-before-writing lease check, the monotonic-allocator cross-check,
// chunked descriptor writes, the publish retry with descriptor rewrites at
// the winning version, the conflict's commit-before-propagate, and the
// inventory commit last. The derivations live in the Python module; this
// port keeps the order and the failure taxonomy.

#ifndef DMI_CATALOG_INDEXER_H
#define DMI_CATALOG_INDEXER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "catalog_writer.h"
#include "pack_index.h"

namespace dmi_catalog {

struct IndexerConfig {
  int max_packs = 64;
  int max_rows_per_insert = 10'000;
  uint64_t max_estimated_bytes = 128ull * 1024 * 1024;
  int max_publish_attempts = 8;
};

struct IndexFailureData {
  std::string pack_id;
  std::string object_key;
  std::string error_type;
  std::string message;
};

struct IndexResultData {
  uint64_t requested_packs = 0;
  uint64_t skipped_packs = 0;
  uint64_t indexed_packs = 0;
  uint64_t indexed_rows = 0;
  uint64_t failed_packs = 0;
  uint64_t descriptor_inserts = 0;
  uint64_t estimated_bytes = 0;
  std::vector<IndexFailureData> failures;
};

class NativeIndexer {
 public:
  NativeIndexer(dmi_store::S3Client* s3, std::string bucket,
                CatalogWriter* writer, IndexerConfig config);

  IndexResultData index(const std::vector<PackRefData>& refs);

 private:
  uint64_t allocate_version();

  dmi_store::S3Client* s3_;
  std::string bucket_;
  CatalogWriter* writer_;
  IndexerConfig config_;
  std::optional<uint64_t> published_version_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_INDEXER_H
