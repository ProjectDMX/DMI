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
#include <functional>
#include <memory>
#include <optional>
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
  // Test seam, unset in production, called with each allocated version
  // just before the publish that carries it. The publish wedges in
  // CatalogWriter exist for the same reason: a version race needs the
  // published head to move between an allocation and its publish, which
  // only something outside this call can do. Left empty, this is nothing.
  std::function<void(uint64_t)> after_allocate;
  // Called before each catalog write commit() makes -- the version claim,
  // each descriptor chunk, the inventory commit -- so that a long pass can
  // renew its lease when that falls due, instead of running the lease down
  // while its caller holds the lease lock. The publish renews on its own.
  // What it throws propagates. Unset: nothing.
  std::function<void()> keep_lease;
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

// One index() pass, split at its object-store reads so that a caller can
// hold the catalog's lease lock for the catalog phases only: plan() and
// commit() talk to the catalog, read() only to the object store. A read that
// stalls then cannot keep the lease from renewing.
struct IndexPlan {
  IndexResultData result;            // requested, skipped and failures so far
  std::vector<PackRefData> pending;  // not yet committed, in read order
  // The lease the replay guard was read under ("" for none). commit() reads
  // the guard again if the lease has changed since: a pass that lost its
  // lease while reading may find the packs committed by another publisher.
  std::string planned_under;
  bool read = false;
  // Filled by read(): the packs read, each with its rendered rows.
  std::vector<PackRefData> indexed;
  std::vector<std::vector<std::string>> rows;
  uint64_t estimated_bytes = 0;
};

class NativeIndexer {
 public:
  NativeIndexer(dmi_store::S3Client* s3, CatalogWriter* writer,
                IndexerConfig config);

  // plan(), read() and commit() in one call.
  IndexResultData index(const std::vector<PackRefData>& refs);

  // Deduplicates refs and reads the replay guard (the catalog).
  IndexPlan plan(const std::vector<PackRefData>& refs);
  // Reads each pending pack's descriptor rows (the object store only).
  // Throws kBatchTooLarge past max_estimated_bytes.
  void read(IndexPlan* plan);
  // Allocates a version, writes the descriptors, publishes and commits the
  // inventory (the catalog). Requires read().
  IndexResultData commit(IndexPlan* plan);

 private:
  uint64_t allocate_version();
  void keep_lease() const;

  dmi_store::S3Client* s3_;
  CatalogWriter* writer_;
  IndexerConfig config_;
  std::optional<uint64_t> published_version_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_INDEXER_H
