#include "indexer.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <set>

namespace dmi_catalog {

namespace {

uint64_t NowWallClockNs() {
  // The wall clock stamps published_at_ns only -- a human-readable record
  // of when the version was published, never part of its ordering.
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

std::vector<PackIdentity> PackIdentitiesFrom(
    const std::vector<const PackRefData*>& refs) {
  std::vector<PackIdentity> out;
  for (const PackRefData* ref : refs) {
    out.emplace_back(ref->store_id, ref->pack_id);
  }
  return out;
}

std::vector<std::string> RenderPackRows(
    const std::vector<const PackRefData*>& refs, uint64_t index_version) {
  std::vector<std::string> rows;
  for (const PackRefData* ref : refs) {
    rows.push_back("toUUID('" + ref->pack_id + "')," + sql_quote(ref->store_id) +
                   "," + sql_quote(ref->object_key) + "," +
                   std::to_string(ref->object_bytes) + "," +
                   sql_quote(ref->checksum) + "," +
                   std::to_string(ref->record_count) + "," +
                   std::to_string(index_version));
  }
  return rows;
}

}  // namespace

NativeIndexer::NativeIndexer(dmi_store::S3Client* s3,
                             std::string bucket, CatalogWriter* writer,
                             IndexerConfig config)
    : s3_(s3), bucket_(std::move(bucket)), writer_(writer),
      config_(config) {}

uint64_t NativeIndexer::allocate_version() {
  const uint64_t version = writer_->allocate_version();
  if (version < 1) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "allocate_version must return a positive integer");
  }
  if (!published_version_.has_value()) {
    // Cross-check against durable state: a broken allocator must fail
    // loudly here rather than publish under a pinned watermark.
    published_version_ = writer_->last_published_version();
  }
  if (version <= *published_version_) {
    throw CatalogError(
        CatalogError::Kind::kAllocation,
        "catalog version allocator returned a non-monotonic version: " +
            std::to_string(version) + " is not above the published head " +
            std::to_string(*published_version_));
  }
  return version;
}

IndexResultData NativeIndexer::index(const std::vector<PackRefData>& refs) {
  IndexResultData result;
  if (static_cast<int>(refs.size()) > config_.max_packs) {
    throw CatalogError(
        CatalogError::Kind::kValue,
        "pack batch exceeds max_packs: " + std::to_string(refs.size()) +
            " > " + std::to_string(config_.max_packs));
  }

  // Deduplicate refs, containing identity conflicts as per-pack failures:
  // two different refs claiming one (store_id, pack_id) means at least one
  // is wrong, and raising here would abort every reconcile pass forever.
  std::map<PackIdentity, const PackRefData*> by_identity;
  std::set<PackIdentity> conflicted;
  for (const PackRefData& ref : refs) {
    const PackIdentity identity{ref.store_id, ref.pack_id};
    const auto current = by_identity.find(identity);
    if (current == by_identity.end()) {
      by_identity[identity] = &ref;
      continue;
    }
    if (current->second->object_key != ref.object_key ||
        current->second->object_bytes != ref.object_bytes ||
        current->second->checksum != ref.checksum ||
        current->second->record_count != ref.record_count) {
      conflicted.insert(identity);
    }
  }

  std::vector<const PackRefData*> unique;
  for (const auto& [identity, ref] : by_identity) {
    if (!conflicted.count(identity)) unique.push_back(ref);
  }
  for (const PackRefData& ref : refs) {
    if (conflicted.count({ref.store_id, ref.pack_id})) {
      result.failures.push_back({ref.pack_id, ref.object_key,
                                 "PackConflictError",
                                 "conflicting pack identity"});
    }
  }
  result.requested_packs = unique.size() + result.failures.size();

  // The replay guard, and nothing else: visibility comes from publish.
  std::set<PackIdentity> committed;
  if (!unique.empty()) {
    std::vector<PackIdentity> identities;
    for (const PackRefData* ref : unique) {
      identities.emplace_back(ref->store_id, ref->pack_id);
    }
    committed = writer_->committed_pack_ids(identities);
  }

  std::vector<const PackRefData*> pending;
  std::vector<std::vector<std::string>> row_batches;
  uint64_t estimated_bytes = 0;
  for (const PackRefData* ref : unique) {
    if (committed.count({ref->store_id, ref->pack_id})) continue;
    pending.push_back(ref);
  }
  result.skipped_packs = unique.size() - pending.size();

  std::vector<std::string> all_rows;
  for (const PackRefData* ref : pending) {
    try {
      const std::vector<std::string> rows =
          read_pack_descriptor_rows(s3_, bucket_, *ref);
      uint64_t pack_bytes = 0;
      for (const std::string& row : rows) pack_bytes += row.size();
      // The rendered-row length is this port's analog of the Python
      // harness's JSON-encoding estimate — a bounded upper-bound proxy.
      if (estimated_bytes + pack_bytes > config_.max_estimated_bytes) {
        // A too-large batch is a caller error, not a property of the pack
        // being read: blaming the pack would silently skip the rest.
        throw CatalogError(
            CatalogError::Kind::kValue,
            "catalog batch exceeds max_estimated_bytes: " +
                std::to_string(estimated_bytes + pack_bytes) + " > " +
                std::to_string(config_.max_estimated_bytes));
      }
      estimated_bytes += pack_bytes;
      all_rows.insert(all_rows.end(), rows.begin(), rows.end());
    } catch (const CatalogError& e) {
      std::string message = e.what();
      if (message.size() > 512) message.resize(512);
      result.failures.push_back(
          {ref->pack_id, ref->object_key, "CatalogError", message});
      pending.erase(std::remove(pending.begin(), pending.end(), ref),
                    pending.end());
    }
  }

  if (!all_rows.empty() || !pending.empty()) {
    // Fail before the batch is written, not after it is wasted: a writer
    // without publishing authority discovers it last otherwise.
    if (writer_->held_lease() == nullptr) {
      throw CatalogError(
          CatalogError::Kind::kLease,
          "the catalog writer holds no publisher lease, and only the "
          "lease holder can make a snapshot visible: acquire one before "
          "indexing. Refused before allocating a version and writing "
          "descriptors, neither of which this pass could have published.");
    }
    uint64_t version = allocate_version();
    uint64_t descriptor_inserts = 0;
    const auto write_batches = [&](uint64_t at_version) {
      for (size_t start = 0; start < all_rows.size();
           start += config_.max_rows_per_insert) {
        std::vector<std::string> chunk(
            all_rows.begin() + static_cast<long>(start),
            all_rows.begin() +
                static_cast<long>(
                    std::min(all_rows.size(),
                             start + config_.max_rows_per_insert)));
        writer_->write_descriptors(chunk, at_version);
        ++descriptor_inserts;
      }
    };
    write_batches(version);

    // Publish before the inventory, never after: committed_pack_ids reads
    // the inventory, so a pack recorded there but never made visible is
    // skipped forever AND invisible. Only a lost VERSION race is retried
    // here, repaired by allocating higher and rewriting the descriptors at
    // the winning version (supersession ranks by index_version).
    uint64_t attempts = 0;
    for (; attempts < static_cast<uint64_t>(config_.max_publish_attempts);
         ++attempts) {
      try {
        writer_->publish_snapshot(
            version, PackIdentitiesFrom(pending), NowWallClockNs(),
            all_rows.size(), pending.size());
      } catch (const CatalogError& e) {
        if (e.kind() != CatalogError::Kind::kPublishRace) {
          if (e.kind() == CatalogError::Kind::kPublishConflict) {
            // Visible, so skippable: record the packs before propagating.
            if (!pending.empty()) {
              writer_->commit_packs(RenderPackRows(pending, version), version);
            }
          }
          throw;
        }
        if (attempts + 1 ==
            static_cast<uint64_t>(config_.max_publish_attempts)) {
          break;
        }
        // Drop the cached head before re-allocating: the race means a
        // competitor published, so the head read on the first allocation
        // is stale at exactly the moment the cross-check matters.
        published_version_.reset();
        version = allocate_version();
        write_batches(version);
        continue;
      }
      published_version_ = version;
      break;
    }
    if (attempts == static_cast<uint64_t>(config_.max_publish_attempts)) {
      throw CatalogError(
          CatalogError::Kind::kPublishRace,
          "could not publish a catalog snapshot after " +
              std::to_string(config_.max_publish_attempts) + " attempts");
    }
    if (!pending.empty()) {
      writer_->commit_packs(RenderPackRows(pending, version), version);
    }
    result.descriptor_inserts = descriptor_inserts;
  }

  result.indexed_packs = pending.size();
  result.indexed_rows = all_rows.size();
  result.failed_packs = result.failures.size();
  result.estimated_bytes = estimated_bytes;
  return result;
}

}  // namespace dmi_catalog
