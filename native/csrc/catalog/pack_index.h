// B3: the pack-index read path, ported from pack.py's PackIndex.from_store
// — trailer, footer, per-record validation — rendering the descriptor
// VALUES rows the catalog write path consumes. The pack footer is EXTERNAL
// data and is validated at the boundary exactly where the Python module
// validates it: identity against the ref, ranges against the footer,
// lengths against dtype and shape, and the v1 uncompressed-only contract.

#ifndef DMI_CATALOG_PACK_INDEX_H
#define DMI_CATALOG_PACK_INDEX_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "../store/s3_client.h"
#include "lease_coordinator.h"

namespace dmi_catalog {

// read_pack_descriptor_rows could not get the object store's answer about
// the pack: a transport error or timeout, a retryable status on every
// attempt, or the S3 client's Cancellation cutting the read (cancelled()
// says which; a cancel that came in once the store had failed it does not
// count, as in S3Client). It says
// nothing about the pack itself, unlike every other refusal the read makes.
// A CatalogError of kind kValue like those, so a caller that treats every
// unreadable pack alike still does.
class StoreUnavailableError : public CatalogError {
 public:
  StoreUnavailableError(const std::string& what, bool cancelled)
      : CatalogError(Kind::kValue, what), cancelled_(cancelled) {}
  bool cancelled() const { return cancelled_; }

 private:
  bool cancelled_;
};

struct PackRefData {
  std::string pack_id;
  std::string store_id;
  std::string object_key;
  uint64_t object_bytes = 0;
  std::string checksum;
  uint64_t record_count = 0;
};

// Reads the pack the ref names and renders one descriptor VALUES row per
// record — the 33 capture_raw columns in schema order, without
// index_version (the batch's own version). Throws CatalogError (kValue)
// on any format violation, mirroring PackFormatError / PackIntegrityError,
// and StoreUnavailableError when the store did not answer a range read.
// The bucket is the S3Client's own config; a bucket parameter here would
// only invite a caller to believe passing a different one redirects the
// read.
//
// `charge`, when given, is called with the exact length of each range
// BEFORE that range is fetched -- the 64-byte trailer, then the footer at
// the length the trailer declares. It is the reader's read budget
// (Python's _ReadBudget.consume): it throws to refuse, and nothing has
// been fetched when it does.
std::vector<std::string> read_pack_descriptor_rows(
    dmi_store::S3Client* s3, const PackRefData& ref,
    const std::function<void(uint64_t)>& charge = nullptr);

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_PACK_INDEX_H
