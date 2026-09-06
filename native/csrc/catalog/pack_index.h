// B3: the pack-index read path, ported from pack.py's PackIndex.from_store
// — trailer, footer, per-record validation — rendering the descriptor
// VALUES rows the catalog write path consumes. The pack footer is EXTERNAL
// data and is validated at the boundary exactly where the Python module
// validates it: identity against the ref, ranges against the footer,
// lengths against dtype and shape, and the v1 uncompressed-only contract.

#ifndef DMI_CATALOG_PACK_INDEX_H
#define DMI_CATALOG_PACK_INDEX_H

#include <cstdint>
#include <string>
#include <vector>

#include "../store/s3_client.h"

namespace dmi_catalog {

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
// on any format violation, mirroring PackFormatError / PackIntegrityError.
std::vector<std::string> read_pack_descriptor_rows(
    dmi_store::S3Client* s3, const std::string& bucket,
    const PackRefData& ref);

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_PACK_INDEX_H
