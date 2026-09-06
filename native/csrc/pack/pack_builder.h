// dmi-pack-v1: byte-conformant native writer for the DMI capture pack format.
//
// The contract is the Python reference in src/dmi/storage/capture/pack.py. A
// native writer is conformant if and only if the same corpus produces the same
// bytes: same header, same aligned record region, same canonical footer JSON
// (sort_keys, ","/" separators, ensure_ascii escaping), same trailer, same
// SHA-256. The golden manifest (tests/data/capture_golden_manifest.json) pins
// the whole-object digest.

#ifndef DMI_PACK_PACK_BUILDER_H_
#define DMI_PACK_PACK_BUILDER_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <openssl/sha.h>

namespace dmi_pack {

// --- format constants (mirror pack.py) ---
inline constexpr uint32_t kPackMajorVersion = 1;
inline constexpr uint32_t kPackMinorVersion = 0;
inline constexpr size_t kPackAlignment = 64;
inline constexpr uint64_t kMaxFooterBytes = 64ull * 1024 * 1024;
inline constexpr uint64_t kMaxRecords = 1'000'000;
inline constexpr size_t kHeaderSize = 64;   // <8sHHI16sQI20s
inline constexpr size_t kTrailerSize = 64;  // <8sHHQQI32s
inline constexpr size_t kMaxRank = 32;

// Errors, carried as codes; the binding maps them onto the Python taxonomy.
enum class Status {
  kOk = 0,
  kSealedState,     // pack already sealed
  kRecordLimit,     // record limit reached
  kDuplicateId,     // duplicate capture id
  kCapacity,        // record/pack would exceed limits
  kEmpty,           // seal of an empty pack
  kBadArgument,     // invalid pack_id / metadata / dtype / shape
};

inline const char* StatusName(Status s) {
  switch (s) {
    case Status::kOk: return "ok";
    case Status::kSealedState: return "pack is already sealed";
    case Status::kRecordLimit: return "pack record limit reached";
    case Status::kDuplicateId: return "duplicate capture id";
    case Status::kCapacity: return "record would exceed max_pack_bytes";
    case Status::kEmpty: return "cannot seal an empty pack";
    case Status::kBadArgument: return "invalid argument";
  }
  return "unknown";
}

// CRC-32 (IEEE, reflected) — bit-identical to zlib.crc32.
uint32_t Crc32(const uint8_t* data, size_t n, uint32_t crc = 0);

// The metadata one record carries. Field order matches CaptureMetadata.
struct RecordMetadata {
  std::string capture_id;
  std::string tenant_id;
  std::string experiment_id;
  std::string run_id;
  std::string session_id;
  std::string request_id;
  std::string sequence_id;
  std::string model_id;
  std::string model_revision;
  std::optional<std::string> adapter_revision;  // nullopt -> JSON null
  std::string capture_policy_version;
  std::string hook_name;
  int64_t layer_number = -1;      // [-1, 2^31-1]
  uint64_t producer_rank = 0;     // <= 2^32-1
  uint64_t step_number = 0;       // <= 2^64-1
  uint64_t token_start = 0;
  uint64_t token_end = 0;
  uint64_t batch_position = 0;    // <= 2^32-1
  std::string dtype;              // one of the ten supported names
  std::vector<uint32_t> shape;    // each <= 2^31-1, rank <= 32
  uint64_t captured_at_ns = 0;
};

// One record to append: metadata by const reference (Append only reads it)
// plus a payload pointer the caller keeps alive for the call.
struct PackRecord {
  const RecordMetadata* metadata = nullptr;
  const uint8_t* payload = nullptr;
  size_t payload_bytes = 0;
};

struct SealedPack {
  std::string pack_id;
  uint64_t created_at_ns = 0;
  std::vector<uint8_t> data;
  uint64_t record_count = 0;
  uint64_t footer_offset = 0;
  std::string checksum;  // sha256 hex of data
};

// Validate one metadata against the same rules the reference enforces.
Status ValidateMetadata(const RecordMetadata& m);

// Canonical JSON for one record row, exactly as _encode_json produces it:
// sort_keys=True, separators=(",", ":"), ensure_ascii=True.
void EncodeRecordRow(const RecordMetadata& m, uint64_t offset,
                     uint64_t stored_length, uint64_t decoded_length,
                     const char* codec, const std::string& checksum,
                     std::string* out);

class PackBuilder {
 public:
  PackBuilder(const std::string& pack_id, uint64_t created_at_ns,
              uint64_t max_pack_bytes, uint64_t max_records = kMaxRecords);

  Status Append(const PackRecord& record);
  Status Seal(SealedPack* out);

  uint64_t record_count() const { return record_count_; }
  bool sealed() const { return sealed_; }

 private:
  std::string pack_id_;
  std::array<uint8_t, 16> uuid_{};
  uint64_t created_at_ns_ = 0;
  uint64_t max_pack_bytes_ = 0;
  uint64_t max_records_ = 0;
  std::vector<uint8_t> buffer_;
  std::vector<std::string> record_json_;
  uint64_t record_json_bytes_ = 0;
  uint64_t record_count_ = 0;
  std::vector<std::string> capture_ids_;  // sorted, deduped by construction
  std::string footer_prefix_cache_;
  std::string footer_suffix_cache_;
  bool sealed_ = false;
};

// Convenience: parse a canonical UUID string; false if invalid.
bool ParseUuid(const std::string& text, std::array<uint8_t, 16>* out,
               std::string* canonical);

}  // namespace dmi_pack

#endif  // DMI_PACK_PACK_BUILDER_H_
