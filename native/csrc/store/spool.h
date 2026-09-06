// Durable pack spool: the crash-safe staging area between pack assembly and
// upload, byte-compatible with the Python DurablePackSpool (spool.py).
//
// On-disk contract (must stay identical — the Python uploader drains
// native-written spools and vice versa):
//   <root>/<key-parent-dirs>/<pack_id>.<created_at_ns>.<record_count>.
//       <checksum>.dmi-pack.ready   — the pack bytes verbatim
//   .<pack_id>.<random>.open                    — in-progress write, same dir
//   <name>.quarantined                          — corrupt ready file, sidelined
// Stage: write temp + fsync → hardlink to ready name → unlink temp → fsync
// the directory chain to the root. Idempotent: re-staging validates the
// existing ready file (name, size, sha256) and returns it; a different pack
// under the same pack_id is a conflict.

#ifndef DMI_STORE_SPOOL_H_
#define DMI_STORE_SPOOL_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace dmi_store {

struct SpoolConfig {
  std::string root;
  uint64_t max_bytes = 0;
};

struct StagedPack {
  std::string pack_id;
  uint64_t created_at_ns = 0;
  uint64_t record_count = 0;
  std::string checksum;  // sha256 hex of the staged bytes
  std::string object_key;  // upload key: <parent>/<pack_id>.dmi-pack
  std::string path;        // absolute ready-file path
  uint64_t object_bytes = 0;
};

struct SpoolSnapshot {
  uint64_t entries = 0;
  uint64_t bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t max_bytes = 0;
};

enum class SpoolStatus {
  kOk = 0,
  kFull,        // byte limit exceeded
  kConflict,    // same pack_id, different content
  kIntegrity,   // ready file fails validation
  kIo,          // filesystem error
  kBadArgument,
};

inline const char* SpoolStatusName(SpoolStatus s) {
  switch (s) {
    case SpoolStatus::kOk: return "ok";
    case SpoolStatus::kFull: return "spool byte limit exceeded";
    case SpoolStatus::kConflict: return "spool already contains a different pack";
    case SpoolStatus::kIntegrity: return "ready pack failed validation";
    case SpoolStatus::kIo: return "spool filesystem error";
    case SpoolStatus::kBadArgument: return "invalid argument";
  }
  return "unknown";
}

class Spool {
 public:
  // Opens (creating) the root. Recovery of pre-existing files is explicit
  // via Recover(), matching the Python constructor + recover() split.
  static SpoolStatus Open(SpoolConfig config, Spool* out, std::string* error);

  Spool() = default;
  Spool(const Spool&) = delete;
  Spool& operator=(const Spool&) = delete;

  // Stage pack bytes. object_key must end in "<pack_id>.dmi-pack".
  SpoolStatus Stage(const std::string& pack_id, uint64_t created_at_ns,
                    uint64_t record_count, const std::string& checksum,
                    const std::string& object_key, const uint8_t* data,
                    size_t n, StagedPack* out, std::string* error);

  // Delete "*.open" leftovers, validate every "*.dmi-pack.ready" (name +
  // sha256), quarantine failures, rebuild accounting. Returns survivors.
  SpoolStatus Recover(std::vector<StagedPack>* out, std::string* error);

  // Remove one staged pack after upload (identity + size verified first).
  SpoolStatus Remove(const StagedPack& staged, std::string* error);

  SpoolSnapshot Snapshot() const;

 private:
  std::string root_;
  uint64_t max_bytes_ = 0;
  mutable std::mutex mutex_;
  uint64_t bytes_ = 0;
  uint64_t entries_ = 0;
  uint64_t peak_bytes_ = 0;
  uint64_t generation_ = 0;
};

}  // namespace dmi_store

#endif  // DMI_STORE_SPOOL_H_
