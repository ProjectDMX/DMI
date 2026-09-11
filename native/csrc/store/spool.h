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
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

  // Startup cleanup: delete abandoned "*.open" files, validate ready packs,
  // quarantine failures, and rebuild accounting. Other writers on this root
  // must be stopped; use ListPending() while they are running.
  SpoolStatus Recover(std::vector<StagedPack>* out, std::string* error);

  // Validate and list ready packs without deleting in-progress writes.
  SpoolStatus ListPending(std::vector<StagedPack>* out, std::string* error);

  // Remove one staged pack after upload (identity + size verified first).
  SpoolStatus Remove(const StagedPack& staged, std::string* error);

  SpoolSnapshot Snapshot() const;

  // Test seam: called by Stage() after its capacity reservation is taken and
  // before the temp file is written, outside the lock. Lets a test hold one
  // stager at exactly the point where its reservation exists but nothing is
  // on disk yet, which is the window a directory scan cannot see.
  void SetStageHookForTesting(std::function<void()> hook);

 private:
  SpoolStatus Scan(std::vector<StagedPack>* out, bool discard_open_files,
                  std::string* error);
  // Count/uncount one ready path in the committed account, at most once each
  // -- Python's _account_ready_locked / _unaccount_ready_locked. `mutex_`
  // must be held. Both return whether they actually changed the account.
  //
  // The aggregate alone cannot decide this. A ready file reached by the retry
  // or EEXIST-loser path may have been created by THIS object (already
  // counted, so adding would double it) or by a second Spool object on the
  // same root after this one's Open() (never counted, so adding nothing
  // leaves the cap judged against 0 -- with max 1500, one 1000-byte file
  // staged elsewhere then retried here left room for another 1000).
  bool AccountReadyLocked(const std::string& path, uint64_t object_bytes);
  bool UnaccountReadyLocked(const std::string& path);

  std::string root_;
  uint64_t max_bytes_ = 0;
  mutable std::mutex mutex_;
  // Two accounts, kept apart on purpose:
  //   committed_bytes_/committed_entries_ -- ready files this object knows
  //     are on disk. This is what a directory scan can re-derive, and what
  //     Recover()/reconciliation overwrite.
  //   reserved_bytes_/reserved_entries_ -- fresh stages between their
  //     capacity check and their link(). Nothing on disk speaks for these
  //     (the temp file, if it exists yet, is named in `inflight_temps_` so a
  //     scan skips it), so no scan may ever overwrite them.
  // Capacity is judged against the SUM. A single counter that a scan
  // replaced wholesale erased the second account: with max 1500, A reserved
  // 1000 and paused before its write, B's 1000 triggered the scan, the scan
  // saw an empty directory, and both were admitted (2000 on disk).
  uint64_t committed_bytes_ = 0;
  uint64_t committed_entries_ = 0;
  uint64_t reserved_bytes_ = 0;
  uint64_t reserved_entries_ = 0;
  // Which ready paths the committed account currently includes, and for how
  // many bytes -- Python's _accounted_ready. It is what makes the accounting
  // idempotent per PATH rather than per call, so the same ready file can be
  // met more than once (retry, EEXIST loser, a scan) and be counted exactly
  // once. Rebuilt wholesale wherever committed_* is. Stale .open bytes are
  // in committed_bytes_ without being here, exactly as in Python's
  // constructor, so this map is a ledger of ready paths, not a second copy
  // of the aggregate.
  std::unordered_map<std::string, uint64_t> accounted_ready_;
  std::unordered_set<std::string> inflight_temps_;
  uint64_t peak_bytes_ = 0;
  uint64_t generation_ = 0;
  std::function<void()> stage_hook_for_testing_;
};

}  // namespace dmi_store

#endif  // DMI_STORE_SPOOL_H_
