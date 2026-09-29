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
//
// One owner per directory (B6). Recover() deletes every .open file this
// object is not writing, so it is only safe while no other PROCESS writes
// there. A spool directory therefore has an owner lock -- flock(LOCK_EX) on
// <root>/.owner.lock, whose content records the holder's host and pid --
// and a second process that tries to take it is refused, told who holds
// it. The lock goes with its holder, even one killed with SIGKILL.
//   - owner_lock=kTake (the default) takes it in Open(), before anything
//     reads the directory, and holds it for the Spool object's life.
//   - owner_lock=kHeldByCaller takes none: the calling process holds a
//     SpoolOwnerLock on the directory already. Two Spools in ONE process
//     that both take refuse each other (flock binds to an open file
//     description, not to the process), so a process running a sink and a
//     storage service on one directory holds one SpoolOwnerLock and opens
//     both Spools with kHeldByCaller. Open() refuses it unless one of THIS
//     process's descriptors holds the lock (SpoolOwnedByThisProcess): kOwned,
//     naming the holder, beside another process's lock, and kBadArgument
//     when nothing holds it. Standalone callers -- the drivers, adoption --
//     take.
// A directory nested under a HELD spool directory, or containing one with
// a .owner.lock file (held or not: a dead directory's packs are for its
// successor to adopt), is refused: Scan walks recursively, so the outer
// spool's Recover would reach into the inner one. An unheld lock file
// ABOVE refuses nothing -- every take leaves its file behind -- since the
// next take of that directory meets this one's lock file below it. The
// check runs after the lock is taken, so of two processes taking an outer
// and a nested directory at once, at least one is refused. <root>/_refs/ is
// never scanned: the upload handoff's ref files live there (plan section
// 2.4). The spool must be node-local: NFS and Lustre are refused by statfs
// f_type unless allow_shared_filesystem is set, since neither guarantees a
// flock that excludes a process on another node.
//
// The Python DurablePackSpool (spool.py) takes no lock, and its recover()
// deletes every .open file under its root; the C++ spool is deliberately
// stricter, and that is not ported to the reference.

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

enum class OwnerLock {
  kTake = 0,      // Open() takes <root>/.owner.lock for the Spool's life
  kHeldByCaller,  // the caller holds a SpoolOwnerLock on <root>
};

// "take" / "held_by_caller".
const char* OwnerLockName(OwnerLock mode);
bool ParseOwnerLock(const std::string& text, OwnerLock* mode);

struct SpoolConfig {
  std::string root;
  uint64_t max_bytes = 0;
  OwnerLock owner_lock = OwnerLock::kTake;
  // Admit a root on NFS or Lustre. Only safe when every process that could
  // open the directory runs on this node.
  bool allow_shared_filesystem = false;
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
  kOwned,       // another owner holds the directory's owner lock
};

inline const char* SpoolStatusName(SpoolStatus s) {
  switch (s) {
    case SpoolStatus::kOk: return "ok";
    case SpoolStatus::kFull: return "spool byte limit exceeded";
    case SpoolStatus::kConflict: return "spool already contains a different pack";
    case SpoolStatus::kIntegrity: return "ready pack failed validation";
    case SpoolStatus::kIo: return "spool filesystem error";
    case SpoolStatus::kBadArgument: return "invalid argument";
    case SpoolStatus::kOwned: return "spool directory is owned by another process";
  }
  return "unknown";
}

// The statfs f_type names of the shared filesystems a spool refuses: "NFS",
// "Lustre", or nullptr for any other. The list needs maintenance as
// deployments meet new ones.
const char* SharedFilesystemName(int64_t f_type);

// Refuses `dir` (which must exist) on a shared filesystem unless
// `allow_shared_filesystem`.
SpoolStatus CheckNodeLocal(const std::string& dir,
                           bool allow_shared_filesystem, std::string* error);

// Test seam: every node-local check in this binary reads `f_type` instead of
// calling statfs(2). A negative value restores statfs.
void SetFilesystemTypeForTesting(int64_t f_type);

// The holder recorded in a directory's owner lock file.
struct SpoolOwner {
  std::string host;
  int64_t pid = 0;
};

// Whether <dir>/.owner.lock is held right now (by any process, this one
// included), and if so who recorded themselves in it. A holder that has
// locked but not yet written its record reads as an empty host and pid 0.
bool ReadSpoolOwner(const std::string& dir, SpoolOwner* owner);

// Whether one of THIS process's descriptors holds <dir>/.owner.lock, as the
// kernel reports it in /proc/self/fdinfo (falling back to the recorded host
// and pid where /proc cannot be read). What kHeldByCaller requires.
bool SpoolOwnedByThisProcess(const std::string& dir);

// The owner lock of one spool directory: flock(LOCK_EX) on <dir>/.owner.lock,
// released with the object (or Release()), and by the kernel when the
// process dies. The descriptor is close-on-exec; a child forked WITHOUT exec
// shares it, and keeps the lock held for as long as it lives.
class SpoolOwnerLock {
 public:
  static constexpr const char* kFileName = ".owner.lock";

  SpoolOwnerLock() = default;
  ~SpoolOwnerLock();
  SpoolOwnerLock(SpoolOwnerLock&& other) noexcept;
  SpoolOwnerLock& operator=(SpoolOwnerLock&& other) noexcept;
  SpoolOwnerLock(const SpoolOwnerLock&) = delete;
  SpoolOwnerLock& operator=(const SpoolOwnerLock&) = delete;

  // Takes the lock on `dir`, creating it when it does not exist -- beside
  // its lock file, already held, and renamed into place, so no scan of the
  // parent ever meets the directory before its owner holds it -- and records
  // this host and pid in it. kOwned, naming the holder, when another holder
  // has it; kBadArgument for a shared filesystem or a directory nested
  // under, or containing, an owned one (see above), after letting go of
  // the lock and of whatever this call created.
  static SpoolStatus Acquire(const std::string& dir,
                             bool allow_shared_filesystem,
                             SpoolOwnerLock* out, std::string* error);

  // Adoption's try-lock: takes the lock of an EXISTING directory whose owner
  // is gone, creating its lock file if it has none. kOwned while its owner
  // lives; never creates the directory.
  static SpoolStatus TryAdopt(const std::string& dir, SpoolOwnerLock* out,
                              std::string* error);

  bool held() const { return fd_ >= 0; }
  // The canonical path of the directory, while held.
  const std::string& directory() const { return dir_; }

  void Release();

  // Releases the lock, first removing the directory if it holds nothing but
  // its lock file and empty subdirectories. Anything else -- a pack, a
  // quarantined file, a ref -- keeps the directory, which the next owner or
  // adopter meets as it was left. Returns whether the directory was removed.
  bool ReleaseAndRemoveIfEmpty(std::string* error);

 private:
  int fd_ = -1;
  std::string dir_;
};

// The spool layout of the plan's section 2.3. A capture process spools into
//   <base>/<catalog_key>/r<producer_rank>-<incarnation>/
// where catalog_key is the first 12 hex digits of
// sha256("<database>/<table_prefix>/<store_id>"), and incarnation is 8 hex
// digits fresh for every process start, so no two processes -- two jobs on
// one node, or a restart of the same rank -- ever share a directory. The
// directories under one catalog key are siblings: packs bound for one
// catalog and store, which a successor on the node adopts once their owner
// has died (CaptureStorageService, adopt_sibling_spools).
std::string SpoolCatalogKey(const std::string& database,
                            const std::string& table_prefix,
                            const std::string& store_id);
bool IsSpoolCatalogKey(const std::string& name);
std::string SpoolRankDirectoryName(uint64_t producer_rank,
                                   const std::string& incarnation);
// "r<rank>-<8 lowercase hex>", the rank in canonical decimal.
bool ParseSpoolRankDirectoryName(const std::string& name,
                                 uint64_t* producer_rank,
                                 std::string* incarnation);
std::string NewSpoolIncarnation();
std::string SpoolRankDirectory(const std::string& base,
                               const std::string& database,
                               const std::string& table_prefix,
                               const std::string& store_id,
                               uint64_t producer_rank,
                               const std::string& incarnation);

class Spool {
 public:
  // Opens (creating) the root, after the node-local check and, with kTake,
  // after taking its owner lock (kOwned when another holder has it);
  // kHeldByCaller is refused unless this process holds it. Recovery of
  // pre-existing files is explicit via Recover(), matching the Python
  // constructor + recover() split.
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
  // must be stopped; use ListPending() while they are running. The owner
  // lock keeps other processes out; writers in this process sharing it
  // (kHeldByCaller) are the caller's to order.
  SpoolStatus Recover(std::vector<StagedPack>* out, std::string* error);

  // Validate and list ready packs without deleting in-progress writes.
  SpoolStatus ListPending(std::vector<StagedPack>* out, std::string* error);

  // Remove one staged pack after upload (identity + size verified first).
  SpoolStatus Remove(const StagedPack& staged, std::string* error);

  SpoolSnapshot Snapshot() const;

  // The canonical root, once opened.
  const std::string& root() const { return root_; }

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
  // Rebuild the COMMITTED account (bytes, entries, path ledger) from the
  // directory, leaving in-flight reservations alone. Run under `mutex_`
  // before refusing a stage: this object's counter is only as fresh as the
  // Remove calls it has seen, so a second Spool object's removals make it
  // stale-high. The retry/EEXIST-loser paths run it too, so a charge that
  // would exceed the cap is judged against the same durable truth.
  void ReconcileCommittedLocked();

  std::string root_;
  uint64_t max_bytes_ = 0;
  // Held for the object's life under OwnerLock::kTake; empty otherwise.
  SpoolOwnerLock owner_lock_;
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
