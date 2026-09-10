#include "spool.h"

#include <openssl/sha.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

#include <fcntl.h>
#include <unistd.h>

namespace dmi_store {
namespace fs = std::filesystem;

namespace {

constexpr const char* kReadySuffix = ".dmi-pack.ready";
constexpr const char* kOpenSuffix = ".open";

bool HasSuffix(const std::string& name, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
}

bool IsHex64(const std::string& s) {  if (s.size() != 64) return false;
  for (char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

bool IsUuid(const std::string& s) {
  if (s.size() != 36) return false;
  for (size_t i = 0; i < 36; ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (s[i] != '-') return false;
      continue;
    }
    const char c = s[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
          (c >= 'A' && c <= 'F'))) {
      return false;
    }
  }
  return true;
}

// <pack_id>.<created>.<records>.<checksum>.dmi-pack.ready
bool ParseReadyName(const std::string& name, std::string* pack_id,
                    uint64_t* created, uint64_t* records,
                    std::string* checksum) {
  constexpr const char* kSuffix = ".dmi-pack.ready";
  const size_t n = name.size(), suffix = std::strlen(kSuffix);
  if (n <= suffix || name.compare(n - suffix, suffix, kSuffix) != 0) {
    return false;
  }
  const std::string stem = name.substr(0, n - suffix);
  // pack_id itself contains dashes, so split from the right: checksum (64
  // hex), records (digits), created (digits), remainder is the pack id.
  const size_t c3 = stem.rfind('.');
  if (c3 == std::string::npos) return false;
  const size_t c2 = stem.rfind('.', c3 - 1);
  if (c2 == std::string::npos) return false;
  const size_t c1 = stem.rfind('.', c2 - 1);
  if (c1 == std::string::npos) return false;
  const std::string id = stem.substr(0, c1);
  const std::string created_s = stem.substr(c1 + 1, c2 - c1 - 1);
  const std::string records_s = stem.substr(c2 + 1, c3 - c2 - 1);
  const std::string checksum_s = stem.substr(c3 + 1);
  if (!IsUuid(id) || !IsHex64(checksum_s)) return false;
  auto all_digits = [](const std::string& s) {
    return !s.empty() &&
           std::all_of(s.begin(), s.end(),
                       [](char c) { return c >= '0' && c <= '9'; });
  };
  if (!all_digits(created_s) || !all_digits(records_s)) return false;
  *pack_id = id;
  *created = std::strtoull(created_s.c_str(), nullptr, 10);
  *records = std::strtoull(records_s.c_str(), nullptr, 10);
  *checksum = checksum_s;
  return true;
}

bool FsyncDir(const std::string& dir, std::string* error) {
  const int fd = ::open(dir.c_str(), O_RDONLY);
  if (fd < 0) {
    if (error) *error = "cannot open directory " + dir;
    return false;
  }
  const bool ok = ::fsync(fd) == 0;
  if (!ok && error) *error = "fsync failed for " + dir;
  ::close(fd);
  return ok;
}

// fsync every directory from `leaf` up to and including `root`.
bool FsyncChain(const std::string& root, const std::string& leaf,
                std::string* error) {
  std::string dir = leaf;
  for (;;) {
    if (!FsyncDir(dir, error)) return false;
    if (dir == root) return true;
    const size_t slash = dir.rfind('/');
    if (slash == std::string::npos) {
      if (error) *error = "directory escapes spool root";
      return false;
    }
    dir = dir.substr(0, slash);
    if (dir.size() < root.size() ||
        dir.compare(0, root.size(), root) != 0) {
      if (error) *error = "directory escapes spool root";
      return false;
    }
  }
}

std::string Sha256HexFile(const std::string& path, std::string* error) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  constexpr size_t kChunk = 1 << 20;
  std::vector<uint8_t> chunk(kChunk);
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    if (error) *error = "cannot open " + path;
    return "";
  }
  for (;;) {
    const ssize_t n = ::read(fd, chunk.data(), chunk.size());
    if (n < 0) {
      if (error) *error = "cannot read " + path;
      ::close(fd);
      return "";
    }
    if (n == 0) break;
    SHA256_Update(&ctx, chunk.data(), static_cast<size_t>(n));
  }
  ::close(fd);
  SHA256_Final(digest, &ctx);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.resize(64);
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0xF];
  }
  return out;
}

bool WriteFileSynced(const std::string& path, const uint8_t* data, size_t n,
                     std::string* error) {
  const int fd =
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    if (error) *error = "cannot create " + path;
    return false;
  }
  size_t written = 0;
  while (written < n) {
    const ssize_t w = ::write(fd, data + written, n - written);
    if (w <= 0) {
      if (error) *error = "cannot write " + path;
      ::close(fd);
      return false;
    }
    written += static_cast<size_t>(w);
  }
  const bool ok = ::fsync(fd) == 0;
  if (!ok && error) *error = "fsync failed for " + path;
  ::close(fd);
  return ok;
}

// One path component of an object key, as Python's _KEY_COMPONENT spells it:
// ^[A-Za-z0-9][A-Za-z0-9._=%-]*$. Note '.' is legal after the first byte, so
// "tenant=a..b" is a legal component -- only "", "." and ".." as *whole*
// components traverse, and those cannot match the regex anyway.
bool IsKeyComponent(const std::string& part) {
  if (part.empty() || part == "." || part == "..") return false;
  const unsigned char first = static_cast<unsigned char>(part[0]);
  if (!((first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z') ||
        (first >= '0' && first <= '9'))) {
    return false;
  }
  for (size_t i = 1; i < part.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(part[i]);
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
          (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '=' ||
          c == '%' || c == '-')) {
      return false;
    }
  }
  return true;
}

// object_key must end in "<pack_id>.dmi-pack"; the upload key is what the
// Python store derives: parent dirs + pack_id + ".dmi-pack".
bool SplitKey(const std::string& object_key, const std::string& pack_id,
              std::string* parent, std::string* error) {
  const std::string leaf = pack_id + ".dmi-pack";
  if (object_key.size() < leaf.size() ||
      object_key.compare(object_key.size() - leaf.size(), leaf.size(),
                         leaf) != 0 ||
      (!object_key.empty() &&
       object_key.size() != leaf.size() &&
       object_key[object_key.size() - leaf.size() - 1] != '/')) {
    if (error) *error = "spool object key must end with the pack ID";
    return false;
  }
  // Component-wise, exactly like Python's validate_object_key: a substring
  // test would refuse the legal "tenant=a..b" and admit other traversals.
  if (object_key.find('\\') != std::string::npos) {
    if (error) *error = "object key is invalid";
    return false;
  }
  for (size_t begin = 0; begin <= object_key.size();) {
    const size_t slash = object_key.find('/', begin);
    const size_t end = (slash == std::string::npos) ? object_key.size() : slash;
    if (!IsKeyComponent(object_key.substr(begin, end - begin))) {
      if (error) *error = "object key is invalid";
      return false;
    }
    if (slash == std::string::npos) break;
    begin = slash + 1;
  }
  const size_t slash = object_key.rfind('/');
  *parent = (slash == std::string::npos) ? "" : object_key.substr(0, slash);
  return true;
}

std::string ReadyName(const std::string& pack_id, uint64_t created,
                      uint64_t records, const std::string& checksum) {
  return pack_id + "." + std::to_string(created) + "." +
         std::to_string(records) + "." + checksum + ".dmi-pack.ready";
}

}  // namespace

SpoolStatus Spool::Open(SpoolConfig config, Spool* out, std::string* error) {
  if (config.max_bytes == 0) {
    if (error) *error = "max_bytes must be positive";
    return SpoolStatus::kBadArgument;
  }
  std::error_code ec;
  fs::create_directories(config.root, ec);
  if (ec) {
    if (error) *error = "cannot create spool root: " + ec.message();
    return SpoolStatus::kIo;
  }
  if (!FsyncDir(config.root, error)) return SpoolStatus::kIo;
  // Canonical root for escape checks below.
  char resolved[4096];
  if (::realpath(config.root.c_str(), resolved) == nullptr) {
    if (error) *error = "cannot resolve spool root";
    return SpoolStatus::kIo;
  }
  out->root_ = resolved;
  out->max_bytes_ = config.max_bytes;
  out->committed_bytes_ = 0;
  out->committed_entries_ = 0;
  out->reserved_bytes_ = 0;
  out->reserved_entries_ = 0;
  out->inflight_temps_.clear();
  out->peak_bytes_ = 0;
  out->generation_ = 0;
  // Account pre-existing files exactly like the Python constructor: ready
  // files plus stale .open files both count until Recover() runs.
  for (const auto& entry :
       fs::recursive_directory_iterator(out->root_, ec)) {
    if (ec) break;
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    const bool is_ready = HasSuffix(name, kReadySuffix);
    const bool is_open = HasSuffix(name, kOpenSuffix);
    if (!is_ready && !is_open) continue;
    out->committed_bytes_ += entry.file_size();
    if (is_ready) ++out->committed_entries_;
  }
  out->peak_bytes_ = out->committed_bytes_;
  return SpoolStatus::kOk;
}

void Spool::SetStageHookForTesting(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(mutex_);
  stage_hook_for_testing_ = std::move(hook);
}

SpoolStatus Spool::Stage(const std::string& pack_id, uint64_t created_at_ns,
                         uint64_t record_count, const std::string& checksum,
                         const std::string& object_key, const uint8_t* data,
                         size_t n, StagedPack* out, std::string* error) {
  if (!IsUuid(pack_id) || !IsHex64(checksum)) {
    if (error) *error = "pack_id must be a UUID and checksum 64 hex";
    return SpoolStatus::kBadArgument;
  }
  std::string parent;
  if (!SplitKey(object_key, pack_id, &parent, error)) {
    return SpoolStatus::kBadArgument;
  }
  const std::string dir =
      parent.empty() ? root_ : root_ + "/" + parent;
  // A well-formed key still escapes if one of its ancestors is a symlink out
  // of the root, and no textual check can see that. Resolve it the way Python
  // does (Path.resolve(strict=False) + is_relative_to): weakly_canonical
  // follows the existing symlinked ancestors and tolerates a tail that does
  // not exist yet. root_ is already a realpath, so this compares canonical to
  // canonical -- and FsyncChain's textual walk then only ever sees a path
  // genuinely contained by the root.
  {
    std::error_code ec;
    const std::string resolved = fs::weakly_canonical(dir, ec).string();
    if (ec || (resolved != root_ &&
               resolved.compare(0, root_.size() + 1, root_ + "/") != 0)) {
      if (error) *error = "object key escapes the spool root";
      return SpoolStatus::kBadArgument;
    }
  }
  const std::string ready =
      dir + "/" + ReadyName(pack_id, created_at_ns, record_count, checksum);
  const std::string upload_key =
      (parent.empty() ? "" : parent + "/") + pack_id + ".dmi-pack";
  // Temp file in the same directory (link atomicity needs it). Named
  // before the reservation so the reservation can register it: a
  // reconciling scan must skip THIS stage's own .open file, which its
  // reservation already accounts for.
  std::string temp = dir + "/." + pack_id + ".";
  {
    std::random_device rd;
    for (int i = 0; i < 8; ++i) {
      static const char* kDigits = "0123456789abcdef";
      temp.push_back(kDigits[rd() & 0xF]);
    }
    temp += ".open";
  }

  // Phase 1 (locked): decide retry vs fresh, reserve capacity. File I/O
  // stays outside the lock so concurrent workers never serialize on disk.
  bool retry = false;
  std::function<void()> stage_hook;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stage_hook = stage_hook_for_testing_;
    std::error_code ec;
    if (fs::exists(ready, ec)) {
      retry = true;
    } else {
      // A different pack intent under the same pack_id is a conflict.
      for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.compare(0, pack_id.size(), pack_id) == 0 &&
            name.size() > pack_id.size() + 1 && name[pack_id.size()] == '.' &&
            HasSuffix(name, kReadySuffix)) {
          if (error) {
            *error =
                "spool already contains a different pack intent: " + pack_id;
          }
          return SpoolStatus::kConflict;
        }
      }
      if (committed_bytes_ + reserved_bytes_ + n > max_bytes_) {
        // The committed counter is only as fresh as the Remove calls THIS
        // Spool object has seen. An uploader running through a second
        // Spool object on the same root (or another process) removes
        // ready files and updates its OWN counter — this object's never
        // learns about the released capacity, so staging eventually
        // refuses on a directory that is actually empty (reproduced: sink
        // with a 1500-byte limit, serial upload-and-remove between two
        // records). Reconcile the COMMITTED account from the directory —
        // the durable truth for what is committed — before refusing.
        //
        // The scan is authoritative for committed files only. The
        // reservations other stagers hold right now are not on disk (or
        // are, as a temp file this object already accounts for), so they
        // are kept as they are and added back on top: replacing a single
        // combined counter with the scan admitted a concurrent stage the
        // reservation should have refused.
        uint64_t actual = 0;
        uint64_t ready_count = 0;
        std::error_code walk_ec;
        for (const auto& entry :
             fs::recursive_directory_iterator(root_, walk_ec)) {
          if (walk_ec) break;
          if (!entry.is_regular_file()) continue;
          const std::string name = entry.path().filename().string();
          if (HasSuffix(name, kReadySuffix)) {
            actual += entry.file_size();
            ++ready_count;
          } else if (HasSuffix(name, kOpenSuffix) &&
                     inflight_temps_.count(entry.path().string()) == 0) {
            // Someone else's in-progress write (another process, or a
            // stale leftover Recover() has not swept yet): counts, as it
            // does at Open(). This object's own temps are reservations.
            actual += entry.file_size();
          }
        }
        committed_bytes_ = actual;
        committed_entries_ = ready_count;
        if (committed_bytes_ + reserved_bytes_ + n > max_bytes_) {
          if (error) {
            *error = "spool byte limit exceeded: " +
                     std::to_string(committed_bytes_ + reserved_bytes_ + n) +
                     " > " + std::to_string(max_bytes_);
          }
          return SpoolStatus::kFull;
        }
      }
      // Reserve now: the link below is the atomic commit, and two workers
      // racing fresh stages must not both pass the capacity check.
      reserved_bytes_ += n;
      ++reserved_entries_;
      inflight_temps_.insert(temp);
      peak_bytes_ = std::max(peak_bytes_, committed_bytes_ + reserved_bytes_);
      ++generation_;
    }
  }

  // Release this stage's reservation without committing anything.
  auto unreserve = [&] {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reserved_bytes_ >= n) reserved_bytes_ -= n;
    if (reserved_entries_ > 0) --reserved_entries_;
    inflight_temps_.erase(temp);
    ++generation_;
  };
  // Move this stage's reservation into the committed account: the ready
  // file now exists, so a scan will see it from here on.
  auto commit_reservation = [&] {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reserved_bytes_ >= n) reserved_bytes_ -= n;
    if (reserved_entries_ > 0) --reserved_entries_;
    inflight_temps_.erase(temp);
    committed_bytes_ += n;
    ++committed_entries_;
    ++generation_;
  };
  auto fill_out = [&] {
    out->pack_id = pack_id;
    out->created_at_ns = created_at_ns;
    out->record_count = record_count;
    out->checksum = checksum;
    out->object_key = upload_key;
    out->path = ready;
    out->object_bytes = n;
  };
  // Validate a ready file that should hold this pack (retry path, or the
  // EEXIST loser). Name, size, and full sha256 — the hash is the slow part
  // and runs without the lock.
  auto validate_ready = [&]() -> bool {
    std::string id, sum;
    uint64_t created = 0, records = 0;
    const std::string name = fs::path(ready).filename().string();
    std::error_code ec;
    if (!ParseReadyName(name, &id, &created, &records, &sum) ||
        id != pack_id || created != created_at_ns ||
        records != record_count || sum != checksum ||
        fs::file_size(ready, ec) != n ||
        Sha256HexFile(ready, nullptr) != checksum) {
      return false;
    }
    return true;
  };

  if (retry) {
    if (!validate_ready()) {
      if (error) *error = "spool contains different content: " + object_key;
      return SpoolStatus::kConflict;
    }
    // A retry adds no accounting: the file was counted by the stage (or
    // process start) that created it. Every successful retry still closes
    // the durability window itself with a fresh fsync chain.
    if (!FsyncChain(root_, dir, error)) return SpoolStatus::kIo;
    fill_out();
    return SpoolStatus::kOk;
  }

  // Reservation held, nothing on disk yet: the window the test seam opens.
  if (stage_hook) stage_hook();

  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    unreserve();
    if (error) *error = "cannot create spool directory: " + ec.message();
    return SpoolStatus::kIo;
  }
  if (!WriteFileSynced(temp, data, n, error)) {
    unreserve();
    ::unlink(temp.c_str());
    return SpoolStatus::kIo;
  }
  if (::link(temp.c_str(), ready.c_str()) != 0) {
    if (errno == EEXIST) {
      // Lost the race: the winner's file is already counted (by its stage
      // or by process start), so release this reservation, then validate
      // the winner exactly like the retry path.
      ::unlink(temp.c_str());
      unreserve();
      if (!validate_ready()) {
        if (error) *error = "spool contains different content: " + object_key;
        return SpoolStatus::kConflict;
      }
      // The winner may still be between link() and its own fsync: do not
      // acknowledge its dirent before independently making the chain
      // durable (mirrors the Python loser's fsync).
      if (!FsyncChain(root_, dir, error)) return SpoolStatus::kIo;
      fill_out();
      return SpoolStatus::kOk;
    }
    unreserve();
    if (error) {
      *error =
          "cannot link ready file: " + std::string(strerror(errno));
    }
    ::unlink(temp.c_str());
    return SpoolStatus::kIo;
  }
  // Linked: the reservation becomes a committed file, the temp name goes,
  // and the directory chain is synced (outside the lock).
  ::unlink(temp.c_str());
  commit_reservation();
  if (!FsyncChain(root_, dir, error)) return SpoolStatus::kIo;
  fill_out();
  return SpoolStatus::kOk;
}


SpoolStatus Spool::Recover(std::vector<StagedPack>* out, std::string* error) {
  return Scan(out, true, error);
}

SpoolStatus Spool::ListPending(std::vector<StagedPack>* out, std::string* error) {
  return Scan(out, false, error);
}

SpoolStatus Spool::Scan(std::vector<StagedPack>* out, bool discard_open_files,
                        std::string* error) {
  out->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  std::error_code ec;
  std::vector<std::string> readies;
  uint64_t bytes = 0;
  for (const auto& entry :
       fs::recursive_directory_iterator(root_, ec)) {
    if (ec) break;
    if (!entry.is_regular_file()) continue;
    const std::string path = entry.path().string();
    const std::string name = entry.path().filename().string();
    if (HasSuffix(name, kOpenSuffix)) {
      // Our own writes are already accounted for by their reservations.
      if (inflight_temps_.count(path) != 0) continue;
      if (discard_open_files) {
        ::unlink(path.c_str());
        FsyncDir(entry.path().parent_path().string(), nullptr);
      } else {
        const uint64_t size = entry.file_size(ec);
        if (!ec) bytes += size;
        ec.clear();  // Another writer may have just committed its temp.
      }
      continue;
    }
    if (HasSuffix(name, kReadySuffix)) {
      readies.push_back(path);
    }
  }
  std::sort(readies.begin(), readies.end());
  for (const std::string& path : readies) {
    const std::string name = fs::path(path).filename().string();
    std::string id, sum;
    uint64_t created = 0, records = 0;
    const uint64_t size = fs::file_size(path, ec);
    if (!ParseReadyName(name, &id, &created, &records, &sum) || ec ||
        Sha256HexFile(path, nullptr) != sum) {
      // Quarantine: keep the bytes, drop the .ready suffix.
      const std::string target = path.substr(0, path.size() - 6) +
                                 ".quarantined";
      ::rename(path.c_str(), target.c_str());
      FsyncDir(fs::path(path).parent_path().string(), nullptr);
      ++generation_;
      continue;
    }
    const std::string rel = fs::relative(path, root_, ec).string();
    const size_t slash = rel.rfind('/');
    const std::string parent = (slash == std::string::npos) ? "" : rel.substr(0, slash);
    StagedPack staged;
    staged.pack_id = id;
    staged.created_at_ns = created;
    staged.record_count = records;
    staged.checksum = sum;
    staged.object_key = (parent.empty() ? "" : parent + "/") + id + ".dmi-pack";
    staged.path = path;
    staged.object_bytes = size;
    out->push_back(std::move(staged));
    bytes += size;
  }
  // Recovery rebuilds the committed account only; a stage in flight on
  // another thread keeps its reservation.
  committed_bytes_ = bytes;
  committed_entries_ = out->size();
  peak_bytes_ = std::max(peak_bytes_, committed_bytes_ + reserved_bytes_);
  ++generation_;
  (void)error;
  return SpoolStatus::kOk;
}

SpoolStatus Spool::Remove(const StagedPack& staged, std::string* error) {
  std::string parent;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::error_code ec;
    if (!fs::exists(staged.path, ec)) {
      if (ec) {
        if (error) *error = "cannot inspect staged pack: " + ec.message();
        return SpoolStatus::kIo;
      }
      // Removal retries must not release another pack's capacity. A stale
      // count after an external removal is reconciled before refusing Stage.
      return SpoolStatus::kOk;
    }
    const std::string name = fs::path(staged.path).filename().string();
    std::string id, sum;
    uint64_t created = 0, records = 0;
    if (!ParseReadyName(name, &id, &created, &records, &sum) ||
        id != staged.pack_id || created != staged.created_at_ns ||
        records != staged.record_count || sum != staged.checksum ||
        fs::file_size(staged.path, ec) != staged.object_bytes) {
      if (error) *error = "staged pack identity changed before removal";
      return SpoolStatus::kIntegrity;
    }
    if (::unlink(staged.path.c_str()) != 0) {
      if (errno == ENOENT) return SpoolStatus::kOk;
      if (error) *error = "cannot remove staged pack: " + std::string(strerror(errno));
      return SpoolStatus::kIo;
    }
    if (committed_bytes_ >= staged.object_bytes) {
      committed_bytes_ -= staged.object_bytes;
    }
    if (committed_entries_ > 0) --committed_entries_;
    ++generation_;
    parent = fs::path(staged.path).parent_path().string();
  }
  // Directory fsync outside the lock: durability without serializing
  // concurrent stagers on it.
  FsyncDir(parent, nullptr);
  return SpoolStatus::kOk;
}

SpoolSnapshot Spool::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  SpoolSnapshot snapshot;
  // The snapshot reports what capacity is judged against: committed plus
  // reserved, exactly as the single counter did before the split.
  snapshot.entries = committed_entries_ + reserved_entries_;
  snapshot.bytes = committed_bytes_ + reserved_bytes_;
  snapshot.peak_bytes = peak_bytes_;
  snapshot.max_bytes = max_bytes_;
  return snapshot;
}

}  // namespace dmi_store
