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
  if (object_key.find("..") != std::string::npos) {
    if (error) *error = "object key escapes the spool root";
    return false;
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
  out->bytes_ = 0;
  out->entries_ = 0;
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
    out->bytes_ += entry.file_size();
    if (is_ready) ++out->entries_;
  }
  out->peak_bytes_ = out->bytes_;
  return SpoolStatus::kOk;
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
  const std::string ready =
      dir + "/" + ReadyName(pack_id, created_at_ns, record_count, checksum);
  const std::string upload_key =
      (parent.empty() ? "" : parent + "/") + pack_id + ".dmi-pack";

  std::lock_guard<std::mutex> lock(mutex_);
  std::error_code ec;
  if (fs::exists(ready, ec)) {
    // Idempotent retry: validate the existing file before blessing it.
    std::string id, sum;
    uint64_t created = 0, records = 0;
    const std::string name = fs::path(ready).filename().string();
    if (!ParseReadyName(name, &id, &created, &records, &sum) ||
        id != pack_id || created != created_at_ns ||
        records != record_count || sum != checksum ||
        fs::file_size(ready, ec) != n ||
        Sha256HexFile(ready, error) != checksum) {
      if (error && error->empty()) {
        *error = "spool contains different content: " + object_key;
      }
      return SpoolStatus::kConflict;
    }
    out->pack_id = pack_id;
    out->created_at_ns = created_at_ns;
    out->record_count = record_count;
    out->checksum = checksum;
    out->object_key = upload_key;
    out->path = ready;
    out->object_bytes = n;
    if (!FsyncChain(root_, dir, error)) return SpoolStatus::kIo;
    return SpoolStatus::kOk;
  }
  // A different pack intent under the same pack_id is a conflict.
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec) break;
    const std::string name = entry.path().filename().string();
    if (name.compare(0, pack_id.size(), pack_id) == 0 &&
        name.size() > pack_id.size() + 1 && name[pack_id.size()] == '.' &&
        HasSuffix(name, kReadySuffix)) {
      if (error) {
        *error = "spool already contains a different pack intent: " + pack_id;
      }
      return SpoolStatus::kConflict;
    }
  }
  if (bytes_ + n > max_bytes_) {
    if (error) {
      *error = "spool byte limit exceeded: " +
               std::to_string(bytes_ + n) + " > " +
               std::to_string(max_bytes_);
    }
    return SpoolStatus::kFull;
  }
  fs::create_directories(dir, ec);
  if (ec) {
    if (error) *error = "cannot create spool directory: " + ec.message();
    return SpoolStatus::kIo;
  }
  // Temp file in the same directory (rename/link atomicity needs it).
  std::random_device rd;
  std::string temp = dir + "/." + pack_id + ".";
  for (int i = 0; i < 8; ++i) {
    static const char* kDigits = "0123456789abcdef";
    temp.push_back(kDigits[rd() & 0xF]);
  }
  temp += ".open";
  bool temp_exists = true;
  if (!WriteFileSynced(temp, data, n, error)) return SpoolStatus::kIo;
  if (::link(temp.c_str(), ready.c_str()) != 0) {
    if (errno == EEXIST) {
      // Lost the race: validate the winner exactly like the retry path.
      ::unlink(temp.c_str());
      temp_exists = false;
      std::string id, sum;
      uint64_t created = 0, records = 0;
      const std::string name = fs::path(ready).filename().string();
      if (!ParseReadyName(name, &id, &created, &records, &sum) ||
          id != pack_id || created != created_at_ns ||
          records != record_count || sum != checksum ||
          fs::file_size(ready, ec) != n ||
          Sha256HexFile(ready, error) != checksum) {
        if (error && error->empty()) {
          *error = "spool contains different content: " + object_key;
        }
        return SpoolStatus::kConflict;
      }
      bytes_ += n;
      ++entries_;
      peak_bytes_ = std::max(peak_bytes_, bytes_);
      ++generation_;
      out->pack_id = pack_id;
      out->created_at_ns = created_at_ns;
      out->record_count = record_count;
      out->checksum = checksum;
      out->object_key = upload_key;
      out->path = ready;
      out->object_bytes = n;
      if (!FsyncChain(root_, dir, error)) return SpoolStatus::kIo;
      return SpoolStatus::kOk;
    }
    if (error) *error = "cannot link ready file: " + std::string(strerror(errno));
    ::unlink(temp.c_str());
    return SpoolStatus::kIo;
  }
  // Account the moment the link exists, before anything else can fail.
  bytes_ += n;
  ++entries_;
  peak_bytes_ = std::max(peak_bytes_, bytes_);
  ++generation_;
  ::unlink(temp.c_str());
  temp_exists = false;
  if (!FsyncChain(root_, dir, error)) return SpoolStatus::kIo;
  (void)temp_exists;
  out->pack_id = pack_id;
  out->created_at_ns = created_at_ns;
  out->record_count = record_count;
  out->checksum = checksum;
  out->object_key = upload_key;
  out->path = ready;
  out->object_bytes = n;
  return SpoolStatus::kOk;
}

SpoolStatus Spool::Recover(std::vector<StagedPack>* out, std::string* error) {
  out->clear();
  // Single pass under the lock: this driver is test-scoped and the uploader
  // runs recovery once at startup. (The Python optimistic multi-pass exists
  // because its recovery re-hashes without the lock; the native re-hash is
  // fast enough that stalling staging once per startup is the smaller cost.)
  std::lock_guard<std::mutex> lock(mutex_);
  std::error_code ec;
  std::vector<std::string> readies;
  for (const auto& entry :
       fs::recursive_directory_iterator(root_, ec)) {
    if (ec) break;
    if (!entry.is_regular_file()) continue;
    const std::string path = entry.path().string();
    const std::string name = entry.path().filename().string();
    if (HasSuffix(name, kOpenSuffix)) {
      ::unlink(path.c_str());
      FsyncDir(entry.path().parent_path().string(), nullptr);
      continue;
    }
    if (HasSuffix(name, kReadySuffix)) {
      readies.push_back(path);
    }
  }
  std::sort(readies.begin(), readies.end());
  uint64_t bytes = 0;
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
  bytes_ = bytes;
  entries_ = out->size();
  peak_bytes_ = std::max(peak_bytes_, bytes_);
  ++generation_;
  (void)error;
  return SpoolStatus::kOk;
}

SpoolStatus Spool::Remove(const StagedPack& staged, std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::error_code ec;
  if (!fs::exists(staged.path, ec)) {
    if (bytes_ >= staged.object_bytes) bytes_ -= staged.object_bytes;
    if (entries_ > 0) --entries_;
    ++generation_;
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
  ::unlink(staged.path.c_str());
  if (bytes_ >= staged.object_bytes) bytes_ -= staged.object_bytes;
  if (entries_ > 0) --entries_;
  ++generation_;
  FsyncDir(fs::path(staged.path).parent_path().string(), nullptr);
  return SpoolStatus::kOk;
}

SpoolSnapshot Spool::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  SpoolSnapshot snapshot;
  snapshot.entries = entries_;
  snapshot.bytes = bytes_;
  snapshot.peak_bytes = peak_bytes_;
  snapshot.max_bytes = max_bytes_;
  return snapshot;
}

}  // namespace dmi_store
