#include "spool.h"

#include <openssl/sha.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/stat.h>
#if defined(__linux__)
#include <sys/vfs.h>
#else
// statfs(2) with f_fstypename: macOS and the BSDs.
#include <sys/mount.h>
#include <sys/param.h>
#endif
#include <unistd.h>

namespace dmi_store {
namespace fs = std::filesystem;

namespace {

constexpr const char* kReadySuffix = ".dmi-pack.ready";
constexpr const char* kOpenSuffix = ".open";
constexpr const char* kOwnerLockFile = ".owner.lock";
// SpoolOwnerLock::Acquire builds a new directory as
// <parent>/.<name>.<8 hex>.creating and renames it into place.
constexpr const char* kClaimStagingSuffix = ".creating";
// <root>/_refs/: the upload handoff's ref files (plan section 2.4). Not a
// legal object-key component (those start with an alphanumeric), so no pack
// is ever staged under it.
constexpr const char* kRefsDirectory = "_refs";

// statfs(2) f_type values of network filesystems, whose flock does not
// keep out a process on another node -- or is not a place a node-local
// spool can be (linux/magic.h has NFS, SMB2, CIFS, FUSE, 9p and both AFS
// values; the others are their own). BeeGFS keeps flock client-local
// unless tuneUseGlobalFileLocks is set, and GPFS (IBM Storage Scale) keeps
// it node-local. FUSE covers network filesystems (sshfs, s3fs, gcsfuse,
// GlusterFS) and local ones alike, and f_type cannot tell them apart, so a
// local one needs the override. AFS is OpenAFS's and kAFS's.
constexpr uint32_t kNfsSuperMagic = 0x6969;
constexpr uint32_t kLustreSuperMagic = 0x0BD00BD0;
constexpr uint32_t kBeeGfsSuperMagic = 0x19830326;
constexpr uint32_t kCifsSuperMagic = 0xFF534D42;
constexpr uint32_t kSmb2SuperMagic = 0xFE534D42;
constexpr uint32_t kFuseSuperMagic = 0x65735546;
constexpr uint32_t kGpfsSuperMagic = 0x47504653;
constexpr uint32_t kV9fsMagic = 0x01021997;
constexpr uint32_t kAfsSuperMagic = 0x5346414F;
constexpr uint32_t kAfsFsMagic = 0x6B414653;
constexpr uint32_t kOrangeFsSuperMagic = 0x20030528;

std::atomic<int64_t> g_filesystem_type_for_testing{-1};

#if !defined(__linux__)
// Where statfs names the filesystem rather than giving Linux's magic: the
// same refusal, by f_fstypename (FreeBSD spells a FUSE mount
// "fusefs.<name>").
const char* SharedFilesystemTypeName(const char* name) {
  static const char* const kShared[][2] = {
      {"nfs", "NFS"},         {"smbfs", "SMB"},       {"afpfs", "AFP"},
      {"webdav", "WebDAV"},   {"lustre", "Lustre"},   {"macfuse", "FUSE"},
      {"osxfuse", "FUSE"},    {"fusefs", "FUSE"},     {"afs", "AFS"}};
  for (const auto& entry : kShared) {
    const size_t n = std::strlen(entry[0]);
    if (std::strncmp(name, entry[0], n) == 0 &&
        (name[n] == '\0' || name[n] == '.')) {
      return entry[1];
    }
  }
  return nullptr;
}
#endif
std::function<void(const std::string&)>& LockOpenHookForTesting() {
  static auto* hook = new std::function<void(const std::string&)>;
  return *hook;
}

// Whether a recursive walk of a spool root is at <root>/_refs, which no scan
// enters.
bool AtRefsDirectory(const fs::recursive_directory_iterator& it) {
  std::error_code ec;
  return it.depth() == 0 && it->path().filename() == kRefsDirectory &&
         it->is_directory(ec);
}

std::string Hostname() {
  char host[256] = {0};
  if (::gethostname(host, sizeof(host) - 1) != 0 || host[0] == '\0') {
    return "unknown-host";
  }
  return host;
}

std::string Errno(int error) { return std::strerror(error); }

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

void FsyncParent(const std::string& path) {
  FsyncDir(fs::path(path).parent_path().string(), nullptr);
}

// "<host> <pid>": whoever holds the lock records itself, so a refused
// process can say who holds the directory.
void WriteOwnerRecord(int fd) {
  const std::string record =
      Hostname() + " " + std::to_string(::getpid()) + "\n";
  if (::ftruncate(fd, 0) == 0) {
    (void)!::pwrite(fd, record.data(), record.size(), 0);
  }
}

void ReadOwnerRecord(int fd, SpoolOwner* owner) {
  char buffer[512];
  const ssize_t n = ::pread(fd, buffer, sizeof(buffer) - 1, 0);
  owner->host.clear();
  owner->pid = 0;
  if (n <= 0) return;
  std::string record(buffer, static_cast<size_t>(n));
  while (!record.empty() && std::isspace(static_cast<unsigned char>(
                                record.back()))) {
    record.pop_back();
  }
  const size_t space = record.rfind(' ');
  if (space == std::string::npos) {
    owner->host = record;
    return;
  }
  owner->host = record.substr(0, space);
  const std::string pid = record.substr(space + 1);
  if (!pid.empty() && pid.size() <= 18 &&
      std::all_of(pid.begin(), pid.end(),
                  [](char c) { return c >= '0' && c <= '9'; })) {
    owner->pid = std::strtoll(pid.c_str(), nullptr, 10);
  }
}

std::string OwnedMessage(const std::string& dir, const SpoolOwner& owner) {
  const std::string file = dir + "/" + kOwnerLockFile;
  if (owner.pid <= 0) {
    return "spool directory " + dir + " is owned by another process, which "
           "holds " + file + " and has not recorded itself yet; a spool "
           "directory has one owner process";
  }
  return "spool directory " + dir + " is owned by pid " +
         std::to_string(owner.pid) + " on host " + owner.host +
         " (it holds " + file + "); a spool directory has one owner process";
}

// Whether `fd` is still the file at `path`: a remover unlinks the lock file
// before it removes a drained directory, and a lock taken on the unlinked
// file guards nothing.
bool IsFileAt(int fd, const std::string& path) {
  struct stat by_fd{}, by_path{};
  return ::fstat(fd, &by_fd) == 0 && ::stat(path.c_str(), &by_path) == 0 &&
         by_fd.st_dev == by_path.st_dev && by_fd.st_ino == by_path.st_ino;
}

// A path that may not exist yet, absolute, with its existing prefix's
// symlinks resolved.
std::string CanonicalPath(const std::string& path, std::string* error) {
  std::error_code ec;
  const fs::path absolute = fs::absolute(path, ec);
  if (ec) {
    if (error) *error = "cannot resolve " + path + ": " + ec.message();
    return "";
  }
  const fs::path canonical = fs::weakly_canonical(absolute, ec);
  if (ec) {
    if (error) *error = "cannot resolve " + path + ": " + ec.message();
    return "";
  }
  std::string out = canonical.string();
  while (out.size() > 1 && out.back() == '/') out.pop_back();
  return out;
}

// A spool directory must not be nested under, or contain, another owned
// directory: Scan walks recursively, so the outer spool's Recover would
// sweep the inner one's .open files and upload its packs under the outer
// spool's keys. Run AFTER `dir`'s own lock is taken, so that of two takes
// racing on an outer directory and one inside it, at least one sees the
// other: each publishes its lock before it looks.
//   - An ancestor refuses while its lock is HELD. A lock file nobody holds
//     is a spool that was (every take leaves its file behind); whoever
//     takes that ancestor next meets this directory's lock file in its own
//     descendant walk, and is refused.
//   - A descendant refuses when it has a lock file at all, held or not: a
//     dead directory's packs are its successor's to adopt, not this
//     spool's to sweep and upload under its own keys. Except a claim's
//     staging copy (IsSpoolClaimStagingName) that nobody holds: a claim
//     killed before its rename, which holds nothing but its lock file. (One
//     that is held is a claim in progress, and refuses; one not yet locked
//     publishes after this walk, so its own ancestor check sees this lock.)
SpoolStatus CheckNotNested(const std::string& dir, std::string* error) {
  fs::path ancestor(dir);
  while (ancestor.has_parent_path() && ancestor.parent_path() != ancestor) {
    ancestor = ancestor.parent_path();
    std::error_code ec;
    SpoolOwner owner;
    if (fs::exists(ancestor / kOwnerLockFile, ec) &&
        ReadSpoolOwner(ancestor.string(), &owner)) {
      if (error) {
        *error = "spool directory " + dir + " is nested under the spool "
                 "directory " + ancestor.string() + ", which " +
                 (owner.pid > 0 ? "pid " + std::to_string(owner.pid) +
                                      " on host " + owner.host
                                : std::string("another owner")) +
                 " holds (" + kOwnerLockFile + "), and whose recovery "
                 "would sweep this one; use a directory outside it";
      }
      return SpoolStatus::kBadArgument;
    }
  }
  std::error_code ec;
  for (auto it = fs::recursive_directory_iterator(
           dir, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (it->path().filename() != kOwnerLockFile) continue;
    const fs::path owned = it->path().parent_path();
    if (owned == fs::path(dir)) continue;
    if (IsSpoolClaimStagingName(owned.filename().string()) &&
        !ReadSpoolOwner(owned.string(), nullptr)) {
      continue;
    }
    if (error) {
      *error = "spool directory " + dir + " contains the owned spool "
               "directory " + owned.string() + " (it has " + kOwnerLockFile +
               "), which this one's recovery would sweep; use a directory "
               "that does not contain it";
    }
    return SpoolStatus::kBadArgument;
  }
  return SpoolStatus::kOk;
}

}  // namespace

const char* OwnerLockName(OwnerLock mode) {
  return mode == OwnerLock::kHeldByCaller ? "held_by_caller" : "take";
}

bool ParseOwnerLock(const std::string& text, OwnerLock* mode) {
  if (text == "take") {
    *mode = OwnerLock::kTake;
    return true;
  }
  if (text == "held_by_caller") {
    *mode = OwnerLock::kHeldByCaller;
    return true;
  }
  return false;
}

const char* SharedFilesystemName(int64_t f_type) {
  switch (static_cast<uint32_t>(f_type)) {
    case kNfsSuperMagic: return "NFS";
    case kLustreSuperMagic: return "Lustre";
    case kBeeGfsSuperMagic: return "BeeGFS";
    case kCifsSuperMagic: return "CIFS";
    case kSmb2SuperMagic: return "SMB2";
    case kFuseSuperMagic: return "FUSE";
    case kGpfsSuperMagic: return "GPFS";
    case kV9fsMagic: return "9p";
    case kAfsSuperMagic: return "AFS";
    case kAfsFsMagic: return "AFS";
    case kOrangeFsSuperMagic: return "OrangeFS";
    default: return nullptr;
  }
}

void SetFilesystemTypeForTesting(int64_t f_type) {
  g_filesystem_type_for_testing.store(f_type);
}

void SetLockOpenHookForTesting(std::function<void(const std::string&)> hook) {
  LockOpenHookForTesting() = std::move(hook);
}

SpoolStatus CheckNodeLocal(const std::string& dir,
                           bool allow_shared_filesystem, std::string* error) {
  const int64_t f_type = g_filesystem_type_for_testing.load();
  const char* shared = nullptr;
  std::string seen;  // what statfs said, for the refusal
  const auto magic = [](int64_t value) {
    char text[48];
    std::snprintf(text, sizeof(text), "statfs f_type 0x%llx",
                  static_cast<unsigned long long>(value));
    return std::string(text);
  };
  if (f_type >= 0) {
    shared = SharedFilesystemName(f_type);
    seen = magic(f_type);
  } else {
    struct statfs info{};
    if (::statfs(dir.c_str(), &info) != 0) {
      if (error) *error = "cannot statfs " + dir + ": " + Errno(errno);
      return SpoolStatus::kIo;
    }
#if defined(__linux__)
    const int64_t type =
        static_cast<int64_t>(static_cast<uint32_t>(info.f_type));
    shared = SharedFilesystemName(type);
    seen = magic(type);
#else
    shared = SharedFilesystemTypeName(info.f_fstypename);
    seen = std::string("statfs f_fstypename ") + info.f_fstypename;
#endif
  }
  if (shared == nullptr || allow_shared_filesystem) return SpoolStatus::kOk;
  if (error) {
    *error = "spool directory " + dir + " is on " + shared + " (" + seen +
             "): a spool must be node-local, "
             "since its owner lock (flock) does not keep out a process on "
             "another node there. Use a local disk, or set "
             "allow_shared_filesystem if no process on another node can "
             "reach this directory" +
             (std::string(shared) == "FUSE"
                  ? " (a FUSE filesystem that is itself local, such as "
                    "fuse-overlayfs or ntfs-3g, is one)"
                  : std::string());
  }
  return SpoolStatus::kBadArgument;
}

bool ReadSpoolOwner(const std::string& dir, SpoolOwner* owner) {
  const std::string file = dir + "/" + kOwnerLockFile;
  const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  // A probe: if the lock can be taken nobody holds it, and it is let go at
  // once. (A take racing the probe retries, see LockInPlace.)
  if (::flock(fd, LOCK_EX | LOCK_NB) == 0) {
    ::flock(fd, LOCK_UN);
    ::close(fd);
    return false;
  }
  const bool held = errno == EWOULDBLOCK;
  if (held && owner != nullptr) ReadOwnerRecord(fd, owner);
  ::close(fd);
  return held;
}

bool IsSpoolClaimStagingName(const std::string& name) {
  // "." + <name> + "." + 8 hex + ".creating", <name> not empty.
  const size_t suffix = std::strlen(kClaimStagingSuffix);
  if (name.size() < 1 + 1 + 1 + 8 + suffix || name[0] != '.' ||
      !HasSuffix(name, kClaimStagingSuffix)) {
    return false;
  }
  const size_t dot = name.size() - suffix - 9;
  if (name[dot] != '.') return false;
  return std::all_of(name.begin() + dot + 1, name.end() - suffix, [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool SpoolOwnedByThisProcess(const std::string& dir) {
  const std::string file = dir + "/" + kOwnerLockFile;
  struct stat target{};
  if (::stat(file.c_str(), &target) != 0) return false;
  // /proc/self/fdinfo/<fd> lists the flocks each open file description
  // holds ("lock: 1: FLOCK  ADVISORY  WRITE ..."), so the kernel says
  // whether one of this process's descriptors on the file holds the lock --
  // a SpoolOwnerLock's, or one a Spool took with kTake. The record in the
  // file is only a fallback: it is written after the lock is taken, and a
  // pid says nothing across pid namespaces.
  DIR* fds = ::opendir("/proc/self/fd");
  if (fds == nullptr) {
    SpoolOwner owner;
    return ReadSpoolOwner(dir, &owner) && owner.pid == ::getpid() &&
           owner.host == Hostname();
  }
  const int listing = ::dirfd(fds);
  bool held = false;
  while (!held) {
    const dirent* entry = ::readdir(fds);
    if (entry == nullptr) break;
    char* end = nullptr;
    const long fd = std::strtol(entry->d_name, &end, 10);
    if (end == entry->d_name || *end != '\0' || fd == listing) continue;
    struct stat by_fd{};
    if (::fstat(static_cast<int>(fd), &by_fd) != 0 ||
        by_fd.st_dev != target.st_dev || by_fd.st_ino != target.st_ino) {
      continue;
    }
    const std::string info =
        std::string("/proc/self/fdinfo/") + entry->d_name;
    std::FILE* in = std::fopen(info.c_str(), "re");
    if (in == nullptr) continue;
    char line[512];
    while (std::fgets(line, sizeof(line), in) != nullptr) {
      if (std::strncmp(line, "lock:", 5) == 0 &&
          std::strstr(line, " FLOCK ") != nullptr &&
          std::strstr(line, " WRITE ") != nullptr) {
        held = true;
        break;
      }
    }
    std::fclose(in);
  }
  ::closedir(fds);
  return held;
}

namespace {

// Locks the lock file of an existing directory, creating the file if it has
// none. Retries a lock lost to a remover's unlink, and a refusal as brief as
// another process's ReadSpoolOwner probe.
SpoolStatus LockInPlace(const std::string& dir, int* fd_out,
                        std::string* error) {
  const std::string file = dir + "/" + kOwnerLockFile;
  for (int attempt = 0; attempt < 8; ++attempt) {
    const int fd = ::open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
      if (error) *error = "cannot open " + file + ": " + Errno(errno);
      return SpoolStatus::kIo;
    }
    if (LockOpenHookForTesting()) LockOpenHookForTesting()(file);
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      const int failure = errno;
      SpoolOwner owner;
      ReadOwnerRecord(fd, &owner);
      ::close(fd);
      if (failure != EWOULDBLOCK) {
        if (error) *error = "cannot lock " + file + ": " + Errno(failure);
        return SpoolStatus::kIo;
      }
      if (attempt < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        continue;
      }
      if (error) *error = OwnedMessage(dir, owner);
      return SpoolStatus::kOwned;
    }
    if (!IsFileAt(fd, file)) {
      ::close(fd);
      if (!fs::is_directory(dir)) {
        if (error) *error = "spool directory " + dir + " was removed while "
                            "it was being locked";
        return SpoolStatus::kIo;
      }
      continue;
    }
    WriteOwnerRecord(fd);
    *fd_out = fd;
    return SpoolStatus::kOk;
  }
  if (error) *error = "cannot lock " + file + ": it keeps being replaced";
  return SpoolStatus::kIo;
}

// Creates `dir` with its lock file already held: built under a hidden name
// beside it, then renamed into place, so a scan of the parent never meets
// the directory unowned (an adopter would otherwise take a brand-new
// sibling for a dead one). Falls back to LockInPlace if `dir` appears
// meanwhile.
SpoolStatus CreateLocked(const std::string& dir, int* fd_out,
                         std::string* error) {
  const fs::path target(dir);
  const std::string parent = target.parent_path().string();
  const std::string name = target.filename().string();
  std::random_device random;
  for (int attempt = 0; attempt < 8; ++attempt) {
    char suffix[16];
    std::snprintf(suffix, sizeof(suffix), "%08x",
                  static_cast<unsigned>(random()));
    // IsSpoolClaimStagingName's pattern.
    const std::string staging =
        parent + "/." + name + "." + suffix + kClaimStagingSuffix;
    if (::mkdir(staging.c_str(), 0755) != 0) {
      if (errno == EEXIST) continue;
      if (error) *error = "cannot create " + staging + ": " + Errno(errno);
      return SpoolStatus::kIo;
    }
    const std::string file = staging + "/" + kOwnerLockFile;
    const int fd =
        ::open(file.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0 || ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      const int failure = errno;
      if (fd >= 0) ::close(fd);
      ::unlink(file.c_str());
      ::rmdir(staging.c_str());
      if (error) *error = "cannot lock " + file + ": " + Errno(failure);
      return SpoolStatus::kIo;
    }
    WriteOwnerRecord(fd);
    ::fsync(fd);
    FsyncDir(staging, nullptr);
    // A rename that refuses an existing target: plain rename() silently
    // replaces an empty directory.
#if defined(RENAME_NOREPLACE)
    const int renamed = ::renameat2(AT_FDCWD, staging.c_str(), AT_FDCWD,
                                    dir.c_str(), RENAME_NOREPLACE);
#elif defined(__APPLE__) && defined(RENAME_EXCL)
    const int renamed =
        ::renamex_np(staging.c_str(), dir.c_str(), RENAME_EXCL);
#else
    // Neither: refuse a target that exists before renaming. The window
    // left is between the check and the rename, and only another claim of
    // the same fresh incarnation could fall into it.
    int renamed = -1;
    if (::access(dir.c_str(), F_OK) == 0) {
      errno = EEXIST;
    } else {
      renamed = ::rename(staging.c_str(), dir.c_str());
    }
#endif
    if (renamed != 0) {
      const int failure = errno;
      ::close(fd);
      ::unlink(file.c_str());
      ::rmdir(staging.c_str());
      if (failure == EEXIST || failure == ENOTEMPTY) {
        return LockInPlace(dir, fd_out, error);
      }
      if (error) {
        *error = "cannot create spool directory " + dir + ": " +
                 Errno(failure);
      }
      return SpoolStatus::kIo;
    }
    FsyncDir(parent, nullptr);
    *fd_out = fd;
    return SpoolStatus::kOk;
  }
  if (error) *error = "cannot create spool directory " + dir;
  return SpoolStatus::kIo;
}

}  // namespace

namespace {
// Every held SpoolOwnerLock in this binary. Leaked on purpose, so no
// static destructor runs while a lock is still registered. Each binary
// that compiles spool.cpp (the store and sink extensions, the drivers)
// keeps its own registry and its own fork handlers, for its own locks.
std::mutex& LockRegistryMutex() {
  static std::mutex* mutex = new std::mutex;
  return *mutex;
}
std::unordered_set<SpoolOwnerLock*>& LockRegistry() {
  static auto* registry = new std::unordered_set<SpoolOwnerLock*>;
  return *registry;
}
}  // namespace

void SpoolOwnerLock::BeforeFork() { LockRegistryMutex().lock(); }
void SpoolOwnerLock::AfterForkInParent() { LockRegistryMutex().unlock(); }

void SpoolOwnerLock::AfterForkInChild() {
  // Close, never LOCK_UN: an unlock on the shared description would drop
  // the parent's hold too, and closing one of its descriptors does not.
  for (SpoolOwnerLock* lock : LockRegistry()) {
    ::close(lock->fd_);
    lock->fd_ = -1;
    lock->dir_.clear();
  }
  LockRegistry().clear();
  LockRegistryMutex().unlock();
}

void SpoolOwnerLock::Track(SpoolOwnerLock* lock) {
  static std::once_flag handlers;
  std::call_once(handlers, [] {
    ::pthread_atfork(&SpoolOwnerLock::BeforeFork,
                     &SpoolOwnerLock::AfterForkInParent,
                     &SpoolOwnerLock::AfterForkInChild);
  });
  std::lock_guard<std::mutex> guard(LockRegistryMutex());
  LockRegistry().insert(lock);
}

void SpoolOwnerLock::Untrack(SpoolOwnerLock* lock) {
  std::lock_guard<std::mutex> guard(LockRegistryMutex());
  LockRegistry().erase(lock);
}

void SpoolOwnerLock::Hold(int fd, std::string dir) {
  fd_ = fd;
  dir_ = std::move(dir);
  Track(this);
}

SpoolOwnerLock::~SpoolOwnerLock() { Release(); }

SpoolOwnerLock::SpoolOwnerLock(SpoolOwnerLock&& other) noexcept {
  if (other.held()) {
    const int fd = other.fd_;
    std::string dir = std::move(other.dir_);
    Untrack(&other);
    other.fd_ = -1;
    other.dir_.clear();
    Hold(fd, std::move(dir));
  }
}

SpoolOwnerLock& SpoolOwnerLock::operator=(SpoolOwnerLock&& other) noexcept {
  if (this != &other) {
    Release();
    if (other.held()) {
      const int fd = other.fd_;
      std::string dir = std::move(other.dir_);
      Untrack(&other);
      other.fd_ = -1;
      other.dir_.clear();
      Hold(fd, std::move(dir));
    }
  }
  return *this;
}

void SpoolOwnerLock::Release() {
  if (fd_ >= 0) {
    Untrack(this);
    ::close(fd_);  // closing the last descriptor unlocks
  }
  fd_ = -1;
  dir_.clear();
}

SpoolStatus SpoolOwnerLock::Acquire(const std::string& dir,
                                    bool allow_shared_filesystem,
                                    SpoolOwnerLock* out, std::string* error) {
  out->Release();
  if (dir.empty()) {
    if (error) *error = "spool directory must not be empty";
    return SpoolStatus::kBadArgument;
  }
  const std::string canonical = CanonicalPath(dir, error);
  if (canonical.empty()) return SpoolStatus::kIo;
  std::error_code ec;
  const bool exists = fs::is_directory(canonical, ec);
  if (!exists && fs::exists(canonical, ec)) {
    if (error) *error = "spool directory " + canonical + " is not a directory";
    return SpoolStatus::kBadArgument;
  }
  const std::string parent = fs::path(canonical).parent_path().string();
  if (!exists) {
    fs::create_directories(parent, ec);
    if (ec) {
      if (error) *error = "cannot create " + parent + ": " + ec.message();
      return SpoolStatus::kIo;
    }
  }
  SpoolStatus status =
      CheckNodeLocal(exists ? canonical : parent, allow_shared_filesystem,
                     error);
  if (status != SpoolStatus::kOk) return status;
  const std::string lock_file = canonical + "/" + kOwnerLockFile;
  const bool had_lock_file = exists && fs::exists(lock_file, ec);
  int fd = -1;
  status = exists ? LockInPlace(canonical, &fd, error)
                  : CreateLocked(canonical, &fd, error);
  if (status != SpoolStatus::kOk) return status;
  out->Hold(fd, canonical);
  // Only now, with this lock published: see CheckNotNested.
  status = CheckNotNested(canonical, error);
  if (status != SpoolStatus::kOk) {
    // Leave nothing of this take behind: the directory it created (while
    // it is still empty), or the lock file it added to one that existed.
    if (!exists) {
      std::string ignored;
      out->ReleaseAndRemoveIfEmpty(&ignored);
    } else {
      if (!had_lock_file) ::unlink(lock_file.c_str());
      out->Release();
    }
    return status;
  }
  return SpoolStatus::kOk;
}

SpoolStatus SpoolOwnerLock::TryAdopt(const std::string& dir,
                                     SpoolOwnerLock* out,
                                     std::string* error) {
  out->Release();
  char resolved[4096];
  if (::realpath(dir.c_str(), resolved) == nullptr ||
      !fs::is_directory(resolved)) {
    if (error) *error = "no spool directory to adopt at " + dir;
    return SpoolStatus::kBadArgument;
  }
  int fd = -1;
  const SpoolStatus status = LockInPlace(resolved, &fd, error);
  if (status != SpoolStatus::kOk) return status;
  out->Hold(fd, resolved);
  return SpoolStatus::kOk;
}

bool SpoolOwnerLock::ReleaseAndRemoveIfEmpty(std::string* error) {
  if (!held()) return false;
  const std::string dir = dir_;
  const fs::path lock_file = fs::path(dir) / kOwnerLockFile;
  std::vector<fs::path> subdirectories;
  std::error_code ec;
  for (auto it = fs::recursive_directory_iterator(dir, ec);
       !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    std::error_code type_ec;
    if (it->is_directory(type_ec) && !it->is_symlink(type_ec)) {
      subdirectories.push_back(it->path());
    } else if (it->path() != lock_file) {
      Release();  // something is left: the directory stays as it is
      return false;
    }
  }
  if (ec) {
    if (error) *error = "cannot list " + dir + ": " + ec.message();
    Release();
    return false;
  }
  // Deepest first, so each is empty when its turn comes.
  std::sort(subdirectories.begin(), subdirectories.end(),
            [](const fs::path& a, const fs::path& b) {
              return a.string().size() > b.string().size();
            });
  for (const fs::path& subdirectory : subdirectories) {
    ::rmdir(subdirectory.c_str());
  }
  // Unlinked while held: a process that opens the file from here on creates
  // a new one (and the rmdir below then fails, leaving it the directory); one
  // that opened the old file first finds, once it locks it, that the file
  // is no longer at the path (IsFileAt), and lets it go.
  ::unlink(lock_file.c_str());
  const bool removed = ::rmdir(dir.c_str()) == 0;
  if (!removed && error) {
    *error = "cannot remove " + dir + ": " + Errno(errno);
  }
  if (removed) FsyncParent(dir);
  Release();
  return removed;
}

std::string SpoolCatalogKey(const SpoolDestination& destination) {
  const std::string text =
      destination.database + "/" + destination.table_prefix + "/" +
      destination.store_id + "\nclickhouse " + destination.clickhouse_host +
      ":" + std::to_string(destination.clickhouse_port) + "\ns3 " +
      destination.s3_endpoint + "/" + destination.s3_bucket;
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(),
         digest);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  for (int i = 0; i < 6; ++i) {
    out.push_back(kHex[digest[i] >> 4]);
    out.push_back(kHex[digest[i] & 0xF]);
  }
  return out;
}

namespace {
bool IsLowerHex(const std::string& text, size_t size) {
  return text.size() == size &&
         std::all_of(text.begin(), text.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
}  // namespace

bool IsSpoolCatalogKey(const std::string& name) { return IsLowerHex(name, 12); }

std::string SpoolRankDirectoryName(uint64_t producer_rank,
                                   const std::string& incarnation) {
  return "r" + std::to_string(producer_rank) + "-" + incarnation;
}

bool ParseSpoolRankDirectoryName(const std::string& name,
                                 uint64_t* producer_rank,
                                 std::string* incarnation) {
  if (name.size() < 4 || name[0] != 'r') return false;
  const size_t dash = name.find('-');
  if (dash == std::string::npos || dash < 2) return false;
  const std::string digits = name.substr(1, dash - 1);
  if (digits.size() > 19 || (digits.size() > 1 && digits[0] == '0') ||
      !std::all_of(digits.begin(), digits.end(),
                   [](char c) { return c >= '0' && c <= '9'; })) {
    return false;
  }
  const std::string tail = name.substr(dash + 1);
  if (!IsLowerHex(tail, 8)) return false;
  *producer_rank = std::strtoull(digits.c_str(), nullptr, 10);
  *incarnation = tail;
  return true;
}

std::string NewSpoolIncarnation() {
  std::random_device random;
  char out[16];
  std::snprintf(out, sizeof(out), "%08x", static_cast<unsigned>(random()));
  return out;
}

std::string SpoolRankDirectory(const std::string& base,
                               const SpoolDestination& destination,
                               uint64_t producer_rank,
                               const std::string& incarnation) {
  std::string root = base;
  while (root.size() > 1 && root.back() == '/') root.pop_back();
  return root + "/" + SpoolCatalogKey(destination) + "/" +
         SpoolRankDirectoryName(producer_rank, incarnation);
}

SpoolStatus Spool::Open(SpoolConfig config, Spool* out, std::string* error) {
  if (config.max_bytes == 0) {
    if (error) *error = "max_bytes must be positive";
    return SpoolStatus::kBadArgument;
  }
  if (config.root.empty()) {
    if (error) *error = "spool root must not be empty";
    return SpoolStatus::kBadArgument;
  }
  // A re-opened object gives up the directory it owned first.
  out->owner_lock_.Release();
  std::error_code ec;
  if (config.owner_lock == OwnerLock::kTake) {
    // Before anything reads the directory: the accounting walk below, and
    // above all Recover(), belong to its one owner. Creates the root.
    const SpoolStatus locked = SpoolOwnerLock::Acquire(
        config.root, config.allow_shared_filesystem, &out->owner_lock_,
        error);
    if (locked != SpoolStatus::kOk) return locked;
  } else {
    // The caller took the lock, so the directory and its lock file exist.
    char held[4096];
    SpoolOwner owner;
    if (::realpath(config.root.c_str(), held) == nullptr ||
        !ReadSpoolOwner(held, &owner)) {
      if (error) {
        *error = "spool owner_lock=held_by_caller, but nothing holds " +
                 config.root + "/" + kOwnerLockFile +
                 ": take a SpoolOwnerLock on the directory first, or open "
                 "it with owner_lock=take";
      }
      return SpoolStatus::kBadArgument;
    }
    // Held, but by THIS process? "Someone holds it" passes exactly when
    // another live process owns the directory, and this Spool's Recover
    // would then delete that owner's in-flight .open files.
    if (!SpoolOwnedByThisProcess(held)) {
      if (error) {
        *error = "spool owner_lock=held_by_caller, but this process does "
                 "not hold the owner lock of " + std::string(held) + ": " +
                 OwnedMessage(held, owner) + ". held_by_caller is for a "
                 "second Spool in the process that holds the directory's "
                 "SpoolOwnerLock";
      }
      return SpoolStatus::kOwned;
    }
    const SpoolStatus local =
        CheckNodeLocal(held, config.allow_shared_filesystem, error);
    if (local != SpoolStatus::kOk) return local;
  }
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
  out->accounted_ready_.clear();
  out->inflight_temps_.clear();
  out->peak_bytes_ = 0;
  out->generation_ = 0;
  // Account pre-existing files exactly like the Python constructor: ready
  // files plus stale .open files both count until Recover() runs, and only
  // the ready PATHS are remembered (_accounted_ready = ready_bytes), so a
  // later retry of one of them is recognised as already counted.
  for (auto it = fs::recursive_directory_iterator(out->root_, ec);
       it != fs::recursive_directory_iterator(); ++it) {
    if (AtRefsDirectory(it)) {
      it.disable_recursion_pending();
      continue;
    }
    const fs::directory_entry& entry = *it;
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    const bool is_ready = HasSuffix(name, kReadySuffix);
    const bool is_open = HasSuffix(name, kOpenSuffix);
    if (!is_ready && !is_open) continue;
    out->committed_bytes_ += entry.file_size();
    if (is_ready) {
      ++out->committed_entries_;
      out->accounted_ready_.emplace(entry.path().string(), entry.file_size());
    }
  }
  out->peak_bytes_ = out->committed_bytes_;
  return SpoolStatus::kOk;
}

bool Spool::AccountReadyLocked(const std::string& path,
                               uint64_t object_bytes) {
  if (accounted_ready_.count(path) != 0) return false;
  accounted_ready_.emplace(path, object_bytes);
  committed_bytes_ += object_bytes;
  ++committed_entries_;
  peak_bytes_ = std::max(peak_bytes_, committed_bytes_ + reserved_bytes_);
  return true;
}

bool Spool::UnaccountReadyLocked(const std::string& path) {
  const auto found = accounted_ready_.find(path);
  if (found == accounted_ready_.end()) return false;
  // Subtract what was RECORDED for this path, not what the caller believes it
  // to be: the recorded value is the one that went into the aggregate.
  if (committed_bytes_ >= found->second) committed_bytes_ -= found->second;
  if (committed_entries_ > 0) --committed_entries_;
  accounted_ready_.erase(found);
  return true;
}

void Spool::ReconcileCommittedLocked() {
  // The committed counter is only as fresh as the Remove calls THIS Spool
  // object has seen. An uploader running through a second Spool object on
  // the same root (or another process) removes ready files and updates its
  // OWN counter -- this object's never learns about the released capacity,
  // so staging eventually refuses on a directory that is actually empty
  // (reproduced: sink with a 1500-byte limit, serial upload-and-remove
  // between two records). Reconcile the COMMITTED account from the
  // directory -- the durable truth for what is committed.
  //
  // The scan is authoritative for committed files only. The reservations
  // other stagers hold right now are not on disk (or are, as a temp file
  // this object already accounts for), so they are kept as they are and
  // added back on top: replacing a single combined counter with the scan
  // admitted a concurrent stage the reservation should have refused.
  uint64_t actual = 0;
  uint64_t ready_count = 0;
  std::unordered_map<std::string, uint64_t> seen_ready;
  std::error_code walk_ec;
  for (auto it = fs::recursive_directory_iterator(root_, walk_ec);
       it != fs::recursive_directory_iterator(); ++it) {
    if (AtRefsDirectory(it)) {
      it.disable_recursion_pending();
      continue;
    }
    const fs::directory_entry& entry = *it;
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    if (HasSuffix(name, kReadySuffix)) {
      actual += entry.file_size();
      ++ready_count;
      seen_ready.emplace(entry.path().string(), entry.file_size());
    } else if (HasSuffix(name, kOpenSuffix) &&
               inflight_temps_.count(entry.path().string()) == 0) {
      // Someone else's in-progress write (another process, or a stale
      // leftover Recover() has not swept yet): counts, as it does at
      // Open(). This object's own temps are reservations.
      actual += entry.file_size();
    }
  }
  committed_bytes_ = actual;
  committed_entries_ = ready_count;
  // The path ledger is rebuilt with the aggregate it describes, so the two
  // never disagree about which files the committed account holds.
  accounted_ready_ = std::move(seen_ready);
  // The scan can raise the committed total (files another object wrote), and
  // peak_bytes_ must never read below what the account holds right now.
  peak_bytes_ = std::max(peak_bytes_, committed_bytes_ + reserved_bytes_);
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
        ReconcileCommittedLocked();
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
    AccountReadyLocked(ready, n);
    ++generation_;
  };
  // The retry and EEXIST-loser paths both end on a ready file that already
  // exists. Whether it costs anything depends on whether THIS object has
  // counted that path before, which only the path ledger can answer. A path
  // it has NOT counted is judged against the durable truth plus in-flight
  // reservations -- the same capacity decision a fresh stage makes, except
  // that the reconciliation scan counts the file's bytes rather than
  // charging them on top of the committed total. Without the decision, a
  // retry could be added on top of an in-flight reservation: with a
  // 1000-byte reservation paused and a 1000-byte ready file created by a
  // second Spool object, admitting the retry put committed + reserved at
  // 2000 under a 1500 cap and both stages completed.
  //
  // The verdict must not depend on which call ran the scan. An
  // already-counted path skips the scan, so it re-checks the same sum
  // instead of returning kOk unconditionally: otherwise the first retry
  // (which reconciles, sees the overage and refuses) and the second (which
  // finds the path ledgered) would disagree about identical state.
  auto account_existing = [&]() -> SpoolStatus {
    std::lock_guard<std::mutex> lock(mutex_);
    if (accounted_ready_.count(ready) == 0) ReconcileCommittedLocked();
    // Only an IN-FLIGHT reservation justifies refusing a file that is already
    // durably on disk. With reserved_bytes_ == 0 there is nothing the refusal
    // protects: the bytes are written, so kFull reclaims none of them and
    // merely withholds the acknowledgement of a pack that is present. The
    // oracle is the contract here and it agrees -- spool.py:116-130 takes
    // `if ready.exists(): _existing -> _account_ready_locked -> return staged`
    // and never consults max_bytes. Gating on reserved_bytes_ keeps the case
    // below refused exactly as before (a paused reservation plus a foreign
    // ready file -- a state Python cannot reach at all, since it holds its
    // lock across the whole of stage()), while a serial retry under a lowered
    // cap is admitted the way the reference admits it.
    if (reserved_bytes_ > 0 && committed_bytes_ + reserved_bytes_ > max_bytes_) {
      if (error) {
        *error = "spool byte limit exceeded: " +
                 std::to_string(committed_bytes_ + reserved_bytes_) + " > " +
                 std::to_string(max_bytes_);
      }
      return SpoolStatus::kFull;
    }
    if (AccountReadyLocked(ready, n)) ++generation_;
    return SpoolStatus::kOk;
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
    // A retry of a file THIS object created adds no accounting; a retry of
    // one created by a second Spool object on the same root after this
    // one's Open() adds its full size, because nothing here has counted it.
    // Python makes exactly that distinction -- spool.py:124 runs the retry
    // through _account_ready_locked, which is a no-op only for a path
    // already in _accounted_ready. Adding nothing unconditionally judged the
    // cap against 0: with max 1500, a 1000-byte file staged through another
    // object and retried here still admitted another 1000.
    //
    // Every successful retry still closes the durability window itself with
    // a fresh fsync chain.
    const SpoolStatus accounted = account_existing();
    if (accounted != SpoolStatus::kOk) return accounted;
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
      // Lost the race: release this reservation, then validate the winner
      // exactly like the retry path. The winner's file is counted here only
      // if this object has not counted that path already -- the winner may
      // be another Spool object entirely, whose stage touched nothing in
      // this account. Python does the same (spool.py:159).
      ::unlink(temp.c_str());
      unreserve();
      if (!validate_ready()) {
        if (error) *error = "spool contains different content: " + object_key;
        return SpoolStatus::kConflict;
      }
      const SpoolStatus accounted = account_existing();
      if (accounted != SpoolStatus::kOk) return accounted;
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
  std::unordered_map<std::string, uint64_t> seen_ready;
  for (auto it = fs::recursive_directory_iterator(root_, ec);
       it != fs::recursive_directory_iterator(); ++it) {
    if (AtRefsDirectory(it)) {
      it.disable_recursion_pending();
      continue;
    }
    const fs::directory_entry& entry = *it;
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
    seen_ready.emplace(path, size);
  }
  // Recovery rebuilds the committed account only; a stage in flight on
  // another thread keeps its reservation. The path ledger is rebuilt with
  // it (_commit_recovery_locked does the same), so the surviving entries are
  // exactly the ones a later retry will recognise as already counted, and
  // the quarantined ones are simply absent.
  committed_bytes_ = bytes;
  committed_entries_ = out->size();
  accounted_ready_ = std::move(seen_ready);
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
      // The file is already gone -- removed through another Spool object, or
      // by an earlier call here. Drop it from this object's account only if
      // this object was counting that PATH: a removal RETRY finds nothing
      // recorded and costs nothing, so it cannot release another pack's
      // capacity (which a blind subtraction did).
      if (UnaccountReadyLocked(staged.path)) ++generation_;
      return SpoolStatus::kOk;
    }
    const std::string name = fs::path(staged.path).filename().string();
    std::string id, sum;
    uint64_t created = 0, records = 0;
    if (!ParseReadyName(name, &id, &created, &records, &sum) ||
        id != staged.pack_id || created != staged.created_at_ns ||
        records != staged.record_count || sum != staged.checksum ||
        fs::file_size(staged.path, ec) != staged.object_bytes) {
      if (ec == std::errc::no_such_file_or_directory) {
        // The file vanished between the exists() check and this one --
        // another Spool object removed it. Uncount the path if THIS object
        // was charging it, exactly like the already-missing branch, or the
        // ledger keeps bytes for a file that is gone and a later retry
        // treats a recreated path as already accounted.
        if (UnaccountReadyLocked(staged.path)) ++generation_;
        return SpoolStatus::kOk;
      }
      if (error) *error = "staged pack identity changed before removal";
      return SpoolStatus::kIntegrity;
    }
    if (::unlink(staged.path.c_str()) != 0) {
      if (errno == ENOENT) {
        // Same removal race, now between the checks above and the unlink:
        // release this object's charge for the path that another Spool
        // object just removed.
        if (UnaccountReadyLocked(staged.path)) ++generation_;
        return SpoolStatus::kOk;
      }
      if (error) *error = "cannot remove staged pack: " + std::string(strerror(errno));
      return SpoolStatus::kIo;
    }
    // Uncount by PATH: the bytes that leave the aggregate are the ones this
    // object recorded for it, and a path this object never counted (removed
    // on behalf of another Spool object) costs nothing, exactly as
    // _unaccount_ready_locked does. Uncounting by anything else would let a
    // path stay in the ledger after its file is gone, and the next stage of
    // the same pack would then be treated as already accounted.
    UnaccountReadyLocked(staged.path);
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
