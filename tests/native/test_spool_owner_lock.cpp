// B6: the spool owner lock, in process and across a fork.
//
//   1. Two Spool objects that both TAKE one directory refuse each other,
//      even in one process: flock binds to an open file description, not to
//      the process. That is why the engine holds one SpoolOwnerLock and both
//      of its Spools (sink and service) open with held_by_caller.
//   2. held_by_caller opens beside a holder in this process, and is
//      refused beside another process's holder or when nothing holds the
//      lock.
//   3. A second process is refused, told the holder's pid and host; the
//      lock goes with its holder, even one killed with SIGKILL.
//   4. Nesting: a directory under, or containing, an owned one is refused.
//   5. The node-local check refuses NFS and Lustre by statfs f_type, unless
//      explicitly allowed (a test seam stands in for statfs).
//   6. Adoption's try-lock never creates a directory, and a released
//      directory that holds nothing but its lock file can be removed.
//   7. The directory layout of the plan's section 2.3.
//
// Built and run by tests/test_native_spool_owner_lock_unit.py.

#include <openssl/sha.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "store/spool.h"

namespace fs = std::filesystem;
using dmi_store::OwnerLock;
using dmi_store::Spool;
using dmi_store::SpoolConfig;
using dmi_store::SpoolOwnerLock;
using dmi_store::SpoolStatus;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                  \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

bool Contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

std::string FreshRoot(const char* tag) {
  const char* base = std::getenv("SPOOL_TEST_ROOT");
  const std::string root =
      std::string(base != nullptr ? base : "/tmp") + "/owner-" + tag;
  fs::remove_all(root);
  fs::create_directories(root);
  return fs::canonical(root).string();
}

std::string Hostname() {
  char host[256] = {0};
  ::gethostname(host, sizeof(host) - 1);
  return host;
}

std::string Sha256Hex(const std::string& data) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(),
         digest);
  static const char* kHex = "0123456789abcdef";
  std::string out(64, '0');
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0xF];
  }
  return out;
}

SpoolStatus StageOne(Spool& spool, int n, std::string* error) {
  char id[40];
  std::snprintf(id, sizeof(id), "018f0000-0000-7000-8000-%012d", n);
  const std::string data(100, static_cast<char>('a' + n));
  dmi_store::StagedPack staged;
  return spool.Stage(id, 1700000000000000000ull + n, 1, Sha256Hex(data),
                     std::string("v1/") + id + ".dmi-pack",
                     reinterpret_cast<const uint8_t*>(data.data()),
                     data.size(), &staged, error);
}

// (1) The regression a per-Spool lock would cause in the engine.
void TestTwoTakesInOneProcessRefuseEachOther() {
  const std::string root = FreshRoot("two-takes") + "/spool";
  Spool service, sink;
  std::string error;
  CHECK(Spool::Open({root, 1 << 20}, &service, &error) == SpoolStatus::kOk);
  error.clear();
  CHECK(Spool::Open({root, 1 << 20}, &sink, &error) == SpoolStatus::kOwned);
  CHECK(Contains(error, "pid " + std::to_string(::getpid())));
  CHECK(Contains(error, Hostname()));
}

// (2) held_by_caller beside a holder -- the engine's shape: one lock, two
// Spools, both staging and listing.
void TestHeldByCallerOpensBesideTheHolder() {
  const std::string root = FreshRoot("held") + "/spool";
  SpoolOwnerLock lock;
  std::string error;
  CHECK(SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
        SpoolStatus::kOk);
  CHECK(lock.held());
  SpoolConfig config{root, 1 << 20};
  config.owner_lock = OwnerLock::kHeldByCaller;
  Spool service, sink;
  CHECK(Spool::Open(config, &service, &error) == SpoolStatus::kOk);
  CHECK(Spool::Open(config, &sink, &error) == SpoolStatus::kOk);
  CHECK(StageOne(sink, 1, &error) == SpoolStatus::kOk);
  std::vector<dmi_store::StagedPack> pending;
  CHECK(service.ListPending(&pending, &error) == SpoolStatus::kOk);
  CHECK(pending.size() == 1);
  // A take is still refused while the engine's lock is held.
  Spool rival;
  CHECK(Spool::Open({root, 1 << 20}, &rival, &error) == SpoolStatus::kOwned);
}

// (2b) held_by_caller is for a Spool in the process that holds the lock.
// Beside ANOTHER process's lock it is refused, naming that holder: were
// "something holds it" enough, any process could open a live writer's
// directory that way, and its Recover would delete the writer's .open file.
void TestHeldByCallerBesideAnotherProcessIsRefused() {
  const std::string root = FreshRoot("held-elsewhere") + "/spool";
  int ready[2];
  CHECK(::pipe(ready) == 0);
  const pid_t child = ::fork();
  if (child == 0) {
    ::close(ready[0]);
    SpoolOwnerLock lock;
    std::string error;
    const bool ok = SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
                    SpoolStatus::kOk;
    const char byte = ok ? '1' : '0';
    if (::write(ready[1], &byte, 1) != 1) ::_exit(3);
    ::pause();  // until killed
    ::_exit(0);
  }
  ::close(ready[1]);
  char byte = 0;
  CHECK(::read(ready[0], &byte, 1) == 1);
  CHECK(byte == '1');
  ::close(ready[0]);

  SpoolConfig config{root, 1 << 20};
  config.owner_lock = OwnerLock::kHeldByCaller;
  Spool spool;
  std::string error;
  CHECK(Spool::Open(config, &spool, &error) == SpoolStatus::kOwned);
  CHECK(Contains(error, "held_by_caller"));
  CHECK(Contains(error, "pid " + std::to_string(child)));
  CHECK(Contains(error, Hostname()));

  ::kill(child, SIGKILL);
  int status = 0;
  ::waitpid(child, &status, 0);
}

void TestHeldByCallerWithoutAHolderIsRefused() {
  const std::string root = FreshRoot("unheld") + "/spool";
  SpoolConfig config{root, 1 << 20};
  config.owner_lock = OwnerLock::kHeldByCaller;
  Spool spool;
  std::string error;
  CHECK(Spool::Open(config, &spool, &error) == SpoolStatus::kBadArgument);
  CHECK(Contains(error, "held_by_caller"));
  // A lock file nobody holds is refused too.
  {
    SpoolOwnerLock lock;
    CHECK(SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
          SpoolStatus::kOk);
  }
  error.clear();
  CHECK(Spool::Open(config, &spool, &error) == SpoolStatus::kBadArgument);
  CHECK(Contains(error, "held_by_caller"));
}

void TestTheLockGoesWithItsSpool() {
  const std::string root = FreshRoot("scope") + "/spool";
  std::string error;
  {
    Spool first;
    CHECK(Spool::Open({root, 1 << 20}, &first, &error) == SpoolStatus::kOk);
  }
  Spool second;
  CHECK(Spool::Open({root, 1 << 20}, &second, &error) == SpoolStatus::kOk);
}

// (3) Across a fork: the child takes the lock, the parent is refused and
// told who holds it; the lock goes with the child, even on SIGKILL.
void TestASecondProcessIsRefusedUntilTheHolderDies() {
  const std::string root = FreshRoot("fork") + "/spool";
  int ready[2];
  CHECK(::pipe(ready) == 0);
  const pid_t child = ::fork();
  if (child == 0) {
    ::close(ready[0]);
    SpoolOwnerLock lock;
    std::string error;
    const bool ok = SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
                    SpoolStatus::kOk;
    const char byte = ok ? '1' : '0';
    if (::write(ready[1], &byte, 1) != 1) ::_exit(3);
    ::pause();  // until killed
    ::_exit(0);
  }
  ::close(ready[1]);
  char byte = 0;
  CHECK(::read(ready[0], &byte, 1) == 1);
  CHECK(byte == '1');
  ::close(ready[0]);

  SpoolOwnerLock lock;
  std::string error;
  CHECK(SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
        SpoolStatus::kOwned);
  CHECK(Contains(error, "pid " + std::to_string(child)));
  CHECK(Contains(error, Hostname()));
  CHECK(Contains(error, root));
  dmi_store::SpoolOwner owner;
  CHECK(dmi_store::ReadSpoolOwner(root, &owner));
  CHECK(owner.pid == child);
  CHECK(owner.host == Hostname());
  Spool spool;
  CHECK(Spool::Open({root, 1 << 20}, &spool, &error) == SpoolStatus::kOwned);

  ::kill(child, SIGKILL);
  int status = 0;
  ::waitpid(child, &status, 0);
  CHECK(!dmi_store::ReadSpoolOwner(root, &owner));
  error.clear();
  CHECK(SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
        SpoolStatus::kOk);
  CHECK(dmi_store::ReadSpoolOwner(root, &owner));
  CHECK(owner.pid == ::getpid());
}

// (4) Nesting, both ways.
void TestNestedDirectoriesAreRefused() {
  const std::string base = FreshRoot("nested");
  std::string error;
  SpoolOwnerLock outer;
  CHECK(SpoolOwnerLock::Acquire(base + "/outer", false, &outer, &error) ==
        SpoolStatus::kOk);
  SpoolOwnerLock inner;
  CHECK(SpoolOwnerLock::Acquire(base + "/outer/inner", false, &inner,
                                &error) == SpoolStatus::kBadArgument);
  CHECK(Contains(error, "nested"));
  CHECK(Contains(error, base + "/outer"));

  SpoolOwnerLock deep;
  CHECK(SpoolOwnerLock::Acquire(base + "/other/a/b", false, &deep, &error) ==
        SpoolStatus::kOk);
  SpoolOwnerLock ancestor;
  error.clear();
  CHECK(SpoolOwnerLock::Acquire(base + "/other", false, &ancestor, &error) ==
        SpoolStatus::kBadArgument);
  CHECK(Contains(error, "contains"));
  CHECK(Contains(error, base + "/other/a/b"));
  Spool spool;
  CHECK(Spool::Open({base + "/other", 1 << 20}, &spool, &error) ==
        SpoolStatus::kBadArgument);
}

// (5) The node-local check, through the test seam.
void TestSharedFilesystemsAreRefusedUnlessAllowed() {
  const std::string root = FreshRoot("statfs") + "/spool";
  CHECK(std::string(dmi_store::SharedFilesystemName(0x6969)) == "NFS");
  CHECK(std::string(dmi_store::SharedFilesystemName(0x0BD00BD0)) ==
        "Lustre");
  CHECK(dmi_store::SharedFilesystemName(0xEF53) == nullptr);  // ext4
  CHECK(dmi_store::SharedFilesystemName(0x58465342) == nullptr);  // xfs

  std::string error;
  for (const int64_t magic : {int64_t{0x6969}, int64_t{0x0BD00BD0}}) {
    dmi_store::SetFilesystemTypeForTesting(magic);
    SpoolOwnerLock lock;
    error.clear();
    CHECK(SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
          SpoolStatus::kBadArgument);
    CHECK(Contains(error, magic == 0x6969 ? "NFS" : "Lustre"));
    CHECK(Contains(error, "node-local"));
    CHECK(!lock.held());
    Spool spool;
    error.clear();
    CHECK(Spool::Open({root, 1 << 20}, &spool, &error) ==
          SpoolStatus::kBadArgument);
    CHECK(Contains(error, "node-local"));
    // The explicit override.
    SpoolConfig allowed{root, 1 << 20};
    allowed.allow_shared_filesystem = true;
    CHECK(Spool::Open(allowed, &spool, &error) == SpoolStatus::kOk);
  }
  {
    dmi_store::SetFilesystemTypeForTesting(0x6969);
    SpoolOwnerLock lock;
    CHECK(SpoolOwnerLock::Acquire(root + "-allowed", true, &lock, &error) ==
          SpoolStatus::kOk);
  }
  dmi_store::SetFilesystemTypeForTesting(-1);
  SpoolOwnerLock lock;
  CHECK(SpoolOwnerLock::Acquire(root + "-real", false, &lock, &error) ==
        SpoolStatus::kOk);
}

// (6) Adoption's try-lock, and removing a drained directory.
void TestAdoptionLocksOnlyWhatExistsAndIsDead() {
  const std::string base = FreshRoot("adopt");
  std::string error;
  SpoolOwnerLock lock;
  CHECK(SpoolOwnerLock::TryAdopt(base + "/missing", &lock, &error) !=
        SpoolStatus::kOk);
  CHECK(!fs::exists(base + "/missing"));

  // A directory with no lock file at all: nobody owns it.
  fs::create_directories(base + "/bare/v1");
  CHECK(SpoolOwnerLock::TryAdopt(base + "/bare", &lock, &error) ==
        SpoolStatus::kOk);
  CHECK(lock.held());
  CHECK(lock.ReleaseAndRemoveIfEmpty(&error));
  CHECK(!lock.held());
  CHECK(!fs::exists(base + "/bare"));

  // A live owner is left alone.
  SpoolOwnerLock live;
  CHECK(SpoolOwnerLock::Acquire(base + "/live", false, &live, &error) ==
        SpoolStatus::kOk);
  SpoolOwnerLock adopter;
  CHECK(SpoolOwnerLock::TryAdopt(base + "/live", &adopter, &error) ==
        SpoolStatus::kOwned);
  CHECK(!adopter.held());

  // Anything but the lock file and empty directories keeps the directory.
  SpoolOwnerLock kept;
  CHECK(SpoolOwnerLock::Acquire(base + "/kept", false, &kept, &error) ==
        SpoolStatus::kOk);
  fs::create_directories(base + "/kept/v1/tenant=t");
  std::ofstream(base + "/kept/v1/tenant=t/x.quarantined") << "bytes";
  CHECK(!kept.ReleaseAndRemoveIfEmpty(&error));
  CHECK(!kept.held());
  CHECK(fs::exists(base + "/kept/v1/tenant=t/x.quarantined"));
  CHECK(fs::exists(base + "/kept/.owner.lock"));
}

void TestANewDirectoryAppearsWithItsLockHeld() {
  // Created beside its lock file and renamed into place, so no scan of the
  // parent can meet the directory before its owner holds it.
  const std::string base = FreshRoot("atomic");
  SpoolOwnerLock lock;
  std::string error;
  CHECK(SpoolOwnerLock::Acquire(base + "/r0-0a1b2c3d", false, &lock,
                                &error) == SpoolStatus::kOk);
  std::set<std::string> names;
  for (const auto& entry : fs::directory_iterator(base)) {
    names.insert(entry.path().filename().string());
  }
  CHECK(names == std::set<std::string>{"r0-0a1b2c3d"});
  CHECK(lock.directory() == base + "/r0-0a1b2c3d");
  CHECK(fs::exists(base + "/r0-0a1b2c3d/.owner.lock"));
}

// (7) The layout: <base>/<catalog_key>/r<rank>-<incarnation>/.
void TestTheDirectoryLayout() {
  const std::string key = dmi_store::SpoolCatalogKey("db", "prefix", "s3");
  CHECK(key == Sha256Hex("db/prefix/s3").substr(0, 12));
  CHECK(dmi_store::IsSpoolCatalogKey(key));
  CHECK(!dmi_store::IsSpoolCatalogKey("0123456789aB"));
  CHECK(!dmi_store::IsSpoolCatalogKey("0123456789a"));
  CHECK(dmi_store::SpoolCatalogKey("db", "prefix", "s3") !=
        dmi_store::SpoolCatalogKey("db", "prefix", "s4"));

  CHECK(dmi_store::SpoolRankDirectoryName(3, "0a1b2c3d") == "r3-0a1b2c3d");
  uint64_t rank = 0;
  std::string incarnation;
  CHECK(dmi_store::ParseSpoolRankDirectoryName("r12-deadbeef", &rank,
                                                &incarnation));
  CHECK(rank == 12 && incarnation == "deadbeef");
  for (const char* bad : {"r-1-deadbeef", "r1-DEADBEEF", "r1-deadbee",
                          "r1-deadbeef0", "rx-deadbeef", "1-deadbeef",
                          "r1deadbeef", ".r1-deadbeef", "r01-deadbeef"}) {
    CHECK(!dmi_store::ParseSpoolRankDirectoryName(bad, &rank, &incarnation));
  }
  std::set<std::string> seen;
  for (int i = 0; i < 64; ++i) {
    const std::string fresh = dmi_store::NewSpoolIncarnation();
    CHECK(dmi_store::ParseSpoolRankDirectoryName("r0-" + fresh, &rank,
                                                  &incarnation));
    seen.insert(fresh);
  }
  CHECK(seen.size() == 64);
  CHECK(dmi_store::SpoolRankDirectory("/b", "db", "prefix", "s3", 2,
                                      "0a1b2c3d") ==
        "/b/" + key + "/r2-0a1b2c3d");
}

}  // namespace

int main() {
  TestTwoTakesInOneProcessRefuseEachOther();
  TestHeldByCallerOpensBesideTheHolder();
  TestHeldByCallerBesideAnotherProcessIsRefused();
  TestHeldByCallerWithoutAHolderIsRefused();
  TestTheLockGoesWithItsSpool();
  TestASecondProcessIsRefusedUntilTheHolderDies();
  TestNestedDirectoriesAreRefused();
  TestSharedFilesystemsAreRefusedUnlessAllowed();
  TestAdoptionLocksOnlyWhatExistsAndIsDead();
  TestANewDirectoryAppearsWithItsLockHeld();
  TestTheDirectoryLayout();
  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
