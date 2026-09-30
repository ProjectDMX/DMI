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
//      lock goes with its holder, even one killed with SIGKILL, and a child
//      it forked without exec does not keep it.
//   4. Nesting: a directory under, or containing, a HELD one is refused --
//      also when two processes take the pair at once -- and a dead one
//      nested in a spool is left alone by all of its walks.
//   5. The node-local check refuses NFS, Lustre, BeeGFS, CIFS/SMB2, FUSE,
//      GPFS, 9p, AFS and OrangeFS by statfs f_type, unless explicitly
//      allowed (a test seam stands in for statfs).
//   6. Adoption's try-lock never creates a directory, and a released
//      directory that holds nothing but its lock file can be removed; an
//      adopter recovers a dead spool a pack at a time.
//   7. The directory layout of the plan's section 2.3.
//   8. The spool budget charges what dead sibling directories hold.
//
// Built and run by tests/test_native_spool_owner_lock_unit.py.

#include <fcntl.h>
#include <openssl/sha.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

// (2c) Where the kernel lists no flocks in /proc/self/fdinfo -- gVisor's
// procfs prints only pos/flags/mnt_id, WSL1's none either -- the check
// fell back to the owner record only when /proc/self/fd could not be
// opened, so the process that really held the lock was refused its own
// held_by_caller Spools, naming its own pid, and no default-mode capture
// could start. There the record decides: this host and pid, or not.
void TestHeldByCallerWhereTheKernelListsNoFlocks() {
  const std::string root = FreshRoot("no-fdinfo-locks") + "/spool";
  dmi_store::SetFdinfoHidesLocksForTesting(true);
  SpoolOwnerLock lock;
  std::string error;
  CHECK(SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
        SpoolStatus::kOk);
  CHECK(dmi_store::SpoolOwnedByThisProcess(root));
  SpoolConfig config{root, 1 << 20};
  config.owner_lock = OwnerLock::kHeldByCaller;
  Spool spool;
  error.clear();
  CHECK(Spool::Open(config, &spool, &error) == SpoolStatus::kOk);
  CHECK(error.empty());

  // Beside another process's lock it is still refused, naming that holder.
  const std::string other = FreshRoot("no-fdinfo-locks-other") + "/spool";
  int ready[2];
  CHECK(::pipe(ready) == 0);
  const pid_t child = ::fork();
  if (child == 0) {
    ::close(ready[0]);
    SpoolOwnerLock held;
    std::string child_error;
    const bool ok = SpoolOwnerLock::Acquire(other, false, &held,
                                            &child_error) == SpoolStatus::kOk;
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
  CHECK(!dmi_store::SpoolOwnedByThisProcess(other));
  SpoolConfig beside{other, 1 << 20};
  beside.owner_lock = OwnerLock::kHeldByCaller;
  Spool refused;
  error.clear();
  CHECK(Spool::Open(beside, &refused, &error) == SpoolStatus::kOwned);
  CHECK(Contains(error, "pid " + std::to_string(child)));
  ::kill(child, SIGKILL);
  int status = 0;
  ::waitpid(child, &status, 0);
  dmi_store::SetFdinfoHidesLocksForTesting(false);
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

// (3b) flock binds to an open file description, which a child forked
// without exec shares. Such a child -- a fork-started worker -- used to keep
// the lock after its parent was SIGKILLed, so the dead parent's directory
// read as owned (naming the dead pid) and was never adopted. A forked child
// now lets go of its copies of every held lock at once, and the parent's
// hold is untouched.
void TestAForkedChildDoesNotKeepTheLockPastItsParent() {
  const std::string root = FreshRoot("fork-child") + "/spool";
  int ready[2];
  CHECK(::pipe(ready) == 0);
  const pid_t owner = ::fork();
  if (owner == 0) {
    ::close(ready[0]);
    SpoolOwnerLock lock;
    std::string error;
    if (SpoolOwnerLock::Acquire(root, false, &lock, &error) !=
        SpoolStatus::kOk) {
      ::_exit(3);
    }
    const pid_t worker = ::fork();  // no exec
    if (worker == 0) {
      // The child's own view: it holds nothing, and the lock is still held
      // (by the parent).
      const char mine = lock.held() ? 'H' : 'h';
      const char held = dmi_store::ReadSpoolOwner(root, nullptr) ? 'P' : 'p';
      if (::write(ready[1], &mine, 1) != 1 || ::write(ready[1], &held, 1) != 1) {
        ::_exit(3);
      }
      ::pause();  // outlives its parent until killed
      ::_exit(0);
    }
    char pid_text[32];
    const int n = std::snprintf(pid_text, sizeof(pid_text), "%d\n", worker);
    if (::write(ready[1], pid_text, n) != n) ::_exit(3);
    ::pause();  // until killed
    ::_exit(0);
  }
  ::close(ready[1]);
  // The worker's two bytes and the owner's "<worker pid>\n", in any order.
  std::string seen;
  char byte = 0;
  int flags = 0;
  while (seen.find('\n') == std::string::npos || flags < 2) {
    if (::read(ready[0], &byte, 1) != 1) break;
    seen.push_back(byte);
    if (byte == 'h' || byte == 'H' || byte == 'p' || byte == 'P') ++flags;
  }
  ::close(ready[0]);
  std::string digits;
  for (const char c : seen) {
    if (c >= '0' && c <= '9') digits.push_back(c);
  }
  const pid_t worker = static_cast<pid_t>(std::atoi(digits.c_str()));
  CHECK(worker > 0);
  CHECK(seen.find('h') != std::string::npos);  // the child holds nothing
  CHECK(seen.find('P') != std::string::npos);  // the parent still does
  dmi_store::SpoolOwner owner_record;
  CHECK(dmi_store::ReadSpoolOwner(root, &owner_record));
  CHECK(owner_record.pid == owner);

  ::kill(owner, SIGKILL);
  int status = 0;
  ::waitpid(owner, &status, 0);
  // The worker lives on, and the lock went with its parent.
  CHECK(::kill(worker, 0) == 0);
  CHECK(!dmi_store::ReadSpoolOwner(root, &owner_record));
  SpoolOwnerLock successor;
  std::string error;
  CHECK(SpoolOwnerLock::TryAdopt(root, &successor, &error) ==
        SpoolStatus::kOk);
  ::kill(worker, SIGKILL);
}

// (3c) A fork from another thread while a lock is being TAKEN: the child
// gets a copy of the lock file's descriptor between its open() and the
// flock -- the flock then locks the description both share. Registered
// with the fork handler only once the lock was held, that copy was never
// closed in the child, which kept the directory looking live after its
// owner died. Here the fork runs in that window, through the lock-open
// test seam. The worker says it has run before its owner is killed: the
// fork handler closes its copy only once the child is first scheduled,
// which on a loaded host can come tens of milliseconds after the owner
// is reaped, and until then the lock outlives its owner.
void TestAForkWhileALockIsTakenLeavesTheChildNothing() {
  const std::string root = FreshRoot("fork-window") + "/spool";
  fs::create_directories(root);
  int ready[2];
  CHECK(::pipe(ready) == 0);
  const pid_t owner = ::fork();
  if (owner == 0) {
    ::close(ready[0]);
    pid_t worker = -1;
    dmi_store::SetLockOpenHookForTesting([&](const std::string&) {
      if (worker >= 0) return;
      worker = ::fork();  // no exec
      if (worker == 0) {
        // Past fork(), so past the fork handler.
        const char ran = 'w';
        if (::write(ready[1], &ran, 1) != 1) ::_exit(3);
        ::pause();  // outlives its parent until killed
        ::_exit(0);
      }
    });
    SpoolOwnerLock lock;
    std::string error;
    const bool ok =
        SpoolOwnerLock::TryAdopt(root, &lock, &error) == SpoolStatus::kOk;
    dmi_store::SetLockOpenHookForTesting(nullptr);
    char text[32];
    const int n = std::snprintf(text, sizeof(text), "%c%d\n", ok ? 'k' : 'x',
                                static_cast<int>(worker));
    if (::write(ready[1], text, n) != n) ::_exit(3);
    ::pause();  // until killed
    ::_exit(0);
  }
  ::close(ready[1]);
  // The worker's 'w' and the owner's "k<worker pid>\n", in any order.
  std::string seen;
  char byte = 0;
  while (seen.find('\n') == std::string::npos ||
         seen.find('w') == std::string::npos) {
    if (::read(ready[0], &byte, 1) != 1) break;
    seen.push_back(byte);
  }
  ::close(ready[0]);
  CHECK(seen.find('w') != std::string::npos);  // the worker has run
  const size_t outcome = seen.find_first_of("kx");
  CHECK(outcome != std::string::npos && seen[outcome] == 'k');
  const pid_t worker = static_cast<pid_t>(std::atoi(
      outcome == std::string::npos ? "" : seen.c_str() + outcome + 1));
  CHECK(worker > 0);
  dmi_store::SpoolOwner record;
  CHECK(dmi_store::ReadSpoolOwner(root, &record));
  CHECK(record.pid == owner);

  ::kill(owner, SIGKILL);
  int status = 0;
  ::waitpid(owner, &status, 0);
  CHECK(worker > 0 && ::kill(worker, 0) == 0);  // the worker lives on
  // And the lock went with its owner.
  CHECK(!dmi_store::ReadSpoolOwner(root, nullptr));
  if (worker > 0) ::kill(worker, SIGKILL);
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
  // A refused take leaves nothing of its own behind: not the directory it
  // created, nor a lock file it added to one that existed.
  CHECK(!fs::exists(base + "/outer/inner"));
  CHECK(!fs::exists(base + "/other/.owner.lock"));
}

// (4b) An outer directory and one nested in it, taken at the same moment by
// two processes: each checks the other's lock only after publishing its
// own, so at most one of them wins. Checked first and locked second, both
// won most of the time (275 of 300 in the review's probe).
void TestAnOuterAndANestedTakeRacingNeverBothWin() {
  const std::string base = FreshRoot("nest-race");
  int both = 0;
  int outer_won = 0;
  int inner_won = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const std::string outer = base + "/t" + std::to_string(trial);
    const std::string inner = outer + "/0123456789ab/r0-0a1b2c3d";
    if (trial % 2 == 0) fs::create_directories(outer);
    int go[2], report[2], done[2];
    CHECK(::pipe(go) == 0 && ::pipe(report) == 0 && ::pipe(done) == 0);
    pid_t children[2];
    for (int side = 0; side < 2; ++side) {
      children[side] = ::fork();
      if (children[side] == 0) {
        ::close(go[1]);
        ::close(report[0]);
        ::close(done[1]);
        char byte = 0;
        (void)!::read(go[0], &byte, 1);  // EOF: the parent let both go
        SpoolOwnerLock lock;
        std::string error;
        const bool won =
            SpoolOwnerLock::Acquire(side == 0 ? outer : inner, false, &lock,
                                    &error) == SpoolStatus::kOk;
        byte = static_cast<char>(side == 0 ? (won ? 'O' : 'o')
                                           : (won ? 'I' : 'i'));
        if (::write(report[1], &byte, 1) != 1) ::_exit(3);
        (void)!::read(done[0], &byte, 1);  // hold it until both reported
        ::_exit(0);
      }
    }
    ::close(go[0]);
    ::close(report[1]);
    ::close(done[0]);
    ::close(go[1]);
    char results[2] = {0, 0};
    CHECK(::read(report[0], &results[0], 1) == 1);
    CHECK(::read(report[0], &results[1], 1) == 1);
    const std::string seen(results, 2);
    const bool o = seen.find('O') != std::string::npos;
    const bool i = seen.find('I') != std::string::npos;
    if (o && i) ++both;
    if (o) ++outer_won;
    if (i) ++inner_won;
    ::close(done[1]);
    ::close(report[0]);
    for (const pid_t child : children) {
      int status = 0;
      ::waitpid(child, &status, 0);
    }
  }
  if (both != 0) {
    std::cerr << "outer and nested both acquired in " << both
              << " of 200 trials\n";
  }
  CHECK(both == 0);
  CHECK(outer_won + inner_won > 0);
}

// (4c) Nesting refuses only while the other directory's lock is HELD. A
// lock file nobody holds is a spool that was -- every take leaves its file
// behind -- and a dead spool directory inside another is never the outer
// one's to sweep, count or upload under its own keys: every walk of a
// spool passes over a subdirectory with its own lock file. So a spool_root
// that a sink-only run once owned still takes rank directories, and a
// spool_root that a crashed default-mode run left a rank directory in
// still takes the sink-only or explicit-record_sink modes (the rollback),
// which leave that directory to adoption.
void TestAStaleLockFileAboveDoesNotRefuseANestedDirectory() {
  const std::string base = FreshRoot("stale-above");
  const std::string root = base + "/root";
  const std::string rank = root + "/0123456789ab/r0-0a1b2c3d";
  std::string error;
  {
    SpoolOwnerLock once;
    CHECK(SpoolOwnerLock::Acquire(root, false, &once, &error) ==
          SpoolStatus::kOk);
  }
  CHECK(fs::exists(root + "/.owner.lock"));
  SpoolOwnerLock inner;
  error.clear();
  CHECK(SpoolOwnerLock::Acquire(rank, false, &inner, &error) ==
        SpoolStatus::kOk);
  CHECK(error.empty());
  SpoolOwnerLock outer;
  CHECK(SpoolOwnerLock::Acquire(root, false, &outer, &error) ==
        SpoolStatus::kBadArgument);
  CHECK(Contains(error, "contains"));
  CHECK(Contains(error, rank));
  CHECK(Contains(error, "pid " + std::to_string(::getpid())));
  // The inner one stages a pack and has a stage in flight, then dies.
  {
    SpoolConfig config{rank, 1 << 20};
    config.owner_lock = OwnerLock::kHeldByCaller;
    Spool writer;
    CHECK(Spool::Open(config, &writer, &error) == SpoolStatus::kOk);
    CHECK(StageOne(writer, 1, &error) == SpoolStatus::kOk);
  }
  const std::string in_flight =
      rank + "/v1/.018f0000-0000-7000-8000-00000000beef.0badf00d.open";
  std::ofstream(in_flight) << "half a pack";
  inner.Release();
  // Its lock file stays, and nobody holds it: the outer take goes through,
  // and nothing of the outer spool touches the dead one.
  Spool flat;
  error.clear();
  CHECK(Spool::Open({root, 150}, &flat, &error) == SpoolStatus::kOk);
  CHECK(flat.Snapshot().bytes == 0);
  std::vector<dmi_store::StagedPack> staged;
  CHECK(flat.Recover(&staged, &error) == SpoolStatus::kOk);
  CHECK(staged.empty());
  CHECK(fs::exists(in_flight));
  CHECK(StageOne(flat, 2, &error) == SpoolStatus::kOk);  // 100 of 150
  CHECK(flat.ListPending(&staged, &error) == SpoolStatus::kOk);
  CHECK(staged.size() == 1 && staged[0].object_key.rfind("v1/", 0) == 0);
  size_t dead_packs = 0;
  for (const auto& entry : fs::recursive_directory_iterator(rank)) {
    if (entry.path().extension() == ".ready") ++dead_packs;
  }
  CHECK(dead_packs == 1);
  // And while the outer one holds the root, the rank directory cannot be
  // taken: an adopter would first have to wait for it.
  SpoolOwnerLock again;
  error.clear();
  CHECK(SpoolOwnerLock::Acquire(rank, false, &again, &error) ==
        SpoolStatus::kBadArgument);
  CHECK(Contains(error, "nested"));
}

// (4e) An adopter takes a dead directory's lock with TryAdopt, which runs
// no nesting check, then Recovers it. A live spool nested inside that dead
// directory -- a root put there, which the nesting rule admits under an
// unheld lock -- had its in-flight .open files swept by that Recover, and
// its packs listed under the outer directory's keys. Every walk passes
// over a subdirectory with its own lock file, so the adoption drains only
// the dead directory's own packs and leaves the directory in place.
void TestAnAdopterLeavesASpoolNestedInADeadOneAlone() {
  const std::string base = FreshRoot("nested-in-dead");
  const std::string dead = base + "/0123456789ab/r0-0000dead";
  const std::string nested = dead + "/inner";
  std::string error;
  {
    Spool gone;
    CHECK(Spool::Open({dead, 1 << 20}, &gone, &error) == SpoolStatus::kOk);
    CHECK(StageOne(gone, 1, &error) == SpoolStatus::kOk);
  }  // its owner died
  SpoolOwnerLock live;
  CHECK(SpoolOwnerLock::Acquire(nested, false, &live, &error) ==
        SpoolStatus::kOk);
  {
    SpoolConfig config{nested, 1 << 20};
    config.owner_lock = OwnerLock::kHeldByCaller;
    Spool writer;
    CHECK(Spool::Open(config, &writer, &error) == SpoolStatus::kOk);
    CHECK(StageOne(writer, 2, &error) == SpoolStatus::kOk);
  }
  const std::string in_flight =
      nested + "/v1/.018f0000-0000-7000-8000-00000000beef.0badf00d.open";
  std::ofstream(in_flight) << "half a pack";

  SpoolOwnerLock adopter;
  CHECK(SpoolOwnerLock::TryAdopt(dead, &adopter, &error) == SpoolStatus::kOk);
  SpoolConfig config{dead, 1 << 20};
  config.owner_lock = OwnerLock::kHeldByCaller;
  Spool adopted;
  CHECK(Spool::Open(config, &adopted, &error) == SpoolStatus::kOk);
  CHECK(adopted.Snapshot().bytes == 100);  // its own pack only
  std::vector<dmi_store::StagedPack> staged;
  CHECK(adopted.Recover(&staged, &error) == SpoolStatus::kOk);
  CHECK(fs::exists(in_flight));
  CHECK(staged.size() == 1);
  if (staged.size() == 1) {
    CHECK(staged[0].object_key.rfind("v1/", 0) == 0);
    CHECK(adopted.Remove(staged[0], &error) == SpoolStatus::kOk);
  }
  // Drained of its own, it still holds the live one: it stays.
  CHECK(!adopter.ReleaseAndRemoveIfEmpty(&error));
  CHECK(fs::exists(in_flight));
  CHECK(live.held());
}

// (4d) A claim killed between creating its directory's staging copy
// (.<name>.<rand>.creating, lock file inside) and renaming it into place
// leaves that copy behind. Nobody holds it, it holds no pack, and it must
// not refuse a take of the directories above it for good. A staging copy
// whose lock IS held is a claim in progress, and still refuses.
void TestAnUnheldClaimStagingDirectoryRefusesNothing() {
  const std::string base = FreshRoot("staging");
  const std::string key = base + "/root/0123456789ab";
  const std::string leftover = key + "/.r0-0a1b2c3d.0badf00d.creating";
  CHECK(dmi_store::IsSpoolClaimStagingName(".r0-0a1b2c3d.0badf00d.creating"));
  CHECK(!dmi_store::IsSpoolClaimStagingName("r0-0a1b2c3d"));
  CHECK(!dmi_store::IsSpoolClaimStagingName(".creating"));
  fs::create_directories(leftover);
  std::ofstream(leftover + "/.owner.lock") << "host 1\n";
  std::string error;
  {
    SpoolOwnerLock lock;
    CHECK(SpoolOwnerLock::Acquire(base + "/root", false, &lock, &error) ==
          SpoolStatus::kOk);
    CHECK(error.empty());
  }
  {
    SpoolOwnerLock lock;
    CHECK(SpoolOwnerLock::Acquire(key, false, &lock, &error) ==
          SpoolStatus::kOk);
    CHECK(error.empty());
  }
  // Held: a claim in the middle of creating its directory.
  const std::string live = base + "/other/0123456789ab";
  SpoolOwnerLock claiming;
  CHECK(SpoolOwnerLock::Acquire(live + "/.r1-0a1b2c3d.00c0ffee.creating",
                                false, &claiming, &error) ==
        SpoolStatus::kOk);
  SpoolOwnerLock outer;
  error.clear();
  CHECK(SpoolOwnerLock::Acquire(base + "/other", false, &outer, &error) ==
        SpoolStatus::kBadArgument);
  CHECK(Contains(error, "contains"));
}

// (5) The node-local check, through the test seam.
void TestSharedFilesystemsAreRefusedUnlessAllowed() {
  const std::string root = FreshRoot("statfs") + "/spool";
  // Each one's flock does not keep out a process on another node: NFS and
  // Lustre (the plan's two), BeeGFS (client-local unless
  // tuneUseGlobalFileLocks), CIFS/SMB2, FUSE, which cannot tell sshfs,
  // s3fs, gcsfuse or GlusterFS from a local filesystem, GPFS (IBM Storage
  // Scale, whose flock is node-local), 9p, AFS (OpenAFS and kAFS) and
  // OrangeFS -- network filesystems all, where a spool is never node-local.
  const std::vector<std::pair<int64_t, std::string>> shared = {
      {0x6969, "NFS"},        {0x0BD00BD0, "Lustre"},
      {0x19830326, "BeeGFS"}, {0xFF534D42, "CIFS"},
      {0xFE534D42, "SMB2"},   {0x65735546, "FUSE"},
      {0x47504653, "GPFS"},   {0x01021997, "9p"},
      {0x5346414F, "AFS"},    {0x6B414653, "AFS"},
      {0x20030528, "OrangeFS"}};
  for (const auto& [magic, name] : shared) {
    const char* named = dmi_store::SharedFilesystemName(magic);
    CHECK(named != nullptr && std::string(named) == name);
  }
  CHECK(dmi_store::SharedFilesystemName(0xEF53) == nullptr);  // ext4
  CHECK(dmi_store::SharedFilesystemName(0x58465342) == nullptr);  // xfs
  CHECK(dmi_store::SharedFilesystemName(0x794C7630) == nullptr);  // overlayfs
  CHECK(dmi_store::SharedFilesystemName(0x01021994) == nullptr);  // tmpfs
  CHECK(dmi_store::SharedFilesystemName(0x9123683E) == nullptr);  // btrfs
  CHECK(dmi_store::SharedFilesystemName(0x2FC12FC1) == nullptr);  // zfs

  std::string error;
  for (const auto& [magic, name] : shared) {
    dmi_store::SetFilesystemTypeForTesting(magic);
    SpoolOwnerLock lock;
    error.clear();
    CHECK(SpoolOwnerLock::Acquire(root, false, &lock, &error) ==
          SpoolStatus::kBadArgument);
    CHECK(Contains(error, " is on " + name + " "));
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

// (6b) A lock taken on a file a remover unlinked meanwhile guards nothing
// (ReleaseAndRemoveIfEmpty unlinks the lock file, then removes the
// directory), so it is let go and the file at the path is locked instead.
void TestALockOnAnUnlinkedFileIsTakenAgain() {
  const std::string dir = FreshRoot("unlinked") + "/spool";
  fs::create_directories(dir);
  std::ofstream(dir + "/.owner.lock") << "";
  int calls = 0;
  dmi_store::SetLockOpenHookForTesting([&calls](const std::string& file) {
    if (calls++ == 0) ::unlink(file.c_str());  // the remover's unlink
  });
  SpoolOwnerLock lock;
  std::string error;
  CHECK(SpoolOwnerLock::TryAdopt(dir, &lock, &error) == SpoolStatus::kOk);
  dmi_store::SetLockOpenHookForTesting(nullptr);
  CHECK(calls == 2);
  CHECK(lock.held());
  // What is held is the lock file at the path, which anyone else meets.
  CHECK(dmi_store::ReadSpoolOwner(dir, nullptr));
  SpoolOwnerLock rival;
  CHECK(SpoolOwnerLock::TryAdopt(dir, &rival, &error) == SpoolStatus::kOwned);
}

// (6c) Liveness is not the lock file's alone. Something other than its
// holder -- an age-based cleaner such as systemd-tmpfiles, a person -- can
// unlink <dir>/.owner.lock while the owner lives: the owner's descriptor is
// then on the unlinked file, and whoever opens the path next meets a new
// one that nobody holds. Judged by that file alone the live directory read
// as dead, an adopter took it, and its Recover swept the owner's in-flight
// .open files. The owner also locks the directory itself, which cannot be
// unlinked while it holds anything.
void TestAReplacedLockFileLeavesTheDirectoryOwned() {
  const std::string dir = FreshRoot("replaced") + "/spool";
  int ready[2];
  CHECK(::pipe(ready) == 0);
  const pid_t owner = ::fork();
  if (owner == 0) {
    ::close(ready[0]);
    SpoolOwnerLock lock;
    std::string error;
    const bool ok = SpoolOwnerLock::Acquire(dir, false, &lock, &error) ==
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

  CHECK(fs::remove(dir + "/.owner.lock"));
  CHECK(dmi_store::ReadSpoolOwner(dir, nullptr));  // still owned
  std::string error;
  SpoolOwnerLock adopter;
  CHECK(SpoolOwnerLock::TryAdopt(dir, &adopter, &error) ==
        SpoolStatus::kOwned);
  CHECK(!adopter.held());
  Spool spool;
  CHECK(Spool::Open({dir, 1 << 20}, &spool, &error) == SpoolStatus::kOwned);

  ::kill(owner, SIGKILL);
  int status = 0;
  ::waitpid(owner, &status, 0);
  CHECK(!dmi_store::ReadSpoolOwner(dir, nullptr));
  CHECK(SpoolOwnerLock::TryAdopt(dir, &adopter, &error) == SpoolStatus::kOk);

  // In the owner's own process too: its held_by_caller Spools still open.
  adopter.Release();
  SpoolOwnerLock mine;
  CHECK(SpoolOwnerLock::Acquire(dir, false, &mine, &error) ==
        SpoolStatus::kOk);
  CHECK(fs::remove(dir + "/.owner.lock"));
  CHECK(dmi_store::SpoolOwnedByThisProcess(dir));
  SpoolConfig beside{dir, 1 << 20};
  beside.owner_lock = OwnerLock::kHeldByCaller;
  Spool held;
  CHECK(Spool::Open(beside, &held, &error) == SpoolStatus::kOk);
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

  // The window itself: a watcher -- an adopter's scan, in effect -- lists
  // the parent over and over and probes each rank directory it has not yet
  // seen held, while this process creates many and keeps holding every
  // one, so their creation is the only moment one could read as unheld.
  // Each take is slowed where it has opened a lock file and not locked it
  // (the lock-open seam), so a directory there to be seen before its lock
  // would be seen. One made first and locked after gives such a scan a
  // dead-looking sibling, which an adopter would take, and remove from
  // under its claimer.
  const std::string parent = base + "/race";
  fs::create_directories(parent);
  int report[2], stop[2];
  CHECK(::pipe(report) == 0 && ::pipe(stop) == 0);
  const pid_t watcher = ::fork();
  if (watcher == 0) {
    ::close(report[0]);
    ::close(stop[1]);
    ::fcntl(stop[0], F_SETFL, O_NONBLOCK);
    std::set<std::string> held, unheld;
    char byte = 0;
    while (::read(stop[0], &byte, 1) < 0 && errno == EAGAIN) {
      std::error_code ec;
      for (fs::directory_iterator it(parent, ec), end; !ec && it != end;
           it.increment(ec)) {
        const std::string name = it->path().filename().string();
        uint64_t rank = 0;
        std::string incarnation;
        if (held.count(name) != 0 ||
            !dmi_store::ParseSpoolRankDirectoryName(name, &rank,
                                                    &incarnation)) {
          continue;
        }
        if (dmi_store::ReadSpoolOwner(it->path().string(), nullptr)) {
          held.insert(name);
        } else {
          unheld.insert(name);
        }
      }
    }
    const int count = static_cast<int>(unheld.size());
    if (::write(report[1], &count, sizeof(count)) != sizeof(count)) {
      ::_exit(3);
    }
    ::_exit(0);
  }
  ::close(report[1]);
  ::close(stop[0]);
  dmi_store::SetLockOpenHookForTesting([](const std::string&) {
    ::usleep(1000);
  });
  std::vector<SpoolOwnerLock> claims(100);
  int taken = 0;
  for (size_t i = 0; i < claims.size(); ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "/r0-%08zx", i);
    if (SpoolOwnerLock::Acquire(parent + name, false, &claims[i], &error) ==
        SpoolStatus::kOk) {
      ++taken;
    }
  }
  dmi_store::SetLockOpenHookForTesting(nullptr);
  ::close(stop[1]);  // the watcher stops at EOF
  int unheld = -1;
  CHECK(::read(report[0], &unheld, sizeof(unheld)) == sizeof(unheld));
  ::close(report[0]);
  int status = 0;
  ::waitpid(watcher, &status, 0);
  CHECK(taken == 100);
  if (unheld != 0) {
    std::cerr << "a scan met " << unheld << " unheld new directories\n";
  }
  CHECK(unheld == 0);
}

// (6e) An adopter recovers a dead spool a pack at a time. Recover() hashes
// every pack of what may be a large backlog in one call, and whatever
// waited for the adopter -- stop(), a flush() behind its cycle -- waited
// for all of it. BeginRecovery sweeps the dead owner's stale .open file and
// lists the ready packs, hashing none, the account left as it was; each
// ContinueRecovery validates the next one, quarantining a corrupt one at
// its turn, and the last rebuilds the account as Recover() does. Until
// then every other ready pack stays where it was: an adopter let go of
// midway leaves nothing lost.
void TestAnAdopterRecoversADeadSpoolAPackAtATime() {
  const std::string dead =
      FreshRoot("adopt-steps") + "/0123456789ab/r0-0000dead";
  std::string error;
  {
    Spool gone;
    CHECK(Spool::Open({dead, 1 << 20}, &gone, &error) == SpoolStatus::kOk);
    for (int n = 1; n <= 3; ++n) {
      CHECK(StageOne(gone, n, &error) == SpoolStatus::kOk);
    }
  }  // its owner died
  const std::string stale =
      dead + "/v1/.018f0000-0000-7000-8000-00000000dead.0badf00d.open";
  std::ofstream(stale) << "half a pack";
  const auto readies = [&dead] {
    std::set<std::string> found;
    for (const auto& entry : fs::recursive_directory_iterator(dead)) {
      const std::string name = entry.path().filename().string();
      if (name.size() > 6 && name.substr(name.size() - 6) == ".ready") {
        found.insert(entry.path().string());
      }
    }
    return found;
  };
  const std::set<std::string> staged_before = readies();
  CHECK(staged_before.size() == 3);
  // The second in listing order no longer matches its checksum.
  const std::string corrupt = *std::next(staged_before.begin());
  std::ofstream(corrupt, std::ios::binary | std::ios::trunc)
      << std::string(100, 'z');

  SpoolOwnerLock adopter;
  CHECK(SpoolOwnerLock::TryAdopt(dead, &adopter, &error) == SpoolStatus::kOk);
  SpoolConfig config{dead, 1 << 20};
  config.owner_lock = OwnerLock::kHeldByCaller;
  Spool adopted;
  CHECK(Spool::Open(config, &adopted, &error) == SpoolStatus::kOk);
  const uint64_t bytes_before = adopted.Snapshot().bytes;

  dmi_store::SpoolRecovery recovery;
  CHECK(adopted.BeginRecovery(&recovery, &error) == SpoolStatus::kOk);
  CHECK(!fs::exists(stale));
  CHECK((std::set<std::string>(recovery.listed.begin(),
                               recovery.listed.end()) == staged_before));
  CHECK(recovery.next == 0 && recovery.valid.empty());
  CHECK(readies() == staged_before);
  CHECK(adopted.Snapshot().bytes == bytes_before);

  CHECK(!adopted.ContinueRecovery(&recovery));  // the first
  CHECK(recovery.next == 1 && recovery.valid.size() == 1);
  CHECK(recovery.valid[0].path == *staged_before.begin());
  CHECK(!adopted.ContinueRecovery(&recovery));  // the corrupt one
  CHECK(recovery.next == 2 && recovery.valid.size() == 1);
  CHECK(!fs::exists(corrupt));
  CHECK(fs::exists(corrupt.substr(0, corrupt.size() - 6) + ".quarantined"));
  CHECK(readies().size() == 2);
  CHECK(adopted.Snapshot().bytes == bytes_before);  // not rebuilt yet

  CHECK(adopted.ContinueRecovery(&recovery));  // the last, and the account
  CHECK(recovery.valid.size() == 2);
  CHECK(recovery.valid[1].path == *staged_before.rbegin());
  CHECK(recovery.valid[1].object_key ==
        "v1/018f0000-0000-7000-8000-000000000003.dmi-pack");
  CHECK(adopted.Snapshot().bytes == 200);
  CHECK(adopted.Snapshot().entries == 2);
  CHECK(adopted.ContinueRecovery(&recovery));  // done stays done
  CHECK(recovery.valid.size() == 2);

  // Nothing listed: done at the first step.
  const std::string empty =
      FreshRoot("adopt-steps-empty") + "/0123456789ab/r0-00000e00";
  { Spool gone; CHECK(Spool::Open({empty, 1 << 20}, &gone, &error) ==
                      SpoolStatus::kOk); }
  SpoolOwnerLock empty_adopter;
  CHECK(SpoolOwnerLock::TryAdopt(empty, &empty_adopter, &error) ==
        SpoolStatus::kOk);
  SpoolConfig empty_config{empty, 1 << 20};
  empty_config.owner_lock = OwnerLock::kHeldByCaller;
  Spool empty_spool;
  CHECK(Spool::Open(empty_config, &empty_spool, &error) == SpoolStatus::kOk);
  dmi_store::SpoolRecovery nothing;
  CHECK(empty_spool.BeginRecovery(&nothing, &error) == SpoolStatus::kOk);
  CHECK(nothing.listed.empty());
  CHECK(empty_spool.ContinueRecovery(&nothing));
  CHECK(nothing.valid.empty());
}

// (8) The budget across incarnations. Every process start gets a fresh
// rank directory, so a spool that charged only its own directory let each
// crash-restart add a full max_bytes while uploads were blocked. With
// charge_dead_siblings, what the sibling rank directories hold counts
// against max_bytes too, while adoption can drain it: a dead one, or one
// this process's adoption holds -- not one another live process holds
// (that is its own budget), one this process holds for its own writing,
// or one an adopter left blocked.
void TestDeadSiblingsCountAgainstTheBudget() {
  const std::string base = FreshRoot("budget");
  std::string error;
  const auto stage_packs = [&](const std::string& dir, int first, int n) {
    Spool spool;
    CHECK(Spool::Open({dir, 1 << 20}, &spool, &error) == SpoolStatus::kOk);
    for (int i = first; i < first + n; ++i) {
      CHECK(StageOne(spool, i, &error) == SpoolStatus::kOk);
    }
  };  // the Spool, and with it its lock, goes: a dead incarnation

  // A dead incarnation left three 100-byte packs beside the new one.
  const std::string key = base + "/0123456789ab";
  stage_packs(key + "/r0-0000000a", 1, 3);
  SpoolConfig config{key + "/r0-0000000b", 450};
  config.charge_dead_siblings = true;
  Spool spool;
  CHECK(Spool::Open(config, &spool, &error) == SpoolStatus::kOk);
  CHECK(spool.Snapshot().sibling_bytes == 300);
  CHECK(StageOne(spool, 4, &error) == SpoolStatus::kOk);  // 300 + 100
  error.clear();
  CHECK(StageOne(spool, 5, &error) == SpoolStatus::kFull);  // 300 + 200
  CHECK(Contains(error, "300 bytes"));
  CHECK(Contains(error, "dead"));
  // As adoption drains the dead one, the capacity comes back.
  for (const auto& entry :
       fs::recursive_directory_iterator(key + "/r0-0000000a")) {
    if (entry.path().extension() == ".ready") fs::remove(entry.path());
  }
  CHECK(StageOne(spool, 5, &error) == SpoolStatus::kOk);
  CHECK(spool.Snapshot().sibling_bytes == 0);

  // Without the charge each incarnation had the whole budget to itself.
  const std::string uncharged = base + "/ba9876543210";
  stage_packs(uncharged + "/r0-0000000a", 1, 3);
  Spool alone;
  CHECK(Spool::Open({uncharged + "/r0-0000000b", 450}, &alone, &error) ==
        SpoolStatus::kOk);
  for (int i = 4; i < 8; ++i) {
    CHECK(StageOne(alone, i, &error) == SpoolStatus::kOk);
  }

  // A sibling another live process holds is its own budget.
  const std::string shared = base + "/aaaaaaaaaaaa";
  int ready[2];
  CHECK(::pipe(ready) == 0);
  const pid_t child = ::fork();
  if (child == 0) {
    ::close(ready[0]);
    Spool live;
    std::string child_error;
    bool ok = Spool::Open({shared + "/r1-0000000d", 1 << 20}, &live,
                          &child_error) == SpoolStatus::kOk;
    for (int i = 1; ok && i <= 3; ++i) {
      ok = StageOne(live, i, &child_error) == SpoolStatus::kOk;
    }
    const char byte = ok ? '1' : '0';
    if (::write(ready[1], &byte, 1) != 1) ::_exit(3);
    ::pause();
    ::_exit(0);
  }
  ::close(ready[1]);
  char byte = 0;
  CHECK(::read(ready[0], &byte, 1) == 1);
  CHECK(byte == '1');
  ::close(ready[0]);
  SpoolConfig beside{shared + "/r0-0000000e", 450};
  beside.charge_dead_siblings = true;
  Spool next;
  CHECK(Spool::Open(beside, &next, &error) == SpoolStatus::kOk);
  CHECK(next.Snapshot().sibling_bytes == 0);
  for (int i = 4; i < 8; ++i) {
    CHECK(StageOne(next, i, &error) == SpoolStatus::kOk);
  }
  ::kill(child, SIGKILL);
  int status = 0;
  ::waitpid(child, &status, 0);

  // Only what adoption can drain is charged. One THIS process holds for
  // its own writing -- an earlier engine's claim, kept owned while its
  // unsealed sink might still stage -- is drained by no adoption of this
  // process's (its service reads it as live), so charging it left every
  // later sink in the process that much less budget until exit.
  const std::string mixed = base + "/bbbbbbbbbbbb";
  const std::string sibling = mixed + "/r0-0000000f";
  SpoolOwnerLock held;
  CHECK(SpoolOwnerLock::Acquire(sibling, false, &held, &error) ==
        SpoolStatus::kOk);
  {
    SpoolConfig kept{sibling, 1 << 20};
    kept.owner_lock = OwnerLock::kHeldByCaller;
    Spool writer;
    CHECK(Spool::Open(kept, &writer, &error) == SpoolStatus::kOk);
    for (int i = 1; i <= 3; ++i) {
      CHECK(StageOne(writer, i, &error) == SpoolStatus::kOk);
    }
  }
  const auto charged = [&]() {
    SpoolConfig own{mixed + "/r0-00000010", 450};
    own.charge_dead_siblings = true;
    Spool mine;
    CHECK(Spool::Open(own, &mine, &error) == SpoolStatus::kOk);
    return mine.Snapshot().sibling_bytes;
  };
  CHECK(charged() == 0);
  {
    SpoolConfig own{mixed + "/r0-00000010", 450};
    own.charge_dead_siblings = true;
    Spool mine;
    CHECK(Spool::Open(own, &mine, &error) == SpoolStatus::kOk);
    for (int i = 4; i < 8; ++i) {
      CHECK(StageOne(mine, i, &error) == SpoolStatus::kOk);
    }
  }
  fs::remove_all(mixed + "/r0-00000010");

  // One this process's adoption holds -- a dead one its service drains --
  // still counts: the room comes back as the adoption drains it.
  held.Release();
  SpoolOwnerLock adopting;
  CHECK(SpoolOwnerLock::TryAdopt(sibling, &adopting, &error) ==
        SpoolStatus::kOk);
  dmi_store::SpoolOwner record;
  CHECK(dmi_store::ReadSpoolOwner(sibling, &record));
  CHECK(record.adopting && record.blocked.empty());
  CHECK(record.pid == ::getpid());
  CHECK(charged() == 300);

  // One an adopter left for good -- blocked, its lock let go -- is drained
  // by nobody here, and is not charged either; its lock file says why.
  CHECK(adopting.MarkBlocked("it holds a pack this service can never upload"));
  adopting.Release();
  CHECK(!dmi_store::ReadSpoolOwner(sibling, &record));
  CHECK(record.blocked == "it holds a pack this service can never upload");
  CHECK(!record.adopting);
  CHECK(charged() == 0);
  // A process that can adopt it after all takes it again, and the mark
  // goes with its take.
  CHECK(SpoolOwnerLock::TryAdopt(sibling, &adopting, &error) ==
        SpoolStatus::kOk);
  CHECK(dmi_store::ReadSpoolOwner(sibling, &record));
  CHECK(record.adopting && record.blocked.empty());
  CHECK(charged() == 300);
  adopting.Release();
  CHECK(charged() == 300);  // dead: still to adopt

  // One this process cannot take and empty at all -- another user's -- is
  // no adoption's here either.
  fs::permissions(sibling + "/.owner.lock", fs::perms::owner_read);
  fs::permissions(sibling, fs::perms::owner_read | fs::perms::owner_exec);
  if (::access(sibling.c_str(), W_OK) != 0) {  // not as root
    CHECK(charged() == 0);
  }
  fs::permissions(sibling, fs::perms::owner_all);
  fs::permissions(sibling + "/.owner.lock",
                  fs::perms::owner_read | fs::perms::owner_write);
  CHECK(charged() == 300);
}

// (7) The layout: <base>/<catalog_key>/r<rank>-<incarnation>/.
dmi_store::SpoolDestination Destination() {
  dmi_store::SpoolDestination destination;
  destination.clickhouse_host = "ch";
  destination.clickhouse_port = 8123;
  destination.database = "db";
  destination.table_prefix = "prefix";
  destination.s3_endpoint = "http://s3:9000";
  destination.s3_bucket = "bucket";
  destination.store_id = "s3";
  return destination;
}

void TestTheDirectoryLayout() {
  const std::string key = dmi_store::SpoolCatalogKey(Destination());
  CHECK(key == Sha256Hex("db/prefix/s3\nclickhouse ch:8123\n"
                         "s3 http://s3:9000/bucket").substr(0, 12));
  CHECK(dmi_store::IsSpoolCatalogKey(key));
  CHECK(!dmi_store::IsSpoolCatalogKey("0123456789aB"));
  CHECK(!dmi_store::IsSpoolCatalogKey("0123456789a"));
  // Every part of where the packs go is in the key: two deployments that
  // share a spool_root and the default names, but not a server, never
  // adopt each other's directories.
  std::set<std::string> keys{key};
  for (int field = 0; field < 7; ++field) {
    dmi_store::SpoolDestination other = Destination();
    switch (field) {
      case 0: other.clickhouse_host = "ch2"; break;
      case 1: other.clickhouse_port = 8124; break;
      case 2: other.database = "db2"; break;
      case 3: other.table_prefix = "prefix2"; break;
      case 4: other.s3_endpoint = "http://s3b:9000"; break;
      case 5: other.s3_bucket = "bucket2"; break;
      case 6: other.store_id = "s4"; break;
    }
    keys.insert(dmi_store::SpoolCatalogKey(other));
  }
  CHECK(keys.size() == 8);

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
  CHECK(dmi_store::SpoolRankDirectory("/b", Destination(), 2, "0a1b2c3d") ==
        "/b/" + key + "/r2-0a1b2c3d");
}

}  // namespace

int main() {
  TestTwoTakesInOneProcessRefuseEachOther();
  TestHeldByCallerOpensBesideTheHolder();
  TestHeldByCallerBesideAnotherProcessIsRefused();
  TestHeldByCallerWhereTheKernelListsNoFlocks();
  TestHeldByCallerWithoutAHolderIsRefused();
  TestTheLockGoesWithItsSpool();
  TestASecondProcessIsRefusedUntilTheHolderDies();
  TestAForkedChildDoesNotKeepTheLockPastItsParent();
  TestAForkWhileALockIsTakenLeavesTheChildNothing();
  TestNestedDirectoriesAreRefused();
  TestAnOuterAndANestedTakeRacingNeverBothWin();
  TestAStaleLockFileAboveDoesNotRefuseANestedDirectory();
  TestAnUnheldClaimStagingDirectoryRefusesNothing();
  TestAnAdopterLeavesASpoolNestedInADeadOneAlone();
  TestSharedFilesystemsAreRefusedUnlessAllowed();
  TestAdoptionLocksOnlyWhatExistsAndIsDead();
  TestALockOnAnUnlinkedFileIsTakenAgain();
  TestAReplacedLockFileLeavesTheDirectoryOwned();
  TestANewDirectoryAppearsWithItsLockHeld();
  TestAnAdopterRecoversADeadSpoolAPackAtATime();
  TestTheDirectoryLayout();
  TestDeadSiblingsCountAgainstTheBudget();
  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
