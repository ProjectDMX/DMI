// Spool capacity accounting under the access patterns the counters have to
// survive at once:
//
//   1. A SECOND Spool object (an uploader in another thread or process) on
//      the same root removes a ready file. The first object's committed
//      counter never saw the Remove, so it has to reconcile from the
//      directory before refusing a stage the directory has room for.
//   2. While one stager holds a reservation but has written NOTHING yet,
//      another stager trips that reconciliation. The scan sees an empty
//      directory; it must not erase the first stager's reservation.
//   3. A ready file created by a SECOND Spool object after this one's
//      Open() is reached by this one's retry or EEXIST-loser path. This
//      object's committed account has never counted it, so it has to count
//      it now -- but exactly once, however many times that path is met.
//
// A single counter cannot satisfy (1) and (2) at once: replacing it with
// the scan fixes (1) and breaks (2) -- with max 1500, A reserved 1000, B's
// 1000 scanned an empty root, and both were admitted (2000 bytes on disk
// against a 1500 limit). Keeping committed bytes and in-flight reservations
// as separate accounts, and letting the scan overwrite only the first,
// satisfies both. Neither aggregate can decide (3) on its own, so the
// committed account also carries the set of ready PATHS it includes.
//
// Built and run by tests/test_native_spool_reservations.py.

#include <openssl/sha.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "store/spool.h"

namespace fs = std::filesystem;

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

std::string Sha256Hex(const std::vector<uint8_t>& data) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(data.data(), data.size(), digest);
  static const char* kHex = "0123456789abcdef";
  std::string out(64, '0');
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0xF];
  }
  return out;
}

std::string PackId(int n) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "018f0000-0000-7000-8000-%012d", n);
  return buf;
}

dmi_store::SpoolStatus StageBytes(dmi_store::Spool& spool, int id,
                                  size_t n, dmi_store::StagedPack* staged,
                                  std::string* error) {
  std::vector<uint8_t> data(n, static_cast<uint8_t>(id));
  const std::string pack_id = PackId(id);
  return spool.Stage(pack_id, 1700000000000000000ull + id, 1,
                     Sha256Hex(data), pack_id + ".dmi-pack", data.data(),
                     data.size(), staged, error);
}

uint64_t BytesOnDisk(const std::string& root) {
  uint64_t total = 0;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    if (name.size() >= 6 && name.compare(name.size() - 6, 6, ".ready") == 0) {
      total += entry.file_size();
    }
  }
  return total;
}

std::string FreshRoot(const char* tag) {
  const char* base = std::getenv("SPOOL_TEST_ROOT");
  const std::string root =
      std::string(base != nullptr ? base : "/tmp") + "/spool-" + tag;
  fs::remove_all(root);
  return root;
}

// (1) The serial uploader case: stage, remove through a second object,
// stage again on the first object. Must succeed, and the accounting must
// end at exactly one file.
void TestSerialRemoveThroughAnotherSpoolIsReconciled() {
  const std::string root = FreshRoot("serial");
  dmi_store::SpoolConfig config{root, 1500};
  dmi_store::Spool writer, uploader;
  std::string error;
  CHECK(dmi_store::Spool::Open(config, &writer, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(dmi_store::Spool::Open(config, &uploader, &error) ==
        dmi_store::SpoolStatus::kOk);

  dmi_store::StagedPack first;
  CHECK(StageBytes(writer, 1, 1000, &first, &error) ==
        dmi_store::SpoolStatus::kOk);

  std::vector<dmi_store::StagedPack> recovered;
  CHECK(uploader.Recover(&recovered, &error) == dmi_store::SpoolStatus::kOk);
  CHECK(recovered.size() == 1);
  CHECK(uploader.Remove(recovered[0], &error) == dmi_store::SpoolStatus::kOk);
  CHECK(BytesOnDisk(root) == 0);

  // The writer's own counter still says 1000; the directory says 0.
  dmi_store::StagedPack second;
  const dmi_store::SpoolStatus status =
      StageBytes(writer, 2, 1000, &second, &error);
  CHECK(status == dmi_store::SpoolStatus::kOk);
  if (status != dmi_store::SpoolStatus::kOk) std::cerr << error << "\n";
  CHECK(BytesOnDisk(root) == 1000);
  const dmi_store::SpoolSnapshot snap = writer.Snapshot();
  CHECK(snap.bytes == 1000);
  CHECK(snap.entries == 1);
}

// (2) The concurrent admission case: A holds a reservation and has written
// nothing; B's stage trips the reconciliation. B must be refused, A must
// complete, and 1000 bytes -- not 2000 -- must be on disk.
void TestReconciliationKeepsAnInflightReservation() {
  const std::string root = FreshRoot("concurrent");
  dmi_store::SpoolConfig config{root, 1500};
  dmi_store::Spool spool;
  std::string error;
  CHECK(dmi_store::Spool::Open(config, &spool, &error) ==
        dmi_store::SpoolStatus::kOk);

  std::mutex mutex;
  std::condition_variable cv;
  bool reserved = false, release = false;
  spool.SetStageHookForTesting([&] {
    std::unique_lock<std::mutex> lock(mutex);
    reserved = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release; });
  });

  dmi_store::StagedPack a_out;
  std::string a_error;
  dmi_store::SpoolStatus a_status = dmi_store::SpoolStatus::kIo;
  std::thread a([&] { a_status = StageBytes(spool, 1, 1000, &a_out, &a_error); });
  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return reserved; });
  }
  // A's reservation exists and nothing is on disk. Drop the hook so B runs
  // straight through, then B asks for 1000 more against a 1500 limit.
  spool.SetStageHookForTesting(nullptr);
  CHECK(BytesOnDisk(root) == 0);
  dmi_store::StagedPack b_out;
  std::string b_error;
  const dmi_store::SpoolStatus b_status =
      StageBytes(spool, 2, 1000, &b_out, &b_error);
  CHECK(b_status == dmi_store::SpoolStatus::kFull);
  if (b_status != dmi_store::SpoolStatus::kFull) {
    std::cerr << "B was admitted: " << dmi_store::SpoolStatusName(b_status)
              << " " << b_error << "\n";
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
    cv.notify_all();
  }
  a.join();
  CHECK(a_status == dmi_store::SpoolStatus::kOk);
  if (a_status != dmi_store::SpoolStatus::kOk) std::cerr << a_error << "\n";
  CHECK(BytesOnDisk(root) == 1000);
  const dmi_store::SpoolSnapshot snap = spool.Snapshot();
  CHECK(snap.bytes == 1000);
  CHECK(snap.entries == 1);

  // Once A has committed, the room B was refused IS available to a stage
  // that fits: 500 more against the 1500 limit.
  dmi_store::StagedPack c_out;
  CHECK(StageBytes(spool, 3, 500, &c_out, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(BytesOnDisk(root) == 1500);
  CHECK(spool.Snapshot().bytes == 1500);
}

// A refused stage and a lost link() race both release their reservation
// (the reserved account must return to zero, or the spool leaks capacity).
void TestARefusedStageLeavesNoReservationBehind() {
  const std::string root = FreshRoot("refused");
  dmi_store::SpoolConfig config{root, 1500};
  dmi_store::Spool spool;
  std::string error;
  CHECK(dmi_store::Spool::Open(config, &spool, &error) ==
        dmi_store::SpoolStatus::kOk);
  dmi_store::StagedPack out;
  CHECK(StageBytes(spool, 1, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(StageBytes(spool, 2, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kFull);
  CHECK(spool.Snapshot().bytes == 1000);
  // A retry of the SAME pack adds no accounting.
  CHECK(StageBytes(spool, 1, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(spool.Snapshot().bytes == 1000);
  CHECK(spool.Snapshot().entries == 1);
}

// (3) A ready file this object NEVER accounted for. The retry above is the
// easy half: the object had already counted that file, so adding nothing is
// right. When a SECOND Spool object on the same root creates the file after
// this one's Open(), the retry path meets a file worth 1000 bytes that this
// object's committed account has never seen -- and adding nothing there
// leaves the cap judged against 0.
//
// Python takes the retry and the EEXIST-loser paths through
// _account_ready_locked (spool.py:124 and :159), which is a no-op only when
// the path is already in _accounted_ready. Its measured answer: the retry
// snapshot reads 1000, and the next 1000-byte stage raises SpoolFullError
// ("2000 > 1500") with 1000 bytes on disk.
void TestARetryOfAnotherObjectsReadyFileIsAccounted() {
  const std::string root = FreshRoot("foreign-retry");
  dmi_store::SpoolConfig config{root, 1500};
  dmi_store::Spool writer, other;
  std::string error;
  CHECK(dmi_store::Spool::Open(config, &writer, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(dmi_store::Spool::Open(config, &other, &error) ==
        dmi_store::SpoolStatus::kOk);

  // `other` writes the ready file; `writer` has never seen it.
  dmi_store::StagedPack out;
  CHECK(StageBytes(other, 1, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(writer.Snapshot().bytes == 0);

  // The same pack through `writer`: the ready file is already there, so this
  // is the retry path. It succeeds -- the content matches -- and it has to
  // leave the 1000 bytes IN the writer's account.
  CHECK(StageBytes(writer, 1, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(writer.Snapshot().bytes == 1000);
  CHECK(writer.Snapshot().entries == 1);

  // So a second, different pack of 1000 does not fit under the 1500 cap.
  const dmi_store::SpoolStatus status =
      StageBytes(writer, 2, 1000, &out, &error);
  CHECK(status == dmi_store::SpoolStatus::kFull);
  if (status != dmi_store::SpoolStatus::kFull) {
    std::cerr << "second pack was admitted: "
              << dmi_store::SpoolStatusName(status) << "\n";
  }
  CHECK(BytesOnDisk(root) == 1000);

  // And a retry stays idempotent: running it twice more must not count the
  // same path again.
  CHECK(StageBytes(writer, 1, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(StageBytes(writer, 1, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(writer.Snapshot().bytes == 1000);
  CHECK(writer.Snapshot().entries == 1);
}

// The EEXIST-loser path reaches the same file by a different route: both
// objects write the same pack, and the one whose link() loses must still
// end up with those bytes in its account.
void TestAnEexistLoserAccountsForTheWinnersFile() {
  const std::string root = FreshRoot("eexist-loser");
  dmi_store::SpoolConfig config{root, 1500};
  dmi_store::Spool loser, winner;
  std::string error;
  CHECK(dmi_store::Spool::Open(config, &loser, &error) ==
        dmi_store::SpoolStatus::kOk);
  CHECK(dmi_store::Spool::Open(config, &winner, &error) ==
        dmi_store::SpoolStatus::kOk);

  // The loser reserves, and while it is paused the winner links the ready
  // file. The loser's own link() then fails with EEXIST -- the retry branch
  // at the top of Stage ran before the file existed, so this is the only way
  // to that code path.
  dmi_store::StagedPack winner_out;
  loser.SetStageHookForTesting([&] {
    std::string inner;
    CHECK(StageBytes(winner, 1, 1000, &winner_out, &inner) ==
          dmi_store::SpoolStatus::kOk);
  });
  dmi_store::StagedPack out;
  CHECK(StageBytes(loser, 1, 1000, &out, &error) ==
        dmi_store::SpoolStatus::kOk);
  loser.SetStageHookForTesting(nullptr);

  CHECK(BytesOnDisk(root) == 1000);
  CHECK(loser.Snapshot().bytes == 1000);
  CHECK(loser.Snapshot().entries == 1);
  const dmi_store::SpoolStatus status =
      StageBytes(loser, 2, 1000, &out, &error);
  CHECK(status == dmi_store::SpoolStatus::kFull);
  if (status != dmi_store::SpoolStatus::kFull) {
    std::cerr << "loser admitted a second pack: "
              << dmi_store::SpoolStatusName(status) << "\n";
  }
  CHECK(BytesOnDisk(root) == 1000);
}

void TestRepeatedRemovalDoesNotReleaseAnotherPacksCapacity() {
  dmi_store::Spool spool;
  std::string error;
  const std::string root = FreshRoot("remove-retry");
  CHECK(dmi_store::Spool::Open({root, 1500}, &spool, &error) ==
        dmi_store::SpoolStatus::kOk);
  dmi_store::StagedPack first, second, third;
  CHECK(StageBytes(spool, 1, 500, &first, &error) == dmi_store::SpoolStatus::kOk);
  CHECK(StageBytes(spool, 2, 1000, &second, &error) == dmi_store::SpoolStatus::kOk);
  CHECK(spool.Remove(first, &error) == dmi_store::SpoolStatus::kOk);
  CHECK(spool.Remove(first, &error) == dmi_store::SpoolStatus::kOk);
  CHECK(spool.Snapshot().bytes == 1000);
  CHECK(spool.Snapshot().entries == 1);
  CHECK(StageBytes(spool, 3, 1000, &third, &error) == dmi_store::SpoolStatus::kFull);
  CHECK(BytesOnDisk(root) == 1000);
}

}  // namespace

int main() {
  TestSerialRemoveThroughAnotherSpoolIsReconciled();
  TestReconciliationKeepsAnInflightReservation();
  TestARefusedStageLeavesNoReservationBehind();
  TestARetryOfAnotherObjectsReadyFileIsAccounted();
  TestAnEexistLoserAccountsForTheWinnersFile();
  TestRepeatedRemovalDoesNotReleaseAnotherPacksCapacity();
  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
