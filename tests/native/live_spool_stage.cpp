// Hold a real Stage after opening its temp file, while another process tries
// the same spool.
//
//   live_spool_stage <root> [packs]
//
// Opens <root> with the default owner lock (take) and stages `packs` packs
// (default 1). When the LAST one has created its temp file it prints OPEN
// and waits for a line on stdin, so the test can act -- or SIGKILL it --
// with a live writer and an in-flight .open file on disk.
#include "store/spool.h"

#include <openssl/sha.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>

namespace {
int g_pause_at = 1;
int g_opened = 0;
}  // namespace

extern "C" int __real_open(const char*, int, ...);
extern "C" int __wrap_open(const char* path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args);
  }
  const int fd = __real_open(path, flags, mode);
  if (fd >= 0 && (flags & O_CREAT) && std::strstr(path, ".open") &&
      ++g_opened == g_pause_at) {
    std::cout << "OPEN" << std::endl;
    std::string release;
    std::getline(std::cin, release);
  }
  return fd;
}

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) return 2;
  const int packs = argc == 3 ? std::atoi(argv[2]) : 1;
  if (packs < 1) return 2;
  g_pause_at = packs;
  dmi_store::Spool spool;
  std::string error;
  if (dmi_store::Spool::Open({argv[1], 50000}, &spool, &error) !=
      dmi_store::SpoolStatus::kOk) {
    std::cout << "open failed: " << error << std::endl;
    return 3;
  }
  for (int n = 1; n <= packs; ++n) {
    const std::vector<uint8_t> data(1000, static_cast<uint8_t>(41 + n));
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(data.data(), data.size(), digest);
    char checksum[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
      std::snprintf(checksum + 2 * i, 3, "%02x", digest[i]);
    }
    char id[40];
    std::snprintf(id, sizeof(id), "018f0000-0000-7000-8000-%012d", n);
    dmi_store::StagedPack staged;
    const auto status = spool.Stage(
        id, 1700000000000000000ull + n, 1, checksum,
        std::string(id) + ".dmi-pack", data.data(), data.size(), &staged,
        &error);
    std::cout << dmi_store::SpoolStatusName(status) << ": " << error << '\n';
    if (status != dmi_store::SpoolStatus::kOk) return 1;
  }
  return 0;
}
