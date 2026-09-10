// Hold a real Stage after opening its temp file, while an uploader scans.
#include "store/spool.h"

#include <openssl/sha.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>

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
  if (fd >= 0 && (flags & O_CREAT) && std::strstr(path, ".open")) {
    std::cout << "OPEN" << std::endl;
    std::string release;
    std::getline(std::cin, release);
  }
  return fd;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  dmi_store::Spool spool;
  std::string error;
  if (dmi_store::Spool::Open({argv[1], 5000}, &spool, &error) !=
      dmi_store::SpoolStatus::kOk) return 3;
  const std::vector<uint8_t> data(1000, 42);
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(data.data(), data.size(), digest);
  char checksum[65];
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
    std::snprintf(checksum + 2 * i, 3, "%02x", digest[i]);
  }
  const std::string id = "018f0000-0000-7000-8000-000000000001";
  dmi_store::StagedPack staged;
  const auto status = spool.Stage(
      id, 1700000000000000000ull, 1, checksum, id + ".dmi-pack",
      data.data(), data.size(), &staged, &error);
  std::cout << dmi_store::SpoolStatusName(status) << ": " << error << '\n';
  return status == dmi_store::SpoolStatus::kOk ? 0 : 1;
}
