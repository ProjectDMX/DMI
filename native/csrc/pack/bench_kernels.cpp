// T0.3: native kernel ceiling for the capture pack path.
// Measures, on the bench corpus shape (10k records x 64 KiB, seed 17):
//   1. crc32: hardware-accelerated (PCLMULQDQ via -mpclmul) CRC32C per record
//   2. sha256: OpenSSL SHA-256 per pack (seal)
//   3. append: payload memcpy into a slab + 32B header per record
//   4. seal_footer: JSON footer build for 10k records
//   5. spool_write: single 640 MiB .ready file, write+fsync+rename (NVMe)
// Reports GiB/s per stage. Not a product artifact: scoping instrument.

#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#ifdef __x86_64__
#include <nmmintrin.h>
#endif

using Clock = std::chrono::steady_clock;

static double gib_per_s(uint64_t bytes, double seconds) {
  return static_cast<double>(bytes) / seconds / (1024.0 * 1024.0 * 1024.0);
}

#ifdef __x86_64__
// CRC32C (Castagnoli), reflected, via PCLMULQDQ-free bit-reflect trick using
// _mm_crc32_u64: this is the same hardware unit zlib's USE_PCLMUL path uses.
static uint32_t crc32c_hw(const uint8_t* data, size_t n, uint32_t crc = 0) {
  uint64_t value = crc;
  while (n >= 8) {
    value = _mm_crc32_u64(value, *reinterpret_cast<const uint64_t*>(data));
    data += 8;
    n -= 8;
  }
  if (n >= 4) {
    value = _mm_crc32_u32(static_cast<uint32_t>(value), *reinterpret_cast<const uint32_t*>(data));
    data += 4;
    n -= 4;
  }
  for (size_t i = 0; i < n; ++i) {
    value = _mm_crc32_u8(static_cast<uint32_t>(value), data[i]);
  }
  return static_cast<uint32_t>(value);
}
#endif

int main() {
  const size_t kRecords = 10'000;
  const size_t kPayload = 64 * 1024;
  const uint64_t kLogical = kRecords * kPayload;
  std::vector<std::vector<uint8_t>> payloads(64, std::vector<uint8_t>(kPayload));
  {
    uint64_t state = 0x9E3779B97F4A7C15ull; // fixed seed, same-shape corpus
    for (auto& p : payloads) {
      for (auto& b : p) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        b = static_cast<uint8_t>(state >> 33);
      }
    }
  }

  // 1. CRC32 per record (hardware, 8 interleaved chains)
  {
    auto started = Clock::now();
    uint64_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int trial = 0; trial < 3; ++trial) {
      for (const auto& p : payloads) {
        const uint64_t* d = reinterpret_cast<const uint64_t*>(p.data());
        size_t n = p.size() / 8, i = 0;
        for (; i + 8 <= n; i += 8) {
          v[0] = _mm_crc32_u64(v[0], d[i + 0]);
          v[1] = _mm_crc32_u64(v[1], d[i + 1]);
          v[2] = _mm_crc32_u64(v[2], d[i + 2]);
          v[3] = _mm_crc32_u64(v[3], d[i + 3]);
          v[4] = _mm_crc32_u64(v[4], d[i + 4]);
          v[5] = _mm_crc32_u64(v[5], d[i + 5]);
          v[6] = _mm_crc32_u64(v[6], d[i + 6]);
          v[7] = _mm_crc32_u64(v[7], d[i + 7]);
        }
        for (; i < n; ++i) v[0] = _mm_crc32_u64(v[0], d[i]);
      }
    }
    uint32_t sink = static_cast<uint32_t>(
        v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7]);
    double seconds = std::chrono::duration<double>(Clock::now() - started).count() / 3.0;
    printf("crc32_hw          %8.3f GiB/s  (sink %u)\n", gib_per_s(kLogical, seconds), sink);
    volatile uint32_t keep = sink;
    (void)keep;
  }

  // 2. SHA-256 over the whole 640 MiB stream (seal); best-of-5, matching the
  // standalone OpenSSL check (first call warms the library).
  {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    double best = 1e9;
    for (int trial = 0; trial < 5; ++trial) {
      auto started = Clock::now();
      SHA256_CTX ctx;
      SHA256_Init(&ctx);
      for (const auto& p : payloads) {
        SHA256_Update(&ctx, p.data(), p.size());
      }
      SHA256_Final(digest, &ctx);
      best = std::min(best, std::chrono::duration<double>(Clock::now() - started).count());
    }
    printf("sha256_seal       %8.3f GiB/s  (%02x%02x)\n", gib_per_s(kLogical, best),
           digest[0], digest[1]);
  }

  // 3. append: memcpy payload + 32-byte little header into one slab
  {
    auto started = Clock::now();
    std::vector<uint8_t> slab(kLogical + kRecords * 32);
    size_t offset = 0;
    for (const auto& p : payloads) {
      std::memcpy(slab.data() + offset, &offset, 8);
      std::memcpy(slab.data() + offset + 8, p.data(), p.size());
      offset += 32 + p.size();
    }
    double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    printf("append_memcpy     %8.3f GiB/s\n", gib_per_s(kLogical, seconds));
  }

  // 4. seal_footer: JSON footer for 10k records (string build only)
  {
    auto started = Clock::now();
    std::string footer;
    footer.reserve(1 << 20);
    char record[256];
    for (size_t i = 0; i < kRecords; ++i) {
      int n = std::snprintf(record, sizeof(record),
          "{\"capture_id\":\"capture-%012zu\",\"offset\":%zu,"
          "\"stored_length\":%zu,\"decoded_length\":%zu,"
          "\"codec\":\"none\",\"checksum\":\"%08x\"},",
          i, i * kPayload, kPayload, kPayload, static_cast<uint32_t>(i));
      footer.append(record, static_cast<size_t>(n));
    }
    footer.back() = ' ';
    double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    printf("seal_footer_json  %8.3f GiB/s  (%zu KiB footer)\n",
           gib_per_s(kLogical, seconds), footer.size() / 1024);
  }

  // 5. spool_write: one 640 MiB .open -> fsync -> rename .ready (NVMe)
  {
    std::vector<uint8_t> slab(kLogical);
    for (const auto& p : payloads) {
      static size_t offset = 0;
      std::memcpy(slab.data() + offset, p.data(), p.size());
      offset += p.size();
    }
    auto started = Clock::now();
    const char* dir = "/tmp/opencode/kbench";
    char path[256], final_path[256];
    snprintf(path, sizeof(path), "%s/pack.dmi-pack.open", dir);
    snprintf(final_path, sizeof(final_path), "%s/pack.dmi-pack.ready", dir);
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    size_t written = 0;
    while (written < kLogical) {
      ssize_t n = ::write(fd, slab.data() + written, kLogical - written);
      if (n <= 0) break;
      written += static_cast<size_t>(n);
    }
    ::fsync(fd);
    ::close(fd);
    ::rename(path, final_path);
    double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    printf("spool_write_fsync %8.3f GiB/s\n", gib_per_s(kLogical, seconds));
    ::unlink(final_path);
  }
  return 0;
}
