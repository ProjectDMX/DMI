#include <nmmintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
using Clock = std::chrono::steady_clock;
int main(){
  const size_t N=10'000, P=64*1024;
  std::vector<std::vector<uint8_t>> payloads(N, std::vector<uint8_t>(P,0xAB));
  const uint64_t logical=(uint64_t)N*P;
  uint64_t v[8]={0,0,0,0,0,0,0,0};
  auto started=Clock::now();
  for (const auto& p: payloads){
    const uint64_t* d=(const uint64_t*)p.data();
    for (size_t i=0;i<P/8;++i) v[i%8]=_mm_crc32_u64(v[i%8],d[i]);
  }
  double s=std::chrono::duration<double>(Clock::now()-started).count();
  uint64_t x=v[0]^v[1]^v[2]^v[3]^v[4]^v[5]^v[6]^v[7];
  printf("crc32c 8chain FULL 640MiB: %.2f GiB/s (%llu)\n",(double)logical/s/1024/1024/1024,(unsigned long long)x);
}
